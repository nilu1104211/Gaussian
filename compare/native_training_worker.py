"""Train one method on its native H5 distribution; test a predeclared final epoch."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_pugaussian_protocol import evaluate, write_csv
from retrain_common import read_json, sha256, stable_seed, write_json
from retrain_worker import isolated_rng, rng_state, restore_rng, save_torch, seed_all, infer_directory
from native_training_protocol import runtime_sources


def gradient_probe(adapter, sample, seed):
    """Probe the learned distance field without changing optimizer, BN or RNG state."""
    if adapter.kind != 'Grad-PU':
        return {}
    training = adapter.net.training
    flags = [p.requires_grad for p in adapter.net.parameters()]
    try:
        adapter.net.eval()
        for p in adapter.net.parameters():
            p.requires_grad_(False)
        with isolated_rng(), torch.enable_grad():
            seed_all(stable_seed(seed, 'native-query-probe'))
            points = sample['input'][:2].cuda().transpose(1, 2).contiguous()
            with torch.no_grad():
                interpolated = adapter.utils.midpoint_interpolate(adapter.args, points)
            query = adapter.utils.get_query_points(interpolated, adapter.args).detach().requires_grad_(True)
            distance = adapter.net(interpolated, query)
            gradient, = torch.autograd.grad(distance.sum(), query)
            norm = float(gradient.norm())
            positive = float((distance > 0).float().mean())
            result = dict(query_gradient_norm=norm, distance_positive_fraction=positive)
            if not np.isfinite(norm) or norm <= 0 or positive <= 0:
                raise RuntimeError(f'Grad-PU distance field is inactive in inference mode: {result}; '
                                   f'head={getattr(adapter, "head_health", {})}')
            return result
    finally:
        for p, flag in zip(adapter.net.parameters(), flags):
            p.requires_grad_(flag)
        adapter.net.train(training)


def smoke_inference(adapter, dataset, seed):
    """Health only: concatenate training targets, never select on benchmark test data."""
    with isolated_rng():
        seed_all(stable_seed(seed, 'native-smoke-inference'))
        first, second = dataset[0], dataset[min(1, len(dataset)-1)]
        points = torch.cat([first['gt'], second['gt']], dim=0).numpy()
        prediction = adapter.infer(points)
    return dict(inference_points=len(prediction), query_gradient_norm=adapter.query_grad_norm,
                inference_source='two_training_target_patches_not_a_quality_evaluation')


def run(config, resume=False):
    from native_training_adapter import PublisherAdapter, PublisherDataset
    destination = Path(config['output'])
    destination.mkdir(parents=True, exist_ok=True)
    status_file, config_file = destination/'status.json', destination/'config.json'
    last_file = destination/'last.pth'
    if config_file.exists() and read_json(config_file) != config:
        raise ValueError('Native run configuration changed; use a new run directory')
    if last_file.exists() and not resume:
        raise FileExistsError('Existing checkpoint; use --resume-run')
    write_json(config_file, config)
    write_json(status_file, dict(status='initializing'))
    dataset, adapter = None, None
    failure_phase = 'initializing'
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('Native training requires CUDA and the server native extensions')
        info = config['data']
        if sha256(info['training_h5']) != info['training_h5_sha256']:
            raise ValueError('Training H5 fingerprint changed')
        seed_all(config['model_seed'])
        adapter = PublisherAdapter(config)
        limit = config['batch_size'] * config['smoke_batches'] if config['smoke'] else None
        dataset = PublisherDataset(info['training_h5'], adapter, config['model_seed'], limit)
        generator = torch.Generator()
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0,
                            generator=generator, drop_last=config['model'] in ('E0', 'E6'))
        if not len(loader):
            raise ValueError('Training H5 is smaller than the native batch size')
        sources = runtime_sources()
        step, start, history = 0, 1, []
        if resume and last_file.exists():
            saved = torch.load(last_file, map_location='cpu', weights_only=False)
            if saved['config'] != config or saved['sources'] != sources:
                raise ValueError('Resume code/config differs from checkpoint')
            adapter.net.load_state_dict(saved['model'], strict=True)
            adapter.optimizer.load_state_dict(saved['optimizer'])
            adapter.scheduler.load_state_dict(saved['scheduler'])
            adapter.health.update(saved['health'])
            if hasattr(adapter, 'native') and hasattr(adapter.native, 'step'):
                adapter.native.step = saved['native_step']
            step, start = saved['global_step'], saved['epoch']+1
            history = saved['epoch_history']
            restore_rng(saved['rng'])
            del saved
        write_json(destination/'manifest.json', dict(config=config, sources=sources,
            torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
            parameter_count=sum(p.numel() for p in adapter.net.parameters()),
            native_arguments=vars(adapter.args), training_patches=len(dataset),
            selection='fixed_final_epoch_no_validation_or_test_selection',
            loader='lazy H5; original method Dataset.__getitem__; num_workers=0'))
        probe_sample = None
        with (destination/'training.jsonl').open('a', encoding='utf-8') as log:
            for epoch in range(start, config['epochs']+1):
                dataset.epoch = epoch
                generator.manual_seed(stable_seed(config['model_seed'], 'native-shuffle', epoch))
                sums, examples = {}, 0
                started = time.monotonic()
                for index, batch in enumerate(loader):
                    if probe_sample is None:
                        probe_sample = {k: v[:2].clone() for k, v in batch.items()}
                    failure_phase = 'training_batch'
                    result = adapter.train_batch(batch, step, epoch)
                    if not all(np.isfinite(v) for v in result.values()):
                        raise FloatingPointError('Nonfinite native training metric')
                    amount = len(batch['input'])
                    for key, value in result.items():
                        sums[key] = sums.get(key, 0.) + float(value)*amount
                    examples += amount
                    step += 1
                    if index == 0 or (index+1) % config['log_every'] == 0 or index+1 == len(loader):
                        record = dict(epoch=epoch, global_step=step, **result)
                        log.write(__import__('json').dumps(record)+'\n')
                        log.flush()
                        print(record, flush=True)
                adapter.epoch_end()
                failure_phase = 'distance_probe'
                probe = gradient_probe(adapter, probe_sample, config['model_seed'])
                row = dict(epoch=epoch, global_step=step, mean_loss=sums.get('total_loss', sums.get('total'))/examples,
                           seconds=time.monotonic()-started, **probe)
                row.update({'mean_'+key: value/examples for key, value in sums.items()
                            if key in ('positive_fraction', 'raw_min', 'raw_max', 'gradient_norm',
                                       'parameter_step_max', 'chamfer_loss', 'img_loss')})
                history.append(row)
                write_csv(destination/'epochs.csv', history)
                save_torch(last_file, dict(model=adapter.net.state_dict(), optimizer=adapter.optimizer.state_dict(),
                    scheduler=adapter.scheduler.state_dict(), rng=rng_state(), sources=sources, config=config,
                    epoch=epoch, global_step=step, epoch_history=history, health=adapter.health,
                    native_step=getattr(getattr(adapter, 'native', None), 'step', 0)))
                write_json(status_file, dict(status='training', epoch=epoch, global_step=step,
                                            selection=config['selection']))
                print(f'Native epoch {epoch}/{config["epochs"]}: mean loss {row["mean_loss"]:.7g}; '
                      f'{row["seconds"]:.1f}s; {probe}', flush=True)
        failure_phase = 'final_inference'
        if adapter.health['gradient_norm'] <= 0 or adapter.health['parameter_step_max'] <= 0:
            raise RuntimeError('Final native training step has no gradient/effective update')
        if config['smoke']:
            inference = smoke_inference(adapter, dataset, config['model_seed'])
            write_json(destination/'health.json', dict(passed=True, final_training=adapter.health,
                       grad_head=getattr(adapter, 'head_health', None), **inference))
            write_json(status_file, dict(status='completed', smoke=True, epoch=config['epochs'],
                       last_sha256=sha256(last_file), test_evaluated=False))
            return
        final_file = destination/'final.pth'
        save_torch(final_file, adapter.net.state_dict())
        metrics_dir, xyz_dir = None, None
        if config['evaluate_test']:
            from prepare_native_data import test_files
            test = Path(info['test_root'])
            if test_files(test, info['test_shapes']) != info['test_sha256']:
                raise ValueError('Shared fixed test fingerprint changed')
            attempt = destination/'test'
            if attempt.exists() and any(attempt.iterdir()):
                attempt = destination/f'test_retry_{time.time_ns()}'
            xyz_dir = attempt/'xyz'
            infer_directory(adapter, test, xyz_dir, config['seed'], info['test_shapes'])
            rows, summary = evaluate(xyz_dir, test/'input_2048', test/'gt_8192', test/'meshes', info['test_shapes'])
            metrics_dir = attempt/'metrics'
            metrics_dir.mkdir(parents=True)
            write_csv(metrics_dir/'per_shape.csv', [dict(model=config['model'], shape=r['shape'],
                **{k+'_x1000': v*1000 for k, v in r.items() if k != 'shape'}) for r in rows])
            write_csv(metrics_dir/'summary.csv', [dict(model=config['model'], dataset=config['dataset'],
                n_shapes=len(rows), epoch=config['epochs'], selection=config['selection'],
                **{k+'_x1000': v*1000 for k, v in summary.items()})])
            write_json(metrics_dir/'protocol.json', dict(config=config,
                CD='independent float32 unit-sphere, bidirectional mean squared NN sum',
                HD='maximum directed squared NN', P2F='saved-coordinate unsigned point-to-triangle',
                scale_factor=1000, test_sha256=info['test_sha256'],
                prediction_sha256={p.name:sha256(p) for p in xyz_dir.glob('*.xyz')}))
        write_json(destination/'health.json', dict(passed=True, final_training=adapter.health,
                   grad_head=getattr(adapter, 'head_health', None),
                   query_gradient_norm=adapter.query_grad_norm, test_evaluated=config['evaluate_test']))
        write_json(destination/'performance.json', dict(
            parameter_count=sum(p.numel() for p in adapter.net.parameters()),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            inference_timing=getattr(adapter, 'last_inference_timing', None)))
        write_json(status_file, dict(status='completed', epoch=config['epochs'],
            selection=config['selection'], final_sha256=sha256(final_file), last_sha256=sha256(last_file),
            metrics_dir=str(metrics_dir) if metrics_dir else None, xyz_dir=str(xyz_dir) if xyz_dir else None,
            test_evaluated=config['evaluate_test']))
    except BaseException as error:
        write_json(status_file, dict(status='failed', error=str(error),
            health=getattr(adapter, 'health', None), grad_head=getattr(adapter, 'head_health', None)))
        if adapter is not None and adapter.kind == 'Grad-PU' and failure_phase in ('training_batch', 'distance_probe'):
            try:
                failed_batch = probe_sample if failure_phase == 'distance_probe' else batch
                np.savez_compressed(destination/'failure_batch.npz',
                                    **{k:v.detach().cpu().numpy() for k,v in failed_batch.items()})
                save_torch(destination/'failure.pth', dict(model=adapter.net.state_dict(),
                    optimizer=adapter.optimizer.state_dict(), epoch=epoch, global_step=step,
                    phase=failure_phase, config=config, sources=sources, rng=rng_state()))
                write_json(destination/'failure.json', dict(error=str(error), phase=failure_phase,
                    epoch=epoch, global_step=step, grad_head=getattr(adapter, 'head_health', None),
                    usage='Diagnostic state only; resume uses the last complete epoch in last.pth'))
            except Exception as diagnostic_error:
                print(f'Could not export failure diagnostic: {diagnostic_error}', flush=True)
        raise
    finally:
        if dataset is not None:
            dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(read_json(args.config), args.resume)


if __name__ == '__main__':
    main()
