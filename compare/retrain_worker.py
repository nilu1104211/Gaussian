"""Train one native model, select best on full held-out objects, then test best."""
import argparse
import contextlib
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from evaluate_pugaussian_protocol import evaluate, point_scores, read_points, write_csv
from retrain_common import read_json, sha256, source_hashes, stable_seed, write_json


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


@contextlib.contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def save_torch(path, value):
    staged = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, staged)
    os.replace(staged, path)


def infer_directory(adapter, source, destination, seed, expected=None):
    destination.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in (source / 'input_2048').glob('*.xyz'))
    if not names:
        raise ValueError(f'No whole-object inputs: {source}')
    if expected is not None and len(names) != expected:
        raise ValueError(f'{source}: expected {expected} whole objects, got {len(names)}')
    with isolated_rng():
        for name in names:
            seed_all(stable_seed(seed, 'inference', name))
            points = read_points(source / 'input_2048' / f'{name}.xyz', 2048)
            prediction = adapter.infer(points)
            np.savetxt(destination / f'{name}.xyz', prediction, fmt='%.10g')
    return len(names)


def validate(adapter, source, seed, scratch, expected):
    count = infer_directory(adapter, source, scratch, seed, expected)
    # Same float32 normalization and saved-point metrics as final test.
    rows, summary = evaluate(scratch, source / 'input_2048', source / 'gt_8192', source / 'meshes', count)
    # These files were created exclusively by this validation call, not user results.
    for name in sorted(p.stem for p in (source / 'gt_8192').glob('*.xyz')):
        (scratch / f'{name}.xyz').unlink()
    return summary


def run(config, resume=False):
    if not torch.cuda.is_available():
        raise RuntimeError('This worker requires the native CUDA training environment')
    from retrain_adapters import NativeAdapter, PatchDataset
    destination = Path(config['output'])
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / 'status.json'
    previous_config = destination / 'config.json'
    if previous_config.is_file() and read_json(previous_config) != config:
        raise ValueError('Run config changed; use a new output directory')
    if (destination / 'last.pth').exists() and not resume:
        raise FileExistsError('Existing checkpoint; use --resume-run')
    write_json(previous_config, config)
    write_json(status_path, dict(status='initializing'))
    try:
        seed_all(config.get('model_seed', config['seed']))
        adapter = NativeAdapter(config)
        dataset_folder = Path(config['prepared']) / ('PU1K' if config['dataset'] == 'pu1k' else 'PU-GAN')
        data_info = read_json(Path(config['prepared']) / 'manifest.json')['datasets'][dataset_folder.name]
        dataset = PatchDataset(dataset_folder / 'train.h5', adapter, config['seed'], config['input_policy'])
        generator = torch.Generator()
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, generator=generator,
                            num_workers=0, drop_last=adapter.kind in ('E0', 'E6'))
        if len(loader) == 0:
            raise ValueError('Training set smaller than batch size')
        best, best_epoch, start, step = float('inf'), None, 1, 0
        post_validation_updates, validation_seen = 0, False
        history = []
        last_path = destination / 'last.pth'
        if resume and last_path.is_file():
            saved = torch.load(last_path, map_location='cuda', weights_only=False)
            adapter.net.load_state_dict(saved['model'], strict=True)
            adapter.optimizer.load_state_dict(saved['optimizer'])
            adapter.scheduler.load_state_dict(saved['scheduler'])
            adapter.health.update(saved['health'])
            if hasattr(adapter, 'native') and hasattr(adapter.native, 'step'):
                adapter.native.step = saved['native_step']
            best = saved['best_val_cd'] if saved['best_val_cd'] is not None else float('inf')
            best_epoch, start, step = saved['best_epoch'], saved['epoch'] + 1, saved['global_step']
            history = saved['validation_history']
            validation_seen = bool(history)
            post_validation_updates = saved.get('post_validation_updates', 0)
            restore_rng(saved['rng'])
        write_json(destination / 'manifest.json', dict(config=config, sources=source_hashes(),
            torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
            data_manifest_sha256=sha256(Path(config['prepared']) / 'manifest.json'),
            native_arguments=vars(adapter.args)))
        with (destination / 'training.jsonl').open('a', encoding='utf-8') as log:
            for epoch in range(start, config['epochs'] + 1):
                dataset.epoch = epoch
                generator.manual_seed(stable_seed(config['seed'], 'shuffle', epoch))
                losses = []
                started = time.monotonic()
                for batch_index, batch in enumerate(loader):
                    result = adapter.train_batch(batch, step, epoch)
                    if not all(np.isfinite(v) for v in result.values()):
                        raise FloatingPointError('Nonfinite training diagnostic')
                    if validation_seen:
                        post_validation_updates += int(result['parameter_step_max'] > 0)
                    step += 1
                    loss = result.get('total_loss', result.get('total'))
                    losses.append(loss)
                    record = dict(epoch=epoch, global_step=step, **result)
                    log.write(__import__('json').dumps(record) + '\n')
                    if batch_index == 0 or (batch_index + 1) % 100 == 0:
                        print(record, flush=True)
                adapter.epoch_end()
                select = epoch % config['val_every'] == 0 or epoch == config['epochs']
                joint = adapter.kind != 'PU-Gaussian' or epoch > config['stage1_epochs']
                if select and joint:
                    summary = validate(adapter, dataset_folder / 'validation', config['seed'], destination / 'validation_xyz', data_info['validation_shapes'])
                    validation_seen = True
                    row = dict(epoch=epoch, **summary)
                    history.append(row)
                    if summary['CD'] < best:
                        best, best_epoch = summary['CD'], epoch
                        save_torch(destination / 'best.pth', adapter.net.state_dict())
                    write_csv(destination / 'validation.csv', history)
                    print(f'Validation epoch {epoch}: {summary}; best epoch {best_epoch}', flush=True)
                save_torch(last_path, dict(model=adapter.net.state_dict(), optimizer=adapter.optimizer.state_dict(),
                    scheduler=adapter.scheduler.state_dict(), epoch=epoch, global_step=step,
                    best_val_cd=best if np.isfinite(best) else None, best_epoch=best_epoch,
                    rng=rng_state(), health=adapter.health, native_step=getattr(getattr(adapter, 'native', None), 'step', 0),
                    validation_history=history, post_validation_updates=post_validation_updates, config=config))
                log.flush()
                write_json(status_path, dict(status='training', epoch=epoch, global_step=step,
                    best_epoch=best_epoch, best_val_cd=best if np.isfinite(best) else None))
                print(f'Epoch {epoch}/{config["epochs"]}: mean loss {np.mean(losses):.6g}, {time.monotonic()-started:.1f}s', flush=True)
        if not (destination / 'best.pth').is_file():
            raise RuntimeError('No best checkpoint selected from whole-object validation')
        if config['smoke'] and post_validation_updates == 0:
            raise RuntimeError('Smoke test did not verify training updates AFTER validation')
        if adapter.health['gradient_norm'] <= 0 or adapter.health['parameter_step_max'] <= 0:
            raise RuntimeError('Final training step has no effective gradient/parameter update')
        adapter.net.load_state_dict(torch.load(destination / 'best.pth', map_location='cuda', weights_only=True))
        count = infer_directory(adapter, dataset_folder / 'test', destination / 'test/xyz', config['seed'], data_info['test_shapes'])
        from evaluate_pugaussian_protocol import run as evaluate_run
        metrics_dir = destination / 'test/metrics'
        if metrics_dir.exists() and any(metrics_dir.iterdir()):
            # A resumed completed training may have an interrupted test stage. Preserve it.
            metrics_dir = destination / 'test' / f'metrics_retry_{time.time_ns()}'
        evaluate_run(argparse.Namespace(dataset='PU1K' if config['dataset'] == 'pu1k' else 'PU-GAN',
            prediction=[(config['model'], destination / 'test/xyz')], expected_shapes=count,
            input_dir=dataset_folder / 'test/input_2048', gt_dir=dataset_folder / 'test/gt_8192',
            mesh_dir=dataset_folder / 'test/meshes', output_dir=metrics_dir))
        write_json(destination / 'health.json', dict(passed=True, final_training=adapter.health,
            grad_head=getattr(adapter, 'head_health', None), query_gradient_norm=adapter.query_grad_norm,
            post_validation_updates=post_validation_updates, test_shapes=count, metrics_dir=str(metrics_dir)))
        write_json(status_path, dict(status='completed', best_epoch=best_epoch, best_val_cd=best,
            best_sha256=sha256(destination / 'best.pth'), last_sha256=sha256(last_path), metrics_dir=str(metrics_dir)))
    except BaseException as error:
        write_json(status_path, dict(status='failed', error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(read_json(args.config), args.resume)


if __name__ == '__main__':
    main()
