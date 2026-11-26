"""PU-GCSNet training, patch inference and point-cloud evaluation."""
import argparse
import csv
import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import H5Dataset
from model import Model
from network.PUCRN import PUGCSNet
from network import operations
from network.model_loss import CD_dist
from network.point_ops import furthest_point_sample, gather_operation
from network.variants import NEW_FLAGS, NEW_BOUNDS, add_structure_arguments
from utils import pc_utils


NETWORK_KEYS = ('stage2', 'sampling', 'k', 'sigma_min', 'sigma_max',
                'center_bound', 'normal_bound', 'height_bound', 'rho',
                'refinement', 'refine_radius', *NEW_FLAGS, *NEW_BOUNDS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train', 'test', 'eval'), default='train')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--name', default='pu_gcsnet')
    parser.add_argument('--log_dir', default='./model')
    parser.add_argument('--result_dir', default='./model/test/result')
    parser.add_argument('--ckpt')
    parser.add_argument('--init_ckpt', help='Warm start matching PU-CRN backbone tensors')
    parser.add_argument('--resume', help='Resume a full training checkpoint')
    parser.add_argument('--restore_epoch', type=int, default=0)
    parser.add_argument('--h5_data', help='Training-only H5, with held-out shapes removed')
    parser.add_argument('--val_h5', help='Fixed held-out validation patches')
    parser.add_argument('--val_data', help='Optional validation XYZ input directory')
    parser.add_argument('--val_gt', help='Validation XYZ ground truth directory')
    parser.add_argument('--test_data', help='Test XYZ input directory')
    parser.add_argument('--gt_path', help='Test XYZ ground truth directory')
    parser.add_argument('--num_point', type=int, default=256)
    parser.add_argument('--num_shape_point', type=int, default=2048)
    parser.add_argument('--up_ratio', type=int, choices=(4,), default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_step', type=int, default=4)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=0, help='Optional smoke-run optimizer step limit')
    parser.add_argument('--decay_iter', type=int, default=50000)
    parser.add_argument('--lr_init', type=float, default=0.001)
    parser.add_argument('--patch_num_ratio', type=float, default=3)
    parser.add_argument('--jitter', action='store_true')
    parser.add_argument('--jitter_sigma', type=float, default=0.005)
    parser.add_argument('--jitter_max', type=float, default=0.01)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--stage2', choices=('baseline', 'gaussian3d', 'tangent', 'quadratic'), default='quadratic')
    parser.add_argument('--sampling', choices=('deterministic', 'random', 'antithetic'), default='deterministic')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--gt_k', type=int, default=24)
    parser.add_argument('--sigma_min', type=float, default=0.15)
    parser.add_argument('--sigma_max', type=float, default=0.75)
    parser.add_argument('--center_bound', type=float, default=0.2)
    parser.add_argument('--normal_bound', type=float, default=0.2)
    parser.add_argument('--height_bound', type=float, default=0.25)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--lambda_surf', type=float, default=0.0)
    parser.add_argument('--lambda_normal', type=float, default=0.0)
    parser.add_argument('--surf_eta', type=float, default=0.0)
    parser.add_argument('--confidence_tau', type=float, default=0.0)
    parser.add_argument('--refinement', choices=('legacy', 'local'), default='legacy')
    parser.add_argument('--refine_radius', type=float, default=0.5,
                        help='Local refinement displacement bound in units of parent kNN scale')
    parser.add_argument('--lambda_tail', type=float, default=0.0)
    parser.add_argument('--lambda_growth', type=float, default=0.0)
    parser.add_argument('--tail_fraction', type=float, default=0.02)
    parser.add_argument('--tail_warmup_epochs', type=int, default=10)
    parser.add_argument('--save_stages', action='store_true', help='Save merged P1/P2 and P3 predictions')
    add_structure_arguments(parser)
    args = parser.parse_args()
    if min(args.batch_size, args.save_step, args.print_step, args.decay_iter, args.max_epoch) <= 0:
        parser.error('Batch size, epochs and step intervals must be positive')
    if min(args.lambda_surf, args.lambda_normal, args.surf_eta, args.confidence_tau) < 0:
        parser.error('Loss weights and confidence threshold must be nonnegative')
    if (not np.isfinite([args.lambda_tail, args.lambda_growth, args.tail_fraction, args.refine_radius]).all()
            or min(args.lambda_tail, args.lambda_growth, args.tail_warmup_epochs) < 0
            or not 0 < args.tail_fraction <= 1 or args.refine_radius <= 0):
        parser.error('HD-Guard requires finite nonnegative weights, fraction in (0,1], and positive radius')
    if args.lambda_normal and args.stage2 in ('baseline', 'gaussian3d'):
        parser.error('Normal loss requires tangent or quadratic Stage-2')
    if bool(args.val_data) != bool(args.val_gt):
        parser.error('--val_data and --val_gt must be supplied together')
    if args.val_h5 and args.val_data:
        parser.error('Choose either --val_h5 or --val_data/--val_gt')
    if args.num_point < 16 or args.patch_num_ratio < 1:
        parser.error('num_point must be >=16 and patch_num_ratio must be >=1')
    return args


def checkpoint_state(path, device):
    # Training checkpoints include Python/NumPy RNG state. Use trusted local files.
    return torch.load(path, map_location=device, weights_only=False)


def make_network(args, device, checkpoint=None):
    config = checkpoint.get('network_config') if checkpoint is not None else None
    if config is None:
        config = {key: getattr(args, key) for key in NETWORK_KEYS}
    # Old E0-E4 checkpoints retain exactly the original refinement behavior.
    config = dict(config)
    config.setdefault('refinement', 'legacy')
    config.setdefault('refine_radius', 0.5)
    net = PUGCSNet(args.up_ratio, **config).to(device)
    if checkpoint is not None:
        net.load_state_dict(checkpoint['net_state_dict'])
    return net, config


def warm_start(net, path, device):
    state = checkpoint_state(path, device)
    state = state.get('net_state_dict', state)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    current = net.state_dict()
    matching = {key: value for key, value in state.items() if key in current and value.shape == current[key].shape}
    if not matching:
        raise ValueError('No compatible tensors found in initialization checkpoint')
    net.load_state_dict(matching, strict=False)
    logging.info('Warm start loaded %d tensors; initialized %d new tensors', len(matching), len(current) - len(matching))


def save_checkpoint(path, net, model, config, epoch, steps, best, args, best_hd=float('inf')):
    state = dict(net_state_dict=net.state_dict(), network_config=config,
                 optimizer=model.optimizer.state_dict(), scheduler=model.lr_scheduler.state_dict(),
                 epoch=epoch, steps=steps, best_val=best, best_hd_val=best_hd, args=vars(args),
                 torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                 python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate_patches(net, loader, device):
    net.eval()
    metric = CD_dist()
    totals = {}
    count = 0
    for inputs, target, _ in loader:
        inputs, target = inputs.to(device), target.to(device).transpose(1, 2)
        final, aux = net(inputs, return_aux=True)
        for stage, pred in ((2, aux['p2']), (3, final)):
            forward, backward = metric(pred, target)
            values = distance_metrics(forward, backward)
            for label, key in (('cd', 'CD'), ('hd', 'HD_squared'),
                               ('hd_forward', 'HD_pred_to_gt_sq'), ('hd_backward', 'HD_gt_to_pred_sq')):
                name = f'{label}{stage}'
                totals[name] = totals.get(name, 0.0) + values[key].sum().item()
        count += len(inputs)
    if count == 0:
        raise ValueError('Empty validation dataset')
    return {name: total / count for name, total in totals.items()}


def train(args, device):
    if not args.h5_data:
        raise ValueError('Training requires --h5_data')
    folder = Path(args.log_dir) / args.name
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(folder / 'train.log'), logging.StreamHandler()])
    resume_path = args.resume
    if args.restore_epoch:
        resume_path = folder / ('model_%d.pth' % args.restore_epoch)
    if resume_path and args.init_ckpt:
        raise ValueError('Choose resume or initialization checkpoint')
    saved = checkpoint_state(resume_path, device) if resume_path else None
    if saved is not None and 'lambda_tail' in saved.get('args', {}):
        # A new objective is an initialization experiment, not an exact resume.
        for key in ('lambda_tail', 'lambda_growth', 'tail_fraction', 'tail_warmup_epochs',
                    'lambda_surf', 'lambda_normal', 'surf_eta', 'gt_k', 'confidence_tau'):
            if getattr(args, key) != saved['args'][key]:
                raise ValueError(f'Resume must preserve {key}={saved["args"][key]}; use the original run command')
    net, config = make_network(args, device, saved)
    if args.lambda_normal and config['stage2'] in ('baseline', 'gaussian3d'):
        raise ValueError('Checkpoint architecture does not provide supervised normals')
    if args.init_ckpt:
        warm_start(net, args.init_ckpt, device)
    logging.info('Parameters %d; network %s', sum(p.numel() for p in net.parameters()), config)
    (folder / 'config.json').write_text(json.dumps(dict(arguments=vars(args), network=config), indent=2), encoding='utf-8')
    code = folder / 'code'
    shutil.copytree(Path(__file__).parent / 'network', code / 'network', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(Path(__file__).parent / 'utils', code / 'utils', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for source in ('main.py', 'model.py', 'data.py'):
        shutil.copy2(Path(__file__).parent / source, code / source)
    dataset = H5Dataset(args.h5_data, args.num_point, args.num_point, up_ratio=4,
                        jitter=args.jitter, jitter_sigma=args.jitter_sigma, jitter_max=args.jitter_max)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=True)
    if len(loader) == 0:
        raise ValueError('Training set is smaller than batch_size')
    val_loader = None
    if args.val_h5:
        val = H5Dataset(args.val_h5, args.num_point, args.num_point, phase='val', up_ratio=4, use_random_input=False)
        val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers)
    if not args.val_h5 and not args.val_data:
        logging.info('No validation supplied; saving epoch checkpoints only')
    model = Model(net, 'train', args)
    steps, start_epoch, best, best_hd = 0, 1, float('inf'), float('inf')
    if saved is not None:
        if 'optimizer' in saved:
            model.optimizer.load_state_dict(saved['optimizer'])
            model.lr_scheduler.load_state_dict(saved['scheduler'])
        steps, start_epoch = saved.get('steps', 0), saved.get('epoch', args.restore_epoch) + 1
        best = saved.get('best_val', best)
        best_hd = saved.get('best_hd_val', best_hd)
        if 'torch_rng' in saved:
            torch.set_rng_state(saved['torch_rng'].cpu())
            np.random.set_state(saved['numpy_rng'])
            random.setstate(saved['python_rng'])
            if device.type == 'cuda' and saved['cuda_rng']:
                torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    with (folder / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(start_epoch, args.max_epoch + 1):
            for inputs, target, radius in loader:
                model.set_input(inputs.to(device), radius.to(device), target.to(device))
                _, lr = model.optimize(steps, epoch)
                steps += 1
                record = dict(epoch=epoch, step=steps, lr=lr, **model.last_metrics)
                log.write(json.dumps(record) + '\n')
                if steps % args.print_step == 0 or steps == 1:
                    logging.info('%s', record)
                if args.max_steps and steps >= args.max_steps:
                    break
            log.flush()
            if epoch % args.save_step == 0 or epoch == args.max_epoch or (args.max_steps and steps >= args.max_steps):
                # Validation RNG is isolated so its frequency cannot alter training samples.
                torch_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
                cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else []
                torch.manual_seed(args.random_seed)
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                scores = None
                if val_loader is not None:
                    scores = validate_patches(net, val_loader, device)
                elif args.val_data:
                    target_folder = test(args, net, device, args.val_data, folder / 'validation')
                    values = evaluate(target_folder, args.val_gt)
                    scores = dict(cd3=values['CD'], hd3=values['HD_squared'],
                                  hd_forward3=values['HD_pred_to_gt_sq'], hd_backward3=values['HD_gt_to_pred_sq'])
                torch.set_rng_state(torch_rng)
                np.random.set_state(numpy_rng)
                random.setstate(python_rng)
                if cuda_rng:
                    torch.cuda.set_rng_state_all(cuda_rng)
                improved = scores is not None and scores['cd3'] < best
                improved_hd = scores is not None and scores['hd3'] < best_hd
                if scores is not None:
                    best = min(best, scores['cd3'])
                    best_hd = min(best_hd, scores['hd3'])
                    logging.info('Validation epoch %d: %s', epoch, scores)
                    log.write(json.dumps(dict(epoch=epoch, validation=scores)) + '\n')
                    log.flush()
                if improved:
                    save_checkpoint(folder / 'best.pth', net, model, config, epoch, steps, best, args, best_hd)
                if improved_hd:
                    save_checkpoint(folder / 'best_hd.pth', net, model, config, epoch, steps, best, args, best_hd)
                save_checkpoint(folder / ('model_%d.pth' % epoch), net, model, config, epoch, steps, best, args, best_hd)
                save_checkpoint(folder / 'latest.pth', net, model, config, epoch, steps, best, args, best_hd)
            if args.max_steps and steps >= args.max_steps:
                break


@torch.no_grad()
def predict_shape(net, points, args, return_candidates=False):
    # Same overlapping 256-point patches and final FPS protocol as PU-CRN.
    count = points.shape[2]
    if count < args.num_point:
        raise ValueError('Shape has fewer points than one inference patch')
    patches_count = min(count, max(1, int(count / args.num_point * args.patch_num_ratio)))
    _, seeds = operations.fps_subsample(points, patches_count)
    patches, _, _ = operations.group_knn(args.num_point, seeds, points)
    predictions = {'p3': []}
    if args.save_stages:
        predictions.update(p1=[], p2=[])
    for i in range(patches_count):
        patch, centroid, scale = operations.normalize_point_batch(patches[:, :, i])
        if args.save_stages:
            final, aux = net(patch, return_aux=True)
            outputs = dict(p1=aux['p1'], p2=aux['p2'], p3=final)
        else:
            outputs = dict(p3=net(patch))
        for name, output in outputs.items():
            predictions[name].append(output.transpose(1, 2) * scale + centroid)
    merged, candidates = {}, {}
    for name, chunks in predictions.items():
        desired = count * (2 if name == 'p1' else 4)
        cloud = torch.cat(chunks, dim=2)
        _, merged[name] = operations.fps_subsample(cloud, desired)
        if return_candidates:
            candidates[name] = cloud
    return (merged, candidates) if return_candidates else merged


@torch.no_grad()
def test(args, net, device, input_dir=None, output_dir=None):
    input_dir = Path(input_dir or args.test_data)
    output_dir = Path(output_dir or args.result_dir) / input_dir.name
    files = sorted(input_dir.glob('*.xyz'))
    if not files:
        raise ValueError('No XYZ inputs found in %s' % input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net.eval()
    seconds = []
    for path in files:
        cloud = pc_utils.load(str(path), args.num_shape_point)[:, :3]
        normalized, center, radius = pc_utils.normalize_point_cloud(cloud[None])
        inputs = torch.as_tensor(normalized, dtype=torch.float32, device=device).transpose(1, 2)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        predictions = predict_shape(net, inputs, args)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds.append(time.perf_counter() - start)
        for name, pred in predictions.items():
            destination = output_dir if name == 'p3' else output_dir / name
            destination.mkdir(parents=True, exist_ok=True)
            restored = pred.transpose(1, 2).cpu().numpy() * radius + center
            np.savetxt(destination / path.name, restored[0], fmt='%.6f')
    (output_dir / 'inference.json').write_text(json.dumps(dict(
        shapes=len(files), mean_shape_seconds=float(np.mean(seconds)),
        includes_patch_extraction_and_fps=True, save_stages=args.save_stages), indent=2), encoding='utf-8')
    return output_dir


def unit_cloud(path):
    xyz = np.loadtxt(path, ndmin=2)[:, :3]
    if xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError(f'Invalid XYZ coordinates: {path}')
    normalized, _, _ = pc_utils.normalize_point_cloud(xyz[None])
    return torch.as_tensor(normalized, dtype=torch.float32)


def distance_metrics(forward, backward):
    """Per-object metrics from squared nearest distances, prediction first."""
    pred_hd = forward.amax(dim=-1)
    gt_hd = backward.amax(dim=-1)
    return dict(CD=forward.mean(-1) + backward.mean(-1),
                HD_squared=torch.maximum(pred_hd, gt_hd),
                HD_legacy_sum=pred_hd + gt_hd,
                HD_pred_to_gt_sq=pred_hd, HD_gt_to_pred_sq=gt_hd)


def evaluate(pred_dir, gt_dir):
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    predictions = sorted(pred_dir.glob('*.xyz'))
    predictions = [p for p in predictions if not p.stem.endswith('_point2mesh_distance')]
    if not predictions:
        raise ValueError('No predictions to evaluate')
    metric = CD_dist()
    rows = []
    all_p2f = []
    for path in predictions:
        reference = gt_dir / path.name
        if not reference.is_file():
            raise ValueError('Missing ground truth: %s' % reference)
        # Preserve the baseline evaluation normalization for existing CD comparisons.
        values = distance_metrics(*metric(unit_cloud(path), unit_cloud(reference)))
        rows.append(dict(name=path.stem, **{key: value.item() for key, value in values.items()}))
        distance_file = path.with_name(path.stem + '_point2mesh_distance.xyz')
        if distance_file.exists():
            distances = np.loadtxt(distance_file, ndmin=2)[:, 3]
            distances = distances[np.isfinite(distances)]
            if distances.size:
                rows[-1].update(P2F_mean=float(distances.mean()), P2F_std=float(distances.std()))
                all_p2f.append(distances)
    scores = {key: float(np.mean([row[key] for row in rows])) for key in values}
    if all_p2f:
        distances = np.concatenate(all_p2f)
        scores.update(P2F_mean=float(distances.mean()), P2F_std=float(distances.std()),
                      P2F_shapes=len(all_p2f))
    with (pred_dir / 'evaluation.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', *scores], restval='')
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(dict(name='mean', **scores))
    (pred_dir / 'metrics.json').write_text(json.dumps(scores, indent=2), encoding='utf-8')
    print(scores)
    return scores


def main():
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = torch.device('cuda', args.gpu) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if args.phase == 'train':
        train(args, device)
    elif args.phase == 'test':
        if not args.ckpt or not args.test_data:
            raise ValueError('Testing requires --ckpt and --test_data')
        net, _ = make_network(args, device, checkpoint_state(args.ckpt, device))
        result = test(args, net, device)
        if args.gt_path:
            evaluate(result, args.gt_path)
    else:
        if not args.test_data or not args.gt_path:
            raise ValueError('Evaluation requires --test_data (input directory name) and --gt_path')
        evaluate(Path(args.result_dir) / Path(args.test_data).name, args.gt_path)


if __name__ == '__main__':
    main()
