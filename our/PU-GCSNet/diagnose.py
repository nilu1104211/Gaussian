"""Paired whole-shape scores, validation height probes, and pre/post-FPS attribution."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import H5Dataset
from main import checkpoint_state, distance_metrics, make_network, predict_shape, unit_cloud
from network.model_loss import CD_dist
from utils import pc_utils


def save_report(folder, filename, rows, summary):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / filename).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = json.dumps(summary, indent=2, allow_nan=False)
    (folder / 'summary.json').write_text(report + '\n', encoding='utf-8')
    print(report, flush=True)


def xyz_names(folder):
    names = {p.stem for p in folder.glob('*.xyz') if not p.stem.endswith('_point2mesh_distance')}
    if not names:
        raise ValueError(f'No XYZ files in {folder}')
    return names


def paired(args):
    names = xyz_names(args.gt)
    for folder in (args.reference, args.candidate):
        actual = xyz_names(folder)
        if actual != names:
            raise ValueError(f'{folder}: missing {sorted(names - actual)}, extra {sorted(actual - names)}')
    pugan = {p.stem for p in args.pugan_meshes.glob('*.off')}
    if not pugan or not pugan <= names:
        raise ValueError(f'Invalid PU-GAN subset in {args.pugan_meshes}; unmatched names: {sorted(pugan - names)}')
    if args.expected_counts and [len(names), len(pugan)] != args.expected_counts:
        raise ValueError(f'Expected PU1K/PU-GAN counts {args.expected_counts}, found {len(names)}/{len(pugan)}')
    shared_inputs = getattr(args, 'shared_inputs', None)
    if shared_inputs and xyz_names(shared_inputs) != names:
        raise ValueError('Shared input names must match the evaluated shapes')

    metric = CD_dist()
    rows = []
    for name in sorted(names):
        filename = name + '.xyz'
        if shared_inputs:
            inputs = np.loadtxt(shared_inputs / filename, dtype=np.float32, ndmin=2)[:, :3]
            if inputs.shape[1] != 3 or not np.isfinite(inputs).all():
                raise ValueError(f'Invalid input cloud: {shared_inputs / filename}')
            _, center, radius = pc_utils.normalize_point_cloud(inputs[None])
            if not np.isfinite(radius).all() or (radius <= 0).any():
                raise ValueError(f'Degenerate input cloud: {shared_inputs / filename}')

            def load_cloud(path):
                xyz = np.loadtxt(path, dtype=np.float32, ndmin=2)[:, :3]
                if xyz.shape[1] != 3 or not np.isfinite(xyz).all():
                    raise ValueError(f'Invalid XYZ cloud: {path}')
                return torch.as_tensor((xyz[None] - center) / radius, dtype=torch.float32)
        else:
            load_cloud = unit_cloud
        gt = load_cloud(args.gt / filename)
        row = dict(name=name, subset='PU-GAN' if name in pugan else 'ShapeNet')
        for label, folder in [('reference', args.reference), ('candidate', args.candidate)]:
            cloud = load_cloud(folder / filename)
            if cloud.shape != gt.shape:
                raise ValueError(f'{folder / filename}: expected {gt.shape[1]} points, got {cloud.shape[1]}')
            values = distance_metrics(*metric(cloud, gt))
            for key, value in values.items():
                row[f'{label}_{key}'] = value.item()
        for key in values:
            row['delta_' + key] = row['candidate_' + key] - row['reference_' + key]
        rows.append(row)
    rows.sort(key=lambda row: row['delta_HD_squared'], reverse=True)
    groups = {}
    for group in ('PU1K', 'ShapeNet', 'PU-GAN'):
        subset = [r for r in rows if group == 'PU1K' or r['subset'] == group]
        if not subset:
            continue
        stats = dict(count=len(subset))
        for key in values:
            delta = np.array([r['delta_' + key] for r in subset])
            stats[key] = dict(reference_mean=np.mean([r['reference_' + key] for r in subset]),
                              candidate_mean=np.mean([r['candidate_' + key] for r in subset]),
                              mean_delta=delta.mean(), improved=int((delta < 0).sum()),
                              worsened=int((delta > 0).sum()))
        groups[group] = stats
    save_report(args.out, 'paired.csv', rows, dict(reference=str(args.reference), candidate=str(args.candidate),
                coordinates=('input centroid and radius shared by both predictions and GT' if shared_inputs
                             else 'independent unit spheres (official project scores)'), groups=groups))


def load_model(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device)
    checkpoint = checkpoint_state(args.ckpt, 'cpu')
    if not checkpoint.get('network_config'):
        raise ValueError('Diagnostics require a training checkpoint containing network_config')
    net, config = make_network(args, device, checkpoint)
    net.eval()
    return net, dict(checkpoint=str(args.ckpt), epoch=checkpoint.get('epoch'),
                     best_val_cd3=checkpoint.get('best_val'), network=config)


def patch_probe(args):
    net, info = load_model(args)
    if args.height_scale != 1:
        if net.stage2 != 'quadratic' or net.upsampling_stage_2.sampling != 'deterministic':
            raise ValueError('Height probes require a deterministic quadratic checkpoint')
        # All three quadratic coefficients are linear in this bound. Change it only on this eval model.
        net.upsampling_stage_2.height_bound *= args.height_scale
    dataset = H5Dataset(str(args.val_h5), args.num_point, args.num_point, phase='val', use_random_input=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metric = CD_dist()
    rows = []
    for inputs, gt, _ in loader:
        p3, aux = net(inputs.to(args.device), return_aux=True)
        gt = gt.transpose(1, 2).contiguous().to(args.device)
        p2 = aux['p2']
        columns = {}
        for stage, cloud in [('p2', p2), ('p3', p3)]:
            for key, value in distance_metrics(*metric(cloud, gt)).items():
                columns[stage + '_' + key] = value
        move = (p3 - p2).norm(dim=-1)
        columns.update(move_mean=move.mean(1), move_p95=move.quantile(0.95, dim=1), move_max=move.amax(1))
        if 'height' in aux:
            d = aux['local_scale']
            height = (aux['height'] / d).abs().flatten(1)
            midpoint = p2.reshape(p2.shape[0], -1, 2, 3).mean(2)
            mid_normal = (((midpoint - aux['p1']) * aux['n_pred']).sum(-1) / d.squeeze(-1)).abs()
            columns.update(height_over_d_p95=height.quantile(0.95, dim=1), height_over_d_max=height.amax(1),
                           midpoint_normal_over_d_p95=mid_normal.quantile(0.95, dim=1),
                           sigma1_over_d_mean=(aux['sigma'] / d)[:, :, 0].mean(1),
                           sigma2_over_d_mean=(aux['sigma'] / d)[:, :, 1].mean(1),
                           pca_min_ratio_p95=aux['pca_ratios'][:, :, 0].quantile(0.95, dim=1))
        keys = list(columns)
        batch = torch.stack(list(columns.values()), dim=1).cpu().tolist()
        for values in batch:
            rows.append(dict(patch_index=len(rows), **dict(zip(keys, values))))
    if not rows:
        raise ValueError(f'Empty validation set: {args.val_h5}')
    means = {key: np.mean([row[key] for row in rows]) for key in keys}
    save_report(args.out, 'patches.csv', rows, dict(**info, val_h5=str(args.val_h5), height_scale=args.height_scale,
                patches=len(rows), coordinates='shared validation patch frame', mean_of_patch_metrics=means))


def fps_probe(args):
    net, info = load_model(args)
    with args.paired_csv.open(encoding='utf-8', newline='') as stream:
        names = [row['name'] for row in csv.DictReader(stream)][:args.top]
    if not names:
        raise ValueError('Empty paired shape ranking')
    metric = CD_dist()
    rows = []
    for name in names:
        filename = name + '.xyz'
        cloud = np.loadtxt(args.inputs / filename, dtype=np.float32, ndmin=2)[:, :3]
        normalized, center, radius = pc_utils.normalize_point_cloud(cloud[None])
        inputs = torch.as_tensor(normalized, device=args.device).transpose(1, 2)
        gt = np.loadtxt(args.gt / filename, dtype=np.float32, ndmin=2)[:, :3]
        target = torch.as_tensor((gt[None] - center) / radius, device=args.device)
        merged, before = predict_shape(net, inputs, args, return_candidates=True)
        move = (before['p3'] - before['p2']).norm(dim=1)
        for stage in ('p2', 'p3'):
            for step, points in [('before', before[stage]), ('after', merged[stage])]:
                values = distance_metrics(*metric(points.transpose(1, 2).contiguous(), target))
                rows.append(dict(name=name, stage=stage, fps=step, points=points.shape[-1],
                                 paired_move_p95_before_fps=move.quantile(0.95).item(),
                                 **{key: value.item() for key, value in values.items()}))
    save_report(args.out, 'fps.csv', rows, dict(**info, shapes=names, paired_csv=str(args.paired_csv),
                coordinates='input centroid and radius shared by all candidates, FPS subsets and GT'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pair = commands.add_parser('paired', help='Compare existing XYZs using the project test protocol on CPU')
    for name in ('reference', 'candidate', 'gt', 'pugan-meshes'):
        pair.add_argument('--' + name, type=Path, required=True)
    pair.add_argument('--expected-counts', type=int, nargs=2, metavar=('PU1K', 'PUGAN'))
    pair.add_argument('--shared-inputs', type=Path,
                      help='Use this input cloud center/radius for both predictions and GT')
    patch = commands.add_parser('patches', help='Evaluate fixed held-out patches in H5 row order')
    patch.add_argument('--val-h5', type=Path, required=True)
    patch.add_argument('--batch-size', type=int, default=16)
    patch.add_argument('--height-scale', type=float, choices=(0, 0.5, 1), default=1)
    fps = commands.add_parser('fps', help='Trace the top HD regressions in paired.csv')
    for name in ('inputs', 'gt', 'paired-csv'):
        fps.add_argument('--' + name, type=Path, required=True)
    fps.add_argument('--top', type=int, default=5)
    fps.set_defaults(patch_num_ratio=3, save_stages=True)
    for command in (patch, fps):
        command.add_argument('--ckpt', type=Path, required=True)
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--num-point', type=int, default=256)
        command.set_defaults(up_ratio=4)
    for command in (pair, patch, fps):
        command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fps' and args.top < 1:
        parser.error('--top must be positive')
    np.random.seed(42)
    torch.manual_seed(42)
    with torch.no_grad():
        if args.command == 'paired':
            paired(args)
        elif args.command == 'patches':
            patch_probe(args)
        else:
            fps_probe(args)


if __name__ == '__main__':
    main()
