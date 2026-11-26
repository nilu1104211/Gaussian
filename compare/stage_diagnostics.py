"""Audit formal raw best.pth on held-out VALIDATION objects, without training.

Uses the unchanged native overlapping-patch/FPS path. P2/P3 have equal final
counts; candidate points retain correspondence before independent final FPS.
P1 has fewer points and must not be interpreted as an equal-count comparison.
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from evaluate_pugaussian_protocol import (cloud_names, independent_unit_sphere,
    mesh_scene, p2f_scores, read_points, write_csv)
from retrain_common import ROOT, PUGCS_MODELS, read_json, sha256, source_hashes, write_json, pugcs_variant_options


def directional_scores(pred, gt):
    forward = ((pred - gt[cKDTree(gt).query(pred)[1]]) ** 2).sum(-1)
    backward = ((gt - pred[cKDTree(pred).query(gt)[1]]) ** 2).sum(-1)
    return dict(CD=float(forward.mean() + backward.mean()),
                HD=float(max(forward.max(), backward.max())),
                mean_pred_to_gt=float(forward.mean()), mean_gt_to_pred=float(backward.mean()),
                HD_pred_to_gt=float(forward.max()), HD_gt_to_pred=float(backward.max()))


def stage_scores(pred, gt, inputs, scene=None):
    scores = directional_scores(independent_unit_sphere(pred), independent_unit_sphere(gt))
    center = inputs.mean(0, keepdims=True)
    radius = max(float(np.linalg.norm(inputs - center, axis=1).max()), 1e-12)
    shared = directional_scores((pred - center) / radius, (gt - center) / radius)
    scores.update({'shared_' + key: value for key, value in shared.items()})
    if scene is not None:
        scores.update(p2f_scores(pred, scene))
    # Spacing is diagnostic and reported in shared input-radius units.
    distances = cKDTree(pred).query(pred, k=2)[0][:, 1] / radius
    scores.update(nn_spacing_mean=float(distances.mean()), nn_spacing_cv=float(distances.std() / max(distances.mean(), 1e-12)))
    return {key + '_x1000': value * 1000 for key, value in scores.items() if key not in ('nn_spacing_mean', 'nn_spacing_cv')} | {
        key: scores[key] for key in ('nn_spacing_mean', 'nn_spacing_cv')}


def load_formal_network(model_dir, device):
    import torch
    sys.path.insert(0, str(ROOT / 'our/PU-GCSNet'))
    from network.PUCRN import PUGCSNet
    from main import NETWORK_KEYS
    job = read_json(model_dir / 'config.json')
    name = job['model']
    if name not in PUGCS_MODELS:
        raise ValueError('Stage diagnostics only supports PU-GCSNet/PU-CRN variants')
    network = dict(stage2='baseline') if name == 'E0' else pugcs_variant_options(name)
    manifest_path = model_dir / 'manifest.json'
    if manifest_path.is_file():
        native = read_json(manifest_path)['native_arguments']
        network.update({key: native[key] for key in NETWORK_KEYS if key in native})
    network.update(job.get('network_config', {}))
    net = PUGCSNet(**network).to(device).eval()
    state = torch.load(model_dir / 'best.pth', map_location=device, weights_only=True)
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Expected formal best.pth raw state_dict, not a legacy/resume container')
    net.load_state_dict(state, strict=True)
    return net, network, job


def audit(args):
    import torch
    sys.path.insert(0, str(ROOT / 'our/PU-GCSNet'))
    from main import predict_shape
    from network.cross_stage_refiner import gather_parent
    from network.geometry_utils import local_pca_frame
    from network.geometry_feature_bridge import geometry_descriptor

    model_dir = args.model_dir.resolve()
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu for reference inference')
    net, network, job = load_formal_network(model_dir, args.device)
    if job['dataset'] != args.dataset:
        raise ValueError('Checkpoint and validation dataset differ')
    source = args.prepared_root / ('PU1K' if args.dataset == 'pu1k' else 'PU-GAN') / 'validation'
    names = sorted(cloud_names(source / 'input_2048'))
    if not names or set(names) != cloud_names(source / 'gt_8192'):
        raise ValueError('Validation inputs and GT must contain identical nonempty names')
    data_manifest = read_json(args.prepared_root / 'manifest.json')
    expected = data_manifest['datasets']['PU1K' if args.dataset == 'pu1k' else 'PU-GAN']['validation_shapes']
    if len(names) != expected:
        raise ValueError('Validation object count differs from prepared manifest')
    if args.limit:
        names = names[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', dict(status='running'))

    class PatchAudit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.records = []

        def forward(self, points, return_aux=False):
            final, aux = self.net(points, return_aux=True)
            move = final - aux['p2']
            if 'n_pred' in aux:
                normal = gather_parent(aux['n_pred'], aux['parent_idx'])
                scale = gather_parent(aux['local_scale'], aux['parent_idx']).squeeze(-1)
            else:
                frame2 = local_pca_frame(aux['p2'])
                normal, scale = frame2['normal'], frame2['scale'].squeeze(-1)
            component = (move * normal).sum(-1)
            tangent = (move.square().sum(-1) - component.square()).clamp_min(0).sqrt()
            geometry, _, residual = geometry_descriptor(local_pca_frame(aux['p1'], return_neighbors=True))
            self.records.append(dict(move_normal_over_d=float((component.abs() / scale).mean()),
                move_tangent_over_d=float((tangent / scale).mean()),
                move_total_over_d=float((move.norm(dim=-1) / scale).mean()),
                pca_quad_fit_rms=float(residual.mean()), pca_neighbor_distance_cv=float(geometry[..., 4].mean()),
                pca_valid_fraction=float(geometry[..., 7].mean())))
            return (final, aux) if return_aux else final

    wrapper = PatchAudit().eval()
    inference_args = SimpleNamespace(num_point=256, patch_num_ratio=3, save_stages=True)
    rows, movements, hashes = [], [], {}
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()
    try:
        for name in names:
            inputs = read_points(source / 'input_2048' / (name + '.xyz'), 2048)
            center = inputs.mean(0, keepdims=True)
            radius = max(float(np.linalg.norm(inputs - center, axis=1).max()), 1e-12)
            tensor = torch.as_tensor((inputs - center) / radius, device=args.device).T[None]
            wrapper.records.clear()
            # Inference finishes before GT or mesh is opened.
            with torch.no_grad():
                merged, candidates = predict_shape(wrapper, tensor, inference_args, return_candidates=True)
            if candidates['p2'].shape != candidates['p3'].shape:
                raise AssertionError('P2/P3 candidate correspondence was lost')
            gt = read_points(source / 'gt_8192' / (name + '.xyz'), 8192)
            mesh = source / 'meshes' / (name + '.off')
            scene = mesh_scene(mesh) if mesh.is_file() else None
            for phase, clouds in (('fps', merged), ('candidates', candidates)):
                for stage, cloud in clouds.items():
                    pred = cloud[0].T.cpu().numpy() * radius + center
                    if not np.isfinite(pred).all():
                        raise FloatingPointError('Nonfinite stage predictions')
                    expected_points = 4096 if stage == 'p1' else 8192
                    if phase == 'fps' and len(pred) != expected_points:
                        raise AssertionError('Unexpected final point count')
                    rows.append(dict(shape=name, stage=stage, phase=phase, n_points=len(pred),
                                     **stage_scores(pred, gt, inputs, scene)))
                    if args.save_points:
                        folder = args.output / phase / stage
                        folder.mkdir(parents=True, exist_ok=True)
                        np.savetxt(folder / (name + '.xyz'), pred, fmt='%.10g')
            movements.append(dict(shape=name, **{key: float(np.mean([r[key] for r in wrapper.records]))
                                                 for key in wrapper.records[0]}))
            hashes[name] = dict(input=sha256(source / 'input_2048' / (name + '.xyz')),
                                gt=sha256(source / 'gt_8192' / (name + '.xyz')),
                                mesh=sha256(mesh) if mesh.is_file() else None)
            print('Audited validation object ' + name, flush=True)
        summaries = []
        for phase in ('fps', 'candidates'):
            for stage in ('p1', 'p2', 'p3'):
                selected = [r for r in rows if r['phase'] == phase and r['stage'] == stage]
                keys = set.intersection(*(set(r) for r in selected)) - {'shape', 'phase', 'stage'}
                summaries.append(dict(stage=stage, phase=phase, n_shapes=len(selected),
                                      **{key: float(np.mean([r[key] for r in selected])) for key in sorted(keys)}))
        # Optional meshes may leave some rows without P2F fields.
        columns = set().union(*(r.keys() for r in rows))
        rows = [{key: row.get(key, '') for key in rows[0]} | {key: row.get(key, '') for key in sorted(columns - set(rows[0]))} for row in rows]
        write_csv(args.output / 'per_shape.csv', rows)
        write_csv(args.output / 'summary.csv', summaries)
        write_csv(args.output / 'movement.csv', movements)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        write_json(args.output / 'protocol.json', dict(status='completed', split='validation',
            model=job['model'], dataset=args.dataset, scope='subset' if args.limit else 'full validation',
            n_shapes=len(names), model_dir=str(model_dir), best_sha256=sha256(model_dir / 'best.pth'),
            prepared_smoke=bool(data_manifest.get('settings', {}).get('smoke', False)),
            checkpoint_job_smoke=bool(job.get('smoke', False)),
            network_config=network, seed=args.seed, device=args.device,
            parameter_count=sum(p.numel() for p in net.parameters()),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
            total_inference_and_metrics_seconds=time.monotonic() - started,
            p1_note='P1=4096, P2/P3=8192 after FPS; P1 is not an equal-count comparison',
            metric_note='formal: independent unit spheres; shared_*: input center/radius; P2F: raw coordinates; x1000 columns only',
            movement_note='P2/P3 paired before FPS; local parent frame and scale, absolute normal component; object-equal means',
            data_manifest_sha256=sha256(args.prepared_root / 'manifest.json'),
            hashes=hashes, sources=source_hashes()))
    except BaseException as error:
        write_json(args.output / 'protocol.json', dict(status='failed', error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True, help='Formal job folder containing raw best.pth and config.json')
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--dataset', choices=('pu1k', 'pugan'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing directories are refused')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='Optional development subset; 0 means all validation objects')
    parser.add_argument('--save-points', action='store_true')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    audit(args)


if __name__ == '__main__':
    main()
