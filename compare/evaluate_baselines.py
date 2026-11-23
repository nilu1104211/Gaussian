"""Infer four trained baselines and evaluate PU1K with the PU-GCSNet protocols."""

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


COMPARE = Path(__file__).resolve().parent
MODELS = ('PU-Gaussian-1-199', 'PU-Gaussian-100-100', 'Grad-PU', 'RepKPU')
DISTANCE_KEYS = ('CD', 'HD_squared', 'HD_legacy_sum', 'HD_pred_to_gt_sq', 'HD_gt_to_pred_sq')
SEEDED_ENTRY = '''
import random
import runpy
import sys
import numpy as np
import torch

seed = int(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''


def write_json(path, value):
    staged = path.with_suffix(path.suffix + '.tmp')
    staged.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    staged.replace(path)


def write_csv(path, rows):
    staged = path.with_suffix(path.suffix + '.tmp')
    with staged.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    staged.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_points(path, count):
    import numpy as np

    points = np.loadtxt(path, ndmin=2)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError(f'{path}: expected {count} finite XYZ points, got {points.shape}')
    return points


def normalize(points):
    import numpy as np

    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.maximum(np.sqrt((centered ** 2).sum(axis=-1)).max(), 1e-12)
    return centered / radius


def distance_scores(prediction, ground_truth):
    from scipy.spatial import cKDTree

    # Match the project's float32 squared distances after exact nearest matching.
    forward_index = cKDTree(ground_truth).query(prediction, k=1)[1]
    backward_index = cKDTree(prediction).query(ground_truth, k=1)[1]
    forward = ((prediction - ground_truth[forward_index]) ** 2).sum(axis=-1)
    backward = ((ground_truth - prediction[backward_index]) ** 2).sum(axis=-1)
    forward_max, backward_max = forward.max(), backward.max()
    return dict(CD=float(forward.mean() + backward.mean()),
                HD_squared=float(max(forward_max, backward_max)),
                HD_legacy_sum=float(forward_max + backward_max),
                HD_pred_to_gt_sq=float(forward_max), HD_gt_to_pred_sq=float(backward_max))


def evaluate_predictions(predictions, data_root, destination, expected_count=127):
    import numpy as np
    import open3d as o3d

    test = data_root / 'PU1K/PU1K/test'
    inputs = test / 'input_2048/input_2048'
    ground_truth = test / 'input_2048/gt_8192'
    meshes = test / 'original_meshes'
    names = {path.stem for path in ground_truth.glob('*.xyz')}
    actual = {path.stem for path in predictions.glob('*.xyz')
              if not path.stem.endswith('_point2mesh_distance')}
    if len(names) != expected_count or actual != names:
        raise ValueError(f'{predictions}: expected {expected_count} matched predictions; '
                         f'missing={sorted(names - actual)}, extra={sorted(actual - names)}')
    pugan = {path.stem for path in (data_root / 'PUGAN/PU-GAN/test').glob('*.off')}
    expected_pugan = 27 if expected_count == 127 else len(pugan)
    if len(pugan) != expected_pugan or not pugan <= names:
        raise ValueError('PU-GAN test mesh names must identify the 27-object PU1K subset')
    destination.mkdir(parents=True, exist_ok=True)
    rows, distances_by_shape = [], {}
    for index, name in enumerate(sorted(names), 1):
        pred = load_points(predictions / (name + '.xyz'), 8192)
        gt = load_points(ground_truth / (name + '.xyz'), 8192)
        input_points = load_points(inputs / (name + '.xyz'), 2048).astype(np.float32)
        row = dict(name=name, subset='PU-GAN' if name in pugan else 'ShapeNet', points=len(pred))
        official = distance_scores(normalize(pred).astype(np.float32), normalize(gt).astype(np.float32))
        center = input_points.mean(axis=0, keepdims=True)
        radius = np.maximum(np.sqrt(((input_points - center) ** 2).sum(axis=-1)).max(), 1e-12)
        shared = distance_scores((pred.astype(np.float32) - center) / radius,
                                 (gt.astype(np.float32) - center) / radius)
        for protocol, values in [('official', official), ('shared_input', shared)]:
            row.update({protocol + '_' + key: value for key, value in values.items()})

        mesh_path = meshes / (name + '.off')
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.has_triangles():
            raise ValueError(f'No triangles in {mesh_path}')
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        distances = scene.compute_distance(o3d.core.Tensor(pred.astype(np.float32))).numpy().astype(np.float64)
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError(f'Invalid P2F distances for {name}')
        distances_by_shape[name] = distances
        row.update(P2F_mean=float(distances.mean()), P2F_std=float(distances.std()),
                   P2F_P99=float(np.percentile(distances, 99)), P2F_max=float(distances.max()))
        np.savetxt(predictions / (name + '_point2mesh_distance.xyz'),
                   np.column_stack((pred, distances)), fmt='%.10g')
        rows.append(row)
        print(f'Evaluate {index}/{len(names)} {name}: CD={official["CD"] * 1000:.6f} '
              f'HD_squared={official["HD_squared"] * 1000:.6f} '
              f'P2F={row["P2F_mean"] * 1000:.6f} (x1000)', flush=True)

    groups = {}
    for group in ('PU1K', 'PU-GAN', 'ShapeNet'):
        selected = [row for row in rows if group == 'PU1K' or row['subset'] == group]
        if not selected:
            continue
        distances = np.concatenate([distances_by_shape[row['name']] for row in selected])
        scores = dict(shapes=len(selected), points=len(distances))
        for protocol in ('official', 'shared_input'):
            scores[protocol] = {key: float(np.mean([row[protocol + '_' + key] for row in selected]))
                                for key in DISTANCE_KEYS}
        scores['p2f'] = dict(P2F_mean=float(distances.mean()), P2F_std=float(distances.std()),
                             P2F_P99=float(np.percentile(distances, 99)),
                             shape_mean_P2F_P99=float(np.mean([row['P2F_P99'] for row in selected])))
        groups[group] = scores
    write_csv(destination / 'evaluation.csv', rows)
    write_json(destination / 'metrics.json', dict(
        groups=groups, p2f_backend=f'Open3D {o3d.__version__} CPU RaycastingScene',
        nearest_neighbor_backend='SciPy cKDTree exact matching, float32 squared differences',
        prediction_dir=str(predictions), data_root=str(data_root)))
    return groups


def training_info(logfile):
    best_epoch, validation = None, []
    with logfile.open(encoding='utf-8', errors='replace') as stream:
        command = stream.readline().strip()
        for line in stream:
            match = re.search(r'Best checkpoint updated at epoch (\d+)', line)
            if match:
                best_epoch = int(match.group(1))
            match = re.search(r'Validation P2P loss: ([\d.eE+-]+)', line)
            if match:
                validation.append(float(match.group(1)))
    warnings = []
    if len(validation) > 1 and len(set(validation)) == 1:
        warnings.append(f'All {len(validation)} logged Grad-PU validation losses equal '
                        f'{validation[0]}; best checkpoint is from epoch {best_epoch}.')
    return dict(command=command, best_epoch=best_epoch, warnings=warnings)


def model_jobs(args):
    jobs = []
    for label in args.only or MODELS:
        source = 'PU-Gaussian' if label.startswith('PU-Gaussian') else label
        run = args.gaussian_100_run if label == 'PU-Gaussian-100-100' else args.queue_run
        folder = run.resolve() / source
        checkpoints = list(folder.rglob('best.pth' if source == 'PU-Gaussian' else 'ckpt-best.pth'))
        if len(checkpoints) != 1 or checkpoints[0].stat().st_size == 0:
            raise ValueError(f'{folder}: expected one nonempty best checkpoint, got {checkpoints}')
        info = training_info(folder / 'train.log')
        if source == 'PU-Gaussian':
            tokens = shlex.split(info['command'])
            stage1 = tokens[tokens.index('--stage_1_max_epochs') + 1]
            epochs = tokens[tokens.index('--max_epochs') + 1]
            expected = '100' if label == 'PU-Gaussian-100-100' else '1'
            if stage1 != expected or epochs != '200':
                raise ValueError(f'{label}: training log has {stage1} stage-one epochs, {epochs} total')
        jobs.append(dict(label=label, source=source, checkpoint=str(checkpoints[0]),
                         checkpoint_sha256=sha256(checkpoints[0]), training=info))
    return jobs


def inference_command(job, data_root, output, seed):
    source = job['source']
    project = COMPARE / (source + '-main')
    input_dir = data_root / 'PU1K/PU1K/test/input_2048/input_2048'
    gt_dir = input_dir.parent / 'gt_8192'
    arguments = ['--dataset', 'pu1k']
    if source == 'PU-Gaussian':
        arguments += ['--test_input_path', str(input_dir), '--test_gt_path', str(gt_dir),
                      '--ckpt', job['checkpoint'], '--save_dir', str(output),
                      '--r', '4', '--up_rate', '4', '--num_points', '256', '--patch_rate', '3',
                      '--num_samples', '6', '--training_stage', '2']
    elif source == 'Grad-PU':
        arguments += ['--test_input_path', str(input_dir), '--ckpt_path', job['checkpoint'],
                      '--save_dir', str(output / 'xyz'), '--up_rate', '4', '--truncate_distance', 'True']
    else:
        arguments += ['--input_dir', str(input_dir), '--gt_dir', str(gt_dir),
                      '--ckpt', job['checkpoint'], '--save_dir', str(output), '--r', '4']
    return [sys.executable, '-B', '-u', '-c', SEEDED_ENTRY, str(seed), str(project / 'test.py'),
            *arguments], project


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=COMPARE.parent / 'data')
    parser.add_argument('--queue-run', type=Path,
                        default=COMPARE / 'training-runs/pu1k-train-seed42-20261121-144843-518795')
    parser.add_argument('--gaussian-100-run', type=Path,
                        default=COMPARE / 'training-runs/pu1k-train-seed42-20261122-182333-902447')
    parser.add_argument('--output-dir', type=Path, help='New result directory; must not already exist')
    parser.add_argument('--only', choices=MODELS, nargs='+', help='Evaluate these configurations only')
    parser.add_argument('--seed', type=int, default=42, help='Inference RNG seed')
    parser.add_argument('--check', action='store_true', help='Check data, weights and runtime, then exit')
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error('--seed must be in [0, 2**32)')
    if args.only and len(args.only) != len(set(args.only)):
        parser.error('--only contains duplicate configurations')
    return args


def main():
    args = parse_args()
    packages = (COMPARE / '.pugaussian-runtime/python').resolve()
    if not packages.is_dir():
        raise FileNotFoundError(f'Missing {packages}; use the existing server comparison runtime')
    sys.path.insert(0, str(packages))
    os.environ.update(PYTHONPATH=str(packages), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                      CUDA_VISIBLE_DEVICES='0', MPLBACKEND='Agg', PYTHONHASHSEED=str(args.seed))
    import numpy as np
    import open3d as o3d
    import scipy
    import torch
    import pointops_cuda
    import chamfer_3D
    from train_baselines import verify_cuda_sources

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment')
    for module in (pointops_cuda, chamfer_3D):
        if not Path(module.__file__).resolve().is_relative_to(packages):
            raise RuntimeError(f'{module.__name__} loaded outside the comparison runtime')
    verify_cuda_sources()
    data_root = args.data_root.resolve()
    test = data_root / 'PU1K/PU1K/test'
    names = {path.stem for path in (test / 'input_2048/input_2048').glob('*.xyz')}
    if len(names) != 127:
        raise ValueError(f'Expected 127 PU1K input clouds under {test}')
    if names != {path.stem for path in (test / 'input_2048/gt_8192').glob('*.xyz')}:
        raise ValueError('PU1K input and ground-truth names differ')
    if not names <= {path.stem for path in (test / 'original_meshes').glob('*.off')}:
        raise ValueError('Some original meshes are missing')
    pugan = {path.stem for path in (data_root / 'PUGAN/PU-GAN/test').glob('*.off')}
    if len(pugan) != 27 or not pugan <= names:
        raise ValueError('Expected the 27-object PU-GAN subset mesh directory')
    jobs = model_jobs(args)
    for job in jobs:
        print(f'{job["label"]}: {job["checkpoint"]}', flush=True)
        for warning in job['training']['warnings']:
            print('Training note: ' + warning, flush=True)
    if args.check:
        print('Preflight passed.', flush=True)
        return

    destination = args.output_dir or COMPARE / 'evaluation-runs' / (
        f'pu1k-seed{args.seed}-{datetime.now():%Y%m%d-%H%M%S-%f}')
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', seed=args.seed, data_root=str(data_root), models=jobs,
                    python=sys.executable, pytorch=torch.__version__, numpy=np.__version__,
                    scipy=scipy.__version__, open3d=o3d.__version__, gpu=torch.cuda.get_device_name(0),
                    script_sha256=sha256(Path(__file__)),
                    protocols=dict(
                        official='Prediction and GT independently normalized in float64, then float32',
                        shared_input='Prediction and GT use the input centroid and radius in float32',
                        CD='Sum of two mean squared nearest-neighbor distances',
                        HD_squared='Maximum of two directed maximum squared distances',
                        HD_legacy_sum='Sum of two directed maximum squared distances',
                        P2F='Unsigned Euclidean point-to-triangle distance in saved coordinates'),
                    input_sha256={name: sha256(test / 'input_2048/input_2048' / (name + '.xyz'))
                                  for name in sorted(names)})
    write_json(destination / 'run.json', manifest)
    print(f'Result directory: {destination}', flush=True)
    summary = []
    try:
        for job in jobs:
            output = destination / job['label']
            output.mkdir()
            command, project = inference_command(job, data_root, output, args.seed)
            job['status'] = 'inference'
            job['inference_command'] = command
            write_json(destination / 'run.json', manifest)
            logfile = output / 'inference.log'
            print(f'Starting {job["label"]}; detailed inference log: {logfile}', flush=True)
            with logfile.open('w', encoding='utf-8') as stream:
                with subprocess.Popen(command, cwd=project, stdout=stream, stderr=subprocess.STDOUT) as process:
                    last_count = -1
                    while True:
                        try:
                            returncode = process.wait(timeout=10)
                            break
                        except subprocess.TimeoutExpired:
                            count = len(list((output / 'xyz').glob('*.xyz')))
                            if count != last_count:
                                print(f'Inference {job["label"]}: {count}/127 shapes saved', flush=True)
                                last_count = count
            if returncode:
                raise RuntimeError(f'{job["label"]} inference failed with exit code {returncode}; see {logfile}')
            job['status'] = 'evaluation'
            write_json(destination / 'run.json', manifest)
            print(f'Starting unified CD/HD/P2F evaluation: {job["label"]}', flush=True)
            groups = evaluate_predictions(output / 'xyz', data_root, output)
            for group, scores in groups.items():
                row = dict(model=job['label'], subset=group, shapes=scores['shapes'], points=scores['points'])
                for protocol in ('official', 'shared_input'):
                    row.update({protocol + '_' + key + '_x1000': value * 1000
                                for key, value in scores[protocol].items()})
                row.update({key + '_x1000': value * 1000 for key, value in scores['p2f'].items()})
                summary.append(row)
            job['status'] = 'completed'
            write_csv(destination / 'summary.csv', summary)
            write_json(destination / 'summary.json', dict(metric_scale=1000, rows=summary))
            write_json(destination / 'run.json', manifest)
            print(f'Completed {job["label"]}: {json.dumps(groups["PU1K"])}', flush=True)
    except BaseException as error:
        job['status'] = 'failed'
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        write_json(destination / 'run.json', manifest)
        raise
    manifest['status'] = 'completed'
    write_json(destination / 'run.json', manifest)
    print(f'PU1K baseline evaluation completed: {destination / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
