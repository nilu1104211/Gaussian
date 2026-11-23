"""One CD/HD/P2F implementation for both datasets and held-out validation.

CD: independently normalized bidirectional mean squared NN distance sum.
HD: independently normalized max of the directed maximum squared distances.
P2F: unsigned Euclidean point-to-triangle distance in SAVED coordinates.
This is an explicit public-code-based protocol, not a historical CGAL equivalence claim.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from retrain_common import sha256, write_json

EXPECTED = {'PU1K': 127, 'PU-GAN': 27}
SCALE = 1000.0


def read_points(path, count):
    points = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError(f'{path}: expected {count} finite XYZ points, got {points.shape}')
    return points


def independent_unit_sphere(points):
    points = np.asarray(points, dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=-1).max()
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError('Degenerate/nonfinite point cloud')
    return centered / radius


def point_scores(pred, gt):
    pred, gt = independent_unit_sphere(pred), independent_unit_sphere(gt)
    forward = ((pred - gt[cKDTree(gt).query(pred, k=1)[1]]) ** 2).sum(-1)
    backward = ((gt - pred[cKDTree(pred).query(gt, k=1)[1]]) ** 2).sum(-1)
    return dict(CD=float(forward.mean() + backward.mean()),
                HD=float(max(forward.max(), backward.max())),
                HD_pred_to_gt=float(forward.max()), HD_gt_to_pred=float(backward.max()))


def cd_hd_squared(pred, gt):
    scores = point_scores(pred, gt)
    return scores['CD'], scores['HD']


def mesh_scene(mesh_path):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError(f'Invalid triangle mesh: {mesh_path}')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def p2f_scores(pred, scene):
    import open3d as o3d
    distances = scene.compute_distance(o3d.core.Tensor(np.asarray(pred, dtype=np.float32))).numpy().astype(np.float64)
    if distances.shape != (len(pred),) or not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError('Invalid point-to-triangle distances')
    return dict(P2F=float(distances.mean()), P2F_P99=float(np.percentile(distances, 99)))


def cloud_names(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return {p.stem for p in folder.glob('*.xyz') if not p.stem.endswith('_point2mesh_distance')}


def evaluate(predictions, inputs, gt, meshes, expected):
    names = cloud_names(gt)
    if len(names) != expected or cloud_names(inputs) != names or cloud_names(predictions) != names:
        raise ValueError(f'Expected exactly {expected} matching input/GT/prediction names')
    rows = []
    for name in sorted(names):
        read_points(Path(inputs) / f'{name}.xyz', 2048)
        pred = read_points(Path(predictions) / f'{name}.xyz', 8192)
        target = read_points(Path(gt) / f'{name}.xyz', 8192)
        scores = point_scores(pred, target)
        scores.update(p2f_scores(pred, mesh_scene(Path(meshes) / f'{name}.off')))
        rows.append(dict(shape=name, **scores))
    summary = {key: float(np.mean([r[key] for r in rows], dtype=np.float64)) for key in rows[0] if key != 'shape'}
    return rows, summary


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    import scipy
    import open3d as o3d
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {args.output_dir}')
    labels = [name for name, _ in args.prediction]
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate model labels')
    expected = args.expected_shapes or EXPECTED[args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, hashes = [], [], {}
    write_json(args.output_dir / 'protocol.json', dict(status='running'))
    try:
        for model, folder in args.prediction:
            rows, scores = evaluate(folder, args.input_dir, args.gt_dir, args.mesh_dir, expected)
            all_rows.extend(dict(model=model, shape=r['shape'], **{k + '_x1000': v * SCALE for k, v in r.items() if k != 'shape'}) for r in rows)
            summaries.append(dict(model=model, dataset=args.dataset, n_shapes=len(rows),
                                  **{k + '_x1000': v * SCALE for k, v in scores.items()}))
            hashes[model] = {p.name: sha256(p) for p in sorted(Path(folder).glob('*.xyz')) if not p.stem.endswith('_point2mesh_distance')}
        write_csv(args.output_dir / 'per_shape.csv', all_rows)
        write_csv(args.output_dir / 'summary.csv', summaries)
        common = {}
        for key, folder, suffix in [('inputs', args.input_dir, '.xyz'), ('gt', args.gt_dir, '.xyz'), ('meshes', args.mesh_dir, '.off')]:
            common[key] = {name + suffix: sha256(Path(folder) / (name + suffix)) for name in sorted(cloud_names(args.gt_dir))}
        write_json(args.output_dir / 'protocol.json', dict(status='completed', dataset=args.dataset,
            expected_shapes=expected, normalization='independent centroid/max radius in float32',
            CD='bidirectional mean squared NN sum', HD='maximum squared NN over both directions',
            P2F='saved-coordinate unsigned Euclidean point-to-triangle mean', scale_factor=SCALE,
            numpy=np.__version__, scipy=scipy.__version__, open3d=o3d.__version__,
            script_sha256=sha256(__file__), common_sha256=sha256(Path(__file__).with_name('retrain_common.py')),
            source_hashes=common, prediction_hashes=hashes,
            prediction_dirs={name: str(folder) for name, folder in args.prediction}))
    except BaseException as error:
        write_json(args.output_dir / 'protocol.json', dict(status='failed', error=str(error)))
        raise
    print(f'Saved {args.output_dir / "summary.csv"}', flush=True)


def parse_prediction(value):
    label, separator, folder = value.partition('=')
    if not separator or not label.strip() or not folder.strip():
        raise argparse.ArgumentTypeError('Expected MODEL=/prediction/xyz/directory')
    return label.strip(), Path(folder).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=EXPECTED, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--gt-dir', type=Path, required=True)
    parser.add_argument('--mesh-dir', type=Path, required=True)
    parser.add_argument('--prediction', type=parse_prediction, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-shapes', type=int, help='Explicit count for validation or smoke data')
    args = parser.parse_args()
    if args.expected_shapes is not None and args.expected_shapes < 1:
        parser.error('--expected-shapes must be positive')
    run(args)


if __name__ == '__main__':
    main()
