"""Build a shape-level train/validation split from the official training meshes."""

import argparse
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np


INPUT_POINTS = 256
TARGET_POINTS = 1024
SURFACE_POINTS = 8192


def off_records(stream):
    for line in stream:
        fields = line.partition(b'#')[0].split()
        if fields:
            yield fields


def read_faces(stream, face_count):
    start = stream.tell()
    try:
        rows = np.loadtxt(stream, dtype=np.int32, max_rows=face_count,
                          usecols=(0, 1, 2, 3), ndmin=2)
        if rows.shape[0] == face_count and np.all(rows[:, 0] == 3):
            return np.ascontiguousarray(rows[:, 1:])
    except ValueError:
        pass

    stream.seek(start)
    records = off_records(stream)
    triangles = []
    for _ in range(face_count):
        fields = next(records, [])
        if not fields or len(fields) < 1 + int(fields[0]):
            raise ValueError('Invalid or incomplete OFF face record')
        indices = list(map(int, fields[1:1 + int(fields[0])]))
        triangles.extend((indices[0], a, b) for a, b in zip(indices[1:], indices[2:]))
    return np.asarray(triangles, dtype=np.int32).reshape(-1, 3)


def read_off(path):
    with path.open('rb') as stream:
        records = off_records(stream)
        header = next(records, [])
        if not header or header[0] not in (b'OFF', b'COFF'):
            raise ValueError('Expected an OFF or COFF mesh')
        counts = header[1:] or next(records, [])
        if len(counts) < 2:
            raise ValueError('Missing OFF vertex and face counts')
        vertex_count, face_count = map(int, counts[:2])
        if vertex_count < 3 or face_count < 1:
            raise ValueError('OFF mesh has no surface')
        vertices = np.loadtxt(stream, dtype=np.float32, max_rows=vertex_count,
                              usecols=(0, 1, 2), ndmin=2)
        faces = read_faces(stream, face_count)
    if len(vertices) != vertex_count or not np.isfinite(vertices).all():
        raise ValueError('Invalid OFF vertices')
    if not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Missing OFF triangles or an out-of-range vertex index')
    return vertices, faces


def sample_surface(vertices, faces, rng):
    # Limit memory when drawing from PU-GAN meshes with millions of faces.
    areas = np.empty(len(faces), dtype=np.float64)
    chunk_size = 262144
    for start in range(0, len(faces), chunk_size):
        end = start + chunk_size
        a, b, c = vertices[faces[start:end]].transpose(1, 0, 2)
        areas[start:end] = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    np.cumsum(areas, out=areas)
    if not np.isfinite(areas[-1]) or areas[-1] <= 0:
        raise ValueError('No finite, positive surface area')
    indices = areas.searchsorted(rng.random(SURFACE_POINTS) * areas[-1], side='right')
    a, b, c = vertices[faces[indices]].transpose(1, 0, 2)
    u, v = rng.random((2, SURFACE_POINTS, 1), dtype=np.float32)
    root_u = np.sqrt(u)
    return (1 - root_u) * a + root_u * (1 - v) * b + root_u * v * c


def make_patches(points, count, rng):
    sparse = np.empty((count, INPUT_POINTS, 3), dtype=np.float32)
    dense = np.empty((count, TARGET_POINTS, 3), dtype=np.float32)
    center = rng.integers(len(points))
    nearest_center = np.full(len(points), np.inf, dtype=np.float32)
    for i in range(count):
        delta = points - points[center]
        distance = np.einsum('ij,ij->i', delta, delta)
        neighbors = np.argpartition(distance, TARGET_POINTS - 1)[:TARGET_POINTS]
        dense[i] = points[neighbors]
        sparse[i] = dense[i, rng.choice(TARGET_POINTS, INPUT_POINTS, replace=False)]
        np.minimum(nearest_center, distance, out=nearest_center)
        nearest_center[center] = -1
        center = nearest_center.argmax()
    return sparse, dense


def assign_splits(data_root, seed):
    shapenet = sorted((data_root / 'PU1K/PU1K_raw_meshes/train/train_meshes').glob('*.off'))
    pugan_root = data_root / 'PUGAN/PU-GAN'
    pugan = sorted(path for group in ('simple', 'medium', 'complex')
                   for path in (pugan_root / group).glob('*.off'))
    rng = np.random.default_rng(seed)
    splits = {'train': [], 'val': []}
    for source, meshes, expected, patches in (
        ('PU1K-ShapeNet', shapenet, 900, 50),
        ('PU-GAN', pugan, 120, 200),
    ):
        if len(meshes) != expected:
            raise ValueError(f'{source}: expected {expected} official training meshes, found {len(meshes)} under {data_root}')
        held_out = set(rng.choice(len(meshes), len(meshes) // 10, replace=False))
        for i, path in enumerate(meshes):
            split = 'val' if i in held_out else 'train'
            splits[split].append((source, path.relative_to(data_root).as_posix(), patches))
    return splits


def write_split(path, meshes, data_root, seed):
    total = sum(count for _, _, count in meshes)
    rng = np.random.default_rng(seed)
    manifest = []
    start = 0
    with h5py.File(path, 'w') as file:
        sparse = file.create_dataset('poisson_256', (total, INPUT_POINTS, 3), dtype='f4')
        dense = file.create_dataset('poisson_1024', (total, TARGET_POINTS, 3), dtype='f4')
        file.attrs['sampling_method'] = 'area-weighted OFF sampling, FPS centers, kNN patches'
        for i, (source, relative, count) in enumerate(meshes, start=1):
            try:
                points = sample_surface(*read_off(data_root / relative), rng)
                inputs, targets = make_patches(points, count, rng)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f'Failed to prepare {relative}: {exc}') from exc
            end = start + count
            sparse[start:end] = inputs
            dense[start:end] = targets
            manifest.append(dict(source=source, mesh=relative, patch_start=start, patch_count=count))
            start = end
            if i % 50 == 0 or i == len(meshes):
                print(f'{path.stem}: {i}/{len(meshes)} meshes, {start}/{total} patches', flush=True)
    return manifest


def prepare(data_root, output, seed):
    splits = assign_splits(data_root, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f'Output must be a new or empty directory: {output}')
        output.rmdir()

    # Publish both H5 files together, so failed runs leave no partial dataset.
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        manifest = dict(seed=seed, validation_fraction=0.1, splits={})
        for name, meshes in splits.items():
            manifest['splits'][name] = write_split(staged / f'{name}.h5', meshes, data_root, seed)
        (staged / 'split.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Directory containing PU1K and PUGAN')
    parser.add_argument('--output-dir', type=Path, required=True, help='Destination for train.h5, val.h5 and split.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    prepare(args.data_root.resolve(), args.output_dir.resolve(), args.seed)
    print(f'Prepared train.h5, val.h5 and split.json in {args.output_dir}')


if __name__ == '__main__':
    main()
