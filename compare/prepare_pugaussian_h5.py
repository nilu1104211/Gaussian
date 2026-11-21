"""Add PU-Gaussian's dense targets to the prepared PU1K training split."""

import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'our' / 'PU-GCSNet'))
from prepare_data import read_off, sample_surface


def patch_count(path):
    with h5py.File(path, 'r') as file:
        count = file['poisson_256'].shape[0]
        for size in (256, 1024):
            if not count or file[f'poisson_{size}'].shape != (count, size, 3):
                raise ValueError(f'{path} needs aligned, nonempty poisson_256 and poisson_1024 arrays')
    return count


def subset_patches(source, destination, limit):
    with h5py.File(source, 'r') as original, h5py.File(destination, 'w') as subset:
        for size in (256, 1024):
            key = f'poisson_{size}'
            subset.create_dataset(key, data=original[key][:limit])


def prepare_gaussian_h5(train, data_root, output, seed, limit=None):
    manifest_path = data_root / 'prepared' / 'split.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))['splits']
    meshes = manifest['train']
    validation_meshes = {row['mesh'] for row in manifest['val']}
    total = 0
    for row in meshes:
        if row['mesh'] in validation_meshes or row['patch_start'] != total or row['patch_count'] <= 0:
            raise ValueError(f'Train and validation split mismatch in {manifest_path}')
        total += row['patch_count']
    if patch_count(train) != total:
        raise ValueError(f'Training H5 row count differs from {manifest_path}')
    count = min(limit, total) if limit is not None else total
    provenance = json.dumps(dict(
        training=str(train), training_mtime=train.stat().st_mtime_ns,
        manifest_mtime=manifest_path.stat().st_mtime_ns, seed=seed, rows=count, version=2))
    if output.is_file():
        with h5py.File(output, 'r') as cached:
            if cached.attrs.get('provenance') == provenance:
                print(f'Reusing {output}', flush=True)
                return

    output.parent.mkdir(parents=True, exist_ok=True)
    meshes = [row for row in meshes if row['patch_start'] < count]
    surface_points = 65536
    gib = 1024 ** 3
    free = shutil.disk_usage(output.parent).free / gib
    size = len(meshes) * surface_points * 12 + count * (4096 * 2 + 4)
    print(f'Preparing {count} PU-Gaussian patches ({size / gib:.2f} GiB before compression, '
          f'{free:.2f} GiB free)', flush=True)
    rng = np.random.default_rng(seed)
    with TemporaryDirectory(prefix='.prepare-pugaussian-', dir=output.parent) as work:
        staged = Path(work) / output.name
        with h5py.File(train, 'r') as original, h5py.File(staged, 'w', libver='latest') as file:
            for size in (256, 1024):
                key = f'poisson_{size}'
                if limit is None:
                    file[key] = h5py.ExternalLink(str(train), key)
                else:
                    file.create_dataset(key, data=original[key][:count])

            # Points shared by overlapping patches are stored once per mesh.
            surfaces = file.create_dataset('sampled_surfaces', (len(meshes), surface_points, 3), dtype='f4',
                                           chunks=(1, surface_points, 3), shuffle=True, compression='lzf')
            patch_indices = file.create_dataset('patch_surface_indices', (count, 4096), dtype='u2')
            patch_mesh = file.create_dataset('patch_mesh', (count,), dtype='i4')

            for index, row in enumerate(meshes, start=1):
                start = row['patch_start']
                stop = min(start + row['patch_count'], count)
                mesh = data_root / row['mesh']
                surface = sample_surface(*read_off(mesh), rng, point_count=surface_points)
                surfaces[index - 1] = surface
                patch_mesh[start:stop] = index - 1
                tree = cKDTree(surface)
                for offset, patch in enumerate(original['poisson_1024'][start:stop]):
                    center = patch.mean(axis=0)
                    radius = np.linalg.norm(patch - center, axis=1).max()
                    neighbors = tree.query_ball_point(center, float(radius))
                    if len(neighbors) < 4096:
                        neighbors = tree.query(center, k=4096)[1]
                    patch_indices[start + offset] = rng.choice(neighbors, size=4096, replace=False)
                if index % 50 == 0 or stop == count:
                    print(f'Prepared {stop}/{count} patches from {index} meshes', flush=True)
                if stop == count:
                    break
            file.attrs['provenance'] = provenance
        os.replace(staged, output)
