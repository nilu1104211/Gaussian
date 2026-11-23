"""Build versioned, shape-disjoint Poisson/FPS data for both benchmarks."""
import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from retrain_common import DATA_VERSION, ROOT, read_json, sha256, stable_seed, write_json

SIZES = (256, 1024, 1536, 5120)


def clean_test_overlap(assigned, tests, seed):
    """Remove test copies from the training pool, then retain the original val size."""
    test_contents = defaultdict(list)
    for row in tests:
        test_contents[row['mesh_sha256']].append(row['mesh'])
    cleaned = {part: [] for part in assigned}
    audit = dict(excluded_test_copies=[], validation_replacements=[], training_duplicates=[])
    for part, rows in assigned.items():
        for row in rows:
            if row['mesh_sha256'] in test_contents:
                audit['excluded_test_copies'].append(dict(row, original_split=part,
                    test_meshes=test_contents[row['mesh_sha256']]))
            else:
                cleaned[part].append(dict(row))
    sources = sorted({row['source'] for rows in assigned.values() for row in rows})
    for source in sources:
        target = sum(row['source'] == source for row in assigned['val'])
        missing = target - sum(row['source'] == source for row in cleaned['val'])
        candidates = sorted((row for row in cleaned['train'] if row['source'] == source),
                            key=lambda row: stable_seed(seed, 'validation-replacement', row['mesh']))
        # Keep duplicate official training entries together; never split their contents.
        train_counts = defaultdict(int)
        for row in cleaned['train']:
            train_counts[row['mesh_sha256']] += 1
        for row in candidates:
            if missing == 0:
                break
            if train_counts[row['mesh_sha256']] != 1:
                continue
            cleaned['train'].remove(row)
            cleaned['val'].append(row)
            audit['validation_replacements'].append(row['mesh'])
            missing -= 1
        if missing:
            raise ValueError(f'Cannot preserve the validation size for {source}')
    for part, rows in cleaned.items():
        contents = defaultdict(list)
        for row in rows:
            contents[row['mesh_sha256']].append(row['mesh'])
        audit['training_duplicates'].extend(dict(split=part, meshes=meshes)
            for meshes in contents.values() if len(meshes) > 1)
    check_split_identity(cleaned['train'], cleaned['val'], tests)
    return cleaned, audit


def mesh_split(data_root, seed, return_audit=False):
    # Reuse the existing shape split algorithm exactly, including the PU-GAN subset.
    sys.path.insert(0, str(ROOT / 'our/PU-GCSNet'))
    from prepare_data import assign_splits
    data_root = Path(data_root)
    assigned = assign_splits(data_root, seed)
    rows = {part: [dict(source=source, mesh=mesh, patches=count, mesh_sha256=sha256(data_root / mesh))
                   for source, mesh, count in entries] for part, entries in assigned.items()}
    test_paths = [*(data_root / 'PU1K/PU1K/test/original_meshes').glob('*.off'),
                  *(data_root / 'PUGAN/PU-GAN/test').glob('*.off')]
    tests = [dict(mesh=p.relative_to(data_root).as_posix(), mesh_sha256=sha256(p)) for p in test_paths]
    cleaned, audit = clean_test_overlap(rows, tests, seed)
    return (cleaned, audit) if return_audit else cleaned


def numpy_fps(points, count):
    distance = np.full(len(points), np.inf, dtype=np.float32)
    result = np.empty(count, dtype=np.int64)
    farthest = 0
    for i in range(count):
        result[i] = farthest
        np.minimum(distance, ((points - points[farthest]) ** 2).sum(-1), out=distance)
        farthest = int(distance.argmax())
    return result


def fps_batches(points, count, device):
    if device == 'cpu':
        return np.stack([numpy_fps(p, count) for p in points])
    import torch
    sys.path.insert(0, str(ROOT / 'compare/PU-Gaussian-main'))
    from pointops.functions import pointops
    with torch.no_grad():
        tensor = torch.as_tensor(points, dtype=torch.float32, device='cuda').contiguous()
        indices = pointops.furthestsampling(tensor, count).long().cpu().numpy()
    return indices


def sample_mesh(path, count, seed):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles():
        raise ValueError(f'Empty mesh: {path}')
    o3d.utility.random.seed(int(seed))
    points = np.asarray(mesh.sample_points_poisson_disk(count).points, dtype=np.float32)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError(f'Invalid Poisson samples: {path}')
    return points


def make_patches(surface, count, device):
    centers = fps_batches(surface[None], count, device)[0]
    indices = cKDTree(surface).query(surface[centers], k=5120)[1]
    dense = np.ascontiguousarray(surface[indices], dtype=np.float32)
    # One nested FPS trajectory produces all resolutions on precisely one support.
    order = fps_batches(dense, 1536, device)
    batches = np.arange(count)[:, None]
    result = {size: dense[batches, order[:, :size]] for size in (256, 1024, 1536)}
    result[5120] = dense
    for size, points in result.items():
        if points.shape != (count, size, 3) or not np.isfinite(points).all():
            raise ValueError('Invalid multi-density patch')
    return result


def write_training(path, rows, data_root, seed, surface_points, device):
    total = sum(row['patches'] for row in rows)
    if not total:
        raise ValueError('Empty training split')
    staged = path.with_suffix('.h5.tmp')
    manifest, offset = [], 0
    with h5py.File(staged, 'w') as file:
        arrays = {size: file.create_dataset(f'poisson_{size}', (total, size, 3), dtype='f4',
                    chunks=(1, size, 3), compression='lzf', shuffle=True) for size in SIZES}
        for row in rows:
            mesh = Path(data_root) / row['mesh']
            shape_seed = stable_seed(seed, row['mesh'], 'train-surface-v1')
            surface = sample_mesh(mesh, surface_points, shape_seed)
            patches = make_patches(surface, row['patches'], device)
            end = offset + row['patches']
            for size in SIZES:
                arrays[size][offset:end] = patches[size]
            manifest.append(dict(row, patch_start=offset, patch_count=row['patches'],
                                 mesh_sha256=sha256(mesh), sampling_seed=shape_seed))
            offset = end
            print(f'{path.parent.name}/{path.name}: {offset}/{total} patches, {mesh.name}', flush=True)
        file.attrs['data_version'] = DATA_VERSION
        file.attrs['sampling'] = 'Open3D Poisson disk; FPS centers; 5120-NN; nested FPS 256/1024/1536'
        file.attrs['coordinates'] = 'original mesh coordinates, normalize jointly in loader'
    staged.replace(path)
    return manifest


def build_clouds(destination, rows, data_root, seed, limit=None):
    rows = rows[:limit] if limit else rows
    if len({Path(row['mesh']).stem for row in rows}) != len(rows):
        raise ValueError(f'Duplicate object names in {destination}')
    for name in ('input_2048', 'gt_8192', 'meshes'):
        (destination / name).mkdir(parents=True, exist_ok=True)
    entries = []
    for row in rows:
        mesh = Path(data_root) / row['mesh']
        shape = mesh.stem
        for size, folder in [(2048, 'input_2048'), (8192, 'gt_8192')]:
            points = sample_mesh(mesh, size, stable_seed(seed, row['mesh'], 'whole-object', size))
            np.savetxt(destination / folder / f'{shape}.xyz', points, fmt='%.10g')
        shutil.copyfile(mesh, destination / 'meshes' / mesh.name)
        entries.append(dict(shape=shape, mesh=row['mesh'], mesh_sha256=sha256(mesh)))
    return entries


def copy_pu1k_test(destination, data_root, limit=None):
    source = Path(data_root) / 'PU1K/PU1K/test'
    names = sorted(p.stem for p in (source / 'input_2048/gt_8192').glob('*.xyz'))
    if len(names) != 127:
        raise ValueError('Expected the original 127-object PU1K test set')
    names = names[:limit] if limit else names
    for folder in ('input_2048', 'gt_8192', 'meshes'):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    entries = []
    for name in names:
        for folder in ('input_2048', 'gt_8192'):
            shutil.copyfile(source / 'input_2048' / folder / f'{name}.xyz', destination / folder / f'{name}.xyz')
        mesh = source / 'original_meshes' / f'{name}.off'
        shutil.copyfile(mesh, destination / 'meshes' / f'{name}.off')
        entries.append(dict(shape=name, mesh=mesh.relative_to(data_root).as_posix(), mesh_sha256=sha256(mesh)))
    return entries


def check_split_identity(train, validation, test):
    # Official PU-GAN contains different train/test meshes both named "star".
    # File stems identify outputs within a split, not object identity across splits.
    groups = [train, validation, test]
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            if {row['mesh'] for row in left} & {row['mesh'] for row in right}:
                raise ValueError('Source mesh path appears in multiple splits')
            if {row['mesh_sha256'] for row in left} & {row['mesh_sha256'] for row in right}:
                raise ValueError('Identical mesh content appears in multiple splits')


def prepare(data_root, output, seed=42, device='cuda', smoke=False, batch_size=16):
    output = Path(output).resolve()
    settings = dict(version=DATA_VERSION, seed=seed, surface_points=40960, smoke=smoke,
                    smoke_patches_per_mesh=batch_size if smoke else None, fps_device=device,
                    data_root=str(Path(data_root).resolve()), builder_sha256=sha256(__file__))
    marker = output / 'manifest.json'
    if marker.is_file():
        previous = read_json(marker)
        if previous.get('settings') != settings or previous.get('status') != 'completed':
            raise ValueError(f'{output}: different/incomplete data version; select a NEW --prepared-root')
        for name, digest in previous['file_sha256'].items():
            if sha256(output / name) != digest:
                raise ValueError(f'Data file changed: {name}')
        return previous
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Incomplete or unrelated data directory: {output}; use a new directory')
    output.mkdir(parents=True, exist_ok=True)
    assigned, split_audit = mesh_split(data_root, seed, return_audit=True)
    result = dict(status='building', settings=settings, split_audit=split_audit, datasets={})
    write_json(marker, result)
    try:
        for dataset in ('PU1K', 'PU-GAN'):
            folder = output / dataset
            folder.mkdir()
            subsets = {part: [dict(row) for row in rows if dataset == 'PU1K' or row['source'] == 'PU-GAN']
                       for part, rows in assigned.items()}
            if smoke:
                for part in subsets:
                    if dataset == 'PU1K':
                        subsets[part] = [next(r for r in subsets[part] if r['source'] == source)
                                         for source in ('PU1K-ShapeNet', 'PU-GAN')]
                    else:
                        subsets[part] = subsets[part][:2]
                    for row in subsets[part]:
                        row['patches'] = batch_size
            train_rows, val_rows = subsets['train'], subsets['val']
            if {r['mesh'] for r in train_rows} & {r['mesh'] for r in val_rows}:
                raise ValueError('Train/validation shape overlap')
            split = dict(train=train_rows, val=val_rows)
            split['train_patches'] = write_training(folder / 'train.h5', train_rows, data_root,
                                                    seed, 40960, device)
            # Patch validation is retained for diagnosis; best selection uses whole objects.
            split['val_patches'] = write_training(folder / 'val.h5', val_rows, data_root,
                                                  seed, 40960, device)
            split['validation_objects'] = build_clouds(folder / 'validation', val_rows, data_root, seed)
            if dataset == 'PU1K':
                split['test_objects'] = copy_pu1k_test(folder / 'test', data_root, 2 if smoke else None)
            else:
                tests = sorted((Path(data_root) / 'PUGAN/PU-GAN/test').glob('*.off'))
                if len(tests) != 27:
                    raise ValueError('Expected 27 PU-GAN test meshes')
                test_rows = [dict(mesh=p.relative_to(data_root).as_posix()) for p in tests]
                split['test_objects'] = build_clouds(folder / 'test', test_rows, data_root, seed, 2 if smoke else None)
            check_split_identity(split['train_patches'], split['val_patches'], split['test_objects'])
            test_names = {row['shape'] for row in split['test_objects']}
            val_names = {Path(row['mesh']).stem for row in val_rows}
            split['identity_check'] = 'disjoint source paths and SHA256 mesh content; stems may coincide across splits'
            for part, expected_names in [('validation', val_names), ('test', test_names)]:
                for directory in ('input_2048', 'gt_8192'):
                    names = {p.stem for p in (folder / part / directory).glob('*.xyz')}
                    if names != expected_names:
                        raise ValueError(f'{dataset}/{part}: incomplete whole-object data')
            write_json(folder / 'split.json', split)
            result['datasets'][dataset] = dict(train_shapes=len(train_rows), validation_shapes=len(val_rows),
                train_patches=sum(r['patches'] for r in train_rows), validation_patches=sum(r['patches'] for r in val_rows),
                test_shapes=len(split['test_objects']))
        result['file_sha256'] = {p.relative_to(output).as_posix(): sha256(p)
                                for p in sorted(output.rglob('*')) if p.is_file() and p != marker}
        result['status'] = 'completed'
        write_json(marker, result)
    except BaseException as error:
        result.update(status='failed', error=str(error))
        write_json(marker, result)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    prepare(args.data_root.resolve(), args.output, args.seed, args.device, args.smoke, args.batch_size)


if __name__ == '__main__':
    main()
