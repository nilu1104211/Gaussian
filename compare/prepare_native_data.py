"""Use publisher H5s directly; build only Gaussian supervision and shared tests."""
import argparse
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

from native_training_protocol import FOLDERS, TEST_COUNTS, VERSION, find_directory, find_h5
from retrain_common import ROOT, read_json, sha256, stable_seed, write_json

SIZES = (256, 1024, 1536, 5120)


def h5_info(path, gaussian=False, scan=True):
    path = Path(path).resolve()
    sizes = SIZES if gaussian else (256, 1024)
    with h5py.File(path, 'r') as file:
        counts = set()
        shapes = {}
        for size in sizes:
            key = f'poisson_{size}'
            if key not in file:
                raise ValueError(f'{path}: missing {key}')
            array = file[key]
            if array.ndim != 3 or array.shape[1:] != (size, 3) or array.dtype.kind != 'f':
                raise ValueError(f'{path}: invalid {key}: {array.shape}/{array.dtype}')
            counts.add(len(array))
            shapes[key] = list(array.shape)
            if scan:
                for start in range(0, len(array), 128):
                    if not np.isfinite(array[start:start+128]).all():
                        raise ValueError(f'{path}: nonfinite {key} near patch {start}')
        if len(counts) != 1 or not next(iter(counts)):
            raise ValueError(f'{path}: empty or unpaired H5 fields')
    return dict(training_h5=str(path), training_h5_sha256=sha256(path),
                train_patches=next(iter(counts)), h5_shapes=shapes,
                origin='author_style_generated_x20' if gaussian else 'publisher_h5_unmodified')


def link_or_copy(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError(f'Existing fixed test file differs: {target}')
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def test_files(folder, expected):
    folder = Path(folder)
    names = sorted(p.stem for p in (folder/'input_2048').glob('*.xyz'))
    if len(names) != expected:
        raise ValueError(f'{folder}: expected {expected} test inputs, found {len(names)}')
    files = {}
    for sub, suffix, count in [('input_2048', '.xyz', 2048), ('gt_8192', '.xyz', 8192), ('meshes', '.off', None)]:
        actual = sorted(p.stem for p in (folder/sub).glob('*'+suffix))
        if actual != names:
            raise ValueError(f'{folder/sub}: test identities differ')
        for name in names:
            p = folder/sub/(name+suffix)
            if count is not None:
                values = np.loadtxt(p, dtype=np.float32, ndmin=2)
                if values.shape != (count, 3) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid test cloud {p}')
            files[f'{sub}/{name+suffix}'] = sha256(p)
    return files


def pugan_mesh_root(data_root):
    return find_directory(data_root, ['PUGAN/PU-GAN', 'PU-GAN', 'PUGAN'])


def raw_meshes(data_root, dataset):
    if dataset == 'pu1k':
        folder = find_directory(data_root, ['PU1K/PU1K_raw_meshes/train/train_meshes',
                                           'PU1K_raw_meshes/train/train_meshes'])
        paths = sorted(folder.glob('*.off'))
        expected = 900
    else:
        folder = pugan_mesh_root(data_root)
        paths = sorted(p for group in ('simple', 'medium', 'complex', 'train')
                       for p in (folder/group).glob('*.off'))
        expected = 120
    if len(paths) != expected:
        raise ValueError(f'{folder}: expected {expected} native training meshes, found {len(paths)}')
    return paths


def prepare_test(data_root, cache, dataset, seed, reuse_root=None):
    target = cache/FOLDERS[dataset]/'test'
    marker = target/'manifest.json'
    if marker.is_file():
        info = read_json(marker)
        if info['seed'] != seed or test_files(target, TEST_COUNTS[dataset]) != info['test_sha256']:
            raise ValueError(f'Fixed test data changed: {target}')
        return info
    source = None
    if reuse_root is not None:
        candidate = Path(reuse_root)/FOLDERS[dataset]/'test'
        if not candidate.is_dir():
            raise FileNotFoundError(f'Missing requested fixed tests: {candidate}')
        source = candidate
    elif dataset == 'pugan':
        candidate = Path(data_root)/f'repro_20261123_seed{seed}'/FOLDERS[dataset]/'test'
        if candidate.is_dir():
            source = candidate
    if source is not None:
        hashes = test_files(source, TEST_COUNTS[dataset])
        for relative in hashes:
            link_or_copy(source/relative, target/relative)
        origin = str(source.resolve())
    elif dataset == 'pu1k':
        source = find_directory(data_root, ['PU1K/PU1K/test', 'PU1K/test'])
        names = sorted(p.stem for p in (source/'input_2048/gt_8192').glob('*.xyz'))
        if len(names) != TEST_COUNTS[dataset]:
            raise ValueError(f'Missing 127 publisher PU1K test pairs: {source}')
        for name in names:
            for sub in ('input_2048', 'gt_8192'):
                link_or_copy(source/'input_2048'/sub/(name+'.xyz'), target/sub/(name+'.xyz'))
            link_or_copy(source/'original_meshes'/(name+'.off'), target/'meshes'/(name+'.off'))
        origin = 'publisher_PU1K_test_inputs_and_GT_unchanged'
    else:
        from prepare_retraining_data import sample_mesh
        meshes = sorted((pugan_mesh_root(data_root)/'test').glob('*.off'))
        if len(meshes) != TEST_COUNTS[dataset]:
            raise ValueError('Expected the 27 publisher PU-GAN test meshes')
        for mesh in meshes:
            relative = mesh.relative_to(data_root).as_posix()
            for size, sub in [(2048, 'input_2048'), (8192, 'gt_8192')]:
                points = sample_mesh(mesh, size, stable_seed(seed, relative, 'whole-object', size))
                path = target/sub/(mesh.stem+'.xyz')
                path.parent.mkdir(parents=True, exist_ok=True)
                # An incomplete cache must not silently resample already written files.
                if path.exists():
                    actual = np.loadtxt(path, dtype=np.float32)
                    if actual.shape != points.shape or not np.allclose(actual, points, rtol=1e-6, atol=1e-8):
                        raise ValueError(f'Partial test cache differs: {path}')
                else:
                    np.savetxt(path, points, fmt='%.10g')
            link_or_copy(mesh, target/'meshes'/mesh.name)
        origin = 'fixed_Poisson_test_pairs_from_publisher_meshes'
    hashes = test_files(target, TEST_COUNTS[dataset])
    info = dict(test_root=str(target.resolve()), test_shapes=TEST_COUNTS[dataset], seed=seed,
                test_origin=origin, test_sha256=hashes)
    write_json(marker, info)
    return info


def gaussian_mesh_patches(path, count, seed, gpu_batch):
    """Same geometric operations as author generate_dataset.py; bounded GPU batches."""
    import torch
    from prepare_retraining_data import sample_mesh
    sys.path.insert(0, str(ROOT/'compare/PU-Gaussian-main'))
    from utils.model_utils import FPS, extract_knn_patch, normalize_point_cloud
    surface = sample_mesh(path, 40960, seed)
    with torch.no_grad():
        points = torch.as_tensor(surface, dtype=torch.float32, device='cuda').T[None].contiguous()
        points, _, _ = normalize_point_cloud(points)
        centers = FPS(points, count)
        dense = extract_knn_patch(5120, points, centers)
        for start in range(0, count, gpu_batch):
            patch = dense[start:start+gpu_batch].contiguous()
            # Three separate calls preserve the released generator's operations.
            values = {n: FPS(patch, n).transpose(1, 2).cpu().numpy() for n in (256, 1024, 1536)}
            values[5120] = patch.transpose(1, 2).cpu().numpy()
            yield values


def write_gaussian_h5(path, meshes, patch_count, seed, data_root, gpu_batch=8, append_h5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [dict(mesh=str(p.resolve()), mesh_sha256=sha256(p), patches=patch_count) for p in meshes]
    appended = h5_info(append_h5, True, scan=False) if append_h5 else None
    recipe = dict(version=VERSION, seed=seed, meshes=entries, append=appended,
                  builder_sha256=sha256(__file__),
                  surface_points=40960, patch_neighbors=5120,
                  operation='author whole-object normalization + FPS centers + KNN + independent FPS calls')
    marker = path.with_suffix('.recipe.json')
    if path.is_file():
        saved = read_json(marker) if marker.is_file() else {}
        if saved.get('recipe') != recipe or saved.get('sha256') != sha256(path):
            raise ValueError(f'Gaussian H5 provenance differs: {path}; use a new cache directory')
        return h5_info(path, True, scan=False)
    total = len(meshes)*patch_count + (appended['train_patches'] if appended else 0)
    staging = path.with_suffix('.building.h5')
    # A failed generation restarts this explicitly named temporary file, never a completed H5.
    with h5py.File(staging, 'w') as file:
        arrays = {n: file.create_dataset(f'poisson_{n}', (total, n, 3), dtype='f4',
                    chunks=(1, n, 3), compression='gzip', compression_opts=1, shuffle=True) for n in SIZES}
        offset = 0
        for mesh in meshes:
            seen = 0
            relative = mesh.relative_to(data_root).as_posix()
            for values in gaussian_mesh_patches(mesh, patch_count,
                    stable_seed(seed, 'native-gaussian', relative), gpu_batch):
                amount = len(values[256])
                if set(values) != set(SIZES) or amount < 1:
                    raise ValueError('Incomplete Gaussian densities')
                for n in SIZES:
                    if values[n].shape != (amount, n, 3) or not np.isfinite(values[n]).all():
                        raise ValueError(f'Invalid Gaussian patch batch: {mesh}/{n}')
                    arrays[n][offset:offset+amount] = values[n]
                offset += amount
                seen += amount
            if seen != patch_count:
                raise ValueError(f'Wrong patch count for {mesh}: {seen}/{patch_count}')
            print(f'Gaussian data: {offset}/{total} patches; {mesh.name}', flush=True)
        if appended:
            with h5py.File(append_h5, 'r') as source:
                for start in range(0, appended['train_patches'], 32):
                    amount = min(32, appended['train_patches']-start)
                    for n in SIZES:
                        arrays[n][offset:offset+amount] = source[f'poisson_{n}'][start:start+amount]
                    offset += amount
        if offset != total:
            raise ValueError('Incomplete Gaussian H5')
        file.attrs['protocol'] = 'author_style_gaussian_x20_v1'
    os.replace(staging, path)
    write_json(marker, dict(recipe=recipe, sha256=sha256(path)))
    return h5_info(path, True, scan=False)


def raw_pool_audit(data_root, datasets):
    """Report known publisher mesh overlaps; H5 rows have no reliable shape IDs."""
    test_hashes = {}
    for row in datasets.values():
        for relative, digest in row['test']['test_sha256'].items():
            if relative.startswith('meshes/'):
                test_hashes.setdefault(digest, []).append(str(Path(row['test']['test_root'])/relative))
    audit = dict(h5_row_shape_disjointness='not asserted; publisher H5 rows lack shape identifiers',
                 original_raw_pool_overlaps=[])
    for name in ('pu1k', 'pugan'):
        try:
            meshes = raw_meshes(data_root, name)
        except (ValueError, FileNotFoundError):
            continue
        for mesh in meshes:
            digest = sha256(mesh)
            if digest in test_hashes:
                audit['original_raw_pool_overlaps'].append(dict(
                    train_mesh=str(mesh), sha256=digest, test_meshes=test_hashes[digest]))
    return audit


def prepare(args):
    data_root, cache = Path(args.data_root).resolve(), Path(args.cache_root).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache/'manifest.json'
    previous = read_json(marker) if marker.is_file() else {}
    settings = dict(version=VERSION, data_root=str(data_root), seed=args.seed)
    if previous and previous['settings'] != settings:
        raise ValueError('Native cache settings changed; use a different --cache-root')
    manifest = dict(settings=settings, status='preparing', datasets=previous.get('datasets', {}))
    write_json(marker, manifest)
    try:
        for dataset in args.datasets:
            original = find_h5(data_root, dataset, getattr(args, dataset+'_h5', None))
            print(f'Checking publisher H5: {dataset}: {original}', flush=True)
            old = manifest['datasets'].get(dataset, {})
            # Scan full source once; subsequent uses still verify SHA256 identity.
            if old.get('publisher', {}).get('training_h5') == str(original) and old['publisher'].get('training_h5_sha256') == sha256(original):
                publisher = old['publisher']
            else:
                publisher = h5_info(original)
            test = prepare_test(data_root, cache, dataset, args.seed, args.reuse_test_root)
            print(f'Publisher patches={publisher["train_patches"]}; fixed tests={test["test_shapes"]}: '
                  f'{test["test_root"]}', flush=True)
            manifest['datasets'][dataset] = dict(publisher=publisher, test=test, **({'gaussian': old['gaussian']} if 'gaussian' in old else {}))
            write_json(marker, manifest)
        if 'PU-Gaussian' in args.models and not args.tests_only:
            for dataset in args.datasets:
                explicit = getattr(args, 'gaussian_'+dataset+'_h5', None)
                if explicit:
                    info = h5_info(explicit, True)
                    info['origin'] = 'explicit_author_x20_h5'
                else:
                    # PU1K author generator appends the complete PU-GAN x20 H5.
                    pugan_path = cache/'gaussian/pugan_x20.h5'
                    pugan_info = write_gaussian_h5(pugan_path, raw_meshes(data_root, 'pugan'), 200,
                        args.seed, data_root, args.gaussian_patch_batch)
                    if dataset == 'pugan':
                        info = pugan_info
                    else:
                        info = write_gaussian_h5(cache/'gaussian/pu1k_x20.h5', raw_meshes(data_root, 'pu1k'), 50,
                            args.seed, data_root, args.gaussian_patch_batch, pugan_path)
                manifest['datasets'][dataset]['gaussian'] = info
                write_json(marker, manifest)
        manifest['publisher_pool_audit'] = raw_pool_audit(data_root, manifest['datasets'])
        manifest['status'] = 'tests_ready' if args.tests_only else 'completed'
        write_json(marker, manifest)
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        write_json(marker, manifest)
        raise
    print(f'Native data {manifest["status"]}: {cache}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--datasets', choices=('pu1k', 'pugan'), nargs='+', default=['pu1k', 'pugan'])
    parser.add_argument('--models', nargs='+', default=['E0', 'E6', 'Grad-PU', 'RepKPU', 'PU-Gaussian'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tests-only', action='store_true')
    parser.add_argument('--reuse-test-root', type=Path)
    parser.add_argument('--gaussian-patch-batch', type=int, default=8)
    for dataset in ('pu1k', 'pugan'):
        parser.add_argument('--'+dataset+'-h5', type=Path)
        parser.add_argument('--gaussian-'+dataset+'-h5', type=Path)
    args = parser.parse_args()
    if args.gaussian_patch_batch < 1:
        parser.error('--gaussian-patch-batch must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
