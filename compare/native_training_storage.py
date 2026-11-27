"""Inventory storage; optionally remove only obsolete generated prepared caches."""
import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from retrain_common import ROOT, read_json, sha256, write_json


def sizes(path):
    apparent, reclaimable, files = 0, 0, 0
    seen = set()
    for folder, dirs, names in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(folder)/d).is_symlink()]
        for name in names:
            p = Path(folder)/name
            if p.is_symlink():
                continue
            stat = p.stat()
            files += 1
            apparent += stat.st_size
            identity = (stat.st_dev, stat.st_ino)
            if stat.st_nlink == 1 and identity not in seen:
                reclaimable += getattr(stat, 'st_blocks', (stat.st_size+511)//512)*512
            seen.add(identity)
    return dict(files=files, apparent_GiB=round(apparent/2**30, 3),
                estimated_reclaimable_GiB=round(reclaimable/2**30, 3))


def old_caches(root):
    root = Path(root).resolve()
    data = root/'data'
    found = []
    if data.is_dir():
        for p in data.iterdir():
            if p.is_symlink() or not p.is_dir():
                continue
            if re.fullmatch(r'repro_20261123_seed\d+(?:_smoke_b\d+)?', p.name) and (p/'manifest.json').is_file():
                marker = read_json(p/'manifest.json')
                if 'datasets' in marker and 'settings' in marker:
                    found.append(p.resolve())
            elif p.name == 'prepared' and (p/'split.json').is_file() and (p/'train.h5').is_file():
                found.append(p.resolve())
    return sorted(found)


def old_generated_h5(root):
    folder = Path(root).resolve()/'compare/training-runs'
    if not folder.is_dir():
        return []
    names = {'train.h5', 'val.h5', 'pu1k_train_x20.h5', 'pu1k_val_x20.h5'}
    return sorted(p.resolve() for p in folder.rglob('*.h5')
                  if p.name in names and not p.is_symlink() and p.resolve().is_relative_to(folder))


def validate_cleanup(root, cache, targets):
    """Never remove original H5s, the new cache, symlinks, or the only test copy."""
    root, cache = Path(root).resolve(), Path(cache).resolve()
    marker = read_json(cache/'manifest.json')
    if marker['status'] not in ('tests_ready', 'completed'):
        raise ValueError('First complete --tests-only or --prepare-only with the native launcher')
    if not {'pu1k', 'pugan'} <= set(marker['datasets']):
        raise ValueError('Both fixed test sets must be cached before old prepared data is removed')
    source_root = Path(marker.get('settings', {}).get('data_root', root/'data')).resolve()
    protected = [cache, source_root]
    protected.extend((source_root/name).resolve() for name in ('PU1K', 'PU1K_raw_meshes', 'PUGAN', 'PU-GAN')
                     if (source_root/name).exists())
    for dataset, expected in [('pu1k', 127), ('pugan', 27)]:
        row = marker['datasets'][dataset]
        for key in ('publisher', 'gaussian'):
            if key in row:
                path = Path(row[key]['training_h5']).resolve()
                if not path.is_file() or sha256(path) != row[key]['training_h5_sha256']:
                    raise ValueError(f'Protected native H5 changed/missing: {path}')
                protected.append(path)
        test = row['test']
        folder = Path(test['test_root']).resolve()
        if test['test_shapes'] != expected or len(test['test_sha256']) != expected*3:
            raise ValueError('Incomplete native test provenance')
        protected.append(folder)
        for relative, digest in test['test_sha256'].items():
            path = (folder/relative).resolve()
            if not path.is_relative_to(folder) or not path.is_file() or sha256(path) != digest:
                raise ValueError(f'Protected fixed test changed/missing: {path}')
    allowed = set(old_caches(root)) | set(old_generated_h5(root))
    for target in targets:
        target = Path(target).resolve()
        if target not in allowed or not target.is_relative_to(root) or target == root:
            raise ValueError(f'Not an explicitly recognized generated cache: {target}')
        if any(p == target or p.is_relative_to(target) for p in protected):
            raise ValueError(f'A native data dependency still lives in {target}')
    return targets


def inventory(root):
    root = Path(root).resolve()
    rows = [dict(path=str(p), category='old_generated_prepared_cache', **sizes(p)) for p in old_caches(root)]
    rows.extend(dict(path=str(p), category='old_generated_training_H5',
                     apparent_GiB=round(p.stat().st_size/2**30, 3)) for p in old_generated_h5(root))
    for relative, category in [('compare/training-runs', 'old_training_history_keep_unless_backed_up'),
            ('compare/retrain-runs', 'old_retraining_and_structure_history_keep_unless_backed_up'),
            ('compare/native-runs', 'new_native_runs_keep'),
            ('compare/.pugaussian-runtime', 'native_dependencies_keep')]:
        p = root/relative
        if p.is_dir():
            rows.append(dict(path=str(p), category=category, **sizes(p)))
    return rows


def archive_metadata(root, cache, targets):
    root = Path(root).resolve()
    archive = root/'compare/native-runs'/f'cleanup-records-{datetime.now():%Y%m%d-%H%M%S-%f}'
    archive.mkdir(parents=True, exist_ok=False)
    copies = {}
    for target in targets:
        if not target.is_dir():
            continue
        for relative in ('manifest.json', 'split.json', 'PU1K/split.json', 'PU-GAN/split.json'):
            source = target/relative
            if source.is_file():
                destination = archive/target.relative_to(root)/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                copies[str(source)] = dict(backup=str(destination), sha256=sha256(destination))
    record = dict(native_cache=str(Path(cache).resolve()), targets=list(map(str, targets)),
                  metadata=copies, completed_targets=[])
    write_json(archive/'cleanup.json', record)
    return archive, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, default=ROOT/'data/native_20261127_seed42')
    parser.add_argument('--delete-old-prepared', action='store_true',
                        help='Delete only recognized old generated data after verifying both new fixed tests and native H5s')
    args = parser.parse_args()
    rows = inventory(ROOT)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    if not args.delete_old_prepared:
        print('Read-only inventory. Original datasets, weights and runtimes have not been removed.')
        return
    import fcntl
    lockpath = ROOT/'compare/.pugaussian-runtime/training.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A training queue is active; no data was removed')
        targets = validate_cleanup(ROOT, args.cache_root, old_caches(ROOT)+old_generated_h5(ROOT))
        if not targets:
            print('No recognized obsolete generated data remains.')
            return
        archive, record = archive_metadata(ROOT, args.cache_root, targets)
        for path in targets:
            print(f'Removing obsolete generated cache: {path}', flush=True)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            record['completed_targets'].append(str(path))
            write_json(archive/'cleanup.json', record)
        print(f'Preserved old split/provenance metadata: {archive}', flush=True)
    print('Old generated caches removed. Native original H5s, meshes, checkpoints and runtime retained.')


if __name__ == '__main__':
    main()
