"""Small, dependency-light utilities shared by the new retraining commands."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('E0', 'E6', 'Grad-PU', 'RepKPU', 'PU-Gaussian')
DEFAULT_MODELS = ('E0', 'E6', 'RepKPU', 'PU-Gaussian')
DATA_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + '.tmp')
    staged.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(staged, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stable_seed(seed, *parts):
    value = '|'.join([str(seed), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'little') & 0x7fffffff


def source_hashes():
    folders = ['compare', 'our/PU-GCSNet/network', 'our/PU-GCSNet/utils']
    paths = [p for folder in folders for p in (ROOT / folder).rglob('*.py')
             if not any(x in p.parts for x in ('training-runs', 'evaluation-runs', 'retrain-runs', '.pugaussian-runtime', '__pycache__'))]
    paths += [ROOT / 'our/PU-GCSNet/model.py', ROOT / 'our/PU-GCSNet/main.py', ROOT / 'our/PU-GCSNet/data.py']
    paths += list((ROOT / 'compare').glob('*.sh'))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(paths))}
