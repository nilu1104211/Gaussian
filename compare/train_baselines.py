"""Train PU-Gaussian, Grad-PU and RepKPU sequentially on PU1K."""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


COMPARE = Path(__file__).resolve().parent
RUNTIME = COMPARE / '.pugaussian-runtime'
MODELS = ('PU-Gaussian', 'Grad-PU', 'RepKPU')


def verify_cuda_sources():
    """Check the bundled native code before sharing the compiled extensions."""
    legacy_declarations = {'#include <THC/THC.h>', 'extern THCState *state;'}

    def sources(project, pointops, chamfer):
        pointops = project / pointops
        chamfer = project / chamfer
        files = [pointops / 'setup.py', *(pointops / 'src').rglob('*'),
                 *(chamfer / filename for filename in ('setup.py', 'chamfer_cuda.cpp', 'chamfer3D.cu'))]
        result = {}
        for path in files:
            if path.is_file() and path.suffix in ('.py', '.cpp', '.cu', '.h', '.hpp', '.cuh'):
                root = pointops if path.is_relative_to(pointops) else chamfer
                lines = (line.strip() for line in path.read_text(encoding='utf-8').splitlines())
                result[(root.name, path.relative_to(root))] = [
                    line for line in lines if line and line not in legacy_declarations]
        return result

    compiled = sources(COMPARE / 'PU-Gaussian-main', 'pointops', 'utils/Chamfer3D')
    for model in ('Grad-PU', 'RepKPU'):
        if sources(COMPARE / f'{model}-main', 'models/pointops', 'models/Chamfer3D') != compiled:
            raise RuntimeError(f'{model} native sources no longer match the PU-Gaussian runtime')


def train_model(model, epochs, batch_size, seed, train, val, run_dir, env, smoke):
    destination = run_dir / model
    destination.mkdir()
    if model == 'PU-Gaussian':
        args = ['--pu1k_path', str(train), '--val_h5_path', str(val), '--max_epochs', str(epochs),
                '--stage_1_max_epochs', str(min(1, epochs - 1)), '--random_seed', str(seed),
                '--model_dir', str(destination / 'checkpoints')]
        checkpoint = 'best.pth'
    else:
        args = ['--h5_file_path', str(train), '--val_h5_path', str(val), '--epochs', str(epochs), '--seed', str(seed),
                '--out_path', str(destination), '--num_workers', '0' if smoke else '4']
        checkpoint = 'ckpt-best.pth'

    command = [sys.executable, '-u', 'train.py', '--dataset', 'pu1k', '--best_only',
               '--batch_size', str(batch_size), *args]
    logfile = destination / 'train.log'
    print(f'[{datetime.now():%F %T}] Starting {model} ({epochs} epochs): {logfile}', flush=True)
    with logfile.open('w', encoding='utf-8') as stream:
        stream.write(shlex.join(command) + '\n')
        stream.flush()
        result = subprocess.run(command, cwd=COMPARE / f'{model}-main', env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{model} exited with status {result.returncode}. See {logfile}')
    saved = list(destination.rglob('*.pth'))
    if len(saved) != 1 or saved[0].name != checkpoint or saved[0].stat().st_size == 0:
        raise RuntimeError(f'{model} should write exactly one nonempty {checkpoint}. See {logfile}')
    print(f'[{datetime.now():%F %T}] Completed {model}: {saved[0]}', flush=True)


def run_queue(args):
    from prepare_pugaussian_h5 import patch_count, prepare_gaussian_h5, subset_patches

    data_root = args.data_root.resolve()
    train = data_root / 'prepared' / 'train.h5'
    val = data_root / 'prepared' / 'val.h5'
    if train.resolve() == val.resolve():
        raise ValueError('Training and validation must use different H5 files')
    print(f'Training H5: {train} ({patch_count(train)} patches)', flush=True)
    print(f'Validation H5: {val} ({patch_count(val)} patches)', flush=True)
    verify_cuda_sources()

    subprocess.run(['bash', str(COMPARE / 'run_pugaussian.sh'), 'setup'], check=True)
    env = dict(os.environ, PYTHONPATH=str((RUNTIME / 'python').resolve()), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0', MPLBACKEND='Agg')

    label = 'smoke' if args.smoke else 'train'
    run_dir = COMPARE / 'training-runs' / f'pu1k-{label}-seed{args.seed}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    run_dir.mkdir(parents=True)
    print(f'Run directory: {run_dir}', flush=True)
    if args.smoke:
        batch = min(4, args.batch_size)
        small_train = run_dir / 'train.h5'
        small_val = run_dir / 'val.h5'
        subset_patches(train, small_train, batch)
        subset_patches(val, small_val, batch)
        epochs = (2, 1, 1)
    else:
        batch, epochs = args.batch_size, args.epochs
        small_train, small_val = train, val

    first = MODELS.index(args.start_at)
    if first == 0:
        dense_train = (run_dir if args.smoke else RUNTIME) / 'pu1k_train_x20.h5'
        prepare_gaussian_h5(train, data_root, dense_train, args.seed, limit=batch if args.smoke else None)
        train_model('PU-Gaussian', epochs[0], batch, args.seed, dense_train, small_val, run_dir, env, args.smoke)
    for index in range(max(1, first), len(MODELS)):
        train_model(MODELS[index], epochs[index], batch, args.seed, small_train, small_val,
                    run_dir, env, args.smoke)
    print(f'PU1K {label} queue completed: {run_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=COMPARE.parent / 'data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, nargs=3, default=(200, 60, 100),
                        metavar=('PU_GAUSSIAN', 'GRAD_PU', 'REPKPU'))
    parser.add_argument('--start-at', choices=MODELS, default=MODELS[0],
                        help='Start a new queue at this model and skip earlier models')
    parser.add_argument('--smoke', action='store_true', help='Run a few batches and check saved weights')
    args = parser.parse_args()
    if not (0 <= args.seed < 2 ** 32) or min(args.batch_size, *args.epochs) < 1:
        parser.error('Use a valid NumPy seed, a positive batch size and positive epoch counts')
    active = Path(os.environ.get('CONDA_PREFIX', ''))
    if active.name != 'scdata-v3' or Path(sys.prefix).resolve() != active.resolve():
        parser.error('Run conda activate scdata-v3 before launching the queue')

    import fcntl

    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / 'training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, 'A comparison training queue is already running.\n')
        run_queue(args)


if __name__ == '__main__':
    main()
