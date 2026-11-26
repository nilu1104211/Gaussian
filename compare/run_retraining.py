"""Prepare -> native smoke gate -> train/validate/test four networks per dataset.

Defaults: PU1K then PU-GAN, E0/E6/RepKPU/PU-Gaussian, seed 42. Grad-PU is deferred.
The only retained model files per job are best.pth and last.pth (resume state).
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from retrain_common import DEFAULT_MODELS, MODELS, PUGCS_MODELS, NEW_PUGCS_MODELS, ROOT, pugcs_variant_options, read_json, sha256, source_hashes, write_json

COMPARE = ROOT / 'compare'
RUNTIME = COMPARE / '.pugaussian-runtime'
RUNS = COMPARE / 'retrain-runs'
CERTIFICATE = RUNS / 'smoke_pass.json'


def structure_overrides(args):
    path = getattr(args, 'structure_config', None)
    if path is None:
        return {}
    values = read_json(path)
    allowed = set(pugcs_variant_options('E6'))
    if not isinstance(values, dict) or set(values) - set(NEW_PUGCS_MODELS):
        raise ValueError('Structure config must map new variant names to network option dictionaries')
    for name, options in values.items():
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError('Unknown network option for ' + name)
    return values


def profile(args):
    value = dict(seed=args.seed, batch_size=args.batch_size, input_policy=args.input_policy,
                data_root=str(args.data_root.resolve()),
                grad_lr=args.grad_lr, grad_head_bias=args.grad_head_bias, grad_seed=args.grad_seed,
                native_losses=True, lr_policy='native optimizer/epoch schedules; step intervals scaled by reference batch',
                validation='whole objects, independent float32 CD, every 5 epochs plus final',
                gaussian_schedule='100 Gaussian + 100 joint epochs')
    if getattr(args, 'structure_config', None) is not None:
        value['structure_overrides'] = structure_overrides(args)
    return value


def fingerprint(args):
    return hashlib.sha256(json.dumps(dict(profile=profile(args), sources=source_hashes()),
                                    sort_keys=True).encode()).hexdigest()


def job_config(args, dataset, model, output, prepared):
    reference_batch = 64 if model in PUGCS_MODELS and dataset == 'pu1k' else 32
    decay_iter = max(1, round(50000 * reference_batch / args.batch_size))
    epochs = 100 if model in PUGCS_MODELS else {'Grad-PU': 60, 'RepKPU': 100, 'PU-Gaussian': 200}[model]
    if model in NEW_PUGCS_MODELS and getattr(args, 'epochs', None) is not None:
        epochs = args.epochs
    stage1 = 100 if model == 'PU-Gaussian' else 0
    if args.smoke:
        epochs = 4 if model == 'PU-Gaussian' else args.smoke_epochs
        stage1 = 2 if model == 'PU-Gaussian' else 0
    config = dict(model=model, dataset=dataset, output=str(output), prepared=str(prepared),
        epochs=epochs, stage1_epochs=stage1, seed=args.seed, batch_size=args.batch_size,
        model_seed=args.grad_seed if model == 'Grad-PU' else args.seed,
        decay_iter=decay_iter, input_policy=args.input_policy, grad_lr=args.grad_lr,
        grad_head_bias=args.grad_head_bias, val_every=1 if args.smoke else 5, smoke=args.smoke)
    if model in NEW_PUGCS_MODELS:
        config['network_config'] = pugcs_variant_options(model)
        config['network_config'].update(structure_overrides(args).get(model, {}))
        config['loss_config'] = dict(lambda_surf=0.1, surf_eta=0., lambda_normal=0.,
            lambda_tail=0.05, lambda_growth=0.05, tail_fraction=0.02, tail_warmup_epochs=10)
        # Structure selection uses validation only. A frozen selected best can
        # later be tested explicitly using test_examples.py --all-shapes.
        config['evaluate_test'] = False
    return config


def environment():
    prefix = Path(os.environ.get('CONDA_PREFIX', ''))
    if prefix.name != 'scdata-v3' or prefix.resolve() != Path(sys.prefix).resolve():
        raise RuntimeError('Run conda activate scdata-v3 on the server first')
    from train_baselines import verify_cuda_sources
    verify_cuda_sources()
    subprocess.run(['bash', str(COMPARE / 'run_pugaussian.sh'), 'setup'], check=True, cwd=ROOT)
    packages = (RUNTIME / 'python').resolve()
    env = dict(os.environ, PYTHONPATH=str(packages), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='0', MPLBACKEND='Agg')
    check = '''
import torch, numpy, scipy, open3d, h5py, einops, plyfile, pointops_cuda, chamfer_3D
assert torch.cuda.is_available()
print('Native preflight:', torch.__version__, torch.cuda.get_device_name(0),
      numpy.__version__, scipy.__version__, open3d.__version__, h5py.__version__)
'''
    subprocess.run([sys.executable, '-B', '-c', check], env=env, check=True, cwd=ROOT)
    return env


def aggregate(run_dir, jobs):
    for dataset in sorted({j['dataset'] for j in jobs}):
        rows = []
        validation_rows = []
        for job in jobs:
            if job['dataset'] != dataset:
                continue
            status = read_json(Path(job['output']) / 'status.json')
            if status['status'] == 'completed':
                if status.get('test_evaluated') is False:
                    with (Path(job['output']) / 'validation.csv').open(encoding='utf-8-sig') as stream:
                        best = next(row for row in csv.DictReader(stream) if int(row['epoch']) == status['best_epoch'])
                    validation_rows.append(dict(model=job['model'], dataset=dataset, **best))
                    continue
                with (Path(status['metrics_dir']) / 'summary.csv').open(encoding='utf-8-sig') as stream:
                    rows.extend(csv.DictReader(stream))
        if rows:
            with (run_dir / f'{dataset}_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if validation_rows:
            with (run_dir / f'{dataset}_validation_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(validation_rows[0]))
                writer.writeheader()
                writer.writerows(validation_rows)


def execute(args):
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_dir = (args.resume_run or args.run_dir or RUNS / f'{"smoke" if args.smoke else "train"}-seed{args.seed}-{stamp}').resolve()
    prepared = (args.prepared_root or args.data_root / (
        f'repro_20261123_seed{args.seed}' + (f'_smoke_b{args.batch_size}' if args.smoke else ''))).resolve()
    current_fingerprint = fingerprint(args)
    keys = [f'{dataset}/{model}' for dataset in args.datasets for model in args.models]
    if not args.smoke:
        certificate = read_json(CERTIFICATE) if CERTIFICATE.is_file() else {}
        if certificate.get('fingerprint') != current_fingerprint or not set(keys) <= set(certificate.get('passed_jobs', [])):
            raise RuntimeError('No matching native smoke PASS for these models/config/source files. Run --smoke first.')
    if args.resume_run:
        manifest = read_json(run_dir / 'run.json')
        if manifest['fingerprint'] != current_fingerprint or manifest['profile'] != profile(args):
            raise ValueError('Resume source/config differs from original run')
        prepared = Path(manifest['prepared'])
        jobs = manifest['jobs']
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        jobs = [job_config(args, dataset, model, run_dir / dataset / model, prepared)
                for dataset in args.datasets for model in args.models]
        manifest = dict(status='preflight', fingerprint=current_fingerprint, profile=profile(args),
                        prepared=str(prepared), smoke=args.smoke, jobs=jobs)
        write_json(run_dir / 'run.json', manifest)
    print(f'Run directory: {run_dir}', flush=True)
    try:
        env = environment()
        manifest['status'] = 'preparing_data'
        write_json(run_dir / 'run.json', manifest)
        command = [sys.executable, '-B', '-u', str(COMPARE / 'prepare_retraining_data.py'),
                   '--data-root', str(args.data_root.resolve()), '--output', str(prepared),
                   '--seed', str(args.seed), '--batch-size', str(args.batch_size)]
        if args.smoke:
            command.append('--smoke')
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        data_manifest = read_json(prepared / 'manifest.json')
        if data_manifest['settings']['smoke'] != args.smoke:
            raise ValueError('Formal training cannot use smoke data')
        if not args.smoke:
            for dataset, counts in [('PU1K', (917, 102, 127)), ('PU-GAN', (108, 12, 27))]:
                actual = data_manifest['datasets'][dataset]
                if tuple(actual[key] for key in ('train_shapes', 'validation_shapes', 'test_shapes')) != counts:
                    raise ValueError(f'Incorrect formal split for {dataset}: {actual}')
        manifest['data_manifest_sha256'] = sha256(prepared / 'manifest.json')
        manifest['status'] = 'training'
        write_json(run_dir / 'run.json', manifest)
        for config in jobs:
            output = Path(config['output'])
            status_file = output / 'status.json'
            if args.resume_run and status_file.is_file() and read_json(status_file)['status'] == 'completed':
                for file, key in [('best.pth', 'best_sha256'), ('last.pth', 'last_sha256')]:
                    if sha256(output / file) != read_json(status_file)[key]:
                        raise ValueError(f'Completed checkpoint changed: {output / file}')
                print(f'Skipping completed {config["dataset"]}/{config["model"]}', flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            config_path = output / 'config.json'
            if config_path.is_file() and read_json(config_path) != config:
                raise ValueError(f'Config changed: {config_path}')
            write_json(config_path, config)
            command = [sys.executable, '-B', '-u', str(COMPARE / 'retrain_worker.py'), '--config', str(config_path)]
            if args.resume_run:
                command.append('--resume')
            print(f'Starting {config["dataset"]}/{config["model"]}; {output / "train.log"}', flush=True)
            with (output / 'train.log').open('a', encoding='utf-8') as log:
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
                    try:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            print(line, end='', flush=True)
                        returncode = process.wait()
                        if returncode:
                            raise subprocess.CalledProcessError(returncode, command)
                    except BaseException:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                process.kill()
                        raise
            if read_json(status_file)['status'] != 'completed' or not read_json(output / 'health.json')['passed']:
                raise RuntimeError(f'Incomplete or unhealthy job: {output}')
            print(f'Completed {config["dataset"]}/{config["model"]}', flush=True)
        aggregate(run_dir, jobs)
        manifest['status'] = 'completed'
        write_json(run_dir / 'run.json', manifest)
        if args.smoke:
            old = read_json(CERTIFICATE) if CERTIFICATE.is_file() else {}
            passed = set(old.get('passed_jobs', [])) if old.get('fingerprint') == current_fingerprint else set()
            passed.update(keys)
            write_json(CERTIFICATE, dict(fingerprint=current_fingerprint, passed_jobs=sorted(passed),
                                        run=str(run_dir), profile=profile(args)))
        print(f'{"SMOKE PASS" if args.smoke else "RETRAINING COMPLETED"}: {run_dir}', flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        write_json(run_dir / 'run.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--prepared-root', type=Path)
    parser.add_argument('--datasets', choices=('pu1k', 'pugan'), nargs='+', default=['pu1k', 'pugan'])
    parser.add_argument('--models', choices=MODELS, nargs='+', default=list(DEFAULT_MODELS))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, help='Epoch budget for new PU-GCSNet structure variants only')
    parser.add_argument('--input-policy', choices=('fixed', 'native'), default='native',
                        help='Native resampling/augmentation on shared patch IDs; fixed is an optional controlled variant')
    parser.add_argument('--grad-lr', type=float, default=1e-3)
    parser.add_argument('--grad-seed', type=int, default=21,
                        help='Published Grad-PU default initialization seed; split/augmentation seed remains --seed')
    parser.add_argument('--grad-head-bias', type=float, default=None,
                        help='Explicit optional initialization repair; does not replace ReLU or change loss')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--structure-config', type=Path,
                        help='Optional JSON mapping new variant names to network option overrides; recorded in fingerprint/config')
    parser.add_argument('--smoke-epochs', type=int, default=3)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run-dir', type=Path)
    group.add_argument('--resume-run', type=Path)
    parser.add_argument('--plan', action='store_true', help='Print configurations without accessing CUDA/data')
    args = parser.parse_args()
    if args.epochs is not None and (args.epochs < 1 or any(model not in NEW_PUGCS_MODELS for model in args.models)):
        parser.error('--epochs requires a positive budget and only new structure variants')
    if args.batch_size < 1 or args.smoke_epochs < 2 or not 0 <= args.seed < 2**32 or not 0 <= args.grad_seed < 2**32 or args.grad_lr <= 0:
        parser.error('Use positive batch/lr, >=2 smoke epochs and a valid seed')
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.models)) != len(args.models):
        parser.error('Duplicate datasets/models')
    if args.plan:
        jobs = [job_config(args, dataset, model, Path('RUN') / dataset / model, Path('PREPARED'))
                for dataset in args.datasets for model in args.models]
        print(json.dumps(dict(profile=profile(args), jobs=jobs), indent=2))
        return
    # One GPU queue lock covers preparation, all selected model families and evaluation.
    import fcntl
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another comparison/retraining queue is running')
        execute(args)


if __name__ == '__main__':
    main()
