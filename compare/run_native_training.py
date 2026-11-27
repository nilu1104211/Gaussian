"""One server command: publisher data -> automatic smoke -> native training -> shared tests."""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from native_training_protocol import DATASETS, MODELS, FOLDERS, make_config, plan, runtime_sources
from retrain_common import ROOT, read_json, sha256, write_json

RUNS = ROOT/'compare/native-runs'
PATH_OPTIONS = ('data_root', 'cache_root', 'reuse_test_root', 'pu1k_h5', 'pugan_h5',
                'gaussian_pu1k_h5', 'gaussian_pugan_h5')


def options_dict(args):
    keys = (*PATH_OPTIONS, 'datasets', 'models', 'seed', 'epochs', 'batch_size', 'grad_lr',
            'log_every', 'no_test', 'gaussian_patch_batch', 'prepare_only', 'tests_only', 'smoke_only')
    return {key: str(value.resolve()) if isinstance(value, Path) else value
            for key in keys for value in [getattr(args, key)]}


def invoke(command, env, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a', encoding='utf-8') as log:
        print('Running:', ' '.join(map(str, command)), flush=True)
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise


def aggregate(run_dir, jobs):
    rows = []
    for job in jobs:
        status = read_json(Path(job['output'])/'status.json')
        if status.get('metrics_dir'):
            with (Path(status['metrics_dir'])/'summary.csv').open(encoding='utf-8-sig') as stream:
                rows.extend(csv.DictReader(stream))
    if rows:
        with (run_dir/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def execute(args):
    from run_retraining import environment
    env = environment()
    run_dir = (args.resume_run or args.run_dir or
               RUNS/f'native-{datetime.now():%Y%m%d-%H%M%S-%f}').resolve()
    sources = runtime_sources()
    if args.resume_run:
        manifest = read_json(run_dir/'run.json')
        if manifest['sources'] != sources or manifest['options'] != options_dict(args):
            raise ValueError('Native run code/options changed; use a new run directory')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = dict(status='preparing', options=options_dict(args), sources=sources,
                        protocol='publisher_H5_native_loaders_fixed_final_epoch')
        write_json(run_dir/'run.json', manifest)
    print(f'NATIVE RUN DIRECTORY: {run_dir}', flush=True)
    try:
        command = [sys.executable, '-B', '-u', str(ROOT/'compare/prepare_native_data.py'),
                   '--data-root', str(args.data_root), '--cache-root', str(args.cache_root),
                   '--seed', str(args.seed), '--datasets', *args.datasets, '--models', *args.models,
                   '--gaussian-patch-batch', str(args.gaussian_patch_batch)]
        if args.tests_only:
            command.append('--tests-only')
        for key in PATH_OPTIONS:
            if key not in ('data_root', 'cache_root') and getattr(args, key) is not None:
                command += ['--'+key.replace('_', '-'), str(getattr(args, key))]
        invoke(command, env, run_dir/'prepare.log')
        data = read_json(args.cache_root/'manifest.json')
        if args.prepare_only or args.tests_only:
            manifest.update(status='data_prepared', cache_root=str(args.cache_root))
            write_json(run_dir/'run.json', manifest)
            print(f'NATIVE DATA READY: {args.cache_root}', flush=True)
            return
        jobs = []
        smoke_jobs = []
        for dataset in args.datasets:
            for model in args.models:
                method_data = dict(data['datasets'][dataset]['gaussian' if model == 'PU-Gaussian' else 'publisher'],
                                   **data['datasets'][dataset]['test'])
                for smoke, collection, folder in [(True, smoke_jobs, run_dir/'smoke'), (False, jobs, run_dir)]:
                    collection.append(make_config(model, dataset, folder/dataset/model, method_data,
                        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, grad_lr=args.grad_lr,
                        smoke=smoke, log_every=args.log_every, evaluate_test=not args.no_test))
        if args.resume_run and 'jobs' in manifest and manifest['jobs'] != jobs:
            raise ValueError('Native training/test data changed since this run started')
        manifest.pop('error', None)
        manifest.update(status='smoke', jobs=jobs, smoke_jobs=smoke_jobs)
        write_json(run_dir/'run.json', manifest)
        for phase, collection in [('smoke', smoke_jobs), ('training', [] if args.smoke_only else jobs)]:
            manifest['status'] = phase
            write_json(run_dir/'run.json', manifest)
            for config in collection:
                output = Path(config['output'])
                status_file = output/'status.json'
                if args.resume_run and status_file.is_file():
                    status = read_json(status_file)
                    if status['status'] == 'completed':
                        for filename, key in [('last.pth', 'last_sha256')]+([] if config['smoke'] else [('final.pth', 'final_sha256')]):
                            if sha256(output/filename) != status[key]:
                                raise ValueError(f'Completed native checkpoint changed: {output/filename}')
                        if not read_json(output/'health.json')['passed']:
                            raise ValueError(f'Unhealthy completed job: {output}')
                        print(f'Skipping completed {phase}: {config["dataset"]}/{config["model"]}', flush=True)
                        continue
                output.mkdir(parents=True, exist_ok=True)
                config_path = output/'config.json'
                if config_path.is_file() and read_json(config_path) != config:
                    raise ValueError(f'Configuration differs: {config_path}')
                write_json(config_path, config)
                command = [sys.executable, '-B', '-u', str(ROOT/'compare/native_training_worker.py'),
                           '--config', str(config_path)]
                if args.resume_run:
                    command.append('--resume')
                print(f'NATIVE {phase.upper()}: {config["dataset"]}/{config["model"]}; '
                      f'epochs={config["epochs"]}, batch={config["batch_size"]}, seed={config["model_seed"]}', flush=True)
                invoke(command, env, output/'train.log')
                if read_json(status_file)['status'] != 'completed' or not read_json(output/'health.json')['passed']:
                    raise RuntimeError(f'Incomplete native job: {output}')
        if not args.smoke_only:
            aggregate(run_dir, jobs)
        manifest['status'] = 'smoke_pass' if args.smoke_only else 'completed'
        write_json(run_dir/'run.json', manifest)
        print(f'NATIVE {manifest["status"].upper()}: {run_dir}', flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        write_json(run_dir/'run.json', manifest)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--cache-root', type=Path)
    p.add_argument('--datasets', choices=DATASETS, nargs='+', default=list(DATASETS))
    p.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    p.add_argument('--seed', type=int, default=42, help='Fixed test/data generation seed; models keep native seeds')
    p.add_argument('--epochs', type=int, help='Total screening budget per model; Gaussian splits it into two phases')
    p.add_argument('--batch-size', type=int, help='Explicit override; default keeps native per-model batch sizes')
    p.add_argument('--grad-lr', type=float, default=.001)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--gaussian-patch-batch', type=int, default=8)
    p.add_argument('--no-test', action='store_true', help='Train/diagnose only; no benchmark test access')
    p.add_argument('--reuse-test-root', type=Path, help='Old prepared root containing PU1K/test and PU-GAN/test')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--tests-only', action='store_true', help='Cache tests first, so old generated data can be cleaned')
    p.add_argument('--smoke-only', action='store_true')
    p.add_argument('--plan', action='store_true', help='Print budgets and native settings without CUDA/data access')
    p.add_argument('--status', action='store_true', help='Read --resume-run status without CUDA')
    for dataset in DATASETS:
        p.add_argument('--'+dataset+'-h5', type=Path)
        p.add_argument('--gaussian-'+dataset+'-h5', type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--run-dir', type=Path)
    group.add_argument('--resume-run', type=Path)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.status:
        if args.resume_run is None:
            p.error('--status requires --resume-run')
        run = read_json(args.resume_run/'run.json')
        print(json.dumps(dict(status=run['status'], error=run.get('error'), jobs=[dict(
            model=c['model'], dataset=c['dataset'], **(read_json(Path(c['output'])/'status.json')
            if (Path(c['output'])/'status.json').is_file() else dict(status='pending')))
            for c in run.get('jobs', [])]), ensure_ascii=False, indent=2))
        return
    if args.resume_run:
        disallowed = [a for a in sys.argv[1:] if a.startswith('--') and a not in ('--resume-run', '--plan')]
        if disallowed:
            p.error('Resume with --resume-run alone; settings are restored from run.json')
        run_dir, print_plan = args.resume_run, args.plan
        saved = read_json(run_dir/'run.json')['options']
        for key in PATH_OPTIONS:
            saved[key] = Path(saved[key]) if saved[key] is not None else None
        args = SimpleNamespace(**saved, resume_run=run_dir, run_dir=None, plan=print_plan)
    if args.cache_root is None:
        args.cache_root = args.data_root/f'native_20261127_seed{args.seed}'
    for key in PATH_OPTIONS:
        if getattr(args, key) is not None:
            setattr(args, key, Path(getattr(args, key)).resolve())
    if len(set(args.models)) != len(args.models) or len(set(args.datasets)) != len(args.datasets):
        p.error('Duplicate model/dataset')
    if not 0 <= args.seed < 2**32 or args.grad_lr <= 0 or args.log_every < 1 or args.gaussian_patch_batch < 1:
        p.error('Invalid seed, learning rate, logging interval or generation batch')
    if args.epochs is not None and (args.epochs < 1 or 'PU-Gaussian' in args.models and args.epochs < 2):
        p.error('Epoch budget must be positive; Gaussian needs at least two epochs')
    if args.batch_size is not None and args.batch_size < 1:
        p.error('Batch size must be positive')
    if args.plan:
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
        return
    import fcntl
    lockpath = ROOT/'compare/.pugaussian-runtime/training.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another comparison/native/structure queue is running')
        execute(args)


if __name__ == '__main__':
    main()
