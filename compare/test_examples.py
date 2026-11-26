"""Run the four retrained models on matched examples in a new, portable folder.

The launcher uses only the standard library. Each native model runs in its own
process, using NativeAdapter.infer exactly as the formal retraining test does.
"""
import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from retrain_common import DEFAULT_MODELS, ROOT, read_json, sha256, stable_seed, write_json

DATASETS = {'pu1k': 'PU1K', 'pugan': 'PU-GAN'}
DEFAULT_RUN = ROOT / 'compare/retrain-runs/train-seed42-20261123-161953-503869'


def prepared_root(args):
    if args.prepared_root:
        return args.prepared_root.expanduser().resolve()
    # Map the archived server path to the current checkout when it was moved.
    recorded = Path(read_json(args.training_run / 'run.json')['prepared'])
    local = ROOT / 'data' / recorded.name
    return local if local.is_dir() else recorded


def available_shapes(source):
    names = sorted(p.stem for p in (source / 'input_2048').glob('*.xyz'))
    if not names:
        raise FileNotFoundError(f'No fixed test inputs in {source / "input_2048"}')
    for name in names:
        for folder, suffix in [('gt_8192', '.xyz'), ('meshes', '.off')]:
            path = source / folder / (name + suffix)
            if not path.is_file():
                raise FileNotFoundError(f'Missing matching label: {path}')
    return names


def select_shapes(names, requested, count, seed, dataset):
    if requested:
        selected = [name[:-4] if name.endswith('.xyz') else name for name in requested]
        if len(set(selected)) != len(selected):
            raise ValueError('Duplicate shape names')
        missing = set(selected) - set(names)
        if missing:
            raise ValueError(f'{dataset}: unknown shapes {sorted(missing)}; use --list-shapes')
        return selected
    if count < 1 or count > len(names):
        raise ValueError(f'{dataset}: --num-examples must be between 1 and {len(names)}')
    return sorted(random.Random(stable_seed(seed, 'examples', dataset)).sample(names, count))


def build_plan(args):
    training = args.training_run.expanduser().resolve()
    archive = read_json(training / 'run.json')
    if archive.get('status') != 'completed' or archive.get('smoke') is not False:
        raise ValueError('Choose a completed formal training run, not a smoke run')
    prepared = prepared_root(args).resolve()
    selections, jobs = {}, []
    for dataset in args.datasets:
        source = prepared / DATASETS[dataset] / 'test'
        names = select_shapes(available_shapes(source), args.shapes, args.num_examples, args.seed, dataset)
        selections[dataset] = []
        for name in names:
            paths = dict(input=source / 'input_2048' / f'{name}.xyz',
                         gt=source / 'gt_8192' / f'{name}.xyz',
                         mesh=source / 'meshes' / f'{name}.off')
            selections[dataset].append(dict(shape=name, sources={k: str(p) for k, p in paths.items()},
                                           sha256={k: sha256(p) for k, p in paths.items()}))
        for model in args.models:
            folder = training / dataset / model
            config = read_json(folder / 'config.json')
            status = read_json(folder / 'status.json')
            if config['model'] != model or config['dataset'] != dataset or status['status'] != 'completed':
                raise ValueError(f'Model/config/status mismatch: {folder}')
            checkpoint = folder / 'best.pth'
            digest = sha256(checkpoint)
            if digest != status['best_sha256']:
                raise ValueError(f'Best checkpoint hash changed: {checkpoint}')
            protocol = read_json(folder / 'test/metrics/protocol.json')
            for sample in selections[dataset]:
                for key, group, suffix in [('input', 'inputs', '.xyz'), ('gt', 'gt', '.xyz'),
                                           ('mesh', 'meshes', '.off')]:
                    expected = protocol['source_hashes'][group][sample['shape'] + suffix]
                    if sample['sha256'][key] != expected:
                        raise ValueError(f'{dataset}/{sample["shape"]}: {key} differs from {model} formal test data')
            jobs.append(dict(dataset=dataset, model=model, checkpoint=str(checkpoint),
                checkpoint_sha256=digest, best_epoch=status['best_epoch'], config=config,
                training_manifest=str(folder / 'manifest.json')))
    return dict(status='planned', training_run=str(training), prepared_root=str(prepared),
        seed=args.seed, models=args.models, selections=selections, jobs=jobs,
        previews=not args.no_previews, archive=not args.no_archive,
        inference='NativeAdapter.infer; same full 2048 -> 8192 path as formal test',
        coordinates='Original saved coordinates; no GT alignment or per-cloud rescaling',
        script_sha256=sha256(__file__), adapter_sha256=sha256(Path(__file__).with_name('retrain_adapters.py')))


def new_run_directory(root, tag):
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder = root / f'test-{stamp}-{tag}'
    folder.mkdir(exist_ok=False)
    return folder


def snapshot_inputs(run_dir, plan):
    for dataset, samples in plan['selections'].items():
        for sample in samples:
            folder = run_dir / dataset / sample['shape']
            folder.mkdir(parents=True, exist_ok=False)
            for key, name in [('input', 'input_2048.xyz'), ('gt', 'gt_8192.xyz'), ('mesh', 'gt_mesh.off')]:
                destination = folder / name
                shutil.copy2(sample['sources'][key], destination)
                if sha256(destination) != sample['sha256'][key]:
                    raise ValueError(f'Source changed while copying: {destination}')
    for index, job in enumerate(plan['jobs']):
        job['run_dir'] = str(run_dir)
        job['shapes'] = [s['shape'] for s in plan['selections'][job['dataset']]]
        job['seed'] = plan['seed']
        job_path = run_dir / 'jobs' / f'{index:02d}_{job["dataset"]}_{job["model"]}.json'
        job['job_file'] = str(job_path.relative_to(run_dir))
        write_json(job_path, job)
        shutil.copy2(job['training_manifest'], job_path.with_suffix('.training_manifest.json'))


def child_environment():
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='0', MPLBACKEND='Agg')
    packages = ROOT / 'compare/.pugaussian-runtime/python'
    if packages.is_dir():
        env['PYTHONPATH'] = str(packages.resolve()) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def run_child(arguments, log_path, env):
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), *arguments]
    with log_path.open('x', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
                if code:
                    raise RuntimeError(f'Worker exited with {code}; see {log_path}')
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise


def infer_job(path):
    import numpy as np
    import torch
    from evaluate_pugaussian_protocol import read_points
    from retrain_adapters import NativeAdapter
    from retrain_worker import seed_all

    job = read_json(path)
    if not torch.cuda.is_available():
        raise RuntimeError('Inference requires the server CUDA environment; activate scdata-v3')
    if sha256(job['checkpoint']) != job['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after planning')
    config = job['config']
    seed_all(config.get('model_seed', config['seed']))
    adapter = NativeAdapter(config)
    adapter.net.load_state_dict(torch.load(job['checkpoint'], map_location='cuda', weights_only=True), strict=True)
    adapter.net.eval()
    results = []
    for index, name in enumerate(job['shapes'], 1):
        # Identical name-based seed and inference path to retrain_worker.infer_directory.
        seed_all(stable_seed(job['seed'], 'inference', name))
        folder = Path(job['run_dir']) / job['dataset'] / name
        points = read_points(folder / 'input_2048.xyz', 2048)
        read_points(folder / 'gt_8192.xyz', 8192)
        torch.cuda.synchronize()
        started = time.monotonic()
        prediction = adapter.infer(points)
        torch.cuda.synchronize()
        duration = time.monotonic() - started
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (8192, 3) or not np.isfinite(prediction).all():
            raise ValueError(f'Invalid prediction: {job["model"]}/{name}')
        destination = folder / f'{job["model"]}.xyz'
        with destination.open('x', encoding='ascii') as stream:
            np.savetxt(stream, prediction, fmt='%.10g')
        results.append(dict(shape=name, points=len(prediction), seconds=duration, sha256=sha256(destination)))
        print(f'{job["dataset"]}/{job["model"]} {index}/{len(job["shapes"])} {name}: {duration:.2f}s', flush=True)
    write_json(Path(path).with_suffix('.result.json'), dict(status='completed', results=results,
        checkpoint_sha256=job['checkpoint_sha256'], torch=torch.__version__, gpu=torch.cuda.get_device_name(0)))


def write_ply(path, points):
    import numpy as np
    with path.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\n'
                     'property float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(stream, points, fmt='%.10g')


def render_comparison(path, clouds, title):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # One shared physical bounding box. Independent centering would hide drift.
    joined = np.concatenate([points for _, points in clouds])
    low, high = joined.min(axis=0), joined.max(axis=0)
    center = (low + high) / 2
    half = max(float((high - low).max()) * .53, 1e-6)
    views = [(20, -60), (20, 30), (90, -90)]
    figure = plt.figure(figsize=(2.8 * len(clouds), 7.8))
    for row, (elevation, azimuth) in enumerate(views):
        for column, (label, points) in enumerate(clouds):
            ax = figure.add_subplot(len(views), len(clouds), row * len(clouds) + column + 1, projection='3d')
            ax.scatter(*points.T, s=.5, c='#24628c', linewidths=0, depthshade=False)
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)
            ax.set_box_aspect((1, 1, 1))
            ax.set_proj_type('ortho')
            ax.view_init(elev=elevation, azim=azimuth)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'{label}\n{len(points)} points', fontsize=10)
    figure.suptitle(title + '\nShared coordinates, scale and camera in each row', fontsize=11)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=.87, wspace=0, hspace=0)
    figure.savefig(path, dpi=150, facecolor='white')
    plt.close(figure)


def export_run(run_dir):
    from evaluate_pugaussian_protocol import read_points

    plan = read_json(run_dir / 'run.json')
    rows = []
    for dataset, samples in plan['selections'].items():
        for sample in samples:
            folder = run_dir / dataset / sample['shape']
            clouds = []
            labels = [('Input', 'input_2048', 2048), ('GT', 'gt_8192', 8192)]
            labels += [(model, model, 8192) for model in plan['models']]
            for label, basename, count in labels:
                xyz = folder / f'{basename}.xyz'
                points = read_points(xyz, count)
                ply = xyz.with_suffix('.ply')
                write_ply(ply, points)
                clouds.append((label, points))
                rows.append(dict(dataset=dataset, shape=sample['shape'], label=label, points=count,
                    xyz=xyz.relative_to(run_dir).as_posix(), ply=ply.relative_to(run_dir).as_posix(),
                    xyz_sha256=sha256(xyz)))
            if plan['previews']:
                render_comparison(folder / 'comparison.png', clouds, f'{DATASETS[dataset]} / {sample["shape"]}')
            print(f'Exported {dataset}/{sample["shape"]}', flush=True)
    with (run_dir / 'files.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / 'README.txt').write_text(
        'Matched 4x point-cloud examples\n\n'
        'Each dataset/shape folder contains the original input_2048.xyz, gt_8192.xyz,\n'
        'gt_mesh.off, and one 8192-point prediction per model, plus matching PLY files.\n'
        'Open PLY/XYZ in CloudCompare or MeshLab. comparison.png has three views;\n'
        'each row uses identical coordinates, physical scale and camera for all clouds.\n'
        'No GT alignment, independent cloud normalization or display subsampling is applied.\n'
        'Pairing means the same object, not point-by-point correspondence with GT.\n\n'
        'files.csv uses portable relative paths. run.json records shapes, seeds, data\n'
        'hashes, original training config and best checkpoint hashes/epochs.\n'
        'jobs/ contains separate inference logs/results and training source manifests.\n'
        'No checkpoints or training data H5 are included in this export.\n', encoding='utf-8')


def package_run(run_dir):
    archive = run_dir.with_suffix('.tar.gz')
    # Exclusive creation: neither previous results nor previous archives are replaced.
    with tarfile.open(archive, 'x:gz') as stream:
        stream.add(run_dir, arcname=run_dir.name)
    return archive


def execute(args):
    plan = build_plan(args)
    if args.plan:
        import json
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    run_dir = new_run_directory(args.output_root.expanduser().resolve(), args.tag)
    plan.update(status='running', created_at=datetime.now().astimezone().isoformat())
    write_json(run_dir / 'run.json', plan)
    print(f'Test directory: {run_dir}', flush=True)
    try:
        snapshot_inputs(run_dir, plan)
        write_json(run_dir / 'run.json', plan)
        env = child_environment()
        for job in plan['jobs']:
            job_path = run_dir / job['job_file']
            print(f'Starting {job["dataset"]}/{job["model"]}, best epoch {job["best_epoch"]}', flush=True)
            run_child(['--worker-job', str(job_path)], job_path.with_suffix('.log'), env)
        run_child(['--export-run', str(run_dir)], run_dir / 'export.log', env)
        plan['status'] = 'completed'
        plan['finished_at'] = datetime.now().astimezone().isoformat()
        write_json(run_dir / 'run.json', plan)
        if plan['archive']:
            archive = package_run(run_dir)
            print(f'Download archive: {archive}', flush=True)
    except BaseException as error:
        plan.update(status='failed', error=str(error) or type(error).__name__)
        write_json(run_dir / 'run.json', plan)
        raise
    print(f'EXAMPLE TEST COMPLETED: {run_dir}', flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--training-run', type=Path, default=DEFAULT_RUN)
    result.add_argument('--prepared-root', type=Path, help='Prepared data root containing PU1K/ and PU-GAN/')
    result.add_argument('--datasets', choices=DATASETS, nargs='+', default=list(DATASETS))
    result.add_argument('--models', choices=DEFAULT_MODELS, nargs='+', default=list(DEFAULT_MODELS))
    result.add_argument('--num-examples', type=int, default=5, help='Fixed random sample size per dataset')
    result.add_argument('--shapes', nargs='+', help='Exact test shape stems (or .xyz names); overrides sample size')
    result.add_argument('--seed', type=int, default=42, help='Example selection and per-shape inference seed')
    result.add_argument('--output-root', type=Path, default=ROOT / 'compare/test-runs')
    result.add_argument('--tag', default='examples', help='Run label: ASCII letters, digits, hyphen, underscore')
    result.add_argument('--no-previews', action='store_true')
    result.add_argument('--no-archive', action='store_true')
    modes = result.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true', help='Validate files/hashes and print plan without inference/writes')
    modes.add_argument('--list-shapes', action='store_true', help='List available paired test shapes without loading models')
    modes.add_argument('--worker-job', type=Path, help=argparse.SUPPRESS)
    modes.add_argument('--export-run', type=Path, help=argparse.SUPPRESS)
    return result


def main():
    options = parser()
    args = options.parse_args()
    if args.worker_job:
        infer_job(args.worker_job)
    elif args.export_run:
        export_run(args.export_run)
    else:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
            options.error('--tag must contain only ASCII letters, digits, hyphen and underscore')
        if len(set(args.datasets)) != len(args.datasets) or len(set(args.models)) != len(args.models):
            options.error('Duplicate datasets/models are not allowed')
        if args.list_shapes:
            prepared = prepared_root(args)
            for dataset in args.datasets:
                print(f'[{dataset}]')
                print('\n'.join(available_shapes(prepared / DATASETS[dataset] / 'test')))
        else:
            execute(args)


if __name__ == '__main__':
    main()
