"""Run matched retrained/pretrained examples in a new, portable folder.

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

from retrain_common import DEFAULT_MODELS, NEW_PUGCS_MODELS, ROOT, read_json, sha256, stable_seed, write_json

DATASETS = {'pu1k': 'PU1K', 'pugan': 'PU-GAN'}
FULL_TEST_COUNTS = {'pu1k': 127, 'pugan': 27}
DEFAULT_RUN = ROOT / 'compare/retrain-runs/train-seed42-20261123-161953-503869'
PRETRAINED_MODEL = 'PU-Gaussian-pretrained'
REPKPU_PRETRAINED_MODEL = 'RepKPU-pretrained'
PUCRN_PRETRAINED_MODEL = 'PU-CRN-pretrained'
PRETRAINED_MODELS = {PRETRAINED_MODEL: 'PU-Gaussian', REPKPU_PRETRAINED_MODEL: 'RepKPU',
                     PUCRN_PRETRAINED_MODEL: 'E0'}
EXAMPLE_MODELS = (*DEFAULT_MODELS, *NEW_PUGCS_MODELS, *PRETRAINED_MODELS)
PRETRAINED_ROOT = ROOT / 'compare/PU-Gaussian-main/pretrained_model'
REPKPU_PRETRAINED_ROOT = ROOT / 'compare/RepKPU-main/pretrain'
PUCRN_PRETRAINED_ROOT = ROOT / 'baseline/3DVision-main/PointUpsampling/PUCRN/model/release'
PUCRN_CHECKPOINT = dict(filename='model.pth',
    sha256='900070c91fd8d4e9c34191d17795e32fb37af0e284d5b4b0c4434dd7fe9870a6',
    origin='https://github.com/hikvision-research/3DVision/blob/main/PointUpsampling/PUCRN/model/release/model.pth')
# Pin the official current-model checkpoints before any evaluation. PU1K has
# three released experiments; exp1 is fixed here, not chosen using our test CD.
REPKPU_CHECKPOINTS = {
    'pu1k': dict(filename='pu1k_exp1_best.pth', variant='official-current-pu1k-exp1',
                file_id='1Y9JXbVHoBATXd6XK_Tv3kTVnY5Owe4_-',
                sha256='53629847526615c5cb9a036b88412b659698eb7fa69b4d4a50ffc69761534266'),
    'pugan': dict(filename='pugan_best.pth', variant='official-current-pugan',
                 file_id='1vYKh5_wpOM8_S7eXe8IFy1j4up4RHjca',
                 sha256='cb917b6905668766ec2bf7954919b1dd45f4ff9dad6338afcc2f278ac5cedd56'),
}


def official_checkpoint(args, model, dataset):
    if model == PRETRAINED_MODEL:
        checkpoint = args.pretrained_root.expanduser().resolve() / f'pu_gaussian_{dataset}_Best.pth'
        return (checkpoint, sha256(checkpoint),
                'https://github.com/mvg-inatech/PU-Gaussian/blob/main/pretrained_model/' + checkpoint.name,
                'official-dataset-specific')
    if model == PUCRN_PRETRAINED_MODEL:
        root = getattr(args, 'pucrn_pretrained_root', PUCRN_PRETRAINED_ROOT)
        checkpoint = root.expanduser().resolve() / PUCRN_CHECKPOINT['filename']
        digest = sha256(checkpoint)
        if digest != PUCRN_CHECKPOINT['sha256']:
            raise ValueError(f'PU-CRN official checkpoint hash mismatch: {checkpoint}')
        # One released checkpoint, not two dataset-specific training runs.
        return (checkpoint, digest, PUCRN_CHECKPOINT['origin'],
                f'official-shared-release-on-{dataset}-PU1K-reference')
    spec = REPKPU_CHECKPOINTS[dataset]
    root = getattr(args, 'repkpu_pretrained_root', REPKPU_PRETRAINED_ROOT)
    checkpoint = root.expanduser().resolve() / spec['filename']
    digest = sha256(checkpoint)
    if digest != spec['sha256']:
        raise ValueError(f'RepKPU official checkpoint hash mismatch: {checkpoint}; '
                         'use the documented PU1K exp1 / current PU-GAN release')
    return checkpoint, digest, 'https://drive.google.com/file/d/' + spec['file_id'] + '/view', spec['variant']


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
    full_test = getattr(args, 'all_shapes', False)
    if full_test and args.shapes:
        raise ValueError('--all-shapes cannot be combined with --shapes')
    selections, jobs = {}, []
    for dataset in args.datasets:
        source = prepared / DATASETS[dataset] / 'test'
        available = available_shapes(source)
        if full_test:
            expected_count = FULL_TEST_COUNTS[dataset]
            if len(available) != expected_count:
                raise ValueError(f'{dataset}: full test requires exactly {expected_count} objects, got {len(available)}')
            names = available
            for subfolder, suffix in [('gt_8192', '.xyz'), ('meshes', '.off')]:
                if {path.stem for path in (source / subfolder).glob('*' + suffix)} != set(names):
                    raise ValueError(f'{dataset}: full test input/GT/mesh name sets differ')
        else:
            names = select_shapes(available, args.shapes, args.num_examples, args.seed, dataset)
        selections[dataset] = []
        for name in names:
            paths = dict(input=source / 'input_2048' / f'{name}.xyz',
                         gt=source / 'gt_8192' / f'{name}.xyz',
                         mesh=source / 'meshes' / f'{name}.off')
            selections[dataset].append(dict(shape=name, sources={k: str(p) for k, p in paths.items()},
                                           sha256={k: sha256(p) for k, p in paths.items()}))
        for model in args.models:
            pretrained = model in PRETRAINED_MODELS
            native_model = PRETRAINED_MODELS.get(model, model)
            folder = training / dataset / native_model
            config = read_json(folder / 'config.json')
            status = read_json(folder / 'status.json')
            if config['model'] != native_model or config['dataset'] != dataset or status['status'] != 'completed':
                raise ValueError(f'Model/config/status mismatch: {folder}')
            if pretrained:
                checkpoint, digest, origin, variant = official_checkpoint(args, model, dataset)
            else:
                checkpoint = folder / 'best.pth'
                digest, origin, variant = sha256(checkpoint), str(folder / 'status.json'), 'retrained'
            if not pretrained and digest != status['best_sha256']:
                raise ValueError(f'Best checkpoint hash changed: {checkpoint}')
            if native_model in NEW_PUGCS_MODELS and status.get('test_evaluated') is False:
                # Explicit evaluation of a structure selected using validation.
                # Bind it to the exact prepared manifest used during training.
                training_manifest = read_json(folder / 'manifest.json')
                if training_manifest['data_manifest_sha256'] != sha256(prepared / 'manifest.json'):
                    raise ValueError('Prepared data changed since structure training')
                protocol = dict(status='completed', dataset=DATASETS[dataset], expected_shapes=len(available),
                    source_hashes={group: {name + suffix: sha256(source / subfolder / (name + suffix)) for name in available}
                        for group, subfolder, suffix in [('inputs', 'input_2048', '.xyz'), ('gt', 'gt_8192', '.xyz'), ('meshes', 'meshes', '.off')]})
            else:
                protocol = read_json(folder / 'test/metrics/protocol.json')
            if full_test:
                if (protocol.get('status') != 'completed' or protocol.get('expected_shapes') != expected_count
                        or protocol.get('dataset') != DATASETS[dataset]):
                    raise ValueError(f'{dataset}: reference protocol is not the completed {expected_count}-object formal test')
                for group, suffix in [('inputs', '.xyz'), ('gt', '.xyz'), ('meshes', '.off')]:
                    if set(protocol['source_hashes'][group]) != {name + suffix for name in names}:
                        raise ValueError(f'{dataset}: full {group} names differ from the formal test protocol')
            for sample in selections[dataset]:
                for key, group, suffix in [('input', 'inputs', '.xyz'), ('gt', 'gt', '.xyz'),
                                           ('mesh', 'meshes', '.off')]:
                    expected = protocol['source_hashes'][group][sample['shape'] + suffix]
                    if sample['sha256'][key] != expected:
                        raise ValueError(f'{dataset}/{sample["shape"]}: {key} differs from {model} formal test data')
            jobs.append(dict(dataset=dataset, model=model, checkpoint=str(checkpoint),
                checkpoint_sha256=digest, best_epoch=None if pretrained else status['best_epoch'], config=config,
                checkpoint_source='official-pretrained' if pretrained else 'retrained-best',
                checkpoint_origin=origin, checkpoint_variant=variant,
                checkpoint_format='net_state_dict' if model == PUCRN_PRETRAINED_MODEL else 'state_dict',
                adapter_config_source=str(folder / 'config.json'),
                training_manifest=None if pretrained else str(folder / 'manifest.json')))
    return dict(status='planned', training_run=str(training), prepared_root=str(prepared),
        seed=args.seed, models=args.models, selections=selections, jobs=jobs,
        previews=not args.no_previews, archive=not args.no_archive, metrics=args.metrics,
        test_scope='full-test-set' if full_test else 'selected-examples',
        purpose=('Full fixed test sets; official pretrained results are separate from formal retraining tables'
                 if full_test else 'Diagnostic matched examples; official pretrained results are separate from formal retraining tables'),
        entrypoint=str(Path(getattr(args, 'entrypoint', __file__)).resolve()),
        entrypoint_sha256=sha256(getattr(args, 'entrypoint', __file__)),
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
        if job['training_manifest']:
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


def inference_state_dict(checkpoint, checkpoint_format='state_dict'):
    if checkpoint_format == 'net_state_dict':
        if not isinstance(checkpoint, dict) or set(checkpoint) != {'net_state_dict'}:
            raise ValueError('Expected the official PU-CRN net_state_dict checkpoint container')
        checkpoint = checkpoint['net_state_dict']
    elif checkpoint_format != 'state_dict':
        raise ValueError(f'Unsupported inference checkpoint format: {checkpoint_format}')
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError('Inference state_dict must be a nonempty mapping')
    return checkpoint


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
    checkpoint = torch.load(job['checkpoint'], map_location='cuda', weights_only=True)
    state = inference_state_dict(checkpoint, job.get('checkpoint_format', 'state_dict'))
    adapter.net.load_state_dict(state, strict=True)
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
        checkpoint_sha256=job['checkpoint_sha256'], checkpoint_source=job['checkpoint_source'],
        best_epoch=job['best_epoch'], native_arguments=vars(adapter.args),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0)))


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
    rows, metric_rows = [], []
    jobs = {(job['dataset'], job['model']): job for job in plan['jobs']}
    if plan.get('metrics'):
        from evaluate_pugaussian_protocol import mesh_scene, p2f_scores, point_scores
    for dataset, samples in plan['selections'].items():
        for sample in samples:
            folder = run_dir / dataset / sample['shape']
            clouds = []
            target = read_points(folder / 'gt_8192.xyz', 8192)
            scene = mesh_scene(folder / 'gt_mesh.off') if plan.get('metrics') else None
            labels = [('Input', 'input_2048', 2048), ('GT', 'gt_8192', 8192)]
            labels += [(model, model, 8192) for model in plan['models']]
            for label, basename, count in labels:
                xyz = folder / f'{basename}.xyz'
                points = read_points(xyz, count)
                ply = xyz.with_suffix('.ply')
                write_ply(ply, points)
                display = label
                if label in PRETRAINED_MODELS:
                    display = label.removesuffix('-pretrained') + '\nOfficial pretrained'
                elif label in ('PU-Gaussian', 'RepKPU'):
                    display = f'{label}\nRetrained (epoch {jobs[dataset, label]["best_epoch"]})'
                clouds.append((display, points))
                rows.append(dict(dataset=dataset, shape=sample['shape'], label=label, points=count,
                    xyz=xyz.relative_to(run_dir).as_posix(), ply=ply.relative_to(run_dir).as_posix(),
                    xyz_sha256=sha256(xyz)))
                if plan.get('metrics') and label in plan['models']:
                    scores = point_scores(points, target)
                    scores.update(p2f_scores(points, scene))
                    job = jobs[dataset, label]
                    metric_rows.append(dict(dataset=dataset, model=label, shape=sample['shape'],
                        checkpoint_source=job['checkpoint_source'], best_epoch=job['best_epoch'],
                        **{key + '_x1000': value * 1000 for key, value in scores.items()}))
            if plan['previews']:
                render_comparison(folder / 'comparison.png', clouds, f'{DATASETS[dataset]} / {sample["shape"]}')
            print(f'Exported {dataset}/{sample["shape"]}', flush=True)
    with (run_dir / 'files.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scope_description = ('Full fixed test sets: all 127 PU1K / 27 PU-GAN objects for each selected dataset.\n'
                         'Official pretrained results remain separate from formal retraining rankings.\n'
                         if plan.get('test_scope') == 'full-test-set' else
                         'Example metrics are not full-dataset or formal retraining rankings.\n')
    (run_dir / 'README.txt').write_text(
        'Matched 4x point-cloud test\n\n'
        'Each dataset/shape folder contains the original input_2048.xyz, gt_8192.xyz,\n'
        'gt_mesh.off, and one 8192-point prediction per model, plus matching PLY files.\n'
        'Open PLY/XYZ in CloudCompare or MeshLab. If previews are enabled,\n'
        'comparison.png has three views;\n'
        'each row uses identical coordinates, physical scale and camera for all clouds.\n'
        'No GT alignment, independent cloud normalization or display subsampling is applied.\n'
        'Pairing means the same object, not point-by-point correspondence with GT.\n\n'
        'files.csv uses portable relative paths. run.json records shapes, seeds, data\n'
        'hashes, adapter config, weight sources and checkpoint hashes/epochs.\n'
        'E0/E6, PU-Gaussian and RepKPU labels use our retrained checkpoints. Labels\n'
        'ending in -pretrained use official checkpoints, with original epoch\n'
        'unknown (null). RepKPU uses fixed PU1K exp1 and current PU-GAN releases;\n'
        'the separate PU-GAN paper-original architecture is not used here.\n'
        'PU-CRN-pretrained uses the same official release model.pth on both test\n'
        'datasets. Its supplied evaluation targets PU1K; it is not a separately\n'
        'released PU-GAN checkpoint. The container has net_state_dict only.\n'
        'jobs/ contains inference logs/results; training manifests apply only to\n'
        'our retrained weights, not the official pretrained checkpoint.\n'
        'With --metrics, metrics/ contains diagnostic per-shape/mean CD, squared\n'
        'HD and saved-coordinate P2F (display values x1000), using the same evaluator.\n'
        + scope_description +
        'No checkpoints or training data H5 are included in this export.\n', encoding='utf-8')
    if metric_rows:
        export_metrics(run_dir, plan, metric_rows)


def export_metrics(run_dir, plan, rows):
    import numpy as np
    from evaluate_pugaussian_protocol import write_csv

    destination = run_dir / 'metrics'
    destination.mkdir(exist_ok=False)
    keys = [key for key in rows[0] if key.endswith('_x1000')]
    summary = []
    for dataset in plan['selections']:
        for model in plan['models']:
            selected = [row for row in rows if row['dataset'] == dataset and row['model'] == model]
            summary.append(dict(dataset=dataset, model=model, n_shapes=len(selected),
                checkpoint_source=selected[0]['checkpoint_source'], best_epoch=selected[0]['best_epoch'],
                **{key: float(np.mean([row[key] for row in selected], dtype=np.float64)) for key in keys}))
    write_csv(destination / 'per_shape.csv', rows)
    write_csv(destination / 'summary.csv', summary)
    write_json(destination / 'protocol.json', dict(status='completed', purpose=plan['purpose'],
        test_scope=plan.get('test_scope', 'selected-examples'),
        n_shapes={dataset: len(samples) for dataset, samples in plan['selections'].items()},
        normalization='prediction and GT independently centered / max-radius normalized in float32 for CD/HD',
        CD='sum of bidirectional mean squared NN distances', HD='max of bidirectional maximum squared NN distances',
        P2F='saved-coordinate mean unsigned Euclidean point-to-triangle distance',
        P2F_P99='per-shape P99 followed by equal-shape mean', display_scale=1000,
        evaluator_sha256=sha256(Path(__file__).with_name('evaluate_pugaussian_protocol.py')),
        checkpoints=[{key: job[key] for key in ('dataset', 'model', 'checkpoint_sha256',
                    'checkpoint_source', 'checkpoint_origin', 'checkpoint_variant', 'checkpoint_format',
                    'best_epoch')} for job in plan['jobs']]))
    for row in summary:
        print(f'{row["dataset"]}/{row["model"]}: CD={row["CD_x1000"]:.6f}, '
              f'HD={row["HD_x1000"]:.6f}, P2F={row["P2F_x1000"]:.6f} (x1000)', flush=True)


def package_run(run_dir, archive=None):
    run_dir = Path(run_dir)
    archive = Path(archive) if archive else run_dir.with_suffix('.tar.gz')
    partial = archive.with_name(archive.name + '.partial')
    if archive.exists():
        raise FileExistsError(archive)
    paths = [run_dir, *sorted(run_dir.rglob('*'))]
    total = sum(p.stat().st_size for p in paths if p.is_file())
    print(f'Packaging {total / 1024**2:.1f} MiB; gzip level 1. '
          'Inference/export are complete; no GPU work remains.', flush=True)
    print(f'Temporary archive: {partial}', flush=True)
    done, reported = 0, time.monotonic()
    # An interruption leaves only an explicitly named .partial file. Final
    # publication is exclusive on both Windows and POSIX, preserving old files.
    with tarfile.open(partial, 'x:gz', compresslevel=1) as stream:
        for path in paths:
            stream.add(path, arcname=path.relative_to(run_dir.parent).as_posix(), recursive=False)
            if path.is_file():
                done += path.stat().st_size
            if time.monotonic() - reported >= 5:
                print(f'Packaging: {done / max(total, 1):.1%} '
                      f'({done / 1024**2:.1f}/{total / 1024**2:.1f} MiB)', flush=True)
                reported = time.monotonic()
    if os.name == 'nt':
        partial.rename(archive)  # Windows rename refuses an existing destination.
    else:
        os.link(partial, archive)  # POSIX rename could overwrite a competing archive.
        partial.unlink()
    print(f'Packaging: 100%; archive ready ({archive.stat().st_size / 1024**2:.1f} MiB)', flush=True)
    return archive


def validate_export_for_packaging(run_dir):
    """Validate existing artifacts without importing models or recomputing metrics."""
    plan = read_json(run_dir / 'run.json')
    if plan.get('status') not in ('completed', 'failed') or not plan.get('finished_at'):
        raise ValueError('Inference/export did not finish; package-only cannot resume inference')

    def rows(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    expected, expected_predictions = set(), set()
    for dataset, samples in plan['selections'].items():
        for sample in samples:
            folder = run_dir / dataset / sample['shape']
            for label in ('Input', 'GT', *plan['models']):
                expected.add((dataset, sample['shape'], label))
            expected_predictions.update((dataset, model, sample['shape']) for model in plan['models'])
            for key, filename in [('input', 'input_2048.xyz'), ('gt', 'gt_8192.xyz'), ('mesh', 'gt_mesh.off')]:
                if sha256(folder / filename) != sample['sha256'][key]:
                    raise ValueError(f'Export source hash mismatch: {folder / filename}')
    files = rows(run_dir / 'files.csv')
    if len(files) != len(expected) or {(r['dataset'], r['shape'], r['label']) for r in files} != expected:
        raise ValueError('Incomplete exported file inventory')
    for row in files:
        for key in ('xyz', 'ply'):
            path = (run_dir / row[key]).resolve()
            if not path.is_relative_to(run_dir) or not path.is_file():
                raise ValueError(f'Missing or invalid exported {key}: {row[key]}')
        if sha256(run_dir / row['xyz']) != row['xyz_sha256']:
            raise ValueError(f'Exported XYZ hash mismatch: {row["xyz"]}')
    if plan.get('metrics'):
        protocol = read_json(run_dir / 'metrics/protocol.json')
        per_shape = rows(run_dir / 'metrics/per_shape.csv')
        summary = rows(run_dir / 'metrics/summary.csv')
        counts = {dataset: len(samples) for dataset, samples in plan['selections'].items()}
        if protocol.get('status') != 'completed' or protocol.get('n_shapes') != counts:
            raise ValueError('Metric export did not complete')
        if (len(per_shape) != len(expected_predictions)
                or {(r['dataset'], r['model'], r['shape']) for r in per_shape} != expected_predictions):
            raise ValueError('Incomplete per-shape metrics')
        expected_summary = {(ds, model, str(count)) for ds, count in counts.items() for model in plan['models']}
        if (len(summary) != len(expected_summary)
                or {(r['dataset'], r['model'], r['n_shapes']) for r in summary} != expected_summary):
            raise ValueError('Incomplete summary metrics')
    return plan


def archive_completed_run(run_dir, plan, archive=None):
    try:
        result = package_run(run_dir, archive)
    except BaseException as error:
        # A compression failure must not relabel completed inference as failed.
        plan['archive_error'] = str(error) or type(error).__name__
        write_json(run_dir / 'run.json', plan)
        print(f'Inference/export remain completed. To package only: '
              f'python compare/test_examples.py --package-run "{run_dir}"', flush=True)
        raise
    print(f'Download archive: {result}', flush=True)
    return result


def repackage_run(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    print(f'Checking existing export: {run_dir}', flush=True)
    plan = validate_export_for_packaging(run_dir)
    # Older launchers labelled the whole run failed when compression was interrupted.
    if plan.get('error'):
        plan['recovered_archive_error'] = plan.pop('error')
    plan.pop('archive_error', None)
    plan['status'] = 'completed'
    plan['repackaged_at'] = datetime.now().astimezone().isoformat()
    write_json(run_dir / 'run.json', plan)
    archive = run_dir.with_suffix('.tar.gz')
    if archive.exists() or archive.with_name(archive.name + '.partial').exists():
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        archive = run_dir.with_name(run_dir.name + f'-repacked-{stamp}.tar.gz')
    result = archive_completed_run(run_dir, plan, archive)
    print(f'PACKAGE COMPLETED: {result}', flush=True)
    return result


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
            checkpoint_info = ('official pretrained, original epoch unknown'
                               if job['checkpoint_source'] == 'official-pretrained'
                               else f'best epoch {job["best_epoch"]}')
            print(f'Starting {job["dataset"]}/{job["model"]}, {checkpoint_info}, '
                  f'{len(job["shapes"])} test objects', flush=True)
            run_child(['--worker-job', str(job_path)], job_path.with_suffix('.log'), env)
        run_child(['--export-run', str(run_dir)], run_dir / 'export.log', env)
        plan['status'] = 'completed'
        plan['finished_at'] = datetime.now().astimezone().isoformat()
        write_json(run_dir / 'run.json', plan)
    except BaseException as error:
        plan.update(status='failed', error=str(error) or type(error).__name__)
        write_json(run_dir / 'run.json', plan)
        raise
    if plan['archive']:
        archive_completed_run(run_dir, plan)
    completed = 'FULL TEST COMPLETED' if plan['test_scope'] == 'full-test-set' else 'EXAMPLE TEST COMPLETED'
    print(f'{completed}: {run_dir}', flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--training-run', type=Path, default=DEFAULT_RUN)
    result.add_argument('--prepared-root', type=Path, help='Prepared data root containing PU1K/ and PU-GAN/')
    result.add_argument('--datasets', choices=DATASETS, nargs='+', default=list(DATASETS))
    result.add_argument('--models', choices=EXAMPLE_MODELS, nargs='+', default=list(DEFAULT_MODELS),
                        help='Labels ending in -pretrained use official weights; other labels are retrained')
    result.add_argument('--pretrained-root', type=Path, default=PRETRAINED_ROOT,
                        help='Folder containing official pu_gaussian_{pu1k,pugan}_Best.pth')
    result.add_argument('--repkpu-pretrained-root', type=Path, default=REPKPU_PRETRAINED_ROOT,
                        help='Folder containing pinned official pu1k_exp1_best.pth and pugan_best.pth')
    result.add_argument('--pucrn-pretrained-root', type=Path, default=PUCRN_PRETRAINED_ROOT,
                        help='Folder containing the pinned shared official PU-CRN model.pth')
    result.add_argument('--num-examples', type=int, default=5, help='Fixed random sample size per dataset')
    result.add_argument('--shapes', nargs='+', help='Exact test shape stems (or .xyz names); overrides sample size')
    result.add_argument('--all-shapes', action='store_true', help='Use all 127 PU1K / 27 PU-GAN test objects; no sampling')
    result.add_argument('--seed', type=int, default=42, help='Example selection and per-shape inference seed')
    result.add_argument('--output-root', type=Path, default=ROOT / 'compare/test-runs')
    result.add_argument('--tag', default='examples', help='Run label: ASCII letters, digits, hyphen, underscore')
    result.add_argument('--no-previews', action='store_true')
    result.add_argument('--no-archive', action='store_true')
    result.add_argument('--metrics', action='store_true', help='Also compute diagnostic CD/HD/P2F on the chosen examples')
    modes = result.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true', help='Validate files/hashes and print plan without inference/writes')
    modes.add_argument('--list-shapes', action='store_true', help='List available paired test shapes without loading models')
    modes.add_argument('--worker-job', type=Path, help=argparse.SUPPRESS)
    modes.add_argument('--export-run', type=Path, help=argparse.SUPPRESS)
    modes.add_argument('--package-run', type=Path, help='Package an existing completed export without inference')
    return result


def main():
    options = parser()
    args = options.parse_args()
    if args.worker_job:
        infer_job(args.worker_job)
    elif args.export_run:
        export_run(args.export_run)
    elif args.package_run:
        repackage_run(args.package_run)
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
