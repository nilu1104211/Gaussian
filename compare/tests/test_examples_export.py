"""CPU coverage of matched data, integrity gates, isolated runs and portable export."""
import csv
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_examples as examples
from retrain_common import read_json, sha256, write_json


@pytest.fixture
def options(tmp_path):
    training, prepared = tmp_path / 'training', tmp_path / 'prepared'
    models = ['E0', 'E6']
    write_json(training / 'run.json', dict(status='completed', smoke=False, prepared=str(prepared)))
    source = prepared / 'PU1K/test'
    for folder in ('input_2048', 'gt_8192', 'meshes'):
        (source / folder).mkdir(parents=True)
    hashes = dict(inputs={}, gt={}, meshes={})
    for index, shape in enumerate(('shape.a', 'shape.b', 'shape.c')):
        points = np.random.default_rng(index).normal(size=(8192, 3)).astype(np.float32) + index * 3
        np.savetxt(source / 'input_2048' / f'{shape}.xyz', points[:2048], fmt='%.10g')
        np.savetxt(source / 'gt_8192' / f'{shape}.xyz', points, fmt='%.10g')
        (source / 'meshes' / f'{shape}.off').write_text('OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
        for group, folder, suffix in [('inputs', 'input_2048', '.xyz'), ('gt', 'gt_8192', '.xyz'),
                                      ('meshes', 'meshes', '.off')]:
            hashes[group][shape + suffix] = sha256(source / folder / (shape + suffix))
    for model in models:
        folder = training / 'pu1k' / model
        folder.mkdir(parents=True)
        (folder / 'best.pth').write_bytes(b'test checkpoint ' + model.encode())
        write_json(folder / 'config.json', dict(dataset='pu1k', model=model, seed=42))
        write_json(folder / 'status.json', dict(status='completed', best_epoch=5,
                                               best_sha256=sha256(folder / 'best.pth')))
        write_json(folder / 'test/metrics/protocol.json', dict(source_hashes=hashes))
        write_json(folder / 'manifest.json', dict(sources={'native.py': 'recorded-source-hash'}))
    return examples.parser().parse_args(['--training-run', str(training), '--prepared-root', str(prepared),
        '--datasets', 'pu1k', '--models', *models, '--num-examples', '2',
        '--output-root', str(tmp_path / 'results'), '--no-previews'])


def test_selection_is_shared_reproducible_and_explicit_names_work(options):
    plan = examples.build_plan(options)
    names = [s['shape'] for s in plan['selections']['pu1k']]
    assert len(names) == 2
    assert examples.build_plan(options)['selections'] == plan['selections']
    options.models.reverse()
    assert examples.build_plan(options)['selections'] == plan['selections']
    options.shapes = ['shape.b.xyz', 'shape.a']
    assert [s['shape'] for s in examples.build_plan(options)['selections']['pu1k']] == ['shape.b', 'shape.a']
    options.shapes = ['unknown']
    with pytest.raises(ValueError, match='unknown shapes'):
        examples.build_plan(options)


def test_wrong_labels_and_changed_checkpoints_fail_before_creating_results(options):
    target = options.prepared_root / 'PU1K/test/gt_8192/shape.b.xyz'
    options.shapes = ['shape.b']
    original = target.read_bytes()
    target.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='differs from .* formal test data'):
        examples.execute(options)
    assert not options.output_root.exists()
    target.write_bytes(original)
    checkpoint = options.training_run / 'pu1k/E6/best.pth'
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='checkpoint hash changed'):
        examples.execute(options)
    assert not options.output_root.exists()


def test_plan_has_no_writes_and_missing_gt_is_not_silently_skipped(options, capsys):
    options.plan = True
    examples.execute(options)
    assert 'planned' in capsys.readouterr().out
    assert not options.output_root.exists()
    (options.prepared_root / 'PU1K/test/gt_8192/shape.a.xyz').unlink()
    with pytest.raises(FileNotFoundError, match='Missing matching label'):
        examples.execute(options)


def test_two_datasets_keep_distinct_inputs_for_the_same_shape(options):
    # Same object names in both datasets do not imply identical fixed point sets.
    import shutil
    second = options.prepared_root / 'PU-GAN/test'
    shutil.copytree(options.prepared_root / 'PU1K/test', second)
    hashes = dict(inputs={}, gt={}, meshes={})
    for path in (second / 'input_2048').glob('*.xyz'):
        points = np.loadtxt(path)
        np.savetxt(path, points + .03125)
    for group, folder, suffix in [('inputs', 'input_2048', '.xyz'), ('gt', 'gt_8192', '.xyz'),
                                  ('meshes', 'meshes', '.off')]:
        hashes[group] = {p.name: sha256(p) for p in (second / folder).glob('*' + suffix)}
    for model in options.models:
        folder = options.training_run / 'pugan' / model
        shutil.copytree(options.training_run / 'pu1k' / model, folder)
        config = read_json(folder / 'config.json')
        config['dataset'] = 'pugan'
        write_json(folder / 'config.json', config)
        write_json(folder / 'test/metrics/protocol.json', dict(source_hashes=hashes))
    options.datasets = ['pu1k', 'pugan']
    options.shapes = ['shape.a']
    plan = examples.build_plan(options)
    assert len(plan['jobs']) == 4
    run = examples.new_run_directory(options.output_root, 'both')
    examples.snapshot_inputs(run, plan)
    for dataset in options.datasets:
        assert sha256(run / dataset / 'shape.a/input_2048.xyz') == plan['selections'][dataset][0]['sha256']['input']
        jobs = [job for job in plan['jobs'] if job['dataset'] == dataset]
        assert all(job['shapes'] == ['shape.a'] for job in jobs)
    assert sha256(run / 'pu1k/shape.a/input_2048.xyz') != sha256(run / 'pugan/shape.a/input_2048.xyz')


def test_repeated_runs_export_exact_labels_separate_predictions_and_portable_archives(options, monkeypatch):
    # Substitute only the unavailable native CUDA computation, then use real copying,
    # metadata, XYZ/PLY export, packaging and run lifecycle code end to end.
    def child(arguments, log_path, env):
        log_path.write_text('CPU fixture worker\n')
        if arguments[0] == '--export-run':
            examples.export_run(Path(arguments[1]))
            return
        job = read_json(arguments[1])
        for shape in job['shapes']:
            folder = Path(job['run_dir']) / job['dataset'] / shape
            points = np.loadtxt(folder / 'gt_8192.xyz', dtype=np.float32)
            points += .125 if job['model'] == 'E6' else .25
            np.savetxt(folder / f'{job["model"]}.xyz', points, fmt='%.10g')
    monkeypatch.setattr(examples, 'run_child', child)
    source_hashes = {str(p): sha256(p) for root in (options.prepared_root, options.training_run)
                     for p in root.rglob('*') if p.is_file()}
    examples.execute(options)
    examples.execute(options)
    runs = sorted(p for p in options.output_root.iterdir() if p.is_dir())
    assert len(runs) == 2 and runs[0] != runs[1]
    for run in runs:
        plan = read_json(run / 'run.json')
        assert plan['status'] == 'completed'
        with (run / 'files.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2 * 4
        for row in rows:
            assert not Path(row['xyz']).is_absolute()
            assert sha256(run / row['xyz']) == row['xyz_sha256']
            ply = (run / row['ply']).read_text()
            assert f'element vertex {row["points"]}\n' in ply
        for sample in plan['selections']['pu1k']:
            folder = run / 'pu1k' / sample['shape']
            assert sha256(folder / 'input_2048.xyz') == sample['sha256']['input']
            assert sha256(folder / 'gt_8192.xyz') == sample['sha256']['gt']
            assert sha256(folder / 'gt_mesh.off') == sample['sha256']['mesh']
            assert sha256(folder / 'E0.xyz') != sha256(folder / 'E6.xyz')
            # PLY stores the same coordinates as the XYZ, including any offset.
            assert np.array_equal(np.loadtxt(folder / 'E6.xyz', dtype=np.float32),
                                  np.loadtxt(folder / 'E6.ply', skiprows=7, dtype=np.float32))
        with tarfile.open(run.with_suffix('.tar.gz')) as archive:
            names = archive.getnames()
            assert f'{run.name}/files.csv' in names
            assert all(not n.endswith(('.pth', '.h5')) for n in names)
        with pytest.raises(FileExistsError):
            examples.package_run(run)
    assert {path: sha256(path) for path in source_hashes} == source_hashes


def test_failed_worker_preserves_failed_run_without_success_archive(options, monkeypatch):
    def fail(*unused):
        raise RuntimeError('native CUDA fixture failure')
    monkeypatch.setattr(examples, 'run_child', fail)
    with pytest.raises(RuntimeError, match='native CUDA fixture failure'):
        examples.execute(options)
    run, = (p for p in options.output_root.iterdir() if p.is_dir())
    assert read_json(run / 'run.json')['status'] == 'failed'
    assert list(run.glob('pu1k/*/input_2048.xyz'))
    assert not list(options.output_root.glob('*.tar.gz'))


@pytest.mark.parametrize('legacy_final_archive', [False, True])
def test_interrupted_packaging_recovers_without_inference_or_metric_changes(options, monkeypatch, legacy_final_archive):
    import json
    import evaluate_pugaussian_protocol as metrics
    import test_repkpu_pretrained_full as full

    # GPU inference and optional mesh library are unnecessary for exercising the
    # real export, checksum validation, interruption and archive recovery path.
    def child(arguments, log_path, env):
        log_path.write_text('CPU fixture worker\n')
        if arguments[0] == '--export-run':
            examples.export_run(Path(arguments[1]))
            return
        job = read_json(arguments[1])
        for shape in job['shapes']:
            folder = Path(job['run_dir']) / job['dataset'] / shape
            points = np.loadtxt(folder / 'gt_8192.xyz', dtype=np.float32)
            np.savetxt(folder / f'{job["model"]}.xyz', points, fmt='%.10g')

    options.metrics = True
    monkeypatch.setattr(examples, 'run_child', child)
    monkeypatch.setattr(metrics, 'mesh_scene', lambda path: None)
    monkeypatch.setattr(metrics, 'p2f_scores', lambda points, scene: dict(P2F=0., P2F_P99=0.))
    original_add = tarfile.TarFile.add

    def interrupt(stream, name, *args, **kwargs):
        if Path(name).suffix == '.ply':
            raise KeyboardInterrupt
        return original_add(stream, name, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, 'add', interrupt)
    with pytest.raises(KeyboardInterrupt):
        examples.execute(options)
    run, = (p for p in options.output_root.iterdir() if p.is_dir())
    plan = read_json(run / 'run.json')
    assert plan['status'] == 'completed' and plan['archive_error'] == 'KeyboardInterrupt'
    assert not run.with_suffix('.tar.gz').exists()
    partial = run.with_suffix('.tar.gz.partial')
    assert partial.is_file()
    if legacy_final_archive:
        # Earlier launchers left a final-looking truncated gzip and marked the
        # entire run failed, even though finished_at and all exports were saved.
        partial.rename(run.with_suffix('.tar.gz'))
        plan.update(status='failed', error=plan.pop('archive_error'))
        write_json(run / 'run.json', plan)
    existing_archives = {p: p.read_bytes() for p in options.output_root.iterdir() if p.is_file()}
    artifacts = {p: sha256(p) for p in run.rglob('*') if p.is_file() and p.name != 'run.json'}
    monkeypatch.setattr(tarfile.TarFile, 'add', original_add)

    def no_worker(*args):
        pytest.fail('package-only must never launch inference or metric workers')

    monkeypatch.setattr(examples, 'run_child', no_worker)
    full.main(['--package-run', str(run)])
    assert read_json(run / 'run.json')['status'] == 'completed'
    assert artifacts == {p: sha256(p) for p in artifacts}
    assert existing_archives == {p: p.read_bytes() for p in existing_archives}
    archive, = options.output_root.glob('*-repacked-*.tar.gz')
    with tarfile.open(archive) as stream:
        names = stream.getnames()
        assert f'{run.name}/metrics/summary.csv' in names
        assert json.load(stream.extractfile(f'{run.name}/run.json'))['status'] == 'completed'

    summary = run / 'metrics/summary.csv'
    summary.write_text(summary.read_text(encoding='utf-8-sig').splitlines()[0] + '\n', encoding='utf-8-sig')
    with pytest.raises(ValueError, match='Incomplete summary metrics'):
        full.main(['--package-run', str(run)])
    assert list(options.output_root.glob('*-repacked-*.tar.gz')) == [archive]


def gaussian_fixture(options):
    import shutil
    folder = options.training_run / 'pu1k/PU-Gaussian'
    shutil.copytree(options.training_run / 'pu1k/E0', folder)
    write_json(folder / 'config.json', dict(dataset='pu1k', model='PU-Gaussian', seed=42))
    (folder / 'best.pth').write_bytes(b'retrained-gaussian')
    write_json(folder / 'status.json', dict(status='completed', best_epoch=105,
                                           best_sha256=sha256(folder / 'best.pth')))
    options.pretrained_root = options.training_run.parent / 'official-weights'
    options.pretrained_root.mkdir()
    for dataset in ('pu1k', 'pugan'):
        (options.pretrained_root / f'pu_gaussian_{dataset}_Best.pth').write_bytes(b'official-' + dataset.encode())
    return folder


def repkpu_fixture(options, monkeypatch):
    import shutil
    folder = options.training_run / 'pu1k/RepKPU'
    shutil.copytree(options.training_run / 'pu1k/E0', folder)
    write_json(folder / 'config.json', dict(dataset='pu1k', model='RepKPU', seed=42))
    options.repkpu_pretrained_root = options.training_run.parent / 'repkpu-weights'
    options.repkpu_pretrained_root.mkdir()
    for dataset, spec in examples.REPKPU_CHECKPOINTS.items():
        path = options.repkpu_pretrained_root / spec['filename']
        path.write_bytes(b'official-repkpu-' + dataset.encode())
        monkeypatch.setitem(examples.REPKPU_CHECKPOINTS, dataset, {**spec, 'sha256': sha256(path)})
    return folder


def pucrn_fixture(options, monkeypatch):
    options.pucrn_pretrained_root = options.training_run.parent / 'pucrn-release'
    options.pucrn_pretrained_root.mkdir()
    path = options.pucrn_pretrained_root / 'model.pth'
    path.write_bytes(b'official-shared-pucrn-release')
    monkeypatch.setitem(examples.PUCRN_CHECKPOINT, 'sha256', sha256(path))
    return options.training_run / 'pu1k/E0'


def test_pucrn_release_is_shared_and_never_claimed_as_retrained(options, monkeypatch):
    import shutil
    folder = pucrn_fixture(options, monkeypatch)
    second = options.training_run / 'pugan/E0'
    shutil.copytree(folder, second)
    write_json(second / 'config.json', dict(dataset='pugan', model='E0', seed=42))
    shutil.copytree(options.prepared_root / 'PU1K', options.prepared_root / 'PU-GAN')
    options.datasets = ['pu1k', 'pugan']
    options.models = [examples.PUCRN_PRETRAINED_MODEL]
    (folder / 'best.pth').unlink()
    (second / 'best.pth').unlink()
    plan = examples.build_plan(options)
    assert len(plan['jobs']) == 2
    assert len({j['checkpoint'] for j in plan['jobs']}) == 1
    assert len({j['checkpoint_sha256'] for j in plan['jobs']}) == 1
    for job in plan['jobs']:
        assert job['config']['model'] == 'E0'
        assert job['config']['dataset'] == job['dataset']
        assert job['checkpoint_source'] == 'official-pretrained'
        assert job['best_epoch'] is None and job['training_manifest'] is None
        assert job['checkpoint_format'] == 'net_state_dict'
        assert 'shared-release' in job['checkpoint_variant'] and 'PU1K-reference' in job['checkpoint_variant']
        assert job['checkpoint_origin'] == examples.PUCRN_CHECKPOINT['origin']
    Path(plan['jobs'][0]['checkpoint']).write_bytes(b'wrong-weight')
    with pytest.raises(ValueError, match='PU-CRN official checkpoint hash mismatch'):
        examples.execute(options)
    assert not options.output_root.exists()


def test_official_pucrn_container_is_unwrapped_explicitly_without_partial_loading():
    state = {'layer.weight': object()}
    assert examples.inference_state_dict({'net_state_dict': state}, 'net_state_dict') is state
    assert examples.inference_state_dict(state) is state
    for invalid in (state, {'net_state_dict': {}}, {'net_state_dict': state, 'unexpected': 1}):
        with pytest.raises(ValueError):
            examples.inference_state_dict(invalid, 'net_state_dict')
    with pytest.raises(ValueError, match='Unsupported'):
        examples.inference_state_dict(state, 'guessed-format')


def test_repkpu_official_sources_are_fixed_and_swapped_weights_fail_before_writes(options, monkeypatch):
    import shutil
    folder = repkpu_fixture(options, monkeypatch)
    second = options.training_run / 'pugan/RepKPU'
    shutil.copytree(folder, second)
    write_json(second / 'config.json', dict(dataset='pugan', model='RepKPU', seed=42))
    shutil.copytree(options.prepared_root / 'PU1K', options.prepared_root / 'PU-GAN')
    options.datasets = ['pu1k', 'pugan']
    options.models = [examples.REPKPU_PRETRAINED_MODEL]
    # Official-only evaluation must not require or fall back to retrained weights.
    (folder / 'best.pth').unlink()
    (second / 'best.pth').unlink()
    plan = examples.build_plan(options)
    for job in plan['jobs']:
        spec = examples.REPKPU_CHECKPOINTS[job['dataset']]
        assert job['config']['model'] == 'RepKPU'
        assert Path(job['checkpoint']).name == spec['filename']
        assert job['checkpoint_sha256'] == spec['sha256']
        assert job['checkpoint_variant'] == spec['variant']
        assert spec['file_id'] in job['checkpoint_origin']
        assert job['checkpoint_source'] == 'official-pretrained'
        assert job['best_epoch'] is None and job['training_manifest'] is None
    first, second = (Path(job['checkpoint']) for job in plan['jobs'])
    first.write_bytes(second.read_bytes())
    with pytest.raises(ValueError, match='official checkpoint hash mismatch'):
        examples.execute(options)
    assert not options.output_root.exists()


def test_official_weights_use_dataset_specific_files_without_false_training_provenance(options):
    import shutil
    original_selection = examples.build_plan(options)['selections']['pu1k']
    folder = gaussian_fixture(options)
    second = options.training_run / 'pugan/PU-Gaussian'
    shutil.copytree(folder, second)
    write_json(second / 'config.json', dict(dataset='pugan', model='PU-Gaussian', seed=42))
    shutil.copytree(options.prepared_root / 'PU1K', options.prepared_root / 'PU-GAN')
    options.datasets = ['pu1k', 'pugan']
    options.models = ['PU-Gaussian', examples.PRETRAINED_MODEL]
    plan = examples.build_plan(options)
    assert plan['selections']['pu1k'] == original_selection
    assert len(plan['jobs']) == 4
    run = examples.new_run_directory(options.output_root, 'pretrained')
    examples.snapshot_inputs(run, plan)
    for job in plan['jobs']:
        assert job['config']['model'] == 'PU-Gaussian'
        copied_manifest = (run / job['job_file']).with_suffix('.training_manifest.json')
        if job['model'] == examples.PRETRAINED_MODEL:
            assert Path(job['checkpoint']).name == f'pu_gaussian_{job["dataset"]}_Best.pth'
            assert job['checkpoint_sha256'] == sha256(job['checkpoint'])
            assert job['best_epoch'] is None
            assert job['checkpoint_source'] == 'official-pretrained'
            assert job['training_manifest'] is None and not copied_manifest.exists()
            assert 'pretrained_model/' in job['checkpoint_origin']
        else:
            assert Path(job['checkpoint']).name == 'best.pth'
            assert job['best_epoch'] == 105
            assert job['checkpoint_source'] == 'retrained-best' and copied_manifest.exists()


def test_official_only_needs_its_own_weights_and_missing_weights_fail_before_writes(options):
    folder = gaussian_fixture(options)
    options.models = [examples.PRETRAINED_MODEL]
    # A missing retrained weight must not be silently loaded or required in official-only mode.
    (folder / 'best.pth').unlink()
    plan = examples.build_plan(options)
    assert plan['jobs'][0]['checkpoint_source'] == 'official-pretrained'
    Path(plan['jobs'][0]['checkpoint']).unlink()
    with pytest.raises(FileNotFoundError):
        examples.execute(options)
    assert not options.output_root.exists()


@pytest.mark.parametrize('native,label', [('PU-Gaussian', examples.PRETRAINED_MODEL),
                                         ('RepKPU', examples.REPKPU_PRETRAINED_MODEL),
                                         ('E0', examples.PUCRN_PRETRAINED_MODEL)])
def test_pretrained_comparison_exports_both_sources_and_original_coordinate_metrics(options, monkeypatch, native, label):
    import evaluate_pugaussian_protocol as metrics
    if native == 'PU-Gaussian':
        folder = gaussian_fixture(options)
    elif native == 'RepKPU':
        folder = repkpu_fixture(options, monkeypatch)
    else:
        folder = pucrn_fixture(options, monkeypatch)
    expected_epoch = read_json(folder / 'status.json')['best_epoch']
    options.models = [native, label]
    options.shapes = ['shape.a', 'shape.b']
    options.metrics = True
    source = options.prepared_root / 'PU1K/test'
    protocol = read_json(folder / 'test/metrics/protocol.json')
    # All GT points are inside the fixture triangle, so vertical offsets have known P2F.
    for index, name in enumerate(('shape.a', 'shape.b', 'shape.c')):
        points = np.random.default_rng(index).uniform(.05, .4, size=(8192, 3)).astype(np.float32)
        points[:, 2] = 0
        for group, subfolder, data in [('inputs', 'input_2048', points[:2048]), ('gt', 'gt_8192', points)]:
            path = source / subfolder / f'{name}.xyz'
            np.savetxt(path, data, fmt='%.10g')
            protocol['source_hashes'][group][path.name] = sha256(path)
    write_json(folder / 'test/metrics/protocol.json', protocol)

    def plane_scene(path):
        assert path.read_text() == 'OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n'
        return 'fixture-triangle'

    def analytic_plane_p2f(points, scene):
        # Only the optional Open3D dependency is substituted; use an exact analytic oracle.
        assert scene == 'fixture-triangle'
        assert (points[:, :2] >= 0).all() and (points[:, :2].sum(1) <= 1).all()
        distances = np.abs(points[:, 2]).astype(np.float64)
        return dict(P2F=float(distances.mean()), P2F_P99=float(np.percentile(distances, 99)))

    def child(arguments, log_path, env):
        log_path.write_text('CPU fixture worker\n')
        if arguments[0] == '--export-run':
            examples.export_run(Path(arguments[1]))
            return
        job = read_json(arguments[1])
        offset = .25 if job['checkpoint_source'] == 'official-pretrained' else .125
        for shape in job['shapes']:
            case = Path(job['run_dir']) / job['dataset'] / shape
            points = np.loadtxt(case / 'gt_8192.xyz', dtype=np.float32)
            points[:, 2] += offset
            np.savetxt(case / f'{job["model"]}.xyz', points, fmt='%.10g')

    monkeypatch.setattr(metrics, 'mesh_scene', plane_scene)
    monkeypatch.setattr(metrics, 'p2f_scores', analytic_plane_p2f)
    monkeypatch.setattr(examples, 'run_child', child)
    examples.execute(options)
    run, = (path for path in options.output_root.iterdir() if path.is_dir())
    with (run / 'metrics/summary.csv').open(encoding='utf-8-sig') as stream:
        summary = {row['model']: row for row in csv.DictReader(stream)}
    assert set(summary) == set(options.models)
    assert summary[native]['best_epoch'] == str(expected_epoch)
    assert summary[label]['best_epoch'] == ''
    assert float(summary[native]['P2F_x1000']) == pytest.approx(125.)
    assert float(summary[label]['P2F_x1000']) == pytest.approx(250.)
    assert all(float(row['CD_x1000']) < 2e-6 for row in summary.values())
    assert all(row['n_shapes'] == '2' for row in summary.values())
    metadata = read_json(run / 'metrics/protocol.json')
    assert metadata['display_scale'] == 1000
    assert metadata['HD_convention'] == 'bidirectional_max_squared_sum_v1'
    assert metadata['HD'] == 'sum of bidirectional maximum squared NN distances'
    assert metadata['checkpoints'][1]['best_epoch'] is None
    for name in options.shapes:
        case = run / 'pu1k' / name
        assert (case / f'{native}.ply').is_file()
        assert (case / f'{label}.ply').is_file()
        assert sha256(case / f'{native}.xyz') != sha256(case / f'{label}.xyz')
    with tarfile.open(run.with_suffix('.tar.gz')) as archive:
        assert all(not name.endswith('.pth') for name in archive.getnames())


@pytest.mark.parametrize('module,label', [('test_pugaussian_pretrained_full', examples.PRETRAINED_MODEL),
                                         ('test_repkpu_pretrained_full', examples.REPKPU_PRETRAINED_MODEL),
                                         ('test_pucrn_pretrained_full', examples.PUCRN_PRETRAINED_MODEL)])
def test_full_entrypoint_requires_full_sets_and_has_no_sampling_flags(module, label):
    import importlib
    full = importlib.import_module(module)
    args = full.parse_args([])
    assert args.datasets == ['pu1k', 'pugan']
    assert examples.FULL_TEST_COUNTS == {'pu1k': 127, 'pugan': 27}
    assert args.models == [label]
    assert args.all_shapes and args.metrics and args.no_previews
    assert args.num_examples is None and args.shapes is None
    for flags in (['--num-examples', '5'], ['--shapes', 'shape.a'], ['--models', 'E0']):
        with pytest.raises(SystemExit):
            full.parse_args(flags)


def test_full_mode_never_samples_and_checks_each_model_protocol(options, monkeypatch):
    monkeypatch.setattr(examples, 'FULL_TEST_COUNTS', {'pu1k': 3, 'pugan': 27})
    options.all_shapes = True
    options.num_examples = 1
    for model in options.models:
        path = options.training_run / 'pu1k' / model / 'test/metrics/protocol.json'
        protocol = read_json(path)
        protocol.update(status='completed', expected_shapes=3, dataset='PU1K')
        write_json(path, protocol)
    plan = examples.build_plan(options)
    assert plan['test_scope'] == 'full-test-set'
    assert [row['shape'] for row in plan['selections']['pu1k']] == ['shape.a', 'shape.b', 'shape.c']
    assert len(plan['jobs']) == 2
    options.shapes = ['shape.a']
    with pytest.raises(ValueError, match='cannot be combined'):
        examples.build_plan(options)
    options.shapes = None
    (options.prepared_root / 'PU1K/test/input_2048/shape.a.xyz').unlink()
    with pytest.raises(ValueError, match='exactly 3 objects, got 2'):
        examples.execute(options)
    assert not options.output_root.exists()


def test_full_mode_rejects_partial_protocol_and_unmatched_labels(options, monkeypatch):
    monkeypatch.setattr(examples, 'FULL_TEST_COUNTS', {'pu1k': 3, 'pugan': 27})
    options.models = ['E0']
    options.all_shapes = True
    path = options.training_run / 'pu1k/E0/test/metrics/protocol.json'
    protocol = read_json(path)
    protocol.update(status='completed', expected_shapes=3, dataset='PU1K')
    missing = protocol['source_hashes']['gt'].pop('shape.c.xyz')
    write_json(path, protocol)
    with pytest.raises(ValueError, match='full gt names differ'):
        examples.execute(options)
    assert not options.output_root.exists()
    protocol['source_hashes']['gt']['shape.c.xyz'] = missing
    write_json(path, protocol)
    (options.prepared_root / 'PU1K/test/gt_8192/extra.xyz').write_text('0 0 0\n')
    with pytest.raises(ValueError, match='name sets differ'):
        examples.execute(options)
    assert not options.output_root.exists()


@pytest.mark.parametrize('native,module', [('PU-Gaussian', 'test_pugaussian_pretrained_full'),
                                           ('RepKPU', 'test_repkpu_pretrained_full'),
                                           ('E0', 'test_pucrn_pretrained_full')])
def test_full_pretrained_plan_covers_exactly_127_plus_27_objects(tmp_path, monkeypatch, native, module):
    import importlib
    full = importlib.import_module(module)
    training, prepared, weights = tmp_path / 'training', tmp_path / 'prepared', tmp_path / 'weights'
    write_json(training / 'run.json', dict(status='completed', smoke=False, prepared=str(prepared)))
    weights.mkdir()
    for dataset, count in {'pu1k': 127, 'pugan': 27}.items():
        source = prepared / examples.DATASETS[dataset] / 'test'
        hashes = dict(inputs={}, gt={}, meshes={})
        for group, folder, suffix in [('inputs', 'input_2048', '.xyz'), ('gt', 'gt_8192', '.xyz'),
                                      ('meshes', 'meshes', '.off')]:
            (source / folder).mkdir(parents=True)
            for index in range(count):
                path = source / folder / f'object_{index:03d}{suffix}'
                # Planning checks inventory and hashes, without loading models/point data.
                path.write_text(f'{dataset}/{group}/{index}\n')
                hashes[group][path.name] = sha256(path)
        reference = training / dataset / native
        write_json(reference / 'config.json', dict(model=native, dataset=dataset, seed=42))
        write_json(reference / 'status.json', dict(status='completed', best_epoch=105))
        write_json(reference / 'test/metrics/protocol.json', dict(status='completed', expected_shapes=count,
                   dataset=examples.DATASETS[dataset], source_hashes=hashes))
        filename = (f'pu_gaussian_{dataset}_Best.pth' if native == 'PU-Gaussian' else
                    examples.REPKPU_CHECKPOINTS[dataset]['filename'] if native == 'RepKPU' else 'model.pth')
        path = weights / filename
        path.write_bytes(b'official-shared-release' if native == 'E0' else b'official-' + dataset.encode())
        if native == 'RepKPU':
            spec = examples.REPKPU_CHECKPOINTS[dataset]
            monkeypatch.setitem(examples.REPKPU_CHECKPOINTS, dataset, {**spec, 'sha256': sha256(path)})
        elif native == 'E0':
            monkeypatch.setitem(examples.PUCRN_CHECKPOINT, 'sha256', sha256(path))
    args = full.parse_args(['--training-run', str(training), '--prepared-root', str(prepared),
                            '--pretrained-root', str(weights)])
    plan = examples.build_plan(args)
    assert len(plan['selections']['pu1k']) == 127
    assert len(plan['selections']['pugan']) == 27
    assert sum(map(len, plan['selections'].values())) == 154
    assert len(plan['jobs']) == 2 and plan['metrics'] and not plan['previews']
    assert plan['test_scope'] == 'full-test-set'
    assert all(job['checkpoint_source'] == 'official-pretrained' and job['best_epoch'] is None
               for job in plan['jobs'])
    assert plan['entrypoint'].endswith(module + '.py')
    if native == 'E0':
        assert len({job['checkpoint'] for job in plan['jobs']}) == 1
        assert all(job['checkpoint_format'] == 'net_state_dict' for job in plan['jobs'])
