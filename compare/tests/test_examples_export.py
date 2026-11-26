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
