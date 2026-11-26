"""Structure configuration, validation isolation and diagnostic semantics."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrain_common import NEW_PUGCS_MODELS, pugcs_variant_options, sha256, write_json
from run_retraining import aggregate, job_config, profile
from stage_diagnostics import directional_scores, stage_scores


def arguments(**values):
    base = dict(seed=42, batch_size=16, input_policy='native', grad_lr=.001,
                grad_seed=21, grad_head_bias=None, smoke=False, smoke_epochs=3, data_root=Path('data'))
    return SimpleNamespace(**(base | values))


def test_structure_jobs_keep_e6_protocol_and_do_not_test_candidates():
    for model in NEW_PUGCS_MODELS:
        for dataset, decay in [('pu1k', 200000), ('pugan', 100000)]:
            config = job_config(arguments(), dataset, model, Path('run'), Path('prepared'))
            assert config['epochs'] == 100 and config['batch_size'] == 16
            assert config['decay_iter'] == decay and config['val_every'] == 5
            assert config['evaluate_test'] is False
            assert config['loss_config']['lambda_surf'] == .1
            assert config['loss_config']['lambda_tail'] == config['loss_config']['lambda_growth'] == .05
            assert config['network_config'] == pugcs_variant_options(model)
    assert 'evaluate_test' not in job_config(arguments(), 'pu1k', 'E6', Path('run'), Path('prepared'))


def test_structure_epoch_budget_does_not_change_smoke_or_native_baselines():
    short = arguments(epochs=20)
    config = job_config(short, 'pu1k', 'V1a', Path('run'), Path('prepared'))
    assert config['epochs'] == 20 and config['decay_iter'] == 200000 and config['val_every'] == 5
    assert job_config(arguments(epochs=20, smoke=True), 'pu1k', 'V1a', Path('run'), Path('prepared'))['epochs'] == 3
    assert job_config(short, 'pu1k', 'E6', Path('run'), Path('prepared'))['epochs'] == 100
    assert profile(short) == profile(arguments(epochs=100))


def test_overrides_are_recorded_and_change_smoke_fingerprint_profile(tmp_path):
    path = tmp_path / 'overrides.json'
    write_json(path, {'V3': {'geometry_gate': True, 'learned_normal': True, 'use_neighborhood_refine': True}})
    args = arguments(structure_config=path)
    config = job_config(args, 'pu1k', 'V3', Path('run'), Path('prepared'))
    assert config['network_config']['geometry_gate'] is True
    assert profile(args) != profile(arguments())
    write_json(path, {'V3': {'typo': True}})
    with pytest.raises(ValueError, match='Unknown network option'):
        profile(args)


def test_validation_summary_cannot_be_mistaken_for_test_table(tmp_path):
    job = dict(model='V1a', dataset='pu1k', output=str(tmp_path / 'V1a'))
    write_json(tmp_path / 'V1a/status.json', dict(status='completed', best_epoch=5, test_evaluated=False, metrics_dir=None))
    (tmp_path / 'V1a/validation.csv').write_text('epoch,CD,HD\n5,0.1,0.2\n10,0.3,0.4\n')
    aggregate(tmp_path, [job])
    assert not (tmp_path / 'pu1k_summary.csv').exists()
    content = (tmp_path / 'pu1k_validation_summary.csv').read_text(encoding='utf-8-sig')
    assert 'V1a' in content and '0.1' in content and '0.3' not in content


def test_stage_metrics_separate_official_and_shared_coordinates():
    points = np.random.default_rng(42).normal(size=(128, 3)).astype(np.float32)
    shifted = points * 2 + 5
    values = stage_scores(shifted, points, points[:32])
    assert values['CD_x1000'] < 1e-9
    assert values['shared_CD_x1000'] > 1
    forward = directional_scores(np.array([[0., 0, 0], [4, 0, 0]]), np.zeros((1, 3)))
    assert forward['mean_pred_to_gt'] == 8 and forward['mean_gt_to_pred'] == 0
    assert forward['HD'] == 16 and forward['HD_gt_to_pred'] == 0


def test_inference_timing_excludes_export_and_records_shape_count(tmp_path):
    pytest.importorskip('torch')
    from retrain_worker import infer_directory
    source = tmp_path / 'validation'
    (source / 'input_2048').mkdir(parents=True)
    points = np.random.default_rng(42).normal(size=(2048, 3)).astype(np.float32)
    np.savetxt(source / 'input_2048/one.xyz', points)
    class Adapter:
        def infer(self, value):
            return np.repeat(value, 4, axis=0)
    adapter = Adapter()
    count = infer_directory(adapter, source, tmp_path / 'predictions', 42, expected=1)
    assert count == adapter.last_inference_timing['n_shapes'] == 1
    assert adapter.last_inference_timing['seconds'] > 0
    assert np.loadtxt(tmp_path / 'predictions/one.xyz').shape == (8192, 3)


def test_frozen_validation_selected_model_can_be_explicitly_evaluated(tmp_path):
    import test_examples as examples
    training, prepared = tmp_path / 'training', tmp_path / 'prepared'
    write_json(training / 'run.json', dict(status='completed', smoke=False))
    write_json(prepared / 'manifest.json', dict(fixed_test='fixture'))
    source = prepared / 'PU1K/test'
    for directory in ('input_2048', 'gt_8192', 'meshes'):
        (source / directory).mkdir(parents=True)
    # Plan validation uses provenance and names, without running inference.
    (source / 'input_2048/shape.xyz').write_text('input fixture')
    (source / 'gt_8192/shape.xyz').write_text('GT fixture')
    (source / 'meshes/shape.off').write_text('mesh fixture')
    folder = training / 'pu1k/V1a'
    write_json(folder / 'config.json', dict(model='V1a', dataset='pu1k'))
    (folder / 'best.pth').write_bytes(b'fixture')
    write_json(folder / 'status.json', dict(status='completed', test_evaluated=False,
        best_epoch=5, best_sha256=sha256(folder / 'best.pth')))
    write_json(folder / 'manifest.json', dict(data_manifest_sha256=sha256(prepared / 'manifest.json')))
    args = examples.parser().parse_args(['--training-run', str(training), '--prepared-root', str(prepared),
        '--datasets', 'pu1k', '--models', 'V1a', '--num-examples', '1', '--no-previews'])
    plan = examples.build_plan(args)
    assert plan['jobs'][0]['model'] == 'V1a' and plan['jobs'][0]['best_epoch'] == 5
    write_json(prepared / 'manifest.json', dict(changed=True))
    with pytest.raises(ValueError, match='Prepared data changed'):
        examples.build_plan(args)
