"""Scientific protocol and lifecycle regressions; no server data is required."""
import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_pugaussian_protocol import cd_hd_squared, evaluate, point_scores, run
from prepare_retraining_data import check_split_identity, clean_test_overlap, make_patches, mesh_split
from retrain_common import ROOT, stable_seed
from run_retraining import job_config, profile


def args(**kwargs):
    value = dict(seed=42, batch_size=16, input_policy='native', grad_lr=.001, grad_seed=21,
                 grad_head_bias=None, smoke=False, smoke_epochs=3, data_root=Path('data'))
    value.update(kwargs)
    return SimpleNamespace(**value)


def test_metric_semantics_and_raw_mesh_coordinates(tmp_path):
    pytest.importorskip('open3d')
    for folder in ('input', 'gt', 'mesh', 'pred'):
        (tmp_path / folder).mkdir()
    rng = np.random.default_rng(8)
    points = rng.random((8192, 3), dtype=np.float32)
    points[:, :2] *= .3
    points[:, 2] = 0
    np.savetxt(tmp_path / 'input/a.xyz', points[:2048])
    np.savetxt(tmp_path / 'gt/a.xyz', points)
    np.savetxt(tmp_path / 'pred/a.xyz', points + [0, 0, .125])
    (tmp_path / 'mesh/a.off').write_text('OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
    rows, summary = evaluate(tmp_path / 'pred', tmp_path / 'input', tmp_path / 'gt', tmp_path / 'mesh', 1)
    assert summary['CD'] < 1e-12 and summary['HD'] < 1e-12
    assert summary['P2F'] == pytest.approx(.125, abs=1e-7)
    with pytest.raises(ValueError):
        evaluate(tmp_path / 'pred', tmp_path / 'input', tmp_path / 'gt', tmp_path / 'mesh', 27)
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'existing.csv').write_text('keep')
    with pytest.raises(FileExistsError):
        run(SimpleNamespace(output_dir=output))
    assert (output / 'existing.csv').read_text() == 'keep'
    exported = tmp_path / 'sum-metrics'
    run(SimpleNamespace(output_dir=exported, dataset='PU1K', expected_shapes=1,
        prediction=[('fixture', tmp_path / 'pred')], input_dir=tmp_path / 'input',
        gt_dir=tmp_path / 'gt', mesh_dir=tmp_path / 'mesh'))
    from retrain_common import read_json
    metadata = read_json(exported / 'protocol.json')
    assert metadata['HD_convention'] == 'bidirectional_max_squared_sum_v1'
    assert metadata['HD'] == 'sum of bidirectional maximum squared NN distances'
    assert metadata['scale_factor'] == 1000


def test_cd_hd_against_exhaustive_distances():
    from evaluate_pugaussian_protocol import independent_unit_sphere
    # Different point counts and unequal directed maxima also rule out 2*max.
    p = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
    q = np.array([[-1, 0, 0], [1, 0, 0], [0, .6, 0], [0, -.6, 0], [0, 0, 0]], dtype=np.float32)
    d = ((independent_unit_sphere(p)[:, None] - independent_unit_sphere(q)[None]) ** 2).sum(-1)
    cd, hd = cd_hd_squared(p, q)
    assert cd == pytest.approx(float(d.min(1).mean() + d.min(0).mean()))
    assert hd == pytest.approx(float(d.min(1).max() + d.min(0).max()))
    assert hd != pytest.approx(float(max(d.min(1).max(), d.min(0).max())))
    assert hd != pytest.approx(float(2 * max(d.min(1).max(), d.min(0).max())))
    scores = point_scores(p, q)
    assert hd == pytest.approx(scores['HD_pred_to_gt'] + scores['HD_gt_to_pred'])
    assert cd_hd_squared(p[::-1], q[[2, 0, 4, 3, 1]]) == pytest.approx((cd, hd))
    assert cd_hd_squared(p, p[::-1]) == (0, 0)


def test_multidensity_fps_really_uses_one_support():
    surface = np.random.default_rng(3).normal(size=(5200, 3)).astype(np.float32)
    result = make_patches(surface, 1, 'cpu')
    assert result[5120].shape == (1, 5120, 3)
    assert np.array_equal(result[256], result[1024][:, :256])
    assert np.array_equal(result[1024], result[1536][:, :1024])
    dense = {tuple(p) for p in result[5120][0]}
    assert all(tuple(p) in dense for p in result[1536][0])
    assert len({tuple(p) for p in result[1536][0]}) == 1536


def test_split_identity_uses_mesh_contents_instead_of_names():
    train = [dict(mesh='simple/star.off', mesh_sha256='flat-star')]
    validation = [dict(mesh='simple/sphere.off', mesh_sha256='sphere')]
    test = [dict(mesh='test/star.off', mesh_sha256='different-3d-star')]
    check_split_identity(train, validation, test)
    test[0]['mesh_sha256'] = 'flat-star'
    with pytest.raises(ValueError, match='Identical mesh content'):
        check_split_identity(train, validation, test)


def test_test_copies_are_excluded_and_validation_is_replenished():
    def row(name, digest, source='ShapeNet'):
        return dict(mesh=name, mesh_sha256=digest, source=source, patches=50)
    original = dict(train=[row('train/a', 'A'), row('train/b', 'B'),
                           row('train/c', 'C'), row('train/p', 'P', 'PU-GAN')],
                    val=[row('val/copy', 'TEST'), row('val/q', 'Q', 'PU-GAN')])
    cleaned, audit = clean_test_overlap(original, [row('test/original', 'TEST')], 42)
    assert len(cleaned['train']) == 3 and len(cleaned['val']) == 2
    assert 'TEST' not in {r['mesh_sha256'] for rows in cleaned.values() for r in rows}
    assert cleaned['val'][-1]['source'] == 'ShapeNet'
    assert sum(r['source'] == 'PU-GAN' for r in cleaned['train']) == 1
    assert sum(r['source'] == 'PU-GAN' for r in cleaned['val']) == 1
    assert audit['excluded_test_copies'][0]['mesh'] == 'val/copy'
    assert len(audit['validation_replacements']) == 1
    assert clean_test_overlap(original, [row('test/original', 'TEST')], 42) == (cleaned, audit)


def test_pugan_parsers_accept_shared_validation_and_integer_seed(monkeypatch):
    for relative, root in [('compare/Grad-PU-main/args/pugan_args.py', 'compare/Grad-PU-main'),
                           ('compare/RepKPU-main/cfgs/upsampling/pugan_args.py', 'compare/RepKPU-main')]:
        monkeypatch.syspath_prepend(str(ROOT / root))
        spec = importlib.util.spec_from_file_location('pugan_parser_probe', ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(sys, 'argv', ['probe', '--seed', '42', '--val_h5_path', 'val.h5', '--best_only'])
        parsed = module.parse_pugan_args()
        assert type(parsed.seed) is int and parsed.seed == 42
        assert parsed.val_h5_path == 'val.h5' and parsed.best_only


def test_five_native_schedules_and_smoke_cover_joint_phase():
    expected = {'E0': 100, 'E6': 100, 'Grad-PU': 60, 'RepKPU': 100, 'PU-Gaussian': 200}
    for dataset in ('pu1k', 'pugan'):
        for model, epochs in expected.items():
            job = job_config(args(), dataset, model, Path('run'), Path('prepared'))
            assert job['epochs'] == epochs
            assert job['stage1_epochs'] == (100 if model == 'PU-Gaussian' else 0)
            assert job['seed'] == 42 and job['model_seed'] == (21 if model == 'Grad-PU' else 42)
    a = job_config(args(), 'pu1k', 'E0', Path('run'), Path('prepared'))
    b = job_config(args(), 'pu1k', 'E6', Path('run'), Path('prepared'))
    assert a['decay_iter'] == b['decay_iter'] == 200000
    smoke = job_config(args(smoke=True), 'pugan', 'PU-Gaussian', Path('run'), Path('prepared'))
    assert smoke['epochs'] == 4 and smoke['stage1_epochs'] == 2
    assert smoke['val_every'] == 1
    assert profile(args()) != profile(args(grad_lr=.0005))


def test_seed_does_not_depend_on_model_or_generation_order():
    assert stable_seed(42, 'mesh-A', 'train') == stable_seed(42, 'mesh-A', 'train')
    assert stable_seed(42, 'mesh-A', 'train') != stable_seed(42, 'mesh-A', 'validation')


def test_default_plan_excludes_deferred_grad_pu(monkeypatch, capsys):
    import json
    from run_retraining import main
    monkeypatch.setattr(sys, 'argv', ['run_retraining.py', '--plan'])
    main()
    plan = json.loads(capsys.readouterr().out)
    assert len(plan['jobs']) == 8
    assert {job['model'] for job in plan['jobs']} == {'E0', 'E6', 'RepKPU', 'PU-Gaussian'}
    assert {job['dataset'] for job in plan['jobs']} == {'pu1k', 'pugan'}


def test_formal_run_cannot_bypass_native_smoke_gate(tmp_path, monkeypatch):
    import run_retraining as launcher
    monkeypatch.setattr(launcher, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(launcher, 'CERTIFICATE', tmp_path / 'missing_certificate.json')
    monkeypatch.setattr(launcher, 'environment', lambda: pytest.fail('Must not touch CUDA before the gate'))
    options = args(smoke=False, datasets=['pu1k'], models=['E0'], prepared_root=None,
                   resume_run=None, run_dir=None)
    with pytest.raises(RuntimeError, match='smoke PASS'):
        launcher.execute(options)


def test_grad_inference_restores_training_flags_and_truncation():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA tensor lifecycle regression')
    from retrain_adapters import NativeAdapter
    adapter = object.__new__(NativeAdapter)
    adapter.kind = 'Grad-PU'
    adapter.args = SimpleNamespace(truncate_distance=False)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, device='cuda'))
        def regress_distance(self, original, query, global_feats, local_feats):
            return query.square().sum(1, keepdim=True)
    adapter.net = Net().train()
    def normalize(t):
        center = t.mean(-1, keepdim=True)
        radius = (t - center).norm(dim=1, keepdim=True).amax(-1, keepdim=True)
        return (t - center) / radius, center, radius
    adapter.utils = SimpleNamespace(normalize_point_cloud=normalize,
        midpoint_interpolate=lambda a, t: t.repeat_interleave(4, dim=-1))
    def update(a, net, points):
        for p in net.parameters():
            p.requires_grad_(False)
        query = points.detach().requires_grad_(True)
        net.regress_distance(points, query, None, None).mean().backward()
        return query - query.grad
    adapter.test_module = SimpleNamespace(pcd_update=update)
    original_method = adapter.net.regress_distance.__func__
    output = adapter.infer(np.random.default_rng(0).normal(size=(2048, 3)).astype(np.float32))
    assert output.shape == (8192, 3) and np.isfinite(output).all()
    assert adapter.net.training and adapter.net.weight.requires_grad
    assert adapter.args.truncate_distance is False
    assert adapter.net.regress_distance.__func__ is original_method
    assert adapter.query_grad_norm > 0
