"""Queue lifecycle regressions with fake native workers, no GPU/data required."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_structure_study as study
from retrain_common import read_json, sha256, write_json


@pytest.fixture
def queue_env(tmp_path, monkeypatch):
    prepared, baseline = tmp_path / 'prepared', tmp_path / 'baseline'
    write_json(prepared / 'manifest.json', dict(status='completed', settings=dict(smoke=False), datasets={
        'PU1K': dict(train_shapes=917, validation_shapes=102, test_shapes=127),
        'PU-GAN': dict(train_shapes=108, validation_shapes=12, test_shapes=27)}))
    data_sha = sha256(prepared / 'manifest.json')
    metrics = dict(CD=.001, HD=.004, HD_pred_to_gt=.004, HD_gt_to_pred=.002, P2F=.003, P2F_P99=.01)
    for dataset, count in study.COUNTS.items():
        write_json(baseline / dataset / 'protocol.json', dict(status='completed', split='validation',
            scope='full validation', model='E6', dataset=dataset, n_shapes=count,
            prepared_smoke=False, checkpoint_job_smoke=False, data_manifest_sha256=data_sha, best_sha256='E6-fixture'))
        study.write_csv(baseline / dataset / 'summary.csv', [dict(stage='p3', phase='fps',
            n_shapes=count, **{k + '_x1000': v * 1000 for k, v in metrics.items()})])
    monkeypatch.setattr(study.native, 'CERTIFICATE', tmp_path / 'smoke_pass.json')
    monkeypatch.setattr(study.native, 'fingerprint', lambda args: 'fixture-source-profile')
    monkeypatch.setattr(study, 'source_hashes', lambda: {'fixture.py': 'fixture-source'})
    settings = dict(models=list(study.DEFAULT_MODELS), datasets=['pu1k', 'pugan'], seed=42, batch_size=16,
        data_root=str(tmp_path / 'data'), prepared_root=str(prepared),
        smoke_prepared_root=str(tmp_path / 'smoke-prepared'), input_policy='native',
        baseline_run=str(tmp_path / 'no-archive-weights'), baseline_stages=str(baseline),
        combination_policy='validation', structure_config=None, structure_overrides={})
    calls, diag_calls = [], []
    cd_values = dict(V1a=.00096, V1=.0009, V2a=.00097, V2=.00095, V3=.00088)

    def launch(args):
        calls.append(('smoke' if args.smoke else 'train', list(args.models), bool(args.resume_run)))
        folder = args.resume_run or args.run_dir
        jobs = []
        for dataset in args.datasets:
            for model in args.models:
                job = folder / dataset / model
                config = study.native.job_config(args, dataset, model, job, args.prepared_root)
                write_json(job / 'config.json', config)
                (job / 'best.pth').write_bytes(('best:' + model + dataset).encode())
                (job / 'last.pth').write_bytes(('resume:' + model + dataset).encode())
                values = metrics | dict(CD=cd_values.get(model, .001))
                study.write_csv(job / 'validation.csv', [dict(epoch=1 if args.smoke else 5, **values)])
                write_json(job / 'status.json', dict(status='completed', test_evaluated=False,
                    best_epoch=1 if args.smoke else 5, best_sha256=sha256(job / 'best.pth'), last_sha256=sha256(job / 'last.pth')))
                write_json(job / 'health.json', dict(passed=True, post_validation_updates=1))
                jobs.append(config)
        write_json(folder / 'run.json', dict(status='completed', fingerprint=study.native.fingerprint(args),
            profile=study.native.profile(args), smoke=args.smoke, jobs=jobs))
        if args.smoke:
            write_json(study.native.CERTIFICATE, dict(fingerprint=study.native.fingerprint(args),
                passed_jobs=[f'{d}/{m}' for d in args.datasets for m in args.models]))

    def diagnose(command, log_path, env):
        def option(name):
            return command[command.index(name) + 1]
        dataset, model_dir, output = option('--dataset'), Path(option('--model-dir')), Path(option('--output'))
        model, count = model_dir.name, study.COUNTS[dataset]
        diag_calls.append((model, dataset, output))
        output.mkdir(parents=True)
        log_path.write_text('Fake diagnostic process\n')
        write_json(output / 'protocol.json', dict(status='completed', split='validation', scope='full validation',
            model=model, dataset=dataset, n_shapes=count, prepared_smoke=False, checkpoint_job_smoke=False,
            best_sha256=sha256(model_dir / 'best.pth'), data_manifest_sha256=data_sha,
            sources=study.source_hashes()))
        study.write_csv(output / 'summary.csv', [dict(stage=s, phase=p, n_shapes=count,
            **{k + '_x1000': v * 1000 for k, v in metrics.items()})
            for p in ('fps', 'candidates') for s in ('p1', 'p2', 'p3')])
        study.write_csv(output / 'per_shape.csv', [dict(shape=str(i), phase=p, stage=s)
            for i in range(count) for p in ('fps', 'candidates') for s in ('p1', 'p2', 'p3')])
        study.write_csv(output / 'movement.csv', [dict(shape=str(i), move=.05) for i in range(count)])

    return SimpleNamespace(folder=tmp_path / 'queue', settings=settings, calls=calls, diag_calls=diag_calls,
                           launch=launch, diagnose=diagnose, cd_values=cd_values)


def run(env, resume=False, diagnose=None, launch=None):
    return study.execute(env.folder, env.settings if not resume else None, resume=resume,
                         launch=launch or env.launch, diagnose=diagnose or env.diagnose)


def test_default_plan_is_standard_library_only_and_does_not_run_workers(capsys, monkeypatch):
    monkeypatch.setattr(study.native, 'environment', lambda: pytest.fail('Plan touched CUDA'))
    study.main(['--plan'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['formal_jobs'] == 10 and plan['formal_epochs_per_job'] == 20
    assert plan['settings']['combination_policy'] == 'always'
    assert plan['automatic_test'] is False
    assert len(plan['steps']) == 14
    assert plan['steps'][0]['smoke'] is True


def test_complete_queue_and_idempotent_resume(queue_env):
    env = queue_env
    queue = run(env)
    assert queue['status'] == 'completed' and queue['skipped_steps'] == []
    assert [call[0] for call in env.calls] == ['smoke', 'train', 'train']
    assert env.calls[1][1] == ['V1a', 'V1', 'V2a', 'V2'] and env.calls[2][1] == ['V3']
    assert len(env.diag_calls) == 10
    assert queue['combination_gate']['passed'] is True
    rows = study.read_csv(env.folder / 'validation_comparison.csv')
    assert len(rows) == 12 and {r['model'] for r in rows} == {'E6', *study.DEFAULT_MODELS}
    assert all(float(r['CD_change_vs_E6_percent']) < 0 for r in rows if r['model'] != 'E6')
    assert len(study.read_csv(env.folder / 'stage_comparison.csv')) == 60
    assert run(env, resume=True)['status'] == 'completed'
    assert len(env.calls) == 3 and len(env.diag_calls) == 10


def test_missing_global_smoke_certificate_is_recovered_without_retraining_components(queue_env):
    env = queue_env
    run(env)
    study.native.CERTIFICATE.unlink()
    queue = run(env, resume=True)
    assert queue['status'] == 'completed'
    assert len(env.calls) == 4 and env.calls[-1][0] == 'smoke' and env.calls[-1][2] is True
    assert len(env.diag_calls) == 10


def test_failed_diagnostic_resumes_new_attempt_and_preserves_previous(queue_env):
    env = queue_env
    def fail(command, log_path, process_env):
        if 'diagnostics' not in str(command):
            pytest.fail('Invalid diagnostic call')
        env.diagnose(command, log_path, process_env)
        if len(env.diag_calls) == 2:
            raise RuntimeError('Simulated interrupted diagnostic')
    with pytest.raises(RuntimeError, match='Simulated'):
        run(env, diagnose=fail)
    failed_queue = read_json(env.folder / 'queue.json')
    failed = next(s for s in failed_queue['steps'] if s['status'] == 'failed')
    old_output = Path(failed['result'])
    assert old_output.is_dir()
    queue = run(env, resume=True)
    resumed = next(s for s in queue['steps'] if s['id'] == failed['id'])
    assert Path(resumed['result']) != old_output and old_output.is_dir()
    assert len(env.calls) == 3 and len(env.diag_calls) == 11


def test_validation_gate_skips_v3_without_aborting_components(queue_env):
    env = queue_env
    env.cd_values['V2'] = .00105
    queue = run(env)
    assert queue['status'] == 'completed' and queue['combination_gate']['passed'] is False
    assert len(queue['skipped_steps']) == 3
    assert len(env.calls) == 2 and len(env.diag_calls) == 8
    assert not (env.folder / 'combined').exists()
    assert all(r['model'] != 'V3' for r in study.read_csv(env.folder / 'validation_comparison.csv'))


def test_explicit_always_policy_can_run_only_combination(queue_env):
    env = queue_env
    env.settings.update(models=['V3'], combination_policy='always')
    queue = run(env)
    assert queue['combination_gate']['passed'] and len(env.calls) == 2
    assert env.calls[1][1] == ['V3'] and len(env.diag_calls) == 2


def test_completed_weights_are_verified_before_skipping(queue_env):
    env = queue_env
    run(env)
    (env.folder / 'components/pu1k/V1/best.pth').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        run(env, resume=True)
    assert len(env.calls) == 3


def test_completed_diagnostics_are_verified_before_skipping(queue_env):
    env = queue_env
    queue = run(env)
    step = next(s for s in queue['steps'] if s['type'] == 'diagnose')
    summary = Path(step['result']) / 'summary.csv'
    summary.write_text(summary.read_text(encoding='utf-8-sig').replace('4.0', '4.1'), encoding='utf-8-sig')
    with pytest.raises(ValueError, match='diagnostic files changed'):
        run(env, resume=True)


def test_prepared_manifest_cannot_change_during_resume(queue_env):
    env = queue_env
    run(env)
    write_json(Path(env.settings['prepared_root']) / 'manifest.json', dict(changed=True))
    with pytest.raises(ValueError, match='Prepared manifest changed'):
        run(env, resume=True)


def test_baseline_preflight_failure_does_not_leave_an_unresumable_directory(queue_env):
    env = queue_env
    env.settings['baseline_stages'] = str(env.folder / 'missing')
    with pytest.raises(ValueError, match='baseline-stages'):
        run(env)
    assert not env.folder.exists() and not env.calls


def test_missing_resume_checkpoint_cannot_restart_a_trained_job(queue_env):
    env = queue_env
    def fail(args):
        env.launch(args)
        if not args.smoke:
            path = args.run_dir / 'run.json'
            value = read_json(path)
            value['status'] = 'failed'
            write_json(path, value)
            raise RuntimeError('Interrupted training launcher')
    with pytest.raises(RuntimeError):
        run(env, launch=fail)
    (env.folder / 'components/pu1k/V1/last.pth').unlink()
    with pytest.raises(ValueError, match='Cannot resume without last.pth'):
        run(env, resume=True)


def test_frozen_override_copy_is_used_and_checked(queue_env, tmp_path):
    env = queue_env
    env.settings['structure_overrides'] = {'V3': {'geometry_gate': True}}
    run(env)
    snapshot = env.folder / 'structure_options.json'
    assert snapshot.is_file()
    assert read_json(snapshot) == env.settings['structure_overrides']
    write_json(snapshot, {'V3': {'geometry_gate': False}})
    with pytest.raises(ValueError, match='Frozen structure options changed'):
        run(env, resume=True)


def test_cli_rejects_ambiguous_v3_gate_and_resume_overrides(capsys):
    with pytest.raises(SystemExit):
        study.main(['--plan', '--models', 'V3', '--combination-policy', 'validation'])
    assert 'requires V1 and V2' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        study.main(['--resume-run', 'existing', '--batch-size', '8'])
    assert 'restores saved options' in capsys.readouterr().err


def test_twenty_epoch_queue_keeps_smoke_at_three_epochs(queue_env):
    env = queue_env
    env.settings.update(epochs=20, combination_policy='always')
    queue = run(env)
    assert queue['purpose'] == 'screening' and queue['settings']['epochs'] == 20
    for group, expected in [('smoke', 3), ('components', 20), ('combined', 20)]:
        manifest = read_json(env.folder / group / 'run.json')
        assert all(job['epochs'] == expected for job in manifest['jobs'])
    rows = study.read_csv(env.folder / 'validation_comparison.csv')
    assert all(r['training_epochs'] == '20' for r in rows if r['model'] != 'E6')


def test_continuation_uses_last_state_in_new_directory_and_preserves_parent(queue_env):
    env = queue_env
    env.settings.update(epochs=20, combination_policy='always')
    run(env)
    original = env.folder / 'components/pu1k/V1'
    hashes = {name: sha256(original / name) for name in ('best.pth', 'last.pth', 'config.json', 'validation.csv')}
    settings = dict(env.settings, epochs=100, continue_from=str(env.folder), structure_config=None)
    target = env.folder.with_name('continued-100')
    def launch(args):
        if not args.smoke:
            assert args.resume_run is not None
            sidecar = read_json(args.resume_run / 'pu1k' / args.models[0] / 'continuation.json')
            assert sidecar['from_epochs'] == 20 and sidecar['to_epochs'] == 100
        env.launch(args)
    queue = study.execute(target, settings, launch=launch, diagnose=env.diagnose)
    assert queue['status'] == 'completed'
    assert all(sha256(original / name) == digest for name, digest in hashes.items())
    assert read_json(original / 'config.json')['epochs'] == 20
    assert read_json(target / 'components/pu1k/V1/config.json')['epochs'] == 100


def test_continuation_can_select_only_one_promising_model(queue_env):
    env = queue_env
    env.settings.update(epochs=20, combination_policy='always')
    run(env)
    target = env.folder.with_name('continue-v1-only')
    settings = dict(env.settings, models=['V1'], epochs=100, continue_from=str(env.folder), structure_config=None)
    study.execute(target, settings, launch=env.launch, diagnose=env.diagnose)
    assert (target / 'components/pu1k/V1/continuation.json').is_file()
    assert not (target / 'components/pu1k/V1a').exists()
    assert not (target / 'combined').exists()


def test_early_e6_reference_uses_only_matching_epoch_budget(queue_env):
    env = queue_env
    archive = env.folder.with_name('e6-history')
    data_sha = sha256(Path(env.settings['prepared_root']) / 'manifest.json')
    for dataset in env.settings['datasets']:
        folder = archive / dataset / 'E6'
        write_json(folder / 'status.json', dict(status='completed', best_epoch=40, best_sha256='epoch40-weight'))
        write_json(folder / 'manifest.json', dict(data_manifest_sha256=data_sha))
        study.write_csv(folder / 'validation.csv', [dict(epoch=epoch,
            **{key: cd if key == 'CD' else .003 for key in study.METRICS})
            for epoch, cd in [(5, .0012), (20, .001), (40, .0008), (100, .0009)]])
    settings = dict(env.settings, epochs=20, baseline_run=str(archive))
    baseline = study.load_baseline(settings, data_sha)
    for row in baseline.values():
        assert row['epoch'] == 20 and row['metrics']['CD'] == .001
        assert row['budget_matched'] and row['reference_budget'] == 20
        assert row['best_sha256'] is None and row['full_run_best_sha256'] == 'epoch40-weight'


def test_full_best_reference_is_labeled_when_early_history_is_unavailable(queue_env):
    env = queue_env
    env.settings.update(epochs=20)
    baseline = study.load_baseline(env.settings, sha256(Path(env.settings['prepared_root']) / 'manifest.json'))
    assert all(not row['budget_matched'] for row in baseline.values())
    gate = study.gate_combination(env.settings, baseline, env.folder)
    assert gate['passed'] is False and 'matching-budget' in gate['note']


def test_plan_continuation_restores_saved_training_options(queue_env, capsys):
    env = queue_env
    env.settings.update(epochs=20, combination_policy='always')
    run(env)
    capsys.readouterr()
    study.main(['--continue-from', str(env.folder), '--epochs', '100', '--models', 'V1', '--plan'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['formal_epochs_per_job'] == 100 and plan['settings']['models'] == ['V1']
    assert plan['settings']['continue_from'] == str(env.folder)


def test_real_worker_continuation_restores_optimizer_scheduler_rng_and_history(queue_env, monkeypatch):
    torch = pytest.importorskip('torch')
    import retrain_adapters
    import retrain_worker as worker
    env = queue_env
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: None)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda states: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'CPU test worker')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(worker, 'source_hashes', study.source_hashes)
    real_load = torch.load
    monkeypatch.setattr(torch, 'load', lambda path, **kwargs: real_load(path, **(kwargs | {'map_location': 'cpu'})))
    class Adapter:
        def __init__(self, config):
            self.kind, self.args = config['model'], SimpleNamespace()
            self.net = torch.nn.Linear(1, 1)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=.01)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=.9)
            self.native = SimpleNamespace(step=0)
            self.health, self.query_grad_norm = dict(updates=0), 0.
        def train_batch(self, batch, step, epoch):
            self.optimizer.zero_grad()
            before = self.net.weight.detach().clone()
            loss = self.net(batch['x']).square().mean()
            loss.backward()
            norm = sum(float(p.grad.norm().square()) for p in self.net.parameters()) ** .5
            self.optimizer.step()
            self.native.step += 1
            self.health.update(updates=self.health['updates'] + 1, gradient_norm=norm,
                parameter_step_max=float((self.net.weight.detach() - before).abs().max()))
            return dict(total=float(loss.detach()), **self.health)
        def epoch_end(self):
            self.scheduler.step()
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, *args):
            self.epoch = 0
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return {'x': torch.tensor([float(index + 1)])}
    def validate(adapter, *args):
        with torch.no_grad():
            value = float(adapter.net(torch.ones(1, 1)).square().mean())
        adapter.last_inference_timing = dict(seconds=.001, seconds_per_shape=.001)
        return {key: value for key in study.METRICS}
    monkeypatch.setattr(retrain_adapters, 'NativeAdapter', Adapter)
    monkeypatch.setattr(retrain_adapters, 'PatchDataset', Dataset)
    monkeypatch.setattr(worker, 'validate', validate)
    base = dict(env.settings, models=['V1a'], datasets=['pu1k'], batch_size=1)
    full = env.folder.with_name('worker-full10')
    short = env.folder.with_name('worker-short5')
    continued = env.folder.with_name('worker-continued10')
    full_args = study.launcher_args(dict(base, epochs=10), full, ['V1a'])
    full_config = study.native.job_config(full_args, 'pu1k', 'V1a', full / 'pu1k/V1a', full_args.prepared_root)
    worker.run(full_config)
    short_args = study.launcher_args(dict(base, epochs=5), short, ['V1a'])
    short_config = study.native.job_config(short_args, 'pu1k', 'V1a', short / 'pu1k/V1a', short_args.prepared_root)
    worker.run(short_config)
    write_json(short / 'run.json', dict(status='completed', fingerprint='worker-fixture',
        profile=study.native.profile(short_args), smoke=False, jobs=[short_config]))
    next_args = study.launcher_args(dict(base, epochs=10), continued, ['V1a'])
    study.continue_native(short, continued, next_args, 'worker-fixture')
    config = read_json(continued / 'run.json')['jobs'][0]
    worker.run(config, resume=True)
    first = torch.load(full / 'pu1k/V1a/last.pth', weights_only=False)
    second = torch.load(continued / 'pu1k/V1a/last.pth', weights_only=False)
    for name in first['model']:
        torch.testing.assert_close(first['model'][name], second['model'][name], atol=0, rtol=0)
    assert first['scheduler'] == second['scheduler']
    assert first['optimizer']['param_groups'] == second['optimizer']['param_groups']
    for key, state in first['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, second['optimizer']['state'][key][name], atol=0, rtol=0)
    torch.testing.assert_close(first['rng']['torch'], second['rng']['torch'], atol=0, rtol=0)
    assert first['epoch'] == second['epoch'] == 10
    assert first['global_step'] == second['global_step'] == 20
    assert first['validation_history'] == second['validation_history']
    assert [row['epoch'] for row in second['validation_history']] == [5, 10]
