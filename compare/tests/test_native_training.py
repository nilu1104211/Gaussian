"""Native data provenance, method budgets, fixed-epoch selection and safe cleanup."""
import json
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrain_common import sha256, write_json, read_json
from native_training_protocol import MODELS, FOLDERS, make_config, native_defaults, find_h5
from prepare_native_data import h5_info, prepare_test, test_files as fixed_test_files, write_gaussian_h5
from native_training_storage import validate_cleanup, old_caches, old_generated_h5, archive_metadata


def make_h5(path, count=4, gaussian=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    with h5py.File(path, 'w') as f:
        for size in ((256, 1024, 1536, 5120) if gaussian else (256, 1024)):
            f.create_dataset(f'poisson_{size}', data=rng.normal(size=(count, size, 3)).astype('f4'))
    return path


def test_native_budgets_and_paired_baseline_are_not_global_batch16():
    for dataset in ('pu1k', 'pugan'):
        a = make_config('E0', dataset, 'A', {})
        b = make_config('E6', dataset, 'B', {})
        for key in ('batch_size', 'model_seed', 'epochs', 'decay_iter', 'selection'):
            assert a[key] == b[key]
        assert a['batch_size'] == (64 if dataset == 'pu1k' else 32)
        for name in ('Grad-PU', 'RepKPU'):
            c = make_config(name, dataset, 'C', {})
            assert c['batch_size'] == 32 and c['model_seed'] == 21
        g = make_config('PU-Gaussian', dataset, 'G', {})
        assert (g['epochs'], g['stage1_epochs'], g['decay_iter']) == (200, 100, 50000)
        assert g['gaussian_sampling'] == dict(stage1=20, stage2=6, inference=6, final_ratio=4)
    for name in MODELS:
        c = make_config(name, 'pu1k', 'C', {}, epochs=20)
        assert c['epochs'] == 20 and c['selection'] == 'fixed_final_epoch_no_test_selection'
        assert c['stage1_epochs'] == (10 if name == 'PU-Gaussian' else 0)
    assert make_config('E0', 'pu1k', 'C', {}, batch_size=16)['decay_iter'] == 200000
    assert make_config('PU-Gaussian', 'pu1k', 'C', {}, smoke=True)['stage1_epochs'] == 2
    with pytest.raises(ValueError):
        make_config('PU-Gaussian', 'pu1k', 'C', {}, epochs=1)


def test_original_h5_is_validated_without_rewriting_or_reordering(tmp_path):
    path = make_h5(tmp_path/'original.h5')
    before = path.read_bytes()
    info = h5_info(path)
    assert info['train_patches'] == 4 and info['origin'] == 'publisher_h5_unmodified'
    assert before == path.read_bytes()
    with h5py.File(path, 'a') as f:
        f['poisson_1024'][3, 0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        h5_info(path)


def test_different_original_h5_copies_require_explicit_path(tmp_path):
    name = 'pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5'
    a = make_h5(tmp_path/'PU1K/PU1K/train'/name)
    b = make_h5(tmp_path/'PU1K/train'/name, count=2)
    with pytest.raises(ValueError, match='Different'):
        find_h5(tmp_path, 'pu1k')
    assert find_h5(tmp_path, 'pu1k', a) == a.resolve()


def write_test_fixture(folder, names=('shape',)):
    for name in names:
        for sub, size in [('input_2048', 2048), ('gt_8192', 8192)]:
            path = folder/sub/(name+'.xyz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, np.zeros((size, 3), dtype='f4'))
        mesh = folder/'meshes'/(name+'.off')
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_text('OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')


def test_cached_tests_survive_removal_of_old_prepared_source(tmp_path, monkeypatch):
    import prepare_native_data as builder
    source = tmp_path/'old'
    write_test_fixture(source/'PU1K/test')
    monkeypatch.setitem(builder.TEST_COUNTS, 'pu1k', 1)
    cache = tmp_path/'native'
    info = prepare_test(tmp_path, cache, 'pu1k', 42, source)
    assert source.resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(source)
    assert fixed_test_files(cache/'PU1K/test', 1) == info['test_sha256']
    assert prepare_test(tmp_path, cache, 'pu1k', 42) == info


def test_gaussian_stream_writer_appends_all_densities_and_rejects_corruption(tmp_path, monkeypatch):
    import prepare_native_data as builder
    mesh = tmp_path/'mesh.off'
    mesh.write_text('mesh fixture')
    def batches(path, count, seed, gpu_batch):
        for start in range(0, count, 2):
            amount = min(2, count-start)
            yield {n: np.full((amount, n, 3), start+1, dtype='f4') for n in builder.SIZES}
    monkeypatch.setattr(builder, 'gaussian_mesh_patches', batches)
    appended = make_h5(tmp_path/'pugan.h5', 3, gaussian=True)
    output = tmp_path/'generated/pu1k.h5'
    info = write_gaussian_h5(output, [mesh], 4, 42, tmp_path, append_h5=appended)
    assert info['train_patches'] == 7
    with h5py.File(output) as f, h5py.File(appended) as g:
        for n in builder.SIZES:
            assert np.array_equal(f[f'poisson_{n}'][4:], g[f'poisson_{n}'][:])
    assert write_gaussian_h5(output, [mesh], 4, 42, tmp_path, append_h5=appended) == info
    with h5py.File(output, 'a') as f:
        f['poisson_5120'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='provenance differs'):
        write_gaussian_h5(output, [mesh], 4, 42, tmp_path, append_h5=appended)


def cleanup_fixture(root):
    cache = root/'data/native_20261127_seed42'
    datasets = {}
    original = root/'data/PU1K/original.h5'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'publisher source')
    for dataset, count in [('pu1k', 127), ('pugan', 27)]:
        folder = cache/FOLDERS[dataset]/'test'
        hashes = {}
        for index in range(count):
            for sub, ext in [('input_2048', '.xyz'), ('gt_8192', '.xyz'), ('meshes', '.off')]:
                path = folder/sub/(str(index)+ext)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixed fixture')
                hashes[path.relative_to(folder).as_posix()] = sha256(path)
        datasets[dataset] = dict(publisher=dict(training_h5=str(original), training_h5_sha256=sha256(original)),
            test=dict(test_root=str(folder), test_shapes=count, test_sha256=hashes))
    write_json(cache/'manifest.json', dict(status='tests_ready', datasets=datasets))
    old = root/'data/repro_20261123_seed42'
    write_json(old/'manifest.json', dict(settings={}, datasets={}))
    return cache, old, original


def test_cleanup_protects_original_h5_and_requires_complete_shared_test_cache(tmp_path):
    cache, old, original = cleanup_fixture(tmp_path)
    assert old_caches(tmp_path) == [old.resolve()]
    legacy = tmp_path/'compare/training-runs/old/pu1k_train_x20.h5'
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b'old generated')
    assert old_generated_h5(tmp_path) == [legacy.resolve()]
    assert validate_cleanup(tmp_path, cache, [legacy]) == [legacy]
    archive, record = archive_metadata(tmp_path, cache, [old, legacy])
    backup = archive/old.relative_to(tmp_path)/'manifest.json'
    assert backup.read_bytes() == (old/'manifest.json').read_bytes()
    assert record['metadata'][str(old/'manifest.json')]['sha256'] == sha256(backup)
    assert record['completed_targets'] == []  # metadata backup precedes any deletion
    assert validate_cleanup(tmp_path, cache, [old]) == [old]
    with pytest.raises(ValueError, match='recognized'):
        validate_cleanup(tmp_path, cache, [tmp_path/'data'])
    marker = read_json(cache/'manifest.json')
    marker['datasets']['pu1k']['publisher']['training_h5'] = str(old/'needed.h5')
    (old/'needed.h5').write_bytes(original.read_bytes())
    write_json(cache/'manifest.json', marker)
    with pytest.raises(ValueError, match='dependency'):
        validate_cleanup(tmp_path, cache, [old])
    marker['datasets']['pu1k']['publisher']['training_h5'] = str(original)
    write_json(cache/'manifest.json', marker)
    (cache/'PU1K/test/input_2048/0.xyz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='fixed test changed'):
        validate_cleanup(tmp_path, cache, [old])


def test_launcher_smokes_all_methods_before_training_and_can_resume(tmp_path, monkeypatch):
    import run_native_training as launcher
    import run_retraining
    args = launcher.parser().parse_args(['--data-root', str(tmp_path/'data'), '--cache-root', str(tmp_path/'cache'),
                                         '--run-dir', str(tmp_path/'run'), '--epochs', '20', '--no-test'])
    monkeypatch.setattr(run_retraining, 'environment', lambda: {})
    monkeypatch.setattr(launcher, 'runtime_sources', lambda: {'fixture': 'source'})
    calls = []
    def invoke(command, env, log):
        if any(str(p).endswith('prepare_native_data.py') for p in command):
            write_json(args.cache_root/'manifest.json', dict(datasets={d: dict(
                publisher=dict(training_h5='original', training_h5_sha256='hash'),
                gaussian=dict(training_h5='gaussian', training_h5_sha256='hash'),
                test=dict(test_root='fixed', test_shapes=1, test_sha256={})) for d in args.datasets}))
            return
        config = read_json(Path(command[command.index('--config')+1]))
        calls.append(config)
        out = Path(config['output'])
        (out/'last.pth').write_bytes(b'resume fixture')
        status = dict(status='completed', last_sha256=sha256(out/'last.pth'), metrics_dir=None)
        if not config['smoke']:
            (out/'final.pth').write_bytes(b'final fixture')
            status['final_sha256'] = sha256(out/'final.pth')
        write_json(out/'status.json', status)
        write_json(out/'health.json', dict(passed=True))
    monkeypatch.setattr(launcher, 'invoke', invoke)
    launcher.execute(args)
    assert len(calls) == 20 and all(c['smoke'] for c in calls[:10]) and not any(c['smoke'] for c in calls[10:])
    assert all(c['epochs'] == 20 for c in calls[10:])
    assert read_json(args.run_dir/'run.json')['status'] == 'completed'
    args.resume_run, args.run_dir = args.run_dir, None
    launcher.execute(args)
    assert len(calls) == 20  # no completed model is trained twice
    monkeypatch.setattr(launcher, 'runtime_sources', lambda: {'fixture': 'changed'})
    with pytest.raises(ValueError, match='code/options changed'):
        launcher.execute(args)


def test_worker_resume_matches_uninterrupted_training_and_does_not_touch_tests(tmp_path, monkeypatch):
    import torch
    import native_training_adapter as adapters
    import native_training_worker as worker
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'CPU lifecycle fixture')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(worker, 'runtime_sources', lambda: {'fixture': 'unchanged'})
    monkeypatch.setattr(worker, 'infer_directory', lambda *a, **k: pytest.fail('No-test run accessed test data'))
    def state():
        return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
    def restore(saved):
        torch.set_rng_state(saved['torch'].cpu()); np.random.set_state(saved['numpy']); random.setstate(saved['python'])
    def seed(value):
        torch.manual_seed(value); np.random.seed(value); random.seed(value)
    monkeypatch.setattr(worker, 'rng_state', state)
    monkeypatch.setattr(worker, 'restore_rng', restore)
    monkeypatch.setattr(worker, 'seed_all', seed)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, *a): self.epoch = 1
        def __len__(self): return 6
        def __getitem__(self, index):
            x = torch.rand(8, 3)+np.random.random()*.01
            return dict(input=x, gt=x*.25, radius=torch.ones(1))
        def close(self): pass
    class Adapter:
        def __init__(self, config):
            self.kind = 'RepKPU'; self.args = SimpleNamespace(); self.net = torch.nn.Linear(3, 3)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=.001)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, .9)
            self.health = dict(updates=0)
            self.query_grad_norm = 0.
        def train_batch(self, batch, step, epoch):
            self.optimizer.zero_grad(); pred = self.net(batch['input']); loss = (pred-batch['gt']).square().mean()
            loss.backward(); before = self.net.weight.detach().clone()
            norm = float(self.net.weight.grad.norm()); self.optimizer.step()
            self.health.update(updates=self.health['updates']+1, gradient_norm=norm,
                parameter_step_max=float((self.net.weight.detach()-before).abs().max()))
            return dict(total_loss=float(loss.detach()), **self.health)
        def epoch_end(self): self.scheduler.step()
    monkeypatch.setattr(adapters, 'PublisherAdapter', Adapter)
    monkeypatch.setattr(adapters, 'PublisherDataset', Dataset)
    source = tmp_path/'source.h5'; source.write_bytes(b'fixture')
    data = dict(training_h5=str(source), training_h5_sha256=sha256(source))
    uninterrupted = make_config('RepKPU', 'pu1k', tmp_path/'full', data, epochs=3, batch_size=2, evaluate_test=False)
    worker.run(uninterrupted)
    interrupted = dict(uninterrupted, output=str(tmp_path/'resumed'))
    original_save = worker.save_torch
    def stop_after_saved_epoch(path, saved):
        original_save(path, saved)
        if Path(path).name == 'last.pth' and saved['epoch'] == 1:
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(worker, 'save_torch', stop_after_saved_epoch)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        worker.run(interrupted)
    monkeypatch.setattr(worker, 'save_torch', original_save)
    worker.run(interrupted, resume=True)
    a = torch.load(tmp_path/'full/last.pth', weights_only=False)
    b = torch.load(tmp_path/'resumed/last.pth', weights_only=False)
    assert a['global_step'] == b['global_step'] == 9
    assert a['scheduler'] == b['scheduler']
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key])
    for key in a['optimizer']['state']:
        for field in a['optimizer']['state'][key]:
            assert torch.equal(a['optimizer']['state'][key][field], b['optimizer']['state'][key][field])
    assert read_json(tmp_path/'resumed/status.json')['selection'] == 'fixed_final_epoch_no_test_selection'
    assert not (tmp_path/'resumed/test').exists()


def test_dead_grad_distance_probe_rejects_zero_field_and_restores_model_flags(monkeypatch):
    import torch
    import native_training_worker as worker
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
        def forward(self, points, query):
            return torch.relu(query.sum(1, keepdim=True)*0-1+self.weight*0)
    net = Field().train()
    adapter = SimpleNamespace(kind='Grad-PU', net=net, args=SimpleNamespace(),
        utils=SimpleNamespace(midpoint_interpolate=lambda args,x:x,
                              get_query_points=lambda x,args:x+torch.randn_like(x)*.02))
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='inactive'):
        worker.gradient_probe(adapter, dict(input=torch.ones(2, 256, 3)), 21)
    assert net.training and net.weight.requires_grad
    assert torch.equal(torch.get_rng_state(), before)
