"""Protocol for publisher H5 training, method-specific loaders and fixed-epoch tests."""
from pathlib import Path

from retrain_common import ROOT, sha256, source_hashes

VERSION = 1
MODELS = ('E0', 'E6', 'Grad-PU', 'RepKPU', 'PU-Gaussian')
DATASETS = ('pu1k', 'pugan')
FOLDERS = {'pu1k': 'PU1K', 'pugan': 'PU-GAN'}
TEST_COUNTS = {'pu1k': 127, 'pugan': 27}
H5_NAMES = {'pu1k': 'pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5',
            'pugan': 'PUGAN_poisson_256_poisson_1024.h5'}


def native_defaults(model, dataset):
    if model not in MODELS or dataset not in DATASETS:
        raise ValueError('Unknown native model/dataset')
    backbone = model in ('E0', 'E6')
    return dict(batch_size=64 if backbone and dataset == 'pu1k' else 32,
                model_seed=42 if backbone or model == 'PU-Gaussian' else 21,
                epochs=200 if model == 'PU-Gaussian' else 60 if model == 'Grad-PU' else 100,
                stage1_epochs=100 if model == 'PU-Gaussian' else 0,
                lr=0.001, decay_iter=50000)


def make_config(model, dataset, output, data, *, seed=42, epochs=None, batch_size=None,
                grad_lr=.001, smoke=False, log_every=100, evaluate_test=True):
    defaults = native_defaults(model, dataset)
    budget = epochs if epochs is not None else defaults['epochs']
    stage1 = budget // 2 if model == 'PU-Gaussian' and epochs is not None else defaults['stage1_epochs']
    if smoke:
        budget, stage1 = (4, 2) if model == 'PU-Gaussian' else (3, 0)
    if budget < 1 or model == 'PU-Gaussian' and not 0 < stage1 < budget:
        raise ValueError('Gaussian training needs at least two epochs and both phases')
    batch = batch_size if batch_size is not None else defaults['batch_size']
    if batch < 1 or log_every < 1 or grad_lr <= 0:
        raise ValueError('Batch size, logging interval and learning rate must be positive')
    # An explicit smaller batch keeps the step decay at the same seen-sample count.
    interval = round(defaults['decay_iter'] * defaults['batch_size'] / batch)
    config = dict(protocol='publisher_h5_native_loaders_fixed_epoch_v1', model=model, dataset=dataset,
        output=str(Path(output).resolve()), epochs=budget, stage1_epochs=stage1,
        seed=seed, model_seed=defaults['model_seed'], batch_size=batch,
        native_batch_size=defaults['batch_size'], decay_iter=max(1, interval),
        grad_lr=grad_lr, grad_head_bias=None, input_policy='native', smoke=smoke,
        smoke_batches=2, log_every=log_every, evaluate_test=evaluate_test and not smoke,
        selection='fixed_final_epoch_no_test_selection', data=data)
    if model == 'PU-Gaussian':
        config['gaussian_sampling'] = dict(stage1=20, stage2=6, inference=6, final_ratio=4)
    return config


def h5_candidates(data_root, dataset):
    base = Path(data_root)
    folders = (['PU1K/PU1K/train', 'PU1K/train'] if dataset == 'pu1k' else
               ['PUGAN', 'PUGAN/PU-GAN/train', 'PU-GAN/train', 'PUGAN/train'])
    return [base / folder / H5_NAMES[dataset] for folder in folders]


def find_h5(data_root, dataset, explicit=None):
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    found = [p.resolve() for p in h5_candidates(data_root, dataset) if p.is_file()]
    if not found:
        raise FileNotFoundError('Missing publisher training H5. Expected one of: ' +
                                ', '.join(str(p) for p in h5_candidates(data_root, dataset)))
    if len(found) > 1 and len({sha256(p) for p in found}) != 1:
        raise ValueError(f'Different {dataset} H5 copies found; specify --{dataset}-h5 explicitly')
    return found[0]


def find_directory(data_root, candidates):
    found = [Path(data_root) / p for p in candidates if (Path(data_root) / p).is_dir()]
    if not found:
        raise FileNotFoundError('Missing data directory: ' + ', '.join(candidates))
    return found[0].resolve()


def plan(args):
    jobs = []
    for dataset in args.datasets:
        for model in args.models:
            jobs.append(make_config(model, dataset, ROOT/'compare/native-runs/PLAN'/dataset/model,
                dict(training_h5='publisher H5' if model != 'PU-Gaussian' else 'author-style x20 H5',
                     test_root=str(Path(args.cache_root) / FOLDERS[dataset] / 'test')),
                seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                grad_lr=args.grad_lr, log_every=args.log_every, evaluate_test=not args.no_test))
    return dict(protocol_version=VERSION, data_root=str(Path(args.data_root).resolve()),
                cache_root=str(Path(args.cache_root).resolve()), models=list(args.models),
                checkpoint_selection='Fixed final epoch; no old validation set and no test-based selection',
                automatic_smoke=True, jobs=jobs)


def runtime_sources():
    sources = source_hashes()
    # The baseline loader is invoked directly; include it in provenance too.
    path = ROOT/'baseline/3DVision-main/PointUpsampling/PUCRN/data.py'
    sources[path.relative_to(ROOT).as_posix()] = sha256(path)
    return sources
