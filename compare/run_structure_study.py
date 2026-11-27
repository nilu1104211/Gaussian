"""One queue: native smoke -> component training -> diagnostics -> gated V3.

All training jobs select best using validation CD; this entry point never tests
the test set. Resume with --resume-run alone. --plan/--status need no CUDA.
"""
import argparse
import csv
import copy
import math
import os
import signal
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import run_retraining as native
from evaluate_pugaussian_protocol import HD_CONVENTION
from retrain_common import NEW_PUGCS_MODELS, ROOT, read_json, sha256, source_hashes, write_json

RUNS = ROOT / 'compare/retrain-runs'
BASELINE_RUN = RUNS / 'train-seed42-20261123-161953-503869'
DEFAULT_MODELS = ('V1a', 'V1', 'V2a', 'V2', 'V3')
METRICS = ('CD', 'HD', 'HD_pred_to_gt', 'HD_gt_to_pred', 'P2F', 'P2F_P99')
COUNTS = {'pu1k': 102, 'pugan': 12}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def stamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S-%f')


def launcher_args(settings, folder, models, smoke=False):
    return SimpleNamespace(data_root=Path(settings['data_root']),
        prepared_root=Path(settings['smoke_prepared_root'] if smoke else settings['prepared_root']),
        datasets=settings['datasets'], models=list(models), batch_size=settings['batch_size'],
        seed=settings['seed'], input_policy=settings['input_policy'], grad_lr=.001,
        grad_seed=21, grad_head_bias=None, smoke=smoke, smoke_epochs=3,
        epochs=settings.get('epochs', 100),
        structure_config=Path(settings['structure_config']) if settings['structure_config'] else None,
        run_dir=folder if not (folder / 'run.json').is_file() else None,
        resume_run=folder if (folder / 'run.json').is_file() else None)


def recipe(settings):
    components = [model for model in settings['models'] if model != 'V3']
    stages = [dict(id='smoke', type='native', models=settings['models'], smoke=True, output='smoke')]
    if components:
        stages.append(dict(id='components', type='native', models=components, smoke=False, output='components'))
        stages += [dict(id=f'diagnose-{model}-{dataset}', type='diagnose', model=model,
                        dataset=dataset, training='components') for model in components for dataset in settings['datasets']]
    if 'V3' in settings['models']:
        stages.append(dict(id='combination-gate', type='gate'))
        stages.append(dict(id='combined', type='native', models=['V3'], smoke=False, output='combined'))
        stages += [dict(id=f'diagnose-V3-{dataset}', type='diagnose', model='V3', dataset=dataset,
                        training='combined') for dataset in settings['datasets']]
    return stages


def load_baseline(settings, data_sha):
    """Prefer the completed E6 stage audit, without needing any old weights."""
    if settings.get('epochs', 100) < 100:
        try:
            return history_baseline(settings, data_sha, settings['epochs'])
        except (OSError, KeyError, ValueError, StopIteration):
            # A full-best reference is still useful, but must be labeled as a
            # different budget and cannot establish an equal-budget benefit.
            pass
    explicit = settings['baseline_stages']
    candidates = [Path(explicit)] if explicit else sorted(RUNS.glob('e6-validation-stages-*'), reverse=True)
    for folder in candidates:
        try:
            result = {}
            for dataset in settings['datasets']:
                protocol_path = folder / dataset / 'protocol.json'
                summary_path = folder / dataset / 'summary.csv'
                protocol = read_json(protocol_path)
                if not (protocol['status'] == 'completed' and protocol['split'] == 'validation'
                        and protocol['scope'] == 'full validation' and protocol['model'] == 'E6'
                        and protocol['dataset'] == dataset and protocol['n_shapes'] == COUNTS[dataset]
                        and not protocol['prepared_smoke'] and not protocol['checkpoint_job_smoke']
                        and protocol['data_manifest_sha256'] == data_sha):
                    raise ValueError('E6 stage baseline is not the matching full validation run')
                row = next(r for r in read_csv(summary_path) if r['phase'] == 'fps' and r['stage'] == 'p3')
                metrics = {key: float(row[key + '_x1000']) / 1000 for key in METRICS}
                # Archived max-HD reports already retain both directed maxima.
                metrics['HD'] = metrics['HD_pred_to_gt'] + metrics['HD_gt_to_pred']
                if not all(math.isfinite(value) and value >= 0 for value in metrics.values()):
                    raise ValueError('Nonfinite E6 baseline')
                result[dataset] = dict(metrics=metrics, HD_convention=HD_CONVENTION, n_shapes=COUNTS[dataset], epoch=None,
                    reference_budget=100, budget_matched=settings.get('epochs', 100) == 100,
                    best_sha256=protocol['best_sha256'], source=str(summary_path),
                    evidence_sha256={str(path): sha256(path) for path in (protocol_path, summary_path)})
            return result
        except (OSError, KeyError, ValueError, StopIteration):
            if explicit:
                raise ValueError(f'Invalid --baseline-stages: {folder}')
    try:
        return history_baseline(settings, data_sha, 100)
    except (OSError, KeyError, ValueError, StopIteration) as error:
        raise ValueError('Matching E6 validation reports are required. Use --baseline-stages with the returned E6 diagnosis directory; old E6 weights are not required.') from error


def history_baseline(settings, data_sha, budget):
    """Use archived E6 validation within the same early-epoch budget."""
    result = {}
    for dataset in settings['datasets']:
        folder = Path(settings['baseline_run']) / dataset / 'E6'
        status, manifest = read_json(folder / 'status.json'), read_json(folder / 'manifest.json')
        if status['status'] != 'completed' or manifest['data_manifest_sha256'] != data_sha:
            raise ValueError('E6 archive does not match prepared data')
        candidates = [r for r in read_csv(folder / 'validation.csv') if int(r['epoch']) <= budget]
        if not candidates or max(int(r['epoch']) for r in candidates) < budget:
            raise ValueError('E6 history does not cover the requested budget')
        row = min(candidates, key=lambda r: float(r['CD']))
        metrics = {key: float(row[key]) for key in METRICS}
        metrics['HD'] = metrics['HD_pred_to_gt'] + metrics['HD_gt_to_pred']
        if not all(math.isfinite(value) and value >= 0 for value in metrics.values()):
            raise ValueError('Nonfinite E6 archive')
        epoch = int(row['epoch'])
        result[dataset] = dict(metrics=metrics, HD_convention=HD_CONVENTION, n_shapes=COUNTS[dataset], epoch=epoch,
            reference_budget=budget, budget_matched=settings.get('epochs', 100) == budget,
            best_sha256=status['best_sha256'] if epoch == status['best_epoch'] else None,
            full_run_best_sha256=status['best_sha256'], source=str(folder / 'validation.csv'),
            evidence_sha256={str(folder / name): sha256(folder / name)
                             for name in ('status.json', 'manifest.json', 'validation.csv')})
    return result


def continue_native(parent, folder, args, expected):
    """Copy completed shorter-budget state into a NEW run; keep the parent intact."""
    if not (parent / 'run.json').is_file():
        return False  # E.g. V3 was skipped by the parent queue.
    source = read_json(parent / 'run.json')
    if source['status'] != 'completed' or source['fingerprint'] != expected or source['profile'] != native.profile(args):
        raise ValueError('Continuation source does not match source/profile: ' + str(parent))
    jobs = []
    staging = folder.with_name(folder.name + '.bootstrap-' + stamp())
    for dataset in args.datasets:
        for model in args.models:
            source_job = parent / dataset / model
            result = model_result(parent, dataset, model)
            old_config = read_json(source_job / 'config.json')
            recorded = next(j for j in source['jobs'] if j['dataset'] == dataset and j['model'] == model)
            if old_config != recorded:
                raise ValueError('Continuation source job configuration changed')
            target = staging / dataset / model
            config = native.job_config(args, dataset, model, folder / dataset / model, args.prepared_root)
            comparable_old = {k: v for k, v in old_config.items() if k not in ('epochs', 'output')}
            comparable_new = {k: v for k, v in config.items() if k not in ('epochs', 'output')}
            if comparable_old != comparable_new or config['epochs'] <= old_config['epochs']:
                raise ValueError('Continuation may only increase epochs and change the output directory')
            target.mkdir(parents=True, exist_ok=False)
            for name in ('best.pth', 'last.pth', 'validation.csv', 'training.jsonl', 'train.log'):
                if (source_job / name).is_file():
                    shutil.copy2(source_job / name, target / name)
            write_json(target / 'config.json', config)
            write_json(target / 'continuation.json', dict(source=str(source_job),
                from_epochs=old_config['epochs'], to_epochs=config['epochs'],
                source_config_sha256=sha256(source_job / 'config.json'),
                source_best_sha256=result['best_sha256'], source_last_sha256=sha256(source_job / 'last.pth')))
            write_json(target / 'status.json', dict(status='continuation_pending',
                best_epoch=result['epoch'], test_evaluated=False))
            jobs.append(config)
    write_json(staging / 'run.json', dict(source, status='continuation_pending', jobs=jobs,
        continuation_source=str(parent)))
    staging.rename(folder)
    return True


def model_result(folder, dataset, model):
    job = folder / dataset / model
    status = read_json(job / 'status.json')
    if status['status'] != 'completed' or status.get('test_evaluated') is not False:
        raise ValueError(f'Expected completed validation-only job: {job}')
    for name, key in [('best.pth', 'best_sha256'), ('last.pth', 'last_sha256')]:
        if sha256(job / name) != status[key]:
            raise ValueError(f'Checkpoint changed or missing: {job / name}')
    row = next(r for r in read_csv(job / 'validation.csv') if int(r['epoch']) == status['best_epoch'])
    metrics = {key: float(row[key]) for key in METRICS}
    metrics['HD'] = metrics['HD_pred_to_gt'] + metrics['HD_gt_to_pred']
    if not all(math.isfinite(value) and value >= 0 for value in metrics.values()):
        raise ValueError(f'Nonfinite validation metrics: {job}')
    return dict(epoch=status['best_epoch'], metrics=metrics, HD_convention=HD_CONVENTION,
                job=str(job), best_sha256=status['best_sha256'])


def verify_native(folder, args, fingerprint):
    manifest = read_json(folder / 'run.json')
    if manifest['status'] != 'completed' or manifest['fingerprint'] != fingerprint or manifest['profile'] != native.profile(args):
        raise ValueError(f'Completed native run no longer matches this queue: {folder}')
    jobs = manifest['jobs']
    if {(j['dataset'], j['model']) for j in jobs} != {(d, m) for d in args.datasets for m in args.models}:
        raise ValueError('Completed native run has different jobs')
    for config in jobs:
        if config['smoke'] != args.smoke or config.get('evaluate_test') is not False:
            raise ValueError('Queue jobs must use the selected smoke mode and validation-only evaluation')
        expected_config = native.job_config(args, config['dataset'], config['model'],
            folder / config['dataset'] / config['model'], args.prepared_root)
        if config != expected_config:
            raise ValueError('Completed job configuration changed')
        health = read_json(Path(config['output']) / 'health.json')
        if not health['passed'] or (args.smoke and health['post_validation_updates'] <= 0):
            raise ValueError('Native health verification failed')
        model_result(folder, config['dataset'], config['model'])


def gate_combination(settings, baseline, folder):
    if settings['combination_policy'] == 'always':
        return dict(passed=True, policy='always', epochs=settings.get('epochs', 100),
                    note='Preset combination experiment; early-budget screening does not establish independent module benefits.')
    if any(not baseline[d].get('budget_matched', True) for d in settings['datasets']):
        return dict(passed=False, policy='validation', checks=[], note='No matching-budget E6 reference; do not gate using unequal training budgets.')
    checks = []
    for model in ('V1', 'V2'):
        for dataset in settings['datasets']:
            cd = model_result(folder / 'components', dataset, model)['metrics']['CD']
            reference = baseline[dataset]['metrics']['CD']
            checks.append(dict(model=model, dataset=dataset, CD=cd, E6_CD=reference, passed=cd < reference))
    return dict(passed=all(row['passed'] for row in checks), policy='validation', checks=checks,
                note='Single-seed screening: V1 and V2 must each beat E6 validation CD on every selected dataset. This is not statistical proof or a guarantee on other metrics.')


def run_diagnostic(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise


def verify_diagnostic(output, model_dir, dataset, model, queue):
    protocol = read_json(output / 'protocol.json')
    if not (protocol['status'] == 'completed' and protocol['split'] == 'validation'
            and protocol['scope'] == 'full validation' and protocol['n_shapes'] == COUNTS[dataset]
            and protocol['dataset'] == dataset and protocol['model'] == model
            and not protocol['prepared_smoke'] and not protocol['checkpoint_job_smoke']
            and protocol['best_sha256'] == sha256(model_dir / 'best.pth')
            and protocol['data_manifest_sha256'] == queue['data_manifest_sha256']
            and protocol['sources'] == queue['sources']):
        raise ValueError(f'Diagnostic provenance mismatch: {output}')
    if len(read_csv(output / 'per_shape.csv')) != COUNTS[dataset] * 6 or len(read_csv(output / 'movement.csv')) != COUNTS[dataset]:
        raise ValueError('Incomplete diagnostic tables')
    if len(read_csv(output / 'summary.csv')) != 6:
        raise ValueError('Incomplete diagnostic summary')


def summarize(folder, queue):
    rows, stage_rows = [], []
    settings, baseline = queue['settings'], queue['baseline']
    for dataset in settings['datasets']:
        reference = baseline[dataset]
        rows.append(dict(dataset=dataset, model='E6', best_epoch=reference['epoch'],
            HD_convention=HD_CONVENTION,
            training_epochs=reference.get('reference_budget', 100), comparison_budget_matched=reference.get('budget_matched', True),
            n_shapes=COUNTS[dataset], **{k + '_x1000': reference['metrics'][k] * 1000 for k in METRICS},
            CD_change_vs_E6_percent=0., source=reference['source']))
        for model in settings['models']:
            group = 'combined' if model == 'V3' else 'components'
            status_file = folder / group / dataset / model / 'status.json'
            if not status_file.is_file() or read_json(status_file).get('status') != 'completed':
                continue
            result = model_result(folder / group, dataset, model)
            reference_cd = reference['metrics']['CD']
            rows.append(dict(dataset=dataset, model=model, best_epoch=result['epoch'], n_shapes=COUNTS[dataset],
                HD_convention=HD_CONVENTION,
                training_epochs=settings.get('epochs', 100), comparison_budget_matched=reference.get('budget_matched', True),
                **{key + '_x1000': result['metrics'][key] * 1000 for key in METRICS},
                CD_change_vs_E6_percent=100 * (result['metrics']['CD'] / reference_cd - 1) if reference_cd else None,
                source=result['job']))
    for step in queue['steps']:
        if step['type'] == 'diagnose' and step['status'] == 'completed':
            stage_rows += [dict(model=step['model'], dataset=step['dataset'], **row)
                           for row in read_csv(Path(step['result']) / 'summary.csv')]
    write_csv(folder / 'validation_comparison.csv', rows)
    write_csv(folder / 'stage_comparison.csv', stage_rows)


def execute(folder, settings=None, resume=False, launch=None, diagnose=None):
    launch, diagnose = launch or native.execute, diagnose or run_diagnostic
    queue_path = folder / 'queue.json'
    if resume:
        queue = read_json(queue_path)
        settings = queue['settings']
    else:
        data_path = Path(settings['prepared_root']) / 'manifest.json'
        data = read_json(data_path)
        if data['status'] != 'completed' or data['settings']['smoke']:
            raise ValueError('Use the existing complete formal prepared dataset')
        for dataset in settings['datasets']:
            name = 'PU1K' if dataset == 'pu1k' else 'PU-GAN'
            counts = (917, 102, 127) if dataset == 'pu1k' else (108, 12, 27)
            if tuple(data['datasets'][name][k] for k in ('train_shapes', 'validation_shapes', 'test_shapes')) != counts:
                raise ValueError('Unexpected formal split: ' + dataset)
        data_sha = sha256(data_path)
        if settings.get('continue_from'):
            parent_queue = read_json(Path(settings['continue_from']) / 'queue.json')
            if parent_queue['status'] != 'completed' or parent_queue['sources'] != source_hashes() or parent_queue['data_manifest_sha256'] != data_sha:
                raise ValueError('Continuation requires a completed parent queue with matching source and prepared data')
        baseline = load_baseline(settings, data_sha)
        folder.mkdir(parents=True, exist_ok=False)
        if settings['structure_overrides']:
            write_json(folder / 'structure_options.json', settings['structure_overrides'])
            settings['structure_config'] = str(folder / 'structure_options.json')
        queue = dict(version=2, status='initialized', settings=settings,
            purpose='screening' if settings.get('epochs', 100) < 100 else 'full-budget validation',
            fingerprint=native.fingerprint(launcher_args(settings, folder / 'smoke', settings['models'], True)),
            sources=source_hashes(), data_manifest_sha256=data_sha,
            baseline=baseline, steps=[dict(step, status='pending') for step in recipe(settings)])
        write_json(queue_path, queue)
    expected = native.fingerprint(launcher_args(settings, folder / 'smoke', settings['models'], True))
    if queue['fingerprint'] != expected or queue['sources'] != source_hashes():
        raise ValueError('Queue source/options changed; restore the original code or start a new queue')
    if sha256(Path(settings['prepared_root']) / 'manifest.json') != queue['data_manifest_sha256']:
        raise ValueError('Prepared manifest changed since this queue was created')
    if settings['structure_config'] and read_json(settings['structure_config']) != settings['structure_overrides']:
        raise ValueError('Frozen structure options changed')
    print(f'Structure queue: {folder}', flush=True)
    queue['status'] = 'running'
    queue.pop('error', None)
    write_json(queue_path, queue)
    current = None
    try:
        for index, step in enumerate(queue['steps'], 1):
            current = step
            print(f'[{index}/{len(queue["steps"])}] {step["id"]}: {step["status"]}', flush=True)
            if step['status'] == 'skipped':
                continue
            if step['type'] == 'native':
                child = folder / step['output']
                args = launcher_args(settings, child, step['models'], step['smoke'])
                if step['status'] == 'completed':
                    verify_native(child, args, expected)
                    if not step['smoke']:
                        continue
                    certificate = read_json(native.CERTIFICATE) if native.CERTIFICATE.is_file() else {}
                    required = {f'{d}/{m}' for d in settings['datasets'] for m in settings['models']}
                    if certificate.get('fingerprint') == expected and required <= set(certificate.get('passed_jobs', [])):
                        continue
                    # Revalidate the completed native smoke and republish its
                    # certificate; native resume skips the finished training.
                if not step['smoke'] and step.get('models') == ['V3'] and not queue['combination_gate']['passed']:
                    step.update(status='skipped', reason='V1/V2 validation-CD gate did not pass')
                    write_json(queue_path, queue)
                    continue
                if not step['smoke'] and settings.get('continue_from') and not (child / 'run.json').is_file():
                    continue_native(Path(settings['continue_from']) / step['output'], child, args, expected)
                    args = launcher_args(settings, child, step['models'], step['smoke'])
                step.update(status='running')
                write_json(queue_path, queue)
                if (child / 'run.json').is_file():
                    for config in read_json(child / 'run.json')['jobs']:
                        job = Path(config['output'])
                        if not (job / 'last.pth').is_file() and ((job / 'best.pth').exists() or (job / 'validation.csv').exists()):
                            raise ValueError(f'Cannot resume without last.pth: {job}')
                launch(args)
                verify_native(child, args, expected)
            elif step['type'] == 'gate':
                queue['combination_gate'] = gate_combination(settings, queue['baseline'], folder)
                write_json(folder / 'combination_gate.json', queue['combination_gate'])
                print('V3 gate: ' + ('PASS' if queue['combination_gate']['passed'] else 'SKIP'), flush=True)
            else:
                if step['model'] == 'V3' and not queue['combination_gate']['passed']:
                    step.update(status='skipped', reason='V3 was not trained')
                    write_json(queue_path, queue)
                    continue
                model_dir = folder / step['training'] / step['dataset'] / step['model']
                if step['status'] == 'completed':
                    verify_diagnostic(Path(step['result']), model_dir, step['dataset'], step['model'], queue)
                    if any(sha256(Path(step['result']) / name) != digest for name, digest in step['artifact_sha256'].items()):
                        raise ValueError('Completed diagnostic files changed: ' + step['id'])
                    continue
                output = folder / 'diagnostics' / step['model'] / step['dataset'] / ('attempt-' + stamp())
                step.update(status='running', result=str(output))
                write_json(queue_path, queue)
                command = [sys.executable, '-B', '-u', str(ROOT / 'compare/stage_diagnostics.py'),
                    '--model-dir', str(model_dir), '--prepared-root', settings['prepared_root'],
                    '--dataset', step['dataset'], '--seed', str(settings['seed']), '--output', str(output)]
                env = dict(os.environ, PYTHONPATH=str(native.RUNTIME / 'python'), PYTHONNOUSERSITE='1',
                           CUDA_VISIBLE_DEVICES='0', MPLBACKEND='Agg')
                diagnose(command, output.with_suffix('.log'), env)
                verify_diagnostic(output, model_dir, step['dataset'], step['model'], queue)
                step['artifact_sha256'] = {name: sha256(output / name) for name in
                                          ('protocol.json', 'summary.csv', 'per_shape.csv', 'movement.csv')}
            step.update(status='completed')
            step.pop('error', None)
            write_json(queue_path, queue)
            summarize(folder, queue)
        queue.update(status='completed', skipped_steps=[s['id'] for s in queue['steps'] if s['status'] == 'skipped'])
        write_json(queue_path, queue)
        summarize(folder, queue)
        print(f'STRUCTURE STUDY COMPLETED: {folder}; skipped={queue["skipped_steps"]}', flush=True)
    except BaseException as error:
        if current is not None:
            current.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        queue.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        write_json(queue_path, queue)
        raise
    return queue


def new_settings(args):
    seed = args.seed if args.seed is not None else 42
    batch = args.batch_size if args.batch_size is not None else 16
    data_root = (args.data_root or ROOT / 'data').resolve()
    models, datasets = args.models or list(DEFAULT_MODELS), args.datasets or ['pu1k', 'pugan']
    epochs = args.epochs if args.epochs is not None else 20
    policy = args.combination_policy or ('always' if epochs < 100 else 'validation')
    if not 5 <= epochs <= 100 or epochs % 5:
        raise ValueError('Epoch budget must be 5..100 in steps of 5')
    if batch < 1 or not 0 <= seed < 2**32 or len(set(models)) != len(models) or len(set(datasets)) != len(datasets):
        raise ValueError('Use a positive batch size, a valid seed and unique model/dataset lists')
    if 'V3' in models and policy == 'validation' and not {'V1', 'V2'} <= set(models):
        raise ValueError('Validation-gated V3 requires V1 and V2 in --models; use --combination-policy always for an explicit preset combination experiment')
    return dict(models=models, datasets=datasets, seed=seed, batch_size=batch, epochs=epochs, data_root=str(data_root),
        input_policy='native', prepared_root=str((args.prepared_root or data_root / f'repro_20261123_seed{seed}').resolve()),
        smoke_prepared_root=str((args.smoke_prepared_root or data_root / f'repro_20261123_seed{seed}_smoke_b{batch}').resolve()),
        baseline_run=str((args.baseline_run or BASELINE_RUN).resolve()),
        baseline_stages=str(args.baseline_stages.resolve()) if args.baseline_stages else None,
        combination_policy=policy, structure_config=None,
        structure_overrides=native.structure_overrides(args))


def continuation_settings(args):
    parent = args.continue_from.resolve()
    queue = read_json(parent / 'queue.json')
    settings = copy.deepcopy(queue['settings'])
    budget = args.epochs if args.epochs is not None else 100
    if queue['status'] != 'completed' or not settings.get('epochs', 100) < budget <= 100 or budget % 5:
        raise ValueError('Continue a completed shorter-budget queue to a larger budget, up to 100 epochs in steps of 5')
    for key in ('data_root', 'prepared_root', 'smoke_prepared_root', 'datasets', 'batch_size', 'seed',
                'baseline_run', 'baseline_stages', 'structure_config'):
        if getattr(args, key) is not None:
            raise ValueError('Continuation preserves data/seed/optimizer/structure options; only models, epochs and combination-policy may be selected')
    models = args.models or settings['models']
    if len(set(models)) != len(models) or not set(models) <= set(settings['models']):
        raise ValueError('Continuation models must be a unique subset of the parent queue')
    policy = args.combination_policy or ('always' if budget < 100 else 'validation')
    if 'V3' in models and policy == 'validation' and not {'V1', 'V2'} <= set(models):
        raise ValueError('Validation-gated V3 continuation also requires V1/V2, or use --combination-policy always')
    settings.update(models=models, epochs=budget, combination_policy=policy,
                    continue_from=str(parent), structure_config=None)
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--run-dir', type=Path)
    modes.add_argument('--resume-run', type=Path)
    modes.add_argument('--status', type=Path)
    parser.add_argument('--models', choices=NEW_PUGCS_MODELS, nargs='+')
    parser.add_argument('--datasets', choices=('pu1k', 'pugan'), nargs='+')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int, help='Per-model budget; default 20 for screening, 100 for continuation')
    parser.add_argument('--continue-from', type=Path, help='Continue selected completed shorter-budget jobs into a NEW queue, preserving the parent')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--prepared-root', type=Path)
    parser.add_argument('--smoke-prepared-root', type=Path)
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--baseline-stages', type=Path)
    parser.add_argument('--structure-config', type=Path)
    parser.add_argument('--combination-policy', choices=('validation', 'always'))
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    overrides = ('models', 'datasets', 'batch_size', 'epochs', 'continue_from', 'seed', 'data_root', 'prepared_root',
                 'smoke_prepared_root', 'baseline_run', 'baseline_stages', 'structure_config', 'combination_policy')
    if (args.resume_run or args.status) and any(getattr(args, key) is not None for key in overrides):
        parser.error('Resume/status restores saved options; pass only its queue directory')
    if args.status:
        queue = read_json(args.status.resolve() / 'queue.json')
        print(__import__('json').dumps(dict(status=queue['status'], steps=[{k: s[k] for k in ('id', 'status', 'error', 'reason') if k in s} for s in queue['steps']]), indent=2))
        return
    folder = (args.resume_run or args.run_dir or RUNS / (f'structure-study-seed{args.seed if args.seed is not None else 42}-' + stamp())).resolve()
    try:
        settings = (read_json(folder / 'queue.json')['settings'] if args.resume_run else
                    continuation_settings(args) if args.continue_from else new_settings(args))
    except ValueError as error:
        parser.error(str(error))
    if args.plan:
        print(__import__('json').dumps(dict(queue=str(folder), settings=settings,
            steps=recipe(settings), formal_jobs=len(settings['models']) * len(settings['datasets']),
            formal_epochs_per_job=settings.get('epochs', 100), automatic_test=False,
            note='Short budgets are screening; default screening runs V3 as a preset candidate. At 100 epochs V3 defaults to the validation gate. Smoke uses 3 epochs.'), indent=2))
        return
    # Hold the same GPU queue lock as the existing launcher. execute() reuses
    # its Python implementation and does not try to acquire the lock again.
    import fcntl
    native.RUNTIME.mkdir(parents=True, exist_ok=True)
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    with (native.RUNTIME / 'training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another training queue is running')
        execute(folder, settings, resume=bool(args.resume_run))


if __name__ == '__main__':
    main()
