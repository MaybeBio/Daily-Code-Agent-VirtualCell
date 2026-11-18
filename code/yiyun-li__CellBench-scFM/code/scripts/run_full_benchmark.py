#!/usr/bin/env python3
"""Persistent full-data benchmark queue. Supports continuing past failures for final review."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, CancelledError
import signal
import threading
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bio.config import config_loader, MergeCfgsDict
from bio.data.catalog import suite_datasets
from bio.io.embeddings import fingerprint, SCHEMA, model_code_files
from bio.models import list_models
from bio.runtime.suite import parse_seeds, task_plan
from scripts.smoke_pipeline import run_job
from bio.runtime.devices import add_gpu_options, gpu_schedule, cuda_selector

HELD = {'CellFM'}
ORDER = ('PCA_model', 'CellPLM', 'Geneformer', 'LangCell', 'scGPT', 'Nicheformer',
         'UCE', 'scFoundation', 'scPRINT2', 'scVI', 'UMAP', 'CellFM',
         'TranscriptFormer', 'GeneCompass', 'scLong', 'Tabula')


def config_name(model):
    return 'pca' if model == 'PCA_model' else model.lower()


def shared_source_digest():
    digest = hashlib.sha256()
    for path in sorted(model_code_files('PCA_model')):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_plan(seeds, held):
    return [dict(model=model, dataset=dataset, seed=seed, task=task,
                 status='held' if model in held else 'pending', attempts=[])
            for model in ORDER for dataset in suite_datasets() for seed in seeds
            for task in task_plan(dataset, {'data_cfg': {}})[0]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'output/full_benchmark_20261029')
    parser.add_argument('--models', nargs='+', choices=list_models(), help='Dispatch only these models; retain the complete plan')
    parser.add_argument('--datasets', nargs='+', choices=suite_datasets())
    parser.add_argument('--seeds', default='41,42,43')
    add_gpu_options(parser)
    parser.add_argument('--timeout', type=int, default=86400)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Record failures, skip dependent tasks, and continue the queue')
    parser.add_argument('--release-model', nargs='+', choices=list_models(),
                        help='Only after the official source/checkpoint issue has been resolved')
    parser.add_argument('--hold-model', nargs='+', choices=list_models())
    parser.add_argument('--hold-pair', nargs='+', help='MODEL:DATASET awaiting a protocol decision')
    parser.add_argument('--release-pair', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.timeout <= 0 or (args.limit is not None and args.limit < 1):
        parser.error('timeout and limit must be positive')
    try:
        args.gpu_ids, args.workers = gpu_schedule(args.gpu, args.gpus, args.max_workers, args.tasks_per_gpu)
        for gpu in args.gpu_ids:
            cuda_selector(gpu)
        seeds = parse_seeds(args.seeds)
        if len(seeds) != len(set(seeds)):
            raise ValueError('duplicate seeds are not allowed')
    except ValueError as exc:
        parser.error(str(exc))
    folder = args.run_dir.resolve()
    path = folder / 'progress.json'
    report = json.loads(path.read_text()) if path.exists() else dict(
        started_at=datetime.now(timezone.utc).isoformat(), seeds=seeds,
        protocol='full prepared datasets, native mouse vocabularies or approved human 1:1 orthologs',
        held_models=sorted(HELD), jobs=make_plan(seeds, HELD))
    if seeds != report['seeds']:
        parser.error('Seed set differs from this run; use a new run directory')
    held = (set(report['held_models']) - set(args.release_model or [])) | set(args.hold_model or [])
    pairs = (set(report.get('held_pairs', [])) - set(args.release_pair or [])) | set(args.hold_pair or [])
    for pair in pairs:
        model, separator, dataset = pair.partition(':')
        if not separator or model not in list_models() or dataset not in suite_datasets():
            parser.error(f'Invalid held pair: {pair}')
    report['held_models'] = sorted(held)
    report['held_pairs'] = sorted(pairs)
    report['run_kind'] = 'full_benchmark'
    for job in report['jobs']:
        is_held = job['model'] in held or f'{job["model"]}:{job["dataset"]}' in pairs
        if is_held:
            job.update(status='held', reason='Official-source, hardware, or input-protocol resolution pending')
        elif job['status'] == 'held':
            job['status'] = 'pending'
            job.pop('reason', None)
    selected = [j for j in report['jobs']
                if (not args.models or j['model'] in args.models)
                and (not args.datasets or j['dataset'] in args.datasets)]
    print(f'Full plan: {len(report["jobs"])} steps; selected: {len(selected)}; held models: {sorted(held)}', flush=True)
    print(f'GPU slots: {args.gpu_ids}; max concurrent tasks: {args.workers}; per GPU: {args.tasks_per_gpu}', flush=True)
    if args.dry_run:
        print(json.dumps(dict(Counter(j['status'] for j in selected)), indent=2))
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('A controller is already running in this directory')
        report.pop('controller_error', None)
        report['selected_complete'] = False
        report['controller_pid'] = __import__('os').getpid()
        report['controller_active'] = True
        report['source_changed_during_controller'] = False
        source_at_start = shared_source_digest()
        overlay = dict(output_dir=str(folder), data_cfg=dict(max_cells=0,
                       species_input_policy='native_or_human_one_to_one'), model_cfg=dict(head_epochs=500))
        import yaml
        effective = folder / 'effective.yaml'
        effective.write_text(yaml.safe_dump(overlay))

        def save():
            report['status_counts'] = dict(Counter(j['status'] for j in report['jobs']))
            report['complete'] = (not report['controller_active']
                                  and not report['source_changed_during_controller']
                                  and all(j['status'] == 'passed' for j in report['jobs']))
            report['updated_at'] = datetime.now(timezone.utc).isoformat()
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            temp.replace(path)

        gpu_ids, workers = args.gpu_ids, args.workers
        report['scheduler'] = dict(gpus=gpu_ids, max_workers=workers,
                                   tasks_per_gpu=args.tasks_per_gpu,
                                   lane='model_dataset', gpu_indices='within_CUDA_VISIBLE_DEVICES')
        save()
        dispatched = 0
        failed_embeddings = set()
        pending = list(selected)
        active = {}
        busy_lanes = set()
        gpu_load = dict.fromkeys(gpu_ids, 0)
        cancel = threading.Event()
        stop_code = None
        signal_code = 130
        pool = ThreadPoolExecutor(max_workers=workers)

        def finish(future):
            nonlocal stop_code
            job, attempt, start, gpu = active.pop(future)
            busy_lanes.remove((job['model'], job['dataset']))
            gpu_load[gpu] -= 1
            try:
                code = future.result()
                attempt.update(returncode=code, status='passed' if code == 0 else 'failed')
                signature = attempt['signature']
                if code == 0 and job['task'] == 'embedding_extraction':
                    cache = folder / 'embeddings' / f'{job["dataset"]}_{job["model"]}' / 'v2' / signature
                    manifest = json.loads((cache / 'manifest.json').read_text())
                    if manifest['schema'] != SCHEMA or manifest['signature'] != signature or not (cache / 'all.pkl').is_file():
                        raise ValueError('Missing or incompatible embedding artifact')
                    attempt['embedding_shape'] = manifest['shape']
                elif code == 0:
                    from bio.io.export import PRIMARY
                    records = []
                    for line in Path(attempt['log']).read_text(errors='replace').splitlines():
                        if line.startswith('Metrics saved: '):
                            record_path = Path(line.split('Metrics saved: ', 1)[1])
                            record = json.loads(record_path.read_text())
                            if all(record[k] == job[k] for k in ('model', 'dataset', 'seed', 'task')):
                                records.append((record_path, record))
                    if not records:
                        raise ValueError('Worker exited successfully but no matching metric artifact was emitted')
                    metric_path, record = records[-1]
                    attempt.update(metrics=str(metric_path),
                        null_primary=[k for k in PRIMARY[job['task']] if record['summary'].get(k) is None])
            except subprocess.TimeoutExpired:
                attempt.update(status='timeout', error=f'timeout after {args.timeout}s')
            except (InterruptedError, CancelledError, KeyboardInterrupt):
                attempt.update(status='interrupted', error='Task cancelled by controller')
            except Exception as exc:
                attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            attempt['seconds'] = round(time.monotonic() - start, 1)
            job.update(status=attempt['status'], latest=attempt)
            if job['status'] != 'passed':
                if job['task'] == 'embedding_extraction':
                    failed_embeddings.add((job['model'], job['dataset'], job['seed']))
                if not args.continue_on_error or job['status'] == 'interrupted':
                    stop_code = stop_code or (130 if job['status'] == 'interrupted' else 1)
                    cancel.set()
            save()
            print(f'{job["status"].upper()} {job["model"]}/{job["dataset"]}/seed{job["seed"]}/{job["task"]} '
                  f'gpu={gpu} ({attempt["seconds"]}s)', flush=True)

        def interrupted(signum, frame):
            nonlocal signal_code
            signal_code = 128 + signum
            raise KeyboardInterrupt

        previous_term = None
        if threading.current_thread() is threading.main_thread():
            previous_term = signal.signal(signal.SIGTERM, interrupted)
        try:
            while pending or active:
                if stop_code is not None or (args.limit is not None and dispatched >= args.limit):
                    pending.clear()
                index = 0
                while index < len(pending) and len(active) < workers and stop_code is None:
                    if args.limit is not None and dispatched >= args.limit:
                        pending.clear()
                        break
                    job = pending[index]
                    lane = (job['model'], job['dataset'])
                    if lane in busy_lanes:
                        index += 1
                        continue
                    pending.pop(index)
                    if job['status'] == 'held':
                        continue
                    dependency = (job['model'], job['dataset'], job['seed'])
                    if job['task'] != 'embedding_extraction' and dependency in failed_embeddings:
                        if job['status'] != 'passed':
                            job.update(status='skipped', reason='Required embedding extraction failed')
                            save()
                        continue
                    if job['status'] in ('failed', 'timeout', 'interrupted') and not args.retry_failed:
                        print(f'Unresolved {job["model"]}/{job["dataset"]}/{job["seed"]}/{job["task"]}; '
                              'repair then use --retry-failed', flush=True)
                        if job['task'] == 'embedding_extraction':
                            failed_embeddings.add(dependency)
                        if not args.continue_on_error:
                            stop_code = 1
                            cancel.set()
                        continue
                    model_path = ROOT / 'configs' / f'{config_name(job["model"])}.yaml'
                    cfg = config_loader(model_path)
                    MergeCfgsDict(overlay, cfg)
                    cfg.update(seed=job['seed'], topk=0, phase='adapter_finetuning')
                    cfg['data_cfg']['dataset_name'] = job['dataset']
                    cfg['model_cfg']['tasks'] = job['task']
                    signature = fingerprint(cfg)[0]
                    if job['status'] == 'passed' and job.get('signature') == signature:
                        continue
                    gpu = min((g for g in gpu_ids if gpu_load[g] < args.tasks_per_gpu), key=gpu_load.get)
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                    log = folder / 'logs' / job['model'] / f'{job["dataset"]}_seed{job["seed"]}_{job["task"]}_{stamp}.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-u', str(ROOT / 'main.py'), '--cfgs', str(model_path),
                               '--extra-cfg', str(effective), '--dataset_name', job['dataset'],
                               '--task', job['task'], '--seed', str(job['seed']), '--gpu', str(gpu)]
                    attempt = dict(started_at=datetime.now(timezone.utc).isoformat(), command=command,
                                   signature=signature, log=str(log), status='running',
                                   gpu=gpu, cuda_visible_devices=cuda_selector(gpu))
                    job['attempts'].append(attempt)
                    job.update(status='running', signature=signature, log=str(log), gpu=gpu)
                    job.pop('reason', None)
                    save()
                    busy_lanes.add(lane)
                    gpu_load[gpu] += 1
                    print(f'RUN {job["model"]}/{job["dataset"]}/seed{job["seed"]}/{job["task"]} '
                          f'gpu={gpu}: {log}', flush=True)
                    future = pool.submit(run_job, command, log, args.timeout, cancel_event=cancel)
                    active[future] = (job, attempt, time.monotonic(), gpu)
                    dispatched += 1
                if active:
                    done, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        finish(future)
        except KeyboardInterrupt:
            stop_code = signal_code
        except Exception as exc:
            report['controller_error'] = f'{type(exc).__name__}: {exc}'
            stop_code = 1
        finally:
            cancel.set()
            # run_job observes cancellation and kills the entire child process group.
            pool.shutdown(wait=True)
            for future in list(active):
                finish(future)
            if previous_term is not None:
                signal.signal(signal.SIGTERM, previous_term)
            report['controller_active'] = False
            report['source_changed_during_controller'] = shared_source_digest() != source_at_start
            report['selected_status_counts'] = dict(Counter(j['status'] for j in selected))
            report['selected_complete'] = all(j['status'] == 'passed' for j in selected) and not report['source_changed_during_controller']
            save()
            failures = [j for j in selected if j['status'] in ('failed', 'timeout', 'interrupted', 'skipped')]
            (folder / 'failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2) + '\n')
        print(f'State: {report["status_counts"]}; full completion: {report["complete"]}; report: {path}', flush=True)
        return stop_code or int(bool(failures) or report['source_changed_during_controller'])


if __name__ == '__main__':
    raise SystemExit(main())
