#!/usr/bin/env python3
"""Run small-data checks in a separate directory; resume with --resume RUN_DIR."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bio.config import config_loader
from bio.data.catalog import suite_datasets
from bio.data.foundation import input_path
from bio.models import list_models
from bio.runtime.suite import task_plan


def config_name(model):
    return 'pca' if model == 'PCA_model' else model.lower()


def plan(models, datasets):
    jobs = []
    for model in models:
        for dataset in datasets:
            tasks, _ = task_plan(dataset, {'data_cfg': {}})
            for task in tasks:
                jobs.append(dict(model=model, dataset=dataset, task=task,
                                 config=config_name(model), status='pending'))
    return jobs


def run_job(command, log, timeout, cancel_event=None):
    """Kill the entire worker group, including isolated model backends, on timeout."""
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError('Task cancelled before launch')
    with log.open('w') as handle:
        worker = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if cancel_event is None:
                return worker.wait(timeout=timeout)
            deadline = time.monotonic() + timeout
            while True:
                if cancel_event.is_set():
                    raise InterruptedError('Task cancelled by controller')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    return worker.wait(timeout=min(0.2, remaining))
                except subprocess.TimeoutExpired:
                    pass
        except (subprocess.TimeoutExpired, KeyboardInterrupt, InterruptedError):
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait()
            raise


def contract(jobs, extra, seed):
    """Reject resume after source/config/data/checkpoint changes."""
    from bio.config import MergeCfgsDict
    import yaml
    payload = dict(seed=seed, extra=extra, files=[], resources=[])
    files = set((ROOT / 'bio').rglob('*.py')) | {Path(__file__)}
    files.add(ROOT / 'configs/default.yaml')
    for model in dict.fromkeys(j['config'] for j in jobs):
        path = ROOT / 'configs' / f'{model}.yaml'
        files.add(path)
        cfg = config_loader(path)
        MergeCfgsDict(yaml.safe_load(extra) or {}, cfg)
        weights = ROOT / cfg['model_cfg'].get('weights_dir', f"weights/{cfg['model_cfg']['model']}")
        for p in sorted(weights.rglob('*')):
            if p.is_file():
                stat = p.stat()
                payload['resources'].append([str(p), stat.st_size, stat.st_mtime_ns])
    for dataset in dict.fromkeys(j['dataset'] for j in jobs):
        cfg = {'data_cfg': {'dataset_name': dataset}}
        # Respect alternate input roots in the smoke overlay.
        MergeCfgsDict(yaml.safe_load(extra) or {}, cfg)
        cfg['data_cfg']['dataset_name'] = dataset
        path = input_path(cfg)
        stat = path.stat()
        payload['resources'].append([str(path), stat.st_size, stat.st_mtime_ns])
    for p in sorted(files):
        payload['files'].append([str(p), hashlib.sha256(p.read_bytes()).hexdigest()])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='Default: 41, or the saved seed when resuming')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--extra-cfg', help='Default: configs/smoke.yaml, or the saved overlay when resuming')
    parser.add_argument('--models', nargs='+', choices=list_models())
    parser.add_argument('--datasets', nargs='+', choices=suite_datasets())
    parser.add_argument('--run-dir', type=Path, help='New, empty directory for all outputs')
    parser.add_argument('--resume', type=Path, help='Existing run directory; retry unfinished/failed jobs')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.resume and args.run_dir:
        parser.error('--resume and --run-dir are mutually exclusive')
    report = {}
    if args.resume:
        folder = args.resume.resolve()
        report = json.loads((folder / 'progress.json').read_text())
    args.seed = args.seed if args.seed is not None else report.get('seed', 41)
    extra_path = ROOT / (args.extra_cfg or report.get('extra_cfg', 'configs/smoke.yaml'))
    extra = (report['extra_yaml'] if args.resume and not args.extra_cfg and 'extra_yaml' in report
             else extra_path.read_text())
    if args.resume:
        jobs = report['jobs']
        if args.models or args.datasets:
            parser.error('--resume uses the models/datasets saved in the run')
        if args.seed != report['seed'] or contract(jobs, extra, args.seed) != report.get('contract'):
            parser.error('Source/config/data/weights or seed changed; start a new run')
    else:
        models = args.models or list_models()
        datasets = args.datasets or suite_datasets()
        jobs = plan(models, datasets)
        folder = (args.run_dir or ROOT / 'output/smoke/runs' /
                  (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])).resolve()
        report = dict(started_at=datetime.now(timezone.utc).isoformat(), seed=args.seed,
                      extra_cfg=str(extra_path), extra_yaml=extra, n_jobs=len(jobs), jobs=jobs)
    print(f'Smoke plan: {len(jobs)} jobs; outputs: {folder}', flush=True)
    if args.dry_run:
        for job in jobs:
            print(f'  {job["model"]}/{job["dataset"]}/{job["task"]}: {job["status"]}')
        return 0
    if not args.resume:
        report['contract'] = contract(jobs, extra, args.seed)
        folder.mkdir(parents=True, exist_ok=False)
    log_dir = folder / 'logs'
    log_dir.mkdir(exist_ok=True)
    import yaml
    overlay = yaml.safe_load(extra) or {}
    overlay['output_dir'] = str(folder)
    effective = folder / 'effective.yaml'
    effective.write_text(yaml.safe_dump(overlay))
    report.update(timeout=args.timeout, controller_pid=os.getpid())
    report.pop('finished_at', None)

    def save():
        report['passed'] = sum(j['status'] == 'passed' for j in jobs)
        report['failed'] = sum(j['status'] in ('failed', 'timeout') for j in jobs)
        report['skipped'] = sum(j['status'] == 'skipped' for j in jobs)
        path = folder / 'progress.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(path)

    save()
    failed_embeddings = set()
    for job in jobs:
        key = job['model'], job['dataset']
        if job['status'] == 'passed':
            continue
        if job['task'] != 'embedding_extraction' and key in failed_embeddings:
            job.update(status='skipped', reason='Required embedding extraction failed')
            save()
            continue
        command = [sys.executable, '-u', str(ROOT / 'main.py'), '--cfgs',
                   str(ROOT / 'configs' / f'{job["config"]}.yaml'),
                   '--extra-cfg', str(effective), '--dataset_name', job['dataset'],
                   '--task', job['task'], '--seed', str(args.seed)]
        if args.gpu is not None:
            command += ['--gpu', str(args.gpu)]
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        log = log_dir / f'{job["model"]}_{job["dataset"]}_{job["task"]}_{stamp}.log'
        job.pop('reason', None)
        job.pop('error', None)
        job.pop('returncode', None)
        job.update(status='running', command=command, log=str(log))
        save()
        print(f'RUN {job["model"]}/{job["dataset"]}/{job["task"]}', flush=True)
        t0 = time.monotonic()
        try:
            rc = run_job(command, log, args.timeout)
            job.update(returncode=rc, status='passed' if rc == 0 else 'failed')
        except subprocess.TimeoutExpired:
            job.update(status='timeout', error=f'timeout after {args.timeout}s')
        except OSError as exc:
            job.update(status='failed', error=str(exc))
        except KeyboardInterrupt:
            job['status'] = 'interrupted'
            save()
            return 130
        job['seconds'] = round(time.monotonic() - t0, 1)
        save()
        print(f'{job["status"].upper()} {job["model"]}/{job["dataset"]}/{job["task"]} ({job["seconds"]}s)', flush=True)
        if job['status'] != 'passed':
            if job['task'] == 'embedding_extraction':
                failed_embeddings.add(key)
            if args.fail_fast:
                break
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    print(f'Smoke finished: passed={report["passed"]} failed={report["failed"]} skipped={report["skipped"]}; report: {folder / "progress.json"}', flush=True)
    return int(any(j['status'] != 'passed' for j in jobs))


if __name__ == '__main__':
    raise SystemExit(main())
