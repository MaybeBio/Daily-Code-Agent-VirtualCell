#!/usr/bin/env python3
"""Sequentially fill missing organ embeddings without overwriting older caches.

Existing manifests are inventoried, not certified as compatible with new code.
Failures are recorded and the queue continues with the next model/dataset.
Re-running preserves failures unless --retry-failed is explicitly requested.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bio.data.catalog import panel_datasets
from bio.models import list_models


def inventory(dataset, model):
    results = []
    for path in sorted((ROOT / 'output' / 'embeddings' / f'{dataset}_{model}').glob('v2/*/manifest.json')):
        cache = path.parent / 'all.pkl'
        if cache.is_file() and cache.stat().st_size:
            record = json.loads(path.read_text())
            results.append(dict(path=str(cache), shape=record.get('shape'),
                                signature=record.get('signature')))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry previously recorded failures instead of skipping them')
    args = parser.parse_args()
    folder = ROOT / 'output' / 'organ_embedding_continuation'
    folder.mkdir(parents=True, exist_ok=True)
    previous_path = folder / 'progress.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    failures = {(t['model'], t['dataset']): t for t in previous.get('tasks', [])
                if t['status'] == 'failed'}
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), seed=args.seed,
                  controller_pid=os.getpid(), waiting_for_pid=args.wait_pid, tasks=[])
    for model in list_models():
        for dataset in panel_datasets('organ'):
            caches = inventory(dataset, model)
            if (model, dataset) in failures and not caches and not args.retry_failed:
                report['tasks'].append(dict(failures[(model, dataset)], skipped_previous_failure=True))
                continue
            report['tasks'].append(dict(
                model=model, dataset=dataset, existing_caches=caches,
                status='existing_unverified' if caches else 'pending'))
    def save():
        path = folder / 'progress.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(path)
    save()
    if args.wait_pid and not args.dry_run:
        while Path(f'/proc/{args.wait_pid}').exists():
            time.sleep(5)
    report['waiting_for_pid'] = None
    save()
    for entry in report['tasks']:
        model, dataset = entry['model'], entry['dataset']
        if entry['status'] == 'failed':
            print(f'Skipping recorded failure {model}/{dataset}; continuing queue.', flush=True)
            continue
        config = 'pca' if model == 'PCA_model' else model.lower()
        caches = inventory(dataset, model)
        entry.update(existing_caches=caches,
                     status='existing_unverified' if caches else 'pending')
        save()
        if caches:
            print(f'Existing {model}/{dataset}: {caches[-1]["shape"]}', flush=True)
            continue
        command = [sys.executable, '-u', str(ROOT / 'main.py'), '--cfgs',
                   str(ROOT / 'configs' / f'{config}.yaml'), '--dataset_name', dataset,
                   '--task', 'embedding_extraction', '--seed', str(args.seed), '--gpu', str(args.gpu)]
        if args.dry_run:
            print(' '.join(command), flush=True)
            continue
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        log = folder / f'{model}_{dataset}_{stamp}.log'
        entry.update(status='running', command=command, log=str(log))
        save()
        print(f'Extracting {model}/{dataset}; log: {log}', flush=True)
        try:
            with log.open('w') as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            outputs = inventory(dataset, model)
            entry.update(returncode=result.returncode, outputs=outputs,
                         status='passed' if result.returncode == 0 and outputs else 'failed')
        except OSError as exc:
            entry.update(status='failed', error=str(exc))
        except KeyboardInterrupt:
            entry.update(status='interrupted')
            save()
            return 130
        save()
        print(f'{model}/{dataset}: {entry["status"]}', flush=True)
        if entry['status'] == 'failed':
            print('Recorded failure; continuing with the next task.', flush=True)
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report['failed_tasks'] = sum(t['status'] == 'failed' for t in report['tasks'])
    save()
    return int(report['failed_tasks'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
