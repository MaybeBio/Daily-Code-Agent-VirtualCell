#!/usr/bin/env python3
"""Audit task artifacts and export all five tasks by panel after the queue finishes."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bio.io.export import PRIMARY, export_panels
from bio.io.embeddings import SCHEMA


def finish(folder):
    folder = Path(folder).resolve()
    report = json.loads((folder / 'progress.json').read_text())
    errors, nulls, exports, rows = [], [], {}, []
    for job in report['jobs']:
        latest = job.get('latest', {})
        rows.append({k: job.get(k) for k in ('model', 'dataset', 'seed', 'task', 'status')} | dict(error=latest.get('error'), log=latest.get('log'), metrics=latest.get('metrics')))
        if job['status'] != 'passed':
            continue
        try:
            if job['task'] == 'embedding_extraction':
                cache = folder / 'embeddings' / f"{job['dataset']}_{job['model']}" / 'v2' / job['signature']
                artifact = json.loads((cache / 'manifest.json').read_text())
                if artifact.get('schema') != SCHEMA or artifact['signature'] != job['signature'] or not (cache / 'all.pkl').is_file():
                    raise ValueError('Embedding artifact missing or incompatible')
            else:
                artifact = json.loads(Path(latest['metrics']).read_text())
                if artifact.get('metric_version') != 2 or any(artifact.get(k) != job[k] for k in ('model', 'dataset', 'seed', 'task')):
                    raise ValueError('Metric identity does not match its task')
                missing = [k for k in PRIMARY[job['task']] if not isinstance(artifact['summary'].get(k), (int, float)) or not math.isfinite(artifact['summary'][k])]
                if missing:
                    nulls.append({k: job[k] for k in ('model', 'dataset', 'seed', 'task')} | dict(null_primary=missing))
        except (OSError, ValueError, KeyError) as exc:
            errors.append({k: job[k] for k in ('model', 'dataset', 'seed', 'task')} | dict(error=str(exc)))
    for task in PRIMARY:
        try:
            exports[task] = export_panels(task, folder)
        except FileNotFoundError as exc:
            exports[task] = dict(unavailable=str(exc))
            if any(j['task'] == task and j['status'] == 'passed' for j in report['jobs']):
                errors.append(dict(task=task, export_error=str(exc)))
        except Exception as exc:
            errors.append(dict(task=task, export_error=str(exc)))
    counts = dict(Counter(j['status'] for j in report['jobs']))
    passed = bool(report.get('complete')) and not errors
    summary = dict(all_planned_passed=passed, status_counts=counts, artifact_errors=errors,
                   null_primary=nulls, all_primary_metrics_available=passed and not nulls,
                   exports=exports, source_changed=report.get('source_changed_during_controller'),
                   note='See panel scores, ranks and coverage under exports/v2; null metrics and incomplete model blocks are not silently scored.')
    (folder / 'final_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (folder / 'task_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'dataset', 'seed', 'task', 'status', 'error', 'log', 'metrics'])
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({k: summary[k] for k in ('all_planned_passed', 'status_counts', 'all_primary_metrics_available')}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    raise SystemExit(finish(parser.parse_args().run_dir))
