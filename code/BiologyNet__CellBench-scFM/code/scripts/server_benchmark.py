#!/usr/bin/env python3
"""Portable handoff entry: plan/check/run/status for the full prepared-data benchmark."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bio.config import config_loader
from bio.data.catalog import CURATED, suite_datasets
from bio.data.foundation import MODELS as EXTERNAL
from bio.diagnostics.resources import REQUIRED
from bio.io.paths import project_path
from bio.runtime.suite import parse_seeds
from scripts.run_full_benchmark import ORDER, make_plan
from bio.runtime.devices import add_gpu_options, gpu_schedule, cuda_selector


def selected_jobs(models, datasets, seeds):
    return [j for j in make_plan(seeds, set()) if j['model'] in models and j['dataset'] in datasets]


def command(args):
    gpus, workers = gpu_schedule(getattr(args, 'gpu', None), getattr(args, 'gpus', None),
                                 getattr(args, 'max_workers', None), getattr(args, 'tasks_per_gpu', 1))
    return [sys.executable, '-u', str(ROOT / 'scripts/run_full_benchmark.py'),
            '--run-dir', str(args.run_dir.resolve()), '--models', *args.models,
            '--datasets', *args.datasets, '--seeds', args.seeds, '--gpus', ','.join(map(str, gpus)),
            '--max-workers', str(workers), '--tasks-per-gpu', str(getattr(args, 'tasks_per_gpu', 1)),
            '--timeout', str(args.timeout), '--release-model', 'CellFM', '--continue-on-error',
            *(['--retry-failed'] if args.retry_failed else [])]


def uc_sidecars(path):
    # Match _attach_uc_metadata: modern prepared matrices carry metadata internally.
    if not (path.parent / 'uc_epi_cell.tsv').is_file():
        return []
    return [path.parent / f'uc_epi_{suffix}.tsv' for suffix in
            ('cell', 'celltype', 'gene', 'batch_health', 'batch_location', 'batch_patient')]


def preflight(models, datasets, gpu):
    """Check exact transferred resources and interpreter imports; never load checkpoints."""
    errors, warnings, details = [], [], {}
    def require(path):
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f'Missing/empty resource: {path}')
    for dataset in datasets:
        path = ROOT / 'datasets' / CURATED[dataset]['filename']
        require(path)
        if not path.with_suffix('.provenance.json').is_file():
            warnings.append(f'{dataset}: provenance sidecar absent; copy all dataset sidecars')
        if dataset == 'uc_epi':
            for sidecar in uc_sidecars(path):
                require(sidecar)
        if dataset == 'mouse_heart':
            require(path.parent / 'mouse_heart_10x.h5ad')
        if path.is_file():
            try:
                import h5py
                with h5py.File(path, 'r') as data:
                    details[dataset] = {k: len(data[k][data[k].attrs.get('_index', '_index')]) for k in ('obs', 'var')}
                    if 'celltype' not in data['obs']:
                        errors.append(f'{dataset}: missing celltype labels')
                    if CURATED[dataset].get('batch') and 'batch' not in data['obs'] and not (dataset == 'uc_epi' and uc_sidecars(path)):
                        errors.append(f'{dataset}: missing batch metadata')
            except Exception as exc:
                errors.append(f'{dataset}: H5AD metadata check failed: {exc}')
    for model in models:
        for name in REQUIRED.get(model, []):
            require(ROOT / 'weights' / model / name)
        if model in REQUIRED:
            source = ROOT / 'public_code' / model
            if not source.is_dir() or not any(source.iterdir()):
                errors.append(f'Missing upstream code: {source}')
        if model in EXTERNAL:
            cfg = config_loader(ROOT / 'configs' / (model.lower() + '.yaml'))['model_cfg']
            python = project_path(os.environ.get(f'CELLBENCH_{model.upper()}_PYTHON', cfg.get('backend_python', sys.executable)))
            if not python.is_file():
                errors.append(f'{model}: missing backend interpreter: {python}')
                continue
            code = 'import torch; print(torch.__version__)' if model != 'CellFM' else 'import mindspore; print(mindspore.__version__)'
            if model == 'GeneCompass':
                code += "; import transformers; assert transformers.__version__ == '4.30.0', transformers.__version__"
            try:
                checked = subprocess.run([str(python), '-c', code], capture_output=True, text=True, timeout=120)
                if checked.returncode:
                    errors.append(f'{model}: backend import failed: {checked.stderr[-1200:]}')
                details[model] = dict(python=str(python), backend_device=cfg.get('backend_device'), version=checked.stdout.strip())
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f'{model}: {exc}')
    # Shared symbol recovery is also used by human-only encoders on the mouse panel.
    if any(d.startswith('mouse_') for d in datasets):
        require(ROOT / 'weights/CellFM/updated_hgcn.tsv')
    try:
        import torch
        import scanpy, scib, anndata  # noqa: F401
        if 'scVI' in models:
            import scvi  # noqa: F401
        if 'UMAP' in models:
            import umap  # noqa: F401
        if any(m in models for m in ('Geneformer', 'LangCell')):
            import geneformer  # noqa: F401
        details['main_python'] = sys.executable
        details['torch'] = torch.__version__
        details['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            # Parent is not CUDA-masked; child main.py applies --gpu before importing torch.
            gpus = [gpu] if isinstance(gpu, int) else gpu
            details['gpus'] = []
            for index in gpus:
                tensor = torch.ones(1, device=f'cuda:{index}')
                details['gpus'].append(dict(index=index, name=torch.cuda.get_device_name(tensor.device)))
            details['gpu'] = details['gpus'][0]['name']
        else:
            warnings.append('CUDA unavailable; main task heads will use CPU')
    except Exception as exc:
        errors.append(f'Main environment/GPU check failed: {exc}')
    warnings.append('Preflight does not validate peak memory, model loading, numerical equivalence, or full-scale runtime.')
    return dict(ready=not errors, errors=errors, warnings=warnings, details=details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'check', 'run', 'status'])
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--models', nargs='+', choices=ORDER, default=list(ORDER))
    parser.add_argument('--datasets', nargs='+', choices=suite_datasets(), default=list(suite_datasets()))
    parser.add_argument('--seeds', default='41,42,43')
    add_gpu_options(parser)
    parser.add_argument('--timeout', type=int, default=86400, help='Seconds per task, default 24 hours')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    try:
        gpus, workers = gpu_schedule(args.gpu, args.gpus, args.max_workers, args.tasks_per_gpu)
        for gpu in gpus:
            cuda_selector(gpu)
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if len(seeds) != len(set(seeds)):
        parser.error('duplicate seeds are not allowed')
    if args.action == 'status':
        if args.run_dir is None:
            parser.error('--run-dir is required for status')
        report = json.loads((args.run_dir / 'progress.json').read_text())
        jobs = [j for j in report['jobs'] if j['model'] in args.models and j['dataset'] in args.datasets and j['seed'] in seeds]
        print(json.dumps(dict(status_counts=dict(Counter(j['status'] for j in jobs)), controller_active_recorded=report.get('controller_active'), scheduler=report.get('scheduler'), updated_at=report.get('updated_at'), running=[j for j in jobs if j['status']=='running']), indent=2))
        return 0
    jobs = selected_jobs(args.models, args.datasets, seeds)
    print(json.dumps(dict(models=args.models, datasets=args.datasets, seeds=seeds, selected_steps=len(jobs), max_cells=0, head_epochs=500, scvi_epochs=400, continue_on_error=True, gpus=gpus, max_workers=workers, tasks_per_gpu=args.tasks_per_gpu), indent=2), flush=True)
    if args.action == 'plan':
        return 0
    report = preflight(args.models, args.datasets, gpus)
    print(json.dumps(report, indent=2), flush=True)
    if not report['ready']:
        return 2
    if args.action == 'check':
        return 0
    if args.run_dir is None:
        parser.error('--run-dir is required for run; use a fresh directory on the new server')
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'server_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    cmd = command(args)
    print('Launching:', shlex.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, PYTHONNOUSERSITE='1'))


if __name__ == '__main__':
    raise SystemExit(main())
