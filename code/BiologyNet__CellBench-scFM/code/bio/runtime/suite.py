"""Run dataset-appropriate tasks in isolated subprocesses through the same CLI."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from bio.io.paths import PROJECT_ROOT

from bio.data.catalog import CURATED, panel_datasets, suite_datasets

STANDARD = ('embedding_extraction', 'clustering', 'annotation', 'reconstruction')
DATASETS = suite_datasets()
BATCHED = {name for name, spec in CURATED.items() if spec.get('batch')}
PERTURBATION = {name for name, spec in CURATED.items() if spec.get('perturbation')}


def parse_seeds(value):
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        values = [int(v) for v in value]
        if not values:
            raise ValueError('At least one seed is required')
        return values
    parts = [p.strip() for p in str(value).split(',') if p.strip()]
    if not parts:
        raise ValueError('At least one seed is required')
    return [int(p) for p in parts]


def resolve_datasets(opt):
    panel = getattr(opt, 'panel', None)
    dataset = opt.dataset_name
    if panel:
        names = panel_datasets(panel)
        if dataset and dataset not in names:
            raise ValueError(f'{dataset!r} is not in the {panel} panel ({", ".join(names)})')
        return [dataset] if dataset else names
    if dataset:
        if dataset not in CURATED:
            raise ValueError(f'Unsupported dataset {dataset!r}; choose from {", ".join(CURATED)}')
        return [dataset]
    if not getattr(opt, 'cfgs', None):
        raise ValueError('Select a dataset with --dataset/--dataset_name or a section with --panel')
    from bio.config import config_loader
    from bio.io.paths import project_path
    name = config_loader(project_path(opt.cfgs))['data_cfg']['dataset_name']
    if name not in CURATED:
        raise ValueError(f'Unsupported dataset {name!r}; choose from {", ".join(CURATED)}')
    return [name]


def task_plan(dataset, cfg):
    if dataset not in CURATED:
        raise ValueError(f'Unsupported suite dataset {dataset!r}; choose from {", ".join(CURATED)}')
    skipped = {}
    if dataset in PERTURBATION:
        tasks = ['embedding_extraction', 'perturbation']
        for task in ('clustering', 'annotation', 'reconstruction', 'correction'):
            skipped[task] = 'Not part of the perturbation dataset evaluation protocol.'
    else:
        tasks = list(STANDARD)
        batch_key = cfg['data_cfg'].get('dataset_batch_keys', {}).get(dataset)
        if dataset in BATCHED or batch_key:
            tasks.insert(2, 'correction')
        else:
            skipped['correction'] = 'No batch mapping declared for this dataset; configure data_cfg.dataset_batch_keys to enable it.'
        skipped['perturbation'] = 'Not a perturbation dataset.'
    return tasks, skipped


def selected_configs(opt, cfg=None):
    from bio.config import config_loader
    from bio.models import list_models
    from bio.io.paths import project_path
    if cfg is not None:
        return [(project_path(opt.cfgs), cfg)]
    if opt.cfgs:
        path = project_path(opt.cfgs)
        return [(path, config_loader(path))]
    result = []
    for model in list_models():
        name = 'pca' if model == 'PCA_model' else model.lower()
        path = project_path(f'configs/{name}.yaml')
        config = config_loader(path)
        if config['model_cfg']['model'] != model:
            raise ValueError(f'{path} does not configure registered model {model}')
        result.append((path, config))
    return result


def run_suite(opt, cfg=None):
    configs = selected_configs(opt, cfg)
    datasets = resolve_datasets(opt)
    seeds = parse_seeds(opt.seed)
    commands = []
    skipped = {}
    model_names = []
    for config_path, config in configs:
        model = config['model_cfg']['model']
        model_names.append(model)
        for dataset in datasets:
            tasks, reasons = task_plan(dataset, config)
            skipped.setdefault(dataset, {})[model] = reasons
            requested = getattr(opt, 'task', 'all')
            if requested and requested != 'all':
                if requested in tasks:
                    tasks = [requested]
                else:
                    reasons = dict(reasons)
                    reasons[requested] = 'Not in this dataset task plan.'
                    skipped[dataset][model] = reasons
                    tasks = []
            for seed in seeds:
                for task in tasks:
                    command = [sys.executable, str(PROJECT_ROOT/'main.py'), '--cfgs', str(config_path),
                               '--dataset_name', dataset, '--task', task,
                               '--phase', opt.phase, '--seed', str(seed)]
                    extra = getattr(opt, 'extra_cfg', None)
                    if extra:
                        command.extend(['--extra-cfg', str(extra)])
                    for flag in ('gpu', 'hvg'):
                        if getattr(opt, flag) is not None:
                            command.extend(['--'+flag, str(getattr(opt, flag))])
                    commands.append((model, dataset, seed, task, command))
    print(f'Datasets: {", ".join(datasets)}; models: {len(model_names)}; seeds: {seeds}; planned tasks: {len(commands)}', flush=True)
    for model, dataset, seed, task, command in commands:
        print(f'[{model}/{dataset}/seed{seed}/{task}] {shlex.join(command)}', flush=True)
    for dataset, models in skipped.items():
        for model, reasons in models.items():
            for task, reason in reasons.items():
                print(f'[skip {model}/{dataset}/{task}] {reason}', flush=True)
    if opt.dry_run:
        return 0
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    folder = PROJECT_ROOT/'output'/'suites'/run_id
    folder.mkdir(parents=True)
    report = dict(datasets=datasets, models=model_names, seeds=seeds, panel=getattr(opt, 'panel', None),
                  skipped=skipped, tasks=[])
    def save():
        path = folder/'summary.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        temp.replace(path)
    failed = set()
    stop = False
    for model, dataset, seed, task, command in commands:
        dependency_failed = task != 'embedding_extraction' and (model, dataset, seed, 'embedding_extraction') in failed
        if stop or dependency_failed:
            reason = 'Stopped by --fail-fast.' if stop else 'Required task failed for this model.'
            report['tasks'].append(dict(model=model, dataset=dataset, seed=seed, task=task,
                                        command=command, status='skipped', reason=reason))
            save()
            continue
        log = folder/model/f'{dataset}_seed{seed}_{task}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(model=model, dataset=dataset, seed=seed, task=task, command=command, log=str(log), status='running')
        report['tasks'].append(entry)
        save()
        print(f'Running {model}/{dataset}/seed{seed}/{task}; log: {log}', flush=True)
        try:
            with log.open('w') as handle:
                result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
            entry.update(status='passed' if result.returncode == 0 else 'failed', returncode=result.returncode)
        except OSError as exc:
            entry.update(status='failed', error=str(exc))
        except KeyboardInterrupt:
            entry.update(status='interrupted')
            save()
            print(f'Interrupted. Summary: {folder / "summary.json"}', flush=True)
            return 130
        if entry['status'] == 'failed':
            failed.add((model, dataset, seed, task))
            stop = opt.fail_fast
        save()
        print(f'{model}/{dataset}/seed{seed}/{task}: {entry["status"]}', flush=True)
    print(f'Summary: {folder / "summary.json"}', flush=True)
    return int(bool(failed))
