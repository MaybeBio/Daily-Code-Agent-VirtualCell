"""Export v2 scores and scale-invariant ranks on complete paired benchmark blocks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bio.metrics import METRIC_VERSION
from bio.io.paths import project_path
from bio.data.catalog import study_id, PANEL_ORDER, panel_datasets

PRIMARY = {
    'clustering': {'ari': True, 'nmi': True, 'asw': True, 'acc': True},
    'annotation': {'acc': True, 'f1_m': True, 'balanced_accuracy': True, 'precision_macro': True},
    'reconstruction': {'Pearson_expr': True, 'Spearman_expr': True,
                       'MAE_expr': False, 'TopK_expression_overlap': True},
    'perturbation': {'Pearson_delta': True, 'Spearman_delta': True, 'MAE_delta': False, 'DES': True},
    'correction': {'ARI': True, 'ASW_celltype': True, 'iLISI_scaled': True, 'kBET': True},
}

# These describe a fitted representation, not the evaluation protocol shared
# by competing models. They can vary by model, dataset and random seed.
REPRESENTATION_FIELDS = {
    'hvg_source', 'hvg_split_mode', 'n_hvg', 'n_train_cells_for_hvg',
    'fit_split_mode', 'n_fit_cells',
}


def paired_ranks(frame, primary, expected_models=None):
    """Rank only complete dataset/seed blocks, then average seeds, studies, metrics.

    Missing methods/metrics exclude a whole block, never reward missing failures.
    Different shot budgets, phases and metric protocols are never pooled.
    """
    keys = ['topk', 'phase', 'protocol']
    ranks, coverage = [], []
    for protocol, part in frame.groupby(keys, dropna=False):
        models = set(part['model']) if expected_models is None else set(expected_models)
        for (dataset, seed), block in part.groupby(['dataset','seed'], dropna=False):
            complete = len(models) >= 2 and set(block.model) == models
            complete &= not block.model.duplicated().any()
            complete &= all(key in block and np.isfinite(pd.to_numeric(block[key], errors='coerce')).all() for key in primary)
            coverage.append(dict(zip(keys,protocol), dataset=dataset, seed=seed, complete=bool(complete),
                                 expected_models=len(models), observed_models=len(set(block.model))))
            if not complete:
                continue
            for metric, higher in primary.items():
                values=block[metric].rank(method='average',ascending=not higher)
                for model, rank in zip(block.model,values):
                    ranks.append(dict(zip(keys,protocol),dataset=dataset,seed=seed,model=model,metric=metric,rank=rank))
    if not ranks:
        return pd.DataFrame(), pd.DataFrame(coverage)
    r=pd.DataFrame(ranks).groupby(keys+['dataset','model','metric'],dropna=False)['rank'].mean().reset_index()
    r['study']=r['dataset'].map(study_id)
    r=r.groupby(keys+['study','model','metric'],dropna=False)['rank'].mean().reset_index()
    r=r.groupby(keys+['model','metric'],dropna=False)['rank'].mean().reset_index()
    overall=r.groupby(keys+['model'],dropna=False)['rank'].mean().reset_index(name='mean_metric_rank')
    return overall.sort_values(keys+['mean_metric_rank']), pd.DataFrame(coverage)


def _load_task_frame(task, root):
    rows=[]
    for path in sorted((root/'metrics'/task).glob('*.json')):
        record=json.loads(path.read_text())
        if record.get('metric_version',1) != METRIC_VERSION:
            continue
        cfg=record['config']
        options = {k: v for k, v in cfg.get('evaluation_cfg', {}).items()
                   if k not in REPRESENTATION_FIELDS}
        protocol=dict(evaluation=options,
                      pipeline_revision=record.get('pipeline_revision', 0),
                      scale=cfg.get('data_cfg',{}).get('evaluation_expression_scale','dataset_defined'))
        row={k:record.get(k) for k in ['dataset','model','seed','topk','phase','updated_at','run_id']}
        row['phase']=row['phase'] or 'unspecified'
        row['protocol']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
        row['metric_version']=METRIC_VERSION
        row.update({k:v for k,v in record['summary'].items() if isinstance(v,(int,float)) or v is None})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No v{METRIC_VERSION} {task} results in {root}')
    frame=pd.DataFrame(rows).sort_values('updated_at')
    frame=frame.drop_duplicates(['dataset','model','seed','topk','phase','protocol'],keep='last')
    return frame


def _write_export(task, frame, root, panel=None):
    destination=root/'exports'/'v2'
    if panel:
        destination=destination/panel
    destination.mkdir(parents=True,exist_ok=True)
    frame.to_csv(destination/f'{task}_scores.csv',index=False)
    expected = None
    progress = root / 'progress.json'
    if progress.is_file():
        plan = json.loads(progress.read_text())
        if plan.get('run_kind') == 'full_benchmark':
            expected = {job['model'] for job in plan['jobs'] if job['task'] == task}
    rank,coverage=paired_ranks(frame,PRIMARY[task],expected_models=expected)
    rank.to_csv(destination/f'{task}_ranks.csv',index=False)
    coverage.to_csv(destination/f'{task}_coverage.csv',index=False)
    print(f'Exported {len(frame)} {task} runs to {destination}; {int(coverage.complete.sum()) if len(coverage) else 0} complete comparison blocks')
    return frame,rank,coverage


def export_task(task, root='output', datasets=None, panel=None):
    if task not in PRIMARY:
        raise ValueError(f'Unknown task {task}')
    root=project_path(root)
    if panel is not None:
        datasets = panel_datasets(panel)
    frame=_load_task_frame(task, root)
    if datasets is not None:
        allowed=set(datasets)
        frame=frame[frame['dataset'].isin(allowed)]
        if frame.empty:
            raise FileNotFoundError(f'No v{METRIC_VERSION} {task} results for {sorted(allowed)} in {root}')
    return _write_export(task, frame, root, panel=panel)


def export_panels(task, root='output'):
    """Write one rank table per main-text panel; never average organ/disease/species together."""
    if task not in PRIMARY:
        raise ValueError(f'Unknown task {task}')
    root=project_path(root)
    frame=_load_task_frame(task, root)
    written=[]
    for name in PANEL_ORDER:
        part=frame[frame['dataset'].isin(panel_datasets(name))]
        if part.empty:
            continue
        written.append(name)
        _write_export(task, part, root, panel=name)
    if not written:
        raise FileNotFoundError(f'No v{METRIC_VERSION} {task} results in any main-text panel')
    return written
