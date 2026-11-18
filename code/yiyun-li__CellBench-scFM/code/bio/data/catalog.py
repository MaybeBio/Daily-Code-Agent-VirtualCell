"""Curated benchmark datasets; raw downloads are separate from runnable matrices."""
import json

from bio.io.paths import project_path

CURATED = {
    'bmmc': {'filename': 'bmmc.h5ad', 'batch': True},
    'hlca': {'filename': 'hlca.h5ad', 'batch': True},
    'heart': {'filename': 'heart.h5ad', 'batch': True},
    'kidney': {'filename': 'kidney.h5ad', 'batch': True},
    'replogleK562': {'filename': 'replogleK562.h5ad', 'perturbation': True},
    'replogleRPE1': {'filename': 'replogleRPE1.h5ad', 'perturbation': True},
    'immune': {'filename': 'immune.h5ad', 'batch': True},
    'ms': {'filename': 'filtered_ms_adata.h5ad', 'batch': True},
    'uc_epi': {'filename': 'ucepi.h5ad', 'batch': True},
    'uc_imm': {'filename': 'ucimm.h5ad', 'batch': True},
    'adamson': {'filename': 'adamson.h5ad', 'perturbation': True},
    'norman': {'filename': 'norman.h5ad', 'perturbation': True},
    'covid_lung': {'filename': 'covid_lung.h5ad', 'batch': True},
    'tnbc': {'filename': 'tnbc.h5ad', 'batch': True},
    'mouse_lung': {'filename': 'mouse_lung.h5ad', 'batch': True},
    'mouse_kidney': {'filename': 'mouse_kidney.h5ad', 'batch': True},
    'mouse_marrow': {'filename': 'mouse_marrow.h5ad', 'batch': True},
    'mouse_heart': {'filename': 'mouse_heart.h5ad', 'batch': True},
}

# Each dataset is its own study vote. uc_imm stays in CURATED but not in the disease panel.
STUDY_ID = {}

PANEL_ORDER = ('organ', 'disease', 'species', 'perturbation')


def study_id(dataset):
    return STUDY_ID.get(str(dataset), str(dataset))


def load_panels():
    path = project_path('resources/experiment_panels.json')
    return json.loads(path.read_text())


def panel_datasets(name):
    panels = load_panels()
    if name not in panels:
        raise ValueError(f'Unknown panel {name!r}; choose from {", ".join(PANEL_ORDER)}')
    names = list(panels[name]['datasets'])
    missing = [d for d in names if d not in CURATED]
    if missing:
        raise ValueError(f'Panel {name} lists datasets not in CURATED: {missing}')
    return names


def suite_datasets():
    """Union of the four main-text panels, excluding leftover extras such as immune."""
    seen = []
    panels = load_panels()
    for name in PANEL_ORDER:
        for dataset in panels[name]['datasets']:
            if dataset not in seen:
                seen.append(dataset)
    return tuple(seen)


def read_curated(name, data_cfg):
    from bio.data.datasets import _read_flat_dataset
    result = _read_flat_dataset(data_cfg, CURATED[name]['filename'],
                                batch_fields=('batch',), subset_hvg=False)
    if result is None:
        raise FileNotFoundError(f"Curated dataset {name} missing: run dataset download/preparation first")
    n = int(data_cfg.get('max_cells') or 0)
    if n > 0:
        from bio.data.subset import subset_adata
        result = subset_adata(result, n, seed=int(data_cfg.get('subset_seed', 41)))
    return result
