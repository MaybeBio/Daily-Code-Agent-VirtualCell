"""Train-split HVG for statistical baselines; foundation models keep full genes."""
import numpy as np


def train_cell_mask(adata, seed=41, split_by='donor', test_size=None,
                    perturbation=False, perturbation_split='auto', task='annotation'):
    """Fit exactly the task training rows, plus observed controls for perturbation."""
    from bio.data.splits import evaluation_indices
    task = 'perturbation' if perturbation else task
    train, _, _, info = evaluation_indices(
        adata, task=task, seed=seed, split_by=split_by,
        perturbation_split=perturbation_split, test_size=test_size)
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[train] = True
    if perturbation:
        mask[np.asarray(adata.obs['control']) == 1] = True
    return mask, dict(info, n_fit_cells=int(mask.sum()))


def hvg_indices(adata, n_top=3000):
    """Select genes on a private log-expression copy; preserve evaluation counts."""
    import scanpy as sc
    from scipy import sparse
    work = adata.copy()
    values = work.X.data if sparse.issparse(work.X) else np.asarray(work.X)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('HVG input must contain finite nonnegative expression')
    if np.ravel(work.X.sum(1)).max() == 0:
        raise ValueError('Train split has no counts for HVG selection')
    # Integer counts can have maxima below 10; magnitude alone is insufficient.
    counts = np.allclose(values, np.rint(values), atol=1e-6, rtol=0)
    if counts or (values.size and values.max() > 10):
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
    sc.pp.highly_variable_genes(work, n_top_genes=min(int(n_top), work.n_vars))
    return np.flatnonzero(work.var['highly_variable'].to_numpy())


def apply_train_hvg(adata, n_top=3000, seed=41, split_by='donor', test_size=None,
                    perturbation=False, perturbation_split='auto', task='annotation'):
    """Subset genes by HVG computed on train cells only.

    Foundation-model loaders must not call this; they keep the full count matrix.
    """
    mask, info = train_cell_mask(adata, seed=seed, split_by=split_by, test_size=test_size,
                                 perturbation=perturbation, perturbation_split=perturbation_split,
                                 task=task)
    train = np.flatnonzero(mask)
    selected = hvg_indices(adata[train], n_top=n_top)
    subset = adata[:, selected].copy()
    report = dict(hvg_source='train_split', hvg_split_mode=info.get('mode'),
                  n_hvg=int(subset.n_vars), n_train_cells_for_hvg=int(len(train)))
    return subset, report
