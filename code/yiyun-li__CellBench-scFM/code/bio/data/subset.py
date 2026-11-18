"""Bounded cell subset for smoke runs; keeps labels, donors, batches and conditions."""
import numpy as np


def subset_adata(adata, n, seed=41):
    n = int(n)
    if n < 1:
        raise ValueError('max_cells must be positive')
    if adata.n_obs <= n:
        return adata
    rng = np.random.default_rng(seed)
    obs = adata.obs
    selected = []

    def take(mask, count):
        idx = np.flatnonzero(mask)
        if idx.size == 0 or count <= 0:
            return
        count = min(int(count), idx.size)
        selected.extend(rng.choice(idx, size=count, replace=False).tolist())

    if 'condition' in obs:
        conditions = obs['condition'].astype(str).to_numpy()
        is_ctrl = np.isin(np.array([c.lower() for c in conditions]), ['ctrl', 'control'])
        if 'control' in obs:
            is_ctrl = is_ctrl | (np.asarray(obs['control']) == 1)
        take(is_ctrl, min(24, n // 4))
        others = [c for c in np.unique(conditions) if c.lower() not in {'ctrl', 'control'}]
        rng.shuffle(others)
        per = max(2, (n - len(set(selected))) // max(1, min(len(others), 20)))
        for name in others[:20]:
            take(conditions == name, per)
    for key, groups in (('donor_id', 2), ('batch', 2)):
        if key not in obs:
            continue
        values = obs[key].astype(str).to_numpy()
        names = np.unique(values)
        rng.shuffle(names)
        per = max(4, n // (2 * max(1, len(names[:groups]))))
        for name in names[:groups]:
            take(values == name, per)
    labels = obs['celltype'].astype(str).to_numpy()
    types = np.unique(labels)
    leftover = n - len(set(selected))
    if leftover > 0:
        per = max(1, leftover // max(1, len(types)))
        for name in types:
            take(labels == name, per)
    unique = []
    seen = set()
    for i in selected:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    if len(unique) < n:
        rest = np.setdiff1d(np.arange(adata.n_obs), np.array(unique, dtype=int), assume_unique=False)
        extra = min(n - len(unique), rest.size)
        if extra:
            unique.extend(rng.choice(rest, size=extra, replace=False).tolist())
    unique = np.array(sorted(unique[:n]), dtype=int)
    if unique.size < 8:
        unique = np.sort(rng.choice(adata.n_obs, size=min(n, adata.n_obs), replace=False))
    return adata[unique].copy()
