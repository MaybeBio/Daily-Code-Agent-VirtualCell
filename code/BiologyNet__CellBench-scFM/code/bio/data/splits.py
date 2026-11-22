"""Train/test assignment: hold out whole donors when at least two groups exist."""
import numpy as np
from sklearn.model_selection import train_test_split


def _cell_split(n, labels, seed, test_size, val_size):
    idx = np.arange(n)
    labels = np.asarray(labels)
    # Integer counts avoid ceil(100 * (0.2 + 0.1)) == 31 from floating point.
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size))) if val_size > 0 else 0
    extra = n_test + n_val
    if extra >= n:
        raise ValueError('Not enough cells for nonempty train/validation/test splits')
    strat = labels if _can_stratify(labels, extra / n) else None
    train, temp = train_test_split(idx, test_size=extra, random_state=seed, stratify=strat)
    if val_size <= 0:
        return train, np.array([], dtype=int), temp, dict(mode='cell', n_train=len(train), n_test=len(temp), n_val=0)
    rel = n_test / extra
    strat_t = labels[temp] if _can_stratify(labels[temp], rel) else None
    val, test = train_test_split(temp, test_size=n_test, random_state=seed, stratify=strat_t)
    return train, val, test, dict(mode='cell', n_train=len(train), n_test=len(test), n_val=len(val))


def _can_stratify(labels, test_size):
    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 2:
        return False
    n_test = max(1, int(round(len(labels) * test_size)))
    return counts.min() >= 2 and n_test >= len(values) and (len(labels) - n_test) >= len(values)


def split_indices(n, labels, groups=None, seed=41, test_size=0.2, val_size=0.1):
    """Return train, val, test index arrays and a small report.

    When ``groups`` has at least two unique values (typically donor_id), whole
    groups go to train, val, or test. Test cells whose labels are absent from
    train are dropped. One or zero groups fall back to a cell-level split.
    """
    labels = np.asarray(labels)
    if n != len(labels):
        raise ValueError('labels length must match n')
    if groups is None:
        return _cell_split(n, labels, seed, test_size, val_size)
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if unique.size < 2:
        return _cell_split(n, labels, seed, test_size, val_size)
    rng = np.random.default_rng(seed)
    order = unique.copy()
    rng.shuffle(order)
    n_test = min(max(1, int(round(unique.size * test_size))), unique.size - 1)
    test_groups = set(order[:n_test])
    rest = order[n_test:]
    val_groups = set()
    if val_size > 0 and rest.size >= 2:
        n_val = min(max(1, int(round(unique.size * val_size))), rest.size - 1)
        val_groups = set(rest[:n_val])
        train_groups = set(rest[n_val:])
    else:
        train_groups = set(rest)
    train = np.flatnonzero(np.isin(groups, list(train_groups)))
    val = np.flatnonzero(np.isin(groups, list(val_groups))) if val_groups else np.array([], dtype=int)
    test = np.flatnonzero(np.isin(groups, list(test_groups)))
    if len(train) == 0 or len(test) == 0:
        raise ValueError('Donor holdout produced an empty train or test set')
    seen = np.unique(labels[train])
    test = test[np.isin(labels[test], seen)]
    if val.size:
        val = val[np.isin(labels[val], seen)]
    if len(test) == 0:
        raise ValueError('Donor holdout has no test cells with labels seen in training')
    info = dict(mode='donor', n_train=len(train), n_val=len(val), n_test=len(test),
                n_train_donors=len(train_groups), n_val_donors=len(val_groups),
                n_test_donors=len(test_groups))
    return train, val, test, info


def evaluation_indices(adata, task='annotation', seed=41, split_by='donor',
                       perturbation_split='auto', test_size=None):
    """Single source of split indices for encoder fitting and task heads.

    Returned positions refer to adata rows. Controls are not task targets; the
    statistical encoder may fit them in addition to training perturbations.
    """
    if task == 'perturbation':
        from bio.data.pert_genes import is_valid_perturbation
        conditions = adata.obs['condition'].astype(str).to_numpy()
        controls = np.asarray(adata.obs['control']) == 1
        valid = np.array([is_valid_perturbation(c) for c in conditions])
        positions = np.flatnonzero(~controls & valid)
        train, val, test, info = split_perturbations(
            conditions[positions], seed=seed,
            test_size=0.2 if test_size is None else test_size, val_size=0.1,
            mode=perturbation_split)
    else:
        positions = np.arange(adata.n_obs)
        groups = (adata.obs['donor_id'].astype(str).to_numpy()
                  if split_by == 'donor' and 'donor_id' in adata.obs else None)
        train, val, test, info = split_indices(
            adata.n_obs, adata.obs['celltype'].astype(str).to_numpy(), groups=groups,
            seed=seed, test_size=0.2 if test_size is None else test_size,
            val_size=0.1)
    if len(val) == 0:
        train, val = train_test_split(train, test_size=0.1, random_state=seed)
        info = dict(info, val_from='seen_train_cells', n_train=len(train), n_val=len(val))
    return positions[train], positions[val], positions[test], info


def split_conditions(conditions, seed=41, test_size=0.2, val_size=0.1):
    """Hold out whole perturbation conditions so train and test share no knockout."""
    conditions = np.asarray(conditions).astype(str)
    unique = np.unique(conditions)
    if unique.size < 2:
        raise ValueError('Need at least two perturbation conditions for unseen-condition holdout')
    rng = np.random.default_rng(seed)
    order = unique.copy()
    rng.shuffle(order)
    n_test = min(max(1, int(round(unique.size * test_size))), unique.size - 1)
    test_set = set(order[:n_test])
    rest = order[n_test:]
    val_set = set()
    if val_size > 0 and rest.size >= 2:
        n_val = min(max(1, int(round(unique.size * val_size))), rest.size - 1)
        val_set = set(rest[:n_val])
        train_set = set(rest[n_val:])
    else:
        train_set = set(rest)
    train = np.flatnonzero(np.isin(conditions, list(train_set)))
    val = np.flatnonzero(np.isin(conditions, list(val_set))) if val_set else np.array([], dtype=int)
    test = np.flatnonzero(np.isin(conditions, list(test_set)))
    if len(train) == 0 or len(test) == 0:
        raise ValueError('Unseen-condition split produced an empty train or test set')
    info = dict(mode='unseen_condition', n_train=len(train), n_val=len(val), n_test=len(test),
                n_train_conditions=len(train_set), n_val_conditions=len(val_set),
                n_test_conditions=len(test_set),
                train_conditions=sorted(train_set), val_conditions=sorted(val_set),
                test_conditions=sorted(test_set))
    return train, val, test, info


def _split_combos(conditions, singles, combos, seed, val_size):
    if not singles or not combos:
        raise ValueError('Combo holdout needs both single-gene and combinatorial conditions')
    rng = np.random.default_rng(seed)
    order = np.array(list(combos))
    rng.shuffle(order)
    val_set = set()
    if val_size > 0 and order.size >= 3:
        n_val = min(max(1, int(round(order.size * val_size))), order.size - 1)
        val_set = set(order[:n_val])
        test_set = set(order[n_val:])
    else:
        test_set = set(order)
    train_set = set(singles)
    train = np.flatnonzero(np.isin(conditions, list(train_set)))
    val = np.flatnonzero(np.isin(conditions, list(val_set))) if val_set else np.array([], dtype=int)
    test = np.flatnonzero(np.isin(conditions, list(test_set)))
    if len(train) == 0 or len(test) == 0:
        raise ValueError('Combo holdout produced an empty train or test set')
    info = dict(mode='unseen_combo', n_train=len(train), n_val=len(val), n_test=len(test),
                n_train_conditions=len(train_set), n_val_conditions=len(val_set),
                n_test_conditions=len(test_set),
                train_conditions=sorted(train_set), val_conditions=sorted(val_set),
                test_conditions=sorted(test_set))
    return train, val, test, info


def split_perturbations(conditions, seed=41, test_size=0.2, val_size=0.1, mode='auto'):
    """Unseen-condition holdout, or combo holdout when singles and doubles both exist.

    ``mode='auto'`` uses combo holdout when there are at least 10 single-gene and
    10 combinatorial conditions (Norman). Otherwise whole knockout identities
    are held out. Combinatorial names use ``+`` or ``_`` (Norman ``AHR_FEV``).
    """
    from bio.data.pert_genes import parse_perturbation_genes
    conditions = np.asarray(conditions).astype(str)
    unique = np.unique(conditions)
    parsed = {name: parse_perturbation_genes(name) for name in unique}
    singles = [name for name, genes in parsed.items() if len(genes) <= 1]
    combos = [name for name, genes in parsed.items() if len(genes) >= 2]
    resolved = mode
    if mode == 'auto':
        resolved = 'combo' if len(singles) >= 10 and len(combos) >= 10 else 'unseen_condition'
    if resolved in ('combo', 'unseen_combo'):
        return _split_combos(conditions, singles, combos, seed, val_size)
    if resolved in ('unseen_condition', 'condition'):
        return split_conditions(conditions, seed=seed, test_size=test_size, val_size=val_size)
    raise ValueError(f'Unknown perturbation_split {mode!r}')
