"""perturbation metrics; no model training or result persistence."""
import json
import numpy as np
from scipy.stats import mannwhitneyu
from ._common import matrices, correlations


def _bh_qvalues(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _celltype(group):
    try:
        parsed = json.loads(group)
        if isinstance(parsed, list) and len(parsed) >= 2:
            return str(parsed[1])
    except (TypeError, json.JSONDecodeError):
        pass
    return str(group)


def _significant_by_fdr(pert, ctrl, fdr):
    if pert.shape[0] < 3 or ctrl.shape[0] < 3 or pert.shape[1] != ctrl.shape[1]:
        return np.array([], dtype=int)
    pvals = _mannwhitney_pvalues(pert, ctrl)
    lfc = pert.mean(0) - ctrl.mean(0)
    sig = np.flatnonzero(_bh_qvalues(pvals) < fdr)
    if not len(sig):
        return np.array([], dtype=int)
    return sig[np.argsort(-np.abs(lfc[sig]), kind='stable')]


def _mannwhitney_pvalues(pert, ctrl, chunk_size=256):
    """Vectorize the original per-gene SciPy auto rule with bounded memory.

    SciPy chooses exact for min(n1,n2)<=8 without ties; otherwise asymptotic.
    Its vectorized auto rule pools tie detection across columns, so classify
    columns explicitly to preserve the original independent per-gene results.
    """
    pvals = np.empty(pert.shape[1])
    for start in range(0, pert.shape[1], chunk_size):
        end = min(start + chunk_size, pert.shape[1])
        a, b = pert[:, start:end], ctrl[:, start:end]
        exact = np.zeros(end - start, dtype=bool)
        if min(len(pert), len(ctrl)) <= 8:
            ordered = np.sort(np.concatenate((a, b), axis=0), axis=0)
            exact = ~np.any(ordered[1:] == ordered[:-1], axis=0)
        for columns, method in ((exact, 'exact'), (~exact, 'asymptotic')):
            if columns.any():
                pvals[start:end][columns] = mannwhitneyu(
                    a[:, columns], b[:, columns], axis=0,
                    alternative='two-sided', method=method).pvalue
    return pvals


def _overlap_at_n(real_idx, pred_idx):
    if len(real_idx) == 0:
        return None
    n = len(real_idx)
    pred = pred_idx[:n]
    return float(len(set(real_idx.tolist()) & set(pred.tolist())) / n)


def perturbation(pred, truth, control, groups, top_k=50, min_effect=0.0,
                 control_cells=None, control_labels=None, fdr=0.05):
    """Equal-weight condition × cell-type means in the supplied expression space.

    Delta is a difference in that space, NOT an inferred log fold change.
    DES is VCC overlap_at_N: BH-FDR significant genes ranked by |mean difference|,
    using cell-level Wilcoxon tests against same-cell-type controls.
    """
    p, t = matrices(pred, truth)
    c, _ = matrices(control, truth)
    g = np.asarray(groups).astype(str)
    if g.shape != (len(p),):
        raise ValueError('Require one condition × cell-type group per predicted cell')
    names = np.unique(g)
    aggregate = lambda x: np.stack([x[g == key].mean(0) for key in names])
    pm, tm, cm = map(aggregate, (p, t, c))
    dp, dt = pm - cm, tm - cm
    result = dict(MSE_expr=float(np.mean((pm - tm) ** 2)), MAE_expr=float(np.mean(np.abs(pm - tm))),
                  MSE_delta=float(np.mean((dp - dt) ** 2)), MAE_delta=float(np.mean(np.abs(dp - dt))),
                  control_baseline_MSE=float(np.mean(dt ** 2)),
                  control_baseline_MAE=float(np.mean(np.abs(dt))), n_groups=len(names),
                  aggregation='equal_weight_condition_celltype',
                  delta_definition='difference_in_evaluation_expression_space')
    baseline = result['control_baseline_MSE']
    result['MSE_skill_vs_control'] = 1 - result['MSE_expr'] / baseline if baseline > 0 else None
    for method in ['pearson', 'spearman']:
        result[f'{method.title()}_delta'], result[f'{method.title()}_delta_valid_n'] = correlations(dp, dt, method)
    if min_effect < 0 or int(top_k) < 1:
        raise ValueError('Require nonnegative min_effect and positive perturbation_top_k')
    sign_scores, overlaps, des_scores, per_group = [], [], [], {}
    k = min(int(top_k), pm.shape[1])
    ctrl_cells = None if control_cells is None else np.asarray(control_cells, dtype=float)
    ctrl_lab = None if control_labels is None else np.asarray(control_labels).astype(str)
    for key, a, b, pr, tr in zip(names, dp, dt, pm, tm):
        mask = np.abs(b) > min_effect
        sign = float(np.mean(np.sign(a[mask]) == np.sign(b[mask]))) if mask.any() else None
        if sign is not None:
            sign_scores.append(sign)
        directional = []
        for sign_dir in [1, -1]:
            true_idx = np.flatnonzero(sign_dir * b > min_effect)
            pred_idx = np.flatnonzero(sign_dir * a > min_effect)
            true_idx = true_idx[np.argsort(-sign_dir * b[true_idx], kind='stable')[:k]]
            pred_idx = pred_idx[np.argsort(-sign_dir * a[pred_idx], kind='stable')[:k]]
            if len(true_idx):
                directional.append(len(set(true_idx) & set(pred_idx)) / len(true_idx))
        overlap = float(np.mean(directional)) if directional else None
        if overlap is not None:
            overlaps.append(overlap)
        des = None
        if ctrl_cells is not None and ctrl_lab is not None:
            ctype = _celltype(key)
            ctrl = ctrl_cells[ctrl_lab == ctype]
            pert_t = t[g == key]
            pert_p = p[g == key]
            real_idx = _significant_by_fdr(pert_t, ctrl, fdr)
            pred_idx = _significant_by_fdr(pert_p, ctrl, fdr)
            des = _overlap_at_n(real_idx, pred_idx)
            if des is not None:
                des_scores.append(des)
        per_group[key] = dict(MAE_delta=float(np.mean(np.abs(a - b))), MAE_expr=float(np.mean(np.abs(pr - tr))),
                              MSE_expr=float(np.mean((pr - tr) ** 2)),
                              sign_agreement=sign, top_k_directional_recall=overlap, DES=des,
                              changed_genes=int(mask.sum()))
    result.update(Sign_agree_changed=float(np.mean(sign_scores)) if sign_scores else None,
                  TopK_directional_recall=float(np.mean(overlaps)) if overlaps else None,
                  DES=float(np.mean(des_scores)) if des_scores else None,
                  DES_valid_groups=len(des_scores),
                  DES_definition='vcc_overlap_at_N_fdr_bh',
                  fdr=float(fdr),
                  top_k=k, min_effect=min_effect, sign_valid_groups=len(sign_scores), per_group=per_group)
    return result
