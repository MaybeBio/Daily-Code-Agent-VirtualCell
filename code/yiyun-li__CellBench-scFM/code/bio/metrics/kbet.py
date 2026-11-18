"""Büttner et al. 2019 kBET in Python; same scaled score as scIB (1 - rejection rate)."""
import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors


def _rejection_rate(x, batch, k0, alpha=0.05):
    batch = np.asarray(batch)
    cats, counts = np.unique(batch, return_counts=True)
    if len(cats) < 2 or len(x) <= 2:
        return np.nan
    k0 = int(max(2, min(k0, len(x) - 1)))
    freq = counts / counts.sum()
    expected = k0 * freq
    if np.any(expected <= 0):
        return np.nan
    codes = {c: i for i, c in enumerate(cats)}
    b = np.array([codes[v] for v in batch], dtype=int)
    idx = NearestNeighbors(n_neighbors=k0 + 1).fit(x).kneighbors(x, return_distance=False)[:, 1:]
    dof = len(cats) - 1
    rejects = 0
    for row in idx:
        obs = np.bincount(b[row], minlength=len(cats)).astype(float)
        stat = np.sum((obs - expected) ** 2 / expected)
        if chi2.sf(stat, dof) < alpha:
            rejects += 1
    return rejects / len(idx)


def kbet(embedding, batch, labels, scaled=True):
    """Mean per-label kBET. Labels with <10 cells or a single batch are skipped.

    k0 follows scIB: min(70, max(10, floor(mean batch size / 4))).
    Neighbors are computed on the embedding (correction output type).
    """
    x = np.asarray(embedding, dtype=float)
    batch = np.asarray(batch)
    labels = np.asarray(labels).astype(str)
    scores = []
    for label in np.unique(labels):
        mask = labels == label
        sub_b = batch[mask]
        if mask.sum() < 10 or np.unique(sub_b).size < 2:
            continue
        quarter = int(np.floor(np.mean(np.unique(sub_b, return_counts=True)[1]) / 4))
        k0 = int(min(70, max(10, quarter)))
        score = _rejection_rate(x[mask], sub_b, k0)
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        return np.nan
    observed = float(np.mean(scores))
    return 1.0 - observed if scaled else observed
