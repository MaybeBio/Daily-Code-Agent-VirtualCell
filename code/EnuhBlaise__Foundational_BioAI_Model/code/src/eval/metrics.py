"""Embedding quality metrics.

Two families that must always be reported together:

**Biological conservation** -- does the embedding separate cell types?
**Batch correction** -- has technical structure been removed?

Reporting either alone is uninformative, because each is trivially
maximizable at the other's expense: random noise scores perfectly on batch
mixing, and an embedding that encodes the raw batch identity scores well on
clustering whenever composition differs across batches.

Implemented directly (rather than importing scib) to keep the dependency
surface small and to make every definition inspectable.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)

__all__ = [
    "bio_conservation",
    "batch_correction",
    "evaluate_embedding",
    "library_size_leakage",
    "knn_mixing",
    "graph_connectivity",
    "kbet_acceptance",
    "pcr_batch",
]


def _subsample(n: int, max_n: int, seed: int = 0) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))


def _encode(labels: Sequence) -> np.ndarray:
    _, inv = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return inv


# --- biological conservation -------------------------------------------
def bio_conservation(
    Z: np.ndarray, labels: Sequence, seed: int = 0, max_cells: int = 20_000
) -> Dict[str, float]:
    """ARI/NMI from k-means at the true k, plus cell-type silhouette."""
    idx = _subsample(len(Z), max_cells, seed)
    Z, y = np.asarray(Z)[idx], _encode(labels)[idx]
    k = int(np.unique(y).size)
    out: Dict[str, float] = {}
    if k < 2:
        return {"ari": float("nan"), "nmi": float("nan"), "celltype_asw": float("nan")}
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    out["ari"] = float(adjusted_rand_score(y, km.labels_))
    out["nmi"] = float(normalized_mutual_info_score(y, km.labels_))
    # silhouette rescaled from [-1,1] to [0,1]; higher = tighter cell types
    out["celltype_asw"] = float((silhouette_score(Z, y) + 1) / 2)
    return out


# --- batch correction ---------------------------------------------------
def knn_mixing(
    Z: np.ndarray, batch: Sequence, k: int = 30, seed: int = 0,
    max_cells: int = 20_000,
) -> float:
    """iLISI-style: effective number of batches among each cell's neighbours,
    normalized to [0, 1]. 1 = neighbourhoods match the global batch mix."""
    idx = _subsample(len(Z), max_cells, seed)
    Z, b = np.asarray(Z)[idx], _encode(batch)[idx]
    n_b = int(np.unique(b).size)
    if n_b < 2:
        return float("nan")
    k = int(min(k, len(Z) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, ind = nn.kneighbors(Z)
    nb = b[ind[:, 1:]]
    # inverse Simpson index per cell
    counts = np.stack([(nb == j).sum(1) for j in range(n_b)], axis=1) / k
    inv_simpson = 1.0 / np.maximum((counts ** 2).sum(1), 1e-12)
    # normalize against the index implied by the global composition
    glob = np.bincount(b, minlength=n_b) / b.size
    ideal = 1.0 / max((glob ** 2).sum(), 1e-12)
    return float(np.clip(inv_simpson.mean() / ideal, 0, 1))


def graph_connectivity(
    Z: np.ndarray, labels: Sequence, k: int = 15, seed: int = 0,
    max_cells: int = 20_000,
) -> float:
    """Fraction of each cell type lying in its own largest connected kNN
    component. Low values mean a cell type was split across batches."""
    from scipy.sparse.csgraph import connected_components

    idx = _subsample(len(Z), max_cells, seed)
    Z, y = np.asarray(Z)[idx], _encode(labels)[idx]
    k = int(min(k, len(Z) - 1))
    scores = []
    for c in np.unique(y):
        sel = np.flatnonzero(y == c)
        if sel.size < 3:
            continue
        kk = int(min(k, sel.size - 1))
        A = NearestNeighbors(n_neighbors=kk).fit(Z[sel]).kneighbors_graph(Z[sel])
        n_comp, lab = connected_components(A, directed=False)
        scores.append(np.bincount(lab).max() / sel.size)
    return float(np.mean(scores)) if scores else float("nan")


def kbet_acceptance(
    Z: np.ndarray, batch: Sequence, k: int = 30, alpha: float = 0.05,
    seed: int = 0, max_cells: int = 10_000,
) -> float:
    """kBET-style acceptance rate: fraction of neighbourhoods whose batch
    composition is not rejected against the global composition (chi-square)."""
    from scipy.stats import chi2

    idx = _subsample(len(Z), max_cells, seed)
    Z, b = np.asarray(Z)[idx], _encode(batch)[idx]
    n_b = int(np.unique(b).size)
    if n_b < 2:
        return float("nan")
    k = int(min(k, len(Z) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, ind = nn.kneighbors(Z)
    nb = b[ind[:, 1:]]
    obs = np.stack([(nb == j).sum(1) for j in range(n_b)], axis=1).astype(float)
    exp = (np.bincount(b, minlength=n_b) / b.size) * k
    exp = np.maximum(exp, 1e-8)
    stat = ((obs - exp) ** 2 / exp).sum(axis=1)
    p = 1.0 - chi2.cdf(stat, df=n_b - 1)
    return float((p >= alpha).mean())


def batch_asw(Z: np.ndarray, batch: Sequence, labels: Optional[Sequence] = None,
              seed: int = 0, max_cells: int = 20_000) -> float:
    """1 - |silhouette on batch|, computed within each cell type.

    Within-type is the important detail: computed globally, this rewards an
    embedding that merges all cell types together.
    """
    idx = _subsample(len(Z), max_cells, seed)
    Z, b = np.asarray(Z)[idx], _encode(batch)[idx]
    if np.unique(b).size < 2:
        return float("nan")
    if labels is None:
        s = silhouette_samples(Z, b)
        return float(1 - np.abs(s).mean())
    y = _encode(labels)[idx]
    vals = []
    for c in np.unique(y):
        sel = np.flatnonzero(y == c)
        if sel.size < 10 or np.unique(b[sel]).size < 2:
            continue
        vals.append(1 - np.abs(silhouette_samples(Z[sel], b[sel])).mean())
    return float(np.mean(vals)) if vals else float("nan")


def pcr_batch(Z: np.ndarray, batch: Sequence, n_comps: int = 20) -> float:
    """Principal-component regression: variance in the embedding explained by
    batch. 0 is ideal. Reported as explained variance, not 1-minus, so the
    direction is unambiguous."""
    Z = np.asarray(Z)
    n_comps = int(min(n_comps, Z.shape[1], Z.shape[0] - 1))
    if n_comps < 2:
        return float("nan")
    p = PCA(n_components=n_comps).fit(Z)
    S = p.transform(Z)
    b = _encode(batch)
    B = np.eye(int(b.max()) + 1)[b]
    r2 = []
    for i in range(n_comps):
        lr = LinearRegression().fit(B, S[:, i])
        pred = lr.predict(B)
        ss_res = ((S[:, i] - pred) ** 2).sum()
        ss_tot = ((S[:, i] - S[:, i].mean()) ** 2).sum()
        r2.append(0.0 if ss_tot <= 0 else max(0.0, 1 - ss_res / ss_tot))
    w = p.explained_variance_ratio_
    return float(np.sum(np.array(r2) * w) / max(w.sum(), 1e-12))


def library_size_leakage(Z: np.ndarray, library_size: Sequence) -> float:
    """R^2 of log library size regressed on the embedding.

    A diagnostic that standard panels omit. An embedding can score acceptably
    on clustering while mostly encoding sequencing depth; if this is high, the
    representation is substantially a depth readout.
    """
    Z = np.asarray(Z)
    y = np.log1p(np.asarray(library_size, dtype=float))
    if y.std() < 1e-8:
        return float("nan")
    lr = LinearRegression().fit(Z, y)
    pred = lr.predict(Z)
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(max(0.0, 1 - ss_res / ss_tot))


def batch_correction(
    Z: np.ndarray, batch: Sequence, labels: Optional[Sequence] = None,
    seed: int = 0,
) -> Dict[str, float]:
    return {
        "knn_mixing": knn_mixing(Z, batch, seed=seed),
        "kbet": kbet_acceptance(Z, batch, seed=seed),
        "batch_asw": batch_asw(Z, batch, labels, seed=seed),
        "graph_connectivity": graph_connectivity(Z, labels if labels is not None else batch, seed=seed),
        "pcr_batch": pcr_batch(Z, batch),
    }


def evaluate_embedding(
    Z: np.ndarray,
    labels: Sequence,
    batch: Optional[Sequence] = None,
    library_size: Optional[Sequence] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Full panel plus a single aggregate, weighted the way scIB weights it
    (60% biology / 40% batch) -- stated explicitly because any single number
    here is a choice, not a fact."""
    out = dict(bio_conservation(Z, labels, seed=seed))
    if batch is not None:
        out.update(batch_correction(Z, batch, labels, seed=seed))
    if library_size is not None:
        out["library_size_r2"] = library_size_leakage(Z, library_size)

    bio_keys = ["ari", "nmi", "celltype_asw"]
    bat_keys = ["knn_mixing", "kbet", "batch_asw", "graph_connectivity"]
    bio = [out[k] for k in bio_keys if k in out and np.isfinite(out[k])]
    bat = [out[k] for k in bat_keys if k in out and np.isfinite(out[k])]
    if bio:
        out["bio_score"] = float(np.mean(bio))
    if bat:
        out["batch_score"] = float(np.mean(bat))
    if bio and bat:
        out["overall"] = 0.6 * out["bio_score"] + 0.4 * out["batch_score"]
    elif bio:
        out["overall"] = out["bio_score"]
    return out
