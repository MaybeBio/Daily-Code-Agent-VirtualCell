"""Frozen-embedding probes.

The protocol is deliberately rigid: fit the probe on the training split's
embeddings, select nothing on test, report test performance with a bootstrap
interval. Probe hyperparameters are chosen on validation, never on test.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

__all__ = ["linear_probe", "knn_probe", "bootstrap_ci"]


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, metric="macro_f1",
    n_boot: int = 200, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI over resampled cells.

    Note this captures sampling noise over *cells*, which understates true
    uncertainty when cells are correlated within donor. With few held-out
    donors, the honest interval is wider than this; the report says so.
    """
    fn = (
        (lambda a, b: f1_score(a, b, average="macro", zero_division=0))
        if metric == "macro_f1"
        else accuracy_score
    )
    point = float(fn(y_true, y_pred))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.unique(y_true[i]).size < 2:
            continue
        vals.append(fn(y_true[i], y_pred[i]))
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def _clean(Z_tr, y_tr, Z_te, y_te):
    keep_tr = y_tr >= 0
    keep_te = y_te >= 0
    return Z_tr[keep_tr], y_tr[keep_tr], Z_te[keep_te], y_te[keep_te]


def linear_probe(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_test: np.ndarray, y_test: np.ndarray,
    C: float = 1.0, max_iter: int = 2000, seed: int = 0,
    standardize: bool = True, n_boot: int = 200,
) -> Dict[str, float]:
    Z_train, y_train, Z_test, y_test = _clean(
        np.asarray(Z_train), np.asarray(y_train),
        np.asarray(Z_test), np.asarray(y_test),
    )
    if np.unique(y_train).size < 2 or len(y_test) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    if standardize:
        sc = StandardScaler().fit(Z_train)
        Z_train, Z_test = sc.transform(Z_train), sc.transform(Z_test)
    clf = LogisticRegression(
        C=C, max_iter=max_iter, random_state=seed, n_jobs=1
    ).fit(Z_train, y_train)
    pred = clf.predict(Z_test)
    f1, lo, hi = bootstrap_ci(y_test, pred, "macro_f1", n_boot, seed)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": f1,
        "macro_f1_lo": lo,
        "macro_f1_hi": hi,
        "n_test": int(len(y_test)),
    }


def knn_probe(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_test: np.ndarray, y_test: np.ndarray,
    k: int = 15, n_boot: int = 200, seed: int = 0,
) -> Dict[str, float]:
    """Nonparametric probe: measures local neighbourhood structure rather
    than linear separability. A representation can be good for one and poor
    for the other, so both are reported."""
    Z_train, y_train, Z_test, y_test = _clean(
        np.asarray(Z_train), np.asarray(y_train),
        np.asarray(Z_test), np.asarray(y_test),
    )
    if np.unique(y_train).size < 2 or len(y_test) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    k = int(min(k, len(Z_train)))
    clf = KNeighborsClassifier(n_neighbors=k).fit(Z_train, y_train)
    pred = clf.predict(Z_test)
    f1, lo, hi = bootstrap_ci(y_test, pred, "macro_f1", n_boot, seed)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": f1,
        "macro_f1_lo": lo,
        "macro_f1_hi": hi,
        "n_test": int(len(y_test)),
    }
