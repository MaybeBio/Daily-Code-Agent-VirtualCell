"""Shared helpers for the exploration notebooks.

Only plumbing lives here -- chart styling, data loading and a couple of
measurement shortcuts.  The methods under study are always called explicitly
in the notebooks themselves, so nothing interesting is hidden in this file.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap

# --- palette: fixed categorical order, one sequential hue -----------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW = SERIES[:4]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
NEUTRAL = "#9a9992"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eaf2fc", "#2a78d6", "#0d3568"])


def set_style() -> None:
    """Thin marks, recessive axes, no chart junk."""
    logging.getLogger().setLevel(logging.WARNING)
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 110,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 9.5, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "text.color": INK, "font.size": 10,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def load_data(path=None, obs_map=None, raw_layer=None, **synthetic_kwargs):
    """Return an AnnData of raw counts.

    ``path=None``  -> the synthetic simulator (``src/data/synthetic``).
    ``path="x.h5ad"`` -> a real dataset through the repo's own H5ADSource,
    so the same raw-count and metadata guards apply as in a training run.
    """
    if path is None:
        from src.data.synthetic import make_synthetic
        return make_synthetic(**synthetic_kwargs)
    from src.data.sources.h5ad_source import H5ADSource
    return H5ADSource(path=path, obs_map=obs_map, raw_layer=raw_layer).load()


def describe(X, label="") -> dict:
    """The handful of numbers that characterize a count matrix."""
    X = sp.csr_matrix(X)
    lib = np.asarray(X.sum(1)).ravel()
    return {
        "stage": label,
        "cells": X.shape[0],
        "genes": X.shape[1],
        "sparsity_%": round(100 * (1 - X.nnz / max(X.shape[0] * X.shape[1], 1)), 1),
        "median_depth": round(float(np.median(lib)), 1),
        "median_genes_per_cell": int(np.median(np.diff(X.indptr))),
    }


def pca(X_fit, *X_apply, k=30, seed=0):
    """Fit truncated SVD on the first matrix, apply to all."""
    from sklearn.decomposition import TruncatedSVD
    k = int(min(k, min(X_fit.shape) - 1))
    m = TruncatedSVD(n_components=k, random_state=seed).fit(X_fit)
    out = [m.transform(X_fit)] + [m.transform(x) for x in X_apply]
    return out if len(out) > 1 else out[0]


def codes(values) -> np.ndarray:
    return np.unique(np.asarray(values).astype(str), return_inverse=True)[1]


def probe_f1(Ztr, ytr, Zte, yte, seed=0) -> float:
    """Macro-F1 of a logistic-regression probe (the repo's own protocol)."""
    from src.eval.probes import linear_probe
    return linear_probe(Ztr, ytr, Zte, yte, seed=seed, n_boot=0)["macro_f1"]


def r2_by_group(Z, groups) -> np.ndarray:
    """Per-column fraction of variance explained by a categorical covariate."""
    Z = np.asarray(Z, dtype=float)
    g = codes(groups)
    tot = ((Z - Z.mean(0)) ** 2).sum(0)
    between = np.zeros(Z.shape[1])
    for c in np.unique(g):
        sel = g == c
        between += sel.sum() * (Z[sel].mean(0) - Z.mean(0)) ** 2
    return np.divide(between, tot, out=np.zeros_like(tot), where=tot > 0)


def label_bars(ax, bars, fmt="{:.2f}", dy=0.01):
    """Value labels in ink (never the series color)."""
    top = ax.get_ylim()[1]
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + dy * top, fmt.format(h),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
