"""Non-foundation-model controls.

These are not a courtesy. The claim a single-cell foundation model makes is
that pretraining buys something over classical pipelines, and independent
benchmarks (Kedzierska et al. 2023; Boiarsky et al. 2023; Ahlmann-Eltze et
al. 2024) repeatedly found that on cell-type labelling and integration the
margin over PCA, HVG selection and logistic regression is small, absent, or
negative. A result reported without these numbers next to it is not
interpretable, so the evaluation runner computes them unconditionally.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from ..core.base import BaseBaseline
from ..core.registry import register_baseline

log = logging.getLogger(__name__)

__all__ = ["PCABaseline", "HVGBaseline", "RandomProjectionBaseline", "ScaledPCABaseline"]


@register_baseline("pca")
class PCABaseline(BaseBaseline):
    """Truncated SVD on log-normalized expression. The standard workhorse."""

    def __init__(self, n_components: int = 50, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.model: Optional[TruncatedSVD] = None

    def fit(self, X, obs=None) -> "PCABaseline":
        k = int(min(self.n_components, min(np.shape(X)) - 1))
        self.model = TruncatedSVD(n_components=k, random_state=self.seed).fit(X)
        return self

    def embed(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit first")
        return self.model.transform(X)


@register_baseline("scaled_pca")
class ScaledPCABaseline(BaseBaseline):
    """Z-score genes, then PCA -- the Seurat/scanpy default pipeline.

    Scaling is fitted on train only; applying a scaler fitted on the full
    dataset is the most common leak in published single-cell comparisons.
    """

    def __init__(self, n_components: int = 50, clip: float = 10.0, seed: int = 0):
        self.n_components = n_components
        self.clip = clip
        self.seed = seed
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

    def fit(self, X, obs=None) -> "ScaledPCABaseline":
        X = np.asarray(X)
        self.scaler = StandardScaler().fit(X)
        Xs = np.clip(self.scaler.transform(X), -self.clip, self.clip)
        k = int(min(self.n_components, min(Xs.shape) - 1))
        self.pca = PCA(n_components=k, random_state=self.seed).fit(Xs)
        return self

    def embed(self, X) -> np.ndarray:
        Xs = np.clip(self.scaler.transform(np.asarray(X)), -self.clip, self.clip)
        return self.pca.transform(Xs)


@register_baseline("hvg")
class HVGBaseline(BaseBaseline):
    """No learning at all: the log-normalized values of the top-variance genes.

    The floor any representation must clear. It frequently does not.
    """

    def __init__(self, n_genes: int = 50):
        self.n_genes = n_genes
        self.idx: Optional[np.ndarray] = None

    def fit(self, X, obs=None) -> "HVGBaseline":
        X = np.asarray(X)
        var = X.var(axis=0)
        self.idx = np.argsort(-var)[: int(min(self.n_genes, X.shape[1]))]
        return self

    def embed(self, X) -> np.ndarray:
        return np.asarray(X)[:, self.idx]


@register_baseline("random_projection")
class RandomProjectionBaseline(BaseBaseline):
    """Gaussian random projection -- the null model.

    Any metric on which this scores well is a metric that is not measuring
    representation quality. Useful for catching metrics that reward geometry
    rather than biology.
    """

    def __init__(self, n_components: int = 50, seed: int = 0):
        self.n_components = n_components
        self.seed = seed
        self.model: Optional[GaussianRandomProjection] = None

    def fit(self, X, obs=None) -> "RandomProjectionBaseline":
        k = int(min(self.n_components, np.shape(X)[1]))
        self.model = GaussianRandomProjection(
            n_components=k, random_state=self.seed
        ).fit(X)
        return self

    def embed(self, X) -> np.ndarray:
        return self.model.transform(X)
