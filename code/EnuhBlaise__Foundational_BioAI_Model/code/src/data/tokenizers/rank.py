"""Rank-value tokenization (Geneformer-style).

Genes are ordered by expression normalized against a corpus-wide median, and
only the ordering is kept.  This discards magnitude entirely, which buys
strong robustness to depth and protocol at the cost of being unable to
express "twice as much".

The corpus median must be fit on the **training split only**; fitting it on
the full dataset leaks test-set expression statistics into the tokenizer.
``fit`` enforces this by refusing to run twice.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp

from ...core.base import BaseTokenizer
from ...core.registry import register_tokenizer


@register_tokenizer("rank")
class RankTokenizer(BaseTokenizer):
    produces_bins = False

    def __init__(self, max_genes: int = 1024, normalize_ranks: bool = True,
                 seed: int = 0):
        self.max_genes = int(max_genes)
        self.normalize_ranks = bool(normalize_ranks)
        self.gene_medians: Optional[np.ndarray] = None
        self._fitted = False
        self._rng = np.random.default_rng(seed)

    @property
    def n_value_bins(self) -> int:
        return 1

    def fit(self, X_train) -> "RankTokenizer":
        """Compute nonzero median expression per gene on the training split."""
        if self._fitted:
            raise RuntimeError(
                "RankTokenizer is already fitted. Re-fitting on a different "
                "split would leak statistics across the split boundary; "
                "construct a new tokenizer instead."
            )
        if sp.issparse(X_train):
            X = X_train.tocsc()
            med = np.ones(X.shape[1], dtype=np.float64)
            for j in range(X.shape[1]):
                d = X.data[X.indptr[j] : X.indptr[j + 1]]
                if d.size:
                    med[j] = float(np.median(d))
        else:
            X = np.asarray(X_train)
            med = np.ones(X.shape[1], dtype=np.float64)
            for j in range(X.shape[1]):
                d = X[:, j]
                d = d[d > 0]
                if d.size:
                    med[j] = float(np.median(d))
        med[med <= 0] = 1.0
        self.gene_medians = med
        self._fitted = True
        return self

    def __call__(self, values: np.ndarray, gene_ids: np.ndarray) -> Dict[str, Any]:
        values = np.asarray(values, dtype=np.float32).ravel()
        gene_ids = np.asarray(gene_ids, dtype=np.int64).ravel()
        if self.gene_medians is None:
            scaled = values
        else:
            if self.gene_medians.size != values.size:
                raise ValueError(
                    f"tokenizer was fitted on {self.gene_medians.size} genes "
                    f"but received {values.size}; gene spaces must match."
                )
            scaled = values / self.gene_medians.astype(np.float32)
        keep = np.flatnonzero(values > 0)
        if keep.size == 0:
            return {"gene_ids": gene_ids[:0], "values": values[:0], "value_bins": None}
        order = keep[np.argsort(-scaled[keep], kind="stable")][: self.max_genes]
        n = order.size
        if self.normalize_ranks:
            vals = (1.0 - np.arange(n, dtype=np.float32) / max(n, 1)).astype(np.float32)
        else:
            vals = np.arange(n, dtype=np.float32)
        return {"gene_ids": gene_ids[order], "values": vals, "value_bins": None}
