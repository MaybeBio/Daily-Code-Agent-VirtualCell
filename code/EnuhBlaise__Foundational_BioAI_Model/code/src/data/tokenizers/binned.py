"""Value binning (scGPT-style).

Nonzero expression within a cell is mapped to ``n_bins`` equal-frequency
bins.  Binning is computed **per cell**, which makes the representation
invariant to *rescaling* a cell (multiplying its counts by a constant), at
the cost that absolute expression level is no longer recoverable.

It is NOT invariant to real loss of sequencing depth: thinning changes which
genes are detected and how many are tied, and bins are quantiles of the
detected set.  ``Notebooks/04_tokenization.ipynb`` measures both effects.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...core.base import BaseTokenizer
from ...core.registry import register_tokenizer


@register_tokenizer("binned")
class BinnedValueTokenizer(BaseTokenizer):
    produces_bins = True

    def __init__(self, n_bins: int = 51, max_genes: int = 1024,
                 nonzero_only: bool = True, seed: int = 0):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = int(n_bins)
        self.max_genes = int(max_genes)
        self.nonzero_only = bool(nonzero_only)
        self._rng = np.random.default_rng(seed)

    @property
    def n_value_bins(self) -> int:
        # bin 0 is reserved for "zero / padding"
        return self.n_bins + 1

    def __call__(self, values: np.ndarray, gene_ids: np.ndarray) -> Dict[str, Any]:
        values = np.asarray(values, dtype=np.float32).ravel()
        gene_ids = np.asarray(gene_ids, dtype=np.int64).ravel()
        keep = np.flatnonzero(values > 0) if self.nonzero_only else np.arange(values.size)
        if keep.size == 0:
            return {
                "gene_ids": gene_ids[:0],
                "values": values[:0],
                "value_bins": np.zeros(0, dtype=np.int64),
            }
        if keep.size > self.max_genes:
            order = np.argsort(-values[keep])[: self.max_genes]
            keep = keep[np.sort(order)]
        v = values[keep]
        # Equal-frequency bins within the cell.  Ties must share a bin: UMI
        # data is dominated by ties (most detected genes have count 1), and
        # splitting equal values across bins by gene order would inject noise
        # the model then has to "predict".  Average ranks give every tied
        # value the same quantile, so occupied bins are uneven by design.
        from scipy.stats import rankdata

        q = rankdata(v, method="average") / v.size      # quantile in (0, 1]
        bins = np.ceil(q * self.n_bins).astype(np.int64)
        bins = np.clip(bins, 1, self.n_bins)            # 0 stays reserved
        return {
            "gene_ids": gene_ids[keep],
            "values": bins.astype(np.float32),
            "value_bins": bins,
        }
