"""Continuous-value tokenization -- the control condition.

Keeps log1p(CP10K) as a real number and projects it linearly.  No vocabulary,
no discretization, no information discarded.  It exists so that any claimed
benefit of rank or bin tokenization has to be demonstrated against it rather
than assumed; in several published comparisons the discretization schemes do
not clearly beat simply feeding the value.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...core.base import BaseTokenizer
from ...core.registry import register_tokenizer


@register_tokenizer("continuous")
class ContinuousTokenizer(BaseTokenizer):
    produces_bins = False

    def __init__(self, max_genes: int = 1024, nonzero_only: bool = True,
                 seed: int = 0):
        self.max_genes = int(max_genes)
        self.nonzero_only = bool(nonzero_only)
        self._rng = np.random.default_rng(seed)

    @property
    def n_value_bins(self) -> int:
        return 1

    def __call__(self, values: np.ndarray, gene_ids: np.ndarray) -> Dict[str, Any]:
        values = np.asarray(values, dtype=np.float32).ravel()
        gene_ids = np.asarray(gene_ids, dtype=np.int64).ravel()
        if self.nonzero_only:
            keep = np.flatnonzero(values > 0)
        else:
            keep = np.arange(values.size)
        if keep.size > self.max_genes:
            # subsample by expression magnitude, not uniformly: keeping the
            # top-expressed genes is standard, but pure top-k makes the token
            # set deterministic given depth, so we keep top-k.
            order = np.argsort(-values[keep])[: self.max_genes]
            keep = keep[np.sort(order)]
        return {
            "gene_ids": gene_ids[keep],
            "values": values[keep],
            "value_bins": None,
        }
