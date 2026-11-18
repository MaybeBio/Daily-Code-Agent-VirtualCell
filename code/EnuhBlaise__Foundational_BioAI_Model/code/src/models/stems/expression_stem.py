"""Gene-token + binned-value stem.

Each token is one gene.  Its embedding is the sum of a learned gene embedding
and a learned embedding of the expression bin, which is the scGPT-style
construction.  There is deliberately **no positional encoding**: a cell's
genes are an unordered set, and adding positions would make the model
sensitive to an arbitrary ordering.  That property is what lets the same
checkpoint accept a different gene panel at inference.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ...core.base import BaseStem
from ...core.registry import register_stem
from ...core.types import CellBatch


@register_stem("expression")
class ExpressionStem(BaseStem):
    def __init__(
        self,
        d_model: int = 256,
        n_genes: int = 2003,
        n_value_bins: int = 52,
        dropout: float = 0.1,
        use_cls: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__(d_model)
        self.gene_emb = nn.Embedding(n_genes, d_model, padding_idx=padding_idx)
        self.value_emb = nn.Embedding(n_value_bins, d_model, padding_idx=0)
        self.use_cls = use_cls
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        nn.init.normal_(self.value_emb.weight, std=0.02)
        with torch.no_grad():
            self.gene_emb.weight[padding_idx].zero_()
            self.value_emb.weight[0].zero_()

    def forward(self, batch: CellBatch) -> torch.Tensor:
        if batch.value_bins is None:
            raise ValueError(
                "ExpressionStem needs discrete value bins; use the 'binned' "
                "tokenizer or switch to the 'continuous' stem."
            )
        x = self.gene_emb(batch.gene_ids) + self.value_emb(batch.value_bins)
        return self.drop(self.norm(x))
