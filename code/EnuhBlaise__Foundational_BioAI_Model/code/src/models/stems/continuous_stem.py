"""Gene-token + continuous-value stem (no discretization).

The value is passed through a small MLP and added to the gene embedding.
Compared with ``ExpressionStem`` this keeps expression magnitude exactly,
which is the control for whether binning helps or merely regularizes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...core.base import BaseStem
from ...core.registry import register_stem
from ...core.types import CellBatch


@register_stem("continuous")
class ContinuousStem(BaseStem):
    def __init__(
        self,
        d_model: int = 256,
        n_genes: int = 2003,
        dropout: float = 0.1,
        hidden: int = 64,
        padding_idx: int = 0,
    ):
        super().__init__(d_model)
        self.gene_emb = nn.Embedding(n_genes, d_model, padding_idx=padding_idx)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        with torch.no_grad():
            self.gene_emb.weight[padding_idx].zero_()

    def forward(self, batch: CellBatch) -> torch.Tensor:
        v = batch.values.unsqueeze(-1)
        x = self.gene_emb(batch.gene_ids) + self.value_mlp(v)
        return self.drop(self.norm(x))
