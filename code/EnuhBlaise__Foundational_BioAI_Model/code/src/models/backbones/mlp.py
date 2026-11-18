"""Bag-of-genes MLP backbone -- an architectural control.

Mean-pools token states immediately and applies an MLP, then broadcasts back.
It has no capacity to model gene-gene interaction, so the gap between it and
the transformer is the part of performance that attention is actually buying.
Several reported single-cell transformer results do not clearly beat this.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...core.base import BaseBackbone
from ...core.registry import register_backbone


@register_backbone("mean_pool_mlp", aliases=("mlp",))
class MeanPoolMLPBackbone(BaseBackbone):
    def __init__(self, d_model: int = 256, n_layers: int = 3, dropout: float = 0.1,
                 mlp_ratio: float = 4.0):
        super().__init__(d_model)
        hidden = int(d_model * mlp_ratio)
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers += [
                nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, d_model),
            ]
        self.mlp = nn.Sequential(*layers)
        self.ln_f = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if pad_mask is not None:
            m = pad_mask.unsqueeze(-1).float()
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = x.mean(1)
        h = self.mlp(pooled)
        return self.ln_f(x + h.unsqueeze(1))
