"""Masked expression modelling -- the main self-supervised objective.

Two modes:

``classification``
    Predict the discrete expression bin.  Loss is cross-entropy.

``regression``
    Predict the continuous value.  Loss is MSE.

Both compute loss **only at masked positions**.  A subtlety specific to
single-cell data: if masking is uniform over the token set and the token set
is already restricted to nonzero genes, the model never has to predict a
zero, and reconstruction accuracy is then not evidence that it understands
dropout.  ``zero_fraction`` deliberately injects zero-valued targets so the
objective is not trivially biased toward expressed genes.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.base import BaseHead
from ...core.registry import register_head
from ...core.types import CellBatch


@register_head("masked_value")
class MaskedValueHead(BaseHead):
    output_key = "masked_value"

    def __init__(
        self,
        d_model: int = 256,
        n_value_bins: int = 52,
        mode: str = "classification",
        loss_weight: float = 1.0,
        hidden_ratio: float = 1.0,
    ):
        super().__init__(d_model, loss_weight)
        if mode not in ("classification", "regression"):
            raise ValueError("mode must be 'classification' or 'regression'")
        self.mode = mode
        out = n_value_bins if mode == "classification" else 1
        hidden = int(d_model * hidden_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(
        self, token_states: torch.Tensor, cell_embedding: torch.Tensor,
        batch: CellBatch,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        logits = self.net(token_states)
        out = {"logits": logits}
        mask = batch.mask_positions
        if mask is None or not mask.any():
            return None, out

        if self.mode == "classification":
            target = batch.targets.get("value_bins")
            if target is None:
                raise KeyError("masked_value(classification) needs targets['value_bins']")
            loss = F.cross_entropy(logits[mask], target[mask])
            with torch.no_grad():
                out["accuracy"] = (logits[mask].argmax(-1) == target[mask]).float().mean()
        else:
            target = batch.targets.get("values")
            if target is None:
                raise KeyError("masked_value(regression) needs targets['values']")
            pred = logits.squeeze(-1)
            loss = F.mse_loss(pred[mask], target[mask])
            with torch.no_grad():
                # variance-explained at masked positions; negative means the
                # head is doing worse than predicting the mean
                t = target[mask]
                out["r2"] = 1 - loss / t.var().clamp(min=1e-8)
        return loss, out
