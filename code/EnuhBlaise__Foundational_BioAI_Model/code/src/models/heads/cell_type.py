"""Supervised cell-type classification from the pooled cell embedding."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.base import BaseHead
from ...core.registry import register_head
from ...core.types import CellBatch


@register_head("cell_type")
class CellTypeHead(BaseHead):
    output_key = "cell_type"

    def __init__(
        self,
        d_model: int = 256,
        n_classes: int = 8,
        loss_weight: float = 1.0,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        class_weights: Optional[list] = None,
    ):
        super().__init__(d_model, loss_weight)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes)
        )
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None,
            persistent=False,
        )

    def forward(
        self, token_states: torch.Tensor, cell_embedding: torch.Tensor,
        batch: CellBatch,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        logits = self.net(cell_embedding)
        out = {"logits": logits}
        target = batch.targets.get("cell_type")
        if target is None:
            return None, out
        loss = F.cross_entropy(
            logits, target, weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        with torch.no_grad():
            out["accuracy"] = (logits.argmax(-1) == target).float().mean()
        return loss, out
