"""Gradient-reversal batch discriminator.

Attached with a positive ``loss_weight`` it pushes the backbone toward an
embedding from which technical batch cannot be predicted.  This is the only
head here whose objective is to *destroy* information.

It is double-edged and the repo says so rather than enabling it by default:
when cell-type composition differs across batches -- which it almost always
does -- removing all batch-predictive signal also removes real biology, and
over-correction shows up as artificially good batch metrics alongside
degraded biological conservation.  Both are reported side by side in the
evaluation so the trade is visible instead of hidden.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.base import BaseHead
from ...core.registry import register_head
from ...core.types import CellBatch


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


@register_head("batch_adversarial")
class BatchAdversarialHead(BaseHead):
    output_key = "batch_adv"

    def __init__(
        self,
        d_model: int = 256,
        n_batches: int = 4,
        loss_weight: float = 1.0,
        lambd: float = 1.0,
        dropout: float = 0.1,
        warmup_steps: int = 0,
    ):
        super().__init__(d_model, loss_weight)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_batches),
        )
        self.lambd = float(lambd)
        self.warmup_steps = int(warmup_steps)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

    def current_lambda(self) -> float:
        if self.warmup_steps <= 0:
            return self.lambd
        t = float(self._step.item()) / self.warmup_steps
        return self.lambd * min(1.0, t)

    def forward(
        self, token_states: torch.Tensor, cell_embedding: torch.Tensor,
        batch: CellBatch,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        lam = self.current_lambda()
        if self.training:
            self._step += 1
        logits = self.net(_GradReverse.apply(cell_embedding, lam))
        out = {"logits": logits, "lambda": torch.tensor(lam)}
        target = batch.targets.get("batch")
        if target is None:
            return None, out
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            # high accuracy => batch is still linearly decodable
            out["batch_accuracy"] = (logits.argmax(-1) == target).float().mean()
        return loss, out
