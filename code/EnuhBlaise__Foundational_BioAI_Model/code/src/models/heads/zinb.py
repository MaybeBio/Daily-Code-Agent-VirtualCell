"""Zero-inflated negative-binomial reconstruction head.

Models raw counts directly rather than log-normalized values, which is the
scVI formulation.  Worth keeping available because it is the principled
treatment of count noise, and because comparing it against the masked-value
objective on the same backbone isolates how much the likelihood choice
matters versus the architecture.

The zero-inflation term is optional (``zero_inflated=False`` gives plain NB).
For UMI protocols the evidence is that plain NB is usually sufficient and
zero-inflation mostly absorbs model misspecification.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.base import BaseHead
from ...core.registry import register_head
from ...core.types import CellBatch


def _nb_log_prob(x, mu, theta, eps=1e-8):
    log_theta_mu = torch.log(theta + mu + eps)
    return (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )


@register_head("zinb")
class ZINBDecoderHead(BaseHead):
    output_key = "zinb"

    def __init__(
        self,
        d_model: int = 256,
        loss_weight: float = 1.0,
        zero_inflated: bool = False,
        hidden_ratio: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(d_model, loss_weight)
        hidden = int(d_model * hidden_ratio)
        self.zero_inflated = zero_inflated
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU()
        )
        self.rate = nn.Linear(hidden, 1)       # per-gene mean fraction (logit)
        self.log_theta = nn.Linear(hidden, 1)  # per-gene dispersion
        self.dropout_logit = nn.Linear(hidden, 1) if zero_inflated else None
        self.eps = eps

    def forward(
        self, token_states: torch.Tensor, cell_embedding: torch.Tensor,
        batch: CellBatch,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        h = self.trunk(token_states)
        pad = batch.pad_mask
        logits = self.rate(h).squeeze(-1).masked_fill(~pad, -1e4)
        frac = torch.softmax(logits, dim=-1)  # composition over retained genes
        theta = torch.exp(self.log_theta(h).squeeze(-1).clamp(-8, 8))
        out = {"rate": frac, "theta": theta}

        counts = batch.targets.get("counts")
        if counts is None:
            return None, out
        lib = counts.sum(dim=-1, keepdim=True)
        mu = frac * lib + self.eps
        lp = _nb_log_prob(counts, mu, theta, self.eps)
        if self.zero_inflated and self.dropout_logit is not None:
            pi = self.dropout_logit(h).squeeze(-1)
            log_nb_zero = _nb_log_prob(torch.zeros_like(counts), mu, theta, self.eps)
            zero_case = F.softplus(-pi + log_nb_zero) - F.softplus(-pi)
            nonzero_case = -F.softplus(-pi) + lp
            lp = torch.where(counts < self.eps, zero_case, nonzero_case)
            out["dropout_logit"] = pi
        loss = -(lp * pad).sum() / pad.sum().clamp(min=1)
        return loss, out
