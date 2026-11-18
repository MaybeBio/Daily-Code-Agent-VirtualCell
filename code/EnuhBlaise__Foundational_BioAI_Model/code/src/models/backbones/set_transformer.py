"""Permutation-equivariant transformer over the gene set.

Pre-LayerNorm blocks with ``scaled_dot_product_attention`` (which dispatches
to a fused/flash kernel when one is available).  No positional encoding, for
the reason given in the stem: genes are a set.

Memory note: attention over L genes is O(L^2).  With the usual L = 1-2k
highly expressed genes per cell this is the dominant cost, and it is the
practical reason single-cell transformers truncate to a few thousand genes
rather than modelling the full ~20k transcriptome.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.base import BaseBackbone
from ...core.registry import register_backbone


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = (
            t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.proj(o))
        return x + self.drop(self.mlp(self.ln2(x)))


@register_backbone("set_transformer", aliases=("transformer",))
class SetTransformerBackbone(BaseBackbone):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__(d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_mask = None
        if pad_mask is not None:
            # [B, 1, 1, L]: keys that are padding are masked for every query
            attn_mask = pad_mask[:, None, None, :]
            # a fully-padded row would produce NaN; keep at least one key
            empty = ~attn_mask.any(dim=-1, keepdim=True)
            attn_mask = attn_mask | empty
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.ln_f(x)
