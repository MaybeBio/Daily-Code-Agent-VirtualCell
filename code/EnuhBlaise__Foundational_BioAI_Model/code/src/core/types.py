"""The data contract between every component.

``CellBatch`` is the *only* object that crosses a stem -> backbone -> head
boundary.  A new modality is added by writing a stem that produces one; a new
objective by writing a head that consumes one.  Nothing else changes.

Shape conventions
-----------------
B  batch (cells)
L  tokens per cell (genes retained after tokenization)
D  model width
G  gene vocabulary size
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import torch

__all__ = ["CellBatch", "ModelOutput"]


@dataclass
class CellBatch:
    """One minibatch of tokenized cells.

    Attributes
    ----------
    gene_ids : LongTensor [B, L]
        Index into the gene vocabulary.  0 is reserved for PAD.
    values : FloatTensor [B, L]
        Tokenizer-dependent expression signal.  For the continuous tokenizer
        this is log1p(CP10K); for the binned tokenizer it is the bin index as
        a float; for the rank tokenizer it is the (normalized) rank.
    value_bins : LongTensor [B, L] or None
        Discrete bin indices when the tokenizer produces them.
    pad_mask : BoolTensor [B, L]
        True where the position holds a real gene (not padding).
    mask_positions : BoolTensor [B, L] or None
        True where the value was corrupted for masked-value modelling.
    targets : dict
        Supervision.  Conventional keys: ``values`` (pre-corruption values for
        MLM), ``counts`` (raw counts for a likelihood head), ``cell_type``
        (long labels), ``batch`` (long labels for an adversarial head).
    meta : dict
        Non-tensor per-cell metadata carried through for evaluation and for
        leakage auditing: ``donor_id``, ``batch_id``, ``study_id``, ``assay``,
        ``tissue``, ``cell_type_name``, ``library_size``.
    """

    gene_ids: torch.Tensor
    values: torch.Tensor
    pad_mask: torch.Tensor
    value_bins: Optional[torch.Tensor] = None
    mask_positions: Optional[torch.Tensor] = None
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gene_ids.shape != self.values.shape:
            raise ValueError(
                f"gene_ids {tuple(self.gene_ids.shape)} and values "
                f"{tuple(self.values.shape)} must match"
            )
        if self.pad_mask.shape != self.gene_ids.shape:
            raise ValueError("pad_mask must have shape [B, L] matching gene_ids")
        if self.pad_mask.dtype != torch.bool:
            raise TypeError("pad_mask must be a bool tensor")

    @property
    def batch_size(self) -> int:
        return int(self.gene_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.gene_ids.shape[1])

    @property
    def device(self) -> torch.device:
        return self.gene_ids.device

    def to(self, device: torch.device | str) -> "CellBatch":
        def mv(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return replace(
            self,
            gene_ids=mv(self.gene_ids),
            values=mv(self.values),
            pad_mask=mv(self.pad_mask),
            value_bins=mv(self.value_bins),
            mask_positions=mv(self.mask_positions),
            targets={k: mv(v) for k, v in self.targets.items()},
            meta=self.meta,
        )

    def __repr__(self) -> str:
        return (
            f"CellBatch(B={self.batch_size}, L={self.seq_len}, "
            f"bins={'y' if self.value_bins is not None else 'n'}, "
            f"masked={'y' if self.mask_positions is not None else 'n'}, "
            f"targets={sorted(self.targets)}, meta={sorted(self.meta)})"
        )


@dataclass
class ModelOutput:
    """What a full forward pass returns."""

    loss: Optional[torch.Tensor] = None
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    cell_embedding: Optional[torch.Tensor] = None  # [B, D]
    token_states: Optional[torch.Tensor] = None  # [B, L, D]
    logits: Dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
