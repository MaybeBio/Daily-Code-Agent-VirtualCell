"""Abstract interfaces enforcing the Stem / Backbone / Head contract.

The backbone only ever sees ``[B, L, D]`` continuous states plus a padding
mask.  It has no knowledge of genes, counts, bins or assays.  That is what
makes a new modality additive rather than invasive.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .registry import (
    BACKBONES,
    BASELINES,
    HEADS,
    SOURCES,
    SPLITTERS,
    STEMS,
    TOKENIZERS,
)
from .types import CellBatch

__all__ = [
    "BaseStem",
    "BaseBackbone",
    "BaseHead",
    "BaseTokenizer",
    "BaseSource",
    "BaseSplitter",
    "BaseBaseline",
]


class BaseStem(nn.Module, ABC):
    """Modality-specific encoder: raw tokens -> ``[B, L, D]``."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward(self, batch: CellBatch) -> torch.Tensor:
        """Return ``[B, L, D]`` token states."""

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"


class BaseBackbone(nn.Module, ABC):
    """Modality-agnostic sequence model: ``[B, L, D] -> [B, L, D]``."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``x`` is [B, L, D]; ``pad_mask`` is [B, L] with True = real token."""

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"


class BaseHead(nn.Module, ABC):
    """Objective-specific decoder consuming backbone states.

    A head returns ``(loss_or_None, dict_of_logits_or_stats)``.  Heads are
    additive: several run on the same backbone output and their losses are
    combined with configurable weights.
    """

    #: name of the key this head writes into ``ModelOutput.logits``
    output_key: str = "head"
    #: whether this head needs ``CellBatch.targets`` to compute a loss
    requires_targets: bool = True

    def __init__(self, d_model: int, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight = float(loss_weight)

    @abstractmethod
    def forward(
        self,
        token_states: torch.Tensor,
        cell_embedding: torch.Tensor,
        batch: CellBatch,
    ) -> tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        ...

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, loss_weight={self.loss_weight}"


class BaseTokenizer(ABC):
    """Dense expression vector -> token sequence.

    Implementations differ in what they throw away, which is the whole
    question in single-cell tokenization: ranks discard magnitude, bins
    discard resolution, continuous values discard nothing but give the model
    no discrete vocabulary to predict over.
    """

    #: does this tokenizer emit ``value_bins``?
    produces_bins: bool = False

    @abstractmethod
    def __call__(
        self,
        counts: "Any",
        gene_ids: "Any",
    ) -> Dict[str, Any]:
        """Return dict with keys ``gene_ids``, ``values`` and optionally
        ``value_bins`` for a single cell."""

    @property
    @abstractmethod
    def n_value_bins(self) -> int:
        """Size of the value vocabulary (1 when values stay continuous)."""


class BaseSource(ABC):
    """A provider of ``AnnData``.

    Sources are responsible for returning a matrix of **raw counts** in
    ``.X`` plus the obs columns this repo relies on for honest evaluation:
    ``donor_id``, ``batch_id``, ``study_id``, ``assay``, ``tissue``,
    ``cell_type``.  Missing columns must be filled explicitly (with
    ``"unknown"``) rather than silently dropped, so a splitter can refuse to
    pretend a dataset has donor structure it does not have.
    """

    @abstractmethod
    def load(self) -> "Any":
        """Return an ``anndata.AnnData`` with raw counts in ``.X``."""

    #: human-readable provenance string written into run manifests
    @property
    def provenance(self) -> str:
        return self.__class__.__name__


class BaseSplitter(ABC):
    """Produce train/val/test index arrays from cell metadata."""

    @abstractmethod
    def split(self, obs: "Any") -> Dict[str, "Any"]:
        """Return ``{'train': idx, 'val': idx, 'test': idx}``."""

    #: a splitter declares which obs column it partitions on so the leakage
    #: audit can check that column does not straddle splits
    group_key: Optional[str] = None


class BaseBaseline(ABC):
    """A non-foundation-model control.

    These exist because the central empirical claim of a single-cell
    foundation model is that it beats them.  Several published benchmarks
    found that it often does not, so the repo treats them as mandatory
    rather than optional.
    """

    @abstractmethod
    def fit(self, X: "Any", obs: "Any") -> "BaseBaseline":
        ...

    @abstractmethod
    def embed(self, X: "Any") -> "Any":
        """Return ``[N, d]`` cell embeddings."""


# Bind base classes to their registries now that both exist.  Registration of
# a class that does not subclass the right base will fail loudly.
STEMS.base = BaseStem
BACKBONES.base = BaseBackbone
HEADS.base = BaseHead
TOKENIZERS.base = BaseTokenizer
SOURCES.base = BaseSource
SPLITTERS.base = BaseSplitter
BASELINES.base = BaseBaseline
