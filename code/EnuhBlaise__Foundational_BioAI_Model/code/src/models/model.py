"""Composition of stem -> backbone -> heads.

Everything is built from config through the registry, so an experiment
changes architecture by editing YAML, not this file.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..core.registry import BACKBONES, HEADS, STEMS
from ..core.types import CellBatch, ModelOutput

log = logging.getLogger(__name__)

__all__ = ["CellFoundationModel", "build_model"]


class CellFoundationModel(nn.Module):
    """A stem, a backbone, and any number of heads.

    Pooling to a single cell embedding is masked-mean over real tokens.  The
    embedding is L2-normalized optionally; note that normalizing removes
    library-size magnitude from the representation, which is usually what you
    want and occasionally destroys signal you needed.
    """

    def __init__(
        self,
        stem: nn.Module,
        backbone: nn.Module,
        heads: Optional[Mapping[str, nn.Module]] = None,
        pool: str = "mean",
        normalize_embedding: bool = False,
    ):
        super().__init__()
        if stem.d_model != backbone.d_model:
            raise ValueError(
                f"stem d_model {stem.d_model} != backbone d_model {backbone.d_model}"
            )
        self.stem = stem
        self.backbone = backbone
        self.heads = nn.ModuleDict(dict(heads or {}))
        for name, h in self.heads.items():
            if getattr(h, "d_model", backbone.d_model) != backbone.d_model:
                raise ValueError(f"head '{name}' d_model mismatch")
        if pool not in ("mean", "max"):
            raise ValueError("pool must be 'mean' or 'max'")
        self.pool = pool
        self.normalize_embedding = normalize_embedding

    @property
    def d_model(self) -> int:
        return self.backbone.d_model

    def _pool(self, states: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        m = pad_mask.unsqueeze(-1).to(states.dtype)
        if self.pool == "mean":
            emb = (states * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            emb = states.masked_fill(~pad_mask.unsqueeze(-1), float("-inf")).max(1).values
            emb = torch.nan_to_num(emb, neginf=0.0)
        if self.normalize_embedding:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    def encode(self, batch: CellBatch) -> torch.Tensor:
        """Cell embedding only -- what the evaluation layer consumes."""
        states = self.backbone(self.stem(batch), batch.pad_mask)
        return self._pool(states, batch.pad_mask)

    def forward(self, batch: CellBatch, head_subset: Optional[Sequence[str]] = None) -> ModelOutput:
        states = self.backbone(self.stem(batch), batch.pad_mask)
        emb = self._pool(states, batch.pad_mask)
        out = ModelOutput(cell_embedding=emb, token_states=states)
        total = None
        names = list(self.heads) if head_subset is None else list(head_subset)
        for name in names:
            head = self.heads[name]
            loss, extra = head(states, emb, batch)
            for k, v in extra.items():
                if k == "logits":
                    out.logits[name] = v
                elif isinstance(v, torch.Tensor) and v.ndim == 0:
                    out.metrics[f"{name}/{k}"] = float(v.detach())
            if loss is not None:
                w = getattr(head, "loss_weight", 1.0)
                out.losses[name] = loss.detach()
                total = loss * w if total is None else total + loss * w
        out.loss = total
        return out

    def n_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)


def build_model(cfg: Mapping[str, Any], **context: Any) -> CellFoundationModel:
    """Instantiate from a model config.

    ``context`` carries values known only at runtime -- vocabulary size,
    number of classes, number of batches -- and is injected into any component
    whose config declares the placeholder string ``AUTO``.  This keeps
    dataset-dependent sizes out of the YAML, which otherwise have to be kept
    in sync by hand and silently corrupt a model when they drift.

    A top-level ``d_model`` propagates to the stem, backbone and every head
    that does not set its own.  Width is a property of the model, not of each
    part, and repeating it three times is a reliable way to produce a silent
    dimension mismatch.
    """
    d_model = cfg.get("d_model")

    def _resolve(d: Mapping[str, Any]) -> Dict[str, Any]:
        out = {}
        if d_model is not None and "d_model" not in d:
            out["d_model"] = d_model
        for k, v in dict(d).items():
            if isinstance(v, str) and v.strip() in ("AUTO", "${auto}"):
                if k not in context:
                    raise KeyError(
                        f"config asks for auto value '{k}' but the runner did "
                        f"not supply it; available: {sorted(context)}"
                    )
                out[k] = context[k]
            else:
                out[k] = v
        return out

    stem = STEMS.build(_resolve(cfg["stem"]))
    backbone = BACKBONES.build(_resolve(cfg["backbone"]))
    heads = {}
    for name, hcfg in (cfg.get("heads") or {}).items():
        if hcfg is None or hcfg.get("enabled", True) is False:
            continue
        hc = {k: v for k, v in dict(hcfg).items() if k != "enabled"}
        heads[name] = HEADS.build(_resolve(hc))
    model = CellFoundationModel(
        stem=stem,
        backbone=backbone,
        heads=heads,
        pool=cfg.get("pool", "mean"),
        normalize_embedding=bool(cfg.get("normalize_embedding", False)),
    )
    log.info(
        "model: %s + %s + heads[%s] = %.2fM params",
        type(stem).__name__, type(backbone).__name__, ", ".join(heads) or "-",
        model.n_parameters() / 1e6,
    )
    return model
