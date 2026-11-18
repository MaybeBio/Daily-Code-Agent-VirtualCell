"""Device selection with an honest fallback."""
from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            log.warning("cuda requested but unavailable; falling back to cpu")
            return torch.device("cpu")
        if dev.type == "mps" and not torch.backends.mps.is_available():
            log.warning("mps requested but unavailable; falling back to cpu")
            return torch.device("cpu")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
