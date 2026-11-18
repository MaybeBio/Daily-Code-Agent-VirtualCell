"""Registry wrapper around the synthetic simulator."""
from __future__ import annotations

from ...core.base import BaseSource
from ...core.registry import register_source
from ..synthetic import SyntheticConfig, SyntheticCellSimulator


@register_source("synthetic")
class SyntheticSource(BaseSource):
    def __init__(self, **kwargs):
        self.cfg = SyntheticConfig(**kwargs)

    def load(self):
        return SyntheticCellSimulator(self.cfg).generate()

    @property
    def provenance(self) -> str:
        return f"synthetic(seed={self.cfg.seed}, n_cells={self.cfg.n_cells})"
