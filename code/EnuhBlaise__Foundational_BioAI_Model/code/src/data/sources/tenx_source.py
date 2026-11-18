"""Load a 10x Genomics CellRanger matrix directory (``matrix.mtx`` + TSVs)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from ...core.base import BaseSource
from ...core.registry import register_source
from ..preprocessing import ensure_obs_columns

log = logging.getLogger(__name__)


@register_source("tenx_mtx")
class TenXMTXSource(BaseSource):
    """
    Parameters
    ----------
    path : str
        Directory containing ``matrix.mtx(.gz)``, ``barcodes.tsv(.gz)`` and
        ``features.tsv(.gz)`` / ``genes.tsv(.gz)``.
    obs_defaults : dict
        Constant metadata to attach (e.g. ``{"donor_id": "PBMC_donorA"}``).
        A raw 10x directory carries no donor or batch annotation, so these
        must be supplied by whoever knows the experimental design.
    """

    def __init__(
        self,
        path: str,
        obs_defaults: Optional[Dict[str, str]] = None,
        var_names: str = "gene_symbols",
        make_unique: bool = True,
    ):
        self.path = Path(path).expanduser()
        self.obs_defaults = dict(obs_defaults or {})
        self.var_names = var_names
        self.make_unique = make_unique

    def load(self):
        import scanpy as sc

        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found")
        adata = sc.read_10x_mtx(
            self.path, var_names=self.var_names, make_unique=self.make_unique
        )
        adata.var_names_make_unique()
        for k, v in self.obs_defaults.items():
            adata.obs[k] = v
        ensure_obs_columns(adata)
        log.info("loaded 10x %s: %d cells x %d genes", self.path.name, *adata.shape)
        return adata

    @property
    def provenance(self) -> str:
        return f"tenx_mtx:{self.path}"
