"""Load a local ``.h5ad`` -- the format every public single-cell portal exports.

This is the path to use for real data downloaded by hand:

    CELLxGENE Discover   https://cellxgene.cziscience.com/datasets
    Human Cell Atlas     https://data.humancellatlas.org
    Tabula Sapiens       https://tabula-sapiens.sf.czbiohub.org
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from ...core.base import BaseSource
from ...core.registry import register_source
from ..preprocessing import ensure_obs_columns, is_raw_counts

log = logging.getLogger(__name__)


@register_source("h5ad")
class H5ADSource(BaseSource):
    """
    Parameters
    ----------
    path : str
        Path to the ``.h5ad`` file.
    obs_map : dict
        Rename map from the file's obs columns to this repo's required names,
        e.g. ``{"donor": "donor_id", "sample": "batch_id"}``.
    raw_layer : str, optional
        Layer holding raw counts.  Many portal files put normalized values in
        ``.X`` and counts in ``.raw`` or ``layers['counts']``; using the
        normalized matrix with a count likelihood is a silent, serious bug, so
        this source checks and refuses.
    """

    def __init__(
        self,
        path: str,
        obs_map: Optional[Dict[str, str]] = None,
        raw_layer: Optional[str] = None,
        use_raw: bool = False,
        require_counts: bool = True,
        backed: bool = False,
    ):
        self.path = Path(path).expanduser()
        self.obs_map = dict(obs_map or {})
        self.raw_layer = raw_layer
        self.use_raw = use_raw
        self.require_counts = require_counts
        self.backed = backed

    def load(self):
        import anndata as ad

        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Download a dataset from "
                "https://cellxgene.cziscience.com/datasets (the 'Download "
                "h5ad' button) and point data.source.path at it."
            )
        adata = ad.read_h5ad(self.path, backed="r" if self.backed else None)
        if self.use_raw and adata.raw is not None:
            adata = adata.raw.to_adata()
        if self.raw_layer:
            if self.raw_layer not in adata.layers:
                raise KeyError(
                    f"layer '{self.raw_layer}' not in {list(adata.layers)}"
                )
            adata.X = adata.layers[self.raw_layer]
        if self.obs_map:
            adata.obs = adata.obs.rename(columns=self.obs_map)
        ensure_obs_columns(adata)
        if self.require_counts and not is_raw_counts(adata.X):
            raise ValueError(
                f"{self.path.name}: .X does not look like raw integer counts. "
                "Portal files frequently store normalized values in .X. Pass "
                "raw_layer='counts' or use_raw=true, or set "
                "require_counts=false if you intend to model normalized data."
            )
        log.info("loaded %s: %d cells x %d genes", self.path.name, *adata.shape)
        return adata

    @property
    def provenance(self) -> str:
        return f"h5ad:{self.path}"
