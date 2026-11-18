"""CELLxGENE Census -- the largest harmonized public single-cell corpus.

Census is the right default for pretraining: tens of millions of cells with
consistent ontology terms for ``donor_id``, ``assay``, ``tissue`` and
``cell_type``, which is exactly the metadata the honest splitters need and
which most individual datasets lack.

Requires the optional dependency::

    pip install cellxgene-census

and outbound network access to
``cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com``.  In a
sandbox without that access this source raises a clear error rather than
silently falling back to simulation; use ``source: synthetic`` for offline
development and switch the config over when you have network.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ...core.base import BaseSource
from ...core.registry import register_source
from ..preprocessing import ensure_obs_columns

log = logging.getLogger(__name__)

#: obs columns Census exposes, mapped onto this repo's required names
CENSUS_OBS_MAP = {
    "donor_id": "donor_id",
    "dataset_id": "study_id",
    "assay": "assay",
    "tissue_general": "tissue",
    "cell_type": "cell_type",
    "suspension_type": "suspension_type",
    "disease": "disease",
    "sex": "sex",
}


@register_source("census")
class CensusSource(BaseSource):
    """
    Parameters
    ----------
    value_filter : str
        A Census ``obs`` filter, e.g.
        ``"tissue_general == 'blood' and is_primary_data == True"``.
        Always keep ``is_primary_data == True`` unless you want the same cell
        several times: Census contains overlapping re-analyses of the same
        studies, and ignoring this flag duplicates cells across your splits.
    n_cells : int, optional
        Subsample cap, applied reproducibly with ``seed``.
    """

    def __init__(
        self,
        value_filter: str = "tissue_general == 'blood' and is_primary_data == True",
        organism: str = "Homo sapiens",
        census_version: str = "stable",
        n_cells: Optional[int] = 50_000,
        column_names: Optional[List[str]] = None,
        seed: int = 0,
    ):
        self.value_filter = value_filter
        self.organism = organism
        self.census_version = census_version
        self.n_cells = n_cells
        self.column_names = column_names or list(CENSUS_OBS_MAP)
        self.seed = seed

    def load(self):
        try:
            import cellxgene_census
        except ImportError as exc:
            raise ImportError(
                "CensusSource needs the 'cellxgene-census' package:\n"
                "    pip install cellxgene-census\n"
                "For offline development use data.source.name=synthetic."
            ) from exc

        import numpy as np

        log.info("opening Census '%s' (%s)", self.census_version, self.organism)
        try:
            with cellxgene_census.open_soma(census_version=self.census_version) as census:
                adata = cellxgene_census.get_anndata(
                    census=census,
                    organism=self.organism,
                    obs_value_filter=self.value_filter,
                    obs_column_names=self.column_names,
                )
        except Exception as exc:  # network / policy failures are common
            raise RuntimeError(
                f"could not read CELLxGENE Census: {type(exc).__name__}: {exc}\n"
                "This source needs outbound access to "
                "cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com. "
                "If that host is blocked, run with data.source.name=synthetic "
                "or download an .h5ad by hand and use the 'h5ad' source."
            ) from exc

        adata.obs = adata.obs.rename(columns=CENSUS_OBS_MAP)
        if "batch_id" not in adata.obs:
            # Census has no explicit lane id; donor x assay is the closest
            # honest proxy for a technical batch.
            adata.obs["batch_id"] = (
                adata.obs["donor_id"].astype(str)
                + "|"
                + adata.obs.get("assay", "na").astype(str)
            )
        ensure_obs_columns(adata)
        if self.n_cells and adata.n_obs > self.n_cells:
            rng = np.random.default_rng(self.seed)
            keep = rng.choice(adata.n_obs, self.n_cells, replace=False)
            adata = adata[np.sort(keep)].copy()
        adata.uns["synthetic"] = False
        log.info("Census returned %d cells x %d genes", *adata.shape)
        return adata

    @property
    def provenance(self) -> str:
        return (
            f"cellxgene-census[{self.census_version}] {self.organism} "
            f"filter={self.value_filter!r} n={self.n_cells}"
        )
