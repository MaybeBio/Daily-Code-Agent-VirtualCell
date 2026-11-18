"""Synthetic single-cell data.

Kept deliberately separate from ``src/data/sources`` so that no run can
confuse simulated data for measured data: a synthetic AnnData is stamped with
``adata.uns['synthetic'] = True`` and the evaluation report refuses to omit
that flag.
"""
from .generator import SyntheticConfig, SyntheticCellSimulator, make_synthetic

__all__ = ["SyntheticConfig", "SyntheticCellSimulator", "make_synthetic"]
