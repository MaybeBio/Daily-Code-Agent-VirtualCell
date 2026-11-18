import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.registry import autodiscover  # noqa: E402

autodiscover()


@pytest.fixture(scope="session")
def small_adata():
    from src.data.synthetic import make_synthetic
    return make_synthetic(n_cells=400, n_genes=200, n_donors=6, seed=0)


@pytest.fixture(scope="session")
def datamodule():
    from src.data.datamodule import SingleCellDataModule
    return SingleCellDataModule(
        source_cfg={"name": "synthetic", "n_cells": 600, "n_genes": 250,
                    "n_donors": 6, "seed": 0},
        splitter_cfg={"name": "donor_held_out", "val": 0.2, "test": 0.2},
        tokenizer_cfg={"name": "binned", "n_bins": 11, "max_genes": 64},
        n_hvg=150, batch_size=16, return_counts=True,
    ).setup()
