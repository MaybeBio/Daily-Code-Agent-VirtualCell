"""scGPT whole-human features for the cells of an AnnData file with human gene symbols.

Uses the package's own embed_data function, without fine-tuning.

    uv run python embed.py --input cells.h5ad --output features.npy
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scgpt.tasks import embed_data

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

source = ad.read_h5ad(args.input)
symbols = np.array([str(s) for s in source.var["symbol"]], dtype=object)
named = (symbols != "") & ~pd.Series(symbols).duplicated().to_numpy()
cells = ad.AnnData(
    X=source.X[:, named],
    obs=pd.DataFrame(index=pd.Index([str(i) for i in range(source.n_obs)], dtype=object)),
    var=pd.DataFrame({"gene_name": symbols[named]}, index=pd.Index(symbols[named], dtype=object)),
)
device = "cuda" if torch.cuda.is_available() else "cpu"
embedded = embed_data(cells, HERE / "checkpoint", gene_col="gene_name", max_length=1200, batch_size=64,
                      device=device, use_fast_transformer=False, return_new_adata=True)
features = np.asarray(embedded.X, dtype=np.float32)
if features.shape[0] != source.n_obs:
    raise SystemExit(f"scGPT returned {features.shape[0]} rows for {source.n_obs} cells")
np.save(args.output, features)
print(f"cells in: {source.n_obs}; genes with a unique symbol: {int(named.sum())}; feature width: {features.shape[1]}; device: {device}")
