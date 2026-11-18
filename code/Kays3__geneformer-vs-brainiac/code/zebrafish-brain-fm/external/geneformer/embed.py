"""Geneformer V2-104M features for the cells of an AnnData file with human Ensembl gene ids.

The tokenizer and the feature (the <cls> position of the second-to-last layer) follow
quickstart/run_quickstart.py of Kays3/geneformer-petadimension-starter (MIT).

    uv run python embed.py --input cells.h5ad --output features.npy
"""

import argparse
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer
from transformers import BertModel

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "Geneformer" / "Geneformer-V2-104M"

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

out = Path(args.output)
work = out.parent / (out.stem + "_work")
shutil.rmtree(work, ignore_errors=True)
(work / "input").mkdir(parents=True)

source = ad.read_h5ad(args.input)
n_cells = source.n_obs
# Rebuild the tables with plain string indexes: this environment's anndata does not
# write the string type that the project environment's pandas produces.
genes = np.array([str(g) for g in source.var_names], dtype=object)
obs = pd.DataFrame({"n_counts": np.asarray(source.X.sum(axis=1)).ravel().astype(float), "cell_index": np.arange(n_cells)},
                   index=pd.Index([str(i) for i in range(n_cells)], dtype=object))
var = pd.DataFrame({"ensembl_id": genes}, index=pd.Index(genes, dtype=object))
ad.AnnData(X=source.X, obs=obs, var=var).write_h5ad(work / "input" / "cells.h5ad")

tokenizer = TranscriptomeTokenizer(custom_attr_name_dict={"cell_index": "cell_index"}, nproc=4, model_version="V2")
tokenizer.tokenize_data(data_directory=str(work / "input"), output_directory=str(work / "tokens"),
                        output_prefix="cells", file_format="h5ad")
tokens = load_from_disk(str(work / "tokens" / "cells.dataset"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BertModel.from_pretrained(MODEL_DIR).to(device).eval()
features = np.zeros((n_cells, model.config.hidden_size), dtype=np.float32)
seen = np.zeros(n_cells, dtype=bool)
lengths = []
with torch.no_grad():
    for i, (index, input_ids) in enumerate(zip(tokens["cell_index"], tokens["input_ids"])):
        ids = torch.tensor([input_ids], device=device)
        hidden = model(input_ids=ids, output_hidden_states=True).hidden_states[-2]
        features[index] = hidden[0][0].float().cpu().numpy()
        seen[index] = True
        lengths.append(len(input_ids))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} / {len(tokens)} cells", flush=True)

np.save(out, features)
print(f"cells in: {n_cells}; cells embedded: {int(seen.sum())}; cells with no tokens, left as zeros: {int((~seen).sum())}")
print(f"median tokens per cell: {int(np.median(lengths))}; feature width: {features.shape[1]}; device: {device}")
shutil.rmtree(work, ignore_errors=True)
