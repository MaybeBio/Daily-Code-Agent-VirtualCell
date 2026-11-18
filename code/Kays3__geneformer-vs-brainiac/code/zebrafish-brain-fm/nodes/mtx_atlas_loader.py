"""Stage 1 node: load a deposited count matrix and draw a sample of cells."""

import re
from pathlib import Path
from typing import Any, Dict

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class MtxAtlasLoaderNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_mtx_atlas_loader",
        stage="io",
        tool="AnnData",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Load a single-cell count matrix in Matrix Market form (genes in rows) with its gene and cell tables, keep the full matrix in a cache file, and write a random sample of cells as an AnnData file with cell type and sample of origin.",
        parameters={
            "mtx_path": ParameterDefinition(default_value="", description="Matrix Market file, genes in rows and cells in columns."),
            "genes_path": ParameterDefinition(default_value="", description="Gene table: one header line, then one 'gene_id~~symbol' per line."),
            "cells_path": ParameterDefinition(default_value="", description="Cell table, tab-separated with a header: cell_id, celltype, and further columns."),
            "sample_pattern": ParameterDefinition(default_value=r"_[ACGTN]+-\d+$", description="Regular expression removed from the cell id to give the sample name."),
            "cache_path": ParameterDefinition(default_value="data/derived/atlas_full.h5ad", description="Where the full matrix is kept after the first read."),
            "n_cells": ParameterDefinition(default_value=10000, description="Cells to draw at random, without replacement. 0 keeps all."),
            "seed": ParameterDefinition(default_value=0, description="Random seed of the draw."),
            "out_path": ParameterDefinition(default_value="data/derived/atlas_sample.h5ad", description="File to write."),
        },
        inputs={},
        outputs={
            "data_path": PortDefinition(type=PortType.STR, description="AnnData file: raw counts, obs columns celltype and sample, var index of gene ids."),
            "n_cells": PortDefinition(type=PortType.INT, description="Cells written."),
            "n_samples": PortDefinition(type=PortType.INT, description="Distinct samples among the cells written."),
        },
        methods={
            "load": MethodDefinition(description="Read or reuse the full matrix, draw the sample, write it.", inputs=[], outputs=["data_path", "n_cells", "n_samples"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("load", self.load, method_key="load")

    def load(self) -> Dict[str, Any]:
        p = self._parameters
        cache = Path(p["cache_path"])
        if cache.is_file():
            adata = ad.read_h5ad(cache)
        else:
            counts = scipy.io.mmread(p["mtx_path"]).T.tocsr().astype(np.float32)
            genes = [line.strip() for line in pd.read_csv(p["genes_path"], header=0, sep="\t").iloc[:, 0]]
            cells = pd.read_csv(p["cells_path"], sep="\t")
            if counts.shape != (len(cells), len(genes)):
                raise ValueError(f"matrix {counts.shape} does not match {len(cells)} cells and {len(genes)} genes")
            obs = pd.DataFrame({
                "celltype": cells["celltype"].astype(str).to_numpy(),
                "sample": [re.sub(p["sample_pattern"], "", c) for c in cells["cell_id"]],
            }, index=cells["cell_id"].astype(str).to_numpy())
            var = pd.DataFrame({"symbol": [g.split("~~")[-1] for g in genes]}, index=[g.split("~~")[0] for g in genes])
            adata = ad.AnnData(X=counts, obs=obs, var=var)
            adata.var_names_make_unique()
            cache.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(cache)
        n = int(p["n_cells"])
        if 0 < n < adata.n_obs:
            keep = np.sort(np.random.default_rng(int(p["seed"])).choice(adata.n_obs, n, replace=False))
            adata = adata[keep].copy()
        out = Path(p["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out)
        return {"data_path": str(out), "n_cells": int(adata.n_obs), "n_samples": int(adata.obs["sample"].nunique())}
