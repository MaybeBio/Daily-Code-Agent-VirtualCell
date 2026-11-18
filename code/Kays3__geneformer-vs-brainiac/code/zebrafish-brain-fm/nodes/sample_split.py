"""Stage 1 node: hold out whole samples."""

import random
from pathlib import Path
from typing import Any, Dict

import anndata as ad
import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class SampleSplitNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_sample_split",
        stage="analysis",
        tool="AnnData",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Split cells into training and held-out sets by sample of origin, so that no sample contributes cells to both. The held-out samples are drawn at random from the sorted sample names with a fixed seed.",
        parameters={
            "holdout_fraction": ParameterDefinition(default_value=0.2, description="Share of samples to hold out, rounded to a whole number of samples, at least one."),
            "seed": ParameterDefinition(default_value=0, description="Random seed of the draw."),
            "out_path": ParameterDefinition(default_value="data/derived/atlas_split.h5ad", description="File to write."),
        },
        inputs={
            "data_path": PortDefinition(type=PortType.STR, description="AnnData file with an obs column 'sample'."),
        },
        outputs={
            "data_path": PortDefinition(type=PortType.STR, description="The same cells with an obs column 'split' holding 'train' or 'heldout'."),
            "heldout_samples": PortDefinition(type=PortType.LIST, description="Names of the held-out samples."),
            "n_heldout_cells": PortDefinition(type=PortType.INT, description="Cells in the held-out samples."),
        },
        methods={
            "split": MethodDefinition(description="Draw the held-out samples and label every cell.", inputs=["data_path"], outputs=["data_path", "heldout_samples", "n_heldout_cells"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("split", self.split, method_key="split")

    def split(self, data_path) -> Dict[str, Any]:
        p = self._parameters
        adata = ad.read_h5ad(data_path)
        samples = sorted(adata.obs["sample"].unique())
        n_hold = max(1, round(float(p["holdout_fraction"]) * len(samples)))
        heldout = sorted(random.Random(int(p["seed"])).sample(samples, n_hold))
        adata.obs["split"] = np.where(adata.obs["sample"].isin(heldout), "heldout", "train")
        out = Path(p["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out)
        return {"data_path": str(out), "heldout_samples": heldout, "n_heldout_cells": int((adata.obs["split"] == "heldout").sum())}
