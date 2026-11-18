"""Stage 1 baseline node: principal components of log-scaled counts."""

from pathlib import Path
from typing import Any, Dict

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class PcaFeaturesNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_pca_features",
        stage="analysis",
        tool="scikit-learn",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Baseline cell features: counts scaled to 10,000 per cell and log-transformed, then projected on components fitted on the training cells only. Uses truncated SVD on the sparse matrix, without centring.",
        parameters={
            "n_components": ParameterDefinition(default_value=128, description="Number of components; set to the width of the model it is compared with."),
            "seed": ParameterDefinition(default_value=0, description="Random seed of the solver."),
            "out_path": ParameterDefinition(default_value="runs/pca_features.npy", description="File to write."),
        },
        inputs={
            "data_path": PortDefinition(type=PortType.STR, description="AnnData file with raw counts and an obs column 'split'."),
        },
        outputs={
            "features_path": PortDefinition(type=PortType.STR, description="Array of cells x components, in the cell order of the input file."),
        },
        methods={
            "project": MethodDefinition(description="Fit on training cells, project all cells, save.", inputs=["data_path"], outputs=["features_path"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("project", self.project, method_key="project")

    def project(self, data_path) -> Dict[str, Any]:
        p = self._parameters
        adata = ad.read_h5ad(data_path)
        counts = sp.csr_matrix(adata.X, dtype=np.float64)
        totals = np.asarray(counts.sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        logged = (sp.diags(1e4 / totals) @ counts).tocsr()
        logged.data = np.log1p(logged.data)
        is_train = (adata.obs["split"] == "train").to_numpy()
        svd = TruncatedSVD(n_components=int(p["n_components"]), random_state=int(p["seed"])).fit(logged[is_train])
        out = Path(p["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, svd.transform(logged).astype(np.float32))
        return {"features_path": str(out)}
