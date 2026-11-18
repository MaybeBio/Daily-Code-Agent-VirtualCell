"""Stage 1b node: cell features from a published model that runs in its own environment."""

import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class ExternalEmbedNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_external_embed",
        stage="analysis",
        tool="custom",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Run an embedding script of a published model in that model's own Python environment and return its cell features. The script receives --input (AnnData file) and --output (array file) and must write one row per cell, in the cell order of the input.",
        parameters={
            "project_dir": ParameterDefinition(default_value="external/geneformer", description="Folder with the model's own pyproject.toml and its embed.py."),
            "out_path": ParameterDefinition(default_value="runs/external_features.npy", description="File the script writes."),
            "timeout_minutes": ParameterDefinition(default_value=60.0, description="The script is stopped after this time."),
        },
        inputs={
            "data_path": PortDefinition(type=PortType.STR, description="AnnData file of raw counts in the gene identifiers the model expects."),
        },
        outputs={
            "features_path": PortDefinition(type=PortType.STR, description="Array of cells x features, in the cell order of the input file."),
            "n_features": PortDefinition(type=PortType.INT, description="Width of the feature vectors."),
        },
        methods={
            "embed": MethodDefinition(description="Run the script and check its output.", inputs=["data_path"], outputs=["features_path", "n_features"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("embed", self.embed, method_key="embed")

    def embed(self, data_path) -> Dict[str, Any]:
        p = self._parameters
        project, out = Path(p["project_dir"]).resolve(), Path(p["out_path"]).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        command = ["uv", "run", "--project", str(project), "python", str(project / "embed.py"),
                   "--input", str(Path(data_path).resolve()), "--output", str(out)]
        done = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=60 * float(p["timeout_minutes"]))
        (out.parent / (out.stem + ".log")).write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
        if done.returncode != 0:
            raise RuntimeError(f"{project.name} embedding failed: " + done.stderr.strip()[-600:])
        features = np.load(out)
        if not np.isfinite(features).all():
            raise RuntimeError(f"{project.name} wrote features that are not all finite")
        return {"features_path": str(out), "n_features": int(features.shape[1])}
