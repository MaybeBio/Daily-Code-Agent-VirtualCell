"""Stage 0 node: read the data manifest."""

import csv
from typing import Any, Dict

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class ManifestLoaderNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_manifest_loader",
        stage="io",
        tool="custom",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Load the tab-separated data manifest (one row per public data set) as a list of dictionaries.",
        parameters={
            "manifest_path": ParameterDefinition(
                default_value="data/MANIFEST.tsv",
                description="Path to the manifest file. Its columns include layer, tissue, dataset, accession, reference_doi, local_path, sha256, licence and stage_used.",
            ),
        },
        inputs={},
        outputs={
            "rows": PortDefinition(
                type=PortType.LIST,
                description="One dictionary per data set, keyed by the manifest column names.",
            ),
        },
        methods={
            "load": MethodDefinition(
                description="Read the manifest file.",
                inputs=[],
                outputs=["rows"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("load", self.load, method_key="load")

    def load(self) -> Dict[str, Any]:
        with open(self._parameters["manifest_path"], newline="") as handle:
            return {"rows": list(csv.DictReader(handle, delimiter="\t"))}
