"""Stage 0 node: say which data sets may be used."""

from typing import Any, Dict

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    PortDefinition,
)

REQUIRED = ("local_path", "sha256", "licence")


class ManifestCheckNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_manifest_check",
        stage="analysis",
        tool="custom",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Split the data sets of a manifest into ready (path, sha256 and licence recorded) and blocked (any of the three missing).",
        parameters={},
        inputs={
            "rows": PortDefinition(
                type=PortType.LIST,
                description="Manifest rows as dictionaries, from the manifest loader.",
            ),
        },
        outputs={
            "ready": PortDefinition(
                type=PortType.LIST,
                description="Names of the data sets that may be used.",
            ),
            "blocked": PortDefinition(
                type=PortType.DICT,
                description="For each data set that may not be used yet, the list of its empty manifest fields.",
            ),
        },
        methods={
            "check": MethodDefinition(
                description="Check the three required fields of each row.",
                inputs=["rows"],
                outputs=["ready", "blocked"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("check", self.check, method_key="check")

    def check(self, rows) -> Dict[str, Any]:
        ready, blocked = [], {}
        for row in rows:
            missing = [field for field in REQUIRED if not (row.get(field) or "").strip()]
            if missing:
                blocked[row["dataset"]] = missing
            else:
                ready.append(row["dataset"])
        return {"ready": ready, "blocked": blocked}
