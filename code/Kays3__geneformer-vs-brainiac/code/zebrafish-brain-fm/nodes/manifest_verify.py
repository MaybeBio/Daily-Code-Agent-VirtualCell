"""Stage 0 node: verify downloaded data sets against the manifest."""

import hashlib
from pathlib import Path
from typing import Any, Dict

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ManifestVerifyNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_manifest_verify",
        stage="analysis",
        tool="custom",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="For each ready data set, check on this machine that its folder holds a SHA256SUMS file whose own sha256 equals the manifest value, and that every listed file matches its checksum.",
        parameters={
            "root": ParameterDefinition(default_value=".", description="Folder that the manifest's local_path values are relative to."),
        },
        inputs={
            "rows": PortDefinition(type=PortType.LIST, description="Manifest rows as dictionaries, from the manifest loader."),
            "ready": PortDefinition(type=PortType.LIST, description="Names of the data sets with path, sha256 and licence recorded, from the manifest check."),
        },
        outputs={
            "verified": PortDefinition(type=PortType.LIST, description="Names of the data sets whose files all match."),
            "failed": PortDefinition(type=PortType.DICT, description="For each data set that is present but does not match, the reason."),
            "absent": PortDefinition(type=PortType.LIST, description="Names of ready data sets whose folder is not on this machine."),
        },
        methods={
            "verify": MethodDefinition(description="Recompute and compare the checksums.", inputs=["rows", "ready"], outputs=["verified", "failed", "absent"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("verify", self.verify, method_key="verify")

    def verify(self, rows, ready) -> Dict[str, Any]:
        root = Path(self._parameters["root"])
        verified, failed, absent = [], {}, []
        for row in rows:
            name = row["dataset"]
            if name not in ready:
                continue
            folder = root / row["local_path"]
            sums = folder / "SHA256SUMS"
            if not sums.is_file():
                absent.append(name)
            elif sha256_of(sums) != row["sha256"].strip():
                failed[name] = "SHA256SUMS does not match the manifest"
            else:
                bad = [f for h, f in (line.split() for line in sums.read_text().splitlines() if line.strip())
                       if not (folder / f).is_file() or sha256_of(folder / f) != h]
                if bad:
                    failed[name] = "files do not match: " + ", ".join(bad)
                else:
                    verified.append(name)
        return {"verified": verified, "failed": failed, "absent": absent}
