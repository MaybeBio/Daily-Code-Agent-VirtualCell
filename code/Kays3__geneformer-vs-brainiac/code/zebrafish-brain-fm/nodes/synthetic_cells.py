"""Smoke-test node: make synthetic ranked-gene cells. No real data."""

from pathlib import Path
from typing import Any, Dict

import torch

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class SyntheticCellsNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_synthetic_cells",
        stage="io",
        tool="PyTorch",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Generate synthetic cells as ranked gene-token lists from a few random cell-type profiles plus noise, and save them to a file. For smoke tests and timing only; the cells carry no biology.",
        parameters={
            "n_cells": ParameterDefinition(default_value=10000, description="Number of synthetic cells."),
            "vocab_size": ParameterDefinition(default_value=30000, description="Number of gene tokens."),
            "seq_len": ParameterDefinition(default_value=2048, description="Genes kept per cell, ranked from highest to lowest score."),
            "n_types": ParameterDefinition(default_value=8, description="Number of synthetic cell types."),
            "noise": ParameterDefinition(default_value=1.0, description="Standard deviation of the per-cell noise added to the type profile (profile SD is 1)."),
            "seed": ParameterDefinition(default_value=0, description="Random seed."),
            "out_path": ParameterDefinition(default_value="data/derived/synthetic_cells.pt", description="File to write."),
        },
        inputs={},
        outputs={
            "data_path": PortDefinition(type=PortType.STR, description="Path of the saved file: a dict with 'tokens' (int32, n_cells x seq_len) and 'cell_type' (int64, n_cells)."),
            "n_cells": PortDefinition(type=PortType.INT, description="Number of cells written."),
        },
        methods={
            "generate": MethodDefinition(description="Generate and save the synthetic cells.", inputs=[], outputs=["data_path", "n_cells"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("generate", self.generate, method_key="generate")

    def generate(self) -> Dict[str, Any]:
        p = self._parameters
        n, vocab, seq_len = int(p["n_cells"]), int(p["vocab_size"]), int(p["seq_len"])
        gen = torch.Generator().manual_seed(int(p["seed"]))
        profiles = torch.randn(int(p["n_types"]), vocab, generator=gen)
        cell_type = torch.randint(0, int(p["n_types"]), (n,), generator=gen)
        tokens = torch.empty(n, seq_len, dtype=torch.int32)
        for start in range(0, n, 500):
            types = cell_type[start:start + 500]
            scores = profiles[types] + float(p["noise"]) * torch.randn(len(types), vocab, generator=gen)
            tokens[start:start + 500] = scores.topk(seq_len, dim=1).indices.to(torch.int32)
        out = Path(p["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tokens": tokens, "cell_type": cell_type, "vocab_size": vocab}, out)
        return {"data_path": str(out), "n_cells": n}
