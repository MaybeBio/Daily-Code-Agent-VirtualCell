"""Stage 1 node: cell embeddings from the encoder."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from rank_encoder import RankEncoder, trim
from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class EmbedCellsNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_embed_cells",
        stage="analysis",
        tool="PyTorch",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Compute one vector per cell as the mean hidden state of the rank-token encoder over the cell's genes. With random_init the trained weights are ignored and a freshly initialised model of the same shape is used, as a control for what training adds.",
        parameters={
            "random_init": ParameterDefinition(default_value=False, description="Use an untrained model of the same shape."),
            "seed": ParameterDefinition(default_value=0, description="Random seed of the untrained model."),
            "batch_size": ParameterDefinition(default_value=64, description="Cells per forward pass."),
            "out_path": ParameterDefinition(default_value="runs/embeddings.npy", description="File to write."),
        },
        inputs={
            "tokens_path": PortDefinition(type=PortType.STR, description="Token file from the rank tokenizer."),
            "checkpoint_path": PortDefinition(type=PortType.STR, description="Checkpoint from the encoder training node; it also gives the model shape."),
        },
        outputs={
            "features_path": PortDefinition(type=PortType.STR, description="Array of cells x width, in the cell order of the token file."),
        },
        methods={
            "embed": MethodDefinition(description="Run every cell through the encoder and save the mean hidden states.", inputs=["tokens_path", "checkpoint_path"], outputs=["features_path"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("embed", self.embed, method_key="embed")

    def embed(self, tokens_path, checkpoint_path) -> Dict[str, Any]:
        p = self._parameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        torch.manual_seed(int(p["seed"]))
        model = RankEncoder(**checkpoint["config"])
        if not bool(p["random_init"]):
            model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        tokens = torch.load(tokens_path)["tokens"].long()
        tokens = torch.where(tokens < 0, torch.full_like(tokens, model.pad_id), tokens)
        parts, batch = [], int(p["batch_size"])
        with torch.no_grad():
            for start in range(0, len(tokens), batch):
                parts.append(model.embed(trim(tokens[start:start + batch], model.pad_id).to(device)).cpu())
        out = Path(p["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, torch.cat(parts).numpy())
        return {"features_path": str(out)}
