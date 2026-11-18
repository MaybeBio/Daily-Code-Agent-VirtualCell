"""Train a rank-token encoder by masked-gene prediction. The defaults are tier S."""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from rank_encoder import RankEncoder, trim

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class EncoderTrainNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_encoder_train",
        stage="training",  # proposed stage; not in the Neuro-Workflow stage list yet
        tool="PyTorch",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Train a small transformer encoder on ranked gene-token cells by masked-gene prediction, with a timed start check against a time ceiling, and report held-out loss beside a gene-frequency baseline.",
        parameters={
            "layers": ParameterDefinition(default_value=4, description="Transformer layers."),
            "width": ParameterDefinition(default_value=128, description="Hidden width."),
            "heads": ParameterDefinition(default_value=4, description="Attention heads."),
            "steps": ParameterDefinition(default_value=600, description="Training steps."),
            "batch_size": ParameterDefinition(default_value=32, description="Cells per step."),
            "mask_fraction": ParameterDefinition(default_value=0.15, description="Fraction of gene tokens masked."),
            "learning_rate": ParameterDefinition(default_value=1e-3, description="AdamW learning rate."),
            "holdout_fraction": ParameterDefinition(default_value=0.1, description="Fraction of cells held out, never trained on."),
            "seed": ParameterDefinition(default_value=0, description="Random seed."),
            "ceiling_minutes": ParameterDefinition(default_value=6.0, description="Time ceiling. The run does not start if the start check projects more, and stops itself at the ceiling."),
            "start_check_steps": ParameterDefinition(default_value=10, description="Timed steps used to project the total time."),
            "out_dir": ParameterDefinition(default_value="runs/tier_s_smoke", description="Folder for the checkpoint and metrics."),
        },
        inputs={
            "data_path": PortDefinition(type=PortType.STR, description="File with 'tokens' (cells x seq_len, -1 is padding) and 'vocab_size'; 'is_heldout' if the cells were split upstream."),
        },
        outputs={
            "metrics": PortDefinition(type=PortType.DICT, description="Device, weight count, steps done, timing, first and last training loss, held-out loss, baseline losses, status."),
            "checkpoint_path": PortDefinition(type=PortType.STR, description="Path of the saved model weights; empty if the run did not start."),
        },
        methods={
            "train": MethodDefinition(description="Run the start check, train, evaluate on held-out cells, save.", inputs=["data_path"], outputs=["metrics", "checkpoint_path"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("train", self.train, method_key="train")

    def train(self, data_path) -> Dict[str, Any]:
        p = self._parameters
        torch.manual_seed(int(p["seed"]))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data = torch.load(data_path)
        tokens, vocab = data["tokens"].long(), int(data["vocab_size"])
        config = {"vocab_size": vocab, "seq_len": tokens.shape[1], "width": int(p["width"]), "layers": int(p["layers"]), "heads": int(p["heads"])}
        model = RankEncoder(**config).to(device)
        tokens = torch.where(tokens < 0, torch.full_like(tokens, model.pad_id), tokens)
        if "is_heldout" in data:  # real cells: the split was made by sample, upstream
            is_held = data["is_heldout"].bool()
            held, train = tokens[is_held], tokens[~is_held]
            split = "by sample, from the token file"
        else:  # synthetic cells have no samples
            n_hold = max(1, int(len(tokens) * float(p["holdout_fraction"])))
            held, train = tokens[:n_hold], tokens[n_hold:]
            split = "first rows of the file"
        optim = torch.optim.AdamW(model.parameters(), lr=float(p["learning_rate"]))
        steps, batch, frac = int(p["steps"]), int(p["batch_size"]), float(p["mask_fraction"])
        ceiling_s = 60.0 * float(p["ceiling_minutes"])
        gen = torch.Generator().manual_seed(int(p["seed"]))

        def masked_loss(cells: torch.Tensor) -> torch.Tensor:
            cells = trim(cells, model.pad_id).to(device)
            is_gene = cells != model.pad_id
            mask = (torch.rand(cells.shape, generator=gen).to(device) < frac) & is_gene
            if not mask.any():
                mask = is_gene
            hidden = model.hidden(cells.masked_fill(mask, model.mask_id))
            return nn.functional.cross_entropy(model.gene_logits(hidden[mask]), cells[mask])

        def sync() -> None:
            if device == "cuda":
                torch.cuda.synchronize()

        gene_tokens = (train != model.pad_id).sum(dim=1)
        metrics: Dict[str, Any] = {
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "weights": sum(w.numel() for w in model.parameters()),
            "train_cells": len(train), "heldout_cells": len(held), "split": split,
            "vocab_size": vocab, "seq_len": tokens.shape[1],
            "median_tokens_per_train_cell": int(gene_tokens.median()),
            "steps_planned": steps, "batch_size": batch, "precision": "fp32", "seed": int(p["seed"]),
        }
        losses, check_n = [], int(p["start_check_steps"])
        start = time.perf_counter()
        for step in range(steps):
            cells = train[torch.randint(0, len(train), (batch,), generator=gen)]
            loss = masked_loss(cells)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            if step + 1 == check_n:
                sync()
                projected = (time.perf_counter() - start) / check_n * steps
                metrics["start_check_projected_s"] = round(projected, 1)
                print(f"start check: projected {projected:.0f} s against a ceiling of {ceiling_s:.0f} s", flush=True)
                if projected > ceiling_s:
                    metrics["status"] = "not started: start check over the ceiling"
                    return {"metrics": metrics, "checkpoint_path": ""}
            if time.perf_counter() - start > ceiling_s:
                metrics["status"] = "stopped at the ceiling"
                break
        else:
            metrics["status"] = "completed"
        sync()
        elapsed = time.perf_counter() - start
        done = len(losses)

        model.eval()
        with torch.no_grad():
            held_loss = sum(masked_loss(held[i:i + batch]).item() for i in range(0, len(held), batch)) / math.ceil(len(held) / batch)
        counts = torch.bincount(train[train != model.pad_id], minlength=vocab)[:vocab].float() + 1.0
        log_freq = (counts / counts.sum()).log()
        metrics.update({
            "steps_done": done, "train_seconds": round(elapsed, 1), "steps_per_second": round(done / elapsed, 2),
            "loss_first10": round(sum(losses[:10]) / 10, 3), "loss_last10": round(sum(losses[-10:]) / 10, 3),
            "heldout_loss": round(held_loss, 3),
            "baseline_uniform_loss": round(math.log(vocab), 3),
            "baseline_gene_frequency_loss": round(-log_freq[held[held != model.pad_id]].mean().item(), 3),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
            "losses_finite": all(math.isfinite(x) for x in losses),
        })
        out = Path(p["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config}, out / "model.pt")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return {"metrics": metrics, "checkpoint_path": str(out / "model.pt")}
