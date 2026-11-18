"""Self-supervised pretraining loop.

Plain PyTorch with explicit AMP, gradient clipping and cosine decay. Kept
dependency-light on purpose: the distributed strategy is a configuration
concern (``execution/``), and wrapping the loop in Fabric or FSDP is a
drop-in change at the two marked points rather than a rewrite.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..core.types import CellBatch
from ..data.datamodule import SingleCellDataModule
from ..models import build_model
from ..utils.device import resolve_device

log = logging.getLogger(__name__)

__all__ = ["pretrain", "embed_split"]


def _lr_at(step: int, total: int, base: float, warmup: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def embed_split(model, dm: SingleCellDataModule, split: str, device) -> Dict[str, np.ndarray]:
    """Frozen embeddings plus aligned labels for the evaluation layer."""
    model.eval()
    Z, y, b, lib = [], [], [], []
    obs = dm.obs.iloc[dm.splits[split]].reset_index(drop=True)
    for batch in dm.loader(split, shuffle=False):
        batch = batch.to(device)
        Z.append(model.encode(batch).float().cpu().numpy())
        y.append(batch.targets["cell_type"].cpu().numpy())
        if "batch_id" in batch.targets:
            b.append(batch.targets["batch_id"].cpu().numpy())
        idx = batch.meta["index"]
        lib.append(obs["library_size"].to_numpy()[idx])
    return {
        "Z": np.concatenate(Z),
        "y": np.concatenate(y),
        "batch": np.concatenate(b) if b else None,
        "library_size": np.concatenate(lib),
        "obs": obs,
    }


def pretrain(cfg: Dict[str, Any], dm: Optional[SingleCellDataModule] = None):
    """Train and return ``(model, datamodule, history)``."""
    train_cfg = cfg["train"]
    device = resolve_device(cfg.get("execution", {}).get("device", "auto"))
    log.info("device: %s", device)

    if dm is None:
        dm = SingleCellDataModule(
            source_cfg=cfg["data"]["source"],
            splitter_cfg=cfg["data"]["splitter"],
            tokenizer_cfg=cfg["data"]["tokenizer"],
            **{k: v for k, v in cfg["data"].items()
               if k not in ("source", "splitter", "tokenizer")},
        ).setup()

    model = build_model(
        cfg["model"],
        n_genes=dm.vocab_size,
        n_value_bins=dm.n_value_bins,
        n_classes=max(dm.n_classes, 2),
        n_batches=max(dm.n_batches, 2),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
    )
    epochs = int(train_cfg["epochs"])
    loader = dm.loader("train")
    total_steps = max(1, epochs * len(loader))
    warmup = int(train_cfg.get("warmup_steps", max(1, total_steps // 20)))
    clip = float(train_cfg.get("grad_clip", 1.0))
    base_lr = float(train_cfg["lr"])
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = int(train_cfg.get("accumulate_grad_batches", 1))

    history: list[dict] = []
    step = 0
    best_val = float("inf")
    best_state: Optional[dict] = None
    patience = int(train_cfg.get("early_stop_patience", 0))
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        dm.datasets["train"].set_epoch(epoch)   # fresh masks each epoch
        running: Dict[str, float] = {}
        n_seen = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            lr = _lr_at(step, total_steps, base_lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch)  # <- Fabric/FSDP wraps here
                if out.loss is None:
                    raise RuntimeError(
                        "no head produced a loss; check model.heads and that "
                        "the data module supplies the targets they need."
                    )
                loss = out.loss / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                if clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)  # <- and here
                scaler.update()
                opt.zero_grad(set_to_none=True)
            step += 1
            bs = batch.batch_size
            n_seen += bs
            running["loss"] = running.get("loss", 0.0) + float(out.loss.detach()) * bs
            for k, v in out.losses.items():
                running[k] = running.get(k, 0.0) + float(v) * bs
            for k, v in out.metrics.items():
                running[k] = running.get(k, 0.0) + float(v) * bs

        tr = {k: v / max(n_seen, 1) for k, v in running.items()}
        va = _validate(model, dm, device)
        rec = {"epoch": epoch, "lr": lr, **{f"train/{k}": v for k, v in tr.items()},
               **{f"val/{k}": v for k, v in va.items()}}
        history.append(rec)
        log.info(
            "epoch %2d | train loss %.4f | val loss %.4f | %s | %.1fs",
            epoch, tr.get("loss", float("nan")), va.get("loss", float("nan")),
            " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in tr.items()
                     if "accuracy" in k or k.endswith("r2")),
            time.time() - t0,
        )

        if va.get("loss", float("inf")) < best_val - 1e-5:
            best_val = va["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                log.info("early stop at epoch %d (no val improvement in %d)", epoch, patience)
                break

    if best_state is not None and train_cfg.get("restore_best", True):
        model.load_state_dict(best_state)
        log.info("restored best checkpoint (val loss %.4f)", best_val)

    out_dir = Path(cfg.get("output_dir", "outputs/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_cfg": cfg["model"],
            "vocab": dm.vocab.to_dict(),
            "label_maps": dm.label_maps,
            "val_loss": best_val,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    log.info("saved checkpoint to %s", out_dir / "checkpoint.pt")
    return model, dm, history


@torch.no_grad()
def _validate(model, dm, device) -> Dict[str, float]:
    model.eval()
    acc: Dict[str, float] = {}
    n = 0
    for batch in dm.loader("val", shuffle=False):
        batch = batch.to(device)
        out = model(batch)
        if out.loss is None:
            continue
        bs = batch.batch_size
        n += bs
        acc["loss"] = acc.get("loss", 0.0) + float(out.loss.detach()) * bs
        for k, v in out.metrics.items():
            acc[k] = acc.get(k, 0.0) + float(v) * bs
    return {k: v / max(n, 1) for k, v in acc.items()}
