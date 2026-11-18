"""Evaluation: the model against its controls, on the same split.

Deliberate design choice: baselines are not optional and cannot be disabled
from the experiment config. A run that produces a model number without the
baseline numbers beside it produces a result nobody can interpret.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from ..core.registry import BASELINES
from ..data.datamodule import SingleCellDataModule
from ..eval.metrics import evaluate_embedding
from ..eval.probes import knn_probe, linear_probe
from ..eval.report import ComparisonReport
from .pretrain import embed_split

log = logging.getLogger(__name__)

__all__ = ["evaluate"]

#: controls that always run
DEFAULT_BASELINES = [
    {"name": "pca", "n_components": 50},
    {"name": "scaled_pca", "n_components": 50},
    {"name": "hvg", "n_genes": 50},
    {"name": "random_projection", "n_components": 50},
]


def _probe_block(Ztr, ytr, Zte, yte, seed: int) -> Dict[str, float]:
    lp = linear_probe(Ztr, ytr, Zte, yte, seed=seed)
    kp = knn_probe(Ztr, ytr, Zte, yte, seed=seed)
    return {
        "probe_macro_f1": lp["macro_f1"],
        "probe_macro_f1_lo": lp.get("macro_f1_lo", float("nan")),
        "probe_macro_f1_hi": lp.get("macro_f1_hi", float("nan")),
        "probe_accuracy": lp["accuracy"],
        "knn_macro_f1": kp["macro_f1"],
        "knn_accuracy": kp["accuracy"],
    }


def evaluate(
    model,
    dm: SingleCellDataModule,
    cfg: Dict[str, Any],
    device,
    split: str = "test",
) -> ComparisonReport:
    seed = int(cfg.get("seed", 0))
    splitter_name = cfg["data"]["splitter"].get("name", "?")
    group_key = cfg["data"]["splitter"].get("group_key") or (
        {"donor_held_out": "donor_id", "study_held_out": "study_id",
         "tissue_held_out": "tissue"}.get(splitter_name)
    )

    tr = embed_split(model, dm, "train", device)
    te = embed_split(model, dm, split, device)

    n_groups = 0
    if group_key and group_key in te["obs"].columns:
        n_groups = int(te["obs"][group_key].nunique())

    rep = ComparisonReport(
        split_regime=splitter_name,
        group_key=group_key,
        is_synthetic=bool(getattr(dm, "is_synthetic", False)),
        provenance=getattr(dm, "provenance", "?"),
        n_train=int(tr["Z"].shape[0]),
        n_test=int(te["Z"].shape[0]),
        n_test_groups=n_groups,
        leakage=str(dm.leakage) if dm.leakage is not None else None,
    )

    # --- the model ---
    m = _probe_block(tr["Z"], tr["y"], te["Z"], te["y"], seed)
    m.update(
        evaluate_embedding(
            te["Z"], te["y"], te["batch"], te["library_size"], seed=seed
        )
    )
    rep.add("foundation_model", m)

    # --- the controls, on identical splits and identical probes ---
    Xtr, obs_tr = dm.dense_split("train")
    Xte, obs_te = dm.dense_split(split)
    ytr_raw = obs_tr["cell_type"].astype(str).to_numpy()
    yte_raw = obs_te["cell_type"].astype(str).to_numpy()
    cmap = dm.label_maps["cell_type"]
    ytr = np.array([cmap.get(v, -100) for v in ytr_raw])
    yte = np.array([cmap.get(v, -100) for v in yte_raw])
    btest = obs_te["batch_id"].astype(str).to_numpy()
    lib_te = obs_te["library_size"].to_numpy()

    for bcfg in cfg.get("eval", {}).get("baselines", DEFAULT_BASELINES):
        bcfg = dict(bcfg)
        name = bcfg["name"]
        try:
            bl = BASELINES.build(bcfg)
            bl.fit(Xtr, obs_tr)  # fitted on train only
            Ztr, Zte = bl.embed(Xtr), bl.embed(Xte)
            row = _probe_block(Ztr, ytr, Zte, yte, seed)
            row.update(evaluate_embedding(Zte, yte_raw, btest, lib_te, seed=seed))
            rep.add(f"baseline:{name}", row)
        except Exception as exc:
            log.warning("baseline '%s' failed: %s: %s", name, type(exc).__name__, exc)
            rep.notes.append(f"baseline '{name}' failed: {exc}")

    if rep.is_synthetic:
        rep.notes.append(
            "data is simulated; conclusions transfer only insofar as the "
            "simulator's assumptions hold."
        )
    if n_groups and n_groups < 3:
        rep.notes.append(
            f"only {n_groups} held-out {group_key}(s): treat the ranking as "
            "provisional, cell-level CIs do not capture between-donor variance."
        )
    return rep
