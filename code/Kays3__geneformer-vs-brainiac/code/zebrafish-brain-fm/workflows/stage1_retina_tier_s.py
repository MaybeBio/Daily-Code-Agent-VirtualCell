"""Stage 1 on real retina cells, tier S, as a Neuro-Workflow graph.

    Loader -> Split -> Tokenizer -> Train -> Embed -> Probe               the model
    Train -> Embed (untrained weights) -> Probe                          control: what training adds
    Embed -> Probe with permuted training labels                         control: must fail
    Split -> PCA -> Probe                                                the baseline

Data: GSE239410, single-nucleus RNA of the zebrafish retina (Lyu et al. 2023).
The benchmarks are fixed in STAGE1-RETINA.md. Run from the zebrafish-brain-fm folder:

    uv run python workflows/stage1_retina_tier_s.py --tiny
    uv run python workflows/stage1_retina_tier_s.py --steps 3000 --ceiling 45 --seed 0
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))

from neuroworkflow import WorkflowBuilder  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cell_type_probe import CellTypeProbeNode  # noqa: E402
from embed_cells import EmbedCellsNode  # noqa: E402
from encoder_train import EncoderTrainNode  # noqa: E402
from pca_features import PcaFeaturesNode  # noqa: E402
from retina_common import paired_bootstrap, retina_inputs  # noqa: E402

TITLE = "Stage 1: tier S model on real retina nuclei"
PROBES = ("ProbeModel", "ProbeUntrained", "ProbePCA", "ProbePermuted")


def option(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def build(argv=()):
    """Build the graph without running it."""
    argv = list(argv)
    tiny = "--tiny" in argv
    seed = int(option(argv, "--seed", 0))
    n_cells = 1000 if tiny else int(option(argv, "--cells", 10000))
    name = "stage1_retina_tiny" if tiny else f"stage1_retina_tier_s_seed{seed}"
    run = ROOT / "runs" / name

    loader, split, tokenizer = retina_inputs(n_cells)
    train, pca = EncoderTrainNode("Train"), PcaFeaturesNode("PCA")
    embed, embed_untrained = EmbedCellsNode("Embed"), EmbedCellsNode("EmbedUntrained")
    probes = {k: CellTypeProbeNode(k) for k in PROBES}
    train.configure(steps=50 if tiny else int(option(argv, "--steps", 3000)),
                    ceiling_minutes=float(option(argv, "--ceiling", 45)), seed=seed, out_dir=str(run))
    pca.configure(n_components=128, out_path=str(run / "pca_features.npy"))
    embed.configure(out_path=str(run / "embeddings.npy"))
    embed_untrained.configure(random_init=True, seed=seed, out_path=str(run / "embeddings_untrained.npy"))
    for key, probe in probes.items():
        probe.configure(out_path=str(run / f"{key}_predictions.npy"), permute_labels=(key == "ProbePermuted"))

    builder = WorkflowBuilder(f"zfbrain_{name}", context={"species": "zebrafish", "tissue": "retina", "accession": "GSE239410"})
    for node in (loader, split, tokenizer, train, pca, embed, embed_untrained, *probes.values()):
        builder.add_node(node)
    builder.connect("Loader", "data_path", "Split", "data_path")
    builder.connect("Split", "data_path", "Tokenizer", "data_path")
    builder.connect("Split", "data_path", "PCA", "data_path")
    builder.connect("Tokenizer", "tokens_path", "Train", "data_path")
    for node in ("Embed", "EmbedUntrained"):
        builder.connect("Tokenizer", "tokens_path", node, "tokens_path")
        builder.connect("Train", "checkpoint_path", node, "checkpoint_path")
    for features, probe in (("Embed", "ProbeModel"), ("EmbedUntrained", "ProbeUntrained"), ("PCA", "ProbePCA"), ("Embed", "ProbePermuted")):
        builder.connect(features, "features_path", probe, "features_path")
        builder.connect("Tokenizer", "tokens_path", probe, "tokens_path")
    return builder, {"split": split, "tokenizer": tokenizer, "train": train, "probes": probes,
                     "name": name, "seed": seed, "n_cells": n_cells, "run": run}


def main():
    builder, n = build(sys.argv)
    success = builder.build().execute()
    if not success:
        sys.exit("The workflow failed: a node raised an error. See the messages above.")
    data = torch.load(n["tokenizer"].get_output("tokens_path"))
    truth = data["cell_type"].numpy()[data["is_heldout"].numpy()]
    classes = np.unique(truth)
    model_pred, pca_pred = (np.load(n["probes"][k].get_output("predictions_path")) for k in ("ProbeModel", "ProbePCA"))
    results = {
        "workflow_success": bool(success), "name": n["name"], "seed": n["seed"], "cells": n["n_cells"],
        "heldout_samples": n["split"].get_output("heldout_samples"), "heldout_cells": n["split"].get_output("n_heldout_cells"),
        "vocab_size": n["tokenizer"].get_output("vocab_size"),
        "median_genes_per_cell": n["tokenizer"].get_output("median_genes_per_cell"),
        "cell_types": data["cell_type_names"],
        "heldout_cells_per_type": {data["cell_type_names"][c]: int((truth == c).sum()) for c in classes},
        "training": n["train"].get_output("metrics"),
        "probes": {k: probe.get_output("metrics") for k, probe in n["probes"].items()},
        "model_minus_pca_macro_f1": paired_bootstrap(truth, model_pred, pca_pred),
    }
    n["run"].mkdir(parents=True, exist_ok=True)
    (n["run"] / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n--- stage 1, retina, tier S ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
