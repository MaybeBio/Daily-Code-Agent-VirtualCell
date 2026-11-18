"""Stage 1b: published models on the stage 1 retina test, as a Neuro-Workflow graph.

    Loader -> Split -> OrthologMap -> Geneformer (frozen) -> Probe
                                   -> scGPT (frozen) -> Probe          only with --with-scgpt
                                   -> PCA of mapped counts -> Probe    control for the gene mapping
    Split -> PCA of all genes -> Probe                                 the stage 1 baseline, repeated

The benchmarks are fixed in STAGE1B-EXTERNAL-MODELS.md. Each published model runs in
its own environment under external/. Run from the zebrafish-brain-fm folder:

    uv run python workflows/stage1b_retina_external.py --tiny
    uv run python workflows/stage1b_retina_external.py
    uv run python workflows/stage1b_retina_external.py --with-scgpt
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
from external_embed import ExternalEmbedNode  # noqa: E402
from ortholog_map import OrthologMapNode  # noqa: E402
from pca_features import PcaFeaturesNode  # noqa: E402
from retina_common import paired_bootstrap, retina_inputs  # noqa: E402

TITLE = "Stage 1b: Geneformer and scGPT on the same retina test"


def option(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def build(argv=()):
    """Build the graph without running it."""
    argv = list(argv)
    tiny = "--tiny" in argv
    n_cells = 1000 if tiny else int(option(argv, "--cells", 10000))
    models = ["Geneformer"] + (["scGPT"] if "--with-scgpt" in argv else [])
    name = "stage1b_retina_tiny" if tiny else "stage1b_retina_external"
    run = ROOT / "runs" / name

    loader, split, tokenizer = retina_inputs(n_cells)
    orthologs = OrthologMapNode("OrthologMap")
    orthologs.configure(table_path=str(ROOT / "data" / "raw" / "ensembl_orthologs" / "zebrafish_human_orthologs.tsv"),
                        out_path=str(ROOT / "data" / "derived" / f"GSE239410_rna_{n_cells}_human_genes.h5ad"))
    pca_all, pca_mapped = PcaFeaturesNode("PCAAllGenes"), PcaFeaturesNode("PCAMappedGenes")
    pca_all.configure(n_components=128, out_path=str(run / "pca_all_genes.npy"))
    pca_mapped.configure(n_components=128, out_path=str(run / "pca_mapped_genes.npy"))
    embedders = {m: ExternalEmbedNode(m) for m in models}
    for m, node in embedders.items():
        node.configure(project_dir=str(ROOT / "external" / m.lower()), out_path=str(run / f"{m.lower()}_features.npy"))
    probes = {k: CellTypeProbeNode(f"Probe{k}") for k in ("PCAAllGenes", "PCAMappedGenes", *models)}
    for k, probe in probes.items():
        probe.configure(out_path=str(run / f"Probe{k}_predictions.npy"))

    builder = WorkflowBuilder(f"zfbrain_{name}", context={"species": "zebrafish", "tissue": "retina", "accession": "GSE239410"})
    for node in (loader, split, tokenizer, orthologs, pca_all, pca_mapped, *embedders.values(), *probes.values()):
        builder.add_node(node)
    builder.connect("Loader", "data_path", "Split", "data_path")
    builder.connect("Split", "data_path", "Tokenizer", "data_path")
    builder.connect("Split", "data_path", "PCAAllGenes", "data_path")
    builder.connect("Split", "data_path", "OrthologMap", "data_path")
    builder.connect("OrthologMap", "data_path", "PCAMappedGenes", "data_path")
    for m in models:
        builder.connect("OrthologMap", "data_path", m, "data_path")
    for k in probes:
        builder.connect(k, "features_path", f"Probe{k}", "features_path")
        builder.connect("Tokenizer", "tokens_path", f"Probe{k}", "tokens_path")
    return builder, {"split": split, "tokenizer": tokenizer, "orthologs": orthologs, "embedders": embedders,
                     "probes": probes, "models": models, "name": name, "n_cells": n_cells, "run": run}


def main():
    builder, n = build(sys.argv)
    success = builder.build().execute()
    if not success:
        sys.exit("The workflow failed: a node raised an error. See the messages above.")
    data = torch.load(n["tokenizer"].get_output("tokens_path"))
    truth = data["cell_type"].numpy()[data["is_heldout"].numpy()]
    pred = {k: np.load(probe.get_output("predictions_path")) for k, probe in n["probes"].items()}
    results = {
        "workflow_success": bool(success), "name": n["name"], "cells": n["n_cells"],
        "heldout_samples": n["split"].get_output("heldout_samples"), "heldout_cells": n["split"].get_output("n_heldout_cells"),
        "gene_mapping": n["orthologs"].get_output("mapping"),
        "feature_width": {m: node.get_output("n_features") for m, node in n["embedders"].items()},
        "probes": {k: probe.get_output("metrics") for k, probe in n["probes"].items()},
        "minus_pca_mapped_genes_macro_f1": {m: paired_bootstrap(truth, pred[m], pred["PCAMappedGenes"]) for m in n["models"]},
        "minus_pca_all_genes_macro_f1": {m: paired_bootstrap(truth, pred[m], pred["PCAAllGenes"]) for m in n["models"]},
    }
    n["run"].mkdir(parents=True, exist_ok=True)
    (n["run"] / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n--- stage 1b, retina, published models ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
