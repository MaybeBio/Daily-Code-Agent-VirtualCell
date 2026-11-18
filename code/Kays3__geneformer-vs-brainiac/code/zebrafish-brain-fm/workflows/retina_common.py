"""Parts shared by the retina workflows."""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))

from mtx_atlas_loader import MtxAtlasLoaderNode  # noqa: E402
from rank_tokenizer import RankTokenizerNode  # noqa: E402
from sample_split import SampleSplitNode  # noqa: E402


def retina_inputs(n_cells):
    """Loader, split and tokenizer of the GSE239410 nuclei. The sample and the split always use seed 0."""
    raw, derived = ROOT / "data" / "raw" / "GSE239410", ROOT / "data" / "derived"
    stem = "GSE239410_muti_omic_injury_snRNAseq_mat"
    loader, split, tokenizer = MtxAtlasLoaderNode("Loader"), SampleSplitNode("Split"), RankTokenizerNode("Tokenizer")
    loader.configure(mtx_path=str(raw / f"{stem}.mtx.gz"), genes_path=str(raw / f"{stem}_genes.txt.gz"),
                     cells_path=str(raw / f"{stem}_cells.txt.gz"), cache_path=str(derived / "GSE239410_rna_full.h5ad"),
                     n_cells=n_cells, seed=0, out_path=str(derived / f"GSE239410_rna_{n_cells}.h5ad"))
    split.configure(holdout_fraction=0.2, seed=0, out_path=str(derived / f"GSE239410_rna_{n_cells}_split.h5ad"))
    tokenizer.configure(out_path=str(derived / f"GSE239410_rna_{n_cells}_tokens.pt"))
    return loader, split, tokenizer


def paired_bootstrap(truth, first, second, draws=1000):
    """Macro F1 of `first` minus macro F1 of `second`, resampling held-out cells."""
    classes = np.unique(truth)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(draws):
        pick = rng.integers(0, len(truth), len(truth))
        diffs.append(f1_score(truth[pick], first[pick], labels=classes, average="macro", zero_division=0)
                     - f1_score(truth[pick], second[pick], labels=classes, average="macro", zero_division=0))
    return {"difference": round(float(np.mean(diffs)), 4),
            "interval_95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            "method": f"paired bootstrap over held-out cells, {draws:,} draws"}
