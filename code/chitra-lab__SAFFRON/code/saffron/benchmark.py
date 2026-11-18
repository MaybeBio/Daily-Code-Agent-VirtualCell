"""
SAFFRON benchmark: the paper's 8 spatial transcriptomics tasks, hosted at
https://huggingface.co/datasets/ChitraLab/saffron-benchmark, and the paper's metrics for scoring any representation.

    from saffron import benchmark
    adata = benchmark.load("colorectal")          # one of benchmark.TASKS
    benchmark.score(adata, H)                      # H: (cells x features), e.g. SAFFRON SAE features of your embeddings
    benchmark.score(adata, adata.obsm["X_pca"])    # the paper's PCA baseline (also: obsm["gene_sae_features"])

Global tasks (Fig. 2a) return one number: the mean over spatial domains of the maximum |Spearman rho| between any
feature and tau. Local tasks (Fig. 4a) return per-unit |Spearman rho| for each plaque-associated metric
(see `local_scores`).
"""
import json

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr

REPO_ID = "ChitraLab/saffron-benchmark"
GLOBAL_TASKS = ("colorectal", "kidney", "cortex_v1", "cortex_v2", "gi_ileum", "colon")
LOCAL_TASKS = ("starmap_ad", "merfish_ad")
TASKS = GLOBAL_TASKS + LOCAL_TASKS
_MASKS = {"tau_dam_score": "eval_dam", "tau_micro_density": "eval_density", "tau_distance_to_plaque": "eval_distance"}


def load(task, cache_dir=None, revision=None):
    """Download (once) and read one benchmark task as AnnData."""
    import anndata as ad
    from huggingface_hub import hf_hub_download

    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
    sub = "global" if task in GLOBAL_TASKS else "local"
    path = hf_hub_download(REPO_ID, f"{sub}/{task}.h5ad", repo_type="dataset", cache_dir=cache_dir, revision=revision)
    return ad.read_h5ad(path)


def info(adata):
    """Task metadata stored with the file (source, technology, tau definition, scoring protocol)."""
    return json.loads(adata.uns["saffron"])


def score(adata, H):
    """Score a cell-by-feature matrix H on a benchmark task with the paper's metric."""
    if np.asarray(H.shape)[0] != adata.n_obs:
        raise ValueError(f"H has {H.shape[0]} rows but the task has {adata.n_obs} cells")
    return global_score(adata, H) if info(adata)["task"] == "global" else local_scores(adata, H)


def _rows(H, idx=None):
    H = H[idx] if idx is not None else H
    return H.toarray() if sp.issparse(H) else np.asarray(H)


def _abs_spearman(Z, y):
    """|Spearman rho| (tie-averaged ranks) of every column of Z with y; constant columns give 0."""
    n = Z.shape[0]
    ry = rankdata(y).astype(np.float64) - (n + 1) / 2
    ry /= np.sqrt((ry ** 2).sum())
    Zr = np.apply_along_axis(rankdata, 0, Z).astype(np.float64) - (n + 1) / 2
    nrm = np.sqrt((Zr ** 2).sum(axis=0))
    nrm[nrm == 0] = 1.0
    return np.abs((Zr * ry[:, None]).sum(axis=0) / nrm)


def global_score(adata, H, min_cells=10, return_domains=False):
    """Fig. 2a: mean over spatial domains (>= min_cells scored cells) of max_feature |Spearman rho(feature, tau)|."""
    tau = adata.obs["tau"].to_numpy(dtype=float)
    dom = adata.obs["domain"].astype(object).to_numpy()
    ok = np.isfinite(tau) & np.array([d is not None and d == d for d in dom])
    per = {}
    for d in sorted(set(dom[ok]), key=str):
        idx = np.where(ok & (dom == d))[0]
        if len(idx) >= min_cells:
            per[d] = float(_abs_spearman(_rows(H, idx), tau[idx]).max())
    s = float(np.mean(list(per.values())))
    return (s, per) if return_domains else s


def local_scores(adata, H, min_cells=20):
    """Fig. 4a: per-unit |Spearman rho| for each plaque-associated metric.

    DAM score / microglial density: the feature with the highest |rho| over all evaluated cells pooled is selected,
    then its |rho| is reported within each sample. Distance to plaque: within each unit, the max over features."""
    obs = adata.obs
    out = {}
    for metric in ("tau_dam_score", "tau_micro_density"):
        y = obs[metric].to_numpy(dtype=float)
        mask = obs[_MASKS[metric]].to_numpy(dtype=bool)
        idx = np.where(mask)[0]
        best = int(np.argmax(_abs_spearman(_rows(H, idx), y[idx])))
        feat = np.asarray(H[:, best].todense()).ravel() if sp.issparse(H) else np.asarray(H)[:, best]
        unit = obs["sample"].astype(str).to_numpy()
        vals = []
        for u in sorted(set(unit[mask])):
            m = mask & (unit == u)
            if m.sum() >= min_cells:
                r = spearmanr(feat[m], y[m]).statistic
                vals.append(abs(float(r)) if np.isfinite(r) else 0.0)
        out[metric] = vals
    y = obs["tau_distance_to_plaque"].to_numpy(dtype=float)
    mask = obs["eval_distance"].to_numpy(dtype=bool)
    unit = obs["distance_unit"].astype(object).to_numpy()
    vals = []
    for u in sorted({x for x in unit[mask] if x is not None and x == x}, key=str):
        idx = np.where(mask & (unit == u))[0]
        if len(idx) >= min_cells:
            Z = _rows(H, idx)
            live = np.where(Z.std(axis=0) > 0)[0]
            rs = np.nan_to_num([spearmanr(y[idx], Z[:, k]).statistic for k in live])
            vals.append(float(np.abs(rs).max()))
    out["tau_distance_to_plaque"] = vals
    return out
