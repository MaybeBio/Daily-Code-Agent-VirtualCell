#!/usr/bin/env python3
"""Reproduce direct frozen-embedding linear-probe diagnostics (data/direct_embedding_program_probe.csv and Table S27).

Evaluates 1,000-cell subsamples (RandomState(42)) across four representative cohorts
(hlca, ms, covid_lung, and cross-assay 10x+Smart-seq2 mouse_heart) and all 16 benchmark methods:
- Continuous cellular programs via 5-fold donor-grouped Ridge regression (alpha=10.0, solver='cholesky',
  within-fold StandardScaler) using pooled out-of-fold R^2:
  1. Library depth: log10(total_counts + 1)
  2. Cell-cycle proliferation markers: mean log1p(CP10K) over ['MKI67', 'TOP2A', 'PCNA', 'MCM2', 'CDK1', 'CCNB1', 'TYMS']
  3. Interferon / innate-immune markers: mean log1p(CP10K) over ['ISG15', 'IFIT1', 'IFIT3', 'MX1', 'STAT1', 'OAS1', 'IRF7', 'B2M', 'HLA-A', 'H2-K1']
  4. Ribosomal transcript fraction: sum of RPL*/RPS* counts divided by total_counts
  5. Mitochondrial transcript fraction: sum of MT-* counts divided by total_counts (defined on hlca and covid_lung)
- Rare (<1% whole-cohort frequency) vs abundant (>=5% whole-cohort frequency) cell-subtype classification
  via 5-fold cell-stratified RidgeClassifier (alpha=10.0, solver='cholesky', random_state=42, within-fold StandardScaler).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import pickle
import warnings

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_ROOT = os.environ.get(
    "CELLBENCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

MODELS = [
    "scFoundation",
    "TranscriptFormer",
    "UCE",
    "scGPT",
    "Nicheformer",
    "LangCell",
    "Geneformer",
    "scPRINT2",
    "CellPLM",
    "PCA_model",
    "scVI",
    "CellFM",
    "GeneCompass",
    "Tabula",
    "scLong",
    "UMAP",
]

COHORTS = [
    ("hlca", "hlca"),
    ("ms", "filtered_ms_adata"),
    ("covid_lung", "covid_lung"),
    ("mouse_heart", "mouse_heart"),
]

CYCLE_MARKERS = ["MKI67", "TOP2A", "PCNA", "MCM2", "CDK1", "CCNB1", "TYMS"]
IFN_MARKERS = ["ISG15", "IFIT1", "IFIT3", "MX1", "STAT1", "OAS1", "IRF7", "B2M", "HLA-A", "H2-K1"]


def read_h5_col(group: h5py.Group, key: str) -> np.ndarray:
    item = group[key]
    if isinstance(item, h5py.Group):
        codes = item["codes"][:]
        cats = (
            item["categories"].asstr()[:]
            if item["categories"].dtype.kind in "OSU"
            else item["categories"][:]
        )
        return np.array([cats[i] if i >= 0 else "" for i in codes]).astype(str)
    return item.asstr()[:] if item.dtype.kind in "OSU" else item[:]


def main() -> None:
    out_rows: list[dict[str, object]] = []
    manifest_meta: list[dict[str, object]] = []

    for cohort, h5_stem in COHORTS:
        h5_path = os.path.join(BENCHMARK_ROOT, "datasets", f"{h5_stem}.h5ad")
        with h5py.File(h5_path, "r") as f:
            obs = f["obs"]
            ids = read_h5_col(obs, "_index")
            labels = read_h5_col(obs, "celltype")
            donors = read_h5_col(obs, "donor_id")
            n_total = len(ids)
            ix = np.sort(np.random.RandomState(42).choice(n_total, min(1000, n_total), replace=False))

            var = f["var"]
            gene_key = next((k for k in ["gene_name", "gene_symbols", "feature_name"] if k in var), "_index")
            genes = read_h5_col(var, gene_key)
            if cohort == "mouse_heart":
                ref_path = os.path.join(BENCHMARK_ROOT, "datasets", "mouse_heart_10x.h5ad")
                with h5py.File(ref_path, "r") as ref:
                    rvar = ref["var"]
                    mapping = dict(zip(read_h5_col(rvar, "_index"), read_h5_col(rvar, "gene_name")))
                    genes = np.array([mapping.get(g, g) for g in genes])

            x_group = f["X"]
            ptr = x_group["indptr"][:]
            ind = x_group["indices"][:]
            data = x_group["data"][:]
            X = csr_matrix((data, ind, ptr), shape=tuple(x_group.attrs["shape"]))[ix].astype(float)

            total = np.asarray(X.sum(1)).ravel()
            gu = np.char.upper(genes.astype(str))
            cc_mask = np.isin(gu, CYCLE_MARKERS)
            inf_mask = np.isin(gu, IFN_MARKERS)
            ribo_mask = np.array([s.startswith(("RPL", "RPS")) for s in gu])
            mito_mask = np.char.startswith(gu, "MT-")

            norm = X.multiply((1e4 / np.maximum(total, 1))[:, None]).tocsr()
            norm.data = np.log1p(norm.data)

            Y = np.column_stack(
                [
                    np.log10(total + 1.0),
                    np.asarray(norm[:, cc_mask].mean(1)).ravel(),
                    np.asarray(norm[:, inf_mask].mean(1)).ravel(),
                    np.asarray(X[:, ribo_mask].sum(1)).ravel() / np.maximum(total, 1.0),
                    np.asarray(X[:, mito_mask].sum(1)).ravel() / np.maximum(total, 1.0),
                ]
            )

            yy = labels[ix]
            dd = donors[ix]
            n_groups = len(np.unique(dd))
            reg_folds = (
                list(GroupKFold(5).split(ix, yy, dd))
                if n_groups >= 5
                else list(StratifiedKFold(5, shuffle=True, random_state=42).split(ix, yy))
            )
            clf_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(ix, yy))

            counts = dict(zip(*np.unique(labels, return_counts=True)))
            classes = np.unique(yy)
            rare_classes = [c for c in classes if counts[c] / n_total < 0.01]
            abundant_classes = [c for c in classes if counts[c] / n_total >= 0.05]

            manifest_meta.append(
                {
                    "cohort": cohort,
                    "n_total": int(n_total),
                    "n_sample": int(len(ix)),
                    "sampled_donors": int(n_groups),
                    "marker_counts": {
                        "cycle": int(cc_mask.sum()),
                        "ifn": int(inf_mask.sum()),
                        "ribo": int(ribo_mask.sum()),
                        "mito": int(mito_mask.sum()),
                    },
                    "rare_classes": list(rare_classes),
                    "abundant_classes": list(abundant_classes),
                }
            )

        for model in MODELS:
            pattern = os.path.join(
                BENCHMARK_ROOT,
                "output",
                "full_benchmark",
                "embeddings",
                f"{cohort}_{model}",
                "**",
                "all.pkl",
            )
            paths = glob.glob(pattern, recursive=True)
            if not paths:
                raise FileNotFoundError(f"Missing cached all.pkl for {cohort}_{model}")
            latest_pkl = max(paths, key=os.path.getmtime)
            with open(latest_pkl, "rb") as pf:
                z = pickle.load(pf)
            E = np.asarray(z["cell_embs"])
            ci = np.asarray(z["cell_ids"])
            if ci.dtype.kind in "iu":
                lookup = {int(c): j for j, c in enumerate(ci)}
                rows = np.array([lookup[int(i)] for i in ix])
            else:
                lookup = {str(c): j for j, c in enumerate(ci)}
                rows = np.array([lookup[str(ids[i])] for i in ix])

            A = E[rows].astype(np.float64)
            pred = np.zeros_like(Y)
            yp = np.empty(len(ix), dtype=object)

            for tr, te in reg_folds:
                scaler = StandardScaler()
                a_tr = scaler.fit_transform(A[tr])
                a_te = scaler.transform(A[te])
                reg = Ridge(alpha=10.0, solver="cholesky").fit(a_tr, Y[tr])
                pred[te] = reg.predict(a_te)

            for tr, te in clf_folds:
                scaler = StandardScaler()
                a_tr = scaler.fit_transform(A[tr])
                a_te = scaler.transform(A[te])
                clf = RidgeClassifier(alpha=10.0, solver="cholesky").fit(a_tr, yy[tr])
                yp[te] = clf.predict(a_te)

            f1_vals = f1_score(yy, yp, labels=classes, average=None, zero_division=0)
            f1_map = dict(zip(classes, f1_vals))
            r2_vals = r2_score(Y, pred, multioutput="raw_values", force_finite=False)

            rare_list = [float(f1_map[c]) for c in rare_classes]
            abund_list = [float(f1_map[c]) for c in abundant_classes]

            out_rows.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "log_total_counts_r2": float(r2_vals[0]),
                    "cell_cycle_r2": float(r2_vals[1]),
                    "ifn_r2": float(r2_vals[2]),
                    "ribo_r2": float(r2_vals[3]),
                    "mito_r2": float(r2_vals[4]) if not np.isnan(r2_vals[4]) else "NaN",
                    "rare_f1": json.dumps(rare_list, separators=(",", ":")),
                    "abundant_f1": json.dumps(abund_list, separators=(",", ":")),
                    "rare_f1_mean": round(float(np.mean(rare_list)), 4) if rare_list else "NaN",
                    "abundant_f1_mean": round(float(np.mean(abund_list)), 4) if abund_list else "NaN",
                }
            )

    out_csv = os.path.join(ROOT, "data", "direct_embedding_program_probe.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    manifest_path = os.path.join(ROOT, "data", "direct_embedding_program_probe_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_meta, fh, indent=2)

    print(f"Verified and wrote {len(out_rows)} rows to {out_csv} and manifest to {manifest_path}")


if __name__ == "__main__":
    main()
