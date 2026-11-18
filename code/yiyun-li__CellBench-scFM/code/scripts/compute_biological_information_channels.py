#!/usr/bin/env python3
"""Compute reproducible biological-information channel metrics across 12 cohorts and 16 models.

Quantifies four biological-information channels for Model Axis 1 (C1):
1. Rare & minority subtype discriminability: Macro-F1, Balanced Accuracy, Macro-AUPRC vs Overall Accuracy (and gain over PCA).
2. Qualitative marker-gene set identity vs quantitative expression amplitude & cross-cell gene co-variation:
   TopK_expression_overlap vs MAE_expr and Pearson_gene / Spearman_gene.
3. Unsupervised dominant lineage topology: unsupervised Leiden ARI (uARI), ASW_celltype, cLISI_scaled, Graph_connectivity.
4. Native technical/donor mixing vs biological conservation: iLISI_scaled, kBET, Batch_ASW.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_task(task: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = os.path.join(ROOT, "data", "exports", "*", f"{task}_scores.csv")
    for path in sorted(glob.glob(pattern)):
        panel = path.split(os.sep)[-2]
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["panel"] = panel
                rows.append(row)
    return rows


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def main() -> None:
    ann = load_task("annotation")
    clu = load_task("clustering")
    cor = load_task("correction")
    rec = load_task("reconstruction")

    def group_means(rows: list[dict[str, str]], cols: list[str]) -> dict[tuple[str, str, str], dict[str, float]]:
        acc_map: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            key = (r["panel"], r["dataset"], r["model"])
            for c in cols:
                val = r.get(c, "")
                if val != "":
                    acc_map[key][c].append(float(val))
        return {k: {c: statistics.mean(v) for c, v in d.items()} for k, d in acc_map.items()}

    ann_m = group_means(ann, ["acc", "f1_m", "balanced_accuracy", "AUPRC_macro", "AUROC_macro", "n_configured_classes"])
    clu_m = group_means(clu, ["ari", "nmi", "asw", "acc", "AvgCC"])
    cor_m = group_means(cor, ["ARI", "NMI", "ASW_celltype", "Batch_ASW", "iLISI_scaled", "cLISI_scaled", "Graph_connectivity", "kBET"])
    rec_m = group_means(rec, ["Pearson_expr", "Spearman_expr", "Pearson_gene", "Spearman_gene", "MAE_expr", "TopK_expression_overlap"])

    datasets = sorted({(k[0], k[1]) for k in ann_m})
    models = sorted({k[2] for k in ann_m})

    summary_rows = []
    raw_by_model: dict[str, dict[str, float]] = {}
    for m in models:
        acc_12 = statistics.mean(ann_m[(p, d, m)]["acc"] for p, d in datasets)
        f1_12 = statistics.mean(ann_m[(p, d, m)]["f1_m"] for p, d in datasets)
        bal_12 = statistics.mean(ann_m[(p, d, m)]["balanced_accuracy"] for p, d in datasets)
        auprc_12 = statistics.mean(ann_m[(p, d, m)]["AUPRC_macro"] for p, d in datasets)
        d_acc_pca = statistics.mean(ann_m[(p, d, m)]["acc"] - ann_m[(p, d, "PCA_model")]["acc"] for p, d in datasets)
        d_f1_pca = statistics.mean(ann_m[(p, d, m)]["f1_m"] - ann_m[(p, d, "PCA_model")]["f1_m"] for p, d in datasets)
        d_bal_pca = statistics.mean(
            ann_m[(p, d, m)]["balanced_accuracy"] - ann_m[(p, d, "PCA_model")]["balanced_accuracy"] for p, d in datasets
        )
        d_auprc_pca = statistics.mean(
            ann_m[(p, d, m)]["AUPRC_macro"] - ann_m[(p, d, "PCA_model")]["AUPRC_macro"] for p, d in datasets
        )
        rare_vs_abundant_shift = d_f1_pca - d_acc_pca

        uari_12 = statistics.mean(clu_m[(p, d, m)]["ari"] for p, d in datasets)
        asw_ct_12 = statistics.mean(cor_m[(p, d, m)]["ASW_celltype"] for p, d in datasets)
        clisi_12 = statistics.mean(cor_m[(p, d, m)]["cLISI_scaled"] for p, d in datasets)
        grcon_12 = statistics.mean(cor_m[(p, d, m)]["Graph_connectivity"] for p, d in datasets)
        ilisi_12 = statistics.mean(cor_m[(p, d, m)]["iLISI_scaled"] for p, d in datasets)
        kbet_12 = statistics.mean(cor_m[(p, d, m)]["kBET"] for p, d in datasets)
        batch_asw_12 = statistics.mean(cor_m[(p, d, m)]["Batch_ASW"] for p, d in datasets)

        p_expr_12 = statistics.mean(rec_m[(p, d, m)]["Pearson_expr"] for p, d in datasets)
        p_gene_12 = statistics.mean(rec_m[(p, d, m)]["Pearson_gene"] for p, d in datasets)
        s_gene_12 = statistics.mean(rec_m[(p, d, m)]["Spearman_gene"] for p, d in datasets)
        topk_12 = statistics.mean(rec_m[(p, d, m)]["TopK_expression_overlap"] for p, d in datasets)
        mae_12 = statistics.mean(rec_m[(p, d, m)]["MAE_expr"] for p, d in datasets)

        raw_by_model[m] = {
            "macro_f1_12ds": f1_12,
            "pearson_gene_12ds": p_gene_12,
            "topk_marker_overlap_12ds": topk_12,
            "neg_mae_expr_12ds": -mae_12,
            "mae_expr_12ds": mae_12,
            "unsupervised_ari_12ds": uari_12,
            "asw_celltype_12ds": asw_ct_12,
            "clisi_scaled_12ds": clisi_12,
            "ilisi_scaled_12ds": ilisi_12,
        }

        f1_wins_vs_pca = sum(1 for p, d in datasets if ann_m[(p, d, m)]["f1_m"] > ann_m[(p, d, "PCA_model")]["f1_m"])
        auprc_wins_vs_pca = sum(
            1 for p, d in datasets if ann_m[(p, d, m)]["AUPRC_macro"] > ann_m[(p, d, "PCA_model")]["AUPRC_macro"]
        )
        topk_wins_vs_pca = sum(
            1 for p, d in datasets if rec_m[(p, d, m)]["TopK_expression_overlap"] > rec_m[(p, d, "PCA_model")]["TopK_expression_overlap"]
        )
        pgene_wins_vs_pca = sum(
            1 for p, d in datasets if rec_m[(p, d, m)]["Pearson_gene"] > rec_m[(p, d, "PCA_model")]["Pearson_gene"]
        )

        summary_rows.append(
            {
                "model": m,
                "acc_12ds": round(acc_12, 4),
                "macro_f1_12ds": round(f1_12, 4),
                "balanced_acc_12ds": round(bal_12, 4),
                "auprc_macro_12ds": round(auprc_12, 4),
                "delta_acc_vs_pca": round(d_acc_pca, 4),
                "delta_macro_f1_vs_pca": round(d_f1_pca, 4),
                "delta_balanced_acc_vs_pca": round(d_bal_pca, 4),
                "delta_auprc_vs_pca": round(d_auprc_pca, 4),
                "rare_subtype_excess_gain_f1_minus_acc": round(rare_vs_abundant_shift, 4),
                "unsupervised_ari_12ds": round(uari_12, 4),
                "asw_celltype_12ds": round(asw_ct_12, 4),
                "clisi_scaled_12ds": round(clisi_12, 4),
                "graph_connectivity_12ds": round(grcon_12, 4),
                "ilisi_scaled_12ds": round(ilisi_12, 4),
                "kbet_12ds": round(kbet_12, 4),
                "batch_asw_12ds": round(batch_asw_12, 4),
                "pearson_expr_12ds": round(p_expr_12, 4),
                "pearson_gene_12ds": round(p_gene_12, 4),
                "spearman_gene_12ds": round(s_gene_12, 4),
                "topk_marker_overlap_12ds": round(topk_12, 4),
                "mae_expr_12ds": round(mae_12, 4),
                "macro_f1_dataset_wins_vs_pca_of_12": f1_wins_vs_pca,
                "auprc_dataset_wins_vs_pca_of_12": auprc_wins_vs_pca,
                "topk_marker_wins_vs_pca_of_12": topk_wins_vs_pca,
                "pearson_gene_wins_vs_pca_of_12": pgene_wins_vs_pca,
            }
        )

    summary_rows.sort(key=lambda r: -raw_by_model[r["model"]]["macro_f1_12ds"])

    out_csv = os.path.join(ROOT, "data", "biological_information_channels.csv")
    s26_csv = os.path.join(ROOT, "data", "supplementary_tables", "S26_biological_information_channels.csv")
    for path in (out_csv, s26_csv):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Cross-model Spearman rank correlations among biological channels using unrounded 12-cohort means
    scfm_models_13 = [r["model"] for r in summary_rows if r["model"] not in {"PCA_model", "UMAP", "scVI"}]
    scfm_models_10 = [m for m in scfm_models_13 if m not in {"Tabula", "CellFM", "scLong"}]
    diagnostics = {
        "n_models_total": len(summary_rows),
        "n_scfms": len(scfm_models_13),
        "n_scfms_non_collapsed": len(scfm_models_10),
        "spearman_13scfms": {
            "macro_f1_vs_unsupervised_ari": round(
                spearman_rho([raw_by_model[m]["macro_f1_12ds"] for m in scfm_models_13], [raw_by_model[m]["unsupervised_ari_12ds"] for m in scfm_models_13]), 4
            ),
            "macro_f1_vs_pearson_gene": round(
                spearman_rho([raw_by_model[m]["macro_f1_12ds"] for m in scfm_models_13], [raw_by_model[m]["pearson_gene_12ds"] for m in scfm_models_13]), 4
            ),
            "macro_f1_vs_ilisi": round(
                spearman_rho([raw_by_model[m]["macro_f1_12ds"] for m in scfm_models_13], [raw_by_model[m]["ilisi_scaled_12ds"] for m in scfm_models_13]), 4
            ),
            "asw_celltype_vs_ilisi": round(
                spearman_rho([raw_by_model[m]["asw_celltype_12ds"] for m in scfm_models_13], [raw_by_model[m]["ilisi_scaled_12ds"] for m in scfm_models_13]), 4
            ),
            "pearson_gene_vs_neg_mae_expr": round(
                spearman_rho([raw_by_model[m]["pearson_gene_12ds"] for m in scfm_models_13], [raw_by_model[m]["neg_mae_expr_12ds"] for m in scfm_models_13]), 4
            ),
            "topk_marker_vs_neg_mae_expr": round(
                spearman_rho([raw_by_model[m]["topk_marker_overlap_12ds"] for m in scfm_models_13], [raw_by_model[m]["neg_mae_expr_12ds"] for m in scfm_models_13]), 4
            ),
        },
        "dataset_level_cross_assay_and_disease_highlights": {
            ds: {
                m: {
                    "macro_f1": round(ann_m[(p, ds, m)]["f1_m"], 4),
                    "auprc_macro": round(ann_m[(p, ds, m)]["AUPRC_macro"], 4),
                    "unsupervised_ari": round(clu_m[(p, ds, m)]["ari"], 4),
                    "asw_celltype": round(cor_m[(p, ds, m)]["ASW_celltype"], 4),
                    "ilisi_scaled": round(cor_m[(p, ds, m)]["iLISI_scaled"], 4),
                    "pearson_gene": round(rec_m[(p, ds, m)]["Pearson_gene"], 4),
                    "topk_marker_overlap": round(rec_m[(p, ds, m)]["TopK_expression_overlap"], 4),
                    "mae_expr": round(rec_m[(p, ds, m)]["MAE_expr"], 4),
                }
                for m in ["scFoundation", "TranscriptFormer", "UCE", "scGPT", "Nicheformer", "LangCell", "Geneformer", "scPRINT2", "PCA_model", "scVI"]
            }
            for p, ds in datasets
        },
    }
    out_json = os.path.join(ROOT, "data", "biological_information_diagnostics.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)

    print("Wrote:", out_csv)
    print("Wrote:", s26_csv)
    print("Wrote:", out_json)
    print("Spearman (13 scFMs):", json.dumps(diagnostics["spearman_13scfms"], indent=2))


if __name__ == "__main__":
    main()
