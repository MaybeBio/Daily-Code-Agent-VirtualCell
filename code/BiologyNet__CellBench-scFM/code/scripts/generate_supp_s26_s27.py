#!/usr/bin/env python3
"""Generate LaTeX Supplementary Tables S26 and S27 from exported CSV files."""

from __future__ import annotations

import csv
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt_signed(x: float) -> str:
    return f"{x:.4f}" if x >= 0 else f"$-${abs(x):.4f}"


def main() -> None:
    with open(os.path.join(ROOT, "data", "biological_information_channels.csv"), encoding="utf-8") as fh:
        ch = list(csv.DictReader(fh))
    with open(os.path.join(ROOT, "data", "direct_embedding_program_probe.csv"), encoding="utf-8") as fh:
        pr = list(csv.DictReader(fh))

    name_map = {"PCA_model": "PCA"}
    lines: list[str] = []

    lines.append(r"\suppneedspace{.45\textheight}")
    lines.append(r"\begingroup\footnotesize\setlength{\tabcolsep}{3.4pt}\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{longtable}{>{\raggedright\arraybackslash}p{2.35cm}rrrrrrrrrr}")
    lines.append(
        r"\caption{Twelve-cohort biological-information channel profiles across all 16 evaluated methods "
        r"(averaged equally over three random seeds per cohort across human organs, human diseases and mouse organs; "
        r"576 runs per representation task; four-decimal channel summaries archived in \texttt{data/biological\_information\_channels.csv} "
        r"and full-precision run-level scores in \texttt{data/exports/}). Columns group supervised cell-subtype identity (Accuracy, Macro-F1, Macro-AUPRC, "
        r"and Macro-F1 difference relative to PCA), unsupervised lineage topology (Leiden ARI and cell-type ASW), "
        r"native donor/batch mixing (scaled iLISI), and supervised gene-expression reconstruction "
        r"(across-cell gene Pearson $r_{\mathrm{gene}}$, Top-50 expressed-gene overlap, and expression MAE).}\label{tab:supp_bio_channels}\\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"Method & Acc & Mac-F1 & AUPRC & $\Delta$F1$_{\mathrm{PCA}}$ & uARI & ASW$_{\mathrm{ct}}$ & iLISI & $r_{\mathrm{gene}}$ & Top-50 & MAE\\"
    )
    lines.append(r"\midrule\endfirsthead")
    lines.append(r"\multicolumn{11}{l}{\small\itshape Table \thetable\ (continued)}\\\toprule")
    lines.append(
        r"Method & Acc & Mac-F1 & AUPRC & $\Delta$F1$_{\mathrm{PCA}}$ & uARI & ASW$_{\mathrm{ct}}$ & iLISI & $r_{\mathrm{gene}}$ & Top-50 & MAE\\\midrule\endhead"
    )
    lines.append(r"\bottomrule\endfoot")

    for r in ch:
        m = name_map.get(r["model"], r["model"])
        df1 = float(r["delta_macro_f1_vs_pca"])
        df1_s = f"+{df1:.4f}" if df1 > 0 else (f"$-${abs(df1):.4f}" if df1 < 0 else "0.0000")
        lines.append(
            f"{m} & {float(r['acc_12ds']):.4f} & {float(r['macro_f1_12ds']):.4f} & {float(r['auprc_macro_12ds']):.4f} & {df1_s} & "
            f"{float(r['unsupervised_ari_12ds']):.4f} & {float(r['asw_celltype_12ds']):.4f} & {float(r['ilisi_scaled_12ds']):.4f} & "
            f"{float(r['pearson_gene_12ds']):.4f} & {float(r['topk_marker_overlap_12ds']):.4f} & {float(r['mae_expr_12ds']):.4f} \\\\"
        )
    lines.append(r"\end{longtable}\endgroup")
    lines.append("")

    lines.append(r"\suppneedspace{.45\textheight}")
    lines.append(r"\begingroup\footnotesize\setlength{\tabcolsep}{3.8pt}\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{longtable}{>{\raggedright\arraybackslash}p{2.5cm}rrrrrrr}")
    lines.append(
        r"\caption{Direct frozen-embedding linear-probe diagnostics across four representative cohorts "
        r"(\texttt{hlca}, \texttt{ms}, \texttt{covid\_lung} and cross-assay 10x+Smart-seq2 \texttt{mouse\_heart}; "
        r"1,000 cells per cohort, seed 42; full-precision outputs archived in \texttt{data/direct\_embedding\_program\_probe.csv} "
        r"and reproduced via \texttt{scripts/run\_direct\_embedding\_program\_probe.py}). Continuous cellular program columns report mean five-fold donor-grouped "
        r"Ridge regression ($\alpha=10$) out-of-fold $R^2$ for $\log_{10}(\text{UMI}+1)$ library depth, canonical cell-cycle "
        r"proliferation markers (\texttt{MKI67}, \texttt{TOP2A}, \texttt{PCNA}, \texttt{MCM2}, \texttt{CDK1}, \texttt{CCNB1}, \texttt{TYMS}; summarized across \texttt{hlca} and \texttt{covid\_lung}), "
        r"interferon/innate-immune response markers (\texttt{ISG15}, \texttt{IFIT1}, \texttt{IFIT3}, \texttt{MX1}, \texttt{STAT1}, \texttt{OAS1}, \texttt{IRF7}, \texttt{B2M}, \texttt{HLA-A}/\texttt{H2-K1}; all four cohorts), "
        r"ribosomal transcript fraction (\texttt{RPL*}/\texttt{RPS*}), and mitochondrial transcript fraction (\texttt{MT-*}, summarized across \texttt{hlca} and \texttt{covid\_lung}, "
        r"where mitochondrial genes are retained). Classification columns report five-fold cell-stratified Ridge-classifier F1 ($\alpha=10$) "
        r"averaged across all sampled rare cell types ($<1\%$ whole-cohort frequency; 11 classes: 4 in \texttt{hlca}, 1 in \texttt{ms}, 2 in \texttt{covid\_lung} and 4 in \texttt{mouse\_heart}, including classes with $<5$ sampled cells) versus abundant cell types "
        r"($\ge 5\%$ frequency; 26 classes: 10, 3, 8 and 5 per cohort). Probes were fitted with within-fold standardization on cached benchmark embeddings without refitting the underlying representations inside probe folds.}\label{tab:supp_direct_probe}\\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"Method & Depth $R^2$ & Cycle $R^2$ & IFN $R^2$ & Ribo $R^2$ & Mito $R^2$ & Rare F1 ($<1\%$) & Abund. F1 ($\ge 5\%$)\\"
    )
    lines.append(r"\midrule\endfirsthead")
    lines.append(r"\multicolumn{8}{l}{\small\itshape Table \thetable\ (continued)}\\\toprule")
    lines.append(
        r"Method & Depth $R^2$ & Cycle $R^2$ & IFN $R^2$ & Ribo $R^2$ & Mito $R^2$ & Rare F1 ($<1\%$) & Abund. F1 ($\ge 5\%$)\\\midrule\endhead"
    )
    lines.append(r"\bottomrule\endfoot")

    for r in ch:
        m_raw = r["model"]
        mr = [x for x in pr if x["model"] == m_raw]
        m = name_map.get(m_raw, m_raw)
        depth = statistics.mean(float(x["log_total_counts_r2"]) for x in mr)
        cyc = statistics.mean(float(x["cell_cycle_r2"]) for x in mr if x["cohort"] in ("hlca", "covid_lung"))
        ifn = statistics.mean(float(x["ifn_r2"]) for x in mr)
        ribo = statistics.mean(float(x["ribo_r2"]) for x in mr)
        mito = statistics.mean(float(x["mito_r2"]) for x in mr if x["mito_r2"] != "NaN")
        rare_all = [v for x in mr for v in json.loads(x["rare_f1"])]
        abun_all = [v for x in mr for v in json.loads(x["abundant_f1"])]
        lines.append(
            f"{m} & {fmt_signed(depth)} & {fmt_signed(cyc)} & {fmt_signed(ifn)} & {fmt_signed(ribo)} & {fmt_signed(mito)} & "
            f"{statistics.mean(rare_all):.4f} & {statistics.mean(abun_all):.4f} \\\\"
        )
    lines.append(r"\end{longtable}\endgroup")

    out_path = os.path.join(ROOT, "appendix", "supplementary_tables_s26_s27.tex")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
