"""Redraw supplemental 3x4 metric grids (S1-S16) and 4x4 perturbation grid (S22)
with clean, non-overlapping vertical 90-deg x-axis method labels and balanced spacing.

Preserves all underlying values, box/whisker statistics, per-panel y-scales, and palette.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    "CellFM", "CellPLM", "GeneCompass", "Geneformer", "LangCell", "Nicheformer",
    "Tabula", "TranscriptFormer", "UCE", "scFoundation", "scGPT", "scLong",
    "scPRINT2", "PCA_model", "UMAP", "scVI",
]
DISPLAY = [
    "CellFM", "CellPLM", "GeneCompass", "Geneformer", "LangCell", "Nicheformer",
    "Tabula", "TranscriptFormer", "UCE", "scFoundation", "scGPT", "scLong",
    "scPRINT2", "PCA", "UMAP", "scVI",
]
PALETTE = [
    "#1F77B4", "#AEC7E8", "#FF7F0E", "#2CA02C", "#98DF8A", "#D62728",
    "#9467BD", "#C5B0D5", "#8C564B", "#E377C2", "#F7B6D2", "#7F7F7F",
    "#BCBD22", "#DBDB8D", "#17BECF", "#9EDAE5",
]
SEEDS = [41, 42, 43]
PANELS = ["organ", "disease", "species"]
DATASETS = [
    ["bmmc", "hlca", "heart", "kidney"],
    ["ms", "uc_epi", "covid_lung", "tnbc"],
    ["mouse_marrow", "mouse_lung", "mouse_heart", "mouse_kidney"],
]
DS_NAMES = ["BM", "HL", "AH", "AK", "MS", "UC", "C19", "BC", "MM", "ML", "MH", "MK"]
ROW_NAMES = ["Human organs", "Human diseases", "Mouse organs"]
TASKS = ["clustering", "correction", "annotation", "reconstruction"]
METRICS = {
    "clustering": [
        ("ari", "ARI"),
        ("nmi", "NMI"),
        ("asw", "ASW"),
        ("acc", "Accuracy"),
    ],
    "correction": [
        ("iLISI_scaled", "Scaled iLISI"),
        ("kBET", "kBET"),
        ("ARI", "ARI"),
        ("ASW_celltype", "Cell-type ASW"),
    ],
    "annotation": [
        ("f1_m", "Macro-F1"),
        ("acc", "Accuracy"),
        ("balanced_accuracy", "Balanced acc."),
        ("precision_macro", "Macro-precision"),
    ],
    "reconstruction": [
        ("Pearson_expr", "Pearson $r$"),
        ("Spearman_expr", "Spearman"),
        ("MAE_expr", "MAE"),
        ("TopK_expression_overlap", "Top-50 overlap"),
    ],
}
INK = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 7.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.75,
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def readcsv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_static():
    raw = {}
    for task in TASKS:
        arr = np.empty((12, 3, 16, 4))
        for pi, panel in enumerate(PANELS):
            rows = readcsv(ROOT / "data" / "exports" / panel / f"{task}_scores.csv")
            lookup = {(r["dataset"], int(r["seed"]), r["model"]): r for r in rows}
            for di, ds in enumerate(DATASETS[pi]):
                for si, seed in enumerate(SEEDS):
                    for mi, m in enumerate(MODELS):
                        row = lookup[ds, seed, m]
                        arr[pi * 4 + di, si, mi] = [float(row[k]) for k, _ in METRICS[task]]
        raw[task] = arr
    return raw


def draw_boxes(ax, values):
    bp = ax.boxplot(
        [values[:, i] for i in range(16)],
        positions=np.arange(16),
        widths=0.60,
        whis=(0, 100),
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        boxprops={"linewidth": 0.55, "edgecolor": INK},
        medianprops={"color": INK, "linewidth": 0.9},
        whiskerprops={"color": INK, "linewidth": 0.55},
        capprops={"color": INK, "linewidth": 0.55},
    )
    for box, color in zip(bp["boxes"], PALETTE):
        box.set_facecolor(color)
        box.set_alpha(0.92)
    ax.set_xlim(-0.75, 15.75)
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-6)
    pad = span * 0.08
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="#E5E9EC", lw=0.45)
    ax.tick_params(axis="y", length=2.5, pad=2, labelsize=7.0)


def apply_model_ticks(ax, show: bool):
    ax.set_xticks(range(16))
    if show:
        ax.set_xticklabels(DISPLAY, rotation=90, ha="center", va="top")
        ax.tick_params(axis="x", labelsize=6.2, length=2.2, width=0.55, pad=2.5)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)


def draw_metric_grids(raw):
    for ti, task in enumerate(TASKS):
        for ki, (key, label) in enumerate(METRICS[task]):
            num = ti * 4 + ki + 1
            slug = f"s{num:02d}_{task}_{key}"
            fig, axes = plt.subplots(3, 4, figsize=(8.2, 8.6))
            fig.subplots_adjust(left=0.085, right=0.988, bottom=0.125, top=0.96, wspace=0.25, hspace=0.25)
            vals = raw[task][:, :, :, ki]
            for di, ax in enumerate(axes.flat):
                draw_boxes(ax, vals[di])
                apply_model_ticks(ax, show=(di >= 8))
                ax.set_title(DS_NAMES[di], loc="center", fontsize=9.0, pad=4, color=INK)
                if di % 4 == 0:
                    ax.set_ylabel(label, fontsize=8.0)
            for row, rname in enumerate(ROW_NAMES):
                pos = axes[row, 0].get_position()
                fig.text(0.015, (pos.y0 + pos.y1) / 2, rname, rotation=90, ha="center", va="center", fontsize=8.5, color=INK)
            out_pdf = OUT / f"{slug}.pdf"
            fig.savefig(out_pdf, facecolor="white")
            plt.close(fig)
            print("wrote", out_pdf.name, flush=True)


def draw_perturbation():
    rows = readcsv(ROOT / "data" / "exports" / "perturbation" / "perturbation_scores.csv")
    screens = ["norman", "adamson", "replogleK562", "replogleRPE1"]
    names = ["Norman", "Adamson", "K562", "RPE1"]
    specs = [
        ("Pearson_delta", "Pearson $\\Delta$"),
        ("Spearman_delta", "Spearman $\\Delta$"),
        ("MAE_delta", "MAE $\\Delta$"),
        ("DES", "DES"),
    ]
    lookup = {(r["dataset"], int(r["seed"]), r["model"]): r for r in rows}
    fig, axes = plt.subplots(4, 4, figsize=(8.2, 9.7))
    fig.subplots_adjust(left=0.085, right=0.988, bottom=0.11, top=0.96, wspace=0.28, hspace=0.28)
    for ri, (key, label) in enumerate(specs):
        for ci, (screen, sname) in enumerate(zip(screens, names)):
            ax = axes[ri, ci]
            vals = np.array([
                [float(lookup[screen, seed, m][key]) for m in MODELS]
                for seed in SEEDS
            ])
            draw_boxes(ax, vals)
            apply_model_ticks(ax, show=(ri == 3))
            ax.set_title(sname, loc="center", fontsize=9.0, pad=4, color=INK)
            if ci == 0:
                ax.set_ylabel(label, fontsize=8.0)
                ax.text(-0.24, 1.09, "abcd"[ri], transform=ax.transAxes, fontsize=11.0, fontweight="bold", va="bottom", ha="left", color=INK)
    out_pdf = OUT / "s22_perturbation_four_metrics.pdf"
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    print("wrote", out_pdf.name, flush=True)


if __name__ == "__main__":
    raw = load_static()
    draw_metric_grids(raw)
    draw_perturbation()
