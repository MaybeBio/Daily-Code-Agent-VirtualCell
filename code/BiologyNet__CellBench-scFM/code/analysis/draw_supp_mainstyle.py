"""Redraw supplementary figures in the main-figure style.

On-figure text is limited to panel labels, axis names, and method names.
Explanatory sentences stay in the caption file, not on the canvas.
Exports a vector PDF (fonttype 42), a 600 dpi PNG, and a same-size PPT.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Emu

ROOT = Path(r"E:\CellBench\article\CellBench")
OUT = ROOT / "figures" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    "CellFM", "CellPLM", "GeneCompass", "Geneformer", "LangCell", "Nicheformer",
    "Tabula", "TranscriptFormer", "UCE", "scFoundation", "scGPT", "scLong", "scPRINT2",
    "PCA_model", "UMAP", "scVI",
]
SHORT = {
    "CellFM": "CellFM", "CellPLM": "CellPLM", "GeneCompass": "GeneComp.",
    "Geneformer": "Geneform.", "LangCell": "LangCell", "Nicheformer": "Niche.",
    "Tabula": "Tabula", "TranscriptFormer": "TransF.", "UCE": "UCE",
    "scFoundation": "scFound.", "scGPT": "scGPT", "scLong": "scLong",
    "scPRINT2": "scP2", "PCA_model": "PCA", "UMAP": "UMAP", "scVI": "scVI",
}
COLORS = {
    "CellFM": "#332288", "CellPLM": "#F28E2B", "GeneCompass": "#882255",
    "Geneformer": "#CCAA44", "LangCell": "#009988", "Nicheformer": "#AA77BB",
    "Tabula": "#999933", "TranscriptFormer": "#8C6D31", "UCE": "#4477AA",
    "scFoundation": "#AA3377", "scGPT": "#EE6677", "scLong": "#78A2C5",
    "scPRINT2": "#44AA99", "PCA_model": "#59A14F", "UMAP": "#79706E",
    "scVI": "#66CCEE",
}
SEEDS = [41, 42, 43]
PANELS = ["organ", "disease", "species"]
DATASETS = [
    ["bmmc", "hlca", "heart", "kidney"],
    ["ms", "uc_epi", "covid_lung", "tnbc"],
    ["mouse_marrow", "mouse_lung", "mouse_heart", "mouse_kidney"],
]
DS = sum(DATASETS, [])
DS_NAMES = ["BMMC", "HLCA", "Heart", "Kidney", "MS", "UC", "COVID", "TNBC",
            "Marrow", "Lung", "Heart", "Kidney"]
ROW_NAMES = ["Human organs", "Human diseases", "Mouse organs"]
TASKS = ["clustering", "correction", "annotation", "reconstruction"]
METRICS = {
    "clustering": [("ari", "ARI", True), ("nmi", "NMI", True), ("asw", "ASW", True), ("acc", "Accuracy", True)],
    "correction": [("iLISI_scaled", "iLISI", True), ("kBET", "kBET", True), ("ARI", "ARI", True), ("ASW_celltype", "ASW", True)],
    "annotation": [("f1_m", "Macro-F1", True), ("acc", "Accuracy", True), ("balanced_accuracy", "Balanced acc.", True), ("precision_macro", "Macro-precision", True)],
    "reconstruction": [("Pearson_expr", "Pearson r", True), ("Spearman_expr", "Spearman", True), ("MAE_expr", "MAE", False), ("TopK_expression_overlap", "Top-50", True)],
}
INK = "#1F2A33"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Arial"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.7,
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "savefig.dpi": 600,
})


def readcsv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_raw():
    raw = {}
    for task in TASKS:
        array = np.empty((12, 3, 16, 4))
        for pi, panel in enumerate(PANELS):
            rows = readcsv(ROOT / "data" / "exports" / panel / f"{task}_scores.csv")
            lookup = {(r["dataset"], int(r["seed"]), r["model"]): r for r in rows}
            for di, dataset in enumerate(DATASETS[pi]):
                for si, seed in enumerate(SEEDS):
                    for mi, model in enumerate(MODELS):
                        row = lookup[dataset, seed, model]
                        array[pi * 4 + di, si, mi] = [float(row[key]) for key, _, _ in METRICS[task]]
        raw[task] = array
    return raw


def closed(ax):
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(INK)
        side.set_linewidth(0.7)


def boxes(ax, values, high):
    drawn = ax.boxplot(
        [values[:, i] for i in range(values.shape[1])],
        positions=np.arange(16), widths=0.62, whis=(0, 100), showfliers=False,
        patch_artist=True, manage_ticks=False,
        boxprops={"linewidth": 0.5, "edgecolor": INK},
        medianprops={"color": INK, "linewidth": 0.8},
        whiskerprops={"color": INK, "linewidth": 0.5},
        capprops={"color": INK, "linewidth": 0.5},
    )
    for i, box in enumerate(drawn["boxes"]):
        box.set_facecolor(COLORS[MODELS[i]])
        box.set_alpha(0.85)
    ax.set_xlim(-0.7, 15.7)
    if not high:
        lo, hi = values.min(), values.max()
        ax.set_yscale("log")
        ax.set_ylim(lo / 1.15, hi * 1.15)
        ax.yaxis.set_major_locator(LogLocator(numticks=3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"1e{int(round(np.log10(v)))}"))
        ax.yaxis.set_minor_locator(plt.NullLocator())
    else:
        lo, hi = values.min(), values.max()
        pad = max(0.03, (hi - lo) * 0.08)
        ax.set_ylim(min(0, lo - pad), max(hi + pad, 0.1) if hi <= 1.2 else hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    closed(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="#E6EBEF", lw=0.4)
    ax.tick_params(length=2, pad=1)


def method_ticks(ax, show):
    ax.set_xticks(range(16))
    if show:
        ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=70, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", labelsize=6, length=0, pad=1)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)


def save_all(fig, slug, caption):
    folder = OUT / slug
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{slug}.pdf", facecolor="white")
    fig.savefig(folder / f"{slug}.png", dpi=600, facecolor="white")
    with Image.open(folder / f"{slug}.png") as image:
        image.convert("RGB").save(folder / f"{slug}.png", dpi=(600, 600))
    width, height = fig.get_size_inches()
    deck = Presentation()
    deck.slide_width = Inches(width)
    deck.slide_height = Inches(height)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.shapes.add_picture(str(folder / f"{slug}.png"), Emu(0), Emu(0), Inches(width), Inches(height))
    deck.save(folder / f"{slug}.pptx")
    (folder / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    flat_pdf = OUT / f"{slug}.pdf"
    flat_ppt = OUT / f"{slug}.pptx"
    flat_pdf.write_bytes((folder / f"{slug}.pdf").read_bytes())
    flat_ppt.write_bytes((folder / f"{slug}.pptx").read_bytes())
    plt.close(fig)
    print("wrote", slug, flush=True)


def draw_metric_grids(raw):
    for ti, task in enumerate(TASKS):
        for ki, (metric, label, high) in enumerate(METRICS[task]):
            number = ti * 4 + ki + 1
            slug = f"s{number:02d}_{task}_{metric}"
            fig, axes = plt.subplots(3, 4, figsize=(7.4, 8.4), sharey="row")
            fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.96, wspace=0.18, hspace=0.28)
            values = raw[task][:, :, :, ki]
            for di, ax in enumerate(axes.flat):
                boxes(ax, values[di], high)
                method_ticks(ax, show=di >= 8)
                ax.set_title(DS_NAMES[di], loc="center", fontsize=8, pad=2, color=INK, fontweight="regular")
                if di % 4:
                    ax.tick_params(axis="y", labelleft=False)
                else:
                    ax.set_ylabel(label)
                closed(ax)
            for row, name in enumerate(ROW_NAMES):
                pos = axes[row, 0].get_position()
                fig.text(0.012, (pos.y0 + pos.y1) / 2, name, rotation=90, ha="center", va="center", fontsize=8)
            caption = (
                f"{label} across 12 cohorts. Rows are human organs, human diseases, and mouse organs. "
                "Each box is seeds 41, 42, and 43. Boxes are quartiles, the line is the median, and whiskers are the observed minimum and maximum."
            )
            save_all(fig, slug, caption)


def draw_audits(raw):
    # S17 seed range of mean metric rank
    ranked = {}
    for task in TASKS:
        ranked[task] = np.empty_like(raw[task])
        for di, ki in np.ndindex(12, 4):
            for si in range(3):
                high = METRICS[task][ki][2]
                order = np.argsort(np.argsort(-raw[task][di, si, :, ki] if high else raw[task][di, si, :, ki]))
                ranked[task][di, si, :, ki] = order + 1
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.96, wspace=0.28, hspace=0.42)
    for ti, (ax, task) in enumerate(zip(axes.flat, TASKS)):
        span = ranked[task].mean(axis=3).max(axis=1) - ranked[task].mean(axis=3).min(axis=1)
        im = ax.imshow(span.T, cmap="Blues", vmin=0, vmax=8, aspect="auto")
        ax.set_xticks(range(12), DS_NAMES, rotation=70, ha="right")
        ax.set_yticks(range(16), [SHORT[m] for m in MODELS] if ti % 2 == 0 else [])
        ax.set_title("abcd"[ti], loc="left", fontsize=9, fontweight="bold", pad=2)
        closed(ax)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    save_all(fig, "s17_seed_stability", "Seed range of the mean within-dataset metric rank, seeds 41-43.")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.12, top=0.96, wspace=0.35, hspace=0.4)
    for ti, (ax, task) in enumerate(zip(axes.flat, TASKS)):
        labels = [item[1] for item in METRICS[task]]
        matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                rhos = []
                for di in range(12):
                    for si in range(3):
                        a = ranked[task][di, si, :, i]
                        b = ranked[task][di, si, :, j]
                        rhos.append(np.corrcoef(a, b)[0, 1])
                matrix[i, j] = np.mean(rhos)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(4), labels, rotation=40, ha="right")
        ax.set_yticks(range(4), labels)
        ax.set_title("abcd"[ti], loc="left", fontsize=9, fontweight="bold")
        closed(ax)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    save_all(fig, "s18_metric_agreement", "Spearman agreement of method rankings between primary metrics.")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.6))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.94, wspace=0.2)
    for pi, ax in enumerate(axes):
        keep = [i for i in range(pi * 4, pi * 4 + 4) if i != (2 if pi == 0 else 4)]
        full = ranked["annotation"][pi * 4:pi * 4 + 4].mean(axis=(0, 1, 3))
        excl = ranked["annotation"][keep].mean(axis=(0, 1, 3))
        show = np.c_[full, excl]
        ax.imshow(show, cmap="Blues_r", vmin=1, vmax=16, aspect="auto")
        ax.set_xticks([0, 1], ["Full", "Excluded"])
        ax.set_yticks(range(16), [SHORT[m] for m in MODELS] if pi == 0 else [])
        for mi in range(16):
            for j, value in enumerate(show[mi]):
                ax.text(j, mi, f"{value:.1f}", ha="center", va="center", fontsize=6, color=INK)
        ax.set_title("ab"[pi], loc="left", fontsize=9, fontweight="bold")
        closed(ax)
    save_all(fig, "s19_overlap_sensitivity", "Annotation ranks before and after excluding heart or MS.")

    sizes = {"CellFM": 100, "GeneCompass": 101.76842, "Geneformer": 30, "LangCell": 27.5,
             "Nicheformer": 110, "TranscriptFormer": 57, "UCE": 36, "scFoundation": 50,
             "scGPT": 33, "scLong": 48.024242}
    params = {"CellFM": 797.4, "CellPLM": 65.8, "GeneCompass": 18.1, "Geneformer": 39.6,
              "LangCell": 39.4, "Nicheformer": 38.5, "Tabula": 13.2, "TranscriptFormer": 429.9,
              "UCE": 848.9, "scFoundation": 119.3, "scGPT": 50.8, "scLong": 1006.1, "scPRINT2": 101.9}
    for slug, info, xlabel in (
        ("s20_parameters", params, "Params (M)"),
        ("s21_corpus", sizes, "Corpus (M)"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.96, wspace=0.28, hspace=0.38)
        for ti, (ax, task) in enumerate(zip(axes.flat, TASKS)):
            ids = [i for i, model in enumerate(MODELS[:13]) if model in info]
            x = np.array([info[MODELS[i]] for i in ids])
            y = ranked[task].mean(axis=(0, 1, 3))[ids]
            ax.scatter(x, y, s=18, c=[COLORS[MODELS[i]] for i in ids], zorder=3)
            ax.set_xscale("log")
            ax.set_ylim(16.5, 0.5)
            ax.set_ylabel("Mean rank")
            ax.set_xlabel(xlabel)
            ax.set_title("abcd"[ti], loc="left", fontsize=9, fontweight="bold")
            closed(ax)
        save_all(fig, slug, f"{xlabel} against mean task rank for pretrained checkpoints with a resolved value.")


def draw_perturbation():
    rows = readcsv(ROOT / "data" / "exports" / "perturbation" / "perturbation_scores.csv")
    screens = ["norman", "adamson", "replogleK562", "replogleRPE1"]
    names = ["Norman", "Adamson", "K562", "RPE1"]
    specs = [
        ("s22_perturbation_Pearson_delta", "Pearson_delta", "Pearson $\\Delta$", True),
        ("s23_perturbation_Spearman_delta", "Spearman_delta", "Spearman $\\Delta$", True),
        ("s24_perturbation_MAE_delta", "MAE_delta", "MAE $\\Delta$", False),
        ("s25_perturbation_DES", "DES", "DES", True),
    ]
    lookup = {(r["dataset"], int(r["seed"]), r["model"]): r for r in rows}
    for slug, key, label, high in specs:
        fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.6), sharey=True)
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.34, top=0.88, wspace=0.12)
        for di, (screen, name, ax) in enumerate(zip(screens, names, axes)):
            values = np.array([
                [float(lookup[screen, seed, model][key]) for model in MODELS]
                for seed in SEEDS
            ])
            boxes(ax, values, high)
            method_ticks(ax, show=True)
            ax.set_title(name, fontsize=8, pad=2)
            if di == 0:
                ax.set_ylabel(label)
            else:
                ax.tick_params(axis="y", labelleft=False)
            closed(ax)
        save_all(fig, slug, f"{label} on four Perturb-seq screens. Each box is three seeds.")


if __name__ == "__main__":
    raw = load_raw()
    draw_metric_grids(raw)
    draw_audits(raw)
    draw_perturbation()
