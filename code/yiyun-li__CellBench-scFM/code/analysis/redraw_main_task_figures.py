#!/usr/bin/env python3
"""Redraw main-text Figures 2-5 and Figure 7 (manuscript Figures 2-6) in Figure 1's visual style.

Synchronizes canvas width (183 mm), outer left/right alignment grid, typography hierarchy,
color palette, spacing, and angled x-axis method labels across all five benchmark figures:
  - figures/Figure2.pdf & .pptx  (Unsupervised clustering)
  - figures/Figure3.pdf & .pptx  (Native batch assessment)
  - figures/Figure4.pdf & .pptx  (Supervised cell-type annotation)
  - figures/Figure5.pdf & .pptx  (Supervised HVG expression reconstruction)
  - figures/Figure7.pdf & .pptx  (Genetic perturbation response prediction; manuscript Figure 6)

All numerical values and rankings are computed directly from read-only benchmark exports
and verified against data/all_panel_ranks.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image
import pymupdf
from lxml import etree
from pptx import Presentation
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR / "figure_qa"))
sys.path.insert(0, str(ANALYSIS_DIR / "vendor" / "scientific_visualization" / "scripts"))

import task_layout as t
from audit_panel_alignment import require_matplotlib_panel_alignment
from figure_export import export_figure

# ---------------------------------------------------------------------------
# Unified Canvas & Grid Geometry (mm) — strictly aligned with Figure 1 (183 mm)
# ---------------------------------------------------------------------------
W = 183.0
H = 148.8

LEFT = 3.5               # Outer left grid line (panel letters a, b, d; Method banner left)
RIGHT = 179.5            # Outer right grid line (Panel a right; panels c, e right spine)
CONTENT_W = RIGHT - LEFT # 176.0 mm

# Panel a (Composite ranking table) vertical coordinates (mm from bottom)
SEC_A_Y = 144.3          # 'a  Composite ranking' baseline
BANNER_Y = 138.3         # Colored group banners [138.3, 142.5]
BANNER_H = 4.2
SUBHDR_Y = 136.0         # 'Rank' and dataset subheaders center
TABLE_BOTTOM = 85.4      # 16 rows in [85.4, 134.2] -> 3.05 mm per row
TABLE_H = 48.8

# Panels b-e (2x2 metric boxplots) compact symmetric column & row coordinates (mm)
# Left slot:  letter at 3.5  -> 9.0 mm y-tick reserve -> box [12.5, 88.5] (76.0 mm)
# Gutter:     [88.5, 94.5] (6.0 mm)
# Right slot: letter at 94.5 -> 9.0 mm y-tick reserve -> box [103.5, 179.5] (76.0 mm)
LETTER_X = (LEFT, 94.5)
METRIC_X = (12.5, 103.5)
METRIC_W = 76.0
METRIC_H = 20.4
METRIC_Y = (59.2, 17.2)
METRIC_LABEL_Y = (80.7, 38.7)

# ---------------------------------------------------------------------------
# Figure 1 Harmonized Palette & Method Colors
# ---------------------------------------------------------------------------
INK = "#1E293B"          # Primary dark slate ink (matches Figure 1)
SUBINK = "#334155"       # Axis labels & secondary headers
MUTED = "#475569"        # Method banner & subtle text
SPINE = "#475569"        # Crisp slate subplot frame
GRID = "#E2E8F0"         # Hairline horizontal grid (matches Figure 1)
ROW_EVEN = "#F1F5F9"     # Soft slate zebra stripe
ROW_ODD = "#F8FAFC"      # Light architectural card tint
OVERALL_COL = "#3B6E8C"  # Overall rank steel blue
OVERALL_BAR = "#7FA6C2"

PANEL_COLORS_STATIC = ["#2B6CB0", "#B84A73", "#178573"]  # Human organs, Human diseases, Mouse organs
PANEL_COLORS_PERT = ["#2B6CB0", "#B84A73", "#178573", "#C07A1C"]  # Norman, Adamson, K562, RPE1

METHOD_COLORS = {
    "CellFM": "#332288",
    "CellPLM": "#F28E2B",
    "GeneCompass": "#882255",
    "Geneformer": "#CCAA44",
    "LangCell": "#009988",
    "Nicheformer": "#AA77BB",
    "Tabula": "#999933",
    "TranscriptFormer": "#8C6D31",
    "UCE": "#4477AA",
    "scFoundation": "#AA3377",
    "scGPT": "#EE6677",
    "scLong": "#78A2C5",
    "scPRINT2": "#44AA99",
    "PCA_model": "#59A14F",
    "UMAP": "#79706E",
    "scVI": "#66CCEE",
}

STATIC_SPECS = [
    dict(
        task="clustering",
        out_stem="Figure2",
        panel_a_title="Composite ranking across biological panels",
        plot_seq=[0, 1, 3, 2],
        metric_titles={"ari": "ARI", "nmi": "NMI", "acc": "Accuracy", "asw": "Scaled ASW"},
    ),
    dict(
        task="correction",
        out_stem="Figure3",
        panel_a_title="Composite ranking on unintegrated embeddings",
        plot_seq=[0, 1, 2, 3],
        metric_titles={
            "iLISI_scaled": "Scaled iLISI",
            "kBET": "kBET",
            "cLISI_scaled": "Scaled cLISI",
            "Graph_connectivity": "Graph connectivity",
        },
    ),
    dict(
        task="annotation",
        out_stem="Figure4",
        panel_a_title="Composite ranking across biological panels",
        plot_seq=[0, 1, 2, 3],
        metric_titles={
            "acc": "Accuracy",
            "f1_m": "Macro-F1",
            "balanced_accuracy": "Balanced accuracy",
            "precision_macro": "Macro-precision",
        },
    ),
    dict(
        task="reconstruction",
        out_stem="Figure5",
        panel_a_title="Composite ranking across biological panels",
        plot_seq=[3, 2, 0, 1],
        metric_titles={
            "TopK_expression_overlap": "Top-50 overlap",
            "MAE_expr": "MAE rank",
            "Pearson_expr": "Pearson r",
            "Spearman_expr": "Spearman \u03c1",
        },
    ),
]

PERT_Datasets = ["norman", "adamson", "replogleK562", "replogleRPE1"]
PERT_LABELS = ["Norman", "Adamson", "K562", "RPE1"]
PERT_METRICS = [
    ("Pearson_delta", "Pearson \u0394", True),
    ("Spearman_delta", "Spearman \u0394", True),
    ("MAE_delta", "MAE \u0394 rank", False),
    ("DES", "DES", True),
]


def configure_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.6,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.grid": False,
    })


def axmm(fig: plt.Figure, x: float, y: float, width: float, height: float) -> plt.Axes:
    return fig.add_axes([x / W, y / H, width / W, height / H])


def contrasting_ink(rgba: tuple[float, ...]) -> str:
    rgb = np.array(rgba[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
    ink_rgb = np.array(to_rgb(INK))
    ink_lin = np.where(ink_rgb <= 0.04045, ink_rgb / 12.92, ((ink_rgb + 0.055) / 1.055) ** 2.4)
    ink_lum = float(ink_lin @ np.array([0.2126, 0.7152, 0.0722]))
    return "white" if 1.05 / (luminance + 0.05) > (luminance + 0.05) / (ink_lum + 0.05) else INK


def draw_banner(fig: plt.Figure, x_local: float, w_local: float, label: str, color: str) -> None:
    patch = FancyBboxPatch(
        ((LEFT + x_local) / W, BANNER_Y / H),
        w_local / W,
        BANNER_H / H,
        boxstyle=f"round,pad=0,rounding_size={0.65 / W}",
        mutation_aspect=W / H,
        transform=fig.transFigure,
        fc=color,
        ec="none",
        lw=0,
    )
    fig.add_artist(patch)
    fig.text(
        (LEFT + x_local + w_local / 2.0) / W,
        (BANNER_Y + BANNER_H / 2.0) / H,
        label,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="white",
    )


def draw_panel_letter(fig: plt.Figure, col: int, y_mm: float, letter: str, title: str | None = None) -> None:
    fig.text(
        LETTER_X[col] / W,
        y_mm / H,
        letter,
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )
    if title:
        fig.text(
            (LETTER_X[col] + 4.5) / W,
            (y_mm + 0.15) / H,
            title,
            fontsize=8.8,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=INK,
        )


# ---------------------------------------------------------------------------
# Static 4-Task Load & Table Drawing
# ---------------------------------------------------------------------------

def load_static_task(task: str) -> tuple[np.ndarray, np.ndarray]:
    lookup = {}
    for panel in t.PANELS:
        path = REPO_ROOT / "data" / "exports" / panel / f"{task}_scores.csv"
        for row in t.readcsv(path):
            key = (row["dataset"], int(row["seed"]), row["model"])
            assert key not in lookup
            lookup[key] = row
    assert len(lookup) == 576
    metrics = t.METRICS[task]
    raw = np.array([
        [[[float(lookup[d, s, m][k]) for k, _, _ in metrics] for m in t.MODELS] for s in t.SEEDS]
        for d in t.DS
    ])
    assert np.isfinite(raw).all()
    ranked = np.empty_like(raw)
    for di, si, ki in np.ndindex(12, 3, 4):
        ranked[di, si, :, ki] = t.ranks(raw[di, si, :, ki], high=metrics[ki][2])

    reference = {
        (row["panel"], row["model"]): float(row["mean_metric_rank"])
        for row in t.readcsv(REPO_ROOT / "data" / "all_panel_ranks.csv")
        if row["task"] == task
    }
    for pi, panel in enumerate(t.PANELS):
        mr = ranked[pi * 4 : pi * 4 + 4].mean(axis=(0, 1, 3))
        for mi, model in enumerate(t.MODELS):
            assert np.isclose(mr[mi], reference[panel, model], atol=1e-12)
    return raw, ranked


def static_rank_summary(ranked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.column_stack(
        [ranked.sum(axis=(0, 1, 3))]
        + [ranked[i * 4 : i * 4 + 4].sum(axis=(0, 1, 3)) for i in range(3)]
    )
    ordinal = np.column_stack([t.ranks(sums[:, j]) for j in range(4)])
    means = sums / np.array([144.0, 48.0, 48.0, 48.0])
    order = np.argsort(sums[:, 0], kind="stable")
    return order, means, ordinal


def draw_static_table(
    fig: plt.Figure,
    models: list[str],
    ordinal: np.ndarray,
    dataset_means: np.ndarray,
    title: str,
) -> plt.Axes:
    ax = axmm(fig, LEFT, TABLE_BOTTOM, CONTENT_W, TABLE_H)
    ax.set(xlim=(0, CONTENT_W), ylim=(15.5, -0.5))
    ax.axis("off")

    for i, model in enumerate(models):
        ax.add_patch(Rectangle((0, i - 0.5), CONTENT_W, 1.0, fc=ROW_EVEN if i % 2 == 0 else ROW_ODD, lw=0))
        ax.text(
            24.2,
            i,
            t.display(model),
            ha="right",
            va="center",
            fontsize=7.3,
            color=INK,
            fontweight="normal",
        )

    draw_panel_letter(fig, 0, SEC_A_Y, "a", title)
    draw_banner(fig, 0.0, 25.4, "Method", MUTED)
    draw_banner(fig, 26.9, 19.2, "Overall", OVERALL_COL)

    def draw_rank_col(x_start: float, num_w: float, bar_max: float, col_idx: int, bar_fc: str | tuple) -> None:
        bar_x = x_start + num_w + 0.9
        for i in range(16):
            val = ordinal[i, col_idx]
            ax.text(
                x_start + num_w,
                i,
                f"{val:g}",
                ha="right",
                va="center",
                fontsize=7.2,
                color=INK,
            )
            ax.add_patch(Rectangle(
                (bar_x, i - 0.30),
                bar_max * (17.0 - val) / 16.0,
                0.60,
                fc=bar_fc,
                ec="#334155",
                lw=0.45,
            ))
        center_x = x_start + (num_w + 0.9 + bar_max) / 2.0
        fig.text(
            (LEFT + center_x) / W,
            SUBHDR_Y / H,
            "Rank",
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            color=SUBINK,
        )

    draw_rank_col(27.1, 5.1, 12.6, 0, OVERALL_BAR)

    # Three panel groups across [47.6, 176.0] (128.4 mm total, 1.65 mm gaps -> 41.70 mm per group)
    group_w = 41.70
    group_gap = 1.65
    for pi, (panel_label, pcol) in enumerate(zip(t.PANEL_NAMES, PANEL_COLORS_STATIC)):
        g0 = 47.6 + pi * (group_w + group_gap)
        draw_banner(fig, g0, group_w, panel_label, pcol)
        bar_tint = tuple(0.42 + 0.58 * np.array(to_rgb(pcol)))
        draw_rank_col(g0 + 0.2, 5.1, 7.6, pi + 1, bar_tint)

        cmap = LinearSegmentedColormap.from_list(f"panel_{pi}", ["#F4F7FA", pcol])
        tile_start = g0 + 15.06
        tile_pitch = 6.76
        tile_w = 6.36
        for di in range(4):
            col = pi * 4 + di
            tx = tile_start + di * tile_pitch
            fig.text(
                (LEFT + tx + tile_w / 2.0) / W,
                SUBHDR_Y / H,
                t.DS_NAMES[col],
                fontsize=7.1,
                fontweight="bold",
                ha="center",
                va="center",
                color=SUBINK,
            )
            col_min = float(dataset_means[:, col].min())
            for i in range(16):
                val = float(dataset_means[i, col])
                rgba = cmap((16.0 - val) / 15.0)
                ax.add_patch(Rectangle((tx, i - 0.42), tile_w, 0.84, fc=rgba, lw=0))
                ax.text(
                    tx + tile_w / 2.0,
                    i,
                    f"{val:.1f}",
                    fontsize=6.6,
                    ha="center",
                    va="center",
                    color=contrasting_ink(rgba),
                    fontweight="bold" if np.isclose(val, col_min) else "normal",
                )

    return ax


# ---------------------------------------------------------------------------
# Perturbation Load & Table Drawing
# ---------------------------------------------------------------------------

def load_perturbation() -> tuple[np.ndarray, np.ndarray]:
    rows = t.readcsv(REPO_ROOT / "data" / "exports" / "perturbation" / "perturbation_scores.csv")
    lookup = {(row["dataset"], int(row["seed"]), row["model"]): row for row in rows}
    assert len(lookup) == 192
    raw = np.array([
        [[[float(lookup[d, s, m][key]) for key, _, _ in PERT_METRICS] for m in t.MODELS] for s in t.SEEDS]
        for d in PERT_Datasets
    ])
    assert np.isfinite(raw).all()
    ranked = np.empty_like(raw)
    for di, si, ki in np.ndindex(4, 3, 4):
        ranked[di, si, :, ki] = t.ranks(raw[di, si, :, ki], high=PERT_METRICS[ki][2])

    reference = {
        row["model"]: float(row["mean_metric_rank"])
        for row in t.readcsv(REPO_ROOT / "data" / "all_panel_ranks.csv")
        if row["task"] == "perturbation"
    }
    for mi, model in enumerate(t.MODELS):
        assert np.isclose(ranked[:, :, mi, :].mean(), reference[model], atol=1e-12)
    return raw, ranked


def pert_rank_summary(ranked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.column_stack(
        [ranked.sum(axis=(0, 1, 3))] + [ranked[i].sum(axis=(0, 2)) for i in range(4)]
    )
    ordinal = np.column_stack([t.ranks(sums[:, j]) for j in range(5)])
    means = sums / np.array([48.0, 12.0, 12.0, 12.0, 12.0])
    order = np.argsort(sums[:, 0], kind="stable")
    return order, means, ordinal


def draw_pert_table(
    fig: plt.Figure,
    models: list[str],
    ordinal: np.ndarray,
    dataset_means: np.ndarray,
) -> plt.Axes:
    ax = axmm(fig, LEFT, TABLE_BOTTOM, CONTENT_W, TABLE_H)
    ax.set(xlim=(0, CONTENT_W), ylim=(15.5, -0.5))
    ax.axis("off")

    for i, model in enumerate(models):
        ax.add_patch(Rectangle((0, i - 0.5), CONTENT_W, 1.0, fc=ROW_EVEN if i % 2 == 0 else ROW_ODD, lw=0))
        ax.text(
            24.2,
            i,
            t.display(model),
            ha="right",
            va="center",
            fontsize=7.3,
            color=INK,
        )

    draw_panel_letter(fig, 0, SEC_A_Y, "a", "Composite ranking across Perturb-seq screens")
    draw_banner(fig, 0.0, 25.4, "Method", MUTED)
    draw_banner(fig, 26.9, 20.8, "Overall", OVERALL_COL)

    def draw_rank_col(x_start: float, num_w: float, bar_max: float, col_idx: int, bar_fc: str | tuple) -> None:
        bar_x = x_start + num_w + 0.9
        for i in range(16):
            val = ordinal[i, col_idx]
            ax.text(
                x_start + num_w,
                i,
                f"{val:g}",
                ha="right",
                va="center",
                fontsize=7.2,
                color=INK,
            )
            ax.add_patch(Rectangle(
                (bar_x, i - 0.30),
                bar_max * (17.0 - val) / 16.0,
                0.60,
                fc=bar_fc,
                ec="#334155",
                lw=0.45,
            ))
        center_x = x_start + (num_w + 0.9 + bar_max) / 2.0
        fig.text(
            (LEFT + center_x) / W,
            SUBHDR_Y / H,
            "Rank",
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            color=SUBINK,
        )

    draw_rank_col(27.2, 5.4, 13.6, 0, OVERALL_BAR)

    # Four screen groups across [49.3, 176.0] (126.7 mm total, 1.633 mm gaps -> 30.45 mm per group)
    group_w = 30.45
    group_gap = (176.0 - 49.3 - 4 * group_w) / 3.0
    for di, (label, pcol) in enumerate(zip(PERT_LABELS, PANEL_COLORS_PERT)):
        g0 = 49.3 + di * (group_w + group_gap)
        draw_banner(fig, g0, group_w, label, pcol)
        bar_tint = tuple(0.44 + 0.56 * np.array(to_rgb(pcol)))
        draw_rank_col(g0 + 0.3, 5.3, 9.5, di + 1, bar_tint)

        cmap = LinearSegmentedColormap.from_list(f"pert_{di}", ["#F4F7FA", pcol])
        tx = g0 + 17.2
        tw = group_w - 17.2  # 13.25 mm
        fig.text(
            (LEFT + tx + tw / 2.0) / W,
            SUBHDR_Y / H,
            "Mean",
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            va="center",
            color=SUBINK,
        )
        col_min = float(dataset_means[:, di].min())
        for i in range(16):
            val = float(dataset_means[i, di])
            rgba = cmap((16.0 - val) / 15.0)
            ax.add_patch(Rectangle((tx, i - 0.42), tw, 0.84, fc=rgba, lw=0))
            ax.text(
                tx + tw / 2.0,
                i,
                f"{val:.1f}",
                fontsize=7.0,
                ha="center",
                va="center",
                color=contrasting_ink(rgba),
                fontweight="bold" if np.isclose(val, col_min) else "normal",
            )

    return ax


# ---------------------------------------------------------------------------
# Shared 2x2 Boxplot Panel Drawing (Panels b-e)
# ---------------------------------------------------------------------------

def draw_metric_boxplots(
    fig: plt.Figure,
    models: list[str],
    seed_means: np.ndarray,
    rank_means: np.ndarray,
    panel_specs: list[tuple[int, str, bool]],
) -> list[plt.Axes]:
    axes: list[plt.Axes] = []
    for plot_i, (ki, title, higher) in enumerate(panel_specs):
        shown = seed_means[:, :, ki] if higher else rank_means[:, :, ki]
        direction = -1 if higher else 1
        metric_order = np.argsort(direction * shown.mean(axis=0), kind="stable")
        panel_models = [models[i] for i in metric_order]
        panel_values = shown[:, metric_order]

        col = plot_i % 2
        row = 0 if plot_i < 2 else 1
        x, y = METRIC_X[col], METRIC_Y[row]
        label_y = METRIC_LABEL_Y[row]

        ax = axmm(fig, x, y, METRIC_W, METRIC_H)
        draw_panel_letter(fig, col, label_y, "bcde"[plot_i])
        fig.text(
            (x + METRIC_W / 2.0) / W,
            (label_y + 0.1) / H,
            title,
            fontsize=8.1,
            fontweight="bold",
            ha="center",
            va="bottom",
            color=INK,
        )

        bp = ax.boxplot(
            [panel_values[:, mi] for mi in range(16)],
            positions=np.arange(16),
            widths=0.55,
            whis=(0, 100),
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(lw=0.70),
            medianprops=dict(lw=0.96),
            whiskerprops=dict(lw=0.65),
            capprops=dict(lw=0.65),
        )
        for mi, model in enumerate(panel_models):
            color = METHOD_COLORS[model]
            edge = tuple(0.72 * np.array(to_rgb(color)))
            bp["boxes"][mi].set_facecolor(color)
            bp["boxes"][mi].set_edgecolor(edge)
            bp["medians"][mi].set_color(contrasting_ink(to_rgb(color)))
            for line in bp["whiskers"][2 * mi : 2 * mi + 2] + bp["caps"][2 * mi : 2 * mi + 2]:
                line.set_color(edge)
            assert np.isclose(bp["medians"][mi].get_ydata()[0], np.median(panel_values[:, mi]))

        ax.set_xlim(-0.68, 15.68)
        ax.set_xticks(range(16), [t.display(m) for m in panel_models])
        ax.tick_params(
            axis="x",
            which="both",
            top=False,
            labeltop=False,
            bottom=True,
            labelbottom=True,
            length=2.0,
            width=0.65,
            pad=1.3,
            direction="out",
            colors=SUBINK,
        )
        ax.tick_params(
            axis="y",
            which="both",
            right=False,
            labelright=False,
            left=True,
            labelleft=True,
            length=2.4,
            width=0.65,
            pad=2.0,
            direction="out",
            colors=SUBINK,
        )
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")
            lbl.set_va("top")
            lbl.set_rotation_mode("anchor")
            lbl.set_fontsize(6.8)
            lbl.set_color(INK)
            lbl.set_clip_on(False)
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(7.1)
            lbl.set_color(SUBINK)

        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(SPINE)
            side.set_linewidth(0.65)
            side.set_zorder(3)

        if higher:
            assert np.all(np.diff(panel_values.mean(axis=0)) <= 1e-12)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
        else:
            assert np.all(np.diff(panel_values.mean(axis=0)) >= -1e-12)
            ax.set_ylim(16.5, 0.5)
            ax.set_yticks([1, 4, 8, 12, 16])

        ax.grid(axis="y", color=GRID, lw=0.52, zorder=0)
        ax.set_axisbelow(True)
        axes.append(ax)
    return axes


# ---------------------------------------------------------------------------
# Export & Quality Verification (Vector PDF + Hybrid SVG-in-PPTX)
# ---------------------------------------------------------------------------

def verify_bounds_and_export(fig: plt.Figure, axes: list[plt.Axes], out_stem: str, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        panel_ids=list("abcde"),
        row_groups=[["b", "c"], ["d", "e"]],
        column_groups=[["b", "d"], ["c", "e"]],
        json_out=str(tmp_dir / f"{out_stem}_alignment.json"),
        overlay_svg=str(tmp_dir / f"{out_stem}_alignment.svg"),
        tolerance_pt=1.0,
        gutter_tolerance_pt=1.0,
        strict=True,
    )

    # Verify all text bounding boxes stay strictly within figure canvas [0, W] x [0, H]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    for txt in fig.findobj(matplotlib.text.Text):
        if not txt.get_visible() or not txt.get_text().strip():
            continue
        bb = txt.get_window_extent(renderer=renderer)
        assert bb.x0 >= 1.0, f"{out_stem}: text '{txt.get_text()}' clips left edge (x0={bb.x0:.2f}px)"
        assert bb.x1 <= fig_bbox.width - 1.0, f"{out_stem}: text '{txt.get_text()}' clips right edge"
        assert bb.y0 >= 1.0, f"{out_stem}: text '{txt.get_text()}' clips bottom edge (y0={bb.y0:.2f}px)"
        assert bb.y1 <= fig_bbox.height - 1.0, f"{out_stem}: text '{txt.get_text()}' clips top edge"

    # Verify Row 1 (panels b, c) tilted x-labels never collide with Row 2 (panels d, e) titles
    row1_min_y = min(
        lbl.get_window_extent(renderer=renderer).y0
        for ax in axes[1:3]
        for lbl in ax.get_xticklabels()
    )
    row2_title_y_px = ((METRIC_LABEL_Y[1] + 3.0) / H) * fig_bbox.height
    assert row1_min_y >= row2_title_y_px + 2.0, (
        f"{out_stem}: Row 1 x-labels ({row1_min_y:.1f}px) too close to Row 2 title ({row2_title_y_px:.1f}px)"
    )

    base_out = tmp_dir / out_stem
    export_figure(
        fig,
        base_out,
        formats=["pdf", "png"],
        dpi=600,
        font_mode="truetype",
        bbox_inches=None,
        transparent=False,
        overwrite=True,
        write_manifest=False,
    )
    export_figure(
        fig,
        tmp_dir / f"{out_stem}_outlined",
        formats=["svg"],
        font_mode="paths",
        bbox_inches=None,
        overwrite=True,
        write_manifest=False,
    )

    png_path = tmp_dir / f"{out_stem}.png"
    with Image.open(png_path) as img:
        img.convert("RGB").save(png_path, dpi=(600, 600))

    pdf_src = tmp_dir / f"{out_stem}.pdf"
    with pymupdf.open(pdf_src) as doc:
        assert len(doc) == 1 and not doc[0].get_images(), f"{out_stem}.pdf must be 100% vector"
        assert all("Type3" not in str(f) for f in doc[0].get_fonts()), f"{out_stem}.pdf has Type3 fonts"

    pdf_dst = REPO_ROOT / "figures" / f"{out_stem}.pdf"
    pdf_dst.write_bytes(pdf_src.read_bytes())

    # Build matching hybrid SVG+PNG PowerPoint slide
    deck = Presentation()
    w_in, h_in = W / 25.4, H / 25.4
    deck.slide_width = Inches(w_in)
    deck.slide_height = Inches(h_in)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    pic = slide.shapes.add_picture(str(png_path), 0, 0, width=Inches(w_in), height=Inches(h_in))
    svg_bytes = (tmp_dir / f"{out_stem}_outlined.svg").read_bytes()
    part = Part(PackURI(f"/ppt/media/{out_stem.lower()}.svg"), "image/svg+xml", deck.part.package, svg_bytes)
    rid = slide.part.relate_to(part, RT.IMAGE)
    exts = OxmlElement("a:extLst")
    ext = OxmlElement("a:ext")
    ext.set("uri", "{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
    ns = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
    element = etree.Element("{" + ns + "}svgBlip", nsmap={"asvg": ns})
    element.set(qn("r:embed"), rid)
    ext.append(element)
    exts.append(ext)
    pic._pic.blipFill.blip.append(exts)
    deck.save(REPO_ROOT / "figures" / f"{out_stem}.pptx")


def main() -> None:
    configure_rcparams()
    tmp_dir = REPO_ROOT / "figures" / "_main_redesign_build"

    # 1. Static tasks: Figures 2-5
    for spec in STATIC_SPECS:
        task = spec["task"]
        raw, ranked = load_static_task(task)
        order, _, ordinal = static_rank_summary(ranked)
        models = [t.MODELS[i] for i in order]
        raw_ord = raw[:, :, order, :]
        ranked_ord = ranked[:, :, order, :]
        ordinal_ord = ordinal[order]
        dataset_means = ranked_ord.mean(axis=(1, 3)).T

        fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor="white")
        ax_a = draw_static_table(fig, models, ordinal_ord, dataset_means, spec["panel_a_title"])

        panel_specs = []
        for ki in spec["plot_seq"]:
            key, _, higher = t.METRICS[task][ki]
            panel_specs.append((ki, spec["metric_titles"][key], higher))
        metric_axes = draw_metric_boxplots(
            fig,
            models,
            raw_ord.mean(axis=1),
            ranked_ord.mean(axis=1),
            panel_specs,
        )
        verify_bounds_and_export(fig, [ax_a] + metric_axes, spec["out_stem"], tmp_dir)
        plt.close(fig)
        print(f"Generated and verified figures/{spec['out_stem']}.pdf & .pptx")

    # 2. Perturbation task: Figure 7 (manuscript Figure 6)
    raw_p, ranked_p = load_perturbation()
    order_p, _, ordinal_p = pert_rank_summary(ranked_p)
    models_p = [t.MODELS[i] for i in order_p]
    raw_p_ord = raw_p[:, :, order_p, :]
    ranked_p_ord = ranked_p[:, :, order_p, :]
    ordinal_p_ord = ordinal_p[order_p]
    dataset_means_p = ranked_p_ord.mean(axis=(1, 3)).T

    fig_p = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor="white")
    ax_pa = draw_pert_table(fig_p, models_p, ordinal_p_ord, dataset_means_p)
    pert_specs = [(ki, title, higher) for ki, (_, title, higher) in enumerate(PERT_METRICS)]
    pert_axes = draw_metric_boxplots(
        fig_p,
        models_p,
        raw_p_ord.mean(axis=1),
        ranked_p_ord.mean(axis=1),
        pert_specs,
    )
    verify_bounds_and_export(fig_p, [ax_pa] + pert_axes, "Figure7", tmp_dir)
    plt.close(fig_p)
    print("Generated and verified figures/Figure7.pdf & .pptx")


if __name__ == "__main__":
    main()
