"""Top-journal multi-panel overview figure for CellBench (zip manuscript edition).

Dual-backend declarative scene architecture:
1. Vector PDF / SVG / 600-DPI RGB PNG rendered via Matplotlib (TrueType Type 42).
2. 100% Native Editable PowerPoint (.pptx) built from the exact same SCENE primitives
   (MSO_SHAPE, MSO_CONNECTOR with XML arrowheads/dashes, calibrated Arial textboxes;
   zero raster images embedded in the native PPTX).
3. Bonus hybrid SVG-in-PPTX (.svgBlip + 600-DPI PNG fallback) for single-object slide scaling.

Synchronized with main.tex, task_layout.py, and Figures 2-8:
- Streamlined low-text overview across 4 balanced panels (a-d).
- Merges primary task metrics (Slots 1-5) and cross-task diagnostic audits (Slot 6)
  into one 2x3 statistical layer in Panel d (reading order matches Panel c).
- Drops undefined input codes and per-glyph micro-labels; one shared Observed/Predicted legend.
- Keeps all 13 frozen scFMs (with species scope cues) + 3 dataset-fitted baselines
  and all 20 primary evaluation metrics with zero raster images in PDF and native PPTX.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image
import pymupdf

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "fig01_optimized"

# Exact full-text width (183 mm) with compact 4-panel (a-d) height
W, H = 518.74, 508.0

# Harmonized top-journal palette aligned with CellBench Figures 2-8
INK = "#1E293B"        # Primary dark slate ink
MUTED = "#475569"      # Secondary slate ink (WCAG > 7:1 on white)
SUBTLE = "#64748B"     # Structural frame / connector stroke
BORDER = "#CBD5E1"     # Crisp card border
HAIRLINE = "#E2E8F0"   # Subtle internal divider
BG_CARD = "#F8FAFC"    # Soft architectural surface
BG_PANEL = "#FFFFFF"

BLUE = "#2B6CB0"       # Human organs / Task 1 Clustering / Human-pretrained
BLUE_TINT = "#EBF3FA"
BLUE_EDGE = "#93B8DF"

ROSE = "#B84A73"       # Human diseases / Task 2 Batch / Multispecies
ROSE_TINT = "#F9ECF1"
ROSE_EDGE = "#DFA5BB"

TEAL = "#178573"       # Mouse organs / Task 3 Annotation
TEAL_TINT = "#E7F5F2"
TEAL_EDGE = "#8CCAC0"

AMBER = "#C07A1C"      # Task 4 Reconstruction / Predicted profile
AMBER_TINT = "#FDF4E7"
AMBER_EDGE = "#E5C185"

PLUM = "#6B5294"       # Cohort 4 & Task 5 Perturbation / Shared overlap
PLUM_TINT = "#F2EDF9"
PLUM_EDGE = "#B9A7D6"

AUDIT_BG = "#F1F5F9"   # Slot 6 Cross-task diagnostic audits (neutral slate)
AUDIT_EDGE = "#94A3B8"
AUDIT_ACCENT = "#334155"

OBS_COL = "#334155"    # Observed / Reference in Tasks 4-5
PRED_COL = "#D97706"   # Predicted in Tasks 4-5

SCENE: list[dict] = []

CAPTION_LATEX = (
    r"\textbf{Standardized frozen-backbone evaluation across static tissue panels and genetic perturbation screens.} "
    r"(a) Four evaluation panels cover human organs, human diseases, anatomically matched mouse organs, and CRISPR Perturb-seq screens. "
    r"(b) Thirteen frozen scFM checkpoints process model-specific tokenized inputs without weight updates alongside "
    r"256-dimensional PCA, UMAP, and scVI baselines fitted strictly on each seed's training split. "
    r"(c) Donor-partitioned cohorts feed shared unsupervised ($k$-NN/Leiden) and supervised (MLP) probes for clustering (1), "
    r"native batch assessment (2), cell-type annotation (3), and HVG reconstruction (4); Perturb-seq screens feed a matched-control "
    r"residual perturbation decoder (5). "
    r"(d) Twenty primary task--metric slots (1--5, four per task) are ranked within matched $(\text{dataset}, \text{seed}, \text{metric})$ "
    r"blocks alongside cross-task diagnostic audits (6) covering linear probes, active-vs-lookup scale, cross-species and $\Delta=0$ skill, "
    r"and pretraining overlap/LODO stability."
)


# ---------------------------------------------------------------------------
# Declarative Primitives (1:1 shared between Matplotlib and Native PowerPoint)
# ---------------------------------------------------------------------------

def add_rect(
    uid: str,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str = "white",
    edge: str = BORDER,
    radius: float = 0.0,
    lw: float = 0.6,
    dash: str = "solid",
    skip_qa_clip: bool = False,
) -> None:
    SCENE.append(
        dict(
            kind="rect",
            id=uid,
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h),
            fill=fill,
            edge=edge,
            radius=float(radius),
            lw=float(lw),
            dash=dash,
            skip_qa_clip=skip_qa_clip,
        )
    )


def add_circle(
    uid: str,
    x: float,
    y: float,
    r: float,
    fill: str = BLUE,
    edge: str | None = None,
    lw: float = 0.6,
    dash: str = "solid",
) -> None:
    SCENE.append(
        dict(
            kind="circle",
            id=uid,
            x=float(x),
            y=float(y),
            r=float(r),
            fill=fill,
            edge=fill if edge is None else edge,
            lw=float(lw),
            dash=dash,
        )
    )


def add_diamond(
    uid: str,
    x: float,
    y: float,
    r: float,
    fill: str = ROSE,
    edge: str | None = None,
    lw: float = 0.5,
) -> None:
    SCENE.append(
        dict(
            kind="diamond",
            id=uid,
            x=float(x),
            y=float(y),
            r=float(r),
            fill=fill,
            edge=fill if edge is None else edge,
            lw=float(lw),
        )
    )


def add_line(
    uid: str,
    points: list[tuple[float, float]],
    color: str = SUBTLE,
    lw: float = 0.75,
    arrow: bool = False,
    dash: str = "solid",
) -> None:
    SCENE.append(
        dict(
            kind="line",
            id=uid,
            points=[(float(px), float(py)) for px, py in points],
            color=color,
            lw=float(lw),
            arrow=arrow,
            dash=dash,
        )
    )


def add_text(
    uid: str,
    x: float,
    y: float,
    text: str,
    size: float = 7.6,
    bold: bool = False,
    italic: bool = False,
    color: str = INK,
    ha: str = "left",
    va: str = "center",
    box: list[float] | None = None,
    allow_overlap_group: str | None = None,
) -> None:
    entry = dict(
        kind="text",
        id=uid,
        x=float(x),
        y=float(y),
        text=text,
        size=float(size),
        bold=bold,
        italic=italic,
        color=color,
        ha=ha,
        va=va,
    )
    if box is not None:
        entry["layout_box"] = [float(v) for v in box]
    if allow_overlap_group is not None:
        entry["allow_overlap_group"] = allow_overlap_group
    SCENE.append(entry)


# ---------------------------------------------------------------------------
# Reusable Composite Glyphs
# ---------------------------------------------------------------------------

def draw_section_header(letter: str, title: str, y: float, right_note: str = "") -> None:
    add_text(f"sec_{letter}", 10, y, letter, size=10.5, bold=True, color=INK, ha="left", va="center")
    add_text(f"sec_{letter}_title", 23, y, title, size=9.3, bold=True, color=INK, ha="left", va="center")
    if right_note:
        add_text(f"sec_{letter}_note", 508.74, y + 0.3, right_note, size=7.5, color=MUTED, ha="right", va="center")


def draw_badge_card(
    uid: str,
    x: float,
    y: float,
    w: float,
    h: float,
    accent: str,
    fill: str = "white",
    edge: str = BORDER,
    top_bar_h: float = 3.2,
) -> None:
    add_rect(uid, x, y, w, h, fill=fill, edge=edge, radius=3.0, lw=0.65)
    add_rect(f"{uid}_accent", x + 0.5, y + 0.5, w - 1.0, top_bar_h, fill=accent, edge="none", radius=1.5, lw=0)


def add_path(
    uid: str,
    commands: list[tuple],
    fill: str = "white",
    edge: str = BORDER,
    lw: float = 0.8,
    dash: str = "solid",
) -> None:
    SCENE.append(
        dict(
            kind="path",
            id=uid,
            commands=commands,
            fill=fill,
            edge=edge,
            lw=float(lw),
            dash=dash,
        )
    )


def draw_human_emblem(uid: str, cx: float, cy: float, color: str, tint: str, disease: bool = False) -> None:
    # Vitruvian-proportioned standing human silhouette (head + rounded shoulders + natural mid-thigh arms + clean leg gap)
    add_circle(f"{uid}_head", cx, cy - 9.9, 2.55, fill=tint, edge=color, lw=0.78)
    body_cmds = [
        ("M", cx, cy - 6.1),
        ("L", cx + 2.50, cy - 6.1),
        ("C", cx + 4.75, cy - 6.1, cx + 6.00, cy - 4.95, cx + 6.20, cy - 2.8),
        ("L", cx + 6.55, cy + 3.65),
        ("C", cx + 6.60, cy + 4.50, cx + 6.05, cy + 4.90, cx + 5.55, cy + 4.90),
        ("C", cx + 5.05, cy + 4.90, cx + 4.62, cy + 4.50, cx + 4.56, cy + 3.65),
        ("L", cx + 4.25, cy - 2.25),
        ("C", cx + 4.20, cy - 2.95, cx + 3.22, cy - 2.95, cx + 3.20, cy - 2.25),
        ("L", cx + 3.12, cy + 3.0),
        ("L", cx + 2.98, cy + 11.15),
        ("C", cx + 2.98, cy + 12.05, cx + 2.42, cy + 12.45, cx + 1.86, cy + 12.45),
        ("C", cx + 1.30, cy + 12.45, cx + 0.80, cy + 12.05, cx + 0.80, cy + 11.15),
        ("L", cx + 0.72, cy + 3.55),
        ("C", cx + 0.72, cy + 2.75, cx - 0.72, cy + 2.75, cx - 0.72, cy + 3.55),
        ("L", cx - 0.80, cy + 11.15),
        ("C", cx - 0.80, cy + 12.05, cx - 1.30, cy + 12.45, cx - 1.86, cy + 12.45),
        ("C", cx - 2.42, cy + 12.45, cx - 2.98, cy + 12.05, cx - 2.98, cy + 11.15),
        ("L", cx - 3.12, cy + 3.0),
        ("L", cx - 3.20, cy - 2.25),
        ("C", cx - 3.22, cy - 2.95, cx - 4.20, cy - 2.95, cx - 4.25, cy - 2.25),
        ("L", cx - 4.56, cy + 3.65),
        ("C", cx - 4.62, cy + 4.50, cx - 5.05, cy + 4.90, cx - 5.55, cy + 4.90),
        ("C", cx - 6.05, cy + 4.90, cx - 6.60, cy + 4.50, cx - 6.55, cy + 3.65),
        ("L", cx - 6.20, cy - 2.8),
        ("C", cx - 6.00, cy - 4.95, cx - 4.75, cy - 6.1, cx - 2.50, cy - 6.1),
        ("L", cx, cy - 6.1),
        ("Z",),
    ]
    add_path(f"{uid}_body", body_cmds, fill=tint, edge=color, lw=0.78)
    if disease:
        # Crisp filled medical cross centered on the chest
        ky = cy - 1.45
        a, t = 1.90, 0.60
        cross_cmds = [
            ("M", cx - t, ky - a), ("L", cx + t, ky - a), ("L", cx + t, ky - t),
            ("L", cx + a, ky - t), ("L", cx + a, ky + t), ("L", cx + t, ky + t),
            ("L", cx + t, ky + a), ("L", cx - t, ky + a), ("L", cx - t, ky + t),
            ("L", cx - a, ky + t), ("L", cx - a, ky - t), ("L", cx - t, ky - t),
            ("Z",),
        ]
        add_path(f"{uid}_cross", cross_cmds, fill=color, edge="none", lw=0.0)


def draw_mouse_emblem(uid: str, cx: float, cy: float, color: str, tint: str) -> None:
    # C1-smooth laboratory mouse profile: sweeping S-tail, far ear, tucked paws, arched back -> conical snout
    add_path(
        f"{uid}_tail",
        [("M", cx - 7.8, cy + 2.5), ("C", cx - 11.0, cy + 3.6, cx - 11.8, cy - 2.0, cx - 14.8, cy - 1.5)],
        fill="none",
        edge=color,
        lw=0.92,
    )
    add_circle(f"{uid}_ear_bg", cx + 4.5, cy - 5.0, 2.10, fill="#CBECE5", edge=color, lw=0.72)
    add_rect(f"{uid}_paw_b", cx - 6.8, cy + 3.5, 4.6, 2.05, fill=tint, edge=color, radius=1.0, lw=0.72)
    add_rect(f"{uid}_paw_f", cx + 1.9, cy + 3.5, 3.8, 1.95, fill=tint, edge=color, radius=0.95, lw=0.72)
    mouse_body = [
        ("M", cx - 6.0, cy + 4.35),
        ("C", cx - 9.0, cy + 4.2, cx - 9.5, cy - 0.9, cx - 6.8, cy - 4.1),
        ("C", cx - 4.3, cy - 6.6, cx - 0.2, cy - 6.4, cx + 2.8, cy - 4.6),
        ("C", cx + 5.6, cy - 3.9, cx + 8.6, cy - 1.8, cx + 11.5, cy + 1.7),
        ("C", cx + 11.7, cy + 2.45, cx + 10.4, cy + 3.2, cx + 8.5, cy + 3.45),
        ("C", cx + 6.5, cy + 3.7, cx + 5.0, cy + 4.25, cx + 3.1, cy + 4.35),
        ("C", cx + 0.2, cy + 4.45, cx - 3.2, cy + 4.45, cx - 6.0, cy + 4.35),
        ("Z",),
    ]
    add_path(f"{uid}_body", mouse_body, fill=tint, edge=color, lw=0.82)
    add_circle(f"{uid}_ear_fg", cx + 2.5, cy - 3.8, 2.60, fill=tint, edge=color, lw=0.80)
    add_circle(f"{uid}_ear_in", cx + 2.7, cy - 3.7, 1.22, fill="white", edge="none", lw=0.0)
    add_circle(f"{uid}_eye", cx + 7.0, cy - 0.85, 0.82, fill=color, edge=color, lw=0.2)
    add_circle(f"{uid}_eye_hi", cx + 6.75, cy - 1.10, 0.25, fill="white", edge="none", lw=0.0)
    add_circle(f"{uid}_nose", cx + 11.45, cy + 1.80, 0.65, fill=color, edge=color, lw=0.2)


def draw_perturb_emblem(uid: str, cx: float, cy: float, color: str, tint: str) -> None:
    add_circle(f"{uid}_ctrl", cx - 7.8, cy - 4.0, 3.3, fill="white", edge=BLUE, lw=0.85)
    add_circle(f"{uid}_ctrl_n", cx - 7.8, cy - 4.0, 1.0, fill=BLUE, edge=BLUE, lw=0.3)
    add_line(f"{uid}_arr", [(cx - 3.3, cy - 4.0), (cx + 3.0, cy - 4.0)], color=color, lw=0.8, arrow=True)
    add_circle(f"{uid}_pert", cx + 7.8, cy - 4.0, 3.3, fill=tint, edge=color, lw=0.85)
    add_diamond(f"{uid}_pert_n", cx + 7.8, cy - 4.0, 1.45, fill=color)
    add_rect(f"{uid}_tag", cx - 11.5, cy + 3.0, 23.0, 8.2, fill=tint, edge=PLUM_EDGE, radius=1.8, lw=0.55)
    add_text(f"{uid}_tag_txt", cx, cy + 7.1, "gRNA", size=6.6, bold=True, color=color, ha="center", va="center",
             box=[cx - 11.5, cy + 3.0, 23.0, 8.2])


def draw_matrix(
    uid: str,
    x: float,
    y: float,
    rows: int = 6,
    cols: int = 6,
    step: float = 5.2,
    palette: list[str] | None = None,
) -> tuple[float, float]:
    pal = palette or ["#DBEAFE", "#93C5FD", "#3B82F6", "#EFF6FF", "#60A5FA"]
    for r in range(rows):
        for c in range(cols):
            col = pal[(r * 3 + c * 2 + (r * c) % 3) % len(pal)]
            add_rect(
                f"{uid}_{r}_{c}",
                x + c * step,
                y + r * step,
                step - 0.55,
                step - 0.55,
                fill=col,
                edge="none",
                radius=0.0,
                lw=0.0,
            )
    w, h = cols * step - 0.55, rows * step - 0.55
    add_rect(f"{uid}_frame", x - 0.6, y - 0.6, w + 1.2, h + 1.2, fill="none", edge=SUBTLE, radius=0.8, lw=0.55)
    return w, h


def draw_mlp(
    uid: str,
    x: float,
    y: float,
    dx: float = 15.0,
    dy: float = 7.8,
    accent: str = TEAL,
    layers: tuple[int, ...] = (3, 4, 3),
) -> tuple[float, float]:
    max_n = max(layers)
    total_h = (max_n - 1) * dy
    cols: list[list[tuple[float, float]]] = []
    for li, n in enumerate(layers):
        offset_y = (total_h - (n - 1) * dy) / 2.0
        cols.append([(x + li * dx, y + offset_y + ni * dy) for ni in range(n)])
    for li in range(len(cols) - 1):
        for i, p in enumerate(cols[li]):
            for j, q in enumerate(cols[li + 1]):
                add_line(f"{uid}_e_{li}_{i}_{j}", [p, q], color="#CBD5E1", lw=0.4)
    for li, col in enumerate(cols):
        is_mid = 0 < li < len(cols) - 1
        for ni, (nx, ny) in enumerate(col):
            add_circle(
                f"{uid}_n_{li}_{ni}",
                nx,
                ny,
                2.1,
                fill="white" if is_mid else accent,
                edge=accent,
                lw=0.65,
            )
    return (len(layers) - 1) * dx, total_h


def draw_snowflake(uid: str, cx: float, cy: float, r: float = 4.4, color: str = BLUE) -> None:
    for i in range(6):
        ang = i * math.pi / 3.0
        px, py = cx + r * math.cos(ang), cy + r * math.sin(ang)
        add_line(f"{uid}_arm_{i}", [(cx, cy), (px, py)], color=color, lw=0.7)
        bx, by = cx + 0.62 * r * math.cos(ang), cy + 0.62 * r * math.sin(ang)
        for sgn in (-1, 1):
            tx = bx + 0.36 * r * math.cos(ang + sgn * math.pi / 3.0)
            ty = by + 0.36 * r * math.sin(ang + sgn * math.pi / 3.0)
            add_line(f"{uid}_br_{i}_{sgn}", [(bx, by), (tx, ty)], color=color, lw=0.55)


def draw_plus_node(uid: str, cx: float, cy: float, r: float = 3.2, color: str = BLUE) -> None:
    add_circle(uid, cx, cy, r, fill="white", edge=color, lw=0.7)
    add_line(f"{uid}_h", [(cx - 1.5, cy), (cx + 1.5, cy)], color=color, lw=0.65)
    add_line(f"{uid}_v", [(cx, cy - 1.5), (cx, cy + 1.5)], color=color, lw=0.65)


# ---------------------------------------------------------------------------
# Full Scene Construction (Panels a - d)
# ---------------------------------------------------------------------------

def build_scene() -> None:
    SCENE.clear()

    # =========================================================================
    # PANEL a: Benchmark cohorts
    # =========================================================================
    draw_section_header("a", "Benchmark cohorts", y=10.5)

    card_y, card_h = 20.0, 48.0
    # Content-proportional widths across 498.74 pt (3 gaps of 6 pt -> 480.74 pt)
    card_ws = [116.0, 114.0, 119.0, 131.74]
    card_xs = [10.0]
    for cw in card_ws[:-1]:
        card_xs.append(card_xs[-1] + cw + 6.0)

    cohorts = [
        ("Human organs", BLUE, BLUE_TINT, [("BM", "Marrow"), ("HL", "Lung"), ("AH", "Heart"), ("AK", "Kidney")]),
        ("Human diseases", ROSE, ROSE_TINT, [("MS", "Brain"), ("UC", "Colitis"), ("C19", "Lung"), ("BC", "Breast")]),
        ("Mouse organs", TEAL, TEAL_TINT, [("MM", "Marrow"), ("ML", "Lung"), ("MH", "Heart"), ("MK", "Kidney")]),
    ]
    for i, (title, col, tint, pairs) in enumerate(cohorts):
        cx0, cw = card_xs[i], card_ws[i]
        draw_badge_card(f"cohort_{i}", cx0, card_y, cw, card_h, accent=col, fill="white", edge=BORDER)
        add_text(f"cohort_title_{i}", cx0 + 7.0, card_y + 10.5, title, size=8.1, bold=True, color=INK)
        if i == 0:
            draw_human_emblem("icon_human_org", cx0 + 15.0, card_y + 31.5, BLUE, BLUE_TINT, disease=False)
        elif i == 1:
            draw_human_emblem("icon_human_dis", cx0 + 15.0, card_y + 31.5, ROSE, ROSE_TINT, disease=True)
        else:
            draw_mouse_emblem("icon_mouse_org", cx0 + 19.3, card_y + 32.0, TEAL, TEAL_TINT)
        tx_left = cx0 + (29.5 if i < 2 else 35.0)
        tx_col2 = tx_left + (43.5 if i < 2 else 44.2)
        for idx, (code, organ) in enumerate(pairs):
            row = idx // 2
            col_x = tx_left if idx % 2 == 0 else tx_col2
            yy = card_y + 25.0 + row * 13.0
            add_text(f"ds_{i}_{code}", col_x, yy, f"{code} {organ}", size=7.25, bold=False, color=INK)

    # Cohort 4: Perturbations (equal visual weight: white card + BORDER + PLUM accent bar)
    px0, pw = card_xs[3], card_ws[3]
    draw_badge_card("cohort_3", px0, card_y, pw, card_h, accent=PLUM, fill="white", edge=BORDER)
    add_text("cohort_title_3", px0 + 7.0, card_y + 10.5, "Genetic perturbations", size=8.1, bold=True, color=INK)
    draw_perturb_emblem("icon_pert", px0 + 18.5, card_y + 32.0, PLUM, PLUM_TINT)
    add_text("pert_row1", px0 + 35.5, card_y + 22.5, "2-gene: Norman", size=7.2, bold=False, color=INK)
    add_text("pert_row2", px0 + 35.5, card_y + 32.5, "Unseen KO: Adamson", size=7.2, bold=False, color=INK)
    add_text("pert_row3", px0 + 35.5, card_y + 42.0, "Replogle K562 / RPE1", size=7.0, bold=False, color=MUTED)

    # Clean collector bus from Panel a into Panel b
    bus_y = 73.0
    for i in range(4):
        mid_x = card_xs[i] + card_ws[i] / 2.0
        add_line(f"a_drop_{i}", [(mid_x, card_y + card_h), (mid_x, bus_y)], color=SUBTLE, lw=0.65)
    add_line("a_bus_h", [(card_xs[0] + card_ws[0] / 2.0, bus_y), (card_xs[3] + card_ws[3] / 2.0, bus_y)], color=SUBTLE, lw=0.65)
    add_line("a_to_b", [(259.37, bus_y), (259.37, 94.0)], color=INK, lw=0.85, arrow=True)

    # =========================================================================
    # PANEL b: Representation extraction & model taxonomy
    # =========================================================================
    draw_section_header("b", "Representation extraction & model taxonomy", y=84.5)

    b_top, b_h = 94.0, 140.0
    add_rect("b_frame", 10.0, b_top, 498.74, b_h, fill="white", edge=SUBTLE, radius=3.0, lw=0.8)

    # Left: scRNA-seq input matrix
    in_mx, in_my = 22.0, b_top + 19.0
    mw, mh = draw_matrix("in_mat", in_mx, in_my, rows=6, cols=6, step=5.0,
                         palette=["#DBEAFE", "#93C5FD", "#2B6CB0", "#EFF6FF", "#60A5FA"])
    in_cx = in_mx + mw / 2.0
    flow_y = in_my + mh / 2.0
    add_text("in_title", in_cx, b_top + 9.8, "scRNA-seq", size=8.0, bold=True, color=INK, ha="center")

    # Arrow from input matrix to tokenization
    add_line("in_to_tok", [(in_mx + mw + 4.0, flow_y), (74.0, flow_y)], color=INK, lw=0.8, arrow=True)

    # Tokenization / Priors block
    tok_x = 77.0
    tok_specs = [
        ("Gene ID", b_top + 10.5, ["#D882A6", "#7FB0D8", "#3B78B0", "#D882A6", "#94A3B8"]),
        ("Expression", b_top + 28.5, ["#3B78B0", "#BDD7EE", "#1E4E79", "#7FB0D8", "#E2EEF8"]),
        ("Priors", b_top + 46.5, ["#52A896", "#C8EAE2", "#178573", "#9AD5C8", "#D9CEF0"]),
    ]
    for ti, (tlabel, ty, tcols) in enumerate(tok_specs):
        add_text(f"tok_lbl_{ti}", tok_x, ty, tlabel, size=7.0, color=INK, ha="left", va="center")
        for tj, tc in enumerate(tcols):
            add_rect(f"tok_tile_{ti}_{tj}", tok_x + tj * 14.2, ty + 5.0, 12.6, 6.4, fill=tc, edge="#64748B", radius=0.6, lw=0.4)

    add_line("tok_to_enc", [(tok_x + 5 * 14.2 + 2.0, flow_y), (168.0, flow_y)], color=INK, lw=0.8, arrow=True)

    # Central Frozen Backbone Module
    enc_x, enc_y, enc_w, enc_h = 170.0, b_top + 5.5, 246.0, 55.0
    add_rect("enc_box", enc_x, enc_y, enc_w, enc_h, fill="#F1F5F9", edge="#64748B", radius=3.0, lw=0.7)
    add_text("enc_hdr", enc_x + 7.0, enc_y + 8.2, "Frozen backbones", size=8.0, bold=True, color=INK)
    draw_snowflake("enc_snow", enc_x + enc_w - 10.5, enc_y + 8.7, r=4.1, color=BLUE)

    # Inside encoder block: Q/K/V Multi-head attention -> (+) -> LayerNorm -> FFN -> (+) -> LayerNorm
    blk_y = enc_y + 34.0
    # Residual bypass arches
    add_line("res_attn", [(enc_x + 7.0, blk_y), (enc_x + 7.0, enc_y + 17.0), (enc_x + 86.0, enc_y + 17.0), (enc_x + 86.0, blk_y - 3.5)],
             color=BLUE, lw=0.65, arrow=True)
    add_line("res_ffn", [(enc_x + 131.0, blk_y), (enc_x + 131.0, enc_y + 17.0), (enc_x + 188.0, enc_y + 17.0), (enc_x + 188.0, blk_y - 3.5)],
             color=BLUE, lw=0.65, arrow=True)

    add_line("enc_in_seg", [(enc_x + 2.0, blk_y), (enc_x + 13.0, blk_y)], color=INK, lw=0.75, arrow=True)
    # Attention box
    att_x, att_w = enc_x + 14.0, 61.0
    add_rect("att_box", att_x, enc_y + 20.5, att_w, 29.5, fill=BLUE_TINT, edge=BLUE, radius=2.0, lw=0.65)
    for qi, (ql, qc) in enumerate([("Q", BLUE), ("K", TEAL), ("V", ROSE)]):
        qx = att_x + 4.5 + qi * 18.2
        add_rect(f"qkv_{ql}", qx, enc_y + 22.7, 15.5, 9.2, fill="white", edge=qc, radius=1.0, lw=0.55)
        add_text(f"qkv_t_{ql}", qx + 7.75, enc_y + 27.3, ql, size=7.0, bold=True, color=INK, ha="center",
                 box=[qx, enc_y + 22.7, 15.5, 9.2])
    add_text("att_lbl", att_x + att_w / 2.0, enc_y + 41.5, "Attention", size=6.9, color=INK, ha="center",
             box=[att_x, enc_y + 34.5, att_w, 14.0])

    # Add + Norm 1
    add_line("att_to_add1", [(att_x + att_w, blk_y), (enc_x + 82.6, blk_y)], color=INK, lw=0.7)
    draw_plus_node("add1", enc_x + 86.0, blk_y, r=3.1, color=BLUE)
    add_line("add1_to_ln1", [(enc_x + 89.2, blk_y), (enc_x + 95.5, blk_y)], color=INK, lw=0.7, arrow=True)
    ln1_x = enc_x + 96.0
    add_rect("ln1", ln1_x, enc_y + 23.0, 29.0, 22.5, fill=TEAL_TINT, edge=TEAL, radius=2.0, lw=0.65)
    add_text("ln1_t", ln1_x + 14.5, enc_y + 34.25, "Norm", size=6.9, color=INK, ha="center", box=[ln1_x, enc_y + 23.0, 29.0, 22.5])

    # FFN box
    add_line("ln1_to_ffn", [(ln1_x + 29.0, blk_y), (enc_x + 137.5, blk_y)], color=INK, lw=0.7, arrow=True)
    ffn_x = enc_x + 138.0
    add_rect("ffn_box", ffn_x, enc_y + 21.0, 38.0, 28.0, fill=PLUM_TINT, edge=PLUM, radius=2.0, lw=0.65)
    add_text("ffn_lbl", ffn_x + 19.0, enc_y + 26.3, "FFN", size=6.7, color=INK, ha="center",
             box=[ffn_x, enc_y + 21.5, 38.0, 10.0])
    ffn_cols = [
        [(ffn_x + 8.0, enc_y + 37.5), (ffn_x + 8.0, enc_y + 43.5)],
        [(ffn_x + 19.0, enc_y + 35.1), (ffn_x + 19.0, enc_y + 40.5), (ffn_x + 19.0, enc_y + 45.9)],
        [(ffn_x + 30.0, enc_y + 37.5), (ffn_x + 30.0, enc_y + 43.5)],
    ]
    for fl in range(2):
        for p in ffn_cols[fl]:
            for q in ffn_cols[fl + 1]:
                add_line(f"ffn_e_{fl}_{p}_{q}", [p, q], color=PLUM_EDGE, lw=0.45)
    for fl, col in enumerate(ffn_cols):
        for ni, (nx, ny) in enumerate(col):
            add_circle(f"ffn_n_{fl}_{ni}", nx, ny, 1.15, fill=PLUM)

    # Add + Norm 2
    add_line("ffn_to_add2", [(ffn_x + 38.0, blk_y), (enc_x + 184.6, blk_y)], color=INK, lw=0.7)
    draw_plus_node("add2", enc_x + 188.0, blk_y, r=3.1, color=BLUE)
    add_line("add2_to_ln2", [(enc_x + 191.2, blk_y), (enc_x + 197.5, blk_y)], color=INK, lw=0.7, arrow=True)
    ln2_x = enc_x + 198.0
    add_rect("ln2", ln2_x, enc_y + 23.0, 29.0, 22.5, fill=TEAL_TINT, edge=TEAL, radius=2.0, lw=0.65)
    add_text("ln2_t", ln2_x + 14.5, enc_y + 34.25, "Norm", size=6.9, color=INK, ha="center", box=[ln2_x, enc_y + 23.0, 29.0, 22.5])
    add_line("enc_out_seg", [(ln2_x + 29.0, blk_y), (enc_x + enc_w - 2.0, blk_y)], color=INK, lw=0.75)

    # Right: Cell embeddings matrix
    out_mx, out_my = 436.0, b_top + 19.0
    omw, omh = draw_matrix("out_mat", out_mx, out_my, rows=6, cols=7, step=5.0,
                           palette=["#C8EAE2", "#38A38F", "#E7F5F2", "#178573", "#8CCAC0"])
    out_cx = out_mx + omw / 2.0
    add_line("enc_to_out", [(enc_x + enc_w, flow_y), (out_mx - 4.0, flow_y)], color=INK, lw=0.85, arrow=True)
    add_text("out_title", out_cx, b_top + 9.8, "Cell embeddings", size=8.0, bold=True, color=INK, ha="center")

    # Parallel Dataset-Fitted Baseline Lane (concise, zero parenthetical small-print)
    base_y = b_top + 69.5
    add_line("in_to_base", [(in_cx, in_my + mh + 4.5), (in_cx, base_y), (156.0, base_y)], color=SUBTLE, lw=0.75, arrow=True)
    add_rect("base_pill", 158.0, base_y - 7.3, 174.0, 14.6, fill="#F8FAFC", edge=BLUE_EDGE, radius=3.0, lw=0.65)
    add_text("base_lbl_l", 170.0, base_y, "Fitted baselines:", size=7.5, bold=True, color=INK)
    add_text("base_lbl_m", 240.0, base_y, "PCA · UMAP · scVI", size=7.6, bold=True, color=BLUE)
    add_line("base_to_out", [(332.0, base_y), (out_cx, base_y), (out_cx, out_my + omh + 4.5)], color=SUBTLE, lw=0.75, arrow=True)

    # Structured 3-Lane Taxonomy Table of the 13 Checkpoints (with dedicated header + species legend capsule)
    tax_y, tax_h = b_top + 81.0, 54.0
    add_rect("tax_band", 18.0, tax_y, 482.74, tax_h, fill="#F8FAFC", edge=BORDER, radius=2.5, lw=0.55)
    add_rect("tax_hdr_bg", 18.4, tax_y + 0.4, 481.94, 13.4, fill="#E8EEF5", edge="none", radius=2.0, lw=0)
    add_line("tax_hdr_div", [(18.0, tax_y + 13.8), (500.74, tax_y + 13.8)], color=BORDER, lw=0.6)
    add_line("tax_stub_div", [(106.0, tax_y + 2.5), (106.0, tax_y + tax_h - 2.5)], color=BORDER, lw=0.55)

    hdr_cy = tax_y + 7.1
    add_text("tax_hdr_l", 24.0, hdr_cy, "Tokenization", size=7.1, bold=True, color=INK)
    add_text("tax_hdr_m", 114.0, hdr_cy, "13 scFM checkpoints", size=7.1, bold=True, color=INK)

    # Dedicated framed legend capsule inside the header bar (clearly separated from model rows)
    add_rect("sp_leg_box", 366.0, tax_y + 1.8, 130.5, 10.4, fill="white", edge=BORDER, radius=2.0, lw=0.5)
    add_circle("sp_leg_h_dot", 374.0, hdr_cy, r=1.85, fill=BLUE)
    add_text("sp_leg_h_txt", 379.0, hdr_cy, "Human-only", size=6.9, color=INK)
    add_diamond("sp_leg_m_dot", 435.0, hdr_cy, r=2.1, fill=ROSE)
    add_text("sp_leg_m_txt", 440.0, hdr_cy, "Multispecies", size=6.9, color=ROSE)

    # 5-column grid: Human-only (blue) first, Multispecies (rose) consistently at the end of each row
    col_x = [114.0 + 76.5 * k for k in range(5)]
    taxonomy_rows = [
        ("Rank / Binned", [("Geneformer", False), ("LangCell", False), ("scGPT", False),
                           ("Nicheformer", True), ("GeneCompass", True)]),
        ("Value / Full-panel", [("scFoundation", False), ("scLong", False), ("TranscriptFormer", False),
                                ("CellPLM", False), ("Tabula", False)]),
        ("Aligned / Sampled", [("CellFM", False), ("UCE", True), ("scPRINT2", True)]),
    ]
    for ri, (fam_label, items) in enumerate(taxonomy_rows):
        ry = tax_y + 20.6 + ri * 13.2
        if ri > 0:
            add_line(f"tax_rdiv_{ri}", [(22.0, ry - 6.6), (496.0, ry - 6.6)], color=HAIRLINE, lw=0.45)
        add_text(f"tax_fam_{ri}", 24.0, ry, fam_label, size=7.2, bold=True, color=MUTED)
        for ci, (mname, is_multi) in enumerate(items):
            mx = col_x[ci]
            if is_multi:
                add_diamond(f"sp_{mname}", mx + 2.5, ry, r=2.2, fill=ROSE)
            else:
                add_circle(f"sp_{mname}", mx + 2.5, ry, r=1.9, fill=BLUE)
            add_text(f"model_{mname}", mx + 7.2, ry, mname, size=7.25, color=ROSE if is_multi else INK)

    # Flow arrow b -> c
    add_line("b_to_c", [(259.37, b_top + b_h), (259.37, 252.0)], color=INK, lw=0.85, arrow=True)

    # =========================================================================
    # PANEL c: Standardized downstream probes (Tasks 1-5)
    # =========================================================================
    draw_section_header("c", "Standardized downstream probes", y=242.5)

    c_top = 252.5
    r1_h, r2_h = 52.0, 52.0
    r2_top = c_top + r1_h + 4.5

    # Harmonized, equal-luminance pastel card washes & borders for the 5 tasks
    T1_BG, T1_EDGE = "#EFF5FC", "#B4CEE8"   # Task 1: Clustering (Blue)
    T2_BG, T2_EDGE = "#FCF1F5", "#E4B8CA"   # Task 2: Batch assessment (Rose)
    T3_BG, T3_EDGE = "#EEF9F6", "#ABD9D0"   # Task 3: Annotation (Teal)
    T4_BG, T4_EDGE = "#FDF6EC", "#E8C99B"   # Task 4: Reconstruction (Amber)
    T5_BG, T5_EDGE = "#F5F1FB", "#C9BCE2"   # Task 5: Perturbation (Plum)

    def task_card_header(idx: int, x: float, y: float, w: float, h: float, title: str, col: str, fill: str, edge: str) -> None:
        add_rect(f"task_card_{idx}", x, y, w, h, fill=fill, edge=edge, radius=3.0, lw=0.7)
        add_rect(f"task_badge_{idx}", x + 6.0, y + 5.0, 12.0, 11.5, fill=col, edge="none", radius=2.0, lw=0)
        add_text(f"task_num_{idx}", x + 12.0, y + 10.75, str(idx), size=7.6, bold=True, color="white", ha="center",
                 box=[x + 6.0, y + 5.0, 12.0, 11.5])
        add_text(f"task_title_{idx}", x + 21.5, y + 10.75, title, size=8.0, bold=True, color=INK)

    # Task 1: Cell clustering (w = 152)
    t1_x, t1_w = 10.0, 152.0
    task_card_header(1, t1_x, c_top, t1_w, r1_h, "Cell clustering", BLUE, T1_BG, T1_EDGE)
    t1_cy = c_top + 34.0
    cl_cols = [BLUE, TEAL, ROSE]
    pts_unassigned = [(-14, -6), (-7, -9), (-4, -1), (-12, 3), (-3, 7), (4, -7), (6, 2), (-8, 9), (3, 9), (10, -2)]
    for pi, (dx, dy) in enumerate(pts_unassigned):
        add_circle(f"t1_in_{pi}", t1_x + 38.0 + dx, t1_cy + dy, 1.75, fill=cl_cols[pi % 3])
    add_line("t1_arr", [(t1_x + 56.0, t1_cy), (t1_x + 79.0, t1_cy)], color=INK, lw=0.75, arrow=True)
    for gi, (gx, gy) in enumerate([(t1_x + 98.0, t1_cy - 6.0), (t1_x + 125.0, t1_cy - 5.0), (t1_x + 111.5, t1_cy + 7.5)]):
        for pj, (dx, dy) in enumerate([(-3.2, -1.8), (2.8, -2.0), (0.0, 2.5), (-2.5, 2.8), (3.4, 2.0)]):
            add_circle(f"t1_out_{gi}_{pj}", gx + dx, gy + dy, 1.7, fill=cl_cols[gi])

    # Task 2: Batch assessment (w = 156)
    t2_x, t2_w = t1_x + t1_w + 5.0, 156.0
    task_card_header(2, t2_x, c_top, t2_w, r1_h, "Batch assessment", ROSE, T2_BG, T2_EDGE)
    t2_cy = c_top + 33.5
    b_pts = [(-15, -6), (-7, -8), (-2, -2), (-12, 2), (-4, 6), (4, -7), (9, -1), (2, 4), (10, 6), (-9, 8)]
    for bi, (dx, dy) in enumerate(b_pts):
        bx, by = t2_x + 39.0 + dx, t2_cy + dy
        if bi % 2 == 0:
            add_circle(f"t2_p_{bi}", bx, by, 1.85, fill=BLUE if bi < 5 else TEAL)
        else:
            c_use = BLUE if bi < 5 else TEAL
            add_rect(f"t2_s_{bi}", bx - 1.8, by - 1.8, 3.6, 3.6, fill="white", edge=c_use, radius=0.3, lw=0.75)
    add_line("t2_arr_mix", [(t2_x + 56.0, t2_cy - 2.5), (t2_x + 76.0, t2_cy - 7.5)], color=SUBTLE, lw=0.7, arrow=True)
    add_line("t2_arr_bio", [(t2_x + 56.0, t2_cy + 2.5), (t2_x + 76.0, t2_cy + 7.5)], color=SUBTLE, lw=0.7, arrow=True)
    add_text("t2_mix_lbl", t2_x + 80.0, t2_cy - 8.0, "Donor mixing", size=7.2, color=INK)
    add_text("t2_bio_lbl", t2_x + 80.0, t2_cy + 8.0, "Type conservation", size=7.2, color=INK)

    # Task 3: Cell-type annotation (w = 180.74)
    t3_x, t3_w = t2_x + t2_w + 5.0, 180.74
    task_card_header(3, t3_x, c_top, t3_w, r1_h, "Cell-type annotation", TEAL, T3_BG, T3_EDGE)
    t3_my = c_top + 22.0
    m3w, m3h = draw_matrix("t3_mat", t3_x + 12.0, t3_my, rows=4, cols=4, step=5.0)
    t3_cy = t3_my + m3h / 2.0
    add_line("t3_a1", [(t3_x + 12.0 + m3w + 3.5, t3_cy), (t3_x + 51.0, t3_cy)], color=INK, lw=0.75, arrow=True)
    net3_w, _ = draw_mlp("t3_mlp", t3_x + 57.0, c_top + 21.0, dx=14.5, dy=7.0, accent=TEAL, layers=(3, 4, 3))
    add_line("t3_a2", [(t3_x + 57.0 + net3_w + 5.0, t3_cy), (t3_x + 106.0, t3_cy)], color=INK, lw=0.75, arrow=True)
    cls_specs = [("Type A", BLUE), ("Type B", TEAL), ("Rare <1%", ROSE)]
    for ci, (clbl, ccol) in enumerate(cls_specs):
        cy_box = c_top + 19.5 + ci * 9.5
        add_rect(f"t3_cls_{ci}", t3_x + 109.0, cy_box, 63.0, 8.0, fill="white", edge=ccol, radius=1.5, lw=0.65)
        add_rect(f"t3_cls_bar_{ci}", t3_x + 109.4, cy_box + 0.4, 2.8, 7.2, fill=ccol, edge="none", radius=0.8, lw=0)
        add_text(f"t3_cls_t_{ci}", t3_x + 141.5, cy_box + 4.0, clbl, size=6.8, color=INK, ha="center",
                 box=[t3_x + 112.0, cy_box, 60.0, 8.0])

    # Task 4: Expression reconstruction (Row 2 left, w = 228.0)
    t4_x, t4_w = 10.0, 228.0
    task_card_header(4, t4_x, r2_top, t4_w, r2_h, "Expression reconstruction", AMBER, T4_BG, T4_EDGE)
    t4_my = r2_top + 23.0
    m4w, m4h = draw_matrix("t4_in", t4_x + 18.0, t4_my, rows=4, cols=4, step=5.0)
    t4_cy = t4_my + m4h / 2.0
    add_line("t4_a1", [(t4_x + 18.0 + m4w + 5.5, t4_cy), (t4_x + 68.0, t4_cy)], color=INK, lw=0.75, arrow=True)
    net4_w, _ = draw_mlp("t4_mlp", t4_x + 76.0, r2_top + 22.5, dx=16.5, dy=6.8, accent=AMBER, layers=(3, 4, 4, 3))
    add_line("t4_a2", [(t4_x + 76.0 + net4_w + 6.0, t4_cy), (t4_x + 160.0, t4_cy)], color=INK, lw=0.75, arrow=True)
    draw_matrix("t4_out", t4_x + 166.0, t4_my, rows=4, cols=5, step=5.0,
                palette=["#FEF3C7", "#F59E0B", "#FFFBEB", "#D97706", "#FDE68A"])

    # Task 5: Perturbation prediction (Row 2 right, w = 265.74)
    t5_x, t5_w = t4_x + t4_w + 5.0, 265.74
    task_card_header(5, t5_x, r2_top, t5_w, r2_h, "Perturbation prediction", PLUM, T5_BG, T5_EDGE)
    t5_cy = r2_top + 35.0
    # Matched control cell + target gene condition vector
    ctrl_cx = t5_x + 22.0
    add_circle("t5_ctrl", ctrl_cx, t5_cy, 4.8, fill="white", edge=BLUE, lw=0.8)
    add_circle("t5_ctrl_n", ctrl_cx, t5_cy, 1.5, fill=BLUE)
    add_text("t5_plus", t5_x + 33.5, t5_cy, "+", size=8.2, bold=True, color=INK, ha="center")
    cond_x, cond_w = t5_x + 40.0, 34.0
    add_rect("t5_cond", cond_x, t5_cy - 6.6, cond_w, 13.2, fill="white", edge=PLUM, radius=2.0, lw=0.7)
    add_text("t5_cond_t", cond_x + cond_w / 2.0, t5_cy, "Target g", size=6.8, bold=True, color=PLUM, ha="center",
             box=[cond_x, t5_cy - 6.6, cond_w, 13.2])
    add_line("t5_a1", [(cond_x + cond_w + 3.0, t5_cy), (t5_x + 93.0, t5_cy)], color=INK, lw=0.75, arrow=True)

    # Residual MLP + overhead control expression bypass into (+) node
    net5_w, _ = draw_mlp("t5_mlp", t5_x + 99.0, r2_top + 26.6, dx=15.0, dy=5.6, accent=PLUM, layers=(3, 4, 3))
    sum_cx = t5_x + 156.0
    add_line("t5_mlp_to_sum", [(t5_x + 99.0 + net5_w + 4.0, t5_cy), (sum_cx - 3.6, t5_cy)], color=INK, lw=0.75, arrow=True)
    draw_plus_node("t5_sum", sum_cx, t5_cy, r=3.4, color=PLUM)
    # Overhead residual skip from control cell
    skip_y = r2_top + 22.0
    add_line("t5_skip", [(ctrl_cx, t5_cy - 5.2), (ctrl_cx, skip_y), (sum_cx, skip_y), (sum_cx, t5_cy - 3.6)],
             color=BLUE, lw=0.65, arrow=True, dash="dashed")

    add_line("t5_sum_to_out", [(sum_cx + 3.6, t5_cy), (t5_x + 181.0, t5_cy)], color=INK, lw=0.75, arrow=True)
    # Predicted expression & shift profile bars
    bar_x0, bar_max_w = t5_x + 185.0, 62.0
    for bi, bw_pred in enumerate([48, 24, 55, 34]):
        by = r2_top + 24.3 + bi * 5.8
        add_rect(f"t5_bref_{bi}", bar_x0, by, bar_max_w, 4.0, fill="white", edge="#DCE3EC", radius=0.5, lw=0.4)
        add_rect(f"t5_bpred_{bi}", bar_x0, by, float(bw_pred), 4.0, fill=OBS_COL if bi % 2 == 0 else PRED_COL, edge="none", radius=0.5, lw=0)

    # Flow arrow c -> d
    add_line("c_to_d", [(259.37, r2_top + r2_h), (259.37, r2_top + r2_h + 21.0)], color=INK, lw=0.85, arrow=True)

    # =========================================================================
    # PANEL d: Primary metrics (Tasks 1-5) + cross-task audits (6) in ONE aligned 2x3 grid.
    #          Each task card features 3 crisp white rounded plot-wells for balanced visual density.
    # =========================================================================
    d_head_y = r2_top + r2_h + 11.5
    draw_section_header("d", "Primary metrics & cross-task audits", y=d_head_y)

    # One shared legend for the Observed / Predicted colors used by Tasks 4-5
    leg_y = d_head_y + 0.3
    add_circle("d_leg_obs_dot", 426.0, leg_y, 2.0, fill=OBS_COL)
    add_text("d_leg_obs", 431.0, leg_y, "Observed", size=7.0, color=MUTED)
    add_circle("d_leg_prd_dot", 472.0, leg_y, 2.0, fill=PRED_COL)
    add_text("d_leg_prd", 477.0, leg_y, "Predicted", size=7.0, color=MUTED)

    th = 58.0                                  # tile height
    d_gap_x, d_gap_y = 4.5, 4.0
    d_top = d_head_y + 10.0                    # row 1: Tasks 1-3
    d_top2 = d_top + th + d_gap_y              # row 2: Tasks 4-5 + audits (6)
    col_w = [163.0, 167.0, 159.74]             # aligned 2x3 column widths
    col_d_x = [10.0, 10.0 + col_w[0] + d_gap_x, 10.0 + col_w[0] + col_w[1] + 2 * d_gap_x]
    wh = 22.0                                  # uniform plot-well height

    def metric(uid: str, x: float, y: float, text: str) -> None:
        add_text(uid, x, y, text, size=7.05, color=INK, ha="center")

    def tile(idx: int, x: float, y: float, w: float, fill: str, edge: str, title: str, col: str,
             dash: str = "solid") -> None:
        add_rect(f"d_card_{idx}", x, y, w, th, fill=fill, edge=edge, radius=3.0, lw=0.7, dash=dash)
        add_text(f"d{idx}_hdr", x + 7.0, y + 7.6, f"{idx}  {title}", size=7.8, bold=True, color=col)

    def plot_well(uid: str, x: float, y: float, w: float, edge: str) -> float:
        add_rect(uid, x, y, w, wh, fill="white", edge=edge, radius=2.4, lw=0.58)
        return x + w / 2.0

    # --- 1  Clustering (3 plot-wells: Partition barcode | Hungarian 1:1 matching | Silhouette hulls) ---
    x1, w1 = col_d_x[0], col_w[0]
    tile(1, x1, d_top, w1, T1_BG, T1_EDGE, "Clustering", BLUE)
    wy1 = d_top + 14.2
    c1a = plot_well("d1_w0", x1 + 6.0, wy1, 48.0, T1_EDGE)
    c1b = plot_well("d1_w1", x1 + 59.0, wy1, 43.0, T1_EDGE)
    c1c = plot_well("d1_w2", x1 + 107.0, wy1, 50.0, T1_EDGE)

    for j, (rc, fc) in enumerate(zip(
        [BLUE, BLUE, TEAL, TEAL, ROSE, ROSE],
        [BLUE, BLUE, TEAL, ROSE, TEAL, ROSE],
    )):
        tx = c1a - 19.3 + j * 6.6
        add_rect(f"d1_r_{j}", tx, wy1 + 3.0, 5.6, 5.6, fill=rc, edge="none", lw=0)
        add_line(f"d1_link_{j}", [(tx + 2.8, wy1 + 9.0), (tx + 2.8, wy1 + 13.0)], color="#94A3B8", lw=0.45)
        add_rect(f"d1_f_{j}", tx, wy1 + 13.4, 5.6, 5.6, fill=fc, edge="none", lw=0)
    metric("d1_m1", c1a, d_top + 46.8, "ARI · NMI ↑")

    l_cols = [TEAL, BLUE, ROSE]
    r_cols = [BLUE, TEAL, ROSE]
    perm = [1, 0, 2]
    for ri_m in range(3):
        ly = wy1 + 5.0 + ri_m * 6.0
        ry_m = wy1 + 5.0 + perm[ri_m] * 6.0
        add_line(f"d1_hm_{ri_m}", [(c1b - 8.5, ly), (c1b + 8.0, ry_m)], color=SUBTLE, lw=0.6, arrow=True)
        add_circle(f"d1_hl_{ri_m}", c1b - 11.2, ly, 2.0, fill=l_cols[ri_m])
        add_rect(f"d1_hr_{ri_m}", c1b + 9.2, ry_m - 2.0, 4.0, 4.0, fill="white", edge=r_cols[ri_m], radius=0.6, lw=0.75)
    metric("d1_m2", c1b, d_top + 46.8, "Accuracy ↑")

    for grp, (gcx, gcol, gtint) in enumerate([(c1c - 13.5, BLUE, BLUE_TINT), (c1c + 13.5, TEAL, TEAL_TINT)]):
        add_circle(f"d1_hull_{grp}", gcx, wy1 + 11.0, 6.8, fill=gtint, edge=gcol, lw=0.55, dash="dashed")
        for pj, (dx, dy) in enumerate([(-2.6, -2.2), (2.4, -2.0), (0.0, 0.2), (-2.0, 2.8), (2.6, 2.5)]):
            add_circle(f"d1_cl_{grp}_{pj}", gcx + dx, wy1 + 11.0 + dy, 1.35, fill=gcol)
    add_line("d1_gap_r", [(c1c - 5.5, wy1 + 11.0), (c1c + 5.5, wy1 + 11.0)], color=INK, lw=0.65, arrow=True)
    add_line("d1_gap_l", [(c1c + 2.0, wy1 + 11.0), (c1c - 5.5, wy1 + 11.0)], color=INK, lw=0.65, arrow=True)
    metric("d1_m3", c1c, d_top + 46.8, "Scaled ASW ↑")

    # --- 2  Batch assessment (3 plot-wells: k-NN donor diversity | kBET ratios | Pure connected subgraphs) ---
    x2, w2 = col_d_x[1], col_w[1]
    tile(2, x2, d_top, w2, T2_BG, T2_EDGE, "Batch assessment", ROSE)
    c2a = plot_well("d2_w0", x2 + 6.0, wy1, 45.0, T2_EDGE)
    c2b = plot_well("d2_w1", x2 + 56.0, wy1, 42.0, T2_EDGE)
    c2c = plot_well("d2_w2", x2 + 103.0, wy1, 58.0, T2_EDGE)

    wy2_mid = wy1 + 11.0
    add_circle("d2_knn_ring", c2a, wy2_mid, 8.2, fill="#F8FAFC", edge=SUBTLE, lw=0.55, dash="dashed")
    for ni in range(6):
        ang = ni * math.pi / 3.0 + 0.25
        nx, ny = c2a + 5.8 * math.cos(ang), wy2_mid + 5.8 * math.sin(ang)
        add_line(f"d2_sp_{ni}", [(c2a, wy2_mid), (nx, ny)], color="#CBD5E1", lw=0.45)
        if ni % 2 == 0:
            add_circle(f"d2_nb_{ni}", nx, ny, 1.5, fill=BLUE)
        else:
            add_rect(f"d2_ns_{ni}", nx - 1.4, ny - 1.4, 2.8, 2.8, fill="white", edge=TEAL, lw=0.7)
    add_circle("d2_q", c2a, wy2_mid, 1.75, fill=ROSE)
    metric("d2_m1", c2a, d_top + 46.8, "Scaled iLISI ↑")

    for j, frac in enumerate([0.45, 0.54, 0.48]):
        bx, by = c2b - 15.5, wy1 + 3.4 + j * 5.6
        add_rect(f"d2_bb_{j}", bx, by, 31.0 * frac, 4.0, fill=BLUE, edge="none", lw=0)
        add_rect(f"d2_bt_{j}", bx + 31.0 * frac, by, 31.0 * (1.0 - frac), 4.0, fill=TEAL, edge="none", lw=0)
    add_line("d2_exp", [(c2b, wy1 + 1.8), (c2b, wy1 + wh - 1.8)], color=ROSE, lw=0.75, dash="dashed")
    metric("d2_m2", c2b, d_top + 46.8, "kBET ↑")

    add_line("d2_sep", [(c2c, wy1 + 2.8), (c2c, wy1 + wh - 2.8)], color=HAIRLINE, lw=0.55, dash="dashed")
    for grp, (gcx, gcol) in enumerate([(c2c - 14.0, BLUE), (c2c + 14.0, TEAL)]):
        nodes = [
            (gcx - 7.0, wy1 + 6.2), (gcx + 6.5, wy1 + 5.8),
            (gcx, wy1 + 11.0),
            (gcx - 5.5, wy1 + 16.2), (gcx + 6.8, wy1 + 16.0),
        ]
        for n1, n2 in [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]:
            add_line(f"d2_ge_{grp}_{n1}_{n2}", [nodes[n1], nodes[n2]], color=gcol, lw=0.52)
        for ni, (nx, ny) in enumerate(nodes):
            if ni % 2 == 0:
                add_circle(f"d2_gn_{grp}_{ni}", nx, ny, 1.5, fill=gcol)
            else:
                add_rect(f"d2_gs_{grp}_{ni}", nx - 1.4, ny - 1.4, 2.8, 2.8, fill="white", edge=gcol, lw=0.65)
    metric("d2_m3", c2c, d_top + 42.6, "Scaled cLISI ↑")
    metric("d2_m4", c2c, d_top + 51.6, "Graph conn. ↑")

    # --- 3  Annotation (3 plot-wells: 3x3 confusion matrix | Balanced recall | Macro averaging) ---
    x3, w3 = col_d_x[2], col_w[2]
    tile(3, x3, d_top, w3, T3_BG, T3_EDGE, "Annotation", TEAL)
    c3a = plot_well("d3_w0", x3 + 5.5, wy1, 39.0, T3_EDGE)
    c3b = plot_well("d3_w1", x3 + 49.0, wy1, 49.0, T3_EDGE)
    c3c = plot_well("d3_w2", x3 + 102.5, wy1, 51.74, T3_EDGE)

    cm_fills = [
        [TEAL, "#F8FAFC", "#F4C7D7"],
        ["#E6F4F1", "#229A86", "#F8FAFC"],
        ["#F8FAFC", "#F4C7D7", TEAL],
    ]
    for r in range(3):
        for c in range(3):
            add_rect(
                f"d3_cm_{r}_{c}",
                c3a - 8.5 + c * 5.8,
                wy1 + 2.5 + r * 5.8,
                5.4,
                5.4,
                fill=cm_fills[r][c],
                edge="#CBD5E1",
                lw=0.35,
            )
    metric("d3_m1", c3a, d_top + 46.8, "Accuracy ↑")

    for j, (frac, ccol) in enumerate([(0.88, BLUE), (0.80, TEAL), (0.74, ROSE)]):
        bx, by = c3b - 17.5, wy1 + 3.4 + j * 5.6
        add_rect(f"d3_tr_{j}", bx, by, 35.0, 4.0, fill="#F1F5F9", edge="#DCE3EC", radius=0.5, lw=0.35)
        add_rect(f"d3_br_{j}", bx, by, 35.0 * frac, 4.0, fill=ccol, edge="none", radius=0.5, lw=0)
    metric("d3_m2", c3b, d_top + 46.8, "Balanced acc. ↑")

    for j, ccol in enumerate([BLUE, TEAL, ROSE]):
        yy = wy1 + 5.0 + j * 6.0
        add_rect(f"d3_cb_{j}", c3c - 19.5, yy - 2.1, 10.5, 4.2, fill=ccol, edge="none", radius=0.6, lw=0)
        add_line(f"d3_ca_{j}", [(c3c - 8.5, yy), (c3c + 3.5, wy1 + 11.0)], color="#94A3B8", lw=0.55)
    add_circle("d3_mean", c3c + 3.5, wy1 + 11.0, 2.3, fill="white", edge=TEAL, lw=0.75)
    add_circle("d3_mean_in", c3c + 3.5, wy1 + 11.0, 0.85, fill=TEAL)
    add_line("d3_to_macro", [(c3c + 6.2, wy1 + 11.0), (c3c + 18.5, wy1 + 11.0)], color=TEAL, lw=0.7, arrow=True)
    metric("d3_m3", c3c, d_top + 42.6, "Macro-F1 ↑")
    metric("d3_m4", c3c, d_top + 51.6, "Macro-prec. ↑")

    # --- 4  Reconstruction (3 plot-wells: Matched profile curves | 3-gene MAE | Top-50 HVG overlap) ---
    x4, w4 = col_d_x[0], col_w[0]
    tile(4, x4, d_top2, w4, T4_BG, T4_EDGE, "Reconstruction", AMBER)
    wy4 = d_top2 + 14.2
    c4a = plot_well("d4_w0", x4 + 6.0, wy4, 48.0, T4_EDGE)
    c4b = plot_well("d4_w1", x4 + 59.0, wy4, 38.0, T4_EDGE)
    c4c = plot_well("d4_w2", x4 + 102.0, wy4, 55.0, T4_EDGE)

    add_line("d4_axes", [(c4a - 18.5, wy4 + 3.2), (c4a - 18.5, wy4 + 18.8), (c4a + 19.5, wy4 + 18.8)], color="#CBD5E1", lw=0.5)
    obs_pts = [(c4a - 15.5 + j * 6.4, wy4 + 7.5 + v) for j, v in enumerate([0.8, -2.5, 2.4, -1.8, 1.2, -2.0])]
    prd_pts = [(c4a - 15.5 + j * 6.4, wy4 + 14.2 + v) for j, v in enumerate([1.2, -1.8, 2.0, -2.2, 1.5, -1.4])]
    add_line("d4_obs_p", obs_pts, color=OBS_COL, lw=0.90)
    add_line("d4_prd_p", prd_pts, color=PRED_COL, lw=0.90)
    for j, (ox, oy) in enumerate(obs_pts):
        add_circle(f"d4_od_{j}", ox, oy, 1.05, fill=OBS_COL)
        px_d, py_d = prd_pts[j]
        add_circle(f"d4_pd_{j}", px_d, py_d, 1.05, fill=PRED_COL)
    metric("d4_m1", c4a, d_top2 + 42.6, "Pearson r ↑")
    metric("d4_m2", c4a, d_top2 + 51.6, "Spearman ρ ↑")

    for j, (o1, o2) in enumerate([(-10.5, 3.5), (-4.5, 9.5), (-8.5, 6.0)]):
        yy = wy4 + 5.2 + j * 5.8
        add_line(f"d4_eg_{j}", [(c4b + o1, yy), (c4b + o2, yy)], color="#94A3B8", lw=0.7)
        add_circle(f"d4_er_{j}", c4b + o1, yy, 1.6, fill=OBS_COL)
        add_circle(f"d4_ep_{j}", c4b + o2, yy, 1.6, fill=PRED_COL)
    metric("d4_m3", c4b, d_top2 + 46.8, "MAE ↓")

    vy4 = wy4 + 11.0
    add_circle("d4_v_obs", c4c - 5.0, vy4, 7.6, fill="#F1F5F9", edge=OBS_COL, lw=0.75)
    add_circle("d4_v_prd", c4c + 5.0, vy4, 7.6, fill="none", edge=PRED_COL, lw=0.75)
    for gi, (gx, gy, gc) in enumerate([
        (c4c - 8.8, vy4 - 2.0, OBS_COL), (c4c - 8.4, vy4 + 2.2, OBS_COL),
        (c4c, vy4 - 2.8, TEAL), (c4c, vy4, TEAL), (c4c, vy4 + 2.8, TEAL),
        (c4c + 8.8, vy4 - 2.0, PRED_COL), (c4c + 8.4, vy4 + 2.2, PRED_COL),
    ]):
        add_circle(f"d4_gd_{gi}", gx, gy, 1.15, fill=gc)
    metric("d4_m4", c4c, d_top2 + 46.8, "Top-50 overlap ↑")

    # --- 5  Perturbation (3 plot-wells: Paired shift bars | 3-gene MAE Δ | Top-DE overlap DES) ---
    x5, w5 = col_d_x[1], col_w[1]
    tile(5, x5, d_top2, w5, T5_BG, T5_EDGE, "Perturbation", PLUM)
    c5a = plot_well("d5_w0", x5 + 6.0, wy4, 54.0, T5_EDGE)
    c5b = plot_well("d5_w1", x5 + 65.0, wy4, 42.0, T5_EDGE)
    c5c = plot_well("d5_w2", x5 + 112.0, wy4, 49.0, T5_EDGE)

    z0_y = wy4 + 11.2
    add_line("d5_zero", [(c5a - 23.0, z0_y), (c5a + 23.0, z0_y)], color=SUBTLE, lw=0.6, dash="dashed")
    for j, (vo, vp) in enumerate([(7.0, 4.8), (-5.5, -3.0), (4.8, 6.8), (-6.6, -5.0)]):
        for k, (val, col) in enumerate([(vo, OBS_COL), (vp, PRED_COL)]):
            bx = c5a - 20.0 + j * 11.0 + k * 4.1
            by = z0_y - max(val, 0.0)
            add_rect(f"d5_bar_{j}_{k}", bx, by, 3.3, abs(val), fill=col, edge="none", lw=0)
    metric("d5_m1", c5a, d_top2 + 42.6, "Pearson Δ ↑")
    metric("d5_m2", c5a, d_top2 + 51.6, "Spearman Δ ↑")

    for j, (o1, o2) in enumerate([(-10.0, 3.5), (-4.5, 9.0), (-8.0, 6.0)]):
        yy = wy4 + 5.2 + j * 5.8
        add_line(f"d5_eg_{j}", [(c5b + o1, yy), (c5b + o2, yy)], color="#94A3B8", lw=0.7)
        add_circle(f"d5_er_{j}", c5b + o1, yy, 1.6, fill=OBS_COL)
        add_circle(f"d5_ep_{j}", c5b + o2, yy, 1.6, fill=PRED_COL)
    metric("d5_m3", c5b, d_top2 + 46.8, "MAE Δ ↓")

    vy5 = wy4 + 11.0
    add_circle("d5_vo", c5c - 5.0, vy5, 7.6, fill="#F1F5F9", edge=OBS_COL, lw=0.75)
    add_circle("d5_vp", c5c + 5.0, vy5, 7.6, fill="none", edge=PRED_COL, lw=0.75)
    for gi, (gx, gy, gc) in enumerate([
        (c5c - 8.8, vy5 - 2.0, OBS_COL), (c5c - 8.4, vy5 + 2.2, OBS_COL),
        (c5c, vy5 - 2.8, PLUM), (c5c, vy5, PLUM), (c5c, vy5 + 2.8, PLUM),
        (c5c + 8.8, vy5 - 2.0, PRED_COL), (c5c + 8.4, vy5 + 2.2, PRED_COL),
    ]):
        add_circle(f"d5_gd_{gi}", gx, gy, 1.15, fill=gc)
    metric("d5_m4", c5c, d_top2 + 46.8, "DES ↑")

    # --- 6  Cross-task audits (diagnostic; outside the 20 ranked slots) -------
    x6, w6 = col_d_x[2], col_w[2]
    tile(6, x6, d_top2, w6, AUDIT_BG, AUDIT_EDGE, "Cross-task audits", AUDIT_ACCENT, dash="dashed")
    chip_gap, chip_h = 4.0, 17.5
    chip_w = (w6 - 14.0 - chip_gap) / 2.0
    audits = [("Linear probes", TEAL), ("Scale audit", BLUE), ("Species transfer", PLUM), ("Provenance", ROSE)]
    for ai, (atitle, acol) in enumerate(audits):
        ax0 = x6 + 7.0 + (ai % 2) * (chip_w + chip_gap)
        ay0 = d_top2 + 15.8 + (ai // 2) * (chip_h + 3.8)
        add_rect(f"d6_chip_{ai}", ax0, ay0, chip_w, chip_h, fill="white", edge=BORDER, radius=2.4, lw=0.55)
        add_rect(f"d6_bar_{ai}", ax0 + 0.5, ay0 + 0.5, 2.6, chip_h - 1.0, fill=acol, edge="none", radius=1.1, lw=0)
        add_text(f"d6_t_{ai}", ax0 + 7.0, ay0 + chip_h / 2.0, atitle, size=6.95, color=INK)


# ---------------------------------------------------------------------------
# Backend 1: Matplotlib Vector Renderer (PDF / SVG / 600-DPI RGB PNG)
# ---------------------------------------------------------------------------

def render_matplotlib() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(W / 72.0, H / 72.0), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="white")
    ax.set_xlim(0.0, W)
    ax.set_ylim(H, 0.0)
    ax.axis("off")

    for s in SCENE:
        k = s["kind"]
        if k == "rect":
            fc = "none" if s["fill"] == "none" else s["fill"]
            ec = "none" if s["edge"] == "none" or s["lw"] == 0 else s["edge"]
            ls = "--" if s["dash"] == "dashed" else "-"
            if s["radius"] > 0:
                patch = FancyBboxPatch(
                    (s["x"], s["y"]),
                    s["w"],
                    s["h"],
                    boxstyle=f"round,pad=0,rounding_size={s['radius']}",
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=s["lw"],
                    linestyle=ls,
                )
            else:
                patch = Rectangle(
                    (s["x"], s["y"]),
                    s["w"],
                    s["h"],
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=s["lw"],
                    linestyle=ls,
                )
            patch.set_gid(s["id"])
            ax.add_patch(patch)

        elif k == "circle":
            fc = "none" if s["fill"] == "none" else s["fill"]
            ec = "none" if s["edge"] == "none" or s["lw"] == 0 else s["edge"]
            ls = "--" if s["dash"] == "dashed" else "-"
            patch = Circle(
                (s["x"], s["y"]),
                s["r"],
                facecolor=fc,
                edgecolor=ec,
                linewidth=s["lw"],
                linestyle=ls,
            )
            patch.set_gid(s["id"])
            ax.add_patch(patch)

        elif k == "diamond":
            r = s["r"]
            cx, cy = s["x"], s["y"]
            verts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            patch = Polygon(
                verts,
                closed=True,
                facecolor=s["fill"],
                edgecolor=s["edge"],
                linewidth=s["lw"],
            )
            patch.set_gid(s["id"])
            ax.add_patch(patch)

        elif k == "path":
            fc = "none" if s["fill"] == "none" else s["fill"]
            ec = "none" if s["edge"] == "none" or s["lw"] <= 0 else s["edge"]
            ls = "--" if s["dash"] == "dashed" else "-"
            verts: list[tuple[float, float]] = []
            codes: list[int] = []
            for cmd in s["commands"]:
                op = cmd[0]
                if op == "M":
                    codes.append(MplPath.MOVETO)
                    verts.append((float(cmd[1]), float(cmd[2])))
                elif op == "L":
                    codes.append(MplPath.LINETO)
                    verts.append((float(cmd[1]), float(cmd[2])))
                elif op == "C":
                    codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                    verts.extend([
                        (float(cmd[1]), float(cmd[2])),
                        (float(cmd[3]), float(cmd[4])),
                        (float(cmd[5]), float(cmd[6])),
                    ])
                elif op == "Z":
                    codes.append(MplPath.CLOSEPOLY)
                    verts.append((0.0, 0.0))
            patch = PathPatch(
                MplPath(verts, codes),
                facecolor=fc,
                edgecolor=ec,
                linewidth=s["lw"],
                linestyle=ls,
                capstyle="round",
                joinstyle="round",
            )
            patch.set_gid(s["id"])
            ax.add_patch(patch)

        elif k == "line":
            pts = s["points"]
            ls = (0, (3.0, 2.2)) if s["dash"] == "dashed" else "-"
            for j, (p, q) in enumerate(zip(pts, pts[1:])):
                is_last = j == len(pts) - 2
                if s["arrow"] and is_last:
                    arr = FancyArrowPatch(
                        p,
                        q,
                        arrowstyle="-|>",
                        mutation_scale=6.2,
                        linewidth=s["lw"],
                        color=s["color"],
                        linestyle=ls,
                        shrinkA=0,
                        shrinkB=0,
                    )
                    arr.set_gid(f"{s['id']}_{j}")
                    ax.add_patch(arr)
                else:
                    ax.plot(
                        [p[0], q[0]],
                        [p[1], q[1]],
                        color=s["color"],
                        lw=s["lw"],
                        linestyle=ls,
                        solid_capstyle="round",
                        gid=f"{s['id']}_{j}",
                    )

        elif k == "text":
            t = ax.text(
                s["x"],
                s["y"],
                s["text"],
                fontsize=s["size"],
                fontweight="bold" if s["bold"] else "normal",
                fontstyle="italic" if s["italic"] else "normal",
                color=s["color"],
                ha=s["ha"],
                va=s["va"],
            )
            t.set_gid(s["id"])

    return fig, ax


# ---------------------------------------------------------------------------
# Backend 2: 100% Native Editable PowerPoint Renderer (Zero Raster Shapes!)
# ---------------------------------------------------------------------------

def to_rgb_color(val: str) -> RGBColor:
    if val == "white":
        return RGBColor(0xFF, 0xFF, 0xFF)
    return RGBColor.from_string(val.removeprefix("#").upper())


def sample_bezier_commands(commands: list[tuple], steps: int = 16) -> tuple[list[tuple[float, float]], bool]:
    pts: list[tuple[float, float]] = []
    closed = False
    cx, cy = 0.0, 0.0
    for cmd in commands:
        op = cmd[0]
        if op == "M":
            cx, cy = float(cmd[1]), float(cmd[2])
            pts.append((cx, cy))
        elif op == "L":
            cx, cy = float(cmd[1]), float(cmd[2])
            pts.append((cx, cy))
        elif op == "C":
            x1, y1, x2, y2, x3, y3 = (float(v) for v in cmd[1:])
            for step in range(1, steps + 1):
                t = step / steps
                mt = 1.0 - t
                px = mt**3 * cx + 3.0 * mt**2 * t * x1 + 3.0 * mt * t**2 * x2 + t**3 * x3
                py = mt**3 * cy + 3.0 * mt**2 * t * y1 + 3.0 * mt * t**2 * y2 + t**3 * y3
                pts.append((px, py))
            cx, cy = x3, y3
        elif op == "Z":
            closed = True
    return pts, closed


def export_native_pptx(fig: plt.Figure, ax: plt.Axes, pptx_path: Path) -> int:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_artists = {t.get_gid(): t for t in ax.texts}

    prs = Presentation()
    prs.slide_width = Pt(W)
    prs.slide_height = Pt(H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    for s in SCENE:
        k = s["kind"]
        if k in ("rect", "circle", "diamond", "path"):
            if k == "path":
                ppts, closed = sample_bezier_commands(s["commands"], steps=16)
                fb = slide.shapes.build_freeform(Pt(ppts[0][0]), Pt(ppts[0][1]))
                fb.add_line_segments([(Pt(px), Pt(py)) for px, py in ppts[1:]], close=closed)
                shape = fb.convert_to_shape()
            else:
                if k == "rect":
                    stype = MSO_SHAPE.ROUNDED_RECTANGLE if s["radius"] > 0 else MSO_SHAPE.RECTANGLE
                    x, y, w, h = s["x"], s["y"], s["w"], s["h"]
                elif k == "circle":
                    stype = MSO_SHAPE.OVAL
                    x, y, w, h = s["x"] - s["r"], s["y"] - s["r"], 2.0 * s["r"], 2.0 * s["r"]
                else:
                    stype = MSO_SHAPE.DIAMOND
                    x, y, w, h = s["x"] - s["r"], s["y"] - s["r"], 2.0 * s["r"], 2.0 * s["r"]

                shape = slide.shapes.add_shape(stype, Pt(x), Pt(y), Pt(max(w, 0.2)), Pt(max(h, 0.2)))
                if k == "rect" and s["radius"] > 0 and min(w, h) > 0:
                    shape.adjustments[0] = min(0.5, s["radius"] / min(w, h))

            if s["fill"] == "none":
                shape.fill.background()
            else:
                shape.fill.solid()
                shape.fill.fore_color.rgb = to_rgb_color(s["fill"])

            if s["edge"] == "none" or s["lw"] <= 0:
                shape.line.fill.background()
            else:
                shape.line.color.rgb = to_rgb_color(s["edge"])
                shape.line.width = Pt(s["lw"])
                if s.get("dash") == "dashed":
                    shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
            shape.shadow.inherit = False
            for st in shape._element.xpath("./*[local-name()='style']"):
                st.getparent().remove(st)
            shape.name = s["id"]

        elif k == "line":
            pts = s["points"]
            for j, (p, q) in enumerate(zip(pts, pts[1:])):
                conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Pt(p[0]), Pt(p[1]), Pt(q[0]), Pt(q[1]))
                conn.line.color.rgb = to_rgb_color(s["color"])
                conn.line.width = Pt(s["lw"])
                if s.get("dash") == "dashed":
                    conn.line.dash_style = MSO_LINE_DASH_STYLE.DASH
                if s["arrow"] and j == len(pts) - 2:
                    end = OxmlElement("a:tailEnd")
                    end.set("type", "triangle")
                    end.set("w", "sm")
                    end.set("len", "sm")
                    conn.line._get_or_add_ln().append(end)
                conn.shadow.inherit = False
                for st in conn._element.xpath("./*[local-name()='style']"):
                    st.getparent().remove(st)
                conn.name = f"{s['id']}_{j}"

        elif k == "text":
            ha, va = s["ha"], s["va"]
            if "layout_box" in s:
                bx, by, bw, bh = s["layout_box"]
            else:
                art = text_artists[s["id"]]
                bb = art.get_window_extent(renderer)
                bw = bb.width * 72.0 / fig.dpi + 5.0
                bh = s["size"] * 1.32
                bx = s["x"] - ({"left": 0.0, "center": bw / 2.0, "right": bw}[ha])
                by = s["y"] - ({"top": 0.0, "center": bh / 2.0, "bottom": bh}[va])

            tb = slide.shapes.add_textbox(Pt(bx), Pt(by), Pt(bw), Pt(bh))
            tf = tb.text_frame
            tf.clear()
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            tf.word_wrap = False
            tf.vertical_anchor = {
                "top": MSO_ANCHOR.TOP,
                "center": MSO_ANCHOR.MIDDLE,
                "bottom": MSO_ANCHOR.BOTTOM,
            }[va]
            p = tf.paragraphs[0]
            p.space_before = p.space_after = Pt(0)
            p.line_spacing = 1.0
            p.alignment = {
                "left": PP_ALIGN.LEFT,
                "center": PP_ALIGN.CENTER,
                "right": PP_ALIGN.RIGHT,
            }[ha]
            run = p.add_run()
            run.text = s["text"]
            run.font.name = "Arial"
            run.font.size = Pt(s["size"])
            run.font.bold = s["bold"]
            run.font.italic = s["italic"]
            run.font.color.rgb = to_rgb_color(s["color"])
            tb.name = s["id"]

    slide.notes_slide.notes_text_frame.text = (
        CAPTION_LATEX
        + "\n\n100% native editable PowerPoint vector shapes and Arial textboxes (0 raster images)."
    )
    prs.save(pptx_path)
    return len(slide.shapes)


# ---------------------------------------------------------------------------
# Backend 3: Bonus Hybrid SVG-Blip + 600-DPI PNG PPTX (Single-Object Vector)
# ---------------------------------------------------------------------------

def export_hybrid_svg_pptx(png_path: Path, outlined_svg_path: Path, pptx_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Pt(W)
    prs.slide_height = Pt(H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    pic = slide.shapes.add_picture(str(png_path), Pt(0), Pt(0), width=Pt(W), height=Pt(H))
    part = Part(
        PackURI("/ppt/media/Figure1_vector.svg"),
        "image/svg+xml",
        prs.part.package,
        outlined_svg_path.read_bytes(),
    )
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
    slide.notes_slide.notes_text_frame.text = CAPTION_LATEX
    prs.save(pptx_path)


# ---------------------------------------------------------------------------
# Strict Automated Geometry, Collision & Content QA
# ---------------------------------------------------------------------------

def run_qa(fig: plt.Figure, ax: plt.Axes, pdf_path: Path, native_pptx_path: Path, n_shapes: int) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 72.0 / fig.dpi

    # 1. Text bounds & pairwise collision check
    text_boxes = []
    out_of_bounds = []
    for t in ax.texts:
        bb = t.get_window_extent(renderer)
        x0, y0, x1, y1 = bb.x0 * scale, bb.y0 * scale, bb.x1 * scale, bb.y1 * scale
        if x0 < -0.5 or y0 < -0.5 or x1 > W + 0.5 or y1 > H + 0.5:
            out_of_bounds.append({"id": t.get_gid(), "text": t.get_text(), "bbox_pt": [x0, y0, x1, y1]})
        text_boxes.append((t.get_gid(), t.get_text(), bb))

    overlaps = []
    for i, (id1, txt1, bb1) in enumerate(text_boxes):
        # Shrink bbox by 0.6 px to avoid false positives on adjacent baselines
        s1 = bb1.shrunk(0.96, 0.88)
        for id2, txt2, bb2 in text_boxes[i + 1:]:
            s2 = bb2.shrunk(0.96, 0.88)
            if s1.overlaps(s2):
                overlaps.append({"a": f"{id1}: {txt1}", "b": f"{id2}: {txt2}"})

    assert not out_of_bounds, f"Out-of-bounds text: {out_of_bounds}"
    assert not overlaps, f"Overlapping text detected: {overlaps}"

    # 2. Vector PDF content & zero-raster verification
    required_strings = [
        # All 13 scFMs + 3 baselines
        "CellFM", "CellPLM", "GeneCompass", "Geneformer", "LangCell", "Nicheformer",
        "Tabula", "TranscriptFormer", "UCE", "scFoundation", "scGPT", "scLong", "scPRINT2",
        "PCA", "UMAP", "scVI",
        # Updated zip manuscript metrics (especially Task 2 Batch assessment!)
        "Scaled iLISI", "kBET", "Scaled cLISI", "Graph conn.",
        "ARI", "NMI", "Scaled ASW", "Macro-F1", "Balanced acc.",
        "Pearson r", "Spearman", "MAE", "Top-50 overlap",
        "Pearson Δ", "Spearman Δ", "MAE Δ", "DES",
        # Perturbation cohorts
        "Norman", "Adamson", "Replogle K562 / RPE1",
        # Merged Panel d: cross-task audit tile (replaces the former Panel e)
        "Cross-task audits", "Linear probes", "Provenance",
    ]
    forbidden_stale = ["Cell-type ASW", "Diagnostic & provenance audits", "input code"]  # stale labels must be gone

    with pymupdf.open(pdf_path) as doc:
        assert len(doc) == 1, "PDF must be 1 page"
        assert len(doc[0].get_images()) == 0, "PDF must contain 0 raster images"
        pdf_text = doc[0].get_text()
        missing = [s for s in required_strings if s not in pdf_text]
        assert not missing, f"Missing required labels in PDF: {missing}"
        stale = [s for s in forbidden_stale if s in pdf_text]
        assert not stale, f"Stale labels still in PDF: {stale}"
        assert abs(doc[0].rect.width - W) < 0.05 and abs(doc[0].rect.height - H) < 0.05

    # 3. Native PPTX zero-raster verification
    deck = Presentation(native_pptx_path)
    assert len(deck.slides) == 1
    assert len(deck.slides[0].shapes) == n_shapes
    assert all(sh.shape_type != 13 for sh in deck.slides[0].shapes), "Native PPTX must contain 0 picture shapes"

    return {
        "status": "PASS",
        "dimensions_pt": [W, H],
        "dimensions_mm": [round(W * 25.4 / 72.0, 2), round(H * 25.4 / 72.0, 2)],
        "minimum_font_pt": min(t.get_fontsize() for t in ax.texts),
        "text_elements": len(ax.texts),
        "text_overlaps": 0,
        "out_of_bounds": 0,
        "pdf_raster_images": 0,
        "native_pptx_shapes": n_shapes,
        "native_pptx_raster_images": 0,
        "all_16_methods_verified": True,
        "panels": "a-d (d merges primary metrics 1-5 with cross-task audits 6)",
        "batch_metrics_synced_to_zip_manuscript": ["Scaled iLISI", "kBET", "Scaled cLISI", "Graph conn."],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_scene()

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(rc):
        fig, ax = render_matplotlib()

        pdf_path = OUT_DIR / "Figure1.pdf"
        svg_path = OUT_DIR / "Figure1.svg"
        outlined_svg_path = OUT_DIR / "Figure1_outlined.svg"
        png_path = OUT_DIR / "Figure1.png"
        preview_path = OUT_DIR / "Figure1_preview.png"
        native_pptx_path = OUT_DIR / "Figure1.pptx"
        hybrid_pptx_path = OUT_DIR / "Figure1_svg_vector.pptx"

        fig.savefig(pdf_path, format="pdf", facecolor="white")
        fig.savefig(svg_path, format="svg", facecolor="white")
        with plt.rc_context({"svg.fonttype": "path"}):
            fig.savefig(outlined_svg_path, format="svg", facecolor="white")

        # Clean SVG DTD for network-free XML parsers
        for sp in (svg_path, outlined_svg_path):
            xml = re.sub(r"<!DOCTYPE[^>]*>\s*", "", sp.read_text(encoding="utf-8"))
            sp.write_text(xml, encoding="utf-8")

        fig.savefig(png_path, format="png", dpi=600, facecolor="white")
        fig.savefig(preview_path, format="png", dpi=300, facecolor="white")
        for p, dpi in [(png_path, 600), (preview_path, 300)]:
            with Image.open(p) as im:
                assert im.getchannel("A").getextrema() == (255, 255)
                im.convert("RGB").save(p, dpi=(dpi, dpi))

        n_shapes = export_native_pptx(fig, ax, native_pptx_path)
        export_hybrid_svg_pptx(png_path, outlined_svg_path, hybrid_pptx_path)
        qa_report = run_qa(fig, ax, pdf_path, native_pptx_path, n_shapes)
        plt.close(fig)

    shutil.copy2(OUT_DIR / "Figure1.pdf", REPO_ROOT / "figures" / "Figure1.pdf")
    shutil.copy2(OUT_DIR / "Figure1.pptx", REPO_ROOT / "figures" / "Figure1.pptx")
    (OUT_DIR / "caption.tex").write_text(CAPTION_LATEX + "\n", encoding="utf-8")
    (OUT_DIR / "scene.json").write_text(
        json.dumps({"width_pt": W, "height_pt": H, "elements": SCENE}, indent=2),
        encoding="utf-8",
    )
    qa_report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT_DIR / "qa_report.json").write_text(json.dumps(qa_report, indent=2), encoding="utf-8")
    print(json.dumps(qa_report, indent=2))


if __name__ == "__main__":
    main()
