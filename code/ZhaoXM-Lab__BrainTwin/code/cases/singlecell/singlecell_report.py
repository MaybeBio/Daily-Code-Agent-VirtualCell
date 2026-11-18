"""
Single-cell annotation pipeline for Brain Chat UI.

Ported from v2_RAG/singlecell_Anno.ipynb:
  load h5ad -> detect raw/log scale -> preprocess -> Leiden -> DEG -> LLM annotate
  -> UMAP + composition pie chart
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import CODE as V2_RAG_DIR, DATA

DEFAULT_DEMO_H5AD = os.environ.get(
    "SINGLECELL_DEMO_H5AD",
    str(DATA / "singlecell" / "grubman_2019_alzheimer_entorhinal_cortex_clean.h5ad"),
)
JAKEL_H5AD = os.environ.get(
    "SINGLECELL_JAKEL_H5AD",
    str(DATA / "singlecell" / "jakel_2019_multiple_sclerosis_white_matter.h5ad"),
)
MAX_CELLS = int(os.environ.get("SINGLECELL_MAX_CELLS", "8000"))
DEMO_MAX_CELLS = int(os.environ.get("SINGLECELL_DEMO_MAX_CELLS", "5000"))
LEIDEN_RESOLUTION = float(os.environ.get("SINGLECELL_LEIDEN_RES", "0.5"))
N_TOP_GENES = int(os.environ.get("SINGLECELL_N_HVG", "2000"))
MARKERS_PER_CLUSTER = int(os.environ.get("SINGLECELL_MARKERS_PER_CLUSTER", "100"))

BATCH_KEY_CANDIDATES = (
    "batch",
    "sampleNumber",
    "sample",
    "sample_id",
    "donor_id",
    "library",
    "plate",
    "seq_batch",
)

CACHE_DIR = Path(os.environ.get("SINGLECELL_RESULTS_DIR", DATA / "singlecell" / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_PROMPT = """The following are the top marker genes for one scRNA-seq cluster:
{genes}

What is the most likely cell type?
Reply with ONLY one name: OPC, Microglia, Astrocyte, Oligodendrocyte, Excitatory neuron, Inhibitory neuron, or Endothelial cell.
Use OPC only if at least two of OPCML, PTPRZ1, TNR, VCAN, LHFPL3 are in the list.
Use Microglia if P2RY12, C1QB, CSF1R, CX3CR1, or CD74 is in the list and the OPC rule above is not met.
"""

NORMALIZE_SYSTEM = """You standardize scRNA-seq cluster cell-type labels to concise standard names."""

NORMALIZE_USER = """Standardize these cluster cell-type labels.

{lines}

Return ONLY a JSON object: {{"<cluster_id>": "<short cell type>", ...}}

Rules:
- Use concise standard brain cell type names only.
- Never use sentences or prefixes like "This cell is".
- Map long gene-specific descriptions to the closest standard type
  (e.g. "...-expressing cell" -> appropriate neuron/glia type).
- Keep OPC as OPC and Microglia as Microglia. Do not rename them to Oligodendrocyte or Immune cell.
- Preserve every cluster id from the input.
"""

STANDARD_CELL_TYPES = (
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "Oligodendrocyte progenitor cell",
    "OPC",
    "Microglia",
    "Endothelial cell",
    "Pericyte",
    "Immature neuron",
    "Choroid plexus cell",
    "Radial glia",
    "Cajal-Retzius cell",
    "Ependymal cell",
)

ANNOTATION_PATTERNS = (
    re.compile(r"^This cell is\s+(.+?)\.?\s*$", re.IGNORECASE),
    re.compile(r"^The cell'?s celltype is\s+(.+?)\.?\s*$", re.IGNORECASE),
    re.compile(r"^Cell type:\s*(.+?)\.?\s*$", re.IGNORECASE),
    re.compile(r"celltype is\s+(.+?)[\.\n]", re.IGNORECASE),
    re.compile(r"cell type is\s+(.+?)[\.\n]", re.IGNORECASE),
)


def save_uploaded_h5ad(upload_bytes: bytes, filename: str) -> str:
    suffix = ".h5ad" if filename.endswith(".h5ad") else ".h5ad"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(upload_bytes)
    handle.close()
    return handle.name


def _nonzero_expression_values(adata) -> np.ndarray:
    """Nonzero finite values from the whole matrix, not a leading gene block."""
    import scipy.sparse as sp

    matrix = adata.X
    if sp.issparse(matrix):
        values = np.asarray(matrix.data)
    else:
        array = np.asarray(matrix)
        n_genes = array.shape[1]
        if n_genes > 2000:
            gene_idx = np.linspace(0, n_genes - 1, 2000).astype(int)
            array = array[:, gene_idx]
        values = array.ravel()
    values = values[np.isfinite(values)]
    return values[values != 0]


def detect_expression_scale(adata) -> dict[str, Any]:
    """Heuristically distinguish raw counts vs log-normalized expression."""
    values = _nonzero_expression_values(adata)
    if values.size == 0:
        return {
            "scale": "unknown",
            "label": "Unknown",
            "max_value": 0.0,
            "median_value": 0.0,
            "integer_fraction": 0.0,
            "has_log1p_uns": "log1p" in adata.uns,
            "has_counts_layer": "counts" in adata.layers,
            "will_normalize": True,
            "detail": "Empty expression matrix; defaulting to normalize + log1p.",
        }

    max_val = float(np.max(values))
    median_val = float(np.median(values))
    integer_fraction = float(np.mean(np.abs(values - np.round(values)) < 1e-6))
    has_log1p_uns = "log1p" in adata.uns
    has_counts_layer = "counts" in adata.layers
    reasons: list[str] = []

    if has_log1p_uns:
        scale = "log_normalized"
        reasons.append("AnnData.uns contains log1p metadata")
    elif integer_fraction >= 0.85:
        scale = "raw_counts"
        reasons.append(
            f"nonzero values are integer-like ({integer_fraction:.2f}); "
            f"max={max_val:.1f}, nonzero median={median_val:.2f}"
        )
    elif has_counts_layer and max_val > 50:
        scale = "raw_counts"
        reasons.append("counts layer present and X looks like high-magnitude data")
    elif max_val > 30:
        scale = "raw_counts"
        reasons.append(f"nonzero max {max_val:.1f} is above log-normalized range")
    else:
        scale = "log_normalized"
        reasons.append(
            f"fractional nonzero values (integer fraction {integer_fraction:.2f}, "
            f"max {max_val:.2f})"
        )

    label = "Raw counts" if scale == "raw_counts" else "Log-normalized"
    will_normalize = scale == "raw_counts"

    return {
        "scale": scale,
        "label": label,
        "max_value": round(max_val, 3),
        "median_value": round(median_val, 3),
        "integer_fraction": round(integer_fraction, 3),
        "has_log1p_uns": has_log1p_uns,
        "has_counts_layer": has_counts_layer,
        "will_normalize": will_normalize,
        "detail": "; ".join(reasons),
    }


def infer_batch_key(adata) -> str | None:
    for key in BATCH_KEY_CANDIDATES:
        if key not in adata.obs.columns:
            continue
        n_unique = int(adata.obs[key].nunique(dropna=True))
        if 1 < n_unique <= max(50, adata.n_obs // 10):
            return key
    return None


def load_h5ad(path: str | Path, *, max_cells: int | None = None) -> tuple[Any, bool]:
    path = Path(path)
    if max_cells is None:
        return sc.read_h5ad(path), False

    backed = sc.read_h5ad(path, backed="r")
    total_cells = backed.n_obs
    if total_cells > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(total_cells, size=max_cells, replace=False)
        idx.sort()
        adata = backed[idx].to_memory()
        return adata, True
    adata = backed.to_memory()
    return adata, False


def preprocess_adata(adata, batch_key: str | None, scale_info: dict[str, Any]) -> Iterator[tuple[str, dict[str, str] | None]]:
    steps: dict[str, str] = {"expression_scale": scale_info["label"]}

    if scale_info.get("will_normalize"):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        steps["normalization"] = "normalize_total + log1p"
    else:
        steps["normalization"] = "skipped (already log-normalized)"
    yield f"Normalization done: {steps['normalization']}", None

    adata.raw = adata.copy()
    yield "Raw expression saved", None

    hvg_kwargs: dict[str, Any] = {"n_top_genes": N_TOP_GENES}
    if batch_key and batch_key in adata.obs.columns and adata.obs[batch_key].nunique(dropna=True) > 1:
        hvg_kwargs["batch_key"] = batch_key
    sc.pp.highly_variable_genes(adata, **hvg_kwargs)
    steps["hvg"] = f"{N_TOP_GENES} highly variable genes"
    yield f"HVG done: {steps['hvg']}", None

    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_comps)
    yield f"PCA done: {n_comps} components", None
    use_rep = "X_pca"
    if batch_key and batch_key in adata.obs.columns and adata.obs[batch_key].nunique(dropna=True) > 1:
        try:
            sc.external.pp.harmony_integrate(adata, batch_key, max_iter_harmony=20)
            use_rep = "X_pca_harmony"
            steps["integration"] = f"Harmony on {batch_key}"
        except Exception as exc:
            steps["integration"] = f"Harmony skipped ({exc})"
    else:
        steps["integration"] = "none (single batch or no batch key)"
    yield f"Integration done: {steps['integration']}", None

    sc.pp.neighbors(adata, use_rep=use_rep)
    yield "Neighbors done", None
    if "X_umap" not in adata.obsm or adata.obsm["X_umap"].shape[0] != adata.n_obs:
        sc.tl.umap(adata)
        steps["umap"] = "computed"
    else:
        steps["umap"] = "reused existing X_umap"
    yield f"UMAP done: {steps['umap']}", steps


def cluster_and_deg(adata, resolution: float = LEIDEN_RESOLUTION) -> Iterator[tuple[str, pd.DataFrame | None]]:
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
    n_clusters = int(adata.obs["leiden"].nunique())
    yield f"Leiden done: {n_clusters} clusters at resolution={resolution}", None
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=True)

    frames: list[pd.DataFrame] = []
    for cluster in sorted(adata.obs["leiden"].astype(str).unique(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        df = df.copy()
        df["group"] = str(cluster)
        if "logfoldchanges" in df.columns:
            positive = df[df["logfoldchanges"] > 0]
            if not positive.empty:
                df = positive
        frames.append(df.head(MARKERS_PER_CLUSTER))

    if not frames:
        raise ValueError("No marker genes found for Leiden clusters")
    markers = pd.concat(frames, ignore_index=True)
    yield f"Cluster markers done: {len(markers)} genes", markers


def parse_celltype(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return "Unknown"

    for pattern in ANNOTATION_PATTERNS:
        match = pattern.search(raw)
        if match:
            value = match.group(1).strip(" .\"'")
            if value:
                return _finalize_celltype_label(value)

    first_line = raw.split("\n")[0].strip(" .\"'")
    first_line = re.sub(r"^(This cell is|The cell type is)\s+", "", first_line, flags=re.IGNORECASE)
    return _finalize_celltype_label(first_line)


def _finalize_celltype_label(label: str) -> str:
    value = label.strip(" .\"'")
    value = re.sub(r"\s+", " ", value)
    if not value:
        return "Unknown"
    if len(value) > 80:
        value = value[:80].rsplit(" ", 1)[0]
    return value


def _extract_json_mapping(text: str) -> dict[str, str] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return {str(k): _finalize_celltype_label(str(v)) for k, v in payload.items()}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return {str(k): _finalize_celltype_label(str(v)) for k, v in payload.items()}
        except json.JSONDecodeError:
            return None
    return None


def normalize_cluster_labels(raw_mapping: dict[str, str]) -> dict[str, str]:
    from LLM_utils import CustomOpenAIProxyChat
    from langchain_core.messages import HumanMessage, SystemMessage
    from multiagent_system.config import Config

    cleaned = {str(k): parse_celltype(v) for k, v in raw_mapping.items()}
    if not cleaned:
        return cleaned

    lines = "\n".join(f"{cluster} | {label}" for cluster, label in sorted(cleaned.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])))
    standard_hint = ", ".join(STANDARD_CELL_TYPES)

    config = Config()
    llm = CustomOpenAIProxyChat(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model_id,
        streaming=False,
    )
    response = llm.invoke(
        [
            SystemMessage(content=NORMALIZE_SYSTEM),
            HumanMessage(
                content=NORMALIZE_USER.format(lines=lines)
                + f"\n\nPreferred standard names include: {standard_hint}"
            ),
        ]
    )
    content = getattr(response, "content", None) or str(response)
    normalized = _extract_json_mapping(content)
    if not normalized:
        return cleaned

    merged: dict[str, str] = {}
    for cluster, label in cleaned.items():
        merged[cluster] = normalized.get(cluster, normalized.get(str(cluster), label))
    return merged


def annotate_clusters_llm(top_markers: pd.DataFrame, adata) -> Iterator[tuple[str, dict[str, str]]]:
    from LLM_utils import CustomOpenAIProxyChat
    from langchain_core.messages import HumanMessage, SystemMessage
    from multiagent_system.config import Config

    config = Config()
    llm = CustomOpenAIProxyChat(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model_id,
        streaming=False,
    )

    mapping: dict[str, str] = {}
    clusters = sorted(top_markers["group"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    if top_markers.empty:
        clusters = sorted(adata.obs["leiden"].astype(str).unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    for cluster in clusters:
        if not top_markers.empty:
            genes = top_markers.loc[top_markers["group"].astype(str) == str(cluster), "names"].astype(str).tolist()
        else:
            genes = adata.var_names[:50].astype(str).tolist()
        if not genes:
            genes = adata.var_names[:50].astype(str).tolist()
        prompt = CLUSTER_PROMPT.format(genes=", ".join(genes[:50]))
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You annotate scRNA-seq clusters from marker genes. "
                        "Reply with the cell type name only."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        content = getattr(response, "content", None) or str(response)
        mapping[str(cluster)] = parse_celltype(content)
        yield f"Annotated cluster {cluster}: {mapping[str(cluster)]}", dict(mapping)


def save_annotated_h5ad(adata, report_id: str) -> str:
    path = CACHE_DIR / f"{report_id}.h5ad"
    adata.write_h5ad(path)
    return str(path)


def get_cached_h5ad_path(report_id: str) -> Path | None:
    path = CACHE_DIR / f"{report_id}.h5ad"
    return path if path.is_file() else None


def save_deg_csv(tables: list[pd.DataFrame], report_id: str) -> bool:
    if not tables:
        return False
    table = pd.concat(tables, ignore_index=True)
    padj = pd.to_numeric(table["padj"], errors="coerce") if "padj" in table.columns else None
    if padj is None:
        return False
    table = table.loc[padj < 0.05]
    if table.empty:
        return False
    path = CACHE_DIR / f"{report_id}_deg.csv"
    table.to_csv(path, index=False)
    return True


def get_cached_deg_path(report_id: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{32}", report_id or ""):
        return None
    path = CACHE_DIR / f"{report_id}_deg.csv"
    return path if path.is_file() else None


def apply_cluster_labels(adata, mapping: dict[str, str], obs_key: str = "predicted_cell_type") -> None:
    adata.obs[obs_key] = adata.obs["leiden"].astype(str).map(mapping).fillna("Unknown")


def build_cluster_rows(adata, mapping: dict[str, str], top_markers: pd.DataFrame) -> list[dict[str, Any]]:
    counts = adata.obs["leiden"].astype(str).value_counts()
    total = int(adata.n_obs)
    rows: list[dict[str, Any]] = []
    for cluster in sorted(mapping.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        n_cells = int(counts.get(cluster, 0))
        markers = top_markers.loc[top_markers["group"].astype(str) == str(cluster), "names"].astype(str).tolist()
        rows.append(
            {
                "cluster": cluster,
                "predicted_cell_type": mapping.get(cluster, "Unknown"),
                "n_cells": n_cells,
                "pct": round(100.0 * n_cells / total, 2) if total else 0.0,
                "degs": ", ".join(markers[:8]),
            }
        )
    return rows


_DISEASE_TOKENS = {"case", "disease", "ad", "ms", "patient", "affected"}
_CONTROL_TOKENS = {"control", "ctrl", "ct", "healthy", "hc", "normal"}
_GROUP_COLUMNS = (
    "case_control",
    "Condition",
    "condition",
    "oupSample.batchCond",
    "disease_status",
    "diagnosis",
    "status",
    "group",
)
_SKIP_CELL_TYPES = {"doublet", "doublets", "unknown", "unid", "unassigned", "undetermined"}
MIN_DEG_CELLS = 20
DEG_GENES = 15


def _group_pole(value: str) -> str | None:
    token = str(value).strip().lower()
    if token in _DISEASE_TOKENS:
        return "disease"
    if token in _CONTROL_TOKENS:
        return "control"
    return None


def find_disease_contrast(adata) -> dict[str, str] | None:
    columns = [col for col in _GROUP_COLUMNS if col in adata.obs.columns]
    columns.extend(col for col in adata.obs.columns if col not in columns)
    for column in columns:
        values = [str(value) for value in adata.obs[column].dropna().unique()]
        if len(values) != 2:
            continue
        poles = {value: _group_pole(value) for value in values}
        if set(poles.values()) != {"disease", "control"}:
            continue
        disease_value = next(value for value, pole in poles.items() if pole == "disease")
        control_value = next(value for value, pole in poles.items() if pole == "control")
        disease_name = ""
        if "disease" in adata.obs.columns and column != "disease":
            names = [str(value) for value in adata.obs["disease"].dropna().unique()]
            if len(names) == 1:
                disease_name = names[0]
        return {
            "column": column,
            "disease_value": disease_value,
            "control_value": control_value,
            "disease_name": disease_name or disease_value,
        }
    return None


def _deg_gene_table(frame: pd.DataFrame, *, higher_in_disease: bool) -> list[dict[str, Any]]:
    if frame.empty or "logfoldchanges" not in frame.columns:
        return []
    selected = frame[frame["logfoldchanges"] > 0] if higher_in_disease else frame[frame["logfoldchanges"] < 0]
    if "pvals_adj" in selected.columns:
        selected = selected[selected["pvals_adj"] < 0.05]
    selected = selected[np.isfinite(selected["logfoldchanges"])]
    if selected.empty:
        return []
    selected = selected.sort_values("logfoldchanges", ascending=not higher_in_disease)
    rows = []
    for record in selected.head(DEG_GENES).to_dict(orient="records"):
        logfc = record.get("logfoldchanges")
        padj = record.get("pvals_adj")
        rows.append(
            {
                "gene": str(record.get("names") or ""),
                "logfc": round(float(logfc), 3) if logfc is not None and np.isfinite(logfc) else None,
                "padj": round(float(padj), 4) if padj is not None and np.isfinite(padj) else None,
            }
        )
    return [row for row in rows if row["gene"]]


def _full_deg_table(frame: pd.DataFrame, cell_type: str, n_disease: int, n_control: int) -> pd.DataFrame:
    gene = frame["names"].astype(str) if "names" in frame.columns else pd.Series([""] * len(frame))
    return pd.DataFrame(
        {
            "cell_type": cell_type,
            "n_disease": n_disease,
            "n_control": n_control,
            "gene": gene.to_numpy(),
            "logfoldchange": frame["logfoldchanges"] if "logfoldchanges" in frame.columns else pd.NA,
            "score": frame["scores"] if "scores" in frame.columns else pd.NA,
            "pval": frame["pvals"] if "pvals" in frame.columns else pd.NA,
            "padj": frame["pvals_adj"] if "pvals_adj" in frame.columns else pd.NA,
        }
    )


def disease_group_counts(adata, contrast: dict[str, str]) -> dict[str, Any]:
    column = contrast["column"]
    disease_value = contrast["disease_value"]
    control_value = contrast["control_value"]
    labels = adata.obs["predicted_cell_type"].astype(str)
    groups = adata.obs[column].astype(str)
    by_cell_type = []
    for cell_type in labels.value_counts().index.astype(str):
        mask = labels == cell_type
        by_cell_type.append(
            {
                "cell_type": cell_type,
                "n_disease": int((groups.loc[mask] == disease_value).sum()),
                "n_control": int((groups.loc[mask] == control_value).sum()),
            }
        )
    return {
        "n_disease": int((groups == disease_value).sum()),
        "n_control": int((groups == control_value).sum()),
        "by_cell_type": by_cell_type,
    }


def disease_deg_by_celltype(adata, contrast: dict[str, str]) -> Iterator[tuple[str, dict[str, Any], pd.DataFrame | None]]:
    column = contrast["column"]
    disease_value = contrast["disease_value"]
    control_value = contrast["control_value"]
    labels = adata.obs["predicted_cell_type"].astype(str)
    rows: list[dict[str, Any]] = []
    tables: list[pd.DataFrame] = []
    for cell_type in labels.value_counts().index.astype(str):
        if cell_type.strip().lower() in _SKIP_CELL_TYPES:
            continue
        mask = labels == cell_type
        groups = adata.obs.loc[mask, column].astype(str)
        n_disease = int((groups == disease_value).sum())
        n_control = int((groups == control_value).sum())
        row: dict[str, Any] = {
            "cell_type": cell_type,
            "n_disease": n_disease,
            "n_control": n_control,
            "up_in_disease": [],
            "down_in_disease": [],
            "homeostasis_genes": [],
            "explanation": "",
        }
        if n_disease < MIN_DEG_CELLS or n_control < MIN_DEG_CELLS:
            row["skipped"] = f"Need at least {MIN_DEG_CELLS} cells in both the disease group and the control group."
            rows.append(row)
            yield f"DEG skipped for {cell_type}: {row['skipped']}", row, None
            continue
        subset = adata[mask].copy()
        table = None
        try:
            sc.tl.rank_genes_groups(
                subset,
                groupby=column,
                groups=[disease_value],
                reference=control_value,
                method="wilcoxon",
                use_raw=adata.raw is not None,
            )
            frame = sc.get.rank_genes_groups_df(subset, group=disease_value)
            row["up_in_disease"] = _deg_gene_table(frame, higher_in_disease=True)
            row["down_in_disease"] = _deg_gene_table(frame, higher_in_disease=False)
            row["homeostasis_genes"] = _homeostasis_stats(frame)
            if not frame.empty:
                table = _full_deg_table(frame, cell_type, n_disease, n_control)
                tables.append(table)
        except Exception as exc:
            row["skipped"] = str(exc)
        rows.append(row)
        if row.get("skipped"):
            yield f"DEG failed for {cell_type}: {row['skipped']}", row, None
        else:
            yield (
                f"DEG done for {cell_type}: {len(row['up_in_disease'])} up, "
                f"{len(row['down_in_disease'])} down "
                f"(disease {n_disease}, control {n_control})"
            ), row, table


HOMEOSTASIS_GENES = ("CX3CR1", "P2RY12", "P2RY13")


def _homeostasis_stats(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty or "names" not in frame.columns:
        return []
    selected = frame[frame["names"].astype(str).isin(HOMEOSTASIS_GENES)]
    rows = []
    for record in selected.to_dict(orient="records"):
        logfc = record.get("logfoldchanges")
        padj = record.get("pvals_adj")
        rows.append(
            {
                "gene": str(record.get("names") or ""),
                "logfc": round(float(logfc), 3) if logfc is not None and np.isfinite(logfc) else None,
                "padj": float(padj) if padj is not None and np.isfinite(padj) else None,
            }
        )
    order = {gene: index for index, gene in enumerate(HOMEOSTASIS_GENES)}
    return sorted(rows, key=lambda item: order.get(item["gene"], 99))


def _status(text: str) -> dict[str, str]:
    print(f"STATUS {text}", flush=True)
    return {"type": "status", "text": text}


def _format_deg_genes(genes: list[dict[str, Any]]) -> str:
    if not genes:
        return "none"
    return ", ".join(f"{item['gene']} (logFC {item['logfc']})" for item in genes)


_MAS = None


def _get_mas():
    global _MAS
    if _MAS is not None:
        return _MAS
    cwd = os.getcwd()
    os.chdir(V2_RAG_DIR)
    try:
        from multiagent_system.main import create_system

        _MAS = create_system()
    finally:
        os.chdir(cwd)
    return _MAS


def _mas_deg_question(row: dict[str, Any], contrast: dict[str, str]) -> str:
    disease_name = contrast.get("disease_name") or contrast["disease_value"]
    return (
        f"Disease: {disease_name}. Cell type: {row['cell_type']}. "
        f"The comparison is {contrast['disease_value']} relative to {contrast['control_value']}.\n"
        "Both gene lists have adjusted p below 0.05 and are ordered by absolute log fold change.\n"
        f"Up in the disease group: {_format_deg_genes(row['up_in_disease'])}\n"
        f"Down in the disease group: {_format_deg_genes(row['down_in_disease'])}\n\n"
        "Write two paragraphs in English, with a blank line between them. Do not label the paragraphs, do not invent genes outside these lists, and do not diagnose.\n"
        "The first paragraph summarizes the functions of the up-regulated genes, then the down-regulated genes. Name genes and results only. Do not write log fold change or p values.\n"
        "The second paragraph is only about this cell type and what these expression changes may imply. Do not add a closing caveat that the change cannot be inferred as an established functional alteration, or that not every cell of this type changes in the same way."
    )


def _format_homeostasis(genes: list[dict[str, Any]]) -> str:
    if not genes:
        return "none"
    parts = []
    for item in genes:
        padj = item.get("padj")
        padj_text = f"{padj:.3g}" if isinstance(padj, float) else "NA"
        parts.append(f"{item['gene']} (logFC {item['logfc']}, padj {padj_text})")
    return ", ".join(parts)


def _mas_answer_failed(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return True
    return (
        text.startswith("error querying")
        or text.startswith("graph rag not available")
        or text.startswith("document rag not available")
        or "neo.clienterror" in text
    )


def _explain_with_ft_llm(question: str) -> str:
    from LLM_utils import CustomOpenAIProxyChat
    from langchain_core.messages import HumanMessage, SystemMessage
    from multiagent_system.config import Config

    config = Config()
    llm = CustomOpenAIProxyChat(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model_id,
        streaming=False,
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You explain single-cell differential expression. Answer only from the gene lists and statistics supplied by the user. "
                    "Do not write Cypher, do not query a database, and do not invent genes or statistics outside the list. "
                    "Do not label paragraphs."
                )
            ),
            HumanMessage(content=question),
        ]
    )
    return str(getattr(response, "content", None) or response).strip()


def explain_celltype_with_mas(row: dict[str, Any], contrast: dict[str, str]) -> tuple[str, str]:
    question = _mas_deg_question(row, contrast)
    answer = ""
    try:
        mas = _get_mas()
        mas.workflow.cfg = {"configurable": {"thread_id": f"sc-deg-{uuid.uuid4().hex}"}}
        answer = str(mas.query(question, stream=False) or "").strip()
    except Exception as exc:
        answer = f"Error querying Graph RAG: {exc}"
    if not _mas_answer_failed(answer):
        return answer, "mas"
    return _explain_with_ft_llm(question), "ft"


def composition_summary(adata, obs_key: str = "predicted_cell_type") -> list[dict[str, Any]]:
    counts = adata.obs[obs_key].astype(str).value_counts()
    total = int(counts.sum())
    return [
        {
            "cell_type": str(cell_type),
            "n_cells": int(n),
            "pct": round(100.0 * n / total, 2) if total else 0.0,
        }
        for cell_type, n in counts.items()
    ]


def _fig_to_data_url(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def render_umap_png(adata, color_key: str = "predicted_cell_type", title: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc.pl.umap(
        adata,
        color=color_key,
        ax=ax,
        show=False,
        frameon=False,
        title=title or color_key,
    )
    return _fig_to_data_url(fig)


def render_composition_pie(composition: list[dict[str, Any]]) -> str:
    if not composition:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No composition data", ha="center", va="center")
        ax.axis("off")
        return _fig_to_data_url(fig)

    labels = [row["cell_type"] for row in composition]
    sizes = [row["n_cells"] for row in composition]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Predicted cell type composition")
    fig.tight_layout()
    return _fig_to_data_url(fig)


class SingleCellReportService:
    def analyze_h5ad(
        self,
        h5ad_path: str | Path,
        *,
        label: str,
        source: str,
        max_cells: int | None = None,
    ) -> dict[str, Any]:
        events = list(
            self.iter_analyze_events(
                h5ad_path,
                label=label,
                source=source,
                max_cells=max_cells,
            )
        )
        for event in reversed(events):
            if event.get("type") == "done":
                return event
        raise RuntimeError("Single-cell analysis finished without a report")

    def iter_analyze_events(
        self,
        h5ad_path: str | Path,
        *,
        label: str,
        source: str = "upload",
        max_cells: int | None = None,
        full: bool = False,
    ) -> Iterator[dict[str, Any]]:
        cap = None if full else (max_cells if max_cells is not None else MAX_CELLS)

        yield {"type": "status", "text": "Loading h5ad"}
        adata, was_subsampled = load_h5ad(h5ad_path, max_cells=cap)

        yield {
            "type": "status",
            "text": (
                f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes"
                + (" (subsampled for analysis)" if was_subsampled else "")
            ),
        }

        scale_info = detect_expression_scale(adata)
        yield {
            "type": "status",
            "text": (
                f"Expression scale: {scale_info['label']} "
                f"(max={scale_info['max_value']}, median={scale_info['median_value']})"
            ),
        }

        batch_key = infer_batch_key(adata)
        batch_label = batch_key or "none"
        yield {"type": "status", "text": f"Batch key: {batch_label}"}

        preprocess_steps: dict[str, str] = {}
        for text, steps in preprocess_adata(adata, batch_key, scale_info):
            yield _status(text)
            if steps is not None:
                preprocess_steps = steps

        top_markers = None
        for text, markers in cluster_and_deg(adata, resolution=LEIDEN_RESOLUTION):
            yield _status(text)
            if markers is not None:
                top_markers = markers
        if top_markers is None:
            raise RuntimeError("Cluster markers were not produced")
        n_clusters = int(adata.obs["leiden"].nunique())

        yield _status(f"Annotating {n_clusters} clusters")
        mapping: dict[str, str] = {}
        for text, partial in annotate_clusters_llm(top_markers, adata):
            mapping = partial
            yield _status(text)
        yield _status("Normalizing cluster labels")
        mapping = normalize_cluster_labels(mapping)
        apply_cluster_labels(adata, mapping)
        yield _status("Cluster labels ready: " + ", ".join(f"{k}={v}" for k, v in sorted(mapping.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))))

        report_id = uuid.uuid4().hex
        contrast = find_disease_contrast(adata)
        if contrast is None:
            disease_deg = {
                "available": False,
                "detail": "No disease-versus-control grouping was found, so cell-type differential expression was skipped.",
            }
            yield _status("No disease/control column; skipped cell-type DEG")
        else:
            yield _status(
                f"DEG by cell type: {contrast['disease_value']} vs {contrast['control_value']} "
                f"({contrast['column']})"
            )
            deg_rows = []
            deg_tables = []
            for text, row, table in disease_deg_by_celltype(adata, contrast):
                deg_rows.append(row)
                if table is not None:
                    deg_tables.append(table)
                yield _status(text)
            disease_deg_csv = save_deg_csv(deg_tables, report_id)
            for row in deg_rows:
                if row.get("skipped") or not (row.get("up_in_disease") or row.get("down_in_disease")):
                    continue
                yield _status(f"Explaining {row['cell_type']}")
                try:
                    explanation, source_name = explain_celltype_with_mas(row, contrast)
                    row["explanation"] = explanation
                    if source_name == "ft":
                        yield _status(f"Explanation for {row['cell_type']} used FT LLM")
                    else:
                        yield _status(f"Explanation done: {row['cell_type']}")
                except Exception as exc:
                    row["explanation"] = ""
                    yield _status(f"Explanation skipped for {row['cell_type']} ({exc})")
            disease_deg = {
                "available": True,
                "column": contrast["column"],
                "disease_value": contrast["disease_value"],
                "control_value": contrast["control_value"],
                "disease_name": contrast["disease_name"],
                "detail": (
                    f"{contrast['disease_name']}: {contrast['disease_value']} relative to {contrast['control_value']}, "
                    f"group column {contrast['column']}. The comparison is run inside each predicted cell type."
                ),
                "csv_available": disease_deg_csv,
                "group_counts": disease_group_counts(adata, contrast),
                "rows": deg_rows,
            }

        save_annotated_h5ad(adata, report_id)

        yield _status("Saved annotated h5ad")
        yield _status("Rendering UMAP and composition chart")
        if disease_deg.get("available") and contrast is not None:
            disease_deg["umap_data_url"] = render_umap_png(
                adata,
                color_key=contrast["column"],
                title=f"{contrast['disease_value']} vs {contrast['control_value']}",
            )
        composition = composition_summary(adata)
        cluster_rows = build_cluster_rows(adata, mapping, top_markers)

        yield {
            "type": "done",
            "label": label,
            "source": source,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_clusters": n_clusters,
            "leiden_resolution": LEIDEN_RESOLUTION,
            "expression_scale": scale_info,
            "preprocess_steps": preprocess_steps,
            "batch_key": batch_key,
            "subsampled": was_subsampled,
            "composition": composition,
            "cluster_rows": cluster_rows,
            "disease_deg": disease_deg,
            "report_id": report_id,
            "umap_data_url": render_umap_png(adata),
            "pie_data_url": render_composition_pie(composition),
        }


_service: SingleCellReportService | None = None


def get_singlecell_service() -> SingleCellReportService:
    global _service
    if _service is None:
        _service = SingleCellReportService()
    return _service
