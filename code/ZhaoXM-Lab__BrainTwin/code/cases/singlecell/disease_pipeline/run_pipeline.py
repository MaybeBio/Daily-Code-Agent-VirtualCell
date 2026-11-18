#!/usr/bin/env python3
"""Default single-cell disease pipeline.

Reuses scanpy Wilcoxon tests, gseapy Enrichr, and the existing BrainTwin
multi-agent query. It does not recluster, does not rewrite the input file,
and does not invent model answers when the model is unavailable.

Example:
  python run_pipeline.py --config config_grubman2019.yaml --cell-types astro
  python run_pipeline.py --config config_grubman2019.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    }
)

V2_RAG = Path(__file__).resolve().parents[3] / "braintwin"
PROMPT_VERSION = "disease_chain_v3"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _as_dense_vector(matrix, column: int) -> np.ndarray:
    values = matrix[:, column]
    if sp.issparse(values):
        values = values.toarray()
    return np.asarray(values).ravel()


def _matrix_kind(matrix) -> str:
    data = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    if data.size == 0:
        return "empty"
    sample = data if data.size <= 200000 else data[:200000]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return "empty"
    xmax = float(np.max(finite))
    near_int = float(np.mean(np.abs(finite - np.rint(finite)) < 1e-6))
    if xmax > 30 and near_int > 0.98:
        return "counts"
    if xmax <= 30:
        return "log_normalized"
    return "unknown"


def prepare_expression(adata, layer_name: str | None) -> tuple[str, str]:
    """Return the layer used for DE and a short description of what was done."""
    if layer_name and layer_name != "auto":
        if layer_name not in adata.layers:
            raise KeyError(f"Requested layer {layer_name!r} is not in the file")
        kind = _matrix_kind(adata.layers[layer_name])
        if kind == "counts":
            raise ValueError(f"Layer {layer_name!r} looks like raw counts; refuse to test it")
        return layer_name, f"used existing layer {layer_name} ({kind})"

    kind = _matrix_kind(adata.X)
    if kind == "log_normalized":
        adata.layers["log1p_norm"] = adata.X.copy()
        return "log1p_norm", "X already looks log-normalized; copied to layer log1p_norm"
    if kind != "counts":
        raise ValueError(f"Cannot tell whether X is counts or log-normalized (state={kind})")

    working = adata.copy()
    sc.pp.normalize_total(working, target_sum=1e4)
    sc.pp.log1p(working)
    adata.layers["log1p_norm"] = working.X.copy()
    return "log1p_norm", "X looked like raw counts; normalize_total + log1p on a working copy"


def _group_counts(obs: pd.DataFrame, condition_key: str, donor_key: str | None, group: str) -> dict[str, Any]:
    mask = obs[condition_key].astype(str) == str(group)
    n_cells = int(mask.sum())
    n_donors = None
    if donor_key and donor_key in obs.columns:
        n_donors = int(obs.loc[mask, donor_key].astype(str).nunique())
    return {"n_cells": n_cells, "n_donors": n_donors}


def _extract_rank_table(adata, group: str, control: str) -> pd.DataFrame:
    payload = adata.uns["rank_genes_groups"]
    frame = pd.DataFrame(
        {
            "gene": list(payload["names"][group]),
            "score": list(payload["scores"][group]),
            "logfoldchanges": list(payload["logfoldchanges"][group]),
            "pvals": list(payload["pvals"][group]),
            "pvals_adj": list(payload["pvals_adj"][group]),
        }
    )
    # pts tables are indexed by gene, not by the ranked-gene order.
    # With an explicit reference group, scanpy stores both fractions in pts
    # and does not fill pts_rest.
    if "pts" in payload:
        frame["pts_disease"] = frame["gene"].map(payload["pts"][group]).astype(float)
        control_key = control if control in payload["pts"].columns else group
        if "pts_rest" in payload and group in getattr(payload["pts_rest"], "columns", []):
            frame["pts_control"] = frame["gene"].map(payload["pts_rest"][group]).astype(float)
        elif control_key in payload["pts"].columns:
            frame["pts_control"] = frame["gene"].map(payload["pts"][control_key]).astype(float)
    frame["logfoldchanges"] = pd.to_numeric(frame["logfoldchanges"], errors="coerce")
    frame["pvals_adj"] = pd.to_numeric(frame["pvals_adj"], errors="coerce")
    return frame


def _direction_check(adata, table: pd.DataFrame, layer: str, condition_key: str, disease: str, control: str) -> dict[str, Any]:
    ordered = table.dropna(subset=["logfoldchanges"]).sort_values("logfoldchanges", ascending=False)
    if ordered.empty:
        return {"ok": False, "reason": "no logfoldchanges"}
    gene = str(ordered.iloc[0]["gene"])
    logfc = float(ordered.iloc[0]["logfoldchanges"])
    column = adata.var_names.get_loc(gene)
    values = _as_dense_vector(adata.layers[layer], column)
    disease_mask = adata.obs[condition_key].astype(str).to_numpy() == str(disease)
    control_mask = adata.obs[condition_key].astype(str).to_numpy() == str(control)
    mean_disease = float(np.mean(values[disease_mask])) if disease_mask.any() else float("nan")
    mean_control = float(np.mean(values[control_mask])) if control_mask.any() else float("nan")
    agrees = (logfc > 0 and mean_disease > mean_control) or (logfc < 0 and mean_disease < mean_control)
    return {
        "ok": bool(agrees),
        "gene": gene,
        "log2fc_disease_vs_control": logfc,
        "mean_log1p_disease": mean_disease,
        "mean_log1p_control": mean_control,
        "positive_logfc_means": "higher in the disease group",
    }


def run_de_for_cell_type(adata, cell_type: str, cfg: dict[str, Any], layer: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    condition_key = cfg["condition_key"]
    disease = str(cfg["disease_group"])
    control = str(cfg["control_group"])
    subset = adata[adata.obs[cfg["cell_type_key"]].astype(str) == str(cell_type)].copy()
    disease_info = _group_counts(subset.obs, condition_key, cfg.get("donor_key"), disease)
    control_info = _group_counts(subset.obs, condition_key, cfg.get("donor_key"), control)
    meta = {
        "cell_type": cell_type,
        "n_disease_cells": disease_info["n_cells"],
        "n_control_cells": control_info["n_cells"],
        "n_disease_donors": disease_info["n_donors"],
        "n_control_donors": control_info["n_donors"],
        "donor_key": cfg.get("donor_key"),
        "statistical_note": "Cell-level Wilcoxon test. Donor differences are not controlled.",
    }
    if disease_info["n_cells"] == 0 or control_info["n_cells"] == 0:
        meta["skipped"] = "missing disease or control cells"
        return None, meta

    sc.tl.rank_genes_groups(
        subset,
        groupby=condition_key,
        groups=[disease],
        reference=control,
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        pts=True,
        use_raw=False,
        layer=layer,
        key_added="rank_genes_groups",
    )
    table = _extract_rank_table(subset, disease, control)
    meta["direction_check"] = _direction_check(subset, table, layer, condition_key, disease, control)
    meta["n_genes_tested"] = int(len(table))
    meta["skipped"] = ""
    return table, meta


def split_candidates(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    padj = float(cfg["padj_max"])
    min_fc = float(cfg["min_abs_log2fc"])
    passed = table[table["pvals_adj"] < padj].copy()
    passed = passed[passed["logfoldchanges"].abs() > min_fc]
    up = passed[passed["logfoldchanges"] > 0].sort_values(["pvals_adj", "logfoldchanges"], ascending=[True, False])
    down = passed[passed["logfoldchanges"] < 0].sort_values(["pvals_adj", "logfoldchanges"], ascending=[True, True])
    return up, down


def run_enrichment(genes: list[str], background: list[str], cfg: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    if len(genes) < 3:
        return pd.DataFrame(), "fewer than 3 genes; enrichment not run"
    import gseapy as gp

    result = gp.enrichr(
        gene_list=genes,
        gene_sets=cfg["gene_set"],
        organism=cfg.get("organism", "human"),
        background=background,
        outdir=None,
        cutoff=float(cfg["padj_max"]),
        no_plot=True,
    )
    frame = result.results.copy()
    if frame.empty:
        return frame, ""
    adj_col = "Adjusted P-value" if "Adjusted P-value" in frame.columns else None
    if adj_col:
        frame = frame[frame[adj_col] <= float(cfg["padj_max"])]
    keep = [c for c in ["Gene_set", "Term", "Overlap", "P-value", "Adjusted P-value", "Odds Ratio", "Combined Score", "Genes"] if c in frame.columns]
    frame = frame[keep].sort_values("Adjusted P-value") if "Adjusted P-value" in keep else frame
    return frame, ""


def _gene_records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    rows = []
    for record in frame.head(limit).to_dict(orient="records"):
        rows.append(
            {
                "gene": record.get("gene"),
                "log2fc": None if pd.isna(record.get("logfoldchanges")) else round(float(record["logfoldchanges"]), 4),
                "pvals_adj": None if pd.isna(record.get("pvals_adj")) else float(record["pvals_adj"]),
                "pts_disease": None if pd.isna(record.get("pts_disease", np.nan)) else round(float(record["pts_disease"]), 4),
                "pts_control": None if pd.isna(record.get("pts_control", np.nan)) else round(float(record["pts_control"]), 4),
            }
        )
    return rows


def build_model_input(cell_type: str, cfg: dict[str, Any], meta: dict[str, Any], up: pd.DataFrame, down: pd.DataFrame, function_frame: pd.DataFrame, function_source: str) -> dict[str, Any]:
    limit = int(cfg["top_n_genes"])
    label = (cfg.get("cell_type_labels") or {}).get(cell_type, cell_type)
    terms = []
    if function_source == "enrich" and not function_frame.empty:
        chosen = []
        if "direction" in function_frame.columns:
            for direction in ("up", "down"):
                chosen.append(function_frame[function_frame["direction"] == direction].head(8))
            selected = pd.concat(chosen, ignore_index=True) if chosen else function_frame.head(0)
        else:
            selected = function_frame.head(8)
        for record in selected.to_dict(orient="records"):
            terms.append(
                {
                    "term": record.get("Term"),
                    "genes": record.get("Genes"),
                    "adjusted_p": record.get("Adjusted P-value"),
                    "odds_ratio": record.get("Odds Ratio"),
                    "direction": record.get("direction"),
                }
            )
    return {
        "disease": cfg["disease"],
        "tissue": cfg["tissue"],
        "species": cfg["species"],
        "cell_type": label,
        "cell_type_id": cell_type,
        "n_disease_cells": meta["n_disease_cells"],
        "n_control_cells": meta["n_control_cells"],
        "n_disease_donors": meta["n_disease_donors"],
        "n_control_donors": meta["n_control_donors"],
        "contrast": f"{cfg['disease_group']} versus {cfg['control_group']}",
        "positive_log2fc": "higher in the disease group",
        "statistical_note": meta["statistical_note"],
        "function_source": function_source,
        "up_genes": _gene_records(up, limit),
        "down_genes": _gene_records(down, limit),
        "function_terms": terms,
    }


def _fmt_p(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{float(value):.1e}"


def _fmt_gene_line(records: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{row['gene']} {float(row['log2fc']):+.2f} {_fmt_p(row['pvals_adj'])}"
        for row in records
        if row.get("log2fc") is not None
    )


def _fmt_terms(terms: list[dict[str, Any]]) -> str:
    lines = []
    for row in terms:
        genes = str(row.get("genes") or "")
        shown = ";".join(genes.split(";")[:6])
        lines.append(f"{row.get('direction')} | {row.get('term')} | padj={_fmt_p(row.get('adjusted_p'))} | genes={shown}")
    return "\n".join(lines) if lines else "none"


def build_prompt(payload: dict[str, Any]) -> str:
    source = payload["function_source"]
    if source == "enrich":
        function_rule = (
            "Function entries come from Enrichr statistical enrichment. "
            "You may cite these entries and their adjusted P values. "
            "Do not treat a gene direction as evidence that the whole pathway is activated or inhibited."
        )
    elif source == "model":
        function_rule = (
            "No enrichment tool was called. Summarize functions from the genes and existing knowledge, "
            "and label this as a model-proposed functional interpretation. "
            "Do not invent enrichment scores, P values, or citations."
        )
    else:
        function_rule = (
            "Enrichment did not succeed. Do not invent enrichment results. "
            "Any functional summary must be labeled as a model interpretation."
        )
    # Graph RAG turns this question into Cypher. Do not ask for a code block or
    # a machine-readable object here; that instruction replaces the Cypher query.
    return (
        f"Cell type {payload['cell_type']} in {payload['species']} {payload['tissue']}, "
        f"disease {payload['disease']}. Contrast {payload['contrast']}. "
        f"Positive log2FC means {payload['positive_log2fc']}. "
        f"Cells disease/control: {payload['n_disease_cells']}/{payload['n_control_cells']}. "
        f"Donors disease/control: {payload['n_disease_donors']}/{payload['n_control_donors']}. "
        f"{payload['statistical_note']} {function_rule}\n"
        f"Genes higher in disease, written as gene log2FC adjusted P: {_fmt_gene_line(payload['up_genes'])}\n"
        f"Genes lower in disease, written as gene log2FC adjusted P: {_fmt_gene_line(payload['down_genes'])}\n"
        f"Function entries from {source}:\n{_fmt_terms(payload['function_terms'])}\n"
        "Using the knowledge graph, describe GO terms and trait associations for these genes and this cell type. "
        "Separate what the differential result shows, what the graph supports, and what remains a hypothesis. "
        "Do not invent citations. Do not state that a whole pathway is activated or inhibited only because member genes changed. "
        "An association is not a proven causal path."
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _gene_clause(line: str, keep: int) -> str:
    if ":" not in line:
        return line
    prefix, rest = line.split(":", 1)
    items = [item.strip() for item in rest.split(";") if item.strip()]
    return prefix + ": " + "; ".join(items[:keep])


def _compact_kg_question(prompt: str) -> str:
    """Shorter question for a second BrainTwin attempt.

    The full gene table can overflow the Cypher prompt.
    """
    lines = prompt.splitlines()
    head = lines[0] if lines else prompt[:400]
    gene_lines = []
    function_lines = []
    in_function = False
    for line in lines[1:]:
        if line.startswith("Genes higher") or line.startswith("Genes lower"):
            gene_lines.append(_gene_clause(line, 8))
            in_function = False
        elif line.startswith("Function entries"):
            in_function = True
            function_lines.append(line)
        elif in_function and (line.startswith("up |") or line.startswith("down |")):
            function_lines.append(line)
        else:
            in_function = False
    ups = [line for line in function_lines if line.startswith("up |")][:4]
    downs = [line for line in function_lines if line.startswith("down |")][:4]
    header = next((line for line in function_lines if line.startswith("Function entries")), "Function entries:")
    closing = (
        "Using the knowledge graph, describe GO terms and trait associations for these genes and this cell type. "
        "Separate what the differential result shows, what the graph supports, and what remains a hypothesis. "
        "Do not invent citations. An association is not a proven causal path."
    )
    return "\n".join([head, *gene_lines, header, *ups, *downs, closing])


def _structure_chains(prompt: str, braintwin_answer: str, model_id: str, base_url: str) -> str:
    import requests

    instruction = (
        "Turn the BrainTwin answer and the differential result into one compact object with keys report and chains. "
        "report is one short English sentence. Write at most 4 chains. "
        "gene_changes is required and must be one short string naming a few supporting genes and their direction, not the full gene table. "
        "Each chain also has cell_type, function_process, function_basis, function_source, disease_role, support, "
        "external_sources, uncertain_steps. "
        "support is only observed, knowledge_supported, or hypothesis. "
        "observed is only for the differential result. knowledge_supported is only for statements present in the BrainTwin answer. "
        "Everything else is hypothesis. external_sources is an empty list unless the BrainTwin answer names a source. "
        "Do not invent scores, P values, or citations. Do not fill a chain when evidence is missing. "
        "Write no text outside the object.\n\n"
        f"Differential result:\n{_compact_kg_question(prompt)}\n\nBrainTwin answer:\n{braintwin_answer[:1200]}"
    )
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": instruction}],
            "temperature": 0.1,
            "stream": False,
            "max_tokens": 900,
        },
        timeout=180,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"])


def _write_kg_trace(kg_dir: Path, cell_type: str, details: dict[str, Any], question: str) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    hits = []
    for step in details.get("intermediate_steps") or []:
        if not isinstance(step, dict):
            continue
        context = step.get("context")
        hits.append({"cypher": step.get("query") or details.get("cypher"), "context": context})
    payload = {
        "cell_type": cell_type,
        "question": question,
        "cypher": details.get("cypher"),
        "hits": hits,
        "reference_available": False,
        "kg_name": "BrainTwin Neo4j",
        "relation_source": "unknown",
        "note": "Hits are the raw graph-query context. They are not literature verification. No edge id is invented when the result has none.",
    }
    (kg_dir / f"{cell_type}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def query_braintwin(prompt: str, cell_type: str, kg_dir: Path | None = None) -> tuple[str, str]:
    if str(V2_RAG) not in sys.path:
        sys.path.insert(0, str(V2_RAG))
    from multiagent_system.config import Config
    from multiagent_system.main import create_system

    original = Config.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.base_url = str(self.base_url).rstrip("/")
        self.full_url = str(self.full_url).rstrip("/")

    Config.__init__ = patched  # type: ignore[method-assign]
    system = create_system()
    model_id = system.config.model_id
    system.workflow.cfg = {"configurable": {"thread_id": f"disease-{cell_type}"}}

    def _ask(text: str) -> tuple[str, dict[str, Any]]:
        traced = system.workflow.query_with_trace(text)
        return str(traced.get("answer") or ""), traced.get("kg_details") or {}

    last_error: Exception | None = None
    answer = ""
    details: dict[str, Any] = {}
    for _attempt in range(2):
        try:
            answer, details = _ask(prompt)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    used_prompt = prompt
    if answer.startswith("Error querying"):
        compact = _compact_kg_question(prompt)
        system.workflow.cfg = {"configurable": {"thread_id": f"disease-{cell_type}-retry"}}
        try:
            retried, retry_details = _ask(compact)
        except Exception as exc:
            return f"{answer}\n\n[compact retry failed: {type(exc).__name__}: {exc}]", model_id
        if retried.startswith("Error querying"):
            if kg_dir is not None:
                _write_kg_trace(kg_dir, cell_type, details, prompt)
            return f"{answer}\n\n[compact retry]\n{compact}\n\n{retried}", model_id
        answer = retried
        details = retry_details
        used_prompt = compact
    if kg_dir is not None:
        _write_kg_trace(kg_dir, cell_type, details, used_prompt)
    try:
        structured = _structure_chains(used_prompt, answer, model_id, system.config.base_url)
    except Exception as exc:
        return f"{answer}\n\n[structure failed: {type(exc).__name__}: {exc}]", model_id
    return f"{answer}\n\n{structured}", model_id


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def plot_dotplot(gene_rows: pd.DataFrame, output_dir: Path) -> None:
    if gene_rows.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_types = list(dict.fromkeys(gene_rows["cell_type"]))
    genes = list(dict.fromkeys(gene_rows["gene"]))
    fig_w = max(6.5, 0.42 * len(genes) + 2)
    fig_h = max(3.2, 0.55 * len(cell_types) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    xs = {gene: i for i, gene in enumerate(genes)}
    ys = {cell_type: i for i, cell_type in enumerate(cell_types)}
    logfc = gene_rows["log2fc"].astype(float)
    limit = float(np.nanmax(np.abs(logfc))) if len(logfc) else 1.0
    limit = max(limit, 0.25)
    for _, row in gene_rows.iterrows():
        size = 40 + 220 * float(row["pts"] if pd.notna(row["pts"]) else 0)
        ax.scatter(
            xs[row["gene"]],
            ys[row["cell_type"]],
            s=size,
            c=[float(row["log2fc"])],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            edgecolor="#4A4A4A",
            linewidth=0.3,
        )
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=60, ha="right")
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels(cell_types)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cell type")
    ax.set_title("Disease versus control log2 fold change")
    norm = plt.Normalize(vmin=-limit, vmax=limit)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, label="log2FC (disease / control)")
    fig.tight_layout()
    fig.savefig(output_dir / "deg_dotplot.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "deg_dotplot.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_function_overview(function_frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    enrich = function_frame[function_frame["source"] == "enrich"] if not function_frame.empty else function_frame
    n_bars = 0 if enrich.empty else min(20, len(enrich))
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.38 * max(n_bars, 1) + 1.2)))
    if enrich.empty:
        ax.text(0.5, 0.5, "No Enrichr terms passed the cutoff", ha="center", va="center")
        ax.set_axis_off()
    else:
        enrich = enrich.sort_values("adjusted_p").head(20).iloc[::-1]
        labels = [f"{row.cell_type}: {row.term}"[:80] for row in enrich.itertuples()]
        values = -np.log10(enrich["adjusted_p"].clip(lower=1e-300))
        colors = ["#E8542B" if row.direction == "up" else "#4C78A8" for row in enrich.itertuples()]
        ax.barh(range(len(enrich)), values, color=colors)
        ax.set_yticks(range(len(enrich)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("-log10 adjusted P (Enrichr)")
        ax.set_title("Functional enrichment overview")
        ax.scatter([], [], c="#E8542B", label="Up in disease")
        ax.scatter([], [], c="#4C78A8", label="Down in disease")
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "function_overview.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "function_overview.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_chains(chains: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shown = []
    seen = set()
    for chain in chains:
        label = str(chain.get("cell_type", ""))
        if label in seen:
            continue
        seen.add(label)
        shown.append(chain)
    shown = shown[:8]
    fig_h = max(3.2, 0.7 * max(len(shown), 1) + 1.2)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, max(len(shown), 1))
    ax.axis("off")
    ax.set_title("Inferred disease role, not a validated causal path", loc="left")
    headers = [(0.1, "Cell type"), (1.7, "Gene change"), (4.3, "Function"), (7.2, "Possible disease role")]
    for x, header in headers:
        ax.text(x, len(shown) - 0.02, header, fontsize=8, fontweight="bold", va="bottom")
    if not shown:
        ax.text(0.1, 0.4, "No inferred chain was returned", fontsize=10)
    for row_i, chain in enumerate(shown):
        y = len(shown) - row_i - 1
        genes = chain.get("gene_changes") or ""
        if isinstance(genes, list):
            gene_text = ", ".join(
                f"{item.get('gene')} {item.get('direction', '')}".strip() for item in genes[:4] if isinstance(item, dict)
            )
        else:
            gene_text = str(genes)
        if not gene_text.strip():
            gene_text = str(chain.get("function_basis") or "")
        values = [
            (0.1, str(chain.get("cell_type", ""))),
            (1.7, gene_text),
            (4.3, str(chain.get("function_process", ""))),
            (7.2, str(chain.get("disease_role", ""))),
        ]
        for x, value in values:
            ax.text(x, y + 0.35, value[:42], fontsize=7, va="center")
        for x in (1.45, 4.05, 6.95):
            ax.annotate(
                "",
                xy=(x + 0.18, y + 0.35),
                xytext=(x, y + 0.35),
                arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#888888"},
            )
    fig.tight_layout()
    fig.savefig(output_dir / "inferred_chains.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "inferred_chains.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _top_gene_rows(cell_type: str, up: pd.DataFrame, down: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for frame, direction in ((up.head(8), "up"), (down.head(8), "down")):
        for record in frame.to_dict(orient="records"):
            pts_col = "pts_disease" if direction == "up" else "pts_control"
            rows.append(
                {
                    "cell_type": cell_type,
                    "gene": record.get("gene"),
                    "direction": direction,
                    "log2fc": record.get("logfoldchanges"),
                    "pts": record.get(pts_col, record.get("pts_disease")),
                }
            )
    return rows


def run(cfg: dict[str, Any], cell_types: list[str] | None) -> None:
    output_dir = Path(cfg["output_dir"])
    de_dir = output_dir / "de"
    function_dir = output_dir / "function"
    chain_dir = output_dir / "chains"
    report_dir = output_dir / "reports"
    prompt_dir = output_dir / "prompts"
    raw_dir = output_dir / "raw_answers"
    fig_dir = output_dir / "figures"
    for path in (de_dir, function_dir, chain_dir, report_dir, prompt_dir, raw_dir, fig_dir):
        path.mkdir(parents=True, exist_ok=True)

    print(f"[info] reading {cfg['h5ad']}")
    adata = sc.read_h5ad(cfg["h5ad"])
    layer, expression_note = prepare_expression(adata, cfg.get("layer"))
    print(f"[info] expression: {expression_note}")
    background = [str(gene) for gene in adata.var_names]

    present = adata.obs[cfg["cell_type_key"]].astype(str)
    selected = cell_types or [str(item) for item in present.value_counts().index]
    reuse = bool(cfg.get("reuse_intermediates", True))
    run_model = bool(cfg.get("run_braintwin", True))
    model_id = ""
    summaries = []
    dot_rows: list[dict[str, Any]] = []
    function_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []

    for cell_type in selected:
        label = (cfg.get("cell_type_labels") or {}).get(cell_type, cell_type)
        safe = str(cell_type).replace("/", "_")
        full_path = de_dir / f"{safe}_de_full.tsv"
        up_path = de_dir / f"{safe}_up.tsv"
        down_path = de_dir / f"{safe}_down.tsv"
        meta_path = de_dir / f"{safe}_meta.json"
        print(f"[info] cell type {cell_type}")

        if reuse and full_path.exists() and meta_path.exists():
            table = _read_table(full_path)
            up = _read_table(up_path)
            down = _read_table(down_path)
            meta = json.loads(meta_path.read_text())
            print("  reused differential expression")
        else:
            table, meta = run_de_for_cell_type(adata, cell_type, cfg, layer)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            if table is None:
                print(f"  skipped: {meta['skipped']}")
                summaries.append({**meta, "n_up": 0, "n_down": 0, "function_source": "", "n_chains": 0})
                continue
            if not meta["direction_check"]["ok"]:
                _write_table(table, full_path)
                raise SystemExit(f"Contrast direction check failed for {cell_type}: {meta['direction_check']}")
            up, down = split_candidates(table, cfg)
            _write_table(table, full_path)
            _write_table(up, up_path)
            _write_table(down, down_path)
            print(f"  tested {meta['n_genes_tested']} genes; up {len(up)}; down {len(down)}; direction ok")

        if meta.get("skipped"):
            summaries.append({**meta, "n_up": 0, "n_down": 0, "function_source": "", "n_chains": 0})
            continue

        dot_rows.extend(_top_gene_rows(label, up, down))
        function_path = function_dir / f"{safe}_function.tsv"
        function_source = cfg["function_mode"]
        if reuse and function_path.exists():
            function_frame = _read_table(function_path)
        else:
            pieces = []
            errors = []
            if cfg["function_mode"] == "enrich":
                for direction, genes in (("up", up["gene"].astype(str).tolist()), ("down", down["gene"].astype(str).tolist())):
                    try:
                        enriched, message = run_enrichment(genes, background, cfg)
                    except Exception as exc:  # network or library errors stay visible
                        enriched = pd.DataFrame()
                        message = f"{type(exc).__name__}: {exc}"
                    if message:
                        errors.append(f"{direction}: {message}")
                    if not enriched.empty:
                        enriched["direction"] = direction
                        enriched["source"] = "enrich"
                        pieces.append(enriched)
                function_source = "enrich"
            else:
                function_source = "model"
            function_frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            _write_table(function_frame, function_path)
            (function_dir / f"{safe}_function_note.txt").write_text("\n".join(errors))
        if not function_frame.empty:
            for record in function_frame.to_dict(orient="records"):
                function_rows.append(
                    {
                        "cell_type": label,
                        "source": record.get("source", function_source),
                        "direction": record.get("direction", ""),
                        "term": record.get("Term", ""),
                        "adjusted_p": record.get("Adjusted P-value", np.nan),
                        "genes": record.get("Genes", ""),
                    }
                )

        payload = build_model_input(cell_type, cfg, meta, up, down, function_frame, function_source)
        prompt = build_prompt(payload)
        (prompt_dir / f"{safe}_prompt.txt").write_text(prompt)
        (prompt_dir / f"{safe}_input.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        raw_path = raw_dir / f"{safe}_raw.txt"
        chain_path = chain_dir / f"{safe}_chains.json"
        if reuse and raw_path.exists() and chain_path.exists():
            raw_answer = raw_path.read_text()
            parsed = json.loads(chain_path.read_text())
            print("  reused BrainTwin answer")
        elif not run_model:
            raw_answer = ""
            parsed = {"report": "BrainTwin was not run.", "chains": [], "model_status": "not_run"}
            raw_path.write_text(raw_answer)
            chain_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
        else:
            try:
                raw_answer, model_id = query_braintwin(prompt, safe, output_dir / "kg")
                parsed = _extract_json(raw_answer) or {"report": "", "chains": [], "model_status": "unparsed"}
                parsed["model_status"] = parsed.get("model_status", "ok")
                if isinstance(parsed.get("chains"), list):
                    allowed = {"observed", "knowledge_supported", "hypothesis"}
                    for chain in parsed["chains"]:
                        if isinstance(chain, dict):
                            support = str(chain.get("support", "")).strip().lower()
                            if support in allowed:
                                chain["support"] = support
            except Exception as exc:
                raw_answer = traceback.format_exc()
                parsed = {"report": "BrainTwin call failed. No answer was fabricated.", "chains": [], "model_status": f"{type(exc).__name__}: {exc}"}
                print(f"  BrainTwin failed: {exc}")
            raw_path.write_text(raw_answer)
            chain_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
        report = str(parsed.get("report") or "No model report.")
        report_dir.joinpath(f"{safe}_report.md").write_text(
            f"# {label}\n\n{report}\n\nContrast: {cfg['disease_group']} versus {cfg['control_group']}. "
            "Positive log2FC means higher in the disease group. "
            "Chains below are inferred, not validated causal paths.\n"
        )
        chains = parsed.get("chains") if isinstance(parsed.get("chains"), list) else []
        for chain in chains:
            if isinstance(chain, dict):
                chain.setdefault("cell_type", label)
                chain_rows.append(chain)
        summaries.append(
            {
                **{k: meta.get(k) for k in ["cell_type", "n_disease_cells", "n_control_cells", "n_disease_donors", "n_control_donors", "skipped"]},
                "n_up": int(len(up)),
                "n_down": int(len(down)),
                "function_source": function_source,
                "n_chains": len(chains),
                "direction_ok": meta.get("direction_check", {}).get("ok"),
            }
        )

    summary = pd.DataFrame(summaries)
    _write_table(summary, output_dir / "celltype_summary.tsv")
    _write_table(pd.DataFrame(chain_rows), output_dir / "chain_summary.tsv")
    plot_dotplot(pd.DataFrame(dot_rows), fig_dir)
    plot_function_overview(pd.DataFrame(function_rows), fig_dir)
    plot_chains(chain_rows, fig_dir)

    run_meta = {
        "dataset": cfg.get("dataset"),
        "expression": expression_note,
        "layer": layer,
        "use_raw": False,
        "method": "wilcoxon",
        "corr_method": "benjamini-hochberg",
        "padj_max": cfg["padj_max"],
        "min_abs_log2fc": cfg["min_abs_log2fc"],
        "function_mode": cfg["function_mode"],
        "gene_set": cfg.get("gene_set"),
        "prompt_version": PROMPT_VERSION,
        "model_id": model_id,
        "braintwin_interface": "multiagent_system.main.create_system().query",
        "donor_controlled": False,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2))
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(summary.to_string(index=False))
    print(f"[ok] wrote {output_dir}")
    print("[ok] explanation chains are not written here. Use --generate-mas so MAS returns them.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the default single-cell disease pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell-types", nargs="*", default=None, help="Optional subset, otherwise all labels")
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--skip-braintwin", action="store_true")
    parser.add_argument("--chains-v2-only", action="store_true", help="Rebuild audited chains from saved tables without rerunning tests")
    parser.add_argument(
        "--generate-mas",
        action="store_true",
        help="Reuse saved modules and graph rows, then ask the model to infer from that evidence. Does not rewrite model answers.",
    )
    args = parser.parse_args()
    cfg = _load_config(Path(args.config))
    if args.generate_mas:
        from module_mas import generate_with_mas

        generate_with_mas(Path(cfg["output_dir"]), cfg, reuse=not args.no_reuse)
        return
    if args.chains_v2_only:
        from chain_v2 import build_chain_v2

        build_chain_v2(Path(cfg["output_dir"]), cfg)
        return
    if args.no_reuse:
        cfg["reuse_intermediates"] = False
    if args.skip_braintwin:
        cfg["run_braintwin"] = False
    run(cfg, args.cell_types)


if __name__ == "__main__":
    main()
