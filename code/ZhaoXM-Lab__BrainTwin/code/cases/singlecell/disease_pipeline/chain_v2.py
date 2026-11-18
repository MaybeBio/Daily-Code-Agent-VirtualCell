#!/usr/bin/env python3
"""Build audited disease chains from saved differential and enrichment tables.

Version 2 does not take the model JSON as the chain. A model sentence is not
knowledge-graph evidence. Missing KG hit files are reported and left empty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

GO_ID = re.compile(r"(GO:\d+)")
GENE_TOKEN = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")

THEMES: list[tuple[str, tuple[str, ...]]] = [
    ("protein_quality", ("unfolded", "chaperone", "heat shock", "inclusion body", "protein folding", "protein stabilization", "protein complex assembly")),
    ("cell_death", ("apopt", "cell death", "programmed cell death")),
    ("synapse", ("synap", "axon", "neuron projection", "nervous system", "cell-cell adhesion", "learning")),
    ("viral", ("viral", "virus", "interferon", "symbiont")),
    ("immune", ("immune", "inflamm", "cytokine", "macrophage", "microglia", "leukocyte")),
    ("mitochondrion", ("mitochond",)),
    ("microtubule", ("microtubule",)),
]


def build_chain_v2(output_dir: Path, cfg: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    v2_dir = output_dir / "v2"
    v2_dir.mkdir(parents=True, exist_ok=True)
    labels = cfg.get("cell_type_labels") or {}
    disease = str(cfg.get("disease") or "")
    padj_max = float(cfg.get("padj_max", 0.05))
    missing = _missing_inputs(output_dir)
    records: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    cell_ids = _cell_ids(output_dir)
    for cell_id in cell_ids:
        label = labels.get(cell_id, cell_id)
        built, cell_records = _chains_for_cell(output_dir, cell_id, label, disease, padj_max)
        records.extend(cell_records)
        chains.extend(built)
    audits.extend(_audit(chains, records))
    payload = {
        "dataset": cfg.get("dataset"),
        "disease": disease,
        "tissue": cfg.get("tissue"),
        "species": cfg.get("species"),
        "limitations": [
            "No donor or sample identifier is present. Cell-level Wilcoxon tests do not control donor differences.",
            "Positive log2FC means higher in the disease group.",
            "Overlap with a GO term is not evidence that the process is activated or inhibited.",
            "GO terms were merged by shared genes and keyword themes. The full GO hierarchy was not available.",
            "Knowledge-graph hit records were not saved for this run, so no edge is kg_supported.",
            "No literature retrieval record was saved, so no edge is literature_supported.",
        ],
        "missing_inputs": missing,
        "stage_diagnosis": _stage_diagnosis(),
        "records": records,
        "chains": chains,
    }
    (v2_dir / "chains_v2.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    pd.DataFrame(audits).to_csv(v2_dir / "chain_audit.tsv", sep="\t", index=False)
    (v2_dir / "chains_and_conclusions_v2.md").write_text(_markdown(payload, chains))
    (v2_dir / "change_log.md").write_text(_change_log(output_dir, payload, chains, labels))
    print(f"[ok] chain v2 wrote {v2_dir} ({len(chains)} chains)")
    return v2_dir


def _missing_inputs(output_dir: Path) -> list[dict[str, str]]:
    missing = []
    if not (output_dir / "kg").exists():
        missing.append(
            {
                "item": "kg/",
                "reason": "Graph RAG intermediate steps were dropped before saving. No cypher result, triple, path, or edge id is available.",
            }
        )
    if not list((output_dir / "raw_answers").glob("*_raw.txt")):
        missing.append({"item": "raw_answers/", "reason": "Model answers are absent."})
    return missing


def _stage_diagnosis() -> dict[str, str]:
    return {
        "differential_and_enrichment": "Tables are present and are the source for v2 gene statistics and GO overlaps.",
        "generation": "The question asked for at most four chains and one support label for the whole chain. Disease-role text was unconstrained, so names such as protective, neuroinflammation, or the disease name were written as conclusions.",
        "structured_extraction": "The second model call copied or rewrote those sentences into JSON. It did not join genes back to the differential table. Some GO names in the neuron chains are not in the enrichment file.",
        "report": "The markdown report repeated the model paragraph and did not check it against the tables.",
        "knowledge_graph": "The query interface returned only the final sentence. Hit paths were not stored, so they cannot be reconstructed from that sentence.",
    }


def _cell_ids(output_dir: Path) -> list[str]:
    return sorted(path.name[: -len("_de_full.tsv")] for path in (output_dir / "de").glob("*_de_full.tsv"))


def _chains_for_cell(
    output_dir: Path,
    cell_id: str,
    label: str,
    disease: str,
    padj_max: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full = pd.read_csv(output_dir / "de" / f"{cell_id}_de_full.tsv", sep="\t")
    up = _read_gene_list(output_dir / "de" / f"{cell_id}_up.tsv")
    down = _read_gene_list(output_dir / "de" / f"{cell_id}_down.tsv")
    stats = {}
    for row in full.to_dict(orient="records"):
        gene = str(row["gene"])
        logfc = float(row["logfoldchanges"])
        stats[gene] = {
            "gene": gene,
            "logFC": logfc,
            "adjusted_P": float(row["pvals_adj"]),
            "direction": "up" if logfc > 0 else "down" if logfc < 0 else "unchanged",
            "record_id": f"de:{cell_id}:{gene}",
        }
    enrich_path = output_dir / "function" / f"{cell_id}_function.tsv"
    terms = []
    if enrich_path.exists() and enrich_path.stat().st_size > 0:
        frame = pd.read_csv(enrich_path, sep="\t")
        if not frame.empty:
            for index, row in enumerate(frame.to_dict(orient="records"), start=1):
                term = _term_record(cell_id, index, row, up, down, stats, padj_max)
                if term is not None:
                    terms.append(term)
    clusters = _cluster_terms(terms)
    records = [stats[gene] for gene in stats if gene in up or gene in down]
    records.extend(terms)
    chains = []
    for index, cluster in enumerate(clusters, start=1):
        chains.append(_chain_from_cluster(cell_id, label, disease, index, cluster, stats))
    return chains, records


def _read_gene_list(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, sep="\t")
    if frame.empty or "gene" not in frame.columns:
        return set()
    return set(frame["gene"].astype(str))


def _term_record(
    cell_id: str,
    index: int,
    row: dict[str, Any],
    up: set[str],
    down: set[str],
    stats: dict[str, dict[str, Any]],
    padj_max: float,
) -> dict[str, Any] | None:
    adjusted = row.get("Adjusted P-value")
    if pd.isna(adjusted) or float(adjusted) > padj_max:
        return None
    direction = str(row.get("direction") or "")
    expected = up if direction == "up" else down
    listed = [gene for gene in str(row.get("Genes") or "").split(";") if gene]
    overlap = [gene for gene in listed if gene in expected]
    outside = [gene for gene in listed if gene not in expected]
    sign_conflict = [
        gene
        for gene in overlap
        if stats.get(gene, {}).get("direction") not in {direction, ""}
    ]
    name = str(row.get("Term") or "")
    match = GO_ID.search(name)
    return {
        "record_id": f"enrich:{cell_id}:{direction}:{index}",
        "term_id": match.group(1) if match else "",
        "name": name,
        "source": str(row.get("source") or "enrich"),
        "input_direction": direction,
        "listed_genes": listed,
        "overlap_genes": overlap,
        "genes_outside_input": outside,
        "sign_conflict": sign_conflict,
        "adjusted_P": float(adjusted),
        "p_value": None if pd.isna(row.get("P-value")) else float(row["P-value"]),
        "odds_ratio": None if pd.isna(row.get("Odds Ratio")) else float(row["Odds Ratio"]),
        "combined_score": None if pd.isna(row.get("Combined Score")) else float(row["Combined Score"]),
        "polarity": _polarity(name),
        "themes": _themes(name),
    }


def _polarity(name: str) -> str:
    low = name.lower()
    if "negative regulation" in low:
        return "negative"
    if "positive regulation" in low:
        return "positive"
    return "unspecified"


def _themes(name: str) -> list[str]:
    low = name.lower()
    return [theme for theme, words in THEMES if any(word in low for word in words)]


def _cluster_terms(terms: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(terms)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(terms):
        for right_index in range(left_index + 1, len(terms)):
            if _pair_merge(left, terms[right_index]):
                parent[find(left_index)] = find(right_index)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, term in enumerate(terms):
        grouped.setdefault(find(index), []).append(term)
    clusters = [cluster for cluster in grouped.values() if any(term["overlap_genes"] for term in cluster)]
    clusters.sort(key=lambda cluster: min(term["adjusted_P"] for term in cluster))
    return clusters


def _pair_merge(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["input_direction"] != right["input_direction"]:
        return False
    left_genes = set(left["overlap_genes"])
    right_genes = set(right["overlap_genes"])
    if not left_genes or not right_genes:
        return False
    score = _jaccard(left_genes, right_genes)
    shared = left_genes & right_genes
    subset = left_genes <= right_genes or right_genes <= left_genes
    smaller = left_genes if len(left_genes) <= len(right_genes) else right_genes
    shared_theme = bool(set(left["themes"]) & set(right["themes"]))
    if left_genes == right_genes or score >= 0.4:
        return True
    if subset and len(smaller) >= 2:
        return True
    if subset and shared_theme:
        return True
    return shared_theme and bool(shared)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _chain_from_cluster(
    cell_id: str,
    label: str,
    disease: str,
    index: int,
    cluster: list[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    direction = cluster[0]["input_direction"]
    genes = []
    for term in cluster:
        for gene in term["overlap_genes"]:
            if gene not in {item["gene"] for item in genes} and gene in stats:
                genes.append(dict(stats[gene]))
    genes.sort(key=lambda item: item["adjusted_P"])
    polarities = sorted({term["polarity"] for term in cluster})
    conflicted = "positive" in polarities and "negative" in polarities
    viral = any("viral" in term["themes"] for term in cluster)
    chain_id = f"{cell_id}-{index:02d}"
    interpretation = _interpretation(direction, conflicted, viral, genes)
    edges = _edges(chain_id, label, genes, cluster)
    limitations = [
        "Cell-level Wilcoxon test. Donor differences are not controlled.",
        "Enrichment of a gene set does not establish that the process increased or decreased.",
    ]
    if any(len(term["overlap_genes"]) < 2 for term in cluster):
        limitations.append("At least one term overlaps only one input gene. The enrichment statistic is weak.")
    if any(term["genes_outside_input"] for term in cluster):
        limitations.append("Some genes listed by enrichment were not in the input gene set and were excluded from overlap_genes.")
    conflicts = []
    if conflicted:
        conflicts.append(
            {
                "issue": "The same gene set overlaps both positive-regulation and negative-regulation terms.",
                "terms": [term["name"] for term in cluster],
                "consequence": "The activity direction of the process is not determined.",
            }
        )
    return {
        "chain_id": chain_id,
        "cell_type": label,
        "cell_type_id": cell_id,
        "disease": disease,
        "stopped_at": "functional_annotation",
        "gene_changes": genes,
        "functional_evidence": [
            {
                "term_id": term["term_id"],
                "name": term["name"],
                "source": term["source"],
                "input_direction": term["input_direction"],
                "overlap_genes": term["overlap_genes"],
                "adjusted_P": term["adjusted_P"],
                "odds_ratio": term["odds_ratio"],
                "combined_score": term["combined_score"],
                "record_id": term["record_id"],
                "genes_outside_input": term["genes_outside_input"],
            }
            for term in cluster
        ],
        "functional_interpretation": interpretation,
        "disease_role": {
            "text": "",
            "evidence": [],
            "missing_evidence": "No saved knowledge-graph hit or literature record supports a specific disease action.",
            "empty_reason": "A disease name or an expression direction is not a disease action, and none is invented here.",
        },
        "edges": edges,
        "conflicts": conflicts,
        "limitations": limitations,
    }


def _interpretation(direction: str, conflicted: bool, viral: bool, genes: list[dict[str, Any]]) -> dict[str, Any]:
    direction_text = "higher" if direction == "up" else "lower"
    names = ", ".join(item["gene"] for item in genes[:8])
    extra = "" if len(genes) <= 8 else f" and {len(genes) - 8} more"
    text = (
        f"Genes {direction_text} in the disease group overlap the listed GO terms ({names}{extra}). "
        "This records an annotation overlap. It does not show that the process itself increased or decreased."
    )
    if conflicted:
        text += " Positive-regulation and negative-regulation terms both match, so the activity direction is ambiguous."
    if viral:
        text += " A viral-process annotation does not show that these samples had a viral infection."
    return {
        "text": text,
        "activity_direction_supported": False,
        "activity_direction_reason": "The sign of the input gene set and words such as positive or negative regulation are not the direction of the biological process.",
        "evidence_types": ["data_supported"],
    }


def _edges(chain_id: str, label: str, genes: list[dict[str, Any]], cluster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for number, gene in enumerate(genes, start=1):
        claim = f"{gene['gene']} is {gene['direction']} in the disease group (log2FC {gene['logFC']:.4g}, adjusted P {gene['adjusted_P']:.3g})."
        edges.append(
            {
                "edge_id": f"{chain_id}-expr-{number:02d}",
                "source": label,
                "target": gene["gene"],
                "relation": "differentially_expressed",
                "evidence": [
                    {
                        "evidence_type": "data_supported",
                        "source_record_id": gene["record_id"],
                        "supported_claim": claim,
                        "limitations": "Cell-level Wilcoxon test. Donor differences are not controlled.",
                    }
                ],
                "open_questions": [],
            }
        )
    for number, term in enumerate(cluster, start=1):
        edges.append(
            {
                "edge_id": f"{chain_id}-overlap-{number:02d}",
                "source": f"{term['input_direction']} genes",
                "target": term["term_id"] or term["name"],
                "relation": "overlaps_annotation",
                "evidence": [
                    {
                        "evidence_type": "data_supported",
                        "source_record_id": term["record_id"],
                        "supported_claim": (
                            f"The {term['input_direction']} gene set overlaps {term['name']}. "
                            f"Overlap genes: {', '.join(term['overlap_genes'])}."
                        ),
                        "limitations": "Overlap does not establish activation, inhibition, or a disease action.",
                    }
                ],
                "open_questions": ["Whether the process is increased or decreased is unresolved."],
            }
        )
    return edges


def _audit(chains: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    de_ids = {record["record_id"] for record in records if str(record.get("record_id", "")).startswith("de:")}
    enrich_ids = {record["record_id"] for record in records if str(record.get("record_id", "")).startswith("enrich:")}
    de_by_id = {record["record_id"]: record for record in records if str(record.get("record_id", "")).startswith("de:")}
    rows = []
    for chain in chains:
        for gene in chain["gene_changes"]:
            source = de_by_id.get(gene["record_id"])
            match = (
                source is not None
                and source["gene"] == gene["gene"]
                and source["direction"] == gene["direction"]
                and abs(source["logFC"] - gene["logFC"]) < 1e-6
            )
            rows.append(_audit_row(chain["chain_id"], gene["record_id"], "gene_matches_differential_table", "pass" if match else "fail", "Statistics and direction are copied from the full differential table." if match else "Gene statistics do not match the full table."))
        for edge in chain["edges"]:
            for evidence in edge["evidence"]:
                kind = evidence["evidence_type"]
                record_id = evidence["source_record_id"]
                if kind == "data_supported":
                    ok = record_id in de_ids or record_id in enrich_ids
                    rows.append(_audit_row(chain["chain_id"], edge["edge_id"], "data_supported_cites_real_record", "pass" if ok else "fail", record_id if ok else "Cited record is missing."))
                elif kind == "kg_supported":
                    rows.append(_audit_row(chain["chain_id"], edge["edge_id"], "kg_supported_has_hit", "fail", "No knowledge-graph hit record was saved."))
                elif kind == "literature_supported":
                    rows.append(_audit_row(chain["chain_id"], edge["edge_id"], "literature_supported_has_source", "fail", "No literature record was saved."))
                causal = any(word in edge["relation"] for word in ("cause", "activat", "inhibit", "promot"))
                rows.append(_audit_row(chain["chain_id"], edge["edge_id"], "relation_is_not_causal", "fail" if causal else "pass", edge["relation"]))
        whole = all(
            evidence["evidence_type"] == "data_supported" and "disease action" in evidence["supported_claim"].lower()
            for edge in chain["edges"]
            for evidence in edge["evidence"]
        )
        rows.append(_audit_row(chain["chain_id"], chain["chain_id"], "whole_chain_not_marked_observed", "pass", "Expression edges and annotation overlaps are separate. No whole-chain observed label is used."))
        role = chain["disease_role"]["text"]
        role_bad = role.strip().lower() in {"alzheimer disease", "alzheimer's disease", "higher in disease", "疾病组更高"}
        rows.append(_audit_row(chain["chain_id"], chain["chain_id"], "disease_name_not_used_as_action", "fail" if role_bad else "pass", "Disease role is empty pending a specific supported action." if not role else role))
        rows.append(_audit_row(chain["chain_id"], chain["chain_id"], "activity_direction_not_inferred", "pass" if chain["functional_interpretation"]["activity_direction_supported"] is False else "fail", chain["functional_interpretation"]["activity_direction_reason"]))
    if not any(row["check"] == "kg_supported_has_hit" and row["status"] == "fail" for row in rows):
        rows.append(_audit_row("dataset", "kg", "kg_records_absent", "not_applicable", "No edge claims kg_supported. The hit files themselves are missing and are listed in missing_inputs."))
    return rows


def _audit_row(chain_id: str, edge_id: str, check: str, status: str, reason: str) -> dict[str, str]:
    return {"chain_id": chain_id, "edge_id": edge_id, "check": check, "status": status, "reason": reason}


def _markdown(payload: dict[str, Any], chains: list[dict[str, Any]]) -> str:
    lines = [
        "# Grubman 2019 链条 v2",
        "",
        "链条从完整差异表和 Enrichr 原始结果重新连接。旧的 `chains/` 与 `chains_and_conclusions.md` 仍保留。",
        "",
        "正的 log2FC 表示疾病组更高。没有供体编号，细胞级 Wilcoxon 没有控制供体差异。",
        "基因集与 GO 条目重叠只说明注释重叠，不说明该过程增强或减弱。",
        "本次没有保存知识图谱命中记录，也没有文献检索记录，因此没有 `kg_supported` 或 `literature_supported` 连接。",
        "",
        "## 证据类别",
        "",
        "- data_supported：当前差异表或富集表直接支持的连接。",
        "- kg_supported：本次无可用命中，未使用。",
        "- literature_supported：本次无检索记录，未使用。",
        "- hypothesis：具体疾病作用没有上述证据，字段留空，不补写保护、代偿、神经炎症或疾病名称。",
        "",
    ]
    current = None
    for chain in chains:
        if chain["cell_type"] != current:
            current = chain["cell_type"]
            lines.extend(["", f"## {current}", "", "| 链条 | 方向 | 代表条目 | 条目数 | 基因数 |", "|---|---|---|---:|---:|"])
            for item in chains:
                if item["cell_type"] != current:
                    continue
                lead = item["functional_evidence"][0]["name"]
                direction = item["gene_changes"][0]["direction"] if item["gene_changes"] else ""
                lines.append(
                    f"| {item['chain_id']} | {direction} | {lead} | {len(item['functional_evidence'])} | {len(item['gene_changes'])} |"
                )
            lines.append("")
        lines.extend(_chain_markdown(chain))
    if not chains:
        lines.append("没有可连接的富集条目。")
    lines.append("")
    return "\n".join(lines)


def _chain_markdown(chain: dict[str, Any]) -> list[str]:
    lines = [f"### {chain['chain_id']}", ""]
    shown_genes = chain["gene_changes"][:12]
    lines.append("数据观察：")
    lines.append("")
    lines.append("| 基因 | log2FC | 校正后 P | 方向 | 记录 |")
    lines.append("|---|---:|---:|---|---|")
    for gene in shown_genes:
        lines.append(
            f"| {gene['gene']} | {gene['logFC']:.4g} | {gene['adjusted_P']:.3g} | {gene['direction']} | {gene['record_id']} |"
        )
    hidden_genes = len(chain["gene_changes"]) - len(shown_genes)
    if hidden_genes:
        lines.append("")
        lines.append(f"其余 {hidden_genes} 个重叠基因的统计值在 `chains_v2.json` 的 `{chain['chain_id']}.gene_changes`。")
    lines.extend(["", "功能条目重叠："])
    lines.append("")
    for term in chain["functional_evidence"][:8]:
        overlap = ", ".join(term["overlap_genes"][:8])
        more = len(term["overlap_genes"]) - min(8, len(term["overlap_genes"]))
        suffix = f" 等 {len(term['overlap_genes'])} 个" if more else ""
        lines.append(
            f"- {term['name']}（{term['record_id']}，输入方向 {term['input_direction']}，校正后 P {term['adjusted_P']:.3g}）。"
            f"重叠基因：{overlap}{suffix}。"
        )
    hidden_terms = len(chain["functional_evidence"]) - min(8, len(chain["functional_evidence"]))
    if hidden_terms:
        lines.append(f"- 其余 {hidden_terms} 条原始条目保留在该链的 `functional_evidence`。")
    lines.extend(["", "功能解释：", "", chain["functional_interpretation"]["text"], ""])
    if chain["conflicts"]:
        lines.append("方向冲突：同一重叠不能同时被读成过程增强和减弱。")
        lines.append("")
    lines.extend(
        [
            "知识图谱支持：无已保存的命中关系。",
            "",
            "文献支持：无已保存的检索记录。",
            "",
            "疾病作用：未填写。缺少能把注释重叠连接到具体作用的证据。",
            "",
        ]
    )
    return lines


def _change_log(
    output_dir: Path,
    payload: dict[str, Any],
    chains: list[dict[str, Any]],
    labels: dict[str, str],
) -> str:
    lines = [
        "# Chain v2 变更记录",
        "",
        "## 问题发生在哪一层",
        "",
    ]
    for key, text in payload["stage_diagnosis"].items():
        lines.append(f"- {key}：{text}")
    lines.extend(["", "## 缺失而不能重建的材料", ""])
    if payload["missing_inputs"]:
        for item in payload["missing_inputs"]:
            lines.append(f"- {item['item']}：{item['reason']}")
    else:
        lines.append("- 无。")
    lines.extend(["", "## 相对旧链条的删除、合并和降级", ""])
    enrich_names = {
        term["name"].lower()
        for chain in chains
        for term in chain["functional_evidence"]
    }
    enrich_genes = {}
    for chain in chains:
        for term in chain["functional_evidence"]:
            enrich_genes.setdefault(term["name"].lower(), set()).update(term["overlap_genes"])
    old_dir = output_dir / "chains"
    if not old_dir.exists():
        lines.append("旧 `chains/` 不存在，没有可对照的条目。")
        return "\n".join(lines) + "\n"
    for path in sorted(old_dir.glob("*_chains.json")):
        old = json.loads(path.read_text())
        cell_id = path.name[: -len("_chains.json")]
        label = labels.get(cell_id, cell_id)
        lines.append(f"### {label}")
        lines.append("")
        for number, chain in enumerate(old.get("chains") or [], start=1):
            lines.extend(_old_chain_decision(number, chain, enrich_names, enrich_genes))
        if not old.get("chains"):
            lines.append("- 旧文件没有链条。")
            lines.append("")
    lines.extend(
        [
            "## 没有改动的部分",
            "",
            "- 差异分析阈值、归一化和细胞类型标签没有重算。",
            "- 旧的差异表、富集表、模型原文、旧链条和旧报告都保留。",
            "",
        ]
    )
    return "\n".join(lines)


def _match_term_name(
    process: str,
    enrich_names: set[str],
    enrich_genes: dict[str, set[str]],
    mentioned: set[str],
) -> str:
    key = process.lower()
    substring = [name for name in enrich_names if key and (key in name or name in key)]
    if substring:
        return max(substring, key=lambda name: len(mentioned & enrich_genes.get(name, set())))
    old_tokens = _term_tokens(process)
    best = ""
    best_score = 0.0
    for name in enrich_names:
        tokens = _term_tokens(name)
        if not old_tokens or not tokens:
            continue
        score = len(old_tokens & tokens) / len(old_tokens | tokens)
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= 0.45 else ""


def _term_tokens(text: str) -> set[str]:
    stop = {"of", "the", "a", "and", "in", "to", "via", "for"}
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stop}


def _old_chain_decision(
    number: int,
    chain: dict[str, Any],
    enrich_names: set[str],
    enrich_genes: dict[str, set[str]],
) -> list[str]:
    process = str(chain.get("function_process") or "")
    basis = str(chain.get("function_basis") or "")
    role = str(chain.get("disease_role") or "")
    support = str(chain.get("support") or "")
    skip = {"GENES", "GENE", "UP", "DOWN", "UP-REGULATED", "DOWN-REGULATED", "REGULATED"}
    mentioned = {gene for gene in GENE_TOKEN.findall(basis.upper()) if gene not in skip}
    matched_name = _match_term_name(process, enrich_names, enrich_genes, mentioned)
    lines = [f"- 旧链条 {number}：{process or '未命名'}。支持等级曾标为 `{support}`，这是整条链共用的标签，已取消。"]
    if not matched_name:
        lines.append(f"  - 删除：功能条目不在该细胞类型的 Enrichr 原始结果中（{process}）。")
    else:
        allowed = enrich_genes.get(matched_name, set())
        absent = sorted(gene for gene in mentioned if gene not in allowed)
        if absent:
            lines.append(f"  - 删除基因连接：{', '.join(absent)} 不在对应条目的 overlap 中。")
        lines.append(f"  - 合并：与原始条目 “{matched_name}” 属于同一重叠簇，原始条目编号保留在 v2。")
    if role.strip():
        lines.append(f"  - 降级：疾病作用“{role}”没有对应的 KG 命中或文献记录，v2 不保留为事实。")
    if "observed" in support.lower():
        lines.append("  - 降级：整条链的 observed 只可能覆盖表达变化，不能覆盖功能或疾病作用。")
    return lines


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Rebuild audited chains from an existing disease-pipeline output")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    build_chain_v2(Path(cfg["output_dir"]), cfg)
