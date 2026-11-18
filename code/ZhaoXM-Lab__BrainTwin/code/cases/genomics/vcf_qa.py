"""
VCF question answering.

Uses the same PRS scoring, snpEff lookup, and knowledge-graph trait query as
Genetic Risk Analysis. The model only answers the disease named in the question.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA

_SAVED_SCORES: dict[str, dict[str, Any]] | None = None
_EXPERIMENT_Z: dict[str, dict[str, dict[str, float]]] | None = None
DEMO_SAMPLE_ID = "4033490"
EXPERIMENT_Z_CSV = str(DATA / "genomics" / "ukb" / "UKB_PRS_result_zscore.csv")

DISEASES: list[dict[str, Any]] = [
    {"id": "AD", "aliases": ["alzheimer"], "traits": ["alzheimer", "dementia", "amyloid", "phf-tau", "tau"]},
    {"id": "SCZ", "aliases": ["schizophrenia"], "traits": ["schizophrenia", "schizoaffective"]},
    {"id": "MDD", "aliases": ["depression", "depressive"], "traits": ["depression", "depressive"]},
    {"id": "Anxiety", "aliases": ["anxiety"], "traits": ["anxiety", "panic"]},
    {"id": "Bipolar", "aliases": ["bipolar"], "traits": ["bipolar", "manic"]},
    {"id": "epilepsy", "aliases": ["epilepsy", "seizure"], "traits": ["epilep", "seizure"]},
    {"id": "PTSD", "aliases": ["ptsd"], "traits": ["ptsd", "post-traumatic", "posttraumatic"]},
    {"id": "Parkinsons-disease", "aliases": ["parkinson"], "traits": ["parkinson"]},
    {"id": "ADHD", "aliases": ["adhd"], "traits": ["adhd", "attention deficit", "hyperactiv"]},
    {"id": "Stroke", "aliases": ["stroke"], "traits": ["stroke", "cerebrovascular"]},
    {"id": "ASD", "aliases": ["autism"], "traits": ["autism"]},
]


def _matches_alias(question: str, alias: str) -> bool:
    if any("\u4e00" <= ch <= "\u9fff" for ch in alias):
        return alias.lower() in question.lower()
    return re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", question, flags=re.IGNORECASE) is not None


def diseases_in_question(question: str) -> list[dict[str, Any]]:
    found = []
    for disease in DISEASES:
        if any(_matches_alias(question, alias) for alias in disease["aliases"]):
            found.append(disease)
    return found


def _variant_index(report: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for row in report.get("all_variants") or report.get("top_variants") or []:
        if str(row.get("impact") or "").upper() != "HIGH":
            continue
        gene = str(row.get("gene") or "").strip()
        rsid = str(row.get("rsid") or "").strip()
        if not gene:
            continue
        rows.append(
            {
                "gene": gene,
                "rsid": rsid,
                "chrom": str(row.get("chrom") or ""),
                "pos": str(row.get("pos") or ""),
            }
        )
    return rows


def _query_traits(genes: list[str], keywords: list[str]) -> list[dict[str, Any]]:
    if not genes or not keywords:
        return []
    try:
        from neo4j import GraphDatabase
        from multiagent_system.config import Config

        config = Config()
        driver = GraphDatabase.driver(config.neo4j_url, auth=(config.neo4j_username, config.neo4j_password))
        query = """
        MATCH (g:Gene)-[:ASSOCIATE_WITH_TRAIT]->(t:Trait)
        WHERE g.Gene_Name IN $genes
          AND any(k IN $keywords WHERE toLower(t.Trait_Name) CONTAINS k)
        WITH t, COLLECT(DISTINCT g.Gene_Name) AS genes
        RETURN t.Trait_Name AS trait, size(genes) AS gene_count, genes AS genes
        ORDER BY gene_count DESC
        LIMIT 20
        """
        try:
            with driver.session() as session:
                return [dict(record) for record in session.run(query, genes=genes, keywords=keywords)]
        finally:
            driver.close()
    except Exception:
        return []


def _risk_phrase(z_score: float) -> str:
    if z_score >= 2:
        return "well above the reference group"
    if z_score >= 1:
        return "above the reference group"
    if z_score <= -2:
        return "well below the reference group"
    if z_score <= -1:
        return "below the reference group"
    return "near the reference average"


def _variants_by_gene(variants: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in variants:
        grouped.setdefault(item["gene"], []).append(item)
    return grouped


def _format_gene(gene: str, by_gene: dict[str, list[dict[str, str]]]) -> str:
    hits = by_gene.get(gene) or []
    if not hits:
        return gene
    hit = hits[0]
    location = f"{hit['chrom']}:{hit['pos']}" if hit.get("chrom") and hit.get("pos") else ""
    detail = " ".join(part for part in (hit.get("rsid") or "", location) if part)
    return f"{gene} ({detail})" if detail else gene


def _trait_bucket(trait: str, diseases: list[dict[str, Any]]) -> str:
    name = trait.lower()
    if any(marker in name for marker in ("pleiotrop", "educational attainment", "fasting glucose", "serum amyloid", "cognitive decline", "intelligence")):
        return "related"
    core = []
    for disease in diseases:
        core.extend(alias.lower() for alias in disease["aliases"] if alias.isascii())
        core.extend(token for token in disease["traits"] if token in {"dementia", "schizophrenia", "depression", "bipolar", "epilep", "stroke", "autism", "parkinson", "anxiety", "ptsd", "adhd"})
    if any(token and token in name for token in core):
        return "direct"
    return "related"


def _evidence(question: str, rows: list[dict[str, Any]], genes: list[str], variants: list[dict[str, str]]) -> tuple[str, list[str]]:
    selected = diseases_in_question(question)
    if not selected:
        return "", []
    by_id = {row["disorder"]: row for row in rows}
    by_gene = _variants_by_gene(variants)
    prs_lines = []
    keywords: list[str] = []
    for disease in selected:
        keywords.extend(disease["traits"])
        row = by_id.get(disease["id"])
        if not row:
            prs_lines.append(f"{disease['id']}: this PRS is not available.")
            continue
        prs_lines.append(
            f"{row['label']}: z={row['z_score']}, percentile={row['percentile']}%, "
            f"relative to the reference group: {_risk_phrase(float(row['z_score']))}. "
            "The overall direction must follow this phrase and must not be reversed."
        )
    traits = _query_traits(genes, keywords)
    direct_lines = []
    related_lines = []
    for trait in traits:
        trait_name = str(trait.get("trait") or "")
        trait_genes = [gene for gene in (trait.get("genes") or []) if gene in genes]
        if not trait_name or not trait_genes or "taurine" in trait_name.lower():
            continue
        joined = "；".join(_format_gene(gene, by_gene) for gene in trait_genes[:8])
        line = f"{trait_name}：{joined}"
        if _trait_bucket(trait_name, selected) == "direct":
            direct_lines.append(line)
        else:
            related_lines.append(line)
    sections = [
        "[Polygenic scores]",
        *prs_lines,
        "[Traits named by the disease]",
        *(direct_lines or ["None."]),
        "[Indirect traits, including tau, amyloid, cognitive decline, and pleiotropy]",
        *(related_lines or ["None."]),
    ]
    return "\n".join(sections), [disease["id"] for disease in selected]


class VcfQaSession:
    def __init__(self, sample_id: str, rows: list[dict[str, Any]], genes: list[str], variants: list[dict[str, str]]):
        self.sample_id = sample_id
        self.rows = rows
        self.genes = genes
        self.variants = variants
        self._llm = None

    def summary(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "n_disorders": len(self.rows),
            "n_high_genes": len(self.genes),
        }

    def _complete(self, system: str, user: str) -> str:
        if self._llm is None:
            from LLM_utils import CustomOpenAIProxyChat
            from multiagent_system.config import Config

            config = Config()
            self._llm = CustomOpenAIProxyChat(
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model_id,
                streaming=False,
            )
        result = self._llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return getattr(result, "content", str(result))

    def iter_ask_events(self, question: str) -> Iterator[dict[str, Any]]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        yield {"type": "status", "text": "Matching the disease in the question"}
        evidence, diseases = _evidence(question, self.rows, self.genes, self.variants)
        if not diseases:
            names = "、".join(sorted({row["label"] for row in self.rows}))
            answer = f"Name one disease in the question. PRS values were computed for: {names}."
            yield {"type": "done", "answer": answer, "features": [], "evidences": answer}
            return
        yield {"type": "status", "text": "Reading PRS, HIGH-impact genes, and knowledge-graph traits"}
        chinese = any("\u4e00" <= ch <= "\u9fff" for ch in question)
        if chinese:
            system = (
                "Answer only the disease named in the question. Use the genes, rs IDs, and trait names in the evidence. Do not invent any. "
                "Do not say this person will develop the disease. Write four paragraphs in English, with a blank line between paragraphs.\n"
                "Paragraph 1 is the overall judgment. The first sentence must keep the evidence phrase about the reference group and must include the z score and percentile. "
                "State that this is polygenic risk relative to the reference group, not a diagnosis.\n"
                "Paragraph 2 explains how the polygenic score pushes the judgment higher, near average, or lower.\n"
                "Paragraph 3 explains the gene evidence. First cover traits named by the disease, then indirect traits such as tau, amyloid, cognitive decline, and pleiotropy. "
                "Use one sentence per trait, in the form 'gene (rs ID) is associated with this trait'. Indirect traits are supporting evidence and must not reverse the overall direction.\n"
                "Paragraph 4 uses two or three sentences to say whether these evidence classes agree, and whether that keeps or limits the overall judgment."
            )
        else:
            system = (
                "Answer only the disease named in the question. Use genes, rsIDs, and trait names from the evidence. "
                "Do not invent associations, and do not say the person will develop the disease. "
                "Write four paragraphs separated by a blank line.\n"
                "Paragraph 1 is the overall judgment. The first sentence must keep the evidence wording for whether risk is "
                "higher than, lower than, or near the reference population, and must include the z-score and percentile. "
                "State that this is polygenic risk relative to the reference population, not a diagnosis.\n"
                "Paragraph 2 explains how the polygenic score moves that judgment.\n"
                "Paragraph 3 explains the gene evidence: disease-named traits first, then indirect traits such as tau, "
                "amyloid, cognitive decline, and pleiotropy. One trait per sentence, as “gene (rsID) is associated with the trait”. "
                "Indirect traits are supporting context and must not reverse the overall judgment.\n"
                "Paragraph 4 says whether the evidence agrees, and whether that keeps or limits the judgment."
            )
        try:
            answer = self._complete(system, f"Question:\n{question}\n\nEvidence:\n{evidence}").strip()
        except Exception as exc:
            answer = evidence or str(exc)
        if not answer:
            answer = evidence
        yield {
            "type": "done",
            "answer": answer,
            "features": diseases,
            "evidences": evidence,
        }


def experiment_z_scores(sample_id: str) -> dict[str, dict[str, float]] | None:
    global _EXPERIMENT_Z
    if _EXPERIMENT_Z is None:
        columns: dict[str, list[tuple[str, float]]] = {}
        with open(EXPERIMENT_Z_CSV, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sid = row.get("") or row.get("sample_id") or ""
                for disease, value in row.items():
                    if not disease or not sid:
                        continue
                    columns.setdefault(disease, []).append((sid, float(value)))
        lookup: dict[str, dict[str, dict[str, float]]] = {}
        for disease, pairs in columns.items():
            ordered = sorted(pairs, key=lambda item: item[1])
            total = len(ordered)
            for rank, (sid, value) in enumerate(ordered, start=1):
                lookup.setdefault(sid, {})[disease] = {
                    "z": value,
                    "percentile": rank / total * 100,
                }
        _EXPERIMENT_Z = lookup
    return _EXPERIMENT_Z.get(str(sample_id))


def apply_experiment_z(sample_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    saved = experiment_z_scores(sample_id)
    if not saved:
        return rows
    updated = []
    for row in rows:
        score = saved.get(row["disorder"])
        if not score:
            updated.append(row)
            continue
        copied = dict(row)
        copied["z_score"] = round(score["z"], 3)
        copied["percentile"] = round(score["percentile"], 1)
        updated.append(copied)
    return updated


def saved_raw_scores(sample_id: str) -> dict[str, Any] | None:
    global _SAVED_SCORES
    if not sample_id:
        return None
    if _SAVED_SCORES is None:
        from prs_report import DEFAULT_REFERENCE_JSON

        with open(DEFAULT_REFERENCE_JSON, encoding="utf-8") as handle:
            _SAVED_SCORES = json.load(handle)
    return _SAVED_SCORES.get(str(sample_id))


def iter_session_events(sample_id: str | None = None, vcf_path: str | None = None) -> Iterator[dict[str, Any]]:
    from prs_report import calculate_prs_batch, enrich_scores, get_prs_service
    from variant_prioritization_service import extract_sample_id, find_precomputed_dir, iter_variant_events

    if vcf_path and not sample_id:
        sample_id = extract_sample_id(vcf_path)
    saved = saved_raw_scores(sample_id or "")
    service = get_prs_service()
    if saved is not None:
        yield {"type": "status", "text": "Using saved results for this sample"}
        rows = apply_experiment_z(sample_id or "", enrich_scores(saved, service.reference))
    elif vcf_path:
        yield {"type": "status", "text": "Scoring the VCF against the PRS panels"}
        raw_scores = calculate_prs_batch(vcf_path, service.pgs_dicts)
        rows = enrich_scores(raw_scores, service.reference)
        sample_id = extract_sample_id(vcf_path)
        rows = apply_experiment_z(sample_id or "", rows)
    else:
        raise ValueError("No saved scores for this sample.")

    report: dict[str, Any] = {"available": False}
    precomputed = find_precomputed_dir(sample_id or "") is not None
    if precomputed:
        yield {"type": "status", "text": "Using saved variant annotation"}
    for event in iter_variant_events(sample_id or "", vcf_path):
        if event.get("type") == "status" and not precomputed:
            yield event
        elif event.get("type") == "variant_done":
            report = event.get("variant_prioritization") or report
    genes = list(report.get("hit_genes_high") or [])
    variants = _variant_index(report) if report.get("available") else []
    session = VcfQaSession(sample_id or "", rows, genes, variants)
    yield {"type": "ready", "session": session}
