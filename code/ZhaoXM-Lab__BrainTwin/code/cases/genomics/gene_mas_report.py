"""
Batch gene function / trait analysis for variant prioritization hit genes.

Uses deterministic BrainTwin KG batch queries + two focused LLM calls (not per-gene,
not Graph-RAG Cypher generation).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import CODE as V2_RAG_DIR, DATA

MAS_GENE_CAP = int(os.environ.get("GENE_MAS_CAP", "80"))
MAS_MODERATE_CAP = int(os.environ.get("GENE_MAS_MODERATE_CAP", "40"))

FUNCTIONS_SYSTEM = """
You are a genomics expert summarizing biological functions for batches of hit genes
in brain genetic risk reports. Synthesize shared themes; do not write one paragraph per gene.
"""

FUNCTIONS_USER = """
Hit genes from snpEff variant prioritization (HIGH impact first):
HIGH: {high_genes}
MODERATE subset: {moderate_genes}

BrainTwin KG — GO terms (batch query):
{go_table}

BrainTwin KG — cell types (batch query):
{celltype_table}

Summarize the main biological functions and pathways represented by this gene set.
Group by functional themes (e.g. synaptic signaling, lipid metabolism), not by individual genes.
Write 1-2 short paragraphs (under 180 words). Avoid diagnosis language and disclaimer sections.
"""

TRAITS_SYSTEM = """
You are a genomics expert linking gene sets to brain-related traits and phenotypes
for polygenic risk reports. Synthesize themes; do not analyze genes one-by-one.
"""

TRAITS_USER = """
Hit genes from snpEff variant prioritization (HIGH impact first):
HIGH: {high_genes}
MODERATE subset: {moderate_genes}

Polygenic risk context (UK Biobank reference):
{prs_context}

BrainTwin KG — associated traits (batch query):
{trait_table}

Enrichr (ClinVar / OMIM):
{enrichr_context}

What traits, phenotypes, or brain-related conditions might this gene set be associated with?
Connect to the PRS profile where plausible. Write 1-2 short paragraphs (under 180 words).
Avoid deterministic diagnosis language and disclaimer sections.
"""


def select_genes_for_mas(variant: dict[str, Any]) -> dict[str, Any]:
    high = list(variant.get("hit_genes_high") or [])
    moderate = list(variant.get("hit_genes_moderate") or [])

    if not high and not moderate:
        gene_impact: dict[str, str] = {}
        for row in variant.get("all_variants") or variant.get("top_variants") or []:
            gene = str(row.get("gene") or "").strip()
            if not gene:
                continue
            impact = str(row.get("impact") or "MODERATE").upper()
            if impact == "HIGH":
                gene_impact[gene] = "HIGH"
            elif gene not in gene_impact:
                gene_impact[gene] = "MODERATE"
        high = sorted(
            [gene for gene, impact in gene_impact.items() if impact == "HIGH"],
            key=str.upper,
        )
        moderate = sorted(
            [gene for gene, impact in gene_impact.items() if impact != "HIGH"],
            key=str.upper,
        )

    selected_high = high[:MAS_GENE_CAP]
    remaining = max(0, MAS_GENE_CAP - len(selected_high))
    selected_moderate = moderate[: min(remaining, MAS_MODERATE_CAP)]

    return {
        "high": selected_high,
        "moderate": selected_moderate,
        "all": selected_high + selected_moderate,
        "total_high": len(high),
        "total_moderate": len(moderate),
        "truncated": (len(high) + len(moderate)) > (len(selected_high) + len(selected_moderate)),
    }


def _fetch_kg_batch_context(genes: list[str]) -> dict[str, Any]:
    if not genes:
        return {"available": False, "message": "No genes provided"}

    try:
        from neo4j import GraphDatabase
        from multiagent_system.config import Config

        config = Config()
        driver = GraphDatabase.driver(
            config.neo4j_url,
            auth=(config.neo4j_username, config.neo4j_password),
        )

        go_query = """
        MATCH (g:Gene)-[]->(go:GOTerm)
        WHERE g.Gene_Name IN $gene_list
        WITH go, COLLECT(DISTINCT g.Gene_Name) AS genes
        RETURN go.Go_Term_Name AS term,
               size(genes) AS gene_count,
               genes AS genes
        ORDER BY gene_count DESC
        LIMIT 25
        """
        trait_query = """
        MATCH (g:Gene)-[:ASSOCIATE_WITH_TRAIT]->(t:Trait)
        WHERE g.Gene_Name IN $gene_list
        WITH t, COLLECT(DISTINCT g.Gene_Name) AS genes
        RETURN t.Trait_Name AS trait,
               size(genes) AS gene_count,
               genes AS genes
        ORDER BY gene_count DESC
        LIMIT 25
        """
        celltype_query = """
        MATCH (g:Gene)-[]->(ct:CellType)
        WHERE g.Gene_Name IN $gene_list
        WITH ct, COLLECT(DISTINCT g.Gene_Name) AS genes
        RETURN ct.CellType_Name AS celltype,
               size(genes) AS gene_count,
               genes AS genes
        ORDER BY gene_count DESC
        LIMIT 15
        """

        with driver.session() as session:
            go_rows = [dict(record) for record in session.run(go_query, gene_list=genes)]
            trait_rows = [dict(record) for record in session.run(trait_query, gene_list=genes)]
            celltype_rows = [dict(record) for record in session.run(celltype_query, gene_list=genes)]
        driver.close()

        return {
            "available": True,
            "go_terms": go_rows,
            "traits": trait_rows,
            "cell_types": celltype_rows,
        }
    except Exception as exc:
        return {"available": False, "message": str(exc)}


def _format_gene_list(genes: list[str], limit: int = 60) -> str:
    if not genes:
        return "(none)"
    preview = genes[:limit]
    text = ", ".join(preview)
    if len(genes) > limit:
        text += f", ... (+{len(genes) - limit} more)"
    return text


def _format_kg_table(rows: list[dict[str, Any]], label_key: str) -> str:
    if not rows:
        return "(no matches in BrainTwin KG)"
    lines = [f"| {label_key} | # genes | example genes |", "| --- | ---: | --- |"]
    for row in rows[:20]:
        genes = row.get("genes") or []
        if isinstance(genes, list):
            gene_preview = ", ".join(genes[:5])
            if len(genes) > 5:
                gene_preview += f", ... (+{len(genes) - 5})"
        else:
            gene_preview = str(genes)
        lines.append(
            f"| {row.get(label_key, '')} | {row.get('gene_count', '')} | {gene_preview} |"
        )
    return "\n".join(lines)


def _format_prs_context(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no PRS rows)"
    elevated = sorted(rows, key=lambda item: item.get("percentile", 0), reverse=True)[:5]
    lowered = sorted(rows, key=lambda item: item.get("percentile", 100))[:3]
    lines = ["Elevated PRS percentiles:"]
    for row in elevated:
        lines.append(
            f"- {row.get('label', row.get('disorder', ''))}: "
            f"{row.get('percentile')}% (z={row.get('z_score')})"
        )
    lines.append("Lower PRS percentiles:")
    for row in lowered:
        lines.append(
            f"- {row.get('label', row.get('disorder', ''))}: "
            f"{row.get('percentile')}% (z={row.get('z_score')})"
        )
    return "\n".join(lines)


def _format_enrichr_context(variant: dict[str, Any]) -> str:
    terms = variant.get("enrichr_terms") or []
    if not terms:
        return "(no Enrichr terms)"
    lines = []
    for term in terms[:8]:
        lines.append(
            f"- {term.get('library', '')}: {term.get('term', '')} "
            f"(adj.p={term.get('adjusted_p')}, genes={term.get('genes', '')})"
        )
    return "\n".join(lines)


def build_functions_prompt(
    gene_selection: dict[str, Any],
    kg_context: dict[str, Any],
) -> str:
    return FUNCTIONS_USER.format(
        high_genes=_format_gene_list(gene_selection.get("high") or []),
        moderate_genes=_format_gene_list(gene_selection.get("moderate") or []),
        go_table=_format_kg_table(kg_context.get("go_terms") or [], "term"),
        celltype_table=_format_kg_table(kg_context.get("cell_types") or [], "celltype"),
    ).strip()


def build_traits_prompt(
    gene_selection: dict[str, Any],
    variant: dict[str, Any],
    prs_rows: list[dict[str, Any]],
    kg_context: dict[str, Any],
) -> str:
    return TRAITS_USER.format(
        high_genes=_format_gene_list(gene_selection.get("high") or []),
        moderate_genes=_format_gene_list(gene_selection.get("moderate") or []),
        prs_context=_format_prs_context(prs_rows),
        trait_table=_format_kg_table(kg_context.get("traits") or [], "trait"),
        enrichr_context=_format_enrichr_context(variant),
    ).strip()


def _run_llm(system_prompt: str, user_prompt: str) -> str:
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
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content=user_prompt),
        ]
    )
    return (getattr(response, "content", None) or str(response)).strip()


def analyze_hit_genes_with_mas(
    variant: dict[str, Any] | None,
    prs_rows: list[dict[str, Any]],
    *,
    stage: str = "all",
) -> dict[str, Any]:
    if not variant or not variant.get("available"):
        return {"available": False, "message": "Variant prioritization not available"}

    if os.environ.get("GENE_MAS_DISABLE") == "1":
        return {"available": False, "message": "Gene analysis disabled"}

    gene_selection = select_genes_for_mas(variant)
    if not gene_selection["all"]:
        return {"available": False, "message": "No hit genes to analyze"}

    kg_context = _fetch_kg_batch_context(gene_selection["all"])
    functions_prompt = build_functions_prompt(gene_selection, kg_context)
    traits_prompt = build_traits_prompt(gene_selection, variant, prs_rows, kg_context)

    gene_functions = ""
    associated_traits = ""

    try:
        if stage in {"all", "functions"}:
            gene_functions = _run_llm(FUNCTIONS_SYSTEM, functions_prompt)
        if stage in {"all", "traits"}:
            associated_traits = _run_llm(TRAITS_SYSTEM, traits_prompt)
    except Exception as exc:
        return {
            "available": False,
            "message": f"Gene analysis failed: {exc}",
            "gene_selection": gene_selection,
            "kg_context": kg_context,
            "prompts": {
                "functions": functions_prompt,
                "traits": traits_prompt,
            },
        }

    if not gene_functions and not associated_traits:
        return {
            "available": False,
            "message": "Gene analysis returned empty responses",
            "gene_selection": gene_selection,
            "kg_context": kg_context,
        }

    return {
        "available": True,
        "gene_functions": gene_functions,
        "associated_traits": associated_traits,
        "gene_selection": gene_selection,
        "kg_context": kg_context,
        "prompts": {
            "functions": functions_prompt,
            "traits": traits_prompt,
        },
    }


def iter_gene_mas_events(
    variant: dict[str, Any] | None,
    prs_rows: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    if not variant or not variant.get("available"):
        yield {
            "type": "gene_mas_done",
            "gene_mas_analysis": {"available": False, "message": "Variant prioritization not available"},
        }
        return

    gene_selection = select_genes_for_mas(variant)
    if not gene_selection["all"]:
        yield {
            "type": "gene_mas_done",
            "gene_mas_analysis": {"available": False, "message": "No hit genes to analyze"},
        }
        return

    kg_context = _fetch_kg_batch_context(gene_selection["all"])
    functions_prompt = build_functions_prompt(gene_selection, kg_context)
    traits_prompt = build_traits_prompt(gene_selection, variant, prs_rows, kg_context)

    yield {"type": "status", "text": "Summarizing hit gene functions (batch)"}
    try:
        gene_functions = _run_llm(FUNCTIONS_SYSTEM, functions_prompt)
    except Exception as exc:
        yield {
            "type": "gene_mas_done",
            "gene_mas_analysis": {
                "available": False,
                "message": f"Gene function analysis failed: {exc}",
                "prompts": {"functions": functions_prompt, "traits": traits_prompt},
            },
        }
        return

    yield {"type": "status", "text": "Summarizing associated traits (batch)"}
    try:
        associated_traits = _run_llm(TRAITS_SYSTEM, traits_prompt)
    except Exception as exc:
        yield {
            "type": "gene_mas_done",
            "gene_mas_analysis": {
                "available": False,
                "message": f"Trait analysis failed: {exc}",
                "gene_functions": gene_functions,
                "prompts": {"functions": functions_prompt, "traits": traits_prompt},
            },
        }
        return

    yield {"type": "status", "text": "Gene function and trait summaries ready"}
    yield {
        "type": "gene_mas_done",
        "gene_mas_analysis": {
            "available": True,
            "gene_functions": gene_functions,
            "associated_traits": associated_traits,
            "gene_selection": gene_selection,
            "kg_context": kg_context,
            "prompts": {
                "functions": functions_prompt,
                "traits": traits_prompt,
            },
        },
    }
