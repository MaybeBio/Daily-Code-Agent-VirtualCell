"""Resume KG full run for remaining diseases (ADHD, Stroke), then score all.

Self-contained (previous helper scripts were deleted). Checkpoint-skips existing files.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "braintwin"))
sys.path.insert(0, str(ROOT / "cases" / "genomics"))
UKB_DIR = Path(os.environ.get("BRAINTWIN_UKB_DIR", ROOT / "data" / "genomics" / "ukb"))

OUTPUT_DIR = ROOT / "cases" / "genomics" / "output" / "disease_risk_kg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KG_METHOD = "geneset"
PROMPT_VERSION = "v6"
ACTIVE_PROMPT_VERSION = "v6"
MIN_PATHOLOGY_GENES = 2  # pathology_traits require >= this many genes supporting the trait

EXPERT_URL = os.environ.get("BRAINTWIN_LLM_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_ID = os.environ.get("BRAINTWIN_MODEL_ID", "gpt-3.5-turbo")
NEO4J_URL = os.environ.get("BRAINTWIN_NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("BRAINTWIN_NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("BRAINTWIN_NEO4J_PASSWORD", "")

DISEASES = ["AD", "SCZ", "MDD", "Anxiety", "Bipolar", "epilepsy",
            "PTSD", "Parkinsons-disease", "ADHD", "Stroke"]
DISEASE_TO_META_COL = {
    "AD": "AD", "SCZ": "SCZ", "MDD": "Depression", "Anxiety": "Anxiety",
    "Bipolar": "Bipolar-Disorder", "epilepsy": "Epilepsy", "PTSD": "PTSD",
    "Parkinsons-disease": "Parkinsons-disease", "ADHD": "ADHD", "Stroke": "Stroke",
}
DISORDER_LABELS = {
    "AD": "Alzheimer's disease", "SCZ": "Schizophrenia", "MDD": "Major depressive disorder",
    "ADHD": "ADHD", "Anxiety": "Anxiety disorder", "Stroke": "Stroke",
    "Bipolar": "Bipolar disorder", "epilepsy": "Epilepsy",
    "PTSD": "PTSD", "Parkinsons-disease": "Parkinson's disease",
}
DISEASE_PATHOLOGY_KEYWORDS = {
    "AD": ["alzheimer", "dementia", "amyloid", "phf-tau", "tau", "neurofibrillary",
           "abeta", "ab1-42", "cognitive decline in alzheimer", "late onset", "csf"],
    "SCZ": ["schizophrenia", "schizoaffective", "psychosis", "psychotic"],
    "MDD": ["major depressive", "depression", "depressive disorder", "mdd"],
    "Anxiety": ["anxiety disorder", "generalized anxiety", "panic disorder", "phobia"],
    "Bipolar": ["bipolar disorder", "manic", "mania"],
    "epilepsy": ["epilepsy", "seizure disorder", "convulsion"],
    "PTSD": ["ptsd", "post-traumatic stress", "posttraumatic stress"],
    "Parkinsons-disease": ["parkinson", "parkinson's disease"],
    "ADHD": ["adhd", "attention deficit hyperactivity"],
    "Stroke": ["stroke", "ischemic stroke", "cerebrovascular", "intracerebral"],
}
GENERIC_TRAIT_IGNORE = [
    "educational attainment", "intelligence", "cognitive performance", "cognitive ability",
    "brain morphology", "personality", "neuroticism", "memory in normal cognition",
    "cognitive aspects of educational", "mtag", "pleiotropy",
]
DISEASE_TRAIT_KEYWORDS = {
    "AD": ["alzheimer", "dementia", "amyloid", "tau", "cognitive", "educational attainment", "neurodegener", "memory"],
    "SCZ": ["schizophrenia", "schizoaffective", "psychosis", "psychotic"],
    "MDD": ["depression", "depressive", "mood", "neuroticism"],
    "Anxiety": ["anxiety", "neuroticism", "worry", "panic"],
    "Bipolar": ["bipolar", "manic", "mania"],
    "epilepsy": ["epilep", "seizure", "convulsion", "febrile"],
    "PTSD": ["ptsd", "post-traumatic", "posttraumatic", "traumatic"],
    "Parkinsons-disease": ["parkinson", "levodopa", "motor", "tremor"],
    "ADHD": ["attention deficit", "adhd", "hyperactiv", "inattention"],
    "Stroke": ["stroke", "cerebrovascular", "ischemic stroke", "intracerebral"],
}

CONCURRENCY = 8
LLM_TEMPERATURE = 0.1
_GENE_KG_CACHE: dict[str, dict[str, list[str]]] = {}
_GO_TOTAL_CACHE: dict[str, int] = {}
_Z_LOOKUP: dict = {}
_PCT_LOOKUP: dict = {}


def get_prs_z(sample_id: str, disease: str) -> tuple[float | None, float | None]:
    global _Z_LOOKUP, _PCT_LOOKUP
    if not _Z_LOOKUP:
        df = pd.read_csv(UKB_DIR / "UKB_PRS_result_zscore.csv", index_col=0)
        df.index = df.index.astype(str)
        _Z_LOOKUP = {sid: {d: float(df.loc[sid, d]) for d in df.columns} for sid in df.index}
        # percentiles optional
        _PCT_LOOKUP = {}
    z = _Z_LOOKUP.get(sample_id, {}).get(disease)
    p = _PCT_LOOKUP.get(sample_id, {}).get(disease)
    return z, p


def get_all_high_genes(sample_id: str) -> list[str]:
    from variant_prioritization_service import load_variant_report
    variant = load_variant_report(sample_id)
    if not variant.get("available"):
        return []
    return list(variant.get("hit_genes_high") or [])


def _kg_run(cypher: str) -> list[dict]:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS),
                                  connection_timeout=10, max_connection_lifetime=30)
    try:
        with driver.session() as s:
            return [dict(r) for r in s.run(cypher)]
    except Exception:
        return []
    finally:
        driver.close()


def gene_kg_relations(gene: str) -> dict[str, list[str]]:
    if gene in _GENE_KG_CACHE:
        return _GENE_KG_CACHE[gene]
    traits = [r["name"] for r in _kg_run(
        f"MATCH (g:Gene {{Gene_Name:'{gene}'}})-[:ASSOCIATE_WITH_TRAIT]->(t:Trait) "
        f"RETURN t.Trait_Name AS name LIMIT 50")]
    gos = [r["name"] for r in _kg_run(
        f"MATCH (g:Gene {{Gene_Name:'{gene}'}})-[:ANNOTATAED_AS]->(t:GOTerm) "
        f"RETURN t.Go_Term_Name AS name LIMIT 20")]
    cts = [r["name"] for r in _kg_run(
        f"MATCH (g:Gene {{Gene_Name:'{gene}'}})-[:IS_MARKER]->(c:CellType) "
        f"RETURN c.CellType_Name AS name LIMIT 20")]
    regions = []
    for ct in cts[:5]:
        r = _kg_run(f"MATCH (c:CellType {{CellType_Name:'{ct}'}})<-[:CONTAINS]-(br:BrainSubRegion) "
                    f"RETURN br.BrainSubRegion_Name AS name LIMIT 10")
        regions.extend(x["name"] for x in r)
    regions = list(dict.fromkeys(regions))
    rel = {"traits": traits, "gos": gos, "celltypes": cts, "regions": regions}
    _GENE_KG_CACHE[gene] = rel
    return rel


def trait_relevant(trait: str, disease: str) -> bool:
    kws = DISEASE_TRAIT_KEYWORDS.get(disease, [])
    tl = trait.lower()
    return any(k in tl for k in kws)


def trait_pathology(trait: str, disease: str) -> bool:
    """Strict disease-pathology match (for gene-set convergence)."""
    tl = trait.lower()
    if any(g in tl for g in GENERIC_TRAIT_IGNORE):
        return False
    kws = DISEASE_PATHOLOGY_KEYWORDS.get(disease, [])
    return any(k in tl for k in kws)


def trait_is_generic(trait: str) -> bool:
    tl = trait.lower()
    return any(g in tl for g in GENERIC_TRAIT_IGNORE)


def _cypher_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def go_total_gene_count(go_term: str) -> int:
    """Total distinct genes in KG annotated to this GO term (global, no cohort leakage)."""
    if go_term in _GO_TOTAL_CACHE:
        return _GO_TOTAL_CACHE[go_term]
    safe = _cypher_escape(go_term)
    rows = _kg_run(
        f"MATCH (g:Gene)-[:ANNOTATAED_AS]->(go:GOTerm {{Go_Term_Name:'{safe}'}}) "
        f"RETURN count(DISTINCT g) AS n"
    )
    n = int(rows[0]["n"]) if rows else 0
    _GO_TOTAL_CACHE[go_term] = n
    return n


def _enrich_go_rows(go_rows: list[dict], n_sample_genes: int) -> list[dict]:
    enriched = []
    for row in go_rows:
        sample_hits = row["count"]
        enriched.append({
            **row,
            "kg_total_genes": go_total_gene_count(row["term"]),
            "sample_hits": sample_hits,
            "sample_total": n_sample_genes,
        })
    return enriched


def _collect_gene_kg_records(all_high: list[str], disease: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for gene in all_high:
        rel = gene_kg_relations(gene)
        rel_traits = [t for t in rel["traits"] if trait_relevant(t, disease)]
        if not (rel["gos"] or rel["celltypes"] or rel["regions"] or rel_traits or rel["traits"]):
            records.append({"gene": gene, "gos": [], "celltypes": [], "regions": [], "relevant_traits": []})
            continue
        records.append({
            "gene": gene,
            "gos": rel["gos"],
            "celltypes": rel["celltypes"],
            "regions": rel["regions"],
            "relevant_traits": rel_traits,
        })
    return records


def aggregate_gene_set_kg(records: list[dict[str, Any]], disease: str) -> dict[str, Any]:
    """Aggregate per-gene KG hits into gene-set convergence tables."""
    go_genes: dict[str, set[str]] = {}
    trait_genes: dict[str, set[str]] = {}
    ct_genes: dict[str, set[str]] = {}
    region_genes: dict[str, set[str]] = {}
    genes_with_kg: set[str] = set()

    for rec in records:
        gene = rec["gene"]
        has = bool(rec["gos"] or rec["celltypes"] or rec["regions"] or rec["relevant_traits"])
        if has:
            genes_with_kg.add(gene)
        for go in rec["gos"]:
            go_genes.setdefault(go, set()).add(gene)
        for trait in rec["relevant_traits"]:
            trait_genes.setdefault(trait, set()).add(gene)
        for ct in rec["celltypes"]:
            ct_genes.setdefault(ct, set()).add(gene)
        for reg in rec["regions"]:
            region_genes.setdefault(reg, set()).add(gene)

    def _trait_score(item: tuple[str, set[str]]) -> tuple:
        trait, genes = item
        pathol = trait_pathology(trait, disease)
        return (len(genes), int(pathol), trait.lower())

    def _rows(mapping: dict[str, set[str]], min_genes: int = 1, limit: int = 12) -> list[dict]:
        ranked = sorted(mapping.items(), key=lambda x: (-len(x[1]), x[0]))
        out = []
        for term, genes in ranked:
            if len(genes) < min_genes:
                continue
            out.append({"term": term, "genes": sorted(genes), "count": len(genes)})
            if len(out) >= limit:
                break
        return out

    convergent_traits = _rows(trait_genes, min_genes=2, limit=10)
    pathology_traits = [
        {"term": t, "genes": sorted(gs), "count": len(gs)}
        for t, gs in sorted(trait_genes.items(), key=_trait_score, reverse=True)
        if len(gs) >= MIN_PATHOLOGY_GENES and trait_pathology(t, disease)
    ][:8]
    if len(convergent_traits) < 6:
        seen = {r["term"] for r in convergent_traits}
        for term, genes in sorted(trait_genes.items(), key=_trait_score, reverse=True):
            if term in seen:
                continue
            convergent_traits.append({"term": term, "genes": sorted(genes), "count": len(genes)})
            seen.add(term)
            if len(convergent_traits) >= 10:
                break

    n_sample = len(records)
    if ACTIVE_PROMPT_VERSION == "v6":
        raw_gos = _rows(go_genes, min_genes=2, limit=8)
        gos_out = _enrich_go_rows(raw_gos, n_sample)
    else:
        raw_gos = _rows(go_genes, min_genes=2, limit=8) or _rows(go_genes, min_genes=1, limit=6)
        gos_out = [{"term": r["term"], "genes": r["genes"], "count": r["count"]} for r in raw_gos]
    return {
        "n_hit_genes": n_sample,
        "n_with_kg": len(genes_with_kg),
        "convergent_traits": convergent_traits,
        "pathology_traits": pathology_traits,
        "convergent_gos": gos_out,
        "convergent_celltypes": _rows(ct_genes, min_genes=1, limit=5),
        "convergent_regions": _rows(region_genes, min_genes=1, limit=5),
    }


def format_kg_evidence_aggregate(agg: dict[str, Any]) -> str:
    """Format gene-set aggregate for the LLM prompt."""
    lines = [
        f"GENE SET: {agg['n_hit_genes']} HIGH-impact genes; {agg['n_with_kg']} with KG links.",
        "Interpret at gene-set level. Multi-gene pathology convergence is the key KG signal.",
        "",
        f"PATHOLOGY CONVERGENCE (disease-specific, >={MIN_PATHOLOGY_GENES} genes — USE THESE for kg_used):",
    ]
    pathology = agg.get("pathology_traits") or []
    if pathology:
        for row in pathology:
            genes = ", ".join(row["genes"][:8])
            lines.append(f"  [{row['count']} genes] {row['term']}  <- {genes}")
    else:
        lines.append(f"  (none with >={MIN_PATHOLOGY_GENES} genes)")

    lines.append("")
    lines.append("OTHER CONVERGENT TRAITS (generic/pleiotropic — IGNORE for kg_used):")
    ignored = [r for r in agg.get("convergent_traits", []) if r["term"] not in {p["term"] for p in pathology}]
    if ignored:
        for row in ignored[:6]:
            lines.append(f"  [{row['count']} genes] {row['term']}")
    else:
        lines.append("  (none)")

    lines.append("")
    if ACTIVE_PROMPT_VERSION == "v6":
        lines.append("SHARED GO FUNCTIONS (>=2 sample genes; sample_hits/sample_total | KG_total_genes):")
        if agg.get("convergent_gos"):
            for row in agg["convergent_gos"][:5]:
                genes = ", ".join(row["genes"][:6])
                sample_hits = row.get("sample_hits", row["count"])
                sample_total = row.get("sample_total", agg.get("n_hit_genes", 0))
                kg_total = row.get("kg_total_genes", "?")
                lines.append(
                    f"  [{sample_hits}/{sample_total} sample hits | {kg_total} genes in KG] "
                    f"{row['term']}  <- {genes}"
                )
        else:
            lines.append("  (none with >=2 genes)")
    else:
        lines.append("SHARED GO FUNCTIONS (supporting context only):")
        if agg.get("convergent_gos"):
            for row in agg["convergent_gos"][:5]:
                genes = ", ".join(row["genes"][:6])
                lines.append(f"  [{row['count']} genes] {row['term']}  <- {genes}")
        else:
            lines.append("  (none)")

    if agg.get("convergent_celltypes"):
        lines.append("")
        lines.append("CELL TYPES (from hit gene set):")
        for row in agg["convergent_celltypes"][:4]:
            lines.append(f"  [{row['count']} genes] {row['term']}  <- {', '.join(row['genes'][:5])}")

    if agg.get("convergent_regions"):
        lines.append("")
        lines.append("BRAIN REGIONS (from hit gene set):")
        for row in agg["convergent_regions"][:4]:
            lines.append(f"  [{row['count']} genes] {row['term']}  <- {', '.join(row['genes'][:5])}")

    return "\n".join(lines)


def build_kg_evidence(sample_id: str, disease: str) -> tuple[str, list[str], dict[str, Any]]:
    all_high = get_all_high_genes(sample_id)
    if not all_high:
        return "", [], {}
    records = _collect_gene_kg_records(all_high, disease)
    agg = aggregate_gene_set_kg(records, disease)
    return format_kg_evidence_aggregate(agg), all_high, agg


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024, timeout: int = 60) -> str:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "max_tokens": max_tokens,
        "temperature": LLM_TEMPERATURE,
        "stream": False,
    }
    r = requests.post(f"{EXPERT_URL}/v1/chat/completions", json=payload, timeout=(10, timeout))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _try_loads(candidate: str) -> Any | None:
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    parsed = _try_loads(text)
    if isinstance(parsed, dict):
        return parsed
    fenced = re.search(r"```[a-zA-Z]*\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        parsed = _try_loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    parsed = _try_loads(text[start : i + 1])
                    if isinstance(parsed, dict):
                        return parsed
                    break
    raise ValueError("No valid JSON object found")


def _z_fallback(z: float | None) -> dict[str, Any]:
    if z is None:
        return {"assessment": "uncertain", "confidence": 0.5, "rationale": "no PRS data, fallback"}
    if z >= 1.0:
        a, c = "supported", 0.7
    elif z <= -1.0:
        a, c = "not_supported", 0.3
    else:
        a, c = "uncertain", 0.5
    return {"assessment": a, "confidence": c, "rationale": "LLM parse failed, z-score heuristic fallback"}


def _pathology_gene_phrase() -> str:
    if MIN_PATHOLOGY_GENES <= 1:
        return "disease-pathology trait linked to >=1 HIGH-impact gene"
    return f"the same disease-pathology trait supported by >={MIN_PATHOLOGY_GENES} genes"


SYSTEM_KG_V3 = """You are a research-grade genetic risk synthesis assistant.
Never say the person will develop a disease.
Assess ONE target disease only. Do not mention or judge any other disease.
Return valid JSON with keys: disease, assessment, confidence, risk_assessment, rationale,
kg_used (boolean), prs_only_confidence (number).

STEP A - PRS-only baseline (do this first):
  From the PRS z-score alone, set prs_only_confidence in [0,1] using your own judgement.
  Do not use any fixed formula. Record this before looking at KG.

STEP B - Gene-set KG (secondary evidence):
  KG is aggregated across ALL HIGH-impact genes. Focus on PATHOLOGY CONVERGENCE:
  {path_phrase}.
  IGNORE "OTHER CONVERGENT TRAITS" (generic cognitive, educational attainment, pleiotropy).
  Shared GO terms are context only.

STEP C - Synthesize final confidence:
  Compare PRS direction/strength with pathology convergence.
  - Consistent (both point toward higher risk, or both toward lower): you may move confidence
    further in that direction, proportional to how strong the convergence is.
  - Inconsistent (PRS high but KG weak/generic, or PRS low but KG suggests pathology without
    clear disease specificity): temper confidence toward uncertain or stay near PRS.
  - No useful pathology convergence: final confidence should stay close to prs_only_confidence.
  Set kg_used=true only if pathology convergence materially informed your final answer.
  Set confidence in [0,1]; assessment must match confidence.

Rules:
- assessment: "supported" | "uncertain" | "not_supported"
- rationale: (1) PRS-only view (2) key pathology convergence or why ignored (3) how consistent/
  inconsistent evidence led to final confidence and kg_used.
FORMATTING: output ONLY the JSON object; keep rationale under 220 words."""


SYSTEM_KG = """You are a research-grade genetic risk synthesis assistant.
Never say the person will develop a disease.
Assess ONE target disease only. Do not mention or judge any other disease.
Return valid JSON with keys: disease, assessment, confidence, risk_assessment, rationale,
kg_used (boolean), prs_only_confidence (number).

STEP A - PRS-only baseline (do this first):
  From the PRS z-score alone, set prs_only_confidence in [0,1] using your own judgement.
  Do not use any fixed formula. Record this before looking at KG.

STEP B - Gene-set KG (secondary evidence):
  KG is aggregated across ALL HIGH-impact genes. Focus on PATHOLOGY CONVERGENCE:
  {path_phrase}.
  IGNORE "OTHER CONVERGENT TRAITS" (generic cognitive, educational attainment, pleiotropy).
  For SHARED GO FUNCTIONS (>=2 sample genes), each line shows sample_hits/sample_total and
  KG_total_genes. Use as supporting context: higher sample_hits/sample_total suggests stronger
  within-sample convergence; lower KG_total_genes suggests a more specific function; very high
  KG_total_genes with few sample hits suggests a generic background term.

STEP C - Synthesize final confidence:
  Compare PRS direction/strength with pathology convergence.
  - Consistent (both point toward higher risk, or both toward lower): you may move confidence
    further in that direction, proportional to how strong the convergence is.
  - Inconsistent (PRS high but KG weak/generic, or PRS low but KG suggests pathology without
    clear disease specificity): temper confidence toward uncertain or stay near PRS.
  - No useful pathology convergence: final confidence should stay close to prs_only_confidence.
  Set kg_used=true only if pathology convergence materially informed your final answer.
  Set confidence in [0,1]; assessment must match confidence.

Rules:
- assessment: "supported" | "uncertain" | "not_supported"
- rationale: (1) PRS-only view (2) key pathology convergence or why ignored (3) how consistent/
  inconsistent evidence led to final confidence and kg_used.
FORMATTING: output ONLY the JSON object; keep rationale under 220 words."""


def _system_prompt() -> str:
    tpl = SYSTEM_KG if ACTIVE_PROMPT_VERSION == "v6" else SYSTEM_KG_V3
    return tpl.format(path_phrase=_pathology_gene_phrase())


def build_kg_prompt(
    sample_id: str,
    disease: str,
    prs_txt: str,
    all_high: list[str],
    kg_ev: str,
    agg: dict[str, Any] | None = None,
) -> tuple[str, str]:
    label = DISORDER_LABELS.get(disease, disease)
    genes_str = ", ".join(all_high) if all_high else "(none)"
    n = len(all_high)
    path_kws = ", ".join(DISEASE_PATHOLOGY_KEYWORDS.get(disease, [])[:8])
    n_path = len((agg or {}).get("pathology_traits") or [])
    kg_block = (f"\nKnowledge-graph gene-set synthesis:\n{kg_ev}\n"
                if kg_ev.strip() else "\n(No KG evidence retrieved.)\n")
    go_step = ""
    if ACTIVE_PROMPT_VERSION == "v6":
        go_step = "2b) For GO lines, weigh sample_hits/sample_total vs KG_total_genes.\n"
    user = (
        f"Sample: {sample_id}\nTarget disease (ONLY this disease): {label}\n\n"
        f"PRS analysis (primary signal):\n{prs_txt}\n\n"
        f"Hit genes (all HIGH-impact, {n} genes): {genes_str}\n"
        f"Pathology keywords for {label}: {path_kws}\n"
        f"Pathology convergence items (>={MIN_PATHOLOGY_GENES} genes): {n_path}\n"
        f"{kg_block}\n"
        f"REQUIRED procedure:\n"
        f"1) Set prs_only_confidence from PRS z-score alone (your judgement, no fixed formula).\n"
        f"2) Read PATHOLOGY CONVERGENCE; ignore OTHER CONVERGENT TRAITS.\n"
        f"{go_step}"
        f"3) Synthesize final confidence: strengthen if PRS and pathology convergence agree; "
        f"temper or stay near PRS if they conflict or KG is non-informative.\n"
        f"4) Set kg_used=true only if pathology convergence materially changed your answer.\n"
        f"5) Do not discuss any disease other than {label}.\n\n"
        f"Produce final JSON assessment."
    )
    system = _system_prompt()
    return system, user


def build_prs_text(sample_id: str, disease: str) -> str:
    z, p = get_prs_z(sample_id, disease)
    label = DISORDER_LABELS.get(disease, disease)
    if z is None:
        return f"Disease: {label}. No PRS data available."
    pct = f"{p:.1f}" if p is not None else "N/A"
    return (f"Disease: {label}. PRS z-score = {z:.3f} "
            f"(percentile {pct}%). Interpret this z-score for disease risk.")


def run_one(sample_id: str, disease: str, truth: int, force: bool = False) -> dict[str, Any]:
    out_dir = OUTPUT_DIR / disease
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"sample_{sample_id}.json"
    if out_file.exists() and not force:
        return json.load(out_file.open())

    z, p = get_prs_z(sample_id, disease)
    kg_ev, all_high, kg_agg = "", [], {}
    kg_res: dict[str, Any] = _z_fallback(z)
    try:
        prs_txt = build_prs_text(sample_id, disease)
        kg_ev, all_high, kg_agg = build_kg_evidence(sample_id, disease)
        sys_p, usr_p = build_kg_prompt(sample_id, disease, prs_txt, all_high, kg_ev, kg_agg)
        raw = call_llm(sys_p, usr_p)
        kg_res = extract_json_block(raw)
        if not isinstance(kg_res, dict):
            kg_res = _z_fallback(z)
            kg_res["_fallback"] = f"non-dict JSON: {type(kg_res)}"
    except Exception as e:
        kg_res = _z_fallback(z)
        kg_res["_fallback"] = str(e)

    rec = {
        "sample_id": sample_id,
        "disease": disease,
        "truth": truth,
        "prs_z": z,
        "prs_percentile": p,
        "n_high_genes": len(all_high),
        "hit_genes": all_high,
        "kg_method": KG_METHOD,
        "prompt_version": ACTIVE_PROMPT_VERSION,
        "min_pathology_genes": MIN_PATHOLOGY_GENES,
        "llm_calls": 1,
        "kg_evidence": kg_ev,
        "kg_aggregate": kg_agg,
        "braintwin_kg": {
            "assessment": kg_res.get("assessment"),
            "confidence": kg_res.get("confidence"),
            "rationale": kg_res.get("rationale", ""),
            "kg_used": kg_res.get("kg_used"),
            "prs_only_confidence": kg_res.get("prs_only_confidence"),
        },
    }
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    return rec


def _vp_available(sample_id: str) -> bool:
    return (UKB_DIR / "variant_prioritization_results" / sample_id).is_dir()


def run_disease(disease: str, md: pd.DataFrame, concurrency: int = CONCURRENCY,
                force: bool = False, sample_ids: list[str] | None = None) -> list[dict]:
    col = DISEASE_TO_META_COL.get(disease, disease)
    if sample_ids:
        truth_map = {}
        cases = set(str(s) for s in md[(md[col] == 1) & (md["Disease_Count"] > 0)]["eid"].tolist())
        for sid in sample_ids:
            truth_map[str(sid)] = 1 if str(sid) in cases else 0
        samples = [(str(sid), truth_map[str(sid)]) for sid in sample_ids]
    else:
        cases = [s for s in md[(md[col] == 1) & (md["Disease_Count"] > 0)]["eid"].tolist() if _vp_available(s)]
        ctrls = [s for s in md[md["Disease_Count"] == 0]["eid"].tolist() if _vp_available(s)]
        samples = [(s, 1) for s in cases] + [(s, 0) for s in ctrls]
    print(f"\n[{disease}] running {len(samples)} samples"
          f"{' (forced re-run)' if force else ''}", flush=True)

    results = []
    done = 0

    def _process(args):
        sid, truth = args
        return run_one(sid, disease, truth, force=force)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process, s): s for s in samples}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  FAILED {futures[fut]}: {e}", flush=True)
            done += 1
            if done % 50 == 0:
                print(f"  [{disease}] {done}/{len(samples)} done", flush=True)
    print(f"  [{disease}] complete: {len(results)}/{len(samples)}", flush=True)
    return results


def rebuild_summary_and_score() -> None:
    rows = []
    for disease in DISEASES:
        ddir = OUTPUT_DIR / disease
        if not ddir.is_dir():
            continue
        for f in ddir.glob("sample_*.json"):
            rec = json.load(f.open())
            rows.append({
                "sample_id": rec["sample_id"],
                "disease": rec["disease"],
                "truth": rec["truth"],
                "prs_z": rec.get("prs_z"),
                "n_high_genes": rec.get("n_high_genes"),
                "braintwin_kg_conf": rec.get("braintwin_kg", {}).get("confidence"),
                "braintwin_kg_assess": rec.get("braintwin_kg", {}).get("assessment"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"\nSummary saved: {len(df)} rows -> {OUTPUT_DIR / 'summary.csv'}")

    # merge with LLM results if available
    llm_path = ROOT / "cases" / "genomics" / "output" / "structured_risk" / "summary.csv"
    print(f"\n{'Disease':<22}{'N':<8}{'PRS':<10}{'General':<10}{'Expert':<10}{'KG':<10}")
    print("-" * 70)
    llm = pd.read_csv(llm_path) if llm_path.exists() else None
    for d in DISEASES:
        sub = df[df["disease"] == d]
        if len(sub) < 2 or sub["truth"].nunique() < 2:
            continue
        y = sub["truth"].values
        prs = sub["prs_z"].fillna(0).values
        kg = sub["braintwin_kg_conf"].fillna(0.5).values
        prs_auc = roc_auc_score(y, prs)
        kg_auc = roc_auc_score(y, kg)
        gen_auc = exp_auc = float("nan")
        if llm is not None and {"general_llm_conf", "expert_llm_conf"}.issubset(llm.columns):
            ls = llm[llm["disease"] == d]
            if len(ls) > 1 and ls["truth"].nunique() > 1:
                gen_auc = roc_auc_score(ls["truth"], ls["general_llm_conf"].fillna(0.5))
                exp_auc = roc_auc_score(ls["truth"], ls["expert_llm_conf"].fillna(0.5))
        print(f"{d:<22}{len(sub):<8}{prs_auc:<10.3f}{gen_auc:<10.3f}{exp_auc:<10.3f}{kg_auc:<10.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diseases", nargs="*", default=None)
    parser.add_argument("--samples", nargs="*", default=None,
                        help="Optional sample IDs to run (with truth inferred from metadata)")
    parser.add_argument("--llm-url", type=str, default=None,
                        help="Override LLM base URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (for small tests)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if sample JSON already exists")
    parser.add_argument("--prompt-version", type=str, default="v6",
                        choices=["v3", "v6"],
                        help="Prompt/KG format version (default: v6)")
    parser.add_argument("--min-pathology-genes", type=int, default=2,
                        help="Min genes per trait for pathology_traits / kg_used signal (default: 2)")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    global OUTPUT_DIR, EXPERT_URL, ACTIVE_PROMPT_VERSION, PROMPT_VERSION, MIN_PATHOLOGY_GENES
    ACTIVE_PROMPT_VERSION = args.prompt_version
    PROMPT_VERSION = ACTIVE_PROMPT_VERSION
    MIN_PATHOLOGY_GENES = max(1, int(args.min_pathology_genes))
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.llm_url:
        EXPERT_URL = args.llm_url.rstrip("/")
        print(f"Using LLM: {EXPERT_URL}", flush=True)
    print(f"Prompt: {ACTIVE_PROMPT_VERSION}, temperature: {LLM_TEMPERATURE}, "
          f"min_pathology_genes: {MIN_PATHOLOGY_GENES}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)

    if args.score_only:
        rebuild_summary_and_score()
        return

    diseases = args.diseases or DISEASES
    md = pd.read_csv(UKB_DIR / "Metadata.csv")
    md["eid"] = md["eid"].astype(str)

    t0 = time.time()
    for disease in diseases:
        run_disease(disease, md, args.concurrency, force=args.force, sample_ids=args.samples)
    print(f"\nElapsed: {time.time()-t0:.0f}s")
    rebuild_summary_and_score()


if __name__ == "__main__":
    main()
