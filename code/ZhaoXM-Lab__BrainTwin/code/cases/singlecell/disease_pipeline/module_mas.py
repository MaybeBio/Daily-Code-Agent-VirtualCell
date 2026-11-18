#!/usr/bin/env python3
"""Prepare functional modules, query the graph with fixed Cypher, then ask for an inference.

The model does not write the Cypher. Queries are gene-to-GO and gene-to-trait
for every gene in the batch. A second prompt sends the saved expression,
enrichment, and graph rows back to the model for a written inference.
Model text is stored unchanged.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent
V2_RAG = PIPELINE_DIR.parents[3] / "braintwin"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from chain_v2 import _cell_ids, _read_gene_list, _term_record  # noqa: E402

PROMPT_VERSION = "module_mas_v3"
INFERENCE_PROMPT_VERSION = "module_inference_v2"
MAX_PROMPT_CHARS = 2800
INFERENCE_MARK = "Inference pass over saved expression, enrichment, and graph rows"
DISEASE_TRAIT_WORDS = ("alzheimer", "dementia", "amyloid", "tau", "neurofibrillary")
_LLM_ORIGIN = os.environ.get("BRAINTWIN_LLM_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_URL = _LLM_ORIGIN + "/v1/models"
MODEL_ID = os.environ.get("BRAINTWIN_MODEL_ID", "Qwen3_8B_Instruct_Finetuned")
EMPTY_GRAPH_ANSWER = "The knowledge graph returned no records for this query."
HYPOTHESIS_MARK = "Model hypothesis (not graph support):"
QUERY_MARK = "Graph query: fixed GO and trait Cypher"
GO_CYPHER = """MATCH (g:Gene)-[:ANNOTATAED_AS]->(t:GOTerm)
WHERE g.Gene_Name IN $genes
RETURN g.Gene_Name AS gene, t.Go_Term_Name AS term
ORDER BY gene, term"""
TRAIT_CYPHER = """MATCH (g:Gene)-[:ASSOCIATE_WITH_TRAIT]->(t:Trait)
WHERE g.Gene_Name IN $genes
RETURN g.Gene_Name AS gene, t.Trait_Name AS trait
ORDER BY gene, trait"""

# More specific themes are listed first. Adhesion, transmission, and
# development stay apart so a shared gene cannot pull them into one module.
PRIMARY_THEMES: list[tuple[str, tuple[str, ...]]] = [
    ("synaptic_adhesion", ("cell-cell adhesion", "cell adhesion", "adhesion")),
    (
        "synaptic_transmission",
        (
            "synaptic transmission",
            "trans-synaptic",
            "neurotransmitter",
            "postsynaptic potential",
            "synaptic vesicle",
            "signal release from synapse",
            "vesicle-mediated transport in synapse",
        ),
    ),
    ("synapse_organization", ("synap",)),
    ("axon_guidance", ("axon guidance", "axonogenesis", "axon development", "neuron projection guidance", "axon")),
    (
        "neural_development",
        (
            "nervous system",
            "neuron development",
            "neuron projection",
            "brain development",
            "central nervous system",
            "morphogenesis",
            "differentiation",
            "dendrite",
        ),
    ),
    ("learning", ("learning", "memory")),
    ("protein_quality", ("unfolded", "chaperone", "heat shock", "inclusion body", "protein folding", "protein stabilization")),
    ("cell_death", ("apopt", "cell death", "programmed cell death")),
    ("viral", ("viral", "virus", "interferon", "symbiont")),
    ("immune", ("immune", "inflamm", "cytokine", "macrophage", "microglia", "leukocyte")),
    ("mitochondrion", ("mitochond",)),
    ("microtubule", ("microtubule",)),
]


def generate_with_mas(output_dir: Path, cfg: dict[str, Any], reuse: bool = True) -> Path:
    output_dir = Path(output_dir)
    mas_dir = output_dir / "mas"
    prompt_dir = mas_dir / "prompts"
    raw_dir = mas_dir / "raw_answers"
    kg_dir = mas_dir / "kg"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    modules = build_functional_modules(output_dir, cfg)
    questions = _questions(modules, cfg)
    for item in questions:
        (prompt_dir / item["prompt_file"]).write_text(item["prompt"])
    payload = {
        "dataset": cfg.get("dataset"),
        "disease": cfg.get("disease"),
        "tissue": cfg.get("tissue"),
        "species": cfg.get("species"),
        "prompt_version": PROMPT_VERSION,
        "inference_prompt_version": INFERENCE_PROMPT_VERSION,
        "what_this_file_is": "Functional modules prepared as MAS input. They are not explanation chains.",
        "limitations": [
            "No donor or sample identifier is present. Cell-level Wilcoxon tests do not control donor differences.",
            "Positive log2FC means higher in the disease group.",
            "A module groups GO annotations. It does not state a disease mechanism.",
            "A single-gene module has one expression change. Its GO terms are alternative annotations, not independent supports.",
            "Odds ratios are stored from the enrichment table and are not treated as extra support.",
            "Cypher is written by this pipeline. The model does not generate the graph query.",
            "The two queries are gene-to-GO and gene-to-trait. Marker edges are not queried.",
            "A model statement that does not correspond to a returned record is a hypothesis, not graph support.",
        ],
        "modules": modules,
        "questions": [
            {
                "cell_type_id": item["cell_type_id"],
                "batch": item["batch"],
                "module_ids": item["module_ids"],
                "prompt_file": item["prompt_file"],
            }
            for item in questions
        ],
    }
    (mas_dir / "modules.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    reachable = _model_reachable()
    answers: list[dict[str, Any]] = []
    if not reachable:
        status = _status_text(modules, questions, answers, [], reached=False)
        (mas_dir / "generation_status.md").write_text(status)
        print(f"[mas] model endpoint {MODEL_URL} is not reachable. Modules and prompts were written. No chain text was generated.")
        return mas_dir
    pending = []
    for item in questions:
        raw_path = raw_dir / item["raw_file"]
        existing = raw_path.read_text() if raw_path.exists() else ""
        if reuse and QUERY_MARK in existing:
            answers.append({"prompt_file": item["prompt_file"], "raw_file": item["raw_file"], "status": "reused"})
            print(f"[mas] reused {item['raw_file']}")
            continue
        pending.append(item)
    if pending:
        driver = _graph_driver()
        try:
            for item in pending:
                raw_path = raw_dir / item["raw_file"]
                try:
                    genes = _module_genes(item["modules"])
                    details = _query_graph(driver, genes)
                    kg_dir.mkdir(parents=True, exist_ok=True)
                    (kg_dir / item["kg_file"]).write_text(json.dumps(details, ensure_ascii=False, indent=2))
                    interpretation = _interpret_records(item["prompt"], item["modules"], details, prompt_dir, item["prompt_file"])
                    raw_path.write_text(f"{QUERY_MARK}\n\n{interpretation.strip()}\n")
                    status = "records" if details["n_records"] else "no_records"
                    answers.append({"prompt_file": item["prompt_file"], "raw_file": item["raw_file"], "status": status})
                    print(f"[mas] wrote {item['raw_file']} ({status}, {details['n_records']} records)")
                except Exception as exc:
                    failure = traceback.format_exc()
                    raw_path.write_text(f"[Graph query failed. No explanation chain was written.]\n\n{failure}")
                    answers.append({"prompt_file": item["prompt_file"], "raw_file": item["raw_file"], "status": f"{type(exc).__name__}: {exc}"})
                    print(f"[mas] failed {item['prompt_file']}: {exc}")
        finally:
            driver.close()
    inferences = write_inferences(output_dir, cfg, questions, reuse=reuse)
    (mas_dir / "generation_status.md").write_text(_status_text(modules, questions, answers, inferences, reached=True))
    print(f"[ok] MAS module stage wrote {mas_dir} ({len(modules)} modules, {len(questions)} questions, {len(inferences)} inferences)")
    return mas_dir


def _graph_driver():
    if str(V2_RAG) not in sys.path:
        sys.path.insert(0, str(V2_RAG))
    from neo4j import GraphDatabase
    from multiagent_system.config import Config

    config = Config()
    return GraphDatabase.driver(config.neo4j_url, auth=(config.neo4j_username, config.neo4j_password))


def _module_genes(modules: list[dict[str, Any]]) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    for module in modules:
        for gene in module["gene_changes"]:
            name = str(gene["gene"])
            if name not in seen:
                seen.add(name)
                genes.append(name)
    return genes


def _query_graph(driver, genes: list[str]) -> dict[str, Any]:
    queries = [
        {"name": "gene_go", "relation": "ANNOTATAED_AS", "cypher": GO_CYPHER},
        {"name": "gene_trait", "relation": "ASSOCIATE_WITH_TRAIT", "cypher": TRAIT_CYPHER},
    ]
    with driver.session() as session:
        for query in queries:
            rows = [dict(record) for record in session.run(query["cypher"], genes=genes)]
            query["context"] = rows
            query["n_records"] = len(rows)
    return {
        "cypher_source": "disease_pipeline",
        "parameters": {"genes": genes},
        "queries": queries,
        "n_records": sum(query["n_records"] for query in queries),
        "note": "Cypher is fixed by the disease pipeline. The model does not write the query. Marker edges are not queried.",
    }


def _interpret_records(
    question: str,
    modules: list[dict[str, Any]],
    details: dict[str, Any],
    prompt_dir: Path,
    prompt_file: str,
) -> str:
    shown, hidden = _select_records(modules, details)
    if details["n_records"] == 0:
        record_text = "No rows were returned."
    else:
        lines = [f"{row['gene']} | {row['relation']} | {row['target']}" for row in shown]
        record_text = "\n".join(lines)
        if hidden:
            record_text += f"\n{hidden} additional rows are stored in the kg file and are not repeated here."
    prompt = (
        "The graph was queried with fixed Cypher for two relations: "
        "gene to GO term through ANNOTATAED_AS, and gene to trait through ASSOCIATE_WITH_TRAIT. "
        "Marker relations were not queried.\n"
        f"Returned rows, written as gene | relation | target:\n{record_text}\n\n"
        "Write two parts.\n"
        "Knowledge graph: state only relations present in the returned rows, one relation at a time. "
        f"If no rows were returned, write exactly: {EMPTY_GRAPH_ANSWER}\n"
        f"{HYPOTHESIS_MARK} relations suggested by the differential modules that are absent from the returned rows. "
        "If you have none, write None.\n"
        "Do not invent citations, graph records, or enrichment statistics. "
        "A GO annotation in the graph is not evidence that the process increased or decreased. "
        "An association is not a proven causal path.\n\n"
        f"{question}"
    )
    stem = prompt_file.removesuffix(".txt")
    (prompt_dir / f"{stem}_interpretation.txt").write_text(prompt)
    return _ask_ft(prompt)


def _select_records(modules: list[dict[str, Any]], details: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    names_by_gene: dict[str, list[str]] = defaultdict(list)
    for module in modules:
        for annotation in module["annotations"]:
            for gene in annotation["overlap_genes"]:
                names_by_gene[str(gene)].append(str(annotation["name"]))
    matched: list[dict[str, str]] = []
    other: list[dict[str, str]] = []
    for query in details["queries"]:
        relation = "GO" if query["name"] == "gene_go" else "trait"
        target_key = "term" if query["name"] == "gene_go" else "trait"
        for row in query["context"]:
            item = {"gene": str(row["gene"]), "relation": relation, "target": str(row[target_key])}
            names = names_by_gene.get(item["gene"], [])
            if relation == "trait" or any(_term_match(name, item["target"]) for name in names):
                matched.append(item)
            else:
                other.append(item)
    shown = matched[:40]
    if len(shown) < 12:
        shown.extend(other[: 12 - len(shown)])
    hidden = details["n_records"] - len(shown)
    return shown, max(hidden, 0)


def _term_match(annotation: str, graph_term: str, min_jaccard: float = 0.6) -> bool:
    left = _term_tokens(_short_term(annotation))
    right = _term_tokens(graph_term)
    if not left or not right:
        return False
    if left <= right or right <= left:
        return True
    return len(left & right) / len(left | right) >= min_jaccard


def _term_tokens(text: str) -> set[str]:
    stop = {"of", "the", "a", "and", "in", "to", "via", "for", "go"}
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stop and not token.isdigit()}


def _ask_ft(prompt: str, max_tokens: int = 700) -> str:
    import requests

    response = requests.post(
        _LLM_ORIGIN + "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "stream": False,
            "max_tokens": max_tokens,
        },
        timeout=180,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"]).strip()


def write_inferences(
    output_dir: Path,
    cfg: dict[str, Any],
    questions: list[dict[str, Any]],
    reuse: bool = True,
    only_stems: set[str] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Send the saved stage-one evidence back to the model for a written inference."""
    mas_dir = Path(output_dir) / "mas"
    prompt_dir = mas_dir / "prompts"
    raw_dir = mas_dir / "raw_answers"
    kg_dir = mas_dir / "kg"
    results: list[dict[str, Any]] = []
    for item in questions:
        stem = item["raw_file"].removesuffix(".txt")
        if only_stems is not None and stem not in only_stems:
            continue
        raw_name = f"{stem}_inference.txt"
        raw_path = raw_dir / raw_name
        existing = raw_path.read_text() if raw_path.exists() else ""
        if (
            reuse
            and not force
            and existing.startswith(INFERENCE_MARK)
            and "[Inference failed" not in existing
        ):
            results.append({"raw_file": raw_name, "status": "reused"})
            print(f"[infer] reused {raw_name}")
            continue
        stage1_path = raw_dir / item["raw_file"]
        stage1 = stage1_path.read_text() if stage1_path.exists() else ""
        if QUERY_MARK not in stage1:
            results.append({"raw_file": raw_name, "status": "skipped_missing_stage1"})
            print(f"[infer] skipped {raw_name}: stage-one reading is missing")
            continue
        kg_path = kg_dir / item["kg_file"]
        details = json.loads(kg_path.read_text()) if kg_path.exists() else {}
        prompt = _inference_prompt(item, stage1, details, cfg)
        (prompt_dir / raw_name).write_text(prompt)
        try:
            text = _ask_ft(prompt, max_tokens=900)
            text, repairs = _repair_direction(item, text)
            raw_path.write_text(f"{INFERENCE_MARK}\n\n{text.strip()}\n")
            left = _evidence_conflicts(item["modules"], text)
            status = "rewritten" if repairs else "written"
            if left:
                status = f"{status}_conflicts:{len(left)}"
            results.append({"raw_file": raw_name, "status": status, "prompt_chars": len(prompt), "conflicts": left})
            print(f"[infer] wrote {raw_name} ({status}, {repairs} repairs, {len(left)} conflicts left)")
        except Exception as exc:
            failure = traceback.format_exc()
            raw_path.write_text(f"[Inference failed. No inference was written.]\n\n{failure}")
            results.append({"raw_file": raw_name, "status": f"{type(exc).__name__}: {exc}"})
            print(f"[infer] failed {raw_name}: {exc}")
    return results


def _inference_prompt(
    item: dict[str, Any],
    stage1: str,
    details: dict[str, Any],
    cfg: dict[str, Any],
) -> str:
    modules = item["modules"]
    disease = cfg.get("disease") or modules[0].get("disease") or "the disease"
    rows, hidden = _inference_rows(modules, details)
    if rows:
        row_text = "\n".join(f"{row['gene']} | {row['relation']} | {row['target']}" for row in rows)
        if hidden:
            row_text += f"\n{hidden} additional matched rows are stored in the kg file and are not repeated here."
    elif details:
        row_text = "No saved graph row matched these module terms or an Alzheimer-related trait."
    else:
        row_text = "Saved graph rows are missing for this batch."
    evidence = "\n".join(_module_line(module) for module in modules)
    reading = _clip_reading(stage1)
    prompt = (
        f"{INFERENCE_MARK}\n\n"
        "The lines below are saved evidence for one cell type. "
        f"Write an inference about how the expression change and the enriched function may relate to {disease}. "
        "Write in English only. Do not use Chinese.\n"
        "Do not run a new graph query.\n\n"
        f"{_header(modules[0], cfg)}\n\n"
        "Differential expression and enrichment:\n"
        f"{evidence}\n\n"
        "Saved graph rows that match these genes and module terms, written as gene | relation | target. "
        "GO means the gene is annotated with that term. trait means the gene is associated with that trait. "
        "Trait rows that do not name Alzheimer disease, dementia, amyloid, or tau stay in the kg file and are omitted here. "
        "These rows do not show whether a process increased or decreased.\n"
        f"{row_text}\n\n"
        "Earlier reading of the same query. It can omit or misstate records. "
        "When it disagrees with the saved graph rows, follow the saved graph rows.\n"
        f"{reading}\n\n"
        "Direction lock. A negative log2FC means lower in the disease group. "
        "A positive log2FC means higher. Copy the sign and the direction word below. "
        "Do not write higher next to a negative number, or lower next to a positive number.\n"
        f"{_direction_lock(modules)}\n\n"
        "For each module, write one inference about one gene and one enriched term:\n"
        "- name the module id, cell type, and gene\n"
        "- use the direction word from the direction lock and copy that gene's log2FC with its sign\n"
        "- name the enriched term\n"
        "- cite a saved graph row only when the row starts with the same gene. "
        "If the row names another gene, say the saved rows do not name this gene and term\n"
        "- state a possible relation to Alzheimer disease\n"
        "Do not add a trait, gene, statistic, or identifier that is not printed above.\n"
        "Keep the expression result, the enrichment overlap, a graph record, and a hypothesis in separate clauses.\n"
        "A GO row does not show that the process increased or decreased.\n"
        "An association is not a proven causal path.\n"
        "A single-gene module is one expression change.\n"
        "Do not invent citations, statistics, graph rows, or a disease action that the evidence does not state.\n"
        "If a module has no matching graph row, still state the expression and enrichment, and say the graph rows do not name that pair."
    )
    if len(prompt) > 9000:
        prompt = prompt.replace(reading, "[Earlier reading omitted to fit the context.]")
    return prompt


def _inference_rows(modules: list[dict[str, Any]], details: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    go_rows: list[dict[str, str]] = []
    trait_rows: list[dict[str, str]] = []
    for query in details.get("queries", []):
        relation = "GO" if query.get("name") == "gene_go" else "trait"
        target_key = "term" if query.get("name") == "gene_go" else "trait"
        bucket = go_rows if relation == "GO" else trait_rows
        for row in query.get("context", []):
            bucket.append({"gene": str(row["gene"]), "relation": relation, "target": str(row[target_key])})
    chosen: list[dict[str, str]] = []
    matched = 0
    for module in modules:
        names = [_short_term(annotation["name"]) for annotation in module["annotations"][:3]]
        genes = [item["gene"] for item in sorted(module["gene_changes"], key=lambda item: item["adjusted_P"])[:6]]
        module_go = [
            row
            for row in go_rows
            if row["gene"] in genes and any(_term_match(name, row["target"], min_jaccard=0.75) for name in names)
        ]
        module_traits = [
            row
            for row in trait_rows
            if row["gene"] in genes and any(word in row["target"].lower() for word in DISEASE_TRAIT_WORDS)
        ]
        matched += len(module_go) + len(module_traits)
        chosen.extend(module_go[:2])
        chosen.extend(module_traits[:2])
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in chosen:
        key = (row["gene"], row["relation"], row["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    shown = unique[:16]
    return shown, max(matched - len(shown), 0)


def _clip_reading(text: str, limit: int = 700) -> str:
    body = text.replace(QUERY_MARK, "").strip()
    if not body:
        return "No first-pass reading was saved."
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + "\n[First-pass reading truncated.]"


def _direction_word(logfc: float) -> str:
    if logfc > 0:
        return "higher"
    if logfc < 0:
        return "lower"
    return "unchanged"


def _direction_lock(modules: list[dict[str, Any]]) -> str:
    lines = []
    for module in modules:
        for gene in module["gene_changes"][:4]:
            word = _direction_word(float(gene["logFC"]))
            lines.append(
                f"{module['module_id']} {gene['gene']}: log2FC {float(gene['logFC']):+.2f} requires the word {word}."
            )
    return "\n".join(lines)


def _repair_direction(item: dict[str, Any], text: str) -> tuple[str, int]:
    repairs = 0
    current = text
    for _ in range(2):
        conflicts = _evidence_conflicts(item["modules"], current)
        if not conflicts:
            break
        repairs += 1
        current = _ask_ft(_repair_prompt(item, current, conflicts), max_tokens=900)
    return current, repairs


def _repair_prompt(item: dict[str, Any], draft: str, conflicts: list[str]) -> str:
    listed = "\n".join(f"- {item}" for item in conflicts)
    return (
        "Rewrite the inference in English only. Do not use Chinese. "
        "The draft contradicts the saved differential evidence.\n"
        "Conflicts:\n"
        f"{listed}\n\n"
        "Direction lock. A negative log2FC means lower in the disease group. "
        "A positive log2FC means higher. Copy the sign and the required word.\n"
        f"{_direction_lock(item['modules'])}\n\n"
        "Use the required direction word. Copy log2FC with its sign.\n"
        "Cite a graph row only when it starts with the same gene as that inference. "
        "Otherwise say the saved rows do not name that gene and term.\n"
        "Do not add a trait, gene, statistic, or identifier that is not already in the direction lock or the draft evidence.\n"
        "Keep the expression result separate from the hypothesis. An association is not a proven causal path.\n\n"
        f"Draft:\n{draft.strip()}\n"
    )


_HIGHER_RE = re.compile(r"\b(higher|increased|up-regulated|upregulated)\b|(?<![A-Za-z])up(?![A-Za-z-])", re.I)
_LOWER_RE = re.compile(r"\b(lower|decreased|down-regulated|downregulated)\b|(?<![A-Za-z])down(?![A-Za-z-])", re.I)
_LOGFC_RE = re.compile(r"log2FC\s*[:=]?\s*([+-]?\d+\.\d+)", re.I)
_ROW_RE = re.compile(r"\b([A-Z][A-Z0-9]+|MT-[A-Z0-9]+)\s*\|\s*(?:GO|trait)\s*\|", re.I)
_GENE_RE = re.compile(r"\b([A-Z][A-Z0-9]+|MT-[A-Z0-9]+)\b")


def _module_markers(module_id: str) -> list[re.Pattern[str]]:
    number = int(module_id.rsplit("M", 1)[-1])
    return [
        re.compile(rf"\b{re.escape(module_id)}\b"),
        re.compile(rf"\bModule\s+0*{number}(?!\d)", re.I),
    ]


def _module_span(module: dict[str, Any], modules: list[dict[str, Any]], text: str) -> tuple[int, int] | None:
    starts = []
    for pattern in _module_markers(module["module_id"]):
        match = pattern.search(text)
        if match:
            starts.append(match.start())
    if not starts:
        return None
    start = min(starts)
    end = len(text)
    for other in modules:
        if other["module_id"] == module["module_id"]:
            continue
        for pattern in _module_markers(other["module_id"]):
            match = pattern.search(text, start + 1)
            if match:
                end = min(end, match.start())
    return start, end


def _evidence_conflicts(modules: list[dict[str, Any]], text: str) -> list[str]:
    conflicts: list[str] = []
    for module in modules:
        span = _module_span(module, modules, text)
        if span is None:
            continue
        module_id = module["module_id"]
        chunk = text[span[0] : span[1]]
        genes = {str(item["gene"]): float(item["logFC"]) for item in module["gene_changes"]}
        sentences = re.split(r"(?<=[.])\s+", chunk)
        named = next((gene for gene in _GENE_RE.findall(chunk) if gene in genes), "")
        for sentence in sentences:
            present = [gene for gene in genes if re.search(rf"\b{re.escape(gene)}\b", sentence)]
            if not present:
                continue
            says_higher = _HIGHER_RE.search(sentence) is not None
            says_lower = _LOWER_RE.search(sentence) is not None
            printed = [float(value) for value in _LOGFC_RE.findall(sentence)]
            for gene in present:
                true = genes[gene]
                word = _direction_word(true)
                for value in printed:
                    if value == 0 or true == 0:
                        continue
                    if (value > 0) != (true > 0) and abs(abs(value) - abs(true)) < 0.051:
                        conflicts.append(
                            f"{module_id} {gene}: printed log2FC {value:+.2f} but the evidence is {true:+.2f}. "
                            f"Copy {true:+.2f} and write {word}."
                        )
                if true < 0 and says_higher and not says_lower:
                    conflicts.append(f"{module_id} {gene}: the text says higher, but log2FC {true:+.2f} requires lower.")
                if true > 0 and says_lower and not says_higher:
                    conflicts.append(f"{module_id} {gene}: the text says lower, but log2FC {true:+.2f} requires higher.")
        if named:
            for cited in _ROW_RE.findall(chunk):
                if cited != named:
                    conflicts.append(
                        f"{module_id}: the inference names {named}, but the graph row cites {cited}. "
                        f"Cite {named} only, or say the saved rows do not name {named}."
                    )
    for token in re.findall(r"\bX-\d+\b", text):
        conflicts.append(f"Remove {token}. It is not in the saved evidence.")
    # One conflict per identical sentence keeps the repair prompt short.
    unique: list[str] = []
    seen: set[str] = set()
    for item in conflicts:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:12]


def build_functional_modules(output_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    labels = cfg.get("cell_type_labels") or {}
    disease = str(cfg.get("disease") or "")
    padj_max = float(cfg.get("padj_max", 0.05))
    modules: list[dict[str, Any]] = []
    for cell_id in _cell_ids(output_dir):
        stats, terms, meta = _load_cell(output_dir, cell_id, padj_max)
        grouped = _group_terms(terms)
        for index, cluster in enumerate(grouped, start=1):
            modules.append(
                _module_record(
                    cell_id,
                    labels.get(cell_id, cell_id),
                    disease,
                    index,
                    cluster,
                    stats,
                    meta,
                )
            )
    return modules


def _load_cell(
    output_dir: Path,
    cell_id: str,
    padj_max: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    full = pd.read_csv(output_dir / "de" / f"{cell_id}_de_full.tsv", sep="\t")
    up = _read_gene_list(output_dir / "de" / f"{cell_id}_up.tsv")
    down = _read_gene_list(output_dir / "de" / f"{cell_id}_down.tsv")
    stats: dict[str, dict[str, Any]] = {}
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
    terms: list[dict[str, Any]] = []
    enrich_path = output_dir / "function" / f"{cell_id}_function.tsv"
    if enrich_path.exists() and enrich_path.stat().st_size > 0:
        frame = pd.read_csv(enrich_path, sep="\t")
        for index, row in enumerate(frame.to_dict(orient="records"), start=1):
            term = _term_record(cell_id, index, row, up, down, stats, padj_max)
            if term is None or not term["overlap_genes"]:
                continue
            term["theme"] = _primary_theme(term["name"])
            terms.append(term)
    meta_path = output_dir / "de" / f"{cell_id}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return stats, terms, meta


def _primary_theme(name: str) -> str:
    low = name.lower()
    for theme, words in PRIMARY_THEMES:
        if any(word in low for word in words):
            return theme
    return "unspecified"


def _group_terms(terms: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    single: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    multi: list[dict[str, Any]] = []
    for term in terms:
        genes = term["overlap_genes"]
        if len(genes) == 1:
            single[(term["input_direction"], genes[0])].append(term)
        else:
            multi.append(term)
    clusters = list(single.values())
    clusters.extend(_cluster_multi(multi))
    clusters.sort(key=lambda cluster: min(term["adjusted_P"] for term in cluster))
    return clusters


def _cluster_multi(terms: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(terms)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(terms):
        for right_index in range(left_index + 1, len(terms)):
            if _same_module(left, terms[right_index]):
                parent[find(left_index)] = find(right_index)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, term in enumerate(terms):
        grouped[find(index)].append(term)
    return list(grouped.values())


def _same_module(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["input_direction"] != right["input_direction"]:
        return False
    if left["theme"] != right["theme"]:
        return False
    left_genes = set(left["overlap_genes"])
    right_genes = set(right["overlap_genes"])
    if not left_genes or not right_genes:
        return False
    if left_genes == right_genes:
        return True
    union = left_genes | right_genes
    if len(left_genes & right_genes) / len(union) >= 0.4:
        return True
    smaller = left_genes if len(left_genes) <= len(right_genes) else right_genes
    subset = left_genes <= right_genes or right_genes <= left_genes
    return subset and len(smaller) >= 2


def _module_record(
    cell_id: str,
    label: str,
    disease: str,
    index: int,
    cluster: list[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    genes = []
    seen: set[str] = set()
    for term in cluster:
        for gene in term["overlap_genes"]:
            if gene not in seen and gene in stats:
                seen.add(gene)
                genes.append(dict(stats[gene]))
    genes.sort(key=lambda item: item["adjusted_P"])
    annotations = sorted(cluster, key=lambda term: term["adjusted_P"])
    single = len(genes) == 1
    theme_counts = Counter(term["theme"] for term in annotations)
    return {
        "module_id": f"{cell_id}-M{index:02d}",
        "kind": "functional_module",
        "cell_type": label,
        "cell_type_id": cell_id,
        "disease": disease,
        "input_direction": annotations[0]["input_direction"],
        "theme": "single_gene" if single else annotations[0]["theme"],
        "evidence_weight": "single_gene" if single else "multi_gene",
        "independent_expression_evidence": 1 if single else len(genes),
        "annotation_count": len(annotations),
        "annotation_themes": dict(theme_counts),
        "n_disease_cells": meta.get("n_disease_cells"),
        "n_control_cells": meta.get("n_control_cells"),
        "statistical_note": meta.get("statistical_note", ""),
        "gene_changes": genes,
        "annotations": [
            {
                "term_id": term["term_id"],
                "name": term["name"],
                "theme": term["theme"],
                "input_direction": term["input_direction"],
                "overlap_genes": term["overlap_genes"],
                "adjusted_P": term["adjusted_P"],
                "odds_ratio": term["odds_ratio"],
                "record_id": term["record_id"],
            }
            for term in annotations
        ],
    }


def _questions(modules: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for module in modules:
        by_cell[module["cell_type_id"]].append(module)
    questions = []
    for cell_id, cell_modules in by_cell.items():
        batches = _batches(cell_modules, cfg)
        for batch_index, batch in enumerate(batches, start=1):
            prompt = _prompt(batch, cfg)
            stem = f"{cell_id}_b{batch_index:02d}"
            questions.append(
                {
                    "cell_type_id": cell_id,
                    "batch": batch_index,
                    "module_ids": [module["module_id"] for module in batch],
                    "modules": batch,
                    "prompt": prompt,
                    "prompt_file": f"{stem}.txt",
                    "raw_file": f"{stem}.txt",
                    "kg_file": f"{stem}.json",
                    "thread_id": f"module-{stem}",
                }
            )
    return questions


def _batches(modules: list[dict[str, Any]], cfg: dict[str, Any]) -> list[list[dict[str, Any]]]:
    header = _header(modules[0], cfg)
    closing = _closing()
    budget = MAX_PROMPT_CHARS - len(header) - len(closing) - 2
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for module in modules:
        line = _module_line(module)
        if current and size + len(line) + 1 > budget:
            batches.append(current)
            current = []
            size = 0
        current.append(module)
        size += len(line) + 1
    if current:
        batches.append(current)
    return batches


def _header(module: dict[str, Any], cfg: dict[str, Any]) -> str:
    tissue = cfg.get("tissue") or ""
    species = cfg.get("species") or ""
    disease = cfg.get("disease") or module.get("disease") or ""
    return (
        f"Cell type {module['cell_type']} in {species} {tissue}, disease {disease}. "
        f"Cells disease/control: {module.get('n_disease_cells')}/{module.get('n_control_cells')}. "
        "Positive log2FC means higher in the disease group. "
        f"{module.get('statistical_note') or 'Cell-level Wilcoxon test. Donor differences are not controlled.'} "
        "The numbered groups below are functional modules, not explanation chains."
    )


def _closing() -> str:
    return (
        "Using the knowledge graph, pick specific gene-term relations from these modules and describe explanation chains. "
        "One chain should be one relation, not a whole module. "
        "A single-gene module is one expression change no matter how many GO terms are listed. Do not count those terms as separate supports. "
        "Separate what the differential result shows, what the graph supports, and what remains a hypothesis. "
        "Do not invent citations. Do not state that a whole pathway is activated or inhibited only because member genes changed. "
        "An association is not a proven causal path."
    )


def _prompt(modules: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    lines = [_header(modules[0], cfg)]
    lines.extend(_module_line(module) for module in modules)
    lines.append(_closing())
    return "\n".join(lines)


def _module_line(module: dict[str, Any]) -> str:
    examples = "; ".join(_short_term(item["name"]) for item in module["annotations"][:3])
    if module["evidence_weight"] == "single_gene":
        gene = module["gene_changes"][0]
        ranked = sorted(module["annotation_themes"].items(), key=lambda item: (-item[1], item[0]))
        themes = ", ".join(f"{name} {count}" for name, count in ranked[:4])
        return (
            f"{module['module_id']}. Single-gene evidence, not {module['annotation_count']} supports. "
            f"{gene['gene']} is {module['input_direction']} (log2FC {gene['logFC']:+.2f}, adjusted P {gene['adjusted_P']:.1e}). "
            f"This one gene is the only overlap of {module['annotation_count']} GO terms, for example {examples}. "
            f"Annotation themes: {themes}. Count this as one expression change."
        )
    shown = module["gene_changes"][:6]
    gene_text = "; ".join(f"{item['gene']} {item['logFC']:+.2f} {item['adjusted_P']:.1e}" for item in shown)
    extra = ""
    hidden = len(module["gene_changes"]) - len(shown)
    if hidden:
        extra = f"; and {hidden} more genes"
    return (
        f"{module['module_id']}. Functional module, theme {module['theme']}, direction {module['input_direction']}, "
        f"{module['annotation_count']} GO terms, {len(module['gene_changes'])} genes. "
        f"Genes: {gene_text}{extra}. Example terms: {examples}. "
        "This module is not itself an explanation chain."
    )


def _short_term(name: str) -> str:
    return name.split(" (GO")[0]


def _model_reachable() -> bool:
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _status_text(
    modules: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    inferences: list[dict[str, Any]],
    reached: bool,
) -> str:
    single = [module for module in modules if module["evidence_weight"] == "single_gene"]
    lines = [
        "# MAS 功能模块提问",
        "",
        "这里的模块来自已保存的差异表和富集表，用来组织提问。它们不是解释链。",
        "图谱查询由 pipeline 固定写成两条 Cypher：基因到 GO，以及基因到性状。模型不生成查询，也不查 marker。",
        "微调模型先根据返回的记录作答。记录里没有的关系标成模型假设。",
        "这一步只转述证据。转述保存后，同一批证据再交给模型写一次推论，原文在 raw_answers/*_inference.txt。",
        "推论由模型生成，不在这里改写成疾病结论。关联不是已证明的因果。",
        "",
        f"- 模块数：{len(modules)}",
        f"- 其中单基因模块：{len(single)}",
        f"- 提问数：{len(questions)}",
        f"- 模型接口：{MODEL_URL}",
        "",
    ]
    if not reached:
        lines.extend(
            [
                "微调模型接口没有响应。本次没有生成解释，也没有用手写结论代替。",
                "模型恢复后，用同一命令再跑即可。带有固定查询标记的原文会被复用。",
                "",
            ]
        )
    else:
        lines.extend(["## 调用结果", ""])
        for item in answers:
            lines.append(f"- `{item['raw_file']}`：{item['status']}")
        lines.extend(["", "## 推论", ""])
        if inferences:
            for item in inferences:
                lines.append(f"- `{item['raw_file']}`：{item['status']}")
        else:
            lines.append("- 没有推论文件。")
        lines.append("")
    lines.extend(["## 提问与模块", ""])
    for item in questions:
        joined = ", ".join(item["module_ids"])
        lines.append(f"- `{item['prompt_file']}`：{joined}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Ask MAS to write chains from functional modules")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    loaded = yaml.safe_load(Path(args.config).read_text())
    generate_with_mas(Path(loaded["output_dir"]), loaded, reuse=not args.no_reuse)
