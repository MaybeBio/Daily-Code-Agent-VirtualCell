#!/usr/bin/env python3
"""Figure 5 with the existing MRI session.

A disease name is the question. MRIParsingSession returns feature keys from
that sample's FreeSurfer schema. Cohort numbers are then read from those keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

from mri_session import MRIParsingSession

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output" / "cohort_questions"
UCLA_JSON = ROOT / "data" / "mri" / "UCLA" / "all_sample_structured.json"
UCLA_META = ROOT / "data" / "mri" / "UCLA" / "subjects_meta.csv"
ADNI_JSON = ROOT / "data" / "mri" / "ADNI" / "all_sample_structured.json"
ADNI_ASEG = ROOT / "data" / "mri" / "ADNI" / "aseg_features.csv"
V2_RAG = ROOT / "braintwin"
BROAD_PHRASES = {
    "frontal",
    "temporal",
    "parietal",
    "occipital",
    "cingulate",
    "cerebellum",
    "cerebellar",
    "ventricle",
    "cortex",
    "lobe",
    "subcortical",
}

TASKS = [
    {
        "id": "ucla_schizophrenia",
        "panel": "main",
        "cohort": "UCLA",
        "disease": "schizophrenia",
        "question": "Which structural MRI features should be examined for schizophrenia?",
    },
    {
        "id": "ucla_bipolar",
        "panel": "main",
        "cohort": "UCLA",
        "disease": "bipolar disorder",
        "question": "Which structural MRI features should be examined for bipolar disorder?",
    },
    {
        "id": "ucla_adhd",
        "panel": "main",
        "cohort": "UCLA",
        "disease": "ADHD",
        "question": "Which structural MRI features should be examined for ADHD?",
    },
    {
        "id": "adni_alzheimer",
        "panel": "adni_subplot",
        "cohort": "ADNI",
        "disease": "Alzheimer's disease",
        "question": "Which structural MRI features should be examined for Alzheimer's disease?",
    },
]


def load_ucla() -> tuple[dict, dict[str, str]]:
    samples = json.loads(UCLA_JSON.read_text())
    labels = {}
    with UCLA_META.open(newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["subject_id"]] = row["diagnosis_from_id_prefix"]
    return samples, labels


def load_adni() -> tuple[dict, dict[str, str]]:
    samples = json.loads(ADNI_JSON.read_text())
    labels = {}
    with ADNI_ASEG.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Group"] in {"AD", "CN"} and row["Subject_ID"] in samples:
                labels[row["Subject_ID"]] = row["Group"]
    return {sid: samples[sid] for sid in labels}, labels


def feature_value(sample: dict, path: str):
    current = sample
    for step in path.split("."):
        if not isinstance(current, dict) or step not in current:
            return None
        current = current[step]
    if isinstance(current, dict):
        return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def unit_of(path: str) -> str:
    metric = path.rsplit(".", 1)[-1]
    if metric == "Volume" or metric == "GrayVol":
        return "mm^3"
    if metric in {"ThickAvg", "ThickStd"}:
        return "mm"
    if metric == "SurfArea":
        return "mm^2"
    return ""


def roi_of(path: str) -> str:
    parts = path.split(".")
    if len(parts) >= 3 and parts[1] == "regional":
        if parts[0] == "surface" and len(parts) >= 4:
            return f"{parts[2]}.{parts[3]}"
        return parts[2]
    return path


def summarize(samples: dict, labels: dict[str, str], features: list[str]) -> list[dict]:
    rows = []
    for path in features:
        grouped: dict[str, list[float]] = {}
        for sid, sample in samples.items():
            label = labels.get(sid)
            value = feature_value(sample, path)
            if label is None or value is None:
                continue
            grouped.setdefault(label, []).append(value)
        digits = 3 if unit_of(path) == "mm" else 1
        by_label = []
        for label in sorted(grouped):
            values = grouped[label]
            by_label.append(
                {
                    "label": label,
                    "n": len(values),
                    "mean": round(statistics.fmean(values), digits),
                    "sd": round(statistics.stdev(values), digits) if len(values) > 1 else None,
                }
            )
        rows.append(
            {
                "feature": path,
                "roi": roi_of(path),
                "unit": unit_of(path),
                "by_label": by_label,
            }
        )
    return rows


def run_task(task: dict, samples: dict, labels: dict[str, str], template_id: str) -> dict:
    started = time.time()
    session = MRIParsingSession(samples[template_id])
    result = session.ask(task["question"])
    raw_features = list(result.get("features") or [])
    kept = []
    dropped = []
    for path in raw_features:
        if path in kept or path in dropped:
            continue
        if feature_value(samples[template_id], path) is None:
            dropped.append(path)
        else:
            kept.append(path)
    return {
        "id": task["id"],
        "panel": task["panel"],
        "cohort": task["cohort"],
        "disease": task["disease"],
        "question": task["question"],
        "template_subject": template_id,
        "n_subjects": len(samples),
        "seconds": round(time.time() - started, 2),
        "regions": [roi_of(path) for path in kept],
        "features": kept,
        "dropped_features": dropped,
        "volume_selection_reason": result.get("volume_selection_reason"),
        "surface_selection_reason": result.get("surface_selection_reason"),
        "session_answer_about_template_subject": result.get("answer"),
        "descriptive": summarize(samples, labels, kept),
        "caveat": (
            "Means are computed from the selected keys. Age, sex, and QC are unavailable. "
            "These are descriptive values, not atrophy or a confirmed disease effect."
        ),
    }


def _roi_phrases(sample: dict) -> list[tuple[str, str, str | None, str]]:
    entries = []
    for key in sample["volume"]["regional"]:
        core = key[5:] if key.startswith("Left_") else key[6:] if key.startswith("Right_") else key
        entries.append(("volume", key, None, core.replace("_", " ").lower()))
    for key in sample["surface"]["regional"]:
        spaced = re.sub(r"(frontal|temporal|parietal|occipital|cingulate)", r" \1", key)
        phrase = re.sub(r"\s+", " ", spaced).strip().lower()
        entries.append(("surface", key, "left_hemisphere", phrase))
        entries.append(("surface", key, "right_hemisphere", phrase))
    return entries


def _phrase_hits(entries, text: str) -> dict[str, list[tuple[str, str, str | None]]]:
    hits: dict[str, list[tuple[str, str, str | None]]] = {}
    for kind, roi, hemisphere, phrase in entries:
        if len(phrase) < 5:
            continue
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            hits.setdefault(phrase, []).append((kind, roi, hemisphere))
    return hits


def _feature_path(kind: str, roi: str, hemisphere: str | None) -> str:
    if kind == "volume":
        return f"volume.regional.{roi}.Volume"
    return f"surface.regional.{roi}.{hemisphere}.ThickAvg"


def ask_hybrid(question: str) -> dict:
    if str(V2_RAG) not in sys.path:
        sys.path.insert(0, str(V2_RAG))
    from multiagent_system.main import create_system

    system = create_system()
    for llm in (system.llm_manager.llm_base, system.llm_manager.llm_ft):
        if llm is not None:
            llm.streaming = False
    document_error = None
    doc_state: dict = {}
    print("document rag", flush=True)
    try:
        app = system.rag_manager.get_document_rag()
        if app is None:
            document_error = "Document RAG not available"
        else:
            doc_state = app.invoke({"question": question})
    except Exception as exc:
        document_error = f"{type(exc).__name__}: {exc}"
        store = system.rag_manager.vector_store
        if store is not None:
            print(f"document generation failed ({document_error}); keeping retrieved passages", flush=True)
            doc_state = {
                "answer": None,
                "context": store.similarity_search(question, k=8),
                "sources": [],
            }
    print("graph rag", flush=True)
    graph = system.rag_manager.query_graph_rag(question)
    passages = []
    evidence_parts = []
    for doc in doc_state.get("context") or []:
        content = getattr(doc, "page_content", "") or ""
        meta = getattr(doc, "metadata", {}) or {}
        evidence_parts.append(content)
        passages.append(
            {
                "title": meta.get("title") or "",
                "source": meta.get("source") or "",
                "page": meta.get("page") or "",
                "text": content[:800],
            }
        )
    graph_text = str(graph.get("qa_result_text") or graph.get("final_answer") or "")
    evidence_parts.append(graph_text)
    return {
        "document_answer": doc_state.get("answer"),
        "document_error": document_error,
        "document_query": doc_state.get("query"),
        "sources": doc_state.get("sources") or [],
        "passages": passages,
        "graph_answer": graph.get("final_answer"),
        "graph_cypher": (graph.get("details") or {}).get("cypher"),
        "graph_result": graph_text[:4000],
        "evidence_text": "\n".join(evidence_parts).lower(),
    }


def run_hybrid(task: dict, samples: dict, labels: dict[str, str], template_id: str) -> dict:
    started = time.time()
    question = (
        f"Which brain regions are reported to change on structural MRI in {task['disease']}? "
        "Name the regions and the reported direction of change."
    )
    hybrid = ask_hybrid(question)
    entries = _roi_phrases(samples[template_id])
    evidence_hits = _phrase_hits(entries, hybrid.pop("evidence_text"))
    answer_hits = _phrase_hits(entries, str(hybrid.get("document_answer") or "").lower())
    specific = {}
    broad = {}
    for phrase, matches in evidence_hits.items():
        if phrase in BROAD_PHRASES or phrase.endswith(" lobe"):
            broad[phrase] = matches
        else:
            specific[phrase] = matches
    features = []
    for phrase, matches in specific.items():
        for kind, roi, hemisphere in matches:
            path = _feature_path(kind, roi, hemisphere)
            if path not in features and feature_value(samples[template_id], path) is not None:
                features.append(path)
    answer_only = sorted(set(answer_hits) - set(evidence_hits))
    hybrid["passages"] = hybrid["passages"][:8]
    return {
        "id": task["id"],
        "panel": task["panel"],
        "cohort": task["cohort"],
        "disease": task["disease"],
        "source": "braintwin_document_rag_and_graph_rag",
        "question": question,
        "n_subjects": len(samples),
        "seconds": round(time.time() - started, 2),
        "attested_regions": sorted(specific),
        "broad_regions_not_substituted": sorted(broad),
        "answer_regions_absent_from_retrieval": answer_only,
        "features": features,
        "descriptive": summarize(samples, labels, features),
        "hybrid": hybrid,
        "caveat": (
            "Region names are kept only when they appear in the retrieved passages or the graph result. "
            "Means are descriptive. Age, sex, and QC are unavailable."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="ucla_schizophrenia")
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    selected = [task for task in TASKS if task["id"] == args.id]
    if not selected:
        known = ", ".join(task["id"] for task in TASKS)
        raise SystemExit(f"Unknown subplot {args.id}. Known: {known}")
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading cohorts", flush=True)
    loaders = {"UCLA": load_ucla, "ADNI": load_adni}
    templates = {
        "UCLA": "sub-10159",
        "ADNI": "4552",
    }
    needed = {task["cohort"] for task in selected}
    cohorts = {name: loaders[name]() for name in needed}
    for name, (samples, _labels) in cohorts.items():
        print(f"{name} {len(samples)} template {templates[name]}", flush=True)
    records = []
    for task in selected:
        samples, labels = cohorts[task["cohort"]]
        print(f"ask {task['id']} hybrid={args.hybrid}", flush=True)
        if args.hybrid:
            record = run_hybrid(task, samples, labels, templates[task["cohort"]])
        else:
            record = run_task(task, samples, labels, templates[task["cohort"]])
        records.append(record)
        out_name = "hybrid_results.json" if args.hybrid else "results.json"
        (OUT / out_name).write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(
            f"done {task['id']} features={len(record['features'])}",
            flush=True,
        )
        for path in record["features"]:
            print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
