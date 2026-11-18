"""Small-scale test: structured pure-LLM prompt (v2) vs legacy full_llm_2methods.

Output: output/llm_structured_test_ad50/
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

import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cases" / "genomics"))
UKB_DIR = Path(os.environ.get("BRAINTWIN_UKB_DIR", ROOT / "data" / "genomics" / "ukb"))
PROJECT_ROOT = ROOT

OUTPUT_DIR = ROOT / "cases" / "genomics" / "output" / "structured_risk"
LLM_URL = os.environ.get("BRAINTWIN_LLM_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_ID = os.environ.get("BRAINTWIN_MODEL_ID", "gpt-3.5-turbo")
PROMPT_VERSION = "llm_v3"
MODEL_TAG = "ft"
LLM_TEMPERATURE = 0.1
CONCURRENCY = 8
DISEASE = "AD"
DISORDER_LABEL = "Alzheimer's disease"
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
ALL_DISEASES = ["AD", "SCZ", "MDD", "Anxiety", "Bipolar", "epilepsy",
                "PTSD", "Parkinsons-disease", "ADHD", "Stroke"]

_Z_LOOKUP: dict = {}


def _vp_available(sample_id: str) -> bool:
    return (UKB_DIR / "variant_prioritization_results" / str(sample_id)).is_dir()


def build_full_samples(disease: str, md: pd.DataFrame) -> list[tuple[str, int]]:
    col = DISEASE_TO_META_COL.get(disease, disease)
    cases = [
        str(s) for s in md[(md[col] == 1) & (md["Disease_Count"] > 0)]["eid"].tolist()
        if _vp_available(s)
    ]
    ctrls = [
        str(s) for s in md[md["Disease_Count"] == 0]["eid"].tolist()
        if _vp_available(s)
    ]
    return [(s, 1) for s in cases] + [(s, 0) for s in ctrls]


def get_prs_z(sample_id: str) -> tuple[float | None, float | None]:
    global _Z_LOOKUP
    if not _Z_LOOKUP:
        df = pd.read_csv(UKB_DIR / "UKB_PRS_result_zscore.csv", index_col=0)
        df.index = df.index.astype(str)
        _Z_LOOKUP = {sid: {d: float(df.loc[sid, d]) for d in df.columns} for sid in df.index}
    z = _Z_LOOKUP.get(sample_id, {}).get(DISEASE)
    return z, None


def get_all_high_genes(sample_id: str) -> list[str]:
    from variant_prioritization_service import load_variant_report
    variant = load_variant_report(sample_id)
    if not variant.get("available"):
        return []
    return list(variant.get("hit_genes_high") or [])


def build_prs_text(z: float | None, pct: float | None = None) -> str:
    if z is None:
        return f"Disease: {DISORDER_LABEL}. No PRS data available."
    if pct is None and z is not None:
        from scipy.stats import norm
        pct = float(norm.cdf(z) * 100.0)
    pct_s = f"{pct:.1f}" if pct is not None else "N/A"
    return (f"Disease: {DISORDER_LABEL}. PRS z-score = {z:.3f} "
            f"(percentile {pct_s}%). Interpret this z-score for disease risk.")


SYSTEM_LLM_V3 = """You are a research-grade genetic risk synthesis assistant.
Never say the person will develop a disease.
Assess ONE target disease only. Do not mention or judge any other disease.

Return valid JSON with keys:
  disease, assessment, confidence, prs_only_confidence, evidence_used, genes_cited, rationale

STEP A - PRS-only baseline (do this first):
  From the PRS z-score and percentile alone, set prs_only_confidence in [0,1].
  Use your own judgement (no fixed formula), but it MUST reflect PRS direction:
  high percentile / positive z → higher score; low percentile / negative z → lower score.
  Do NOT default to 0.5 unless z is near 0 and you explicitly justify neutrality.

STEP B - Hit gene set (secondary evidence; REQUIRED when genes are provided):
  You receive ALL HIGH-impact hit genes. Use TWO complementary signals:
  (1) BURDEN: the count of HIGH-impact genes is itself genetic evidence.
      Many hits (roughly 20+) suggest a substantive rare/strong-variant contribution.
  (2) NAMED GENES: pick 1-5 genes copied exactly from the provided hit list.
      Briefly explain plausible brain, synaptic, immune, or disease-relevant roles.
      You do NOT need GWAS-level proof — functional plausibility is enough.

  If the hit-gene list is non-empty you MUST:
    - include "hit_genes" in evidence_used
    - populate genes_cited with 1-5 symbols from that list (never leave it empty)
    - discuss gene burden and named genes in rationale part (2)

STEP C - Synthesize final confidence:
  Start from prs_only_confidence. Adjust modestly when gene burden / named genes
  support or conflict with the PRS direction. Final confidence may equal
  prs_only_confidence only if genes are truly uninformative (rare when many hits exist).

Rules:
- assessment: "supported" | "uncertain" | "not_supported"
- evidence_used: include "PRS"; add "hit_genes" whenever a hit list is provided
- genes_cited: 1-5 gene symbols from the hit list when genes are provided
- rationale: three parts — (1) PRS-only view (2) gene burden + cited genes (3) synthesis
FORMATTING: output ONLY the JSON object; keep rationale under 220 words."""


def build_user_prompt(sample_id: str, prs_txt: str, all_high: list[str]) -> str:
    genes_str = ", ".join(all_high) if all_high else "(none)"
    gene_req = (
        "2) REQUIRED: treat hit genes as burden + named evidence. "
        "Set evidence_used to include both PRS and hit_genes. "
        "Copy 1-5 gene symbols from the list into genes_cited and explain them.\n"
        if all_high else
        "2) No hit genes provided; evidence_used may be [\"PRS\"] only.\n"
    )
    return (
        f"Sample: {sample_id}\nTarget disease (ONLY this disease): {DISORDER_LABEL}\n\n"
        f"PRS analysis:\n{prs_txt}\n\n"
        f"Hit genes (all HIGH-impact, {len(all_high)} genes): {genes_str}\n\n"
        f"REQUIRED procedure:\n"
        f"1) Set prs_only_confidence from PRS z-score/percentile alone (must vary with PRS).\n"
        f"{gene_req}"
        f"3) Synthesize final confidence and assessment from PRS + gene evidence.\n"
        f"4) Do not discuss any disease other than {DISORDER_LABEL}.\n\n"
        f"Produce JSON."
    )


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
    r = requests.post(f"{LLM_URL}/v1/chat/completions", json=payload, timeout=(10, timeout))
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
        return {"assessment": "uncertain", "confidence": 0.5, "prs_only_confidence": 0.5,
                "evidence_used": ["PRS"], "genes_cited": [], "rationale": "no PRS, fallback"}
    if z >= 1.0:
        a, c = "supported", 0.7
    elif z <= -1.0:
        a, c = "not_supported", 0.3
    else:
        a, c = "uncertain", 0.5
    return {"assessment": a, "confidence": c, "prs_only_confidence": c,
            "evidence_used": ["PRS"], "genes_cited": [], "rationale": "LLM parse failed, fallback"}


def run_one(sample_id: str, truth: int, force: bool = False) -> dict[str, Any]:
    out_file = OUTPUT_DIR / DISEASE / f"sample_{sample_id}.json"
    if out_file.exists() and not force:
        return json.load(out_file.open())

    z, _ = get_prs_z(sample_id)
    all_high = get_all_high_genes(sample_id)
    prs_txt = build_prs_text(z)
    t0 = time.time()
    try:
        raw = call_llm(SYSTEM_LLM_V3, build_user_prompt(sample_id, prs_txt, all_high))
        res = extract_json_block(raw)
        if not isinstance(res, dict):
            res = _z_fallback(z)
            res["_fallback"] = "non-dict"
    except Exception as e:
        res = _z_fallback(z)
        res["_fallback"] = str(e)
    elapsed = time.time() - t0

    rec = {
        "sample_id": sample_id,
        "disease": DISEASE,
        "truth": truth,
        "prs_z": z,
        "n_high_genes": len(all_high),
        "hit_genes": all_high,
        "prompt_version": PROMPT_VERSION,
        "model_tag": MODEL_TAG,
        "llm_url": LLM_URL,
        "llm_time_s": round(elapsed, 2),
        "structured_llm": {
            "assessment": res.get("assessment"),
            "confidence": res.get("confidence"),
            "prs_only_confidence": res.get("prs_only_confidence"),
            "evidence_used": res.get("evidence_used"),
            "genes_cited": res.get("genes_cited"),
            "rationale": res.get("rationale", ""),
            "_fallback": res.get("_fallback"),
        },
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    return rec


def main():
    import argparse
    global OUTPUT_DIR, LLM_URL, DISEASE, DISORDER_LABEL, PROMPT_VERSION, MODEL_TAG

    parser = argparse.ArgumentParser()
    parser.add_argument("--diseases", nargs="+", default=["AD"],
                        help="Disease code(s); use 'all' for 10-disease full run")
    parser.add_argument("--full", action="store_true", help="Run all case/control samples with variant data")
    parser.add_argument("--samples-file", type=str, default=None, help="Optional fixed sample ID list")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--llm-url", type=str, default=None)
    parser.add_argument("--model-tag", choices=["ft", "base"], default="ft",
                        help="Model variant label: ft (fine-tuned) or base")
    parser.add_argument("--prompt-version", type=str, default=None,
                        help="Override prompt version tag stored in outputs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    MODEL_TAG = args.model_tag
    if args.prompt_version:
        PROMPT_VERSION = args.prompt_version
    elif MODEL_TAG == "base":
        PROMPT_VERSION = "llm_v3_base"
    else:
        PROMPT_VERSION = "llm_v3"

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    elif args.full:
        OUTPUT_DIR = ROOT / "cases" / "genomics" / "output" / (
            "structured_risk_full_base" if MODEL_TAG == "base" else "structured_risk_full"
        )
    if args.llm_url:
        LLM_URL = args.llm_url.rstrip("/")

    diseases = ALL_DISEASES if len(args.diseases) == 1 and args.diseases[0].lower() == "all" else args.diseases

    md = pd.read_csv(UKB_DIR / "Metadata.csv")
    md["eid"] = md["eid"].astype(str)

    all_results: list[dict] = []
    for disease in diseases:
        DISEASE = disease
        DISORDER_LABEL = DISORDER_LABELS.get(disease, disease)
        _Z_LOOKUP.clear()

        if args.full:
            samples = build_full_samples(disease, md)
            sample_ids = [sid for sid, _ in samples]
        else:
            samples_file = Path(args.samples_file) if args.samples_file else (
                ROOT / "cases" / "genomics" / "output" / "sample_ids.txt"
            )
            sample_ids = [ln.strip() for ln in samples_file.read_text().splitlines() if ln.strip()]
            col = DISEASE_TO_META_COL.get(disease, disease)
            cases = set(md[(md[col] == 1) & (md["Disease_Count"] > 0)]["eid"].tolist())
            samples = [(sid, 1 if sid in cases else 0) for sid in sample_ids]

        print(f"LLM structured test: {DISEASE}, n={len(samples)}, model={MODEL_TAG}, "
              f"prompt={PROMPT_VERSION}, url={LLM_URL}, T={LLM_TEMPERATURE}, out={OUTPUT_DIR}",
              flush=True)

        t0 = time.time()
        results = []

        def _process(item):
            sid, truth = item
            return run_one(sid, truth, force=args.force)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_process, s): s for s in samples}
            done = 0
            progress_step = 50 if args.full else 10
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % progress_step == 0:
                    print(f"  [{DISEASE}] {done}/{len(samples)} done", flush=True)

        elapsed = time.time() - t0
        times = [r["llm_time_s"] for r in results]
        print(f"\n[{DISEASE}] Done: {len(results)} samples in {elapsed:.1f}s "
              f"({elapsed/len(results):.2f}s/sample avg, "
              f"per-call median {sorted(times)[len(times)//2]:.2f}s)", flush=True)

        rows = []
        for r in results:
            sl = r["structured_llm"]
            rows.append({
                "sample_id": r["sample_id"], "disease": DISEASE, "truth": r["truth"], "prs_z": r["prs_z"],
                "structured_conf": sl.get("confidence"),
                "prs_only_conf": sl.get("prs_only_confidence"),
                "n_genes_cited": len(sl.get("genes_cited") or []),
                "evidence_used": json.dumps(sl.get("evidence_used") or []),
                "llm_time_s": r["llm_time_s"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / f"summary_{DISEASE}.csv", index=False)
        if not args.full:
            df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

        y = df.truth.values
        label = f"full {DISEASE}" if args.full else f"50 {DISEASE}"
        print(f"\n=== AUC ({label}, n={len(df)}) ===")
        print(f"  PRS z:              {roc_auc_score(y, df.prs_z.fillna(0)):.3f}")
        print(f"  structured {PROMPT_VERSION} conf:   {roc_auc_score(y, df.structured_conf.fillna(0.5)):.3f}")
        print(f"  structured prs_only:{roc_auc_score(y, df.prs_only_conf.fillna(0.5)):.3f}")

        old_dir = ROOT / "cases" / "genomics" / "output" / "legacy_llm" / DISEASE
        old_rows = []
        for sid in sample_ids:
            f = old_dir / f"sample_{sid}.json"
            if f.exists():
                o = json.load(f.open())
                old_rows.append({
                    "sample_id": sid, "truth": o["truth"],
                    "legacy_general": o.get("general_llm", {}).get("confidence"),
                    "legacy_expert": o.get("expert_llm", {}).get("confidence"),
                })
        if old_rows:
            old = pd.DataFrame(old_rows)
            m = df.merge(old, on=["sample_id", "truth"])
            y2 = m.truth.values
            print(f"  legacy General LLM: {roc_auc_score(y2, m.legacy_general.fillna(0.5)):.3f}")
            print(f"  legacy Expert LLM:  {roc_auc_score(y2, m.legacy_expert.fillna(0.5)):.3f}")

        kg_dir = ROOT / "cases" / "genomics" / "output" / "disease_risk_kg" / DISEASE
        kg_rows = []
        for sid in sample_ids:
            f = kg_dir / f"sample_{sid}.json"
            if f.exists():
                o = json.load(f.open())
                kg_rows.append({
                    "sample_id": sid,
                    "v6_kg": o.get("braintwin_kg", {}).get("confidence"),
                })
        if kg_rows:
            kg = pd.DataFrame(kg_rows)
            m2 = df.merge(kg, on="sample_id")
            print(f"  v6 KG:              {roc_auc_score(m2.truth, m2.v6_kg.fillna(0.5)):.3f}")

        fb = sum(1 for r in results if r["structured_llm"].get("_fallback"))
        cited = sum(1 for r in results if (r["structured_llm"].get("genes_cited") or []))
        print(f"\n  fallback: {fb}/{len(results)}")
        print(f"  genes_cited non-empty: {cited}/{len(results)}")
        print(f"  avg genes_cited: {df.n_genes_cited.mean():.1f}")
        all_results.extend(results)


if __name__ == "__main__":
    main()
