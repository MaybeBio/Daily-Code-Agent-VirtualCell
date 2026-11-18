"""
Polygenic risk score (PRS) report for brain disorders.

Ported from v2_RAG/PSR_report.ipynb:
  1. Load PGS scoring files (PGS Catalog)
  2. Score a sample VCF in one pass
  3. Compare to UK Biobank reference distribution
  4. Summarize with an LLM narrative
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cyvcf2 import VCF  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import CODE as V2_RAG_DIR, DATA

from LLM_utils import CustomOpenAIProxyChat  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from multiagent_system.config import Config  # noqa: E402

DEFAULT_PRS_DIR = os.environ.get(
    "PRS_PGS_DIR",
    str(DATA / "genomics" / "prs"),
)
DEFAULT_REFERENCE_JSON = os.environ.get(
    "PRS_REFERENCE_JSON",
    str(DATA / "genomics" / "ukb" / "UKB_PRS_result.json"),
)
DEFAULT_DEMO_VCF = os.environ.get(
    "PRS_DEMO_VCF",
    str(DATA / "genomics" / "ukb" / "vcf_by_sample" / "1000323.vcf.gz"),
)
DEFAULT_DEMO_SAMPLE_ID = os.environ.get("PRS_DEMO_SAMPLE_ID", "1000323")
CACHE_DIR = Path(os.environ.get("PRS_CACHE_DIR", DATA / "genomics" / "prs_cache"))
REFERENCE_STATS_PATH = CACHE_DIR / "reference_stats.json"

DISORDER_LABELS: dict[str, str] = {
    "AD": "Alzheimer's disease",
    "SCZ": "Schizophrenia",
    "MDD": "Major depressive disorder",
    "ADHD": "ADHD",
    "Anxiety": "Anxiety disorder",
    "ASD": "Autism spectrum disorder",
    "Stroke": "Stroke",
    "Bipolar": "Bipolar disorder",
    "epilepsy": "Epilepsy",
    "PTSD": "PTSD",
    "Parkinsons-disease": "Parkinson's disease",
}

SUMMARY_SYSTEM = """
You are a genetic risk communication assistant for polygenic risk scores (PRS).

Summarize the provided PRS results clearly and objectively.
Highlight disorders with relatively higher or lower percentiles and z-scores.
"""

SUMMARY_USER = """
Sample notes:
{notes}

PRS results (relative to UK Biobank reference, n={reference_n}):
{table}
{variant_section}

Write a concise markdown report with:
1. A short overall summary (3-5 sentences) covering polygenic risk
2. Top PRS traits to note (elevated and low)
3. If variant prioritization data is provided, briefly connect high/moderate-impact GWAS variants and hit genes to the PRS profile

Keep it under 250 words.
"""


def _format_variant_section(variant: dict[str, Any] | None) -> str:
    if not variant or not variant.get("available"):
        return ""

    summary = variant.get("summary", {})
    lines = [
        "",
        "Variant prioritization (GWAS catalog SNPs -> snpEff -> genes):",
        (
            f"- Annotated GWAS variants: {summary.get('n_gwas_variants_annotated', 0)}; "
            f"high/moderate impact: {summary.get('n_hit_variants', 0)}; "
            f"hit genes: {summary.get('n_hit_genes', 0)}"
        ),
    ]

    top_variants = variant.get("top_variants") or []
    if top_variants:
        preview = ", ".join(
            f"{item['rsid']}->{item['gene']}({item['impact']})"
            for item in top_variants[:8]
        )
        lines.append(f"- Example prioritized variants: {preview}")

    hit_genes = variant.get("hit_genes") or []
    if hit_genes:
        lines.append(f"- Example hit genes: {', '.join(hit_genes[:20])}")

    enrichr_terms = variant.get("enrichr_terms") or []
    if enrichr_terms:
        preview = "; ".join(
            f"{term['library']}: {term['term']} ({term['genes']})"
            for term in enrichr_terms[:5]
        )
        lines.append(f"- Enrichr trait terms: {preview}")

    return "\n".join(lines)


def _discover_pgs_files(prs_dir: str | Path) -> dict[str, str]:
    prs_dir = Path(prs_dir)
    catalog: dict[str, str] = {}
    for path in sorted(prs_dir.glob("*_GRCh37.txt.gz")):
        disorder = path.name.split("_")[0]
        catalog[disorder] = str(path)
    if not catalog:
        raise FileNotFoundError(f"No PGS scoring files found in {prs_dir}")
    return catalog


def load_pgs(pgs_file: str | Path) -> dict[str, tuple[str, float]]:
    pgs = pd.read_csv(pgs_file, sep="\t", comment="#", low_memory=False)
    if "chr_name" in pgs.columns and "chr_position" in pgs.columns:
        frame = pgs[["chr_name", "chr_position", "effect_allele", "effect_weight"]]
    else:
        frame = pgs[["hm_chr", "hm_pos", "effect_allele", "effect_weight"]].rename(
            columns={"hm_chr": "chr_name", "hm_pos": "chr_position"}
        )
    frame = frame.dropna(subset=["chr_name", "chr_position"])
    frame["chr_name"] = frame["chr_name"].astype(int).astype(str)
    frame["chr_position"] = frame["chr_position"].astype(int).astype(str)
    keys = (frame["chr_name"] + ":" + frame["chr_position"]).tolist()
    values = list(zip(frame["effect_allele"], frame["effect_weight"].astype(float)))
    return dict(zip(keys, values))


def _cache_path(disorder: str, pgs_file: str) -> Path:
    stem = Path(pgs_file).stem.replace(".txt", "")
    return CACHE_DIR / f"{disorder}_{stem}.pkl"


def load_pgs_cached(disorder: str, pgs_file: str) -> dict[str, tuple[str, float]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(disorder, pgs_file)
    if cache_file.is_file():
        with cache_file.open("rb") as handle:
            return pickle.load(handle)

    data = load_pgs(pgs_file)
    with cache_file.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def load_pgs_catalog(prs_dir: str | Path | None = None) -> dict[str, dict[str, tuple[str, float]]]:
    catalog_files = _discover_pgs_files(prs_dir or DEFAULT_PRS_DIR)
    return {
        disorder: load_pgs_cached(disorder, path)
        for disorder, path in catalog_files.items()
    }


def calculate_prs_batch(
    vcf_path: str | Path,
    pgs_dicts: dict[str, dict[str, tuple[str, float]]],
) -> dict[str, dict[str, float]]:
    vcf = VCF(str(vcf_path))
    results = {
        disorder: {"prs": 0.0, "matched_snps": 0}
        for disorder in pgs_dicts
    }

    for variant in vcf:
        key = f"{variant.CHROM.strip('chr')}:{variant.POS}"
        alleles = [variant.REF] + variant.ALT
        genotype = variant.genotypes[0][:2]

        for disorder, pgs_dict in pgs_dicts.items():
            if key not in pgs_dict:
                continue
            effect_allele, beta = pgs_dict[key]
            if effect_allele not in alleles:
                continue
            dosage = sum(
                1 for allele_index in genotype
                if allele_index >= 0 and alleles[allele_index] == effect_allele
            )
            results[disorder]["prs"] += dosage * beta
            results[disorder]["matched_snps"] += 1

    return results


def build_reference_stats(reference_json: str | Path | None = None) -> dict[str, Any]:
    reference_json = Path(reference_json or DEFAULT_REFERENCE_JSON)
    if not reference_json.is_file():
        raise FileNotFoundError(f"Reference PRS file not found: {reference_json}")

    with reference_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    disorders = list(next(iter(payload.values())).keys())
    distributions: dict[str, list[float]] = {disorder: [] for disorder in disorders}
    matched: dict[str, list[int]] = {disorder: [] for disorder in disorders}

    for sample_scores in payload.values():
        for disorder in disorders:
            entry = sample_scores[disorder]
            distributions[disorder].append(float(entry["prs"]))
            matched[disorder].append(int(entry["matched_snps"]))

    stats_bundle: dict[str, Any] = {
        "reference_n": len(payload),
        "disorders": {},
    }
    for disorder in disorders:
        values = np.asarray(distributions[disorder], dtype=float)
        stats_bundle["disorders"][disorder] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0) or 1.0),
            "median": float(np.median(values)),
            "values": values.tolist(),
        }
    return stats_bundle


def load_reference_stats(reference_json: str | Path | None = None) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reference_json = Path(reference_json or DEFAULT_REFERENCE_JSON)

    if REFERENCE_STATS_PATH.is_file():
        source_mtime = reference_json.stat().st_mtime if reference_json.is_file() else 0
        if REFERENCE_STATS_PATH.stat().st_mtime >= source_mtime:
            with REFERENCE_STATS_PATH.open(encoding="utf-8") as handle:
                return json.load(handle)

    stats_bundle = build_reference_stats(reference_json)
    with REFERENCE_STATS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(stats_bundle, handle)
    return stats_bundle


def enrich_scores(
    raw_scores: dict[str, dict[str, float]],
    reference: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for disorder, entry in raw_scores.items():
        prs = float(entry["prs"])
        matched = int(entry["matched_snps"])
        ref = reference["disorders"].get(disorder)
        if not ref:
            continue

        ref_values = np.asarray(ref["values"], dtype=float)
        z_score = (prs - ref["mean"]) / ref["std"] if ref["std"] else 0.0
        percentile = float(stats.percentileofscore(ref_values, prs, kind="rank"))

        rows.append(
            {
                "disorder": disorder,
                "label": DISORDER_LABELS.get(disorder, disorder),
                "prs": round(prs, 4),
                "matched_snps": matched,
                "z_score": round(z_score, 3),
                "percentile": round(percentile, 1),
                "reference_mean": round(ref["mean"], 4),
                "reference_std": round(ref["std"], 4),
            }
        )

    rows.sort(key=lambda item: item["percentile"], reverse=True)
    return rows


def render_prs_chart(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("No PRS rows to plot")

    ordered = sorted(rows, key=lambda item: item["percentile"])
    labels = [item["label"] for item in ordered]
    percentiles = [item["percentile"] for item in ordered]
    colors = ["#dc2626" if p >= 75 else "#f59e0b" if p >= 50 else "#16a34a" for p in percentiles]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(labels))))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, percentiles, color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile vs UK Biobank reference")
    ax.set_title("Polygenic risk profile")
    ax.axvline(50, color="#9ca3af", linestyle="--", linewidth=1)
    ax.axvline(75, color="#f87171", linestyle=":", linewidth=1)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, facecolor="white")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _format_results_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Disorder | PRS | Percentile | Z-score | Matched SNPs |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['prs']} | {row['percentile']}% | "
            f"{row['z_score']} | {row['matched_snps']} |"
        )
    return "\n".join(lines)


def generate_summary(
    rows: list[dict[str, Any]],
    notes: str,
    reference_n: int,
    variant: dict[str, Any] | None = None,
) -> str:
    config = Config()
    llm = CustomOpenAIProxyChat(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model_id,
        streaming=False,
    )
    prompt = SUMMARY_USER.format(
        notes=notes.strip() or "No additional notes.",
        reference_n=reference_n,
        table=_format_results_table(rows),
        variant_section=_format_variant_section(variant),
    )
    response = llm.invoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )
    content = getattr(response, "content", None) or str(response)
    return content.strip()


class PRSReportService:
    def __init__(
        self,
        prs_dir: str | Path | None = None,
        reference_json: str | Path | None = None,
    ):
        self.prs_dir = prs_dir or DEFAULT_PRS_DIR
        self.reference_json = reference_json or DEFAULT_REFERENCE_JSON
        self._pgs_dicts: dict[str, dict[str, tuple[str, float]]] | None = None
        self._reference: dict[str, Any] | None = None

    @property
    def pgs_dicts(self) -> dict[str, dict[str, tuple[str, float]]]:
        if self._pgs_dicts is None:
            self._pgs_dicts = load_pgs_catalog(self.prs_dir)
        return self._pgs_dicts

    @property
    def reference(self) -> dict[str, Any]:
        if self._reference is None:
            self._reference = load_reference_stats(self.reference_json)
        return self._reference

    def get_demo_payload(self) -> dict[str, Any]:
        with Path(self.reference_json).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        sample_id = DEFAULT_DEMO_SAMPLE_ID
        if sample_id not in payload:
            sample_id = next(iter(payload))
        raw_scores = payload[sample_id]
        rows = enrich_scores(raw_scores, self.reference)
        return self._finalize_report(
            sample_id=sample_id,
            label=f"UKB demo sample {sample_id}",
            source="demo",
            rows=rows,
            notes="Demo profile from precomputed UK Biobank PRS reference.",
            skip_llm=False,
        )

    def analyze_vcf(self, vcf_path: str | Path, notes: str = "") -> dict[str, Any]:
        raw_scores = calculate_prs_batch(vcf_path, self.pgs_dicts)
        rows = enrich_scores(raw_scores, self.reference)
        sample_id = Path(vcf_path).name.replace(".vcf.gz", "").replace(".vcf", "")
        return self._finalize_report(
            sample_id=sample_id,
            label=Path(vcf_path).name,
            source="upload",
            rows=rows,
            notes=notes,
            skip_llm=False,
        )

    def _finalize_report(
        self,
        *,
        sample_id: str,
        label: str,
        source: str,
        rows: list[dict[str, Any]],
        notes: str,
        variant: dict[str, Any] | None = None,
        gene_mas: dict[str, Any] | None = None,
        skip_llm: bool = False,
    ) -> dict[str, Any]:
        chart = render_prs_chart(rows)
        summary = ""
        if not skip_llm:
            summary = generate_summary(
                rows,
                notes,
                self.reference["reference_n"],
                variant,
            )

        return {
            "sample_id": sample_id,
            "label": label,
            "source": source,
            "disorders_tested": len(rows),
            "reference_n": self.reference["reference_n"],
            "rows": rows,
            "table_markdown": _format_results_table(rows),
            "summary": summary,
            "chart_data_url": chart,
            "variant_prioritization": variant or {"available": False},
            "gene_mas_analysis": gene_mas or {"available": False},
        }

    def iter_analyze_events(
        self,
        vcf_path: str | Path,
        notes: str = "",
        *,
        source: str = "upload",
        label: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        from variant_prioritization_service import extract_sample_id, iter_variant_events
        from gene_mas_report import iter_gene_mas_events

        yield {"type": "status", "text": "Loading PGS scoring panels"}
        pgs = self.pgs_dicts
        yield {
            "type": "status",
            "text": f"Loaded {len(pgs)} disorder panels ({', '.join(sorted(pgs)[:4])}...)",
        }

        if source == "demo":
            yield {"type": "status", "text": "Loading demo PRS from UK Biobank reference"}
            with Path(self.reference_json).open(encoding="utf-8") as handle:
                reference_payload = json.load(handle)
            sample_id = DEFAULT_DEMO_SAMPLE_ID
            if sample_id not in reference_payload:
                sample_id = next(iter(reference_payload))
            raw_scores = reference_payload[sample_id]
            rows = enrich_scores(raw_scores, self.reference)
            report_label = label or f"UKB demo sample {sample_id}"
        else:
            yield {"type": "status", "text": "Scoring variants in uploaded VCF (this may take a few minutes)"}
            raw_scores = calculate_prs_batch(vcf_path, pgs)
            yield {"type": "status", "text": "Comparing scores to UK Biobank reference"}
            rows = enrich_scores(raw_scores, self.reference)
            sample_id = extract_sample_id(vcf_path)
            report_label = label or Path(vcf_path).name

        yield {"type": "status", "text": "Rendering PRS risk profile chart"}

        variant_report: dict[str, Any] | None = None
        for event in iter_variant_events(sample_id, vcf_path):
            if event.get("type") == "status":
                yield event
            elif event.get("type") == "variant_done":
                variant_report = event.get("variant_prioritization")

        gene_mas_report: dict[str, Any] | None = None
        for event in iter_gene_mas_events(variant_report, rows):
            if event.get("type") == "status":
                yield event
            elif event.get("type") == "gene_mas_done":
                gene_mas_report = event.get("gene_mas_analysis")

        yield {"type": "status", "text": "Writing combined interpretation"}
        payload = self._finalize_report(
            sample_id=sample_id,
            label=report_label,
            source=source,
            rows=rows,
            notes=notes,
            variant=variant_report,
            gene_mas=gene_mas_report,
        )
        yield {"type": "done", **payload}


_prs_service: PRSReportService | None = None


def get_prs_service() -> PRSReportService:
    global _prs_service
    if _prs_service is None:
        _prs_service = PRSReportService()
    return _prs_service


def save_uploaded_vcf(upload_bytes: bytes, filename: str) -> str:
    suffix = ".vcf.gz" if filename.endswith(".gz") else ".vcf" if filename.endswith(".vcf") else ".vcf"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(upload_bytes)
    handle.close()
    return handle.name
