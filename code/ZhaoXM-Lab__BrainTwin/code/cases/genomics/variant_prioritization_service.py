"""
Load / run variant prioritization (snpEff -> gene -> Enrichr) for Genetic Risk Analysis.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA

V2_RAG_DIR = DATA
VARIANT_ROOT = DATA / "variant_prioritization"
DEMO_OUTPUT_DIR = Path(
    os.environ.get(
        "VARIANT_DEMO_DIR",
        str(VARIANT_ROOT / "demo_output"),
    )
)
RESULTS_ROOT = Path(
    os.environ.get(
        "VARIANT_RESULTS_ROOT",
        str(DATA / "genomics" / "ukb" / "variant_prioritization_results"),
    )
)
GWAS_LIST = Path(
    os.environ.get(
        "VARIANT_GWAS_LIST",
        str(VARIANT_ROOT / "resources" / "GWAS_SNP_list.txt"),
    )
)
PIPELINE_PY = Path(
    os.environ.get(
        "VARIANT_PIPELINE_PY",
        str(VARIANT_ROOT / "run_pipeline.py"),
    )
)
SNPEFF_JAR = os.environ.get(
    "SNPEFF_JAR",
    str(DATA / "snpeff" / "snpEff.jar"),
)
SNPSIFT_JAR = os.environ.get(
    "SNPSIFT_JAR",
    str(DATA / "snpeff" / "SnpSift.jar"),
)
SNPEFF_ENV = os.environ.get("SNPEFF_ENV", "vep_env")
DEMO_SAMPLE_IDS = {"1000323", "1001471", "1004223"}


def extract_sample_id(vcf_path: str | Path) -> str:
    name = Path(vcf_path).name
    for suffix in (".vcf.gz", ".vcf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def find_precomputed_dir(sample_id: str) -> Path | None:
    for root in (DEMO_OUTPUT_DIR, RESULTS_ROOT):
        candidate = root / sample_id
        payload = candidate / "final_payload.json"
        if payload.is_file():
            return candidate
    return None


DISPLAY_VARIANT_LIMIT = 50
DISPLAY_GENE_LIMIT = 60


def _group_hit_genes(
    all_variants: list[dict[str, Any]],
    *,
    display_limit: int = DISPLAY_GENE_LIMIT,
) -> dict[str, Any]:
    gene_impact: dict[str, str] = {}
    for row in all_variants:
        gene = str(row.get("gene") or "").strip()
        if not gene:
            continue
        impact = str(row.get("impact") or "").upper()
        if impact == "HIGH":
            gene_impact[gene] = "HIGH"
        elif gene not in gene_impact:
            gene_impact[gene] = "MODERATE"

    high_genes = sorted(
        [gene for gene, impact in gene_impact.items() if impact == "HIGH"],
        key=str.upper,
    )
    moderate_genes = sorted(
        [gene for gene, impact in gene_impact.items() if impact != "HIGH"],
        key=str.upper,
    )

    shown_high = high_genes[:display_limit]
    remaining = max(0, display_limit - len(shown_high))
    shown_moderate = moderate_genes[:remaining]

    return {
        "hit_genes_high": high_genes,
        "hit_genes_moderate": moderate_genes,
        "display_gene_groups": {
            "high": shown_high,
            "moderate": shown_moderate,
        },
        "n_hit_genes_high": len(high_genes),
        "n_hit_genes_moderate": len(moderate_genes),
        "n_display_genes": len(shown_high) + len(shown_moderate),
    }


def _parse_hit_variants(
    ann_best_tsv: Path,
    display_limit: int = DISPLAY_VARIANT_LIMIT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not ann_best_tsv.is_file():
        return [], []

    df = pd.read_csv(ann_best_tsv, sep="\t", dtype=str)
    if "BEST_IMPACT_SCORE" not in df.columns:
        return [], []

    df = df[df["CHROM"] != "CHROM"].copy()

    df["BEST_IMPACT_SCORE"] = pd.to_numeric(df["BEST_IMPACT_SCORE"], errors="coerce").fillna(-1)
    hits = df[df["BEST_IMPACT_SCORE"] >= 3].copy()
    impact_rank = {"HIGH": 2, "MODERATE": 1}
    hits["_rank"] = hits["BEST_IMPACT"].map(lambda x: impact_rank.get(str(x), 0))
    hits = hits.sort_values(["_rank", "BEST_IMPACT_SCORE"], ascending=[False, False])

    all_rows: list[dict[str, Any]] = []
    for _, row in hits.iterrows():
        all_rows.append(
            {
                "chrom": row.get("CHROM", ""),
                "pos": row.get("POS", ""),
                "rsid": row.get("ID", ""),
                "ref": row.get("REF", ""),
                "alt": row.get("ALT", ""),
                "gene": row.get("BEST_GENE", ""),
                "impact": row.get("BEST_IMPACT", ""),
            }
        )

    display_rows = all_rows[:display_limit]
    return display_rows, all_rows


def _parse_enrichr_terms(payload: dict[str, Any], limit_per_library: int = 10) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    enrichr_blob = payload.get("enrichr", {}).get("top_terms") or {}

    if not enrichr_blob:
        enrichr_path = payload.get("files", {}).get("enrichr_results_json")
        if enrichr_path and Path(enrichr_path).is_file():
            with Path(enrichr_path).open(encoding="utf-8") as handle:
                enrichr_blob = json.load(handle)

    for library, records in enrichr_blob.items():
        if not isinstance(records, list):
            continue
        for record in records[:limit_per_library]:
            terms.append(
                {
                    "library": library,
                    "term": record.get("Term", ""),
                    "adjusted_p": record.get("Adjusted P-value"),
                    "genes": record.get("Genes", ""),
                }
            )
    return terms


def load_variant_report(sample_id: str) -> dict[str, Any]:
    outdir = find_precomputed_dir(sample_id)
    if outdir is None:
        return {
            "available": False,
            "sample_id": sample_id,
            "message": (
                f"No precomputed variant prioritization for sample {sample_id}. "
                f"Demo samples: {', '.join(sorted(DEMO_SAMPLE_IDS))}."
            ),
        }

    payload_path = outdir / "final_payload.json"
    with payload_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    ann_best = outdir / "ann.best.tsv"
    if not ann_best.is_file():
        ann_best = Path(payload.get("files", {}).get("ann_tsv_best", ""))

    summary = payload.get("summary", {})
    hit_genes = payload.get("hit_genes", [])
    top_variants, all_variants = _parse_hit_variants(Path(ann_best))
    enrichr_terms = _parse_enrichr_terms(payload)
    total_hit_variants = summary.get("n_hit_variants_after_filter", len(all_variants))
    gene_groups = _group_hit_genes(all_variants)

    return {
        "available": True,
        "sample_id": sample_id,
        "source_dir": str(outdir),
        "summary": {
            "n_gwas_variants_annotated": summary.get("n_total_variants_in_ann", 0),
            "n_hit_variants": total_hit_variants,
            "n_hit_genes": summary.get("n_hit_genes", len(hit_genes)),
            "n_hit_genes_high": gene_groups["n_hit_genes_high"],
            "n_hit_genes_moderate": gene_groups["n_hit_genes_moderate"],
            "impact_rule": summary.get("impact_rule", ""),
            "protein_coding_required": summary.get("protein_coding_required", True),
        },
        "top_variants": top_variants,
        "all_variants": all_variants,
        "n_display_variants": len(top_variants),
        "n_total_hit_variants": total_hit_variants or len(all_variants),
        "hit_genes": hit_genes[:200],
        "hit_genes_high": gene_groups["hit_genes_high"],
        "hit_genes_moderate": gene_groups["hit_genes_moderate"],
        "display_gene_groups": gene_groups["display_gene_groups"],
        "n_display_genes": gene_groups["n_display_genes"],
        "enrichr_terms": enrichr_terms,
    }


def pipeline_available() -> bool:
    return (
        PIPELINE_PY.is_file()
        and Path(SNPEFF_JAR).is_file()
        and Path(SNPSIFT_JAR).is_file()
        and GWAS_LIST.is_file()
    )


def run_variant_pipeline(vcf_path: str | Path, sample_id: str) -> dict[str, Any]:
    if not pipeline_available():
        return {
            "available": False,
            "sample_id": sample_id,
            "message": "Variant prioritization pipeline tools are not configured on this server.",
        }

    outdir = RESULTS_ROOT / sample_id
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PIPELINE_PY),
        "--vcf",
        str(Path(vcf_path).resolve()),
        "--gwas_list",
        str(GWAS_LIST),
        "--outdir",
        str(outdir),
        "--snpeff_env",
        SNPEFF_ENV,
        "--snpeff_jar",
        SNPEFF_JAR,
        "--snpsift_jar",
        SNPSIFT_JAR,
        "--require_protein_coding",
    ]
    subprocess.run(cmd, check=True)
    return load_variant_report(sample_id)


def build_variant_report(
    sample_id: str,
    vcf_path: str | Path | None = None,
    *,
    allow_run: bool = False,
) -> dict[str, Any]:
    report = load_variant_report(sample_id)
    if report.get("available"):
        return report

    if allow_run and vcf_path is not None and os.environ.get("VARIANT_PRIORITIZATION_RUN") == "1":
        try:
            return run_variant_pipeline(vcf_path, sample_id)
        except Exception as exc:
            return {
                "available": False,
                "sample_id": sample_id,
                "message": f"Variant prioritization failed: {exc}",
            }

    return report


def iter_variant_events(
    sample_id: str,
    vcf_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    yield {"type": "status", "text": "Loading GWAS variant prioritization"}
    precomputed = find_precomputed_dir(sample_id)
    if precomputed:
        yield {
            "type": "status",
            "text": f"Using precomputed snpEff results for sample {sample_id}",
        }
    elif os.environ.get("VARIANT_PRIORITIZATION_RUN") == "1" and vcf_path:
        yield {
            "type": "status",
            "text": "Running snpEff variant prioritization pipeline (may take several minutes)",
        }
    else:
        yield {
            "type": "status",
            "text": f"No precomputed variant results for {sample_id}; skipping live snpEff run",
        }

    report = build_variant_report(
        sample_id,
        vcf_path,
        allow_run=os.environ.get("VARIANT_PRIORITIZATION_RUN") == "1",
    )
    if report.get("available"):
        summary = report["summary"]
        yield {
            "type": "status",
            "text": (
                f"Variant prioritization: {summary['n_hit_variants']} high/moderate variants, "
                f"{summary.get('n_hit_genes_high', 0)} HIGH genes, "
                f"{summary.get('n_hit_genes_moderate', 0)} MODERATE genes"
            ),
        }
    yield {"type": "variant_done", "variant_prioritization": report}
