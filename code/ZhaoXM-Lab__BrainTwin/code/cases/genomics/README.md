# Genomics cases

These scripts start from a VCF, an SNP list, or genes already linked to variants. They call the knowledge graph and, where noted, the chat model. They do not read an MRI or a single-cell count matrix, except `SNP_report.py`, which scores hit genes on the STAB2 reference atlas.

Set `BRAINTWIN_LLM_URL`, `BRAINTWIN_MODEL_ID`, `BRAINTWIN_NEO4J_URL`, and `BRAINTWIN_NEO4J_PASSWORD` before a script that calls the model or the graph.

| Script | What it does | Inputs under `data/` |
|---|---|---|
| `prs_report.py` | Scores one VCF with PGS files and compares the score with the UKB reference distribution. | `genomics/prs/`, `genomics/ukb/UKB_PRS_result.json`, `genomics/ukb/vcf_by_sample/` |
| `PSR_report_SCZ.py` | Scores a VCF from the same PGS tables without the narrative report. | `genomics/prs/`, `genomics/` |
| `vcf_qa.py` | Answers only the disease named in the question, using that sample's PRS and graph traits. | `genomics/ukb/UKB_PRS_result_zscore.csv` |
| `variant_prioritization_service.py` | Runs snpEff, maps variants to genes, and calls the prioritization pipeline. | `snpeff/`, `variant_prioritization/`, `genomics/ukb/variant_prioritization_results/` |
| `gene_mas_report.py` | Summarizes functions and traits for a batch of hit genes with two model calls and direct graph queries. | Knowledge graph only |
| `SNP_report.py` | Tests an SNP list for disease overlap, then writes genetic, cellular, and imaging-hypothesis sections. | `singlecell/stab2_hs_anno.h5ad` as the reference atlas |
| `run_structured_risk.py` | Asks the model for a 0–1 confidence from the disease z-score and all HIGH-impact genes. | `genomics/ukb/Metadata.csv`, `UKB_PRS_result_zscore.csv`, `variant_prioritization_results/` |
| `run_disease_risk.py` | Queries gene–GO and gene–trait rows for those HIGH-impact genes, then asks for a second confidence. | Same UKB files, plus the Neo4j environment variables |

Run both from the repository root. A small structured run needs a sample list at `cases/genomics/output/sample_ids.txt`. `--full` uses every sample that has a variant report.

```bash
python cases/genomics/run_structured_risk.py --diseases AD --full
python cases/genomics/run_disease_risk.py --diseases AD
```

`stab2_hs_anno.h5ad` is not the subject's own single-cell experiment. `SNP_report.py` uses its `cell_type`, `region`, and `region_fine` labels to see where the hit genes are expressed.
