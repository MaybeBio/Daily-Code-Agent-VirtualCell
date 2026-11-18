# Cases

Three downstream uses of BrainTwin. The cross-scale report is not included.

- `genomics/`: VCF and SNP questions, polygenic scores, variant prioritization, gene summaries, and the UK Biobank disease-risk runs in `run_structured_risk.py` and `run_disease_risk.py`.
- `singlecell/`: cluster a sample h5ad file and annotate the clusters. `disease_pipeline/` runs the author-cell-type differential test, Enrichr, fixed graph queries, and the written inference.
- `mri/`: read one person's structured FreeSurfer record and answer a question about those measurements. `cohort_questions.py` asks the same session over a UCLA or ADNI cohort.

Each folder has its own notes. Shared path setup is `cases/_paths.py`. Model and graph endpoints come from `BRAINTWIN_LLM_URL`, `BRAINTWIN_MODEL_ID`, `BRAINTWIN_NEO4J_URL`, and `BRAINTWIN_NEO4J_PASSWORD`. Input files are listed in `DATA.md`.
