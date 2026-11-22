# Preregistration: Ancestry bias in single-cell foundation models

Frozen at git tag `prereg-v1`. Registered on OSF (Secondary Data Preregistration): https://doi.org/10.17605/OSF.IO/XED6M
Any departure will be reported in a "Deviations from preregistration" section with its reason.

## 1. Research questions and hypotheses
- H1 (primary): zero-shot per-donor macro-F1 is lower for AA than EA donors, for each foundation model.
- H1b (key comparison): the AA-EA gap is larger for foundation models than for the PCA and scVI baselines.
- H2: fine-tuning on ancestry-balanced data reduces the gap compared with EA-only fine-tuning.
- H3: any gap is concentrated in specific cell types.
- H4: ancestry separates cells more in foundation-model embeddings than in scVI/PCA embeddings.
Tests are two-sided. A finding of no meaningful gap will be reported with equal prominence.

## 2. Data
- Primary: Randolph et al. 2021 (GSE162632; Zenodo 10.5281/zenodo.4273999). 45 AA + 45 EA healthy males,
  mock (NI) and influenza conditions, both ancestries pooled within every 10x capture. Genetic African
  ancestry proportion (YRI) per donor.
- Replication: GSE270387 prostate cancer PBMC. 33 AA + 24 EA after QC; disease group recorded.
- Audit only: CELLxGENE Census release 2025-11-08 (4 of 2,252 healthy blood donors recorded as AA).
- Excluded before analysis: GSE189050 (lupus; 20 of 23 AA donors in separate shallower runs, median 129 vs 256
  genes/cell); Aquino et al. 2023 (controlled access).

## 3. What was done before registration
Done: data download and assembly; QC; labels; donor splits; a blinded power pilot.
- QC (all data): 200-6,000 genes/cell, <15% mitochondrial reads; Scrublet doublets (prostate); influenza genes
  removed (Randolph); prostate donors with >25% of cells failing QC excluded (2 donors, both AA). A first donor
  rule (3 MADs) was dropped before any results because it removed good-quality donors.
- Labels: Randolph coarse labels from the original authors; prostate coarse labels from clustering + canonical
  markers; fine labels from marker gating, keeping only subtypes resolvable in both cohorts. No tested method
  was used to make labels.
- Splits: results/splits.json (seed 2026 primary; 2027-2030 robustness).
- Blinded pilot: CellTypist and PCA+kNN run; only pooled per-donor macro-F1 summaries viewed
  (PCA+kNN mean 0.92-0.94, SD 0.029-0.034). No ancestry-stratified result was computed or viewed.
Not done: no foundation model has been run; no ancestry-stratified accuracy, gap or embedding metric has been seen.

## 4. Variables
- Unit of analysis: donor.
- Exposure: ancestry (AA vs EA); Randolph also continuous YRI.
- Methods (7): scGPT (whole-human), Geneformer (V2-104M), UCE (33-layer; 4-layer if a 10k-cell timing run
  projects >48 GPU-hours), scFoundation; baselines scVI, CellTypist (Immune_All_Low), PCA.
- Primary outcome: per-donor macro-F1 vs coarse labels (B, CD4_T, CD8_T, DC, Monocyte, NK), counting cell types
  with >=10 cells in that donor.
- Secondary: balanced accuracy; per-cell-type F1; fine-label macro-F1 (B_naive, B_memory, CD4_naive,
  CD4_memory, CD8_T, DC, Monocyte, NK); ancestry iLISI and silhouette (H4).
- Covariates: age (Randolph), log(cells per donor), disease group (prostate).
- Randolph primary analysis uses mock (NI) cells.

## 5. Analysis plan
- Zero-shot annotation (H1): 5 donor folds (seed 2026). kNN (k=15) trained on EA donors in the other folds
  predicts every donor in the held-out fold. AA donors are never used for training.
- Primary model, per method and dataset: F1 ~ AA + age + log(cells). Estimand: AA coefficient with 95% CI;
  two-sided alpha 0.05; Benjamini-Hochberg across 7 methods.
- Key comparison (H1b): f1 ~ AA * method + age + log(cells) + (1 | donor), PCA as reference. Confirmatory:
  Randolph + prostate pooled (+ dataset, + disease group). Supportive: Randolph alone.
- Smallest effect of interest: 2 macro-F1 points.
  - AA coef <= -0.02 and 95% CI excludes 0: meaningful gap
  - CI excludes 0 but |coef| < 0.02: detectable but negligible
  - 90% CI within +/-0.02 (TOST): equivalent, no meaningful gap
  - otherwise: inconclusive; report power
- Power (pilot SD ~0.03): H1 Randolph 0.90; prostate 0.58; H1b pooled 0.83; H1b Randolph 0.63.
  Re-reported with the observed SD.
- H2: scGPT and Geneformer fine-tuned EA-only / balanced / random, 3 seeds, fixed split (Randolph 40 test /
  50 train); f1 ~ AA * regime + (1 | donor). Hyperparameters tuned on EA validation donors only.
- H3: per-cell-type F1 differences; pseudobulk DESeq2 in the cell types with the largest gaps.
- H4: ancestry iLISI/silhouette per cell type vs scVI, paired Wilcoxon across cell types.
- Sensitivity: influenza cells; composition-matched test sets; continuous YRI; fine labels; all 59 prostate
  donors; prostate controls only and patients only; CV seeds 2027-2030; 5-MAD QC; scGPT continual-pretrained.
- Exploratory: ancestry x disease (prostate); Burkina Faso cohort (GSE273781/5); depth confounding in lupus.

## 6. Other
- Checkpoint SHA-256 hashes committed before use. A model that cannot run after 2 days of troubleshooting is
  dropped and reported.
- UCE reproducibility: 5,000 cells embedded with seeds 2026 and 2027; kNN label agreement >= 0.95 expected.
