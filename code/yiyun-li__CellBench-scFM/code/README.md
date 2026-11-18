# CellBench: Frozen-Backbone Benchmark of Single-Cell Foundation Models

Official benchmark codebase, model adapters, downstream evaluation heads, statistical analysis scripts, and dataset/checkpoint manifests for **CellBench**:

> **What Frozen Single-Cell Foundation Models Preserve, Miss, and Leak**

CellBench evaluates **13 publicly released single-cell foundation model (scFM) checkpoints** in frozen-backbone mode alongside **3 leak-free training-split-fitted baselines** (`PCA`, `UMAP`, and `scVI`) across:
- **12 static tissue and disease cohorts** spanning healthy human organs, human diseases, and matched mouse organs (**2,304 representation runs** across unsupervised clustering, native batch assessment, supervised cell-type annotation, and HVG expression reconstruction);
- **4 CRISPR Perturb-seq screens** (**192 perturbation runs** across unseen two-gene combinations and unseen single-gene knockouts, anchored to zero-change baseline skill $\hat{\Delta}_g = \mathbf{0}$);
- **Direct frozen-embedding linear probes** for rare (<1%) subtypes and continuous cellular programs, **instantiated checkpoint tensor accounting**, and a **208-pair pretraining provenance and overlap audit**.

---

## Evaluated Models & Pretrained Checkpoint Links (13 scFMs + 3 Baselines)

Machine-readable table: [`manifests/models.csv`](https://github.com/yiyun-li/CellBench-scFM/blob/HEAD/manifests/models.csv) (detailed download hashes and asset records in [`data/provenance/`](https://github.com/yiyun-li/CellBench-scFM/blob/HEAD/data/provenance) and [`public_code/sources.json`](https://github.com/yiyun-li/CellBench-scFM/blob/HEAD/public_code/sources.json)).

| Method | Category | Audited Params | Width ($d$) | Checkpoint | Official Code & Commit | Paper | Pretrained Weights |
| :--- | :--- | ---: | ---: | :--- | :--- | :--- | :--- |
| **scFoundation** | Foundation model | 119.3M | 3072 | `models.ckpt` | [Repo](https://github.com/biomap-research/scFoundation) (`397631c495ed`) | [Paper](https://www.nature.com/articles/s41592-024-02305-7) | [Weights](https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk) |
| **UCE** | Foundation model | 848.9M | 1280 | `4layer_model.torch` | [Repo](https://github.com/snap-stanford/UCE) (`9c416007be15`) | [Paper](https://doi.org/10.1038/s41586-026-10689-z) | [Weights](https://ndownloader.figshare.com/files/42706576) |
| **TranscriptFormer** | Foundation model | 429.9M | 2048 | `tf_sapiens/model_weights.pt` | [Repo](https://github.com/czi-ai/transcriptformer) (`c943a89a4de9`) | [Paper](https://github.com/czi-ai/transcriptformer) | [Weights](https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz) |
| **scGPT** | Foundation model | 50.8M | 512 | `human/best_model.pt` | [Repo](https://github.com/bowang-lab/scGPT) (`cebd6fae655b`) | [Paper](https://www.nature.com/articles/s41592-024-02201-0) | [Weights](https://drive.google.com/uc?id=14AebJfGOUF047Eg40hk57HCtrb0fyDTm) |
| **Nicheformer** | Foundation model | 38.5M | 512 | `model.safetensors` | [Repo](https://github.com/theislab/nicheformer) (`485cadbc5caa`) | [Paper](https://www.nature.com/articles/s41592-025-02814-z) | [Weights](https://huggingface.co/theislab/Nicheformer/resolve/0ba4ba162f9ad3d6c8b713761954c31be2fa69ed/model.safetensors) |
| **LangCell** | Foundation model | 39.4M | 256 | `cell_bert/pytorch_model.bin + cell_proj.bin` | [Repo](https://github.com/PharMolix/LangCell) (`69e41ef2fae4`) | [Paper](https://proceedings.mlr.press/v235/zhao24u.html) | [Weights](https://drive.google.com/uc?id=119ic1oiZr6Lxa5Bguv240L-OL5lPGEAq) |
| **scPRINT2** | Foundation model | 101.9M | 424 | `small-v2.ckpt` | [Repo](https://github.com/cantinilab/scPRINT-2) (`db0ef5f1fd12`) | [Paper](https://github.com/cantinilab/scPRINT-2) | [Weights](https://huggingface.co/jkobject/scPRINT/resolve/8e1ad1a1bc40c1fc2f4f1700fe27923eea9a2bed/small-v2.ckpt) |
| **CellPLM** | Foundation model | 65.8M | 512 | `20230926_85M.best.ckpt` | [Repo](https://github.com/OmicsML/CellPLM) (`b59ee7688b1b`) | [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/173bcdbd820edbe43284e0941109ada9-Paper-Conference.pdf) | [Weights](https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&dl=1) |
| **Geneformer** | Foundation model | 39.6M | 512 | `gf-12L-30M-i2048/pytorch_model.bin` | [Repo](https://huggingface.co/ctheodoris/Geneformer) (`31bf641d0f17`) | [Paper](https://www.nature.com/articles/s41586-023-06139-9) | [Weights](https://huggingface.co/ctheodoris/Geneformer/resolve/31bf641d0f173a18a5e06d9a6574bb0d3fe5edc1/gf-12L-30M-i2048/pytorch_model.bin) |
| **GeneCompass** | Foundation model | 18.1M | 256 | `GeneCompass_Small/pytorch_model.bin` | [Repo](https://github.com/xCompass-AI/GeneCompass) (`59e5e485cb94`) | [Paper](https://www.nature.com/articles/s41422-024-01034-y) | [Weights](https://www.scidb.cn/en/anonymous/SUZOdk1y) |
| **CellFM** | Foundation model | 797.4M | 1536 | `base_weight.ckpt` | [Repo](https://github.com/biomed-AI/CellFM) (`72c9f4a9580a`) | [Paper](https://www.nature.com/articles/s41467-025-59926-5) | [Weights](https://huggingface.co/ShangguanNingyuan/CellFM/resolve/69361b50d1814ade0e3749c9d8eb829590c2ce3c/base_weight.ckpt) |
| **scLong** | Foundation model | 1,006.1M | 200 | `gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth` | [Repo](https://github.com/BaiDing1234/scLong) (`41b720215409`) | [Paper](https://www.nature.com/articles/s41467-026-69102-y) | [Weights](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EpvKzQW4hI5Bnb88-iM7vE0B_e2_U5r_ZGXb_FILCLTw3Q?e=TAmKk5) |
| **Tabula** | Foundation model | 13.2M | 192 | `pancreas.pth` | [Repo](https://github.com/aristoteleo/tabula) (`65b5f7ebf36a`) | [Paper](https://papers.nips.cc/paper_files/paper/2025/hash/95d590995a8722259c61e094b62b25ac-Abstract-Conference.html) | [Weights](https://drive.google.com/uc?id=1xvUCfbi3K0Gn7iCniYg2S4_FwzwWk6St) |
| **PCA** | Dataset-fitted baseline | Dataset-fitted | 256 | `Fitted per donor training split` | [Repo](https://github.com/scikit-learn/scikit-learn) (`scikit-learn`) | [Paper](https://jmlr.org/papers/v12/pedregosa11a.html) | Fitted on train split |
| **UMAP** | Dataset-fitted baseline | Dataset-fitted | 256 | `Fitted per donor training split` | [Repo](https://github.com/lmcinnes/umap) (`umap-learn`) | [Paper](https://doi.org/10.21105/joss.00861) | Fitted on train split |
| **scVI** | Dataset-fitted baseline | Dataset-fitted | 256 | `Fitted per donor training split` | [Repo](https://github.com/scverse/scvi-tools) (`scvi-tools`) | [Paper](https://www.nature.com/articles/s41592-018-0229-2) | Fitted on train split |

---

## Evaluation Datasets & Download Links (16 Public Cohorts)

Machine-readable table: [`manifests/datasets.csv`](https://github.com/yiyun-li/CellBench-scFM/blob/HEAD/manifests/datasets.csv) (exact preparation records and MD5/version IDs in [`data/dataset_assets.json`](https://github.com/yiyun-li/CellBench-scFM/blob/HEAD/data/dataset_assets.json) and [`data/provenance/`](https://github.com/yiyun-li/CellBench-scFM/blob/HEAD/data/provenance)).

| Panel | Code | Dataset ID | Cells | Genes | Cohort Description | Portal / Collection | Direct Download |
| :--- | :--- | :--- | ---: | ---: | :--- | :--- | :--- |
| Human organs | `BM` | `bmmc` | 30,000 | 13,953 | Human bone marrow mononuclear cells (NeurIPS 2021 CITE-seq RNA) | [Portal](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122) | [Download](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz) |
| Human organs | `HL` | `hlca` | 30,000 | 27,402 | Human Lung Cell Atlas (core) | [Portal](https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293) | [Download](https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad) |
| Human organs | `AH` | `heart` | 30,000 | 32,383 | Cells of the adult human heart | [Portal](https://cellxgene.cziscience.com/collections/b52eb423-5d0d-4645-b217-e1c6d38b2e72) | [Download](https://datasets.cellxgene.cziscience.com/49312f8c-a75a-4672-9c8a-9078acd35bdb.h5ad) |
| Human organs | `AK` | `kidney` | 30,000 | 26,837 | Single-cell RNA-seq of the adult human kidney | [Portal](https://cellxgene.cziscience.com/collections/bcb61471-2a44-4d00-a0af-ff085512674c) | [Download](https://datasets.cellxgene.cziscience.com/3ae68468-89f7-4941-a945-f7bae9b2cbea.h5ad) |
| Human diseases | `MS` | `ms` | 17,799 | 20,110 | Multiple sclerosis brain white matter (Jakel et al.) | [Portal](https://cellxgene.cziscience.com/collections/16c1e722-96ae-4bf6-b408-cd7f8918484f) | [Download](https://datasets.cellxgene.cziscience.com/8bbea7c5-c742-4c27-a597-0f1e1383bb48.h5ad) |
| Human diseases | `UC` | `uc_epi` | 30,000 | 20,028 | Ulcerative colitis colon epithelium (Smillie et al.) | [Portal](https://singlecell.broadinstitute.org/single_cell/study/SCP259/intra-and-inter-cellular-rewiring-of-the-human-colon-during-ulcerative-colitis) | [Download](https://singlecell.broadinstitute.org/single_cell/study/SCP259) |
| Human diseases | `C19` | `covid_lung` | 30,000 | 33,523 | Lethal COVID-19 lung atlas (Melms et al.) | [Portal](https://cellxgene.cziscience.com/collections/e4c9ed14-e560-4900-a3bf-b0f8d2ce6a10) | [Download](https://datasets.cellxgene.cziscience.com/1792df55-7cfe-4564-afcf-4bb3ffb73bc0.h5ad) |
| Human diseases | `BC` | `tnbc` | 30,000 | 32,354 | Untreated triple-negative breast cancer atlas | [Portal](https://cellxgene.cziscience.com/collections/ceef2841-5333-46ac-92ef-ccbe0c20fe55) | [Download](https://datasets.cellxgene.cziscience.com/e94bd3cc-6271-424a-baac-12f8eb320a0e.h5ad) |
| Mouse organs | `MM` | `mouse_marrow` | 30,000 | 17,943 | Tabula Muris Senis bone marrow (10x) | [Portal](https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb) | [Download](https://datasets.cellxgene.cziscience.com/88dcbe63-d6eb-4fae-9362-e21b98365a42.h5ad) |
| Mouse organs | `ML` | `mouse_lung` | 24,540 | 17,943 | Tabula Muris Senis lung (10x) | [Portal](https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb) | [Download](https://datasets.cellxgene.cziscience.com/e67a08e2-96fc-4244-b8a3-fca092b22f77.h5ad) |
| Mouse organs | `MH` | `mouse_heart` | 18,282 | 16,669 | Tabula Muris Senis heart (10x + Smart-seq2 merged) | [Portal](https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb) | [10x h5ad](https://datasets.cellxgene.cziscience.com/951e10f3-050f-4059-abc3-5adc360b777b.h5ad) / [SS2 h5ad](https://datasets.cellxgene.cziscience.com/dcda4e26-53aa-47cd-9be1-3dea00e8f902.h5ad) |
| Mouse organs | `MK` | `mouse_kidney` | 21,647 | 17,943 | Tabula Muris Senis kidney (10x) | [Portal](https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb) | [Download](https://datasets.cellxgene.cziscience.com/cfd31d35-ceb2-4990-b0b5-3434ecbb7967.h5ad) |
| Perturbation | `Norman` | `norman` | 30,000 | 33,694 | Norman et al. 2019 K562 single- and two-gene CRISPRa Perturb-seq | [Portal](https://zenodo.org/records/13350497) | [Download](https://zenodo.org/api/records/13350497/files/NormanWeissman2019_filtered.h5ad/content) |
| Perturbation | `Adamson` | `adamson` | 30,000 | 32,738 | Adamson et al. 2016 K562 UPR single-gene Perturb-seq | [Portal](https://zenodo.org/records/13350497) | [Download](https://zenodo.org/api/records/13350497/files/AdamsonWeissman2016_GSM2406681_10X010.h5ad/content) |
| Perturbation | `Replogle K562` | `replogleK562` | 30,000 | 8,563 | Replogle et al. 2022 K562 essential-gene Perturb-seq | [Portal](https://zenodo.org/records/13350497) | [Download](https://zenodo.org/api/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad/content) |
| Perturbation | `Replogle RPE1` | `replogleRPE1` | 30,000 | 8,749 | Replogle et al. 2022 RPE1 essential-gene Perturb-seq | [Portal](https://zenodo.org/records/13350497) | [Download](https://zenodo.org/api/records/13350497/files/ReplogleWeissman2022_rpe1.h5ad/content) |

---

## Repository Structure

- **`bio/`** — Standardized frozen-backbone model adapters for all 13 scFMs and 3 training-split-fitted baselines, ortholog mappers, and automated dataset/weight downloaders (`bio/assets/datasets.py`, `bio/assets/weights.py`, `bio/assets/frontier.py`, `bio/assets/prepare_datasets.py`).
- **`cell_eval/`** — Unified downstream evaluation engine: fixed MLP probes for annotation and HVG reconstruction, Leiden clustering, native `scIB` batch assessment, paired-control Perturb-seq residual lookup decoding, and evaluation metrics.
- **`configs/`** — Per-model YAML execution and preprocessing configurations (`cellfm.yaml`, `cellplm.yaml`, `genecompass.yaml`, `geneformer.yaml`, `langcell.yaml`, `nicheformer.yaml`, `pca.yaml`, `scfoundation.yaml`, `scgpt.yaml`, `sclong.yaml`, `scprint2.yaml`, `scvi.yaml`, `tabula.yaml`, `transcriptformer.yaml`, `uce.yaml`, `umap.yaml`).
- **`scripts/`** — Command-line entry points for environment setup, downloading datasets and model weights, running the full benchmark, and executing direct linear-probe diagnostics:
  - `scripts/download_datasets.py`: Download and verify raw public datasets into `datasets/source_files/`.
  - `scripts/download_weights.py` & `scripts/download_frontier_weights.py`: Download pretrained checkpoints with SHA-256 verification into `weights/`.
  - `scripts/run_full_benchmark.py` & `scripts/run_batch.py`: Run multi-seed frozen-backbone extraction and downstream evaluation across cohorts and tasks.
  - `scripts/run_direct_embedding_program_probe.py`: Reproduce 5-fold Ridge linear probes for rare (<1%) subtypes and continuous cellular programs.
  - `scripts/compute_biological_information_channels.py`: Reproduce 12-cohort biological-channel profiles and concordance statistics.
- **`analysis/`** — Non-parametric rank aggregation (`task_layout.py`), checkpoint tensor parameter accounting (`checkpoint_parameter_audit.py`), leave-study-out overlap sensitivity (`overlap_sensitivity.py`, `match_scfoundation_sources.py`), Perturb-seq zero-change skill analysis (`perturbation_rq.py`), supplementary table generation (`generate_supplementary_tables.py`), and figure generation scripts.
- **`manifests/`** — Curated CSV and Markdown link indices for all 16 evaluation cohorts (`datasets.csv`) and 16 evaluated methods (`models.csv`).
- **`data/` & `results/`** — Run-level score exports (`data/exports/`), donor split counts, checkpoint parameter manifests (`data/parameter_audit/`), the 208-pair pretraining-overlap audit (`data/pretraining_overlap_audit.csv`), direct linear-probe outputs (`data/direct_embedding_program_probe.csv`), and multi-sheet Excel workbooks (`results/supplementary_tables.xlsx`, `results/perturbation_seed_results.xlsx`).
- **`resources/` & `tests/`** — Ensembl 1-to-1 human–mouse orthology tables, isolated environment lockfiles for frontier checkpoints, and unit tests.

---

## Quick Start

### 1. Environment Setup

```bash
conda env create -f environment-cellbench.yml
conda activate cellbench
pip install -r requirements-cellbench.txt

# Optional isolated environments for framework-specific checkpoints (CellFM, GeneCompass, scPRINT2, TranscriptFormer)
bash scripts/setup_frontier_environments.sh
```

### 2. Download Public Datasets & Pretrained Checkpoints

```bash
# Download public evaluation datasets and prepare standardized .h5ad cohorts
python scripts/download_datasets.py
python -m bio.assets.prepare_datasets

# Download pretrained foundation model weights and verify SHA-256 digests
python scripts/download_weights.py
python scripts/download_frontier_weights.py
python scripts/download_public_code.py
```

### 3. Run Frozen-Backbone Benchmark Evaluations

```bash
# Smoke test adapter contracts
python scripts/smoke_pipeline.py

# Full multi-seed benchmark (seeds 41, 42, 43 across 16 methods x 16 cohorts)
python scripts/run_full_benchmark.py --output-dir output/cellbench_runs
python scripts/export_metrics.py --run-dir output/cellbench_runs
```

### 4. Reproduce Diagnostic Probes, Provenance Audits, & Figures

```bash
pip install -r requirements-analysis.txt

# Direct frozen-embedding cellular-program and rare-subtype linear probes
CELLBENCH_ROOT=. python scripts/run_direct_embedding_program_probe.py
python scripts/compute_biological_information_channels.py

# Perturbation zero-change skill, overlap sensitivity, and figure generation
python analysis/perturbation_rq.py
python analysis/overlap_sensitivity.py
python analysis/redraw_main_task_figures.py
python analysis/draw_rq678_figures.py
```

---

## License

Code and benchmark scripts in this repository are released under the [MIT License](https://github.com/yiyun-li/CellBench-scFM/blob/HEAD/LICENSE). Third-party foundation model checkpoints and public transcriptomic datasets remain subject to their respective upstream licenses and terms of use.
