# BMFM-RNA zero-shot on PBMC 3k and kidney cancer T cells

## Question
I wanted to learn how a single-cell foundation model works by running one myself on public data and looking at what it returns.

What goes into IBM's BMFM-RNA single-cell foundation model, what comes out, and how do its zero-shot outputs compare with standard cell-type labels on a small public dataset?

Second part: on T cells from a kidney cancer patient with T-cell receptor (TCR) data, do cells from the same expanded clone sit close together in the model's embedding? The model never sees the TCR. Cells of one expanded clone share an antigen and often a cell state, so if the embedding groups them, it captures biology finer than broad cell type. See [Part 2](#part-2-kidney-cancer-t-cells-with-tcr-data).

A foundation model is a large neural network trained once on a lot of data, then reused for other tasks. Zero-shot means I used the released model as is, with no training or fine-tuning on my data.

## Data
- 10x Genomics "3k PBMCs from a Healthy Donor" (peripheral blood mononuclear cells, Chromium v1). Public 10x dataset, no GEO accession. Downloaded with `scanpy.datasets.pbmc3k()`.
- Reference labels: the 8 cell types from the Scanpy clustering tutorial, which reproduces the Seurat tutorial (`scanpy.datasets.pbmc3k_processed()`), for the 2,638 cells that pass quality control. These labels come from clustering plus marker genes, not from an independent ground truth.
- Model: [ibm-research/biomed.rna.bert.110m.wced.multitask.v1](https://huggingface.co/ibm-research/biomed.rna.bert.110m.wced.multitask.v1), code from [BiomedSciAI/biomed-multi-omic](https://github.com/BiomedSciAI/biomed-multi-omic), paper: [arXiv 2506.14861](https://arxiv.org/abs/2506.14861).

## How to run
Needs Linux, an NVIDIA GPU, [uv](https://github.com/astral-sh/uv), and build tools for the `hic-straw` dependency:

```bash
sudo apt install build-essential libcurl4-openssl-dev zlib1g-dev
git clone https://github.com/BiomedSciAI/biomed-multi-omic.git
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -e ./biomed-multi-omic
```

Then:

```bash
python prepare_data.py
HF_HOME=$PWD/hf_cache bmfm-targets-run --config-dir $PWD/biomed-multi-omic/run -cn predict input_file=$PWD/data/pbmc3k_raw_labeled.h5ad working_dir=$PWD/outputs/zero_shot checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1
python compare.py
```

The `--config-dir` flag is not in the package README; without it the command can't find the `predict` config.

Tested with Python 3.12, PyTorch 2.15 (CUDA 13.2) on an RTX 5070 Ti. The prediction step took about 30 seconds for 2,638 cells.

## What the model takes and returns
Input: an `.h5ad` file with raw counts and gene symbols. The model keeps protein-coding genes (17,569 of 32,738 here), log-normalizes the counts, and turns each cell into a list of up to 1,024 tokens, each a gene with its expression value. A token is one unit of input, like a word in a sentence.

The model is a transformer (a type of neural network first built for text) with BERT-style layers. It was pre-trained on CELLxGENE data with two kinds of objectives:
- WCED (whole-cell expression decoding): the model sees part of a cell's genes, compresses the cell into one vector, and has to reconstruct whether each gene is detected and how much, for all genes.
- Multitask: predict cell type, tissue, general tissue and donor ID from CELLxGENE metadata.

Output, per cell:
- An embedding: 768 numbers that summarize the cell, used like principal component scores.
- A predicted cell type, tissue, general tissue and donor ID, with a probability for each of the model's labels (794 cell types, 365 tissues, 67 general tissues, 7,735 donors). Labels follow the Cell Ontology, not the tutorial's names.

## Result
Predicted cell type vs reference label (`results/crosstab_reference_vs_predicted.csv`):
- B cells: 327 of 342 "B cell".
- NK cells: 145 of 154 "natural killer cell".
- CD14+ monocytes: 467 of 480 a monocyte label ("monocyte", "classical monocyte", "CD14-positive monocyte", "non-classical monocyte").
- FCGR3A+ monocytes: all 150 a monocyte label, 36 of them "non-classical monocyte".
- Dendritic cells: 32 of 37 "dendritic cell" or "conventional dendritic cell".
- Megakaryocytes: 11 of 15 "platelet".
- CD4 T cells: 1,053 of 1,144 just "T cell".
- CD8 T cells: 264 of 316 just "T cell", and 39 "natural killer cell".

Tissue: 1,965 of 2,638 cells were predicted as "carotid artery segment" and 482 as "blood".

Do the embeddings separate the reference cell types? For each cell I took its 15 nearest neighbours and counted the share with the same reference label (`results/knn_agreement.csv`):
- BMFM-RNA embedding: 0.906.
- PCA on the same data (log-normalized, 2,000 highly variable genes, 50 components): 0.913.
- By cell type the two are close, except dendritic cells (0.87 vs 0.71).

UMAP of the embeddings, coloured by reference label (left) and predicted cell type (right):

![UMAP](https://github.com/tom-snir/bmfm-rna-zero-shot/raw/HEAD/results/umap_bmfm_embedding.png)

## Write-up
Without seeing this dataset before, the model put B cells, NK cells, monocytes and dendritic cells in the right broad groups, and the embedding separates them into clear islands. It did not split CD4 from CD8 T cells; most T cells got the parent label "T cell". The main disagreement was 39 CD8 T cells called NK cells. These two cell types share cytotoxic genes, so the mix-up makes biological sense.

The embedding separated the reference cell types about as well as PCA. This comparison favours PCA, because the reference labels were made by clustering on PCA of these same cells.

The tissue prediction did not transfer: most blood cells were called artery. My guess is that the training data holds a large immune-cell dataset from carotid artery plaques, but I did not check this. The donor prediction names a training donor, which has no meaning for new data.

Limits: one small dataset from one donor, reference labels that are not independent, and the model's labels are at a different level of detail than the reference, which I did not map to a shared set.

## Files
- `prepare_data.py`: downloads the data and builds the labelled raw-count `.h5ad`.
- `compare.py`: compares predictions with the reference, runs the neighbour test, makes the UMAP and saves CSVs.
- `results/cells_predictions.csv`: per cell, reference label, predictions and cell-type probability.
- `results/embeddings.csv`: per cell, the 768 embedding values. Not in the repo (22 MB); `compare.py` creates it.
- `prepare_wu_renal1.py`, `compare_wu_renal1.py`: the same for Part 2.

## Part 2: kidney cancer T cells with TCR data

### Data
- Wu et al. 2020, Nature, "Peripheral T cell expansion predicts tumour infiltration and clinical response". GEO [GSE139555](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139555).
- One patient (Renal1): T cells from tumour (GSM4143678), adjacent normal kidney (GSM4143679) and blood (GSM4143680). 10x single-cell RNA plus TCR from the same cells.
- I kept the 6,276 T cells in the authors' metadata (`GSE139555_tcell_metadata.txt.gz`), with their cluster labels, tissue source and clone IDs. Clone size is the number of this patient's cells that share a clone ID. 3,603 cells have a TCR; 2,322 are in clones of 2 or more cells.

### How to run
Download the three samples' `barcodes.tsv.gz`, `genes.tsv.gz` and `matrix.mtx.gz` files, plus `GSE139555_tcell_metadata.txt.gz`, from GEO into `data/wu_renal1/downloads/`. Then:

```bash
python prepare_wu_renal1.py
HF_HOME=$PWD/hf_cache bmfm-targets-run --config-dir $PWD/biomed-multi-omic/run -cn predict input_file=$PWD/data/wu_renal1/renal1_raw_labeled.h5ad working_dir=$PWD/outputs/wu_renal1 checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1
python compare_wu_renal1.py
```

### Result
Predicted cell type vs the authors' clusters:
- CD4 vs CD8 mostly agreed: authors' naive-like CD4 cluster (4.3-TCF7) 313 CD4 vs 108 CD8; Tregs 124 CD4 vs 3 CD8; CD8 effectors (8.1-Teff) 801 CD8 vs 13 CD4.
- 1,786 of 6,276 cells (28%) were predicted "unknown", across all clusters.
- Many CD8 cells were called NK cells, most in one tissue-resident cluster (8.3b-Trm: 372 of 599).

Predicted tissue vs actual source:
- Blood: 988 of 1,210 blood cells predicted "blood".
- Kidney: 9 cells in total. Tumour and normal kidney T cells were mostly predicted as blood, brain, breast or lung.

Clones: for each cell in a clone of 2 or more, the share of its 15 nearest neighbours from the same clone (mean over 2,322 cells):
- BMFM-RNA embedding: 0.088.
- PCA on the same data: 0.082.
- Expected if neighbours were random cells from the same patient: 0.013; from the same cluster: 0.028; from the same cluster and tissue source: 0.036.
- Clones of 20+ cells: embedding 0.144, PCA 0.135, expected from same cluster and source 0.059.

UMAP of the embeddings, coloured by tissue source, authors' cluster, and the 10 largest clones:

![UMAP Renal1](https://github.com/tom-snir/bmfm-rna-zero-shot/raw/HEAD/results/wu_renal1/umap_renal1_bmfm_embedding.png)

### Write-up
On these T cells the model separated CD4 from CD8, which it did not do on PBMC 3k. As with PBMC, it called many cytotoxic CD8 T cells NK cells. About a quarter of the cells got "unknown", which is a label in the CELLxGENE training data for unannotated cells.

The tissue prediction recognised blood but not kidney.

Cells from the same expanded clone were neighbours in the embedding about 2.4 times more often than expected from their cluster and tissue source alone, and more so for large clones. PCA gave almost the same result, so this reflects the cells' expression profiles rather than something only the foundation model picks up.

Limits: one patient. Each tissue is a separate 10x sample, so tissue and batch are mixed together. The authors' clusters come from their own integrated analysis.
