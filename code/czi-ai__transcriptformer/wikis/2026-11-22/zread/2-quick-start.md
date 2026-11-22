

Get TranscriptFormer up and running in minutes — from installation to your first cell embeddings. This page walks you through the three essential stages: **install the package**, **download a model**, and **run inference** on your single-cell data. By the end, you will have produced an AnnData file containing context-aware embeddings ready for downstream analysis.

![TranscriptFormer Overview](https://github.com/czi-ai/transcriptformer/blob/main/assets/model_overview.png?raw=true)

Sources: [README.md](/README.md#L1-L8), [README.md](/README.md#L15-L32)

---

## Prerequisites

Before you begin, ensure your environment meets these requirements:

| Requirement | Detail |
|---|---|
| **Python** | ≥ 3.11 (3.11 or 3.12 supported) |
| **Package manager** | [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip |
| **GPU** | NVIDIA GPU recommended (A100 40GB ideal); 16 GB VRAM works with batch size 1–4 |
| **PyTorch** | ≤ 2.5.1 (2.6.0+ may cause pickle errors) |

> [!TIP]
> On systems with limited VRAM, set `--batch-size 1` or `--batch-size 4` to avoid out-of-memory errors. You can also use the `--oom-dataloader` flag for datasets that don't fit in memory.

Sources: [README.md](/README.md#L35-L86), [pyproject.toml](/pyproject.toml#L12-L34)

---

## Installation

TranscriptFormer can be installed from PyPI or from source. The `transcriptformer` CLI becomes available after installation.

### Install from PyPI (Recommended)

```bash
# Create and activate a virtual environment
uv venv --python=3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from PyPI
uv pip install transcriptformer
```

### Install from Source

Choose this path if you plan to contribute or need the latest unreleased changes.

```bash
# Clone the repository
git clone https://github.com/czi-ai/transcriptformer.git
cd transcriptformer

# Create and activate a virtual environment
uv venv --python=3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development (editable) mode
uv pip install -e .
```

### Verify Installation

```bash
transcriptformer --help
```

You should see three subcommands: `inference`, `download`, and `download-data`.

Sources: [README.md](/README.md#L35-L63), [pyproject.toml](/pyproject.toml#L6-L55)

---

## Workflow Overview

The quickest path from zero to embeddings follows three sequential steps:

```mermaid
flowchart LR
    A["📦 Install\ntranscriptformer"] --> B["⬇️ Download\nModel Weights"]
    B --> C["🧬 Run\nInference"]
    C --> D["📊 Embeddings\nin .h5ad"]

    style A fill:#6b3fa0,color:#fff
    style B fill:#7e4fbf,color:#fff
    style C fill:#9168d9,color:#fff
    style D fill:#a98ef0,color:#fff
```

Each step maps to a single CLI command — no Python scripting required for the standard workflow.

---

## Step 1 — Download Model Weights

TranscriptFormer ships three model variants, each pretrained on a different scope of species. Choose the one that matches your analysis needs:

| Model | Species Scope | Trainable Params | Vocab Size | Use When |
|---|---|---|---|---|
| **TF-Sapiens** | Human only | 368 M | 23,829 | You only need human cell embeddings |
| **TF-Exemplar** | Human + 4 model organisms | 542 M | 110,290 | You need cross-species analysis across common model organisms |
| **TF-Metazoa** | 12 species (broadest) | 444 M | 247,388 | You need the widest evolutionary coverage including invertebrates |

Run the download command for your chosen model:

```bash
# Download a specific model
transcriptformer download tf-sapiens
# Or: transcriptformer download tf-exemplar
# Or: transcriptformer download tf-metazoa

# Download all models and extra species embeddings at once
transcriptformer download all

# Download only the extra protein embeddings (for out-of-distribution species)
transcriptformer download all-embeddings

# Specify a custom checkpoint directory (default: ./checkpoints)
transcriptformer download tf-sapiens --checkpoint-dir /path/to/custom/dir
```

After download, the checkpoint directory has this structure:

```
checkpoints/
├── tf_sapiens/
│   ├── model_weights.pt      # Trained model weights
│   ├── config.json           # Model architecture config
│   └── vocabs/               # Gene vocabulary files
├── tf_exemplar/
├── tf_metazoa/
└── all_embeddings/           # Protein embeddings for 24 species
    ├── mus_musculus_gene.h5
    ├── rattus_norvegicus_gene.h5
    └── ...
```

> [!TIP]
> Models come bundled with protein embeddings for their training species. If you need to run inference on a species *not* in the training set (e.g., rat with TF-Sapiens), also download `all-embeddings` or provide a custom embedding file via `--pretrained-embedding`.

Sources: [README.md](/README.md#L92-L151), [src/transcriptformer/cli/download_artifacts.py](/src/transcriptformer/cli/download_artifacts.py#L64-L122)

---

## Step 2 — Prepare Your Input Data

TranscriptFormer accepts **H5AD files** (the standard AnnData format) with minimal preparation:

| Requirement | Details |
|---|---|
| **Gene identifiers** | `adata.var` must contain an `ensembl_id` column with Ensembl gene IDs (e.g., `ENSG00000000003`). Changeable via `--gene-col-name`. |
| **Expression data** | **Raw (unnormalized) count data** — the model handles all transformations internally. It reads `adata.raw.X` first, then falls back to `adata.X`. |
| **Count clipping** | Counts are clipped at 30 by default (matching training). Adjustable via `--clip-counts`. |
| **Vocabulary filtering** | Out-of-vocabulary genes are automatically filtered. Only genes in the model's vocabulary are used. |

**No additional normalization, log-transformation, or scaling is needed.** The model processes raw counts as-is.

For experimenting, the repository ships small validation datasets you can use immediately:

```
test/data/
├── human_val.h5ad
├── mouse_val.h5ad
└── chicken_val.h5ad
```

Sources: [README.md](/README.md#L303-L341)

---

## Step 3 — Run Inference

With a model downloaded and an H5AD file ready, run inference with a single command:

### Basic Inference (In-Distribution Species)

```bash
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/human_val.h5ad \
  --output-path ./inference_results \
  --batch-size 8
```

### Out-of-Distribution Species

To run a model on a species it was *not* trained on, supply the protein embedding file:

```bash
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/mouse_val.h5ad \
  --pretrained-embedding ./checkpoints/all_embeddings/mus_musculus_gene.h5 \
  --batch-size 8
```

### Extract Contextual Gene Embeddings

By default the CLI produces **cell embeddings** (mean-pooled across all genes). To get per-gene contextual embeddings instead, switch the embedding type:

```bash
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/human_val.h5ad \
  --emb-type cge \
  --batch-size 8
```

### Key CLI Options at a Glance

| Flag | Default | Purpose |
|---|---|---|
| `--checkpoint-path` | *(required)* | Path to the model checkpoint directory |
| `--data-file` | *(required)* | Path to input H5AD file |
| `--output-path` | `./inference_results` | Directory for saving results |
| `--output-filename` | `embeddings.h5ad` | Name of the output file |
| `--batch-size` | `8` | Samples per batch (reduce for low VRAM) |
| `--emb-type` | `cell` | `cell` = mean-pooled embeddings; `cge` = contextual gene embeddings |
| `--pretrained-embedding` | `None` | Path to protein embeddings for OOD species |
| `--precision` | `16-mixed` | Numerical precision (`16-mixed` or `32`) |
| `--num-gpus` | `1` | Number of GPUs (`-1` for all available) |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, or `mps` |
| `--oom-dataloader` | `False` | Enable OOM-safe data loading for large datasets |
| `--config-override` | — | Override any config value (e.g., `model.data_config.normalize_to_scale=10000`) |

Sources: [README.md](/README.md#L215-L301), [src/transcriptformer/cli/__init__.py](/src/transcriptformer/cli/__init__.py#L93-L221)

---

## Understanding Your Output

Inference produces an AnnData file (default: `./inference_results/embeddings.h5ad`) with the following structure:

**Cell embeddings** (`--emb-type cell`, default):

| Key | Location | Shape | Description |
|---|---|---|---|
| `obsm['embeddings']` | AnnData object | `(n_cells, embedding_dim)` | Mean-pooled cell representations |
| `obs` | AnnData object | — | Original cell metadata preserved |
| `uns['llh']` | AnnData object | — | Log-likelihood scores (if available) |

**Contextual gene embeddings** (`--emb-type cge`):

| Key | Location | Shape | Description |
|---|---|---|---|
| `uns['cge_embeddings']` | AnnData object | `(n_gene_instances, embedding_dim)` | Per-gene contextual representations |
| `uns['cge_cell_indices']` | AnnData object | `(n_gene_instances,)` | Cell index each gene embedding belongs to |
| `uns['cge_gene_names']` | AnnData object | `(n_gene_instances,)` | Gene name for each embedding |

### Quick Post-Inference Access

```python
import anndata as ad

# Load cell embeddings
adata = ad.read_h5ad("./inference_results/embeddings.h5ad")
cell_embeddings = adata.obsm['embeddings']  # (n_cells, embedding_dim)

# Load contextual gene embeddings
cge = ad.read_h5ad("./inference_results/cge_embeddings.h5ad")
gene_embs = cge.uns['cge_embeddings']       # (n_gene_instances, embedding_dim)
cell_idx = cge.uns['cge_cell_indices']       # which cell each gene belongs to
gene_names = cge.uns['cge_gene_names']       # gene identifier strings
```

Sources: [README.md](/README.md#L343-L397)

---

## Troubleshooting Common Issues

| Problem | Cause | Solution |
|---|---|---|
| `pickle` / `Unpickling` error | PyTorch ≥ 2.6.0 | Downgrade: `uv pip install torch==2.5.1` |
| CUDA out-of-memory | Batch size too large for GPU | Reduce `--batch-size` to 1–4 |
| Dataset too large for RAM | Full H5AD loaded into memory | Add `--oom-dataloader --n-data-workers 4` |
| Low gene overlap warning | Many genes filtered from vocabulary | Check that `adata.var` contains correct Ensembl IDs |
| Slow CPU inference | Block mask compilation overhead | Add `--disable-compile-block-mask` when using `--device cpu` |

Sources: [README.md](/README.md#L66-L86), [src/transcriptformer/cli/__init__.py](/src/transcriptformer/cli/__init__.py#L194-L210)

---

## Where to Go Next

Now that you can install, download, and run TranscriptFormer, explore deeper based on your goals:

- **Understand the models** → [Model Variants and Species Coverage](3-model-variants-and-species-coverage)
- **Learn how the architecture works** → [Architecture Overview](4-architecture-overview)
- **Customize inference for your use case** → [CLI Inference Commands](13-cli-inference-commands)
- **Work with cell or gene embeddings** → [Cell and Contextual Gene Embeddings](14-cell-and-contextual-gene-embeddings)
- **Scale to large datasets or multi-GPU** → [Multi-GPU and OOM-Safe Inference](15-multi-gpu-and-oom-safe-inference)
- **Tune every configuration parameter** → [Configuration Reference](16-configuration-reference)