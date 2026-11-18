# Getting Started

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.readthedocs.yaml](.readthedocs.yaml)
- [README.md](README.md)
- [docs/environment.yml](docs/environment.yml)
- [docs/requirements.txt](docs/requirements.txt)
- [pyproject.toml](pyproject.toml)
- [scgpt/data_collator.py](scgpt/data_collator.py)

</details>



This page guides you through the initial setup and first steps with scGPT, from installation to running your first cell embedding task. It covers the essential prerequisites, basic installation via pip, and a minimal working example using pretrained models.

For detailed installation instructions including optional dependencies and troubleshooting, see [Installation](#2.1). For comprehensive examples of pretrained model usage, see [Quick Start with Pretrained Models](#2.2).

---

## Prerequisites

scGPT requires the following environment:

| Component | Requirement | Purpose |
|-----------|-------------|---------|
| **Python** | >= 3.7.13 | Core language runtime |
| **R** | >= 3.6.1 | Required by some dependencies (scvi-tools, scanpy) |
| **CUDA** | 11.7+ recommended | For GPU acceleration with flash-attention |
| **GPU** | Optional but recommended | Significantly faster training and inference |

**Sources:** [README.md:30](), [pyproject.toml:12]()

---

## Basic Installation

The simplest installation uses pip to install the scGPT package from PyPI:

```bash
pip install scgpt "flash-attn<1.0.5"
```

The `flash-attn` dependency is optional but strongly recommended for performance. If you encounter installation issues with `flash-attn`, scGPT will automatically fall back to standard PyTorch attention mechanisms.

**Known issue (as of 2023.09):** If you encounter errors related to Google's `orbax` package, use:

```bash
pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
```

### Optional: Experiment Tracking

For logging training metrics and visualizations, install Weights & Biases:

```bash
pip install wandb
```

**Sources:** [README.md:28-54](), [pyproject.toml:1-41]()

---

## Installation Dependencies Flow

```mermaid
graph TB
    subgraph "Package Management"
        PIP["pip install scgpt"]
        POETRY["poetry install<br/>(deprecated)"]
        PYPROJECT["pyproject.toml<br/>Package Definition"]
    end
    
    subgraph "Core Dependencies"
        TORCH["torch >= 1.13.0<br/>Deep Learning Backend"]
        SCANPY["scanpy ^1.9.1<br/>Single-cell Analysis"]
        SCVI["scvi-tools >= 0.16.0<br/>Variational Inference"]
        TORCHTEXT["torchtext<br/>Vocabulary Management"]
        DATASETS["datasets ^2.3.0<br/>Data Loading"]
    end
    
    subgraph "Optional Dependencies"
        FLASHATTN["flash-attn < 1.0.5<br/>GPU Optimization"]
        WANDB["wandb ^0.12.3<br/>Experiment Tracking"]
        FAISS["faiss-cpu or faiss-gpu<br/>Reference Mapping"]
    end
    
    subgraph "Python Package"
        SCGPT_INIT["scgpt/__init__.py<br/>v0.2.4"]
        SCGPT_INIT --> MODEL_MODULE["scgpt.model"]
        SCGPT_INIT --> TOKENIZER_MODULE["scgpt.tokenizer"]
        SCGPT_INIT --> TASKS_MODULE["scgpt.tasks"]
        SCGPT_INIT --> TRAINER_MODULE["scgpt.trainer"]
    end
    
    PIP --> PYPROJECT
    POETRY --> PYPROJECT
    
    PYPROJECT --> TORCH
    PYPROJECT --> SCANPY
    PYPROJECT --> SCVI
    PYPROJECT --> TORCHTEXT
    PYPROJECT --> DATASETS
    
    PYPROJECT -.->|"dev dependency"| FLASHATTN
    PYPROJECT -.->|"optional"| WANDB
    
    TORCH --> SCGPT_INIT
    TORCHTEXT --> SCGPT_INIT
    SCANPY --> SCGPT_INIT
    
    FLASHATTN -.->|"if available"| MODEL_MODULE
    FAISS -.->|"if needed"| TASKS_MODULE
```

**Sources:** [pyproject.toml:11-27](), [pyproject.toml:28-36](), [README.md:28-44]()

---

## Your First scGPT Application

The most common use case is generating cell embeddings from scRNA-seq data using a pretrained model. This requires three steps: loading data, loading a pretrained model, and generating embeddings.

### Step 1: Prepare Your Data

scGPT expects data in AnnData format, commonly used in single-cell analysis:

```python
import scanpy as sc

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Basic preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
```

### Step 2: Load Pretrained Model and Vocabulary

```python
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained

# Load gene vocabulary
vocab = GeneVocab.from_file("vocab.json")

# Initialize model architecture
model = TransformerModel(
    ntoken=len(vocab),
    d_model=512,
    nhead=8,
    d_hid=512,
    nlayers=12,
    vocab=vocab,
    pad_token=vocab["<pad>"],
    pad_value=0,
)

# Load pretrained weights
load_pretrained(model, torch.load("best_model.pt", map_location="cpu"))
model.eval()
```

### Step 3: Generate Cell Embeddings

```python
from scgpt.tasks.cell_emb import embed_data

# Generate embeddings
adata = embed_data(
    model=model,
    adata=adata,
    gene_ids=adata.var.index.tolist(),
    batch_size=64,
    return_new_adata=True,
)

# Embeddings stored in adata.obsm['X_scGPT']
print(adata.obsm['X_scGPT'].shape)  # (n_cells, d_model)
```

**Sources:** [README.md:56-69](), [README.md:18]()

---

## Basic Usage Workflow: From Data to Embeddings

```mermaid
flowchart LR
    subgraph "User Input"
        RAWDATA["AnnData Object<br/>adata.h5ad"]
        CKPT["Pretrained Checkpoint<br/>best_model.pt"]
        VOCAB["Gene Vocabulary<br/>vocab.json"]
    end
    
    subgraph "Loading Phase"
        VOCAB --> GENEVOCAB["GeneVocab.from_file()<br/>scgpt/tokenizer/gene_tokenizer.py"]
        GENEVOCAB --> VOCABOBJ["GeneVocab object<br/>gene_name to id mapping"]
        
        CKPT --> LOADPRETRAINED["load_pretrained()<br/>scgpt/utils/util.py:304"]
        VOCABOBJ --> TRANSMODEL["TransformerModel<br/>scgpt/model/model.py"]
        LOADPRETRAINED --> TRANSMODEL
        TRANSMODEL --> LOADEDMODEL["Loaded Model<br/>model.eval()"]
    end
    
    subgraph "Inference Phase"
        RAWDATA --> EMBEDDATA["embed_data()<br/>scgpt/tasks/cell_emb.py"]
        LOADEDMODEL --> EMBEDDATA
        EMBEDDATA --> FORWARD["model.encode_batch()<br/>No gradient computation"]
        FORWARD --> CELLEMB["Cell Embeddings<br/>adata.obsm['X_scGPT']"]
    end
    
    subgraph "Downstream Applications"
        CELLEMB --> REFMAP["Reference Mapping<br/>tutorials/Tutorial_Reference_Mapping.ipynb"]
        CELLEMB --> INTEGRATION["Batch Integration<br/>tutorials/zero-shot/"]
        CELLEMB --> ANNOTATION["Cell Type Annotation<br/>Fine-tuning"]
    end
```

This diagram maps the typical zero-shot workflow to specific code entities. The `embed_data()` function in [scgpt/tasks/cell_emb.py]() is the primary interface for generating embeddings, while `load_pretrained()` in [scgpt/utils/util.py:304]() handles checkpoint loading across different backends (CPU, GPU, flash-attn).

**Sources:** [README.md:18](), [README.md:56-69]()

---

## Pretrained Model Zoo

scGPT provides several pretrained models for different use cases:

| Model Name | Training Data | Download Link | Use Case |
|------------|---------------|---------------|----------|
| **whole-human** | 33M normal human cells | [Google Drive](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y) | **Recommended default** |
| **continual pretrained** | Continual pretraining on diverse batches | [Google Drive](https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB) | Zero-shot embedding tasks |
| **brain** | 13.2M brain cells | [Google Drive](https://drive.google.com/drive/folders/1vf1ijfQSk7rGdDGpBntR5bi5g6gNt-Gx) | Brain-specific analysis |
| **blood** | 10.3M blood/bone marrow cells | [Google Drive](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU) | Hematology studies |
| **pan-cancer** | 5.7M cancer cells | [Google Drive](https://drive.google.com/drive/folders/13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa) | Cancer research |

Each checkpoint folder contains:
- `best_model.pt`: Model weights
- `vocab.json`: Gene vocabulary mapping gene names to token IDs
- `args.json`: Training configuration

**Note:** The `whole-human` model is recommended as the default starting point for most applications. Organ-specific models may provide competitive performance when the fine-tuning dataset shares similar cell type context with the training data.

**Sources:** [README.md:56-69]()

---

## Model Loading Architecture

```mermaid
graph TB
    subgraph "Checkpoint Files"
        CKPT["best_model.pt<br/>state_dict"]
        VOCAB_FILE["vocab.json<br/>gene mappings"]
        ARGS["args.json<br/>hyperparameters"]
    end
    
    subgraph "Vocabulary System"
        VOCAB_FILE --> GENEVOCAB_CLASS["GeneVocab class<br/>scgpt/tokenizer/gene_tokenizer.py"]
        GENEVOCAB_CLASS --> VOCAB_INSTANCE["vocab object<br/>len(vocab) = ntoken"]
        VOCAB_INSTANCE --> SPECIAL_TOKENS["Special tokens:<br/>&lt;pad&gt;, &lt;cls&gt;, &lt;eoc&gt;"]
    end
    
    subgraph "Model Initialization"
        VOCAB_INSTANCE --> TRANSMODEL_INIT["TransformerModel.__init__()<br/>scgpt/model/model.py"]
        ARGS --> TRANSMODEL_INIT
        
        TRANSMODEL_INIT --> GENE_ENCODER["GeneEncoder<br/>Embedding layer"]
        TRANSMODEL_INIT --> VALUE_ENCODER["ContinuousValueEncoder<br/>Expression values"]
        TRANSMODEL_INIT --> TRANSFORMER["TransformerEncoder<br/>12 layers default"]
    end
    
    subgraph "Weight Loading"
        CKPT --> TORCH_LOAD["torch.load()<br/>map_location"]
        TORCH_LOAD --> LOAD_PRETRAINED["load_pretrained()<br/>scgpt/utils/util.py:304"]
        LOAD_PRETRAINED --> CHECK_BACKEND["Check attention backend:<br/>FlashAttention vs Standard"]
        CHECK_BACKEND --> STATE_DICT["model.load_state_dict()"]
    end
    
    TRANSMODEL_INIT --> LOAD_PRETRAINED
    STATE_DICT --> READY_MODEL["model.eval()<br/>Ready for inference"]
    
    READY_MODEL --> EMBED_DATA_FUNC["embed_data()<br/>scgpt/tasks/cell_emb.py:258"]
```

The `load_pretrained()` function at [scgpt/utils/util.py:304]() handles automatic compatibility across different attention backends (flash-attn, standard PyTorch). This allows the same checkpoint to be loaded on CPU, GPU with flash-attention, or GPU with standard attention without modification.

**Sources:** [README.md:18](), [scgpt/data_collator.py:1-203]()

---

## Data Collation for Model Input

Once a model is loaded, raw AnnData needs to be converted into model-compatible batches. The `DataCollator` class handles this transformation:

```mermaid
flowchart TD
    subgraph "Input Data Structure"
        EXAMPLE["Example dict:<br/>{{'id': tensor,<br/>'genes': tensor[gene_ids],<br/>'expressions': tensor[values]}}"]
    end
    
    subgraph "DataCollator Processing"
        EXAMPLE --> COLLATOR["DataCollator<br/>scgpt/data_collator.py"]
        
        COLLATOR --> BINNING["Value Binning<br/>binning(row, n_bins=51)<br/>scgpt/preprocess.py"]
        BINNING --> SAMPLE_TRUNC["Sample/Truncate<br/>max_length limit"]
        SAMPLE_TRUNC --> PADDING["Padding<br/>pad_token_id, pad_value"]
        PADDING --> MASKING["MLM Masking<br/>mlm_probability=0.15"]
    end
    
    subgraph "Output Tensors"
        MASKING --> OUTPUT["Batch dict:<br/>{{'gene': padded_genes,<br/>'expr': padded_expressions,<br/>'masked_expr': masked_expr}}"]
    end
    
    OUTPUT --> DATALOADER["DataLoader<br/>Batch iterator"]
    DATALOADER --> MODEL_INPUT["Model forward pass"]
```

The `DataCollator` parameters:
- `do_padding=True`: Pad sequences to `max_length`
- `pad_token_id`: Token ID for padding (typically vocab["<pad>"])
- `do_mlm=True`: Apply masked language modeling
- `mlm_probability=0.15`: Fraction of tokens to mask
- `do_binning=True`: Discretize expression values into 51 bins
- `sampling=True`: Randomly sample genes if length > max_length

**Sources:** [scgpt/data_collator.py:10-203]()

---

## Next Steps

After completing basic installation and running your first embedding example, explore:

- **[Installation](#2.1)** - Detailed installation instructions, troubleshooting flash-attention, and environment setup
- **[Quick Start with Pretrained Models](#2.2)** - Comprehensive examples using different pretrained models
- **[Zero-Shot Applications](#4)** - Use scGPT without fine-tuning for reference mapping and integration
- **[Cell Embedding System](#5)** - Deep dive into the `embed_data()` function and embedding generation
- **[Fine-tuning Applications](#6)** - Adapt scGPT to specific downstream tasks
- **[Pretrained Models](#7)** - Complete catalog of available checkpoints and their specifications

For immediate hands-on learning, the tutorial notebooks in `tutorials/zero-shot/` provide complete working examples that can be run directly.

**Sources:** [README.md:1-113]()

---