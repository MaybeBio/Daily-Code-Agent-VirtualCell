# Quick Start with Example Data

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [example_data/README.md](example_data/README.md)
- [example_data/data/h5ad/ZEN38.h5ad](example_data/data/h5ad/ZEN38.h5ad)
- [example_data/data/image/ZEN38.jpg](example_data/data/image/ZEN38.jpg)

</details>



## Purpose and Scope

This page guides you through running DeepSpot2Cell on the provided example dataset to verify your installation and see immediate results. This quick start uses pre-processed data and a pre-trained model to demonstrate inference capabilities without requiring full dataset preprocessing or model training.

For complete installation instructions, see [Installation and Setup](#2.1). For detailed preprocessing, training, and inference workflows, see the tutorial series starting at [Tutorials](#3).

**Sources**: [README.md:24-33](), [example_data/README.md:1-12]()

---

## Example Data Overview

The repository includes a sample dataset from the HEST collection, specifically the **ZEN38** sample, which contains Xenium single-cell spatial transcriptomics data aligned with H&E histology images. This example data allows you to immediately test inference functionality without downloading the full HEST-1k dataset.

### Data Structure

The example data is organized in the `example_data/` directory with the following structure:

| File Path | Content | Format | Purpose |
|-----------|---------|--------|---------|
| `data/h5ad/ZEN38.h5ad` | Xenium single-cell spatial transcriptomics | AnnData H5AD | Ground truth expression data for validation |
| `data/image/ZEN38.jpg` | H&E histology image | JPEG | Input image for morphological feature extraction |
| `data/meta/ZEN38.json` | Sample metadata | JSON | Pixel size, spot diameter, spatial coordinates |

```mermaid
graph TB
    subgraph "example_data Directory"
        ROOT["example_data/"]
        
        subgraph "Data Files"
            H5AD["data/h5ad/ZEN38.h5ad<br/>AnnData format<br/>Single-cell expressions"]
            IMAGE["data/image/ZEN38.jpg<br/>H&E histology<br/>High-resolution WSI"]
            META["data/meta/ZEN38.json<br/>Metadata<br/>pixel_size_um_embedded<br/>spot_diameter_fullres"]
        end
        
        README["README.md<br/>Documentation"]
    end
    
    subgraph "HEST Format Compatibility"
        HEST_LIB["HEST Library<br/>github.com/mahmoodlab/hest"]
        HEST_1K["HEST-1k Dataset<br/>huggingface.co/datasets/MahmoodLab/hest"]
    end
    
    ROOT --> H5AD
    ROOT --> IMAGE
    ROOT --> META
    ROOT --> README
    
    H5AD --> HEST_LIB
    IMAGE --> HEST_LIB
    META --> HEST_LIB
    
    HEST_LIB -.."follows format of"..-> HEST_1K
```

**Diagram**: Example data directory structure and HEST format integration

**Sources**: [example_data/README.md:1-12]()

---

## Verifying Example Data

After installation, verify that the example data is present and accessible:

```bash
# Check data directory structure
ls -R example_data/data/

# Expected output:
# example_data/data/h5ad:
# ZEN38.h5ad
#
# example_data/data/image:
# ZEN38.jpg
#
# example_data/data/meta:
# ZEN38.json
```

### Data Format Specifications

The example data follows the HEST format specification. Key metadata fields in `ZEN38.json` include:

| Field | Description | Typical Value |
|-------|-------------|---------------|
| `pixel_size_um_embedded` | Micrometers per pixel at embedded resolution | ~0.5 μm/pixel |
| `spot_diameter_fullres` | Spot diameter in pixels at full resolution | Scaled to embedded resolution |
| Coordinate mappings | Spatial alignment between image and expression data | Transformation matrices |

**Sources**: [example_data/README.md:7-11](), [README.md:47-51]()

---

## Quick Inference Workflow

The fastest way to see DeepSpot2Cell in action is through the inference tutorial notebook, which uses the example data to demonstrate cell-level transcriptomic predictions.

```mermaid
graph LR
    subgraph "Input: Example Data"
        EX_IMG["ZEN38.jpg<br/>H&E Image"]
        EX_META["ZEN38.json<br/>Metadata"]
        EX_H5AD["ZEN38.h5ad<br/>Ground Truth<br/>(optional)"]
    end
    
    subgraph "Inference Pipeline"
        TUTORIAL["Tutorial 3 Notebook<br/>tutorials/GettingStartedWithDeepSpot2Cell_3_inference.ipynb"]
        
        subgraph "Key Functions"
            LOAD_PFM["load_PFM_model()<br/>deepspot2cell/utils/utils_image.py"]
            LOAD_MODEL["DeepSpot2Cell.load_from_checkpoint()<br/>deepspot2cell/model.py"]
            PREDICT["predict_cell_from_image_path()<br/>deepspot2cell/utils/utils_image.py"]
        end
    end
    
    subgraph "Output: Predictions"
        CELL_EXPR["Cell-Level<br/>Gene Expression<br/>numpy array"]
        VIZ["Visualizations<br/>Spatial plots<br/>Gene overlays"]
    end
    
    EX_IMG --> TUTORIAL
    EX_META --> TUTORIAL
    EX_H5AD -.."validation"..-> TUTORIAL
    
    TUTORIAL --> LOAD_PFM
    LOAD_PFM --> LOAD_MODEL
    LOAD_MODEL --> PREDICT
    
    PREDICT --> CELL_EXPR
    CELL_EXPR --> VIZ
```

**Diagram**: Quick inference workflow using example data, mapping to specific code functions

**Sources**: [README.md:29-33]()

---

## Running the Inference Tutorial

### Step 1: Open the Notebook

Navigate to the inference tutorial:

```bash
cd tutorials/
jupyter notebook GettingStartedWithDeepSpot2Cell_3_inference.ipynb
```

The tutorial notebook provides an interactive environment with pre-configured cells that execute the inference pipeline on the ZEN38 example data.

### Step 2: Key Notebook Sections

The inference tutorial is structured into the following sections:

| Section | Purpose | Key Operations |
|---------|---------|----------------|
| **Setup** | Import dependencies, configure paths | Load `deepspot2cell` package, set data paths |
| **Load PFM** | Initialize pathology foundation model | Call `load_PFM_model()` for UNI/H-optimus-0/Phikon |
| **Load Model** | Load pre-trained DeepSpot2Cell checkpoint | `DeepSpot2Cell.load_from_checkpoint()` |
| **Prepare Data** | Load example image and metadata | Read `ZEN38.jpg` and `ZEN38.json` |
| **Run Inference** | Predict cell-level expressions | `predict_cell_from_image_path()` or `predict_spots()` |
| **Visualize** | Generate spatial plots and gene maps | Overlay predictions on tissue image |
| **Validate** | Compare with ground truth (optional) | Correlation with `ZEN38.h5ad` if available |

### Step 3: Expected Runtime

| Operation | Approximate Time | Notes |
|-----------|------------------|-------|
| PFM model loading | 1-2 minutes | One-time per session |
| DeepSpot2Cell model loading | 10-30 seconds | Depends on checkpoint size |
| Inference on ZEN38 | 2-5 minutes | GPU-accelerated recommended |
| Visualization generation | 30-60 seconds | Depends on number of cells |

**Sources**: [README.md:29-33]()

---

## Understanding the Outputs

### Predicted Expression Matrix

The primary output is a cell-by-gene expression matrix where each row represents a cell and each column represents a gene:

```
Shape: (n_cells, n_genes)
Type: numpy.ndarray or pandas.DataFrame
Values: Predicted gene expression levels (normalized)
```

### Spatial Visualization

The tutorial generates spatial visualizations showing:

1. **Cell Segmentation Overlay**: Individual cells highlighted on the H&E image
2. **Gene Expression Maps**: Spatial distribution of specific genes
3. **Cell Type Predictions**: If integrated with downstream clustering/annotation

### Quality Metrics (If Ground Truth Available)

When the ground truth data from `ZEN38.h5ad` is available, the tutorial can compute:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Pearson Correlation** | Gene-wise correlation between predicted and actual | Higher is better (>0.5 typical) |
| **Spearman Correlation** | Rank-based correlation | Robust to outliers |
| **MSE** | Mean squared error per gene | Lower is better |

```mermaid
graph TB
    subgraph "Inference Outputs"
        PRED["predictions<br/>numpy array<br/>(n_cells, n_genes)"]
        
        subgraph "Visualization Outputs"
            SPATIAL["Spatial Plots<br/>matplotlib figures"]
            GENE_MAP["Gene Expression Maps<br/>Per-gene overlays"]
            HEATMAP["Expression Heatmaps<br/>Cell clustering"]
        end
        
        subgraph "Validation Metrics (Optional)"
            CORR["Pearson/Spearman<br/>Per-gene correlations"]
            MSE_VAL["MSE<br/>Per-gene errors"]
            SUMMARY["Summary Statistics<br/>Overall performance"]
        end
    end
    
    subgraph "Code Functions"
        PREDICT_FN["predict_cell_from_image_path()<br/>deepspot2cell/utils/utils_image.py"]
        SAFE_PEARSON["safe_pearson()<br/>deepspot2cell/loss.py"]
    end
    
    PREDICT_FN --> PRED
    PRED --> SPATIAL
    PRED --> GENE_MAP
    PRED --> HEATMAP
    
    PRED --> CORR
    PRED --> MSE_VAL
    CORR --> SUMMARY
    MSE_VAL --> SUMMARY
    
    SAFE_PEARSON -.."computes"..-> CORR
```

**Diagram**: Output types from inference and their associated code functions

**Sources**: [README.md:29-33]()

---

## Troubleshooting Quick Start

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Missing example data** | `FileNotFoundError` when loading `ZEN38.jpg` | Verify repository was cloned completely with `git clone --recurse-submodules` |
| **PFM model not found** | Error loading foundation model | Download PFM weights from HuggingFace (see [Pathology Foundation Models](#10.2)) |
| **CUDA out of memory** | GPU memory error during inference | Reduce batch size or use CPU with `device='cpu'` |
| **Slow inference** | Takes >10 minutes for ZEN38 | Ensure GPU is being used; check with `torch.cuda.is_available()` |

### Verifying Installation

Run these checks before proceeding:

```python
import deepspot2cell
import torch
import numpy as np
from pathlib import Path

# Check package installation
print(f"DeepSpot2Cell version: {deepspot2cell.__version__}")

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Check example data
data_path = Path("example_data/data")
assert (data_path / "h5ad/ZEN38.h5ad").exists(), "Missing ZEN38.h5ad"
assert (data_path / "image/ZEN38.jpg").exists(), "Missing ZEN38.jpg"
assert (data_path / "meta/ZEN38.json").exists(), "Missing ZEN38.json"
print("Example data verified ✓")
```

**Sources**: [README.md:16-22]()

---

## Next Steps

After completing the quick start with example data, you can proceed to more advanced workflows:

### For Custom Data Processing

If you want to work with your own HEST data or custom tissue samples:
- **[Tutorial 1: Data Preprocessing](#3.1)** - Learn how to process raw HEST samples, extract morphological features using PFMs, and generate training-ready embeddings
- **[Dataset Configuration](#6.1)** - Configure dataset creation parameters for your specific data

### For Model Training

To train your own DeepSpot2Cell model:
- **[Tutorial 2: Model Training](#3.2)** - Step-by-step guide for training with custom hyperparameters
- **[Training Configuration](#6.2)** - Detailed documentation of all training parameters
- **[DeepSpot2Cell Model Architecture](#4.1)** - Understand the model components (Phi and Rho ensembles)

### For Production Inference

To deploy trained models for large-scale inference:
- **[Inference Workflows](#8.2)** - Different inference modes (spot-level vs cell-level)
- **[Image Processing Utilities](#7.1)** - Functions for batch processing and tile extraction

### For Advanced Usage

To extend or customize DeepSpot2Cell:
- **[Core Components](#4)** - Deep dive into model, dataset, and loss function implementations
- **[Development Guide](#11)** - Contributing guidelines and project structure

**Sources**: [README.md:24-33]()

---

## Quick Start Summary

This quick start demonstrated:

1. ✓ Verifying the example data structure in `example_data/`
2. ✓ Running inference using the pre-built tutorial notebook
3. ✓ Understanding output formats (expression matrices, visualizations, metrics)
4. ✓ Troubleshooting common installation and runtime issues

The ZEN38 example provides a minimal but complete demonstration of DeepSpot2Cell's capabilities. For production use, you'll want to follow the full tutorial series to preprocess your own data, train custom models, and perform large-scale inference on whole slide images.

**Sources**: [README.md:1-70](), [example_data/README.md:1-12]()

---