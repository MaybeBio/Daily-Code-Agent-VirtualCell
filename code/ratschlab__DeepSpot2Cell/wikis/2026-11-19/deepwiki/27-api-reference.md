# API Reference

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [deepspot2cell/__init__.py](deepspot2cell/__init__.py)
- [deepspot2cell/dataloader.py](deepspot2cell/dataloader.py)
- [deepspot2cell/loss.py](deepspot2cell/loss.py)
- [deepspot2cell/model.py](deepspot2cell/model.py)

</details>



This document provides a complete reference for all public classes, functions, and modules in the DeepSpot2Cell codebase. It covers the main `deepspot2cell` package and its `utils` subpackage.

For detailed information about specific modules:
- Main package classes and functions: see [deepspot2cell Package](#9.1)
- Utility functions and helpers: see [deepspot2cell.utils Package](#9.2)

For higher-level conceptual information about these components, see:
- Model architecture details: [DeepSpot2Cell Model Architecture](#4.1)
- Dataset loading details: [DS2CDataset and Data Loading](#4.2)
- Loss functions and metrics: [Loss Functions](#4.3)

---

## Public API Overview

The `deepspot2cell` package exports the following core components through [deepspot2cell/\_\_init\_\_.py:1-4]():

### Core Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `DeepSpot2Cell` | `model.py` | Main model class implementing Phi and Rho ensembles with training/validation logic |
| `Phi` | `model.py` | Feature transformation network for cell and spot embeddings |
| `Rho` | `model.py` | Aggregation network that produces gene expression predictions |
| `DS2CDataset` | `dataloader.py` | PyTorch Dataset class for loading preprocessed embeddings and expressions |

### Loss Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `mse` | `loss.py` | Mean squared error loss |
| `pearsonr` | `loss.py` | Pearson correlation loss (1 - mean correlation) |
| `pearson_mse` | `loss.py` | Combined Pearson and MSE loss |
| `wmse` | `loss.py` | Weighted MSE loss with gene-specific weights |
| `safe_pearson` | `loss.py` | Pearson correlation metric with NaN handling |

**Sources:** [deepspot2cell/\_\_init\_\_.py:1-4]()

---

## Module Organization

The following diagram shows the structure of the `deepspot2cell` package and how modules depend on each other:

```mermaid
graph TB
    subgraph "deepspot2cell Package"
        INIT["__init__.py<br/>Public API Exports"]
        
        subgraph "Core Modules"
            MODEL["model.py<br/>DeepSpot2Cell, Phi, Rho"]
            LOADER["dataloader.py<br/>DS2CDataset"]
            LOSS["loss.py<br/>mse, pearsonr, wmse, safe_pearson"]
            DATASET["dataset_creation.py<br/>process_sample"]
        end
        
        subgraph "utils Subpackage"
            UTILS_INIT["utils/__init__.py"]
            UTILS_CORE["utils/utils.py<br/>load_config, fix_seed, get_gene_order"]
            UTILS_IMG["utils/utils_image.py<br/>load_pfm_model, predict_spot, predict_cell"]
            UTILS_DATA["utils/utils_dataloader.py<br/>pickle_load, spatial_upsample, balanced_sample"]
        end
    end
    
    INIT --> MODEL
    INIT --> LOADER
    INIT --> LOSS
    
    MODEL --> LOSS
    MODEL --> UTILS_CORE
    
    LOADER --> UTILS_CORE
    DATASET --> UTILS_IMG
    DATASET --> UTILS_CORE
    
    UTILS_INIT --> UTILS_CORE
    UTILS_INIT --> UTILS_IMG
    UTILS_INIT --> UTILS_DATA
```

**Sources:** [deepspot2cell/\_\_init\_\_.py:1-4](), [deepspot2cell/model.py:1-281](), [deepspot2cell/dataloader.py:1-390](), [deepspot2cell/loss.py:1-48]()

---

## Class Hierarchy

The following diagram shows the inheritance structure and relationships between core classes:

```mermaid
graph TB
    TORCH_DATASET["torch.utils.data.Dataset"]
    LIGHTNING_MODULE["lightning.LightningModule"]
    TORCH_MODULE["torch.nn.Module"]
    
    DS2CDATASET["DS2CDataset"]
    DEEPSPOT2CELL["DeepSpot2Cell"]
    PHI["Phi"]
    RHO["Rho"]
    WMSE["wmse"]
    
    TORCH_DATASET --> DS2CDATASET
    LIGHTNING_MODULE --> DEEPSPOT2CELL
    TORCH_MODULE --> PHI
    TORCH_MODULE --> RHO
    TORCH_MODULE --> WMSE
    
    DEEPSPOT2CELL -.contains.-> PHI
    DEEPSPOT2CELL -.contains.-> RHO
    
    PHI -.sequential.-> LINEAR1["nn.Linear"]
    PHI -.sequential.-> DROPOUT1["nn.Dropout"]
    PHI -.sequential.-> RELU1["nn.ReLU"]
    
    RHO -.sequential.-> LINEAR2["nn.Linear"]
    RHO -.sequential.-> DROPOUT2["nn.Dropout"]
    RHO -.sequential.-> RELU2["nn.ReLU"]
```

**Sources:** [deepspot2cell/model.py:11-40](), [deepspot2cell/model.py:43-281](), [deepspot2cell/dataloader.py:9-390](), [deepspot2cell/loss.py:25-41]()

---

## Data Flow Through API Components

The following diagram illustrates how data flows through the main API components during training and inference:

```mermaid
graph LR
    subgraph "Data Loading"
        H5["HDF5 Files<br/>*_expressions.h5"]
        NPY["NPY Files<br/>patch_*.npy"]
        
        DS2C["DS2CDataset<br/>__getitem__"]
        
        H5 --> DS2C
        NPY --> DS2C
    end
    
    subgraph "Data Preprocessing"
        PREPROCESS["DS2CDataset.preprocess<br/>normalize, log1p, scale"]
        STATS["_compute_dataset_statistics<br/>StandardScaler, min-max"]
        
        DS2C --> PREPROCESS
        STATS -.computes.-> PREPROCESS
    end
    
    subgraph "Model Forward Pass"
        BATCH["DataLoader Batch<br/>cell_embeddings, spot_expression"]
        
        PHI_CELL["phi_cell ensemble<br/>_apply_phi_cell"]
        PHI_SPOT["phi_spot ensemble<br/>_apply_phi"]
        RHO_ENS["rho ensemble<br/>_apply_rho"]
        
        FORWARD["forward / _forward_superspot<br/>or _forward_single_cell"]
        
        BATCH --> FORWARD
        FORWARD --> PHI_CELL
        FORWARD --> PHI_SPOT
        PHI_CELL --> RHO_ENS
        PHI_SPOT --> RHO_ENS
    end
    
    subgraph "Loss Computation"
        PRED["Predictions"]
        TARGET["Targets"]
        
        MSE_LOSS["mse"]
        PEARSON["safe_pearson"]
        
        RHO_ENS --> PRED
        BATCH --> TARGET
        
        PRED --> MSE_LOSS
        TARGET --> MSE_LOSS
        
        PRED --> PEARSON
        TARGET --> PEARSON
    end
    
    PREPROCESS --> BATCH
```

**Sources:** [deepspot2cell/dataloader.py:164-278](), [deepspot2cell/dataloader.py:280-319](), [deepspot2cell/model.py:84-147](), [deepspot2cell/loss.py:5-48]()

---

## Core API Components

### DeepSpot2Cell Model

The `DeepSpot2Cell` class [deepspot2cell/model.py:43-281]() is a PyTorch Lightning module that implements the main model architecture.

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `__init__` | Initialize model with Phi/Rho ensembles and hyperparameters |
| `forward` | Forward pass for spot-level prediction (calls `_forward_superspot`) |
| `_forward_single_cell` | Forward pass for single-cell prediction |
| `_forward_superspot` | Forward pass for superspot (batch) prediction |
| `_apply_phi` | Apply Phi ensemble to spot embeddings (median aggregation) |
| `_apply_phi_cell` | Apply Phi ensemble to cell embeddings (median aggregation) |
| `_apply_rho` | Apply Rho ensemble to concatenated features (mean aggregation) |
| `training_step` | Training logic with loss computation |
| `validation_step` | Validation logic with metrics |
| `configure_optimizers` | Configure AdamW optimizer |

**Sources:** [deepspot2cell/model.py:43-281]()

### DS2CDataset

The `DS2CDataset` class [deepspot2cell/dataloader.py:9-390]() handles data loading and preprocessing.

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `__init__` | Initialize dataset with paths, normalization options |
| `load_data` | Load all sample data and collect patch barcodes |
| `load_spot_data` | Load expression data from HDF5 files |
| `__getitem__` | Retrieve preprocessed batch with embeddings and expressions |
| `preprocess` | Normalize, log-transform, and scale expression data |
| `inverse_transform` | Reverse preprocessing transformations |
| `_compute_dataset_statistics` | Compute dataset-wide statistics for scaling |

**Sources:** [deepspot2cell/dataloader.py:9-390]()

### Phi and Rho Networks

The `Phi` [deepspot2cell/model.py:11-24]() and `Rho` [deepspot2cell/model.py:27-40]() classes are simple feedforward networks:

**Phi Architecture:**
- Linear layer (input_size → input_size)
- Dropout
- ReLU activation
- Linear layer (input_size → output_size)
- Dropout
- ReLU activation

**Rho Architecture:**
- Dropout
- Linear layer (input_size → input_size)
- ReLU activation
- Linear layer (input_size → output_size)
- ReLU activation
- Dropout

**Sources:** [deepspot2cell/model.py:11-40]()

---

## Import Patterns and Usage

The following diagram shows typical import patterns and how to use the API:

```mermaid
graph TB
    subgraph "User Code"
        IMPORT["from deepspot2cell import DeepSpot2Cell, DS2CDataset, mse"]
        UTILS["from deepspot2cell.utils import load_config, fix_seed"]
    end
    
    subgraph "Dataset Setup"
        CONFIG["config = load_config('config.yaml')"]
        SEED["fix_seed(2024)"]
        
        TRAIN_DS["train_ds = DS2CDataset(<br/>dataset_variant='',<br/>ids_list=['SAMPLE1'],<br/>data_path='/path/to/data',<br/>model_name='UNI',<br/>normalize=True<br/>)"]
        
        LOADER["train_loader = DataLoader(<br/>train_ds,<br/>batch_size=32,<br/>shuffle=True<br/>)"]
    end
    
    subgraph "Model Setup"
        MODEL_INIT["model = DeepSpot2Cell(<br/>input_size=1024,<br/>output_size=300,<br/>lr=1e-4,<br/>n_ensemble=10,<br/>phi2rho_size=512<br/>)"]
        
        TRAINER["trainer = L.Trainer(<br/>max_epochs=100,<br/>callbacks=[...]<br/>)"]
    end
    
    subgraph "Training"
        FIT["trainer.fit(<br/>model,<br/>train_loader,<br/>val_loader<br/>)"]
    end
    
    IMPORT --> CONFIG
    UTILS --> SEED
    
    CONFIG --> TRAIN_DS
    SEED --> TRAIN_DS
    
    TRAIN_DS --> LOADER
    LOADER --> FIT
    
    MODEL_INIT --> FIT
    TRAINER --> FIT
```

**Sources:** [deepspot2cell/\_\_init\_\_.py:1-4](), [deepspot2cell/model.py:43-50](), [deepspot2cell/dataloader.py:9-36]()

---

## API Design Patterns

### Ensemble Aggregation

The model uses two distinct aggregation strategies for ensembles [deepspot2cell/model.py:139-146]():

- **Phi ensembles**: Median aggregation across ensemble members (more robust to outliers)
- **Rho ensembles**: Mean aggregation across ensemble members (smoother predictions)

### Preprocessing Pipeline

The dataset applies a consistent preprocessing pipeline [deepspot2cell/dataloader.py:280-319]():

1. **Normalization**: Scale to `norm_counts` (default 1e4)
2. **Log transformation**: Apply `log1p` to normalized counts
3. **Standard scaling** (optional): Z-score standardization using `StandardScaler`
4. **Min-max scaling** (optional): Scale to [0, 1] range

Separate scalers are maintained for spot-level and cell-level data.

### Masking Strategy

The codebase uses several masking mechanisms:

| Mask Type | Purpose | Code Reference |
|-----------|---------|----------------|
| `cell_mask` | Indicates valid (non-padded) cells | [deepspot2cell/dataloader.py:223-224]() |
| `is_inside_spot` | Indicates cells inside vs outside spot | [deepspot2cell/dataloader.py:190]() |
| `neighb_masks` | Indicates missing neighboring spots | [deepspot2cell/dataloader.py:245-254]() |

**Sources:** [deepspot2cell/model.py:139-146](), [deepspot2cell/dataloader.py:280-319](), [deepspot2cell/dataloader.py:223-254]()

---

## Batch Data Structure

The `DS2CDataset.__getitem__` method returns a dictionary with the following structure:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `patch_embedding` | `torch.Tensor` | `(embed_dim,)` | Spot-level embedding from PFM |
| `cell_embeddings` | `torch.Tensor` | `(max_cells, embed_dim)` | Cell-level embeddings (padded) |
| `spot_expression` | `torch.Tensor` | `(num_genes,)` | Ground truth spot expression |
| `cell_expressions` | `torch.Tensor` | `(max_cells, num_genes)` | Ground truth cell expressions (padded) |
| `cell_mask` | `torch.Tensor` | `(max_cells,)` | Binary mask for valid cells |
| `is_inside_spot` | `torch.Tensor` | `(max_cells,)` | Binary mask for cells inside spot |
| `num_cells` | `int` | scalar | Actual number of cells (before padding) |
| `cell_centroids` | `torch.Tensor` | `(max_cells, 2)` | Cell centroid coordinates (padded) |
| `patch_barcode` | `str` | - | Unique identifier for the patch |
| `neighb_embeddings` | `torch.Tensor` | `(num_neighbors+1, embed_dim)` | Spot and neighbor embeddings (optional) |
| `neighb_masks` | `torch.Tensor` | `(num_neighbors+1,)` | Binary mask for missing neighbors (optional) |
| `cell_classes` | `List[str]` | `(max_cells,)` | Cell type annotations (optional) |

**Sources:** [deepspot2cell/dataloader.py:226-265]()

---

## Detailed Subsections

For detailed API documentation of specific components:

- **[deepspot2cell Package](#9.1)**: Complete reference for `DeepSpot2Cell`, `Phi`, `Rho`, `DS2CDataset`, and loss functions
- **[deepspot2cell.utils Package](#9.2)**: Complete reference for utility functions including configuration, image processing, and data loading helpers

---