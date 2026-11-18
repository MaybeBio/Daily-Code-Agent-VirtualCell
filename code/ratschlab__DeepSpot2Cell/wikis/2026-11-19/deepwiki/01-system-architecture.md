# System Architecture

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [figures/architecture.jpg](figures/architecture.jpg)

</details>



## Purpose and Scope

This document provides a detailed technical explanation of the DeepSpot2Cell system architecture, including:

- The distinction between training mode (spot-level supervision) and inference mode (cell-level prediction)
- Multi-scale image processing using cell tiles, spot tiles, and neighboring tiles
- The resolution transfer mechanism that enables cell-level predictions from spot-level training
- Data flow through the system components
- Module organization and dependencies

For conceptual definitions and terminology, see [Key Concepts and Terminology](#1.2). For implementation details of specific components, see [DeepSpot2Cell Model Architecture](#4.1), [DS2CDataset and Data Loading](#4.2), and [Dataset Creation](#5.1).

---

## Architectural Overview

DeepSpot2Cell is designed around a key innovation: training with readily available spot-level spatial transcriptomics data (55μm resolution) to enable predictions at the cellular level during inference. The system architecture reflects this dual-mode operation.

### System Components

The DeepSpot2Cell system consists of three primary layers:

| Layer | Components | Purpose |
|-------|-----------|---------|
| **Data Preparation** | `dataset_creation.py`, `dataloader.py` | Process raw HEST data, extract tiles, compute PFM embeddings, load training batches |
| **Model Layer** | `model.py` (DeepSpot2Cell, Phi, Rho), `loss.py` | Multi-scale feature aggregation, ensemble prediction, loss computation |
| **Execution Layer** | `scripts/train.py`, `utils/*` | Training orchestration, configuration management, utilities |

Sources: [README.md:1-70]()

### High-Level Data Flow

```mermaid
graph TB
    subgraph Input["Input Data"]
        HE["H&E Whole Slide Images"]
        ST["Spatial Transcriptomics<br/>(HEST-1k Dataset)"]
        SEG["Cell Segmentation Masks"]
    end
    
    subgraph Preprocessing["Preprocessing Layer"]
        DC["dataset_creation.py<br/>process_sample()"]
        PFM["Pathology Foundation Model<br/>(UNI/H-optimus-0/Phikon v2)"]
        TILES["Tile Extraction:<br/>- Cell tiles<br/>- Spot tiles (55μm)<br/>- Neighbor tiles"]
    end
    
    subgraph Storage["Persistent Storage"]
        EMB["Embeddings<br/>(.npy files)"]
        EXPR["Expression Matrices<br/>(.h5 files)"]
    end
    
    subgraph Training["Training Pipeline"]
        LOADER["DS2CDataset<br/>(dataloader.py)"]
        MODEL["DeepSpot2Cell Model<br/>(model.py)"]
        LOSS["Loss Functions<br/>(loss.py)"]
    end
    
    subgraph Output["Outputs"]
        CKPT["Trained Checkpoint<br/>(.pt files)"]
        PRED["Cell-Level Predictions<br/>(Virtual Transcriptomics)"]
    end
    
    HE --> TILES
    SEG --> TILES
    ST --> DC
    TILES --> PFM
    PFM --> EMB
    DC --> EXPR
    
    EMB --> LOADER
    EXPR --> LOADER
    LOADER --> MODEL
    MODEL --> LOSS
    LOSS --> CKPT
    CKPT --> PRED
```

Sources: [deepspot2cell/dataset_creation.py](), [deepspot2cell/dataloader.py](), [deepspot2cell/model.py](), [deepspot2cell/loss.py](), [scripts/train.py]()

---

## Training Architecture

During training, DeepSpot2Cell learns to predict spot-level gene expression from multi-scale morphological features. The model is supervised with measurements from 55μm diameter spots that aggregate signals from multiple cells.

### Multi-Scale Input Processing

Training uses three types of image tiles extracted around each cell:

```mermaid
graph TB
    subgraph SpatialContext["Spatial Context (Training)"]
        CELL["Cell Tile<br/>(cell_tile)<br/>Cropped by segmentation mask"]
        SPOT["Spot Tile<br/>(spot_tile)<br/>55μm region containing cell"]
        NEIGH["Neighbor Tiles<br/>(neighbor_tiles)<br/>Adjacent spot regions"]
    end
    
    subgraph FeatureExtraction["Feature Extraction"]
        PFM["Pathology Foundation Model<br/>get_pfm_predictions()"]
        CELL_EMB["cell_emb<br/>(1536-dim)"]
        SPOT_EMB["spot_emb<br/>(1536-dim)"]
        NEIGH_EMB["neighbor_emb<br/>(N × 1536-dim)"]
    end
    
    subgraph ModelInputs["Model Inputs"]
        CELL_IN["x_cell"]
        SPOT_IN["x_spot"]
        NEIGH_IN["x_neighbors"]
        MASK["neighbor_mask"]
    end
    
    CELL --> PFM
    SPOT --> PFM
    NEIGH --> PFM
    
    PFM --> CELL_EMB
    PFM --> SPOT_EMB
    PFM --> NEIGH_EMB
    
    CELL_EMB --> CELL_IN
    SPOT_EMB --> SPOT_IN
    NEIGH_EMB --> NEIGH_IN
    NEIGH_EMB --> MASK
```

**Tile Types:**

| Tile Type | Variable Name | Dimensions | Purpose |
|-----------|--------------|------------|---------|
| Cell Tile | `cell_tile` | Varies (segmentation-based) | Captures individual cell morphology |
| Spot Tile | `spot_tile` | 55μm diameter | Provides local tissue context |
| Neighbor Tiles | `neighbor_tiles` | 55μm diameter each | Captures spatial neighborhood information |

Sources: [deepspot2cell/dataloader.py](), [deepspot2cell/utils/utils_image.py]()

### Model Components

The DeepSpot2Cell model architecture consists of ensemble components that process multi-scale features:

```mermaid
graph TB
    subgraph Inputs["Inputs (Batch)"]
        X_CELL["x_cell<br/>(B × 1536)"]
        X_SPOT["x_spot<br/>(B × 1536)"]
        X_NEIGH["x_neighbors<br/>(B × N × 1536)"]
        MASK["neighbor_mask<br/>(B × N)"]
    end
    
    subgraph PhiEnsemble["Phi Ensemble (K models)"]
        PHI1["Phi Model 1<br/>(model.py: Phi)"]
        PHI2["Phi Model 2"]
        PHIK["Phi Model K"]
    end
    
    subgraph PhiOutputs["Phi Outputs"]
        PHI_OUT["phi_outputs<br/>List[B × D]"]
    end
    
    subgraph RhoEnsemble["Rho Ensemble"]
        RHO["Rho Model<br/>(model.py: Rho)<br/>Aggregates Phi outputs"]
    end
    
    subgraph FinalOutput["Output"]
        PRED["predictions<br/>(B × num_genes)<br/>Spot-level expression"]
    end
    
    X_CELL --> PHI1
    X_SPOT --> PHI1
    X_NEIGH --> PHI1
    MASK --> PHI1
    
    X_CELL --> PHI2
    X_SPOT --> PHI2
    X_NEIGH --> PHI2
    MASK --> PHI2
    
    X_CELL --> PHIK
    X_SPOT --> PHIK
    X_NEIGH --> PHIK
    MASK --> PHIK
    
    PHI1 --> PHI_OUT
    PHI2 --> PHI_OUT
    PHIK --> PHI_OUT
    
    PHI_OUT --> RHO
    RHO --> PRED
```

**Component Details:**

- **`Phi` class** ([deepspot2cell/model.py]()): Individual ensemble member that processes multi-scale inputs
  - Aggregates cell, spot, and neighbor embeddings
  - Uses attention mechanism for neighbor pooling (when `neighbor_mask` is provided)
  - Produces intermediate predictions
  
- **`Rho` class** ([deepspot2cell/model.py]()): Meta-learner that combines Phi outputs
  - Takes concatenated Phi predictions as input
  - Produces final gene expression predictions
  
- **`DeepSpot2Cell` class** ([deepspot2cell/model.py]()): Main model coordinating the ensemble
  - Manages K Phi models and 1 Rho model
  - Handles training/validation steps
  - Implements PyTorch Lightning interface

Sources: [deepspot2cell/model.py]()

### Loss Computation

Training optimizes a combination of Mean Squared Error (MSE) and Pearson correlation:

```mermaid
graph LR
    PRED["Model Predictions<br/>(B × G genes)"]
    TARGET["Spot-level Ground Truth<br/>(B × G genes)"]
    
    subgraph LossFunctions["Loss Functions (loss.py)"]
        MSE["mse_loss()<br/>Regression accuracy"]
        PEARSON["pearson_loss()<br/>1 - correlation"]
        COMBINED["Combined Loss<br/>mse + pearson"]
    end
    
    METRICS["Training Metrics:<br/>- train_loss<br/>- train_pearson<br/>- val_loss<br/>- val_pearson"]
    
    PRED --> MSE
    TARGET --> MSE
    PRED --> PEARSON
    TARGET --> PEARSON
    
    MSE --> COMBINED
    PEARSON --> COMBINED
    COMBINED --> METRICS
```

**Loss Functions:**

| Function | File Location | Purpose |
|----------|--------------|---------|
| `mse_loss()` | [deepspot2cell/loss.py]() | Standard MSE between predictions and targets |
| `pearson_loss()` | [deepspot2cell/loss.py]() | Returns `1 - correlation` to minimize negative correlation |
| `safe_pearson()` | [deepspot2cell/loss.py]() | Robust correlation computation with numerical stability |
| `wmse_loss()` | [deepspot2cell/loss.py]() | Weighted MSE for gene-specific importance |

Sources: [deepspot2cell/loss.py](), [deepspot2cell/model.py:90-150]()

---

## Inference Architecture

During inference, the same model architecture is applied to individual cells to produce cell-level transcriptomic predictions. This is the resolution transfer: spot-supervised training enables cell-level inference.

### Cell-Level Prediction Mode

```mermaid
graph TB
    subgraph CellInput["Single Cell Input"]
        TARGET_CELL["Target Cell<br/>(from segmentation)"]
        CELL_TILE["Cell Tile<br/>(cropped by mask)"]
    end
    
    subgraph ContextExtraction["Context Extraction"]
        SPOT_CTX["Containing Spot Tile<br/>(55μm region)"]
        NEIGH_CTX["Neighboring Spot Tiles<br/>(adjacent regions)"]
    end
    
    subgraph PFMProcessing["PFM Processing"]
        PFM_INF["PFM Feature Extraction<br/>predict_cell()"]
        FEAT["Feature Vectors:<br/>- cell_emb<br/>- spot_emb<br/>- neighbor_emb"]
    end
    
    subgraph ModelInference["Model Forward Pass"]
        PHI_INF["Phi Ensemble<br/>(K models)"]
        RHO_INF["Rho Aggregation"]
        CELL_PRED["Cell-Level<br/>Gene Expression<br/>(num_genes values)"]
    end
    
    TARGET_CELL --> CELL_TILE
    TARGET_CELL --> SPOT_CTX
    TARGET_CELL --> NEIGH_CTX
    
    CELL_TILE --> PFM_INF
    SPOT_CTX --> PFM_INF
    NEIGH_CTX --> PFM_INF
    
    PFM_INF --> FEAT
    FEAT --> PHI_INF
    PHI_INF --> RHO_INF
    RHO_INF --> CELL_PRED
```

**Key Inference Functions:**

| Function | File Location | Purpose |
|----------|--------------|---------|
| `predict_cell()` | [deepspot2cell/utils/utils_image.py]() | Predicts expression for a single cell from image path |
| `get_pfm_predictions()` | [deepspot2cell/utils/utils_image.py]() | Extracts PFM features for tiles |
| `forward()` | [deepspot2cell/model.py]() | Main model forward pass (shared between training/inference) |

Sources: [deepspot2cell/utils/utils_image.py](), [deepspot2cell/model.py]()

### Training vs Inference Comparison

| Aspect | Training Mode | Inference Mode |
|--------|--------------|----------------|
| **Supervision** | Spot-level expression (55μm aggregate) | No supervision (prediction only) |
| **Input** | Cell + spot + neighbors from training data | Cell + spot + neighbors from any image |
| **Output** | Spot-level predictions (for loss computation) | Cell-level predictions (virtual transcriptomics) |
| **Model State** | `model.train()`, gradients enabled | `model.eval()`, gradients disabled |
| **Batch Processing** | `DS2CDataset` batches via DataLoader | Single cell or custom batching |
| **Code Entry Point** | `scripts/train.py` → `trainer.fit()` | `utils/utils_image.py` → `predict_cell()` |

Sources: [scripts/train.py](), [deepspot2cell/utils/utils_image.py](), [deepspot2cell/model.py]()

---

## Resolution Transfer Mechanism

The core architectural innovation is the resolution transfer from spot-level training to cell-level inference. This is achieved through the multi-scale input design:

```mermaid
graph TB
    subgraph Training["Training Phase (Spot-Level Supervision)"]
        TR_CELL["Cell Morphology<br/>(high resolution)"]
        TR_SPOT["Spot Context<br/>(55μm tissue region)"]
        TR_NEIGH["Spatial Neighbors<br/>(surrounding spots)"]
        TR_LABEL["Ground Truth:<br/>Spot Expression<br/>(aggregated from cells)"]
        
        TR_LEARN["Model Learns:<br/>morphology → expression<br/>+ spatial patterns"]
    end
    
    subgraph Inference["Inference Phase (Cell-Level Prediction)"]
        INF_CELL["Cell Morphology<br/>(same resolution)"]
        INF_SPOT["Spot Context<br/>(same 55μm region)"]
        INF_NEIGH["Spatial Neighbors<br/>(same structure)"]
        
        INF_APPLY["Apply Learned Patterns:<br/>Individual cell features<br/>+ learned spatial context"]
        
        INF_OUT["Output:<br/>Cell-Level Expression<br/>(disaggregated)"]
    end
    
    TR_CELL --> TR_LEARN
    TR_SPOT --> TR_LEARN
    TR_NEIGH --> TR_LEARN
    TR_LABEL --> TR_LEARN
    
    TR_LEARN -.."learned weights".-> INF_APPLY
    
    INF_CELL --> INF_APPLY
    INF_SPOT --> INF_APPLY
    INF_NEIGH --> INF_APPLY
    INF_APPLY --> INF_OUT
```

**How Resolution Transfer Works:**

1. **Training**: The model learns to map morphological features (visible at cell resolution) to gene expression (measured at spot resolution). Even though the supervision is at spot level, the model sees individual cell morphologies within each spot.

2. **Key Insight**: The model learns fine-grained morphology-expression relationships because:
   - Cell tiles capture individual cell morphology
   - Spot tiles provide tissue context
   - Neighbor tiles encode spatial patterns
   - The combination allows the model to associate specific morphological patterns with transcriptional states

3. **Inference**: When predicting for an individual cell, the model applies learned patterns to that cell's morphology while incorporating the same contextual information (spot and neighbors), enabling cell-level predictions.

4. **Implementation**: The `forward()` method in `DeepSpot2Cell` is identical for training and inference—only the interpretation of the output changes (spot-level during training, cell-level during inference).

Sources: [README.md:1-10](), [deepspot2cell/model.py]()

---

## Module Organization

The codebase is organized into clear functional modules:

```mermaid
graph TB
    subgraph Package["deepspot2cell/ Package"]
        INIT["__init__.py<br/>Exports: DeepSpot2Cell, DS2CDataset,<br/>mse_loss, pearson_loss"]
        
        subgraph Core["Core Modules"]
            MODEL["model.py<br/>Classes: DeepSpot2Cell, Phi, Rho"]
            LOADER["dataloader.py<br/>Class: DS2CDataset"]
            LOSS["loss.py<br/>Functions: mse_loss, pearson_loss,<br/>wmse_loss, safe_pearson"]
            DATASET["dataset_creation.py<br/>Function: process_sample"]
        end
        
        subgraph Utils["utils/ Subpackage"]
            UTILS_INIT["utils/__init__.py"]
            UTILS_CORE["utils/utils.py<br/>config, seed, gene utilities"]
            UTILS_IMG["utils/utils_image.py<br/>PFM loading, predict_cell,<br/>predict_spot"]
            UTILS_DATA["utils/utils_dataloader.py<br/>augmentation, smoothing,<br/>clustering"]
        end
    end
    
    subgraph Scripts["scripts/ Directory"]
        TRAIN["train.py<br/>Main training script"]
    end
    
    subgraph Configs["configs/ Directory"]
        CFG_DATASET["dataset.example.yaml<br/>Preprocessing config"]
        CFG_TRAIN["training.example.yaml<br/>Training config"]
    end
    
    INIT --> MODEL
    INIT --> LOADER
    INIT --> LOSS
    
    MODEL --> LOSS
    LOADER --> UTILS_CORE
    LOADER --> UTILS_DATA
    DATASET --> UTILS_IMG
    
    UTILS_INIT --> UTILS_CORE
    UTILS_INIT --> UTILS_IMG
    UTILS_INIT --> UTILS_DATA
    
    TRAIN --> MODEL
    TRAIN --> LOADER
    TRAIN --> UTILS_CORE
    
    CFG_DATASET --> DATASET
    CFG_TRAIN --> TRAIN
```

**Module Responsibilities:**

| Module | Primary Classes/Functions | Responsibility |
|--------|--------------------------|----------------|
| `model.py` | `DeepSpot2Cell`, `Phi`, `Rho` | Neural network architecture and training logic |
| `dataloader.py` | `DS2CDataset` | Data loading, preprocessing, batching |
| `loss.py` | `mse_loss()`, `pearson_loss()` | Loss function implementations |
| `dataset_creation.py` | `process_sample()` | Preprocessing HEST data, extracting embeddings |
| `utils/utils.py` | `load_config()`, `fix_seed()` | Configuration and general utilities |
| `utils/utils_image.py` | `predict_cell()`, `get_pfm_predictions()` | Image processing and PFM inference |
| `utils/utils_dataloader.py` | `smooth_expression()`, `cluster_spots()` | Data augmentation and processing |
| `scripts/train.py` | `main()` | Training orchestration and CLI |

Sources: [deepspot2cell/__init__.py](), [deepspot2cell/model.py](), [deepspot2cell/dataloader.py](), [deepspot2cell/loss.py](), [deepspot2cell/dataset_creation.py](), [deepspot2cell/utils/](), [scripts/train.py]()

---

## Data Flow Through Components

The complete data flow from raw inputs to predictions:

```mermaid
graph TB
    subgraph Stage1["Stage 1: Raw Data"]
        HEST_DB["HEST-1k Dataset<br/>(Hugging Face)"]
        WSI["Whole Slide Images"]
        SEG_MASKS["Segmentation Masks"]
    end
    
    subgraph Stage2["Stage 2: Preprocessing<br/>(dataset_creation.py)"]
        PROC["process_sample()<br/>Extract tiles, compute embeddings"]
        PFM_MODEL["Load PFM Model<br/>load_image_model()"]
    end
    
    subgraph Stage3["Stage 3: Persistent Data"]
        EMB_FILES["embeddings/<br/>sample_id_cell_emb.npy<br/>sample_id_spot_emb.npy<br/>sample_id_neighbor_emb.npy"]
        EXPR_FILES["expressions/<br/>sample_id.h5<br/>(AnnData format)"]
    end
    
    subgraph Stage4["Stage 4: Training Data<br/>(dataloader.py)"]
        DATASET_CLS["DS2CDataset<br/>__init__, __getitem__"]
        BATCH_DATA["Batched Tensors:<br/>x_cell, x_spot,<br/>x_neighbors, neighbor_mask,<br/>target_expression"]
    end
    
    subgraph Stage5["Stage 5: Model Training<br/>(model.py, train.py)"]
        FWD["forward()<br/>Phi ensemble → Rho"]
        LOSS_COMP["training_step()<br/>Compute loss"]
        OPT["Optimizer Step<br/>Update weights"]
    end
    
    subgraph Stage6["Stage 6: Inference<br/>(utils_image.py)"]
        LOAD_MODEL["Load Checkpoint<br/>DeepSpot2Cell.load_from_checkpoint()"]
        PREDICT["predict_cell()<br/>Single cell prediction"]
        RESULTS["Cell Expression Vector<br/>(num_genes values)"]
    end
    
    HEST_DB --> PROC
    WSI --> PROC
    SEG_MASKS --> PROC
    PFM_MODEL --> PROC
    
    PROC --> EMB_FILES
    PROC --> EXPR_FILES
    
    EMB_FILES --> DATASET_CLS
    EXPR_FILES --> DATASET_CLS
    DATASET_CLS --> BATCH_DATA
    
    BATCH_DATA --> FWD
    FWD --> LOSS_COMP
    LOSS_COMP --> OPT
    OPT -.checkpoint.-> LOAD_MODEL
    
    LOAD_MODEL --> PREDICT
    PREDICT --> RESULTS
```

**Processing Stages:**

1. **Stage 1-2**: [Dataset Creation](#5.1) - One-time preprocessing to generate embeddings
2. **Stage 3**: Persistent storage in `.npy` and `.h5` formats for reuse
3. **Stage 4**: [DS2CDataset](#4.2) - Dynamic loading and augmentation during training
4. **Stage 5**: [Model Training](#8.1) - Iterative optimization using PyTorch Lightning
5. **Stage 6**: [Inference](#8.2) - Apply trained model to new cells

Sources: [deepspot2cell/dataset_creation.py](), [deepspot2cell/dataloader.py](), [deepspot2cell/model.py](), [scripts/train.py](), [deepspot2cell/utils/utils_image.py]()

---