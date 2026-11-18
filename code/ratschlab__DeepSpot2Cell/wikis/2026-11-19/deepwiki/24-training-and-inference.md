# Training and Inference

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [deepspot2cell/utils/utils_image.py](deepspot2cell/utils/utils_image.py)
- [scripts/train.py](scripts/train.py)

</details>



This page provides an overview of the training and inference workflows in DeepSpot2Cell. It explains the conceptual differences between these two operational modes, their implementation in the codebase, and how they relate to the overall system architecture.

For detailed documentation of the training script implementation, see [Training Script](#8.1). For specific inference workflows and their usage, see [Inference Workflows](#8.2). For hands-on guidance, consult [Tutorial 2: Model Training](#3.2) and [Tutorial 3: Inference and Prediction](#3.3).

## Training vs Inference: Core Distinction

The fundamental innovation of DeepSpot2Cell lies in the asymmetry between training and inference. During **training**, the model learns from spot-level gene expression data (55μm regions containing multiple cells). During **inference**, the trained model predicts gene expression at the single-cell level. This resolution transfer is what enables the system to generate virtual single-cell spatial transcriptomics from more readily available spot-level training data.

```mermaid
graph TB
    subgraph "Training Phase"
        direction TB
        TR_INPUT["H&E Image Patches<br/>(multi-scale)"]
        TR_EMB["PFM Embeddings<br/>(cell + spot + neighbors)"]
        TR_MODEL["DeepSpot2Cell Model<br/>[model.py:DeepSpot2Cell]"]
        TR_TARGET["Spot-Level Expression<br/>(ground truth)"]
        TR_LOSS["Loss Computation<br/>[loss.py:mse, pearsonr]"]
        TR_BACKWARD["Gradient Descent<br/>(optimizer step)"]
        
        TR_INPUT --> TR_EMB
        TR_EMB --> TR_MODEL
        TR_TARGET --> TR_LOSS
        TR_MODEL --> TR_LOSS
        TR_LOSS --> TR_BACKWARD
        TR_BACKWARD -.update weights.-> TR_MODEL
    end
    
    subgraph "Inference Phase"
        direction TB
        INF_INPUT["H&E Image Patches<br/>(multi-scale)"]
        INF_EMB["PFM Embeddings<br/>(cell + spot + neighbors)"]
        INF_MODEL["Trained DeepSpot2Cell<br/>[model.py:_forward_single_cell]"]
        INF_OUTPUT["Cell-Level Expression<br/>(prediction)"]
        
        INF_INPUT --> INF_EMB
        INF_EMB --> INF_MODEL
        INF_MODEL --> INF_OUTPUT
    end
    
    TR_MODEL -.checkpoint.-> INF_MODEL
    
    style TR_LOSS fill:#f9f9f9
    style INF_OUTPUT fill:#f9f9f9
```

**Sources:** [README.md:9](), [deepspot2cell/model.py](), [deepspot2cell/loss.py]()

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Objective** | Learn spot-to-expression mapping | Predict cell-level expression |
| **Supervision** | Spot-level gene expression (55μm) | None (uses learned patterns) |
| **Input Structure** | Cell tile + Spot tile + Neighbors | Cell tile + Spot tile + Neighbors |
| **Output** | Spot-level prediction + loss | Cell-level prediction |
| **Forward Method** | `forward()` in training mode | `_forward_single_cell()` |
| **Data Source** | `DS2CDataset` with ground truth | Raw images or embeddings |

## Training Workflow

The training workflow is orchestrated by the `train.py` script, which integrates data loading, model instantiation, and PyTorch Lightning training loops.

```mermaid
graph TB
    START["train.py execution"]
    
    CONFIG["load_config()<br/>[train.py:15-17]"]
    SEED["fix_seed()<br/>[utils.py:fix_seed]"]
    
    subgraph "Data Preparation"
        BUILD_DS["build_datasets()<br/>[train.py:20-60]"]
        TRAIN_DS["DS2CDataset (train)<br/>[dataloader.py:DS2CDataset]"]
        VAL_DS["DS2CDataset (val)<br/>[dataloader.py:DS2CDataset]"]
        BUILD_DL["build_dataloaders()<br/>[train.py:63-87]"]
        TRAIN_DL["PyTorch DataLoader (train)"]
        VAL_DL["PyTorch DataLoader (val)"]
    end
    
    subgraph "Model Setup"
        BUILD_MODEL["build_model()<br/>[train.py:90-110]"]
        DS2C_MODEL["DeepSpot2Cell<br/>[model.py:DeepSpot2Cell]"]
    end
    
    subgraph "Trainer Setup"
        BUILD_TRAINER["build_trainer()<br/>[train.py:113-152]"]
        CALLBACKS["Callbacks:<br/>EarlyStopping<br/>ModelCheckpoint"]
        LOGGER["CSVLogger"]
        TRAINER["L.Trainer"]
    end
    
    FIT["trainer.fit()<br/>[train.py:175]"]
    CKPT["Checkpoint saved<br/>(.pt file)"]
    
    START --> CONFIG
    CONFIG --> SEED
    SEED --> BUILD_DS
    
    BUILD_DS --> TRAIN_DS
    BUILD_DS --> VAL_DS
    TRAIN_DS --> BUILD_DL
    VAL_DS --> BUILD_DL
    BUILD_DL --> TRAIN_DL
    BUILD_DL --> VAL_DL
    
    BUILD_DS --> BUILD_MODEL
    BUILD_MODEL --> DS2C_MODEL
    
    CONFIG --> BUILD_TRAINER
    BUILD_TRAINER --> CALLBACKS
    BUILD_TRAINER --> LOGGER
    BUILD_TRAINER --> TRAINER
    
    TRAIN_DL --> FIT
    VAL_DL --> FIT
    DS2C_MODEL --> FIT
    TRAINER --> FIT
    
    FIT --> CKPT
```

**Sources:** [scripts/train.py:1-179]()

### Training Pipeline Components

The training script follows a modular design pattern with separate functions for each setup phase:

**Configuration Loading** [scripts/train.py:15-17]()
- Reads YAML configuration file
- Contains experiment settings, data paths, model hyperparameters, and trainer options

**Dataset Construction** [scripts/train.py:20-60]()
- Instantiates `DS2CDataset` objects for training and validation splits
- Configures data preprocessing: normalization, scaling, neighbor degree
- Shares scaling statistics between train and validation sets

**DataLoader Building** [scripts/train.py:63-87]()
- Creates PyTorch `DataLoader` instances
- Configures batch size, workers, and memory pinning
- Enables shuffling for training, disables for validation

**Model Instantiation** [scripts/train.py:90-110]()
- Infers input/output dimensions from dataset samples
- Initializes `DeepSpot2Cell` with ensemble configurations
- Sets dropout rates, learning rate, and weight decay

**Trainer Setup** [scripts/train.py:113-152]()
- Configures PyTorch Lightning `Trainer`
- Attaches callbacks: `EarlyStopping`, `ModelCheckpoint`
- Sets up logging with `CSVLogger`
- Specifies accelerator, precision, and gradient clipping

**Training Execution** [scripts/train.py:175]()
- Calls `trainer.fit()` to start training loop
- Automatically handles epoch iteration, validation, and checkpointing

**Sources:** [scripts/train.py:1-179]()

## Inference Workflows

DeepSpot2Cell supports three distinct inference modes, each designed for different use cases. All modes share the common pattern of processing multi-scale image features through pathology foundation models before prediction.

```mermaid
graph LR
    subgraph "Inference Mode Selection"
        MODE_SPOT["Spot-Level Inference<br/>[utils_image.py:246-297]"]
        MODE_CELL["Cell-Level Inference<br/>[utils_image.py:300-361]"]
        MODE_DS2C["DeepSpot2Cell Inference<br/>[utils_image.py:364-457]"]
    end
    
    subgraph "Common Processing"
        IMAGE["H&E Images"]
        CROP["crop_tile()<br/>[utils_image.py:195-203]"]
        PFM["Pathology Foundation Model<br/>get_morphology_model_and_preprocess()"]
        EMBED["Feature Embeddings"]
    end
    
    subgraph "Model Forward Pass"
        SPOT_FORWARD["forward() with spot tiles<br/>[model.py:forward]"]
        CELL_FORWARD["_forward_single_cell()<br/>[model.py:_forward_single_cell]"]
        DS2C_FORWARD["_forward_single_cell()<br/>with image paths"]
    end
    
    subgraph "Outputs"
        SPOT_EXPR["Spot Expression<br/>(55μm aggregated)"]
        CELL_EXPR["Cell Expression<br/>(single-cell)"]
        DS2C_EXPR["Cell Expression<br/>(from paths)"]
    end
    
    IMAGE --> CROP
    CROP --> PFM
    PFM --> EMBED
    
    EMBED --> MODE_SPOT
    EMBED --> MODE_CELL
    
    MODE_SPOT --> SPOT_FORWARD
    MODE_CELL --> CELL_FORWARD
    MODE_DS2C --> DS2C_FORWARD
    
    SPOT_FORWARD --> SPOT_EXPR
    CELL_FORWARD --> CELL_EXPR
    DS2C_FORWARD --> DS2C_EXPR
```

**Sources:** [deepspot2cell/utils/utils_image.py:246-457]()

### Inference Mode Comparison

| Function | Purpose | Input Requirements | Output Format | Typical Use Case |
|----------|---------|-------------------|---------------|------------------|
| `predict_spot_spatial_transcriptomics_from_image_path` | Predict at spot level | WSI path, spot coordinates, AnnData | Spot-level expression matrix | Validation against ground truth |
| `predict_cell_spatial_transcriptomics_from_image_path` | Predict at cell level | WSI path, cell coordinates, AnnData | Cell-level expression matrix | Virtual single-cell transcriptomics |
| `predict_spot2cell_from_image_paths` | Predict from individual images | Cell image path, spot path, neighbor paths | Gene expression dictionary | Fine-grained analysis of individual cells |

**Sources:** [deepspot2cell/utils/utils_image.py:246-457]()

### Spot-Level Inference

Function: `predict_spot_spatial_transcriptomics_from_image_path` [deepspot2cell/utils/utils_image.py:246-297]()

This mode predicts gene expression at the spot level (55μm diameter), matching the training supervision resolution. It is primarily used for:
- Validating model performance against ground truth spot-level data
- Predicting expression for spots in samples without transcriptomics data
- Benchmarking against standard spatial transcriptomics platforms

**Key Steps:**
1. Iterate over spot coordinates in `adata.obs`
2. For each spot, crop the spot tile and compute mini-tiles
3. Identify and crop neighboring spots within specified radius
4. Extract PFM embeddings for all tiles
5. Forward pass through DeepSpot2Cell model
6. Inverse transform predictions to original expression scale

**Sources:** [deepspot2cell/utils/utils_image.py:246-297]()

### Cell-Level Inference

Function: `predict_cell_spatial_transcriptomics_from_image_path` [deepspot2cell/utils/utils_image.py:300-361]()

This mode generates virtual single-cell spatial transcriptomics by predicting at the individual cell level. This is the primary application of DeepSpot2Cell's resolution transfer capability.

**Key Steps:**
1. Build spatial neighbor index using `NearestNeighbors` [deepspot2cell/utils/utils_image.py:309-320]()
2. For each cell, crop the cell tile at specified `cell_diameter`
3. Crop neighboring cell tiles within `radius_neighbors`
4. Extract PFM embeddings for cell and neighbors
5. Use `_forward_single_cell()` method for prediction
6. Inverse transform to expression values

**Sources:** [deepspot2cell/utils/utils_image.py:300-361]()

### DeepSpot2Cell-Specific Inference

Function: `predict_spot2cell_from_image_paths` [deepspot2cell/utils/utils_image.py:364-457]()

This specialized mode operates on pre-extracted image tiles rather than whole slide images, providing flexibility for custom workflows and fine-grained analysis.

**Input Structure:**
- `cell_image_path`: Path to target cell image tile
- `spot_image_path`: Path to spot tile containing the cell
- `neighbor_image_paths`: List of paths to neighboring image tiles

**Workflow:**
1. Load individual image tiles using PIL [deepspot2cell/utils/utils_image.py:398]()
2. Extract embeddings for cell, spot, and each neighbor
3. Construct context tensor by concatenating embeddings
4. Create context mask to handle missing neighbors [deepspot2cell/utils/utils_image.py:435-437]()
5. Call `_forward_single_cell()` with cell embedding and context
6. Return gene expression as dictionary mapping gene names to values

**Sources:** [deepspot2cell/utils/utils_image.py:364-457]()

## Training and Inference in Practice

The complete DeepSpot2Cell workflow integrates training and inference through the tutorial series:

```mermaid
graph TB
    subgraph "Tutorial 2: Training"
        T2_CONFIG["Configure training.yaml<br/>(data paths, hyperparameters)"]
        T2_SCRIPT["Execute train.py<br/>or notebook cells"]
        T2_TRAIN["Training loop<br/>(DS2CDataset + DeepSpot2Cell)"]
        T2_MONITOR["Monitor metrics<br/>(loss, Pearson correlation)"]
        T2_CKPT["Save checkpoint<br/>(best model .pt)"]
    end
    
    subgraph "Tutorial 3: Inference"
        T3_LOAD["Load checkpoint<br/>(torch.load)"]
        T3_MODEL["Restore DeepSpot2Cell"]
        T3_DATA["Prepare images<br/>(WSI or tiles)"]
        T3_INFER["Run inference function<br/>(spot/cell/image paths)"]
        T3_RESULTS["Gene expression predictions"]
        T3_VIZ["Visualize results<br/>(spatial plots)"]
    end
    
    T2_CONFIG --> T2_SCRIPT
    T2_SCRIPT --> T2_TRAIN
    T2_TRAIN --> T2_MONITOR
    T2_MONITOR --> T2_CKPT
    
    T2_CKPT -.checkpoint file.-> T3_LOAD
    T3_LOAD --> T3_MODEL
    T3_MODEL --> T3_INFER
    T3_DATA --> T3_INFER
    T3_INFER --> T3_RESULTS
    T3_RESULTS --> T3_VIZ
```

**Sources:** [README.md:26-33]()

### Command-Line Training

The training script can be executed directly from the command line:

```bash
python scripts/train.py --config configs/training.example.yaml --output runs
```

**Parameters:**
- `--config`: Path to training configuration YAML file (default: `configs/training.example.yaml`)
- `--output`: Directory for checkpoints and logs (default: `runs`)

The script creates an experiment directory at `{output}/{experiment.name}/` containing:
- `checkpoints/`: Model checkpoint files (.pt)
- `logs/`: CSV training logs with metrics per epoch

**Sources:** [scripts/train.py:156-176]()

### Inference from Checkpoints

To use a trained model for inference, load the checkpoint and call the appropriate inference function:

```python
# Load model
checkpoint = torch.load('runs/experiment_name/checkpoints/best.pt')
model = DeepSpot2Cell(...)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Run cell-level inference
predictions = predict_cell_spatial_transcriptomics_from_image_path(
    image_path='path/to/wsi.tif',
    adata=cell_annotations,
    cell_diameter=128,
    radius_neighbors=100,
    preprocess=preprocess,
    morphology_model=pfm,
    model_expression=model,
    device='cuda'
)
```

For detailed usage examples, see [Tutorial 3: Inference and Prediction](#3.3).

**Sources:** [deepspot2cell/utils/utils_image.py:300-361]()

## Key Implementation Details

### Mixed Precision Training

Both training and inference support mixed precision (FP16) for memory efficiency and speed:

**Training:** Configured via `trainer.precision` in YAML [scripts/train.py:146]()
```yaml
trainer:
  precision: 16  # or 32 for full precision
```

**Inference:** Automatic via `torch.autocast` [deepspot2cell/utils/utils_image.py:260]()
```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    with torch.inference_mode():
        # inference code
```

### Batch Processing vs Sequential Inference

**Training:** Processes batches for efficiency via `DataLoader` [scripts/train.py:69-85]()

**Inference:** Typically processes one sample at a time in a loop [deepspot2cell/utils/utils_image.py:262]()
- Enables progress tracking with `tqdm`
- Reduces memory requirements for large WSIs
- Allows early stopping if needed

### Gradient Management

**Training:** Gradients computed and accumulated during backward pass [scripts/train.py:147]()
- Optional gradient clipping via `gradient_clip_val`

**Inference:** Gradients disabled for performance [deepspot2cell/utils/utils_image.py:261]()
- Uses `torch.inference_mode()` (more efficient than `torch.no_grad()`)

**Sources:** [scripts/train.py:113-152](), [deepspot2cell/utils/utils_image.py:260-296]()

## Summary

The training and inference workflows in DeepSpot2Cell are designed to be modular and flexible:

- **Training** uses spot-level supervision through the `train.py` script and `DS2CDataset` class
- **Inference** provides three modes for different use cases: spot-level, cell-level, and image-path-based
- Both workflows share the same multi-scale feature extraction pipeline (cell + spot + neighbors → PFM → embeddings)
- The resolution transfer from spot-supervised training to cell-level inference is the core methodological innovation

For implementation details, see [Training Script](#8.1) and [Inference Workflows](#8.2).

**Sources:** [scripts/train.py](), [deepspot2cell/utils/utils_image.py](), [deepspot2cell/model.py](), [README.md]()

---