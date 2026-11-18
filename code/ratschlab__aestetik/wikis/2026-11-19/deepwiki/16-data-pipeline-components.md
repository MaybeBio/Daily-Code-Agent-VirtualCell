# Data Pipeline Components

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/aestetik/data_modules/data_module.py](src/aestetik/data_modules/data_module.py)
- [src/aestetik/utils/utils_data.py](src/aestetik/utils/utils_data.py)
- [src/aestetik/utils/utils_grid.py](src/aestetik/utils/utils_grid.py)

</details>



## Purpose and Scope

This page provides technical reference documentation for the data pipeline components that prepare spatial transcriptomics data for model training. The pipeline transforms raw AnnData objects into batched spatial grid tensors suitable for the autoencoder architecture.

This document covers:
- `AESTETIKDataModule`: PyTorch Lightning DataModule for data orchestration
- Data preparation utilities in `utils_data.py`: clustering, weight calibration, and preprocessing coordination
- Spatial grid construction in `utils_grid.py`: KDTree-based neighborhood feature extraction

For information about the main AESTETIK orchestrator class that initializes these components, see [AESTETIK Class](#5.1). For details about the model architecture that consumes the prepared data, see [Model Architecture](#5.3). For clustering algorithms used in preprocessing, see [Utility Functions](#5.4).

---

## Component Overview

The data pipeline consists of three primary layers that transform raw spatial omics data into training-ready batches:

```mermaid
graph TB
    subgraph "Layer 1: Orchestration"
        DataModule["AESTETIKDataModule<br/>(L.LightningDataModule)"]
    end
    
    subgraph "Layer 2: Preprocessing Coordination"
        PrepareInput["prepare_input_for_model()"]
        BuildGrid["build_grid()"]
        Calibrate["calibrate_transcriptomics_<br/>morphology_ratio()"]
    end
    
    subgraph "Layer 3: Spatial Grid Construction"
        CreateSTGrid["create_st_grid()"]
        BuildTrees["_build_trees()"]
        CreateBatchGrid["_create_batch_grid()"]
        CreateSpot["_create_spot()"]
    end
    
    subgraph "External Dependencies"
        Clustering["clustering()<br/>(utils_clustering)"]
        CustomDS["CustomDataset<br/>(dataloader.py)"]
    end
    
    subgraph "Input/Output"
        Input["AnnData<br/>Raw Features"]
        Output["DataLoader<br/>Batched Grids"]
    end
    
    Input --> DataModule
    DataModule --> PrepareInput
    PrepareInput --> Clustering
    PrepareInput --> Calibrate
    PrepareInput --> BuildGrid
    BuildGrid --> CreateSTGrid
    CreateSTGrid --> BuildTrees
    CreateSTGrid --> CreateBatchGrid
    CreateBatchGrid --> CreateSpot
    DataModule --> CustomDS
    CustomDS --> Output
    
    style DataModule fill:#f9f9f9,stroke:#333,stroke-width:2px
    style PrepareInput fill:#f9f9f9,stroke:#333,stroke-width:2px
    style CreateSTGrid fill:#f9f9f9,stroke:#333,stroke-width:2px
```

**Sources**: [src/aestetik/data_modules/data_module.py:1-159](), [src/aestetik/utils/utils_data.py:1-112](), [src/aestetik/utils/utils_grid.py:1-188]()

---

## AESTETIKDataModule Class

### Class Definition

`AESTETIKDataModule` is a PyTorch Lightning DataModule that orchestrates the complete data preparation pipeline and provides dataloaders for training and validation.

**File Location**: [src/aestetik/data_modules/data_module.py:12-159]()

### Initialization Parameters

The class is initialized with structured parameter dictionaries:

| Parameter | Type | Description |
|-----------|------|-------------|
| `adata` | `anndata.AnnData` | Input AnnData object containing transcriptomics and morphology features |
| `validation_split` | `float` | Proportion of data for validation (0.0 to 1.0) |
| `used_obsm_transcriptomics` | `str` | Key in `adata.obsm` for transcriptomics features (e.g., PCA) |
| `used_obsm_morphology` | `str` | Key in `adata.obsm` for morphology features |
| `used_obsm_combined` | `str` | Key to store concatenated features |
| `used_obs_batch` | `Optional[str]` | Key in `adata.obs` for batch labels (for multi-sample data) |
| `dataloader_params` | `dict` | Contains `batch_size`, `num_workers` |
| `clustering_params` | `dict` | Contains `nCluster`, `clustering_method`, `n_neighbors` |
| `grid_params` | `dict` | Contains `morphology_dim` (window size) |
| `loss_regularization_params` | `dict` | Contains `multi_triplet_loss`, `n_repeats`, `morphology_weight`, `transcriptomics_weight`, `total_weight` |
| `data_handling_params` | `dict` | Contains `n_jobs`, `train_size` |

**Sources**: [src/aestetik/data_modules/data_module.py:13-93]()

### Key Methods

#### setup()

The `setup()` method triggers the complete preprocessing pipeline:

```mermaid
sequenceDiagram
    participant DM as "DataModule.setup()"
    participant Prep as "prepare_input_for_model()"
    participant CDS as "CustomDataset"
    participant Split as "random_split()"
    
    DM->>Prep: "Pass all parameters"
    Note over Prep: "1. Cluster transcriptomics"
    Note over Prep: "2. Cluster morphology"
    Note over Prep: "3. Calibrate weights"
    Note over Prep: "4. Build spatial grids"
    Prep-->>DM: "Return adjusted weights"
    
    DM->>CDS: "Create dataset with grids"
    Note over CDS: "Stores X_st_grid from obsm"
    
    alt "validation_split > 0"
        DM->>Split: "Split into train/val"
        Split-->>DM: "train_dataset, val_dataset"
    else "validation_split == 0"
        Note over DM: "Use all data for training"
    end
```

**Sources**: [src/aestetik/data_modules/data_module.py:96-134]()

**Key Operations**:
1. Calls `prepare_input_for_model()` with all configuration parameters [line 98-108]()
2. Receives calibrated `transcriptomics_weight` and `morphology_weight` [line 98]()
3. Creates `CustomDataset` instance with processed data [line 109-114]()
4. Splits dataset into training and validation sets if `validation_split > 0` [line 116-134]()

#### train_dataloader()

Returns a PyTorch DataLoader for training data:

```python
# Returns: torch.utils.data.DataLoader
# Configuration: uses self.dataloader_params (batch_size, num_workers)
# Shuffle: True
```

**Sources**: [src/aestetik/data_modules/data_module.py:137-140]()

#### val_dataloader()

Returns a PyTorch DataLoader for validation data, or empty iterator if no validation split:

```python
# Returns: torch.utils.data.DataLoader or empty iterator
# Configuration: uses self.dataloader_params
# Shuffle: False
```

**Sources**: [src/aestetik/data_modules/data_module.py:142-145]()

#### _validate_params()

Validates that all required parameter keys are present in the configuration dictionaries:

| Dictionary | Required Keys |
|------------|---------------|
| `dataloader_params` | `batch_size`, `num_workers` |
| `clustering_params` | `nCluster`, `clustering_method`, `n_neighbors` |
| `grid_params` | `morphology_dim` |
| `loss_regularization_params` | `multi_triplet_loss`, `n_repeats`, `morphology_weight`, `transcriptomics_weight` |
| `data_handling_params` | `n_jobs`, `train_size` |

**Sources**: [src/aestetik/data_modules/data_module.py:147-159]()

---

## Data Preparation Functions (utils_data.py)

### prepare_input_for_model()

**Function Signature**: [src/aestetik/utils/utils_data.py:14-72]()

The main preprocessing orchestrator that coordinates clustering, weight calibration, and grid construction.

**Preprocessing Pipeline**:

```mermaid
graph TD
    Start["Input: AnnData with<br/>transcriptomics & morphology<br/>in adata.obsm"]
    
    ClusterT["clustering()<br/>transcriptomics modality"]
    ClusterM["clustering()<br/>morphology modality"]
    
    Calibrate["calibrate_transcriptomics_<br/>morphology_ratio()"]
    
    Concat["Concatenate modalities<br/>into used_obsm_combined"]
    
    BuildG["build_grid()<br/>Create spatial grids"]
    
    End["Output: Adjusted weights<br/>Grid stored in adata.obsm"]
    
    Start --> ClusterT
    Start --> ClusterM
    ClusterT --> Calibrate
    ClusterM --> Calibrate
    Calibrate --> Concat
    Concat --> BuildG
    BuildG --> End
```

**Sources**: [src/aestetik/utils/utils_data.py:14-72]()

**Parameters**:
- `adata`: AnnData object to preprocess
- `nCluster`: Number of clusters (int) or resolution (float)
- `used_obsm_transcriptomics`: Key for transcriptomics features in `adata.obsm`
- `used_obsm_morphology`: Key for morphology features in `adata.obsm`
- `used_obsm_combined`: Key to store concatenated features
- `clustering_method`: One of `"bgm"`, `"kmeans"`, `"louvain"`, `"leiden"`
- `n_neighbors`: Number of neighbors for spatial cluster refinement
- `window_size`: Grid window dimension (e.g., 5 for 5×5 grids)
- `n_jobs`: Number of parallel jobs for grid construction
- `total_weight`: Total loss weight across modalities
- `morphology_weight`: Initial weight for morphology modality
- `used_obs_batch`: Optional batch column key in `adata.obs`

**Returns**: 
- `transcriptomics_weight`: Calibrated weight for transcriptomics (may be 0 if clustering failed)
- `morphology_weight`: Calibrated weight for morphology (may be 0 if clustering failed)

**Side Effects**:
- Stores cluster labels in `adata.obs[f"{used_obsm}_cluster"]` for each modality [lines 35-51]()
- Creates concatenated features in `adata.obsm[used_obsm_combined]` [lines 62-63]()
- Stores spatial grids in `adata.obsm["X_st_grid"]` via `build_grid()` [lines 65-70]()

### build_grid()

**Function Signature**: [src/aestetik/utils/utils_data.py:74-89]()

Coordinates spatial grid construction for both modalities and concatenates results.

**Processing Flow**:

```mermaid
graph LR
    Trans["Transcriptomics<br/>Features"]
    Morph["Morphology<br/>Features"]
    
    GridT["create_st_grid()<br/>Transcriptomics Grid"]
    GridM["create_st_grid()<br/>Morphology Grid"]
    
    Concat["np.concatenate()<br/>axis=1"]
    
    Store["adata.obsm['X_st_grid']"]
    
    Trans --> GridT
    Morph --> GridM
    GridT --> Concat
    GridM --> Concat
    Concat --> Store
```

**Sources**: [src/aestetik/utils/utils_data.py:74-89]()

**Key Operations**:
1. Calls `create_st_grid()` for transcriptomics features [lines 81-82]()
2. Calls `create_st_grid()` for morphology features [lines 85-86]()
3. Concatenates grids along channel axis [lines 88-89]()
4. Final shape: `(n_spots, transcriptomics_dim + morphology_dim, window_size, window_size)`

### calibrate_transcriptomics_morphology_ratio()

**Function Signature**: [src/aestetik/utils/utils_data.py:91-112]()

Adjusts modality weights when clustering fails to produce the expected number of clusters.

**Calibration Logic**:

```mermaid
graph TD
    Start["Input: Initial weights<br/>and cluster results"]
    
    CheckT{"Transcriptomics<br/>resulted in<br/>1 cluster?"}
    CheckM{"Morphology<br/>resulted in<br/>1 cluster?"}
    
    SetT["Set transcriptomics_weight = 0"]
    SetM["Set morphology_weight = 0"]
    
    Keep["Keep original weights"]
    
    End["Return adjusted weights"]
    
    Start --> CheckT
    CheckT -->|"Yes & weight > 0"| SetT
    CheckT -->|"No"| CheckM
    SetT --> CheckM
    
    CheckM -->|"Yes & weight > 0"| SetM
    CheckM -->|"No"| Keep
    SetM --> End
    Keep --> End
```

**Sources**: [src/aestetik/utils/utils_data.py:91-112]()

**Rationale**: When a modality produces only 1 cluster instead of the requested `nCluster`, it indicates the features are not discriminative for spatial domain identification. Setting the weight to 0 prevents this modality from contributing to triplet loss, allowing the other modality to guide learning.

**Implementation Details**:
- Checks if `adata.obs[f"{used_obsm}_cluster"].unique().size == 1` [lines 101, 107]()
- Sets weight to 0 if condition is met and original weight was positive [lines 103-104, 109-110]()
- Logs weight adjustments for transparency [lines 102-103, 108-109]()

---

## Spatial Grid Construction (utils_grid.py)

### create_st_grid()

**Function Signature**: [src/aestetik/utils/utils_grid.py:54-108]()

Constructs spatial grids for all spots using KDTree-based neighbor queries and parallel processing.

**Architecture Overview**:

```mermaid
graph TB
    Input["Input:<br/>adata.obsm[used_obsm]<br/>adata.obs['x_array', 'y_array']"]
    
    Scale["MinMaxScaler<br/>Scale features to [0,1]"]
    
    Batch["Build KDTrees per batch<br/>_build_trees()"]
    
    Parallel["Parallel processing<br/>Split spots across n_jobs"]
    
    CreateBatch["_create_batch_grid()<br/>Process spot subset"]
    
    CreateSpot["_create_spot()<br/>Per-spot grid construction"]
    
    Concat["Concatenate results<br/>Reshape to (N,C,H,W)"]
    
    Output["Output:<br/>np.ndarray<br/>(n_spots, dim, window_size, window_size)"]
    
    Input --> Scale
    Scale --> Batch
    Batch --> Parallel
    Parallel --> CreateBatch
    CreateBatch --> CreateSpot
    CreateSpot --> Concat
    Concat --> Output
```

**Sources**: [src/aestetik/utils/utils_grid.py:54-108]()

**Key Processing Steps**:

1. **Feature Scaling** [lines 64-65]():
   - Applies `MinMaxScaler` with range `(0, 1)` to all features
   - Ensures consistent feature magnitudes across modalities

2. **Batch Handling** [lines 69-80]():
   - Extracts `used_obs_batch` column if specified
   - Converts to categorical codes for efficient indexing
   - Treats all data as single batch if no batch column exists

3. **KDTree Construction** [lines 78-80]():
   - Calls `_build_trees()` to create spatial indices per batch
   - Returns `trees` dict and `batch_to_indices` mapping

4. **Offset Computation** [lines 82-83]():
   - Calls `_compute_offsets_flat()` to generate grid position offsets
   - For `window_size=5`, creates 25 (x,y) offsets from center

5. **Parallel Processing** [lines 91-105]():
   - Splits spots into `cpu_count` chunks
   - Uses `joblib.Parallel` with `prefer="threads"`
   - Passes immutable (write-protected) arrays to workers [lines 85-89]()

6. **Grid Assembly** [lines 106-107]():
   - Concatenates spot grids from all workers
   - Reshapes from `(N, H, W, C)` to `(N, C, H, W)` using `np.moveaxis()`

**Return Shape**: `(num_spots, dim_emb, window_size, window_size)`

### _build_trees()

**Function Signature**: [src/aestetik/utils/utils_grid.py:110-123]()

Creates KDTree spatial indices for efficient neighbor queries, handling multi-batch data.

**Per-Batch Processing**:

```mermaid
graph TD
    BatchLabels["Input: batch_labels<br/>(one per spot)"]
    
    UniqueBatches["Extract unique batch IDs"]
    
    Loop["For each batch_id"]
    
    Filter["Filter spots:<br/>spot_indices = where(batch_labels == batch_id)"]
    
    Stack["Stack coordinates:<br/>np.column_stack([x_array, y_array])"]
    
    BuildTree["KDTree(coords)"]
    
    Store["trees[batch_id] = tree<br/>batch_to_indices[batch_id] = spot_indices"]
    
    BatchLabels --> UniqueBatches
    UniqueBatches --> Loop
    Loop --> Filter
    Filter --> Stack
    Stack --> BuildTree
    BuildTree --> Store
    Store --> Loop
```

**Sources**: [src/aestetik/utils/utils_grid.py:110-123]()

**Purpose**: Separating batches ensures spots from different tissue samples or experimental batches don't query neighbors across batch boundaries, preserving spatial coherence within each sample.

### _create_batch_grid()

**Function Signature**: [src/aestetik/utils/utils_grid.py:125-146]()

Processes a subset of spots assigned to a single parallel worker.

**Implementation**:
- Iterates over assigned `spot_indices` [line 135]()
- Calls `_create_spot()` for each spot [lines 136-144]()
- Collects results in list for efficient concatenation [line 145]()

**Sources**: [src/aestetik/utils/utils_grid.py:125-146]()

### _create_spot()

**Function Signature**: [src/aestetik/utils/utils_grid.py:148-182]()

Constructs the spatial feature grid for a single spot using neighbor queries and median imputation.

**Grid Construction Algorithm**:

```mermaid
graph TD
    Start["Input: spot_idx, spatial coords, features"]
    
    Init["Initialize grid:<br/>np.full((H, W, C), np.nan)"]
    
    GetBatch["Extract batch_id for spot"]
    
    GetTree["Get KDTree for this batch"]
    
    Loop["For each offset (dx, dy)"]
    
    Query["Query position = center + offset<br/>using tree.query()"]
    
    CheckDist{"distance > 0?"}
    
    Skip["Continue to next offset"]
    
    Place["Place neighbor features<br/>in grid[row, col]"]
    
    Median["Compute median across grid<br/>for each feature"]
    
    Impute["Fill NaN positions<br/>with median values"]
    
    End["Return completed grid"]
    
    Start --> Init
    Init --> GetBatch
    GetBatch --> GetTree
    GetTree --> Loop
    Loop --> Query
    Query --> CheckDist
    CheckDist -->|"Yes (no exact match)"| Skip
    CheckDist -->|"No (exact match)"| Place
    Skip --> Loop
    Place --> Loop
    Loop --> Median
    Median --> Impute
    Impute --> End
```

**Sources**: [src/aestetik/utils/utils_grid.py:148-182]()

**Key Implementation Details**:

1. **Grid Initialization** [lines 163-165]():
   - Shape: `(window_size, window_size, dim_emb)`
   - Filled with `np.nan` to track missing positions

2. **Neighbor Query** [lines 168-172]():
   - Computes target position: `center + [dx_offset, dy_offset]`
   - Uses `tree.query(position)` to find nearest neighbor
   - Only accepts exact matches (`distance == 0`)
   - Skips positions without exact coordinate matches

3. **Median Imputation** [lines 178-180]():
   - Computes `np.nanmedian(grid, axis=(0, 1))` across spatial dimensions
   - Replaces all `NaN` entries with corresponding feature medians
   - Ensures complete grids even for edge spots with fewer neighbors

**Rationale for Distance Threshold**: The strict `distance > 0` check [line 171]() ensures only spots at exact grid positions are included, preserving the regular spatial lattice structure typical of spatial transcriptomics platforms.

### _compute_offsets_flat()

**Function Signature**: [src/aestetik/utils/utils_grid.py:184-188]()

Generates flattened (x, y) offset arrays for grid positions.

**Example for window_size=5**:
```python
# start = -2, end = 3
# Returns shape (25, 2):
[[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
 [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
 ...
 [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]]
```

**Sources**: [src/aestetik/utils/utils_grid.py:184-188]()

---

## Data Flow Integration

### Complete Pipeline Data Flow

```mermaid
graph TB
    subgraph "Input Stage"
        A["AnnData Object<br/>adata.obsm['X_pca_transcriptomics']<br/>adata.obsm['X_pca_morphology']<br/>adata.obs['x_array', 'y_array']"]
    end
    
    subgraph "AESTETIKDataModule.__init__()"
        B["Store parameters:<br/>- used_obsm keys<br/>- clustering_params<br/>- grid_params<br/>- loss_regularization_params"]
    end
    
    subgraph "DataModule.setup()"
        C["prepare_input_for_model()"]
        
        C1["clustering()<br/>for transcriptomics"]
        C2["clustering()<br/>for morphology"]
        C3["calibrate_transcriptomics_<br/>morphology_ratio()"]
        C4["build_grid()"]
        
        C4a["create_st_grid()<br/>for transcriptomics"]
        C4b["create_st_grid()<br/>for morphology"]
        
        C4a1["MinMaxScaler"]
        C4a2["_build_trees()"]
        C4a3["_create_batch_grid()"]
        C4a4["_create_spot()"]
        
        D["CustomDataset()"]
        E["random_split()"]
    end
    
    subgraph "Output Stage"
        F["train_dataloader()"]
        G["val_dataloader()"]
        H["Batched Tensors<br/>(N, C, H, W)"]
    end
    
    A --> B
    B --> C
    C --> C1
    C --> C2
    C1 --> C3
    C2 --> C3
    C3 --> C4
    C4 --> C4a
    C4 --> C4b
    C4a --> C4a1
    C4a1 --> C4a2
    C4a2 --> C4a3
    C4a3 --> C4a4
    C4b --> C4a1
    C4a --> D
    C4b --> D
    D --> E
    E --> F
    E --> G
    F --> H
    G --> H
    
    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style C fill:#f9f9f9,stroke:#333,stroke-width:2px
    style D fill:#f9f9f9,stroke:#333,stroke-width:2px
    style H fill:#f9f9f9,stroke:#333,stroke-width:2px
```

**Sources**: [src/aestetik/data_modules/data_module.py:1-159](), [src/aestetik/utils/utils_data.py:1-112](), [src/aestetik/utils/utils_grid.py:1-188]()

### Key Data Transformations

| Stage | Input Shape | Output Shape | Storage Location |
|-------|-------------|--------------|------------------|
| Raw Features | `(n_spots, n_features)` | - | `adata.obsm[used_obsm_*]` |
| Clustering | - | `(n_spots,)` | `adata.obs[f"{used_obsm}_cluster"]` |
| Feature Scaling | `(n_spots, n_features)` | `(n_spots, n_features)` | In-memory |
| create_st_grid() | `(n_spots, n_features)` | `(n_spots, n_features, H, W)` | In-memory |
| Concatenate Grids | 2×`(n_spots, n_features, H, W)` | `(n_spots, 2*n_features, H, W)` | `adata.obsm["X_st_grid"]` |
| CustomDataset | - | `(batch_size, 2*n_features, H, W)` | DataLoader batches |

### Memory and Performance Considerations

**Parallelization**:
- Grid construction uses `joblib.Parallel` with `prefer="threads"` [line 94-95]()
- Number of workers controlled by `n_jobs` parameter
- Arrays are made read-only before parallel processing [lines 85-89]()

**Memory Efficiency**:
- Uses `np.float32` for features to reduce memory footprint [line 23]()
- Processes spots in chunks for parallel workers
- KDTree provides O(log n) neighbor queries vs O(n) brute force

**Batch Processing**:
- Separate KDTrees per batch prevent cross-batch neighbor queries
- Enables processing of multi-sample datasets
- Maintains spatial coherence within each tissue section

**Sources**: [src/aestetik/utils/utils_grid.py:29-108]()

---

## Usage Example

```python
from aestetik.data_modules.data_module import AESTETIKDataModule
import anndata as ad

# Load preprocessed AnnData with PCA features
adata = ad.read_h5ad("spatial_data.h5ad")

# Configure data module
data_module = AESTETIKDataModule(
    adata=adata,
    validation_split=0.1,
    used_obsm_transcriptomics="X_pca_transcriptomics",
    used_obsm_morphology="X_pca_morphology", 
    used_obsm_combined="X_pca_combined",
    used_obs_batch="batch_id",
    dataloader_params={"batch_size": 32, "num_workers": 4},
    clustering_params={"nCluster": 10, "clustering_method": "leiden", "n_neighbors": 15},
    grid_params={"morphology_dim": 5},
    loss_regularization_params={
        "multi_triplet_loss": True,
        "n_repeats": 3,
        "morphology_weight": 1.0,
        "transcriptomics_weight": None,  # Will be calibrated
        "total_weight": 2.0
    },
    data_handling_params={"n_jobs": 8, "train_size": None}
)

# Trigger preprocessing
data_module.setup(stage="fit")

# Access processed data
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Check results stored in AnnData
print(adata.obs.columns)  # Contains cluster labels
print(adata.obsm["X_st_grid"].shape)  # (n_spots, 2*dim, 5, 5)
```

**Sources**: [src/aestetik/data_modules/data_module.py:12-159]()

---