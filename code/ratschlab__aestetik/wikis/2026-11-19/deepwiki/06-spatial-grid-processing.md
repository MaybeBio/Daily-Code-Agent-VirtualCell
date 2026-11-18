# Spatial Grid Processing

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [figures/maynard_human_brain_analysis_151676_Transcriptomics_Morphology_AESTETIK.png](figures/maynard_human_brain_analysis_151676_Transcriptomics_Morphology_AESTETIK.png)
- [llms.txt](llms.txt)
- [src/aestetik/utils/utils_grid.py](src/aestetik/utils/utils_grid.py)

</details>



## Purpose and Scope

This document explains the spatial grid construction system in AESTETIK. Spatial grids transform spot-level feature vectors into structured spatial neighborhoods that capture local topology for convolutional processing. This system is a critical component of AESTETIK's multi-modal integration approach.

For information about how spatial grids are used within the broader data pipeline, see [Data Pipeline Components](#5.2). For details on the multi-modal integration that occurs before grid construction, see [Multi-Modal Data Integration](#3.1).

---

## Overview

AESTETIK uses convolutional autoencoders to learn representations from spatial transcriptomics data. Standard neural networks process feature vectors, but spatial transcriptomics data has an additional dimension: the physical location of each spot in tissue. Spatial grid processing bridges this gap by constructing structured 2D grids around each spot that encode neighborhood topology.

### What is a Spatial Grid?

A spatial grid is a fixed-size window (e.g., 5×5 or 7×7) centered on each spot. Each position in the grid contains feature values from the nearest neighboring spot at that spatial location. If no spot exists at a grid position, the position is filled with a representative value (median).

**Input**: Spot-level features (e.g., PCA-reduced transcriptomics or morphology) and spatial coordinates  
**Output**: 3D tensor per spot with shape `(num_features, window_size, window_size)`  
**Purpose**: Enable convolutional operations that leverage spatial topology

Sources: [src/aestetik/utils/utils_grid.py:54-108](), [README.md:29-36]()

---

## Grid Construction Workflow

The following diagram shows the complete grid construction process for a single modality (transcriptomics or morphology):

```mermaid
flowchart TD
    Input["Input: adata.obsm[used_obsm]<br/>adata.obs[x_array, y_array]"]
    Scale["MinMaxScaler normalization<br/>(0, 1) range"]
    BuildTree["Build KDTree per batch<br/>_build_trees()"]
    ComputeOffsets["Compute grid offsets<br/>_compute_offsets_flat()"]
    Parallel["Parallel processing<br/>_create_batch_grid()"]
    CreateSpot["For each spot:<br/>_create_spot()"]
    Query["KDTree query at<br/>each grid position"]
    FillMissing["Fill missing positions<br/>with median"]
    Output["Output: st_grid<br/>(n_spots, dim_emb, W, W)"]
    
    Input --> Scale
    Scale --> BuildTree
    Scale --> ComputeOffsets
    BuildTree --> Parallel
    ComputeOffsets --> Parallel
    Parallel --> CreateSpot
    CreateSpot --> Query
    Query --> FillMissing
    FillMissing --> Output
    
    style Input fill:#f9f9f9
    style Output fill:#f9f9f9
```

Sources: [src/aestetik/utils/utils_grid.py:54-108]()

---

## Main Function: create_st_grid

The entry point for spatial grid construction is `create_st_grid()`:

```mermaid
graph TB
    Function["create_st_grid()"]
    
    Params["Parameters:<br/>adata: AnnData<br/>used_obsm: str<br/>window_size: int<br/>cpu_count: int<br/>used_obs_batch: Optional[str]"]
    
    Extract["Extract data:<br/>x_array, y_array from adata.obs<br/>embs from adata.obsm[used_obsm]"]
    
    Normalize["MinMaxScaler:<br/>embs → [0, 1]"]
    
    BatchCheck{"used_obs_batch<br/>specified?"}
    
    BatchLabels["batch_labels from<br/>adata.obs[used_obs_batch]"]
    
    SingleBatch["batch_labels = zeros<br/>(treat as single slice)"]
    
    BuildTrees["_build_trees()<br/>Create KDTree per batch"]
    
    Offsets["_compute_offsets_flat()<br/>Grid position offsets"]
    
    Freeze["Set arrays read-only<br/>(thread safety)"]
    
    Split["Split spots into<br/>cpu_count batches"]
    
    ParallelCall["Parallel():<br/>_create_batch_grid()<br/>for each batch"]
    
    Concat["Concatenate results"]
    
    Reshape["Moveaxis:<br/>(n, h, w, c) → (n, c, h, w)"]
    
    Return["Return spot_grid"]
    
    Function --> Params
    Params --> Extract
    Extract --> Normalize
    Normalize --> BatchCheck
    BatchCheck -->|"Yes"| BatchLabels
    BatchCheck -->|"No"| SingleBatch
    BatchLabels --> BuildTrees
    SingleBatch --> BuildTrees
    BuildTrees --> Offsets
    Offsets --> Freeze
    Freeze --> Split
    Split --> ParallelCall
    ParallelCall --> Concat
    Concat --> Reshape
    Reshape --> Return
```

Sources: [src/aestetik/utils/utils_grid.py:54-108]()

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `adata` | `anndata.AnnData` | AnnData object containing spatial data |
| `used_obsm` | `str` | Key in `adata.obsm` containing features (e.g., `"X_pca_transcriptomics"`) |
| `window_size` | `int` | Size of spatial grid (e.g., 5 for 5×5 grid) |
| `cpu_count` | `int` | Number of parallel jobs for processing |
| `used_obs_batch` | `Optional[str]` | Key in `adata.obs` for batch labels (tissue slices) |

### Return Value

Returns `np.ndarray` with shape `(num_spots, dim_emb, window_size, window_size)` where:
- `num_spots`: Total number of spots across all batches
- `dim_emb`: Feature dimensionality (e.g., number of PCA components)
- `window_size`: Grid dimensions

Sources: [src/aestetik/utils/utils_grid.py:54-65]()

---

## Spatial Indexing with KDTree

### Building Trees per Batch

The `_build_trees()` function creates a separate KDTree for each batch (tissue slice). This ensures that grid queries only find neighbors within the same tissue section.

```mermaid
graph LR
    Input["x_array, y_array<br/>batch_labels"]
    
    UniqueIDs["Get unique<br/>batch IDs"]
    
    Loop["For each batch_id"]
    
    Filter["Filter spots:<br/>batch_labels == batch_id"]
    
    Stack["Stack coordinates:<br/>np.column_stack([x, y])"]
    
    Tree["KDTree(coords)"]
    
    StoreTrees["trees[batch_id] = tree"]
    
    StoreIndices["batch_to_indices[batch_id]<br/>= spot_indices"]
    
    Return["Return trees,<br/>batch_to_indices"]
    
    Input --> UniqueIDs
    UniqueIDs --> Loop
    Loop --> Filter
    Filter --> Stack
    Stack --> Tree
    Tree --> StoreTrees
    Filter --> StoreIndices
    StoreTrees --> Return
    StoreIndices --> Return
```

**Data Structures**:
- `trees`: `Dict[int, KDTree]` - Mapping from batch ID to spatial index
- `batch_to_indices`: `Dict[int, np.ndarray]` - Mapping from batch ID to global spot indices

This separation is critical for multi-sample processing: spots from different tissue sections should not contribute to each other's grids, even if they have similar coordinates.

Sources: [src/aestetik/utils/utils_grid.py:110-123]()

---

## Grid Construction for Individual Spots

### The _create_spot Function

For each spot, `_create_spot()` constructs a grid by querying the KDTree at regular spatial offsets:

```mermaid
flowchart TD
    Start["Input: spot_idx"]
    
    GetCenter["center = (x_array[spot_idx],<br/>y_array[spot_idx])<br/>batch_id = batch_labels[spot_idx]"]
    
    InitGrid["Initialize grid:<br/>shape = (W, W, dim_emb)<br/>fill_value = np.nan"]
    
    GetIndices["indices_in_batch =<br/>batch_to_indices[batch_id]"]
    
    LoopOffsets["For each (dx, dy)<br/>in offsets_flat"]
    
    ComputePos["position = center +<br/>[dx, dy]"]
    
    Query["distance, neighbor_idx =<br/>tree[batch_id].query(position)"]
    
    CheckDist{"distance > 0?"}
    
    Skip["Continue to<br/>next offset"]
    
    Unravel["grid_row, grid_col =<br/>unravel_index(offset_idx)"]
    
    Assign["grid[row, col] =<br/>embs[indices_in_batch[neighbor_idx]]"]
    
    ComputeMedian["median_spot =<br/>nanmedian(grid, axis=(0,1))"]
    
    FillNaN["Fill NaN positions<br/>with median_spot"]
    
    Return["Return grid"]
    
    Start --> GetCenter
    GetCenter --> InitGrid
    InitGrid --> GetIndices
    GetIndices --> LoopOffsets
    LoopOffsets --> ComputePos
    ComputePos --> Query
    Query --> CheckDist
    CheckDist -->|"Yes (no exact match)"| Skip
    CheckDist -->|"No (exact match)"| Unravel
    Skip --> LoopOffsets
    Unravel --> Assign
    Assign --> LoopOffsets
    LoopOffsets -->|"Done"| ComputeMedian
    ComputeMedian --> FillNaN
    FillNaN --> Return
```

Sources: [src/aestetik/utils/utils_grid.py:148-182]()

### Grid Offset Computation

The `_compute_offsets_flat()` function generates all grid positions relative to the center spot:

| Window Size | Half | Range | Total Positions |
|-------------|------|-------|-----------------|
| 5 | 2 | -2 to 2 | 25 |
| 7 | 3 | -3 to 3 | 49 |
| 9 | 4 | -4 to 4 | 81 |

For a 5×5 grid centered at spot (10, 10), the function generates positions:
```
(8, 8), (8, 9), (8, 10), (8, 11), (8, 12),
(9, 8), (9, 9), (9, 10), (9, 11), (9, 12),
...
(12, 8), (12, 9), (12, 10), (12, 11), (12, 12)
```

Sources: [src/aestetik/utils/utils_grid.py:184-188]()

---

## Missing Value Imputation

Not all grid positions have corresponding spots in the tissue. AESTETIK fills missing positions with the median feature values from the populated positions in that grid:

```mermaid
graph LR
    Grid["Grid with NaN values"]
    
    Median["Compute median per feature:<br/>median_spot = nanmedian(grid, axis=(0,1))"]
    
    FindNaN["Find NaN positions:<br/>nan_indices = np.where(np.isnan(grid))"]
    
    Fill["Fill NaN positions:<br/>grid[nan_indices] =<br/>median_spot[nan_indices[1]]"]
    
    Complete["Complete grid"]
    
    Grid --> Median
    Median --> FindNaN
    FindNaN --> Fill
    Fill --> Complete
```

This imputation strategy:
- Uses the local neighborhood context (median of existing neighbors)
- Preserves feature-wise statistics
- Avoids introducing artifacts from global statistics

Sources: [src/aestetik/utils/utils_grid.py:178-180]()

---

## Parallel Processing

Grid construction is computationally intensive (querying KDTree for every position of every spot). AESTETIK uses `joblib.Parallel` to distribute work:

```mermaid
graph TB
    subgraph "Main Process"
        Split["Split spots into<br/>cpu_count batches"]
        Freeze["Set arrays read-only<br/>(thread safety)"]
        Create["Create delayed function:<br/>delayed(_create_batch_grid)"]
        Launch["Launch parallel jobs"]
        Concat["Concatenate results"]
    end
    
    subgraph "Worker 1"
        W1["Process batch 1:<br/>spots 0-999"]
        W1Grid["_create_batch_grid()"]
        W1Return["Return grids"]
    end
    
    subgraph "Worker 2"
        W2["Process batch 2:<br/>spots 1000-1999"]
        W2Grid["_create_batch_grid()"]
        W2Return["Return grids"]
    end
    
    subgraph "Worker N"
        WN["Process batch N:<br/>spots 9000-9999"]
        WNGrid["_create_batch_grid()"]
        WNReturn["Return grids"]
    end
    
    Split --> Freeze
    Freeze --> Create
    Create --> Launch
    Launch --> W1
    Launch --> W2
    Launch --> WN
    W1 --> W1Grid --> W1Return
    W2 --> W2Grid --> W2Return
    WN --> WNGrid --> WNReturn
    W1Return --> Concat
    W2Return --> Concat
    WNReturn --> Concat
```

**Key implementation details**:
- `batch_indices = np.array_split(range(n_spots), cpu_count)` divides spots evenly
- `prefer="threads"` uses thread-based parallelism (more efficient for numpy operations)
- Arrays are marked read-only with `setflags(write=False)` for thread safety
- `delayed()` creates deferred function calls executed in parallel

Sources: [src/aestetik/utils/utils_grid.py:85-106]()

---

## Feature Normalization

Before grid construction, features are normalized using `MinMaxScaler` to the range [0, 1]:

```python
embs = MinMaxScaler(feature_range=(0, 1)).fit_transform(embs)
```

**Rationale**:
- Ensures all features have comparable scales
- Prevents features with larger magnitudes from dominating
- Stabilizes convolutional operations in the autoencoder
- Makes median imputation more meaningful across features

Sources: [src/aestetik/utils/utils_grid.py:65]()

---

## Output Format and Axis Transformation

The grid construction produces output with shape `(num_spots, window_size, window_size, dim_emb)`, but PyTorch convolutions expect channels-first format `(batch, channels, height, width)`. The final step transforms the axes:

```python
spot_grid = np.moveaxis(spot_grid, 3, 1)  
# (num_spots, window_size, window_size, dim_emb) → (num_spots, dim_emb, window_size, window_size)
```

This matches the expected input format for the convolutional autoencoder.

Sources: [src/aestetik/utils/utils_grid.py:107-108]()

---

## Integration with Data Pipeline

The spatial grid construction integrates into the broader AESTETIK data processing pipeline:

```mermaid
sequenceDiagram
    participant DataModule as "AESTETIKDataModule"
    participant UtilsData as "utils_data.prepare_input_for_model()"
    participant UtilsGrid as "utils_grid.create_st_grid()"
    participant UtilsClustering as "utils_clustering"
    
    DataModule->>UtilsData: prepare_input_for_model()
    activate UtilsData
    
    Note over UtilsData: Extract PCA features
    
    UtilsData->>UtilsClustering: Cluster transcriptomics
    UtilsClustering-->>UtilsData: cluster labels
    
    UtilsData->>UtilsClustering: Cluster morphology
    UtilsClustering-->>UtilsData: cluster labels
    
    Note over UtilsData: Calibrate weights
    
    UtilsData->>UtilsGrid: create_st_grid(adata,<br/>used_obsm="X_pca_transcriptomics",<br/>window_size=5)
    activate UtilsGrid
    Note over UtilsGrid: Build spatial grids
    UtilsGrid-->>UtilsData: X_st_grid_transcriptomics
    deactivate UtilsGrid
    
    UtilsData->>UtilsGrid: create_st_grid(adata,<br/>used_obsm="X_pca_morphology",<br/>window_size=5)
    activate UtilsGrid
    Note over UtilsGrid: Build spatial grids
    UtilsGrid-->>UtilsData: X_st_grid_morphology
    deactivate UtilsGrid
    
    Note over UtilsData: Concatenate grids
    
    UtilsData-->>DataModule: Combined grids + metadata
    deactivate UtilsData
```

**Key points**:
1. Grid construction occurs **after** clustering (needed for triplet loss labels)
2. Grids are built **separately** for transcriptomics and morphology
3. Both modalities use the **same spatial coordinates** (`x_array`, `y_array`)
4. Grids are later **concatenated** along the channel dimension
5. The combined grid becomes input to the autoencoder

Sources: [src/aestetik/utils/utils_grid.py:54-108](), High-level diagrams (Diagram 4)

---

## Code Entity Reference

### Main Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `create_st_grid()` | [utils_grid.py:54-108]() | Main entry point for grid construction |
| `_build_trees()` | [utils_grid.py:110-123]() | Build KDTree spatial indices per batch |
| `_create_batch_grid()` | [utils_grid.py:125-146]() | Process a batch of spots in parallel |
| `_create_spot()` | [utils_grid.py:148-182]() | Create grid for a single spot |
| `_compute_offsets_flat()` | [utils_grid.py:184-188]() | Generate grid position offsets |

### Data Flow

```mermaid
graph LR
    AnnData["adata.obsm[used_obsm]<br/>adata.obs[x_array]<br/>adata.obs[y_array]"]
    
    Function["create_st_grid()"]
    
    Output["np.ndarray<br/>(n_spots, dim_emb, W, W)"]
    
    Storage["Stored in adata.obsm<br/>as 'X_st_grid'"]
    
    Model["Fed to CustomDataset<br/>→ DataLoader<br/>→ Autoencoder"]
    
    AnnData --> Function
    Function --> Output
    Output --> Storage
    Storage --> Model
```

Sources: [src/aestetik/utils/utils_grid.py:1-188]()

---

## Performance Considerations

### Computational Complexity

For `N` spots, window size `W`, and feature dimension `D`:
- **KDTree construction**: O(N log N) per batch
- **Grid queries**: O(W² log N) per spot
- **Total**: O(N × W² log N)

### Memory Usage

Each grid requires `W × W × D × dtype_size` bytes. For a typical configuration:
- Window size: 5×5 = 25 positions
- Features: 50 dimensions
- Data type: float32 (4 bytes)
- **Per spot**: 25 × 50 × 4 = 5,000 bytes ≈ 5 KB
- **For 10,000 spots**: 50 MB

### Optimization Strategies

1. **Parallel processing**: Distributes work across CPU cores
2. **Read-only arrays**: Prevents memory copying in threads
3. **KDTree indexing**: O(log N) queries vs O(N) linear search
4. **Batch-wise separation**: Reduces search space for multi-sample data

Sources: [src/aestetik/utils/utils_grid.py:91-105]()

---

## Thread Safety

The implementation ensures thread safety through immutability:

```python
x_array.setflags(write=False)
y_array.setflags(write=False)
embs.setflags(write=False)
batch_labels.setflags(write=False)
offsets_flat.setflags(write=False)
```

This prevents race conditions when multiple threads access the same arrays during parallel processing.

Sources: [src/aestetik/utils/utils_grid.py:85-89]()

---