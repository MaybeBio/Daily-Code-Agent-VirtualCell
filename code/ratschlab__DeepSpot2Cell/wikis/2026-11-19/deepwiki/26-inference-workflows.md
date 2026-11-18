# Inference Workflows

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [deepspot2cell/model.py](deepspot2cell/model.py)
- [deepspot2cell/utils/utils_image.py](deepspot2cell/utils/utils_image.py)
- [tutorials/GettingStartedWithDeepSpot2Cell_3_inference.ipynb](tutorials/GettingStartedWithDeepSpot2Cell_3_inference.ipynb)

</details>



This page documents the different inference workflows available in DeepSpot2Cell for predicting gene expression from H&E images. DeepSpot2Cell provides three distinct inference approaches: spot-level prediction, cell-level prediction, and DeepSpot2Cell-specific inference from image paths. Each workflow is designed for different use cases and operates at different spatial resolutions.

For information about training the model, see [Training Script](#8.1). For information about preprocessing data before inference, see [Tutorial 1: Data Preprocessing](#3.1).

## Overview of Inference Modes

DeepSpot2Cell supports two fundamental prediction modes built into the model architecture, plus specialized functions for end-to-end inference directly from image files:

| Inference Mode | Method | Resolution | Use Case |
|---|---|---|---|
| **Superspot Mode** | `_forward_superspot()` | Spot-level (55μm) | Batch prediction during training/validation |
| **Single Cell Mode** | `_forward_single_cell()` | Cell-level | Individual cell predictions |
| **Spot from Image** | `predict_spot_spatial_transcriptomics_from_image_path()` | Spot-level | End-to-end spot inference from WSI |
| **Cell from Image** | `predict_cell_spatial_transcriptomics_from_image_path()` | Cell-level | End-to-end cell inference from WSI |
| **DeepSpot2Cell from Image** | `predict_spot2cell_from_image_paths()` | Cell-level | Inference from individual image tiles |

## Model-Level Inference Modes

The `DeepSpot2Cell` class implements two forward pass methods that operate at different spatial resolutions. These methods share the same underlying architecture (Phi and Rho ensembles) but differ in how they aggregate features.

### Superspot Mode (_forward_superspot)

The superspot mode is used during training and validation to predict spot-level gene expression. It aggregates embeddings from all cells within a spot to produce a single spot-level prediction.

```mermaid
graph TB
    subgraph Input["Input Tensors"]
        X["x: [B, C, D]<br/>Cell embeddings"]
        M["mask: [B, C]<br/>Valid cell mask"]
        CTX["context: [B, N, D]<br/>Spot + neighbors"]
        CTX_M["context_mask: [B, N-1]<br/>Neighbor validity"]
    end
    
    subgraph Processing["Feature Processing"]
        PHI_CELL["phi_cell ensemble<br/>Apply to all cells"]
        PHI_SPOT["phi_spot ensemble<br/>Apply to spot + neighbors"]
        
        OWN["Own spot features<br/>context[:, 0, :]"]
        NEIGH["Neighbor features<br/>context[:, 1:, :]"]
        
        MEAN["Mean neighbor features<br/>masked average"]
    end
    
    subgraph Aggregation["Cell Aggregation"]
        CONCAT["Concatenate per-cell:<br/>cell + spot + neighbors"]
        MASK_AGG["Mask invalid cells<br/>multiply by mask"]
        SUM["Sum across cells<br/>sum(dim=1)"]
    end
    
    subgraph Output["Output"]
        RHO["rho ensemble<br/>Apply to aggregated"]
        PRED["Spot prediction<br/>[B, genes]"]
    end
    
    X --> PHI_CELL
    CTX --> PHI_SPOT
    
    PHI_SPOT --> OWN
    PHI_SPOT --> NEIGH
    
    CTX_M --> MEAN
    NEIGH --> MEAN
    
    PHI_CELL --> CONCAT
    OWN --> CONCAT
    MEAN --> CONCAT
    
    CONCAT --> MASK_AGG
    M --> MASK_AGG
    
    MASK_AGG --> SUM
    SUM --> RHO
    RHO --> PRED
```

**Sources:** [deepspot2cell/model.py:114-137]()

The superspot forward pass processes batches of spots:

1. **Cell Feature Extraction**: Cell embeddings `x` of shape `[batch, num_cells, embedding_dim]` are processed through the `phi_cell` ensemble to produce cell-specific features.

2. **Context Processing**: The context tensor contains the spot's own embedding at position `context[:, 0, :]` and neighboring spot embeddings at `context[:, 1:, :]`. These are processed through the `phi_spot` ensemble.

3. **Neighbor Aggregation**: Neighboring spot features are masked using `context_mask` and averaged to produce a mean neighbor representation. Invalid neighbors (masked as `True`) are excluded from the average.

4. **Per-Cell Concatenation**: For each cell, the system concatenates three feature vectors: the cell's own features, the spot's features, and the mean neighbor features. This produces a tensor of shape `[batch, num_cells, phi2rho_size * 3]`.

5. **Cell Masking and Aggregation**: The concatenated features are multiplied by the cell mask (expanded to match dimensions) to zero out invalid cells. The features are then summed across the cell dimension to produce a single feature vector per spot.

6. **Final Prediction**: The aggregated features pass through the `rho` ensemble to produce spot-level gene expression predictions of shape `[batch, num_genes]`.

### Single Cell Mode (_forward_single_cell)

The single cell mode predicts gene expression for individual cells. Unlike superspot mode, it does not aggregate across cells but instead produces a prediction for each cell independently.

```mermaid
graph TB
    subgraph Input["Input Tensors"]
        X_C["x: [N, D] or [1, D]<br/>Single or multiple cells"]
        CTX_C["context: [M, D]<br/>Spot + neighbors"]
        CTX_M_C["context_mask: [M-1]<br/>Neighbor validity"]
    end
    
    subgraph Processing["Feature Extraction"]
        PHI_CELL_C["phi_cell ensemble<br/>Process cells"]
        PHI_SPOT_C["phi_spot ensemble<br/>Process context"]
        
        OWN_C["Own spot: context[0]"]
        NEIGH_C["Neighbors: context[1:]"]
        
        MEAN_C["Mean neighbors<br/>masked average"]
    end
    
    subgraph Broadcast["Feature Broadcasting"]
        EXPAND_SPOT["Expand spot features<br/>to [N, phi2rho_size]"]
        EXPAND_NEIGH["Expand neighbor features<br/>to [N, phi2rho_size]"]
    end
    
    subgraph Combine["Feature Combination"]
        CONCAT_C["Concatenate:<br/>cell + spot + neighbors"]
    end
    
    subgraph Output["Output"]
        RHO_C["rho ensemble"]
        PRED_C["Cell predictions<br/>[N, genes]"]
    end
    
    X_C --> PHI_CELL_C
    CTX_C --> PHI_SPOT_C
    
    PHI_SPOT_C --> OWN_C
    PHI_SPOT_C --> NEIGH_C
    
    NEIGH_C --> MEAN_C
    CTX_M_C --> MEAN_C
    
    PHI_CELL_C --> CONCAT_C
    OWN_C --> EXPAND_SPOT
    MEAN_C --> EXPAND_NEIGH
    
    EXPAND_SPOT --> CONCAT_C
    EXPAND_NEIGH --> CONCAT_C
    
    CONCAT_C --> RHO_C
    RHO_C --> PRED_C
```

**Sources:** [deepspot2cell/model.py:88-112]()

The single cell forward pass operates differently from superspot mode:

1. **Input Handling**: Accepts a tensor of cell embeddings with shape `[num_cells, embedding_dim]` or `[1, embedding_dim]`. If a 1D tensor is provided, it is unsqueezed to add a batch dimension.

2. **Cell Features**: Each cell embedding is processed through the `phi_cell` ensemble independently, producing cell-specific features.

3. **Context Processing**: The spot containing the cell is at `context[0]` and neighboring spots are at `context[1:]`. The spot is processed through `phi_spot`, while neighbors are processed and averaged (with masking).

4. **Feature Broadcasting**: Unlike superspot mode which aggregates, single cell mode broadcasts the spot and neighbor features to match the number of cells. Each cell receives the same spot and neighbor context.

5. **Per-Cell Prediction**: Features are concatenated for each cell: `[cell_features, spot_features, neighbor_features]`. Each concatenated vector passes through the `rho` ensemble to produce an independent gene expression prediction for that cell.

6. **Output Shape**: Returns a tensor of shape `[num_cells, num_genes]`, with one prediction per cell.

This mode is used during validation (when `cell_gt_available=True`) to compute cell-level metrics, and during inference to generate virtual single-cell transcriptomics.

**Sources:** [deepspot2cell/model.py:168-191](), [deepspot2cell/model.py:211-232]()

## Image-Based Inference Functions

DeepSpot2Cell provides three high-level functions for performing inference directly from H&E whole slide images. These functions handle image loading, tile extraction, feature computation, and prediction in a single call.

### Spot-Level Inference from Image Path

The `predict_spot_spatial_transcriptomics_from_image_path()` function performs end-to-end spot-level prediction from a whole slide image.

```mermaid
graph LR
    subgraph Inputs["Function Inputs"]
        IMG_PATH["image_path"]
        ADATA["adata<br/>(spot locations)"]
        SPOT_DIA["spot_diameter"]
        N_TILES["n_mini_tiles"]
        PREPROC["preprocess<br/>(PFM transform)"]
        MORPH["morphology_model<br/>(PFM)"]
        EXPR_MODEL["model_expression<br/>(DeepSpot2Cell)"]
    end
    
    subgraph Loading["Image Loading"]
        PYVIPS["pyvips.Image<br/>Load WSI"]
    end
    
    subgraph PerSpot["For Each Spot"]
        CROP_SPOT["crop_tile()<br/>Extract spot region"]
        MINI["compute_mini_tiles()<br/>Split into n tiles"]
        FIND_NEIGH["compute_neighbors()<br/>Find spatial neighbors"]
        CROP_NEIGH["crop_tile()<br/>Extract neighbor spots"]
    end
    
    subgraph Embed["Feature Extraction"]
        PREP_ALL["preprocess<br/>Transform images"]
        MORPH_FWD["morphology_model()<br/>Extract embeddings"]
    end
    
    subgraph Predict["Prediction"]
        MODEL_FWD["model_expression()<br/>Predict expression"]
        INV_TRANS["inverse_transform()<br/>Denormalize"]
    end
    
    subgraph Output["Output"]
        COUNTS["counts array<br/>[n_spots, n_genes]"]
    end
    
    IMG_PATH --> PYVIPS
    ADATA --> PerSpot
    PYVIPS --> CROP_SPOT
    
    SPOT_DIA --> CROP_SPOT
    CROP_SPOT --> MINI
    N_TILES --> MINI
    
    ADATA --> FIND_NEIGH
    FIND_NEIGH --> CROP_NEIGH
    PYVIPS --> CROP_NEIGH
    
    MINI --> PREP_ALL
    CROP_NEIGH --> PREP_ALL
    PREPROC --> PREP_ALL
    
    PREP_ALL --> MORPH_FWD
    MORPH --> MORPH_FWD
    
    MORPH_FWD --> MODEL_FWD
    EXPR_MODEL --> MODEL_FWD
    
    MODEL_FWD --> INV_TRANS
    INV_TRANS --> COUNTS
```

**Sources:** [deepspot2cell/utils/utils_image.py:246-297]()

#### Workflow Details

1. **Image Loading**: Opens the whole slide image using `pyvips.Image.new_from_file()` for memory-efficient access to large images.

2. **Spot Iteration**: Iterates over all spots in the provided AnnData object (`adata.obs`), processing each spot independently.

3. **Tile Extraction**: For each spot:
   - Uses `crop_tile()` to extract a square region centered on the spot coordinates (`x_pixel`, `y_pixel`) with size `spot_diameter`
   - Applies `compute_mini_tiles()` to split the spot into `n_mini_tiles` sub-regions (e.g., 4x4=16 tiles)
   - Uses `compute_neighbors()` to find neighboring spots within a specified radius

4. **Neighbor Processing**: Extracts image tiles for all neighboring spots identified in step 3. If no neighbors exist, creates a zero array as placeholder.

5. **Preprocessing and Embedding**: 
   - Applies the PFM preprocessing transformation to all image tiles
   - Passes preprocessed tiles through the morphology model to extract feature embeddings
   - Uses mixed precision (`torch.autocast`) and inference mode (`torch.inference_mode()`) for efficiency

6. **Prediction**: Combines spot, sub-spot, and neighbor embeddings into a list `[X_spot, X_subspot, X_neighbors]` and passes to the expression model for prediction.

7. **Post-processing**: Applies inverse transformation to denormalize predictions back to gene expression space.

#### Input Requirements

| Parameter | Type | Description |
|---|---|---|
| `image_path` | str | Path to whole slide image (H&E) |
| `adata` | AnnData | Spot locations with `x_pixel`, `y_pixel`, `barcode` in `.obs` |
| `spot_diameter` | int | Diameter in pixels for spot tile extraction |
| `n_mini_tiles` | int | Number of sub-tiles to split each spot into |
| `preprocess` | callable | PFM preprocessing function |
| `morphology_model` | torch.nn.Module | Pre-trained PFM for feature extraction |
| `model_expression` | Model | Trained DeepSpot2Cell model with `inverse_transform()` |
| `device` | str | 'cuda' or 'cpu' |
| `super_resolution` | bool | If True, extracts only center tile |
| `neighbor_radius` | int | Radius for neighbor identification (default=1) |

**Sources:** [deepspot2cell/utils/utils_image.py:246-297]()

### Cell-Level Inference from Image Path

The `predict_cell_spatial_transcriptomics_from_image_path()` function performs end-to-end cell-level prediction from a whole slide image.

```mermaid
graph LR
    subgraph Inputs["Function Inputs"]
        IMG_PATH_C["image_path"]
        ADATA_C["adata<br/>(cell locations)"]
        CELL_DIA["cell_diameter"]
        RAD_NEIGH["radius_neighbors"]
        PREPROC_C["preprocess"]
        MORPH_C["morphology_model"]
        EXPR_MODEL_C["model_expression"]
    end
    
    subgraph Preprocessing["Neighbor Computation"]
        NEIGH_FIT["NearestNeighbors<br/>fit on cell locations"]
        NEIGH_FIND["radius_neighbors()<br/>Find cells in radius"]
        STORE_NEIGH["Store in adata.obs"]
    end
    
    subgraph Loading["Image Loading"]
        PYVIPS_C["pyvips.Image<br/>Load WSI"]
    end
    
    subgraph PerCell["For Each Cell"]
        CROP_CELL["crop_tile()<br/>Extract cell region"]
        PARSE_NEIGH["Parse neighbors<br/>from adata.obs"]
        CROP_CELL_NEIGH["crop_tile()<br/>Extract neighbor cells"]
    end
    
    subgraph Embed_C["Feature Extraction"]
        PREP_ALL_C["preprocess<br/>Transform images"]
        MORPH_FWD_C["morphology_model()<br/>Extract embeddings"]
    end
    
    subgraph Predict_C["Prediction"]
        MODEL_FWD_C["model_expression()<br/>Predict expression"]
        INV_TRANS_C["inverse_transform()<br/>Denormalize"]
    end
    
    subgraph Output_C["Output"]
        COUNTS_C["counts array<br/>[n_cells, n_genes]"]
    end
    
    ADATA_C --> NEIGH_FIT
    NEIGH_FIT --> NEIGH_FIND
    RAD_NEIGH --> NEIGH_FIND
    NEIGH_FIND --> STORE_NEIGH
    
    IMG_PATH_C --> PYVIPS_C
    STORE_NEIGH --> PARSE_NEIGH
    
    PYVIPS_C --> CROP_CELL
    CELL_DIA --> CROP_CELL
    
    PARSE_NEIGH --> CROP_CELL_NEIGH
    PYVIPS_C --> CROP_CELL_NEIGH
    
    CROP_CELL --> PREP_ALL_C
    CROP_CELL_NEIGH --> PREP_ALL_C
    PREPROC_C --> PREP_ALL_C
    
    PREP_ALL_C --> MORPH_FWD_C
    MORPH_C --> MORPH_FWD_C
    
    MORPH_FWD_C --> MODEL_FWD_C
    EXPR_MODEL_C --> MODEL_FWD_C
    
    MODEL_FWD_C --> INV_TRANS_C
    INV_TRANS_C --> COUNTS_C
```

**Sources:** [deepspot2cell/utils/utils_image.py:300-361]()

#### Workflow Details

1. **Neighbor Precomputation**: Uses `sklearn.neighbors.NearestNeighbors` to identify all cells within `radius_neighbors` of each cell. Stores results in `adata.obs["neighbors"]` as a triple-underscore-separated string of cell barcodes.

2. **Image Loading**: Opens the whole slide image using `pyvips` for efficient memory access.

3. **Cell Iteration**: Processes each cell in `adata.obs` sequentially.

4. **Tile Extraction**: For each cell:
   - Extracts a square tile centered on the cell using `crop_tile()` with `cell_diameter`
   - Parses the neighbor list and extracts tiles for all neighboring cells
   - Creates zero-filled placeholder if no neighbors exist

5. **Feature Extraction**: Preprocesses and embeds cell and neighbor tiles through the morphology model.

6. **Prediction**: Combines cell and neighbor embeddings into list `[X_cell, X_neighbors]` and passes to model.

7. **Post-processing**: Applies inverse transformation to convert predictions to gene expression values.

#### Key Differences from Spot-Level Inference

- **No sub-tiles**: Cells are not split into mini-tiles
- **No spot context**: Uses neighboring cells directly instead of spot-level context
- **Radius-based neighbors**: Uses distance-based neighbor finding instead of spot adjacency
- **Different model input**: Model receives `[cell, neighbors]` instead of `[spot, subspots, neighbor_spots]`

**Sources:** [deepspot2cell/utils/utils_image.py:300-361]()

### DeepSpot2Cell-Specific Inference from Image Paths

The `predict_spot2cell_from_image_paths()` function provides a specialized interface for DeepSpot2Cell that accepts individual image tile paths rather than extracting tiles from a whole slide image.

```mermaid
graph TB
    subgraph Inputs["Function Inputs"]
        CELL_IMG["cell_image_path<br/>(target cell tile)"]
        SPOT_IMG["spot_image_path<br/>(spot containing cell)"]
        NEIGH_IMGS["neighbor_image_paths<br/>(list of neighbor tiles)"]
        DS2C_MODEL["deepspot2cell_model<br/>(trained model)"]
        MORPH_DS2C["morphology_model<br/>(PFM)"]
        PREPROC_DS2C["preprocess<br/>(PFM transform)"]
        GENES["gene_names<br/>(optional)"]
    end
    
    subgraph EmbedExtract["extract_embedding() helper"]
        PIL_LOAD["PIL.Image.open()"]
        CHECK_SQ["Check if square<br/>resize if needed"]
        PREP_IMG["preprocess()<br/>transform to tensor"]
        MORPH_EXTRACT["morphology_model()<br/>extract embedding"]
    end
    
    subgraph CellEmbed["Cell Embedding"]
        CELL_PIL["Load cell image"]
        CELL_EMB["Extract cell embedding<br/>shape: [1, D]"]
    end
    
    subgraph SpotEmbed["Spot Embedding"]
        SPOT_PIL["Load spot image"]
        SPOT_EMB["Extract spot embedding<br/>shape: [1, D]"]
    end
    
    subgraph NeighEmbed["Neighbor Embeddings"]
        NEIGH_LOOP["For each neighbor path"]
        CHECK_EXISTS["Check if file exists"]
        NEIGH_EMB["Extract embedding<br/>or create zeros"]
        NEIGH_LIST["Collect embeddings<br/>list of [1, D] tensors"]
    end
    
    subgraph ContextBuild["Build Context"]
        CONCAT_CTX["Concatenate:<br/>[spot] + neighbors"]
        CTX_TENSOR["context tensor<br/>shape: [M, D]"]
        CTX_MASK_BUILD["Build context_mask<br/>False for spot, exists() for neighbors"]
        CTX_MASK["context_mask<br/>shape: [M]"]
    end
    
    subgraph Prediction["Prediction"]
        FWD_SINGLE["model._forward_single_cell(<br/>  cell_emb,<br/>  context,<br/>  context_mask<br/>)"]
        PRED_TENSOR["predicted_expression<br/>shape: [1, num_genes]"]
    end
    
    subgraph Format["Format Results"]
        TO_NUMPY["Convert to numpy"]
        MAP_GENES["Map to gene names<br/>if provided"]
        RESULTS_DICT["results dict<br/>{gene: expression}"]
    end
    
    CELL_IMG --> CELL_PIL
    CELL_PIL --> EmbedExtract
    EmbedExtract --> CELL_EMB
    
    SPOT_IMG --> SPOT_PIL
    SPOT_PIL --> EmbedExtract
    EmbedExtract --> SPOT_EMB
    
    NEIGH_IMGS --> NEIGH_LOOP
    NEIGH_LOOP --> CHECK_EXISTS
    CHECK_EXISTS --> NEIGH_EMB
    CHECK_EXISTS --> EmbedExtract
    NEIGH_EMB --> NEIGH_LIST
    
    SPOT_EMB --> CONCAT_CTX
    NEIGH_LIST --> CONCAT_CTX
    CONCAT_CTX --> CTX_TENSOR
    
    CHECK_EXISTS --> CTX_MASK_BUILD
    CTX_MASK_BUILD --> CTX_MASK
    
    CELL_EMB --> FWD_SINGLE
    CTX_TENSOR --> FWD_SINGLE
    CTX_MASK --> FWD_SINGLE
    DS2C_MODEL --> FWD_SINGLE
    
    FWD_SINGLE --> PRED_TENSOR
    PRED_TENSOR --> TO_NUMPY
    TO_NUMPY --> MAP_GENES
    GENES --> MAP_GENES
    MAP_GENES --> RESULTS_DICT
```

**Sources:** [deepspot2cell/utils/utils_image.py:364-457]()

#### Function Architecture

This function is designed for scenarios where image tiles have already been extracted and saved as individual files. It directly calls the model's `_forward_single_cell()` method.

#### Helper Function: extract_embedding()

Defined inline at [deepspot2cell/utils/utils_image.py:396-410](), this helper extracts embeddings from individual image files:

1. Opens image using PIL
2. Ensures square dimensions (resizes if necessary)
3. Applies preprocessing transformation
4. Passes through morphology model
5. Returns numpy array of embedding

#### Main Workflow

1. **Cell Embedding Extraction**: Loads the target cell image and extracts its embedding using the helper function.

2. **Spot Embedding Extraction**: Loads the spot image (the spot region containing the target cell) and extracts its embedding.

3. **Neighbor Embedding Extraction**: Iterates through neighbor image paths:
   - If file exists: extracts embedding
   - If file missing: creates zero-filled tensor matching embedding dimensions
   - Collects all neighbor embeddings in a list

4. **Context Construction**: 
   - Concatenates spot embedding and neighbor embeddings: `[spot_embedding] + neighbor_embeddings`
   - Creates context mask: `[False]` for the spot (always present), then `True/False` for each neighbor based on file existence

5. **Model Inference**: Calls `deepspot2cell_model._forward_single_cell()` with:
   - Cell embedding tensor
   - Context tensor (spot + neighbors)
   - Context mask (validity indicators)

6. **Result Formatting**: If `gene_names` provided, returns a dictionary mapping gene names to predicted expression values. Otherwise, uses generic names like "gene_0", "gene_1", etc.

#### Input Requirements

| Parameter | Type | Description |
|---|---|---|
| `cell_image_path` | str | Path to target cell image tile |
| `spot_image_path` | str | Path to spot image tile containing the cell |
| `neighbor_image_paths` | list[str] | List of paths to neighboring image tiles |
| `deepspot2cell_model` | DeepSpot2Cell | Trained model instance |
| `morphology_model` | torch.nn.Module | Pre-trained PFM |
| `preprocess` | callable | PFM preprocessing function |
| `device` | str | 'cuda' or 'cpu' |
| `gene_names` | list[str] (optional) | Gene names for result mapping |

#### Return Value

Returns a dictionary mapping gene names/indices to predicted expression values:

```python
{
    "gene1": 5.234,
    "gene2": 1.032,
    ...
}
```

#### Use Cases

This function is particularly useful for:
- **Interactive applications**: Where users select individual cells for prediction
- **Pre-extracted datasets**: Where tiles have been saved during preprocessing
- **Debugging**: Testing model predictions on specific tiles
- **Integration**: Embedding DeepSpot2Cell in larger pipelines that work with tile-based data

**Sources:** [deepspot2cell/utils/utils_image.py:364-457]()

## Inference Workflow Comparison

The three image-based inference functions serve different purposes and have different input/output characteristics:

| Aspect | Spot from Image | Cell from Image | DeepSpot2Cell from Paths |
|---|---|---|---|
| **Input** | WSI + spot coordinates | WSI + cell coordinates | Individual tile paths |
| **Resolution** | Spot-level (55μm) | Cell-level (~10μm) | Cell-level |
| **Tile Extraction** | Automatic from WSI | Automatic from WSI | Pre-extracted tiles |
| **Sub-tiles** | Yes (splits spots) | No | No |
| **Context** | Neighboring spots | Neighboring cells | Spot + neighbor spots |
| **Batch Processing** | Sequential per spot | Sequential per cell | Single prediction |
| **Model Method** | Custom forward | Custom forward | `_forward_single_cell()` |
| **Output** | Array [n_spots, n_genes] | Array [n_cells, n_genes] | Dict {gene: value} |
| **Use Case** | Spot-level ST prediction | Cell-level ST prediction | Interactive/debugging |

## Implementation Notes

### Memory and Performance Considerations

1. **Mixed Precision**: All image-based inference functions use `torch.autocast(device_type="cuda", dtype=torch.float16)` for efficient GPU memory usage and faster inference.

2. **Inference Mode**: Functions use `torch.inference_mode()` to disable gradient computation, reducing memory overhead.

3. **Sequential Processing**: Both `predict_spot_spatial_transcriptomics_from_image_path()` and `predict_cell_spatial_transcriptomics_from_image_path()` process locations sequentially with a progress bar (`tqdm`). For large datasets, this can be time-consuming.

4. **WSI Loading**: Uses `pyvips` for memory-efficient loading of large whole slide images, loading only required regions on demand.

**Sources:** [deepspot2cell/utils/utils_image.py:260-261](), [deepspot2cell/utils/utils_image.py:327-328]()

### Ensemble Predictions

The model uses ensemble methods for robust predictions:

- **Phi Ensembles**: Multiple Phi modules process features independently, and predictions are combined using median: `torch.median(torch.stack([p(x) for p in self.phi_spot], 1), 1).values`

- **Rho Ensemble**: Multiple Rho modules process aggregated features, and predictions are averaged: `torch.stack([r(x) for r in self.rho], 1).mean(1)`

This ensemble approach provides more stable predictions compared to single-model inference.

**Sources:** [deepspot2cell/model.py:139-146]()

### Missing Neighbors Handling

All inference functions handle missing neighbors gracefully:

- In `predict_spot_spatial_transcriptomics_from_image_path()`: Creates zero array if no neighbors found [deepspot2cell/utils/utils_image.py:274]()

- In `predict_cell_spatial_transcriptomics_from_image_path()`: Creates zero array if no neighbors found [deepspot2cell/utils/utils_image.py:340]()

- In `predict_spot2cell_from_image_paths()`: Creates zero tensor for non-existent neighbor paths [deepspot2cell/utils/utils_image.py:428]()

The context mask ensures that missing neighbors are properly excluded during feature aggregation in the model's forward pass.

---