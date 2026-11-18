# System Architecture

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.codeboarding/AESTETIK_API_Orchestration.md](.codeboarding/AESTETIK_API_Orchestration.md)
- [.codeboarding/on_boarding.md](.codeboarding/on_boarding.md)
- [src/aestetik/AESTETIK.py](src/aestetik/AESTETIK.py)
- [src/aestetik/modules/aestetik_module.py](src/aestetik/modules/aestetik_module.py)

</details>



## Purpose and Scope

This document provides a comprehensive technical overview of AESTETIK's internal architecture, component interactions, and design patterns. It is intended for developers who need to understand the system's internals for contribution, extension, or debugging purposes.

This page focuses on architectural design and component relationships. For specific API usage patterns, see [User Guide](#4). For detailed class and method documentation, see [API Reference](#5). For development environment setup, see [Development Setup](#6.2).

---

## Architectural Layers

AESTETIK follows a layered architecture with clear separation of concerns. The system is organized into five primary layers, each with distinct responsibilities.

### Layer Diagram

```mermaid
graph TB
    subgraph "Layer 1: Public API"
        AESTETIK["AESTETIK<br/>(AESTETIK.py)"]
    end
    
    subgraph "Layer 2: Orchestration"
        DataModule["AESTETIKDataModule<br/>(data_module.py)"]
        LitModel["AESTETIKModel<br/>(aestetik_module.py)"]
        Trainer["Lightning Trainer<br/>(external)"]
    end
    
    subgraph "Layer 3: Core Components"
        AE["AE Autoencoder<br/>(model.py)"]
        Dataset["CustomDataset<br/>(dataloader.py)"]
        Loss["compute_loss<br/>(loss_function.py)"]
    end
    
    subgraph "Layer 4: Utilities"
        UtilsData["utils_data.py"]
        UtilsGrid["utils_grid.py"]
        UtilsClustering["utils_clustering.py"]
        UtilsViz["utils_visualization.py"]
    end
    
    subgraph "Layer 5: External Dependencies"
        PyTorch["PyTorch"]
        Lightning["PyTorch Lightning"]
        AnnData["AnnData/Scanpy"]
    end
    
    AESTETIK --> DataModule
    AESTETIK --> LitModel
    AESTETIK --> Trainer
    AESTETIK --> UtilsClustering
    AESTETIK --> UtilsData
    
    DataModule --> UtilsData
    DataModule --> Dataset
    
    LitModel --> AE
    LitModel --> Loss
    LitModel --> DataModule
    
    UtilsData --> UtilsGrid
    UtilsData --> UtilsClustering
    
    AE --> PyTorch
    LitModel --> Lightning
    Trainer --> Lightning
    Dataset --> PyTorch
    UtilsData --> AnnData
    UtilsClustering --> AnnData
```

**Sources:** [src/aestetik/AESTETIK.py](), [src/aestetik/modules/aestetik_module.py](), [src/aestetik/data_modules/data_module.py]()

### Layer Responsibilities

| Layer | Components | Primary Responsibilities |
|-------|-----------|-------------------------|
| **Public API** | `AESTETIK` class | User-facing interface, parameter validation, workflow orchestration |
| **Orchestration** | `AESTETIKDataModule`, `AESTETIKModel`, `Trainer` | Training/inference coordination, data preparation, checkpoint management |
| **Core Components** | `AE`, `CustomDataset`, `compute_loss` | Model architecture, data batching, loss computation |
| **Utilities** | `utils_*` modules | Preprocessing, grid construction, clustering, visualization |
| **External Dependencies** | PyTorch, Lightning, AnnData | Framework primitives and data structures |

**Sources:** [src/aestetik/AESTETIK.py:24-173](), [src/aestetik/modules/aestetik_module.py:14-65]()

---

## Core Component Interactions

### AESTETIK Class: The Orchestrator

The `AESTETIK` class acts as a facade, exposing three primary methods while managing complex internal state through parameter dictionaries.

#### Parameter Organization

```mermaid
graph LR
    AESTETIK["AESTETIK.__init__"]
    
    GridParams["grid_params<br/>{morphology_dim, num_input_channels,<br/>obsm_transcriptomics_dim}"]
    ModelParams["model_architecture_params<br/>{latent_dim, c_hid, kernel_size,<br/>p, n_ensemble_encoder/decoder}"]
    DataLoaderParams["dataloader_params<br/>{batch_size, num_workers}"]
    TrainingParams["training_params<br/>{lr, weight_decay, max_epochs}"]
    ClusteringParams["clustering_params<br/>{nCluster, clustering_method,<br/>n_neighbors, refine_cluster}"]
    LossParams["loss_regularization_params<br/>{multi_triplet_loss, rec_alpha,<br/>triplet_alpha, n_repeats,<br/>morphology_weight, total_weight}"]
    DataHandlingParams["data_handling_params<br/>{n_jobs, train_size}"]
    
    AESTETIK --> GridParams
    AESTETIK --> ModelParams
    AESTETIK --> DataLoaderParams
    AESTETIK --> TrainingParams
    AESTETIK --> ClusteringParams
    AESTETIK --> LossParams
    AESTETIK --> DataHandlingParams
```

**Sources:** [src/aestetik/AESTETIK.py:120-166]()

#### Method Delegation Pattern

The `AESTETIK` class follows a delegation pattern where public methods orchestrate private methods:

```mermaid
graph TB
    fit["fit()"]
    predict["predict()"]
    fit_predict["fit_predict()"]
    
    validate_fit["_validate_fit_inputs()"]
    set_fit["_set_fit_params()"]
    build_model["_build_model()"]
    create_callbacks["_create_callbacks()"]
    
    check_fitted["_check_fitted()"]
    validate_predict["_validate_predict_inputs()"]
    calibrate_predict["_calibrate_predict_inputs()"]
    set_predict["_set_predict_params()"]
    compute_latent["_compute_latent_space()"]
    postprocess["_postprocess_predictions()"]
    
    fit --> validate_fit
    fit --> set_fit
    fit --> build_model
    fit --> create_callbacks
    
    predict --> check_fitted
    predict --> validate_predict
    predict --> calibrate_predict
    predict --> set_predict
    predict --> compute_latent
    predict --> postprocess
    
    fit_predict --> fit
    fit_predict --> set_predict
    fit_predict --> compute_latent
    fit_predict --> postprocess
```

**Sources:** [src/aestetik/AESTETIK.py:179-332]()

---

## Data Flow Architecture

### Training Data Flow

The training data flow involves transformation through multiple stages before reaching the model.

```mermaid
graph TB
    AnnData["AnnData Object<br/>.obsm['X_pca_transcriptomics']<br/>.obsm['X_pca_morphology']<br/>.obs['x_array', 'y_array']"]
    
    prepare_input["prepare_input_for_model()<br/>(utils_data.py)"]
    
    clustering_trans["clustering()<br/>transcriptomics"]
    clustering_morph["clustering()<br/>morphology"]
    
    calibrate["calibrate_weights()<br/>(utils_data.py)"]
    
    build_grid["build_grid()<br/>(utils_grid.py)"]
    
    CustomDataset["CustomDataset<br/>(dataloader.py)"]
    
    DataLoader["DataLoader<br/>(PyTorch)"]
    
    AESTETIKModel["AESTETIKModel.training_step()"]
    
    AnnData --> prepare_input
    prepare_input --> clustering_trans
    prepare_input --> clustering_morph
    clustering_trans --> calibrate
    clustering_morph --> calibrate
    calibrate --> build_grid
    build_grid --> CustomDataset
    CustomDataset --> DataLoader
    DataLoader --> AESTETIKModel
```

**Sources:** [src/aestetik/data_modules/data_module.py](), [src/aestetik/utils/utils_data.py](), [src/aestetik/utils/utils_grid.py]()

### Data Transformations Table

| Stage | Input | Output | Key Operations |
|-------|-------|--------|----------------|
| **Clustering** | `X.obsm[used_obsm_*]` | `X.obs['cluster_transcriptomics']`, `X.obs['cluster_morphology']` | BGM/KMeans/Louvain/Leiden clustering |
| **Weight Calibration** | Cluster labels | `transcriptomics_weight`, `morphology_weight` | Silhouette score comparison |
| **Grid Construction** | Features + spatial coords | `X.obsm['X_st_grid']` (N × C × H × W) | KDTree neighbor search, MinMaxScaling |
| **Dataset Creation** | Grids + cluster labels | Batches of (anchor, pos_list, neg_list) | Triplet sampling with `n_repeats` |

**Sources:** [src/aestetik/utils/utils_data.py](), [src/aestetik/utils/utils_grid.py](), [src/aestetik/dataloader.py]()

### Prediction Data Flow

```mermaid
sequenceDiagram
    participant User
    participant AESTETIK
    participant build_grid
    participant DataLoader
    participant Trainer
    participant AESTETIKModel
    participant AE_encoder
    participant clustering
    
    User->>AESTETIK: predict(adata, num_repeats=1000)
    AESTETIK->>AESTETIK: _check_fitted()
    AESTETIK->>AESTETIK: _validate_predict_inputs()
    AESTETIK->>AESTETIK: _calibrate_predict_inputs()
    AESTETIK->>build_grid: build_grid(adata)
    build_grid-->>AESTETIK: adata.obsm['X_st_grid']
    AESTETIK->>DataLoader: TensorDataset(X_st_grid)
    AESTETIK->>Trainer: predict(model, dataloader)
    
    loop num_repeats times per batch
        Trainer->>AESTETIKModel: predict_step(batch)
        AESTETIKModel->>AESTETIKModel: model.train()
        AESTETIKModel->>AE_encoder: encoder(batch)
        AE_encoder-->>AESTETIKModel: latent
    end
    
    AESTETIKModel->>AESTETIKModel: torch.mean(latent_list)
    AESTETIKModel-->>Trainer: avg_latent
    Trainer-->>AESTETIK: all_latent_space
    AESTETIK->>AESTETIK: adata.obsm[save_emb] = latent
    
    alt cluster=True
        AESTETIK->>clustering: clustering(adata, used_obsm=save_emb)
        clustering-->>AESTETIK: adata.obs['cluster_aestetik']
    end
    
    AESTETIK-->>User: None (adata modified)
```

**Sources:** [src/aestetik/AESTETIK.py:240-282](), [src/aestetik/modules/aestetik_module.py:142-161]()

---

## Model Architecture Components

### AESTETIKModel: Lightning Module Wrapper

The `AESTETIKModel` wraps the core autoencoder in a PyTorch Lightning module, managing the training loop.

#### Component Relationships

```mermaid
graph TB
    AESTETIKModel["AESTETIKModel<br/>(LightningModule)"]
    
    setup["setup()<br/>Initialize loss functions"]
    configure_model["configure_model()<br/>Build AE model"]
    training_step["training_step()<br/>Forward + loss computation"]
    validation_step["validation_step()<br/>Validation forward pass"]
    predict_step["predict_step()<br/>Stochastic encoding"]
    configure_optimizers["configure_optimizers()<br/>Adam optimizer"]
    
    AE["AE Autoencoder"]
    compute_loss["compute_loss()"]
    DataModule["AESTETIKDataModule"]
    
    AESTETIKModel --> setup
    AESTETIKModel --> configure_model
    AESTETIKModel --> training_step
    AESTETIKModel --> validation_step
    AESTETIKModel --> predict_step
    AESTETIKModel --> configure_optimizers
    
    configure_model --> AE
    training_step --> AE
    training_step --> compute_loss
    validation_step --> AE
    validation_step --> compute_loss
    predict_step --> AE
    
    AESTETIKModel -.->|holds reference| DataModule
```

**Sources:** [src/aestetik/modules/aestetik_module.py:14-185]()

#### Training Step Breakdown

The `training_step` method [src/aestetik/modules/aestetik_module.py:93-117]() performs:

1. **Batch unpacking**: `anchor, pos_transcriptomics_list, neg_transcriptomics_list, pos_morphology_list, neg_morphology_list = batch`
2. **Forward pass**: `anchor_encode, anchor_decode = self.model(x=anchor)`
3. **Loss computation**: Calls `compute_loss()` with all components
4. **Logging**: Logs `L` (total), `RC` (reconstruction transcriptomics), `RI` (reconstruction morphology), `TC` (triplet transcriptomics), `TI` (triplet morphology)

**Sources:** [src/aestetik/modules/aestetik_module.py:93-117]()

### AE Autoencoder: Ensemble Architecture

The actual model architecture is defined in `src/aestetik/models/model.py`. It implements an ensemble of encoder-decoder pairs with median aggregation.

#### Ensemble Structure

```mermaid
graph LR
    Input["Input Grid<br/>N × C × H × W"]
    
    Encoder1["Encoder 1"]
    Encoder2["Encoder 2"]
    EncoderN["Encoder N"]
    
    MedianEnc["torch.median<br/>(ensemble dim)"]
    
    Latent["Latent Space<br/>N × latent_dim"]
    
    MedianDec["torch.median<br/>(ensemble dim)"]
    
    Decoder1["Decoder 1"]
    Decoder2["Decoder 2"]
    DecoderM["Decoder M"]
    
    Output["Reconstructed Grid<br/>N × C × H × W"]
    
    Input --> Encoder1
    Input --> Encoder2
    Input --> EncoderN
    
    Encoder1 --> MedianEnc
    Encoder2 --> MedianEnc
    EncoderN --> MedianEnc
    
    MedianEnc --> Latent
    
    Latent --> Decoder1
    Latent --> Decoder2
    Latent --> DecoderM
    
    Decoder1 --> MedianDec
    Decoder2 --> MedianDec
    DecoderM --> MedianDec
    
    MedianDec --> Output
```

**Sources:** [src/aestetik/models/model.py]()

---

## Design Patterns and Principles

### 1. Parameter Dictionary Pattern

AESTETIK organizes related parameters into dictionaries that flow through the system:

```mermaid
graph LR
    AESTETIK_init["AESTETIK.__init__<br/>Creates param dicts"]
    
    DataModule["AESTETIKDataModule<br/>Receives subset of dicts"]
    
    AESTETIKModel["AESTETIKModel<br/>Receives subset of dicts"]
    
    AE["AE<br/>Receives unpacked params"]
    
    AESTETIK_init -->|"grid_params<br/>clustering_params<br/>loss_regularization_params<br/>data_handling_params<br/>dataloader_params"| DataModule
    
    AESTETIK_init -->|"grid_params<br/>model_architecture_params<br/>training_params<br/>optimizer_params"| AESTETIKModel
    
    AESTETIKModel -->|"**grid_params<br/>**model_architecture_params"| AE
```

This pattern provides:
- **Encapsulation**: Related parameters grouped together
- **Flexibility**: Easy to pass subsets to different components
- **Validation**: Centralized parameter checking in `_validate_params()`

**Sources:** [src/aestetik/AESTETIK.py:120-166](), [src/aestetik/modules/aestetik_module.py:15-21]()

### 2. Lazy Initialization Pattern

The `AESTETIKModel` defers model construction until data characteristics are known:

```mermaid
sequenceDiagram
    participant Trainer
    participant AESTETIKModel
    participant DataModule
    participant AE
    
    Trainer->>AESTETIKModel: fit()
    AESTETIKModel->>AESTETIKModel: configure_model()
    
    alt model_built == False
        AESTETIKModel->>DataModule: Get adata
        DataModule-->>AESTETIKModel: adata with X_st_grid
        AESTETIKModel->>AESTETIKModel: Extract grid shape
        AESTETIKModel->>AE: Initialize with grid shape
        AESTETIKModel->>AESTETIKModel: model_built = True
    else model_built == True
        AESTETIKModel->>AESTETIKModel: return (skip)
    end
```

**Sources:** [src/aestetik/modules/aestetik_module.py:80-90]()

### 3. Strategy Pattern for Clustering

Multiple clustering algorithms are supported through a unified interface:

| Method | Implementation | Parameters |
|--------|---------------|-----------|
| `"bgm"` | Bayesian Gaussian Mixture | `nCluster` (int) |
| `"kmeans"` | KMeans | `nCluster` (int) |
| `"louvain"` | Scanpy Louvain | `nCluster` (float resolution) |
| `"leiden"` | Scanpy Leiden | `nCluster` (float resolution) |

The `clustering()` function [src/aestetik/utils/utils_clustering.py]() selects the appropriate strategy based on `method` parameter.

**Sources:** [src/aestetik/utils/utils_clustering.py](), [src/aestetik/AESTETIK.py:146-151]()

### 4. Template Method Pattern in DataModule

`AESTETIKDataModule` implements PyTorch Lightning's `LightningDataModule` interface:

```mermaid
graph TB
    LightningDataModule["LightningDataModule<br/>(Lightning base class)"]
    AESTETIKDataModule["AESTETIKDataModule<br/>(AESTETIK implementation)"]
    
    prepare_data["prepare_data()<br/>Called once"]
    setup["setup(stage)<br/>Called per process"]
    train_dataloader["train_dataloader()<br/>Returns training loader"]
    val_dataloader["val_dataloader()<br/>Returns validation loader"]
    
    LightningDataModule --> AESTETIKDataModule
    AESTETIKDataModule --> prepare_data
    AESTETIKDataModule --> setup
    AESTETIKDataModule --> train_dataloader
    AESTETIKDataModule --> val_dataloader
```

**Sources:** [src/aestetik/data_modules/data_module.py]()

---

## Integration with External Frameworks

### PyTorch Lightning Integration Points

AESTETIK leverages PyTorch Lightning's infrastructure:

| Lightning Component | AESTETIK Usage | Location |
|---------------------|----------------|----------|
| `LightningModule` | `AESTETIKModel` extends it | [src/aestetik/modules/aestetik_module.py:14]() |
| `LightningDataModule` | `AESTETIKDataModule` extends it | [src/aestetik/data_modules/data_module.py]() |
| `Trainer` | Created in `AESTETIK.fit()` | [src/aestetik/AESTETIK.py:234-237]() |
| `Callback` | `LossHistoryCallback`, `EarlyStopping`, `ModelCheckpoint` | [src/aestetik/callbacks/callbacks.py](), [src/aestetik/AESTETIK.py:449-467]() |

**Trainer Configuration:**

```python
self.trainer = Trainer(
    max_epochs=self.training_params["max_epochs"],
    callbacks=callbacks,
    num_sanity_val_steps=0
)
```

**Sources:** [src/aestetik/AESTETIK.py:234-237]()

### AnnData Integration Pattern

AESTETIK follows scanpy conventions for data storage:

```mermaid
graph TB
    AnnData["AnnData Object"]
    
    obsm_input["adata.obsm<br/>(Input Features)"]
    obsm_output["adata.obsm<br/>(Output Embeddings)"]
    obs_metadata["adata.obs<br/>(Metadata)"]
    
    Input1["'X_pca_transcriptomics'<br/>(N × D_trans)"]
    Input2["'X_pca_morphology'<br/>(N × D_morph)"]
    Input3["'X_pca'<br/>(Combined, optional)"]
    
    Grid["'X_st_grid'<br/>(N × C × H × W)"]
    Embedding["'AESTETIK'<br/>(N × latent_dim)"]
    
    Spatial["'x_array', 'y_array'<br/>(Spatial coordinates)"]
    Clusters1["'cluster_transcriptomics'<br/>'cluster_morphology'"]
    Clusters2["'cluster_aestetik'<br/>(Final clusters)"]
    
    AnnData --> obsm_input
    AnnData --> obsm_output
    AnnData --> obs_metadata
    
    obsm_input --> Input1
    obsm_input --> Input2
    obsm_input --> Input3
    
    obsm_output --> Grid
    obsm_output --> Embedding
    
    obs_metadata --> Spatial
    obs_metadata --> Clusters1
    obs_metadata --> Clusters2
```

**Sources:** [src/aestetik/AESTETIK.py:337-393](), [src/aestetik/utils/utils_data.py]()

---

## Component Lifecycle Management

### Fit Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Uninitialized: AESTETIK.__init__()
    
    Uninitialized --> Validating: fit() called
    
    Validating --> ParameterSetup: _validate_fit_inputs()
    
    ParameterSetup --> DataPreparing: _set_fit_params()
    
    DataPreparing --> ModelBuilding: AESTETIKDataModule initialized
    
    ModelBuilding --> Training: _build_model()
    
    Training --> Fitted: Trainer.fit()
    
    Fitted --> [*]
    
    note right of Validating
        Checks obsm keys exist
        Checks obs columns exist
    end note
    
    note right of DataPreparing
        prepare_input_for_model()
        Clustering both modalities
        Weight calibration
        Grid construction
    end note
    
    note right of Training
        Multiple epochs
        Callbacks active
        Checkpointing (if validation)
    end note
```

**Sources:** [src/aestetik/AESTETIK.py:179-238]()

### Predict Lifecycle

```mermaid
stateDiagram-v2
    [*] --> CheckFitted: predict() called
    
    CheckFitted --> Validating: _check_fitted()
    
    Validating --> Calibrating: _validate_predict_inputs()
    
    Calibrating --> GridBuilding: _calibrate_predict_inputs()
    
    GridBuilding --> Computing: build_grid() (if needed)
    
    Computing --> Postprocessing: _compute_latent_space()
    
    Postprocessing --> [*]: _postprocess_predictions()
    
    note right of CheckFitted
        Raises if trainer is None
    end note
    
    note right of Calibrating
        Trim features to match
        training dimensions
    end note
    
    note right of Computing
        num_repeats forward passes
        Mean aggregation
        Model in train mode
    end note
    
    note right of Postprocessing
        Store in adata.obsm
        Optional clustering
        Optional refinement
    end note
```

**Sources:** [src/aestetik/AESTETIK.py:240-282](), [src/aestetik/AESTETIK.py:369-426]()

### State Management

The `AESTETIK` class maintains minimal mutable state:

| Attribute | Type | Set During | Purpose |
|-----------|------|------------|---------|
| `lit_aestetik_model` | `Optional[AESTETIKModel]` | `fit()` | Reference to trained model |
| `trainer` | `Optional[Trainer]` | `fit()` | Reference to Lightning trainer |
| `losses` | `dict` | `fit()` | Training history from callbacks |
| Various `*_params` dicts | `dict` | `__init__()` | Configuration storage |

**Sources:** [src/aestetik/AESTETIK.py:170-171](), [src/aestetik/AESTETIK.py:238]()

---

## Key Implementation Details

### Stochastic Prediction with Dropout

During prediction, the model is set to train mode to enable dropout, and multiple forward passes are averaged:

```python
self.model.train()  # Enable dropout
batch_latent_space = []

for _ in range(self.predict_params["num_repeats"]):
    out = self.model.encoder(batch_spots)
    batch_latent_space.append(out)

batch_latent_space = torch.stack(batch_latent_space, dim=0)
batch_latent_space = torch.mean(batch_latent_space, dim=0)
```

**Sources:** [src/aestetik/modules/aestetik_module.py:151-160]()

### Grid Construction with KDTree

Spatial grids are constructed using efficient KDTree neighbor search:

1. Build KDTree from `(x_array, y_array)` coordinates
2. For each spot, query `window_size × window_size` nearest neighbors
3. Fill grid with MinMaxScaled features from neighbors
4. Parallelize across spots using `n_jobs`

**Sources:** [src/aestetik/utils/utils_grid.py]()

### Weight Calibration Logic

The system automatically balances modality contributions based on clustering quality:

```python
def calibrate_weights(silhouette_transcriptomics, silhouette_morphology,
                      morphology_weight, total_weight):
    if silhouette_transcriptomics > silhouette_morphology:
        # Morphology provides weaker signal
        transcriptomics_weight = total_weight - morphology_weight
    else:
        # Transcriptomics provides weaker signal
        transcriptomics_weight = morphology_weight
        morphology_weight = total_weight - morphology_weight
    return transcriptomics_weight, morphology_weight
```

**Sources:** [src/aestetik/utils/utils_data.py]()

### Triplet Sampling Strategy

The `CustomDataset` implements triplet sampling with configurable repeats:

- **Anchor**: Random spot from dataset
- **Positives**: `n_repeats` spots from same cluster (both modalities)
- **Negatives**: `n_repeats` spots from different cluster (both modalities)
- Sampling ensures diverse triplets across epochs

**Sources:** [src/aestetik/dataloader.py]()

---

## Summary

AESTETIK's architecture demonstrates several key design principles:

1. **Separation of Concerns**: Clear boundaries between API, orchestration, core components, and utilities
2. **Framework Integration**: Deep integration with PyTorch Lightning and AnnData ecosystems
3. **Flexible Configuration**: Parameter dictionaries enable modular component configuration
4. **Lazy Initialization**: Components constructed only when data characteristics are known
5. **Ensemble Robustness**: Multiple encoder/decoder instances with median aggregation
6. **Stochastic Inference**: Monte Carlo dropout for uncertainty-aware predictions

The system's modular design allows for extension at multiple points: custom clustering methods, alternative model architectures, different loss functions, and additional preprocessing utilities.

**Sources:** [src/aestetik/AESTETIK.py](), [src/aestetik/modules/aestetik_module.py](), [src/aestetik/data_modules/data_module.py](), [src/aestetik/models/model.py]()

---