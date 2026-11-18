# Quick Start Tutorial

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [example/gettingStartedWithAESTETIK.ipynb](example/gettingStartedWithAESTETIK.ipynb)
- [example/gettingStartedWithAESTETIKwithSimulatedData.ipynb](example/gettingStartedWithAESTETIKwithSimulatedData.ipynb)
- [figures/maynard_human_brain_analysis_151676_Transcriptomics_Morphology_AESTETIK.png](figures/maynard_human_brain_analysis_151676_Transcriptomics_Morphology_AESTETIK.png)
- [llms.txt](llms.txt)

</details>



## Purpose and Scope

This page provides a hands-on walkthrough of the basic AESTETIK workflow using the example notebooks provided in the repository. It demonstrates the complete pipeline from loading spatial transcriptomics data through model training to visualization of results.

For installation instructions, see [Installation](#2.1). For detailed parameter tuning and advanced workflows, see [Training Models](#4.2) and [Making Predictions](#4.3). For comprehensive API documentation, see [AESTETIK Class](#5.1).

---

## Tutorial Overview

The AESTETIK workflow consists of five primary stages executed through example notebooks. The tutorial uses the LIBD Human DLPFC dataset (sample 151676) to demonstrate multi-modal spatial transcriptomics analysis.

```mermaid
graph LR
    A["Load Data<br/>(AnnData)"]
    B["Extract<br/>Morphology<br/>Features"]
    C["Train Model<br/>AESTETIK.fit()"]
    D["Predict<br/>Embeddings<br/>AESTETIK.predict()"]
    E["Visualize<br/>Results"]
    
    A --> B
    B --> C
    C --> D
    D --> E
```

**Sources:** [example/gettingStartedWithAESTETIK.ipynb:10-14]()

---

## Tutorial Workflow

### Stage 1: Data Loading and Preprocessing

The tutorial begins by loading spatial transcriptomics data stored in AnnData format. The dataset requires specific spatial coordinates and preprocessed transcriptomics features.

```mermaid
graph TB
    subgraph "Input Files"
        H5AD["151676.h5ad<br/>(AnnData object)"]
        IMG["151676.png<br/>(H&E image)"]
        JSON["151676.json<br/>(metadata)"]
    end
    
    subgraph "Loading Process"
        SC["sc.read_h5ad()"]
        PREP["preprocess_adata()"]
    end
    
    subgraph "Required Data Fields"
        OBS["adata.obs:<br/>x_array, y_array<br/>x_pixel, y_pixel"]
        OBSM["adata.obsm:<br/>X_pca"]
    end
    
    H5AD --> SC
    SC --> PREP
    PREP --> OBS
    PREP --> OBSM
    JSON -.->|"spot_diameter_fullres<br/>dot_size"| PREP
```

The preprocessing step uses the `preprocess_adata()` utility function from the transcriptomics utilities module. This function ensures the AnnData object contains the required spatial coordinate fields (`x_array`, `y_array`, `x_pixel`, `y_pixel`) and performs dimensionality reduction to generate `X_pca` features.

**Key Data Requirements:**
| Field Location | Field Name | Description |
|---|---|---|
| `adata.obs` | `x_array`, `y_array` | Spatial coordinates on the array grid |
| `adata.obs` | `x_pixel`, `y_pixel` | Spatial coordinates in image pixel space |
| `adata.obsm` | `X_pca` | PCA-reduced transcriptomics features |
| metadata JSON | `spot_diameter_fullres` | Spot diameter in full resolution pixels |
| metadata JSON | `dot_size` | Visualization dot size parameter |

**Sources:** [example/gettingStartedWithAESTETIK.ipynb:100-155](), [aestetik/utils/utils_transcriptomics.py]()

---

### Stage 2: Morphology Feature Extraction

Morphology features are extracted from H&E images using a pre-trained CNN (Inception V3). The extraction process crops spot-level image patches and generates embeddings via the model's penultimate layer.

```mermaid
sequenceDiagram
    participant Notebook as Tutorial Notebook
    participant InceptionV3 as "inception_v3<br/>(torchvision)"
    participant ExtractFunc as "extract_morphology_embeddings()"
    participant PCA as "PCA<br/>(sklearn)"
    participant AnnData as "adata.obsm"
    
    Notebook->>InceptionV3: Load weights (DEFAULT)
    Notebook->>InceptionV3: Replace fc layer with Identity()
    Notebook->>ExtractFunc: Pass model, img_path,<br/>coordinates, spot_diameter
    
    ExtractFunc->>ExtractFunc: Crop spot patches from image
    ExtractFunc->>InceptionV3: Forward pass (feature_dim=2048)
    InceptionV3-->>ExtractFunc: Return embeddings
    
    ExtractFunc->>PCA: fit_transform(n_components=15)
    PCA-->>ExtractFunc: Reduced features
    
    ExtractFunc-->>AnnData: Store in X_pca_morphology
```

The extraction process is implemented in `extract_morphology_embeddings()` which performs the following operations:

1. Loads the H&E image using efficient image processing (pyvips)
2. Crops circular patches centered at each spot's pixel coordinates
3. Applies preprocessing transformations (resize, normalize)
4. Extracts features using the CNN's penultimate layer (2048 dimensions)
5. Applies PCA to reduce dimensions to match transcriptomics features (typically 15 components)

After extraction, the notebook stores features in two modality-specific keys:

| Key in `adata.obsm` | Content | Shape |
|---|---|---|
| `X_pca_transcriptomics` | Transcriptomics PCA features | n_spots × 15 |
| `X_pca_morphology` | Morphology CNN+PCA features | n_spots × 15 |

**Sources:** [example/gettingStartedWithAESTETIK.ipynb:194-260](), [aestetik/utils/utils_morphology.py]()

---

### Stage 3: Model Training

Model training is orchestrated through the `AESTETIK` class, which initializes the model architecture and invokes the PyTorch Lightning training loop.

```mermaid
graph TB
    subgraph "Model Initialization"
        INIT["AESTETIK(<br/>nCluster=n,<br/>morphology_weight=0,<br/>refine_cluster=True,<br/>window_size=7<br/>)"]
    end
    
    subgraph "Training Pipeline"
        FIT["model.fit(X=adata)"]
        VALIDATE["_validate_fit_inputs()"]
        DATAMODULE["AESTETIKDataModule"]
        CLUSTER["Clustering:<br/>transcriptomics<br/>morphology"]
        CALIBRATE["calibrate_weights()"]
        GRID["build_grid()"]
        TRAINER["Lightning Trainer"]
        LITMODEL["AESTETIKModel"]
    end
    
    subgraph "Model Architecture"
        ENCODER["Encoder<br/>Ensemble"]
        DECODER["Decoder<br/>Ensemble"]
    end
    
    INIT --> FIT
    FIT --> VALIDATE
    VALIDATE --> DATAMODULE
    DATAMODULE --> CLUSTER
    DATAMODULE --> CALIBRATE
    DATAMODULE --> GRID
    GRID --> TRAINER
    TRAINER --> LITMODEL
    LITMODEL --> ENCODER
    LITMODEL --> DECODER
```

**Key Parameters in Tutorial:**

| Parameter | Value | Purpose |
|---|---|---|
| `nCluster` | Number of ground truth clusters | Determines number of expected spatial domains |
| `morphology_weight` | 0 | Controls morphology contribution (0 = transcriptomics only) |
| `refine_cluster` | True | Enables spatial cluster refinement |
| `window_size` | 7 | Spatial grid size (7×7 neighborhood) |

The `fit()` method performs the following sequence:

1. **Input Validation:** Checks for required fields in `adata.obsm` (`X_pca_transcriptomics`, `X_pca_morphology`) and `adata.obs` (spatial coordinates)
2. **Data Module Setup:** Instantiates `AESTETIKDataModule` which prepares training data
3. **Clustering:** Clusters both modalities separately to generate triplet loss labels
4. **Weight Calibration:** Automatically adjusts modality weights based on cluster quality
5. **Grid Construction:** Builds spatial grids for each spot using KDTree neighbor queries
6. **Model Building:** Constructs the autoencoder architecture
7. **Training Loop:** Executes PyTorch Lightning training with the configured trainer

**Sources:** [example/gettingStartedWithAESTETIK.ipynb:283-320](), [aestetik/aestetik.py:fit()]()

---

### Stage 4: Predictions and Clustering

After training, the `predict()` method generates latent space embeddings and optionally performs clustering on the learned representations.

```mermaid
sequenceDiagram
    participant User as Tutorial Notebook
    participant Model as "AESTETIK instance"
    participant Predict as "predict() method"
    participant Encoder as "Encoder ensemble"
    participant Clustering as "clustering() function"
    participant AnnData as "adata"
    
    User->>Model: predict(X=adata, cluster=True)
    
    Model->>Predict: _check_fitted()
    Model->>Predict: _validate_predict_inputs()
    Model->>Predict: _compute_latent_space()
    
    Predict->>Encoder: Multiple forward passes<br/>(Monte Carlo dropout)
    Encoder-->>Predict: Latent embeddings
    
    Predict->>Predict: Average embeddings
    Predict->>AnnData: Store in adata.obsm["AESTETIK"]
    
    alt cluster=True
        Predict->>Clustering: clustering(adata.obsm["AESTETIK"],<br/>nCluster)
        Clustering-->>AnnData: Store in adata.obs["AESTETIK_cluster"]
    end
    
    Predict-->>User: None (adata modified in-place)
```

**Output Data Structure:**

The `predict()` method modifies the input AnnData object in-place, adding the following fields:

| Field Location | Field Name | Content |
|---|---|---|
| `adata.obsm` | `AESTETIK` | Latent space embeddings (n_spots × latent_dim) |
| `adata.obs` | `AESTETIK_cluster` | Cluster assignments (categorical) |
| `adata.obs` | `X_pca_transcriptomics_cluster` | Transcriptomics-only clusters (from preprocessing) |
| `adata.obs` | `X_pca_morphology_cluster` | Morphology-only clusters (from preprocessing) |

The method uses stochastic forward passes (Monte Carlo dropout) to compute robust embeddings by averaging multiple predictions per spot. The clustering step applies the method specified during initialization (default: Mclust) to the learned representations.

**Sources:** [example/gettingStartedWithAESTETIK.ipynb:334-364](), [aestetik/aestetik.py:predict()]()

---

### Stage 5: Visualization

The tutorial demonstrates two visualization approaches: spatial scatter plots for cluster comparison and training loss plots for model diagnostics.

```mermaid
graph LR
    subgraph "Spatial Visualization"
        SQUIDPY["sq.pl.spatial_scatter()"]
        CLUSTERS["Compare:<br/>ground_truth<br/>transcriptomics_cluster<br/>morphology_cluster<br/>AESTETIK_cluster"]
    end
    
    subgraph "Training Diagnostics"
        VISUALIZE["visualize(model=model,<br/>plot_loss=True)"]
        LOSS["Training loss curves"]
    end
    
    ADATA["adata"] --> SQUIDPY
    SQUIDPY --> CLUSTERS
    
    MODEL["trained model"] --> VISUALIZE
    VISUALIZE --> LOSS
```

**Spatial Cluster Visualization:**

The tutorial uses `sq.pl.spatial_scatter()` from squidpy to visualize cluster assignments in spatial context. This allows comparison of:
- Ground truth annotations (if available)
- Single-modality clusters (transcriptomics-only, morphology-only)
- Multi-modal AESTETIK clusters

**Training Loss Visualization:**

The `visualize()` utility function plots training metrics stored during the Lightning training loop, including:
- Total loss over epochs
- Reconstruction loss components (transcriptomics, morphology)
- Triplet loss components (transcriptomics, morphology)

**Sources:** [example/gettingStartedWithAESTETIK.ipynb:367-398](), [aestetik/utils/utils_visualization.py]()

---

## Alternative Tutorial: Simulated Data

The repository provides a second tutorial demonstrating AESTETIK on simulated spatial transcriptomics data. This tutorial is useful for understanding expected performance when modalities are complementary.

```mermaid
graph TB
    subgraph "Simulated Data Setup"
        SIMDATA["A.h5ad<br/>(simulated dataset)"]
        FEATURES["adata.obsm['X_pca']<br/>adata.obsm['image']"]
    end
    
    subgraph "Baseline Comparisons"
        KMEANS_T["KMeans on<br/>transcriptomics only"]
        KMEANS_M["KMeans on<br/>morphology only"]
        KMEANS_C["KMeans on<br/>concatenated features"]
    end
    
    subgraph "AESTETIK Training"
        PARAMS["morphology_weight=1.5<br/>refine_cluster=1<br/>window_size=3"]
        FITPREDICT["fit_predict(adata)"]
    end
    
    SIMDATA --> FEATURES
    FEATURES --> KMEANS_T
    FEATURES --> KMEANS_M
    FEATURES --> KMEANS_C
    FEATURES --> PARAMS
    PARAMS --> FITPREDICT
```

**Key Differences from Main Tutorial:**

| Aspect | Main Tutorial | Simulated Data Tutorial |
|---|---|---|
| Dataset | LIBD Human DLPFC (151676) | Synthetic spatial transcriptomics |
| Morphology weight | 0 (transcriptomics-only) | 1.5 (morphology important) |
| Window size | 7×7 | 3×3 |
| Comparisons | Ground truth vs AESTETIK | KMeans vs AESTETIK on all modality combinations |
| API usage | Separate `fit()` + `predict()` | Combined `fit_predict()` |

The simulated data tutorial demonstrates how to use `fit_predict()`, which combines training and prediction in a single call. This method is equivalent to calling `fit()` followed by `predict()`.

**Sources:** [example/gettingStartedWithAESTETIKwithSimulatedData.ipynb:1-296]()

---

## Complete Workflow Code Structure

The following diagram maps tutorial steps to concrete code entities in the AESTETIK codebase:

```mermaid
graph TB
    subgraph "Tutorial User Code"
        N1["preprocess_adata()"]
        N2["extract_morphology_<br/>embeddings()"]
        N3["AESTETIK(nCluster=n)"]
        N4["model.fit(X=adata)"]
        N5["model.predict(X=adata)"]
        N6["visualize(model)"]
    end
    
    subgraph "AESTETIK Core Classes"
        C1["aestetik/aestetik.py:<br/>AESTETIK class"]
        C2["aestetik/datamodule.py:<br/>AESTETIKDataModule"]
        C3["aestetik/litmodel.py:<br/>AESTETIKModel"]
        C4["aestetik/model.py:<br/>AE"]
    end
    
    subgraph "Utility Modules"
        U1["utils_transcriptomics.py:<br/>preprocess_adata()"]
        U2["utils_morphology.py:<br/>extract_morphology_embeddings()"]
        U3["utils_data.py:<br/>prepare_input_for_model()"]
        U4["utils_clustering.py:<br/>clustering()"]
        U5["utils_visualization.py:<br/>visualize()"]
    end
    
    N1 --> U1
    N2 --> U2
    N3 --> C1
    N4 --> C1
    N5 --> C1
    N6 --> U5
    
    C1 --> C2
    C1 --> C3
    C2 --> U3
    C2 --> U4
    C3 --> C4
```

**Sources:** [example/gettingStartedWithAESTETIK.ipynb](), [aestetik/aestetik.py](), [aestetik/datamodule.py](), [aestetik/litmodel.py]()

---

## Expected Outputs

After completing the tutorial, the AnnData object contains the following new fields:

```mermaid
graph TB
    subgraph "Input AnnData"
        IN_OBS["adata.obs:<br/>x_array, y_array<br/>x_pixel, y_pixel"]
        IN_OBSM["adata.obsm:<br/>X_pca_transcriptomics<br/>X_pca_morphology"]
    end
    
    subgraph "Output AnnData"
        OUT_OBS["adata.obs:<br/>+ X_pca_transcriptomics_cluster<br/>+ X_pca_morphology_cluster<br/>+ AESTETIK_cluster"]
        OUT_OBSM["adata.obsm:<br/>+ AESTETIK<br/>(latent embeddings)"]
    end
    
    IN_OBS --> OUT_OBS
    IN_OBSM --> OUT_OBSM
```

These outputs can be used for downstream analysis including:
- UMAP visualization of learned representations
- Differential expression analysis between AESTETIK clusters
- Spatial domain characterization
- Cross-sample integration (see [Making Predictions](#4.3))

**Sources:** [example/gettingStartedWithAESTETIK.ipynb:353-378](), [README.md:65-74]()

---