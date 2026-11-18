# Data Management & Storage

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [NEWS.rst](NEWS.rst)

</details>



## Purpose and Scope

This document covers SCimilarity's data management and storage infrastructure, including the evolution from Zarr-based storage to the current CellArr/TileDB framework. It details the data models, storage formats, and integration patterns that enable efficient single-cell data operations at scale.

For information about data loading during training, see [Data Loading Systems](#4.1). For post-training data structure creation, see [Post-Training Workflows](#6).

## Storage Architecture Evolution

SCimilarity has undergone a significant architectural evolution in its data storage approach. Version 0.4.0 introduced a major transition from Zarr-based workflows to CellArr, a TileDB-based framework, as the primary data store to streamline the end-to-end process.

### Current Architecture: CellArr/TileDB Framework

The current storage architecture centers around CellArr, which provides a TileDB-based framework for storing and accessing single-cell data efficiently.

#### CellArr/TileDB Storage System

```mermaid
graph TB
    subgraph "CellArr Storage Layer"
        CELLARR["CellArr Store"]
        TILEDB_DF["TileDB DataFrames"]
        TILEDB_ARRAYS["TileDB Arrays"]
        CELL_META["Cell Metadata"]
    end
    
    subgraph "Data Access Layer"
        QUERY_UTILS["query_tiledb_df"]
        ADATA_EXTRACT["adata_from_tiledb"]
        EMB_EXTRACT["embedding_from_tiledb"]
        ALIGN_UTILS["align_dataset"]
    end
    
    subgraph "Training Integration"
        TILEDB_DM["TileDBDataModule"]
        TILEDB_DS["TileDBDataset"]
        TILEDB_SAMPLER["TileDBSampler"]
    end
    
    subgraph "Core Components"
        CELL_QUERY["CellQuery"]
        CELL_EMBEDDING["CellEmbedding"]
        CELL_ANNOTATION["CellAnnotation"]
    end
    
    CELLARR --> TILEDB_DF
    CELLARR --> TILEDB_ARRAYS
    CELLARR --> CELL_META
    
    TILEDB_DF --> QUERY_UTILS
    TILEDB_ARRAYS --> ADATA_EXTRACT
    TILEDB_ARRAYS --> EMB_EXTRACT
    CELLARR --> ALIGN_UTILS
    
    CELLARR --> TILEDB_DM
    TILEDB_DM --> TILEDB_DS
    TILEDB_DM --> TILEDB_SAMPLER
    
    QUERY_UTILS --> CELL_QUERY
    ADATA_EXTRACT --> CELL_EMBEDDING
    EMB_EXTRACT --> CELL_ANNOTATION
    ALIGN_UTILS --> CELL_EMBEDDING
```

**Sources:** [NEWS.rst:29-41]()

#### Data Storage Components

| Component | Purpose | File Location |
|-----------|---------|---------------|
| `CellArr Store` | Primary TileDB-based data storage | Referenced in training scripts |
| `TileDBDataModule` | PyTorch Lightning data module for TileDB | `scimilarity/tiledb_data_models.py` |
| `TileDBDataset` | Dataset class for TileDB data access | `scimilarity/tiledb_data_models.py` |
| `TileDBSampler` | Sampling strategies for TileDB data | `scimilarity/tiledb_data_models.py` |

### Legacy Architecture: Zarr Storage

Prior to version 0.4.0, SCimilarity used Zarr-based storage workflows. While this approach has been superseded, understanding it provides context for the architectural evolution.

```mermaid
graph TB
    subgraph "Legacy Zarr Storage"
        ZARR_ARRAYS["Zarr Arrays"]
        ZARR_META["Zarr Metadata"]
        ZARR_CHUNKS["Chunked Storage"]
    end
    
    subgraph "Legacy Data Loading"
        ZARR_DM["Zarr DataModule"]
        ZARR_DS["Zarr Dataset"]
    end
    
    subgraph "Migration Path"
        OLD_WORKFLOW["Old Zarr Workflows"]
        NEW_WORKFLOW["New CellArr Workflows"]
        MIGRATION["Data Migration"]
    end
    
    ZARR_ARRAYS --> ZARR_DM
    ZARR_META --> ZARR_DM
    ZARR_CHUNKS --> ZARR_DS
    
    OLD_WORKFLOW --> MIGRATION
    MIGRATION --> NEW_WORKFLOW
    
    style OLD_WORKFLOW fill:#ffebee
    style NEW_WORKFLOW fill:#e8f5e8
```

**Sources:** [NEWS.rst:29-31]()

## Data Models and Access Patterns

### TileDB Data Access Utilities

SCimilarity provides several utility functions for accessing data stored in the TileDB format:

#### Core Data Access Functions

| Function | Purpose | Return Type |
|----------|---------|-------------|
| `query_tiledb_df` | Query TileDB dataframes with filtering | DataFrame |
| `adata_from_tiledb` | Extract cells from TileDB stores by index, including raw counts | AnnData |
| `embedding_from_tiledb` | Extract pre-computed embeddings from TileDB | Array |
| `align_dataset` | Align datasets with reference gene sets | AnnData |

```mermaid
graph LR
    subgraph "Data Query Flow"
        USER_QUERY["User Query"]
        TILEDB_QUERY["query_tiledb_df"]
        CELL_EXTRACT["adata_from_tiledb"]
        EMB_EXTRACT["embedding_from_tiledb"]
    end
    
    subgraph "TileDB Storage"
        TILEDB_STORE["TileDB Store"]
        CELL_DATA["Cell Expression Data"]
        METADATA["Cell Metadata"]
        EMBEDDINGS["Pre-computed Embeddings"]
    end
    
    subgraph "Output Formats"
        ADATA_OUT["AnnData Objects"]
        DF_OUT["Pandas DataFrames"]
        ARRAY_OUT["NumPy Arrays"]
    end
    
    USER_QUERY --> TILEDB_QUERY
    USER_QUERY --> CELL_EXTRACT
    USER_QUERY --> EMB_EXTRACT
    
    TILEDB_QUERY --> METADATA
    CELL_EXTRACT --> CELL_DATA
    EMB_EXTRACT --> EMBEDDINGS
    
    TILEDB_QUERY --> DF_OUT
    CELL_EXTRACT --> ADATA_OUT
    EMB_EXTRACT --> ARRAY_OUT
```

**Sources:** [NEWS.rst:38-41]()

### Data Processing Pipeline Integration

The storage layer integrates with SCimilarity's data processing pipeline through several key interfaces:

#### Data Alignment and Normalization

```mermaid
graph TB
    subgraph "Input Data Sources"
        RAW_ADATA["Raw AnnData"]
        CSV_META["CSV Metadata"]
        REF_GENES["Reference Gene Set"]
    end
    
    subgraph "Processing Layer"
        ALIGN_FUNC["align_dataset"]
        LOGNORM["lognorm_counts"]
        PREPROCESS["Data Preprocessing"]
    end
    
    subgraph "TileDB Storage"
        CELLARR_STORE["CellArr Store"]
        ALIGNED_DATA["Aligned Gene Expression"]
        NORM_DATA["Log-Normalized Counts"]
        META_STORE["Metadata Storage"]
    end
    
    RAW_ADATA --> ALIGN_FUNC
    REF_GENES --> ALIGN_FUNC
    CSV_META --> PREPROCESS
    
    ALIGN_FUNC --> LOGNORM
    LOGNORM --> CELLARR_STORE
    PREPROCESS --> META_STORE
    
    CELLARR_STORE --> ALIGNED_DATA
    CELLARR_STORE --> NORM_DATA
    CELLARR_STORE --> META_STORE
```

**Sources:** [NEWS.rst:38-41]()

## Training Data Integration

The TileDB storage system is tightly integrated with the training infrastructure through specialized data modules and samplers.

### TileDB Training Components

```mermaid
graph TB
    subgraph "TileDB Training Stack"
        TILEDB_DM["TileDBDataModule"]
        TILEDB_DS["TileDBDataset"]
        TILEDB_SAMPLER["TileDBSampler"]
        TRIPLET_SEL["TripletSelector"]
    end
    
    subgraph "PyTorch Lightning Integration"
        PL_TRAINER["PyTorch Lightning Trainer"]
        METRIC_LEARNING["MetricLearning Model"]
        TRAIN_LOOP["Training Loop"]
    end
    
    subgraph "Data Access"
        CELLARR["CellArr Store"]
        BATCH_LOADER["Batch Loading"]
        SAMPLING_WEIGHTS["Sampling Weights"]
    end
    
    CELLARR --> TILEDB_DS
    TILEDB_DS --> TILEDB_DM
    TILEDB_DM --> TILEDB_SAMPLER
    TILEDB_SAMPLER --> TRIPLET_SEL
    
    TILEDB_DM --> PL_TRAINER
    PL_TRAINER --> METRIC_LEARNING
    METRIC_LEARNING --> TRAIN_LOOP
    
    TILEDB_DS --> BATCH_LOADER
    TILEDB_SAMPLER --> SAMPLING_WEIGHTS
```

**Sources:** [NEWS.rst:32-33]()

### Multiprocessing Support

Version 0.4.1 introduced refactoring to support `spawn` as the multiprocessing context for TileDB and AnnData dataloaders, improving compatibility across different systems.

**Sources:** [NEWS.rst:15-16]()

## Storage Performance and Optimization

### TileDB Advantages

The transition to TileDB provides several performance benefits:

| Feature | Benefit | Use Case |
|---------|---------|----------|
| Columnar Storage | Efficient compression and query performance | Large-scale expression matrices |
| Sparse Array Support | Memory-efficient storage of sparse single-cell data | scRNA-seq count matrices |
| Multi-dimensional Indexing | Fast querying by cell and gene indices | Rapid data retrieval |
| Metadata Integration | Co-located data and metadata | Unified data access patterns |

### Data Model Evolution

The evolution from Zarr to TileDB represents a strategic shift toward more integrated and performant data management:

```mermaid
graph LR
    subgraph "Version Evolution"
        V03["v0.3.0<br/>Zarr-based<br/>Workflows"]
        V04["v0.4.0<br/>CellArr/TileDB<br/>Framework"]
        V041["v0.4.1<br/>Multiprocessing<br/>Improvements"]
    end
    
    subgraph "Storage Evolution"
        ZARR_STORAGE["Zarr Arrays<br/>Separate Metadata"]
        TILEDB_STORAGE["TileDB Arrays<br/>Integrated Metadata"]
        OPTIMIZED_STORAGE["Optimized TileDB<br/>Spawn Support"]
    end
    
    subgraph "Workflow Evolution"
        MANUAL_WORKFLOW["Manual Data<br/>Management"]
        STREAMLINED_WORKFLOW["Streamlined<br/>End-to-End Process"]
        ROBUST_WORKFLOW["Robust<br/>Multiprocessing"]
    end
    
    V03 --> V04
    V04 --> V041
    
    ZARR_STORAGE --> TILEDB_STORAGE
    TILEDB_STORAGE --> OPTIMIZED_STORAGE
    
    MANUAL_WORKFLOW --> STREAMLINED_WORKFLOW
    STREAMLINED_WORKFLOW --> ROBUST_WORKFLOW
```

**Sources:** [NEWS.rst:20-41]()

## Integration with Core Components

The storage layer provides the foundation for all major SCimilarity components:

### Component Storage Dependencies

| Component | Storage Dependencies | Key Methods |
|-----------|---------------------|-------------|
| `CellEmbedding` | TileDB arrays, gene alignment | `align_dataset`, `embedding_from_tiledb` |
| `CellQuery` | TileDB metadata, pre-computed embeddings | `query_tiledb_df`, `adata_from_tiledb` |
| `CellAnnotation` | Reference embeddings, cell metadata | `embedding_from_tiledb`, annotation indices |

**Sources:** [NEWS.rst:38-41]()

---