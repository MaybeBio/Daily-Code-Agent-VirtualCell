# SCimilarity Overview

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.rst](README.rst)
- [docs/about.rst](docs/about.rst)
- [setup.cfg](setup.cfg)

</details>



## Purpose and Scope

This document provides a comprehensive overview of the SCimilarity system architecture, core components, and fundamental workflows. SCimilarity is a cell atlas foundation model that enables scalable similarity search across millions of single-cell RNA sequencing profiles.

This overview covers the system's high-level design, key classes, and data flow patterns. For detailed installation instructions, see [Installation & Setup](#2). For specific API documentation, see [Core API Reference](#3). For hands-on examples, see [Tutorials & Examples](#8).

## What is SCimilarity

SCimilarity is a machine learning system that creates unified representations of single-cell expression profiles, enabling quantitative similarity measurements between cellular states. The system provides a novel cell search capability that can efficiently query millions of single-cell profiles to find cells similar to a query state, allowing researchers to systematically leverage massive public scRNA-seq atlases.

Key capabilities include:
- **Cell similarity search**: Find cells with similar expression patterns across large atlases
- **Cell type annotation**: Automatically classify cells based on reference datasets  
- **Cross-study generalization**: Represent new studies without additional model training
- **Scalable embedding**: Process millions of single-cell profiles efficiently

**Sources:** [README.rst:1-43](), [setup.cfg:6-15]()

## System Architecture

SCimilarity follows a hierarchical, data-centric architecture built around efficient storage and retrieval of single-cell embeddings and metadata.

### Core Class Hierarchy

```mermaid
graph TB
    subgraph "Base Infrastructure"
        CSK["CellSearchKNN<br/>• Model loading<br/>• GPU/CPU execution<br/>• kNN search foundation"]
    end
    
    subgraph "Specialized Components"  
        CE["CellEmbedding<br/>• Embedding generation<br/>• Batch processing<br/>• Gene alignment"]
        CQ["CellQuery<br/>• Similarity search<br/>• Metadata compilation<br/>• Result filtering"]
        CA["CellAnnotation<br/>• Cell type prediction<br/>• Reference-based classification<br/>• Label consensus"]
        INT["Interpreter<br/>• Model interpretation<br/>• Gene attribution<br/>• Feature analysis"]
    end
    
    subgraph "Data Layer"
        TDB["TileDB Arrays<br/>CellArr storage"]
        ZARR["Zarr Arrays<br/>Expression matrices"]
        HNSW["HNSW Indices<br/>hnswlib integration"]
        META["Cell Metadata<br/>Study/cell annotations"]
    end
    
    CSK --> CE
    CSK --> CQ  
    CSK --> CA
    CE --> INT
    
    TDB --> CE
    TDB --> CQ
    ZARR --> CE
    HNSW --> CSK
    META --> CQ
    META --> CA
```

**Sources:** Architecture diagrams provided, [setup.cfg:49-62]()

### Data Storage Architecture  

```mermaid
graph LR
    subgraph "Input Data"
        ANNDATA["AnnData<br/>scanpy objects"]
        CSV["CSV Metadata<br/>Cell annotations"]
        EXPR["Expression Matrices<br/>Raw counts"]
    end
    
    subgraph "Storage Systems"
        CELLARR["CellArr<br/>TileDB backend<br/>Aligned expression data"]
        ZARR_STORE["Zarr Arrays<br/>Chunked storage<br/>Compressed matrices"]
        EMB_STORE["Embedding Storage<br/>Dense vectors<br/>Search indices"]
    end
    
    subgraph "Access Patterns"
        BATCH["Batch Processing<br/>Training data loading"]
        SEARCH["Search Queries<br/>kNN retrieval"]
        STREAM["Streaming Access<br/>On-demand loading"]
    end
    
    ANNDATA --> CELLARR
    CSV --> CELLARR
    EXPR --> ZARR_STORE
    
    CELLARR --> BATCH
    CELLARR --> STREAM
    EMB_STORE --> SEARCH
    ZARR_STORE --> BATCH
```

**Sources:** [setup.cfg:54-61]()

## Core Workflows

SCimilarity supports three primary workflows, each implemented through specific class interfaces and data processing pipelines.

### Embedding and Search Pipeline

```mermaid
flowchart TD
    subgraph "Data Input"
        INPUT["New scRNA-seq Data<br/>AnnData format"]
    end
    
    subgraph "CellEmbedding Processing"
        ALIGN["align_dataset()<br/>Gene alignment to model"]
        NORM["lognorm_counts()<br/>Log normalization"]
        EMBED["CellEmbedding.get_embeddings()<br/>Model inference"]
    end
    
    subgraph "CellQuery Search"
        QUERY["CellQuery.search()<br/>Similarity calculation"]
        KNN["kNN Index Lookup<br/>HNSW search"]
        FILTER["Result filtering<br/>Metadata compilation"]
    end
    
    subgraph "Output"
        RESULTS["Similar Cells<br/>Distances & metadata"]
    end
    
    INPUT --> ALIGN
    ALIGN --> NORM
    NORM --> EMBED
    EMBED --> QUERY
    QUERY --> KNN
    KNN --> FILTER
    FILTER --> RESULTS
```

### Annotation Workflow

```mermaid
flowchart TD
    subgraph "Reference Data"
        REF["Reference Atlas<br/>CellArr + metadata"]
        LABELS["Cell Type Labels<br/>Ontology mappings"]
    end
    
    subgraph "CellAnnotation Processing"
        LOAD["CellAnnotation.load()<br/>Reference loading"]
        PREDICT["CellAnnotation.annotate()<br/>kNN classification"]
        CONSENSUS["Label consensus<br/>Voting mechanism"]
        FILTER_ANN["Filter by confidence<br/>Threshold application"]
    end
    
    subgraph "Query Processing"
        QUERY_EMB["Query embeddings<br/>From CellEmbedding"]
    end
    
    subgraph "Output"
        PREDICTIONS["Cell Type Predictions<br/>Confidence scores"]
    end
    
    REF --> LOAD
    LABELS --> LOAD
    QUERY_EMB --> PREDICT
    LOAD --> PREDICT
    PREDICT --> CONSENSUS
    CONSENSUS --> FILTER_ANN
    FILTER_ANN --> PREDICTIONS
```

**Sources:** Architecture diagrams provided

## Key Components and Dependencies

### Core Dependencies

SCimilarity integrates several key technologies to provide its functionality:

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| `torch` | Deep learning framework | Model training and inference |
| `pytorch-lightning` | Training orchestration | Metric learning pipeline |
| `tiledb` | Array storage | CellArr data backend |
| `tiledb-vector-search` | Vector indices | Fast similarity search |
| `hnswlib` | Approximate kNN | HNSW index implementation |
| `scanpy` | Single-cell analysis | Data preprocessing |
| `zarr` | Array storage | Chunked data arrays |
| `captum` | Model interpretation | Gene attribution analysis |

### Training Infrastructure

```mermaid
graph TB
    subgraph "Data Loading"
        TILEDB_DM["TiledbDataModule<br/>PyTorch Lightning"]
        ANNDATA_DM["AnndataDataModule<br/>Direct AnnData loading"]
        TRIPLET["TripletSelector<br/>Training sample selection"]
    end
    
    subgraph "Model Architecture"
        ENCODER["Encoder Network<br/>Expression → Embedding"]
        DECODER["Decoder Network<br/>Embedding → Expression"]
        METRIC["MetricLearning<br/>Triplet loss training"]
    end
    
    subgraph "Training Loop"
        LIGHTNING["PyTorch Lightning<br/>Training orchestration"]
        EVAL["Evaluation Metrics<br/>ASW, NMSE scoring"]
    end
    
    TILEDB_DM --> METRIC
    ANNDATA_DM --> METRIC
    TRIPLET --> METRIC
    
    METRIC --> ENCODER
    METRIC --> DECODER
    METRIC --> LIGHTNING
    
    LIGHTNING --> EVAL
```

**Sources:** [setup.cfg:49-62]()

## Data Flow and Processing

The SCimilarity system processes single-cell data through a series of standardized steps that ensure compatibility across different studies and datasets.

### End-to-End Data Processing

```mermaid
flowchart LR
    subgraph "Input Processing"
        RAW["Raw Counts<br/>Study-specific genes"]
        ALIGN_STEP["Gene Alignment<br/>Map to model features"]
        NORM_STEP["Log Normalization<br/>log1p transformation"]
    end
    
    subgraph "Model Inference"
        MODEL["Pretrained Model<br/>MetricLearning weights"]
        ENCODE["Encoder Forward Pass<br/>Generate embeddings"]
    end
    
    subgraph "Search Infrastructure"
        INDEX["Search Index<br/>HNSW or TileDB-VS"]
        SEARCH_EXEC["Search Execution<br/>kNN retrieval"]
        METADATA["Metadata Lookup<br/>Cell annotations"]
    end
    
    subgraph "Result Compilation"
        COMPILE["Result Compilation<br/>Distances + metadata"]
        FORMAT["Output Formatting<br/>DataFrames/AnnData"]
    end
    
    RAW --> ALIGN_STEP
    ALIGN_STEP --> NORM_STEP
    NORM_STEP --> MODEL
    MODEL --> ENCODE
    ENCODE --> INDEX
    INDEX --> SEARCH_EXEC
    SEARCH_EXEC --> METADATA
    METADATA --> COMPILE
    COMPILE --> FORMAT
```

**Sources:** Architecture diagrams provided

## Deployment and Distribution

SCimilarity is distributed through multiple channels to support different usage patterns:

- **PyPI Package**: Core API installation via `pip install scimilarity`
- **Docker Container**: Complete environment at `ghcr.io/genentech/scimilarity`  
- **Zenodo Assets**: Pretrained models, embeddings, and indices
- **Documentation Site**: API docs and tutorials at GitHub Pages

The system supports both local development and cloud deployment scenarios, with GPU acceleration available for large-scale processing tasks.

**Sources:** [README.rst:19-32](), [setup.cfg:1-43]()

---