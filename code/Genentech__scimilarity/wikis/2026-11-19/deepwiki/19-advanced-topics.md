# Advanced Topics

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/scimilarity/interpreter.py](src/scimilarity/interpreter.py)
- [src/scimilarity/nn_models.py](src/scimilarity/nn_models.py)
- [src/scimilarity/ontologies.py](src/scimilarity/ontologies.py)
- [src/scimilarity/zarr_dataset.py](src/scimilarity/zarr_dataset.py)

</details>



This document covers advanced features and sophisticated use cases of SCimilarity that extend beyond the basic API workflows. Topics include ontology integration for hierarchical cell type analysis, model interpretation for understanding gene contributions, neural network architecture details, and advanced data storage capabilities.

For basic usage patterns, see [Core API Reference](#3). For training custom models, see [Model Training](#4). For standard data management workflows, see [Data Management & Storage](#5).

## Ontology Integration

SCimilarity provides sophisticated ontology integration capabilities that enable hierarchical analysis of cell types, tissues, and diseases. The system supports multiple standard ontologies and provides utilities for computing ontology-aware similarity metrics.

### Supported Ontologies

SCimilarity integrates with several standard biological ontologies through the `ontologies` module:

| Ontology | Function | Purpose |
|----------|----------|---------|
| Cell Ontology (CL) | `import_cell_ontology()` | Hierarchical cell type relationships |
| UBERON | `import_uberon_ontology()` | Anatomical structure and tissue ontology |
| Disease Ontology (DOID) | `import_doid_ontology()` | Disease classification hierarchy |
| MONDO | `import_mondo_ontology()` | Unified disease ontology |

### Ontology-Aware Analysis Workflow

```mermaid
graph TB
    subgraph "Ontology Import"
        OBO["OBO Files"]
        ICO["import_cell_ontology()"]
        IUO["import_uberon_ontology()"]
        IDO["import_doid_ontology()"]
        IMO["import_mondo_ontology()"]
    end
    
    subgraph "Graph Processing"
        NX["NetworkX DiGraph"]
        GIM["get_id_mapper()"]
        GC["get_children()"]
        GP["get_parents()"]
        GAA["get_all_ancestors()"]
        GAD["get_all_descendants()"]
    end
    
    subgraph "Similarity Analysis"
        OS["ontology_similarity()"]
        APS["all_pair_similarities()"]
        OSW["ontology_silhouette_width()"]
        FMVP["find_most_viable_parent()"]
    end
    
    subgraph "SCimilarity Integration"
        CA["CellAnnotation"]
        EMB["Cell Embeddings"]
        PRED["Predictions"]
    end
    
    OBO --> ICO
    OBO --> IUO
    OBO --> IDO
    OBO --> IMO
    
    ICO --> NX
    IUO --> NX
    IDO --> NX
    IMO --> NX
    
    NX --> GIM
    NX --> GC
    NX --> GP
    NX --> GAA
    NX --> GAD
    
    GAA --> OS
    GAD --> OS
    OS --> APS
    EMB --> OSW
    NX --> OSW
    NX --> FMVP
    
    CA --> PRED
    OSW --> EMB
    APS --> CA
```

**Sources:** [src/scimilarity/ontologies.py:30-123]()

### Hierarchical Similarity Computation

The ontology system enables sophisticated similarity calculations that consider hierarchical relationships between cell types:

```mermaid
graph LR
    subgraph "Cell Type Pairs"
        CT1["Cell Type 1"]
        CT2["Cell Type 2"]
    end
    
    subgraph "Ancestry Analysis"
        A1["get_all_ancestors(CT1)"]
        A2["get_all_ancestors(CT2)"]
        CA["Common Ancestors"]
    end
    
    subgraph "Similarity Metrics"
        OSF["ontology_similarity()"]
        COUNT["len(common_ancestors)"]
        OSW_FUNC["ontology_silhouette_width()"]
    end
    
    subgraph "Evaluation"
        INTRA["Intra-cluster Distance"]
        INTER["Inter-cluster Distance"]
        SW["Silhouette Width"]
    end
    
    CT1 --> A1
    CT2 --> A2
    A1 --> CA
    A2 --> CA
    CA --> OSF
    OSF --> COUNT
    
    CT1 --> OSW_FUNC
    CT2 --> OSW_FUNC
    OSW_FUNC --> INTRA
    OSW_FUNC --> INTER
    INTRA --> SW
    INTER --> SW
```

**Sources:** [src/scimilarity/ontologies.py:372-517]()

### Ontology-Aware Silhouette Analysis

The `ontology_silhouette_width()` function provides evaluation metrics that respect cell type hierarchies, where ancestors are not considered inter-cluster and descendants are considered intra-cluster:

**Key Parameters:**
- `embeddings`: Cell embeddings matrix
- `labels`: Cell type names  
- `onto`: NetworkX ontology graph
- `name2id`: Mapping from cell type names to ontology IDs
- `metric`: Distance metric (default: "cosine")

**Returns:**
- Average silhouette width score
- DataFrame with per-cell-type silhouette metrics

**Sources:** [src/scimilarity/ontologies.py:439-517]()

## Model Interpretation

SCimilarity provides model interpretation capabilities through the `Interpreter` class, which uses integrated gradients to identify genes that contribute most to similarity relationships between cells.

### Interpretation Architecture

```mermaid
graph TB
    subgraph "Input Data"
        ANC["Anchor Cells"]
        NEG["Negative Cells"]
        GO["Gene Order"]
    end
    
    subgraph "Interpretation Engine"
        INT["Interpreter"]
        SD["SimpleDist"]
        IG["IntegratedGradients"]
        ENC["Encoder Model"]
    end
    
    subgraph "Attribution Analysis"
        GA["get_attributions()"]
        ATTR["Attribution Matrix"]
        GRG["get_ranked_genes()"]
        PRG["plot_ranked_genes()"]
    end
    
    subgraph "Output"
        RANKED["Ranked Gene DataFrame"]
        PLOT["Attribution Plot"]
        GENES["Top Contributing Genes"]
    end
    
    ANC --> INT
    NEG --> INT
    GO --> INT
    
    INT --> SD
    SD --> ENC
    SD --> IG
    
    ANC --> GA
    NEG --> GA
    GA --> ATTR
    ATTR --> GRG
    ATTR --> PRG
    
    GRG --> RANKED
    PRG --> PLOT
    RANKED --> GENES
```

**Sources:** [src/scimilarity/interpreter.py:43-214]()

### Gene Attribution Workflow

The interpretation system computes gene attributions by measuring how changes in gene expression affect the distance between anchor and negative cell pairs:

#### SimpleDist Distance Function

The `SimpleDist` class implements the distance calculation used for attribution:

```mermaid
graph LR
    subgraph "Distance Calculation"
        A["Anchors"]
        N["Negatives"] 
        ENC1["encoder(anchors)"]
        ENC2["encoder(negatives)"]
        DIST["sum((f_neg - f_anc)²)"]
    end
    
    A --> ENC1
    N --> ENC2
    ENC1 --> DIST
    ENC2 --> DIST
```

**Sources:** [src/scimilarity/interpreter.py:5-41]()

#### Attribution Computation Process

1. **Baseline Integration**: Uses negative cells as baselines for integrated gradients
2. **Directional Masking**: Multiplies attributions by `anchor > negative` to focus on upregulated genes
3. **Absolute Values**: Takes absolute value since attribution signs can be unreliable
4. **GPU Support**: Automatically handles GPU tensors if model is on GPU

**Sources:** [src/scimilarity/interpreter.py:69-131]()

### Attribution Analysis Results

The `get_ranked_genes()` method produces a comprehensive analysis DataFrame:

| Column | Description |
|--------|-------------|
| `gene` | Gene symbol |
| `gene_idx` | Gene index in original order |
| `attribution` | Mean attribution score |
| `attribution_std` | Standard deviation of attributions |
| `cells` | Number of cells in analysis |

**Sources:** [src/scimilarity/interpreter.py:133-163]()

## Neural Network Architecture

SCimilarity employs an encoder-decoder architecture with specific design choices optimized for single-cell data representation learning.

### Encoder-Decoder Design

```mermaid
graph TB
    subgraph "Input Layer"
        GENES["Gene Expression (n_genes)"]
        IDROP["Input Dropout (0.4)"]
    end
    
    subgraph "Encoder Architecture"
        H1["Hidden Layer 1 (1024)"]
        BN1["BatchNorm1d"]
        PRELU1["PReLU"]
        
        DROP["Dropout (0.5)"]
        H2["Hidden Layer 2 (1024)"]
        BN2["BatchNorm1d"] 
        PRELU2["PReLU"]
        
        LATENT["Latent Space (128)"]
        NORM["L2 Normalize"]
    end
    
    subgraph "Decoder Architecture"
        DH1["Hidden Layer 1 (1024)"]
        DBN1["BatchNorm1d"]
        DPRELU1["PReLU"]
        
        DDROP["Dropout (0.5)"]
        DH2["Hidden Layer 2 (1024)"]
        DBN2["BatchNorm1d"]
        DPRELU2["PReLU"]
        
        RECON["Reconstruction (n_genes)"]
    end
    
    GENES --> IDROP
    IDROP --> H1
    H1 --> BN1
    BN1 --> PRELU1
    PRELU1 --> DROP
    DROP --> H2
    H2 --> BN2
    BN2 --> PRELU2
    PRELU2 --> LATENT
    LATENT --> NORM
    
    NORM --> DH1
    DH1 --> DBN1
    DBN1 --> DPRELU1
    DPRELU1 --> DDROP
    DDROP --> DH2
    DH2 --> DBN2
    DBN2 --> DPRELU2
    DPRELU2 --> RECON
```

**Sources:** [src/scimilarity/nn_models.py:12-207]()

### Architecture Configuration

#### Encoder Specifications

| Component | Default Value | Purpose |
|-----------|---------------|---------|
| `latent_dim` | 128 | Embedding dimensionality |
| `hidden_dim` | [1024, 1024] | Hidden layer sizes |
| `dropout` | 0.5 | Hidden layer dropout rate |
| `input_dropout` | 0.4 | Input layer dropout rate |
| Activation | PReLU | Parameterized ReLU activation |
| Normalization | BatchNorm1d | Batch normalization after linear layers |
| Output | L2 Normalize | Unit sphere embeddings |

**Sources:** [src/scimilarity/nn_models.py:31-81]()

#### Decoder Specifications

The decoder mirrors the encoder architecture in reverse:
- **Input**: Latent embeddings (128-dim)
- **Hidden Layers**: [1024, 1024] with BatchNorm1d and PReLU
- **Output**: Reconstructed gene expression (n_genes)
- **Dropout**: Applied to hidden layers (0.5 rate)

**Sources:** [src/scimilarity/nn_models.py:130-176]()

### Model State Management

Both encoder and decoder classes provide state management capabilities:

```mermaid
graph LR
    subgraph "Model Persistence"
        SS["save_state()"]
        LS["load_state()"]
        FILE["Model Checkpoint"]
        GPU["GPU Support"]
    end
    
    subgraph "State Dictionary"
        SD["state_dict"]
        TORCH["torch.save()"]
        LOAD["torch.load()"]
    end
    
    SS --> SD
    SD --> TORCH
    TORCH --> FILE
    
    FILE --> LOAD
    LOAD --> LS
    GPU --> LS
```

**Sources:** [src/scimilarity/nn_models.py:82-111](), [src/scimilarity/nn_models.py:178-206]()

## Advanced Data Storage

SCimilarity provides sophisticated data storage capabilities through the `ZarrDataset` class, enabling efficient access to large-scale single-cell datasets stored in Zarr format.

### Zarr Integration Architecture

```mermaid
graph TB
    subgraph "Storage Layer"
        DS["DirectoryStore"]
        ZG["Zarr Group"]
        CHUNKS["Chunk Store"]
    end
    
    subgraph "Data Components"
        X["Expression Matrix (X)"]
        OBS["Cell Metadata (obs)"]
        VAR["Gene Metadata (var)"]
        LAYERS["Additional Layers"]
        UNS["Unstructured Data (uns)"]
    end
    
    subgraph "Matrix Formats"
        CSR["CSR Matrix"]
        CSC["CSC Matrix"] 
        COO["COO Matrix"]
        ATTRS["Matrix Attributes"]
    end
    
    subgraph "Access Patterns"
        CELL["get_cell()"]
        GENE["get_gene()"]
        ROW["get_row()"]
        COL["get_col()"]
        BATCH["Batch Access"]
    end
    
    DS --> ZG
    ZG --> CHUNKS
    
    ZG --> X
    ZG --> OBS
    ZG --> VAR
    ZG --> LAYERS
    ZG --> UNS
    
    X --> CSR
    X --> CSC
    X --> COO
    CSR --> ATTRS
    CSC --> ATTRS
    COO --> ATTRS
    
    X --> CELL
    X --> GENE
    CSR --> ROW
    CSC --> COL
    ROW --> BATCH
    COL --> BATCH
```

**Sources:** [src/scimilarity/zarr_dataset.py:12-35]()

### Sparse Matrix Access Patterns

The `ZarrDataset` class provides optimized access patterns for different sparse matrix formats:

#### Format-Specific Slicing

| Format | Optimal Access | Method | Use Case |
|--------|----------------|--------|----------|
| CSR | Row-wise | `row_slice_csr()` | Cell-based queries |
| CSC | Column-wise | `col_slice_csc()` | Gene-based queries |
| COO | Flexible | `slice_coo()` | General purpose |

**Sources:** [src/scimilarity/zarr_dataset.py:555-643]()

#### Memory Management

```mermaid
graph LR
    subgraph "Access Modes"
        REF["Reference Mode"]
        MEM["In-Memory Mode"]
        LAZY["Lazy Loading"]
    end
    
    subgraph "Matrix Operations"
        SLICE["Matrix Slicing"]
        APPEND["append_matrix()"]
        SET["set_matrix()"]
    end
    
    subgraph "Performance"
        CHUNK["Chunked Access"]
        STREAM["Streaming"]
        CACHE["Caching"]
    end
    
    REF --> LAZY
    MEM --> CACHE
    LAZY --> SLICE
    CACHE --> SLICE
    
    SLICE --> APPEND
    SLICE --> SET
    CHUNK --> STREAM
    STREAM --> SLICE
```

**Sources:** [src/scimilarity/zarr_dataset.py:645-829]()

### Dynamic Data Manipulation

The system supports dynamic modification of stored datasets:

#### Matrix Appending Operations

- **Row Appending**: For CSR matrices along axis 0
- **Column Appending**: For CSC matrices along axis 1  
- **Flexible Appending**: For COO matrices with explicit axis specification
- **Metadata Updates**: Automatic shape and index updates

**Sources:** [src/scimilarity/zarr_dataset.py:725-829]()

#### Annotation Management

The annotation system handles both categorical and continuous metadata:

```mermaid
graph TB
    subgraph "Annotation Types"
        CAT["Categorical Data"]
        CONT["Continuous Data"]
        STR["String Arrays"]
        CODES["Category Codes"]
    end
    
    subgraph "Storage Encoding"
        JSON["JSON Codec"]
        NUMCODECS["numcodecs"]
        DTYPE["Native dtypes"]
    end
    
    subgraph "Operations"
        SET["set_annotation()"]
        APPEND["append_annotation()"]
        GET["get_annotation_column()"]
    end
    
    CAT --> CODES
    CAT --> JSON
    CONT --> DTYPE
    STR --> JSON
    
    CODES --> SET
    JSON --> SET
    DTYPE --> SET
    
    SET --> APPEND
    SET --> GET
```

**Sources:** [src/scimilarity/zarr_dataset.py:910-999]()