# Post-Training Workflows

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [scripts/build_embeddings.py](scripts/build_embeddings.py)

</details>



## Purpose and Scope

This page documents the workflows and processes that occur after training a SCimilarity model. These workflows transform trained models into production-ready systems by generating embeddings, building search indices, and constructing metadata structures required for similarity search and cell annotation.

Post-training workflows bridge the gap between trained model artifacts and the deployed SCimilarity system that end users interact with. For detailed information about the training process itself, see [Model Training](#4). For specific implementation details about search index construction, see [Building Search Indices](#6.1). For metadata construction processes, see [Metadata Construction](#6.2).

## Post-Training Workflow Overview

The post-training phase consists of several sequential steps that prepare trained models for deployment and use:

### High-Level Post-Training Pipeline

```mermaid
flowchart TD
    TM["Trained Model<br/>(checkpoint files)"] --> BW["Build Workflow"]
    
    subgraph BW ["Post-Training Build Workflow"]
        direction TB
        BE["build_embeddings.py<br/>Generate cell embeddings"]
        BI["Build Search Indices<br/>HNSW + TileDB Vector"]
        BM["Build Metadata<br/>Cell annotations & labels"]
        
        BE --> BI
        BE --> BM
    end
    
    subgraph DS ["Data Sources"]
        CA["CellArr TileDB<br/>counts, gene_annotation"]
        REF["Reference Datasets<br/>Cell type labels"]
    end
    
    subgraph OUT ["Output Artifacts"]
        EMB["cell_embedding<br/>TileDB array"]
        AIDX["Annotation kNN<br/>HNSW index"]
        SIDX["Search kNN<br/>HNSW index"] 
        META["Cell Metadata<br/>Parquet/TileDB"]
    end
    
    CA --> BE
    REF --> BM
    
    BE --> EMB
    BI --> AIDX
    BI --> SIDX
    BM --> META
    
    subgraph DEP ["Deployment Ready"]
        PKG["SCimilarity Package<br/>CellQuery, CellAnnotation"]
    end
    
    EMB --> PKG
    AIDX --> PKG
    SIDX --> PKG
    META --> PKG
```

Sources: [scripts/build_embeddings.py:1-121]()

## Embedding Generation Workflow

The core post-training workflow begins with generating embeddings for all cells in the dataset using the trained model. This process is implemented in the `build_embeddings.py` script.

### Embedding Generation Architecture

```mermaid
flowchart LR
    subgraph INPUT ["Input Components"]
        MODEL["Trained Model Path<br/>CellEmbedding.load()"]
        CELLARR["CellArr TileDB<br/>-t argument"]
    end
    
    subgraph PROC ["Processing Pipeline"]
        ALIGN["Gene Alignment<br/>gene_indices = [genes.index(x) for x in ce.gene_order]"]
        BATCH["Batch Processing<br/>batch_size = 100000"]
        NORM["Log Normalization<br/>get_expression()"]
        EMBED["Generate Embeddings<br/>ce.get_embeddings(X)"]
    end
    
    subgraph STORE ["Storage Components"]
        SCHEMA["TileDB Schema<br/>xdim, ydim, domain"]
        WRITE["Write to TileDB<br/>embedding_tdb[cell_idx] = embedding"]
        OPT["Optimize Array<br/>optimize_tiledb_array()"]
    end
    
    MODEL --> ALIGN
    CELLARR --> ALIGN
    
    ALIGN --> BATCH
    BATCH --> NORM
    NORM --> EMBED
    
    EMBED --> SCHEMA
    SCHEMA --> WRITE
    WRITE --> OPT
    
    subgraph OUTPUT ["Output Structure"]
        EMBDIR["cellsearch/cell_embedding<br/>TileDB dense array"]
    end
    
    OPT --> EMBDIR
```

Sources: [scripts/build_embeddings.py:39-121]()

### Key Processing Steps

| Step | Code Entity | Description | Configuration |
|------|-------------|-------------|---------------|
| **Gene Alignment** | `gene_indices = [genes.index(x) for x in ce.gene_order]` | Maps model gene order to CellArr gene indices | Uses `gene_annotation` TileDB array |
| **Expression Extraction** | `get_expression()` | Retrieves and normalizes cell expression data | `target_sum=1e4`, log1p transformation |
| **Batch Processing** | `for i in tqdm(range(0, matrix_shape[0], batch_size))` | Processes cells in configurable batches | Default batch size: 100,000 cells |
| **Embedding Generation** | `ce.get_embeddings(X)` | Generates embeddings using trained model | Uses `CellEmbedding` class |
| **TileDB Storage** | `embedding_tdb[cell_idx] = embedding` | Stores embeddings in dense TileDB array | LZ4 compression, row-major order |

Sources: [scripts/build_embeddings.py:17-37](), [scripts/build_embeddings.py:103-109]()

### TileDB Array Configuration

The embedding storage uses a specific TileDB schema optimized for efficient access patterns:

```mermaid
graph TB
    subgraph SCHEMA ["TileDB Array Schema"]
        DIMS["Dimensions<br/>x: cells (0 to matrix_shape[0]-1)<br/>y: features (0 to ce.latent_dim-1)"]
        TILES["Tile Configuration<br/>x tile: 10000<br/>y tile: ce.latent_dim"]
        ATTR["Attributes<br/>data: float32<br/>LZ4Filter compression"]
        ORDER["Array Order<br/>cell_order: row-major<br/>tile_order: row-major"]
    end
    
    subgraph CONFIG ["TileDB Config"]
        MEM["Memory Budget<br/>sm.mem.total_budget = 50G"]
        DENSE["Array Type<br/>sparse = False"]
    end
    
    SCHEMA --> CONFIG
    
    subgraph ACCESS ["Access Pattern"]
        BATCH_READ["Batch Read<br/>embedding_tdb[cell_idx]"]
        OPTIM["Post-Write Optimization<br/>optimize_tiledb_array()"]
    end
    
    CONFIG --> ACCESS
```

Sources: [scripts/build_embeddings.py:14-15](), [scripts/build_embeddings.py:74-94](), [scripts/build_embeddings.py:118]()

## Index and Metadata Construction

After embedding generation, the workflow continues with building search indices and constructing metadata structures. These components enable efficient similarity search and cell annotation functionality.

### Post-Embedding Workflow Integration

```mermaid
flowchart TD
    subgraph EMB_OUT ["Embedding Output"]
        CELL_EMB["cell_embedding<br/>TileDB dense array"]
        EMB_SHAPE["embeddings.shape<br/>numpy validation"]
    end
    
    subgraph INDEX_BUILD ["Index Construction"]
        HNSW_SEARCH["Search kNN Index<br/>HNSW for CellQuery"]
        HNSW_ANNOT["Annotation kNN Index<br/>HNSW for CellAnnotation"]
        TILEDB_VEC["TileDB Vector Search<br/>Alternative to HNSW"]
    end
    
    subgraph META_BUILD ["Metadata Construction"]
        CELL_META["Cell Metadata<br/>Labels, ontologies, datasets"]
        GENE_META["Gene Metadata<br/>Symbols, annotations"]
        FILTER_META["Filter Metadata<br/>Quality metrics, selection criteria"]
    end
    
    subgraph DEPLOY_READY ["Deployment Artifacts"]
        CELLQUERY["CellQuery System<br/>Similarity search"]
        CELLANNOT["CellAnnotation System<br/>Cell type prediction"]
        API_READY["API Ready<br/>Production deployment"]
    end
    
    CELL_EMB --> INDEX_BUILD
    EMB_SHAPE --> INDEX_BUILD
    
    CELL_EMB --> META_BUILD
    
    INDEX_BUILD --> DEPLOY_READY
    META_BUILD --> DEPLOY_READY
```

Sources: [scripts/build_embeddings.py:113-116]()

## Integration with SCimilarity Components

The post-training workflows produce artifacts that integrate directly with the main SCimilarity API components:

| Component | Input Artifacts | Purpose | Code Integration |
|-----------|----------------|---------|------------------|
| **CellEmbedding** | Trained model checkpoints | Generate embeddings for new data | `CellEmbedding(model_path)` |
| **CellQuery** | `cell_embedding` TileDB, search kNN indices | Similarity search functionality | Loads embeddings and indices automatically |
| **CellAnnotation** | `cell_embedding` TileDB, annotation kNN indices, cell metadata | Cell type prediction | Uses reference embeddings and labels |
| **CellSearchKNN** | kNN indices (HNSW or TileDB Vector) | Base search infrastructure | Shared by CellQuery and CellAnnotation |

The post-training workflow ensures that all these components have the necessary data structures and indices to operate efficiently in production environments.

Sources: [scripts/build_embeddings.py:51-52]()

---