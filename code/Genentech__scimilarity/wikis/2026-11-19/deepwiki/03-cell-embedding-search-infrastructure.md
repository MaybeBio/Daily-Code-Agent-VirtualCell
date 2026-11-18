# Cell Embedding & Search Infrastructure

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/api.rst](docs/api.rst)
- [src/scimilarity/cell_embedding.py](src/scimilarity/cell_embedding.py)
- [src/scimilarity/cell_search_knn.py](src/scimilarity/cell_search_knn.py)
- [src/scimilarity/triplet_selector.py](src/scimilarity/triplet_selector.py)

</details>



This document covers the foundational classes `CellEmbedding` and `CellSearchKNN` that provide the core infrastructure for generating cell embeddings and performing similarity searches in SCimilarity. These classes form the base layer upon which higher-level functionality is built.

For cell similarity queries and metadata compilation, see [Cell Query System](#3.2). For automated cell type prediction, see [Cell Annotation](#3.3). For data preprocessing and alignment utilities, see [Data Processing Utilities](#3.4).

## Architecture Overview

The embedding and search infrastructure follows a hierarchical design where `CellSearchKNN` extends `CellEmbedding` to add kNN search capabilities on top of the base embedding functionality.

### Class Hierarchy

```mermaid
classDiagram
    class CellEmbedding {
        +str model_path
        +bool use_gpu
        +Encoder model
        +list gene_order
        +int latent_dim
        +dict int2label
        +dict label2int
        +__init__(model_path, use_gpu)
        +get_embeddings(X, num_cells, buffer_size)
    }
    
    class CellSearchKNN {
        +str knn_type
        +object knn
        +list safelist
        +list blocklist
        +load_knn_index(knn_file, memory_budget)
        +get_nearest_neighbors(embeddings, k, ef)
    }
    
    class Encoder {
        +int n_genes
        +int latent_dim
        +list hidden_dim
        +forward(x)
        +load_state(path)
    }
    
    CellSearchKNN --|> CellEmbedding
    CellEmbedding --> Encoder
```

Sources: [src/scimilarity/cell_embedding.py:4-71](), [src/scimilarity/cell_search_knn.py:6-104]()

## CellEmbedding Class

The `CellEmbedding` class handles model loading, gene alignment, and embedding generation from gene expression data.

### Model Loading Infrastructure

```mermaid
graph TB
    subgraph "Model Files"
        EncoderCkpt["encoder.ckpt"]
        GeneOrder["gene_order.tsv"] 
        LayerSizes["layer_sizes.json"]
        LabelInts["label_ints.csv"]
    end
    
    subgraph "CellEmbedding.__init__"
        ModelPath["model_path"]
        LoadEncoder["Load Encoder Model"]
        ParseGenes["Parse gene_order"]
        ParseLayers["Parse layer_sizes"]
        ParseLabels["Parse label mappings"]
    end
    
    subgraph "Runtime Objects"
        EncoderModel["self.model (Encoder)"]
        GeneOrderList["self.gene_order (list)"]
        LatentDim["self.latent_dim (int)"]
        Int2Label["self.int2label (dict)"]
        Label2Int["self.label2int (dict)"]
    end
    
    ModelPath --> EncoderCkpt
    ModelPath --> GeneOrder
    ModelPath --> LayerSizes
    ModelPath --> LabelInts
    
    EncoderCkpt --> LoadEncoder
    GeneOrder --> ParseGenes
    LayerSizes --> ParseLayers
    LabelInts --> ParseLabels
    
    LoadEncoder --> EncoderModel
    ParseGenes --> GeneOrderList
    ParseLayers --> LatentDim
    ParseLabels --> Int2Label
    ParseLabels --> Label2Int
```

Sources: [src/scimilarity/cell_embedding.py:24-71]()

### Embedding Generation Workflow

The `get_embeddings` method processes gene expression data through several stages:

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| Data Type Detection | `scipy.sparse`, `numpy.ndarray`, `torch.Tensor` | Normalized format | Handle multiple input formats |
| Zarr Optimization | Zarr-backed sparse matrices | Loaded arrays | Optimize small batch performance |
| Batch Processing | Expression matrix | Embedding chunks | Memory-efficient processing |
| GPU Transfer | CPU tensors | GPU tensors | Accelerate inference |
| Model Inference | Gene expression | Latent embeddings | Generate cell representations |

```mermaid
graph LR
    subgraph "Input Data Types"
        NumpyArray["numpy.ndarray"]
        TorchTensor["torch.Tensor"] 
        CSRMatrix["scipy.sparse.csr_matrix"]
        CSCMatrix["scipy.sparse.csc_matrix"]
    end
    
    subgraph "get_embeddings Processing"
        TypeCheck["Data Type Detection"]
        ZarrOpt["Zarr Optimization"]
        BatchLoop["Batch Processing Loop"]
        GPUTransfer["GPU Transfer (if use_gpu=True)"]
        ModelInference["self.model(profiles)"]
        Concatenate["torch.vstack()"]
    end
    
    subgraph "Output"
        Embeddings["numpy.ndarray [num_cells x latent_dim]"]
    end
    
    NumpyArray --> TypeCheck
    TorchTensor --> TypeCheck
    CSRMatrix --> TypeCheck
    CSCMatrix --> TypeCheck
    
    TypeCheck --> ZarrOpt
    ZarrOpt --> BatchLoop
    BatchLoop --> GPUTransfer
    GPUTransfer --> ModelInference
    ModelInference --> Concatenate
    Concatenate --> Embeddings
```

Sources: [src/scimilarity/cell_embedding.py:72-156]()

## CellSearchKNN Class

The `CellSearchKNN` class extends `CellEmbedding` to provide approximate nearest neighbor search capabilities using either HNSW or TileDB vector search backends.

### kNN Backend Support

```mermaid
graph TB
    subgraph "CellSearchKNN"
        KnnType["knn_type parameter"]
        LoadMethod["load_knn_index()"]
        SearchMethod["get_nearest_neighbors()"]
    end
    
    subgraph "HNSW Backend"
        HnswlibIndex["hnswlib.Index"]
        HnswFile["*.hnsw file"]
        HnswQuery["knn_query(embeddings, k)"]
    end
    
    subgraph "TileDB Backend" 
        TileDBIndex["vs.IVFFlatIndex"]
        TileDBDir["index directory"]
        TileDBQuery["query(embeddings, k, nprobe)"]
    end
    
    KnnType --> LoadMethod
    LoadMethod --> SearchMethod
    
    LoadMethod --> HnswlibIndex
    LoadMethod --> TileDBIndex
    
    HnswlibIndex --> HnswFile
    TileDBIndex --> TileDBDir
    
    SearchMethod --> HnswQuery
    SearchMethod --> TileDBQuery
```

Sources: [src/scimilarity/cell_search_knn.py:40-103]()

### Search Parameter Configuration

The kNN search behavior is controlled by several parameters:

| Parameter | Backend | Purpose | Default |
|-----------|---------|---------|---------|
| `k` | Both | Number of nearest neighbors | 50 |
| `ef` | HNSW | Dynamic list size for search | 100 |
| `memory_budget` | TileDB | Memory allocation for index | 50000000 |
| `nprobe` | TileDB | Number of clusters to search | `√partitions` |

## GPU Support and Performance

Both classes support GPU acceleration through the `use_gpu` parameter:

```mermaid
graph LR
    subgraph "GPU Configuration"
        UseGPU["use_gpu=True"]
        ModelCUDA["model.cuda()"]
        DataTransfer["profiles.cuda()"]
        InferenceCUDA["GPU Inference"]
        ResultCPU["detach().cpu().numpy()"]
    end
    
    subgraph "CPU Configuration"
        UseCPU["use_gpu=False"]
        ModelCPU["model (CPU)"]
        InferenceCPU["CPU Inference"]
        ResultDirect["detach().numpy()"]
    end
    
    UseGPU --> ModelCUDA
    ModelCUDA --> DataTransfer
    DataTransfer --> InferenceCUDA
    InferenceCUDA --> ResultCPU
    
    UseCPU --> ModelCPU
    ModelCPU --> InferenceCPU
    InferenceCPU --> ResultDirect
```

Sources: [src/scimilarity/cell_embedding.py:62-63](), [src/scimilarity/cell_embedding.py:139-151]()

## Error Handling and Validation

The infrastructure includes several validation mechanisms:

- **Data Type Validation**: Supports `numpy.ndarray`, `torch.Tensor`, and `scipy.sparse` matrices [src/scimilarity/cell_embedding.py:129-137]()
- **NaN Detection**: Validates embedding outputs for numerical stability [src/scimilarity/cell_embedding.py:153-154]()
- **kNN Index Validation**: Ensures index files exist and match expected formats [src/scimilarity/cell_search_knn.py:55-62]()
- **Empty Batch Handling**: Handles edge cases with no valid cells [src/scimilarity/cell_embedding.py:143-144]()

## Integration Points

These base classes integrate with the broader SCimilarity ecosystem:

- **Model Training**: Uses `Encoder` from neural network training pipeline [src/scimilarity/cell_embedding.py:27]()
- **Data Processing**: Works with alignment utilities for gene space matching
- **Higher-Level APIs**: Extended by `CellQuery` and `CellAnnotation` classes
- **Storage Backends**: Interfaces with TileDB and Zarr data stores through preprocessing

Sources: [src/scimilarity/cell_embedding.py:1-157](), [src/scimilarity/cell_search_knn.py:1-104]()

---