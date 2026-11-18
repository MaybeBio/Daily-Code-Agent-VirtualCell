# Cell Query System

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/scimilarity/cell_query.py](src/scimilarity/cell_query.py)

</details>



The Cell Query System provides advanced similarity search capabilities for single-cell data through the `CellQuery` class. This system extends the base cell search infrastructure to enable complex queries with metadata compilation, result filtering, and multiple search strategies including both approximate and exhaustive search methods.

For basic cell embedding and search infrastructure, see [Cell Embedding & Search Infrastructure](#3.1). For cell type annotation functionality, see [Cell Annotation](#3.3).

## Architecture Overview

The `CellQuery` class serves as the primary interface for cell similarity searches, extending `CellSearchKNN` with enhanced query capabilities and metadata management.

### Class Hierarchy and Dependencies

```mermaid
graph TB
    CSK["CellSearchKNN<br/>(Base Class)"]
    CQ["CellQuery<br/>(Advanced Search)"]
    
    subgraph "Data Storage"
        TDB["TileDB Arrays<br/>cell_metadata<br/>cell_embedding"]
        HNSW["HNSW Index<br/>full_kNN.bin"]
        CSV["CSV Metadata<br/>full_kNN_meta.csv"]
        NPY["NumPy Arrays<br/>full_kNN_embedding.npy"]
    end
    
    subgraph "Search Backends"
        HNSWLIB["hnswlib Backend"]
        TDBVS["tiledb_vector_search Backend"]
    end
    
    CSK --> CQ
    CQ --> TDB
    CQ --> HNSW
    CQ --> CSV
    CQ --> NPY
    CQ --> HNSWLIB
    CQ --> TDBVS
```

**Sources:** [src/scimilarity/cell_query.py:1-84]()

### Data Integration Architecture

```mermaid
flowchart LR
    subgraph "Input Data"
        ADATA["AnnData Objects<br/>adata.layers['counts']"]
        QUERIES["Query Specifications<br/>centroid_key<br/>cluster_key"]
    end
    
    subgraph "CellQuery Components"
        META["get_precomputed_embeddings()"]
        ANNOT["annotate_cell_index()"]
        COMP["compile_sample_metadata()"]
    end
    
    subgraph "Search Methods"
        NEAREST["search_nearest()"]
        CENT["search_centroid_nearest()"]
        CLUST["search_cluster_centroids_nearest()"]
        EXHAUST["search_exhaustive()"]
    end
    
    subgraph "Storage Backend"
        CELLEMB["cell_embedding TileDB"]
        CELLMETA["cell_metadata TileDB"]
        KNNINDEX["kNN Index Files"]
    end
    
    subgraph "Outputs"
        INDICES["nn_idxs: List[numpy.ndarray]"]
        DISTANCES["nn_dists: List[numpy.ndarray]"]
        METADATA["metadata: pandas.DataFrame"]
    end
    
    ADATA --> COMP
    QUERIES --> CENT
    QUERIES --> CLUST
    
    META --> CELLEMB
    ANNOT --> CELLMETA
    COMP --> CELLMETA
    
    NEAREST --> KNNINDEX
    CENT --> KNNINDEX
    CLUST --> KNNINDEX
    EXHAUST --> CELLEMB
    
    NEAREST --> INDICES
    CENT --> DISTANCES
    CLUST --> METADATA
    EXHAUST --> OUTPUTS
```

**Sources:** [src/scimilarity/cell_query.py:93-116](), [src/scimilarity/cell_query.py:150-204]()

## Initialization and Configuration

The `CellQuery` class requires a model path and supports extensive configuration for different use cases:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to the model directory |
| `use_gpu` | bool | False | Enable GPU acceleration |
| `metadata_tiledb_uri` | str | "cell_metadata" | TileDB cell metadata storage path |
| `embedding_tiledb_uri` | str | "cell_embedding" | TileDB cell embedding storage path |
| `knn_type` | str | "hnswlib" | kNN backend: "hnswlib" or "tiledb_vector_search" |
| `load_knn` | bool | True | Whether to load the kNN index |

The system automatically constructs file paths within a `cellsearch` subdirectory and initializes connections to TileDB arrays for metadata and embeddings.

**Sources:** [src/scimilarity/cell_query.py:31-92]()

## Core Data Access Methods

### Embedding Retrieval

The `get_precomputed_embeddings()` method provides efficient access to precomputed cell embeddings from TileDB storage:

```python
# Fast retrieval of specific cell embeddings
embeddings = cq.get_precomputed_embeddings([0, 1, 100])
embeddings = cq.get_precomputed_embeddings(slice(0, 1000))
```

**Sources:** [src/scimilarity/cell_query.py:93-116]()

### Cell Index Annotation

The `annotate_cell_index()` method maps global cell indices to sample-specific indices, enabling proper data alignment:

```python
# Convert global indices to per-sample cell indices
metadata_with_cell_index = cq.annotate_cell_index(metadata)
```

This method requires metadata with `study`, `sample`, and `data_type` columns and adds a `cell_index` column representing the position within each sample dataset.

**Sources:** [src/scimilarity/cell_query.py:117-148]()

### Sample Metadata Compilation

The `compile_sample_metadata()` method aggregates nearest neighbor results at the sample level:

```python
# Compile sample-level statistics for search results
sample_metadata = cq.compile_sample_metadata(
    nn_idxs, 
    levels=["study", "sample", "tissue", "disease"]
)
```

This method calculates cell counts and fractions per sample, providing insights into which samples contain cells similar to the query.

**Sources:** [src/scimilarity/cell_query.py:150-204]()

## Search Methods

The `CellQuery` class provides multiple search strategies optimized for different use cases:

### Search Method Comparison

| Method | Search Type | Performance | Use Case |
|--------|-------------|-------------|----------|
| `search_nearest` | Approximate (kNN) | Fast | General similarity search |
| `search_exhaustive` | Exact | Slow | Precise distance filtering |
| `search_centroid_nearest` | Approximate (kNN) | Fast | Single cluster queries |
| `search_cluster_centroids_nearest` | Approximate (kNN) | Fast | Multiple cluster queries |
| `search_centroid_exhaustive` | Exact | Slow | Precise centroid search |
| `search_cluster_centroids_exhaustive` | Exact | Slow | Precise multi-cluster search |

### kNN-Based Search Methods

#### Basic Nearest Neighbor Search

```mermaid
flowchart TD
    EMBED["Input Embeddings<br/>numpy.ndarray"]
    PARAMS["Parameters<br/>k, ef, max_dist"]
    
    subgraph "search_nearest()"
        KNN["get_nearest_neighbors()"]
        FILTER["Distance Filtering<br/>(if max_dist)"]
        METACOMP["Metadata Compilation"]
    end
    
    OUTPUTS["nn_idxs: List[numpy.ndarray]<br/>nn_dists: List[numpy.ndarray]<br/>metadata: pandas.DataFrame"]
    
    EMBED --> KNN
    PARAMS --> KNN
    KNN --> FILTER
    FILTER --> METACOMP
    METACOMP --> OUTPUTS
```

**Sources:** [src/scimilarity/cell_query.py:206-275]()

#### Centroid-Based Search

The system supports searching using centroids computed from marked cells:

```python
# Mark cells for centroid calculation
adata.obs["used_in_query"] = (adata.obs["celltype"] == "macrophage").astype(int)

# Search using centroid
centroid_embedding, nn_idxs, nn_dists, metadata, qc_stats = cq.search_centroid_nearest(
    adata, 
    centroid_key='used_in_query',
    k=10000,
    qc=True
)
```

The method includes quality control metrics to assess query coherence through k-means clustering analysis.

**Sources:** [src/scimilarity/cell_query.py:277-375]()

#### Cluster Centroid Search

For analyzing multiple clusters simultaneously:

```python
# Search all cluster centroids
centroid_embeddings, cluster_idx, nn_idxs, nn_dists, metadata = cq.search_cluster_centroids_nearest(
    adata, 
    cluster_key="leiden",
    k=10000
)
```

This method returns results organized by cluster labels, enabling comparative analysis across cell types.

**Sources:** [src/scimilarity/cell_query.py:377-473]()

### Exhaustive Search Methods

#### Distance-Based Exhaustive Search

```mermaid
flowchart TD
    subgraph "search_exhaustive()"
        BUFFER["Buffer Processing<br/>buffer_size=100000"]
        DIST["Distance Calculation<br/>scipy.spatial.distance.cdist"]
        DISTFILT["Distance Filtering<br/>max_dist threshold"]
        METAFILT["Metadata Filtering<br/>metadata_filter dict"]
        SORT["Sort by Distance"]
    end
    
    subgraph "Inputs"
        EMBEDDINGS["Query Embeddings"]
        MAXDIST["max_dist=0.03"]
        METAFILTER["metadata_filter"]
    end
    
    subgraph "Processing Loop"
        BATCH["Batch Cell Processing"]
        COSINE["Cosine Distance Computation"]
        FILTER["Apply Filters"]
    end
    
    EMBEDDINGS --> BUFFER
    MAXDIST --> DISTFILT
    METAFILTER --> METAFILT
    
    BUFFER --> BATCH
    BATCH --> COSINE
    COSINE --> FILTER
    FILTER --> SORT
```

The exhaustive search provides exact distance calculations but processes cells in batches for memory efficiency.

**Sources:** [src/scimilarity/cell_query.py:475-559]()

#### Exhaustive Centroid Search

Combines centroid calculation with exhaustive search for precise results:

```python
# Exhaustive search with metadata filtering
nn_idxs, nn_dists, metadata = cq.search_centroid_exhaustive(
    adata,
    centroid_key='used_in_query',
    max_dist=0.03,
    metadata_filter={"tissue": "lung", "disease": "healthy"}
)
```

**Sources:** [src/scimilarity/cell_query.py:561-652]()

## Quality Control and Validation

The centroid search methods include built-in quality control through query coherence analysis:

- **K-means clustering** of query cells to assess homogeneity
- **Overlap analysis** between cluster centroids and search results
- **Query coherence score** indicating result reliability

This ensures that centroid-based queries represent coherent cell populations rather than heterogeneous mixtures.

**Sources:** [src/scimilarity/cell_query.py:362-374](), [src/scimilarity/cell_query.py:639-651]()

## Performance Considerations

The `CellQuery` system balances performance and accuracy through:

- **Approximate search** via HNSW indices for fast queries
- **Exhaustive search** for precise distance-based filtering
- **Batch processing** to manage memory usage during exhaustive search
- **TileDB integration** for efficient large-scale data access
- **Configurable parameters** (`ef`, `k`, `buffer_size`) for performance tuning

The choice between approximate and exhaustive methods depends on the specific requirements for speed versus precision in each use case.

**Sources:** [src/scimilarity/cell_query.py:475-559](), [src/scimilarity/cell_query.py:206-275]()

---