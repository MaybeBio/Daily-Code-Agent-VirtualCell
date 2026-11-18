

Metadata filtering and query optimization in scimilarity enables precise cell search operations by constraining results based on biological and experimental attributes while maintaining computational efficiency through strategic buffer management and distance-based pruning.

## Metadata Filtering Architecture

The metadata filtering system operates on a dictionary-based query model where keys represent metadata column names and values specify valid terms for filtering. This design allows for flexible, multi-criteria filtering across biological dimensions such as tissue type, disease state, study origin, and sample identifiers. The filtering occurs during exhaustive search operations, where each batch of cells undergoes both distance-based similarity evaluation and metadata attribute validation before being included in the result set.

Sources: [cell_query.py](src/scimilarity/cell_query.py#L479-L482), [cell_query.py](src/scimilarity/cell_query.py#L521-L532)

```mermaid
flowchart TD
    A[Query Embeddings] --> B{Exhaustive Search}
    B --> C[Batch Processing<br/>buffer_size: 100,000]
    C --> D[Distance Computation<br/>cdist cosine metric]
    D --> E{max_dist Filter}
    E -->|Pass| F[Metadata Filter Application]
    E -->|Fail| G[Discard Results]
    F --> H{metadata_filter<br/>Dictionary}
    H -->|Multiple Keys| I[Logical AND<br/>np.all metadata_hits]
    H -->|Single Key| J[Direct Equality Check]
    I --> K[Combine with Distance Hits<br/>np.vstack[hits, metadata_hits]]
    J --> K
    K --> L[Sort by Distance<br/>np.argsort]
    L --> M[Return Filtered Results<br/>nn_idxs, nn_dists, metadata]
```

> [!TIP]
> The metadata filtering logic applies logical AND operations when multiple filter keys are specified, ensuring results satisfy all specified constraints simultaneously. This is implemented through `np.all(np.vstack(metadata_hits), axis=0)` which combines multiple boolean condition arrays.

## Metadata Schema and Storage

Cell metadata is persisted in TileDB format with a standardized schema containing core biological and experimental attributes. The `CellQuery` class initializes by loading this metadata into memory through the `metadata_tiledb_uri` parameter (default: "cell_metadata"), which provides rapid access during search operations. The metadata structure includes essential columns such as study, sample, tissue, disease, data_type (train/test/NA), and prediction fields derived from cell ontology mappings.

Sources: [cell_query.py](src/scimilarity/cell_query.py#L36-L37), [cell_query.py](src/scimilarity/cell_query.py#L77-L85), [build_cellsearch_metadata.py](scripts/build_cellsearch_metadata.py#L60-L63), [build_cellsearch_metadata.py](scripts/build_cellsearch_metadata.py#L65-L69)

The metadata construction process, implemented in `build_cellsearch_metadata.py`, performs several critical transformations including column renaming (datasetID→study, sampleID→sample, etc.), cell ontology ID to name mapping, and prediction enrichment through KNN-based annotation models. The resulting schema provides a comprehensive view of each cell's provenance and biological classification.

| Metadata Column | Source/Transform | Description | Example Values |
|----------------|------------------|-------------|----------------|
| study | datasetID column | Research study identifier | "GSE123456" |
| sample | sampleID column | Biological sample identifier | "S1", "Tumor_01" |
| tissue | tissue column | Tissue type annotation | "lung", "brain" |
| disease | disease column | Disease state annotation | "normal", "cancer" |
| data_type | train/val indices | Partition membership | "train", "test", "NA" |
| prediction | KNN annotation | Predicted cell type | "macrophage", "T cell" |
| prediction_nn_dist | KNN distances | Annotation confidence metric | 0.015, 0.023 |
| author_label | cellTypeOntologyID mapping | Original cell type label | "CL0000084" |

## Exhaustive Search with Metadata Filtering

The `search_exhaustive` method implements the primary metadata filtering interface, combining distance-based similarity search with attribute-based constraints. This method processes cells in configurable buffer batches (default: 100,000 cells) to balance memory usage with computational throughput, making it suitable for large-scale cell atlases containing millions of cells.

Sources: [cell_query.py](src/scimilarity/cell_query.py#L475-L560)

The search workflow begins by computing pairwise cosine distances between query embeddings and batches of precomputed cell embeddings using scipy's `cdist` function. For each cell in the query batch, the method first applies a distance threshold filter using `max_dist` (default: 0.03), then overlays the metadata filter when specified. The metadata filtering logic creates boolean hit arrays for each key-value pair in the `metadata_filter` dictionary and combines them with the distance-based hits using logical AND operations.

```python
# Distance filtering
hits = nn_dist[row] <= max_dist

# Metadata filtering
if metadata_filter is not None:
    metadata_hits = []
    for k, v in metadata_filter.items():
        metadata_hits.append(
            (self.cell_metadata.loc[nn_idx, k] == v).values
        )
    if len(metadata_hits) > 1:
        metadata_hits = np.all(np.vstack(metadata_hits), axis=0)
    hits = np.all(np.vstack([hits, metadata_hits]), axis=0)
```

This implementation ensures that only cells satisfying both similarity criteria (within max_dist) and metadata constraints (matching all specified key-value pairs) are included in the results. The final results are sorted by increasing distance before being returned.

> [!TIP]
> For optimal query performance, apply the most restrictive metadata filters first and use distance thresholds that match your application's precision requirements. The default max_dist of 0.03 provides a balance between specificity and recall, but can be adjusted based on downstream analysis needs.

## Centroid-Based Query Filtering

Centroid-based search methods extend metadata filtering capabilities to support queries derived from cell groups or clusters. The `search_centroid_exhaustive` method accepts an AnnData object with a binary marker column (`centroid_key`) identifying cells to include in the centroid calculation. This enables queries representing biological conditions, disease states, or experimental treatments while maintaining the same metadata filtering interface as standard cell queries.

Sources: [cell_query.py](src/scimilarity/cell_query.py#L561-L653)

The centroid computation uses the `get_centroid` utility function operating on the counts layer of the input AnnData object, followed by embedding generation through the trained model. The resulting centroid embedding is then passed to the standard `search_exhaustive` method with the same `max_dist`, `metadata_filter`, and `buffer_size` parameters. This architecture ensures consistent filtering behavior across single-cell and group-based queries.

The method optionally performs quality control (QC) through k-means clustering of the query cells and coherence assessment against the search results. The QC statistics include a `query_coherence` metric measuring the overlap between cluster centroids' nearest neighbors and the query results, providing insight into the internal consistency of the query cell population.

## Cluster-Based Query Filtering

For applications requiring queries across multiple cell types or conditions, the `search_cluster_centroids_exhaustive` method provides a comprehensive interface for filtering at the cluster level. This method accepts an AnnData object with cluster labels specified through `cluster_key` and optionally restricts analysis to a specific `cluster_label`.

Sources: [cell_query.py](src/scimilarity/cell_query.py#L654-L744)

Cluster centroids are computed using the `get_cluster_centroids` utility function, which performs gene ordering alignment and handles null/NaN cluster labels based on the `skip_null` parameter. Each centroid embedding is then searched exhaustively against the reference database with consistent metadata filtering applied across all centroids. The results are organized as dictionaries keyed by cluster labels, enabling efficient post-query analysis at the cluster level.

The method adds a `centroid` column to the returned metadata DataFrame, mapping each result to its originating cluster through the `embedding_idx` field. This structure supports downstream analyses such as cluster-specific annotation validation or comparative studies across cell type distributions.

## Sample-Level Metadata Aggregation

Beyond cell-level filtering, the `compile_sample_metadata` method provides sample-level aggregation of nearest neighbor results. This function operates on the indices returned from search operations and aggregates counts by specified hierarchical levels (default: ["study", "sample", "tissue", "disease"]) to enable quantitative comparisons across studies and experimental conditions.

Sources: [cell_query.py](src/scimilarity/cell_query.py#L150-L205)

The aggregation computes both absolute cell counts and fractional representation for each study-sample combination. The fraction calculation uses the total cell count per study-sample pair from the full metadata database, providing normalized measures that account for varying sample sizes across studies. This output is particularly valuable for identifying enriched cell types in specific biological conditions or assessing the generalizability of query results across different experimental batches.

```python
study_sample_cells = self.cell_metadata.groupby(
    ["study", "sample"], observed=True
).size()

for i, row in df.iterrows():
    total_cells = study_sample_cells.loc[(row["study"], row["sample"])]
    fraction.append(row["cells"] / total_cells)
    total.append(total_cells)
df["fraction"] = fraction
df["total"] = total
```

## Performance Optimization Strategies

Query optimization in scimilarity leverages multiple strategies to balance computational efficiency with result quality. The buffer_size parameter controls batch processing granularity, with the default of 100,000 cells representing a practical compromise between memory usage and I/O efficiency. For systems with abundant memory, increasing buffer_size can reduce iteration overhead, while memory-constrained environments may require smaller values to prevent out-of-memory conditions.

Sources: [cell_query.py](src/scimilarity/cell_query.py#L479-L482), [build_cellsearch_metadata.py](scripts/build_cellsearch_metadata.py#L17)

The precomputed embeddings stored in TileDB enable rapid retrieval without re-encoding, and the indexed metadata structure supports efficient subset extraction. The distance threshold (max_dist) provides an early filtering mechanism that eliminates dissimilar cells before metadata filtering is applied, reducing unnecessary boolean operations on metadata attributes.

> [!TIP]
> When working with very large cell atlases (>10 million cells), consider partitioning queries by tissue or study using metadata filters to limit the search space. The TileDB-based storage system supports efficient columnar reads, making it well-suited for this filtering pattern.

## Query Method Comparison

The following table summarizes the metadata filtering capabilities across different query methods in the CellQuery class:

| Method | Input Type | Use Case | Metadata Filter Support | Output Structure |
|--------|-----------|----------|------------------------|------------------|
| search_exhaustive | Embeddings | Single/multiple cell queries | Full dictionary filter | List of arrays per query |
| search_centroid_exhaustive | AnnData + centroid_key | Group-based centroid queries | Full dictionary filter | Single centroid results |
| search_cluster_centroids_exhaustive | AnnData + cluster_key | Multi-cluster comparison | Full dictionary filter | Dictionary keyed by cluster |

All three methods share the same underlying `search_exhaustive` implementation, ensuring consistent filtering behavior and parameter semantics. The choice of method depends primarily on the query structure rather than the filtering requirements.

## Advanced Filtering Patterns

Complex biological queries can be constructed through combinations of metadata filters and distance thresholds. For example, identifying macrophages specifically from lung tissue in normal samples would use:

```python
metadata_filter = {
    "tissue": "lung",
    "disease": "normal",
    "prediction": "macrophage"
}
nn_idxs, nn_dists, metadata = cq.search_exhaustive(
    embeddings,
    max_dist=0.03,
    metadata_filter=metadata_filter,
    buffer_size=100000
)
```

For progressive refinement, queries can be executed with relaxed constraints initially, followed by subsequent filtering on the returned metadata DataFrame. This pattern is particularly useful when the exact cell type names are uncertain or when exploring related biological states.

Sources: [cell_query.py](src/scimilarity/cell_query.py#L544-L549)

## Next Steps

For comprehensive understanding of the search infrastructure, review [K-NN Search with HNSW](10-k-nn-search-with-hnsw) for approximate search methods and [Exhaustive Search Methods](11-exhaustive-search-methods) for additional distance computation strategies. The [Cell Query API Reference](9-cell-query-api-reference) provides complete parameter documentation for all query methods.