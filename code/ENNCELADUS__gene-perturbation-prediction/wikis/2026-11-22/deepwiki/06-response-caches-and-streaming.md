# Response Caches and Streaming

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/data/response_cache.py](src/data/response_cache.py)
- [src/data/response_streaming.py](src/data/response_streaming.py)
- [tests/test_tx1_predicted_response.py](tests/test_tx1_predicted_response.py)
- [tests/test_tx1_response_gene_bags_cache.py](tests/test_tx1_response_gene_bags_cache.py)
- [tests/test_tx1_response_streaming.py](tests/test_tx1_response_streaming.py)

</details>



## Purpose and Scope
This page details the implementation of response gene bags, memory-mapped response target caches, memory-bounded streaming loaders, and predicted-response cache controls within the codebase. These modules govern how observed cellular response states and baseline controls are ingested, stored on disk via memory-mapped layouts, streamed under tight memory budgets, and validated against canonical HVG orders.

Sources: `src/data/response_cache.py:1-46`, `src/data/response_streaming.py:1-43`

---

## 1. Memory-Mapped Response Targets (`ResponseTargetsCache`)

Observed response targets are stored using structured memory-mapped files to allow random access to multi-cell gene bags without holding entire datasets in RAM. The core data structure is `ResponseTargetsCache`, defined in `src/data/response_cache.py:27-36`.

### On-Disk Layout and Schema
The response cache directory contains a manifest file (`manifest.json`), a metadata table (`metadata.parquet`), and three binary array files (`target_cells.npy`, `offsets.npy`, `genes.npy`) adhering to schema version `1` (`src/data/response_cache.py:17-25`).

- `target_cells.npy`: A 2D `float32` array containing concatenated expression vectors for all cells across all bags, shaped `[total_cells, len(hvg_order)]` (`src/data/response_cache.py:126-135`).
- `offsets.npy`: An `int64` array of shape `[n_bags + 1]` marking the start and stop indices in `target_cells.npy` for each response bag (`src/data/response_cache.py:136-151`).
- `metadata.parquet`: A table tracking bag provenance containing columns `perturbation_gene`, `model_id`, and `n_cells` (`src/data/response_cache.py:152-159`).

### Key Functions
- `normalize_response_metadata`: Canonicalizes perturbation symbols by stripping whitespace, converting to uppercase, and checking for duplicate normalized conditions (`src/data/response_cache.py:47-58`).
- `open_response_targets_cache`: Validates manifest schema versions, HVG matching, array data types, offset monotonicity, and returns a initialized `ResponseTargetsCache` instance (`src/data/response_cache.py:61-174`).
- `ResponseTargetsCache.target_bag`: Slices the memory-mapped `target_cells` array using `offsets` to return the cell expression matrix for a specific bag index (`src/data/response_cache.py:41-45`).

```mermaid
graph TD
    A["manifest.json"] --> B["open_response_targets_cache"]
    C["metadata.parquet"] --> B
    D["offsets.npy"] --> B
    E["target_cells.npy"] --> B
    B --> F["ResponseTargetsCache"]
    F --> G["target_bag(index: int)"]
```
Sources: `src/data/response_cache.py:17-174`, `tests/test_tx1_response_gene_bags_cache.py:32-62`

---

## 2. Streaming Loaders and Reservoir Draining

To prevent out-of-memory errors when processing massive datasets (such as HCT116 with over 18,000 perturbed genes where unflattened Python lists consume hundreds of gigabytes), the system uses a memory-bounded streaming and reservoir-draining pipeline (`src/data/response_streaming.py:1-43`).

### Reservoir Draining Algorithm
`drain_gene_reservoirs_to_matrix` builds a compressed sparse row (`csr_matrix`) from per-gene cell reservoirs while actively mutating and draining the reservoir in place (`src/data/response_streaming.py:86-122`).

1. **Total Budget Resolution**: `resolve_total_budget_keep_mask` computes an optional deterministic boolean mask across cells if a total cell budget is specified (`src/data/response_streaming.py:57-84`).
2. **In-Place Draining**: As cells are consumed from reservoirs sorted by gene name, processed slots are explicitly set to `None`. This allows Python's garbage collector to reclaim memory immediately rather than retaining references until the entire builder returns (`src/data/response_streaming.py:32-37`).
3. **Vectorized CSR Construction**: Component arrays (`indices`, `indptr`, `data`) are assembled using vectorized numpy routines rather than plain Python boxed-list iterations (`src/data/response_streaming.py:26-37`).

```mermaid
graph TD
    A["reservoirs: Mapping[str, MutableSequence]"] --> B["drain_gene_reservoirs_to_matrix"]
    C["metadata: pd.DataFrame"] --> B
    B --> D["resolve_total_budget_keep_mask"]
    D --> E["In-place slot nulling"]
    E --> F["csr_matrix"]
```
Sources: `src/data/response_streaming.py:1-146`, `tests/test_tx1_response_streaming.py:68-158`

---

## 3. Predicted-Response Caches and Control Cell Chunking

When evaluating predicted response states or establishing baseline controls, cell indices must be grouped into deterministic batches or windows.

### Control Cell Chunking
The helper `_chunk_control_cell_indices` divides control cell pools into fixed-size sets, handling tail padding deterministically using a configurable random seed (`tests/test_tx1_response_gene_bags_cache.py:148-164`). Because tail window filling depends on the seed, the seed forms part of cache fingerprints.

```mermaid
graph TD
    A["ControlCellPool"] --> B["_chunk_control_cell_indices"]
    B --> C["Seed-sensitive padding"]
    C --> D["Chunked Control Batches"]
```
Sources: `tests/test_tx1_predicted_response.py:13-164`, `tests/test_tx1_response_gene_bags_cache.py:148-164`

---

## 4. Natural Language to Code Entity Mapping

The following diagrams bridge natural language concepts to their concrete code symbols and classes in the repository.

### Diagram: Response Storage Architecture
```mermaid
graph TD
    NLP1["Memory-Mapped Response Cache"] -->|Implemented by| CE1["src/data/response_cache.py"]
    NLP2["Response Bag Container"] -->|Class name| CE2["ResponseTargetsCache"]
    NLP3["Cache Open & Verification"] -->|Function name| CE3["open_response_targets_cache"]
    NLP4["Symbol Normalization"] -->|Function name| CE4["normalize_response_symbols"]
    CE1 --> CE2
    CE1 --> CE3
    CE1 --> CE4
```
Sources: `src/data/response_cache.py:27-174`

### Diagram: Streaming Pipeline Architecture
```mermaid
graph TD
    NLP1["Memory-Bounded Streaming Loader"] -->|Implemented by| CE1["src/data/response_streaming.py"]
    NLP2["Reservoir Draining & CSR Assembly"] -->|Function name| CE2["drain_gene_reservoirs_to_matrix"]
    NLP3["Cell Budget Keep-Mask"] -->|Function name| CE3["resolve_total_budget_keep_mask"]
    CE1 --> CE2
    CE1 --> CE3
```
Sources: `src/data/response_streaming.py:57-146`

---