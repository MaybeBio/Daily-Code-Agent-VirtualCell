# Data Preparation and Caches

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/data/prepared.py](src/data/prepared.py)
- [src/data/tx1_cache.py](src/data/tx1_cache.py)
- [src/experiments/prepare.py](src/experiments/prepare.py)
- [tests/test_joint_integration.py](tests/test_joint_integration.py)
- [tests/test_tx1_embed_cache.py](tests/test_tx1_embed_cache.py)

</details>



The data preparation phase provides an immutable, offline pipeline [src/experiments/prepare.py:1-95]() that transforms raw single-cell UMI counts, perturbation matrices, and protein metadata into deterministic, standardized caches and configuration manifests (`prepared_inputs.json`). By decoupling resource-intensive ingestion—such as executing the Tx1-3B single-cell encoder [src/data/tx1_cache.py:1-48]() or parsing massive H5AD/parquet archives—from the active training loop, the system guarantees reproducible training runs and low I/O latency across distributed workers.

```mermaid
graph TD
    sub5["Raw Inputs"] -->| "src/experiments/prepare.py" | prep["prepare_inputs"]
    prep -->| "src/data/tx1_cache.py" | tx1["Tx1BasalCache"]
    prep -->| "src/data/response_cache.py" | resp["ResponseTargetsCache"]
    prep -->| "src/data/embeddings.py" | esm["ESM2Embeddings"]
    prep -->| "src/data/q_sc.py" | qsc["QScFeatures"]
    tx1 -->| "PreparedInputs" | trainer["src/training/"]
    resp -->| "PreparedInputs" | trainer
    esm -->| "PreparedInputs" | trainer
    qsc -->| "PreparedInputs" | trainer
```
*Sources: [src/experiments/prepare.py:94-172](), [src/data/tx1_cache.py:1-48](), [src/data/prepared.py:30-66]()*

---

## 2.1 Tx1 Basal Embedding Cache

The Tx1 basal embedding subsystem manages encoder-agnostic cell line embeddings [src/data/tx1_cache.py:1-48](). It uses an injected `EncoderFn` callable to process raw control cells, writing per-cell `tahoe_x1_3b` vector arrays alongside provenance sidecars [src/data/tx1_cache.py:59-73]() and HVG gene-order constraints [src/data/tx1_cache.py:31-38](). The verification pipeline (`verify_cache`) enforces strict schema versions, exact embedding widths (`EMBEDDING_WIDTH = 2560`) [src/data/tx1_cache.py:46](), and cryptographic authentication against pinned Phase-A registration manifests [src/data/tx1_cache.py:136-164]().

For detailed implementation, atomicity guarantees, and sharding strategies, see [Tx1 Basal Embedding Cache](#2.1).

```mermaid
graph TD
    raw["RawAnnData"] -->| "embed_lines" | enc["EncoderFn"]
    enc -->| "write_line_cache" | cache["LineCacheDirectory"]
    cache -->| "verify_cache" | val["ValidationAndProvenance"]
```
*Sources: [src/data/tx1_cache.py:1-48](), [src/data/tx1_cache.py:136-164]()*

---

## 2.2 Response Caches and Streaming

During response prediction training, the pipeline loads sparse perturbations and sampled cell-state distributions using memory-mapped structures [src/data/response_cache.py:1-30](). Response gene bags curate target distributions across training anchors, aligning single-cell response counts with high-variable gene panels [src/data/response.py:1-50](). Streaming loaders fetch these records dynamically to minimize memory overhead during batch construction [src/data/prepared.py:30-66]().

For detailed mechanics on gene bags, memory-mapped caching, and streaming loaders, see [Response Caches and Streaming](#2.2).

```mermaid
graph TD
    bags["assemble_train_response_gene_bags"] -->| "write_response_targets_cache" | rc["ResponseTargetsCache"]
    rc -->| "open_response_targets_cache" | stream["StreamingLoader"]
```
*Sources: [src/experiments/prepare.py:141-151](), [src/data/prepared.py:21-25]()*

---

## 2.3 Gene and Protein Feature Tables

Model conditioning relies on static feature tables precomputed prior to training [src/experiments/prepare.py:94-172](). This includes ESM2 protein language model embedding unions [src/data/embeddings.py:1-40](), UniProt offline cache bootstraps, single-cell quality control metrics (`q_sc` caches) [src/data/q_sc.py:1-35](), and copy-number variation priors derived from reference donors like K562 (`ACH-000551`) [src/experiments/prepare.py:125-133](). These tables align gene symbols across disparate DepMap and Perturb-seq datasets into unified evaluation splits [src/data/geneeffect.py:1-60]().

For deeper coverage of ESM2 precomputations, copy priors, and split configurations, see [Gene and Protein Feature Tables](#2.3).

```mermaid
graph TD
    labels["load_geneeffect_long"] -->| "build_coverage_universe" | univ["CoverageUniverse"]
    univ -->| "write_embedding_union" | esm["ESM2UnionTable"]
    univ -->| "build_q_sc_shards" | qsc["QScCache"]
```
*Sources: [src/experiments/prepare.py:124-134](), [src/data/q_sc.py:1-35]() *

---

## 2.4 Raw Data Ingestion Scripts

Raw inputs are gathered and normalized via dedicated builder scripts located under `src/data/prepare/` and executed through orchestration entrypoints [src/experiments/prepare.py:1-95](). These builders ingest raw UMI matrices from Tahoe downloads, Kinker cancer dependency files, and atlas sources, registering them into centralized source registries [src/data/geneeffect.py:1-60](). Scripts validate dataset integrity and guarantee input immutability before downstream training phases begin [src/experiments/prepare.py:118-120]().

For details on source registries, Tahoe downloads, and builder execution scripts, see [Raw Data Ingestion Scripts](#2.4).

```mermaid
graph TD
    cli["hpc/run.sh prepare"] -->| "parse_config" | prep["prepare_inputs"]
    prep -->| "load_source_registry" | reg["SourceRegistry"]
    reg -->| "materialize" | raw["RawUMI_AnnData"]
```
*Sources: [src/experiments/prepare.py:94-135](), [src/data/geneeffect.py:1-60]()*

---

## 2.5 Datasets, Batching and Sampling

The prepared dataset layer bridges immutable on-disk caches to PyTorch data loaders [src/data/prepared.py:1-66](). Modules such as `src/data/datasets.py` and `src/training/sampling.py` ingest the JSON manifest (`prepared_inputs.json`) [src/data/prepared.py:26-28]() to construct context batches, sample paired control-perturbation vectors, and manage memory-mapped pointers during training loops.

For specifications on dataset structures, batching logic, and configuration loading, see [Datasets, Batching and Sampling](#2.5).

```mermaid
graph TD
    meta["prepared_inputs.json"] -->| "load_inputs" | prep["PreparedInputs"]
    prep -->| "construct" | ds["DatasetAndSampler"]
    ds -->| "batches.py" | loader["TorchDataLoader"]
```
*Sources: [src/data/prepared.py:30-66](), [src/data/prepared.py:137-174]() *

---