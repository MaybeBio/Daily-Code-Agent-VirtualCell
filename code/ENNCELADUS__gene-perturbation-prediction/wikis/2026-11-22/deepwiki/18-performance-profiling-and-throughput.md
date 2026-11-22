# Performance Profiling and Throughput

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/results/2026-09-06-joint-batch1024-production.json](docs/results/2026-09-06-joint-batch1024-production.json)
- [docs/results/2026-09-06-joint-batch1024.md](docs/results/2026-09-06-joint-batch1024.md)
- [docs/results/2026-09-06-joint-throughput-diagnosis.md](docs/results/2026-09-06-joint-throughput-diagnosis.md)
- [docs/results/2026-09-06-joint-throughput-probes.json](docs/results/2026-09-06-joint-throughput-probes.json)
- [docs/results/2026-09-06-joint-throughput-production.json](docs/results/2026-09-06-joint-throughput-production.json)
- [docs/results/2026-09-06-joint-vectorization-diagnosis.md](docs/results/2026-09-06-joint-vectorization-diagnosis.md)
- [docs/results/2026-09-06-joint-vectorization-probes.json](docs/results/2026-09-06-joint-vectorization-probes.json)
- [docs/results/2026-09-06-joint-vectorization-production.json](docs/results/2026-09-06-joint-vectorization-production.json)
- [src/experiments/profile_joint.py](src/experiments/profile_joint.py)
- [src/model/state.py](src/model/state.py)
- [tests/test_joint_objective.py](tests/test_joint_objective.py)
- [tests/test_state_core.py](tests/test_state_core.py)

</details>



This page covers the system performance profiling infrastructure, throughput diagnosis procedures, vectorization enhancements, and scaling outcomes for the joint GeneEffect and response-replay training architecture. The implementation centers around bounded profiling harnesses (`src/experiments/profile_joint.py`), adapter-level batching optimizations (`src/model/state.py`), and empirical diagnosis records spanning single-GPU and distributed (DDP) executions on NVIDIA H20 hardware.

---

## 1. The Bounded Profiling Probe (`profile_joint.py`)

To evaluate training iterations without committing checkpoints or running full multi-day training epochs, the codebase provides a specialized timing and profiling probe in `src/experiments/profile_joint.py`. 

The script parses command-line arguments configuring the target YAML file, output path, step count (default 64, strictly required to be a positive multiple of 4 to sample regular and response-replay steps consistently), and an optional dependency batch size override [src/experiments/profile_joint.py:25-39](). It enforces execution under a CUDA accelerator context and disables multi-worker operator profiling unless running a single worker process [src/experiments/profile_joint.py:40-47]().

The probe initialization flow executes the following sequence:
1. Loads prepared inputs using `load_inputs` [src/experiments/profile_joint.py:51]().
2. Instantiates the joint model via `build_joint_model` and transfers it to the accelerator device [src/experiments/profile_joint.py:52]().
3. Fits startup normalization tensors through `fit_startup_standardizer` [src/experiments/profile_joint.py:53-56]().
4. Wraps the model and optimizer with `accelerator.prepare` [src/experiments/profile_joint.py:57-58]().
5. Constructs training and response replay iterators via `make_training_loaders` [src/experiments/profile_joint.py:60]().

During the timed loop, update steps alternate between ordinary GeneEffect updates and response-replay updates (triggered every 4 steps) using `train_update` [src/experiments/profile_joint.py:64-70](). CUDA synchronization ensures accurate hardware timing, discarding the first 8 warmup steps before measuring median durations [src/experiments/profile_joint.py:72-82](). Peak allocated memory is queried via `torch.cuda.max_memory_allocated()` and aggregated across ranks using distributed all-reduce operations if `world_size > 1` [src/experiments/profile_joint.py:83-89]().

```mermaid
title "Natural Language Space to Code Entity Space: Profiling Probe Execution"
graph TD
    subgraph "Natural Language Space"
        NL_Init["H20 Hardware Initialization & Warmup"]
        NL_Step["Bounded Step Timing: Ordinary vs Replay"]
    end

    subgraph "Code Entity Space"
        CE_Config["src/experiments/config.load_config"]
        CE_Inputs["src/data/prepared.load_inputs"]
        CE_Model["src/model/initialization.build_joint_model"]
        CE_Norm["src/model/normalization.fit_startup_standardizer"]
        CE_Train["src/training/trainer.train_update"]
        CE_Profile["src/experiments/profile_joint.py:main"]
    end

    NL_Init --> CE_Profile
    CE_Profile --> CE_Config
    CE_Profile --> CE_Inputs
    CE_Profile --> CE_Model
    CE_Model --> CE_Norm
    NL_Step --> CE_Train
    CE_Profile --> CE_Train
```

Sources: `src/experiments/profile_joint.py`, `docs/results/2026-09-06-joint-throughput-diagnosis.md`

---

## 2. Throughput Diagnosis and Resident Input Optimizations

Initial baseline profiling on single-GPU H20 hardware revealed severe bottlenecks stemming from repeated CPU-to-GPU memory transfers of fixed input structures. Baseline profiling recorded 5,832 `memcpy` calls and 4,559 stream synchronizations across only four updates, where dependency transfers alone accounted for approximately 75 ms per update [docs/results/2026-09-06-joint-throughput-diagnosis.md:68-70]().

To mitigate this, several architectural shifts were introduced:
* **Resident Basal Inputs**: Datasets retain one sampled basal tensor per eligible `ModelID` directly on the worker device, allowing reuse across batches without triggering recurring host-to-device transfers [docs/results/2026-09-06-joint-throughput-diagnosis.md:76-78]().
* **Window Views**: Complete STATE windows leverage tensor views instead of allocating and transferring new index tensors per condition [docs/results/2026-09-06-joint-throughput-diagnosis.md:74-75]().

```mermaid
title "Natural Language Space to Code Entity Space: Resident Input Optimization"
graph TD
    subgraph "Natural Language Space"
        NL_Resident["Device-Resident Basal Tensors"]
        NL_Views["Zero-Copy Window Views"]
    end

    subgraph "Code Entity Space"
        CE_Load["src/data/prepared.load_inputs"]
        CE_Dataset["src/data/datasets.DependencyDataset"]
        CE_Adapter["src/model/state.StateForwardAdapter"]
        CE_Forward["src/model/state.StateForwardAdapter:forward_condition_chunks"]
    end

    NL_Resident --> CE_Dataset
    NL_Views --> CE_Adapter
    CE_Load --> CE_Dataset
    CE_Dataset --> CE_Forward
    CE_Adapter --> CE_Forward
```

Sources: `src/experiments/profile_joint.py`, `docs/results/2026-09-06-joint-throughput-diagnosis.md`

---

## 3. Vectorization and Kernel-Launch Minimization

Subsequent vectorization diagnostics identified further inefficiencies in dtype casting, perturbation expansion, and distance loss computations [docs/results/2026-09-06-joint-vectorization-diagnosis.md:1-16](). 

Key code-level vectorization improvements implemented in `src/model/state.py` and related response modules include:
* **Single FP32 Cast**: Low-precision STATE outputs (`torch.float16` or `torch.bfloat16`) are cast to `torch.float32` *once* at the boundary of `forward_condition_chunks` before splitting into condition views, eliminating redundant casting and backward cast overhead per chunk [src/model/state.py:125-129]().
* **Batched Perturbation Expansion**: Perturbations across multi-condition chunks are expanded as a single matrix operation in `forward_condition_chunks`, avoiding individual reduction kernels per chunk during gradient propagation [src/model/state.py:108-112]().
* **Response Distance Grouping**: Response bags are grouped by shape to enable batched distance computation kernels in `src/model/response.py`, reducing kernel launch counts and stream synchronizations [docs/results/2026-09-06-joint-vectorization-diagnosis.md:58-65]().

These optimizations reduced total kernel launches from 18,128 to 11,069 (a 38.9% reduction) and stream synchronizations by 19.3% [docs/results/2026-09-06-joint-vectorization-diagnosis.md:62-65]().

Sources: `src/model/state.py`, `docs/results/2026-09-06-joint-vectorization-diagnosis.md`

---

## 4. Empirical Performance and Scaling Results

Profiling probes and production logs document steady throughput improvements across successive optimization phases on NVIDIA H20 nodes (`apg3op6hp3v99-0`, 97871 MiB VRAM).

### Throughput and Memory Progression

| Implementation Phase | Revision | GPUs | Dependency Batch / Rank | Throughput (Rows/s) | Peak Allocated (GiB / rank) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline | `7a79352` [docs/results/2026-09-06-joint-throughput-probes.json:8]() | 1 | 256 | 802 | 15.39 [docs/results/2026-09-06-joint-throughput-probes.json:16]() |
| Window Views | `abe57f0` [docs/results/2026-09-06-joint-throughput-probes.json:33]() | 1 | 256 | 849 | 15.31 [docs/results/2026-09-06-joint-throughput-probes.json:41]() |
| Resident Basal Inputs | `bbe4670` [docs/results/2026-09-06-joint-throughput-probes.json:83]() | 1 | 256 | 1,179 | 14.99 [docs/results/2026-09-06-joint-throughput-probes.json:91]() |
| Optimized DDP (2 GPUs) | `f3d6118` [docs/results/2026-09-06-joint-throughput-probes.json:108]() | 2 | 256 | 2,043 | 15.15 [docs/results/2026-09-06-joint-throughput-probes.json:116]() |
| Final Vectorization | `f73f4e0` [docs/results/2026-09-06-joint-vectorization-diagnosis.md:10]() | 2 | 256 | 2,230 | 15.16 [docs/results/2026-09-06-joint-vectorization-diagnosis.md:38]() |
| Batch 1024 Scaling | `36acbd0` [docs/results/2026-09-06-joint-batch1024.md:5]() | 2 | 1,024 | **2,867** | ~60.8 GiB (Occupancy) [docs/results/2026-09-06-joint-batch1024.md:49-53]() |

### Batch Size Scaling and Resource Utilization

Scaling the per-rank dependency batch size from 256 to 1024 (`outputs/geneeffect_joint/joint_seed0_20260906T174818Z_b1024/`) increased row throughput to **2,867 rows/second** while raising GPU compute utilization from ~76% to over **88–90%** across both H20 cards [docs/results/2026-09-06-joint-batch1024.md:43-52](). Observed allocator memory occupancy stabilized at approximately 62.3 GiB per card under batch 1024 [docs/results/2026-09-06-joint-batch1024.md:51-53]().

Sources: `docs/results/2026-09-06-joint-throughput-probes.json`, `docs/results/2026-09-06-joint-vectorization-diagnosis.md`, `docs/results/2026-09-06-joint-batch1024.md`

---