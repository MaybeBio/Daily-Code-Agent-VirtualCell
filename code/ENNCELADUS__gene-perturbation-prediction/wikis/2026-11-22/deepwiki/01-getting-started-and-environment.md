# Getting Started and Environment

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [hpc/run.sh](hpc/run.sh)
- [pyproject.toml](pyproject.toml)
- [tests/conftest.py](tests/conftest.py)
- [uv.lock](uv.lock)

</details>



## Purpose and Scope
This page details the local and high-performance computing (HPC) environment configuration for the `gene-perturbation-prediction` repository. It covers dependency management via `uv` and `pyproject.toml`, core library versions (`torch`, `arc-state`, `scanpy`), environment hardening and test suites via `tests/conftest.py`, and cluster execution orchestration through `hpc/run.sh`. 

---

## 1. Project Initialization and Dependency Management

The project uses `uv` for fast dependency resolution and environment bootstrapping, pinned to Python `>=3.11,<3.13` [pyproject.toml:5-11](). Project metadata and dependencies are declared in `pyproject.toml` [pyproject.toml:5-31](), with strict locking handled in `uv.lock` [uv.lock:1-3]().

The core dependency stack includes scientific computing libraries, deep learning frameworks, and domain-specific single-cell analysis tools:
- **Deep Learning & Acceleration**: `torch>=2.7.0`, `accelerate>=1.13.0`, and the external perturbation backbone `arc-state` loaded directly via Git reference (`@git+https://github.1com/ArcInstitute/state.git@bf4fbc9ea35bf4d1e91afe201b663dec5d8bdd48`) [pyproject.toml:12-14,27]().
- **Single-Cell & Bioinformatics**: `scanpy>=1.11.2`, `anndata>=0.12.10`, `mudata>=0.3.8`, and `scvi-tools>=1.4.0.post1,<1.4.1` [pyproject.toml:13,16,22,24]().
- **Data & Numerical Backends**: `numpy>=2.2.6`, `pandas>=2.2.3`, `scipy>=1.15.0`, `h5py==3.15.1`, and `pyarrow>=14.0.2` [pyproject.toml:15,18-20,25]().

Optional dependencies are partitioned into distinct installation groups:
- `baseline`: Includes `xgboost==2.1.4` for tree-based control models [pyproject.toml:33-36]().
- `research`: Includes `datasets>=2.10.0` and `scib>=1.1.0` for integration benchmarking [pyproject.toml:37-40]().
- `viz`: Includes visualization libraries such as `matplotlib==3.10.7`, `seaborn==0.13.2`, `networkx==3.5`, and `tabulate==0.9.0` [pyproject.toml:41-46]().
- `dev`: Development group containing `pytest>=8.3.5`, `ruff>=0.11.8`, and `ipython>=8.18.1` [pyproject.toml:48-55]().

```mermaid
title "Dependency and Configuration Architecture"
graph TD
    subgraph "PyProjectConfig [pyproject.toml]"
        A[pyproject.toml] --> B[CoreDependencies]
        A --> C[OptionalGroups]
        A --> D[DevGroup]
        A --> E[ToolConfigRuffPytest]
    end
    subgraph "Core Stack"
        B --> F[torch]
        B --> G[arc-state]
        B --> H[scanpy]
        B --> I[anndata]
    end
    subgraph "Tooling"
        E --> J[pytest]
        E --> K[ruff]
    end
    classDef default fill:none,stroke:#000,stroke-width:1px;
```
*Sources: [pyproject.toml:1-101](), [uv.lock:1-7]()*.

---

## 2. Test Environment and Conftest Guards

The test suite is located in `tests/` and executed using `pytest`, configured under `[tool.pytest.ini_options]` [pyproject.toml:99-101](). To prevent platform-specific crashes, race conditions, and backend incompatibilities, `tests/conftest.py` executes critical environment initialization before any test modules or external libraries are loaded [tests/conftest.py:1-30]().

### Key Environment Guards
1. **MPS Fallback Guard**: Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow operations lacking native Apple Silicon MPS backward implementations (such as specific distance metrics or energy distance losses relying on `torch.cdist`) to seamlessly fall back to CPU execution [tests/conftest.py:7-12]().
2. **OpenMP Thread-Pool Pinning**: Sets `OMP_NUM_THREADS=1` to prevent segmentation faults caused by conflicting OpenMP runtimes bundled independently in `scipy` (sparse matrix conversions) and `torch` (sparse tensor construction) [tests/conftest.py:14-19]().
3. **OpenMP Load-Order Guard**: Safely imports `xgboost` prior to `torch` loading to establish a stable dynamic library load order on macOS, avoiding inter-library conflicts [tests/conftest.py:21-29]().

```mermaid
title "Conftest Environment Initialization Flow"
graph TD
    subgraph "Test Execution Entry [tests/conftest.py]"
        Start[pytest initialization] --> MPS[Set PYTORCH_ENABLE_MPS_FALLBACK=1]
        MPS --> OMP[Set OMP_NUM_THREADS=1]
        OMP --> XGB[Import xgboost to pin libomp load order]
        XGB --> Fixtures[Register Synthetic Fixtures: synthetic_benchmark_csv, synthetic_bags_npz]
    end
    classDef default fill:none,stroke:#000,stroke-width:1px;
```
*Sources: [tests/conftest.py:1-155]()*.

---

## 3. HPC Execution Entrypoints (`hpc/run.sh`)

Cluster execution and script orchestration are managed through the shell script `hpc/run.sh` [hpc/run.sh:1-45](). The script standardizes environment paths, validates visible accelerator devices, and handles distributed training launches via Hugging Face `accelerate`.

### Supported Subcommands
- `prepare`: Invokes data preparation modules via `src.experiments.prepare` [hpc/run.sh:6,32]().
- `train`: Dynamically counts available CUDA devices, configures multi-GPU training flags, and launches `src.train` through `accelerate` [hpc/run.sh:7,34-43]().
- `test`: Evaluates a saved checkpoint using `src.evaluate` [hpc/run.sh:8,33]().
- `p1a`, `p1b`, `p1c`: Dispatches diagnostic experiment programs for response-pathway analysis (`src.experiments.p1a`, `src.experiments.p1b`, `src.experiments.p1c`) [hpc/run.sh:9-11,29-31]().

```mermaid
title "HPC Script Command Dispatch and Execution"
graph TD
    subgraph "HPC Entrypoint [hpc/run.sh]"
        CLI[hpc/run.sh command args] --> Switch{Command Type}
        Switch -->|prepare| Prep[Exec python -m src.experiments.prepare]
        Switch -->|train| TrainCheck[Check torch.cuda.device_count]
        TrainCheck -->|Workers > 1| MultiGPU[Launch accelerate multi_gpu bf16]
        TrainCheck -->|Workers <= 1| SingleGPU[Launch accelerate single GPU bf16]
        Switch -->|test| TestEval[Exec python -m src.evaluate]
        Switch -->|p1a/p1b/p1c| ExpDispatch[Exec python -m src.experiments.p1a/b/c]
    end
    classDef default fill:none,stroke:#000,stroke-width:1px;
```
*Sources: [hpc/run.sh:1-45]()*.

---

## 4. Summary of Verification Commands

To verify the local installation and test setup, run the following commands:

| Operation | Command | Target / Module |
| :--- | :--- | :--- |
| **Environment Sync** | `uv sync` | Bootstraps `.venv` from `pyproject.toml` and `uv.lock` |
| **Test Suite** | `uv run pytest` | Executes all unit and integration tests under `tests/` |
| **Data Preparation** | `./hpc/run.sh prepare configs/geneeffect_joint.yaml` | Executes pipeline preparation entrypoint [hpc/run.sh:6]() |
| **Distributed Training** | `./hpc/run.sh train configs/geneeffect_joint.yaml` | Dispatches multi-GPU training via Accelerate [hpc/run.sh:7]() |

*Sources: [pyproject.toml:57-58](), [hpc/run.sh:1-45](), [tests/conftest.py:1-31]()*.

---