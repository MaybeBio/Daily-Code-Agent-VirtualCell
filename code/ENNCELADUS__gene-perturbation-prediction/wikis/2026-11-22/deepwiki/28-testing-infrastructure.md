# Testing Infrastructure

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [tests/conftest.py](tests/conftest.py)

</details>



The testing infrastructure for `gene-perturbation-prediction` validates environment configurations, synthetic data generation, pipeline integration, and end-to-end model execution workflows. The test suite relies on `pytest` combined with specialized environment bootstrap guards and programmatic synthetic data generators defined in `tests/conftest.py` [tests/conftest.py:1-173]().

This page provides a high-level overview of the testing systems. For deep technical details, refer to the child pages:
- [Fixtures and Synthetic Data](#7.1)
- [Integration and End-to-End Tests](#7.2)

---

## 7.1. Fixtures and Synthetic Data

The testing foundation is anchored by global environment setup and deterministic synthetic data generators. Because operations involve PyTorch, NumPy, SciPy sparse matrices, and external libraries like XGBoost, `tests/conftest.py` implements startup guards to prevent deadlocks and segmentation faults before test collection begins [tests/conftest.py:7-29](). 

Synthetic fixtures generate reproducible benchmark CSVs (`synthetic_benchmark_csv`, `synthetic_all_cv_benchmark_csv`) and cell-bag matrices (`synthetic_bags_npz`) without requiring real DepMap or single-cell multi-gigabyte raw files. These simulate structural constraints such as CV folds, gene effects, and cellular response distributions.

```
Title: "Testing Environment and Fixtures Architecture"
graph TD
    subgraph "Environment Guards"
        A["tests/conftest.py"] -->|Sets| B["PYTORCH_ENABLE_MPS_FALLBACK"]
        A -->|Pins| C["OMP_NUM_THREADS=1"]
        A -->|Loads| D["xgboost import order"]
    end
    subgraph "Synthetic Generators"
        A --> E["_write_synthetic_benchmark"]
        A --> F["_write_synthetic_all_cv_benchmark"]
        A --> G["_write_synthetic_bags_npz"]
    end
    E --> H["synthetic_benchmark_csv"]
    F --> I["synthetic_all_cv_benchmark_csv"]
    G --> J["synthetic_bags_npz"]
```
*Sources: [tests/conftest.py:7-154]()*

For detailed breakdowns of how these fixtures construct inputs, configure h5ad stubs, and execute stage0 probes, see [Fixtures and Synthetic Data](#7.1).

---

## 7.2. Integration and End-to-End Tests

Integration and end-to-end (e2e) tests verify that disparate subsystems—such as data loaders, model adapters, response prediction heads, and training loops—interact correctly. These tests run through complete life-cycle steps including joint training updates, model checkpointing, distributed execution state, and command-line interface (CLI) execution paths.

They guard against regression issues during refactoring, ensuring that loss calculations, gradient flow, and file serialization conform to expectations across different deployment modes.

```
Title: "End-to-End Test and Pipeline Integration Flow"
graph TD
    subgraph "Test Execution"
        T1["pytest invocation"] --> T2["Integration Tests"]
        T2 --> T3["CLI Entrypoint Tests"]
        T2 --> T4["Distributed and Resume Tests"]
    end
    subgraph "Guarded Components"
        T3 --> C1["src.train / src.evaluate"]
        T4 --> C2["src/train/checkpoint.py"]
        T4 --> C3["src/train/distributed.py"]
    end
```
*Sources: [tests/conftest.py:1-173]() [src/train/checkpoint.py:1-1]() [src/train/distributed.py:1-1]()*

For detailed coverage of test suites verifying joint training loops, resume states, and CLI behavior, see [Integration and End-to-End Tests](#7.2).

---