# Testing and Quality Assurance

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.github/workflows/python_check.yml](.github/workflows/python_check.yml)
- [.pre-commit-config.yaml](.pre-commit-config.yaml)
- [LICENSE](LICENSE)
- [perturbgen/Model/trainer.py](perturbgen/Model/trainer.py)
- [perturbgen/tests/test_cellgen_training.py](perturbgen/tests/test_cellgen_training.py)
- [setup.py](setup.py)

</details>



This section provides an overview of the testing infrastructure, continuous integration (CI) pipelines, and code quality tools used to maintain the PerturbGen repository. The project employs a multi-layered approach to quality assurance, combining local unit tests with automated cloud-based checks and documentation deployment.

### System Overview: QA and CI/CD Mapping

The following diagram bridges the natural language concepts of quality assurance to the specific code entities and configuration files that implement them.

**QA Infrastructure Mapping**
```mermaid
graph TD
    subgraph "Local Testing"
        ["run_test.sh"] --> ["perturbgen/tests/"]
        ["perturbgen/tests/"] --> ["test_cellgen_training.py"]
        ["perturbgen/tests/"] --> ["test_perturbation.py"]
        ["perturbgen/tests/"] --> ["utils.py"]
    end

    subgraph "Automation & Quality"
        [".pre-commit-config.yaml"] --> ["Black/Isort"]
        [".pre-commit-config.yaml"] --> ["Flake8/Mypy"]
        [".github/workflows/python_check.yml"] --> ["GitHub Actions"]
    end

    subgraph "Documentation"
        ["docs/"] --> ["Sphinx"]
    end

    ["test_cellgen_training.py"] -- "validates" --> ["PerturbGenTrainer"]
    ["test_cellgen_training.py"] -- "validates" --> ["CountDecoderTrainer"]
    ["GitHub Actions"] -- "executes" --> ["mypy"]
    ["GitHub Actions"] -- "executes" --> ["flake8"]
```
Sources: [perturbgen/tests/test_cellgen_training.py:14-17](), [.github/workflows/python_check.yml:1-36](), [.pre-commit-config.yaml:1-60]()

---

## 7.1 Test Suite

The test suite is designed to validate the core components of the PerturbGen pipeline, ranging from model training loops to the accuracy of in silico perturbations. It utilizes `unittest` and `pytorch_lightning` utilities to ensure that changes to the architecture do not break existing functionality.

Key components include:
*   **Training Smoke Tests**: Contained in `test_cellgen_training.py`, these tests verify that both the masking model (`PerturbGenTrainer`) and the count decoder (`CountDecoderTrainer`) can initialize and complete a training step using synthetic data [perturbgen/tests/test_cellgen_training.py:15-17]().
*   **Perturbation Validation**: `test_perturbation.py` ensures that the simulation logic correctly handles different perturbation modes (e.g., delete, overexpress).
*   **Test Utilities**: `tests/utils.py` provides shared fixtures such as `dummy_dataset` and `dummy_cell_gene_matrix` to generate mock single-cell data for rapid testing without requiring massive biological datasets [perturbgen/tests/test_cellgen_training.py:69-92]().

For detailed information on running tests and writing new test cases, see **[Test Suite (#7.1)]**.

Sources: [perturbgen/tests/test_cellgen_training.py:48-190]()

---

## 7.2 Performance Benchmarks

Performance benchmarks measure generation efficiency, training throughput, and optimization settings such as `torch.compile`, the fused Adam optimizer, and skip-context-logits optimization. These scripts allow developers to profile GPU memory consumption and training speed across various hardware backends.

For details on running and interpreting benchmark scripts, see **[Performance Benchmarks (#7.2)]**.

Sources: [perturbgen/Model/trainer.py:49-73]()

---

## 7.3 CI/CD, Documentation Build, and DevOps

PerturbGen uses GitHub Actions to automate code quality checks, style enforcement, and documentation builds.

### Continuous Integration
*   **Linting and Type Checking**: The repository uses `flake8` for style enforcement and `mypy` for static type checking [.github/workflows/python_check.yml:26-32](). These are enforced both via `pre-commit` hooks and automated GitHub Actions [.pre-commit-config.yaml:17-53]().
*   **Formatting**: `black` and `isort` are used to maintain a consistent code style across the project [.pre-commit-config.yaml:25-44]().

### DevOps and Documentation
*   **Environment Management**: The project defines dependencies in `setup.py` to ensure reproducible environments across different CI runners [setup.py:1-15]().

**CI/CD Pipeline Flow**
```mermaid
graph LR
    ["Code Push"] --> ["Pre-commit Hooks"]
    ["Pre-commit Hooks"] --> ["GitHub Actions"]
    subgraph "GitHub Actions (python_check.yml)"
        ["Flake8"]
        ["MyPy"]
        ["Pydocstyle"]
    end
```
Sources: [.github/workflows/python_check.yml:21-36](), [.pre-commit-config.yaml:1-60]()

For details on the CI configuration, Docker environments, and documentation hosting, see **[CI/CD, Documentation Build, and DevOps (#7.3)]**.

---