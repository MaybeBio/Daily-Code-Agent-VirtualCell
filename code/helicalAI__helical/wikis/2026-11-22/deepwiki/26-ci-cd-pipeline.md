# CI/CD Pipeline

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.github/workflows/main.yml](.github/workflows/main.yml)
- [.github/workflows/release.yml](.github/workflows/release.yml)
- [ci/tests/data/cell_type_sample.h5ad](ci/tests/data/cell_type_sample.h5ad)
- [helical/models/evo_2/Dockerfile](helical/models/evo_2/Dockerfile)
- [requirements-dev.txt](requirements-dev.txt)

</details>



This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline implemented for the Helical bio foundation models framework. The pipeline automates testing, verification, and deployment processes to ensure code quality and reliability. For information about testing specifications, see [Testing](#9.2).

## Pipeline Overview

The Helical framework implements two primary CI/CD workflows:

1. **CI Pipeline**: Runs on pull requests to the `main` branch, focusing on testing and verification
2. **Release Pipeline**: Runs on pull requests to the `release` branch, handling both testing and deployment

Both pipelines leverage GitHub Actions to automate these processes, with tests running on self-hosted runners to provide access to GPU resources needed for the bio foundation models.

```mermaid
flowchart TD
    subgraph "CI Pipeline"
        PR_Main["Pull Request to main"] --> CI["CI Pipeline Trigger"]
        CI --> Tests["Unit Tests"]
        CI --> IntegrationTests["Integration Tests"]
        CI --> Notebooks["Notebook Tests"]
    end
    
    subgraph "Release Pipeline"
        PR_Release["Pull Request to release"] --> Release["Release Pipeline Trigger"]
        Release --> FreshInstall["Fresh Installation Test"]
        Release --> RTests["Unit Tests"]
        Release --> RIntegrationTests["Integration Tests"]
        Release --> RNotebooks["Notebook Tests"]
        RNotebooks --> TagVersion["Tag Version"]
        TagVersion --> BuildPackage["Build Package"]
        BuildPackage --> PublishPyPI["Publish to PyPI"]
    end
```

**Diagram: CI/CD Pipeline Overview**

Sources: [.github/workflows/main.yml](), [.github/workflows/release.yml]()

## CI Pipeline Components

The CI pipeline runs automatically on pull requests to the `main` branch and consists of three parallel jobs:

### 1. Unit Tests

The unit tests job:
- Checks out the repository code
- Sets up Python 3.11.8
- Installs development dependencies and the package itself
- Downloads required model files and datasets
- Runs the unit tests with code coverage reporting
- Uploads the coverage report as an artifact

### 2. Integration Tests

The integration tests verify that all models work as expected:
- Runs all models in the framework with a GPU (`CUDA_VISIBLE_DEVICES: 0`)
- Tests both model loading and fine-tuning for each supported model
- Includes tests for all RNA models (Geneformer, scGPT, UCE, Helix-mRNA, Mamba2-mRNA)
- Includes tests for all DNA models (HyenaDNA, Caduceus)

### 3. Notebook Tests

The notebook tests ensure that all example notebooks execute correctly:
- Runs on a GPU-enabled runner
- Makes slight modifications to input data to speed up notebook execution
- Uses pytest with nbmake to execute all notebooks in the `examples/notebooks` directory

```mermaid
flowchart LR
    subgraph "Unit Tests"
        UT1["Checkout Code"] --> UT2["Setup Python 3.11.8"]
        UT2 --> UT3["Install Dependencies"]
        UT3 --> UT4["Download Model Files"]
        UT4 --> UT5["Run pytest with Coverage"]
        UT5 --> UT6["Upload Coverage Report"]
    end
    
    subgraph "Integration Tests"
        IT1["Checkout Code"] --> IT2["Setup Python 3.11.8"]
        IT2 --> IT3["Install Helical"]
        IT3 --> IT4["Download Model Files"]
        IT4 --> IT5["Test All Models"]
        IT5 --> IT6["Test All Fine-tuning"]
    end
    
    subgraph "Notebook Tests"
        NT1["Checkout Code"] --> NT2["Setup Python 3.11.8"]
        NT2 --> NT3["Install Helical"]
        NT3 --> NT4["Reduce Test Datasets"]
        NT4 --> NT5["Run Notebooks with pytest-nbmake"]
    end
```

**Diagram: CI Pipeline Job Details**

Sources: [.github/workflows/main.yml:9-159]()

## Release Pipeline

The release pipeline extends the CI pipeline with additional steps to handle the release process:

### 1. Fresh Installation Test

Before running the full test suite, the release pipeline first verifies that the package can be cleanly installed on a fresh environment:

- Uses a standard Ubuntu 24.04 runner (not self-hosted)
- Installs the package without any pre-existing dependencies
- Verifies basic installation integrity

### 2. Testing Suite

The release pipeline runs the same testing jobs as the CI pipeline:
- Unit tests with code coverage
- Integration tests for all models
- Notebook tests

### 3. Release Process

After all tests pass, the release pipeline:
1. Extracts the version number from `pyproject.toml`
2. Checks if a tag for this version already exists
3. Creates a new Git tag (v{version})
4. Builds the Python package
5. Publishes the package to PyPI using a secure API token

```mermaid
flowchart TD
    subgraph "Release Process"
        RP1["All Tests Pass"] --> RP2["Extract Version from pyproject.toml"]
        RP2 --> RP3["Check if Tag Exists"]
        RP3 --> RP4["Create Git Tag v{version}"]
        RP4 --> RP5["Build Package"]
        RP5 --> RP6["Publish to PyPI"]
    end
```

**Diagram: Release Process**

Sources: [.github/workflows/release.yml:179-224]()

## Test Execution Environment

The CI/CD pipeline runs tests in specific environments to ensure consistent results:

1. **Unit Tests**: Run with CPU only (`CUDA_VISIBLE_DEVICES=-1`) to verify core functionality without GPU dependency
2. **Integration Tests**: Run with GPU enabled (`CUDA_VISIBLE_DEVICES: 0`) to test model execution
3. **Notebook Tests**: Run with GPU enabled to validate end-to-end workflows

## Model Testing Coverage

The integration tests systematically cover all models in the Helical framework:

| Model Category | Model | Run Test | Fine-tuning Test |
|----------------|-------|----------|-----------------|
| RNA Models | Geneformer v1 | ✓ | ✓ |
| RNA Models | Geneformer v2 | ✓ | ✓ |
| RNA Models | scGPT | ✓ | ✓ |
| RNA Models | UCE | ✓ | ✓ |
| RNA Models | Helix-mRNA | ✓ | ✓ |
| RNA Models | Mamba2-mRNA | ✓ | ✓ |
| DNA Models | HyenaDNA | ✓ | ✓ |
| DNA Models | Caduceus | ✓ | ✓ |

Sources: [.github/workflows/main.yml:64-126](), [.github/workflows/release.yml:82-144]()

## Containerization

The Helical framework provides Docker containerization for deployment and execution in isolated environments. The repository includes a Dockerfile specifically for the Evo-2 model:

```mermaid
flowchart TD
    subgraph "Containerization Process"
        C1["PyTorch Base Image"] --> C2["Install System Dependencies"]
        C2 --> C3["Install Helical from GitHub"]
        C3 --> C4["Set NVIDIA Environment Variables"]
        C4 --> C5["Install Lightning and Wandb"]
        C5 --> C6["Install Transformer Engine"]
        C6 --> C7["Clone and Install Evo2"]
        C7 --> C8["Set Correct Torch Version"]
    end
```

**Diagram: Container Build Process**

Sources: [helical/models/evo_2/Dockerfile]()

## CI/CD Infrastructure Components

The CI/CD pipeline is built using the following components:

```mermaid
classDiagram
    class GitHubActions {
        +main.yml
        +release.yml
        +Self-hosted runners
        +Ubuntu runners
    }
    
    class TestingFramework {
        +pytest
        +pytest-cov
        +pytest-mock
        +nbmake
    }
    
    class DeploymentTools {
        +PyPI publishing
        +Docker containers
        +Git tags
    }
    
    class TestAssets {
        +Sample data
        +Pre-configured model weights
    }
    
    GitHubActions --> TestingFramework: uses
    GitHubActions --> DeploymentTools: uses for release
    TestingFramework --> TestAssets: uses during tests
```

**Diagram: CI/CD System Components**

Sources: [.github/workflows/main.yml](), [.github/workflows/release.yml](), [requirements-dev.txt]()

## CI/CD Pipeline in the Helical Architecture

The CI/CD pipeline integrates with the overall Helical architecture by ensuring all components function correctly:

```mermaid
flowchart TD
    subgraph "Helical Framework Architecture"
        BM["Base Models"] --> RNA["RNA Models"]
        BM --> DNA["DNA Models"]
        
        RNA --> GFCI["Geneformer CI Test"]
        RNA --> SGCI["scGPT CI Test"]
        RNA --> UCECI["UCE CI Test"]
        RNA --> HMCI["Helix-mRNA CI Test"]
        RNA --> MMCI["Mamba2-mRNA CI Test"]
        
        DNA --> HDCI["HyenaDNA CI Test"]
        DNA --> CDCI["Caduceus CI Test"]
        DNA --> EVCI["Evo2 CI Test"]
        
        FT["Fine-tuning System"] --> FTCI["Fine-tuning CI Tests"]
        
        CICD["CI/CD Pipeline"] --> UnitTests["Unit Tests"]
        CICD --> IntegrationTests["Integration Tests"]
        CICD --> NotebookTests["Notebook Tests"]
        CICD --> Release["Release Process"]
    end
```

**Diagram: CI/CD Integration with Helical Architecture**

Sources: [.github/workflows/main.yml](), [.github/workflows/release.yml]()

## Test Data

The CI pipeline uses sample data files for testing, including:

- Cell type sample data (h5ad format) for testing gene expression models
- Reduced datasets for notebook testing to ensure faster execution

Sources: [ci/tests/data/cell_type_sample.h5ad](), [.github/workflows/main.yml:147-155]()

## Development Tools

The CI/CD pipeline uses the following development tools, specified in the requirements-dev.txt file:

- pytest 8.2.0: Used for running unit tests
- pytest-cov 5.0.0: Used for code coverage reporting
- pytest-mock 3.14.0: Used for mocking components during testing
- nbmake 1.5.4: Used for notebook testing

Sources: [requirements-dev.txt]()

---