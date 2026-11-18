# Development & Deployment

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.github/workflows/pypi-publish.yaml](.github/workflows/pypi-publish.yaml)
- [.github/workflows/pypi-test.yaml](.github/workflows/pypi-test.yaml)

</details>



## Purpose and Scope

This document covers the development infrastructure, continuous integration/continuous deployment (CI/CD) pipelines, package management, and deployment workflows for the SCimilarity codebase. It provides technical details on how the project is built, tested, and distributed to end users.

For information about building documentation, see [Documentation System](#7.1). For details about model training infrastructure, see [Model Training](#4). For post-training deployment workflows, see [Post-Training Workflows](#6).

## CI/CD Pipeline Overview

SCimilarity uses GitHub Actions to automate testing, building, and publishing workflows. The project maintains separate workflows for testing and deployment to ensure code quality and reliable releases.

### GitHub Actions Workflows

```mermaid
flowchart TD
    subgraph "GitHub Repository"
        CODE["Source Code"]
        WORKFLOWS[".github/workflows/"]
    end
    
    subgraph "Testing Pipeline"
        TEST_WORKFLOW["pypi-test.yaml"]
        TEST_JOB["build job"]
        PYTHON_MATRIX["Python 3.8-3.12 Matrix"]
        TOX_TEST["tox execution"]
    end
    
    subgraph "Publishing Pipeline"
        PUB_WORKFLOW["pypi-publish.yaml"]
        PUB_JOB["build job"]
        TOX_BUILD["tox -e clean,build"]
        PYPI_PUBLISH["pypa/gh-action-pypi-publish"]
    end
    
    subgraph "External Services"
        PYPI["PyPI Registry"]
        UBUNTU["ubuntu-latest runners"]
    end
    
    CODE --> WORKFLOWS
    WORKFLOWS --> TEST_WORKFLOW
    WORKFLOWS --> PUB_WORKFLOW
    
    TEST_WORKFLOW --> TEST_JOB
    TEST_JOB --> PYTHON_MATRIX
    PYTHON_MATRIX --> TOX_TEST
    TEST_JOB --> UBUNTU
    
    PUB_WORKFLOW --> PUB_JOB
    PUB_JOB --> TOX_BUILD
    TOX_BUILD --> PYPI_PUBLISH
    PYPI_PUBLISH --> PYPI
    PUB_JOB --> UBUNTU
```

**Sources:** [.github/workflows/pypi-test.yaml:1-43](), [.github/workflows/pypi-publish.yaml:1-40]()

## Testing Infrastructure

The testing pipeline ensures code quality across multiple Python versions and validates package functionality before release.

### Test Workflow Configuration

| Component | Configuration |
|-----------|---------------|
| **Workflow Name** | `Test the library` |
| **Trigger** | Manual dispatch (`workflow_dispatch`) |
| **Python Versions** | 3.8, 3.9, 3.10, 3.11, 3.12 |
| **Runner** | `ubuntu-latest` |
| **Test Framework** | `tox` |

The test workflow uses a matrix strategy to validate compatibility across Python versions:

```mermaid
flowchart LR
    subgraph "Test Matrix Execution"
        TRIGGER["workflow_dispatch"]
        CHECKOUT["actions/checkout@v2"]
        SETUP["actions/setup-python@v4"]
        DEPS["pip install dependencies"]
        TOX["tox execution"]
    end
    
    subgraph "Python Versions"
        PY38["Python 3.8"]
        PY39["Python 3.9"]
        PY310["Python 3.10"]
        PY311["Python 3.11"]
        PY312["Python 3.12"]
    end
    
    subgraph "Dependencies"
        PIP["pip upgrade"]
        FLAKE8["flake8"]
        PYTEST["pytest"]
        TOX_PKG["tox"]
    end
    
    TRIGGER --> CHECKOUT
    CHECKOUT --> SETUP
    SETUP --> PY38
    SETUP --> PY39
    SETUP --> PY310
    SETUP --> PY311
    SETUP --> PY312
    
    PY38 --> DEPS
    PY39 --> DEPS
    PY310 --> DEPS
    PY311 --> DEPS
    PY312 --> DEPS
    
    DEPS --> PIP
    DEPS --> FLAKE8
    DEPS --> PYTEST
    DEPS --> TOX_PKG
    
    PIP --> TOX
    FLAKE8 --> TOX
    PYTEST --> TOX
    TOX_PKG --> TOX
```

**Sources:** [.github/workflows/pypi-test.yaml:14-42]()

## Package Build and Publishing

The publishing pipeline handles package distribution to PyPI, enabling users to install SCimilarity via `pip install scimilarity`.

### Publishing Workflow Components

The publishing workflow executes the following sequence:

1. **Environment Setup**: Configures Python 3.9 environment with system dependencies
2. **Dependency Installation**: Installs build tools including `pandoc` for documentation processing
3. **Package Building**: Uses `tox -e clean,build` to create distribution packages
4. **PyPI Publishing**: Deploys packages using the official PyPI GitHub Action

```mermaid
sequenceDiagram
    participant DEV as "Developer"
    participant GH as "GitHub Actions"
    participant TOX as "tox"
    participant PYPI as "PyPI Registry"
    
    DEV->>GH: "Trigger workflow_dispatch"
    GH->>GH: "Setup Python 3.9"
    GH->>GH: "Install pandoc"
    GH->>GH: "Install pip dependencies"
    Note over GH: "flake8, pytest, tox"
    
    GH->>TOX: "python -m tox -e clean,build"
    TOX->>GH: "Distribution packages"
    
    GH->>PYPI: "pypa/gh-action-pypi-publish@release/v1"
    Note over PYPI: "Uses PYPI_TOKEN secret"
    PYPI->>GH: "Publish confirmation"
```

**Sources:** [.github/workflows/pypi-publish.yaml:22-39]()

### Build Dependencies and Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **pandoc** | Documentation processing | System package via `apt` |
| **tox** | Build automation and testing | Python package |
| **flake8** | Code linting | Python package |
| **pytest** | Test execution | Python package |

**Sources:** [.github/workflows/pypi-publish.yaml:22-29]()

## Development Workflow

### Manual Trigger Strategy

Both workflows use `workflow_dispatch` triggers, indicating a manual release process rather than automatic deployment on pushes or tags. This provides controlled release management where:

- Testing can be executed on-demand for validation
- Publishing requires explicit developer action
- Release timing is fully controlled

The commented-out trigger configurations suggest the infrastructure supports automatic triggers that can be enabled when needed:

```yaml
# Commented automatic triggers:
# on:
#   push:
#     tags: '^[0-9]+.[0-9]+.[0-9]+'  # Publishing
#   push:
#     branches: [main]                # Testing
#   pull_request:
#     branches: [main]                # Testing
```

**Sources:** [.github/workflows/pypi-publish.yaml:6-9](), [.github/workflows/pypi-test.yaml:6-11]()

## Deployment Methods

SCimilarity supports multiple deployment strategies to accommodate different user environments and use cases:

### Distribution Channels

```mermaid
flowchart TD
    subgraph "Source"
        REPO["GitHub Repository"]
        CODE["Source Code"]
        MODELS["Pretrained Models"]
    end
    
    subgraph "Build Process"
        GH_ACTIONS["GitHub Actions"]
        TOX_BUILD["tox build"]
        PACKAGE["Python Package"]
    end
    
    subgraph "Distribution"
        PYPI_REG["PyPI Registry"]
        DOCKER_HUB["Docker Hub"]
        ZENODO["Zenodo Assets"]
    end
    
    subgraph "Installation Methods"
        PIP_INSTALL["pip install scimilarity"]
        DOCKER_PULL["docker pull"]
        DIRECT_DL["Direct download"]
    end
    
    REPO --> CODE
    REPO --> MODELS
    CODE --> GH_ACTIONS
    GH_ACTIONS --> TOX_BUILD
    TOX_BUILD --> PACKAGE
    
    PACKAGE --> PYPI_REG
    CODE --> DOCKER_HUB
    MODELS --> ZENODO
    
    PYPI_REG --> PIP_INSTALL
    DOCKER_HUB --> DOCKER_PULL
    ZENODO --> DIRECT_DL
```

**Sources:** [.github/workflows/pypi-publish.yaml:32-39]()

### Authentication and Security

The publishing workflow uses secure token-based authentication:

- **PyPI Token**: Stored as `PYPI_TOKEN` GitHub secret
- **User Authentication**: Uses `__token__` as username with token-based auth
- **Action Security**: Leverages official PyPA publishing action for secure deployment

**Sources:** [.github/workflows/pypi-publish.yaml:36-39]()

---