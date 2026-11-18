# Installation

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.github/actions/setup/action.yaml](.github/actions/setup/action.yaml)
- [.readthedocs.yaml](.readthedocs.yaml)
- [docs/index.md](docs/index.md)
- [docs/installation.md](docs/installation.md)
- [docs/release_notes/index.md](docs/release_notes/index.md)
- [docs/tutorials/notebooks/mouse_brain_multimodal.ipynb](docs/tutorials/notebooks/mouse_brain_multimodal.ipynb)
- [docs/tutorials/notebooks/mouse_cns_sample_integration.ipynb](docs/tutorials/notebooks/mouse_cns_sample_integration.ipynb)
- [docs/tutorials/notebooks/mouse_cns_single_sample.ipynb](docs/tutorials/notebooks/mouse_cns_single_sample.ipynb)
- [docs/tutorials/notebooks/mouse_cns_spatial_reference_mapping.ipynb](docs/tutorials/notebooks/mouse_cns_spatial_reference_mapping.ipynb)
- [docs/user_guide/index.md](docs/user_guide/index.md)
- [pyproject.toml](pyproject.toml)
- [src/nichecompass/utils/analysis.py](src/nichecompass/utils/analysis.py)
- [tests/test_basic.py](tests/test_basic.py)

</details>



This document provides instructions for installing NicheCompass, a Python package for end-to-end analysis of spatial multi-omics data. For information about the overall architecture and system design, see [Architecture](#2).

## System Requirements

NicheCompass is compatible with Python 3.9 and does not currently support Apple silicon. The package requires several external dependencies, particularly PyTorch-related libraries which may need GPU support for optimal performance.

```mermaid
flowchart TD
    A["System Requirements"] --> B["Python 3.9"]
    A --> C["CPU or CUDA-compatible GPU"]
    A --> D["Not supported on Apple Silicon"]
    A --> E["Virtual Environment Recommended"]
    
    E --> E1["Conda/Mambaforge"]
    E --> E2["Docker"]
    E --> E3["Python venv"]
```

Sources: [pyproject.toml:10-11](). [docs/installation.md:1-8]()

## Installation Methods

NicheCompass can be installed using pip after setting up the required external dependencies. We recommend using a virtual environment like conda or Docker rather than your system Python.

### Method 1: Using pip

```mermaid
flowchart LR
    A["Install External Libraries"] --> B["Install PyTorch with CUDA"]
    A --> C["Install PyTorch Scatter/Sparse"]
    A --> D["Install bedtools"]
    B & C & D --> E["Install NicheCompass via pip"]
    E --> F["Install Optional Dependencies"]
```

Sources: [docs/installation.md:56-66]()

### Method 2: Using conda environment

```mermaid
flowchart LR
    A["Create conda environment from file"] --> B["Activate environment"]
    B --> C["Install JAX with CUDA support"]
    C --> D["Ready to use NicheCompass"]
```

Sources: [docs/installation.md:39-43]()

## External Libraries Setup

Before installing NicheCompass, you'll need to set up several external libraries:

### PyTorch and PyTorch Extensions

Install PyTorch with appropriate CUDA support if available:

```bash
pip install torch==${TORCH}+${CUDA} --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

For example, for PyTorch 2.0.0 and CUDA 11.7:

```bash
pip install torch==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

### bedtools

Install bedtools using conda:

```bash
conda install bedtools=2.30.0 -c bioconda
```

Sources: [docs/installation.md:12-37](), [.github/actions/setup/action.yaml:30-43]()

## NicheCompass Installation

### Basic Installation

After setting up the external libraries, install NicheCompass via pip:

```bash
pip install nichecompass
```

### Optional Dependencies

NicheCompass provides several optional dependency groups for different functionalities:

```mermaid
graph TD
    A["nichecompass"] --> B["Core Package"]
    A --> C["Optional Dependencies"]
    
    C --> D["benchmarking"]
    C --> E["multimodal"]
    C --> F["tutorials"]
    C --> G["dev"]
    C --> H["docs"]
    C --> I["tests"]
    
    D --> D1["scib-metrics, pynndescent, etc."]
    E --> E1["scglue"]
    F --> F1["jupyter"]
    G --> G1["pre-commit, twine"]
    H --> H1["sphinx packages"]
    I --> I1["pytest, coverage"]
```

Install all optional dependencies:

```bash
pip install nichecompass[all]
```

Or install specific dependency groups:

```bash
pip install nichecompass[benchmarking,multimodal,tutorials]
```

Sources: [pyproject.toml:29-81](), [docs/installation.md:63-66]()

## Package Dependencies

The following table shows the core dependencies required by NicheCompass:

| Dependency | Version Requirement |
|------------|---------------------|
| mlflow | >=1.28.0 |
| pyreadr | >=0.4.6 |
| scanpy | >=1.9.3,<1.10.0 |
| torch-geometric | >=2.2.0 |
| omnipath | >=1.0.7 |
| decoupler | >=1.4.0 |
| numpy | <2 |

Sources: [pyproject.toml:29-37]()

## JAX GPU Support (Optional)

For enabling GPU support for JAX (used in benchmarking), run:

```bash
pip install jaxlib==0.4.7+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Replace with the appropriate CUDA version as needed.

Sources: [docs/installation.md:45-53](), [pyproject.toml:67-74]()

## Verification

After installation, you can verify that NicheCompass is properly installed by running:

```python
import nichecompass
print(nichecompass.__version__)
```

The output should match the version you installed.

Sources: [tests/test_basic.py:1-7]()

## Common Installation Patterns

The following diagram illustrates common installation patterns depending on your needs:

```mermaid
flowchart TD
    start["Start Installation"] --> env["Create Virtual Environment"]
    env --> choose["Choose Installation Pattern"]
    
    choose --> basic["Basic Installation"]
    choose --> multi["Multimodal Analysis"]
    choose --> bench["Benchmarking"]
    choose --> all["Complete Installation"]
    
    basic --> ext["Install External Libraries"]
    ext --> pip_basic["pip install nichecompass"]
    
    multi --> ext2["Install External Libraries"]
    ext2 --> pip_multi["pip install nichecompass[multimodal]"]
    
    bench --> ext3["Install External Libraries"]
    ext3 --> pip_bench["pip install nichecompass[benchmarking]"]
    ext3 --> jax["Install JAX with CUDA"]
    
    all --> ext4["Install External Libraries"]
    ext4 --> pip_all["pip install nichecompass[all]"]
    ext4 --> jax2["Install JAX with CUDA"]
```

Sources: [docs/installation.md:1-73](), [docs/tutorials/notebooks/mouse_brain_multimodal.ipynb:42-49](), [docs/tutorials/notebooks/mouse_cns_single_sample.ipynb:39-43]()

## Package Structure and Components

The following diagram shows how the installation fits into the overall NicheCompass system architecture:

```mermaid
graph TD
    A["NicheCompass Package"] --> B["Core Components"]
    A --> C["Optional Extensions"]
    
    B --> B1["Models"]
    B --> B2["Utils"]
    B --> B3["Base Functions"]
    
    C --> C1["Benchmarking"]
    C --> C2["Multimodal Analysis"]
    C --> C3["Documentation"]
    
    subgraph "Installation Requirements"
        D["PyTorch Ecosystem"]
        E["Dependencies"]
        F["External Tools"]
    end
    
    D --> D1["torch"]
    D --> D2["torch_geometric"]
    D --> D3["torch_scatter"]
    D --> D4["torch_sparse"]
    
    E --> E1["scanpy"]
    E --> E2["omnipath"]
    E --> E3["decoupler"]
    
    F --> F1["bedtools"]
    
    B1 & B2 & B3 -.-> D & E & F
    C1 -.-> JAX["JAX/JAXlib"]
    C2 -.-> SCGLUE["scglue"]
```

Sources: [pyproject.toml:29-81](), [docs/installation.md:12-37]()

---