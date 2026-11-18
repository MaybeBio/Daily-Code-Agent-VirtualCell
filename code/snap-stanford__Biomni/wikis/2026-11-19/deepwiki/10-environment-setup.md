# Conda Environment

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [biomni/eval/__init__.py](biomni/eval/__init__.py)
- [biomni/eval/biomni_eval1.py](biomni/eval/biomni_eval1.py)
- [biomni/version.py](biomni/version.py)
- [biomni_env/environment.yml](biomni_env/environment.yml)

</details>



## Purpose and Scope

This document describes the `biomni_e1` Conda environment, which serves as the foundational execution environment for the Biomni platform. The environment provides a carefully curated set of dependencies spanning bioinformatics tools, scientific computing libraries, machine learning frameworks, and LLM integration packages that enable the A1 agent and associated tools to execute biomedical research tasks.

For information about installing and setting up this environment, see [Installation and Setup](#3). For details on configuring LLM providers within this environment, see [LLM Provider Setup](#3.2). For information about the system-level configuration that runs within this environment, see [Configuration System](#2.5).

## Environment Overview

The `biomni_e1` environment is defined in [biomni_env/environment.yml:1-40]() and represents the complete dependency stack required for Biomni's operation. The environment uses Python 3.11 as its foundation and organizes dependencies across multiple functional layers to support diverse biomedical computing workflows.

```mermaid
graph TB
    subgraph env["biomni_e1 Environment"]
        python["Python 3.11<br/>[environment.yml:7]"]
        pip["pip installer<br/>[environment.yml:8-9]"]
    end
    
    subgraph channels["Channel Priority"]
        cf["conda-forge<br/>Priority 1<br/>[environment.yml:3]"]
        defaults["defaults<br/>Priority 2<br/>[environment.yml:4]"]
        bioconda["bioconda<br/>Priority 3<br/>[environment.yml:5]"]
    end
    
    subgraph pipDeps["Pip Dependencies"]
        core["Core Scientific:<br/>numpy, pandas, scipy,<br/>matplotlib, seaborn<br/>[environment.yml:10-27]"]
        ml["ML/AI:<br/>scikit-learn, transformers<br/>[environment.yml:15,35]"]
        llm["LLM Integration:<br/>langchain, langgraph,<br/>openai, anthropic<br/>[environment.yml:17-23]"]
        utils["Utilities:<br/>requests, beautifulsoup4,<br/>jupyter, pytest<br/>[environment.yml:29-34]"]
        domain["Domain-Specific:<br/>gradio, mcp, tooluniverse<br/>[environment.yml:16,37-38]"]
    end
    
    env --> channels
    python --> pip
    pip --> pipDeps
    
    cf -.provides.-> python
    cf -.provides.-> core
    bioconda -.provides bioinformatics.-> domain
```

**Diagram: biomni_e1 Environment Structure and Dependency Flow**

The environment definition follows a minimalist approach, specifying only essential version constraints while allowing most packages to resolve to their latest compatible versions. This strategy balances stability with access to current features and bug fixes.

Sources: [biomni_env/environment.yml:1-40]()

## Channel Strategy

The environment uses a three-tier channel priority system defined in [biomni_env/environment.yml:2-5]():

| Priority | Channel | Purpose |
|----------|---------|---------|
| 1 | `conda-forge` | Community-maintained packages, latest updates, broad package coverage |
| 2 | `defaults` | Anaconda-maintained packages, stability-focused |
| 3 | `bioconda` | Specialized bioinformatics tools (CLI tools like samtools, BWA, MACS2) |

This ordering ensures that:
- Community packages from `conda-forge` are preferred, providing access to the most up-to-date versions
- Anaconda's curated packages serve as fallbacks when community versions are unavailable
- Bioinformatics-specific tools are sourced from the specialized `bioconda` channel

The channel strategy is particularly important for resolving conflicts between general-purpose scientific packages and domain-specific bioinformatics tools, which may have complex dependency chains involving C/C++ libraries and system tools.

Sources: [biomni_env/environment.yml:2-5]()

## Dependency Layers

### Core Python Stack

The environment is built on Python 3.11 [environment.yml:7](), which provides performance improvements and enhanced type hinting capabilities compared to earlier versions. The `pip` package manager [environment.yml:8]() is included to handle packages not available through conda channels.

```mermaid
graph TB
    python311["python=3.11<br/>[environment.yml:7]"]
    pip["pip<br/>[environment.yml:8]"]
    
    subgraph scientific["Scientific Computing Core"]
        numpy["numpy==2.1<br/>PINNED<br/>[environment.yml:10]"]
        pandas["pandas<br/>[environment.yml:11]"]
        scipy["scipy<br/>[environment.yml:13]"]
        statsmodels["statsmodels<br/>[environment.yml:14]"]
        sklearn["scikit-learn<br/>[environment.yml:15]"]
    end
    
    subgraph viz["Visualization"]
        matplotlib["matplotlib<br/>[environment.yml:12]"]
        seaborn["seaborn<br/>[environment.yml:27]"]
    end
    
    subgraph network["Network & Data"]
        networkx["networkx<br/>[environment.yml:28]"]
        requests["requests<br/>[environment.yml:29]"]
        bs4["beautifulsoup4<br/>[environment.yml:24]"]
        lxml["lxml<br/>[environment.yml:25]"]
    end
    
    python311 --> pip
    pip --> numpy
    numpy --> pandas
    numpy --> scipy
    scipy --> statsmodels
    numpy --> sklearn
    
    pandas --> matplotlib
    matplotlib --> seaborn
    
    requests --> bs4
    bs4 --> lxml
```

**Diagram: Core Python Dependencies and Their Relationships**

The scientific computing stack is anchored by **numpy==2.1** [environment.yml:10](), which is one of only two pinned packages in the environment. NumPy 2.x introduced breaking changes to the C API, making version pinning necessary to ensure compatibility with downstream packages that may not have updated their builds.

Sources: [biomni_env/environment.yml:7-15,24-29]()

### Machine Learning and LLM Integration

The environment includes comprehensive support for modern machine learning workflows and LLM-based agent systems:

```mermaid
graph TB
    subgraph ml["ML Frameworks"]
        sklearn["scikit-learn<br/>[environment.yml:15]"]
        transformers["transformers<br/>[environment.yml:35]"]
        sentencepiece["sentencepiece<br/>[environment.yml:36]"]
    end
    
    subgraph langchain["LangChain Ecosystem"]
        lc["langchain<br/>[environment.yml:17]"]
        lg["langgraph==0.3.18<br/>PINNED<br/>[environment.yml:18]"]
        lc_openai["langchain-openai<br/>[environment.yml:19]"]
        lc_anthropic["langchain-anthropic<br/>[environment.yml:20]"]
        lc_ollama["langchain-ollama<br/>[environment.yml:21]"]
        lc_community["langchain-community<br/>[environment.yml:22]"]
    end
    
    subgraph providers["LLM Provider SDKs"]
        openai["openai<br/>[environment.yml:23]"]
    end
    
    subgraph agent["Agent Infrastructure"]
        mcp["mcp<br/>Model Context Protocol<br/>[environment.yml:37]"]
        tooluniverse["tooluniverse<br/>[environment.yml:38]"]
    end
    
    transformers --> sentencepiece
    
    lc --> lg
    lc --> lc_openai
    lc --> lc_anthropic
    lc --> lc_ollama
    lc --> lc_community
    
    lc_openai --> openai
```

**Diagram: ML and LLM Integration Dependencies**

**langgraph==0.3.18** [environment.yml:18]() is the second pinned package, ensuring API stability for the agent workflow implementation. LangGraph's API evolved rapidly in earlier versions, and version 0.3.18 provides the specific node-based workflow features used by the A1 agent's generate-execute loop (see [A1 Agent Core](#2.1)).

The `mcp` package [environment.yml:37]() enables integration with external tools via the Model Context Protocol, allowing dynamic extension of agent capabilities at runtime (see [MCP Integration](#4.5)).

Sources: [biomni_env/environment.yml:15,17-23,35-38]()

### Development and Utility Tools

The environment includes essential development tools and utilities:

| Package | Purpose | Reference |
|---------|---------|-----------|
| `jupyter` | Interactive notebook interface | [environment.yml:31]() |
| `notebook` | Classic notebook server | [environment.yml:32]() |
| `ipykernel` | Jupyter kernel support | [environment.yml:33]() |
| `pytest` | Testing framework | [environment.yml:34]() |
| `gradio` | Web UI framework | [environment.yml:16]() |
| `pyyaml` | YAML configuration parsing | [environment.yml:30]() |
| `python-dotenv` | Environment variable management | [environment.yml:39]() |
| `tqdm` | Progress bars | [environment.yml:26]() |

These utilities support various interaction modes with the Biomni platform: Jupyter notebooks for interactive exploration, Gradio for web-based interfaces, pytest for development and testing, and python-dotenv for managing API keys and configuration (see [LLM Provider Setup](#3.2)).

Sources: [biomni_env/environment.yml:16,26,30-34,39]()

## Version Pinning Strategy

The environment employs a **minimal pinning strategy**, with only two packages having explicit version constraints:

```mermaid
graph LR
    subgraph pinned["Pinned Packages"]
        numpy["numpy==2.1<br/>[environment.yml:10]"]
        langgraph["langgraph==0.3.18<br/>[environment.yml:18]"]
    end
    
    subgraph rationale["Rationale"]
        numpy_reason["NumPy 2.x C API<br/>breaking changes<br/>Compatibility with<br/>scikit-learn, scipy"]
        lg_reason["LangGraph API stability<br/>Specific node/edge features<br/>Used in A1 agent workflow"]
    end
    
    subgraph floating["Floating Versions<br/>~35+ packages"]
        others["All other dependencies<br/>resolve to latest<br/>compatible versions"]
    end
    
    numpy --> numpy_reason
    langgraph --> lg_reason
```

**Diagram: Version Pinning Strategy and Rationale**

This approach offers several advantages:

1. **Latest Features**: Unpinned packages automatically receive updates with new features and bug fixes
2. **Security Patches**: Security vulnerabilities in dependencies are addressed through regular updates
3. **Reduced Conflicts**: Fewer version constraints reduce the likelihood of dependency resolution failures
4. **Maintenance Burden**: Minimal version specifications reduce the need for frequent environment updates

The two pinned versions address specific stability requirements:
- **numpy==2.1**: NumPy 2.0 introduced breaking changes to the C API that affected binary compatibility with packages built against NumPy 1.x. Pinning to 2.1 ensures all numeric computing packages use compatible builds.
- **langgraph==0.3.18**: The A1 agent's workflow implementation (see [A1 Agent Core](#2.1)) relies on specific LangGraph features for node-based execution graphs and state management that were stabilized in version 0.3.18.

Sources: [biomni_env/environment.yml:10,18]()

## Extended Dependencies

While the core environment definition in [environment.yml]() specifies the foundational packages, the complete Biomni ecosystem includes additional dependencies installed through various mechanisms:

```mermaid
graph TB
    subgraph yml["environment.yml<br/>Core Definition"]
        core_pkg["~40 pip packages<br/>Scientific, ML, LLM"]
    end
    
    subgraph extended["Extended Installation"]
        bioinf["Bioinformatics CLI Tools<br/>samtools, BWA, bowtie2,<br/>bedtools, MACS2, FastQC<br/>Installed via conda/bioconda"]
        
        omics["Single-Cell & Omics<br/>scanpy, anndata, scvi-tools,<br/>squidpy, muon, spatialdata<br/>Installed via pip"]
        
        dl["Deep Learning<br/>PyTorch, JAX<br/>Installed separately<br/>per compute environment"]
        
        spatial["Spatial Data<br/>pyarrow, zarr, h5py,<br/>dask, s3fs<br/>Installed via pip"]
    end
    
    subgraph tools["Tool Dependencies"]
        genomics["Genomics Tools<br/>Use: scanpy, pysam,<br/>biopython"]
        database["Database Tools<br/>Use: requests, pandas"]
        lab["Lab Automation<br/>Use: PyLabRobot<br/>external dependency"]
    end
    
    yml --> extended
    
    omics --> genomics
    bioinf --> genomics
    core_pkg --> database
    lab -.external.-> extended
```

**Diagram: Extended Dependency Ecosystem**

These extended dependencies are installed through installation scripts that supplement the base environment:

- **Bioinformatics CLI tools** are installed via conda from the `bioconda` channel, providing command-line utilities like `samtools`, `BWA`, `bowtie2`, `bedtools`, `MACS2`, and `FastQC`. These tools are invoked by genomics-related functions (see [Genomics and Single-Cell Tools](#5.2)).

- **Single-cell and omics packages** (`scanpy`, `anndata`, `scvi-tools`, `squidpy`, `muon`, `spatialdata`) are installed via pip and provide specialized analysis capabilities for single-cell RNA-seq, spatial transcriptomics, and multi-omics data integration.

- **Deep learning frameworks** (PyTorch, JAX) may be installed separately based on compute requirements (CPU vs GPU) and are not included in the base environment to avoid unnecessary dependencies.

- **Spatial data libraries** (`pyarrow`, `zarr`, `h5py`, `dask`, `s3fs`) enable efficient handling of large-scale biological datasets stored in chunked formats or cloud storage (see [Data Infrastructure](#2.6)).

Sources: [biomni_env/environment.yml:1-40]() (base definition), inferred from high-level architecture diagrams

## Tool Integration Dependencies

Different tool categories within Biomni have specific dependency requirements that map to packages in the environment:

| Tool Category | Key Dependencies | Usage Example |
|--------------|------------------|---------------|
| Database Query Tools | `requests`, `pandas`, `beautifulsoup4` | REST API queries to UniProt, PDB, KEGG (see [Database Query Tools](#5.1)) |
| Genomics Tools | `scanpy`, `anndata`, `pysam`, bioinformatics CLI | Single-cell analysis, variant calling, ChIP-seq (see [Genomics and Single-Cell Tools](#5.2)) |
| Lab Automation | `PyLabRobot` (external) | Liquid handling script generation (see [Lab Automation Tools](#5.3)) |
| Embedding Generation | `transformers`, `sentencepiece` | ESM protein embeddings, UCE cell embeddings |
| Graph Analysis | `networkx` | Protein-protein interaction networks, pathway analysis |

```mermaid
graph TB
    subgraph tools["Tool Categories"]
        db_tools["Database Tools<br/>~30 APIs"]
        genomics_tools["Genomics Tools<br/>scRNA-seq, variants"]
        lab_tools["Lab Automation<br/>PyLabRobot"]
        embed_tools["Embedding Tools<br/>ESM, UCE"]
        graph_tools["Graph Analysis<br/>PPI networks"]
    end
    
    subgraph deps["Environment Dependencies"]
        requests["requests<br/>[environment.yml:29]"]
        pandas["pandas<br/>[environment.yml:11]"]
        bs4["beautifulsoup4<br/>[environment.yml:24]"]
        
        scanpy["scanpy<br/>extended install"]
        anndata["anndata<br/>extended install"]
        samtools["samtools<br/>bioconda"]
        
        pylabr["PyLabRobot<br/>external"]
        
        transformers["transformers<br/>[environment.yml:35]"]
        sentencepiece["sentencepiece<br/>[environment.yml:36]"]
        
        networkx["networkx<br/>[environment.yml:28]"]
    end
    
    db_tools --> requests
    db_tools --> pandas
    db_tools --> bs4
    
    genomics_tools --> scanpy
    genomics_tools --> anndata
    genomics_tools --> samtools
    
    lab_tools --> pylabr
    
    embed_tools --> transformers
    embed_tools --> sentencepiece
    
    graph_tools --> networkx
```

**Diagram: Tool Category Dependencies Mapped to Environment Packages**

The modular dependency structure allows tools to be selectively loaded based on available packages, with graceful degradation when optional dependencies are missing. For example, deep learning-based embedding tools check for PyTorch availability before attempting to load models.

Sources: [biomni_env/environment.yml:11,24,28-29,35-36](), inferred from tool architecture

## Environment Management

### Creating the Environment

The environment is created using the standard conda workflow:

```bash
conda env create -f biomni_env/environment.yml
```

This command:
1. Reads the environment specification from [environment.yml]()
2. Resolves dependencies across the three configured channels (conda-forge, defaults, bioconda)
3. Installs Python 3.11 and pip via conda
4. Uses pip to install the specified Python packages

### Activating and Using the Environment

Once created, activate the environment for all Biomni operations:

```bash
conda activate biomni_e1
```

The A1 agent and all tools expect to run within this environment. Code execution nodes in the agent workflow (see [Execution Pipeline](#2.2)) inherit the activated environment's Python interpreter and installed packages.

### Updating Dependencies

To update unpinned dependencies to their latest compatible versions:

```bash
conda activate biomni_e1
pip install --upgrade pandas scipy scikit-learn matplotlib seaborn \
    langchain langchain-openai langchain-anthropic \
    transformers requests beautifulsoup4 gradio jupyter
```

Note that `numpy` and `langgraph` should **not** be upgraded without testing, as they have pinned versions for stability reasons.

### Verifying the Installation

After environment creation, verify that key packages are available:

```python
import numpy as np
import pandas as pd
import langchain
import langgraph
from transformers import AutoModel

print(f"NumPy: {np.__version__}")        # Should be 2.1.x
print(f"LangGraph: {langgraph.__version__}")  # Should be 0.3.18
```

Sources: [biomni_env/environment.yml:1-40]()

## Relationship to System Components

The conda environment serves as the execution foundation for multiple system components:

```mermaid
graph TB
    subgraph env["biomni_e1 Environment"]
        python["Python 3.11<br/>Runtime"]
        packages["Installed Packages<br/>~40 base + extended"]
    end
    
    subgraph agents["Agent Layer"]
        a1["A1 Agent<br/>Executes in env"]
        react["ReAct Agent<br/>Executes in env"]
        paper["PaperTaskExtractor<br/>Executes in env"]
    end
    
    subgraph execution["Code Execution"]
        py_exec["run_python_repl<br/>Uses env interpreter"]
        r_exec["run_r_code<br/>Requires R in env"]
        bash_exec["run_bash_script<br/>Accesses CLI tools"]
    end
    
    subgraph tools["Tool Ecosystem"]
        db["Database Tools<br/>requests, pandas"]
        genomics["Genomics Tools<br/>scanpy, anndata"]
        lab["Lab Tools<br/>PyLabRobot"]
    end
    
    subgraph llm["LLM Integration"]
        providers["LLM Providers<br/>langchain integrations"]
        workflow["LangGraph Workflows<br/>A1 generate-execute loop"]
    end
    
    env --> agents
    env --> execution
    env --> tools
    env --> llm
    
    agents --> execution
    execution --> tools
    agents --> llm
```

**Diagram: Environment's Role in System Architecture**

Key integration points:

1. **Agent Execution**: The A1 agent (see [A1 Agent Core](#2.1)) runs within the environment, accessing `langchain` and `langgraph` for workflow orchestration
2. **Code Execution Pipeline**: The execution nodes (see [Execution Pipeline](#2.2)) use the environment's Python interpreter for `run_python_repl`, and access CLI tools like `samtools` for `run_bash_script`
3. **Tool Registry**: Tools registered in the ToolRegistry (see [Tool Registry and Retrieval](#2.4)) import from environment packages
4. **LLM Integration**: The `get_llm` function (see [LLM Integration Layer](#2.3)) uses provider-specific packages like `langchain-openai` and `langchain-anthropic`

The environment's minimal pinning strategy allows the system to remain current with ecosystem improvements while maintaining stability for core agent functionality.

Sources: [biomni_env/environment.yml:1-40](), inferred from system architecture

---