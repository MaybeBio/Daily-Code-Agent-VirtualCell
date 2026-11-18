# User API

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/api/developer.md](docs/api/developer.md)
- [docs/api/index.md](docs/api/index.md)
- [docs/api/user.md](docs/api/user.md)
- [docs/tutorials/index.md](docs/tutorials/index.md)

</details>



## Purpose and Scope

The User API provides the public interfaces for researchers and analysts to use NicheCompass for spatial multi-omics data analysis without needing to understand the internal implementation details. This documentation covers the user-facing functions and classes that most users will interact with directly. For information about lower-level components and implementation details, see [Developer API](#6.2).

## User API Overview

The NicheCompass User API consists of several key components organized by functionality:

```mermaid
flowchart TD
    A["User API"] --> B["NicheCompass Model"]
    A --> C["Data Loading"]
    A --> D["Gene Program Utilities"]
    A --> E["Cell-Cell Communication"]
    A --> F["Benchmarking"]
    A --> G["Other Utilities"]
    
    B --> B1["Model Configuration"]
    B --> B2["Training"]
    B --> B3["Analysis Methods"]
    
    C --> C1["load_spatial_adata_from_csv"]
    
    D --> D1["GP Extraction"]
    D --> D2["GP Filtering"]
    D --> D3["GP Addition to AnnData"]
    D --> D4["Multimodal Integration"]
    
    E --> E1["compute_communication_gp_network"]
    E --> E2["visualize_communication_gp_network"]
    
    F --> F1["compute_benchmarking_metrics"]
    F --> F2["Individual Metrics (CLISIS, CAS, etc.)"]
```

Sources: [docs/api/user.md:1-88]()

## NicheCompass Model

The core of the User API is the `NicheCompass` class, which provides a high-level interface for model configuration, training, and analysis.

```mermaid
classDiagram
    class NicheCompass {
        +__init__(hyperparameters)
        +train(adata)
        +fit(adata)
        +get_latent_representation()
        +analyze_niches()
        +analyze_cell_communication()
        +save_model()
        +load_model()
    }
```

The `NicheCompass` class serves as the primary entry point for:
- Configuring model parameters
- Training on spatial multi-omics data
- Extracting latent representations
- Performing downstream analyses such as niche identification and cell-cell communication inference

Sources: [docs/api/user.md:32-41]()

## Data Loading Utilities

NicheCompass provides a utility function for loading spatial data from CSV files:

```mermaid
flowchart LR
    A["CSV Files"] --> B["load_spatial_adata_from_csv()"]
    B --> C["AnnData Object"]
    C --> D["Ready for NicheCompass"]
```

The `load_spatial_adata_from_csv` function processes spatial coordinates and gene expression data to create an AnnData object properly structured for use with NicheCompass.

Sources: [docs/api/user.md:20-29]()

## Gene Program Utilities

Gene Program (GP) utilities form a significant part of the User API, allowing users to extract biological knowledge from various databases, process it, and incorporate it into the analysis workflow.

```mermaid
flowchart TD
    A["External Databases"] --> B["Extraction Functions"]
    B --> B1["extract_gp_dict_from_omnipath_lr_interactions()"]
    B --> B2["extract_gp_dict_from_nichenet_lrt_interactions()"]
    B --> B3["extract_gp_dict_from_mebocost_es_interactions()"]
    B --> B4["extract_gp_dict_from_collectri_tf_network()"]
    
    B1 & B2 & B3 & B4 --> C["GP Dictionaries"]
    C --> D["filter_and_combine_gp_dict_gps()"]
    D --> E["Filtered GP Dictionary"]
    E --> F["add_gps_from_gp_dict_to_adata()"]
    F --> G["AnnData with GP Masks"]
    G --> H["NicheCompass Model"]
    
    I["ATAC-seq Data"] --> J["generate_multimodal_mapping_dict()"]
    J --> K["Multimodal Mapping"]
    K --> L["add_multimodal_mask_to_adata()"]
    L --> G
```

The Gene Program utilities provide:
- Functions to extract gene programs from different biological interaction databases
- Tools to filter and combine gene programs
- Methods to add gene program masks to AnnData objects
- Support for multimodal integration (RNA-seq and ATAC-seq)

Sources: [docs/api/user.md:43-63]()

## Cell-Cell Communication Utilities

NicheCompass provides tools for analyzing and visualizing cell-cell communication based on the model's latent representation:

```mermaid
flowchart LR
    A["Trained NicheCompass Model"] --> B["get_latent_representation()"]
    B --> C["Latent Space (Cells × GPs)"]
    C --> D["compute_communication_gp_network()"]
    D --> E["Communication Network"]
    E --> F["visualize_communication_gp_network()"]
    F --> G["Network Visualization"]
```

These utilities help analyze how different cell types communicate with each other through various gene programs, providing insights into spatial cellular interactions.

Sources: [docs/api/user.md:65-76]()

## Benchmarking Utilities

The User API includes several benchmarking functions to evaluate model performance:

```mermaid
flowchart TD
    A["Trained NicheCompass Model"] --> B["compute_benchmarking_metrics()"]
    B --> C["Comprehensive Evaluation"]
    
    A --> D["Individual Metrics"]
    D --> D1["compute_clisis()"]
    D --> D2["compute_cas()"]
    D --> D3["compute_gcs()"]
    D --> D4["compute_mlami()"]
    D --> D5["compute_nasw()"]
    
    subgraph "Metric Meanings"
        M1["CLISIS: Cell-Level Interpretable Spatial Integration Score"]
        M2["CAS: Cell type Adjacency Score"]
        M3["GCS: Graph Connectivity Score"]
        M4["MLAMI: Multi-Level Adjusted Mutual Information"]
        M5["NASW: Neighborhood Preservation"]
    end
```

The benchmarking utilities provide:
- A comprehensive function to compute multiple metrics at once
- Individual functions for specific metrics
- Tools to evaluate spatial properties, cell type distribution, and neighborhood preservation

Sources: [docs/api/user.md:3-18]()

## Other Utilities

The User API also includes additional utility functions:

```mermaid
flowchart LR
    A["Other Utilities"] --> B["create_new_color_dict()"]
    B --> C["Custom Color Dictionary"]
    C --> D["Use in Visualizations"]
```

These additional utilities support visualization and other ancillary tasks when using NicheCompass.

Sources: [docs/api/user.md:78-87]()

## Code Organization and Imports

The NicheCompass User API is organized into several modules that users can import as needed:

```mermaid
flowchart TD
    A["import nichecompass"] --> B["nichecompass.models"]
    A --> C["nichecompass.data"]
    A --> D["nichecompass.utils"]
    A --> E["nichecompass.benchmarking"]
    
    B --> B1["NicheCompass"]
    
    C --> C1["load_spatial_adata_from_csv"]
    
    D --> D1["Gene Program Utilities"]
    D --> D2["Cell-Cell Communication Utilities"]
    D --> D3["Other Utilities"]
    
    E --> E1["Benchmarking Functions"]
```

Users will typically import specific functions as needed, for example:

- `from nichecompass.models import NicheCompass`
- `from nichecompass.data import load_spatial_adata_from_csv`
- `from nichecompass.utils import extract_gp_dict_from_omnipath_lr_interactions`
- `from nichecompass.benchmarking import compute_clisis`

Sources: [docs/api/index.md:1-14](), [docs/api/user.md:1-88]()

## Typical Analysis Workflow

A typical analysis workflow using the NicheCompass User API follows these steps:

```mermaid
flowchart TD
    A["1. Data Loading"] --> B["2. GP Extraction"]
    B --> C["3. GP Processing"]
    C --> D["4. Add GPs to AnnData"]
    D --> E["5. Initialize NicheCompass Model"]
    E --> F["6. Train Model"]
    F --> G["7. Analyze Results"]
    G --> H["8. Benchmark Performance"]
    
    G --> G1["Niche Identification"]
    G --> G2["Cell-Cell Communication"]
    G --> G3["Generate Visualizations"]
```

For more detailed examples of using these APIs in specific analysis scenarios, refer to the [Tutorials](#4) section.

Sources: [docs/tutorials/index.md:1-12]()

---