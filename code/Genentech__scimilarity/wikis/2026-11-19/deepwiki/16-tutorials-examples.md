# Tutorials & Examples

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/tutorials.rst](docs/tutorials.rst)

</details>



## Purpose and Scope

This page provides an overview of SCimilarity's interactive tutorials and practical examples that demonstrate the system's core functionality. The tutorials cover the complete workflow from basic cell searches to advanced model training and interpretation. Each tutorial is implemented as a Jupyter notebook with step-by-step instructions and working code examples.

For detailed API documentation of the classes used in these tutorials, see [Core API Reference](#3). For information about training custom models beyond the tutorial scope, see [Model Training](#4).

## Tutorial Architecture Overview

SCimilarity's tutorials are organized into distinct learning paths that correspond to the main system components and user workflows. The tutorial structure follows a progression from basic similarity search operations to advanced model training and interpretation.

### Tutorial Workflow Mapping

```mermaid
graph TB
    subgraph "Tutorial Categories"
        BASIC["Cell Search Tutorials<br/>• Individual cells<br/>• Gene signatures<br/>• Cluster centroids"]
        ANNOT["Cell Annotation Tutorial<br/>• Reference loading<br/>• Type prediction<br/>• Result filtering"]
        ADV["Advanced Tutorial<br/>• Gene attribution<br/>• Model interpretation<br/>• Comparison analysis"]
        TRAIN["Training Tutorial<br/>• Data preparation<br/>• Model training<br/>• Index building"]
    end
    
    subgraph "Core Classes Used"
        CE["CellEmbedding<br/>notebooks/cell_search_tutorial_1"]
        CQ["CellQuery<br/>notebooks/cell_search_tutorial_2<br/>notebooks/cell_search_tutorial_3"]
        CA["CellAnnotation<br/>notebooks/cell_annotation_tutorial"]
        INT["Interpreter<br/>notebooks/advanced_tutorial"]
        ML["MetricLearning<br/>notebooks/training_tutorial"]
    end
    
    subgraph "Data Components"
        TDB["TileDB Arrays<br/>CellArr storage"]
        KNN["HNSW Indices<br/>kNN search"]
        META["Cell Metadata<br/>Annotation data"]
        MODELS["Pretrained Models<br/>Zenodo assets"]
    end
    
    BASIC --> CE
    BASIC --> CQ
    ANNOT --> CA
    ADV --> INT
    TRAIN --> ML
    
    CE --> TDB
    CQ --> KNN
    CA --> META
    INT --> MODELS
    ML --> TDB
    
    style BASIC fill:#f0f0f0
    style ANNOT fill:#f0f0f0
    style ADV fill:#f0f0f0
    style TRAIN fill:#f0f0f0
```

### Learning Path Structure

```mermaid
flowchart TD
    START["Getting Started<br/>Installation & Setup"]
    
    subgraph "Basic Tutorials"
        T1["cell_search_tutorial_1<br/>Individual Cell Search<br/>• CellEmbedding basics<br/>• Single cell queries"]
        T2["cell_search_tutorial_2<br/>Gene Signature Search<br/>• CellQuery usage<br/>• Gene expression patterns"]
        T3["cell_search_tutorial_3<br/>Cluster-based Search<br/>• Centroid computation<br/>• Batch processing"]
    end
    
    subgraph "Application Tutorials"
        T4["cell_annotation_tutorial<br/>Cell Type Annotation<br/>• CellAnnotation class<br/>• Reference datasets<br/>• Prediction workflows"]
    end
    
    subgraph "Advanced Tutorials"
        T5["advanced_tutorial<br/>Model Interpretation<br/>• Gene attribution<br/>• Comparison analysis<br/>• Interpreter class"]
        T6["training_tutorial<br/>Custom Model Training<br/>• MetricLearning model<br/>• TileDB integration<br/>• Index building"]
    end
    
    START --> T1
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> T5
    T4 --> T6
    
    style START fill:#e1f5fe
    style T1 fill:#f3e5f5
    style T4 fill:#fff3e0
    style T5 fill:#f0f0f0
    style T6 fill:#f0f0f0
```

Sources: [docs/tutorials.rst:1-14]()

## Tutorial Categories

### Cell Search Tutorials

The cell search tutorials demonstrate SCimilarity's core similarity search capabilities using different query strategies:

| Tutorial | Notebook File | Primary Classes | Focus Area |
|----------|---------------|-----------------|------------|
| Individual Cell Search | `cell_search_tutorial_1` | `CellEmbedding` | Basic embedding generation and single cell queries |
| Gene Signature Search | `cell_search_tutorial_2` | `CellQuery` | Expression pattern-based searches and metadata compilation |
| Cluster-based Search | `cell_search_tutorial_3` | `CellQuery` | Centroid computation and batch query processing |

These tutorials progressively build understanding of the search system, starting with basic `CellEmbedding` operations and advancing to complex `CellQuery` workflows. For comprehensive documentation of these tutorials, see [Cell Search Tutorials](#8.1).

### Cell Annotation Tutorial

The cell annotation tutorial (`cell_annotation_tutorial`) demonstrates automated cell type prediction using the `CellAnnotation` class. This tutorial covers:

- Loading reference datasets and metadata
- Configuring annotation parameters and filters
- Running prediction workflows with consensus voting
- Interpreting annotation results and confidence scores

For detailed documentation of the annotation workflow, see [Cell Annotation Tutorial](#8.2).

### Advanced Features Tutorial

The advanced tutorial (`advanced_tutorial`) explores model interpretation capabilities using the `Interpreter` class. This tutorial demonstrates:

- Gene attribution analysis for understanding model decisions
- Comparative analysis across different cell types and conditions
- Visualization techniques for interpretation results
- Integration with ontology systems for hierarchical analysis

### Model Training Tutorial

The training tutorial (`training_tutorial`) provides a complete walkthrough of training custom SCimilarity models using the `MetricLearning` framework. This tutorial covers:

- Data preparation with `CellArr` and TileDB integration
- Model configuration and hyperparameter settings
- Training loop implementation with PyTorch Lightning
- Post-training index building and validation

Sources: [docs/tutorials.rst:8-13]()

## Tutorial Execution Environment

### Notebook Structure

All tutorials are implemented as Jupyter notebooks located in the `notebooks/` directory. Each notebook follows a consistent structure:

1. **Environment Setup**: Import statements and configuration
2. **Data Loading**: Loading pretrained models and reference datasets  
3. **Core Workflow**: Step-by-step demonstration of functionality
4. **Results Analysis**: Interpretation and visualization of outputs
5. **Next Steps**: Links to related tutorials and advanced topics

### Prerequisites

The tutorials assume basic familiarity with:
- Python data science stack (NumPy, Pandas, matplotlib)
- Single-cell RNA sequencing concepts
- Jupyter notebook environment

### Data Requirements

Most tutorials use publicly available datasets and pretrained models distributed via Zenodo. The notebooks include automated download and caching mechanisms for required data assets.

## Getting Started Guide

### Tutorial Progression

For new users, the recommended tutorial progression is:

1. **Start with Installation**: Complete setup as described in [Installation & Setup](#2)
2. **Basic Search**: Begin with `cell_search_tutorial_1` to understand `CellEmbedding`
3. **Advanced Search**: Progress through `cell_search_tutorial_2` and `cell_search_tutorial_3`
4. **Annotation Workflow**: Complete `cell_annotation_tutorial` for classification tasks
5. **Interpretation**: Explore `advanced_tutorial` for model understanding
6. **Custom Training**: Attempt `training_tutorial` for advanced use cases

### Running Tutorials

```bash
# Clone repository and install dependencies
git clone https://github.com/Genentech/scimilarity
cd scimilarity
pip install -e .

# Launch Jupyter and navigate to notebooks/
jupyter lab notebooks/
```

Each tutorial notebook includes detailed instructions for data download and environment configuration specific to that workflow.

Sources: [docs/tutorials.rst:1-14]()

---