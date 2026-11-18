# Cell Annotation

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/scimilarity/cell_annotation.py](src/scimilarity/cell_annotation.py)

</details>



## Purpose and Scope

This document covers the `CellAnnotation` class and its associated workflows for automated cell type prediction and classification in SCimilarity. The `CellAnnotation` system enables researchers to predict cell types for new single-cell datasets using pre-trained reference models and nearest neighbor search algorithms.

For information about similarity search workflows without cell type prediction, see [Cell Query System](#3.2). For details about the underlying embedding and search infrastructure, see [Cell Embedding & Search Infrastructure](#3.1).

## System Overview

The `CellAnnotation` class extends the `CellSearchKNN` base class to provide specialized functionality for cell type prediction. It combines embedding generation, nearest neighbor search, and consensus voting to assign cell type labels to query cells.

### CellAnnotation Architecture

```mermaid
graph TB
    subgraph "CellAnnotation System"
        CA["CellAnnotation"]
        CSK["CellSearchKNN (Base)"]
        
        subgraph "Core Components"
            EMBED["get_embeddings()"]
            KNN["get_nearest_neighbors()"]
            PRED["get_predictions_knn()"]
            ANNOT["annotate_dataset()"]
        end
        
        subgraph "Filtering System"
            RESET["reset_knn()"]
            BLOCK["blocklist_celltypes()"]
            SAFE["safelist_celltypes()"]
        end
        
        subgraph "Data Sources"
            KNNBIN["labelled_kNN.bin"]
            LABELS["reference_labels.tsv"]
            MODEL["model files"]
        end
        
        subgraph "Label Mappings"
            IDX2LABEL["idx2label"]
            IDX2STUDY["idx2study"]
            CLASSES["classes property"]
        end
    end
    
    CA --> CSK
    CSK --> EMBED
    CSK --> KNN
    CA --> PRED
    CA --> ANNOT
    
    CA --> RESET
    CA --> BLOCK
    CA --> SAFE
    
    KNNBIN --> CA
    LABELS --> IDX2LABEL
    LABELS --> IDX2STUDY
    MODEL --> CSK
    
    IDX2LABEL --> CLASSES
    IDX2LABEL --> PRED
    IDX2STUDY --> PRED
    
    BLOCK --> KNNBIN
    SAFE --> KNNBIN
```

**Sources:** [src/scimilarity/cell_annotation.py:1-320]()

## Core Functionality

### Initialization and Setup

The `CellAnnotation` class initializes by loading pre-built kNN indices and reference cell type labels. The initialization process sets up the annotation-specific file paths and data structures needed for prediction.

```mermaid
graph LR
    subgraph "Initialization Flow"
        INIT["__init__()"]
        SUPER["super().__init__()"]
        PATHS["annotation_path setup"]
        FILES["filenames configuration"]
        LOAD["load_knn_index()"]
        PARSE["parse reference_labels.tsv"]
        MAP["build idx2label/idx2study"]
    end
    
    INIT --> SUPER
    SUPER --> PATHS
    PATHS --> FILES
    FILES --> LOAD
    LOAD --> PARSE
    PARSE --> MAP
```

Key configuration elements:
- `annotation_path`: Directory containing annotation-specific files ([src/scimilarity/cell_annotation.py:37-38]())
- `knn` filename: Binary file containing the labeled kNN index ([src/scimilarity/cell_annotation.py:43-45]())
- `celltype_labels` filename: TSV file mapping indices to cell type labels ([src/scimilarity/cell_annotation.py:46-49]())

**Sources:** [src/scimilarity/cell_annotation.py:23-64]()

### Cell Type Filtering System

The annotation system provides flexible filtering mechanisms to control which cell types can be predicted through blocklists and safelists.

#### Blocklist Functionality

The `blocklist_celltypes()` method prevents specific cell types from being predicted by marking corresponding indices as deleted in the kNN index.

```mermaid
graph TB
    subgraph "Blocklist Workflow"
        INPUT["blocklist_celltypes(labels)"]
        RESET["reset_knn()"]
        SET["self.blocklist = set(labels)"]
        CLEAR["self.safelist = None"]
        ITER["iterate idx2label.items()"]
        CHECK["celltype_name in blocklist?"]
        MARK["knn.mark_deleted(i)"]
    end
    
    INPUT --> RESET
    RESET --> SET
    SET --> CLEAR
    CLEAR --> ITER
    ITER --> CHECK
    CHECK -->|Yes| MARK
    CHECK -->|No| ITER
```

#### Safelist Functionality

The `safelist_celltypes()` method restricts predictions to only specified cell types by marking all indices as deleted, then unmarking only the safelisted ones.

```mermaid
graph TB
    subgraph "Safelist Workflow"
        INPUT["safelist_celltypes(labels)"]
        CLEAR1["self.blocklist = None"]
        SET["self.safelist = set(labels)"]
        MARKALL["mark all indices deleted"]
        ITER["iterate idx2label.items()"]
        CHECK["celltype_name in safelist?"]
        UNMARK["knn.unmark_deleted(i)"]
    end
    
    INPUT --> CLEAR1
    CLEAR1 --> SET
    SET --> MARKALL
    MARKALL --> ITER
    ITER --> CHECK
    CHECK -->|Yes| UNMARK
    CHECK -->|No| ITER
```

**Sources:** [src/scimilarity/cell_annotation.py:92-147]()

### Prediction Engine

The core prediction functionality is implemented in `get_predictions_knn()`, which performs nearest neighbor search and consensus voting to assign cell type labels.

#### Prediction Workflow

```mermaid
graph TB
    subgraph "Prediction Process"
        EMBED["embeddings input"]
        NN["get_nearest_neighbors()"]
        VOTE["consensus voting"]
        STATS["compute statistics"]
        RETURN["return predictions + metadata"]
        
        subgraph "Consensus Voting"
            COUNT["count celltype occurrences"]
            WEIGHT["apply distance weighting"]
            MAX["select max occurrence"]
        end
        
        subgraph "Statistics Computation"
            HITS["hits (unweighted)"]
            HITSWT["hits_weighted"]
            MINDIST["min_dist"]
            MAXDIST["max_dist"]
            VS2ND["vs2nd ratio"]
            VSALL["vsAll ratio"]
        end
    end
    
    EMBED --> NN
    NN --> VOTE
    VOTE --> COUNT
    COUNT --> WEIGHT
    WEIGHT --> MAX
    MAX --> STATS
    STATS --> HITS
    STATS --> HITSWT
    STATS --> MINDIST
    STATS --> MAXDIST
    STATS --> VS2ND
    STATS --> VSALL
    STATS --> RETURN
```

#### Voting Mechanisms

The system supports two voting strategies:
- **Unweighted**: Simple count of cell type occurrences among k nearest neighbors
- **Distance-weighted**: Inverse distance weighting where closer neighbors have higher influence

**Sources:** [src/scimilarity/cell_annotation.py:148-279]()

### Dataset Annotation

The `annotate_dataset()` method provides a high-level interface for annotating entire AnnData objects with cell type predictions and embedding information.

```mermaid
graph LR
    subgraph "Dataset Annotation Flow"
        DATA["AnnData input"]
        ALIGN["align_dataset()"]
        EMBED["get_embeddings()"]
        PRED["get_predictions_knn()"]
        STORE["store results in obs/obsm"]
    end
    
    DATA --> ALIGN
    ALIGN --> EMBED
    EMBED --> PRED
    PRED --> STORE
    
    subgraph "Output Annotations"
        OBSM["obsm['X_scimilarity']"]
        HINT["obs['celltype_hint']"]
        DIST["obs['min_dist']"]
        HITS["obs['celltype_hits']"]
        STAT["obs['celltype_hint_stat']"]
    end
    
    STORE --> OBSM
    STORE --> HINT
    STORE --> DIST
    STORE --> HITS
    STORE --> STAT
```

**Sources:** [src/scimilarity/cell_annotation.py:281-319]()

## Data Structures and File Format

### Reference Label Format

The `reference_labels.tsv` file contains the mapping between kNN index positions and cell type labels:

| Column | Description | Required |
|--------|-------------|----------|
| Cell Type | Human-readable cell type label | Yes |
| Study | Source study identifier | Optional |

### Label Mapping Dictionaries

The class maintains two key mapping structures:
- `idx2label`: Maps integer indices to cell type strings ([src/scimilarity/cell_annotation.py:55]())
- `idx2study`: Maps integer indices to study identifiers ([src/scimilarity/cell_annotation.py:56]())

### Prediction Statistics

The prediction statistics returned by `get_predictions_knn()` include:

| Field | Type | Description |
|-------|------|-------------|
| `hits` | JSON string | Unweighted vote counts per cell type |
| `hits_weighted` | JSON string | Distance-weighted vote counts |
| `min_dist` | float | Minimum distance to nearest neighbors |
| `max_dist` | float | Maximum distance to nearest neighbors |
| `vs2nd` | float | Best vs second-best ratio (unweighted) |
| `vsAll` | float | Best vs all votes ratio (unweighted) |
| `vs2nd_weighted` | float | Best vs second-best ratio (weighted) |
| `vsAll_weighted` | float | Best vs all votes ratio (weighted) |

**Sources:** [src/scimilarity/cell_annotation.py:181-189](), [src/scimilarity/cell_annotation.py:215-274]()

## Integration with SCimilarity Ecosystem

The `CellAnnotation` class integrates with other SCimilarity components through inheritance and shared interfaces:

```mermaid
graph TB
    subgraph "Class Hierarchy"
        CSK["CellSearchKNN"]
        CA["CellAnnotation"]
        CE["CellEmbedding"]
        CQ["CellQuery"]
    end
    
    subgraph "Shared Utilities"
        ALIGN["align_dataset()"]
        LOGNORM["lognorm_counts()"]
        UTILS["data processing utils"]
    end
    
    subgraph "Data Infrastructure"
        TDB["TileDB storage"]
        HNSW["HNSW indices"]
        MODELS["trained models"]
    end
    
    CSK --> CA
    CSK --> CE
    CSK --> CQ
    
    CA --> ALIGN
    CA --> LOGNORM
    CE --> UTILS
    
    CA --> TDB
    CA --> HNSW
    CA --> MODELS
```

**Sources:** [src/scimilarity/cell_annotation.py:3](), [src/scimilarity/cell_annotation.py:306]()

---