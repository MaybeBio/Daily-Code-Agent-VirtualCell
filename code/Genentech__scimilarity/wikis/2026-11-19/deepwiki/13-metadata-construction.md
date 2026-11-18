# Metadata Construction

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [scripts/build_cellsearch_metadata.py](scripts/build_cellsearch_metadata.py)

</details>



This document covers the construction of comprehensive cell metadata structures for the SCimilarity system. The metadata construction process occurs after model training and embedding generation, creating enriched metadata that enables efficient cell querying and search operations.

For information about building search indices, see [Building Search Indices](#6.1). For details about the training workflows that precede metadata construction, see [Model Training](#4).

## Purpose and Scope

The metadata construction system creates a unified metadata structure that combines:

- Original cell metadata from TileDB storage
- Cell type ontology mappings
- Model predictions and confidence scores
- Training/validation dataset annotations
- Nearest neighbor distances for quality assessment

This enriched metadata enables the `CellQuery` and `CellAnnotation` classes to perform efficient filtering, searching, and result compilation.

## Architecture Overview

```mermaid
flowchart TD
    subgraph "Input Data Sources"
        CELLARR["cell_metadata TileDB Array"]
        EMBEDDINGS["cell_embedding TileDB Array"]
        TRAINCSV["train_cells.csv.gz"]
        VALCSV["val_cells.csv.gz"]
        ONTO["Cell Ontology"]
        SAFELIST["safelist_file (optional)"]
    end
    
    subgraph "Processing Components"
        SCRIPT["build_cellsearch_metadata.py"]
        CELLANN["CellAnnotation Model"]
        ONTOMAPPER["get_id_mapper()"]
        BATCHPROC["Batch Processing Loop"]
    end
    
    subgraph "Output Structure"
        METAMETA["Enriched cell_metadata TileDB"]
        SEARCHPATH["cellsearch/ directory"]
    end
    
    CELLARR --> SCRIPT
    EMBEDDINGS --> SCRIPT
    TRAINCSV --> SCRIPT
    VALCSV --> SCRIPT
    ONTO --> ONTOMAPPER
    SAFELIST --> CELLANN
    
    SCRIPT --> CELLANN
    SCRIPT --> ONTOMAPPER
    SCRIPT --> BATCHPROC
    
    ONTOMAPPER --> METAMETA
    CELLANN --> METAMETA
    BATCHPROC --> METAMETA
    
    METAMETA --> SEARCHPATH
    
    style SCRIPT fill:#e1f5fe
    style CELLANN fill:#f3e5f5
    style METAMETA fill:#e8f5e8
```

**Title:** Metadata Construction Data Flow

Sources: [scripts/build_cellsearch_metadata.py:1-123]()

## Data Sources and Input Processing

### Cell Metadata Loading

The system loads base cell metadata from the TileDB `cell_metadata` array, performing column standardization to ensure consistent naming conventions:

| Original Column | Standardized Name | Purpose |
|----------------|-------------------|---------|
| `datasetID` | `study` | Study identifier |
| `sampleID` | `sample` | Sample identifier |
| `tissue` | `tissue` | Tissue type |
| `disease` | `disease` | Disease condition |

```mermaid
flowchart LR
    subgraph "TileDB Loading"
        TDBOPEN["tiledb.open()"]
        DFEXTRACT["cell_tdb.df[:]"]
        RESET["reset_index()"]
        RENAME["rename_columns()"]
    end
    
    subgraph "Column Processing"
        STUDYCOL["study_column → study"]
        SAMPLECOL["sample_column → sample"]
        TISSUECOL["tissue_column → tissue"]
        DISEASECOL["disease_column → disease"]
    end
    
    TDBOPEN --> DFEXTRACT
    DFEXTRACT --> RESET
    RESET --> RENAME
    
    RENAME --> STUDYCOL
    RENAME --> SAMPLECOL
    RENAME --> TISSUECOL
    RENAME --> DISEASECOL
    
    style TDBOPEN fill:#e1f5fe
    style RENAME fill:#f3e5f5
```

**Title:** Cell Metadata Column Standardization Process

Sources: [scripts/build_cellsearch_metadata.py:50-58]()

### Ontology Mapping

Cell type ontology IDs are mapped to human-readable names using the Cell Ontology:

```mermaid
flowchart TB
    subgraph "Ontology Processing"
        IMPORTONTO["import_cell_ontology()"]
        GETMAPPER["get_id_mapper(onto)"]
        MAPIDS["cellTypeOntologyID.map(id2name)"]
    end
    
    subgraph "Data Transformation"
        ONTOID["cellTypeOntologyID"]
        AUTHORLABEL["author_label"]
    end
    
    IMPORTONTO --> GETMAPPER
    GETMAPPER --> MAPIDS
    ONTOID --> MAPIDS
    MAPIDS --> AUTHORLABEL
    
    style IMPORTONTO fill:#e1f5fe
    style MAPIDS fill:#f3e5f5
```

**Title:** Cell Type Ontology Mapping Pipeline

Sources: [scripts/build_cellsearch_metadata.py:61-63]()

## Prediction and Annotation Processing

### Batch Processing Pipeline

The system processes cells in configurable batches to manage memory usage efficiently:

```mermaid
flowchart TD
    subgraph "Batch Processing Loop"
        START["i = 0"]
        CHECKEND["i < cell_metadata.shape[0]"]
        CALCBATCH["n = min(i + buffer_size, total_cells)"]
        GETBATCH["df = cell_metadata.iloc[i:n]"]
        LOADEMBEDDING["embedding = embedding_tdb[cell_idx]"]
        PREDICT["model.get_predictions_knn()"]
        APPENDRESULTS["Append to prediction_list"]
        INCREMENT["i += buffer_size"]
    end
    
    START --> CHECKEND
    CHECKEND -->|"Yes"| CALCBATCH
    CALCBATCH --> GETBATCH
    GETBATCH --> LOADEMBEDDING
    LOADEMBEDDING --> PREDICT
    PREDICT --> APPENDRESULTS
    APPENDRESULTS --> INCREMENT
    INCREMENT --> CHECKEND
    CHECKEND -->|"No"| END["Complete"]
    
    style PREDICT fill:#e1f5fe
    style LOADEMBEDDING fill:#f3e5f5
```

**Title:** Cell Metadata Batch Processing Loop

Sources: [scripts/build_cellsearch_metadata.py:87-108]()

### CellAnnotation Integration

The `CellAnnotation` model generates predictions and confidence scores for each cell:

| Output Field | Source | Description |
|-------------|--------|-------------|
| `prediction` | `predictions.values` | Predicted cell type |
| `prediction_nn_dist` | `distances.min(axis=1)` | Minimum neighbor distance |
| `data_type` | Train/val CSV lookup | Dataset partition |

Sources: [scripts/build_cellsearch_metadata.py:72-112]()

## Output Structure and Storage

### Final Metadata Schema

The enriched metadata combines original fields with computed annotations:

```mermaid
flowchart LR
    subgraph "Original Fields"
        STUDY["study"]
        SAMPLE["sample"] 
        TISSUE["tissue"]
        DISEASE["disease"]
        ONTOID["cellTypeOntologyID"]
    end
    
    subgraph "Computed Fields"
        AUTHORLABEL["author_label"]
        PREDICTION["prediction"]
        NNDIST["prediction_nn_dist"]
        DATATYPE["data_type"]
    end
    
    subgraph "Output Storage"
        TILEDBOUT["cell_metadata TileDB"]
        SEARCHDIR["cellsearch/ directory"]
    end
    
    STUDY --> TILEDBOUT
    SAMPLE --> TILEDBOUT
    TISSUE --> TILEDBOUT
    DISEASE --> TILEDBOUT
    ONTOID --> TILEDBOUT
    
    AUTHORLABEL --> TILEDBOUT
    PREDICTION --> TILEDBOUT
    NNDIST --> TILEDBOUT
    DATATYPE --> TILEDBOUT
    
    TILEDBOUT --> SEARCHDIR
    
    style TILEDBOUT fill:#e8f5e8
    style SEARCHDIR fill:#f3e5f5
```

**Title:** Final Metadata Schema and Storage Structure

Sources: [scripts/build_cellsearch_metadata.py:110-121]()

### TileDB Storage Format

The final metadata is stored using `tiledb.from_pandas()` in the `cellsearch/cell_metadata` directory, enabling efficient querying by the search and annotation systems.

## Configuration and Command Line Interface

The `build_cellsearch_metadata.py` script accepts the following key parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `-t` | Required | TileDB base path |
| `-m` | Required | Model path |
| `-b` | 100000 | Cell buffer size for batch processing |
| `--cellsearch` | "cellsearch" | Relative path to cellsearch folder |
| `--cellmetadata` | "cell_metadata" | Relative path to cell metadata folder |
| `--knn` | "labelled_kNN.bin" | kNN index filename |
| `--labels` | "reference_labels.tsv" | Reference labels filename |
| `--safelist_file` | None | Optional cell type safelist |

The script integrates with the post-training workflow to create searchable metadata structures that support the core SCimilarity query and annotation functionality.

Sources: [scripts/build_cellsearch_metadata.py:15-29]()

---