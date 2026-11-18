# Data Processing Pipeline

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [data/spatialcorpus-110M/dissociated/Lu_2021/preprocess_Lu_2021.py](data/spatialcorpus-110M/dissociated/Lu_2021/preprocess_Lu_2021.py)
- [src/nicheformer/data/constants.py](src/nicheformer/data/constants.py)
- [src/nicheformer/data/tools/niche_compositions.py](src/nicheformer/data/tools/niche_compositions.py)

</details>



## Purpose and Scope

This document describes the data processing pipeline in the Nicheformer system, which transforms raw spatial and single-cell omics data into formats suitable for model training and analysis. The pipeline covers data acquisition, preprocessing, quality control, tokenization, and niche analysis. For detailed information about how the tokenized data is used for model training, see [Training and Fine-tuning](#5).

## Pipeline Overview

The Nicheformer data processing pipeline transforms raw data from various spatial and single-cell technologies into standardized formats used for model training and downstream analyses.

```mermaid
flowchart TD
    subgraph "Data Sources"
        GEO["GEO Datasets"]
        STech["Spatial Technologies"]:::noColor
        STech --> Xenium["Xenium"]:::noColor
        STech --> CosMx["CosMx"]:::noColor
        STech --> MERFISH["MERFISH"]:::noColor
        STech --> Visium["Visium"]:::noColor
    end
    
    subgraph "Raw Data Processing"
        DScripts["Download Scripts"]:::noColor
        RawData["Raw Data"]:::noColor
        PreProcess["Preprocess Scripts"]:::noColor
        QC["Quality Control (qc_filter)"]:::noColor
        Validate["Validation (validate)"]:::noColor
        AnnData["Validated AnnData"]:::noColor
    end
    
    subgraph "Data Transformation"
        Token["Tokenization Process (tokenize_data)"]:::noColor
        NComp["Niche Composition Analysis (niche_compositions)"]:::noColor
        SplitData["Data Splitting"]:::noColor
    end
    
    subgraph "Processed Outputs"
        Parquet["Tokenized Parquet Files"]:::noColor
        NicheMatrices["Niche Composition Matrices"]:::noColor
        ProcessedAnnData["Processed AnnData Objects"]:::noColor
    end
    
    GEO --> DScripts
    Xenium & CosMx & MERFISH & Visium --> DScripts
    DScripts --> RawData
    RawData --> PreProcess
    PreProcess --> QC
    QC --> Validate
    Validate --> AnnData
    
    AnnData --> Token
    AnnData --> NComp
    AnnData --> SplitData
    
    Token --> Parquet
    NComp --> NicheMatrices
    SplitData --> ProcessedAnnData
    NicheMatrices --> ProcessedAnnData
    
    ProcessedAnnData & Parquet --> Training["Training Pipeline"]:::noColor
```

Sources: [data/spatialcorpus-110M/dissociated/Lu_2021/preprocess_Lu_2021.py:1-87](), [src/nicheformer/data/tools/niche_compositions.py:1-51]()

## Data Standards and Constants

The Nicheformer pipeline enforces standardized data structures through defined constants that specify key attributes in AnnData objects.

```mermaid
classDiagram
    class AnnData {
        +X: gene expression matrix
        +obs: cell metadata
        +var: gene metadata
        +obsm: observation matrices
        +uns: unstructured metadata
    }
    
    class ObsConstants {
        +ASSAY_ONTOLOGY_TERM_ID
        +SEX_ONTOLOGY_TERM_ID
        +ORGANISM_ONTOLOGY_TERM_ID
        +TISSUE_ONTOLOGY_TERM_ID
        +SUSPENSION_TYPE
        +SAMPLE_ID
        +DONOR_ID
        +TISSUE_TYPE
        +SPATIAL_X
        +SPATIAL_Y
        +AUTHOR_CELL_TYPE
    }
    
    class ObsmConstants {
        +SPATIAL
        +NICHE
    }
    
    class UnsConstants {
        +TITLE
        +NICHE
    }
    
    class VarConstants {
        +FEATURE_IS_FILTERED
    }
    
    class OntologyTerms {
        +AssayOntologyTermId
        +SuspensionTypeId
        +SexOntologyTermId
        +OrganismOntologyTermId
        +TissueOntologyTermId
    }
    
    AnnData o-- ObsConstants : "uses for .obs keys"
    AnnData o-- ObsmConstants : "uses for .obsm keys"
    AnnData o-- UnsConstants : "uses for .uns keys"
    AnnData o-- VarConstants : "uses for .var keys"
    ObsConstants o-- OntologyTerms : "uses term values"
```

Sources: [src/nicheformer/data/constants.py:1-124]()

## Data Preprocessing

### Data Loading and Standardization

The preprocessing stage involves:
1. Loading raw data (MTX, CSV, TSV, etc.) from source locations
2. Converting to standardized AnnData objects
3. Adding required metadata with controlled vocabulary using ontology terms
4. Ensuring all objects follow the Nicheformer data schema

Here's a typical preprocessing workflow shown with the Lu_2021 dataset as an example:

```mermaid
flowchart LR
    subgraph "Raw Files"
        MTX["Matrix (*.mtx.gz)"]:::noColor
        Features["Features (*.tsv.gz)"]:::noColor
        Barcodes["Barcodes (*.tsv.gz)"]:::noColor
    end
    
    subgraph "Processing"
        Read["read_mtx()"]:::noColor
        AddMeta["Add Metadata"]:::noColor
        Concat["anndata.concat()"]:::noColor
        QC["qc_filter()"]:::noColor
        Validate["validate()"]:::noColor
    end
    
    MTX & Features & Barcodes --> Read
    Read --> AddMeta
    AddMeta --> Concat
    Concat --> QC
    QC --> Validate
    Validate --> Final["Validated AnnData (.h5ad)"]:::noColor
    
    subgraph "Added Metadata"
        Assay["assay_ontology_term_id"]:::noColor
        Sex["sex_ontology_term_id"]:::noColor
        Organism["organism_ontology_term_id"]:::noColor
        Tissue["tissue_ontology_term_id"]:::noColor
        Suspension["suspension_type"]:::noColor
        Condition["condition_id"]:::noColor
        Donor["donor_id"]:::noColor
    end
    
    AddMeta --- Assay & Sex & Organism & Tissue & Suspension & Condition & Donor
```

Sources: [data/spatialcorpus-110M/dissociated/Lu_2021/preprocess_Lu_2021.py:23-75]()

### Quality Control and Validation

Quality control is performed using the `qc_filter` function, which applies standard filtering criteria:
- Removes cells with abnormally high or low gene counts
- Filters out genes with low expression across cells
- Normalizes and transforms data when needed

The validation step (`validate` function) ensures:
- All required metadata fields are present
- Values conform to expected ontology terms
- Data structure is compatible with downstream processing

Sources: [data/spatialcorpus-110M/dissociated/Lu_2021/preprocess_Lu_2021.py:74-87]()

## Niche Composition Analysis

### Purpose and Process

For spatial data, Nicheformer calculates niche compositions representing the cell type environment surrounding each cell. This creates a representation of cellular neighborhoods used for spatial context analysis.

```mermaid
flowchart TD
    subgraph "Inputs"
        AData["AnnData with Spatial Coordinates"]:::noColor
        CellTypes["Author Cell Types (.obs['author_cell_type'])"]:::noColor
        Radii["List of Radii"]:::noColor
    end
    
    subgraph "Processing Steps"
        OneHot["One-hot Encode Cell Types"]:::noColor
        SpatialN["Calculate Spatial Neighbors (squidpy.gr.spatial_neighbors)"]:::noColor
        NicheMx["Calculate Niche Matrix (connectivities @ one_hot_ct)"]:::noColor
    end
    
    subgraph "Outputs"
        ObsmNiche["AnnData.obsm['X_niche_n']"]:::noColor
        UnsNiche["AnnData.uns['niche']"]:::noColor
    end
    
    AData --> SpatialN
    CellTypes --> OneHot
    Radii --> SpatialN
    
    OneHot & SpatialN --> NicheMx
    NicheMx --> ObsmNiche
    NicheMx & Radii --> UnsNiche
```

The `niche_compositions` function:
1. Takes an AnnData object with spatial coordinates and cell type annotations
2. Creates one-hot encoded matrix of cell types
3. For each specified radius:
   - Builds a spatial neighbor graph using squidpy
   - Calculates niche composition by multiplying connectivities with cell type matrix
   - Stores results in `.obsm[f"X_niche_{n}"]` 
4. Records metadata about the niche calculation in `.uns['niche']`

Sources: [src/nicheformer/data/tools/niche_compositions.py:9-51]()

### Output Structure

The niche composition analysis adds:

1. **Obsm entries**: 
   - For each radius `n`: `.obsm[f"X_niche_{n}"]` - Matrix of cell count by type in neighborhood
   - Dimensions: n_cells × n_cell_types

2. **Uns metadata**:
   - `.uns['niche']['columns']` - List of cell type names
   - For each radius `n`: `.uns[f"niche_{n}"]` - Configuration parameters including:
     - `radius`: Spatial distance used
     - `coord_type`: Coordinate system type
     - `node_degree`: Average number of neighbors
     - `library_key`: Library identifier field

Sources: [src/nicheformer/data/tools/niche_compositions.py:27-51]()

## Data Format Standardization

The Nicheformer pipeline enforces several standards through specified constants:

### Key AnnData Attributes

| Component | Key Constants | Description |
|-----------|---------------|-------------|
| `.obs` | `ObsConstants` | Cell metadata (assay type, organism, cell type, spatial coordinates) |
| `.obsm` | `ObsmConstants` | Spatial coordinates (`SPATIAL`) and niche compositions (`NICHE`) |
| `.uns` | `UnsConstants` | Dataset metadata and niche computation parameters |
| `.var` | `VarConstants` | Gene metadata including filtering status |

### Ontology Terms

Controlled vocabulary ensures data consistency through ontology terms:

| Category | Example Terms | Usage |
|----------|--------------|-------|
| Assay | `TENX_3`, `SMART_SEQ2`, `MERFISH_SPATIAL` | Technology identification |
| Organism | `MOUSE`, `HUMAN` | Species identification |
| Tissue | `BRAIN`, `LIVER`, `LUNG` | Tissue source |
| Suspension | `TENX_CELL`, `TENX_NUCLEUS`, `SPATIAL` | Sample preparation type |

Sources: [src/nicheformer/data/constants.py:4-124]()

## Integration with Model Training

The processed data serves as input to the Nicheformer training pipeline. For detailed information about tokenization and training procedures, see [Data Tokenization](#4.2) and [Training Pipeline](#5.1).

---