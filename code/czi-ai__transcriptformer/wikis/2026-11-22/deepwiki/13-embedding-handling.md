# Embedding Handling

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/transcriptformer/model/inference.py](src/transcriptformer/model/inference.py)
- [src/transcriptformer/utils/utils.py](src/transcriptformer/utils/utils.py)

</details>



## Purpose and Scope

This document describes how TranscriptFormer models handle gene embeddings - the numerical vector representations of genes that capture their functional and contextual similarities. The page covers loading pre-trained embeddings, performing embedding surgery for out-of-distribution inference, generating embeddings during model inference, and storing/manipulating embeddings in various file formats. For information about the data processing pipeline prior to embedding generation, see [Data Processing Pipeline](#4.2).

## Embedding Overview

Gene embeddings in TranscriptFormer represent the core method for transforming gene expression data into meaningful numerical representations that can be used for downstream analysis.

```mermaid
flowchart LR
    A["Raw Gene Expression Data"] --> B["Preprocessing Pipeline"]
    B --> C["TranscriptFormer Model"]
    C --> D["Gene Embeddings"]
    D --> E["Cell Type Classification"]
    D --> F["Integration Analysis"]
    D --> G["Trajectory Analysis"]
    
    subgraph "Embedding Types"
        H["Pretrained Embeddings<br/>(HDF5/PKL Files)"]
        I["Model-Generated<br/>Embeddings"]
    end
    
    H -->|"Used for OOD Species"| C
    C --> I
```

Sources: [src/transcriptformer/model/inference.py:36-75](). [src/transcriptformer/utils/utils.py:13-45]()

## Loading Embeddings

TranscriptFormer supports loading pre-trained embeddings from both pickle (`.pkl`) and HDF5 (`.h5`) files. These embeddings map gene identifiers to vector representations.

### Loading Functions

The system provides utility functions for loading embeddings from different file formats:

```mermaid
classDiagram
    class LoadEmbeddings {
        +load_embeddings(embeddings_path) dict
    }
    
    class LoadFromHDF5 {
        +load_from_hdf5(file_path) dict
    }
    
    LoadEmbeddings --> LoadFromHDF5: Uses for H5 files
```

The `load_embeddings()` function determines the appropriate loading method based on the file extension and returns a dictionary mapping gene names to their corresponding embedding vectors.

Sources: [src/transcriptformer/utils/utils.py:13-45]()

### HDF5 Embedding Structure

When loading from HDF5 files, TranscriptFormer expects a specific structure:

```mermaid
graph TD
    A["HDF5 File"] --> B["'keys' dataset"]
    A --> C["'arrays' group"]
    B --> D["Gene names (encoded as bytes)"]
    C --> E["Individual datasets (one per gene)"]
    E --> F["Embedding vectors (numpy arrays)"]
```

The `load_from_hdf5()` function reads this structure and converts it to a Python dictionary where keys are gene names (decoded from bytes to UTF-8 strings) and values are numpy arrays representing embeddings.

Sources: [src/transcriptformer/utils/utils.py:22-45]()

## Embedding Surgery

A key feature of TranscriptFormer is "embedding surgery," which enables out-of-distribution inference by replacing or augmenting the model's embedding layer with pre-trained embeddings for genes not seen during training.

```mermaid
sequenceDiagram
    participant Config as "Inference Config"
    participant InferenceEngine as "run_inference()"
    participant Surgery as "change_embedding_layer()"
    participant Model as "TranscriptFormer Model"
    
    Config->>InferenceEngine: Specify pretrained_embedding paths
    InferenceEngine->>InferenceEngine: Check if embeddings specified
    InferenceEngine->>Surgery: Call with model and embedding paths
    Surgery->>Model: Modify embedding layer
    Model-->>InferenceEngine: Return modified model
    Note over Model: Model now has embeddings for OOD genes
```

This process is triggered during the inference process when `pretrained_embedding` is specified in the inference configuration. The function `change_embedding_layer()` replaces or extends the model's embedding layer with the pre-trained embeddings.

Sources: [src/transcriptformer/model/inference.py:68-75]()

## Generating Embeddings During Inference

TranscriptFormer generates embeddings during the inference process:

```mermaid
flowchart TD
    A["Input AnnData<br/>or Files"] --> B["AnnDataset"]
    B --> C["DataLoader"]
    C --> D["TranscriptFormer<br/>Model"]
    D --> E["Prediction Output"]
    E --> F["Stack & Process<br/>Output"]
    F --> G["Final AnnData<br/>with Embeddings"]
    
    subgraph "output_keys configuration"
        H["Embedding Types<br/>to Generate"]
    end
    
    H -->|"Controls"| D
```

The embeddings to generate are specified in the `output_keys` parameter of the inference configuration. These embeddings are then stored in the `obsm` attribute of the output AnnData object.

Sources: [src/transcriptformer/model/inference.py:118-138]()

## Working with Output Embeddings

The output of TranscriptFormer inference is an AnnData object with embeddings stored in its `obsm` attribute:

```mermaid
classDiagram
    class AnnData {
        +X Raw data matrix
        +obs Cell metadata
        +var Gene metadata
        +obsm Additional cell matrices
        +uns Unstructured metadata
    }
    
    class EmbeddingsDict {
        +embedding_key1 numpy.ndarray
        +embedding_key2 numpy.ndarray
    }
    
    AnnData --> EmbeddingsDict: obsm
```

Each key in `obsm` corresponds to a specific type of embedding specified in the `output_keys` configuration. The values are numpy arrays containing the embeddings for each cell.

Sources: [src/transcriptformer/model/inference.py:128-138]()

## Saving Embeddings

TranscriptFormer provides utilities for saving embeddings to HDF5 files:

```mermaid
flowchart LR
    A["Embeddings Dictionary"] --> B["save_as_hdf5()"]
    B --> C["HDF5 File"]
    
    subgraph "HDF5 Structure"
        D["'keys' dataset<br/>(Gene Names)"]
        E["'arrays' group<br/>(Embedding Vectors)"]
    end
    
    C --> D
    C --> E
```

The `save_as_hdf5()` function takes a dictionary of embeddings and saves it to an HDF5 file with the specified structure. The keys are stored as a dataset, and the embedding arrays are stored in an "arrays" group.

Sources: [src/transcriptformer/utils/utils.py:71-81]()

## Example Usage Flow

Below is an end-to-end overview of how embeddings are handled in a typical TranscriptFormer workflow:

```mermaid
sequenceDiagram
    participant User
    participant Inference as "run_inference()"
    participant Model as "TranscriptFormer Model"
    participant Utils as "Utility Functions"
    participant Output as "AnnData with Embeddings"
    
    User->>Inference: Configure with data and embedding paths
    Inference->>Utils: load_vocabs_and_embeddings()
    Utils-->>Inference: Return vocabs and embeddings
    Inference->>Model: Initialize with embeddings
    
    alt Has pretrained_embedding configured
        Inference->>Model: Perform embedding surgery
    end
    
    Inference->>Model: Run prediction
    Model-->>Inference: Return output with embeddings
    Inference->>Utils: stack_dict() to process outputs
    Utils-->>Inference: Return processed output
    Inference->>Output: Create AnnData with embeddings in obsm
    Inference-->>User: Return AnnData object
```

This diagram illustrates the complete flow of embedding handling during the inference process, from configuration to the final output.

Sources: [src/transcriptformer/model/inference.py:27-140](), [src/transcriptformer/utils/utils.py:13-81]()

## Best Practices

When working with embeddings in TranscriptFormer:

1. **For in-distribution species**: Use the model's built-in embeddings without modification.
2. **For out-of-distribution species**: Provide pre-trained embeddings through the `pretrained_embedding` parameter.
3. **When saving embeddings**: Use the HDF5 format for better compatibility with the loading functions.
4. **For downstream analysis**: Access embeddings through the `obsm` attribute of the output AnnData object.

Sources: [src/transcriptformer/model/inference.py:68-75](), [src/transcriptformer/utils/utils.py:13-81]()

---