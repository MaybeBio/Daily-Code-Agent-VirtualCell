# Model Loading and Inference

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [multiomics_open_research/bulk_rna_bert/pretrained.py](multiomics_open_research/bulk_rna_bert/pretrained.py)

</details>



This document explains how to load pretrained BulkRNABert models and perform inference with them. It covers the key components of the model loading process, the inference pipeline, and provides examples of how to use the loaded models for generating gene embeddings. For information about the available pretrained models and their characteristics, see [Pretrained Models](#3.1).

## 1. Model Loading Process

The central function for loading pretrained models is `get_pretrained_model()`, which handles loading the model configuration, initializing the tokenizer, building the forward function, and loading the model parameters.

```mermaid
flowchart TD
    A["get_pretrained_model()"] --> B["Load Configuration"]
    B --> C["Initialize Tokenizer"]
    C --> D["Build Forward Function"]
    D --> E["Load Parameters"]
    
    A --> F["Return Model Components"]
    F --> G1["parameters"]
    F --> G2["forward_fn"]
    F --> G3["tokenizer"]
    F --> G4["config"]
    
    H["Checkpoint Directory"] --> B
    H --> E
    
    subgraph "Configuration Settings"
        I1["n_genes"]
        I2["n_expressions_bins"]
        I3["embed_dim"]
        I4["num_layers"]
        I5["num_attention_heads"]
    end
    
    I1 -.-> B
    I2 -.-> B
    I3 -.-> B
    I4 -.-> B
    I5 -.-> B
```

Sources: [multiomics_open_research/bulk_rna_bert/pretrained.py:29-86]()

### 1.1 Function Parameters

The `get_pretrained_model()` function accepts several parameters that control the loading process:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | str | Name of the pretrained model to load (e.g., "bulk_rna_bert_tcga") |
| `compute_dtype` | jnp.dtype | Data type used for computations (default: jnp.float32) |
| `param_dtype` | jnp.dtype | Data type for model parameters (default: jnp.float32) |
| `output_dtype` | jnp.dtype | Data type for model outputs (default: jnp.float32) |
| `embeddings_layers_to_save` | tuple[int, ...] | Indices of transformer layers from which to return embeddings |
| `checkpoint_directory` | str | Directory where checkpoints are stored (default: "checkpoints/") |

Sources: [multiomics_open_research/bulk_rna_bert/pretrained.py:29-36]()

### 1.2 Available Pretrained Models

BulkRNABert provides three pretrained models:

| Model Name | Description |
|------------|-------------|
| `bulk_rna_bert_tcga` | BulkRNABert pretrained on TCGA data |
| `bulk_rna_bert_gtex_encode` | BulkRNABert pretrained on GTEx and ENCODE data |
| `bulk_rna_bert_gtex_encode_tcga` | BulkRNABert pretrained on a mix of GTEx, ENCODE, and TCGA data |

Sources: [README.md:49-52]()

## 2. Inference Pipeline

The inference process with BulkRNABert involves several steps: loading the model, preprocessing RNA-seq data, tokenizing, running inference, and extracting embeddings.

```mermaid
flowchart LR
    A["RNA-seq Data (CSV)"] --> B["preprocess_rna_seq_for_bulkrnabert()"]
    B --> C["RNA-seq Array"]
    
    C --> D["tokenizer.batch_tokenize()"]
    D --> E["Token IDs"]
    
    E --> F["forward_fn.apply()"]
    F --> G["Model Outputs"]
    
    G --> H["Gene Embeddings"]
    
    I["get_pretrained_model()"] --> J["parameters"]
    I --> K["forward_fn"]
    I --> L["tokenizer"]
    I --> M["config"]
    
    J --> F
    K --> F
    M --> B
    L --> D
```

Sources: [README.md:19-48]()

### 2.1 Preprocessing Data

Before inference, RNA-seq data must be preprocessed using the `preprocess_rna_seq_for_bulkrnabert()` function. This function:

1. Takes a pandas DataFrame containing RNA-seq data
2. Applies normalization (log or max normalization based on config)
3. Returns a numpy array ready for tokenization

Sources: [README.md:37-38]()

### 2.2 Tokenization

The tokenizer converts normalized RNA-seq values into token IDs that the model can process:

1. The `BinnedExpressionTokenizer` discretizes continuous expression values into bins
2. The `batch_tokenize()` method processes multiple samples at once
3. The output is a 2D array of token IDs (samples × genes)

Sources: [README.md:39-40](), [multiomics_open_research/bulk_rna_bert/pretrained.py:67-72]()

### 2.3 Model Inference

Inference involves:

1. Transforming the model's forward function using `hk.transform()`
2. Converting tokens to JAX arrays
3. Calling the transformed forward function with model parameters
4. Extracting embeddings from the output

Sources: [README.md:43-47]()

## 3. End-to-End Example

The complete process of loading a model and performing inference follows this sequence:

```mermaid
sequenceDiagram
    participant User
    participant get_pretrained_model
    participant preprocess_rna_seq_for_bulkrnabert
    participant BinnedExpressionTokenizer
    participant hk.transform
    participant forward_fn.apply
    
    User->>get_pretrained_model: model_name, embeddings_layers_to_save
    get_pretrained_model-->>User: parameters, forward_fn, tokenizer, config
    
    User->>hk.transform: forward_fn
    hk.transform-->>User: transformed_forward_fn
    
    User->>preprocess_rna_seq_for_bulkrnabert: rna_seq_df, config
    preprocess_rna_seq_for_bulkrnabert-->>User: rna_seq_array
    
    User->>BinnedExpressionTokenizer: batch_tokenize(rna_seq_array)
    BinnedExpressionTokenizer-->>User: tokens_ids
    
    User->>forward_fn.apply: parameters, random_key, tokens
    forward_fn.apply-->>User: model_outputs
    
    User->>User: Extract embeddings
```

Sources: [README.md:19-48]()

### 3.1 Code Implementation

Here's a reference to the essential code flow for loading a model and performing inference:

1. Import required packages and functions
2. Load the pretrained model using `get_pretrained_model()`
3. Transform the forward function using `hk.transform()`
4. Load and preprocess RNA-seq data
5. Tokenize the preprocessed data
6. Run inference using `forward_fn.apply()`
7. Extract the desired embeddings from the output

Sources: [README.md:20-47]()

### 3.2 Extracting Embeddings

After inference, you can extract embeddings from different transformer layers based on your `embeddings_layers_to_save` setting:

- The output dictionary contains keys like `embeddings_4` for the embeddings from layer 4
- You can compute mean embeddings across genes to get a sample-level representation
- These embeddings can be used for downstream tasks like classification or survival analysis

Sources: [README.md:47-48]()

## 4. Advanced Usage

### 4.1 Working with Different Data Types

BulkRNABert supports different precision levels for computations, parameters, and outputs:

- Use `compute_dtype=jnp.float16` for faster inference when precision is less critical
- Use `param_dtype=jnp.float32` for training for better numerical stability
- Match `output_dtype` to your downstream application requirements

Sources: [multiomics_open_research/bulk_rna_bert/pretrained.py:31-33]()

### 4.2 Accessing Multiple Layer Embeddings

To get embeddings from multiple layers, specify them in `embeddings_layers_to_save`:

```
# Get embeddings from layers 1, 2, and 4
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(1, 2, 4),
)
```

This will result in output keys `embeddings_1`, `embeddings_2`, and `embeddings_4` in the model output.

Sources: [multiomics_open_research/bulk_rna_bert/pretrained.py:34](), [multiomics_open_research/bulk_rna_bert/pretrained.py:74]()

## 5. Model Loading and Inference in the BulkRNABert Architecture

The model loading and inference components fit into the overall BulkRNABert architecture as follows:

```mermaid
flowchart TD
    A["get_pretrained_model()"] --> |"Returns"| B["Model Components"]
    B --> B1["parameters"]
    B --> B2["forward_fn"]
    B --> B3["tokenizer"]
    B --> B4["config"]
    
    C["RNA-seq Data"] --> D["preprocess_rna_seq_for_bulkrnabert()"]
    D --> E["Processed Data"]
    E --> F["tokenizer.batch_tokenize()"]
    F --> G["Token IDs"]
    
    B1 --> H["forward_fn.apply()"]
    B2 --> |"Transformed by hk.transform()"| H
    G --> H
    H --> I["Model Outputs"]
    I --> J["Embedding Extraction"]
    J --> K["Downstream Applications"]
    
    K --> K1["Cancer Classification"]
    K --> K2["Survival Analysis"]
    K --> K3["Custom Analysis"]
```

Sources: [README.md:19-48](), [multiomics_open_research/bulk_rna_bert/pretrained.py:29-86]()

This page covers how to load pretrained BulkRNABert models and perform inference to generate embeddings. For information on how to use these embeddings for downstream tasks, see [Cancer Classification](#5.1) and [Survival Analysis](#5.2).

---