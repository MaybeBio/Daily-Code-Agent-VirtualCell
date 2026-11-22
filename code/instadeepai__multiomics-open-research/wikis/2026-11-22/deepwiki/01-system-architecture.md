# System Architecture

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [multiomics_open_research/bulk_rna_bert/pretrained.py](multiomics_open_research/bulk_rna_bert/pretrained.py)

</details>



## Purpose and Scope

This document describes the high-level architecture of BulkRNABert, detailing its main components and how they interact to process RNA-sequencing data through transformer-based models. It covers the data flow from raw RNA-seq data to gene embeddings and explains how these embeddings enable downstream applications. For specific details about the BulkRNABert model itself, see [BulkRNABert Model](#3), and for preprocessing specifics, see [Data Preprocessing](#4).

## System Overview

BulkRNABert is a transformer-based system for analyzing bulk RNA sequencing data. The system processes gene expression data to generate contextualized gene embeddings that can be used for various downstream tasks like cancer classification and survival prediction.

```mermaid
flowchart TD
    A["Raw RNA-seq Data"] --> B["Preprocessing Pipeline"]
    B --> C["Processed RNA-seq Data"]
    C --> D["BinnedExpressionTokenizer"]
    D --> E["Tokenized RNA-seq Data"]
    E --> F["BulkRNABert Model"]
    
    G["Pretrained Model Checkpoints"] --> F
    
    F --> H["Gene Embeddings"]
    H --> I["Downstream Tasks"]
    I --> J1["Cancer Type Classification"]
    I --> J2["Survival Analysis"]
    
    subgraph "Pretrained Models"
        G1["bulk_rna_bert_tcga"]
        G2["bulk_rna_bert_gtex_encode"]
        G3["bulk_rna_bert_gtex_encode_tcga"]
    end
    
    G1 -.-> G
    G2 -.-> G
    G3 -.-> G
    
    K["Common Gene IDs"] --> B
```

Sources: [README.md:5-7](), [multiomics_open_research/bulk_rna_bert/pretrained.py:29-86]()

## Core Components

The BulkRNABert system consists of the following core components:

1. **Data Preprocessing** - Transforms raw RNA-seq data into a standardized format
2. **Tokenization** - Converts numerical gene expression values into token IDs
3. **BulkRNABert Model** - Processes tokenized data to generate gene embeddings
4. **Model Loading** - Facilitates loading of pretrained models
5. **Downstream Applications** - Utilizes embeddings for specific tasks like cancer classification

### Data Preprocessing Pipeline

The preprocessing pipeline converts raw RNA-seq data (such as from TCGA, GTEx, or ENCODE) into a format suitable for the BulkRNABert model.

```mermaid
flowchart LR
    A["Raw RNA-seq Data Files"] --> B["preprocess_tcga_rna_seq.py"]
    K["common_gene_id.txt"] --> B
    B --> C["Processed RNA-seq CSV"]
    
    C --> D["preprocess_rna_seq_for_bulkrnabert()"]
    D --> E["RNA-seq Array"]
    
    subgraph "Normalization Options"
        N1["Log Normalization"]
        N2["Max Normalization"]
    end
    
    N1 -.-> D
    N2 -.-> D
    
    F["BulkRNABertConfig"] --> D
```

Sources: [README.md:55-69]()

Key features of the preprocessing pipeline:
- Uses a standardized set of gene IDs specified in `common_gene_id.txt`
- Supports different normalization strategies (log, max)
- Transforms raw data (like TPM values) into a format ready for tokenization

### Tokenization System

The tokenization system converts normalized gene expression values into token IDs for input to the transformer model.

```mermaid
flowchart TD
    A["RNA-seq Array"] --> B["BinnedExpressionTokenizer"]
    B --> C["Token IDs"]
    
    D["Configuration Parameters"] --> B
    D1["n_expressions_bins"] --> D
    D2["use_max_normalization"] --> D
    D3["normalization_factor"] --> D
    D4["prepend_cls_token"] --> D
    
    C --> E["BulkRNABert Model Input"]
```

Sources: [multiomics_open_research/bulk_rna_bert/pretrained.py:67-72](), [README.md:38-40]()

The `BinnedExpressionTokenizer` discretizes continuous gene expression values into bins, turning them into integer token IDs that can be processed by the transformer model.

### Model Architecture

```mermaid
flowchart TD
    A["Token IDs"] --> B["BulkRNABert Model"]
    
    subgraph "BulkRNABert Internal Structure"
        C["Gene Embeddings Layer"] --> D["Transformer Encoder Layers"]
        D --> E["Contextual Gene Embeddings"]
    end
    
    B --> C
    E --> F["Embeddings Output"]
    
    G["Model Config"] --> B
    H["Pretrained Parameters"] --> B
    
    F --> I["Downstream Applications"]
```

Sources: [multiomics_open_research/bulk_rna_bert/pretrained.py:76-82](), [README.md:41-47]()

BulkRNABert uses a BERT-style transformer architecture that processes token IDs to generate contextual gene embeddings. These embeddings capture complex relationships between genes in the expression profile.

## Model Loading System

The model loading system provides a straightforward way to load pretrained BulkRNABert models.

```mermaid
flowchart TD
    A["get_pretrained_model()"] --> B["Load Configuration"]
    B --> C["Initialize Tokenizer"]
    C --> D["Build Forward Function"]
    D --> E["Load Parameters"]
    
    F["Checkpoint Directory"] --> G["Model-specific Checkpoint Path"] 
    G --> B
    G --> E
    
    A --> H["Return Components"]
    H --> I1["parameters"]
    H --> I2["forward_fn"]
    H --> I3["tokenizer"]
    H --> I4["config"]
```

Sources: [multiomics_open_research/bulk_rna_bert/pretrained.py:29-86](), [README.md:28-33]()

The `get_pretrained_model` function serves as the primary entry point for using pretrained BulkRNABert models. It:
1. Loads the model configuration from the checkpoint directory
2. Initializes the tokenizer with the correct parameters
3. Builds a forward function for the model
4. Loads the pretrained parameters
5. Returns all components needed for inference

## Complete Inference Flow

The following diagram illustrates the complete flow from raw RNA-seq data to embeddings that can be used for downstream tasks:

```mermaid
flowchart LR
    A["RNA-seq Data CSV"] --> B["preprocess_rna_seq_for_bulkrnabert()"]
    B --> C["RNA-seq Array"]
    C --> D["tokenizer.batch_tokenize()"]
    D --> E["Token IDs"]
    E --> F["forward_fn.apply()"]
    F --> G["Embeddings"]
    
    H["get_pretrained_model()"] --> I["parameters, forward_fn, tokenizer, config"]
    I --> D
    I --> F
    
    G --> J["Mean Embedding"]
    J --> K["Downstream Tasks"]
```

Sources: [README.md:19-48]()

## Pretrained Model Variants

BulkRNABert offers three pretrained model variants, each trained on different datasets:

| Model Name | Training Data | Use Case |
|------------|---------------|----------|
| `bulk_rna_bert_tcga` | TCGA | Cancer research focused |
| `bulk_rna_bert_gtex_encode` | GTEx and ENCODE | Normal tissue expression patterns |
| `bulk_rna_bert_gtex_encode_tcga` | GTEx, ENCODE, and TCGA | Comprehensive gene expression understanding |

Sources: [README.md:49-52]()

## Downstream Applications System

BulkRNABert embeddings can be used for various downstream tasks, particularly in cancer research:

```mermaid
flowchart LR
    A["BulkRNABert Embeddings"] --> B["Downstream Applications"]
    
    B --> C["Cancer Classification"]
    B --> D["Survival Analysis"]
    
    C --> E["MLP Classifier"]
    E --> F1["BRCA"]
    E --> F2["BLCA"]
    E --> F3["GBMLGG"]
    E --> F4["LUAD"]
    E --> F5["UCEC"]
    
    D --> G["Survival Predictor"]
    G --> H["Patient Outcome Prediction"]
```

Sources: [README.md:71-73]()

The system provides pretrained downstream models for specific cancer classification tasks and survival analysis, leveraging the gene embeddings generated by BulkRNABert.

## Technical Implementation Details

BulkRNABert is implemented using JAX and Haiku, providing efficient computation and transformation-based building blocks:

```mermaid
flowchart TD
    A["JAX Backend"] --> B["BulkRNABert Implementation"]
    C["Haiku Neural Network Library"] --> B
    
    B --> D["Forward Transformation"]
    D --> E["Embedding Generation"]
    
    F["Model Parameters"] --> D
    G["Input Tokens"] --> D
    
    E --> H["Downstream Applications"]
```

Sources: [README.md:19-48](), [multiomics_open_research/bulk_rna_bert/pretrained.py:29-86]()

The technical implementation uses:
- JAX for high-performance numerical computing
- Haiku for neural network component definitions
- Functional transformations for model application
- Checkpointing for model parameter storage and loading

The model parameters are stored as joblib files in the checkpoint directory, and configuration is stored in JSON format.

## Data Flow Throughout System

The following diagram illustrates how data flows through the entire BulkRNABert system:

```mermaid
flowchart TD
    A["Raw RNA-seq Data"] --> B["Preprocessing"]
    B --> C["Gene Expression Table"]
    C --> D["Normalization"]
    D --> E["Tokenization"]
    E --> F["Token IDs"]
    F --> G["BulkRNABert Model"]
    G --> H["Gene Embeddings"]
    H --> I["Mean Pooling"]
    I --> J["Sample Embedding Vector"]
    J --> K1["Cancer Classification"]
    J --> K2["Survival Analysis"]
    
    L["common_gene_id.txt"] --> B
    M["BulkRNABertConfig"] --> D
    M --> E
    M --> G
    N["Pretrained Parameters"] --> G
```

Sources: [README.md:19-48](), [multiomics_open_research/bulk_rna_bert/pretrained.py:29-86]()

This diagram shows the complete data transformation process from raw RNA-seq data to actionable insights via the downstream applications, illustrating the role of each component in the system.

---