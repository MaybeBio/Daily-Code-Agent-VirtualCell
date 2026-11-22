# Cancer Classification

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [checkpoints/tcga_5_cohorts/config.json](checkpoints/tcga_5_cohorts/config.json)
- [checkpoints/tcga_5_cohorts/params.joblib](checkpoints/tcga_5_cohorts/params.joblib)
- [data/5_cohorts_labels_mapping.pkl](data/5_cohorts_labels_mapping.pkl)
- [multiomics_open_research/bulk_rna_bert/downstream/pretrained.py](multiomics_open_research/bulk_rna_bert/downstream/pretrained.py)

</details>



## Purpose and Scope

This document describes how to use BulkRNABert for cancer type classification. It explains the architecture of the cancer classification module, the available pre-trained models, and provides practical guidance on implementing cancer classification using the BulkRNABert embeddings. For information about survival prediction, see [Survival Analysis](#5.2).

## System Architecture

The cancer classification pipeline in BulkRNABert uses a two-stage approach:

1. A pre-trained BulkRNABert model generates gene embeddings from RNA-seq data
2. A classification head (MLP) takes these embeddings to predict cancer types

```mermaid
flowchart LR
    A["RNA-seq Data"] --> B["BulkRNABert\nEncoder"]
    B --> C["Gene\nEmbeddings"]
    C --> D["MLP\nClassifier"]
    D --> E["Cancer Type\nPrediction"]
    
    subgraph "Classification Head"
        D
    end
    
    subgraph "Pre-trained Model"
        B
        C
    end
```

Sources: [multiomics_open_research/bulk_rna_bert/downstream/pretrained.py:26-29]()

The pre-trained model for cancer classification supports 5 cancer cohorts:

```mermaid
flowchart TD
    A["Cancer Classification\nMLP"] --> B["5 Output Logits"]
    B --> C1["BLCA\n(Bladder Urothelial Carcinoma)"]
    B --> C2["BRCA\n(Breast Invasive Carcinoma)"]
    B --> C3["GBMLGG\n(Glioblastoma Multiforme and\nBrain Lower Grade Glioma)"]
    B --> C4["LUAD\n(Lung Adenocarcinoma)"]
    B --> C5["UCEC\n(Uterine Corpus Endometrial Carcinoma)"]
```

Sources: [data/5_cohorts_labels_mapping.pkl:1](), [checkpoints/tcga_5_cohorts/config.json:1]()

## Model Configuration

The cancer classification model uses the following configuration:

| Component | Configuration |
|-----------|---------------|
| Base Model | BulkRNABert pre-trained on TCGA data |
| Hidden Layer Sizes | [256, 128] |
| Final Activation | Identity |
| MLP Activation | SELU |
| Dropout Rate | 0.0 |
| Layer Normalization | Disabled |
| Number of Output Classes | 5 (BLCA, BRCA, GBMLGG, LUAD, UCEC) |

Sources: [checkpoints/tcga_5_cohorts/config.json:1]()

## Implementation Details

The cancer classification system utilizes the `RNASeqSurvivalMLP` module for the classification head, despite its name suggesting survival analysis. The same underlying architecture is used for both tasks with different output configurations.

```mermaid
classDiagram
    class get_pretrained_downstream_model {
      +loads model parameters
      +configures tokenizer
      +builds forward function
      +returns parameters, forward_fn, tokenizer, config
    }
    
    class RNASeqSurvivalMLP {
      +hidden_sizes
      +final_activation
      +dropout_rate
      +layer_norm
      +mlp_activation
      +can be used for classification or survival prediction
    }
    
    class build_bulk_bert_with_head_fn {
      +combines BulkRNABert with classification head
      +configures compute parameters
    }
    
    get_pretrained_downstream_model --> RNASeqSurvivalMLP: creates
    get_pretrained_downstream_model --> build_bulk_bert_with_head_fn: uses
```

Sources: [multiomics_open_research/bulk_rna_bert/downstream/pretrained.py:38-118](), [multiomics_open_research/bulk_rna_bert/downstream/pretrained.py:103-105]()

## Data Flow

The cancer classification pipeline processes data through the following steps:

```mermaid
flowchart LR
    A["RNA-seq Data\n(CSV format)"] --> B["Preprocessing"]
    B --> C["Tokenization"]
    C --> D["BulkRNABert\nEncoder"]
    D --> E["Gene Embeddings"]
    E --> F["MLP\nClassifier"]
    F --> G["Cancer Type\nPrediction"]
    
    subgraph "Inference Process"
        B
        C
        D
        E
        F
        G
    end
```

Sources: [README.md:19-48]()

## Usage

To use the pre-trained cancer classification model:

1. First, prepare your RNA-seq data according to the expected format (see [Data Preprocessing](#4))
2. Load the pre-trained downstream model using the `get_pretrained_downstream_model` function
3. Preprocess and tokenize your RNA-seq data
4. Run the model to get predictions

Example workflow:

```mermaid
flowchart TD
    A["Load model with\nget_pretrained_downstream_model('tcga_5_cohorts')"] --> B["Preprocess RNA-seq data\nwith preprocess_rna_seq_for_bulkrnabert"]
    B --> C["Tokenize data\nwith tokenizer.batch_tokenize"]
    C --> D["Run model inference\nwith forward_fn.apply"]
    D --> E["Get classification predictions"]
```

Sources: [multiomics_open_research/bulk_rna_bert/downstream/pretrained.py:38-118](), [README.md:19-48]()

## Model Loading and Inference

The system provides the `get_pretrained_downstream_model` function to load the pre-trained cancer classification model:

```mermaid
flowchart TD
    A["get_pretrained_downstream_model"] --> B["Load Configuration"]
    B --> C["Initialize Tokenizer"]
    C --> D["Build Forward Function"]
    D --> E["Load Parameters"]
    
    A --> F["Return Components"]
    F --> G1["parameters"]
    F --> G2["forward_fn"]
    F --> G3["tokenizer"]
    F --> G4["config"]
    F --> G5["mlm_config"]
    
    H["Checkpoint Directory"] --> B
    H --> E
```

Sources: [multiomics_open_research/bulk_rna_bert/downstream/pretrained.py:38-118]()

The model uses "tcga_5_cohorts" as the model name, which maps to "classification_head" in the internal model configuration.

Sources: [multiomics_open_research/bulk_rna_bert/downstream/pretrained.py:33-35]()

## Key Files and Resources

| Resource | Description |
|----------|-------------|
| `get_pretrained_downstream_model` | Function to load pre-trained cancer classification model |
| `tcga_5_cohorts` | Pre-trained model name for 5 cancer cohort classification |
| `data/5_cohorts_labels_mapping.pkl` | Mapping of numerical labels to cancer cohort names |
| `examples/downstream_task_example.ipynb` | Example notebook for cancer classification |

Sources: [README.md:71-73](), [multiomics_open_research/bulk_rna_bert/downstream/pretrained.py:33-35](), [multiomics_open_research/bulk_rna_bert/downstream/pretrained.py:38-118]()

## Notes on Model Performance

The BulkRNABert approach achieves state-of-the-art performance in cancer type classification on the TCGA dataset. The model was trained on the 5 cancer cohorts (BRCA, BLCA, GBMLGG, LUAD, UCEC) and demonstrates the effectiveness of transformer-based models for capturing gene expression patterns specific to different cancer types.

Sources: [README.md:5-7]()

For a complete example demonstrating cancer classification, refer to the notebook at `examples/downstream_task_example.ipynb`.

Sources: [README.md:71-73]()

---