# Usage Examples

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [notebooks/README.md](notebooks/README.md)
- [notebooks/experiments/downstream_fine_tune.ipynb](notebooks/experiments/downstream_fine_tune.ipynb)

</details>



This page provides practical examples of using the Nicheformer system for single-cell and spatial omics data analysis. We'll cover the main workflows and usage patterns, demonstrating how to work with the system from data preparation to downstream analysis. For specific detailed examples, please refer to the subsequent child pages:

- For data acquisition and preparation details, see [Data Download and Preprocessing](#7.1)
- For detailed tokenization workflows, see [Tokenization Examples](#7.2)
- For performing analyses with Nicheformer, see [Downstream Analysis Examples](#7.3)

## Typical Workflow

The typical workflow for using Nicheformer consists of four main stages: data preparation, data tokenization, model training/fine-tuning, and downstream analysis.

```mermaid
flowchart LR
    subgraph "1. Data Preparation"
        A1["Download Data"] --> A2["Preprocess"]
        A2 --> A3["Quality Control"]
        A3 --> A4["Create AnnData"]
    end
    subgraph "2. Tokenization"
        B1["Load Tech Means"] --> B2["Tokenize Data"]
        B2 --> B3["Parquet Files"]
    end
    subgraph "3. Model Training/Fine-tuning"
        C1["Pre-trained Model"] --> C2["Fine-tune Model"]
        C2 --> C3["Trained Model"]
    end
    subgraph "4. Downstream Analysis"
        D1["Generate Embeddings"] --> D2["Regression Analysis"]
        D1 --> D3["Classification Tasks"]
        D1 --> D4["Niche Analysis"]
    end
    
    A4 --> B1
    B3 --> C1
    C3 --> D1
```

Sources: [notebooks/README.md:3-5]()

## Data Preparation Example

Before using Nicheformer, you'll need to prepare your data in the correct format. The system works with AnnData objects, which store gene expression data and associated metadata.

```mermaid
flowchart TD
    Raw["Raw Data Files (.mtx, .h5ad, .csv)"] --> QC["Quality Control"]
    QC --> AnnData["Create AnnData Object"]
    AnnData --> Split["Split into Train/Val/Test"]
    Split --> Save["Save Processed AnnData"]
    
    style AnnData stroke-width:2px
```

Sources: [notebooks/experiments/downstream_fine_tune.ipynb:110-116]()

## Basic Tokenization

Tokenization converts gene expression data into tokens that can be processed by the Nicheformer model. This example shows how to tokenize data using technology-specific mean vectors.

```python
# Example of how to create and use a dataset for tokenization
train_dataset = NicheformerDataset(
    adata=adata,  # Your AnnData object
    technology_mean=technology_mean,  # Technology-specific mean vectors
    split='train',  # Data split to use
    max_seq_len=1500,  # Maximum sequence length
    aux_tokens=30,  # Number of auxiliary tokens
    chunk_size=1000,  # Chunk size for processing
    metadata_fields = {
        'obs': ['author_cell_type'],  # Optional metadata fields
    }
)
```

Sources: [notebooks/experiments/downstream_fine_tune.ipynb:118-129]()

## Fine-tuning for Downstream Tasks

Nicheformer can be fine-tuned for various downstream tasks such as niche regression, cell type classification, and density prediction.

```mermaid
flowchart TD
    PreTrained["Pre-trained Nicheformer Model"] --> |Load| FT["NicheformerFineTune"]
    FT --> |Train| Tasks["Downstream Tasks"]
    Tasks --> |Regression| NicheReg["Niche Regression"]
    Tasks --> |Classification| CellType["Cell Type Classification"]
    Tasks --> |Regression| Density["Density Prediction"]
    
    style FT stroke-width:2px
    style PreTrained stroke-width:2px
```

Sources: [notebooks/experiments/downstream_fine_tune.ipynb:197-222]()

### Example Configuration for Fine-tuning

Below is a table of common configuration parameters for fine-tuning the Nicheformer model:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `supervised_task` | Type of fine-tuning task | `'niche_regression'` |
| `extract_layers` | Model layers to extract features from | `[11]` |
| `dim_prediction` | Dimension of output prediction vector | `33` |
| `freeze` | Whether to freeze backbone model | `True` |
| `pool` | Pooling method for embeddings | `'mean'` |
| `label` | Target variable to predict | `'X_niche_1'` |
| `lr` | Learning rate | `1e-4` |
| `max_epochs` | Maximum training epochs | `100` |

Sources: [notebooks/experiments/downstream_fine_tune.ipynb:55-90]()

## Fine-tuning Complete Workflow

This diagram shows the complete workflow for fine-tuning a pre-trained Nicheformer model and generating predictions.

```mermaid
flowchart TB
    Data["AnnData Object"] --> Split["Split into Train/Val/Test"]
    Split --> DataLoaders["Create DataLoaders"]
    
    PreTrained["Load Pre-trained Model"] --> FineTune["Create NicheformerFineTune"]
    FineTune --> Train["Train with PyTorch Lightning"]
    DataLoaders --> Train
    
    Train --> Test["Test Model"]
    Test --> Predict["Generate Predictions"]
    Predict --> Save["Save Results to AnnData"]
    
    style FineTune stroke-width:2px
    style PreTrained stroke-width:2px
    style DataLoaders stroke-width:2px
```

Sources: [notebooks/experiments/downstream_fine_tune.ipynb:253-313]()

## Code Entity to Function Mapping

The following diagram maps Nicheformer system components to their corresponding code entities:

```mermaid
graph TD
    subgraph "Nicheformer Code Entities"
        NF["Nicheformer Class"] --- FT["NicheformerFineTune"]
        NF --- Dataset["NicheformerDataset"]
        FT --- Tasks["Downstream Tasks"]
        Tasks --- Niche["niche_regression"]
        Tasks --- CellType["cell_type_classification"] 
        Tasks --- Density["density_prediction"]
    end
    
    subgraph "File Structure"
        Main["nicheformer/models/nicheformer.py"] --- FineTune["nicheformer/models/_nicheformer_fine_tune.py"]
        Main --- Dataset2["nicheformer/data/dataset.py"]
    end
    
    NF --> Main
    FT --> FineTune
    Dataset --> Dataset2
```

Sources: [notebooks/experiments/downstream_fine_tune.ipynb:31-33]()

## Example Usage Flow

The following sequence diagram shows an example user workflow with the Nicheformer system:

```mermaid
sequenceDiagram
    participant User
    participant DataProcessing as "Data Processing"
    participant Model as "Nicheformer Model"
    participant Analysis as "Analysis Tools"
    
    User->>DataProcessing: Download & prepare data
    DataProcessing->>DataProcessing: Create AnnData object
    DataProcessing->>DataProcessing: Tokenize data
    DataProcessing->>User: Return tokenized data
    
    User->>Model: Load pre-trained model
    User->>Model: Configure fine-tuning
    Model->>Model: Fine-tune for specific task
    Model->>User: Return fine-tuned model
    
    User->>Analysis: Generate embeddings
    User->>Analysis: Perform downstream analysis
    Analysis->>User: Return analysis results
```

Sources: [notebooks/experiments/downstream_fine_tune.ipynb:40-43](), [notebooks/README.md:3-5]()

## Common Usage Scenarios

Below is a table of common usage scenarios with Nicheformer:

| Scenario | Description | Key Components |
|----------|-------------|----------------|
| Cell Type Classification | Classify cells based on gene expression | `NicheformerFineTune` with `supervised_task='cell_type_classification'` |
| Niche Composition Prediction | Predict the cellular composition of a niche | `NicheformerFineTune` with `supervised_task='niche_regression'` |
| Density Prediction | Predict cell density in spatial data | `NicheformerFineTune` with `predict_density=True` |
| Feature Extraction | Generate cell embeddings for downstream analysis | Use pre-trained model with `extractor=True` |

Sources: [notebooks/experiments/downstream_fine_tune.ipynb:76-90]()

For detailed examples of each workflow stage, please refer to the specific child pages linked at the top of this document.

---