# Training and Fine-tuning

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/nicheformer/_fine_tune.py](src/nicheformer/_fine_tune.py)
- [src/nicheformer/_train.py](src/nicheformer/_train.py)

</details>



This document outlines the training and fine-tuning processes in the Nicheformer system. It covers how to pretrain the Nicheformer foundation model and how to fine-tune it for specific downstream tasks. For details on the model architecture, see [Model Architecture](#3), and for data preprocessing before training, see [Data Processing Pipeline](#4).

## Overview of Training and Fine-tuning Workflow

The Nicheformer training pipeline consists of two main phases:

1. **Pretraining**: Training the foundation model on large-scale single-cell and spatial omics data using masked language modeling
2. **Fine-tuning**: Adapting the pretrained model for specific downstream tasks such as niche regression, cell type classification, or density prediction

```mermaid
flowchart TB
    subgraph "Pretraining Phase"
        direction LR
        TokenData["Tokenized Data"] --> DataModule["MerlinDataModuleDistributed"]
        DataModule --> PreTrain["manual_train_fm()"]
        PretrainConfig["Configuration Parameters"] --> PreTrain
        PreTrain --> PretrainedModel["Pretrained Nicheformer Model"]
        PreTrain --> Checkpoint["Checkpoint Files"]
    end
    
    subgraph "Fine-tuning Phase"
        direction LR
        Checkpoint --> LoadModel["Load Pretrained Model"]
        LoadModel --> FineTuneModel["FineTuningModel"]
        FineTuneConfig["Task Configuration"] --> FineTuneModel
        LabeledData["Labeled Data"] --> FineTuneDataModule["Task-specific DataModule"]
        FineTuneDataModule --> FineTuneTrainer["fine_tune_predictions()"]
        FineTuneModel --> FineTuneTrainer
        FineTuneTrainer --> TaskSpecificModel["Task-specific Model"]
        FineTuneTrainer --> Predictions["Task Predictions"]
    end
    
    PretrainedModel --> LoadModel
```

Sources: [src/nicheformer/_train.py:8-80](), [src/nicheformer/_fine_tune.py:12-129]()

## Pretraining the Nicheformer Model

The pretraining process creates a foundation model by training it on a large corpus of tokenized single-cell and spatial data. This is implemented in the `manual_train_fm` function.

### Pretraining Process

```mermaid
sequenceDiagram
    participant Config as "Configuration"
    participant Model as "Nicheformer"
    participant DataModule as "MerlinDataModuleDistributed"
    participant Trainer as "PyTorch Lightning Trainer"
    participant Logger as "WandbLogger"
    
    Config->>Model: Initialize model parameters
    Config->>DataModule: Configure data paths and settings
    Config->>Trainer: Configure training settings
    
    alt Load Pretrained Model
        Config->>Model: Load from checkpoint
    end
    
    Model->>Trainer: Register model
    DataModule->>Trainer: Register data module
    Logger->>Trainer: Register logger
    
    Trainer->>DataModule: Request batches
    DataModule->>Trainer: Provide tokenized data
    Trainer->>Model: Forward pass (masked tokens)
    Model->>Model: Compute loss
    Trainer->>Model: Update weights
    
    Trainer->>Logger: Log metrics
    Trainer->>Model: Save checkpoints
```

Sources: [src/nicheformer/_train.py:8-80]()

### Pretraining Configuration

The pretraining process is controlled by a configuration dictionary passed to the `manual_train_fm` function, which specifies model architecture and training parameters:

| Parameter | Description |
|-----------|-------------|
| `dim_model` | Dimension of the model's hidden representations |
| `nheads` | Number of attention heads in transformer layers |
| `dim_feedforward` | Dimension of the feedforward network in transformer layers |
| `nlayers` | Number of transformer encoder layers |
| `dropout` | Dropout rate for regularization |
| `masking_p` | Probability of masking tokens for masked language modeling |
| `n_tokens` | Number of tokens in the vocabulary |
| `context_length` | Maximum length of input token sequences |
| `warmup` | Number of warmup steps for learning rate scheduler |
| `lr` | Learning rate |
| `batch_size` | Batch size for training |
| `max_epochs` | Maximum number of training epochs |
| `autoregressive` | Whether to use autoregressive masking |
| `pool` | Pooling method for token representations |
| `supervised_task` | Type of supervised task (if any) |
| `pretrained_path` | Path to a pretrained model checkpoint (if continuing training) |

Sources: [src/nicheformer/_train.py:12-32]()

### Data Loading and Training

The pretraining process uses the `MerlinDataModuleDistributed` class to load tokenized data efficiently:

```mermaid
flowchart LR
    subgraph "Data Loading"
        TokenizedData["Tokenized Data Files"] --> DataModule["MerlinDataModuleDistributed"]
        DataModule --> DataLoader["Train/Val DataLoaders"]
    end
    
    subgraph "Training Loop"
        DataLoader --> Batch["Tokenized Batches"]
        Batch --> Nicheformer["Nicheformer Model"]
        Nicheformer --> Loss["Masked Language Modeling Loss"]
        Loss --> Optimizer["Optimizer Step"]
        Optimizer --> Nicheformer
    end
    
    subgraph "Logging and Checkpointing"
        Nicheformer --> Metrics["Training Metrics"]
        Metrics --> WandbLogger["Weights & Biases Logger"]
        Nicheformer --> Checkpoint["Model Checkpoint"]
    end
```

Sources: [src/nicheformer/_train.py:38-80]()

## Fine-tuning for Downstream Tasks

After pretraining, the Nicheformer model can be fine-tuned for specific downstream tasks. This is implemented in the `fine_tune_predictions` function.

### Fine-tuning Process

```mermaid
flowchart TB
    subgraph "Model Initialization"
        CheckpointPath["Checkpoint Path"] --> LoadModel["Load Pretrained Nicheformer"]
        LoadModel --> FineTuneModel["Create FineTuningModel"]
        Config["Task Configuration"] --> FineTuneModel
    end
    
    subgraph "Fine-tuning"
        FineTuneModel --> Trainer["PyTorch Lightning Trainer"]
        DataModule["MerlinDataModuleDistributed"] --> Trainer
        Trainer --> TrainedModel["Fine-tuned Model"]
    end
    
    subgraph "Prediction Generation"
        TrainedModel --> Predict["Generate Predictions"]
        TestDataLoader["Test DataLoader"] --> Predict
        Predict --> SavePredictions["Save Predictions"]
    end
```

Sources: [src/nicheformer/_fine_tune.py:12-129]()

### Fine-tuning Configuration

The fine-tuning process uses the following key configuration parameters:

| Parameter | Description |
|-----------|-------------|
| `checkpoint_path` | Path to the pretrained Nicheformer model checkpoint |
| `freeze` | Whether to freeze the backbone model parameters |
| `extract_layers` | Which layers to extract features from |
| `function_layers` | Task-specific functional layers |
| `reinit_layers` | Whether to reinitialize certain layers |
| `extractor` | Feature extraction method |
| `baseline` | Whether to use a baseline model |
| `supervised_task` | Type of downstream task (e.g., 'niche_regression', 'niche_classification', 'density_regression') |
| `regress_distribution` | Whether to regress a distribution |
| `pool` | Pooling method for token representations |
| `dim_prediction` | Dimension of the prediction output |
| `n_classes` | Number of classes for classification tasks |
| `predict_density` | Whether to predict density |
| `organ` | Organ/tissue type for data selection |
| `label` | Label key for supervision |

Sources: [src/nicheformer/_fine_tune.py:21-40]()

### Downstream Tasks

The Nicheformer model can be fine-tuned for various downstream tasks:

1. **Niche Regression**: Predicting cellular neighborhood compositions
2. **Cell Type Classification**: Predicting cell types based on gene expression
3. **Density Regression**: Predicting cell density in spatial coordinates

```mermaid
flowchart TB
    subgraph "Foundation Model"
        PretrainedModel["Pretrained Nicheformer"]
    end
    
    subgraph "Task-specific Models"
        PretrainedModel --> NicheRegression["Niche Regression Model"]
        PretrainedModel --> CellTypeClassification["Cell Type Classification Model"]
        PretrainedModel --> DensityRegression["Density Regression Model"]
    end
    
    subgraph "Predictions"
        NicheRegression --> NichePredictions["Predicted Niche Compositions"]
        CellTypeClassification --> CellTypePredictions["Predicted Cell Types"]
        DensityRegression --> DensityPredictions["Predicted Cell Densities"]
    end
```

Sources: [src/nicheformer/_fine_tune.py:21-40](), [src/nicheformer/_fine_tune.py:110-121]()

## Prediction Generation and Saving

After fine-tuning, the model can generate predictions on test data, which are collected and saved for downstream analysis:

```mermaid
sequenceDiagram
    participant Model as "Fine-tuned Model"
    participant DataLoader as "Test DataLoader"
    participant Batch as "Data Batch"
    participant Predictions as "Predictions"
    participant Storage as "Storage"
    
    DataLoader->>Batch: Get next batch
    Batch->>Model: Process batch
    Model->>Predictions: Generate predictions
    Predictions->>Predictions: Accumulate results
    
    loop For each batch
        DataLoader->>Batch: Get next batch
        Batch->>Model: Process batch
        Model->>Predictions: Generate predictions
        Predictions->>Predictions: Accumulate results
    end
    
    Predictions->>Storage: Save predictions
    Predictions->>Storage: Save labels
    Predictions->>Storage: Save indices
```

Sources: [src/nicheformer/_fine_tune.py:99-129]()

## Monitoring and Logging

Both pretraining and fine-tuning processes use PyTorch Lightning's Trainer interface with Weights & Biases (wandb) logging for experiment tracking:

```mermaid
flowchart LR
    subgraph "Training Infrastructure"
        Model["Nicheformer/FineTuningModel"] --> Trainer["PyTorch Lightning Trainer"]
        DataModule["MerlinDataModuleDistributed"] --> Trainer
    end
    
    subgraph "Callbacks"
        Checkpoint["ModelCheckpoint"] --> Trainer
        LRMonitor["LearningRateMonitor"] --> Trainer
    end
    
    subgraph "Logging"
        Trainer --> WandbLogger["WandbLogger"]
        WandbLogger --> Dashboard["Weights & Biases Dashboard"]
    end
    
    subgraph "Artifacts"
        Trainer --> ModelCheckpoints["Model Checkpoints"]
        Trainer --> MetricsLog["Training Metrics"]
    end
```

Sources: [src/nicheformer/_train.py:38-55](), [src/nicheformer/_fine_tune.py:49-64]()

## Hardware Requirements and Performance Optimizations

The training and fine-tuning processes utilize several performance optimizations:

1. **Distributed Data Parallel (DDP)**: Training across multiple GPUs
2. **Mixed Precision**: Using BF16 mixed precision for faster training
3. **Gradient Clipping**: Preventing exploding gradients
4. **Gradient Accumulation**: Effective larger batch sizes with limited memory

These optimizations allow efficient training on large datasets while maintaining numerical stability.

Sources: [src/nicheformer/_train.py:43-55](), [src/nicheformer/_fine_tune.py:54-64]()

---