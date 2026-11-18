# Training System

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [scgpt/trainer.py](scgpt/trainer.py)

</details>



This document provides a technical overview of the training infrastructure in scGPT, explaining the core components responsible for training, evaluation, and testing workflows. For information about model architecture, see [Model Architecture](#2.1), and for data processing before training, see [Data Processing](#2.2).

## Overview

The scGPT training system consists of several core components designed to facilitate model training across different tasks including cell type annotation, dataset integration, perturbation prediction, and multi-omics analysis. The system handles the entire workflow from data preparation through training loops, evaluation, and performance reporting.

```mermaid
flowchart TD
    subgraph "Data Preparation"
        prepData["prepare_data()"] --> tokenizedData["Tokenized Data"]
        tokenizedData --> prepDataloader["prepare_dataloader()"]
        prepDataloader --> DataLoader["PyTorch DataLoaders"]
    end
    
    subgraph "Training Loop"
        DataLoader --> trainFunc["train()"]
        trainFunc --> Optimizer["Optimizer Updates"]
        Optimizer --> Model["TransformerModel"]
        Model --> Loss["Loss Calculation"]
        Loss --> Optimizer
    end
    
    subgraph "Evaluation"
        DataLoader --> evalFunc["evaluate()"]
        evalFunc --> valMetrics["Validation Metrics"]
        DataLoader --> testFunc["test()"]
        testFunc --> testMetrics["Test Metrics"]
    end
    
    subgraph "Monitoring"
        trainFunc --> WandB["Weights & Biases"]
        evalFunc --> WandB
        testFunc --> WandB
    end
```

Sources: [scgpt/trainer.py:1-686]()

## Data Preparation for Training

Before training, data must be prepared with proper tokenization, masking, and batching. This preparation flow is crucial for the model to effectively learn from the data.

### Masked Value Preparation

A key part of scGPT's training approach is random masking of gene expression values, which allows the model to learn to predict these values:

```mermaid
flowchart LR
    subgraph "Masking Process"
        input["Input Gene Expression Data"] --> tokenize["tokenize_and_pad_batch()"]
        tokenize --> tokens["Tokenized Gene Values"]
        tokens --> mask["random_mask_value()"]
        mask --> masked["Masked Values"]
        masked --> ready["Training-Ready Data"]
    end
    
    subgraph "Masked Data Structure"
        GeneIDs["Gene IDs"]
        Values["Masked Values"]
        TargetValues["Target Values"]
        BatchLabels["Batch Labels"]
        CelltypeLabels["Celltype Labels (optional)"]
        ModTypes["Modality Types (optional)"]
    end
```

Sources: [scgpt/trainer.py:24-115]()

### Data Loaders

The `prepare_dataloader` function creates PyTorch DataLoaders with support for batch-aware sampling, which is important for tasks involving multiple batches or domains:

```mermaid
flowchart TD
    data["Training Data Dict"] --> SeqDataset["SeqDataset Class"]
    SeqDataset --> checkSample{"per_seq_batch_sample?"}
    checkSample -->|"Yes"| subsetSampler["SubsetsBatchSampler"]
    checkSample -->|"No"| standardLoader["Standard DataLoader"]
    subsetSampler --> finalLoader["Batch-Aware DataLoader"]
    standardLoader --> regularLoader["Regular DataLoader"]
```

Sources: [scgpt/trainer.py:118-170]()

## Training Process

The training process orchestrates the model training cycle, including forward passes, loss calculation, gradient computation and optimization steps.

### Training Loop

The `train` function in [scgpt/trainer.py:173-371]() handles the training loop for a single epoch:

```mermaid
flowchart TD
    startTrain["Start Training Epoch"] --> iterateLoader["Iterate Through DataLoader"]
    iterateLoader --> forwardPass["Forward Pass through Model"]
    forwardPass --> computeLoss["Compute Loss Components"]
    computeLoss --> backprop["Gradient Calculation & Backpropagation"]
    backprop --> gradClip["Gradient Clipping"]
    gradClip --> optimStep["Optimizer Step"]
    optimStep --> scaleUpdate["Scaler Update (if AMP enabled)"]
    scaleUpdate --> logMetrics["Log Metrics"]
    logMetrics --> nextBatch{"More Batches?"}
    nextBatch -->|"Yes"| iterateLoader
    nextBatch -->|"No"| endEpoch["End Training Epoch"]
```

Sources: [scgpt/trainer.py:173-371]()

### Loss Components

The training system supports multiple loss components depending on the configured training objectives:

| Loss Component | Description | Configuration Flag |
|----------------|-------------|-------------------|
| GEP (Gene Expression Prediction) | MSE loss for masked gene values prediction | `config.GEP` |
| GEPC/MVC (Masked Value Prediction) | Secondary prediction task for cell embedding | `config.GEPC` |
| CLS (Classification) | Cell type classification loss | `config.CLS` |
| ECS (Elastic Cell Similarity) | Cell similarity constraint | `config.ESC` |
| DAB/DAR (Domain Adversarial) | Batch/domain adversarial loss | `config.DAR` |
| Zero-log probability | Bernoulli loss for explicit zero modeling | `config.explicit_zero_prob` |

Sources: [scgpt/trainer.py:235-289](), [examples/finetune_integration.py:466-497]()

### Optimization and Scaling

The training system uses:
- Adam optimizer with configurable learning rate
- Learning rate scheduling
- Gradient clipping for stability
- Automatic Mixed Precision (AMP) for faster training with lower memory usage

Sources: [scgpt/trainer.py:295-309](), [examples/finetune_integration.py:431-435]()

## Evaluation and Testing

The evaluation system provides mechanisms to assess model performance on validation and test data.

### Validation Process

The `evaluate` function computes validation metrics without updating model parameters:

```mermaid
flowchart LR
    validData["Validation Data"] --> evalLoader["Evaluation DataLoader"]
    evalLoader --> modelEval["Model.eval()"]
    modelEval --> noGrad["torch.no_grad() Context"]
    noGrad --> forwardPass["Forward Pass"]
    forwardPass --> calcLoss["Calculate Loss"]
    calcLoss --> calcMetrics["Calculate Metrics"]
    calcMetrics --> logResults["Log Results"]
```

Sources: [scgpt/trainer.py:379-461]()

### Test Process and Metrics

For final evaluation on test data, the system provides:

1. The `test` function for classification tasks, calculating:
   - Accuracy, Precision, Recall
   - Macro and Micro F1 scores

2. The `eval_testdata` function for integration tasks, calculating:
   - scIB metrics for batch integration quality
   - UMAP visualizations of embeddings

Sources: [scgpt/trainer.py:507-686]()

## Configuration and Training Tasks

The scGPT training system is highly configurable to support different training tasks:

| Task | Description | Configuration |
|------|-------------|--------------|
| Annotation | Cell type annotation/classification | `config.task = "annotation"` |
| Integration | Dataset integration across batches | `config.task = "integration"` |
| Perturbation | Cellular perturbation prediction | `config.task = "perturb"` |
| Multi-omics | Integration of multiple omics data types | `config.task = "multiomic"` |

Each task utilizes different combinations of the loss functions and training objectives.

Sources: [scgpt/trainer.py:35](), [examples/finetune_integration.py:49-73]()

## Training Workflow

The complete training workflow combines these components into a coordinated process:

```mermaid
flowchart TD
    startTraining["Start Training"] --> loadData["Load & Preprocess Data"]
    loadData --> prepareModel["Initialize/Load Model"]
    prepareModel --> beginLoop["Begin Training Loop"]
    
    subgraph "Per Epoch"
        beginLoop --> prepareDataEpoch["Prepare Data for Epoch"]
        prepareDataEpoch --> createLoaders["Create DataLoaders"]
        createLoaders --> trainEpoch["Train for One Epoch"]
        trainEpoch --> evaluate["Evaluate on Validation Set"]
        evaluate --> checkBest{"Best Model?"}
        checkBest -->|"Yes"| saveModel["Save Best Model"]
        checkBest -->|"No"| checkInterval{"Save Interval?"}
        saveModel --> checkInterval
        checkInterval -->|"Yes"| evalTest["Evaluate on Test Data"]
        checkInterval -->|"No"| updateLR["Update Learning Rate"]
        evalTest --> updateLR
        updateLR --> nextEpoch{"More Epochs?"}
        nextEpoch -->|"Yes"| prepareDataEpoch
        nextEpoch -->|"No"| endTraining["End Training"]
    end
```

Sources: [examples/finetune_integration.py:708-780]()

## Fine-tuning Existing Models

The system supports fine-tuning pretrained models for specific tasks:

```mermaid
flowchart LR
    pretrainedModel["Pretrained Model"] --> loadWeights["Load Model Weights"]
    loadWeights --> compatCheck["Check Compatibility"]
    compatCheck -->|"Fully Compatible"| loadAll["Load All Parameters"]
    compatCheck -->|"Partially Compatible"| loadPartial["Load Matching Parameters"]
    loadAll --> continueTraining["Continue Training"]
    loadPartial --> continueTraining
```

Sources: [examples/finetune_integration.py:406-422]()

## Monitoring and Visualization

The training system integrates with Weights & Biases (wandb) for experiment tracking:

1. Defining metrics for tracking:
   ```python
   wandb.define_metric("valid/loss", summary="min", step_metric="epoch")
   wandb.define_metric("test/avg_bio", summary="max")
   ```

2. Logging training metrics per batch:
   ```python
   wandb.log(metrics_to_log)  # Log loss components
   ```

3. Visualization outputs:
   - UMAP plots for batch effect assessment
   - UMAP plots for cell type visualization
   - Training and validation curves

Sources: [scgpt/trainer.py:372-378](), [examples/finetune_integration.py:550-556](), [examples/finetune_integration.py:767-778]()

## Usage Example

A typical training workflow follows these steps:

1. Configure training settings (batch size, learning rate, etc.)
2. Prepare and tokenize data
3. Initialize the model or load a pretrained model
4. Create optimizer and scheduler
5. Run the training loop for specified epochs
6. Evaluate and save the best performing model
7. Generate and visualize results

For a complete example, see the `finetune_integration.py` script in the examples directory.

Sources: [examples/finetune_integration.py:1-797]()

---