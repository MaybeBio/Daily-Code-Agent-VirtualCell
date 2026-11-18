# Perturbation Prediction

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/tutorial_reference_mapping.nblink](docs/tutorial_reference_mapping.nblink)
- [scgpt/model/generation_model.py](scgpt/model/generation_model.py)
- [scgpt/model/model.py](scgpt/model/model.py)
- [scgpt/utils/util.py](scgpt/utils/util.py)
- [tutorials/Tutorial_Perturbation.ipynb](tutorials/Tutorial_Perturbation.ipynb)

</details>



This page documents how to fine-tune scGPT for predicting the effects of genetic perturbations on cell state. Given a control cell's gene expression profile and a perturbation specification (e.g., gene knockout, overexpression), the model predicts the resulting gene expression profile after the perturbation is applied.

**Scope**: This page covers the `TransformerGenerator` model architecture, the affine expression decoder, perturbation encoding, training procedures, and evaluation metrics specific to perturbation prediction tasks. For general cell type classification, see [Cell Type Annotation](#6.1). For analyzing gene regulatory relationships, see [Gene Regulatory Network Analysis](#6.4).

## Overview

Perturbation prediction is a supervised fine-tuning task where scGPT learns to predict how genetic interventions alter cellular gene expression. This capability is critical for understanding gene function, designing therapeutic interventions, and modeling cellular responses to perturbations.

The key innovation is the `TransformerGenerator` class, which extends the base transformer architecture with:
- **Perturbation encoding**: Explicit flags indicating which genes are perturbed
- **Affine decoder**: Predicts expression as `coefficient * input_value + bias`
- **Specialized inference method**: `pred_perturb()` for generating predictions

Sources: [scgpt/model/generation_model.py:25-350](), [tutorials/Tutorial_Perturbation.ipynb:1-100]()

## TransformerGenerator Architecture

### Component Diagram

```mermaid
graph TB
    subgraph "Input Encoders"
        GE["GeneEncoder<br/>gene token embeddings"]
        VE["ContinuousValueEncoder<br/>expression values"]
        PE["PerturbationEncoder<br/>nn.Embedding(3, d_model)<br/>padding_idx=pert_pad_id"]
    end
    
    subgraph "Transformer Core"
        TE["TransformerEncoder<br/>12 layers, 512 dim<br/>FlashAttention or standard"]
    end
    
    subgraph "Decoders"
        AED["AffineExprDecoder<br/>coeff_decoder + bias_decoder"]
        CLS["ClsDecoder<br/>optional classification"]
        MVC["MVCDecoder<br/>optional MVC objective"]
    end
    
    INPUT["Input:<br/>gene_ids, values,<br/>pert_flags, mask"] --> GE
    INPUT --> VE
    INPUT --> PE
    
    GE --> SUM["total_embs = gene + value + pert"]
    VE --> SUM
    PE --> SUM
    
    SUM --> TE
    TE --> TRANS_OUT["transformer_output<br/>(batch, seq_len, d_model)"]
    
    TRANS_OUT --> AED
    TRANS_OUT --> CLS
    TRANS_OUT --> MVC
    
    AED --> PRED["pred = coeff * input + bias<br/>(batch, seq_len)"]
    
    style PE fill:#ffcccc
    style AED fill:#ccffcc
    style PRED fill:#ccccff
```

**TransformerGenerator vs TransformerModel**

The `TransformerGenerator` class differs from the base `TransformerModel` in three critical ways:

| Component | TransformerModel | TransformerGenerator |
|-----------|------------------|---------------------|
| **Perturbation Input** | None | `pert_encoder: nn.Embedding(3, d_model)` |
| **Decoder** | `ExprDecoder` (direct prediction) | `AffineExprDecoder` (coefficient + bias) |
| **Inference Method** | `generate()` for autoregressive | `pred_perturb()` for one-shot prediction |

Sources: [scgpt/model/generation_model.py:25-124](), [scgpt/model/model.py:28-160]()

### Affine Expression Decoder

The `AffineExprDecoder` predicts perturbed expression as an affine transformation of the input expression:

```
pred_expression = coefficient * input_expression + bias
```

This formulation is biologically motivated: perturbations often scale existing expression (coefficient) and add a shift term (bias).

```mermaid
graph LR
    subgraph "AffineExprDecoder"
        INPUT["transformer_output<br/>(batch, seq_len, d_model)"]
        VALUES["input values<br/>(batch, seq_len)"]
        
        INPUT --> COEFF_DEC["coeff_decoder<br/>ExprDecoder"]
        INPUT --> BIAS_DEC["bias_decoder<br/>ExprDecoder"]
        
        COEFF_DEC --> COEFF["coefficient<br/>(batch, seq_len)"]
        BIAS_DEC --> BIAS["bias<br/>(batch, seq_len)"]
        
        COEFF --> MULT["multiply"]
        VALUES --> MULT
        MULT --> ADD["add"]
        BIAS --> ADD
        
        ADD --> OUTPUT["predicted expression<br/>(batch, seq_len)"]
    end
    
    style COEFF fill:#ffffcc
    style BIAS fill:#ffffcc
```

**Optional Activations**: The decoder supports optional activation functions (e.g., `ReLU`, `Sigmoid`) and adaptive bias scaling based on non-zero expression means.

Sources: [scgpt/model/generation_model.py:376-442]()

## Training Workflow

### Data Preparation

The training pipeline uses the GEARS framework's `PertData` class to manage perturbation datasets. The data includes:
- **Control cells**: Baseline gene expression
- **Perturbed cells**: Expression after genetic intervention
- **Perturbation flags**: Binary indicators (0=control, 1=perturbed, 2=padding)
- **DE genes**: Differentially expressed genes for evaluation

```mermaid
sequenceDiagram
    participant Data as PertData
    participant Loader as DataLoader
    participant Model as TransformerGenerator
    participant Loss as masked_mse_loss
    
    Data->>Data: load(data_name="adamson"/"norman")
    Data->>Data: prepare_split(split="simulation")
    Data->>Loader: get_dataloader(batch_size=64)
    
    loop Training Epoch
        Loader->>Loader: batch_data with x, y, pert_flags
        Loader->>Loader: select genes (all or batch-wise)
        Loader->>Model: forward(gene_ids, values, pert_flags)
        Model->>Model: encode: gene + value + pert embeddings
        Model->>Model: transformer_encoder
        Model->>Model: AffineExprDecoder(output, values)
        Model->>Loss: pred vs target with mask
        Loss->>Model: backward & optimize
    end
```

Sources: [tutorials/Tutorial_Perturbation.ipynb:180-245](), [tutorials/Tutorial_Perturbation.ipynb:711-807]()

### Training Configuration

Key hyperparameters from the tutorial:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `load_model` | `"../save/scGPT_human"` | Pretrained checkpoint |
| `load_param_prefixs` | `["encoder", "value_encoder", "transformer_encoder"]` | Freeze decoder layers initially |
| `lr` | `1e-4` | Learning rate |
| `batch_size` | `64` | Training batch size |
| `epochs` | `15` | Training epochs |
| `max_seq_len` | `1536` | Maximum genes per batch |
| `include_zero_gene` | `"all"` or `"batch-wise"` | Include zero-expression genes |
| `amp` | `True` | Automatic mixed precision |

The `include_zero_gene` setting controls whether to predict all genes or only expressed genes:
- `"all"`: Predict all genes in vocabulary (slower, more comprehensive)
- `"batch-wise"`: Predict only non-zero genes in current batch (faster, focused)

Sources: [tutorials/Tutorial_Perturbation.ipynb:66-121]()

### Training Loop Implementation

The training loop follows a standard supervised learning pattern with perturbation-specific data handling:

```mermaid
flowchart TD
    START["Start Training"] --> BATCH["Get batch_data"]
    BATCH --> EXTRACT["Extract:<br/>ori_gene_values (batch, n_genes)<br/>pert_flags (batch, n_genes)<br/>target_gene_values (batch, n_genes)"]
    
    EXTRACT --> SELECT{"include_zero_gene?"}
    SELECT -->|"all"| ALL["input_gene_ids = all genes"]
    SELECT -->|"batch-wise"| NONZERO["input_gene_ids = nonzero genes"]
    
    ALL --> SUBSAMPLE{"len > max_seq_len?"}
    NONZERO --> SUBSAMPLE
    SUBSAMPLE -->|yes| RAND["Random sample max_seq_len genes"]
    SUBSAMPLE -->|no| KEEP["Keep all selected genes"]
    
    RAND --> MAP["map_raw_id_to_vocab_id()"]
    KEEP --> MAP
    
    MAP --> FORWARD["model.forward(mapped_ids, values, pert_flags)"]
    FORWARD --> LOSS["masked_mse_loss(pred, target, mask=all_ones)"]
    LOSS --> BACKWARD["scaler.scale(loss).backward()"]
    BACKWARD --> CLIP["clip_grad_norm_(params, 1.0)"]
    CLIP --> STEP["scaler.step(optimizer)"]
    STEP --> UPDATE["scaler.update()"]
    
    UPDATE --> CHECK{"More batches?"}
    CHECK -->|yes| BATCH
    CHECK -->|no| END["End Epoch"]
```

**Key Implementation Details**:
1. **Gene ID Mapping**: Raw gene indices are mapped to vocabulary IDs using `map_raw_id_to_vocab_id()` [scgpt/utils/util.py:273-305]()
2. **Masking**: Unlike MLM training, all positions are used for loss computation (`masked_positions = torch.ones_like(input_values)`)
3. **Gradient Scaling**: Uses `torch.cuda.amp.GradScaler` for mixed precision training
4. **Padding Mask**: Created as `torch.zeros_like(input_values, dtype=torch.bool)` since no actual padding exists

Sources: [tutorials/Tutorial_Perturbation.ipynb:711-807]()

## Inference and Prediction

### pred_perturb Method

The `pred_perturb()` method is the primary inference interface for generating predictions:

```mermaid
graph TB
    INPUT["batch_data:<br/>x (ori_values, pert_flags)<br/>pert (condition names)"]
    
    INPUT --> EXTRACT["Extract:<br/>ori_gene_values<br/>pert_flags"]
    
    EXTRACT --> SELECT{"include_zero_gene"}
    SELECT -->|"all"| ALL["input_gene_ids = all genes"]
    SELECT -->|"batch-wise"| NONZERO["input_gene_ids = nonzero<br/>unique().sort()"]
    
    ALL --> SUBSET["input_values = ori_values[:, ids]<br/>input_pert_flags = pert_flags[:, ids]"]
    NONZERO --> SUBSET
    
    SUBSET --> MAP["mapped_ids = map_raw_id_to_vocab_id(ids, gene_ids)"]
    MAP --> REPEAT["mapped_ids.repeat(batch_size, 1)"]
    
    REPEAT --> FORWARD["forward() with amp"]
    FORWARD --> OUTPUT["output_dict['mlm_output']"]
    
    OUTPUT --> RECONSTRUCT["pred_gene_values = zeros_like(ori_values)<br/>pred[:, input_gene_ids] = output"]
    RECONSTRUCT --> RETURN["Return full prediction<br/>(batch, n_genes)"]
    
    style FORWARD fill:#ccffcc
    style RECONSTRUCT fill:#ffcccc
```

**Method Signature**:
```python
def pred_perturb(
    self,
    batch_data,
    include_zero_gene="batch-wise",
    gene_ids=None,
    amp=True,
) -> Tensor
```

Sources: [scgpt/model/generation_model.py:295-350]()

### Evaluation Function

The `eval_perturb()` function runs the model on a test set and collects predictions:

**Workflow**:
1. Set model to `eval()` mode
2. Iterate through test data loader
3. For each batch, call `pred_perturb()`
4. Extract predictions for differentially expressed (DE) genes using `batch.de_idx`
5. Stack all predictions and ground truth
6. Return dictionary with `pert_cat`, `pred`, `truth`, `pred_de`, `truth_de`

Sources: [tutorials/Tutorial_Perturbation.ipynb:809-858]()

## Evaluation Metrics

### compute_perturbation_metrics

The primary evaluation function computes Pearson correlations across genes and conditions:

```mermaid
graph TB
    subgraph "Input"
        RESULTS["results dict:<br/>pred, truth, pert_cat"]
        CTRL["ctrl_adata:<br/>control expression"]
    end
    
    RESULTS --> CALC1["Compute mean expression<br/>per condition"]
    CTRL --> MEAN_CTRL["mean_ctrl = ctrl.X.mean(0)"]
    
    CALC1 --> MEAN_PERT["true_mean_perturbed<br/>pred_mean_perturbed"]
    MEAN_PERT --> DELTA["true_delta = true - ctrl<br/>pred_delta = pred - ctrl"]
    
    subgraph "Metrics Across Genes"
        DELTA --> CORR1["pearson(true, pred)<br/>per condition"]
        DELTA --> CORR2["pearson(true_delta, pred_delta)<br/>per condition"]
        CORR1 --> M1["pearson"]
        CORR2 --> M2["pearson_delta"]
    end
    
    subgraph "Metrics on DE Genes"
        DELTA --> DE["Extract DE genes<br/>from rank_genes_groups"]
        DE --> CORR3["pearson(true_de, pred_de)<br/>per condition"]
        DE --> CORR4["pearson(true_delta_de, pred_delta_de)<br/>per condition"]
        CORR3 --> M3["pearson_de"]
        CORR4 --> M4["pearson_de_delta"]
    end
    
    M1 --> OUTPUT["Return mean metrics<br/>or raw per-condition"]
    M2 --> OUTPUT
    M3 --> OUTPUT
    M4 --> OUTPUT
    
    style CORR1 fill:#ccffff
    style CORR2 fill:#ccffff
    style CORR3 fill:#ffffcc
    style CORR4 fill:#ffffcc
```

**Metric Definitions**:

| Metric | Description | Biological Meaning |
|--------|-------------|-------------------|
| `pearson` | Correlation between predicted and true expression across genes | Overall prediction accuracy |
| `pearson_de` | Correlation on differentially expressed genes only | Accuracy on most affected genes |
| `pearson_delta` | Correlation of expression changes (perturbed - control) | Ability to predict perturbation effects |
| `pearson_de_delta` | Correlation of changes on DE genes | Accuracy of predicted differential effects |

**Optional Parameters**:
- `non_zero_genes=True`: Only consider non-zero genes in ground truth
- `return_raw=True`: Return per-condition correlations instead of means

Sources: [scgpt/utils/util.py:429-596]()

### Example Results

From the tutorial, typical metrics after fine-tuning on the Adamson dataset:

```
Epoch 7 validation metrics:
- pearson: 0.9904
- pearson_de: 0.9573
- pearson_delta: 0.6970
- pearson_de_delta: 0.8659
```

The high `pearson` and `pearson_de` indicate excellent prediction of absolute expression levels. The lower `pearson_delta` reflects the inherent difficulty of predicting expression changes, which is a more challenging task.

Sources: [tutorials/Tutorial_Perturbation.ipynb:949]()

## Key Implementation Details

### Perturbation Flags

Perturbation flags are encoded as integers:
- `0`: Padding token (default for `pert_encoder`)
- `1`: Gene is perturbed (knockout/overexpression)
- `2`: Gene is not perturbed (control)

The flags are embedded using `nn.Embedding(3, d_model, padding_idx=0)` and added to gene and value embeddings.

Sources: [scgpt/model/generation_model.py:83](), [tutorials/Tutorial_Perturbation.ipynb:75]()

### Model Initialization

When fine-tuning from a pretrained checkpoint, typically only the encoder components are loaded:

```python
load_param_prefixs = [
    "encoder",              # GeneEncoder
    "value_encoder",        # ContinuousValueEncoder
    "transformer_encoder",  # TransformerEncoder layers
]
```

This allows the `pert_encoder` and `AffineExprDecoder` to be trained from scratch, adapting to the perturbation task while leveraging pretrained gene and expression representations.

Sources: [tutorials/Tutorial_Perturbation.ipynb:87-91](), [tutorials/Tutorial_Perturbation.ipynb:663-693]()

### Loss Function

The training uses `masked_mse_loss` with all positions unmasked:

```python
masked_positions = torch.ones_like(input_values, dtype=torch.bool)
loss = masked_mse_loss(output_values, target_values, masked_positions)
```

This is equivalent to standard MSE loss but uses the same interface as other scGPT training objectives.

Sources: [tutorials/Tutorial_Perturbation.ipynb:765-768]()

## Dataset Support

The tutorial demonstrates integration with the GEARS framework's perturbation datasets:

**Supported Datasets**:
- `"adamson"`: K562 cells with single and combinatorial CRISPR perturbations
- `"norman"`: Chronic myelogenous leukemia cells with perturbations
- Other datasets compatible with GEARS `PertData` interface

**Split Types**:
- `"simulation"`: Simulated train/val/test splits
- `"simulation_single"`: Single perturbations only
- Custom splits via GEARS

Sources: [tutorials/Tutorial_Perturbation.ipynb:114-184]()

## Summary

The perturbation prediction task uses the specialized `TransformerGenerator` architecture with three key innovations:

1. **Perturbation Encoding**: Explicit flags indicate which genes are perturbed
2. **Affine Decoder**: Models expression as `coefficient * input + bias`
3. **Specialized Inference**: `pred_perturb()` method for efficient prediction

The training workflow follows standard supervised learning with perturbation-specific data handling, and evaluation focuses on Pearson correlations for both absolute expression and expression changes. Fine-tuning from pretrained scGPT checkpoints typically achieves >0.99 correlation on absolute expression and >0.85 on expression changes.

Sources: [scgpt/model/generation_model.py:25-350](), [tutorials/Tutorial_Perturbation.ipynb:1-1200](), [scgpt/utils/util.py:429-596]()

---