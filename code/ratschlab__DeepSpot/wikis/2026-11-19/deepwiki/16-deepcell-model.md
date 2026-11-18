# DeepCell Model

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.codeboarding/Core Utilities.md](.codeboarding/Core Utilities.md)
- [.codeboarding/DeepSpot Models.md](.codeboarding/DeepSpot Models.md)
- [deepspot/cell/model.py](deepspot/cell/model.py)

</details>



## Purpose and Scope

The `DeepCell` model is the core neural network architecture for predicting gene expression at single-cell resolution from H&E histology images. This page documents the `DeepCell` class implementation, including its initialization parameters, cell context modes, forward pass mechanisms, and training lifecycle.

For information about the underlying Phi and Rho neural layer implementations, see [Phi and Rho Neural Layers](#5.3). For details about available loss functions, see [Loss Functions](#5.4). For the spot-level equivalent model, see [DeepSpot Model](#5.2).

**Sources:** [deepspot/cell/model.py:63-281]()

---

## Class Overview

The `DeepCell` class inherits from `lightning.LightningModule`, leveraging PyTorch Lightning's training orchestration capabilities. The model implements a deep-set architecture that processes cell-level histology features and optionally incorporates spatial neighbor information to predict gene expression profiles.

```mermaid
graph TB
    LightningModule["lightning.LightningModule<br/>(PyTorch Lightning)"]
    DeepCell["DeepCell<br/>deepspot.cell.model.DeepCell"]
    
    PhiModules["phi_cell: nn.ModuleList<br/>(n_ensemble_phi instances)"]
    RhoModules["rho: nn.ModuleList<br/>(n_ensemble_rho instances)"]
    LossFunc["loss_func<br/>(selected from deepspot.cell.loss)"]
    
    LightningModule -->|"inherits from"| DeepCell
    DeepCell -->|"initializes"| PhiModules
    DeepCell -->|"initializes"| RhoModules
    DeepCell -->|"configures"| LossFunc
    
    PhiModules -.->|"feature transformation"| RhoModules
    RhoModules -.->|"gene expression output"| Predictions["Predicted Gene Counts"]
```

**Core Responsibilities:**
- Initialize ensemble of Phi and Rho neural network modules
- Configure loss function and optimizer
- Execute forward passes based on cell context mode
- Track training and validation losses
- Provide loss visualization capabilities

**Sources:** [deepspot/cell/model.py:63-128]()

---

## Initialization Parameters

The `DeepCell.__init__` method accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | `int` | - | Dimension of input feature vectors (foundation model output) |
| `output_size` | `int` | - | Number of genes to predict |
| `loss_func` | `str` | `"mse"` | Loss function identifier: `"mse"`, `"cos"`, `"l1"`, `"huber"`, `"mse_cos"`, `"l1_cos"`, `"huber_cos"`, `"pearson"`, `"mse_pearson"`, `"poisson"` |
| `lr` | `float` | `1e-4` | Learning rate for AdamW optimizer |
| `p` | `float` | `0.2` | Default dropout probability for Phi and Rho |
| `p_phi` | `Union[None, float]` | `None` | Dropout probability for Phi layers (overrides `p` if set) |
| `p_rho` | `Union[None, float]` | `None` | Dropout probability for Rho layers (overrides `p` if set) |
| `n_ensemble` | `int` | `10` | Default number of ensemble models |
| `n_ensemble_phi` | `Union[None, int]` | `None` | Number of Phi ensemble models (overrides `n_ensemble` if set) |
| `n_ensemble_rho` | `Union[None, int]` | `None` | Number of Rho ensemble models (overrides `n_ensemble` if set) |
| `phi2rho_size` | `int` | `1024` | Dimension of Phi output / Rho input |
| `emb_size` | `int` | `1024` | Embedding size (stored but not actively used) |
| `weight_decay` | `float` | `1e-6` | L2 regularization weight for AdamW |
| `random_seed` | `int` | `2024` | Random seed for reproducibility |
| `agg_neighbors` | `str` | `"max"` | Aggregation operation for neighbor features: `"max"`, `"mean"`, `"sum"` |
| `scaler` | - | `None` | Optional scaler object for inverse transformation |
| `cell_context` | `str` | `"cell_neighbors"` | Forward pass mode: `"cell"` or `"cell_neighbors"` |

**Loss Function Mapping:**

The `loss_func` parameter maps to functions in [deepspot/cell/loss.py]():

```mermaid
graph LR
    mse["loss_func='mse'"]
    cos["loss_func='cos'"]
    l1["loss_func='l1'"]
    huber["loss_func='huber'"]
    mse_cos["loss_func='mse_cos'"]
    l1_cos["loss_func='l1_cos'"]
    huber_cos["loss_func='huber_cos'"]
    pearson["loss_func='pearson'"]
    mse_pearson["loss_func='mse_pearson'"]
    poisson["loss_func='poisson'"]
    
    mse -->|"assigns"| loss_mse_function
    cos -->|"assigns"| loss_cosine_function
    l1 -->|"assigns"| loss_l1_function
    huber -->|"assigns"| loss_huber_function
    mse_cos -->|"assigns"| loss_mse_cosine_function
    l1_cos -->|"assigns"| loss_l1_cosine_function
    huber_cos -->|"assigns"| loss_huber_cosine_function
    pearson -->|"assigns"| loss_pearson_function
    mse_pearson -->|"assigns"| loss_mse_pearson_function
    poisson -->|"assigns"| loss_poisson_function
```

**Sources:** [deepspot/cell/model.py:64-128](), [deepspot/cell/model.py:88-107]()

---

## Cell Context Modes

The `cell_context` parameter determines which forward pass function is used. The model supports two modes:

### Mode Comparison

| Mode | Forward Function | Input Structure | Phi Multiplier | Description |
|------|-----------------|-----------------|----------------|-------------|
| `"cell"` | `_forward_cell` | Single tensor | 1 | Processes individual cell features only |
| `"cell_neighbors"` | `_forward_cell_neighbors` | List of 2 tensors | 2 | Processes cell + aggregated neighbor features |

```mermaid
graph TB
    subgraph "cell_context='cell'"
        CellInput["x: torch.Tensor<br/>(batch_size, 1, input_size)"]
        CellPhi["_apply_phi(x, phi_cell,<br/>operation=NONE)"]
        CellRho["_apply_rho(x_phi)"]
        CellOutput["output: torch.Tensor<br/>(batch_size, output_size)"]
        
        CellInput --> CellPhi
        CellPhi --> CellRho
        CellRho --> CellOutput
    end
    
    subgraph "cell_context='cell_neighbors'"
        NeighInput["x: List[torch.Tensor]<br/>[x_cell, x_neighbors]"]
        NeighCellPhi["_apply_phi(x_cell, phi_cell,<br/>operation=NONE)"]
        NeighNeighPhi["_apply_phi(x_neighbors, phi_cell,<br/>operation=agg_neighbors)"]
        Concat["torch.cat((x_cell, x_neighbors),<br/>dim=1)"]
        NeighRho["_apply_rho(x_phi)"]
        NeighOutput["output: torch.Tensor<br/>(batch_size, output_size)"]
        
        NeighInput --> NeighCellPhi
        NeighInput --> NeighNeighPhi
        NeighCellPhi --> Concat
        NeighNeighPhi --> Concat
        Concat --> NeighRho
        NeighRho --> NeighOutput
    end
```

**Phi Multiplier Impact:**

The `_get_phi_multiplier` method determines the Rho input size based on context:
- `"cell"`: multiplier = 1 → Rho input = `phi2rho_size`
- `"cell_neighbors"`: multiplier = 2 → Rho input = `2 * phi2rho_size`

This is because in `"cell_neighbors"` mode, the cell and neighbor Phi outputs are concatenated before passing to Rho.

**Sources:** [deepspot/cell/model.py:130-142](), [deepspot/cell/model.py:179-188]()

---

## Forward Pass Mechanisms

### Overview of Forward Execution

The `forward` method delegates to the appropriate forward function based on the `cell_context` configuration:

```mermaid
graph TB
    ForwardCall["forward(x)"]
    CheckContext{"self._forward_fn<br/>(set during __init__)"}
    
    ForwardCell["_forward_cell(x)"]
    ForwardCellNeighbors["_forward_cell_neighbors(x)"]
    
    ForwardCall --> CheckContext
    CheckContext -->|"cell_context='cell'"| ForwardCell
    CheckContext -->|"cell_context='cell_neighbors'"| ForwardCellNeighbors
```

### _forward_cell

Processes individual cell features without neighbor information:

1. Apply Phi ensemble to cell features (no aggregation)
2. Apply Rho ensemble to transformed features
3. Return predicted gene expression

**Implementation:** [deepspot/cell/model.py:179-181]()

### _forward_cell_neighbors

Processes cell features with neighbor context:

1. Split input into `x_cell` and `x_neighbors`
2. Apply Phi ensemble to `x_cell` (no aggregation)
3. Apply Phi ensemble to `x_neighbors` (aggregate across neighbors using `agg_neighbors` operation)
4. Concatenate transformed cell and neighbor features
5. Apply Rho ensemble to concatenated features
6. Return predicted gene expression

**Implementation:** [deepspot/cell/model.py:183-188]()

### _apply_phi

Core Phi transformation with ensemble and optional aggregation:

```mermaid
graph TB
    Input["x: (batch_size, sample_size, input_size)"]
    Reshape1["Reshape to (-1, input_size)"]
    PhiEnsemble["Apply n_ensemble_phi Phi modules"]
    MedianAgg["Median aggregation across ensemble"]
    Reshape2["Reshape to (batch_size, sample_size, phi2rho_size)"]
    
    CheckOp{"operation type?"}
    OpSum["Sum across sample_size"]
    OpMean["Mean across sample_size"]
    OpMax["Max across sample_size"]
    OpNone["Flatten to (batch_size, sample_size * phi2rho_size)"]
    
    Output["x_phi"]
    
    Input --> Reshape1
    Reshape1 --> PhiEnsemble
    PhiEnsemble --> MedianAgg
    MedianAgg --> Reshape2
    Reshape2 --> CheckOp
    
    CheckOp -->|"SUM"| OpSum
    CheckOp -->|"MEAN"| OpMean
    CheckOp -->|"MAX"| OpMax
    CheckOp -->|"NONE"| OpNone
    
    OpSum --> Output
    OpMean --> Output
    OpMax --> Output
    OpNone --> Output
```

**Key Behavior:**
- Applies each Phi module independently to all samples
- Uses **median aggregation** across the ensemble (robust to outliers)
- `Operation.NONE`: Used for target cell features (preserves spatial structure before Rho)
- `Operation.MAX/MEAN/SUM`: Used for neighbor aggregation

**Implementation:** [deepspot/cell/model.py:190-209]()

### _apply_rho

Final gene expression prediction with ensemble averaging:

1. Apply each Rho module independently to Phi-transformed features
2. **Mean aggregation** across ensemble (smooth predictions)
3. Return final gene expression predictions

**Implementation:** [deepspot/cell/model.py:211-214]()

**Sources:** [deepspot/cell/model.py:170-214]()

---

## Neighbor Aggregation Operations

The `agg_neighbors` parameter controls how neighbor features are combined in `_forward_cell_neighbors` mode. The `Operation` enum defines available operations:

```mermaid
graph LR
    OpEnum["Operation (Enum)"]
    OpSum["SUM = 'sum'<br/>Sum all neighbor features"]
    OpMax["MAX = 'max'<br/>Max pooling across neighbors"]
    OpMean["MEAN = 'mean'<br/>Average neighbor features"]
    OpNone["NONE = 'none'<br/>No aggregation (flatten)"]
    
    OpEnum --> OpSum
    OpEnum --> OpMax
    OpEnum --> OpMean
    OpEnum --> OpNone
```

| Operation | Behavior | Use Case |
|-----------|----------|----------|
| `MAX` | Takes maximum value across neighbors for each feature dimension | Default; captures strongest signals from neighborhood |
| `MEAN` | Averages neighbor features | Smooth representation of neighborhood |
| `SUM` | Sums neighbor features | Accumulates total neighborhood signal |
| `NONE` | Flattens all features without aggregation | Not typically used for neighbors |

**Sources:** [deepspot/cell/model.py:29-33](), [deepspot/cell/model.py:109-114]()

---

## Training Lifecycle

### Training and Validation Steps

The model implements PyTorch Lightning's training interface:

```mermaid
graph TB
    Trainer["Lightning Trainer"]
    
    TrainingStep["training_step(batch, batch_idx)"]
    ValidationStep["validation_step(batch, batch_idx)"]
    LoopStep["loop_step(batch, stage)"]
    
    Forward["self(X)"]
    LossCalc["loss_func(y_hat, y)"]
    Logging["self.log(stage, loss)"]
    Tracking["Append to training_loss or validation_loss"]
    
    Trainer -->|"calls"| TrainingStep
    Trainer -->|"calls"| ValidationStep
    
    TrainingStep -->|"stage='train'"| LoopStep
    ValidationStep -->|"stage='val'"| LoopStep
    
    LoopStep --> Forward
    Forward --> LossCalc
    LossCalc --> Logging
    LossCalc --> Tracking
```

**loop_step Method:**

1. Unpack batch into features `X` and targets `y`
2. Convert to float tensors (handle list inputs for neighbor mode)
3. Execute forward pass: `y_hat = self(X)`
4. Compute loss: `loss = self.loss_func(y_hat, y)`
5. Log loss to Lightning logger with stage prefix
6. Track loss in `self.training_loss` or `self.validation_loss` list
7. Return loss for backpropagation

**Implementation:** [deepspot/cell/model.py:144-168]()

### Loss Tracking

The model maintains two lists for monitoring training progress:

- `self.training_loss`: List of loss values from each training batch
- `self.validation_loss`: List of loss values from each validation batch

These lists are populated during `loop_step` and used by the `plot_loss` method for visualization.

**Sources:** [deepspot/cell/model.py:121-122](), [deepspot/cell/model.py:157-160]()

---

## Optimizer Configuration

The model uses AdamW optimizer with L2 weight decay:

```python
# From deepspot/cell/model.py:216-218
def configure_optimizers(self) -> dict:
    optimizer = torch.optim.AdamW(
        self.parameters(), 
        lr=self.hparams.lr,           # Default: 1e-4
        weight_decay=self.hparams.weight_decay  # Default: 1e-6
    )
    return {"optimizer": optimizer}
```

**Configuration:**
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** Configured via `lr` parameter (default: `1e-4`)
- **Weight Decay:** Configured via `weight_decay` parameter (default: `1e-6`)
- **No Learning Rate Scheduler:** Simple constant learning rate

**Sources:** [deepspot/cell/model.py:216-218]()

---

## Loss Visualization

### plot_loss Method

The `plot_loss` method provides visualization of training progress with exponential smoothing:

**Visualization Pipeline:**

```mermaid
graph TB
    TrainLoss["self.training_loss<br/>(per batch)"]
    ValLoss["self.validation_loss<br/>(per validation interval)"]
    
    Interpolate["Interpolate validation loss<br/>to training batch frequency"]
    Smooth["Apply exponential smoothing<br/>(alpha=0.2)"]
    
    Plot["matplotlib plot:<br/>- Original losses (transparent)<br/>- Smoothed losses (solid)"]
    
    TrainLoss --> Smooth
    ValLoss --> Interpolate
    Interpolate --> Smooth
    Smooth --> Plot
```

**Features:**

1. **Exponential Smoothing:** Applies `smooth_curve` method with `alpha=0.2` to reduce noise
2. **Validation Loss Interpolation:** Validation loss is recorded less frequently than training loss, so it's interpolated to match training step count
3. **Dual Plotting:** Shows both original (transparent) and smoothed (solid) curves
4. **Color Coding:** Blue for training loss, orange for validation loss

**smooth_curve Method:**

Uses exponential weighted moving average:
```
smoothed[0] = data[0]
smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
```

With `alpha=0.2`, each point is 20% current value and 80% smoothed previous value.

**Implementation:** [deepspot/cell/model.py:220-281]()

**Sources:** [deepspot/cell/model.py:220-281]()

---

## Inverse Transformation

The `inverse_transform` method applies inverse scaling if a scaler was provided during initialization:

```python
# From deepspot/cell/model.py:174-177
def inverse_transform(self, x) -> torch.Tensor:
    if self.scaler is not None:
        x = self.scaler.inverse_transform(x)
    return x
```

This is typically used to convert normalized/scaled predictions back to original gene expression scales. The scaler object should implement an `inverse_transform` method compatible with the data preprocessing pipeline.

**Sources:** [deepspot/cell/model.py:174-177]()

---

## Ensemble Architecture Summary

The `DeepCell` model uses a two-stage ensemble architecture:

| Stage | Module | Count | Aggregation | Purpose |
|-------|--------|-------|-------------|---------|
| Phi | `phi_cell` | `n_ensemble_phi` (default: 10) | **Median** | Feature transformation with robustness to outliers |
| Rho | `rho` | `n_ensemble_rho` (default: 10) | **Mean** | Gene expression prediction with smoothed outputs |

**Ensemble Benefits:**
- **Median aggregation (Phi):** Reduces impact of individual module failures or extreme transformations
- **Mean aggregation (Rho):** Produces smooth, averaged predictions across ensemble members
- **Independent training:** Each ensemble member learns slightly different feature representations due to dropout

**Sources:** [deepspot/cell/model.py:125-127](), [deepspot/cell/model.py:195-196](), [deepspot/cell/model.py:212-213]()

---

## Complete Data Flow Example

```mermaid
graph TB
    subgraph "Input Stage"
        BatchData["Batch from DeepCellDataLoader"]
        XCell["x_cell: (batch_size, 1, input_size)"]
        XNeigh["x_neighbors: (batch_size, num_neighbors, input_size)"]
        Y["y: (batch_size, output_size)"]
    end
    
    subgraph "Forward Pass: _forward_cell_neighbors"
        PhiCell["Phi ensemble on x_cell<br/>Median aggregation<br/>→ (batch_size, phi2rho_size)"]
        PhiNeigh["Phi ensemble on x_neighbors<br/>Median then MAX/MEAN/SUM<br/>→ (batch_size, phi2rho_size)"]
        Concat["Concatenate<br/>→ (batch_size, 2*phi2rho_size)"]
        RhoEns["Rho ensemble<br/>Mean aggregation<br/>→ (batch_size, output_size)"]
    end
    
    subgraph "Loss & Optimization"
        LossCalc["loss_func(y_hat, y)"]
        Backward["Backpropagation"]
        AdamW["AdamW.step()"]
    end
    
    BatchData --> XCell
    BatchData --> XNeigh
    BatchData --> Y
    
    XCell --> PhiCell
    XNeigh --> PhiNeigh
    PhiCell --> Concat
    PhiNeigh --> Concat
    Concat --> RhoEns
    
    RhoEns --> LossCalc
    Y --> LossCalc
    LossCalc --> Backward
    Backward --> AdamW
```

**Sources:** [deepspot/cell/model.py:63-218]()

---