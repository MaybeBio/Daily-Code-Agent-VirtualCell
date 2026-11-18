# Model Architecture

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.codeboarding/DeepSpot Models.md](.codeboarding/DeepSpot Models.md)
- [deepspot/cell/model.py](deepspot/cell/model.py)

</details>



## Purpose and Scope

This page provides an in-depth overview of the neural network architecture shared by both `DeepCell` and `DeepSpot` models. It covers the two-stage Phi-Rho ensemble design, context modes for handling spatial relationships, and the training lifecycle. For detailed documentation of individual model classes, see [DeepCell Model](#5.1) and [DeepSpot Model](#5.2). For the underlying neural layer implementations, see [Phi and Rho Neural Layers](#5.3). For optimization objectives, see [Loss Functions](#5.4).

---

## Architectural Overview

Both `DeepCell` and `DeepSpot` implement a two-stage ensemble architecture consisting of **Phi modules** for feature transformation and **Rho modules** for final prediction. This design follows a deep-set architecture pattern that enables the models to process variable-sized inputs (e.g., spots with different numbers of sub-tiles or neighbors) while maintaining permutation invariance.

### Two-Stage Ensemble Architecture

```mermaid
graph TB
    Input["Input Features<br/>(Foundation Model Embeddings)"]
    
    subgraph PhiStage["Phi Stage: Feature Transformation"]
        Phi1["Phi Module 1<br/>Linear → Dropout → ReLU"]
        Phi2["Phi Module 2<br/>Linear → Dropout → ReLU"]
        PhiN["Phi Module 10<br/>Linear → Dropout → ReLU"]
        MedianAgg["Median Aggregation"]
        
        Phi1 --> MedianAgg
        Phi2 --> MedianAgg
        PhiN --> MedianAgg
    end
    
    subgraph RhoStage["Rho Stage: Gene Expression Prediction"]
        Rho1["Rho Module 1<br/>Dropout → ReLU → Linear"]
        Rho2["Rho Module 2<br/>Dropout → ReLU → Linear"]
        RhoN["Rho Module 10<br/>Dropout → ReLU → Linear"]
        MeanAgg["Mean Aggregation"]
        
        Rho1 --> MeanAgg
        Rho2 --> MeanAgg
        RhoN --> MeanAgg
    end
    
    Input --> Phi1
    Input --> Phi2
    Input --> PhiN
    
    MedianAgg --> Rho1
    MedianAgg --> Rho2
    MedianAgg --> RhoN
    
    MeanAgg --> Output["Gene Expression Predictions<br/>(output_size genes)"]
```

**Sources:** [deepspot/cell/model.py:36-60](), [deepspot/cell/model.py:125-127](), [deepspot/cell/model.py:190-214]()

### Key Architectural Properties

| Component | Purpose | Aggregation | Count |
|-----------|---------|-------------|-------|
| **Phi Modules** | Transform input features to intermediate representation | Median across ensemble | 10 (default, configurable via `n_ensemble_phi`) |
| **Rho Modules** | Transform Phi output to gene expression predictions | Mean across ensemble | 10 (default, configurable via `n_ensemble_rho`) |
| **Input Size** | Foundation model embedding dimension | N/A | 1024 (UNI), 1536 (Hoptimus0), 768 (Phikon), 2048 (Inception) |
| **Phi Output** | Intermediate representation dimension | N/A | 1024 (default, configurable via `phi2rho_size`) |
| **Output Size** | Number of genes to predict | N/A | 300-800 (Visium), 500 (Xenium) |

**Sources:** [deepspot/cell/model.py:64-81]()

---

## Model Variants and Context Modes

The architecture supports two model variants operating at different spatial resolutions, each with multiple context modes for incorporating spatial information.

### DeepSpot Context Modes

```mermaid
graph TB
    DeepSpot["DeepSpot Model"]
    
    subgraph SpotContexts["Spot Context Modes"]
        Spot["spot<br/>Single spot features<br/>phi_multiplier=1"]
        SpotSubspot["spot_subspot<br/>Spot + 9 mini-tiles<br/>phi_multiplier=2"]
        SpotNeighbors["spot_neighbors<br/>Spot + neighbor spots<br/>phi_multiplier=2"]
        SpotSubspotNeighbors["spot_subspot_neighbors<br/>Full context<br/>phi_multiplier=3"]
    end
    
    ForwardSpot["_forward_spot()"]
    ForwardSpotSubspot["_forward_spot_subspot()"]
    ForwardSpotNeighbors["_forward_spot_neighbors()"]
    ForwardSpotSubspotNeighbors["_forward_spot_subspot_neighbors()"]
    
    DeepSpot -->|spot_context='spot'| Spot
    DeepSpot -->|spot_context='spot_subspot'| SpotSubspot
    DeepSpot -->|spot_context='spot_neighbors'| SpotNeighbors
    DeepSpot -->|spot_context='spot_subspot_neighbors'| SpotSubspotNeighbors
    
    Spot --> ForwardSpot
    SpotSubspot --> ForwardSpotSubspot
    SpotNeighbors --> ForwardSpotNeighbors
    SpotSubspotNeighbors --> ForwardSpotSubspotNeighbors
    
    ForwardSpot --> PhiRho["Phi-Rho Pipeline"]
    ForwardSpotSubspot --> PhiRho
    ForwardSpotNeighbors --> PhiRho
    ForwardSpotSubspotNeighbors --> PhiRho
```

**Sources:** [deepspot/spot/model.py:110-126](), [deepspot/spot/model.py:163-187]()

### DeepCell Context Modes

```mermaid
graph TB
    DeepCell["DeepCell Model"]
    
    subgraph CellContexts["Cell Context Modes"]
        Cell["cell<br/>Single cell features<br/>phi_multiplier=1"]
        CellNeighbors["cell_neighbors<br/>Cell + neighbor cells<br/>phi_multiplier=2"]
    end
    
    ForwardCell["_forward_cell()"]
    ForwardCellNeighbors["_forward_cell_neighbors()"]
    
    DeepCell -->|cell_context='cell'| Cell
    DeepCell -->|cell_context='cell_neighbors'| CellNeighbors
    
    Cell --> ForwardCell
    CellNeighbors --> ForwardCellNeighbors
    
    ForwardCell --> PhiRho["Phi-Rho Pipeline"]
    ForwardCellNeighbors --> PhiRho
```

**Sources:** [deepspot/cell/model.py:130-142](), [deepspot/cell/model.py:179-188]()

### Context Mode Comparison

| Model | Context Mode | Description | Phi Multiplier | Forward Function |
|-------|--------------|-------------|----------------|------------------|
| **DeepSpot** | `spot` | Individual spot only | 1 | `_forward_spot()` |
| **DeepSpot** | `spot_subspot` | Spot + 9 mini-tiles (3×3 grid) | 2 | `_forward_spot_subspot()` |
| **DeepSpot** | `spot_neighbors` | Spot + neighboring spots | 2 | `_forward_spot_neighbors()` |
| **DeepSpot** | `spot_subspot_neighbors` | Spot + mini-tiles + neighbors (full) | 3 | `_forward_spot_subspot_neighbors()` |
| **DeepCell** | `cell` | Individual cell only | 1 | `_forward_cell()` |
| **DeepCell** | `cell_neighbors` | Cell + neighboring cells | 2 | `_forward_cell_neighbors()` |

The `phi_multiplier` determines the input dimension to Rho modules: `phi2rho_size × phi_multiplier`. This accommodates concatenated Phi outputs from multiple spatial contexts.

**Sources:** [deepspot/cell/model.py:130-135](), [deepspot/spot/model.py:110-117]()

---

## Component-Level Architecture

### Phi Module Design

The `Phi` class implements feature transformation with the following structure:

```mermaid
graph LR
    Input["Input<br/>(batch × input_size)"]
    Linear["Linear Layer<br/>(input_size → output_size)"]
    Dropout["Dropout<br/>(p=p_phi)"]
    ReLU["ReLU Activation<br/>(inplace=True)"]
    Output["Output<br/>(batch × output_size)"]
    
    Input --> Linear
    Linear --> Dropout
    Dropout --> ReLU
    ReLU --> Output
```

**Implementation Details:**
- **Input Size**: Foundation model embedding dimension (e.g., 1024 for UNI)
- **Output Size**: Configurable via `phi2rho_size` parameter (default: 1024)
- **Dropout Rate**: Configurable via `p_phi` parameter (default: 0.2)
- **Activation**: ReLU with in-place operation for memory efficiency

**Ensemble Aggregation:**
For each input, all Phi modules in the ensemble process it independently. The median value across the ensemble is selected for each feature dimension, providing robustness to outlier predictions.

**Sources:** [deepspot/cell/model.py:36-46](), [deepspot/cell/model.py:190-209]()

### Rho Module Design

The `Rho` class implements final gene expression prediction:

```mermaid
graph LR
    Input["Input<br/>(batch × phi2rho_size × phi_multiplier)"]
    Dropout1["Dropout<br/>(p=p_rho)"]
    ReLU["ReLU Activation<br/>(inplace=True)"]
    Linear["Linear Layer<br/>(phi2rho_size × phi_multiplier → output_size)"]
    Dropout2["Dropout<br/>(p=0.1)"]
    Output["Output<br/>(batch × output_size)"]
    
    Input --> Dropout1
    Dropout1 --> ReLU
    ReLU --> Linear
    Linear --> Dropout2
    Dropout2 --> Output
```

**Implementation Details:**
- **Input Size**: `phi2rho_size × phi_multiplier` (e.g., 1024×2 = 2048 for `cell_neighbors` mode)
- **Output Size**: Number of genes to predict (e.g., 500 for Xenium)
- **Dropout Rates**: Configurable `p_rho` before activation (default: 0.2), fixed 0.1 after linear layer
- **Activation**: ReLU with in-place operation

**Ensemble Aggregation:**
The mean of all Rho module outputs is computed, providing a stable final prediction by averaging over the ensemble.

**Sources:** [deepspot/cell/model.py:49-60](), [deepspot/cell/model.py:211-214]()

---

## Forward Pass Mechanisms

### Spatial Context Processing

Different context modes process spatial information differently through the Phi-Rho pipeline:

```mermaid
graph TB
    subgraph CellOnly["Cell-Only Forward (_forward_cell)"]
        C_Input["x: cell features<br/>(batch × 1 × input_size)"]
        C_Phi["Apply Phi<br/>(no aggregation)"]
        C_Rho["Apply Rho"]
        C_Output["predictions"]
        
        C_Input --> C_Phi
        C_Phi --> C_Rho
        C_Rho --> C_Output
    end
    
    subgraph CellNeighbors["Cell + Neighbors Forward (_forward_cell_neighbors)"]
        CN_Input["x: [cell, neighbors]<br/>cell: (batch × 1 × input_size)<br/>neighbors: (batch × n_neighbors × input_size)"]
        CN_PhiCell["Apply Phi to cell<br/>(no aggregation)"]
        CN_PhiNeigh["Apply Phi to neighbors<br/>(max/mean/sum aggregation)"]
        CN_Concat["Concatenate<br/>(batch × phi2rho_size × 2)"]
        CN_Rho["Apply Rho"]
        CN_Output["predictions"]
        
        CN_Input --> CN_PhiCell
        CN_Input --> CN_PhiNeigh
        CN_PhiCell --> CN_Concat
        CN_PhiNeigh --> CN_Concat
        CN_Concat --> CN_Rho
        CN_Rho --> CN_Output
    end
```

**Neighbor Aggregation Strategies:**
The `agg_neighbors` parameter (default: `"max"`) controls how neighbor features are aggregated after Phi transformation:
- **`max`**: Takes maximum value across neighbors for each feature dimension
- **`mean`**: Computes mean across neighbors
- **`sum`**: Sums neighbor features

**Sources:** [deepspot/cell/model.py:179-188](), [deepspot/cell/model.py:109-115]()

### Spot Context Processing

DeepSpot extends this pattern to support sub-spot tiles:

| Forward Function | Inputs | Phi Applications | Concatenation |
|------------------|--------|-----------------|---------------|
| `_forward_spot()` | `x` (spot features) | Spot through `phi_spot` | Single Phi output |
| `_forward_spot_subspot()` | `[x_spot, x_subspot]` | Spot through `phi_spot`, subspots through `phi_subspot` | 2 Phi outputs |
| `_forward_spot_neighbors()` | `[x_spot, x_neighbors]` | Spot through `phi_spot`, neighbors through `phi_spot` (aggregated) | 2 Phi outputs |
| `_forward_spot_subspot_neighbors()` | `[x_spot, x_subspot, x_neighbors]` | Spot through `phi_spot`, subspots through `phi_subspot`, neighbors through `phi_spot` (aggregated) | 3 Phi outputs |

**Sources:** [deepspot/spot/model.py:163-187]()

---

## Training Lifecycle

Both models inherit from `lightning.LightningModule` and implement standard PyTorch Lightning training patterns:

```mermaid
graph TB
    Init["__init__()<br/>Initialize Phi/Rho ensembles<br/>Configure loss function<br/>Set hyperparameters"]
    
    TrainStep["training_step(batch, batch_idx)<br/>→ loop_step(batch, 'train')"]
    ValStep["validation_step(batch, batch_idx)<br/>→ loop_step(batch, 'val')"]
    
    LoopStep["loop_step(batch, stage)"]
    Forward["forward(X)<br/>→ _forward_fn(X)"]
    Loss["loss_func(y_hat, y)"]
    Log["Log metrics<br/>self.log(stage, loss)"]
    
    Optimizer["configure_optimizers()<br/>AdamW optimizer"]
    
    Init --> TrainStep
    Init --> ValStep
    Init --> Optimizer
    
    TrainStep --> LoopStep
    ValStep --> LoopStep
    
    LoopStep --> Forward
    Forward --> Loss
    Loss --> Log
    
    Log -->|accumulate| TrainingLoss["self.training_loss[]"]
    Log -->|accumulate| ValidationLoss["self.validation_loss[]"]
    
    TrainingLoss --> PlotLoss["plot_loss()<br/>Visualize training curves"]
    ValidationLoss --> PlotLoss
```

**Sources:** [deepspot/cell/model.py:144-168](), [deepspot/cell/model.py:216-218](), [deepspot/cell/model.py:235-281]()

### Key Training Components

| Component | Implementation | Configuration |
|-----------|----------------|---------------|
| **Optimizer** | `torch.optim.AdamW` | `lr` (default: 1e-4), `weight_decay` (default: 1e-6) |
| **Loss Function** | Configurable via `loss_func` parameter | See [Loss Functions](#5.4) for available options |
| **Loss Tracking** | `self.training_loss` and `self.validation_loss` lists | Populated during `loop_step()` |
| **Visualization** | `plot_loss()` method | Exponentially smoothed training curves with original overlays |
| **Random Seed** | `fix_seed(random_seed)` | Ensures reproducibility (default: 2024) |

**Sources:** [deepspot/cell/model.py:64-81](), [deepspot/cell/model.py:88-107](), [deepspot/cell/model.py:216-218]()

---

## Initialization Parameters

### Shared Parameters (DeepCell and DeepSpot)

```python
# Core architecture parameters
input_size: int           # Foundation model embedding dimension
output_size: int          # Number of genes to predict
phi2rho_size: int = 1024  # Intermediate representation size
emb_size: int = 1024      # (Deprecated, maintained for compatibility)

# Ensemble configuration
n_ensemble: int = 10           # Default ensemble size
n_ensemble_phi: int = None     # Phi ensemble size (defaults to n_ensemble)
n_ensemble_rho: int = None     # Rho ensemble size (defaults to n_ensemble)

# Regularization
p: float = 0.2                 # Default dropout probability
p_phi: float = None            # Phi dropout (defaults to p)
p_rho: float = None            # Rho dropout (defaults to p)

# Training configuration
loss_func: str = "mse"         # Loss function identifier
lr: float = 1e-4               # Learning rate
weight_decay: float = 1e-6     # AdamW weight decay
random_seed: int = 2024        # Random seed for reproducibility

# Data processing
scaler = None                  # Optional scaler for inverse transforms
```

### DeepCell-Specific Parameters

```python
cell_context: str = 'cell_neighbors'  # Context mode: 'cell' or 'cell_neighbors'
agg_neighbors: str = "max"            # Neighbor aggregation: 'max', 'mean', or 'sum'
```

**Sources:** [deepspot/cell/model.py:64-81]()

### DeepSpot-Specific Parameters

```python
spot_context: str = 'spot_subspot_neighbors'  # Context mode (4 options)
agg_neighbors: str = "max"                    # Neighbor aggregation
```

**Sources:** [deepspot/spot/model.py:59-83]()

---

## Code-to-Architecture Mapping

This diagram maps architectural concepts to their implementations:

```mermaid
graph TB
    subgraph Architecture["Architectural Concepts"]
        TwoStage["Two-Stage Ensemble<br/>Phi → Rho"]
        ContextModes["Context Modes<br/>Spatial relationships"]
        LossOpt["Loss & Optimization<br/>Training objectives"]
    end
    
    subgraph CodeEntities["Code Entities"]
        PhiClass["Phi class<br/>cell/model.py:36-46<br/>spot/model.py:31-41"]
        RhoClass["Rho class<br/>cell/model.py:49-60<br/>spot/model.py:44-55"]
        
        DeepCellClass["DeepCell class<br/>cell/model.py:63-281"]
        DeepSpotClass["DeepSpot class<br/>spot/model.py:58-217"]
        
        ForwardFuncs["_forward_* methods<br/>Context-specific forward passes"]
        ApplyPhi["_apply_phi()<br/>Ensemble median aggregation"]
        ApplyRho["_apply_rho()<br/>Ensemble mean aggregation"]
        
        LossFuncs["loss.py<br/>Loss function implementations"]
        LoopStep["loop_step()<br/>Training/validation logic"]
        ConfigOpt["configure_optimizers()<br/>AdamW setup"]
    end
    
    TwoStage --> PhiClass
    TwoStage --> RhoClass
    TwoStage --> ApplyPhi
    TwoStage --> ApplyRho
    
    ContextModes --> DeepCellClass
    ContextModes --> DeepSpotClass
    ContextModes --> ForwardFuncs
    
    LossOpt --> LossFuncs
    LossOpt --> LoopStep
    LossOpt --> ConfigOpt
    
    DeepCellClass --> PhiClass
    DeepCellClass --> RhoClass
    DeepSpotClass --> PhiClass
    DeepSpotClass --> RhoClass
```

**Sources:** [deepspot/cell/model.py:36-281](), [deepspot/spot/model.py:31-217]()

---

## Summary

The DeepSpot architecture implements a flexible two-stage ensemble design that supports multiple spatial context modes for both spot-level and cell-level spatial transcriptomics prediction:

- **Phi Stage**: Transforms input features through 10 parallel modules with median aggregation for robustness
- **Rho Stage**: Generates final predictions through 10 parallel modules with mean aggregation for stability
- **Context Modes**: Enable incorporation of spatial relationships (sub-tiles, neighbors) while maintaining a unified architecture
- **Training**: Leverages PyTorch Lightning for standardized training loops with configurable loss functions and AdamW optimization

For implementation details of specific model variants, see [DeepCell Model](#5.1) and [DeepSpot Model](#5.2). For neural layer internals, see [Phi and Rho Neural Layers](#5.3). For optimization objectives, see [Loss Functions](#5.4).

---