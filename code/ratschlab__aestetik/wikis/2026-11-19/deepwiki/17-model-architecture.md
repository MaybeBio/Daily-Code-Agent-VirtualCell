# Model Architecture

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/aestetik/callbacks/callbacks.py](src/aestetik/callbacks/callbacks.py)
- [src/aestetik/metrics/loss_function.py](src/aestetik/metrics/loss_function.py)
- [src/aestetik/models/model.py](src/aestetik/models/model.py)
- [src/aestetik/modules/aestetik_module.py](src/aestetik/modules/aestetik_module.py)

</details>



## Purpose and Scope

This page documents the neural network model architecture used in AESTETIK, including the `AESTETIKModel` Lightning module wrapper and the `AE` autoencoder. This covers the convolutional encoder-decoder architecture, ensemble mechanisms, and loss computation.

For information about the AESTETIK orchestrator class that manages model training, see [AESTETIK Class](#5.1). For data preprocessing and batch creation, see [Data Pipeline Components](#5.2).

---

## Component Overview

The model architecture consists of three primary components:

| Component | Class | File | Purpose |
|-----------|-------|------|---------|
| Lightning Wrapper | `AESTETIKModel` | `aestetik_module.py` | PyTorch Lightning module handling training loops, optimization, and prediction |
| Core Model | `AE` | `model.py` | Convolutional autoencoder with ensemble encoder/decoder |
| Loss Computation | `compute_loss()` | `loss_function.py` | Multi-component loss function for reconstruction and triplet losses |

**Sources:** [src/aestetik/modules/aestetik_module.py:1-185](), [src/aestetik/models/model.py:1-98](), [src/aestetik/metrics/loss_function.py:1-90]()

---

## Module Class Hierarchy

```mermaid
classDiagram
    class LightningModule {
        <<PyTorch Lightning>>
    }
    
    class AESTETIKModel {
        +datamodule: AESTETIKDataModule
        +model: AE
        +loss: dict
        +weights: dict
        +predict_params: dict
        +setup()
        +configure_model()
        +training_step()
        +validation_step()
        +predict_step()
        +configure_optimizers()
    }
    
    class AE {
        +encoder_ensemble: ModuleList
        +decoder_ensemble: ModuleList
        +encoder()
        +decoder()
        +forward()
    }
    
    class Encode {
        +encoder: Sequential
        +forward()
    }
    
    class Decode {
        +decoder: Sequential
        +forward()
    }
    
    class Module {
        <<PyTorch>>
    }
    
    LightningModule <|-- AESTETIKModel
    Module <|-- AE
    Module <|-- Encode
    Module <|-- Decode
    AESTETIKModel --> AE : contains
    AE --> Encode : ensemble
    AE --> Decode : ensemble
```

**Sources:** [src/aestetik/modules/aestetik_module.py:14-21](), [src/aestetik/models/model.py:51-98]()

---

## AESTETIKModel Lightning Module

The `AESTETIKModel` class extends `lightning.LightningModule` and serves as the training orchestrator. It wraps the core `AE` autoencoder and implements PyTorch Lightning's interface for training, validation, and prediction.

### Initialization Parameters

```mermaid
graph LR
    AESTETIKModel["AESTETIKModel"]
    
    subgraph "Parameter Groups"
        grid["grid_params<br/>obsm_transcriptomics_dim"]
        arch["model_architecture_params<br/>kernel_size, latent_dim<br/>n_ensemble_encoder, n_ensemble_decoder<br/>c_hid, p"]
        train["training_params<br/>rec_alpha, triplet_alpha"]
        opt["optimizer_params<br/>lr, weight_decay"]
        dm["datamodule<br/>AESTETIKDataModule"]
    end
    
    grid --> AESTETIKModel
    arch --> AESTETIKModel
    train --> AESTETIKModel
    opt --> AESTETIKModel
    dm --> AESTETIKModel
```

**Sources:** [src/aestetik/modules/aestetik_module.py:15-55]()

### Key Methods

| Method | Parameters | Returns | Purpose |
|--------|-----------|---------|---------|
| `setup()` | `stage` | `None` | Initialize loss functions and weights from datamodule |
| `configure_model()` | - | `None` | Build the `AE` model with grid parameters |
| `training_step()` | `batch`, `batch_idx` | `Tensor` | Compute forward pass and loss for training |
| `validation_step()` | `batch`, `batch_idx` | `Tensor` | Compute validation loss with gradient tracking disabled |
| `predict_step()` | `batch`, `batch_idx`, `dataloader_idx` | `Tensor` | Generate latent embeddings with Monte Carlo dropout |
| `configure_optimizers()` | - | `Optimizer` | Configure Adam optimizer with AMSGrad |

**Sources:** [src/aestetik/modules/aestetik_module.py:68-168]()

### Model Configuration

The model is lazily initialized in `configure_model()` to allow dynamic grid parameter determination:

```mermaid
sequenceDiagram
    participant L as Lightning Trainer
    participant M as AESTETIKModel
    participant D as DataModule
    participant A as AE
    
    L->>M: configure_model()
    M->>M: Check model_built flag
    M->>D: Access adata.obsm["X_st_grid"]
    D-->>M: Grid shape information
    M->>M: Set morphology_dim
    M->>M: Set num_input_channels
    M->>A: Initialize AE(**params)
    A-->>M: model instance
    M->>M: Set model_built = True
```

**Sources:** [src/aestetik/modules/aestetik_module.py:80-91]()

---

## AE Autoencoder Architecture

The `AE` class implements the core convolutional autoencoder with ensemble components. Both encoder and decoder use multiple independent networks whose outputs are combined via median aggregation.

### Architecture Overview

```mermaid
graph TB
    Input["Input Grid<br/>N × C × H × W"]
    
    subgraph "Encoder Ensemble"
        E1["Encode 1"]
        E2["Encode 2"]
        EN["Encode N"]
        MedianEnc["Median Aggregation"]
    end
    
    Latent["Latent Space<br/>N × latent_dim"]
    
    subgraph "Decoder Ensemble"
        D1["Decode 1"]
        D2["Decode 2"]
        DM["Decode M"]
        MedianDec["Median Aggregation"]
    end
    
    Output["Reconstructed Grid<br/>N × C × H × W"]
    
    Input --> E1
    Input --> E2
    Input --> EN
    E1 --> MedianEnc
    E2 --> MedianEnc
    EN --> MedianEnc
    MedianEnc --> Latent
    Latent --> D1
    Latent --> D2
    Latent --> DM
    D1 --> MedianDec
    D2 --> MedianDec
    DM --> MedianDec
    MedianDec --> Output
```

**Sources:** [src/aestetik/models/model.py:51-98]()

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_input_channels` | `int` | Total number of channels (transcriptomics + morphology) |
| `morphology_dim` | `int` | Spatial dimension of input grid (height/width) |
| `c_hid` | `int` | Number of hidden channels in convolutional layers |
| `latent_dim` | `int` | Dimensionality of latent representation |
| `kernel_size` | `int` | Size of convolutional kernels |
| `p` | `float` | Dropout probability |
| `n_ensemble_encoder` | `int` | Number of encoder networks (default: 2) |
| `n_ensemble_decoder` | `int` | Number of decoder networks (default: 1) |

**Sources:** [src/aestetik/models/model.py:52-62]()

### Forward Pass Methods

The `AE` class provides three methods for inference:

```mermaid
graph LR
    AE["AE Model"]
    
    encoder["encoder(x)<br/>Returns: latent_space"]
    decoder["decoder(latent)<br/>Returns: reconstruction"]
    forward["forward(x)<br/>Returns: latent, reconstruction"]
    
    AE --> encoder
    AE --> decoder
    AE --> forward
    
    forward -.-> encoder
    forward -.-> decoder
```

**Sources:** [src/aestetik/models/model.py:82-97]()

---

## Encoder Architecture (Encode Class)

The `Encode` class implements a single convolutional encoder network. Multiple instances are used in the ensemble.

### Layer Structure

```mermaid
graph TD
    Input["Input<br/>N × num_input_channels × H × W"]
    
    Conv["Conv2d<br/>kernel_size, stride=1"]
    BN1["BatchNorm2d"]
    Drop1["Dropout2d(p)"]
    ReLU1["ReLU"]
    Pool["MaxPool2d<br/>kernel_size, stride=3"]
    Flat["Flatten"]
    Linear["Linear<br/>conv_out_2² × c_hid → latent_dim"]
    
    Output["Output<br/>N × 1 × latent_dim"]
    
    Input --> Conv
    Conv --> BN1
    BN1 --> Drop1
    Drop1 --> ReLU1
    ReLU1 --> Pool
    Pool --> Flat
    Flat --> Linear
    Linear --> Output
```

**Spatial Dimension Calculations:**

The spatial dimensions are computed during `AE.__init__()`:
- `conv_out_1 = (morphology_dim - kernel_size) + 1`
- `conv_out_2 = floor((conv_out_1 - kernel_size) / 3 + 1)`

**Sources:** [src/aestetik/models/model.py:8-26](), [src/aestetik/models/model.py:65-66]()

### Implementation Details

The encoder is implemented as a `nn.Sequential` module containing all layers. The `forward()` method adds a singleton dimension to output shape `(batch_size, 1, latent_dim)` for ensemble concatenation.

**Sources:** [src/aestetik/models/model.py:13-26]()

---

## Decoder Architecture (Decode Class)

The `Decode` class implements a single convolutional decoder network that reconstructs the input from latent space.

### Layer Structure

```mermaid
graph TD
    Input["Input<br/>N × latent_dim"]
    
    Linear["Linear<br/>latent_dim → conv_out_2² × c_hid"]
    Unflatten["Unflatten<br/>→ (c_hid, conv_out_2, conv_out_2)"]
    ReLU["ReLU"]
    Upsample["UpsamplingNearest2d<br/>→ (conv_out_1, conv_out_1)"]
    Drop["Dropout2d(p)"]
    ConvT["ConvTranspose2d<br/>kernel_size, stride=1"]
    BN["BatchNorm2d"]
    Sigmoid["Sigmoid"]
    
    Output["Output<br/>N × 1 × num_input_channels × H × W"]
    
    Input --> Linear
    Linear --> Unflatten
    Unflatten --> ReLU
    ReLU --> Upsample
    Upsample --> Drop
    Drop --> ConvT
    ConvT --> BN
    BN --> Sigmoid
    Sigmoid --> Output
```

**Sources:** [src/aestetik/models/model.py:29-48]()

### Reconstruction Process

The decoder reverses the encoder operations:
1. Linear expansion from latent space
2. Reshape to spatial feature map
3. Upsampling to intermediate size
4. Transposed convolution to original spatial dimensions
5. Sigmoid activation for bounded output values

**Sources:** [src/aestetik/models/model.py:34-43]()

---

## Ensemble Mechanism

### Median Aggregation Strategy

Both encoder and decoder ensembles use median aggregation for robust predictions:

```mermaid
sequenceDiagram
    participant Input
    participant E1 as Encoder 1
    participant E2 as Encoder 2
    participant EN as Encoder N
    participant Agg as Aggregation
    
    Input->>E1: forward(x)
    Input->>E2: forward(x)
    Input->>EN: forward(x)
    E1-->>Agg: output₁
    E2-->>Agg: output₂
    EN-->>Agg: outputₙ
    Agg->>Agg: torch.cat(outputs, dim=1)
    Agg->>Agg: torch.median(outputs, dim=1)
    Agg-->>Input: median_output
```

**Sources:** [src/aestetik/models/model.py:82-92]()

### Code Implementation

#### Encoder Ensemble

```
x_encode = [unit(x) for unit in self.encoder_ensemble]
x_encode = torch.cat(x_encode, 1)
x_encode_median, _ = torch.median(x_encode, 1)
```

**Sources:** [src/aestetik/models/model.py:82-86]()

#### Decoder Ensemble

```
x_decode = [unit(x) for unit in self.decoder_ensemble]
x_decode = torch.cat(x_decode, 1)
x_decode_median, _ = torch.median(x_decode, 1)
```

**Sources:** [src/aestetik/models/model.py:88-92]()

### Rationale

Median aggregation provides robustness to outlier predictions from individual ensemble members, unlike mean aggregation which can be skewed by extreme values.

---

## Loss Computation

The loss function combines reconstruction and triplet losses for both modalities.

### Loss Components

```mermaid
graph TB
    subgraph "Reconstruction Losses"
        RecT["rec_loss_transcriptomics<br/>L1Loss"]
        RecM["rec_loss_morphology<br/>L1Loss"]
    end
    
    subgraph "Triplet Losses"
        TripT["triplet_loss_transcriptomics<br/>TripletMarginLoss"]
        TripM["triplet_loss_morphology<br/>TripletMarginLoss"]
    end
    
    subgraph "Weighting"
        Weights["transcriptomics_weight<br/>morphology_weight"]
        Alphas["rec_alpha<br/>triplet_alpha"]
    end
    
    RecT --> Total["Total Loss"]
    RecM --> Total
    TripT --> Total
    TripM --> Total
    Weights --> Total
    Alphas --> Total
```

**Sources:** [src/aestetik/metrics/loss_function.py:39-89]()

### Loss Formula

The total loss is computed as:

```
total_loss = transcriptomics_weight × (rec_alpha × rec_loss_transcriptomics + 
                                        triplet_alpha × triplet_loss_transcriptomics) +
             morphology_weight × (rec_alpha × rec_loss_morphology + 
                                  triplet_alpha × triplet_loss_morphology)
```

**Sources:** [src/aestetik/metrics/loss_function.py:86-87]()

### Loss Function Interface

| Function | Parameters | Returns | Purpose |
|----------|-----------|---------|---------|
| `compute_loss()` | `model`, `anchor`, `anchor_encode`, `anchor_decode`, positive/negative lists, weights, alphas | `Tuple[Tensor, ...]` | Compute all loss components |
| `compute_reconstruction_loss()` | `anchor`, `anchor_decode`, `loss` | `Tensor` | L1 loss between input and reconstruction |
| `compute_average_triplet_loss()` | `model`, `anchor_encode`, `pos_list`, `neg_list`, `loss`, `device` | `Tensor` | Average triplet loss over samples |

**Sources:** [src/aestetik/metrics/loss_function.py:4-89]()

### Triplet Loss Computation

Triplet loss uses anchor, positive, and negative samples from the same/different clusters:

```mermaid
sequenceDiagram
    participant A as Anchor
    participant P as Positive Samples
    participant N as Negative Samples
    participant E as Encoder
    participant L as TripletMarginLoss
    
    A->>E: Encode anchor
    E-->>A: anchor_encode
    P->>E: Encode positives (batch)
    E-->>P: pos_encodes
    N->>E: Encode negatives (batch)
    E-->>N: neg_encodes
    A->>L: (anchor, pos, neg)
    P->>L: Multiple samples
    N->>L: Multiple samples
    L-->>A: Average loss
```

**Sources:** [src/aestetik/metrics/loss_function.py:4-29]()

---

## Training Workflow

### Training Step Implementation

The `training_step()` method in `AESTETIKModel` orchestrates the forward pass and loss computation:

```mermaid
sequenceDiagram
    participant T as Trainer
    participant M as AESTETIKModel
    participant AE as AE Model
    participant L as compute_loss
    
    T->>M: training_step(batch, batch_idx)
    M->>M: Unpack batch
    Note over M: anchor, pos_trans_list, neg_trans_list<br/>pos_morph_list, neg_morph_list
    M->>AE: forward(anchor)
    AE-->>M: anchor_encode, anchor_decode
    M->>L: compute_loss(...)
    L-->>M: total_loss, component_losses
    M->>M: Log metrics (L, RC, RI, TC, TI)
    M-->>T: total_loss
```

**Sources:** [src/aestetik/modules/aestetik_module.py:93-117]()

### Logged Metrics

| Metric Key | Description |
|------------|-------------|
| `L` | Total loss |
| `RC` | Reconstruction loss (transcriptomics) |
| `RI` | Reconstruction loss (morphology/image) |
| `TC` | Triplet loss (transcriptomics) |
| `TI` | Triplet loss (morphology/image) |

**Sources:** [src/aestetik/modules/aestetik_module.py:112-116]()

---

## Prediction Workflow

### Monte Carlo Dropout Prediction

The `predict_step()` method uses Monte Carlo dropout for uncertainty estimation:

```mermaid
graph TD
    Input["Input Batch"]
    
    ModelTrain["Set model.train()<br/>Keep dropout active"]
    
    Loop["Repeat num_repeats times"]
    
    Forward["encoder(batch)"]
    Store["Store latent output"]
    
    Stack["Stack outputs<br/>dim=0"]
    Mean["torch.mean(dim=0)"]
    
    Output["Averaged Latent Space"]
    
    Input --> ModelTrain
    ModelTrain --> Loop
    Loop --> Forward
    Forward --> Store
    Store --> Loop
    Loop --> Stack
    Stack --> Mean
    Mean --> Output
```

**Sources:** [src/aestetik/modules/aestetik_module.py:142-161]()

### Implementation Details

The prediction process:
1. Sets model to training mode to keep dropout layers active
2. Performs multiple forward passes (`num_repeats` times)
3. Averages the latent representations across repeats
4. Returns mean latent space for each batch

This approach provides more robust embeddings by sampling from the dropout distribution.

**Sources:** [src/aestetik/modules/aestetik_module.py:151-161]()

---

## Validation Workflow

The validation step is similar to training but with gradients disabled:

```mermaid
graph LR
    Val["validation_step()"]
    Train["Set model.train()"]
    NoGrad["with torch.no_grad()"]
    Forward["Forward pass"]
    Loss["Compute loss"]
    Log["Log val_loss"]
    
    Val --> Train
    Train --> NoGrad
    NoGrad --> Forward
    Forward --> Loss
    Loss --> Log
```

**Key differences from training:**
- Wrapped in `torch.no_grad()` context
- Only logs `val_loss` (total loss)
- Model still in training mode to maintain dropout behavior

**Sources:** [src/aestetik/modules/aestetik_module.py:119-140]()

---

## Optimizer Configuration

The `configure_optimizers()` method sets up the Adam optimizer:

```python
optimizer = torch.optim.Adam(
    self.model.parameters(),
    amsgrad=True,
    lr=optimizer_params["lr"],
    weight_decay=optimizer_params["weight_decay"]
)
```

**Configuration:**
- Algorithm: Adam with AMSGrad variant
- Adaptive learning rates for all model parameters
- L2 regularization via `weight_decay`

**Sources:** [src/aestetik/modules/aestetik_module.py:163-168]()

---

## Parameter Validation

Both `AESTETIKModel` and `AE` validate their parameters during initialization to ensure all required fields are present.

### Required Parameters by Group

| Parameter Group | Required Keys |
|----------------|---------------|
| `grid_params` | `obsm_transcriptomics_dim` |
| `model_architecture_params` | `kernel_size`, `latent_dim`, `n_ensemble_encoder`, `n_ensemble_decoder`, `c_hid`, `p` |
| `training_params` | `rec_alpha`, `triplet_alpha` |
| `optimizer_params` | `lr`, `weight_decay` |

**Sources:** [src/aestetik/modules/aestetik_module.py:171-185]()

---

## Callbacks

AESTETIK provides a custom callback for tracking training loss history:

### LossHistoryCallback

```mermaid
classDiagram
    class Callback {
        <<PyTorch Lightning>>
    }
    
    class LossHistoryCallback {
        +losses: List
        +on_train_batch_end()
    }
    
    Callback <|-- LossHistoryCallback
```

**Purpose:** Records loss values at the end of each training batch for visualization and analysis.

**Usage:** Automatically invoked by Lightning Trainer during training.

**Sources:** [src/aestetik/callbacks/callbacks.py:1-18]()

---

## Summary

The AESTETIK model architecture consists of:

1. **Lightning Wrapper (`AESTETIKModel`)**: Handles training orchestration, validation, and prediction workflows
2. **Core Autoencoder (`AE`)**: Convolutional encoder-decoder with ensemble members
3. **Encoder (`Encode`)**: Conv2d → BatchNorm → Dropout → MaxPool → Linear layers
4. **Decoder (`Decode`)**: Linear → Unflatten → Upsample → ConvTranspose2d → Sigmoid layers
5. **Ensemble Mechanism**: Median aggregation across multiple encoders/decoders
6. **Multi-Component Loss**: Reconstruction (L1) + Triplet losses for both modalities
7. **Monte Carlo Dropout**: Multiple forward passes during prediction for robust embeddings

This architecture enables robust learning of multi-modal spatial transcriptomics embeddings while maintaining spatial neighborhood information through grid-based inputs.

---