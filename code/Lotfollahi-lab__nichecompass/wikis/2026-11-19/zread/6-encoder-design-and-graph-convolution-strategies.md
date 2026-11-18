

The encoder architecture in NicheCompass implements a sophisticated graph neural network approach for transforming spatial omics data into latent gene program representations. This documentation explores the dual-stage encoder design, graph convolution strategies, and integration mechanisms that enable the model to capture both local cellular contexts and global biological patterns through learned gene program activities.

## Architectural Foundation

The encoder operates as the first critical component in the Variational Gene Program Graph Autoencoder (VGPGAE), transforming high-dimensional omics feature spaces into compact latent representations that encode biological gene program activities. The design follows a hybrid architecture combining fully-connected preprocessing layers with graph-based message passing, allowing the model to learn both feature-level transformations and neighborhood-aware representations.

The encoder receives three primary inputs: omics feature vectors representing gene expression or chromatin accessibility measurements, edge indices defining spatial connectivity between cells/spots, and optional categorical covariates embeddings for batch correction or metadata integration. Through a carefully designed transformation pipeline, these inputs are mapped to a latent space characterized by two complementary components: prior gene programs derived from biological knowledge, and addon gene programs learned de novo from the data.

## Dual-Stage Encoder Architecture

The encoder implements a sequential processing pipeline where fully-connected layers perform initial feature transformation followed by graph convolution layers that incorporate spatial context. This two-stage design allows the model to first learn nonlinear transformations of individual node features, then leverage neighborhood information to refine these representations based on cellular microenvironments.

The fully-connected preprocessing stage supports either single or dual-layer configurations. In the two-layer setup, the first layer reduces input dimensionality by half, followed by a second layer that maps to the hidden dimension size with batch normalization applied. This architecture enables gradual dimensionality reduction while preserving information through intermediate representations. The single-layer configuration provides a direct mapping from input to hidden space for simpler scenarios or when computational efficiency is prioritized.

```mermaid
flowchart TD
    A[Input Features<br/>n_input] --> B{Covariate Mode}
    B -->|input| C[Concatenate<br/>Covariate Embedding]
    B -->|hidden| D[FC Layer 1<br/>n_input → n_hidden]
    C --> D
    D --> E{FC Layers}
    E -->|2 layers| F[FC Layer 2<br/>n_input/2 → n_hidden]
    E -->|1 layer| G[Activation & Dropout]
    F --> H[BatchNorm]
    H --> G
    G --> I{Covariate Mode}
    I -->|hidden| J[Concatenate<br/>Covariate Embedding]
    I -->|input| K[Graph Conv Layers]
    J --> K
    K --> L{Conv Layers}
    L -->|2 layers| M[Conv Layer 1<br/>n_hidden → n_hidden]
    L -->|1 layer| N[Final Conv Layers]
    M --> N
    N --> O[Prior GP μ & logσ<br/>n_hidden → n_latent]
    N --> P[Addon GP μ & logσ<br/>n_hidden → n_addon_latent]
    O --> Q[Concatenate<br/>z = [μ_prior, μ_addon]]
    P --> Q
    Q --> R[Latent Representation]
```

The graph convolution stage operates on the transformed feature space from the fully-connected layers, applying message passing along the spatial connectivity graph. Similar to the fully-connected stage, it supports one or two convolutional layers. When two layers are configured, the first layer performs intermediate transformations within the hidden dimension space, while the final layers produce the distribution parameters for the latent space.

A key architectural innovation is the separation of the latent space into two distinct streams: the prior gene program stream (n_latent dimensions) and the addon gene program stream (n_addon_latent dimensions). This dual-stream design allows the model to leverage existing biological knowledge through gene programs while simultaneously discovering novel biological programs from the data. Both streams share the same processing pipeline until the final graph convolution layers, where separate weight matrices produce distinct distribution parameters.

## Graph Convolution Strategies

NicheCompass provides two graph convolution strategies optimized for different analytical scenarios: the Graph Convolutional Network (GCN) approach and the Graph Attention Network v2 (GATv2) approach. These strategies implement fundamentally different aggregation mechanisms for processing neighborhood information, trading off between computational efficiency, interpretability, and representation capacity.

### GCNConv: Standard Graph Convolution

The GCNConv implementation follows the standard graph convolutional network formulation, performing mean aggregation of neighborhood features with learnable weight matrices. This approach provides computationally efficient message passing with strong theoretical foundations, making it suitable for large spatial atlases where processing speed is critical.

The GCN strategy processes each node's features by aggregating information from its immediate neighbors using a symmetric normalized adjacency matrix. This normalization scheme ensures stable gradient flow during training and prevents the vanishing or exploding gradient problems that can occur in deep graph neural networks. The aggregated neighborhood information is then transformed through a learned weight matrix and passed through a nonlinear activation function.

### GATv2Conv: Attention-Based Graph Convolution

The GATv2Conv strategy implements the second-generation graph attention network, which learns dynamic attention weights for each neighbor during training. This approach allows the model to automatically identify which neighboring cells contribute most significantly to each cell's representation, enabling nuanced capture of spatial gradients and microenvironmental influences.

The GATv2 mechanism computes attention coefficients between each pair of connected nodes based on their learned feature representations. These coefficients are normalized across neighbors using softmax, creating a weighted aggregation scheme where each neighbor's contribution is proportional to its learned importance. The architecture supports multi-head attention, where multiple independent attention mechanisms operate in parallel and their outputs are averaged, enhancing representational capacity and stabilizing training.

*Table: Graph Convolution Strategy Comparison*

| **Feature** | **GCNConv** | **GATv2Conv** |
|------------|-------------|---------------|
| **Aggregation Mechanism** | Mean aggregation with symmetric normalization | Learnable attention-weighted aggregation |
| **Computational Complexity** | O(\|E\| × d) | O(\|E\| × d × n_heads) |
| **Interpretability** | Fixed neighbor importance | Learnable neighbor attention weights |
| **Parameter Efficiency** | Lower (single weight matrix) | Higher (attention parameters per head) |
| **Use Case** | Large-scale spatial atlases | Fine-grained microenvironment analysis |
| **Attention Heads** | Not applicable | Configurable (default: 4) |

The choice between GCNConv and GATv2Conv depends on analytical requirements. GCNConv provides faster training and inference, making it suitable for processing large spatial atlases with millions of cells. GATv2Conv offers richer representations through learnable attention, beneficial when understanding cell-cell communication patterns or identifying spatial gradients is paramount.

> [!TIP]
> The GATv2Conv implementation uses the `concat=False` parameter, meaning attention heads are averaged rather than concatenated. This design choice maintains the output dimensionality while still benefiting from multi-head attention's representational advantages and training stability.

## Categorical Covariates Integration

The encoder supports flexible integration of categorical covariates such as batch labels, experimental conditions, or sample identifiers through learned embedding layers. This capability enables batch correction and metadata-aware representation learning without requiring separate post-hoc normalization steps.

Two injection modes determine where categorical covariates are incorporated into the encoder pipeline: input mode and hidden mode. In input mode, categorical embeddings are concatenated with the raw omics features before the first fully-connected layer, allowing the model to learn feature transformations that are conditioned on categorical metadata. In hidden mode, embeddings are concatenated after the fully-connected layers but before graph convolution, enabling the model to learn spatial aggregation strategies that are modulated by categorical information.

The embedding dimensions for each categorical variable are configurable through the `cat_covariates_embeds_nums` parameter. The total embedding dimension is calculated as the sum of all individual embedding dimensions when the encoder is configured to receive covariate embeddings. This flexible design accommodates heterogeneous categorical covariates with varying cardinalities and significance levels.

During the forward pass, categorical indices are first transformed into dense embeddings using the learned embedding tables. These embeddings are then concatenated across all categorical variables and either added to the input features or hidden representations depending on the configured injection mode. The injection location can be independently controlled for the encoder, gene expression decoder, and chromatin accessibility decoder, providing fine-grained control over which components of the model receive categorical information.

## Latent Space Parameterization

The encoder produces parameters for two independent Gaussian distributions in the latent space: one for the prior gene program dimensions and one for the addon gene program dimensions. This dual-distribution design enables the variational autoencoder framework to separate known biological programs from novel discoveries while maintaining statistical rigor through variational inference.

The mean (μ) and log standard deviation (logσ) parameters are computed through separate graph convolution layers with independent weight matrices. This separation allows the model to learn distinct transformation functions for localization and scale parameters, capturing different aspects of the latent space structure. The logσ parameterization ensures numerical stability by preventing negative standard deviations and simplifying gradient computation.

The latent representation is sampled using the reparameterization trick: z = μ + σ ⊙ ε, where ε is sampled from a standard normal distribution. This approach enables end-to-end backpropagation through the sampling operation while maintaining stochasticity necessary for variational inference. The sampled latent vector concatenates the prior and addon gene program dimensions, creating a unified representation that is passed to downstream decoders.

> [!TIP]
> The variational formulation enables regularization through KL divergence loss, preventing overfitting and encouraging the latent space to follow a standard normal distribution. This is particularly important for addon gene programs, which would otherwise be unconstrained and could overfit to spurious patterns in the training data.

## Encoder Configuration Parameters

The encoder exposes a comprehensive set of configuration parameters that control architectural decisions and training behavior. These parameters enable customization for different analytical scenarios, data modalities, and computational constraints.

*Table: Encoder Configuration Parameters*

| **Parameter** | **Type** | **Default** | **Description** |
|--------------|----------|-------------|-----------------|
| `n_input` | int | - | Number of input omics features |
| `n_cat_covariates_embed_input` | int | 0 | Dimension of concatenated covariate embeddings |
| `n_hidden` | int | - | Hidden dimension after FC layers |
| `n_latent` | int | - | Dimension of prior GP latent space |
| `n_addon_latent` | int | 100 | Dimension of addon GP latent space |
| `n_fc_layers` | int | 1 | Number of fully-connected layers (1 or 2) |
| `n_layers` | int | 1 | Number of graph convolution layers (1 or 2) |
| `conv_layer` | str | "gatv2conv" | Graph convolution type: "gcnconv" or "gatv2conv" |
| `cat_covariates_embed_mode` | str | "input" | Covariate injection: "input" or "hidden" |
| `n_attention_heads` | int | 4 | Number of attention heads (GATv2 only) |
| `dropout_rate` | float | 0.0 | Dropout probability for hidden layers |
| `activation` | nn.Module | nn.ReLU | Activation function |
| `use_bn` | bool | True | Apply batch normalization to μ |

The hidden dimension (`n_hidden`) serves as a bottleneck between the input feature space and latent space, controlling the trade-off between representational capacity and compression efficiency. Larger hidden dimensions preserve more information through the transformation pipeline but increase computational cost and memory requirements.

The number of graph convolution layers (`n_layers`) determines the receptive field of each node. With one layer, nodes only incorporate information from immediate neighbors. With two layers, nodes incorporate information from two-hop neighbors, capturing broader spatial context. Deeper architectures are not supported to avoid over-smoothing, where repeated aggregation causes node representations to become indistinguishable.

## Forward Pass Execution

The encoder's forward pass implements a deterministic transformation from input features to latent space parameters, incorporating both fully-connected and graph-based transformations along with optional categorical covariate integration.

Execution begins with optional concatenation of categorical covariates embeddings to the input features when `cat_covariates_embed_mode` is set to "input". This conditioned feature vector then passes through the first fully-connected layer with dropout and activation applied. If two fully-connected layers are configured, the output passes through a second layer followed by batch normalization.

After fully-connected transformation, categorical embeddings are optionally concatenated if the mode is set to "hidden", providing conditioned representations for the graph convolution stage. The graph convolution layers then process the features using the spatial connectivity defined by `edge_index`. For two-layer configurations, an intermediate convolution layer transforms features within the hidden dimension before the final layers.

The final graph convolution layers produce separate μ and logσ parameters for both the prior and addon gene program streams. When `n_addon_latent` is greater than zero, the addon stream parameters are concatenated to the prior stream parameters, creating the complete latent space parameterization. The concatenated parameters are filtered for the current batch using the `batch_idx` parameter, enabling efficient mini-batch training.

## Integration with VGPGAE Module

The encoder operates as a core component within the VGPGAE module, which orchestrates the complete variational autoencoder pipeline. The VGPGAE module instantiates the encoder with parameters derived from the data characteristics and analysis configuration, ensuring appropriate dimensionality matching between input features, hidden representations, and latent space.

Categorical covariates embeddings are created and managed by the VGPGAE module using `nn.Embedding` layers. During the forward pass, these embeddings are retrieved based on categorical indices and passed to the encoder based on the injection configuration. The module supports per-component injection control, allowing covariates to be included selectively in the encoder, gene expression decoder, and chromatin accessibility decoder.

The VGPGAE module applies logarithmic transformation to omics features before encoder input when `log_variational` is enabled. This transformation improves numerical stability by compressing the dynamic range of count data, which is particularly important for RNA-seq and ATAC-seq measurements that span several orders of magnitude.

During training, the encoder outputs are filtered for the current batch using appropriate indexing schemes. For node-level batches (omics decoder), the first `node_batch_size` nodes are selected. For edge-level batches (graph decoder), nodes participating in sampled edges are selected. This batch filtering ensures efficient memory usage and enables mini-batch training on large spatial datasets.

Sources: [encoders.py](src/nichecompass/nn/encoders.py#L12-L209), [vgpgae.py](src/nichecompass/modules/vgpgae.py#L146-L345), [vgpgae.py](src/nichecompass/modules/vgpgae.py#L541-L932)

## Next Steps

The encoder transforms spatial omics data into latent gene program representations, which serve as input to downstream components. To understand how these latent representations are used for reconstruction tasks, continue to [Decoder Architectures for Multi-modal Reconstruction](7-decoder-architectures-for-multi-modal-reconstruction). For comprehensive understanding of the complete model framework, refer to [Variational Gene Program Graph Autoencoder (VGPGAE)](5-variational-gene-program-graph-autoencoder-vgpgae). To learn about the loss functions that train the encoder parameters, see [Loss Functions and Optimization Objectives](8-loss-functions-and-optimization-objectives).