

The decoder architectures in NicheCompass transform latent gene program representations back into observable multi-modal data, enabling reconstruction of both RNA expression and ATAC-seq chromatin accessibility profiles. These decoders serve a dual purpose: they reconstruct the input data through the variational autoencoder framework while providing interpretable mappings between gene programs and biological features. The architecture supports both graph-level edge reconstruction and node-level omics feature reconstruction, with two primary decoder strategies offering different trade-offs between biological interpretability and model flexibility.

## Graph-Level Edge Reconstruction

The **CosineSimGraphDecoder** reconstructs the spatial neighborhood graph by computing pairwise similarities between latent representations of cell nodes. This decoder operates on edge-level batches during training, processing concatenated latent feature vectors from source and target nodes to predict edge existence probabilities.

The decoder employs a simple yet effective architecture: after applying dropout regularization for robustness, it computes element-wise cosine similarity between source and target node embeddings. The resulting logits represent the probability that an edge exists between the nodes, with the sigmoid activation integrated into the binary cross-entropy loss for computational efficiency. This design choice eliminates redundant computations by avoiding explicit sigmoid operations during forward propagation.

**Key Implementation Details:**
- Input tensor dimensions: `4 × edge_batch_size × n_gps` (concatenating positive/negative edges for source and target nodes)
- Output: edge logits of dimension `2 × edge_batch_size`
- The decoder processes pairs `[edge_label_index[0], edge_label_index[1]]` to reconstruct positive and negative edges
- Dropout rate controls regularization strength during graph reconstruction training

Sources: [decoders.py](src/nichecompass/nn/decoders.py#L14-L63)

## Masked Omics Feature Decoders

The **MaskedOmicsFeatureDecoder** represents NicheCompass's core innovation for interpretable multi-modal reconstruction. This architecture leverages prior knowledge gene programs to constrain connections between latent space dimensions and observable features, ensuring that each gene program only reconstructs biologically relevant genes or peaks.

### Architecture Components

The masked decoder utilizes the **AddOnMaskedLayer** as its fundamental building block, which implements three parallel linear transformations:

1. **Masked Layer**: Applies a binary mask ensuring prior gene programs only contribute to reconstruction of features they're biologically associated with
2. **Add-On Layer**: Handles unmasked features through a separate pathway, allowing latent programs to reconstruct genes without prior annotation
3. **Categorical Covariates Embedding Layer**: Incorporates batch or sample information as additional inputs when specified

The output from these three pathways is aggregated and passed through a softmax activation, ensuring the reconstructed gene expression sums to a normalized probability distribution. This normalization is then scaled by the observed library size using `nb_means = exp(log_library_size) × nb_means_normalized`, enabling negative binomial distribution parameter estimation appropriate for count-based omics data.

### Multi-Modal Support

The masked decoder architecture seamlessly handles both RNA and ATAC modalities through separate instantiations:

- **RNA Decoders**: `target_rna_decoder` and `source_rna_decoder` reconstruct gene expression for target and source entities (typically representing different spatial positions or cell types)
- **ATAC Decoders**: `target_atac_decoder` and `source_atac_decoder` reconstruct chromatin accessibility peaks when multi-modal data is available

Critically, the ATAC decoders leverage the gene-peaks mapping matrix (`gene_peaks_mask_`) to ensure consistency across modalities. When a gene is turned off through L1 regularization in the RNA decoder, the corresponding peaks are automatically deactivated in the ATAC decoder via the dynamic masking mechanism.

### Dynamic Masking and Regularization

The decoder supports dynamic masking that adapts during training to implement feature selection:

- **Gene Programs**: Inactive gene programs (those with low average absolute activation) can be masked out entirely, preventing them from influencing reconstruction
- **Individual Features**: Genes can be selectively turned off through L1 regularization, with their corresponding ATAC peaks automatically masked for cross-modal consistency
- **Add-On Programs**: Unmasked genes are handled through the add-on pathway, enabling discovery of novel gene programs not present in prior knowledge

This dynamic masking is computed as the element-wise product of static masks (defined by prior knowledge) and dynamic masks (determined by training progress and regularization):

```
effective_mask = static_mask × dynamic_mask
```

Only features with at least one active connection are reconstructed, preventing the softmax normalization from being influenced by disconnected features.

Sources: [decoders.py](src/nichecompass/nn/decoders.py#L65-L170), [layers.py](src/nichecompass/nn/layers.py#L13-L143)

## Fully Connected Omics Feature Decoders

The **FCOmicsFeatureDecoder** provides an alternative architecture that sacrifices interpretability for modeling flexibility. This decoder uses standard fully connected layers without the constraint of gene program masks, allowing the model to learn arbitrary latent-to-observation mappings.

### Architecture Variants

The FC decoder supports one or two hidden layers:

- **Single Layer**: `Linear → Softmax` for direct mapping from latent space to feature space
- **Two Layers**: `Linear → ReLU → Linear → Softmax` for additional non-linear transformation capacity

The two-layer variant introduces intermediate representations that can capture more complex relationships between gene programs and observed features, though this comes at the cost of reduced interpretability.

### Use Cases

The fully connected decoder is beneficial when:
- Prior knowledge gene programs are incomplete or unavailable
- The modeling task requires capturing subtle, non-linear relationships
- Benchmarking against the masked decoder to assess the value of biological constraints

Despite the unconstrained architecture, the FC decoder maintains the same library size scaling and negative binomial distribution assumptions as the masked decoder, ensuring consistency with the statistical properties of count-based omics data.

Sources: [decoders.py](src/nichecompass/nn/decoders.py#L172-L269)

## Decoder Architecture Comparison

| Feature | Masked Decoder | Fully Connected Decoder |
|---------|---------------|------------------------|
| **Interpretability** | High (gene programs directly map to features) | Low (arbitrary latent-to-observation mapping) |
| **Biological Constraints** | Enforced through prior knowledge masks | None (learned from data) |
| **Feature Selection** | Built-in via L1 regularization and dynamic masking | Requires external implementation |
| **Cross-Modal Consistency** | Automatic via gene-peaks mapping | Manual implementation required |
| **Model Capacity** | Constrained by mask structure | Flexible (configurable depth) |
| **Prior Knowledge** | Required for optimal performance | Optional |
| **Non-Linearity** | Single linear layer per pathway | Configurable (1-2 layers with ReLU) |

## Multi-Modal Decoder Integration

NicheCompass's decoder architecture seamlessly integrates RNA and ATAC reconstruction through coordinated masking strategies. The integration operates through three key mechanisms:

### Gene-to-Peak Mapping

The `gene_peaks_mask_` matrix establishes which ATAC peaks are physically associated with each gene (typically based on genomic proximity). During decoder initialization, this mask is used to propagate masking decisions from RNA to ATAC:

```python
# Create ATAC add-on mask from RNA add-on mask
atac_decoder_addon_mask = torch.mm(
    rna_decoder_addon_mask.to(torch.int),
    gene_peaks_mask_.to(torch.int)
).to(torch.bool)
```

This ensures that if a gene is masked in the RNA decoder (either statically via prior knowledge or dynamically via regularization), all associated peaks are automatically masked in the ATAC decoder.

### Dynamic Decoder Mask Updates

During training, the decoder updates dynamic masks based on gene program activity. The `update_atac_dynamic_decoder_mask` parameter in the forward pass controls whether these updates propagate to the ATAC decoders:

- When enabled: Genes turned off in RNA decoders automatically deactivate their associated peaks in ATAC decoders
- When disabled: ATAC masks remain independent, useful for studying modality-specific programs

### Dispersion Parameter Learning

Both RNA and ATAC decoders learn modality-specific dispersion parameters (`target_rna_theta`, `source_rna_theta`, `target_atac_theta`, `source_atac_theta`) for the negative binomial distribution. These parameters are learned separately for target and source entities, allowing the model to capture different variability patterns across spatial contexts or sample types.

Sources: [vgpgae.py](src/nichecompass/modules/vgpgae.py#L309-L460)

## Decoder Forward Pass and Output Generation

The forward pass through the decoder architecture occurs within the VGPGAE module's forward method, which processes node-level batches for omics reconstruction. The sequence of operations includes:

### Input Preparation

1. **Latent Representation Extraction**: The encoder outputs μ (mean) and logσ (log standard deviation) parameters, from which latent features z are sampled via the reparameterization trick
2. **Optional Active Gene Program Filtering**: When `use_only_active_gps=True`, inactive gene programs are masked out by setting their latent scores to zero
3. **Categorical Covariate Embedding**: Sample-specific covariates (e.g., batch, condition) are embedded and concatenated with latent features if configured

### Decoder Execution

Each decoder (target RNA, source RNA, target ATAC, source ATAC) processes its inputs independently:

```python
# Concatenate covariates if present
if cat_covariates_embed is not None:
    z = torch.cat((z, cat_covariates_embed), dim=-1)

# Compute normalized means via decoder-specific pathway
nb_means_normalized = decoder(z, dynamic_mask)

# Scale by library size
nb_means = torch.exp(log_library_size) * nb_means_normalized
```

The masked decoder additionally receives the `dynamic_mask` parameter, which allows on-the-fly modification of which gene programs can contribute to which features.

### Output Processing

The decoder outputs negative binomial mean parameters, which are combined with the learned dispersion parameters (θ) to compute the reconstruction loss. Only features with active connections (determined by the product of static and dynamic masks) are included in the loss computation, preventing disconnected features from artificially influencing the normalization.

Sources: [vgpgae.py](src/nichecompass/modules/vgpgae.py#L541-L700)

## Architectural Flexibility and Configuration

NicheCompass provides extensive configuration options for tailoring decoder architectures to specific use cases:

### Decoder Selection

The `use_fc_decoder` parameter controls whether masked or fully connected decoders are instantiated:

- `use_fc_decoder=False` (default): Use masked decoders for interpretability
- `use_fc_decoder=True`: Use fully connected decoders for flexibility

### Layer Configuration

For fully connected decoders, `fc_decoder_n_layers` specifies depth:
- `n_layers=1`: Single linear layer with softmax
- `n_layers=2`: Two linear layers with ReLU activation between them

### Categorical Covariate Injection

The `cat_covariates_embeds_injection` list determines where covariate embeddings are added:
- `"encoder"`: Inject into encoder (affects latent space)
- `"gene_expr_decoder"`: Inject into RNA decoders (affects expression reconstruction)
- `"chrom_access_decoder"`: Inject into ATAC decoders (affects accessibility reconstruction)

This fine-grained control allows researchers to experiment with different integration strategies without modifying the core architecture.

Sources: [vgpgae.py](src/nichecompass/modules/vgpgae.py#L146-L300)

## Next Steps

Understanding the decoder architectures provides the foundation for comprehending how NicheCompass optimizes these reconstructions through carefully designed loss functions. The next logical step is to examine [Loss Functions and Optimization Objectives](8-loss-functions-and-optimization-objectives) to understand how reconstruction accuracy is measured and balanced with other training objectives such as edge reconstruction, regularization, and contrastive learning. For context on how the encoder produces the latent representations that feed into these decoders, refer back to [Encoder Design and Graph Convolution Strategies](6-encoder-design-and-graph-convolution-strategies).