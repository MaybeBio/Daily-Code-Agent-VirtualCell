

This page provides a comprehensive guide to evaluating NicheCompass models through training metrics and benchmarking frameworks. The evaluation ecosystem spans two complementary dimensions: **training-time metrics** that monitor model convergence and performance during training, and **post-training benchmarking metrics** that assess the biological and spatial fidelity of learned latent representations against ground truth spatial data. These evaluation tools enable rigorous quantitative assessment of model quality across spatial transcriptomics applications.

## Training-Time Evaluation Metrics

Training metrics provide real-time feedback on model performance during the training process, computed on validation data at epoch intervals and after training completion. The `eval_metrics()` function in `src/nichecompass/train/metrics.py` implements a comprehensive suite of evaluation metrics designed for the graph-based multi-modal architecture of NicheCompass.
Sources: [metrics.py](/src/nichecompass/train/metrics.py#L14-L137)

### Edge Reconstruction Evaluation

The primary evaluation focuses on the graph decoder's ability to reconstruct spatial connectivity patterns. This is assessed through threshold-independent metrics computed from edge reconstruction probabilities and ground truth edge labels:

- **AUROC (Area Under Receiver Operating Characteristic Curve)**: Measures the model's ability to distinguish between true positive edges (actual spatial connections) and negative edges (non-neighbors). Higher values indicate better discriminative performance. Computed using `sklearn.metrics.roc_auc_score()` on edge reconstruction probabilities.
Sources: [metrics.py](/src/nichecompass/train/metrics.py#L100-L101)

- **AUPRC (Area Under Precision-Recall Curve)**: Particularly informative when dealing with imbalanced positive/negative edge ratios, as is typical in sparse spatial graphs. Computed using `sklearn.metrics.average_precision_score()`.
Sources: [metrics.py](/src/nichecompass/train/metrics.py#L102-L104)

- **Best Accuracy Score**: The maximum classification accuracy achievable by optimizing the probability threshold over a grid search from 0.01 to 1.0 with 0.005 increments. The optimal threshold is identified and stored alongside the score.
Sources: [metrics.py](/src/nichecompass/train/metrics.py#L107-L114)

- **Best F1 Score**: The harmonic mean of precision and recall under the optimal threshold, balancing false positives and false negatives. Computed through the same threshold search strategy.
Sources: [metrics.py](/src/nichecompass/train/metrics.py#L117-L121)

### Omics Reconstruction Evaluation

For multi-modal models that predict gene expression and chromatin accessibility, mean squared error (MSE) metrics evaluate the accuracy of omics data reconstruction. The function computes separate MSE scores for both target and source modalities (e.g., RNA and ATAC) by comparing predictions against ground truth values.
Sources: [metrics.py](/src/nichecompass/train/metrics.py#L45-L58)

### Categorical Covariate Evaluation

When categorical covariates are used for contrastive learning, the function evaluates how well the model distinguishes edges connecting cells from different categories. For each categorical covariate, it computes the mean similarity difference by comparing reconstruction probabilities of edges between same-category cells versus different-category cells. Positive values indicate the model successfully learns to separate different categorical groups.
Sources: [metrics.py](/src/nichecompass/train/metrics.py#L61-L72)

### Evaluation Integration with Training

The `Trainer` class integrates these metrics into the training workflow through two key methods:

1. **`eval_epoch()`**: Called during training loops to compute evaluation metrics on validation data, accumulating predictions across batches and computing metrics. This supports monitoring training progress without incurring excessive computational overhead.
Sources: [trainer.py](/src/nichecompass/train/trainer.py#L492-L550)

2. **`eval_end()`**: Called after training completion to compute comprehensive evaluation metrics using all validation data. This final evaluation produces detailed metric summaries including AUROC, AUPRC, best accuracy, best F1, and modality-specific MSE scores, which are printed and optionally logged to MLflow.
Sources: [trainer.py](/src/nichecompass/train/trainer.py#L591-L707)

### Visualization

The `plot_eval_metrics()` function enables visualization of metric trajectories over epochs, supporting analysis of training dynamics and convergence patterns. This visualization is particularly useful for identifying training instabilities, optimal stopping points, and the impact of hyperparameter modifications.
Sources: [metrics.py](/src/nichecompass/train/metrics.py#L138-L167)

## Post-Training Benchmarking Metrics

The `compute_benchmarking_metrics()` function in `src/nichecompass/benchmarking/metrics.py` provides a comprehensive benchmarking framework for evaluating spatial transcriptomics models. This framework assesses how well the learned latent representation preserves spatial organization, biological structure, and enables niche identification compared to ground truth spatial coordinates.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L28-L47)

### Benchmarking Architecture

The benchmarking system operates on a unified computational graph architecture:

```mermaid
flowchart TD
    A[AnnData Object] --> B[Spatial Coordinates<br/>adata.obsm['spatial']]
    A --> C[Latent Representation<br/>adata.obsm['nichecompass_latent']]
    A --> D[Cell Type Annotations<br/>adata.obs['cell_type']]
    A --> E[Batch Annotations<br/>adata.obs['batch']]
    
    B --> F[Spatial KNN Graph<br/>k=15, 50, or 90]
    C --> G[Latent KNN Graph<br/>k=15, 50, or 90]
    
    F --> H[Spatial Conservation Metrics<br/>CAS, MLAMI, CLISIS, GCS]
    G --> H
    
    G --> I[Niche Coherence Metrics<br/>CNMI, CARI, CASW, CLISI, NASW]
    D --> I
    
    G --> J[Batch Correction Metrics<br/>BASW, BLISI, KBET, PCR]
    D --> J
    E --> J
    
    H --> K[Benchmarking Dictionary]
    I --> K
    J --> K
```

This architecture enables parallel computation of multiple metrics by constructing k-nearest neighbor graphs with appropriate neighborhood sizes for different metric classes (k=15 for connectivity metrics, k=50 for KBET, k=90 for LISI-based metrics).
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L83-L125)

### Spatial Conservation Metrics

Spatial conservation metrics evaluate how well the latent space preserves the spatial organization present in the original tissue architecture.

#### Global Spatial Conservation

**CAS (Cell Type Affinity Similarity)**: A supervised metric measuring global preservation of cell-type-pair edges between spatial and latent nearest neighbor graphs. CAS quantifies how accurately the latent representation maintains neighborhood relationships between different cell types, with values ranging from 0 (no preservation) to 1 (perfect preservation). The metric builds cell-cell contact maps for both spatial and latent spaces, then computes their similarity through neighborhood enrichment analysis.
Sources: [cas.py](/src/nichecompass/benchmarking/cas.py#L19-L100)

> [!TIP]
> CAS is particularly valuable for validating that niche-specific cell-type relationships are preserved in the latent representation, which is critical for downstream biological interpretation.

**MLAMI (Maximum Leiden Adjusted Mutual Information)**: An unsupervised global metric assessing spatial organization preservation through clustering consistency. MLAMI computes Leiden clusterings across multiple resolutions for both spatial and latent graphs, then compares cluster assignments using adjusted mutual information. The maximum AMI across resolutions is reported, with higher values indicating better preservation of global spatial patterns.
Sources: [mlami.py](/src/nichecompass/benchmarking/mlami.py#L18-L31)

#### Local Spatial Conservation

**CLISIS (Cell Type Local Inverse Simpson's Index Similarity)**: A supervised local metric evaluating preservation of neighborhood heterogeneity. CLISIS computes the Cell Type Local Inverse Simpson's Index (CLISI) for each cell in both spatial and latent graphs, measuring the effective number of different cell types in local neighborhoods. The similarity between spatial and latent CLISI scores is then computed, with values closer to 1 indicating better preservation of local cell-type mixing patterns.
Sources: [clisis.py](/src/nichecompass/benchmarking/clisis.py#L18-L80)

> [!TIP]
> CLISIS is computed using k=90 neighbors to capture broader neighborhood context, making it sensitive to local organizational patterns at the scale of tissue microenvironments.

**GCS (Graph Connectivity Similarity)**: An unsupervised local metric measuring edge preservation between spatial and latent graphs. GCS computes the normalized Frobenius norm of the element-wise difference between spatial and latent adjacency matrices, with values ranging from 0 (no similarity) to 1 (identical connectivity). When batch information is available, GCS is computed per batch and averaged.
Sources: [gcs.py](/src/nichecompass/benchmarking/gcs.py#L16-L100)

### Niche Coherence Metrics

Niche coherence metrics assess the quality of cluster structure in the latent space, evaluating whether biologically meaningful niches form distinct, compact clusters.

**CNMI (Clustered Normalized Mutual Information)**: Measures the agreement between Leiden clusters computed on the latent space and ground truth cell type annotations using k-means clustering. Higher CNMI indicates that latent clusters align well with biological cell types.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L217-L230)

**CARI (Clustered Adjusted Rand Index)**: Complement to CNMI using adjusted Rand index instead of mutual information, providing another perspective on cluster-label agreement.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L217-L230)

**CASW (Cell Type Affinity Silhouette Width)**: Measures compactness and separation of cell type clusters using silhouette width analysis. Higher values indicate that cells of the same type are tightly clustered and well-separated from other types.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L232-L242)

**CLISI (Cell Type Local Inverse Simpson's Index)**: Directly computed on the latent space without spatial comparison, measuring the effective number of cell types in local neighborhoods. Higher values indicate better mixing of cell types, which can indicate either good biological representation or over-integration depending on the biological context.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L244-L254)

**NASW (Niche Average Silhouette Width)**: Evaluates clusterability of the latent space independent of ground truth annotations. NASW computes Leiden clusterings at multiple resolutions, then calculates the average silhouette width across all resolutions. Higher values indicate that the latent space contains well-defined, compact clusters suitable for niche identification.
Sources: [nasw.py](/src/nichecompass/benchmarking/nasw.py#L17-L80)

### Batch Correction Metrics

For multi-sample or multi-batch datasets, batch correction metrics evaluate how well the model integrates samples while preserving biological variation.

**BASW (Batch Average Silhouette Width)**: Measures batch mixing within cell type clusters using silhouette width analysis. Computed by treating batch as the label and cell type as the group, higher BASW indicates better mixing of batches within cell types.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L267-L277)

**BLISI (Batch Local Inverse Simpson's Index)**: A LISI-based metric evaluating batch mixing in local neighborhoods. Higher values indicate better batch mixing, but extremely high values may indicate over-integration where biological signal is removed along with batch effects.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L279-L289)

**KBET (K-nearest neighbor Batch Effect Test)**: A statistical test for batch mixing that evaluates whether the distribution of batches in local neighborhoods matches the global batch distribution. KBET is computed per cell type with k=50 neighbors, providing detailed insight into batch-specific integration quality.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L291-L300)

**PCR (Principal Component Regression)**: Quantifies the proportion of variance explained by batch in the latent space compared to a pre-integration reference (e.g., PCA on raw counts). Lower PCR values indicate successful batch effect removal while preserving biological variance.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L302-L315)

## Metric Selection Guide

Different applications require different metric combinations depending on the evaluation goals:

*Table: Recommended metric combinations for common evaluation scenarios*

| **Application Scenario** | **Primary Metrics** | **Secondary Metrics** | **Interpretation Focus** |
|------------------------|---------------------|----------------------|--------------------------|
| **Spatial Fidelity Validation** | CAS, CLISIS, GCS, MLAMI | - | Preservation of spatial organization and neighborhood relationships |
| **Niche Identification Assessment** | NASW, CNMI, CARI | CASW, CLISI | Quality of cluster structure for downstream niche analysis |
| **Multi-sample Integration** | BASW, BLISI, KBET, PCR | - | Batch mixing quality and biological signal preservation |
| **Comprehensive Benchmarking** | CAS, CLISIS, GCS, MLAMI, NASW, CNMI | BASW, BLISI, KBET | Full evaluation across spatial, biological, and technical dimensions |

## Implementation Workflow

The typical evaluation workflow follows these steps:

```mermaid
flowchart LR
    A[Train NicheCompass Model] --> B[Extract Latent Representation<br/>adata.obsm['nichecompass_latent']]
    B --> C[Compute Training Metrics<br/>eval_metrics]
    B --> D[Compute Benchmarking Metrics<br/>compute_benchmarking_metrics]
    C --> E[Review AUROC, AUPRC, MSE]
    D --> F[Compute KNN Graphs<br/>k=15, 50, 90]
    F --> G[Calculate Spatial Conservation<br/>CAS, MLAMI, CLISIS, GCS]
    F --> H[Calculate Niche Coherence<br/>NASW, CNMI, CARI]
    F --> I[Calculate Batch Correction<br/>BASW, BLISI, KBET, PCR]
    G --> J[Generate Benchmarking Report]
    H --> J
    I --> J
    E --> K[Combine Training & Benchmarking Results]
    J --> K
```

The `compute_benchmarking_metrics()` function manages this workflow automatically, computing appropriate KNN graphs based on selected metrics and handling graph caching to avoid redundant computations. The function supports MLflow integration for experiment tracking and parallelization through the `n_jobs` parameter for large-scale evaluations.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L28-L47)

## Parameter Configuration

Effective benchmarking requires careful parameter selection based on dataset characteristics:

*Table: Key parameters for benchmarking configuration*

| **Parameter** | **Default Value** | **Description** | **Adjustment Guidelines** |
|---------------|-------------------|----------------|---------------------------|
| `metrics` | ["cas", "mlami", "clisis", "gcs", "cnmi", "nasw", "basw", "blisi", "kbet", "pcr"] | List of metrics to compute | Select based on evaluation goals, subset for faster evaluation |
| `n_neighbors` | 15, 50, 90 | KNN graph sizes | Automatically set per metric class; rarely needs manual adjustment |
| `spatial_key` | "spatial" | Key for spatial coordinates | Must match coordinate storage location in adata.obsm |
| `latent_key` | "nichecompass_latent" | Key for latent representation | Must match model output storage location |
| `cell_type_key` | "cell_type" | Key for cell type annotations | Required for supervised metrics |
| `batch_key` | None | Key for batch annotations | Required for batch correction metrics |
| `n_jobs` | 1 | Parallelization degree | Increase for faster KNN graph computation on large datasets |
| `seed` | 0 | Random seed for reproducibility | Set different seeds for robustness testing |

## Advanced Features

### Batch-Aware Evaluation

For multi-sample datasets, most metrics support batch-aware computation that either (1) computes metrics per batch and averages (e.g., GCS, CAS) or (2) evaluates batch-specific mixing quality (e.g., BASW, BLISI). This ensures that integration quality is assessed across technical variation while preserving biological signal.
Sources: [cas.py](/src/nichecompass/benchmarking/cas.py#L22-L25)

### Graph Caching

The system automatically caches KNN graphs in `adata.obsp` using keys like `"spatial_15knng_connectivities"` to avoid redundant computation across multiple benchmarking runs. Graphs are reused if they exist with the exact key, enabling incremental evaluation without recomputing expensive neighbor searches.
Sources: [utils.py](/src/nichecompass/benchmarking/utils.py#L13-L69)

### MLflow Integration

Both training and benchmarking metrics can be automatically logged to MLflow for experiment tracking by providing `mlflow_experiment_id`. This enables systematic comparison of hyperparameter configurations, model architectures, and preprocessing strategies.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L46), [trainer.py](/src/nichecompass/train/trainer.py#L696-L698)

> [!TIP]
> For large-scale benchmarking experiments, precompute KNN graphs once with multiple neighbor sizes and reuse them across multiple evaluation runs to significantly reduce computational time.

## Troubleshooting

Common issues during benchmarking and their resolutions:

- **CLISIS/BLISI Computation Failures**: These metrics can fail on datasets with extreme cell type imbalance or very small batch sizes. The implementation catches exceptions and returns 0.0, but consider increasing neighborhood size or merging rare cell types if failures are frequent.
Sources: [metrics.py](/src/nichecompass/benchmarking/metrics.py#L203-L207)

- **Memory Issues with Large Datasets**: The full benchmarking suite requires storing multiple KNN graphs (connectivities and distances for k=15, 50, 90). Reduce the number of metrics computed or increase available memory for very large datasets (>1M cells).

- **Slow KBET Computation**: KBET is computationally expensive as it performs statistical tests per cell type. Consider excluding KBET for initial exploratory analysis and only including it for final evaluation.

## Next Steps

After completing model evaluation, consider these follow-up analyses:

- **[Latent Representation Extraction and Visualization](22-latent-representation-extraction-and-visualization)**: Explore the learned latent space through dimensionality reduction and visualization techniques.

- **[Differential Gene Program Testing](23-differential-gene-program-testing)**: Leverage the high-quality latent representation for biological discovery through differential activity analysis.

- **[Niche Characterization and Gene Importance Computation](25-niche-characterization-and-gene-importance-computation)**: Use validated latent representations for niche identification and gene importance scoring.

- **[Hyperparameter Tuning and Best Practices](27-loss-weight-balancing-and-regularization-strategies)**: Optimize model performance based on evaluation metric insights.