# GeneEffect Head and Features

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/model/features.py](src/model/features.py)
- [src/model/geneeffect.py](src/model/geneeffect.py)
- [src/model/initialization.py](src/model/initialization.py)
- [src/model/losses.py](src/model/losses.py)
- [src/model/normalization.py](src/model/normalization.py)
- [tests/test_geneeffect_e2e.py](tests/test_geneeffect_e2e.py)
- [tests/test_geneeffect_features.py](tests/test_geneeffect_features.py)
- [tests/test_geneeffect_head.py](tests/test_geneeffect_head.py)

</details>



## Purpose and Scope
This page details the implementation of the GeneEffect prediction head, the five input feature blocks (`delta_proj`, `s`, `q_sc`, `e_g`, `z_c`), the deterministic sparse Johnson-Lindenstrauss projection, the `BlockStandardizer` normalization layer, checkpoint initialization/warm-start protocols, and the Huber-based training loss.

Sources: [`src/model/geneeffect.py:22-142`]() [`src/model/features.py:42-138`]() [`src/model/normalization.py:25-169`]() [`src/model/initialization.py:52-124`]() [`src/model/losses.py:81-97`]()

---

## 1. Five-Feature Residual Head Architecture
The GeneEffect prediction system evaluates gene dependency scores by combining five distinct feature blocks through a residual MLP head, `GeneEffectResidualHead` [src/model/head.py:10](). The architecture is governed by `GeneEffectBlockConfig` [src/model/head.py:8](), which allows enabling or disabling individual feature blocks (e.g., for ablation studies).

The five feature blocks comprise:
1. `delta_proj`: Projected expression shift vector via `FixedSparseProjection` [src/model/features.py:71]().
2. `s`: Summary distribution statistics (energy distance, mean shift, own-gene shift, etc.) [src/model/features.py:47-54]().
3. `q_sc`: Single-cell query embeddings representing the cell line context.
4. `e_g`: ESM2 protein sequence embedding for the target gene [src/model/initialization.py:15]().
5. `z_c`: Basal state summary of the cell line context [src/model/losses.py:18-21]().

```mermaid
graph TD
    sub5[""OnlineConditionBatch"" --> src_model_geneeffect_GeneEffectE2EModel_condition_features]
    src_model_geneeffect_GeneEffectE2EModel_condition_features --> src_model_features_compute_condition_feature_batch
    src_model_features_compute_condition_feature_batch --> f_delta[""FeatureBatch.delta_proj""]
    src_model_features_compute_condition_feature_batch --> f_s[""FeatureBatch.s""]
    f_q[""FeatureBatch.q_sc""] --> norm[""BlockStandardizer.transform""]
    f_e[""FeatureBatch.e_g""] --> norm
    f_z[""FeatureBatch.z_c""] --> norm
    f_delta --> norm
    f_s --> norm
    norm --> head[""GeneEffectResidualHead""]
    head --> delta_hat[""delta_hat (GeneEffect Residual)""]
```
*Figure 1: Data flow from raw online condition batches through feature extraction, standardization, and the GeneEffect residual head.*

Sources: [`src/model/geneeffect.py:66-94`]() [`src/model/features.py:42-65`](), [`src/model/head.py:7-17`]()

---

## 2. Feature Engineering and Sparse Projection
Feature extraction turns predicted cell bags and basal references into fixed-width tensors suitable for the residual head.

### Sparse Johnson-Lindenstrauss Projection
`FixedSparseProjection` implements a deterministic, data-independent sparse JL projection using the Achlioptas distribution at density $\frac{1}{\sqrt{4000}}$ [src/model/features.py:71-88](). The projection matrix is generated once from a fixed seed (`PROJECTION_SEED = 0`), ensuring reproducible lower-dimensional embeddings of the high-dimensional expression shift space ($\Delta \in \mathbb{R}^{2 \times \text{HVG\_WIDTH}}$) [src/model/features.py:13-16, 71-88]().

```python
# Core sparse projection transformation pattern
projected = delta @ components.transpose(0, 1)
```
*Source code reference:* [`src/model/features.py:100-115`]()

### Vectorized Condition Features
`compute_condition_feature_batch` computes condition features over batch elements without incurring Python/CUDA round-trip overhead per row [src/model/features.py:158-167](). It evaluates summary metrics including energy distance, mean shifts, cosine similarities, and own-gene shift indicators [src/model/features.py:47-54, 100-115]().

Sources: [`src/model/features.py:42-191`]() [`tests/test_geneeffect_features.py:21-40`]()

---

## 3. Normalization via `BlockStandardizer`
Continuous feature blocks (`delta_proj`, `s`, `q_sc`, `e_g`, `z_c`) are standardized using `BlockStandardizer` before entering the residual head [src/model/normalization.py:15-26]().

### Fitting and Streaming
- **Batch Fitting (`fit`)**: Computes exact mean and scale across a pre-materialized training block mapping [src/model/normalization.py:40-75]().
- **Streamed Fitting (`fit_batches`)**: Computes population statistics incrementally using Welford's algorithm across data streams without loading all training rows into memory simultaneously [src/model/normalization.py:77-144]().
- **Constant Columns**: Features with zero standard deviation are automatically tracked and their scale factors are safely clamped to `1.0` to prevent division by zero [src/model/normalization.py:68-71]().

```mermaid
graph TD
    raw_blocks[""train_blocks / train_batches""] --> fit_method[""BlockStandardizer.fit / fit_batches""]
    fit_method --> stats[""_BlockStats (mean, scale, constant_columns)""]
    stats --> transform_method[""BlockStandardizer.transform""]
    transform_method --> standardized_tensor[""Normalized Feature Tensor""]
```
*Figure 2: Statistical fitting and transformation lifecycle inside `BlockStandardizer`.*

Sources: [`src/model/normalization.py:25-169`]()

---

## 4. Initialization and Warm-Start Protocols
Model construction and checkpoint restoration are managed in `src/model/initialization.py` [src/model/initialization.py:52-177]().

### Warm-Start State Dicts
`warm_start_state_dict` loads checkpoints manually via `torch.load` and filters keys by shape compatibility [src/model/initialization.py:52-105](). Any tensors with mismatched shapes are skipped while preserving freshly initialized parameters for new heads or dimensions, returning a `WarmStartReport` detailing loaded, skipped, missing, and unexpected keys [src/model/initialization.py:35-124]().

### Joint Model Assembly
`build_joint_model` constructs the complete `GeneEffectE2EModel` combining the STATE backbone, ESM2 adapters, projection matrices, and standardizers either from scratch or by restoring a fully self-contained checkpoint [src/model/initialization.py:134-177]().

Sources: [`src/model/initialization.py:34-177`]()

---

## 5. Objective Functions and Losses
Dependency prediction models are supervised using robust regression losses combined with permutation-invariant bag pooling [src/model/losses.py:11-97]().

### Moment Pooling
`moment_pool` summarizes per-cell bags into permutation-invariant representations [src/model/losses.py:11-46](). By default (`moments=2`), it computes concatenated mean and biased population variance across cell dimensions [src/model/losses.py:18-31]().

### GeneEffect Huber Loss
`geneeffect_loss` computes the mean FP32 Huber loss ($\delta=1.0$) exclusively over valid, labeled gene-context pairs, incorporating fail-closed checks for non-finite values or empty batches [src/model/losses.py:81-97]().

```python
def geneeffect_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    return F.huber_loss(
        prediction.float()[valid], target.float()[valid], delta=1.0, reduction="mean"
    )
```
*Source code reference:* [`src/model/losses.py:81-97`]()

Sources: [`src/model/losses.py:11-97`]() [`tests/test_geneeffect_head.py:1-12`]()

---