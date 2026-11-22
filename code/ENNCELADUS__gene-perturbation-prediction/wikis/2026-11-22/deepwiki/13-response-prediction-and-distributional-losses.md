# Response Prediction and Distributional Losses

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [src/model/response.py](src/model/response.py)
- [tests/test_response_training.py](tests/test_response_training.py)

</details>



## Purpose and Scope
This page documents the technical design, implementation, and verification of response prediction and distributional loss formulations within the `src/model/response.py` and `tests/test_response_training.py` modules. The response prediction system evaluates gene perturbations at single-cell resolution by processing cell bags through the STATE backbone adapters, applying deterministic padding strategies to conform to fixed sentence-length constraints, and computing rigorous distributional metrics (`energy_distance` and `mean_delta_mse`) to capture both average shift and dispersion variations without mode collapse.

Sources:
- `src/model/response.py:1-250`
- `tests/test_response_training.py:1-177`

---

## 1. Response Prediction and Bag Padding (`predict_bags`)

When performing gene perturbation predictions across multi-cell conditions (bags), cell counts vary dynamically across cell lines and batches. However, the STATE backbone architecture enforces a strict fixed-window cell sentence length (`cell_sentence_len`) [src/model/response.py:18-24](). 

The `predict_bags` function coordinates ingestion of variable-length basal cell tensors, chunks them to match the required window sizes, handles deterministic padding via `_chunk_control_cell_indices`, forwards them through the network, and trims the outputs back to the original cell counts.

```
+-----------------------------------------------------------------+
|                        predict_bags()                           |
|  Inputs: controls (Sequence[Tensor]), genes (Sequence[str])     |
+-----------------------------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|               _state_window(model) Resolution                   |
|  Inspects model, DDP wrappers, and state_adapter for window     |
+-----------------------------------------------------------------+
                                 |
           +---------------------+---------------------+
           |                                           |
    (Window found)                              (No window / Test)
           |                                           |
           v                                           v
+-----------------------------+               +-------------------+
| _chunk_control_cell_indices |               | _seeded_model_    |
| - Deterministic NumPy RNG   |               |   forward()       |
| - Resampling w/ replacement |               +-------------------+
| - Chunks to cell_sentence_len|
+-----------------------------+
           |
           v
+-----------------------------------------------------------------+
|               _seeded_model_forward() execution                 |
|  - Training: standard rank RNG                                  |
|  - Evaluation: torch.random.fork_rng() isolated seed            |
+-----------------------------------------------------------------+
           |
           v
+-----------------------------------------------------------------+
|                     Trimming & Assembly                         |
|  Strips added padding rows; returns tuple of prediction tensors |
+-----------------------------------------------------------------+
```

Sources:
- `src/model/response.py:12-40`
- `src/model/response.py:78-96`
- `src/model/response.py:133-173`

---

## 2. Distributional Losses: Mean-Delta MSE and Energy Distance

To measure the accuracy of predicted cell response bags relative to observed experimental distributions, two primary loss formulations are utilized: `mean_delta_mse` and `energy_distance`.

### Mean-Delta MSE (`mean_delta_mse`)
Computes the mean squared error between the predicted and observed mean shifts relative to the control mean [src/model/response.py:43-55](). Explicitly retaining the control mean prevents the model from winning by simply reproducing absolute basal expression levels while ignoring the actual perturbation effect [src/model/response.py:48-52]().

$$\mathcal{L}_{\text{delta}} = \left\| \left(\frac{1}{N}\sum \hat{x} - \mu_{\text{ctrl}}\right) - \left(\frac{1}{M}\sum x - \mu_{\text{ctrl}}\right) \right\|_2^2$$

### Energy Distance (`energy_distance`)
Calculates the statistical energy distance between two empirical distributions (predicted cells $X$ and observed cells $Y$) [src/model/response.py:58-75]():

$$D_E(X, Y) = 2 \mathbb{E}\|X - Y\| - \mathbb{E}\|X - X'\| - \mathbb{E}\|Y - Y'\|$$

Unlike `mean_delta_mse`, energy distance is sensitive to changes in spread and dispersion [src/model/response.py:58-69](). This penalizes mode collapse—a failure mode where a model predicts the correct average perturbation shift while collapsing single-cell variance [src/model/response.py:62-66]().

Sources:
- `src/model/response.py:43-76`
- `tests/test_response_training.py:94-128`

---

## 3. RNG Isolation and Deterministic Evaluation

To guarantee reproducible evaluation metrics without polluting or mutating the global training random number generator state, `_seeded_model_forward` uses PyTorch's `fork_rng` context manager alongside explicit CUDA device seeding [src/model/response.py:114-131]().

```
+-----------------------------------------------------------------+
|                     _seeded_model_forward                       |
+-----------------------------------------------------------------+
                                 |
                        Is model.training?
                                 |
           +---------------------+---------------------+
           | Yes                                       | No
           v                                           v
+-----------------------------+               +-------------------+
| Execute normal model pass   |               | Resolve CUDA      |
| using rank-local RNG        |               | devices via       |
+-----------------------------+               | _cuda_rng_devices |
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              | fork_rng(devices) |
                                              | - Set CPU seed    |
                                              | - Set CUDA manual |
                                              |   seeds per device|
                                              +-------------------+
                                                       |
                                                       v
                                              +-------------------+
                                              | Execute model     |
                                              | forward pass      |
                                              +-------------------+
```

Sources:
- `src/model/response.py:98-131`

---

## 4. Code Entity Mapping

The following table associates the conceptual requirements of response prediction and distributional losses with their concrete Python definitions.

| Concept / System Element | Code Entity | File Path and Line Range |
| :--- | :--- | :--- |
| **Bag Padding & Resampling** | `_chunk_control_cell_indices` | `src/model/response.py:12-40` |
| **Mean-Delta MSE Loss** | `mean_delta_mse` | `src/model/response.py:43-55` |
| **Energy Distance Metric** | `energy_distance` | `src/model/response.py:58-76` |
| **STATE Window Resolver** | `_state_window` | `src/model/response.py:78-96` |
| **Isolated Seeded Forward** | `_seeded_model_forward` | `src/model/response.py:114-131` |
| **Batched Bag Predictor** | `predict_bags` | `src/model/response.py:133-173` |
| **Response Training Tests** | `test_batched_predictions_and_losses_match_independent_conditions` | `tests/test_response_training.py:12-30` |

Sources:
- `src/model/response.py:12-173`
- `tests/test_response_training.py:12-30`

---