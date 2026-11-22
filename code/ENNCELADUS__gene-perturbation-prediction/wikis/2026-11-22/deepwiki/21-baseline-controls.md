# Baseline Controls

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/04-sl-ranking-protocol.md](docs/04-sl-ranking-protocol.md)
- [docs/results/tx1_gmm_ridge_seed0/README.md](docs/results/tx1_gmm_ridge_seed0/README.md)
- [docs/results/tx1_gmm_ridge_seed0/evidence.json](docs/results/tx1_gmm_ridge_seed0/evidence.json)
- [docs/specs/2026-09-07-tx1-gmm-ridge-design.md](docs/specs/2026-09-07-tx1-gmm-ridge-design.md)
- [docs/specs/2026-09-07-tx1-gmm-ridge-plan.md](docs/specs/2026-09-07-tx1-gmm-ridge-plan.md)
- [src/baselines/tx1_gmm.py](src/baselines/tx1_gmm.py)
- [src/experiments/tx1_gmm_ridge.py](src/experiments/tx1_gmm_ridge.py)
- [tests/test_tx1_gmm_ridge.py](tests/test_tx1_gmm_ridge.py)

</details>



## Purpose and Scope
This page documents the baseline control models implemented in the gene-perturbation prediction codebase. These baselines provide rigorous lower and upper performance bounds against which the full STATE backbone and joint neural network architectures are evaluated. The scope covers five primary control strategies: the context-blind **gene-mean** control, the **copy-prior** structural control, the **nearest-line** non-parametric neighbor control, the **context-PCA-ridge** linear baseline (using either HVG or Tx1 features), and the recently implemented **Tx1 GMM ridge baseline** (`Tx1GMMRidge`).

---

## 1. Overview of Baseline Architecture and Strategy
Baselines isolate whether learned deep representations (such as the STATE backbone and ESM2 perturbation adapters) add predictive value over simpler statistical summaries of basal cellular state and gene identity. 

The baseline controls operate on shared prepared inputs (`PreparedInputs`), enforcing strict separation between training cell lines and held-out validation/test splits to prevent data leakage.

```mermaid
graph TD
    sub-graph-Inputs["Input Data Layer"]
    InputData["PreparedInputs object\nsrc/data/prepared.py"] --> BaselineSelector["Baseline Dispatcher\nsrc/experiments/baselines.py"]

    sub-graph-Baselines["Baseline Implementations"]
    BaselineSelector --> GeneMean["Gene-Mean Control\nPan-essential mean"]
    BaselineSelector --> CopyPrior["Copy-Prior Control\nStructural CNV prior"]
    BaselineSelector --> NearestLine["Nearest-Line Control\nCosine/Euclidean distance"]
    BaselineSelector --> ContextPCA["Context-PCA-Ridge\nsrc/baselines/"]
    BaselineSelector --> Tx1GMM["Tx1 GMM Ridge\nsrc/baselines/tx1_gmm.py"]

    Tx1GMM --> GMMFit["GaussianMixture fit\nK=64 diagonal"]
    GMMFit --> FeatureExtract["Context Feature Vector\noccupancy, entropy, nll"]
    FeatureExtract --> RidgeFit["Per-Gene Ridge Regression\nalpha=1.0"]

    Sources: "Sources: src/experiments/tx1_gmm_ridge.py:74-130, src/baselines/tx1_gmm.py:70-167"
```

Sources: `src/experiments/tx1_gmm_ridge.py:74-130`, `src/baselines/tx1_gmm.py:70-167`

---

## 2. Implemented Baseline Classes and Functions

### Tx1 GMM Ridge Baseline (`Tx1GMMRidge`)
The `Tx1GMMRidge` model is defined in `src/baselines/tx1_gmm.py` [src/baselines/tx1_gmm.py:19-199]() and orchestrated via `src/experiments/tx1_gmm_ridge.py` [src/experiments/tx1_gmm_ridge.py:1-175](). It fits a global cell-dimension `StandardScaler` and a diagonal `GaussianMixture` model ($K=64$) on equal numbers of single-cell rows sampled from each labeled training line.

Key functions and classes include:
- `Tx1GMMRidge.fit(inputs, *, n_components)`: Validates splits, balances cell counts across training lines without replacement, fits scalers and the GMM, extracts context feature matrices, and fits per-gene `Ridge(alpha=1.0)` models [src/baselines/tx1_gmm.py:70-167]().
- `Tx1GMMRidge._features(bag, cell_scaler, gmm)`: Computes posterior responsibilities, occupancy probabilities, occupancy entropy, effective component count, mean assignment confidence, and mean negative log-likelihood (`nll`) for a given cell bag [src/baselines/tx1_gmm.py:42-58]().
- `evaluate_model(model, inputs, *, split)`: Computes evaluation metrics (Huber loss, RMSE, residual Pearson/Spearman correlations, and macro per-gene SD ratios) for train or validation sets [src/experiments/tx1_gmm_ridge.py:8-53]().

Sources: `src/baselines/tx1_gmm.py:19-200`, `src/experiments/tx1_gmm_ridge.py:1-176`

---

## 3. Detailed Data Flow and Fitting Mechanics

The baseline execution pipeline for GMM ridge proceeds through strict preparation, feature extraction, and target residualization steps.

```mermaid
graph TD
    Load["load_config & load_inputs\nsrc/experiments/tx1_gmm_ridge.py"] --> SampleCells["Sample balanced cells\nmin cells per bag, seed=0"]
    SampleCells --> ScaleCells["Fit global cell StandardScaler\nsrc/baselines/tx1_gmm.py"]
    ScaleCells --> FitGMM["Fit GaussianMixture K=64\ndiag covariance, reg_covar=1e-4"]
    FitGMM --> ExtractContext["Compute context features\nEntropy, NLL, Conf, Occupancy"]
    ExtractContext --> ScaleContext["Fit context StandardScaler"]
    ScaleContext --> RidgeReg["Fit Ridge alpha=1.0 per gene\ntarget - gene_mean"]
    RidgeReg --> Export["Export model.joblib & metrics\nsrc/experiments/tx1_gmm_ridge.py"]

    Sources: "Sources: src/baselines/tx1_gmm.py:70-167, docs/specs/2026-09-07-tx1-gmm-ridge-design.md:1-37"
```

Sources: `src/baselines/tx1_gmm.py:70-167`, `docs/specs/2026-09-07-tx1-gmm-ridge-design.md:1-37`

---

## 4. Empirical Performance and Evaluation Results

Empirical comparisons between baselines are archived under `docs/results/tx1_gmm_ridge_seed0/` [docs/results/tx1_gmm_ridge_seed0/README.md:1-69](). The validation set comprises 479,084 observed `(ModelID, gene_symbol)` pairs across 27 validation lines with 4,447 variable genes.

| Method | Huber ↓ | RMSE ↓ | Residual Pearson ↑ | Residual Spearman ↑ | Macro per-gene SD ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tx1 GMM64-ridge, $\alpha=1$ | 0.054734 | 0.344255 | 0.078027 | 0.081147 | 1.312647 |
| Tx1 PCA8-ridge, $\alpha=1$ | 0.016281 | 0.181303 | 0.132973 | 0.123286 | 0.272707 |
| HVG PCA8-ridge, $\alpha=1$ | 0.016451 | 0.182288 | 0.080383 | 0.087934 | 0.203722 |
| Gene mean | 0.016317 | 0.181596 | undefined | undefined | 0 |

As documented in `evidence.json` [docs/results/tx1_gmm_ridge_seed0/evidence.json:1-175](), the fixed GMM64 configuration exhibits a large training-to-validation gap (train Huber `0.00879`, validation Huber `0.05473`), indicating poor out-of-sample transfer compared to linear PCA-based ridge baselines (`Tx1 PCA8-ridge`), which achieve superior residual correlation (`0.133`) and lower validation error.

Sources: `docs/results/tx1_gmm_ridge_seed0/README.md:1-70`, `docs/results/tx1_gmm_ridge_seed0/evidence.json:1-176`

---