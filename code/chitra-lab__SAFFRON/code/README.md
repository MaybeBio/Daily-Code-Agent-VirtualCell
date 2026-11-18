# SAFFRON

SAFFRON is a mechanistic interpretability framework for evaluating whether spatial
transcriptomics foundation model (SFM) embeddings capture known sources of spatial
variation in gene expression, using a Matryoshka sparse autoencoder (SAE) to
decompose embeddings into sparse, human-interpretable features.

Given:
1. a set of embeddings `Z` (from an SFM, a single-cell FM, or even gene expression), and
2. a known measure of spatial variation `τ` for the same cells (e.g. a 1-D spatial axis,
   a disease-signature score, distance to a pathological feature),

SAFFRON trains a sparse autoencoder on `Z` and evaluates which of its learned features track `τ`.

📄 **Paper:** [Evaluating the ability of spatial transcriptomics foundation models to learn multi-scale spatial variation](https://doi.org/10.64898/2026.08.01.742217) (*bioRxiv* preprint, 2026)

<p align="center">
  <img src="https://github.com/chitra-lab/SAFFRON/raw/HEAD/docs/_static/img/saffron_overview.png" alt="SAFFRON overview" width="900">
</p>

## 🚀 Quick Start

```bash
git clone https://github.com/chitra-lab/SAFFRON && cd SAFFRON
pip install -e .
```

## ▶️ Running SAFFRON

```python
import anndata as ad
from saffron import SAFFRON

adata = ad.read_h5ad("my_data.h5ad")
Z = adata.obsm["X_embedding"]        # (N, D) embeddings from an SFM, a single-cell FM, or gene expression
tau = adata.obs["isodepth"].values   # (N,) known spatial variation, e.g. a 1-D spatial axis

model = SAFFRON()                    # Matryoshka SAE with BatchTopK sparsity (M = 4·D, K ≈ 6% of M by default)
model.fit(Z)                         # train the SAE once ...
result = model.evaluate(tau)         # ... then evaluate its features against any spatial measure

print(result.correlation.best_feature, result.correlation.best_rho)  # SAE feature most correlated with tau
if result.omp_selection is not None:                                  # OMP runs when |rho| >= 0.3
    print(result.omp_selection.selected_features, result.omp_selection.r2)  # minimal reconstructing feature set
```

The tutorials below show the full analyses from the paper.

## Getting started

Check out our [readthedocs](https://saffron-sae.readthedocs.io/en/latest/index.html), which
includes tutorials for two analyses:
- `notebooks/global_spatial_variation_tutorial.ipynb` — **Spatial gradients in the colorectal
  tumor-stroma boundary.** Novae embeddings on 10x Visium data, evaluated against a **global**
  1-D tumor-to-stroma gradient, including a check of the SAE features against raw gene expression.
- `notebooks/local_spatial_variation_tutorial.ipynb` — **Local microenvironment variation around
  Aβ plaques (MERFISH).** Compares a gene-expression SAE against a SFM's SAE on **local**
  microenvironment measures in an Alzheimer's disease brain dataset.

## 📊 Benchmark

The paper's 8 benchmark tasks (6 global spatial gradients, 2 Alzheimer's disease plaque microenvironments), with their
spatial measures `τ` and the baselines (PCA and an SAE on gene expression directly), are on Hugging Face at
[ChitraLab/saffron-benchmark](https://huggingface.co/datasets/ChitraLab/saffron-benchmark). To evaluate a new model:

```python
from saffron import SAFFRON, benchmark

adata = benchmark.load("colorectal")      # one of benchmark.TASKS
Z = my_model.embed(adata)                 # your model's (cells x D) embeddings
H = SAFFRON().fit(Z).sae_result_.Z        # SAFFRON SAE features
benchmark.score(adata, H)                 # same metric as the paper
benchmark.score(adata, adata.obsm["X_pca"])   # PCA baseline (also obsm["gene_sae_features"])
```

## 📦 Dependencies

`pip install -e .` installs everything needed for the core SAFFRON pipeline and
both notebooks (numpy, scipy, scikit-learn, torch, anndata, matplotlib, jupyter,
ipykernel).
