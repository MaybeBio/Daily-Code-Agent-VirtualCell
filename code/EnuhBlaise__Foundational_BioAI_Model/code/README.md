# Single-cell transcriptomics foundation model — research repository

A modular repository for building and, more importantly, **honestly
evaluating** representation models for single-cell RNA-seq.

Built on the Stem–Backbone–Head architecture from
`frontier_bio_repo_architecture.md`, specialized to single-cell
transcriptomics.

---

## What makes this different from a training script

The architecture here is conventional on purpose. Transformer design is not
what limits single-cell foundation models. Three independent benchmarks found
that these models often fail to beat PCA, logistic regression or linear
baselines on cell-type annotation, integration and perturbation prediction
(see `docs/BIBLIOGRAPHY.md`). What limits progress is **confounding,
leakage and evaluation**.

So the evaluation layer is a first-class module rather than an afterthought,
and it is built to be hard to fool:

| | |
|---|---|
| **Baselines are mandatory** | PCA, scaled PCA, raw HVG and a random projection run on every evaluation. The config cannot disable them. |
| **Honest splits are the default** | Donor-held-out splitting; a leakage audit runs before training and **raises** on donor overlap. |
| **Random splits are a control, not a result** | `RandomSplitter` warns, and the report labels the split regime so its numbers can't be mistaken for held-out ones. |
| **Fit-on-train is enforced** | HVGs, gene vocabulary, tokenizer statistics and label encoders are all fitted after splitting, on train only. `RankTokenizer` refuses to be re-fitted. |
| **Both metric families, always** | Biological conservation and batch correction are each trivially gameable alone — noise scores ~0.97 on batch mixing. |
| **The report states a verdict** | Plain language on whether the model beat its best control, and it won't call a win a win when the intervals overlap. |
| **Simulated data is branded** | An unsuppressable banner; synthetic numbers can never be quietly presented as biology. |

The rationale for each, with sources, is in **`docs/CONSTRAINTS.md`**.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10. CPU is fine for the smoke test and ablations.

## Run

```bash
python run.py +experiment=smoke_test        # ~10s, end-to-end
python run.py --list                        # every registered component
pytest tests/ -q                            # 96 tests
```

Then the real experiments:

```bash
# reference synthetic run
python run.py +experiment=pretrain_synthetic

# how much would a random split have inflated the result?
python run.py +experiment=leakage_ablation

# architectural control: no attention
python run.py +experiment=pretrain_synthetic model=mlp_control

# tokenization controls (pair the stem with the tokenizer)
python run.py +experiment=pretrain_synthetic model=continuous data/tokenizer=continuous
python run.py +experiment=pretrain_synthetic model=continuous data/tokenizer=rank

# real data (needs `pip install cellxgene-census` + network)
python run.py +experiment=pretrain_census
```

Any field is overridable from the CLI:

```bash
python run.py +experiment=pretrain_synthetic \
  data/splitter=study_held_out \
  model.d_model=512 \
  model.backbone.n_layers=8 \
  train.epochs=50
```

`model`, `data`, `data/tokenizer`, `data/splitter`, `execution` and `eval`
are Hydra config groups, so they are swapped with `group=name` rather than by
overriding individual keys. Model width is a single `model.d_model` that
propagates to the stem, backbone and every head — repeating it per component
is a reliable way to produce a silent dimension mismatch.

Enabling an off-by-default head:

```bash
python run.py +experiment=pretrain_synthetic model.heads.cell_type.enabled=true
```

## What a run prints

```
model                        probeF1      knnF1        ari        nmi      iLISI    PCRbatch        BIO      BATCH
--------------------------------------------------------------------------------------------------------------
foundation_model               0.474      0.094     -0.003      0.029     0.972*      0.008*      0.165     0.964*
baseline:pca                   0.858*     0.687*      0.214      0.317     0.535       0.243       0.350     0.591
baseline:hvg                   0.768      0.623      0.220*     0.349*     0.661       0.134       0.360*    0.672
baseline:random_projection     0.581      0.420       0.082      0.159     0.714       0.110       0.247     0.717

split regime   : donor_held_out (held-out 'donor_id')
                 *** SYNTHETIC DATA -- these numbers characterize the simulator, not biology ***
cells          : train=1506 test=502 over 2 held-out group(s)
                 WARNING: fewer than 3 held-out groups; the cell-level CIs understate real uncertainty badly.

VERDICT: model does NOT beat baseline:pca on probe_macro_f1: 0.474 vs 0.858 (-0.383).
         This is the common outcome in published benchmarks at this scale.
```

That is a real output from a deliberately under-trained model. A harness that
can only produce good news is not a research instrument — the point is that
this one tells you when you have not beaten PCA.

Note the second row of the model's own result: it wins every *batch* metric
and loses every *biology* metric. That is the noise regime, and it is exactly
why both families are always reported.

---

## Data

**Real data is the default intent**; `configs/data/census.yaml` points at
CELLxGENE Census, which supplies the harmonized `donor_id` / `assay` /
`tissue` / `cell_type` metadata the honest splitters need. Local `.h5ad` and
10x matrix directories are also supported.

```bash
pip install cellxgene-census
python run.py +experiment=pretrain_census
# or a file you downloaded from cellxgene.cziscience.com/datasets:
python run.py +experiment=pretrain_synthetic data=h5ad \
  data.source.path=/path/to/file.h5ad \
  'data.source.obs_map={donor: donor_id, sample: batch_id}'
```

> **The repo was built in a sandbox without access to the data hosts**
> (`cellxgene-census-public-us-west-2.s3...`, `cf.10xgenomics.com`,
> `ftp.ncbi.nlm.nih.gov` were all policy-blocked). The real-data loaders are
> therefore written and unit-tested against their interfaces but have not
> been run against the live services. Expect to adjust `obs_map` for a given
> dataset's column names on first use.

**Synthetic data lives in a separate module** (`src/data/synthetic/`) and
reproduces the statistics that make real data hard, with every effect
individually switchable:

- negative-binomial counts with mean-dependent overdispersion (`phi ≈ a + b/mu`)
- ~90% sparsity and ~10% of genes detected per cell, matching 10x
- log-normal library sizes spanning an order of magnitude
- per-gene batch effects (not removable by a global scale factor)
- donors nested inside batches, so donor-held-out splitting tests something
- batch-dependent cell-type composition
- ambient RNA contamination and doublets
- hierarchical cell types, so metrics can distinguish coarse lineages from close subtypes

Switchability is what makes it useful: generate with `batch_effect=0` and a
batch metric must read ~0, or the metric is measuring something else. The
test suite asserts this.

---

## Layout

```
configs/            Hydra composition: experiment / model / data / execution / eval
src/core/           Registry + the Stem/Backbone/Head contract (CellBatch)
src/data/
  preprocessing.py  normalization, batch-aware HVG, frozen gene vocabulary
  splits.py         donor/study/tissue held-out splitters + leakage audit
  tokenizers/       rank (Geneformer-style) | binned (scGPT-style) | continuous (control)
  sources/          census | h5ad | 10x mtx | synthetic
  synthetic/        standalone simulator with known ground truth
src/models/
  stems/            expression (gene+bin) | continuous
  backbones/        set_transformer (no positional encoding) | mean_pool_mlp (control)
  heads/            masked_value | cell_type | zinb | batch_adversarial
src/eval/
  baselines.py      the mandatory controls
  metrics.py        ARI, NMI, ASW, iLISI, kBET, graph connectivity, PCR, library-size R²
  probes.py         linear + kNN probes with bootstrap CIs
  report.py         side-by-side comparison and verdict
src/runners/        pretrain | evaluate
docs/
  CONSTRAINTS.md    the ten recurring failure modes and how each is handled
  BIBLIOGRAPHY.md   verified sources for every idea implemented
tests/              96 tests
```

## Exploring the data methods

`Notebooks/` holds four executed notebooks that call the repo's own curation
and tokenization code and chart what each step does to the data:

| notebook | covers |
|---|---|
| `01_raw_count_characteristics` | depth, mean–variance, zeros vs the negative binomial, confounders, simulator knobs |
| `02_curation_qc_normalization_hvg` | QC thresholds, raw-count guard, CP10K and `log1p`, gene selection flavors, vocabulary alignment |
| `03_splitting_and_leakage` | splitters, leakage audit, what a random split buys, fit-on-train, donor-level uncertainty |
| `04_tokenization` | the three tokenizers, bin occupancy and chance level, truncation, depth robustness, sweeps, masking |

```bash
pip install jupyterlab matplotlib
jupyter lab Notebooks/
```

Each notebook has a `DATA_PATH` switch at the top: `None` uses the simulator,
a path to an `.h5ad` runs the same analysis on real data.

## Extending it

Write a class, decorate it, reference it by name in YAML. No training-loop
edits:

```python
from src.core import BaseBackbone, register_backbone

@register_backbone("mamba")
class MambaBackbone(BaseBackbone):
    def __init__(self, d_model=256, n_layers=8, d_state=16):
        super().__init__(d_model)
        ...
    def forward(self, x, pad_mask=None):
        return ...
```

```yaml
backbone: {name: mamba, d_model: 512, n_layers: 12}
```

`python run.py --list` enumerates everything registered. Unknown names fail
loudly with suggestions; a component registered against the wrong base class
is rejected at import.

## Known gaps

- `configs/execution/fsdp_cluster.yaml` is a placeholder. The training loop
  marks the two call sites a Fabric/FSDP wrapper replaces, but multi-node has
  not been run.
- The real-data sources are unvalidated against live services (see above).
- `graph_connectivity` saturates at 1.0 on small well-separated test sets;
  it discriminates only at realistic scale.
- No trajectory or cell-cycle conservation metrics yet; scIB has both.
