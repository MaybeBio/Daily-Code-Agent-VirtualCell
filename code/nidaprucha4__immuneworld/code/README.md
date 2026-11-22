# A single-cell RNA-sequencing foundation model simulates tumour immune microenvironment evolution to predict pan-cancer immunotherapy response

ImmuneWorld models tumour immune microenvironment evolution as autoregressive cell-state transitions. The release contains continuous pseudotime encoding, trajectory-aware attention, cancer-invariant transfer, perturbation-conditioned rollouts, masked-gene reconstruction, cohort-held-out response modelling, and the evaluation measures used for perturbation, trajectory, deconvolution, and immunotherapy tasks.

## Environment

The reference environment uses Python 3.10.8, PyTorch 1.13.1, CUDA 11.7, scanpy 1.9.1, and scVelo 0.2.5. The pinned stack targets the four-node accelerator configuration used for the main training run.

```bash
conda env create -f environment.yml
conda activate immuneworld
pip install --no-deps -e .
```

The container uses the NVIDIA PyTorch 22.12 image.

```bash
docker build -t immuneworld:1.0 .
```

## Data preparation

Public data locations are listed in `datasets.txt`. The pretraining corpus combines 5.2 million Human Cell Atlas cells, 4.8 million tumour-infiltrating immune cells from CELLxGENE, and 2.4 million GEO Perturb-seq cells. Norman, Adamson, Replogle K562, and Replogle RPE1 are excluded from the pretraining GEO subset. Training-only statistics select 2,000 highly variable genes and normalize every held-out partition without refitting.

Perturbation benchmarks use an 80/10/10 split by perturbation identity. Immunotherapy prediction uses leave-one-cohort-out folds across 14 cohorts and seven cancer types. Patient identifiers must not cross fold boundaries. Trajectory analysis uses the Pancreas, Dentate Gyrus, and Bone Marrow datasets distributed with scVelo.

## Training

The reference run uses 12 transformer layers, eight attention heads, a per-head learned temporal decay, SwiGLU blocks, batch size 128 per accelerator, four A100 80GB accelerators, and an effective batch size of 512. AdamW begins at `1e-4`, warms up for 5,000 steps, and follows cosine decay to `1e-6` for 200 epochs. The run takes approximately 72 hours and 288 GPU-hours. Three independent runs use seeds 42, 123, and 456.

```bash
torchrun --nproc_per_node=4 -m immuneworld.commands.train --settings settings/pretraining.yaml
```

The objective is trajectory mean squared error plus masked-gene binary cross entropy and perturbation outcome error with weights 1.0, 0.5, and 0.3. Cancer-domain discrimination uses gradient reversal with weight 0.1.

## Evaluation

Perturbation reporting uses Pearson correlation for Norman, Adamson, K562, and RPE1. Trajectory reporting uses cross-boundary direction and dynamic time warping averaged over Pancreas, Dentate Gyrus, and Bone Marrow. Deconvolution reports macro F1 over eight immune populations. Immunotherapy evaluation reports leave-one-cohort-out ROC AUC with cohort and cancer summaries.

The reference aggregate results are Norman Pearson correlation `0.914 ± 0.008`, trajectory cross-boundary direction `0.881`, trajectory dynamic time warping `1.387`, deconvolution F1 `0.730 ± 0.019`, and immunotherapy response AUC `0.891 ± 0.014` over three seeds.

## Compute budget

The pretraining corpus contains 12.4 million cells represented over a 2,000-gene vocabulary. A complete run requires four A100 accelerators with 80GB memory each, approximately 72 wall-clock hours, and 288 GPU-hours. Data staging, normalized matrices, fold materialization, optimizer state, and run outputs require at least 3TB of local storage.
