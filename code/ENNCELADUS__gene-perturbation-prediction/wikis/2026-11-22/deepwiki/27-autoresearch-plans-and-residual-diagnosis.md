# Autoresearch Plans and Residual Diagnosis

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [autoresearch/plan-260908-p1c-interface-isolation/handoff.json](autoresearch/plan-260908-p1c-interface-isolation/handoff.json)
- [autoresearch/reason-260907-residual-diagnosis/diagnose_predictions.py](autoresearch/reason-260907-residual-diagnosis/diagnose_predictions.py)
- [autoresearch/reason-260907-residual-diagnosis/handoff.json](autoresearch/reason-260907-residual-diagnosis/handoff.json)
- [autoresearch/reason-260907-residual-diagnosis/lineage.md](autoresearch/reason-260907-residual-diagnosis/lineage.md)
- [autoresearch/reason-260907-residual-diagnosis/prediction_diagnostics.json](autoresearch/reason-260907-residual-diagnosis/prediction_diagnostics.json)
- [autoresearch/reason-260907-residual-diagnosis/reason-results.tsv](autoresearch/reason-260907-residual-diagnosis/reason-results.tsv)
- [autoresearch/reason-260907-residual-diagnosis/summary.md](autoresearch/reason-260907-residual-diagnosis/summary.md)
- [docs/specs/2026-09-08-p1c-interface-isolation-design.md](docs/specs/2026-09-08-p1c-interface-isolation-design.md)
- [src/experiments/p1c.py](src/experiments/p1c.py)

</details>



### Purpose and Scope
This page details the autoresearch orchestration artifacts, handoff plans, read-only residual diagnostic scripts, and empirical findings for the gene perturbation prediction codebase. It covers the transition from joint training failures to targeted diagnostic programs (such as P1-C interface isolation and basal reconstruction) managed under bounded reasoning protocols in `autoresearch/` and `docs/specs/` [docs/specs/2026-09-08-p1c-interface-isolation-design.md:1-10], [autoresearch/plan-260908-p1c-interface-isolation/handoff.json:1-22].

---

## 1. Autoresearch Handoffs and P1-C Orchestration

The autoresearch framework uses structured JSON handoffs and protocol specifications to govern experimental iterations without manual prompt hallucination. The P1-C interface isolation plan is codified in `autoresearch/plan-260908-p1c-interface-isolation/handoff.json`, specifying experimental archetypes, decision predicates, and hypotheses regarding interface learning rates and basal representation mismatches [autoresearch/plan-260908-p1c-interface-isolation/handoff.json:1-22].

The orchestration entrypoint `src/experiments/p1c.py` coordinates fold preparation, native state loading, and variant validation [src/experiments/p1c.py:1-50]. Key functions include:
- `native_inputs(manifest)`: Validates SHA256 digests of checkpoint files and vocabulary maps before loading the native STATE checkpoint and its gene one-hot mapping [src/experiments/p1c.py:35-50].
- `init_check(backbone, view, bundle, variant, device)`: Verifies zero-effect or zero-adapter initialisation invariants for variants like `V1`, `V2`, `V2-null`, and `V3` [src/experiments/p1c.py:82-130].
- `native_null_reference(...)`: Evaluates the untrained variant against the native model's non-targeting identity on shared validation panels [src/experiments/p1c.py:135-177].

```mermaid
title "P1-C Orchestration and Variant Validation Flow"
graph TD
    Handoff["handoff.json"] --> CLI["src/experiments/p1c.py"]
    CLI --> CheckManifest["native_inputs()"]
    CheckManifest --> LoadState["torch.load(checkpoint)"]
    CLI --> InitCheck["init_check()"]
    InitCheck --> VariantValidation{"Variant Type?"}
    VariantValidation -->|V1/V3| ControlCheck["Compare pred to control_hvg"]
    VariantValidation -->|V2/V2-null| AdapterCheck["Verify final layer zero & batch_index=0"]
    VariantValidation -->|V0| SkipCheck["Not Applicable"]
```
Sources: `autoresearch/plan-260908-p1c-interface-isolation/handoff.json`, `src/experiments/p1c.py`

---

## 2. Residual Prediction Diagnosis Scripts

To diagnose failures in joint GeneEffect prediction, read-only CPU diagnostic scripts analyze existing test predictions without retraining or parameter mutation [autoresearch/reason-260907-residual-diagnosis/diagnose_predictions.py:1-13]. 

The diagnostic script `autoresearch/reason-260907-residual-diagnosis/diagnose_predictions.py` performs the following operations:
- Loads joint model predictions and baseline model predictions from parquet outputs [autoresearch/reason-260907-residual-diagnosis/diagnose_predictions.py:7-10].
- Aligns observation rows across models on `(model_id, gene_symbol)` index keys [autoresearch/reason-260907-residual-diagnosis/diagnose_predictions.py:11-20].
- Computes macro Pearson correlation per gene, prediction standard deviation ratios relative to true targets, Huber loss, common line offsets, and SVD singular value spectra over complete gene sets [autoresearch/reason-260907-residual-diagnosis/diagnose_predictions.py:21-71].

```mermaid
title "Residual Prediction Diagnosis Dataflow"
graph TD
    ParquetJoint["outputs/geneeffect_joint/.../predictions.parquet"] --> LoadJoint["pd.read_parquet()"]
    ParquetBase["outputs/.../baselines/test/predictions.parquet"] --> LoadBase["pd.read_parquet()"]
    LoadJoint --> Align["Align Indices (model_id, gene_symbol)"]
    LoadBase --> Align
    Align --> Metrics["Compute Macro Pearson, Huber, & SVD Spectra"]
    Metrics --> OutputJSON["diagnose_predictions.py JSON Output"]
```
Sources: `autoresearch/reason-260907-residual-diagnosis/diagnose_predictions.py`

---

## 3. Empirical Findings and Statistical Breakdown

The diagnostic evaluation of joint run `joint_seed0_20260906T174818Z_b1024` uncovers severe shrinkage and contextual ordering deficits in the joint head [autoresearch/reason-260907-residual-diagnosis/summary.md:9-15]. The numerical findings compiled in `autoresearch/reason-260907-residual-diagnosis/prediction_diagnostics.json` demonstrate that while the model achieves high overall correlation via gene-mean priors, its condition-specific residual predictions are heavily compressed:

| Metric | Joint Model | Tx1 PCA-Ridge Baseline |
| :--- | :---: | :---: |
| Residual Pearson Macro per Gene | 0.0541 | 0.1216 |
| Median Prediction SD | 0.0129 | 0.0568 |
| Median Prediction SD / Target SD | 0.0575 | 0.2531 |
| Genes with Positive Pearson | 61.48% | 72.90% |
| Variable-Gene Huber Loss | 0.0403 | 0.0405 |
| Prediction Spectrum Top-1 Energy | 58.90% | 42.92% |
| Participation Rank | 2.58 | 3.66 |

Sources: `autoresearch/reason-260907-residual-diagnosis/summary.md`, `autoresearch/reason-260907-residual-diagnosis/prediction_diagnostics.json`

---

##  4. Structural Hypotheses and Architectural Remedies

Bounded reasoning reviews (`autoresearch/reason-260907-residual-diagnosis/summary.md`, `lineage.md`) and specification designs (`docs/specs/2026-09-08-p1c-interface-isolation-design.md`) attribute the residual failure to three ranked structural mechanisms [autoresearch/reason-260907-residual-diagnosis/summary.md:69-119], [docs/specs/2026-09-08-p1c-interface-isolation-design.md:1-32]:

1. **Indirect Gene-Dependent Context Mapping:** The joint 6668→256→256→1 head mixes basal, ESM2, and response vectors through a shared non-linear MLP (`src/model/head.py`), struggling to learn fine-grained per-gene context slopes compared to independent per-gene ridge regressions (`src/baselines/residual.py`) [autoresearch/reason-260907-residual-diagnosis/summary.md:73-93].
2. **Response Objective Dominance:** Joint training optimizes response MSE and energy distance on four anchors alongside GeneEffect loss. Global gradient clipping and scale imbalances can attenuate gradient flow to the GeneEffect head [autoresearch/reason-260907-residual-diagnosis/summary.md:94-115].
3. **Basal Representation Mismatch:** A mismatch exists between the composite input architecture (feeding Tx1 embeddings to a new basal encoder) and native STATE (which adds basal encodings to transformer hidden states before a frozen decoder) [docs/specs/2026-09-08-p1c-interface-isolation-design.md:23-30].

To resolve these issues, P1-C prescribes tiered experimental variants (`V0` through `V3`, `N-native`) to isolate interface learning rates, null-subtracted residuals, and direct basal-reconstruction paths [docs/specs/2026-09-08-p1c-interface-isolation-design.md:33-109], [autoresearch/plan-260908-p1c-interface-isolation/handoff.json:13-20].

Sources: `autoresearch/reason-260907-residual-diagnosis/summary.md`, `autoresearch/reason-260907-residual-diagnosis/lineage.md`, `docs/specs/2026-09-08-p1c-interface-isolation-design.md`, `autoresearch/plan-260908-p1c-interface-isolation/handoff.json`

---