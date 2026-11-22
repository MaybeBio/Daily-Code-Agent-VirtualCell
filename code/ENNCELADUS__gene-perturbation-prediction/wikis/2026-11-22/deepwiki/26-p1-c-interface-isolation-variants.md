# P1-C Interface Isolation Variants

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [autoresearch/plan-260908-p1c-interface-isolation/handoff.json](autoresearch/plan-260908-p1c-interface-isolation/handoff.json)
- [docs/results/p1_response_pathway_diagnostics/evidence/p1c/comparison/analysis.md](docs/results/p1_response_pathway_diagnostics/evidence/p1c/comparison/analysis.md)
- [docs/specs/2026-09-08-p1c-interface-isolation-design.md](docs/specs/2026-09-08-p1c-interface-isolation-design.md)
- [hpc/README.md](hpc/README.md)
- [hpc/p1c_pipeline.sh](hpc/p1c_pipeline.sh)
- [src/eval/p1b.py](src/eval/p1b.py)
- [src/eval/p1c.py](src/eval/p1c.py)
- [src/eval/p1c_tier0.py](src/eval/p1c_tier0.py)
- [src/experiments/p1c.py](src/experiments/p1c.py)
- [src/model/p1c.py](src/model/p1c.py)
- [tests/test_p1c.py](tests/test_p1c.py)

</details>



## Purpose and Scope
This page details the P1-C diagnostic program implementation, focusing on the five interface isolation variants (`V0`-`V3`) and the native state model (`N-native`), model construction (`build_variant`), parameter provenance auditing, leave-one-anchor-out fold preparation, pipeline orchestration waves, and GPU queueing mechanics as specified in `docs/specs/2026-09-08-p1c-interface-isolation-design.md` [docs/specs/2026-09-08-p1c-interface-isolation-design.md:1]().

Sources: [docs/specs/2026-09-08-p1c-interface-isolation-design.md:1-7]()

---

## 1. Interface Isolation Variants (V0-V3 and N-native)
The P1-C architecture defines a spectrum of variants designed to isolate basal-reconstruction failures and interface adaptation mechanisms. The variants and their corresponding input layouts are defined in `src/model/p1c.py`:

```python
VARIANTS = ("V0", "V1", "V2-null", "V2", "V3")

INPUT_LAYOUT = {
    "V0": "tx1",
    "V1": "hvg_tx1",
    "V2-null": "hvg",
    "V2": "hvg_tx1",
    "V3": "hvg_tx1",
    "N-native": "hvg",
}
```
*Sources: [src/model/p1c.py:25-34]()*

Trainable parameter counts vary across variants depending on adapter structure and basal encoder freezing:
- `V0`, `V1`, `V2`, `V3`: `2,533,864` trainable parameters.
- `V2-null`: `1,694,184` trainable parameters (null adapter baseline).
- `N-native`: `0` trainable parameters (released checkpoint evaluation) [src/model/p1c.py:40-47]().

Key architectural components include:
- **`NativeOneHotPerturbations`**: Manages fixed one-hot perturbation vectors keyed by gene name without case-folding, supporting native model vocabularies like `"non-targeting"` [src/model/p1c.py:52-58]().
- **`SplitBasalEncoder`**: Combines a frozen native basal encoder over the leading `hvg_width` columns with a zero-initialized linear map over remaining Tx1 columns [src/model/p1c.py:95-102]().
- **`NullSubtractedBackbone`**: Evaluates `control_hvg + (perturbed - null)` to enforce zero-effect initializations for `V1` and `V3` [src/model/p1c.py:118-125]().

Sources: [src/model/p1c.py:25-138]()

---

## 2. Provenance Auditing and `build_variant`
Variants are instantiated from a P1-B checkpoint template (`B-init.pt`) using `build_variant`. This function constructs model components while auditing parameter provenance and applying configuration-specific layers.

```mermaid
title "Variant Construction and Provenance Flow"

graph TD
    Template["B-init.pt Template"] --> Load["_load_template()"]
    Load --> Build["build_variant()"]
    Build --> V0["Variant V0 (Tx1 Backbone)"]
    Build --> V1["Variant V1 (Null-Subtracted)"]
    Build --> V2Null["Variant V2-null (HVG Basal)"]
    Build --> V2["Variant V2 (Split Basal Encoder)"]
    Build --> V3["Variant V3 (Full Adapter + Split)"]
    Build --> Native["N-native (Native Checkpoint)"]
    
    subgraph "Validation"
        V1 --> Check["init_check()"]
        V3 --> Check
        V2Null --> Check2["adapter_final_layer_zero"]
        V2 --> Check2
    end
```
*Sources: [src/model/p1c.py:1](), [src/experiments/p1c.py:75-130]*

Initializations are verified via `init_check`, confirming zero-effect output deviations or zero-initialized adapter layers before training begins [src/experiments/p1c.py:82-130]().

Sources: [src/model/p1c.py:1](), [src/experiments/p1c.py:75-130]()

---

## 3. Fold Preparation and Leave-One-Anchor-Out (LOAO)
P1-C evaluates models across four anchors: `jurkat`, `k562`, `hepg2`, and `hct116` [hpc/p1c_pipeline.sh:32](). Fold preparation restricts bundles and splits to training on three anchors while holding out the fourth as an external evaluation context.

```python
def restrict_to_native(bundle, native_genes):
    """Shallow-copy bundle restricted to conditions the native model can predict."""
    bundle = dict(bundle)
    bundle["splits"] = {
        role: [i for i in indices if bundle["keys"][i][1] in native_genes]
        for role, indices in bundle["splits"].items()
    }
    return bundle
```
*Sources: [src/experiments/p1c.py:60-72]()*

Fold metadata records are verified against manifest checksums to ensure dataset immutability [src/experiments/p1c.py:53-57]().

Sources: [src/experiments/p1c.py:1-73]()

---

## 4. Pipeline Waves and GPU Queueing
Execution of the P1-C experiment grid is automated via `hpc/p1c_pipeline.sh`. The pipeline proceeds in distinct waves: fold preparation (CPU), Tier 0 analysis, native evaluation, Tier 4 head training, and variant training across all folds [hpc/p1c_pipeline.sh:1-8]().

```mermaid
title "Pipeline Waves and GPU Queueing Architecture"

graph TD
    Start["hpc/p1c_pipeline.sh"] --> Wave1["Fold Preparation (CPU)"]
    Wave1 --> Wave2["Tier 0 + Native Eval + Tier 4 Heads"]
    Wave2 --> Wave3["Variant Waves (V0, V1, V2-null, V2)"]
    Wave3 --> Queue{"GPU Queue Manager"}
    
    Queue -->|Job Slot Free| JobBody["job_body()"]
    JobBody --> RunSh["run_sh()"]
    RunSh --> Train["p1c train / evaluate"]
    
    subgraph "Signal & Process Management"
        Signal["signal_job()"] --> Term["TERM / KILL Trap"]
        Term --> Tracked["tracked_pids / tracked_kids"]
    end
```
*Sources: [hpc/p1c_pipeline.sh:1-201]*

The queueing mechanism limits concurrent jobs to match the visible GPU count (`GPUS="0 1"`), polling every `POLL_SECONDS` to dispatch queued tasks and handle interruption signals cleanly through `on_signal` [hpc/p1c_pipeline.sh:14-139]().

Sources: [hpc/p1c_pipeline.sh:1-465]()

---

## 5. Tier 0 Analysis and Evaluation Protocols
Tier 0 analyses existing P1-B exports without requiring GPU inference or weight refitting [src/eval/p1c_tier0.py:1-2](). Key evaluation scripts compute bias decomposition and matched coverage:

```python
def bias_decomposition(predicted, observed):
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(
        observed, dtype=np.float64
    )
    shared = error.mean(axis=0)
    total = float(np.mean(error**2))
    shared_bias = float(np.mean(shared**2))
    specific = float(np.mean((error - shared) ** 2))
    return {
        "total_mse": total,
        "shared_bias": shared_bias,
        "condition_specific": specific,
        "shared_bias_fraction": shared_bias / total if total > 0 else float("nan"),
        "n": int(error.shape[0]),
    }
```
*Sources: [src/eval/p1c_tier0.py:12-28]()*

Fold comparisons pool metrics across folds with equal fold weight using gene resampling replicates (`equal_fold_ratio` and `equal_fold_difference`) to prevent dominant anchor condition counts from skewing pooled confidence intervals [src/eval/p1c.py:27-177]().

Sources: [src/eval/p1c_tier0.py:1-309](), [src/eval/p1c.py:1-179]()

---