# Glossary

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.claude/skills/research-vault/SKILL.md](.claude/skills/research-vault/SKILL.md)
- [.gitignore](.gitignore)
- [AGENTS.md](AGENTS.md)
- [CLAUDE.md](CLAUDE.md)
- [README.md](README.md)
- [docs/01-blueprint.md](docs/01-blueprint.md)
- [docs/02-literature-review.md](docs/02-literature-review.md)
- [docs/data/cell-line-geneeffect-226.md](docs/data/cell-line-geneeffect-226.md)
- [docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md](docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md)
- [hpc/README.md](hpc/README.md)
- [hpc/p1c_pipeline.sh](hpc/p1c_pipeline.sh)
- [src/data/geneeffect.py](src/data/geneeffect.py)
- [src/data/q_sc.py](src/data/q_sc.py)
- [src/data/splits.py](src/data/splits.py)
- [src/data/tx1_cache.py](src/data/tx1_cache.py)
- [src/experiments/prepare.py](src/experiments/prepare.py)
- [src/model/p1c.py](src/model/p1c.py)
- [tests/test_joint_integration.py](tests/test_joint_integration.py)
- [tests/test_p1c.py](tests/test_p1c.py)
- [tests/test_tx1_embed_cache.py](tests/test_tx1_embed_cache.py)

</details>



This wiki page defines 17 core technical terms, data structures, model variants, and diagnostic entities used across the codebase. Each entry details its role, implementation, and code pointers.

---

## 1. Foundation & Backbone Terms

### GeneEffect
* **Definition:** DepMap single-gene dependency score representing cell fitness effects following knockout or knockdown.
* **Implementation & Role:** Managed via ingestion modules [src/data/geneeffect.py]() and evaluated against residuals `r_{g,c} = y_{g,c} - \mu_g` [docs/01-blueprint.md:55-75](). Targets are modeled with Huber loss [docs/01-blueprint.md:78-85]().
* **Sources:** [src/data/geneeffect.py](), [docs/01-blueprint.md:55-75]()

### Tx1
* **Definition:** Tahoe-100M single-cell foundation model pre-trained encoder [docs/02-literature-review.md:43-50]().
* **Implementation & Role:** Provides frozen 2560-dimensional basal cell embeddings (`EMBEDDING_WIDTH = 2560`) cached via [src/data/tx1_cache.py]() and authenticated by `authenticate_tx1_registration` [tests/test_tx1_embed_cache.py:86-102]().
* **Sources:** [src/data/tx1_cache.py](), [tests/test_tx1_embed_cache.py:86-102]()

### STATE
* **Definition:** Arc Institute transformer-based virtual cell perturbation model [docs/02-literature-review.md:43-50]().
* **Implementation & Role:** Wrapped by `StateForwardAdapter` and `ForwardOnlyStateModel` [src/model/state.py]() to predict per-gene expression shifts under specific perturbations.
* **Sources:** [src/model/state.py](), [docs/02-literature-review.md:43-50]()

### ESM2
* **Definition:** Evolutionary Scale Modeling protein language model [CLAUDE.md:52-65]().
* **Implementation & Role:** Precomputes protein sequences into 1280-dimensional embeddings (`Esm2EmbeddingTable`), mapped via `Esm2PerturbationAdapter` [src/model/perturbation.py]() into network input spaces.
* **Sources:** [src/model/perturbation.py](), [src/data/embeddings.py]()

### HVG
* **Definition:** Highly Variable Genes panel [docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md:61-73]().
* **Implementation & Role:** A curated subset of genes (typically 2000 dimensions, `HVG_WIDTH`) forming the output space $B_c$ and transformer target dimensions [src/data/p1c.py]().
* **Sources:** [src/data/p1c.py](), [docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md:61-73]()

### ModelID
* **Definition:** DepMap canonical cell line identifier (e.g., `ACH-000551`) [AGENTS.md:20-25]().
* **Implementation & Role:** Enforced as the primary join key across datasets and metadata parquets, preventing string mismatch errors (`K562` vs `K-562`) [AGENTS.md:20-25]().
* **Sources:** [AGENTS.md:20-25](), [tests/test_p1c.py:157-168]()

```mermaid
graph TD
    "Tx1" --> [src/data/tx1_cache.py]
    "STATE" --> [src/model/state.py]
    "ESM2" --> [src/model/perturbation.py]
    "HVG" --> [src/data/p1c.py]
    "GeneEffect" --> [src/data/geneeffect.py]
    "ModelID" --> [AGENTS.md]
```
Sources: [src/data/tx1_cache.py](), [src/model/state.py](), [src/model/perturbation.py](), [src/data/p1c.py](), [src/data/geneeffect.py](), [AGENTS.md]()

---

## 2. Statistical & Data Structures

### q_sc
* **Definition:** Single-cell basal baseline statistical summary [docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md:82-88]().
* **Implementation & Role:** Encodes mean expression, fraction expressing, and expression variance computed from basal single cells for gene $g$ in context $c$ [src/data/q_sc.py]().
* **Sources:** [src/data/q_sc.py](), [docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md:82-88]()

### mu_bar
* **Definition:** Fold-independent global mean baseline [CLAUDE.md:73-90]().
* **Implementation & Role:** Used as the prediction centering target to prevent data leakage, contrasting with fold-fit training means $\hat{\mu}_g^{(-c)}$ [CLAUDE.md:73-90]().
* **Sources:** [CLAUDE.md:73-90](), [docs/01-blueprint.md:55-65]()

### Energy Distance
* **Definition:** Distributional divergence metric between predicted and observed cell response bags [docs/01-blueprint.md:78-85]().
* **Implementation & Role:** Combines with mean-delta MSE to form the response loss $L_{response} = L_{mean\text{-}delta\ MSE} + L_{energy\ distance}$ [docs/01-blueprint.md:78-85]().
* **Sources:** [docs/01-blueprint.md:78-85]()

### Response Bag
* **Definition:** Sampled collection of single-cell expression vectors representing a cell line's state [docs/01-blueprint.md:65-75]().
* **Implementation & Role:** Processed without pairing individual control cells to perturbation cells, mapped through `predict_bags` [CLAUDE.md:52-65]().
* **Sources:** [CLAUDE.md:52-65](), [docs/01-blueprint.md:65-75]()

### prepared_inputs.json
* **Definition:** Immutable preparation manifest file [hpc/README.md:18-27]().
* **Implementation & Role:** Stores fixed metadata, tensor hashes, and input path pointers generated during `src/experiments/prepare.py` [CLAUDE.md:52-65](). Training pipelines open this manifest read-only [CLAUDE.md:52-65]().
* **Sources:** [CLAUDE.md:52-65](), [hpc/README.md:18-27]()

---

## 3. Joint Training & Execution Concepts

### Replay Step
* **Definition:** Interleaved optimization cycle for auxiliary tasks [CLAUDE.md:52-65]().
* **Implementation & Role:** Executed every fourth update ($t \bmod 4 = 0$), minimizing response reconstruction loss across anchor lines while main GeneEffect Huber steps proceed every update [docs/01-blueprint.md:78-85]().
* **Sources:** [CLAUDE.md:52-65](), [docs/01-blueprint.md:78-85]()

### Anchors
* **Definition:** Designated reference cell lines (K562, Jurkat, HepG2, HCT116) used for response supervision [docs/01-blueprint.md:43-50]().
* **Implementation & Role:** Sampled in balanced batches during response replay steps to anchor multi-context generalizability [CLAUDE.md:52-65]().
* **Sources:** [CLAUDE.md:52-65](), [docs/01-blueprint.md:43-50]()

---

## 4. Diagnostic and Interface Isolation Terms (P1-A / P1-C)

### Tier 0
* **Definition:** Initial diagnostic evaluation suite [tests/test_p1c.py:157-193]().
* **Implementation & Role:** Computes bias decomposition, shared vs. condition-specific variance, and matched coverage metrics on cached model outputs [src/eval/p1c_tier0.py]().
* **Sources:** [src/eval/p1c_tier0.py](), [tests/test_p1c.py:157-193]()

### Readout Arms (A0–A3)
* **Definition:** Feature extraction ablation configurations [hpc/README.md:67-80]().
* **Implementation & Role:** Defined in P1-A diagnostics to isolate the contribution of specific head blocks (delta projections, response dispersion, embeddings) [hpc/README.md:67-80]().
* **Sources:** [hpc/README.md:67-80]()

### Folds
* **Definition:** Cross-validation split groups for P1-C interface isolation experiments [hpc/p1c_pipeline.sh:30-33]().
* **Implementation & Role:** Grouped by cell lines (`jurkat`, `k562`, `hepg2`, `hct116`) to evaluate transfer performance across variant backbones [hpc/p1c_pipeline.sh:30-33]().
* **Sources:** [hpc/p1c_pipeline.sh:30-33]()

### Variants V0–V3 & N-native
* **Definition:** Interface isolation backbone architectures [src/model/p1c.py:25-47]().
* **Implementation & Role:** Variants (`V0`, `V1`, `V2-null`, `V2`, `V3`) and `N-native` test specific input representations, null subtraction mechanisms, and parameter locking protocols [src/model/p1c.py:25-47]().
* **Sources:** [src/model/p1c.py:25-47](), [hpc/p1c_pipeline.sh:30-35]()

```mermaid
graph TD
    "Tier0" --> [src/eval/p1c_tier0.py]
    "ReadoutArmsA0A3" --> [hpc/README.md]
    "Folds" --> [hpc/p1c_pipeline.sh]
    "VariantsV0V3" --> [src/model/p1c.py]
    "Nnative" --> [src/model/p1c.py]
    "ReplayStep" --> [src/training/sampling.py]
    "Anchors" --> [src/training/sampling.py]
```
Sources: [src/eval/p1c_tier0.py](), [hpc/README.md](), [hpc/p1c_pipeline.sh](), [src/model/p1c.py](), [src/training/sampling.py]()