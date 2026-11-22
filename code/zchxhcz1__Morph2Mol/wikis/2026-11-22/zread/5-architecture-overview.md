

Morph2Mol is a foundation model that predicts single-cell spatial gene expression directly from H&E histology images. Its architecture combines three cooperating mechanisms: a masked-pooling image encoder producing per-cell morphological features, a distance-aware spatial decoder that lets neighboring cells exchange information, and a retrieval branch that borrows measured expression profiles from a reference bank. This page maps the repository onto that design — the component boundaries, the fixed training recipe behind the released 25M checkpoint, and the integrity checks that tie inference artifacts together — so you know exactly where each concern lives before diving into the deep-dive pages.

Sources: [README.md](/README.md#L1-L5), [pyproject.toml](/pyproject.toml#L6-L19)

## System Composition and Package Layout

The repository is organized as three package roots with a strict dependency direction: `train/` and `predict/` are thin entry-point layers, while `morph2mol/` holds all reusable machinery. Inference code imports training-produced artifacts but never training drivers — `predict/model.py` explicitly states it contains "the released 25M architecture, without importing training drivers," and `morph2mol/release.py` is documented as requiring "no training corpus at inference." This one-way flow is what makes the released model portable.

| Package root | Responsibility | Key modules |
| --- | --- | --- |
| `morph2mol/` | Shared model, training, retrieval, and packaging library | `encoder.py`, `se2.py`, `head.py`, `retrieval.py`, `gate.py`, `recipe.py`, `release.py`, `training/` |
| `train/` | CLI entry points for the two fixed training stages and corpus preparation | `train_phase1.py`, `train_phase2.py`, `prepare.py`, `embed_scgpt.py`, `export_release.py` |
| `predict/` | End-to-end inference: segmentation, slide I/O, spatial blocking, orchestration | `segment.py`, `spatial_context.py`, `model.py`, `predict.py` |

Sources: [pyproject.toml](/pyproject.toml#L14-L16), [predict/model.py](/predict/model.py#L1-L3), [morph2mol/release.py](/morph2mol/release.py#L1-L3)

## The Dual-Branch Prediction Architecture

Every predicted cell flows through two parallel branches whose outputs are fused per gene. The **direct branch** learns to synthesize expression from morphology alone (encoder → spatial decoder → negative-binomial head), producing a score for all 18,580 model genes. The **retrieval branch** embeds each cell into a shared Phase-1 contrastive space, finds similar reference communities in a precomputed bank, and returns their averaged measured expression. A per-gene fusion weight β, fitted offline on reference slides, blends them: `X = β·A + (1−β)·R`. The prerequisite to reading the diagram below: solid arrows carry data during inference; dashed arrows are offline preparation steps that happened before the release was packaged.

```mermaid
flowchart LR
    subgraph OFFLINE["Offline preparation (dashed)"]
        SC[("scGPT cell + gene embeddings")] --> S1
        CORPUS[("Paired H&E + spatial transcriptomics corpus")] --> S1
        S1["Stage 1<br/>contrastive alignment"] --> S2["Stage 2<br/>NB expression training"]
        S1 --> BANK["Reference bank<br/>(clustered communities)"]
        S1 --> GATE["Fusion weights beta<br/>(41-point grid fit)"]
    end
    subgraph ONLINE["Inference (solid)"]
        IMG["H&E slide"] --> SEG["Cellpose-SAM<br/>nuclei segmentation"]
        SEG --> CROP["Cell crops +<br/>mask pooling"]
        CROP --> ENC["UNI2-h + LoRA<br/>encoder"]
        ENC -->|"stage2 LoRA h [1536]"| DEC["SE(2) spatial<br/>decoder d=256"]
        ENC -->|"phase1 LoRA H1"| RET["Bank retrieval<br/>K=50, tau=0.03"]
        DEC --> HEAD["NB expression head<br/>log1p(mu) = A"]
        HEAD --> FUSE["Per-gene fusion<br/>beta*A + (1-beta)*R"]
        RET -->|"R"| FUSE
        FUSE --> OUT[("Streaming<br/>.h5ad output")]
    end
    S2 -.checkpoints.-> ENC
    S2 -.weights.-> DEC
    S2 -.weights.-> HEAD
    BANK -.summaries.-> RET
    GATE -.beta.json.-> FUSE
```

Sources: [predict/predict.py](/predict/predict.py#L162-L181), [README.md](/README.md#L86-L96)

## Direct Branch: From Pixels to Expression

The direct branch is a three-stage composition with fixed dimensions throughout. The **image encoder** is UNI2-h, a ViT-Huge/14 with 1536-dimensional embeddings, 24 blocks, SwiGLU MLPs, and 8 register tokens, built through `timm` with strict state-dict loading and every parameter frozen ([encoder.py](/morph2mol/encoder.py#L35-L41)). Training touches it only through **LoRA adapters** — rank-16 low-rank factors with alpha 32 injected into the query/key/value and projection matrices of the last 12 blocks ([encoder.py](/morph2mol/encoder.py#L13-L26)). Per-cell features come from mask-weighted pooling: the encoder's 256 patch tokens (a 16×16 grid after stripping the 9 prefix tokens) are multiplied by a mask-weight map and summed into a single 1536-dim vector, so each cell's feature is an average over exactly its own image evidence ([encoder.py](/morph2mol/encoder.py#L27-L33)).

The **spatial decoder** then lets cells within a neighborhood exchange information. A `GaussianModule` converts pairwise inter-cell distances in microns — clipped at the 137 µm context radius — into 128 Gaussian basis features projected to 64-dim pair embeddings ([se2.py](/morph2mol/se2.py#L12-L14), [_se2_arch.py](/morph2mol/_se2_arch.py#L7-L22)). Six identical `SE2Layer` blocks (3 before and 3 after a mid-point, d_model=256, 8 heads) refine both cell features and pair features; each layer converts pair states into an additive attention bias, so geometry directly shapes who attends to whom ([se2.py](/morph2mol/se2.py#L8-L20), [_se2_arch.py](/morph2mol/_se2_arch.py#L24-L58)).

The **expression head** maps decoder states to gene expression under a negative-binomial observation model. `DirectHead` keeps a learnable gene-row matrix `W` (initialized from scGPT gene embeddings during training), concatenates the 256-dim cell state with two cell-area covariates, passes them through a small MLP, and computes `mu` via softplus; a global area exponent scales rates by cell size ([head.py](/morph2mol/head.py#L27-L44)). Areas themselves are computed by the shoelace formula from nucleus boundary polygons, with degenerate polygons replaced by the slide median rather than zero ([head.py](/morph2mol/head.py#L7-L25)). Training minimizes the analytic negative-binomial log-likelihood ([head.py](/morph2mol/head.py#L45-L46)).

| Component | Input → Output | Fixed dimensions |
| --- | --- | --- |
| UNI2-h encoder | `[B,3,224,224]` crops + masks → `[B,1536]` | 24 blocks, embed 1536, LoRA r=16 on last 12 |
| SE(2) decoder | `[N,1536]` + `pos µm` → `[N,256]` | d_model 256, d_pair 64, 6 layers, 8 heads, 137 µm |
| NB head | `[N,256]` + areas + gene ids → `[N,G]` | MLP 258→1024→D, per-gene θ, area exponent |

Sources: [morph2mol/encoder.py](/morph2mol/encoder.py#L13-L41), [morph2mol/se2.py](/morph2mol/se2.py#L8-L22), [morph2mol/head.py](/morph2mol/head.py#L27-L46)

## Retrieval Branch: Reference Bank and Gated Fusion

The retrieval branch operates on a **reference bank**: one directory per reference object (donor/tissue block), each holding community summaries produced by clustering cell features within that slide. During bank construction, an exhaustive float32 K-means-style assignment (with optional Faiss GPU acceleration, no ANN approximation) groups cells into communities of target size 100, and each community is summarized into an fp16 mean key (1536-dim Phase-1 embedding) plus a float32 mean expression vector ([bank_builder.py](/morph2mol/bank_builder.py#L18-L54)). At load time, `ReferenceBank` verifies the bank's schema, cache fingerprint, completion marker, per-artifact SHA-256 hashes, and panel integrity before pooling every community's expression into a single preallocated device tensor — deliberately bounding host and device memory ([retrieval.py](/morph2mol/retrieval.py#L76-L131)).

Retrieval itself is exact, not approximate: query features are L2-normalized in fp16 and scored by full cosine against every community, top-50 neighbors per bank object are merged per coverage group, and ties at the cutoff threshold receive fractional weights at temperature τ = 0.03 ([retrieval.py](/morph2mol/retrieval.py#L143-L200)). Coverage is explicit — a query gene participates only in groups whose reference objects measured it, and genes covered by no object surface as NaN, which downstream logic treats as "no retrieval" ([retrieval.py](/morph2mol/retrieval.py#L62-L67)).

Fusion weights close the loop. `fit_beta` sweeps a 41-point grid of β per gene, selecting the value that maximizes the Pearson correlation of the blended score against measured expression on reference slides; a grid is chosen over a closed form because the objective is smooth but not concave in β ([gate.py](/morph2mol/gate.py#L6-L8), [gate.py](/morph2mol/gate.py#L48-L75)). At inference, `apply_gate` is deliberately label-blind: it looks up the transferred β per gene and fuses, forcing β = 1 (direct-only) wherever retrieval coverage is missing so NaN can never silently blank a gene column ([gate.py](/morph2mol/gate.py#L77-L105)).

> [!TIP]
> The direct branch and the retrieval branch read the **same encoder** with different LoRA weights. `Morph2Mol.encode(which=...)` hot-swaps adapter tensors between the Phase-1 (stage-1 contrastive) and stage-2 parameterizations; stage-2 checkpoints save only the trainable subset, and frozen adapters are restored from the stage-1 checkpoint ([predict/model.py](/predict/model.py#L49-L63)). This is why both stage-1 and stage-2 artifacts must ship in every release.

Sources: [morph2mol/bank_builder.py](/morph2mol/bank_builder.py#L18-L54), [morph2mol/retrieval.py](/morph2mol/retrieval.py#L76-L200), [morph2mol/gate.py](/morph2mol/gate.py#L6-L105)

## Training Pipeline: A Fixed Two-Stage Recipe

Training follows a strictly ordered, hard-coded recipe. **Stage 1** aligns image and RNA representations with a symmetric InfoNCE loss between the LoRA-adapted image tower and a projection tower over 512-dimensional scGPT cell embeddings; it runs three full-coverage epochs with random right-angle rotations and reflections applied jointly to crops and masks, and uses gradient caching so the 2,048-cell effective batch fits through the frozen ViT in chunks ([training/stage1.py](/morph2mol/training/stage1.py#L1-L110), [docs/training.md](/docs/training.md#L60-L77)). **Stage 2** then trains expression prediction end-to-end: `ExpressionModel` wires encoder → `Linear(1536,256)` → `SpatialTransformer` → `DirectHead`, with the first 20% of steps keeping LoRA frozen, negative-binomial plus auxiliary losses, and gradient checkpointing for the SE(2) stack ([training/stage2.py](/morph2mol/training/stage2.py#L77-L90), [morph2mol/recipe.py](/morph2mol/recipe.py#L23-L39)).

Nothing about these settings is negotiable at the module level. `recipe.py` raises a `TypeError` if any caller tries to override a fixed value, and `check_stage2_settings` refuses to load a checkpoint whose args deviate from the Morph2Mol-25M recipe — architectural constants (128 target cells, 64 context cells, 137 µm radius, seed 1) are shared verbatim by training, calibration, and inference code ([morph2mol/recipe.py](/morph2mol/recipe.py#L4-L40), [predict/spatial_context.py](/predict/spatial_context.py#L71-L81)). Downstream artifacts inherit this discipline: entry-point scripts (`train/_entry.py`) accept only path, hardware, and resume arguments, and even chunk sizes must divide the effective batch exactly ([train/_entry.py](/train/_entry.py#L7-L24)).

| | Stage 1 — alignment | Stage 2 — expression |
| --- | --- | --- |
| Objective | Symmetric InfoNCE (τ = 0.07) | NB log-likelihood + aux/MSE terms |
| Trainable | LoRA adapters + projection towers | Linear in-proj, SE(2) decoder, NB head, LoRA (after 20% freeze) |
| Batch / schedule | 2,048/rank, 3 epochs, warmup 150 | Full-coverage blocks, 3 epochs, grad-ckpt on |
| Output | `stage1_final.pt` (LoRA + towers) | `stage2` checkpoint (identity-linked to stage 1) |
| Feeds into | Retrieval keys (phase1 LoRA), bank, gate fitting | Direct branch A, `log1p(mu)` |

Sources: [morph2mol/training/stage1.py](/morph2mol/training/stage1.py#L17-L110), [morph2mol/training/stage2.py](/morph2morph/training/stage2.py#L77-L90), [morph2mol/recipe.py](/morph2mol/recipe.py#L11-L40), [docs/training.md](/docs/training.md#L60-L78)

## Inference Pipeline: Deterministic, Streaming, Self-Verifying

`predict/predict.py` chains the whole system in one process. It first shells out to `segment.py`, which runs Cellpose-SAM on overlapping windows and keeps each nucleus only in the window whose interior contains its centroid — a rule guaranteeing every cell is segmented exactly once ([predict/predict.py](/predict/predict.py#L66-L78), [predict/segment.py](/predict/segment.py#L1-L18)). Cells are then partitioned into blocks of at most 128 targets plus up to 64 halo context cells within 137 µm; the partition is seeded deterministically from the sample name (`zlib.crc32`), reusing the training-time partition functions so inference blocking matches calibration blocking bit-for-bit ([predict/spatial_context.py](/predict/spatial_context.py#L15-L52)). Within each block, all cells attend to each other in the SE(2) decoder, but `commit_target_rows` scatters only the owned target prefix into the output, with a hard assertion that no cell is written twice ([predict/spatial_context.py](/predict/spatial_context.py#L54-L68)).

Output handling is streaming and atomic: the writer creates chunked, compressed HDF5 datasets up front and fills rows block by block, then deletes the model and GPU cache before the second retrieval pass runs in `query_batch` slices; only when every row is verified written does `os.replace` rename the `.partial` file to the final `.h5ad` ([predict/predict.py](/predict/predict.py#L108-L206)). Provenance travels with the data — the `uns` block records checkpoint SHA-256s, retrieval parameters (K=50, τ=0.03), gate source, the spatial-context protocol, and the scales of each stored layer ([predict/predict.py](/predict/predict.py#L98-L106)).

> [!TIP]
> The spatial blocking protocol is itself a verified artifact. `Release` cross-checks that the fusion-weight metadata's `spatial_context` matches `prediction_protocol()`, and that bank, gate, stage-1, stage-2, and gene-space hashes are mutually consistent — so a gate fitted under a different blocking scheme is structurally impossible to pair with this checkpoint ([morph2mol/release.py](/morph2mol/release.py#L33-L46), [predict/spatial_context.py](/predict/spatial_context.py#L71-L91)).

Sources: [predict/predict.py](/predict/predict.py#L62-L206), [predict/segment.py](/predict/segment.py#L1-L47), [predict/spatial_context.py](/predict/spatial_context.py#L15-L91)

## Integrity Chain

The release directory is the architectural seam between training and inference, and it is treated as a chain of custody rather than a folder of files. `Release` validates the `morph2mol-release-v1` schema, resolves the six artifact paths (UNI2-h weights, stage-1, stage-2, gene space, bank, gate), and re-hashes stage-1/stage-2/gene-space against declared SHA-256s ([morph2mol/release.py](/morph2mol/release.py#L6-L22)). It then checks a seven-way equality chain: the bank must be keyed to the stage-1 checkpoint and gene space, the gate metadata to stage-1, stage-2, gene space, and the bank's fingerprint, and retrieval settings must agree between gate and bank ([morph2mol/release.py](/morph2mol/release.py#L23-L46)). Inside the model itself, the stage-2 checkpoint records the SHA-256 of the stage-1 initializer and a gene-order hash, both re-verified at load ([predict/model.py](/predict/model.py#L22-L29)). The consequence: any mismatched mixture of checkpoints, genes, bank, and weights fails fast at startup instead of producing silently wrong predictions.

Sources: [morph2mol/release.py](/morph2mol/release.py#L6-L46), [predict/model.py](/predict/model.py#L22-L29)

## Where to Go Next

This page treated each subsystem at boundary level. The catalog continues depth-first along the same seams:

- Component internals: [Image Encoder and LoRA Adapters](6-image-encoder-and-lora-adapters), [SE(2) Distance Attention Decoder](7-se-2-distance-attention-decoder), [Negative-Binomial Expression Head](8-negative-binomial-expression-head), [scGPT RNA Embedding Backend](9-scgpt-rna-embedding-backend)
- Retrieval branch in detail: [Reference Bank Construction](10-reference-bank-construction), [Cluster Retrieval Mechanics](11-cluster-retrieval-mechanics), [Gated Fusion Weights](12-gated-fusion-weights)
- Training pipeline: [Training Data Contracts](13-training-data-contracts), [Stage 1 Contrastive Alignment](14-stage-1-contrastive-alignment), [Stage 2 Expression Training](15-stage-2-expression-training)
- Inference engineering: [Nuclei Segmentation with Cellpose-SAM](18-nuclei-segmentation-with-cellpose-sam), [Spatial Context Blocking](19-spatial-context-blocking), [Streaming Output and Fault Tolerance](20-streaming-output-and-fault-tolerance), [Release Integrity Verification](21-release-integrity-verification)

For hands-on usage instead of internals, return to [Quick Start](2-quick-start) and the [Inference CLI Reference](3-inference-cli-reference); [Reading Prediction Outputs](4-reading-prediction-outputs) explains the `.h5ad` layout this page's streaming writer produces.
