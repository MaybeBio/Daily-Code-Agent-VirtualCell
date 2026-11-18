# VCDesign: portable code package

This package contains the source code, frozen configuration, synthetic smoke example, and protocol tests for **VCDesign: Finite-Budget Intervention Design for Virtual Cells**.

VCDesign ranks a variable set of feasible interventions toward a target under a finite experimental budget. Its evaluation scores the selected top-$B$ prefix using independently measured held-out outcomes; in the reported CRISPRi benchmark, utility measures the direction of transcriptomic response. VCDesign-CED combines direct candidate scoring with predicted candidate effects when the candidate's own response is absent from its effect atlas.

The manuscript reports a pooled paired BU@20 gain of `+0.0492` over the same base scorer on held-context response-masked pools and `1.11` more top-5% candidates among 20 selections on average. When candidate responses are already measured in other contexts, direct retrieval is stronger (`0.675` versus `0.508` pooled BU@20). Response masking does not mean the base scorer has never seen every candidate identity: the fixed base scorer was trained on 68% of the masked identities. These results concern retrospective directional prioritization, not prospective biological endpoints.

<p align="center"><img src="https://github.com/Boom5426/VCDesign-CED/raw/HEAD/assets/vcdesign_overview.png" alt="VCDesign finite-budget ranking, outcome-based evaluation, and Candidate Effect Distillation" width="960"></p>

## Scope of this package

Included:

- the `gene_open_inverse` Python source tree;
- the frozen paper-run configuration;
- a synthetic, data-free VCDesign-CED example;
- protocol tests and processed-data integrity verification;
- exact Python package versions for a CPU verification environment; and
- the local framework overview image.

Not included:

- Git metadata or commit history;
- the manuscript PDF, author information, or public author-account links;
- raw public H5AD matrices;
- large processed inputs, checkpoints, or cached run records; or
- outputs from local executions.

The manuscript provides the experimental definitions and reported results. Large processed inputs can be supplied separately when needed for reproduction.

## Fast verification

Python 3.11 is required. From the archive root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
python examples/ced_demo.py
python -m pytest -q
```

The expected test result for this release is `169 passed`. The demo checks response-basis construction, ridge effect prediction, missing-knowledge handling, and score fusion. These are software checks, not biological results.

## Reproducing the primary training run

Place the separately supplied processed-data artifact at the archive root with this layout:

```text
DATA_MANIFEST.json
inputs/
checkpoints/
records/
```

Verify it before use:

```bash
python tools/verify_processed_data.py --root .
```

Then train with an explicit, new output directory:

```bash
python -m gene_open_inverse.final_clean_model_v1.train \
  --config configs/paper_run_v1.json \
  --output outputs/final_clean_train \
  --device cuda
```

The training command refuses to overwrite an existing output directory. The configuration resolves assets relative to the archive root and records the selected epoch-8 reference checkpoint.

## Main entry points

| Task | Entry point |
| :--- | :--- |
| Synthetic end-to-end check | `python examples/ced_demo.py` |
| Protocol tests | `python -m pytest -q` |
| Verify processed data | `python tools/verify_processed_data.py --root .` |
| Train the final base scorer | `python -m gene_open_inverse.final_clean_model_v1.train` |
| Run Candidate Effect Distillation | `python -m gene_open_inverse.candidate_effect_distillation_v1.run` |
| Build four-context inputs | `python -m gene_open_inverse.four_context_v1.build` |

## Corrections made after external review

This release differs from the version submitted with the paper in three respects, followed by smaller changes listed below.

- **Ridge penalty selection.** In `four_context_v1.predictors.fit`, cross-validation folds are now cut over gene identities and the response basis is refit inside each fold (`gene_open_inverse/revision_v1/grouped_fit.py`). This is the default, and `selection="frozen"` reproduces the submitted selection exactly. When both select the same penalty, the fitted predictors are identical, and so are the scores and BU@20 values computed from them. They select the same penalty, 1e4, on every pool checked: the three held-out contexts of the main analysis (the full fit and five masking folds each, with raw and unit targets) and the K562 pool of the vocabulary-size curve. The held-out cosine recorded during selection does change, because it is now measured on gene folds. Each selection record names the selection that produced it, and `open_vocab_generalization_v1.verify_masking` refits with that selection. The K562 Candidate Effect Distillation entry point (`candidate_effect_distillation_v1.run`) keeps the submitted selection: its pool has one row per gene, so its folds are already gene folds, but its response basis is fitted before the folds are cut.
- **Fusion normalization.** `four_context_v1.evaluate.fuse` standardizes each branch per query row over every column it is given, which in the published runs is the whole scored pool, and its docstring now says so.
- **An option to standardize over the evaluated candidates.** `four_context_v1.evaluate.fuse(base, effect, standardize_over=mask)` standardizes both branches over the columns selected by a boolean mask or an index array only, and returns 0.0 elsewhere, so those columns must not be graded. `python -m gene_open_inverse.open_vocab_generalization_v1.poolsize --packs PACKS --output OUT --standardize subset` does the same inside every subsample of the vocabulary-size curve. The defaults, `standardize_over=None` and `--standardize pool`, are the published computation. Tests 61 to 63 in `four_context_v1/test_protocol.py` cover the option.

Smaller changes: the protocol checks in `revision_v1`, `external_baselines_v1` and `open_vocab_generalization_v1` now fail their test under pytest when they do not hold; before, only their standalone runners reported a failure. `requirements-cpu.txt` installs the CPU build of PyTorch. The version in `pyproject.toml` matches this release. Two strings that named the venue were made venue neutral, one of them the mission name that `external_baselines_v1` writes into its output records.

## Data provenance

The processed artifact is derived from public K562 and RPE1 Perturb-seq screens, GEO GSE264667 HepG2 and Jurkat screens, STRING v12.0, MAP-KG, ESM-2, Reactome release 97, and NCBI human Gene Info. The manuscript provides complete citations and inclusion criteria.

## Archive integrity

`CODE_PACKAGE_RELEASE.json` records the package version and release date. `CODE_PACKAGE_MANIFEST.sha256` lists the SHA-256 digest of every payload file. To verify it on Linux or macOS:

```bash
sha256sum --check CODE_PACKAGE_MANIFEST.sha256
```

The archive contains no symlinks. All paths are relative, and the build rejects author-account identifiers, machine-specific home paths, generated bytecode, model weights, and bundled datasets.
