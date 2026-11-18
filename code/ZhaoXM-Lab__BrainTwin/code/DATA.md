# Where to put data and weights

The literature index, knowledge graph, benchmark files, and model weights are not in this repository. Download the current public record and place the files in the directories below. Those directories already exist.

https://doi.org/10.6084/m9.figshare.35014504

This version contains:

| File | What to do with it |
|---|---|
| `benchmarks.zip` | Unzip into `data/benchmarks/`. |
| `literature_index.tar.gz` | Unpack, then copy `index.faiss` and `index.pkl` into `data/literature_index/`. |
| `neo4j.dump` | Load into Neo4j 2025.02. |
| `qwen3_8B_lora_r256_checkpoint-43000.tar.gz` | LoRA adapter. It is not a standalone model. |

The full-parameter checkpoint, `qwen3_8B_full_checkpoint-21000.tar.gz`, is not in this version. It will be added to the same record. Until that file is there, `checkpoints/qwen3_8b_braintwin/` stays empty. The embedding model is not in the record either.

Paths below are relative to the `BrainTwin/` root.

## Weights

| Put it here | What it is |
|---|---|
| `checkpoints/qwen3_8b_braintwin/` | BrainTwin full-parameter checkpoint. Not on Figshare yet. |
| `checkpoints/qwen3_8b_base/` | Optional Qwen3-8B base checkpoint, only for a base-model comparison. Download `Qwen/Qwen3-8B` separately. |

When `qwen3_8B_full_checkpoint-21000.tar.gz` is added, unpack it and move the `full_checkpoint-21000` folder to `checkpoints/qwen3_8b_braintwin/`. Keep the four `model-0000N-of-00004.safetensors` files together with `model.safetensors.index.json`. Serve that folder with your own OpenAI-compatible server. Export `BRAINTWIN_LLM_URL` as that server's origin and `BRAINTWIN_MODEL_ID` as the id the server reports. This repository does not record the server address.

The LoRA archive is checkpoint-43000, rank 256, alpha 512. Unpack it, load `Qwen/Qwen3-8B`, then apply the folder with PEFT. `adapter_model.safetensors` and `adapter_config.json` are the adapter. This repository does not load the adapter itself. `rng_state_*.pth`, `scheduler.pt`, `trainer_state.json`, `training_args.bin`, and `zero_to_fp32.py` are only for resuming training.

## Literature index

| Put it here | What it is |
|---|---|
| `data/embedding/multilingual-e5-large-instruct/` | Embedding model directory, including weights and tokenizer files. This is `intfloat/multilingual-e5-large-instruct`. It is not in the Figshare record. |
| `data/literature_index/` | FAISS literature index. Place `index.faiss` and `index.pkl` directly in this directory. |

```bash
tar -xzf literature_index.tar.gz
cp academic_papers_index/index.faiss academic_papers_index/index.pkl data/literature_index/
```

## Knowledge graph

Export `BRAINTWIN_NEO4J_URL` and `BRAINTWIN_NEO4J_PASSWORD` for your own Neo4j server. The default user name is `neo4j`.

| Put it here | What it is |
|---|---|
| `data/knowledge_graph/neo4j.dump` | Neo4j 2025.02 dump from the Figshare record. |

Stop the server, then load the dump with Java 21. The `--from-path` directory is the folder that contains `neo4j.dump`.

```bash
neo4j-admin database load neo4j --from-path=data/knowledge_graph --overwrite-destination=true
```

Set a new password after the load. The dump does not contain one. The default user name is `neo4j`.

The graph uses these labels: CellType, Gene, BrainRegion, BrainSubRegion, SNP, GOTerm, Trait.

## Benchmark questions

```bash
unzip benchmarks.zip -d data/benchmarks
```

The zip also contains a short `README.md`. The four question files are:

| Put it here | What it is |
|---|---|
| `data/benchmarks/singlecell_DEG.json` | Single-cell differential-expression questions. |
| `data/benchmarks/singlecell_region.json` | Single-cell brain-region questions. |
| `data/benchmarks/genomics_SCQ.json` | Genomics multiple-choice questions. |
| `data/benchmarks/image_5K.json` | Imaging questions. |

## Case inputs

These files are not in the Figshare record. Genomics scripts read `data/genomics/`. The single-cell case reads the two h5ad examples below. The one-person MRI case reads a FreeSurfer subject directory or feature tables that you pass in. The cohort script reads the UCLA and ADNI files in this list. `stab2_hs_anno.h5ad` is the reference atlas for the SNP report, not a single-cell case input.

| Put it here | What it is |
|---|---|
| `data/singlecell/grubman_2019_alzheimer_entorhinal_cortex_clean.h5ad` | Single-cell example and the Alzheimer disease-pipeline input. |
| `data/mri/UCLA/all_sample_structured.json` | UCLA cohort used by `cases/mri/cohort_questions.py`. |
| `data/mri/UCLA/subjects_meta.csv` | UCLA subject metadata. |
| `data/mri/ADNI/all_sample_structured.json` | ADNI cohort used by the same script. |
| `data/mri/ADNI/aseg_features.csv` | ADNI subject metadata. |
| `data/genomics/ukb/Metadata.csv` | UK Biobank case and control labels for the disease-risk scripts. |
| `data/singlecell/jakel_2019_multiple_sclerosis_white_matter.h5ad` | Single-cell example. |
| `data/singlecell/stab2_hs_anno.h5ad` | STAB2 human annotation. |
| `data/genomics/prs/` | PGS scoring files. |
| `data/genomics/ukb/UKB_PRS_result.json` | UKB PRS reference distribution. |
| `data/genomics/ukb/UKB_PRS_result_zscore.csv` | UKB PRS z scores. |
| `data/genomics/ukb/vcf_by_sample/` | Per-sample VCF files. |
| `data/genomics/ukb/variant_prioritization_results/` | Variant-prioritization outputs. |
| `data/snpeff/snpEff.jar` | snpEff. |
| `data/snpeff/SnpSift.jar` | SnpSift. |
| `data/variant_prioritization/` | Variant-prioritization code and its `resources/` directory. |
