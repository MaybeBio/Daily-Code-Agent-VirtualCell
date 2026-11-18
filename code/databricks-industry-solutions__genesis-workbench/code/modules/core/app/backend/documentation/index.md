# Genesis Workbench Documentation

## Platform

- [AI-Assisted Workflow Generation (Vortex)](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/ai_assisted_workflows.md) — Visual drag-and-drop canvas for composing and running multi-step pipelines from endpoints, prebuilt workflows, transforms, and data IO; generate a workflow from a plain-language goal with an LLM that self-reviews and repairs its own draft, then track every run with per-node status, JSON view/copy, and one-click Re-run
- [MCP Server](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/mcp_server.md) — Companion Model Context Protocol server (`mcp-genesis-workbench`) that exposes every deployed endpoint and prebuilt workflow as MCP tools over streamable HTTP, callable from the Databricks AI Playground, Claude, Cursor, or custom agents

## Genomics

- [Variant Calling](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/variant_calling.md) — GPU-accelerated germline variant calling from FASTQ files using Parabricks
- [GWAS Analysis](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/gwas_analysis.md) — Genome-wide association testing between genetic variants and phenotypes using Glow
- [VCF Ingestion](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/vcf_ingestion.md) — Convert VCF files to Delta tables for querying and analysis
- [Variant Annotation](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/variant_annotation.md) — Annotate variants with ClinVar clinical significance data

## Large Molecule

- [Protein Structure Prediction](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/protein_structure_prediction.md) — Predict 3D protein structures with ESMFold (fast) or AlphaFold2 (high-accuracy)
- [Protein Design](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/protein_design.md) — Design novel proteins by redesigning specified regions using RFDiffusion + ProteinMPNN
- [Inverse Folding](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/inverse_folding.md) — Design new sequences for a fixed backbone with ProteinMPNN, each validated by re-folding with ESMFold
- [Sequence Similarity Search](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/sequence_search.md) — Fast BLAST-like search across 150M+ sequences using ESM-2 embeddings
- [RFD4-Proteina Design](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/rfd4_proteina_design.md) — Generate novel protein backbones + sequences with the NVIDIA×Baker RFD4-Proteina flow-matching model, served on a GPU_XLARGE (1× H100) endpoint via Express env_pack; de-novo monomer design, first-draft target-conditioned (binder/motif) design, and optional LoRA fine-tuning
- [Antibody Design (VHH)](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/antibody_design.md) — Generate + reward-optimize single-domain (VHH/nanobody) antibodies against an antigen epitope with RFD4-Proteina (in-process, H100), scored on binding (Boltz ipTM), fold confidence (ESMFold pLDDT), and developability (solubility, half-life, Tm, low immunogenicity)
- [Vaccine Immunogen Design](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/vaccine_immunogen.md) — Graft a conserved epitope onto de-novo stable scaffolds with RFD4-Proteina motif-scaffolding (in-process, H100); reward-optimized on epitope-presentation fidelity (motif RMSD), scaffold fold confidence (ESMFold pLDDT), and manufacturability (solubility, Tm, sequence liabilities)

## Small Molecule

- [Guided Molecule Design](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/guided_molecule_design.md) — Closed-loop small-molecule generator: GenMol grows a seed scaffold/binding motif into novel candidates, scored under hard constraints (Min QED, Max ClinTox) with optional ESMFold+DiffDock binding in the reward, reseeding from the best each iteration
- [Molecular Docking](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/molecular_docking.md) — Predict protein-ligand binding poses using DiffDock
- [Protein Binder Design](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/protein_binder_design.md) — Design proteins that bind a target protein using Proteina-Complexa
- [Ligand Binder Design](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/ligand_binder_design.md) — Design proteins that bind a small molecule using Proteina-Complexa-Ligand
- [Motif Scaffolding](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/motif_scaffolding.md) — Transplant functional motifs into new protein scaffolds
- [Guided Enzyme Optimization](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/enzyme_optimization.md) — Reward-weighted optimization loop on top of motif scaffolding; scores each candidate on motif fidelity, fold confidence, optional substrate complex, and four developability axes (solubility, anchor-relative half-life, melting temperature, immunogenic burden). Two generation modes: **Fast** (default, ~30 min, no GPU cost — endpoint-based AME with parent resampling between iterations) and **Accurate** (~30-60 min, ~$22 GPU — in-process AME with Feynman-Kac steering biasing diffusion toward developability)
- [ADMET & Safety](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/admet_safety.md) — Profile molecules for drug-like properties and toxicity using ChemProp (with optional KERMT side-by-side)

## Single Cell

- [Single Cell Analysis](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/single_cell_analysis.md) — End-to-end scRNA-seq processing with Scanpy or RAPIDS, plus interactive results viewer with differential expression, pathway enrichment, and trajectory analysis
- [Cell Type Annotation](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/cell_type_annotation.md) — Predict cell types per cluster using SCimilarity's 23M-cell reference database
- [Cell Similarity Search](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/cell_similarity.md) — Find matching cells across published studies using SCimilarity embeddings
- [Gene Perturbation Prediction](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/perturbation_prediction.md) — Predict effects of gene knockouts or overexpression using scGPT

## NVIDIA BioNeMo

- [ESM2 Fine-tuning & Inference](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/bionemo_esm2.md) — Fine-tune and run inference with NVIDIA's ESM-2 protein language model
- [KERMT (ADMET fine-tune)](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/app/backend/documentation/kermt_admet.md) — Fine-tune NVIDIA-BioNeMo KERMT (Kinetic GROVER Multi-Task GNN) on a SMILES+target assay, deploy it as a serving endpoint, and compare its toxicity/ADMET predictions side-by-side with ChemProp in the Small Molecule ADMET tab
