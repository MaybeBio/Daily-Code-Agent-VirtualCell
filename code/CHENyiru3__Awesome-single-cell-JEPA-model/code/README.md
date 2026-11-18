# Awesome Single-Cell JEPA Models

A curated list of research papers and models on **Joint-Embedding Predictive Architectures (JEPAs)** for single-cell and spatial omics. JEPA is a self-supervised learning paradigm by directly predicting representations in latent space. JEPA also could be an extra objective term in other learning paradigm, which we called the model to be JEPA-inspired. 

For general JEPA model, could check [Awesome JEPA](https://github.com/AI-in-Transportation-Lab/awesome-jepa). Also, [I-JEPA](https://arxiv.org/abs/2301.08243) for images, and [V-JEPA 2](https://arxiv.org/abs/2506.09985) for video, could also be a classic start to learn about JEPA.

## AI assistant disclosure

We will use AI assistant to manage and expand this list, but all updates would be reviewed carefully by repo owner. Any [contribution](#Contributing) is welcomed, but make sure you have reviewed the content and have your own judgement. 

## Contents

- [Single-cell omics](#single-cell-omics)
- [Spatial omics](#spatial-omics)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

Papers are listed newest to oldest within each section. Dates use the arXiv v1 submission record or the first bioRxiv posting. A workshop date is labeled when the first online posting date could not be verified.

**Scope.** Papers must apply a JEPA or a JEPA-inspired latent prediction objective to single-cell or spatial omics data. Hybrid methods qualify when latent prediction is part of the training objective. Papers are grouped by the omics data they model.

**Resources.** Code and checkpoints are checked separately. **Not open-sourced** indicates that no official public implementation was found. **No public checkpoint found** indicates that model weights could not be located in an official release. Resource links were checked on 16 November 2026.

## Single-cell omics

| Model | First release | Paper · Venue | Latent target and context | Code | Checkpoints |
| --- | --- | --- | --- | --- | --- |
| **PerturbJEPA** | 2026-11-16 | [Joint-Embedding Predictive Learning of Cellular Perturbation Responses](https://arxiv.org/abs/2611.20575) · Pang et al. · arXiv | Predicts treated-population embeddings from unpaired control cells and drug–dose inputs. Combines latent population energy matching with effect alignment, state/effect regularization, and intervention ranking; no expression reconstruction. | [Training and evaluation code](https://github.com/BGI-Hangzhou-OmicsAI/PerturbJEPA) · MIT · Released examples use Parse10M | No public checkpoint found |
| **scRM** | 2026-11-02 | [scRM: A single-cell foundation model trained on scRNA-seq reads](https://doi.org/10.64898/2026.10.24.761590) · He et al. · bioRxiv | Predicts unmasked EMA-teacher molecule and cell embeddings from block-masked, UMI-grouped scRNA-seq read sequences. Training also uses masked-base reconstruction and SIGReg. | [Training and evaluation code](https://github.com/altoslabs/scRM) · MIT | Checkpoint availability not verified |
| **DCAF** | [2026-10-19](https://sciety.org/articles/activity/10.64898/2026.10.13.758713) | [Unifying Data Validity, Evaluation Validity and Predictive Modeling for Single-cell Perturbation Simulation](https://doi.org/10.64898/2026.10.13.758713) · Xia et al. · bioRxiv | Combines JEPA with explicit regulatory dynamics to predict single-cell expression responses to perturbations. | **Not open-sourced** | No public checkpoint found |
| **JEPA-Anything** | 2026-09-17 | [Learning Predictive Models across Different Worlds](https://arxiv.org/abs/2609.20800) · Cui et al. · arXiv | The single-cell experiments predict complete cell embeddings from masked scRNA-seq and latent perturbation responses from control states and perturbation inputs. | [Project code](https://github.com/Gen-Verse/JEPA-Anything) · [Norman inference code](https://huggingface.co/datasets/Gen-Verse/jepa-anything/blob/main/02_Single_Cell/Norman_Single_Cell_Perturbation_Prediction/test/Norman_Single_Cell_Perturbation_Prediction_test.py) | [Norman checkpoint](https://huggingface.co/datasets/Gen-Verse/jepa-anything/tree/main/02_Single_Cell/Norman_Single_Cell_Perturbation_Prediction/checkpoints) |
| **GB-JEPA** | 2026-08-06 | [Graph-structured joint-embedding prediction for single-cell transcriptomics](https://arxiv.org/abs/2608.05928v2) · Wang et al. · arXiv | Predicts teacher embeddings pooled over masked, graph-connected gene blocks from the remaining genes in a cell. The first version used the name BioM-JEPA. | **Not open-sourced** · Release announced | No public checkpoint found · Release announced |
| **CellOS** | 2026-06-23 | [Learning a World Model of Cellular State through Joint Embedding Prediction](https://doi.org/10.64898/2026.06.18.733163) · Zhou et al. · bioRxiv | Uses LLM-JEPA to predict the latent embedding of a perception-ranked transcriptome view from the expression-ranked view of the same cell. | **Not open-sourced** | No public checkpoint found |
| **scJEPA** | 2026-04-27 ([workshop date](https://research.google/conferences-and-events/google-at-iclr-2026/)) | [Dual-Space Self-Supervised Learning for Single-Cell Representations](https://openreview.net/pdf?id=KadCjvcLOz) · Zhai et al. · ICLR LMRL Workshop | Matches cell embeddings across masked and unmasked scRNA-seq views. Training also uses expression denoising and SIGReg. | **Not open-sourced** | [scJEPA checkpoint](https://huggingface.co/zhaiyuyao/scJEPA/tree/main) |
| **Cell-JEPA** | 2026-02-02 | [Latent Representation Learning for Single-Cell Transcriptomics](https://arxiv.org/abs/2602.02093) · ElSheikh et al. · arXiv | Predicts the unmasked EMA teacher's whole-cell embedding from masked scRNA-seq and also reconstructs masked gene-expression values. | **Not open-sourced** | No public checkpoint found |
| **GeneJEPA** | 2025-10-15 | [A Predictive World Model of the Transcriptome](https://doi.org/10.1101/2025.10.14.682378) · Litman et al. · bioRxiv | Predicts latent representations of masked gene sets from visible scRNA-seq genes using a Perceiver-style encoder. | [Public training and inference code](https://github.com/BiostateAI/GeneJEPA) · License unspecified in the code repository | [GeneJEPA checkpoint](https://huggingface.co/elonlit/GeneJEPA) |

## Spatial omics

| Model | First release | Paper · Venue | Latent target and context | Code | Checkpoints |
| --- | --- | --- | --- | --- | --- |
| **CellWorld** | 2026-08-07 | [From Gene-Level Reconstruction to Latent Cell Prediction in Spatial Transcriptomics Foundation Models](https://arxiv.org/abs/2608.06659) · Liu et al. · arXiv | Predicts embeddings of masked cells from visible neighboring cells, target coordinates, and a small partial-expression hint. | [Training and inference code](https://github.com/UoM-HealthAI/CellWorld) | [CellWorld checkpoints](https://huggingface.co/Haiping-UoM/CellWorld) · Authorization required |
| **TERRA** | 2026-08-04 | [Multi-scale modeling of human tissues from spatial transcriptomics with TERRA](https://doi.org/10.64898/2026.07.29.741565) · Birk et al. · bioRxiv | Predicts masked gene-token embeddings using the focal cell and its spatial neighbors. The encoder produces gene, cell, and neighborhood representations. | [TERRA code](https://github.com/Lotfollahi-lab/terra) | Author-listed inference weights: [TERRA-96M](https://huggingface.co/lotfollahi-lab/TERRA-96M) (held-out benchmark split) and [TERRA-112M](https://huggingface.co/lotfollahi-lab/TERRA-112M) (full corpus; no held-out split) · CC-BY-NC-4.0 |
| **SpatialJEPA** | 2026-07-22 | [JEPA-inspired graph-context distillation for spatially aware multiomics integration](https://doi.org/10.64898/2026.07.21.739810) · Mann-Krzisnik & Li · bioRxiv | Distills cell embeddings from a spatial graph teacher into a student that sees each cell without its neighbors. The student can embed dissociated RNA–ATAC profiles. | [Training and analysis code](https://github.com/li-lab-mcgill/SpatialJEPA) | [Author-listed Zenodo deposit](https://doi.org/10.5281/zenodo.23933515) |
| **ST-JEPA** | 2026-07-10 ([workshop date](https://genbio-workshop.github.io/2026/)) | [Joint-Embedding Predictive Architecture for Spatial Transcriptomics](https://openreview.net/pdf/d7c8c0703f68e025ff7ce18da069db6da43234b0.pdf) · Birk et al. · ICML Workshop on Generative and Agentic AI for Biology | Predicts masked gene-token embeddings in spatial cell graphs. Its tokenization supports gene, cell, and neighborhood representations. | **Not open-sourced** | No public checkpoint found |

## Contributing

We welcome contributions to this repository! Suggest a paper, or report a broken link, or raise any content suggestion, by [opening an issue](https://github.com/CHENyiru3/Awesome-single-cell-JEPA-model/issues/new) or opening a pull request. 

See [CONTRIBUTING.md](https://github.com/CHENyiru3/Awesome-single-cell-JEPA-model/blob/HEAD/CONTRIBUTING.md) for guidance on checking scope, sources, and proposed updates.

Use the row below for new entries. Describe the context the model sees and the embedding it predicts or matches. Link to the published paper when available, record its first release as YYYY-MM-DD, and note code and checkpoint availability separately. Place each paper once in the most relevant section.

```markdown
| **Model** | YYYY-MM-DD | [Paper title](paper-url) · First author et al. · Venue | Context and latent target. | [Official code](code-url) or **Not open-sourced** | [Official checkpoint](checkpoint-url) or No public checkpoint found |
```

## Citation

If you use this curated list in your work, please cite the repository. For a specific snapshot, replace the repository URL with the GitHub permalink for the commit you used. Machine-readable metadata is available in [CITATION.cff](https://github.com/CHENyiru3/Awesome-single-cell-JEPA-model/blob/HEAD/CITATION.cff).

```bibtex
@misc{chen2026awesomejepa,
  author       = {Chen, Yiru},
  title        = {Awesome Single-Cell {JEPA} Models},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/CHENyiru3/Awesome-single-cell-JEPA-model},
  note         = {Accessed YYYY-MM-DD}
}
```

## License

The original curation and text in this repository are available under the [Creative Commons Attribution 4.0 International license](https://github.com/CHENyiru3/Awesome-single-cell-JEPA-model/blob/HEAD/LICENSE). Linked papers, code, checkpoints, and datasets follow the terms set by their respective rights holders.
