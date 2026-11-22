# fm-ancestry-bias

**Do single-cell foundation models annotate immune cells less accurately for donors of African ancestry?**

A preregistered benchmark of four single-cell foundation models (scGPT, Geneformer, UCE, scFoundation) against
three non-foundation baselines (scVI, CellTypist, PCA), comparing annotation accuracy for African-ancestry (AA)
and European-ancestry (EA) donors.

Author: Akinsola Clement Morakinyo (Clement Akinsola) - Medical Artificial Intelligence Laboratory, Lagos

## Why
Foundation models learn from large single-cell atlases that contain few African-ancestry donors. In CELLxGENE
Census (release 2025-11-08), only 4 of 2,252 healthy blood donors are recorded as African ancestry (0.2%), and
462 have no ancestry recorded. This study tests whether that imbalance makes the models less accurate on
African-ancestry cells.

## Data (all open access)
| Dataset | Role | Donors after QC (AA / EA) | Cells |
|---|---|---|---|
| Randolph et al. 2021, Science (GSE162632 + Zenodo 4273999) | Primary | 45 / 45 | 235,150 |
| Prostate cancer PBMC (GSE270387) | Replication | 33 / 24 | 293,287 |
| Lupus CITE-seq (GSE189050) | Excluded: ancestry confounded with sequencing batch | - | - |

## Preregistration
The full analysis plan is in [`PREREGISTRATION.md`](https://github.com/Ceefour18/fm-ancestry-bias/blob/HEAD/PREREGISTRATION.md), frozen at git tag `prereg-v1`
and registered on OSF: [osf.io/xed6m](https://osf.io/xed6m/overview) (DOI [10.17605/OSF.IO/XED6M](https://doi.org/10.17605/OSF.IO/XED6M)).
Primary outcome: per-donor macro-F1 on the original authors' coarse cell-type labels.
Smallest effect of interest: 2 macro-F1 points.

## Status
- [x] Data preparation, QC, cell-type labels, donor splits (scripts 00-32)
- [x] Blinded power pilot (scripts 40, 45) - pooled summaries only, no ancestry-stratified results
- [ ] Foundation-model embeddings and evaluation (scripts 50+)

## Repository layout
| Folder | Contents |
|---|---|
| `scripts/` | One script per step, numbered by stage: 00 audit, 10 data, 20 QC, 30 labels, 40 baselines/power, 50 embeddings, 60 annotation, 70 metrics, 80 statistics, 90 figures |
| `slurm/` | SLURM job file for each script (same number) |
| `envs/` | Conda environments (`core.yml` Python, `R.yml` R) |
| `results/` | Small result tables, QC and label summaries, `splits.json` (fixed donor splits) |

Raw and processed data are not stored here; scripts `10-17` download and rebuild them from GEO and Zenodo.

## Reproduce
```bash
conda env create -f envs/core.yml
conda env create -f envs/R.yml
# then run scripts in numeric order; on SLURM: sbatch slurm/NN_name.sbatch
```
Key versions: Python 3.11, scanpy 1.11.5, scipy 1.13.1, umap-learn 0.5.7.

## License
MIT (code). Data remain under their original licences.
