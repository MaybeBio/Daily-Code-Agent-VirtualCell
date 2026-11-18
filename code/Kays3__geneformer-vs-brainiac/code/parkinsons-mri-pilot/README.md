# Parkinson’s structural MRI pilot

[Editable slides](https://github.com/Kays3/geneformer-vs-brainiac/blob/HEAD/parkinsons-mri-pilot/Parkinsons-MRI-pilot.pptx) · [PDF slides](https://github.com/Kays3/geneformer-vs-brainiac/blob/HEAD/parkinsons-mri-pilot/Parkinsons-MRI-pilot.pdf) · [Aggregate metrics](https://github.com/Kays3/geneformer-vs-brainiac/blob/HEAD/parkinsons-mri-pilot/results/metrics.json)

We tested whether frozen BrainIAC features from a single T1 scan distinguished Parkinson’s participants with mild cognitive impairment from participants labeled cognitively normal. The study included all 33 Parkinson’s participants in [OpenNeuro ds005892 v1.0.0](https://doi.org/10.18112/openneuro.ds005892.v1.0.0), 15 with mild cognitive impairment and 18 with normal cognition. The dataset also includes healthy controls, who were outside the scope of this comparison.

| Model | Held-out AUROC | Balanced accuracy at 0.5 |
|---|---:|---:|
| Age and sex | 0.533 | 0.628 |
| BrainIAC MRI features | 0.659 | 0.606 |
| Age, sex and BrainIAC MRI features | 0.604 | 0.606 |

We used a single stratified five-fold split by participant. Every participant received one out-of-fold prediction from each model. The image encoder was frozen. Within each training fold we fitted three principal components to the 768 MRI features, standardized the predictors, and fitted class-balanced logistic regression. The table pools the out-of-fold predictions. Model selection and external validation would require additional data.

Scans were rigidly registered to the Nilearn MNI152 template and masked with the positive-intensity template region. The BrainIAC validation transform resized the volume to 96 × 96 × 96 and normalized nonzero voxels. This simplified preprocessing differs from the authors’ individual brain extraction and bias correction. Axial quality review showed residual orbital tissue and mismatched anatomy, particularly in MJF027 and MJF033. All participants were retained. The observed AUROC difference could reflect preprocessing or sample variation. It cannot establish a cognitive biomarker or predict future motor decline.

## Reproduce the run

Use Python 3.11, PyTorch with CUDA, MONAI 1.3.2, einops, nibabel, Nilearn, SimpleITK, scikit-learn, pandas, PyYAML and matplotlib. From this directory, point PYTHONPATH at the src directory of [BrainIAC revision ba60f45](https://github.com/AIM-KannLab/BrainIAC/tree/ba60f45bed832ee0e5683c678caeaeefe2072f0d/src). Download BrainIAC.ckpt using the authors’ checkpoint link. The checkpoint used here had SHA-256 f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693.

```sh
python fetch.py
python prepare.py
PYTHONPATH="$BRAINIAC_SRC" python evaluate.py "$BRAINIAC_CHECKPOINT"
```

The public results folder contains aggregate metrics and a receiver operating characteristic figure. Running the evaluation also generates participant-level scores and an axial image review sheet locally. The proposed 24-month MDS-UPDRS III forecast requires approved longitudinal PPMI records linked to baseline MRI and motor scores with medication state recorded.

## Update the slides with each run

After running the evaluation, use `python build_slides.py` with python-pptx installed to refresh the table and ROC figure. Review cohort counts, methods and interpretation when the experiment changes. Export the reviewed PowerPoint to PDF and commit the slides, aggregate results and analysis changes together.
