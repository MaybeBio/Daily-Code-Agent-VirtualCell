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

## Published 24-month motor trajectory

In a separate early Parkinson’s cohort, Simuni et al. reported mean MDS-UPDRS Part III scores of 20.89 at baseline (423 exams), 24.87 at six months (400), 25.10 at 12 months (334) and 27.13 at 24 months (282) for their untreated + treated OFF group. These visit means summarize available exams. Estimating individual changes requires paired scores and medication-state information. Testing a BrainIAC forecast additionally requires baseline MRI from those same participants. [Table 2, Simuni et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6001458/#mds27361-tbl-0002)

## Proposed study of signs before diagnosis

Anchor each participant’s timeline at the first confirmed Parkinson’s diagnosis and plot dated smell tests, motor exams, dopamine-transporter imaging and any available earlier MRI. Record the last normal and first abnormal visit separately for each marker. If the earliest test is already abnormal, report that the change predates that visit. Define thresholds before examining outcomes and distinguish recalled symptoms from dated measurements.

For a 24-month prediction test, use information available at the prediction visit, include people who remain undiagnosed through follow-up, and account for incomplete follow-up and alternative diagnoses. Compare a clinical model with the same model plus BrainIAC in participants with prediagnostic T1 MRI. Serial central and peripheral markers could test hypotheses about the order of detectable changes. The initiating lesion and its date remain unknown. No participant-level prediagnostic analysis has been run.

[PARS longitudinal study](https://jamanetwork.com/journals/jamaneurology/fullarticle/2630495) · [Multimodal study of brain-first and body-first hypotheses](https://doi.org/10.1093/brain/awaa238)

## Sensitivity to the participant split

Repeating the original five-fold analysis with seeds 0–49 reproduced the original split (seed 42). The median MRI-only AUC was 0.619, with a range of 0.500–0.667. Age and sex gave a median of 0.524 (0.409–0.598), and the combined model gave 0.591 (0.433–0.700). These ranges describe partition sensitivity conditional on the same 33 people. They do not quantify population uncertainty. [All 50 splits](https://github.com/Kays3/geneformer-vs-brainiac/blob/HEAD/parkinsons-mri-pilot/results/auc-by-split.csv)
