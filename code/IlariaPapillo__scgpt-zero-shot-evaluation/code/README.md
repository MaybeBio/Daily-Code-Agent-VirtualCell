# Zero-shot evaluation of scGPT-spatial on spatial transcriptomics

Can a spatial-transcriptomics foundation model find biologically meaningful tissue structure **without any fine-tuning**? This project tests the zero-shot embeddings of [scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial) on public 10x Genomics Visium data. It compares them against expert pathology annotations and against a standard PCA pipeline.

Work carried out as a guest researcher at **Erasmus MC** (Rotterdam), [month–month 2026].

## What I did

1. **Multimodal integration (reproduction + baseline).** Reproduced the official zero-shot integration of Visium and Xenium data (developing fetal lung). Added a PCA baseline to check whether the foundation-model embeddings beat a standard pipeline.
2. **Spatial domain clustering on a pathologist-annotated slide.** Embedded a human prostate cancer FFPE Visium slide zero-shot, clustered the embeddings (Leiden), and mapped the clusters back onto the tissue. Then:
   - evaluated the clusters against the pathologist annotation with ARI, NMI and scIB metrics;
   - interpreted the clusters with marker genes (differential expression);
   - compared against a PCA baseline, across several clustering resolutions and on a random 20% subset of spots.
3. **Generalisation to a second slide.** Applied the same pipeline to an unannotated prostate acinar cell carcinoma slide and compared it with 10x's own graph-based and k-means clusterings.
4. **Gene vocabulary analysis.** Diagnosed why the model silently drops genes, and identified outdated gene symbols as a fixable source of lost signal.

## Repository structure

```
notebooks/
  01_multimodal_integration_fetal_lung.ipynb
  02_spatial_domain_clustering_prostate_cancer.ipynb
  03_generalisation_prostate_acinar_carcinoma.ipynb
requirements.txt
```

## Setup

1. Clone and install scGPT-spatial:
   ```bash
   git clone https://github.com/bowang-lab/scGPT-spatial.git
   ```
2. Download the [pretrained scGPT-spatial checkpoint](https://figshare.com/articles/software/scGPT-spatial_V1_Model_Weights/28356068) into `scGPT_spatial_v1/`, including the `.json` and `.csv` files.
3. Install the other dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the notebooks in a subfolder of the scGPT-spatial repository (e.g. `scGPT-spatial/tutorials/`), since they import `scgpt_spatial` from the parent folder.

### Data (all public)

- **Developing fetal lung (Visium + Xenium):** processed subset from [figshare](https://figshare.com/articles/dataset/_Evaluation_Processed_Multi-modal_Developing_Fetal_Lung_Data_Subset_Visium_Xenium_/28356116), from [Quach et al., 2024](https://www.nature.com/articles/s41467-024-50281-5).
- **Human prostate cancer, FFPE (Visium)**, with pathologist spot annotations.
- **Human prostate acinar cell carcinoma, FFPE (Visium)**, from 10x Genomics, including the Space Ranger clustering output.

Data files are not included in this repository.

## Acknowledgements

Model and pretrained weights: Wang, Cui, Zhang, Xie, Goodarzi and Wang, *scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics*. Notebook 01 starts from their official tutorial.
