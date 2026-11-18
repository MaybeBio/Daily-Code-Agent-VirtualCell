# Single-cell case

`singlecell_report.py` takes one h5ad file, detects whether the counts are raw or already log-transformed, clusters the cells, finds marker genes, and asks the model to name the clusters. It also writes a UMAP and a composition chart.

The two example files go in `data/singlecell/`:

- `grubman_2019_alzheimer_entorhinal_cortex.h5ad`
- `jakel_2019_multiple_sclerosis_white_matter.h5ad`

Cached outputs go to `data/singlecell/cache/` unless `SINGLECELL_RESULTS_DIR` is set. The chat model needs `BRAINTWIN_LLM_URL` and `BRAINTWIN_MODEL_ID`.

This folder does not run the STAB2 atlas. That file is only the reference used by the genomics SNP report.

`disease_pipeline/` is the Alzheimer case on the author cell types. It does not recluster. From the repository root:

```bash
python cases/singlecell/disease_pipeline/run_pipeline.py --config cases/singlecell/disease_pipeline/config_grubman2019.yaml
python cases/singlecell/disease_pipeline/run_pipeline.py --config cases/singlecell/disease_pipeline/config_grubman2019.yaml --generate-mas
```

The first command writes the differential tables and Enrichr terms. The second reads those tables, runs the fixed gene–GO and gene–trait queries, and asks the model to write the inference. The h5ad path in the config is `data/singlecell/grubman_2019_alzheimer_entorhinal_cortex_clean.h5ad`.
