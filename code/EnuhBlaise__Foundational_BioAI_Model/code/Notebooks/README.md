# Exploration notebooks

Run in order; each is self-contained and executes in under 20 seconds on a
laptop CPU.

1. `01_raw_count_characteristics.ipynb` – what the raw matrix looks like
2. `02_curation_qc_normalization_hvg.ipynb` – what each curation step changes
3. `03_splitting_and_leakage.ipynb` – how cells are divided, and why it matters
4. `04_tokenization.ipynb` – what each tokenizer keeps and discards

Set `DATA_PATH` in the second code cell to an `.h5ad` file to use real data
(`OBS_MAP` renames its obs columns to `donor_id`, `batch_id`, `study_id`,
`cell_type`; `RAW_LAYER` names the layer holding raw counts). Cells that need
simulator ground truth skip themselves on real data.

`nbutils.py` holds chart styling and the data loader only. Every method under
study is imported from `src/` and called in the notebook itself.

The outputs saved in these files come from the synthetic simulator. They
describe the simulator, not biology.
