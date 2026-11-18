import numpy as np
import scanpy as sc

# Where Scanpy saves the files it downloads
sc.settings.datasetdir = "data/downloads"

# Raw counts: 2,700 cells x 32,738 genes, gene symbols as names
raw = sc.datasets.pbmc3k()

# Processed version from the Scanpy tutorial: 2,638 filtered cells with cell-type labels
processed = sc.datasets.pbmc3k_processed()

# Keep only the cells that have a label, in the same order
raw = raw[processed.obs_names].copy()
raw.obs["cell_type"] = processed.obs["louvain"].astype(str)

# Checks: counts should be whole numbers, gene names should be symbols
counts = raw.X.toarray() if hasattr(raw.X, "toarray") else raw.X
print(raw)
print("All counts are whole numbers:", np.all(counts == np.round(counts)))
print("Max count:", counts.max())
print("First genes:", list(raw.var_names[:5]))
print(raw.obs["cell_type"].value_counts())

raw.write_h5ad("data/pbmc3k_raw_labeled.h5ad")
