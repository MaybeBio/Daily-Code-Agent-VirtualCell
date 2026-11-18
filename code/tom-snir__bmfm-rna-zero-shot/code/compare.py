import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

out = "outputs/zero_shot"
res = "results"

# Reference labels (from the Scanpy/Seurat tutorial)
adata = sc.read_h5ad("data/pbmc3k_raw_labeled.h5ad")
ref = adata.obs["cell_type"].rename("reference")

# Model outputs. embeddings.csv has no header: first column is the cell barcode
pred = pd.read_csv(f"{out}/predictions.csv", index_col=0)
emb = pd.read_csv(f"{out}/embeddings.csv", header=None, index_col=0)
emb.columns = [f"emb_{i + 1}" for i in range(emb.shape[1])]
prob = pd.read_csv(f"{out}/probabilities.csv", index_col=0)

# Confidence = probability of the predicted cell type
ct_prob = prob.filter(like="cell_type_")
pred["cell_type_prob"] = ct_prob.max(axis=1)

# One table per cell: reference label + model predictions
cells = pd.concat([ref, pred], axis=1, join="inner")
cells.index.name = "barcode"
print("Cells matched:", len(cells))

# 1. Predicted cell type vs reference label
crosstab = pd.crosstab(cells["reference"], cells["cell_type"])
print("\nReference (rows) vs predicted cell type (columns):")
with pd.option_context("display.width", 250, "display.max_columns", 50):
    print(crosstab.loc[:, crosstab.sum() >= 10])
print("\nPredicted types with < 10 cells:", crosstab.loc[:, crosstab.sum() < 10].sum().to_dict())

# 2. What the other heads said
print("\nPredicted tissue_general:\n", cells["tissue_general"].value_counts().head(8))
print("\nPredicted tissue:\n", cells["tissue"].value_counts().head(8))
print("\nMedian cell-type probability by reference label:\n",
      cells.groupby("reference")["cell_type_prob"].median().round(2))

# 3. Do embeddings separate the reference cell types?
# For each cell, the share of its 15 nearest neighbours with the same reference label.
# Same measure on a standard Scanpy PCA (50 components) as a point of comparison.
def knn_agreement(x, labels, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    idx = nn.kneighbors(x, return_distance=False)[:, 1:]
    lab = labels.to_numpy()
    return pd.Series((lab[idx] == lab[:, None]).mean(axis=1), index=labels.index)

pca = adata.copy()
sc.pp.normalize_total(pca, target_sum=1e4)
sc.pp.log1p(pca)
sc.pp.highly_variable_genes(pca, n_top_genes=2000)
sc.pp.pca(pca, n_comps=50, mask_var="highly_variable")

labels = ref.loc[cells.index]
knn = pd.DataFrame({
    "bmfm_embedding": knn_agreement(emb.loc[cells.index].to_numpy(), labels),
    "pca_baseline": knn_agreement(pca[cells.index].obsm["X_pca"], labels),
    "reference": labels,
})
print("\nShare of 15 nearest neighbours with the same reference label (mean):")
print(knn.groupby("reference").mean().round(2))
print("All cells:", knn[["bmfm_embedding", "pca_baseline"]].mean().round(3).to_dict())

# 4. UMAP of the embeddings, coloured by reference label and by prediction
e = sc.AnnData(emb.loc[cells.index].to_numpy(), obs=cells[["reference", "cell_type"]].copy())
sc.pp.neighbors(e, use_rep="X")
sc.tl.umap(e, random_state=0)
sc.settings.figdir = res
sc.pl.umap(e, color=["reference", "cell_type"], wspace=0.6, show=False, save="_bmfm_embedding.png")

# Save CSVs for R
cells.to_csv(f"{res}/cells_predictions.csv")
emb.loc[cells.index].rename_axis("barcode").to_csv(f"{res}/embeddings.csv")
crosstab.to_csv(f"{res}/crosstab_reference_vs_predicted.csv")
knn.rename_axis("barcode").to_csv(f"{res}/knn_agreement.csv")
