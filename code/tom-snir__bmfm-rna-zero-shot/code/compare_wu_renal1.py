import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

out = "outputs/wu_renal1"
res = "results/wu_renal1"

adata = sc.read_h5ad("data/wu_renal1/renal1_raw_labeled.h5ad")
pred = pd.read_csv(f"{out}/predictions.csv", index_col=0)
emb = pd.read_csv(f"{out}/embeddings.csv", header=None, index_col=0).loc[adata.obs_names]
cells = adata.obs.join(pred)

# 1. Predicted cell type vs authors' cluster, with predictions grouped coarsely
def coarse(label):
    if "CD4" in label or "regulatory" in label:
        return "CD4 T"
    if "CD8" in label:
        return "CD8 T"
    if "natural killer" in label:
        return "NK"
    if label in ("T cell", "unknown"):
        return label
    return "other"

cells["predicted_group"] = cells["cell_type"].map(coarse)
print("Authors' cluster (rows) vs predicted cell type, grouped (columns):")
print(pd.crosstab(cells["cluster"], cells["predicted_group"]))
print("\n'other' predictions:", cells.loc[cells["predicted_group"] == "other", "cell_type"].value_counts().head(8).to_dict())

# 2. Predicted tissue vs actual source
print("\nPredicted tissue_general (rows) vs actual source (columns):")
tis = pd.crosstab(cells["tissue_general"], cells["source"])
print(tis.loc[tis.sum(axis=1).sort_values(ascending=False).index].head(10))

# 3. Do cells of the same expanded clone sit close together in the embedding?
# For each cell in a clone of 2+ cells: share of its 15 nearest neighbours in the same clone.
def neighbours(x, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    return nn.kneighbors(x, return_distance=False)[:, 1:]

def same_clone_share(idx, clone):
    c = clone.to_numpy()
    share = (c[idx] == c[:, None]).mean(axis=1)
    return pd.Series(share, index=clone.index)

pca = adata.copy()
sc.pp.normalize_total(pca, target_sum=1e4)
sc.pp.log1p(pca)
sc.pp.highly_variable_genes(pca, n_top_genes=2000)
sc.pp.pca(pca, n_comps=50, mask_var="highly_variable")

idx_emb = neighbours(emb.to_numpy())
idx_pca = neighbours(pca.obsm["X_pca"])

# Cells without a TCR get a unique dummy ID, so they never count as a match
clone = cells["clonotype"].astype(object).fillna(pd.Series(cells.index, index=cells.index))
expanded = cells["clone_size"] >= 2
size_group = pd.cut(cells["clone_size"], [1, 4, 19, 10000], labels=["2-4", "5-19", "20+"])

# Chance levels: the expected share if the 15 neighbours were random cells
# (a) from the whole patient, (b) from the cell's own authors' cluster
n = len(cells)
chance_all = (cells["clone_size"] - 1) / (n - 1)
cluster = cells["cluster"].astype(str)
pair = cluster + "|" + clone.astype(str)
same_in_cluster = pair.map(pair.value_counts())
cluster_size = cluster.map(cluster.value_counts())
chance_cluster = (same_in_cluster - 1) / (cluster_size - 1)

# (c) from the cell's own cluster and same tissue source (each source is a separate 10x sample)
group = cluster + "|" + cells["source"].astype(str)
same_in_group = (group + "|" + clone.astype(str)).pipe(lambda s: s.map(s.value_counts()))
group_size = group.map(group.value_counts())
chance_cluster_source = (same_in_group - 1) / (group_size - 1)

scores = pd.DataFrame({
    "bmfm_embedding": same_clone_share(idx_emb, clone),
    "pca_baseline": same_clone_share(idx_pca, clone),
    "chance_whole_patient": chance_all,
    "chance_same_cluster": chance_cluster,
    "chance_cluster_and_source": chance_cluster_source,
    "clone_size": size_group,
})[expanded]
print("\nShare of 15 nearest neighbours from the same clone (mean), cells in clones of 2+:")
with pd.option_context("display.width", 200, "display.max_columns", 10):
    print(scores.groupby("clone_size", observed=True).mean().round(3))
print("All expanded cells:", scores.drop(columns="clone_size").mean().round(3).to_dict(),
      "n =", len(scores))

# 4. UMAP of the embedding: tissue, authors' cluster, 10 largest clones
top = cells.loc[expanded, "clonotype"].value_counts().head(10).index
cells["top_clone"] = cells["clonotype"].astype(object).where(cells["clonotype"].isin(top))
e = sc.AnnData(emb.to_numpy(), obs=cells[["source", "cluster", "top_clone"]].astype("category"))
sc.pp.neighbors(e, use_rep="X")
sc.tl.umap(e, random_state=0)
sc.settings.figdir = res
sc.pl.umap(e, color=["source", "cluster", "top_clone"], wspace=0.5, na_color="lightgrey",
           show=False, save="_renal1_bmfm_embedding.png")
