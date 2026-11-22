import pandas as pd, scipy.io, anndata as ad
d = "data/processed/lupus/"
X = scipy.io.mmread(d + "counts.mtx").T.tocsr()
obs = pd.read_csv(d + "obs.csv", index_col=0)
var = pd.DataFrame(index=open(d + "genes.txt").read().split())
a = ad.AnnData(X=X, obs=obs, var=var)
a.obs["dataset"] = "GSE189050"
a.write_h5ad(d + "lupus_raw.h5ad", compression="gzip")
print(a)
n = a.obs.groupby(["subject_id", "ancestry"]).size().reset_index(name="cells")
print("\nDonors < 500 cells by ancestry:\n", n[n.cells < 500].groupby("ancestry").size())
print("\nDonors kept (>= 500) by ancestry:\n", n[n.cells >= 500].groupby("ancestry").size())
