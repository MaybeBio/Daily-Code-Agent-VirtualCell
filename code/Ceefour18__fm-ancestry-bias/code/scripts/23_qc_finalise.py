import scanpy as sc, pandas as pd
# Rule: exclude donors with > 25% of cells failing cell-level QC (failed library)
raw = pd.read_csv("results/step0/prostate_donor_cells.csv").set_index("title").cells   # cells before QC
a = sc.read_h5ad("data/processed/prostate_qc_keep_all_donors.h5ad")
kept = a.obs.subject_id.value_counts()
fail = (1 - kept / raw.loc[kept.index]).sort_values(ascending=False)
print("Highest % of cells removed by QC per donor:"); print((fail.head(6) * 100).round(1))
bad = fail[fail > 0.25].index.tolist()
print("Excluded:", {d: a.obs.loc[a.obs.subject_id == d, "ancestry"].iloc[0] for d in bad})
a = a[~a.obs.subject_id.isin(bad)].copy()
a.write_h5ad("data/processed/prostate_qc.h5ad", compression="gzip")
print(a.obs.groupby("ancestry", observed=True).agg(donors=("subject_id", "nunique"), cells=("subject_id", "size")))
