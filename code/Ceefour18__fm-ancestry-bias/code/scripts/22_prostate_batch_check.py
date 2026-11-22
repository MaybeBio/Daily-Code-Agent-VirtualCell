import scanpy as sc, pandas as pd
pd.set_option("display.width", 200)
a = sc.read_h5ad("data/processed/prostate_raw.h5ad")
a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
o = a.obs.assign(low=a.obs.n_genes_by_counts < 200,
                 series=a.obs.subject_id.str[:6])          # SCAF23 vs SCAF27 etc.
print("Donors per series x ancestry:")
print(o.groupby(["series", "ancestry"], observed=True).subject_id.nunique().unstack(fill_value=0))
print("\nMedian genes per series:"); print(o.groupby("series", observed=True).n_genes_by_counts.median())
d = o.groupby(["subject_id", "ancestry", "series"], observed=True).agg(
        cells=("low", "size"), pct_low=("low", "mean"), med_genes=("n_genes_by_counts", "median"))
d["pct_low"] = (d.pct_low * 100).round(1)
print("\nWorst 12 donors by % low-gene cells:")
print(d.sort_values("pct_low", ascending=False).head(12))
