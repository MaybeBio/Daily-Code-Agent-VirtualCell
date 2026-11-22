import scanpy as sc, pandas as pd
pd.set_option("display.width", 200)
for name, f in {"lupus": "data/processed/lupus/lupus_raw.h5ad",
                "prostate": "data/processed/prostate_raw.h5ad"}.items():
    a = sc.read_h5ad(f, backed="r")
    o = a.obs.copy()
    if "nFeature_RNA" in o:  g, mt = o.nFeature_RNA, o.percent_mt
    else:
        a = sc.read_h5ad(f); a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        o = a.obs; g, mt = o.n_genes_by_counts, o.pct_counts_mt
    o = o.assign(genes=g, mt=mt, low_genes=g < 200, high_mt=mt >= 15)
    print(f"\n===== {name}: genes per cell and % mito, by ancestry =====")
    print(o.groupby("ancestry", observed=True)[["genes", "mt"]].median().round(1))
    print("\n% cells failing each filter:")
    print((o.groupby("ancestry", observed=True)[["low_genes", "high_mt"]].mean() * 100).round(1))
    if "run" in o:
        print("\nLupus: donors per run x ancestry (batch confounding check):")
        print(o.groupby(["run", "ancestry"], observed=True).subject_id.nunique().unstack(fill_value=0))
        print("\nMedian genes per run:")
        print(o.groupby("run", observed=True).genes.median())
