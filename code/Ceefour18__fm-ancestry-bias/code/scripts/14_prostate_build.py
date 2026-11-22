import glob, re
import pandas as pd, scanpy as sc, anndata as ad
don = pd.read_csv("results/step0/prostate_donor_cells.csv").set_index("title")
parts = []
for f in sorted(glob.glob("data/raw/GSE270387/*.h5")):
    t = re.sub(r"^GSM\d+_|_filtered_feature_bc_matrix\.h5$", "", f.split("/")[-1])
    a = sc.read_10x_h5(f); a.var_names_make_unique()
    a.obs_names = [f"{t}_{b}" for b in a.obs_names]
    a.obs["subject_id"] = t
    a.obs["ancestry"] = "AA" if "African" in don.loc[t, "race"] else "EA"
    a.obs["classification"] = don.loc[t, "group"] if pd.notna(don.loc[t, "group"]) else "NA"
    parts.append(a)
x = ad.concat(parts, join="outer", merge="same")
x.obs["dataset"] = "GSE270387"
x.write_h5ad("data/processed/prostate_raw.h5ad", compression="gzip")
print(x)
print(x.obs.groupby(["ancestry", "classification"], observed=True).subject_id.nunique())
