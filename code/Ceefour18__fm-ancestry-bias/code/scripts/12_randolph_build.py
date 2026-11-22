import glob, re
import pandas as pd, scipy.io, anndata as ad

raw = "data/raw/GSE162632/"
md = pd.read_csv("data/processed/randolph/cell_metadata.csv", index_col=0)
ind = pd.read_csv(raw + "zenodo/inputs/2_calculate_residuals_and_DE_analyses/"
                  "individual_meta_data_for_GE_with_scaledCovars_with_CTC.txt")
age = ind.drop_duplicates("indiv_ID").set_index("indiv_ID")["age"]

parts = []
for mtx in sorted(glob.glob(raw + "GSM*_matrix.mtx.gz")):
    pre = mtx.replace("_matrix.mtx.gz", "")
    cap = re.search(r"_(B\d+-c\d)-10X", mtx).group(1).replace("-", "_")
    X = scipy.io.mmread(mtx).T.tocsr()
    bc = pd.read_csv(pre + "_barcodes.tsv.gz", header=None, dtype=str)[0]
    ft = pd.read_csv(pre + "_features.tsv.gz", header=None, sep="\t",
                     dtype=str, keep_default_na=False)          # keep genes like "NA" as text
    obs = pd.DataFrame(index=pd.Index((cap + "_" + bc.str.replace("-1$", "", regex=True)).tolist()))
    var = pd.DataFrame({"gene_ids": ft[0].tolist(), "gene_symbol": ft[1].tolist()},
                       index=pd.Index(ft[1].tolist()))
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.var_names_make_unique()
    keep = a.obs_names.isin(md.index)
    print(f"{cap}: {a.n_obs} in GEO, {keep.sum()} matched")
    parts.append(a[keep].copy())

x = ad.concat(parts, join="outer", merge="same")
m = md.loc[x.obs_names]
x.obs["subject_id"] = m.SOC_indiv_ID.astype(str).values
x.obs["ancestry"] = m.SOC_genetic_ancestry.map({"AFR": "AA", "EUR": "EA"}).astype(str).values
x.obs["YRI"] = m.YRI.astype(float).values
x.obs["condition"] = m.SOC_infection_status.astype(str).values
x.obs["batch"] = m.batchID.astype(str).values
x.obs["celltype_author"] = m.celltype.astype(str).values
x.obs["cell_type_coarse"] = m.celltype.replace({"infected_monocytes": "monocytes",
    "NK_high_response": "NK", "highly_infected": "exclude"}).astype(str).values
x.obs["age"] = x.obs.subject_id.map(age).astype(float).values
x.obs["dataset"] = "GSE162632"
x.obs.index = x.obs.index.astype(str); x.obs.index.name = None
x.var.index = x.var.index.astype(str); x.var.index.name = None

x.write_h5ad("data/processed/randolph_raw.h5ad", compression="gzip")
print("\n", x)
print("\nMatched", x.n_obs, "of", len(md))
print(x.obs.groupby(["ancestry", "condition"], observed=True).subject_id.nunique())
print(x.obs.cell_type_coarse.value_counts())
