import anndata as ad
import pandas as pd
import scanpy as sc

d = "data/wu_renal1/downloads"
samples = {
    "RT1": "GSM4143678_SAM24360638-rt1",
    "RN1": "GSM4143679_SAM24360640-rn1",
    "RB1": "GSM4143680_SAM24360639-rb1",
}

# Read each sample's 10x files (genes in rows, cells in columns), flip to cells x genes,
# and name cells like the metadata does: <sample>_<barcode>
parts = []
for name, prefix in samples.items():
    a = sc.read_mtx(f"{d}/{prefix}.matrix.mtx.gz").T
    genes = pd.read_csv(f"{d}/{prefix}.genes.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(f"{d}/{prefix}.barcodes.tsv.gz", sep="\t", header=None)
    a.var_names = genes[1].to_numpy()
    a.var["gene_id"] = genes[0].to_numpy()
    a.obs_names = name + "_" + barcodes[0]
    a.var_names_make_unique()
    parts.append(a)
adata = ad.concat(parts, merge="same")

# Authors' T-cell metadata: cluster, tissue source, clone ID (NA = no TCR found)
meta = pd.read_csv(f"{d}/GSE139555_tcell_metadata.txt.gz", sep="\t", index_col=0)
meta = meta[meta["patient"] == "Renal1"]

# Keep only the T cells the authors kept after quality control
adata = adata[meta.index].copy()
adata.obs["cluster"] = meta["ident"].astype(str)
adata.obs["source"] = meta["source"]
adata.obs["clonotype"] = meta["clonotype"]

# Clone size: how many cells of this patient share the clone ID
sizes = adata.obs["clonotype"].value_counts()
adata.obs["clone_size"] = adata.obs["clonotype"].map(sizes).fillna(0).astype(int)

print(adata)
print("Max count:", adata.X.max())
print(pd.crosstab(adata.obs["cluster"], adata.obs["source"]))
print("\nCells with a TCR:", adata.obs["clonotype"].notna().sum())
print("Clone size distribution (cells):")
print(pd.cut(adata.obs["clone_size"], [-1, 0, 1, 4, 19, 10000],
             labels=["no TCR", "1", "2-4", "5-19", "20+"]).value_counts().sort_index())

adata.write_h5ad("data/wu_renal1/renal1_raw_labeled.h5ad")
