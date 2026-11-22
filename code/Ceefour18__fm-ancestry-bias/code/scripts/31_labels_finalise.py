import scanpy as sc, pandas as pd
FINE_MAP = {"CD4_naive": "CD4_naive", "CD4_memory": "CD4_memory", "Treg": "CD4_memory",
            "CD8_naive": "CD8_T", "CD8_effector_memory": "CD8_T", "MAIT": "CD8_T", "CD8_T": "CD8_T",
            "B_naive": "B_naive", "B_memory": "B_memory",
            "Mono_CD14": "Monocyte", "Mono_CD16": "Monocyte", "Monocyte": "Monocyte",
            "NK_bright": "NK", "NK_dim": "NK", "NK": "NK", "DC": "DC", "Other": "Other"}
for name in ["randolph", "prostate"]:
    a = sc.read_h5ad(f"data/processed/{name}_labelled.h5ad")
    a.obs["label_fine_gated"] = a.obs.label_fine.astype(str)          # keep the raw gating result
    a.obs["label_fine"] = a.obs.label_fine.astype(str).map(FINE_MAP)
    assert a.obs.label_fine.notna().all(), "unmapped label - tell Claude"
    a = a[a.obs.label_coarse != "Other"].copy()                       # excluded from scoring
    a.write_h5ad(f"data/processed/{name}_labelled.h5ad", compression="gzip")
    for lvl in ["label_coarse", "label_fine"]:
        t = pd.crosstab(a.obs[lvl], a.obs.ancestry, normalize="columns").mul(100).round(1)
        t.to_csv(f"results/labels/{name}_{lvl}_final.csv")
        print(f"\n{name} {lvl} (% of cells):\n", t)
