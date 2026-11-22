import sys, numpy as np, pandas as pd, scanpy as sc
sc.settings.figdir = "results/labels"; sc.settings.autoshow = False
import os; os.makedirs("results/labels", exist_ok=True)

COARSE = {"CD4_T": ["CD3E","CD3D","CD4","IL7R"], "CD8_T": ["CD3E","CD8A","CD8B"],
          "NK": ["KLRF1","NCAM1","FCER1G","TYROBP","KLRD1"], "B": ["MS4A1","CD79A","CD79B","BANK1"],
          "Plasma": ["JCHAIN","MZB1","TXNDC5"], "Monocyte": ["CD14","LYZ","S100A8","FCGR3A","MS4A7"],
          "cDC": ["FCER1A","CD1C","CLEC10A"], "pDC": ["LILRA4","IL3RA","CLEC4C"],
          "Platelet": ["PPBP","PF4"], "Erythrocyte": ["HBB","HBA1"]}
FINE = {"CD4_T": {"CD4_naive": ["CCR7","SELL","LEF1","TCF7"], "CD4_memory": ["S100A4","ANXA1","GPR183","AQP3"],
                  "Treg": ["FOXP3","IL2RA","CTLA4"]},
        "CD8_T": {"CD8_naive": ["CCR7","SELL","LEF1"], "CD8_effector_memory": ["GZMB","GZMH","PRF1","GZMK","NKG7"],
                  "MAIT": ["SLC4A10","KLRB1","TRAV1-2"]},
        "B": {"B_naive": ["IGHD","TCL1A","FCER2"], "B_memory": ["CD27","IGHG1","IGHA1","AIM2"]},
        "Monocyte": {"Mono_CD14": ["CD14","VCAN","S100A8","LYZ"], "Mono_CD16": ["FCGR3A","MS4A7","CDKN1C"]},
        "NK": {"NK_bright": ["NCAM1","SELL","GZMK","XCL1"], "NK_dim": ["FCGR3A","PRF1","GZMB","FGFBP2"]}}
AUTHOR_TO_COARSE = {"CD4_T": "CD4_T", "CD8_T": "CD8_T", "B": "B", "monocytes": "Monocyte", "NK": "NK",
                    "DC": "DC", "neutrophils": "Other", "NKT": "Other"}
HARMONISE = {"cDC": "DC", "pDC": "DC", "Plasma": "B", "Platelet": "Other", "Erythrocyte": "Other"}

def prep(a, batch):
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3", layer="counts", batch_key=None)
    sc.pp.pca(a, n_comps=30, mask_var="highly_variable")
    import harmonypy
    ho = harmonypy.run_harmony(a.obsm["X_pca"], a.obs, [batch], max_iter_harmony=20, verbose=False)
    Z = np.asarray(ho.Z_corr)
    a.obsm["X_pca_harmony"] = np.ascontiguousarray(Z if Z.shape[0] == a.n_obs else Z.T, dtype=np.float32)
    sc.pp.neighbors(a, use_rep="X_pca_harmony")
    sc.tl.leiden(a, resolution=1.0, flavor="igraph", n_iterations=2)

def assign(a, markers, tag):
    cols = []
    for lab, genes in markers.items():
        g = [x for x in genes if x in a.var_names]
        if g: sc.tl.score_genes(a, g, score_name=f"s_{lab}"); cols.append(lab)
    m = a.obs.groupby("leiden", observed=True)[[f"s_{c}" for c in cols]].mean()
    m.columns = cols
    mz = (m - m.mean()) / m.std(ddof=0).replace(0, 1)
    best = mz.idxmax(axis=1)
    m["assigned"] = best; m["n_cells"] = a.obs.leiden.value_counts()
    m.round(3).to_csv(f"results/labels/{tag}_cluster_scores.csv")
    return a.obs.leiden.map(best).astype(str)

name, path, batch = sys.argv[1], sys.argv[2], sys.argv[3]
a = sc.read_h5ad(path)
a.layers["counts"] = a.X.copy()
sc.pp.normalize_total(a, 1e4); sc.pp.log1p(a)

if "cell_type_coarse" in a.obs:                      # Randolph: authors' coarse labels
    a.obs["label_coarse"] = a.obs.cell_type_coarse.map(AUTHOR_TO_COARSE).fillna("Other")
else:                                                 # prostate: cluster + markers
    prep(a, batch)
    a.obs["label_coarse"] = assign(a, COARSE, f"{name}_coarse").replace(HARMONISE)
sc.pl.dotplot(a, {k: [g for g in v if g in a.var_names] for k, v in COARSE.items()},
              groupby="label_coarse", save=f"_{name}_coarse.png")

a.obs["label_fine"] = a.obs.label_coarse.astype(str)

def frac(s, g):            # fraction of cells per cluster expressing gene g
    if g not in s.var_names: return pd.Series(0.0, index=s.obs.leiden.cat.categories)
    x = (s[:, g].X > 0)
    x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
    return pd.Series(x, index=s.obs_names).groupby(s.obs.leiden, observed=True).mean()

def score(s, genes, nm):
    g = [x for x in genes if x in s.var_names]
    sc.tl.score_genes(s, g, score_name=nm)
    return s.obs.groupby("leiden", observed=True)[nm].mean()

def gate(s, ct):
    cl = s.obs.leiden.cat.categories
    st = pd.DataFrame(index=cl); lab = pd.Series(index=cl, dtype=object)
    if ct == "CD4_T":
        st["FOXP3"] = frac(s, "FOXP3")
        st["naive"] = score(s, FINE[ct]["CD4_naive"], "sn"); st["memory"] = score(s, FINE[ct]["CD4_memory"], "sm")
        lab[:] = np.where(st.FOXP3 >= .25, "Treg", np.where(st.naive > st.memory, "CD4_naive", "CD4_memory"))
    elif ct == "CD8_T":
        st["SLC4A10"] = frac(s, "SLC4A10"); st["TRAV1-2"] = frac(s, "TRAV1-2")
        st["naive"] = score(s, FINE[ct]["CD8_naive"], "sn"); st["eff"] = score(s, FINE[ct]["CD8_effector_memory"], "se")
        mait = (st.SLC4A10 >= .2) | (st["TRAV1-2"] >= .2)
        lab[:] = np.where(mait, "MAIT", np.where(st.naive > st.eff, "CD8_naive", "CD8_effector_memory"))
    elif ct == "B":
        st["naive"] = score(s, FINE[ct]["B_naive"], "sn"); st["memory"] = score(s, FINE[ct]["B_memory"], "sm")
        lab[:] = np.where(st.naive > st.memory, "B_naive", "B_memory")
    elif ct == "Monocyte":
        st["FCGR3A"] = frac(s, "FCGR3A"); st["CD14"] = frac(s, "CD14")
        lab[:] = np.where((st.FCGR3A >= .5) & (st.CD14 < .5), "Mono_CD16", "Mono_CD14")
    elif ct == "NK":
        st["FCGR3A"] = frac(s, "FCGR3A"); st["GZMK"] = frac(s, "GZMK"); st["XCL1"] = frac(s, "XCL1")
        lab[:] = np.where((st.FCGR3A < .3) & ((st.GZMK >= .3) | (st.XCL1 >= .3)), "NK_bright", "NK_dim")
    st["assigned"] = lab; st["n_cells"] = s.obs.leiden.value_counts()
    st.round(3).to_csv(f"results/labels/{name}_{ct}_gating.csv")
    return s.obs.leiden.map(lab).astype(str)

for ct in FINE:
    idx = a.obs.label_coarse == ct
    if idx.sum() < 500: continue
    s = a[idx].copy()
    prep(s, batch)
    a.obs.loc[idx, "label_fine"] = gate(s, ct).values
    sc.pl.dotplot(s, {k: [g for g in v if g in s.var_names] for k, v in FINE[ct].items()},
                  groupby="leiden", save=f"_{name}_{ct}_subclusters.png")

a.X = a.layers["counts"]; del a.layers["counts"]
for c in [c for c in a.obs if c.startswith("s_")] + ["leiden"]:
    a.obs.pop(c) if c in a.obs else None
a.write_h5ad(f"data/processed/{name}_labelled.h5ad", compression="gzip")

for lvl in ["label_coarse", "label_fine"]:
    t = pd.crosstab(a.obs[lvl], a.obs.ancestry)
    t["AA_%"] = (t["AA"] / t["AA"].sum() * 100).round(1); t["EA_%"] = (t["EA"] / t["EA"].sum() * 100).round(1)
    print(f"\n{name} {lvl} by ancestry:\n", t)
    t.to_csv(f"results/labels/{name}_{lvl}_by_ancestry.csv")
