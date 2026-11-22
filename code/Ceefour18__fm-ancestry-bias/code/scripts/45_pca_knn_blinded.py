import json, numpy as np, pandas as pd, scanpy as sc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

splits = json.load(open("results/splits.json"))
ct = pd.read_csv("results/metrics/celltypist_donor_f1_blinded.csv")
rows = []
for name in ["randolph", "prostate"]:
    a = sc.read_h5ad(f"data/processed/{name}_labelled.h5ad")
    if "condition" in a.obs: a = a[a.obs.condition == "NI"].copy()          # Randolph: mock
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    sc.pp.pca(a, n_comps=50, mask_var="highly_variable")
    X, y = a.obsm["X_pca"], a.obs.label_coarse.astype(str).values
    pred = pd.Series(index=a.obs_names, dtype=object)
    for k, fold_donors in splits[name]["cv"]["2026"].items():
        test = a.obs.subject_id.isin(fold_donors).values
        train = (~test) & (a.obs.ancestry == "EA").values                     # train on EA, other folds
        knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1).fit(X[train], y[train])
        pred[test] = knn.predict(X[test])
        print(f"{name} fold {k}: trained on {train.sum()} EA cells, predicted {test.sum()} cells")
    a.obs["pred"] = pred.values
    for d, g in a.obs.groupby("subject_id", observed=True):
        keep = g.label_coarse.value_counts().loc[lambda s: s >= 10].index
        g = g[g.label_coarse.isin(keep)]
        rows.append({"dataset": name, "subject_id": d,
                     "macro_f1": f1_score(g.label_coarse, g.pred, labels=list(keep), average="macro", zero_division=0)})

r = pd.DataFrame(rows)
r.to_csv("results/metrics/pca_knn_donor_f1_blinded.csv", index=False)        # no ancestry column
m = r.merge(ct, on=["dataset", "subject_id"], suffixes=("_pca", "_ct"))
m["diff"] = m.macro_f1_pca - m.macro_f1_ct
print("\nBLINDED (pooled across ancestries) - for power planning only")
print("PCA+kNN per-donor macro-F1:"); print(r.groupby("dataset").macro_f1.describe()[["count","mean","std"]].round(3))
print("\nWithin-donor difference PCA - CellTypist (donors scored by both):")
print(m.groupby("dataset")["diff"].describe()[["count","mean","std"]].round(3))
print("\nCorrelation of donor scores between methods:")
print(m.groupby("dataset")[["macro_f1_pca","macro_f1_ct"]].corr().round(2))
