import json, numpy as np, pandas as pd, scanpy as sc, celltypist
from celltypist import models
from sklearn.metrics import f1_score

models.download_models(model="Immune_All_Low.pkl", force_update=False)
model = models.Model.load("Immune_All_Low.pkl")
splits = json.load(open("results/splits.json"))

def to_coarse(lbl):
    l = lbl.lower()
    if "cd4" in l or "regulatory t" in l or "tcm/naive helper" in l or "tem/effector helper" in l: return "CD4_T"
    if "cd8" in l or "mait" in l or "gamma-delta" in l or "cytotoxic t" in l: return "CD8_T"
    if "nk" in l or "ilc" in l: return "NK"
    if "b cell" in l or "plasma" in l or "naive b" in l or "memory b" in l: return "B"
    if "monocyte" in l or "macrophage" in l: return "Monocyte"
    if "dc" in l or "dendritic" in l: return "DC"
    return "Other"

rows = []
for name in ["randolph", "prostate"]:
    a = sc.read_h5ad(f"data/processed/{name}_labelled.h5ad")
    a = a[a.obs.subject_id.isin(splits[name]["2026"]["test"])].copy()
    if "condition" in a.obs: a = a[a.obs.condition == "NI"].copy()       # Randolph: mock cells
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    pred = celltypist.annotate(a, model=model, majority_voting=False).predicted_labels
    a.obs["pred_raw"] = pred["predicted_labels"].astype(str).values
    a.obs["pred"] = a.obs.pred_raw.map(to_coarse)
    pd.crosstab(a.obs.pred_raw, a.obs.pred).to_csv(f"results/labels/{name}_celltypist_label_map.csv")
    for d, g in a.obs.groupby("subject_id", observed=True):
        keep = g.label_coarse.value_counts().loc[lambda s: s >= 10].index     # >= 10 cells rule
        g = g[g.label_coarse.isin(keep)]
        f1 = f1_score(g.label_coarse, g.pred, labels=list(keep), average="macro", zero_division=0)
        rows.append({"dataset": name, "subject_id": d, "n_cells": len(g), "macro_f1": f1})

r = pd.DataFrame(rows)
r.to_csv("results/metrics/celltypist_donor_f1_blinded.csv", index=False)   # no ancestry column
print("\nBLINDED summary (pooled across ancestries) - for power planning only:")
print(r.groupby("dataset").macro_f1.describe()[["count", "mean", "std", "min", "max"]].round(3))
