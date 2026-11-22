import glob, re
import pandas as pd
import scanpy as sc

don = pd.read_csv("data/raw/GSE270387/donors.tsv", sep="\t", header=None,
                  names=["title", "race", "group", "strategy"])
don["race"] = don.race.str.replace("race: ", "")
don["group"] = don.group.str.replace("lethal prostate_cancer_group: ", "")

rows = []
for f in sorted(glob.glob("data/raw/GSE270387/*.h5")):
    title = re.sub(r"^GSM\d+_|_filtered_feature_bc_matrix\.h5$", "", f.split("/")[-1])
    a = sc.read_10x_h5(f)
    rows.append({"title": title, "cells": a.n_obs, "genes": a.n_vars})

t = pd.DataFrame(rows).merge(don, on="title", how="left")
t.to_csv("results/step0/prostate_donor_cells.csv", index=False)
print(t.groupby(["race", "group"]).agg(donors=("title", "count"), cells=("cells", "sum"),
      median_cells=("cells", "median")).to_string())
print("\nUnmatched files:", t.race.isna().sum())
print("Total cells:", t.cells.sum())
