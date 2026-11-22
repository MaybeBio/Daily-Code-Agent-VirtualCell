#!/usr/bin/env python
import argparse
from pathlib import Path
import cellxgene_census
import pandas as pd

MIN_DONORS, MIN_CELLS, TARGET_DONORS_PER_GROUP = 5, 200, 20

p = argparse.ArgumentParser()
p.add_argument("--outdir", default="results/step0")
p.add_argument("--census-version", default="stable")
args = p.parse_args()
out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

COLS = ["donor_id", "dataset_id", "assay", "sex", "development_stage",
        "self_reported_ethnicity", "cell_type"]

with cellxgene_census.open_soma(census_version=args.census_version) as c:
    print("Census build:", c["census_info"]["summary"].read().concat()
          .to_pandas().set_index("label").loc["census_build_date", "value"])
    obs = (c["census_data"]["homo_sapiens"].obs
           .read(value_filter="tissue_general == 'blood' and is_primary_data == True "
                              "and disease == 'normal'", column_names=COLS)
           .concat().to_pandas())

for col in COLS:
    obs[col] = obs[col].astype("category")
obs.to_parquet(out / "census_blood_obs.parquet")
print(f"\nCells: {len(obs):,} | donors: {obs.donor_id.nunique():,} | datasets: {obs.dataset_id.nunique()}")

eth = (obs.groupby("self_reported_ethnicity", observed=True)
       .agg(cells=("donor_id", "size"), donors=("donor_id", "nunique"))
       .sort_values("donors", ascending=False))
eth.to_csv(out / "ethnicity_raw_counts.csv")
print("\nEthnicity labels (donors / cells):\n", eth.to_string())

def group(lbl):
    l = str(lbl).lower()
    if "||" in l or "multiethnic" in l: return "mixed"
    if "african" in l: return "AFR"
    if "european" in l: return "EUR"
    if "unknown" in l or l == "na": return "unknown"
    return "other"

obs["anc"] = obs["self_reported_ethnicity"].astype(str).map(group)
print("\nDonors per group:\n", obs.groupby("anc").donor_id.nunique().to_string())

ds = (obs[obs.anc.isin(["AFR", "EUR"])]
      .groupby(["dataset_id", "anc"]).donor_id.nunique().unstack(fill_value=0))
for g in ["AFR", "EUR"]:
    if g not in ds: ds[g] = 0
ds["both"] = (ds.AFR > 0) & (ds.EUR > 0)
ds.sort_values("AFR", ascending=False).to_csv(out / "datasets_by_ancestry.csv")
print(f"\nDatasets with both AFR and EUR donors: {int(ds.both.sum())}")
print(ds[ds.both].sort_values("AFR", ascending=False).head(15).to_string())

sub = obs[obs.anc.isin(["AFR", "EUR"])]
ct = (sub.groupby(["cell_type", "anc"], observed=True)
      .agg(donors=("donor_id", "nunique"), cells=("donor_id", "size"))
      .unstack("anc", fill_value=0))
ct.columns = [f"{m}_{a}" for m, a in ct.columns]
for c_ in ["donors_AFR", "donors_EUR", "cells_AFR", "cells_EUR"]:
    if c_ not in ct: ct[c_] = 0
ct["passes"] = ((ct.donors_AFR >= MIN_DONORS) & (ct.donors_EUR >= MIN_DONORS) &
                (ct.cells_AFR >= MIN_CELLS) & (ct.cells_EUR >= MIN_CELLS))
ct = ct.sort_values("donors_AFR", ascending=False)
ct.to_csv(out / "gate_celltype_by_ancestry.csv")

n_afr = sub[sub.anc == "AFR"].donor_id.nunique()
n_eur = sub[sub.anc == "EUR"].donor_id.nunique()
n_pass = int(ct.passes.sum())
print(f"\nCell types passing: {n_pass} / {len(ct)}")
print(ct[ct.passes].to_string())

verdict = ("GO" if n_afr >= TARGET_DONORS_PER_GROUP and n_pass >= 8
           else "CONDITIONAL" if n_afr >= MIN_DONORS and n_pass >= 4 else "NO-GO")
summary = f"Census gate: AFR donors={n_afr}, EUR donors={n_eur}, cell types passing={n_pass} -> {verdict}"
print("\n" + summary)
(out / "census_gate_verdict.txt").write_text(summary + "\n")