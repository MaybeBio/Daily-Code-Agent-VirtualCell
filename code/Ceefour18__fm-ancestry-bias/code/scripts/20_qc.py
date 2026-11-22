import scanpy as sc, pandas as pd, numpy as np

FILES = {"randolph": "data/processed/randolph_raw.h5ad",
         "prostate": "data/processed/prostate_raw.h5ad",
         "lupus":    "data/processed/lupus/lupus_raw.h5ad"}
MIN_G, MAX_G, MAX_MT, MIN_CELLS = 200, 6000, 15, 500
report = []

def tally(a, step, name):
    for anc, d in a.obs.groupby("ancestry", observed=True):
        report.append(dict(dataset=name, step=step, ancestry=anc,
                           cells=len(d), donors=d.subject_id.nunique()))

for name, f in FILES.items():
    print(f"\n===== {name} =====")
    a = sc.read_h5ad(f)
    a.var_names_make_unique()
    tally(a, "0_raw", name)

    if "gene_ids" in a.var:                                   # drop non-human (flu) genes
        human = a.var.gene_ids.astype(str).str.startswith("ENSG")
        print("non-human genes removed:", (~human).sum())
        a = a[:, human].copy()

    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    if name == "prostate":                                    # authors' doublets not removed
        sc.pp.scrublet(a, batch_key="subject_id")
        print("doublets flagged:", int(a.obs.predicted_doublet.sum()))
        a = a[~a.obs.predicted_doublet].copy()
        tally(a, "1_doublets", name)

    if "cell_type_coarse" in a.obs:                           # Randolph: drop unlabelled state
        a = a[a.obs.cell_type_coarse != "exclude"].copy()

    keep = (a.obs.n_genes_by_counts.between(MIN_G, MAX_G)) & (a.obs.pct_counts_mt < MAX_MT)
    a = a[keep].copy()
    tally(a, "2_cell_qc", name)

    n = a.obs.subject_id.value_counts()
    a.obs["donor_n_cells"] = a.obs.subject_id.map(n).values
    a.write_h5ad(f"data/processed/{name}_qc_all_donors.h5ad", compression="gzip")   # for 200-cell sensitivity
    a = a[a.obs.donor_n_cells >= MIN_CELLS].copy()
    tally(a, "3_donor_qc", name)
    a.write_h5ad(f"data/processed/{name}_qc.h5ad", compression="gzip")
    print(a)

r = pd.DataFrame(report)
r.to_csv("results/step0/qc_by_ancestry.csv", index=False)
w = r.pivot_table(index=["dataset", "ancestry"], columns="step", values=["cells", "donors"])
print("\n", w.to_string())
