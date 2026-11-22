import json, numpy as np, pandas as pd, scanpy as sc

SEEDS = [2026, 2027, 2028, 2029, 2030]        # 2026 = primary; others = robustness
N_FOLDS = 5
PLAN = {
    "randolph": {"strata": ["ancestry"],                   "test": {"AA": 20, "EA": 20}},
    "prostate": {"strata": ["ancestry", "classification"], "test_frac": 0.5},
}
out = {}
for name, p in PLAN.items():
    obs = sc.read_h5ad(f"data/processed/{name}_labelled.h5ad", backed="r").obs
    don = obs.groupby("subject_id", observed=True)[p["strata"]].first().reset_index()
    out[name] = {"fixed": {}, "cv": {}}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # (a) fixed train/test split - used for fine-tuning (H2)
        test = []
        for key, g in don.groupby(p["strata"], observed=True):
            ids = g.subject_id.tolist(); rng.shuffle(ids)
            anc = key[0] if isinstance(key, tuple) else key
            n = p["test"][anc] if "test" in p else int(round(len(ids) * p["test_frac"]))
            test += ids[:n]
        out[name]["fixed"][str(seed)] = {"train": sorted(set(don.subject_id) - set(test)), "test": sorted(test)}
        # (b) 5-fold CV - used for zero-shot (H1): every donor tested once
        fold = {}
        for key, g in don.groupby(p["strata"], observed=True):
            ids = g.subject_id.tolist(); rng.shuffle(ids)
            for i, d in enumerate(ids): fold[d] = i % N_FOLDS
        out[name]["cv"][str(seed)] = {str(k): sorted([d for d, f in fold.items() if f == k]) for k in range(N_FOLDS)}

    s = str(SEEDS[0])
    f = pd.Series({d: k for k, ds in out[name]["cv"][s].items() for d in ds}, name="fold")
    d = don.set_index("subject_id").join(f)
    print(f"\n===== {name} - CV folds (seed {s}): donors per fold x ancestry =====")
    print(pd.crosstab(d.fold, d.ancestry))
    print("Fixed split (H2): test", len(out[name]["fixed"][s]["test"]), "| train", len(out[name]["fixed"][s]["train"]))

json.dump(out, open("results/splits.json", "w"), indent=1)
print("\nSaved results/splits.json  (keys: dataset -> fixed / cv -> seed)")
