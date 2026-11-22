"""Evaluate fixed models on identical participant-held-out folds."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from dataset import get_validation_transform
from model import ViTBackboneNet
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

args = argparse.ArgumentParser()
args.add_argument("checkpoint")
checkpoint = args.parse_args().checkpoint
rows = pd.read_csv("data/participants.tsv", sep="\t").query("group in ['PD-MCI', 'PD-NC']").reset_index(drop=True)
assert rows.participant_id.is_unique and rows.age.notna().all() and set(rows.sex) <= {"M", "F"}
out = Path("results")
out.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTBackboneNet(checkpoint).eval().to(device)
transform = get_validation_transform()
features = []
fig, axes = plt.subplots((len(rows)+5)//6, 6, figsize=(12, 2*((len(rows)+5)//6)))
for ax in axes.flat:
    ax.axis("off")
for ax, participant in zip(axes.flat, rows.participant_id):
    path = f"aligned/{participant}_T1w.nii.gz"
    volume = nib.load(path).get_fdata()
    ax.imshow(volume[:, :, volume.shape[2]//2].T, cmap="gray", origin="lower",
              vmax=np.percentile(volume[volume > 0], 99))
    ax.set_title(participant, fontsize=8)
    image = transform({"image":path})["image"].unsqueeze(0).to(device)
    with torch.inference_mode():
        features.append(model(image).detach().cpu().numpy().ravel())
fig.tight_layout()
fig.savefig(out / "aligned-scans.png", dpi=100)
plt.close(fig)
X = np.array(features)
assert X.shape == (len(rows), 768) and np.isfinite(X).all()
y = (rows.group == "PD-MCI").to_numpy(dtype=int)
clinical = np.column_stack([rows.age, rows.sex == "M"])
names = ["Age and sex", "BrainIAC", "Age, sex and BrainIAC"]
def score_split(seed):
    scores = {name: np.full(len(y), np.nan) for name in names}
    folds = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y), 1):
        folds[test] = fold
        # Retain three scan components, fixed before examining labels or scores.
        pca = PCA(3, svd_solver="full").fit(X[train])
        z = pca.transform(X)
        for name, inputs in zip(names, [clinical, z, np.column_stack([clinical, z])]):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced"))
            clf.fit(inputs[train], y[train])
            scores[name][test] = clf.predict_proba(inputs[test])[:, 1]

    return scores, folds


scores, folds = score_split(42)
pred = rows.assign(fold=folds, mci=y, **scores)
pred.to_csv(out / "held-out-predictions.csv", index=False)
summary = {}
fig, ax = plt.subplots(figsize=(6, 5))
for name, p in scores.items():
    assert np.isfinite(p).all()
    summary[name] = {"AUROC": roc_auc_score(y, p),
                     "balanced_accuracy": balanced_accuracy_score(y, p >= 0.5)}
    RocCurveDisplay.from_predictions(y, p, name=name, ax=ax)
ax.plot([0,1],[0,1], "k--", lw=1)
ax.set_title("Cognitive impairment in Parkinson’s disease\n33 participants · held-out predictions")
fig.tight_layout()
fig.savefig(out / "roc.png", dpi=160)
(out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
print(device, summary)

# Quantify partition sensitivity with the same participants and fixed predictors.
runs = []
for seed in range(50):
    predictions = scores if seed == 42 else score_split(seed)[0]
    runs.append({"seed": seed, **{name: roc_auc_score(y, p) for name, p in predictions.items()}})
pd.DataFrame(runs).to_csv(out / "auc-by-split.csv", index=False)
