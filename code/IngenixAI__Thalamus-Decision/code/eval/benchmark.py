"""Thalamus-Decision on the evaluation subsets shipped in the model repository (``eval/``, the same files as in
Thalamus-Reasoning-9B), asked as typed decisions.

* ``human_diseases`` (geo_disease_subtype, geo_tissue) and ``tabula_sapiens`` (ts_geo_cell_type_top20, ts_geo_tissue):
  1,000 profiles per task. One ``choice`` question per profile over all candidate labels (the task's options file),
  state ``Expression profile: <omics>``. Macro AUROC (one-vs-rest, as Thalamus-Reasoning's eval) and accuracy.
* ``geo_qa_binary`` (geo_qa_binary, geo_qa_binary_minimal): 2 x 500 yes/no questions about GEO samples. Each GEO sample
  ID in the question becomes "profile k" and the state lists ``Profile k: <omics>``; one ``noul`` question. Accuracy.

The profile vectors are the released 20,029-value vectors (the format tr-embed writes), passed as explicit ``<omics>``
vectors.

    python eval/benchmark.py --benchmark tabula_sapiens            # model: $THALAMUS_DECISION_HOME or the Hub
    python eval/benchmark.py --benchmark geo_qa_binary --model /path/to/Thalamus-Decision-9B
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from thalamus_decision import ThalamusDecision
from thalamus_decision.hub import eval_path, model_path

CELL_TYPE_QUESTION = "Identify the cell type of this sample based on the above gene expression profile."
TISSUE_QUESTION = "Identify the tissue of origin for this sample based on the above gene expression profile."
CLASSIFICATION = {
    "human_diseases": {
        "geo_disease_subtype": "What is the disease subtype for this sample based on the above gene expression profile?",
        "geo_tissue": "What tissue of origin does the above gene expression profile suggest for this sample?",
    },
    "tabula_sapiens": {"ts_geo_cell_type_top20": CELL_TYPE_QUESTION, "ts_geo_tissue": TISSUE_QUESTION},
}
GEO_QA = ("geo_qa_binary", "geo_qa_binary_minimal")
GSM_RE = re.compile(r"(?<!\w)GSM\d+(?!\w)")


def macro_auroc(probs: np.ndarray, gold: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(gold == c, probs[:, c]) for c in range(probs.shape[1]) if 0 < (gold == c).sum() < len(gold)]
    return float(np.mean(aucs)) if aucs else 0.0


def classification(td, data: Path, tasks: dict, limit: int) -> dict:
    out = {}
    for name, question in tasks.items():
        options = sorted(json.loads((data / f"{name}_options.json").read_text()))
        labels = pd.read_parquet(data / f"{name}_labels.parquet")
        emb = np.load(data / f"{name}_embeddings.npy")
        n = min(limit, len(labels)) if limit else len(labels)
        reqs = [{"state": "Expression profile: <omics>", "vectors": [emb[i].tolist()], "questions": {"label": {
            "type": "choice", "instructions": question, "criteria": options}}} for i in range(n)]
        res = td.decide_batch(reqs)
        probs = np.array([[r["answers"]["label"]["probabilities"][o] for o in options] for r in res])
        gold = np.array([options.index(g) for g in labels.label.to_numpy()[:n]])
        out[name] = dict(n=n, options=len(options), macro_auroc=round(macro_auroc(probs, gold), 4),
                         accuracy=round(float((probs.argmax(1) == gold).mean()), 4))
        print(name, out[name], flush=True)
    return out


def geo_qa(td, data: Path, limit: int) -> dict:
    npz = np.load(data / "embeddings.npz")
    vec = dict(zip(npz["geo_accession"].astype(str), npz["embeddings"]))
    out = {}
    for name in GEO_QA:
        rows = json.loads((data / f"{name}.json").read_text())[: limit or None]
        reqs, gold = [], []
        for r in rows:
            ids = list(dict.fromkeys(GSM_RE.findall(r["question"])))
            k = {g: i + 1 for i, g in enumerate(ids)}
            question = GSM_RE.sub(lambda m: f"profile {k[m.group(0)]}", r["question"])
            reqs.append({"state": "\n".join(f"Profile {k[g]}: <omics>" for g in ids),
                         "vectors": [vec[g].tolist() for g in ids],
                         "questions": {"yn": {"type": "noul", "instructions": question}}})
            gold.append(r["answer"].strip().lower().startswith("yes"))
        res = td.decide_batch(reqs)
        pred = np.array([r["answers"]["yn"]["noul"] >= 0.5 for r in res])
        out[name] = dict(n=len(rows), accuracy=round(float((pred == np.array(gold)).mean()), 4))
        print(name, out[name], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=[*CLASSIFICATION, "geo_qa_binary", "all"])
    ap.add_argument("--model", default=None, help="local model folder or Hub repo id")
    ap.add_argument("--limit", type=int, default=0, help="first N items per task (default: all)")
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    a = ap.parse_args()
    folder = model_path(a.model)
    data = eval_path(a.model)  # not part of the server's snapshot: fetched here on first use
    td = ThalamusDecision.load(folder, depmap=False)
    names = [*CLASSIFICATION, "geo_qa_binary"] if a.benchmark == "all" else [a.benchmark]
    summary = {"model": td.engine.model_name}
    for b in names:
        if b in CLASSIFICATION:
            summary[b] = classification(td, data / b, CLASSIFICATION[b], a.limit)
        else:
            summary[b] = geo_qa(td, data / "geo_omics_qa", a.limit)
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
