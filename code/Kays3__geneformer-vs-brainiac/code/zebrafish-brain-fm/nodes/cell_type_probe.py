"""Stage 1 node: how well do frozen features predict cell type on held-out samples?"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class CellTypeProbeNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="zfbrain_cell_type_probe",
        stage="analysis",
        tool="scikit-learn",
        model_source="https://github.com/Kays3/geneformer-vs-brainiac/tree/main/zebrafish-brain-fm",
        description="Fit a logistic regression from fixed cell features to cell type on the training cells and score it on the held-out samples. With permute_labels the training labels are shuffled first, as a control that must fall to chance.",
        parameters={
            "permute_labels": ParameterDefinition(default_value=False, description="Shuffle the training labels before fitting."),
            "seed": ParameterDefinition(default_value=0, description="Random seed of the shuffle."),
            "out_path": ParameterDefinition(default_value="runs/probe_predictions.npy", description="File for the predicted held-out labels."),
        },
        inputs={
            "features_path": PortDefinition(type=PortType.STR, description="Array of cells x features, in the cell order of the token file."),
            "tokens_path": PortDefinition(type=PortType.STR, description="Token file; gives 'cell_type' and 'is_heldout'."),
        },
        outputs={
            "metrics": PortDefinition(type=PortType.DICT, description="Macro F1, balanced accuracy and accuracy on held-out cells, the number of classes, chance level, and cell counts."),
            "predictions_path": PortDefinition(type=PortType.STR, description="Predicted class codes of the held-out cells, in order."),
        },
        methods={
            "probe": MethodDefinition(description="Fit on training cells, score on held-out cells.", inputs=["features_path", "tokens_path"], outputs=["metrics", "predictions_path"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("probe", self.probe, method_key="probe")

    def probe(self, features_path, tokens_path) -> Dict[str, Any]:
        p = self._parameters
        features = np.load(features_path)
        data = torch.load(tokens_path)
        labels, held = data["cell_type"].numpy(), data["is_heldout"].numpy()
        train_labels = labels[~held].copy()
        if bool(p["permute_labels"]):
            np.random.default_rng(int(p["seed"])).shuffle(train_labels)
        scaler = StandardScaler().fit(features[~held])
        model = LogisticRegression(max_iter=2000).fit(scaler.transform(features[~held]), train_labels)
        predicted = model.predict(scaler.transform(features[held]))
        truth = labels[held]
        classes = np.unique(truth)
        out = Path(p["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, predicted)
        metrics = {
            "macro_f1": round(float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)), 4),
            "accuracy": round(float(accuracy_score(truth, predicted)), 4),
            "classes_in_heldout": int(len(classes)),
            "chance": round(1.0 / len(classes), 4),
            "train_cells": int((~held).sum()),
            "heldout_cells": int(held.sum()),
            "permuted_labels": bool(p["permute_labels"]),
        }
        return {"metrics": metrics, "predictions_path": str(out)}
