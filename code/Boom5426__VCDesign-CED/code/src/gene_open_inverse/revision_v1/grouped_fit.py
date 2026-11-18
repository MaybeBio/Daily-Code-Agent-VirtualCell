"""Ridge penalty selection with gene-grouped folds and a fold-local response basis.

Correction made during revision, after external review. The submitted
``four_context_v1.predictors.fit`` selects its
penalty with ``identity_folds`` over row indices. In a multi-context pool a gene owns
one row per training context that measured it, so rows of one gene can sit on both
sides of a fold. The frozen fit also builds the response basis from every row before
the folds are cut, so held-out responses shape the basis that held-out predictions are
reconstructed in.

``fit_grouped`` changes the selection only. Folds are cut over gene identities, and
inside each fold the basis is rebuilt from that fold's training rows alone. The final
predictor is then built exactly as the frozen one builds it (basis on every row, the
same weighted ridge path), so when the grouped selection lands on the frozen penalty
the two predictors are identical.
"""
from __future__ import annotations

import numpy as np

from ..candidate_effect_distillation_v1 import basis as bs
from ..candidate_effect_distillation_v1 import contract as ced
from ..four_context_v1 import predictors as pd


def row_identities(keys: np.ndarray) -> np.ndarray:
    """``context|identity`` keys to the identity of each row."""
    return np.asarray([key.split("|", 1)[1] for key in np.asarray(keys, dtype=str).tolist()],
                      dtype=str)


def grouped_folds(identities: np.ndarray, folds: int = ced.RIDGE_FOLDS,
                  seed: int = ced.RIDGE_FOLD_SEED) -> list[np.ndarray]:
    """Row index blocks such that every row of one identity lands in the same block.

    Identities are sorted before they are permuted, so the assignment depends on the
    set of identities and the seed, never on row order.
    """
    identities = np.asarray(identities, dtype=str)
    unique, inverse = np.unique(identities, return_inverse=True)
    if unique.size < folds:
        raise ValueError(f"{unique.size} identities cannot fill {folds} folds")
    order = np.random.default_rng(seed).permutation(unique.size)
    block_of_identity = np.empty(unique.size, dtype=np.int64)
    for block, part in enumerate(np.array_split(order, folds)):
        block_of_identity[part] = block
    block_of_row = block_of_identity[inverse]
    return [np.flatnonzero(block_of_row == block) for block in range(folds)]


def cross_validate_grouped(features: np.ndarray, responses: np.ndarray, keys: np.ndarray,
                           weights: np.ndarray, rank: int = ced.BASIS_RANK,
                           penalties: tuple[float, ...] = ced.RIDGE_PENALTIES) -> dict:
    """Held-out gene-space cosine per penalty, by gene-grouped fold, basis refit per fold.

    The selection criterion is the frozen one: the mean over held rows of the cosine
    between the reconstructed prediction and the row's consensus response.
    """
    features = np.asarray(features, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    identities = row_identities(keys)
    folds = grouped_folds(identities)
    cosine = {penalty: [] for penalty in penalties}
    squared = {penalty: [] for penalty in penalties}
    for held in folds:
        train = np.setdiff1d(np.arange(features.shape[0]), held)
        overlap = np.intersect1d(identities[train], identities[held])
        if overlap.size:
            raise AssertionError(f"{overlap.size} identities sit on both sides of a fold")
        basis = bs.randomized_basis(responses[train], rank=rank)
        path = pd.WeightedRidgePath(features[train], basis.project(responses[train]),
                                    weights[train])
        held_targets = basis.project(responses[held])
        for penalty in penalties:
            predicted = path.predict(features[held], penalty)
            reconstructed = basis.reconstruct(predicted)
            norm = (np.linalg.norm(reconstructed, axis=1)
                    * np.linalg.norm(responses[held], axis=1))
            cosine[penalty].append(np.einsum("ij,ij->i", reconstructed, responses[held])
                                   / np.maximum(norm, 1e-30))
            squared[penalty].append(((predicted - held_targets) ** 2).mean(axis=1))
    record = {
        "selection": "grouped",
        "fold_unit": "gene identity", "basis": "refit on each fold's training rows",
        "folds": len(folds), "fold_sizes": [int(part.size) for part in folds],
        "fold_identities": [int(np.unique(identities[part]).size) for part in folds],
        "held_out_cosine": {str(p): float(np.concatenate(v).mean()) for p, v in cosine.items()},
        "held_out_squared_error": {str(p): float(np.concatenate(v).mean())
                                   for p, v in squared.items()},
    }
    record["selected_penalty"] = float(max(penalties,
                                           key=lambda p: record["held_out_cosine"][str(p)]))
    record["selected_penalty_by_squared_error"] = float(
        min(penalties, key=lambda p: record["held_out_squared_error"][str(p)]))
    record["held_out_cosine_at_selected"] = record["held_out_cosine"][str(record["selected_penalty"])]
    return record


def fit_grouped(features: np.ndarray, responses: np.ndarray, present: np.ndarray,
                keys: np.ndarray, contexts: tuple[str, ...], rank: int = ced.BASIS_RANK,
                weights: np.ndarray | None = None) -> pd.EffectPredictor:
    """``four_context_v1.predictors.fit`` with the grouped, basis-refitting selection."""
    keep = np.flatnonzero(np.asarray(present, dtype=bool))
    if keep.size <= rank:
        raise ValueError(f"pool of {keep.size} usable rows cannot support a rank-{rank} basis")
    row_weights = np.ones(keep.size) if weights is None else np.asarray(weights)[keep]
    keys_kept = np.asarray(keys, dtype=str)[keep]
    record = cross_validate_grouped(features[keep], responses[keep], keys_kept, row_weights,
                                    rank=rank)
    penalty = float(record["selected_penalty"])
    basis = bs.randomized_basis(responses[keep], rank=rank)
    targets = basis.project(responses[keep])
    return pd.EffectPredictor(
        basis=basis, path=pd.WeightedRidgePath(features[keep], targets, row_weights),
        penalty=penalty, rows=int(keep.size),
        identities=int(np.unique(row_identities(keys_kept)).size), contexts=tuple(contexts),
        held_out_cosine=float(record["held_out_cosine_at_selected"]), cross_validation=record)
