#!/usr/bin/env python3
"""Each test names a failure that would otherwise look like a valid selection.

A fold that still splits one gene across train and held, so the "fixed" selection leaks
exactly as the submitted one did. A fold basis that is still fitted on every row, so
held-out responses shape the space their own prediction is reconstructed in. A selection
switch that changes the fitted predictor rather than only the penalty search.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_open_inverse.candidate_effect_distillation_v1 import contract as ced  # noqa: E402
from gene_open_inverse.candidate_effect_distillation_v1 import predictor as pr   # noqa: E402
from gene_open_inverse.four_context_v1 import predictors as pd                  # noqa: E402
from gene_open_inverse.revision_v1 import grouped_fit as gf                     # noqa: E402

RESULTS = []


class CheckFailed(AssertionError):
    """A protocol check that does not hold. It fails the test under pytest, and the
    standalone runner below reports it from ``RESULTS``."""


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    if not condition:
        raise CheckFailed(f"{name}: {detail}")


def pool(identities=90, contexts=("A", "B", "C"), genes=48, features=16, seed=0):
    """Most genes measured in two or three contexts, like the real atlas."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((identities, features))
    mixing = rng.standard_normal((features, genes))
    keys, feats, resp = [], [], []
    for index in range(identities):
        measured = [c for c in contexts if rng.random() < 0.75] or [contexts[0]]
        for context in measured:
            keys.append(f"{context}|G{index}")
            feats.append(latent[index] + 0.05 * rng.standard_normal(features))
            resp.append(latent[index] @ mixing + 0.1 * rng.standard_normal(genes))
    return (np.asarray(feats), np.asarray(resp), np.ones(len(keys), dtype=bool),
            np.asarray(keys, dtype=str))


def test_folds_never_split_a_gene():
    features, responses, present, keys = pool()
    identities = gf.row_identities(keys)
    folds = gf.grouped_folds(identities)
    split = []
    for held in folds:
        train = np.setdiff1d(np.arange(features.shape[0]), held)
        split.append(np.intersect1d(identities[train], identities[held]).size)
    covered = np.sort(np.concatenate(folds))
    leaky = 0
    for held in pr.identity_folds(features.shape[0]):      # the submitted row-index folds
        train = np.setdiff1d(np.arange(features.shape[0]), held)
        leaky += np.intersect1d(identities[train], identities[held]).size
    check("01_folds_never_split_a_gene",
          sum(split) == 0 and np.array_equal(covered, np.arange(features.shape[0])) and leaky > 0,
          f"grouped overlap {sum(split)}, submitted-style overlap {leaky}")


def test_folds_ignore_row_order():
    features, responses, present, keys = pool()
    identities = gf.row_identities(keys)
    order = np.random.default_rng(7).permutation(identities.size)
    first = {name: block for block, part in enumerate(gf.grouped_folds(identities))
             for name in np.unique(identities[part])}
    second = {name: block for block, part in enumerate(gf.grouped_folds(identities[order]))
              for name in np.unique(identities[order][part])}
    check("02_folds_ignore_row_order", first == second, f"{len(first)} identities")


def test_fold_basis_sees_training_rows_only():
    """The recorded selection must come from a basis that never saw the held rows."""
    features, responses, present, keys = pool()
    identities = gf.row_identities(keys)
    folds = gf.grouped_folds(identities)
    from gene_open_inverse.candidate_effect_distillation_v1 import basis as bs
    held = folds[0]
    train = np.setdiff1d(np.arange(features.shape[0]), held)
    fold_basis = bs.randomized_basis(responses[train], rank=8)
    whole_basis = bs.randomized_basis(responses, rank=8)
    spread = float(np.abs(np.abs(fold_basis.basis) - np.abs(whole_basis.basis)).max())
    record = gf.cross_validate_grouped(features, responses, keys, np.ones(features.shape[0]), rank=8)
    check("03_fold_basis_sees_training_rows_only",
          spread > 1e-8 and record["basis"] == "refit on each fold's training rows"
          and record["fold_unit"] == "gene identity"
          and all(n > 0 for n in record["fold_identities"]),
          f"basis difference {spread:.3g}, fold identities {record['fold_identities']}")


def test_selection_switch_changes_only_the_search():
    """At one penalty the two paths must give the same predictor, or a BU change would be
    blamed on the corrected selection when it came from a different fitted model."""
    features, responses, present, keys = pool()
    grouped = pd.fit(features, responses, present, keys, ("A", "B", "C"), rank=8)
    frozen = pd.fit(features, responses, present, keys, ("A", "B", "C"), rank=8,
                    selection="frozen")
    same_basis = float(np.abs(grouped.basis.basis - frozen.basis.basis).max())
    probe = features[:5]
    same_path = float(np.abs(grouped.path.predict(probe, 1.0)
                             - frozen.path.predict(probe, 1.0)).max())
    check("04_selection_switch_changes_only_the_search",
          same_basis == 0.0 and same_path == 0.0
          and grouped.cross_validation["fold_unit"] == "gene identity"
          and frozen.cross_validation["fold_unit"] == "row index (as submitted)"
          and set(grouped.cross_validation["held_out_cosine"]) == {str(p) for p in ced.RIDGE_PENALTIES}
          and grouped.cross_validation["selection"] == "grouped"
          and frozen.cross_validation["selection"] == "frozen"
          and {"selected_penalty", "selected_penalty_by_squared_error", "held_out_cosine_at_selected"}
          <= set(grouped.cross_validation) & set(frozen.cross_validation),
          f"basis diff {same_basis}, path diff {same_path}, "
          f"penalties grouped {grouped.penalty} frozen {frozen.penalty}")


def main():
    for test in (test_folds_never_split_a_gene, test_folds_ignore_row_order,
                 test_fold_basis_sees_training_rows_only,
                 test_selection_switch_changes_only_the_search):
        try:
            test()
        except CheckFailed:
            pass                                    # already recorded by check()
        except Exception as error:                  # a crash is a failure, not a skip
            RESULTS.append((test.__name__, False, f"raised {type(error).__name__}: {error}"))
    failed = [name for name, ok, _ in RESULTS if not ok]
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
