"""Dataset splitting, and the leakage audit that justifies it.

The single most common way a single-cell representation benchmark overstates
performance is splitting cells at random.  Cells from one donor, one 10x lane
and one dissociation batch share technical signal, so a random split puts
near-replicates of every test cell into the training set.  A model can then
score well by recognizing the batch rather than the biology.

This module makes the honest splits the easy ones, and makes the dishonest
one announce itself.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.base import BaseSplitter
from ..core.registry import register_splitter

log = logging.getLogger(__name__)

__all__ = [
    "RandomSplitter",
    "GroupHeldOutSplitter",
    "DonorHeldOutSplitter",
    "StudyHeldOutSplitter",
    "TissueHeldOutSplitter",
    "audit_leakage",
    "LeakageReport",
]


def _as_fractions(val: float, test: float) -> tuple[float, float, float]:
    if not (0 < val < 1) or not (0 < test < 1) or val + test >= 1:
        raise ValueError(f"invalid split fractions val={val}, test={test}")
    return 1.0 - val - test, val, test


@register_splitter("random")
class RandomSplitter(BaseSplitter):
    """Cell-level random split.

    Provided for one reason: to quantify how much the honest splits cost.
    The gap between this and ``donor_held_out`` on the same model is the
    portion of apparent performance that was batch memorization.  Using it as
    a headline number is a reporting error, and it logs a warning saying so.
    """

    group_key = None

    def __init__(self, val: float = 0.1, test: float = 0.1, seed: int = 0,
                 silence_warning: bool = False):
        self.val, self.test, self.seed = val, test, seed
        self.silence_warning = silence_warning

    def split(self, obs: pd.DataFrame) -> Dict[str, np.ndarray]:
        if not self.silence_warning:
            log.warning(
                "RandomSplitter: cells from the same donor/batch will appear "
                "in train and test. Report this only as a leakage control."
            )
        n = len(obs)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val, n_test = int(round(self.val * n)), int(round(self.test * n))
        return {
            "test": np.sort(perm[:n_test]),
            "val": np.sort(perm[n_test : n_test + n_val]),
            "train": np.sort(perm[n_test + n_val :]),
        }


@register_splitter("group_held_out")
class GroupHeldOutSplitter(BaseSplitter):
    """Hold out entire groups (donors, studies, lanes...).

    Groups are visited in a seeded random order and each is assigned to
    whichever split is furthest below its target fraction.  Realized fractions
    are therefore approximate -- whole groups cannot be divided -- and differ
    between seeds, since one donor often contributes many more cells than
    another.  Different seeds hold out different groups.

    If ``stratify_key`` is given, the assignment additionally tries to keep
    every label (e.g. cell type) present in every split, and raises if the
    test split would end up with fewer than ``min_labels_in_test`` labels.
    """

    def __init__(
        self,
        group_key: str = "donor_id",
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 0,
        stratify_key: Optional[str] = "cell_type",
        min_groups: int = 3,
        min_labels_in_test: int = 2,
        allow_fabricated: bool = False,
    ):
        self.group_key = group_key
        self.val, self.test, self.seed = val, test, seed
        self.stratify_key = stratify_key
        self.min_groups = min_groups
        self.min_labels_in_test = min_labels_in_test
        self.allow_fabricated = allow_fabricated

    def split(self, obs: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self.group_key not in obs.columns:
            raise KeyError(
                f"splitter needs obs column '{self.group_key}'; "
                f"available: {list(obs.columns)}"
            )
        groups = obs[self.group_key].astype(str).to_numpy()
        uniq = np.unique(groups)
        if uniq.size == 1 and not self.allow_fabricated:
            raise ValueError(
                f"obs['{self.group_key}'] has a single value "
                f"({uniq[0]!r}): this dataset carries no {self.group_key} "
                "structure, so a held-out split on it would be a random "
                "split wearing a rigorous name. Use a source that supplies "
                "real labels, or pass allow_fabricated=True knowing the "
                "result is not a generalization estimate."
            )
        if uniq.size < self.min_groups:
            raise ValueError(
                f"need >= {self.min_groups} distinct '{self.group_key}' "
                f"values to make train/val/test; found {uniq.size}."
            )

        sizes = {g: int((groups == g).sum()) for g in uniq}
        # Seeded random order: the seed decides WHICH groups are held out, so
        # repeating over seeds resamples held-out donors.  (Largest-first
        # ordering would return the same split for every seed, which makes
        # repeated evaluation -- the only honest way to see donor-level
        # variance -- impossible.)  The greedy deficit rule below still keeps
        # split sizes near the requested fractions.
        rng = np.random.default_rng(self.seed)
        order = [uniq[i] for i in rng.permutation(uniq.size)]

        train_f, val_f, test_f = _as_fractions(self.val, self.test)
        targets = {"train": train_f, "val": val_f, "test": test_f}
        assigned: Dict[str, list] = {"train": [], "val": [], "test": []}
        counts = {k: 0 for k in assigned}
        total = float(sum(sizes.values()))

        for g in order:
            deficits = {
                k: targets[k] - (counts[k] / total if total else 0) for k in assigned
            }
            # guarantee each split receives at least one group
            empty = [k for k in assigned if not assigned[k]]
            pick = max(empty, key=lambda k: targets[k]) if empty else max(
                deficits, key=deficits.get
            )
            assigned[pick].append(g)
            counts[pick] += sizes[g]

        idx = {
            k: np.sort(np.flatnonzero(np.isin(groups, v))) for k, v in assigned.items()
        }

        if self.stratify_key and self.stratify_key in obs.columns:
            labels = obs[self.stratify_key].astype(str).to_numpy()
            n_test_labels = np.unique(labels[idx["test"]]).size
            if n_test_labels < self.min_labels_in_test:
                raise ValueError(
                    f"test split contains only {n_test_labels} distinct "
                    f"'{self.stratify_key}' values. Held-out groups do not "
                    "cover the label space; increase test fraction or choose "
                    "a different group_key."
                )
            missing = set(np.unique(labels)) - set(np.unique(labels[idx["train"]]))
            if missing:
                log.warning(
                    "%d label(s) absent from train and present elsewhere: %s. "
                    "These are zero-shot for any supervised head.",
                    len(missing),
                    sorted(missing)[:5],
                )
        for k, v in idx.items():
            if v.size == 0:
                raise ValueError(f"split '{k}' is empty after group assignment")
        log.info(
            "group split on '%s': train=%d val=%d test=%d cells over "
            "%d/%d/%d groups",
            self.group_key, idx["train"].size, idx["val"].size, idx["test"].size,
            len(assigned["train"]), len(assigned["val"]), len(assigned["test"]),
        )
        return idx


@register_splitter("donor_held_out")
class DonorHeldOutSplitter(GroupHeldOutSplitter):
    """Default honest split: no donor appears in two splits."""

    def __init__(self, **kw):
        kw.setdefault("group_key", "donor_id")
        super().__init__(**kw)


@register_splitter("study_held_out")
class StudyHeldOutSplitter(GroupHeldOutSplitter):
    """Hardest standard split: generalization to an unseen dataset/protocol."""

    def __init__(self, **kw):
        kw.setdefault("group_key", "study_id")
        super().__init__(**kw)


@register_splitter("tissue_held_out")
class TissueHeldOutSplitter(GroupHeldOutSplitter):
    """Transfer to an unseen tissue context."""

    def __init__(self, **kw):
        kw.setdefault("group_key", "tissue")
        kw.setdefault("stratify_key", None)
        super().__init__(**kw)


class LeakageReport(dict):
    """Result of ``audit_leakage``; truthy when leakage was detected."""

    @property
    def clean(self) -> bool:
        return not self.get("violations")

    def __str__(self) -> str:
        lines = [f"Leakage audit ({'CLEAN' if self.clean else 'LEAKAGE DETECTED'})"]
        for k, v in self.get("overlap", {}).items():
            lines.append(f"  {k:12s} shared across splits: {v}")
        for v in self.get("violations", []):
            lines.append(f"  !! {v}")
        for w in self.get("warnings", []):
            lines.append(f"  ~  {w}")
        return "\n".join(lines)


def audit_leakage(
    obs: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    keys: Sequence[str] = ("donor_id", "batch_id", "study_id"),
    strict_keys: Sequence[str] = ("donor_id",),
) -> LeakageReport:
    """Check that grouping variables do not straddle splits.

    Run unconditionally before training; the cost is milliseconds and the
    failure it catches invalidates every number produced afterwards.
    """
    rep = LeakageReport(overlap={}, violations=[], warnings=[], sizes={})
    for name, idx in splits.items():
        rep["sizes"][name] = int(np.asarray(idx).size)

    all_idx = np.concatenate([np.asarray(v) for v in splits.values()])
    if all_idx.size != np.unique(all_idx).size:
        rep["violations"].append(
            "the same cell index appears in more than one split"
        )

    for key in keys:
        if key not in obs.columns:
            rep["warnings"].append(f"obs has no column '{key}'; not audited")
            continue
        vals = obs[key].astype(str).to_numpy()
        per_split = {k: set(np.unique(vals[np.asarray(v)])) for k, v in splits.items()}
        shared = set()
        names = list(per_split)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared |= per_split[names[i]] & per_split[names[j]]
        rep["overlap"][key] = len(shared)
        if shared:
            msg = (
                f"{len(shared)} '{key}' value(s) appear in multiple splits "
                f"(e.g. {sorted(shared)[:3]})"
            )
            (rep["violations"] if key in strict_keys else rep["warnings"]).append(msg)

    if "cell_type" in obs.columns:
        lab = obs["cell_type"].astype(str).to_numpy()
        for name, idx in splits.items():
            if np.asarray(idx).size and np.unique(lab[np.asarray(idx)]).size < 2:
                rep["warnings"].append(
                    f"split '{name}' contains fewer than 2 cell types"
                )
    return rep
