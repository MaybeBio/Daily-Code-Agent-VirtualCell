import numpy as np
import pandas as pd
import pytest

from src.core.registry import SPLITTERS
from src.data.splits import audit_leakage


def _obs(n=600, n_donors=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "donor_id": rng.choice([f"D{i}" for i in range(n_donors)], n),
        "batch_id": rng.choice([f"B{i}" for i in range(3)], n),
        "study_id": rng.choice(["S1", "S2", "S3"], n),
        "cell_type": rng.choice(["T", "B", "NK", "Mono"], n),
    })


def test_donor_split_has_no_donor_overlap():
    obs = _obs()
    s = SPLITTERS.build({"name": "donor_held_out", "val": 0.2, "test": 0.2}).split(obs)
    assert audit_leakage(obs, s).clean
    seen = [set(obs["donor_id"].to_numpy()[v]) for v in s.values()]
    assert seen[0].isdisjoint(seen[1]) and seen[0].isdisjoint(seen[2])


def test_random_split_is_flagged_as_leaky():
    obs = _obs()
    s = SPLITTERS.build({"name": "random", "silence_warning": True}).split(obs)
    rep = audit_leakage(obs, s)
    assert not rep.clean
    assert "donor_id" in str(rep)


def test_splits_partition_every_cell_exactly_once():
    obs = _obs()
    s = SPLITTERS.build({"name": "donor_held_out"}).split(obs)
    allidx = np.concatenate(list(s.values()))
    assert np.array_equal(np.sort(allidx), np.arange(len(obs)))


def test_refuses_when_group_column_is_constant():
    obs = _obs()
    obs["donor_id"] = "only_one"
    with pytest.raises(ValueError, match="single value"):
        SPLITTERS.build({"name": "donor_held_out"}).split(obs)


def test_allows_constant_group_when_explicitly_acknowledged():
    obs = _obs()
    obs["donor_id"] = "only_one"
    sp = SPLITTERS.build(
        {"name": "donor_held_out", "allow_fabricated": True, "min_groups": 1}
    )
    with pytest.raises(ValueError):
        sp.split(obs)   # still cannot make 3 splits from 1 group


def test_too_few_groups_raises():
    obs = _obs(n_donors=2)
    with pytest.raises(ValueError, match=">= 3"):
        SPLITTERS.build({"name": "donor_held_out"}).split(obs)


def test_split_is_deterministic_given_seed():
    obs = _obs()
    a = SPLITTERS.build({"name": "donor_held_out", "seed": 7}).split(obs)
    b = SPLITTERS.build({"name": "donor_held_out", "seed": 7}).split(obs)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_different_seeds_hold_out_different_donors():
    """Repeated evaluation needs the seed to resample held-out groups."""
    obs = _obs(n=1200, n_donors=12)
    held = set()
    for seed in range(6):
        s = SPLITTERS.build({"name": "donor_held_out", "seed": seed}).split(obs)
        assert audit_leakage(obs, s).clean
        held.add(frozenset(obs["donor_id"].to_numpy()[s["test"]]))
    assert len(held) > 1
