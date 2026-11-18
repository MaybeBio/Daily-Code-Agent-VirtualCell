import numpy as np
import pytest

from src.core.registry import TOKENIZERS


@pytest.fixture
def cell():
    rng = np.random.default_rng(0)
    v = rng.poisson(2.0, 50).astype(np.float32)
    v[:10] = 0
    return v, np.arange(50, dtype=np.int64) + 3


def test_continuous_preserves_values(cell):
    v, g = cell
    out = TOKENIZERS.build({"name": "continuous", "max_genes": 100})(v, g)
    assert set(out["values"]) <= set(v[v > 0])
    assert out["value_bins"] is None


def test_binned_respects_bin_range(cell):
    v, g = cell
    tk = TOKENIZERS.build({"name": "binned", "n_bins": 10, "max_genes": 100})
    out = tk(v, g)
    assert out["value_bins"].min() >= 1          # 0 reserved for masked/zero
    assert out["value_bins"].max() <= 10
    assert tk.n_value_bins == 11


def test_binned_is_invariant_to_depth_scaling(cell):
    """Per-cell equal-frequency bins should not move when depth doubles."""
    v, g = cell
    tk = TOKENIZERS.build({"name": "binned", "n_bins": 10, "max_genes": 100})
    assert np.array_equal(tk(v, g)["value_bins"], tk(v * 2.0, g)["value_bins"])


def test_rank_orders_by_normalized_expression(cell):
    v, g = cell
    tk = TOKENIZERS.build({"name": "rank", "max_genes": 100})
    out = tk(v, g)
    assert np.all(np.diff(out["values"]) <= 0)   # descending
    assert out["gene_ids"][0] == g[np.argmax(v)]


def test_rank_refuses_to_refit():
    X = np.random.default_rng(0).poisson(1.0, (20, 10)).astype(np.float32)
    tk = TOKENIZERS.build({"name": "rank"})
    tk.fit(X)
    with pytest.raises(RuntimeError, match="already fitted"):
        tk.fit(X)


def test_tokenizers_cap_at_max_genes(cell):
    v, g = cell
    for name in ("continuous", "binned", "rank"):
        out = TOKENIZERS.build({"name": name, "max_genes": 7})(v, g)
        assert out["gene_ids"].size <= 7


def test_all_zero_cell_does_not_crash():
    v = np.zeros(20, dtype=np.float32)
    g = np.arange(20, dtype=np.int64) + 3
    for name in ("continuous", "binned", "rank"):
        out = TOKENIZERS.build({"name": name})(v, g)
        assert out["gene_ids"].size == 0


def test_binned_ties_share_a_bin():
    """Equal expression values must never land in different bins."""
    v = np.array([1, 1, 1, 1, 1, 1, 2, 2, 3, 7], dtype=np.float32)
    g = np.arange(10, dtype=np.int64) + 3
    out = TOKENIZERS.build({"name": "binned", "n_bins": 5})(v, g)
    for val in np.unique(v):
        assert np.unique(out["value_bins"][v == val]).size == 1
    assert np.all(np.diff(out["value_bins"][np.argsort(v, kind="stable")]) >= 0)
