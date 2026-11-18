"""The simulator must reproduce the statistics that make real data hard."""
import numpy as np
import pytest

from src.data.synthetic import make_synthetic


@pytest.fixture(scope="module")
def ad():
    return make_synthetic(n_cells=1500, n_genes=2000, seed=0)


def test_counts_are_nonnegative_integers(ad):
    d = ad.X.data
    assert (d >= 0).all() and np.allclose(d, np.rint(d))


def test_sparsity_is_realistic(ad):
    sparsity = 1 - ad.X.nnz / np.prod(ad.shape)
    assert 0.85 < sparsity < 0.98, f"sparsity {sparsity:.2f} outside 10x range"


def test_counts_are_overdispersed_not_poisson(ad):
    X = ad.X
    m = np.asarray(X.mean(0)).ravel()
    v = np.asarray(X.multiply(X).mean(0)).ravel() - m ** 2
    ok = m > 0.1
    # Poisson implies var == mean; real UMI data has var > mean
    assert np.median(v[ok] / m[ok]) > 1.2


def test_library_sizes_span_an_order_of_magnitude(ad):
    lib = np.asarray(ad.X.sum(1)).ravel()
    assert lib.max() / max(lib.min(), 1) > 5


def test_batch_effect_is_present_by_default(ad):
    from src.eval.metrics import pcr_batch
    from src.data.preprocessing import normalize_counts
    Xn, _ = normalize_counts(ad.X)
    Z = np.asarray(Xn[:, :60].todense())
    assert pcr_batch(Z, ad.obs["batch_id"]) > 0.02


def test_batch_effect_can_be_switched_off():
    """Required for metric validation: with no batch effect a batch metric
    must read ~0, otherwise the metric is measuring something else."""
    from src.eval.metrics import pcr_batch
    from src.data.preprocessing import normalize_counts
    a = make_synthetic(n_cells=800, n_genes=400, batch_effect=0.0,
                       donor_effect=0.0, batch_library_effect=0.0,
                       composition_shift=0.0, seed=1)
    Xn, _ = normalize_counts(a.X)
    Z = np.asarray(Xn[:, :60].todense())
    b = make_synthetic(n_cells=800, n_genes=400, batch_effect=1.2, seed=1)
    Xn2, _ = normalize_counts(b.X)
    Z2 = np.asarray(Xn2[:, :60].todense())
    assert pcr_batch(Z, a.obs["batch_id"]) < pcr_batch(Z2, b.obs["batch_id"])


def test_ground_truth_layers_present(ad):
    assert "truth" in ad.layers and "latent" in ad.obsm
    assert ad.uns["synthetic"] is True


def test_doublets_are_labelled_and_deeper(ad):
    lib = ad.obs["observed_library_size"].to_numpy()
    d = ad.obs["is_doublet"].to_numpy()
    assert d.sum() > 0
    assert lib[d].mean() > lib[~d].mean()


def test_required_obs_columns_exist(ad):
    from src.data.preprocessing import REQUIRED_OBS
    assert set(REQUIRED_OBS) <= set(ad.obs.columns)


def test_seed_is_reproducible():
    a = make_synthetic(n_cells=200, n_genes=100, seed=5)
    b = make_synthetic(n_cells=200, n_genes=100, seed=5)
    assert (a.X != b.X).nnz == 0
