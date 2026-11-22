"""Large-data acceleration must preserve SciPy's per-gene exact/tie rules."""
import unittest
import numpy as np
from scipy.stats import mannwhitneyu
from bio.metrics.perturbation import _mannwhitney_pvalues, _significant_by_fdr, _bh_qvalues


class VectorizedDES(unittest.TestCase):
    def test_matches_independent_scipy_tests_for_small_large_tied_and_constant_genes(self):
        rng = np.random.default_rng(41)
        for n, m in [(3, 3), (5, 12), (9, 13), (8, 21), (25, 60)]:
            a, b = rng.normal(size=(n, 73)), rng.normal(size=(m, 73))
            # Mix tie-free, tied, constant, and strong-effect columns in one call.
            a[:, :20] = rng.integers(0, 4, (n, 20))
            b[:, :20] = rng.integers(0, 4, (m, 20))
            a[:, 20:25] = b[:, 20:25] = 0
            a[:, 25:40] += 4
            reference = np.array([mannwhitneyu(a[:, j], b[:, j], alternative='two-sided').pvalue
                                  for j in range(a.shape[1])])
            for chunk_size in (1, 7, 256):
                with self.subTest(n=n, m=m, chunk_size=chunk_size):
                    actual = _mannwhitney_pvalues(a, b, chunk_size)
                    np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-15)
            selected = np.flatnonzero(_bh_qvalues(reference) < .05)
            selected = selected[np.argsort(-np.abs((a.mean(0) - b.mean(0))[selected]), kind='stable')]
            np.testing.assert_array_equal(_significant_by_fdr(a, b, .05), selected)
