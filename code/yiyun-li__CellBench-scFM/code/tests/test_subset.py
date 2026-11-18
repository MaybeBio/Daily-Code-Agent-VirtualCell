import unittest

import numpy as np
from anndata import AnnData
import pandas as pd


class SubsetTests(unittest.TestCase):
    def test_subset_keeps_two_donors_and_types(self):
        from bio.data.subset import subset_adata
        obs = pd.DataFrame({
            'celltype': ['T'] * 40 + ['B'] * 40,
            'donor_id': ['a'] * 20 + ['b'] * 20 + ['a'] * 20 + ['b'] * 20,
            'batch': ['x'] * 40 + ['y'] * 40,
        })
        data = AnnData(np.ones((80, 5), dtype=np.float32), obs=obs)
        out = subset_adata(data, 20, seed=41)
        self.assertLessEqual(out.n_obs, 20)
        self.assertGreaterEqual(out.n_obs, 8)
        self.assertGreaterEqual(out.obs['donor_id'].nunique(), 2)
        self.assertGreaterEqual(out.obs['celltype'].nunique(), 2)

    def test_subset_keeps_control_and_conditions(self):
        from bio.data.subset import subset_adata
        cond = ['ctrl'] * 20 + [f'G{i}' for i in range(20) for _ in range(3)]
        obs = pd.DataFrame({
            'celltype': ['K'] * len(cond),
            'condition': cond,
            'control': [1] * 20 + [0] * 60,
        })
        data = AnnData(np.ones((len(cond), 4), dtype=np.float32), obs=obs)
        out = subset_adata(data, 30, seed=3)
        self.assertIn('ctrl', set(out.obs['condition'].astype(str)))
        self.assertGreater(out.obs['condition'].nunique(), 3)
