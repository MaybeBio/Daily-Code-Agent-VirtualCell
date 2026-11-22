"""HVG must be chosen on train donors, not the full matrix."""
import unittest
import numpy as np
import pandas as pd
import anndata as ad

from bio.data.preprocess import apply_train_hvg
from bio.data.splits import split_indices


class PreprocessTests(unittest.TestCase):
    def test_hvg_uses_train_donor_variance_only(self):
        x = np.ones((40, 6), dtype=np.float32)
        x[:20, 0] = np.linspace(1, 20, 20)
        x[20:, 1] = np.linspace(1, 20, 20)
        obs = pd.DataFrame({'celltype': ['T'] * 40, 'donor_id': ['a'] * 20 + ['b'] * 20})
        adata = ad.AnnData(x, obs=obs)
        train, _, _, _ = split_indices(40, obs['celltype'].to_numpy(), groups=obs['donor_id'].to_numpy(),
                                       seed=41, test_size=0.5)
        train_donor = obs['donor_id'].iloc[int(train[0])]
        expected = '0' if train_donor == 'a' else '1'
        out, report = apply_train_hvg(adata, n_top=1, seed=41, split_by='donor', test_size=0.5)
        self.assertEqual(report['hvg_source'], 'train_split')
        self.assertEqual(list(map(str, out.var_names)), [expected])
        self.assertEqual(out.n_obs, 40)

    def test_train_mask_excludes_held_out_donors(self):
        from bio.data.preprocess import train_cell_mask
        obs = pd.DataFrame({'celltype': ['T'] * 40, 'donor_id': ['a'] * 20 + ['b'] * 20})
        adata = ad.AnnData(np.ones((40, 3), dtype=np.float32), obs=obs)
        mask, info = train_cell_mask(adata, seed=41, split_by='donor', test_size=0.5)
        self.assertEqual(info['mode'], 'donor')
        from bio.data.splits import evaluation_indices
        train, val, test, split = evaluation_indices(adata, test_size=0.5)
        self.assertEqual(split['val_from'], 'seen_train_cells')
        self.assertFalse(mask[val].any())
        self.assertFalse(mask[test].any())
        self.assertEqual(set(obs.iloc[train]['donor_id']) & set(obs.iloc[test]['donor_id']), set())
        self.assertEqual(int(mask.sum()), len(train))

    def test_default_correction_mode_is_native(self):
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/default.yaml').read_text())
        self.assertEqual(cfg['evaluation_cfg']['correction_mode'], 'native')
