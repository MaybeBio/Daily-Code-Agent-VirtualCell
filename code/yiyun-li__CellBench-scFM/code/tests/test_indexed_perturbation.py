"""Memory-bounded perturbation batches preserve the original dense tuple."""
import unittest
import numpy as np
import torch
from bio.data.perturbation import (IndexedPerturbationDataset, PerturbationMatchingDataset,
                                    create_perturbation_adata, collate_fn)


class IndexedPerturbationTests(unittest.TestCase):
    def test_rows_matches_ids_and_values_equal_original_dataset(self):
        rng = np.random.default_rng(41)
        raw = rng.poisson(3, (16, 1000)).astype(np.float32)
        embeddings = rng.normal(size=(16, 8)).astype(np.float32)
        labels, ids = np.zeros(16), np.arange(16)
        conditions = np.array(['ctrl'] * 4 + ['G1'] * 6 + ['G2'] * 6)
        controls = np.array([1] * 4 + [0] * 12)
        rows = np.array([5, 8, 15, 4])
        mapping = {5: 2, 8: 0, 15: 3, 4: 1}
        vocab = {'ctrl': 0, 'G1': 1, 'G2': 2}
        old = create_perturbation_adata(rows, mapping, raw, embeddings, labels, ids, conditions, controls)
        old.obs['pert_id'] = old.obs.condition.map(vocab).astype(int)
        original = PerturbationMatchingDataset(old)
        indexed = IndexedPerturbationDataset(raw, embeddings, rows, mapping, [vocab[c] for c in conditions])
        self.assertIs(indexed.raw, raw)
        self.assertIs(indexed.embeddings, embeddings)
        for first, second in zip(collate_fn([original[i] for i in range(4)]),
                                 collate_fn([indexed[i] for i in range(4)])):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
