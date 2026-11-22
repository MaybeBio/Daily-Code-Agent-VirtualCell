"""Approved protocol differences retain native per-gene and input semantics."""
import unittest
import numpy as np
import pandas as pd
import torch
from bio.runtime.frontier_backends import sclong_benchmark_mean
from bio.models.adapters.nicheformer import map_mouse_vocabulary, nicheformer_assay_tokens


class AuthorizedProtocolsTests(unittest.TestCase):
    def test_sclong_mean_matches_official_reindex_then_mean_including_pad(self):
        torch.manual_seed(41)
        encoded = torch.randn(3, 7, 5)
        encoded[:, -1] += 100  # Official filler includes the PAD position.
        for selected, missing in [([4, 1, 3], 2), ([6, 0], 0), ([0], 6)]:
            with self.subTest(selected=selected, missing=missing):
                reference = torch.cat([encoded[:, selected],
                    encoded.mean(1, keepdim=True).expand(-1, missing, -1)], dim=1).mean(1)
                actual = sclong_benchmark_mean(encoded, np.array(selected), len(selected) + missing)
                torch.testing.assert_close(actual, reference)

    def test_nicheformer_mapping_priority_native_genes_coverage_and_collisions(self):
        vocabulary = {'ENSG1', 'ENSG2', 'ENSMUSGnative'}
        mapped, report = map_mouse_vocabulary(
            ['ENSMUSGa.1', 'ENSMUSGb', 'ENSMUSGnative', 'ENSMUSGc', 'ENSMUSGd'], vocabulary,
            {'ENSMUSGa': 'ENSG1', 'ENSMUSGc': 'ENSGabsent'},
            {'ENSMUSGa': 'ENSG2', 'ENSMUSGb': 'ENSG1', 'ENSMUSGc': 'ENSG2'})
        self.assertEqual(mapped, ['ENSG1', 'ENSG1', 'ENSMUSGnative', '', ''])
        self.assertEqual(report['source_counts'], dict(official=1, ensembl_fallback=1, native=1, unmapped=2))
        self.assertEqual(report['mapped_fraction'], .6)
        self.assertEqual(report['unique_matched_genes'], 2)

    def test_smartseq_approximation_is_per_cell_and_scoped_to_mixed_heart(self):
        obs = pd.DataFrame({'assay': ['10x', 'smartseq2', 'Smart-seq2', "10x 3' v2"]})
        np.testing.assert_array_equal(nicheformer_assay_tokens(obs, 15, 'mouse_heart'), [15, 18, 18, 15])
        np.testing.assert_array_equal(nicheformer_assay_tokens(obs, 15, 'mouse_marrow'), [15]*4)
        with self.assertRaises(ValueError):
            nicheformer_assay_tokens(pd.DataFrame(index=range(4)), 15, 'mouse_heart')
