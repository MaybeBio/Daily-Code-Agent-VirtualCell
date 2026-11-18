"""Chunked native Tabula binning must preserve row values and RNG semantics."""
import json
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


class TabulaPreprocessingTests(unittest.TestCase):
    def test_chunked_binning_matches_upstream_and_sparse_dataset(self):
        from bio.models.adapters.tabula import TabulaDataPreprocessor, TabulaDataset
        x = np.random.default_rng(4).poisson(2, (7, 20)).astype(np.float32)
        genes = [f'G{i}' for i in range(20)]
        data = ad.AnnData(sparse.csr_matrix(x),
                          obs=pd.DataFrame({'celltype': ['T'] * 7}, index=[str(i) for i in range(7)]),
                          var=pd.DataFrame({'gene_name': genes}, index=genes))
        with tempfile.TemporaryDirectory() as tmp:
            vocab = Path(tmp) / 'vocab.json'
            vocab.write_text(json.dumps(dict(zip(genes, range(20)))))
            worker = TabulaDataPreprocessor(str(vocab), chunk_size=2)
            reference = data.copy()
            np.random.seed(41)
            worker.preprocessor(reference, batch_key=None)
            np.random.seed(41)
            result = worker.process_adata(data.copy())
            self.assertTrue(sparse.issparse(result.layers['X_binned']))
            np.testing.assert_array_equal(result.layers['X_binned'].toarray(), reference.layers['X_binned'])
            np.testing.assert_array_equal(result.obsm['bin_edges'], reference.obsm['bin_edges'])
            self.assertEqual(result.obs_names.tolist(), data.obs_names.tolist())
            dataset = TabulaDataset(result, str(vocab), in_feature=8)
            self.assertTrue(sparse.issparse(dataset.original_X))
            np.testing.assert_array_equal(dataset[3]['original'].numpy(), x[3])

    def test_aligned_cache_validation_preserves_expression_storage(self):
        from bio.io.embeddings import validate_result
        raw = np.ones((3, 4), dtype=np.float32)
        result = validate_result(raw, np.ones((3, 2)), np.zeros(3), np.arange(3), 3)
        self.assertIs(result[0], raw)
