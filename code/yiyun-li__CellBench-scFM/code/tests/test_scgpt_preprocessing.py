"""Chunked scGPT preprocessing/tokenization must keep native bins and RNG."""
import unittest
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from bio.models.adapters.scgpt import DataPreprocessor, DataTokenizer


class ScGPTPreprocessingTests(unittest.TestCase):
    def test_matches_native_dense_bins_tokens_padding_and_rng(self):
        rng = np.random.default_rng(41)
        x = rng.poisson(3, (24, 71)).astype(np.float32)
        x[-3:, 3:] = 0  # The last chunk needs the global, not local, pad width.
        genes = [f'G{i}' for i in range(71)]
        vocab = GeneVocab.from_dict({g: i for i, g in enumerate(genes + ['<pad>', '<cls>', '<eoc>'])})
        data = ad.AnnData(sparse.csr_matrix(x), var=pd.DataFrame({'gene_name': genes}, index=genes),
                          obs=pd.DataFrame({'celltype_id': [0, 1] * 12, 'batch_id': ['1'] * 24},
                                           index=[f'cell{i}' for i in range(24)]))
        native = data.copy()
        np.random.seed(41)
        DataPreprocessor(vocab).preprocessor(native, batch_key=None)
        expected_bins = native.layers['X_binned'].copy()
        expected = tokenize_and_pad_batch(expected_bins, np.array(vocab(genes)), max_len=32,
                                          vocab=vocab, pad_token='<pad>', pad_value=-2,
                                          append_cls=True, include_zero_gene=False)
        expected_masked = random_mask_value(expected['values'], mask_ratio=0., mask_value=-1, pad_value=-2)
        expected_rng = np.random.get_state()
        np.random.seed(41)
        processed = DataPreprocessor(vocab, chunk_size=7).process_adata(data.copy())
        actual = DataTokenizer(vocab, max_length=32, chunk_size=7).prepare_model_input(processed)
        np.testing.assert_array_equal(processed.layers['X_binned'].toarray(), expected_bins)
        torch.testing.assert_close(actual['gene_ids'], expected['genes'], rtol=0, atol=0)
        torch.testing.assert_close(actual['target_values'], expected['values'], rtol=0, atol=0)
        torch.testing.assert_close(actual['values'], expected_masked, rtol=0, atol=0)
        actual_rng = np.random.get_state()
        np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
        self.assertEqual(actual_rng[2:], expected_rng[2:])
