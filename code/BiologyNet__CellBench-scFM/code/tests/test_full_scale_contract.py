"""Full-size safeguards must preserve target values and species semantics."""
import json
from pathlib import Path
import pickle
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from bio.data.foundation import prepare_input
from bio.models.adapter import FoundationAdapter


class FullScaleTests(unittest.TestCase):
    def test_model_signatures_track_shared_and_selected_implementation_only(self):
        from bio.io.embeddings import model_code_files
        root = Path(__file__).resolve().parents[1]
        pca = {str(p.relative_to(root)) for p in model_code_files('PCA_model')}
        compass = {str(p.relative_to(root)) for p in model_code_files('GeneCompass')}
        self.assertIn('bio/models/adapters/baselines.py', pca)
        self.assertIn('bio/models/heads.py', pca)
        self.assertIn('bio/data/splits.py', pca)
        self.assertNotIn('bio/models/adapters/geneformer.py', pca)
        self.assertNotIn('bio/models/compat/genecompass.py', pca)
        self.assertIn('bio/models/compat/genecompass.py', compass)
        self.assertIn('bio/runtime/frontier_backends.py', compass)

    def test_missing_never_successful_model_excludes_full_comparison_block(self):
        from bio.io.export import paired_ranks
        frame = pd.DataFrame([dict(model=m, dataset='bmmc', seed=41, topk=0,
                                   phase='adapter_finetuning', protocol='same', acc=.5)
                              for m in ('A', 'B')])
        ranks, coverage = paired_ranks(frame, {'acc': True}, expected_models={'A', 'B', 'C'})
        self.assertTrue(ranks.empty)
        self.assertFalse(coverage.complete.any())
        self.assertEqual(int(coverage.expected_models.iloc[0]), 3)

    def test_compact_cache_restores_exact_targets_and_detects_modified_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = ad.AnnData(sparse.csr_matrix(np.ones((10, 1000), dtype=np.float32)),
                              obs=pd.DataFrame({'celltype': ['T', 'B'] * 5}, index=[str(i) for i in range(10)]))
            fake = SimpleNamespace(adata_all=data, model_name='Test', _embedding_fingerprint='test',
                                   _embedding_provenance={})
            fake.get_embedding_dir_and_filename = lambda: (tmp, str(Path(tmp) / 'all.pkl'))
            fake.evaluation_expression = MethodType(FoundationAdapter.evaluation_expression, fake)
            raw = fake.evaluation_expression()
            _, labels = np.unique(data.obs.celltype, return_inverse=True)
            embeddings = np.arange(20, dtype=np.float32).reshape(10, 2)
            FoundationAdapter.save_embeddings(fake, embeddings, labels, raw, np.arange(10))
            with (Path(tmp) / 'all.pkl').open('rb') as f:
                stored = pickle.load(f)
            self.assertNotIn('raw_exprs', stored)
            self.assertLess((Path(tmp) / 'all.pkl').stat().st_size, raw.nbytes / 2)
            restored = FoundationAdapter.load_embeddings(fake)
            np.testing.assert_array_equal(restored[0], data.X.toarray())
            np.testing.assert_array_equal(restored[1], embeddings)
            with self.assertRaisesRegex(ValueError, 'changed.*targets'):
                FoundationAdapter.save_embeddings(fake, embeddings, labels, raw * 2, np.arange(10))
            data.X = data.X * 2
            with self.assertRaisesRegex(ValueError, 'targets do not match'):
                FoundationAdapter.load_embeddings(fake)

    def test_selected_target_loss_has_identical_gradients_to_full_mask(self):
        torch.manual_seed(41)
        pred = torch.randn(64, 1500, requires_grad=True)
        truth = torch.randn(64, 1500)
        selected = torch.arange(0, 1200, 3)
        mask = torch.zeros_like(pred, dtype=torch.bool)
        mask[:, selected] = True
        before = torch.nn.functional.mse_loss(pred[mask], truth[mask])
        old_grad, = torch.autograd.grad(before, pred, retain_graph=True)
        after = torch.nn.functional.mse_loss(pred.index_select(1, selected), truth[:, selected])
        new_grad, = torch.autograd.grad(after, pred)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(old_grad, new_grad, rtol=0, atol=0)

    def test_human_only_mouse_projection_and_native_mouse_keep_counts_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mouse = [f'ENSMUSG{i:011d}' for i in range(120)]
            human = [f'ENSG{i:011d}' for i in range(120)]
            data = ad.AnnData(sparse.csr_matrix(np.arange(240, dtype=np.float32).reshape(2, 120) + 1),
                              obs=pd.DataFrame(index=['a', 'b']),
                              var=pd.DataFrame({'gene_name': [f'MouseGene{i}' for i in range(120)]}, index=mouse))
            path = root / 'source.h5ad'
            data.write_h5ad(path)
            cfg = dict(data_cfg=dict(dataset_name='mouse_marrow', foundation_input_path=str(path)),
                       model_cfg=dict(model='Geneformer'))
            with patch('bio.data.orthologs._MAPPING', dict(zip(mouse, human))):
                prepare_input(cfg, data, root / 'human.h5ad')
            projected = ad.read_h5ad(root / 'human.h5ad')
            self.assertEqual(set(projected.var_names), set(human))
            self.assertEqual(projected.uns['cellbench_gene_projection']['mapped_fraction'], 1.)
            np.testing.assert_array_equal(projected.X.toarray(), data.X.toarray())
            cfg['model_cfg']['model'] = 'scPRINT2'
            prepare_input(cfg, data, root / 'native.h5ad')
            native = ad.read_h5ad(root / 'native.h5ad')
            self.assertEqual(set(native.var_names), set(mouse))
            self.assertEqual(set(native.obs.organism_ontology_term_id), {'NCBITaxon:10090'})
            np.testing.assert_array_equal(native.X.toarray(), data.X.toarray())
