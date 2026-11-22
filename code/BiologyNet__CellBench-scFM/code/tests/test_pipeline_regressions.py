"""Exercise the fitting, target, task-head and export paths together."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from bio.data.preprocess import hvg_indices, train_cell_mask
from bio.data.splits import evaluation_indices
from bio.io.embeddings import fingerprint
from bio.io.export import PRIMARY, _load_task_frame, paired_ranks
from bio.io.results import log_metrics


def dataset():
    x = np.random.default_rng(41).poisson(4, (100, 24)).astype(np.float32)
    x[0, 0] = 1000
    return ad.AnnData(sparse.csr_matrix(x), obs=pd.DataFrame(
        {'celltype': ['T', 'B'] * 50}, index=[f'cell{i}' for i in range(100)]))


class PipelineRegressionTests(unittest.TestCase):
    def test_bound_perturbation_genes_forward_backward_and_device(self):
        from bio.models.heads import PerturbDecoder
        head = PerturbDecoder(4, 3, 5)
        head.bind_genes(np.array([[0, 0], [1, 0], [1, 2]]))
        self.assertIn('gene_index', dict(head.named_buffers()))
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            head = head.to(device)
            head.bind_genes(np.array([[0, 0], [1, 0], [1, 2]]))
            self.assertEqual(head.gene_index.device, head.pert_emb.weight.device)
            head(torch.ones(3, 4, device=device), torch.tensor([0, 1, 2], device=device)).sum().backward()
            self.assertTrue(torch.isfinite(head.pert_emb.weight.grad).all())

    def test_hvg_handles_high_and_low_counts_without_changing_targets(self):
        for high in [False, True]:
            data = dataset()
            if not high:
                data.X.data = np.minimum(data.X.data, 4)
            before = data.X.copy()
            indices = hvg_indices(data, n_top=8)
            self.assertGreater(len(indices), 0)
            self.assertEqual((data.X != before).nnz, 0)

    def test_encoder_and_heads_share_exact_training_rows(self):
        for donors in [None, np.repeat(['a', 'b'], 50), np.repeat(list('abcde'), 20)]:
            data = dataset()
            if donors is not None:
                data.obs['donor_id'] = donors
            for task in ['annotation', 'reconstruction']:
                train, val, test, _ = evaluation_indices(data, task=task)
                mask, _ = train_cell_mask(data, task=task)
                np.testing.assert_array_equal(np.flatnonzero(mask), np.sort(train))
                self.assertFalse(mask[test].any())
                self.assertFalse(mask[val].any())
        data = dataset()
        data.obs['condition'] = ['ctrl'] * 20 + np.repeat([f'G{i}' for i in range(20)], 4).tolist()
        data.obs['control'] = [1] * 20 + [0] * 80
        data.obs.loc['cell20', 'condition'] = '*'
        train, val, test, _ = evaluation_indices(data, task='perturbation')
        mask, _ = train_cell_mask(data, perturbation=True)
        expected = np.zeros(100, dtype=bool)
        expected[:20] = True
        expected[train] = True
        np.testing.assert_array_equal(mask, expected)
        self.assertFalse(mask[np.r_[val, test, 20]].any())

    def test_annotation_restores_validation_best_and_tests_only_once(self):
        from types import SimpleNamespace
        from bio.tasks.annotation import AnnotationTaskMixin

        class Harness(AnnotationTaskMixin):
            def __init__(self):
                self.device, self.cell_type_num = 'cpu', 2
                self.classify = torch.nn.Linear(1, 2)
                with torch.no_grad():
                    self.classify.weight.zero_()
                    self.classify.bias.zero_()
                self.loss = torch.nn.CrossEntropyLoss()
                self.steps, self.eval_rows = 0, []
                self.optimizer_classify = SimpleNamespace(
                    zero_grad=self.classify.zero_grad, step=self.worsen)

            def worsen(self):
                # Validation loss deliberately worsens each epoch.
                self.steps += 1
                with torch.no_grad():
                    self.classify.bias.copy_(torch.tensor([-self.steps, self.steps]))

            def extract_cell_embeddings(self):
                return np.zeros((100, 1)), np.arange(100, dtype=np.float32)[:, None], np.zeros(100, dtype=int), np.arange(100)

            def evaluation_split(self, cell_ids, task):
                return np.arange(70), np.arange(70, 80), np.arange(80, 100), dict(n_train=70, n_val=10, n_test=20)

            def forward_annotation(self, inputs):
                if not self.classify.training:
                    self.eval_rows.append(len(inputs))
                return self.classify(inputs)

            def metric(self, inputs, task):
                self.record = inputs

        model = Harness()
        truth, pred = model.adapter_finetuning_annotation({'model_cfg': {'head_epochs': 3}})
        self.assertEqual(model.steps, 6)
        self.assertEqual(model.record['split']['best_epoch'], 1)
        torch.testing.assert_close(model.classify.bias, torch.tensor([-2., 2.]))
        self.assertEqual(model.eval_rows, [10, 10, 10, 20])
        self.assertEqual(len(truth), 20)
        self.assertEqual(len(pred), 20)

    def test_annotation_and_reconstruction_share_default_split(self):
        for donors in [None, np.repeat(['a', 'b'], 50), np.repeat([f'd{i}' for i in range(10)], 10)]:
            data = dataset()
            if donors is not None:
                data.obs['donor_id'] = donors
            annotation = evaluation_indices(data, task='annotation')
            reconstruction = evaluation_indices(data, task='reconstruction')
            for a, b in zip(annotation[:3], reconstruction[:3]):
                np.testing.assert_array_equal(a, b)
            if donors is None:
                self.assertEqual(tuple(map(len, annotation[:3])), (70, 10, 20))
            reference, _ = train_cell_mask(data, task='annotation')
            for task in ['clustering', 'correction']:
                mask, _ = train_cell_mask(data, task=task)
                np.testing.assert_array_equal(mask, reference)

    def test_pca_reconstruction_keeps_full_counts_and_runs_task(self):
        from bio.models.adapters.baselines import PCA_model
        data = dataset()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(output_dir=tmp, seed=41, phase='adapter_finetuning',
                       data_cfg={'dataset_name': 'bmmc', 'hvg': 8},
                       model_cfg={'model': 'PCA_model', 'tasks': 'reconstruction', 'embedding_dim': 4},
                       evaluation_cfg={'split_by': 'donor'})
            with patch.object(PCA_model, 'read_adata', side_effect=lambda _: data.copy()):
                model = PCA_model(cfg)
            raw, embedding, labels = model.encode_batch(model.all_loader[0])
            np.testing.assert_array_equal(raw, data.X.toarray())
            self.assertEqual(model.adata_all.shape, data.shape)
            self.assertEqual(model.restruct.proj_out.out_features, data.n_vars)
            self.assertLess(model.model.n_features_in_, data.n_vars)
            result = (raw, embedding, labels, np.arange(data.n_obs))
            with patch.object(model, 'extract_cell_embeddings', return_value=result), patch(
                    'bio.tasks.reconstruction.range', return_value=range(1), create=True):
                pred, truth = model.adapter_finetuning_reconstruction(cfg)
            self.assertEqual(pred.shape, truth.shape)
            self.assertEqual(truth.shape[1], data.n_vars)
            self.assertTrue(np.isfinite(pred).all())
            test = evaluation_indices(data, task='reconstruction')[2]
            np.testing.assert_array_equal(truth, data.X.toarray()[test])

    def test_export_pairs_models_despite_representation_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in [41, 42]:
                for name, extra in [('scGPT', {}), ('PCA_model', dict(
                        hvg_source='train_split', n_hvg=8, n_fit_cells=seed,
                        hvg_split_mode='cell', fit_split_mode='cell', n_train_cells_for_hvg=seed))]:
                    cfg = dict(output_dir=tmp, seed=seed, phase='adapter_finetuning',
                               data_cfg={'dataset_name': 'bmmc'},
                               model_cfg={'model': name, 'tasks': 'annotation'},
                               evaluation_cfg=dict(split_by='donor', **extra))
                    log_metrics(cfg, dict(acc=.8, f1_m=.8, balanced_accuracy=.8, precision_macro=.8))
            frame = _load_task_frame('annotation', Path(tmp))
            self.assertEqual(frame.protocol.nunique(), 1)
            ranks, coverage = paired_ranks(frame, PRIMARY['annotation'])
            self.assertEqual(set(ranks.model), {'scGPT', 'PCA_model'})
            self.assertEqual(int(coverage.complete.sum()), 2)
            # Old results and a genuinely different split protocol remain separate.
            paths = list((Path(tmp) / 'metrics' / 'annotation').glob('*.json'))
            import json
            record = json.loads(paths[0].read_text())
            record.pop('pipeline_revision')
            paths[0].write_text(json.dumps(record))
            self.assertEqual(_load_task_frame('annotation', Path(tmp)).protocol.nunique(), 2)
            record['pipeline_revision'] = 1
            record['config']['evaluation_cfg']['split_by'] = 'cell'
            paths[0].write_text(json.dumps(record))
            self.assertEqual(_load_task_frame('annotation', Path(tmp)).protocol.nunique(), 2)

    def test_pca_perturbation_runs_with_bound_genes_and_full_targets(self):
        from bio.models.adapters.baselines import PCA_model
        data = dataset()
        data.obs['condition'] = ['ctrl'] * 20 + np.repeat([f'G{i}' for i in range(20)], 4).tolist()
        data.obs['control'] = [1] * 20 + [0] * 80
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.h5ad'
            source.write_bytes(b'synthetic')
            cfg = dict(output_dir=tmp, seed=41, phase='adapter_finetuning',
                       data_cfg={'dataset_name': 'adamson', 'hvg': 8, 'foundation_input_path': str(source)},
                       model_cfg={'model': 'PCA_model', 'tasks': 'perturbation', 'embedding_dim': 4},
                       evaluation_cfg={'perturbation_split': 'auto'})
            with patch.object(PCA_model, 'read_adata', side_effect=lambda _: data.copy()):
                model = PCA_model(cfg)
            raw, embedding, labels = model.encode_batch(model.all_loader[0])
            np.testing.assert_array_equal(raw, data.X.toarray())
            with patch.object(model, 'extract_cell_embeddings', return_value=(
                    raw, embedding, labels, np.arange(data.n_obs))), patch(
                    'bio.tasks.perturbation.range', return_value=range(1), create=True):
                pred, truth = model.adapter_finetuning_perturbation(cfg)
            self.assertEqual(pred.shape, truth.shape)
            self.assertEqual(truth.shape[1], data.n_vars)
            self.assertTrue(np.isfinite(pred).all())
            frame = _load_task_frame('perturbation', Path(tmp))
            self.assertTrue(np.isfinite(frame.MAE_delta).all())

    def test_statistical_cache_tracks_split_options_and_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.h5ad'
            path.write_bytes(b'counts')
            cfg = dict(seed=41, data_cfg={'dataset_name': 'bmmc', 'foundation_input_path': str(path)},
                       model_cfg={'model': 'PCA_model', 'tasks': 'annotation'},
                       evaluation_cfg={'split_by': 'donor', 'perturbation_split': 'auto'})
            first = fingerprint(cfg)[0]
            for section, key, value in [('evaluation_cfg', 'split_by', 'cell'),
                                        ('evaluation_cfg', 'perturbation_split', 'condition'),
                                        ('model_cfg', 'tasks', 'reconstruction')]:
                other = copy.deepcopy(cfg)
                other[section][key] = value
                self.assertNotEqual(first, fingerprint(other)[0])
            cfg['model_cfg']['tasks'] = 'clustering'
            self.assertEqual(first, fingerprint(cfg)[0])
