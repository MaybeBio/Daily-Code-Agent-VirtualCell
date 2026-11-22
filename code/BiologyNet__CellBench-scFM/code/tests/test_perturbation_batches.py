"""Regression coverage for the singleton tail in perturbation-head training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from bio.data.batching import MergeSingletonBatchSampler
from bio.models.heads import PerturbDecoder
from bio.tasks.perturbation import PerturbationTaskMixin


class PerturbationBatchTests(unittest.TestCase):
    def test_boundaries_preserve_each_sample_once_and_avoid_singletons(self):
        for count in [2, 3, 63, 64, 65, 66, 127, 128, 129, 130, 18881]:
            with self.subTest(count=count):
                sampler = MergeSingletonBatchSampler(SequentialSampler(range(count)), 64)
                batches = list(sampler)
                self.assertEqual(len(batches), len(sampler))
                self.assertEqual([i for batch in batches for i in batch], list(range(count)))
                self.assertTrue(all(2 <= len(batch) <= 65 for batch in batches))
                if count % 64 == 1:
                    self.assertEqual(len(batches[-1]), 65)

    def test_shuffling_remains_seeded_and_covers_all_rows(self):
        def batches(seed):
            sampler = RandomSampler(range(129), generator=torch.Generator().manual_seed(seed))
            return list(MergeSingletonBatchSampler(sampler, 64))
        first = batches(43)
        self.assertEqual(first, batches(43))
        self.assertNotEqual(first, batches(42))
        self.assertEqual(sorted(i for batch in first for i in batch), list(range(129)))

    def test_insufficient_training_data_fails_clearly(self):
        for count in [0, 1]:
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'at least two'):
                MergeSingletonBatchSampler(SequentialSampler(range(count)), 64)

    def test_original_failure_and_fixed_real_decoder_backward(self):
        head = PerturbDecoder(4, 2, 5).train()
        with self.assertRaisesRegex(ValueError, 'more than 1 value per channel'):
            head(torch.ones(1, 4), torch.ones(1, dtype=torch.long))
        data = TensorDataset(torch.randn(129, 4), torch.ones(129, dtype=torch.long))
        sampler = MergeSingletonBatchSampler(SequentialSampler(data), 64)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        count = 0
        for embeddings, conditions in DataLoader(data, batch_sampler=sampler):
            optimizer.zero_grad()
            prediction = head(embeddings, conditions)
            loss = prediction.square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(head.proj_out.weight.grad).all())
            optimizer.step()
            count += len(conditions)
        self.assertEqual(count, 129)
        self.assertEqual(head.bn1.num_batches_tracked.item(), 3)  # failed forward plus two good batches
        head.eval()
        self.assertEqual(head(torch.ones(1, 4), torch.ones(1, dtype=torch.long)).shape, (1, 5))

    def test_task_uses_safe_training_batches_and_keeps_validation_test_rows(self):
        class Harness(PerturbationTaskMixin):
            def __init__(self):
                self.cfgs = dict(seed=43, model_cfg={'head_epochs': 1})
                self.dataset_name, self.model_name = 'synthetic', 'probe'
                self.device = 'cpu'
                self.raw = np.random.default_rng(43).random((71, 5), dtype=np.float32)
                self.embeddings = np.random.default_rng(43).random((71, 4), dtype=np.float32)
                self.labels, self.ids = np.zeros(71, dtype=int), np.arange(71)
                self.adata_all = ad.AnnData(self.raw, obs=pd.DataFrame(
                    {'condition': ['ctrl'] * 3 + ['G1'] * 68,
                     'control': [1] * 3 + [0] * 68}, index=[f'c{i}' for i in range(71)]))
                self.pert_decoder = PerturbDecoder(4, 2, 5)
                self.optimizer_pert = torch.optim.Adam(self.pert_decoder.parameters(), lr=1e-3)
                self.train_batches = []

            def extract_cell_embeddings(self):
                return self.raw, self.embeddings, self.labels, self.ids

            def load_control_map(self, *args):
                return {i: 0 for i in range(3, 71)}

            def evaluation_split(self, *args):
                return np.arange(3, 68), np.array([68]), np.array([69, 70]), {'n_train': 65, 'n_val': 1, 'n_test': 2}

            def forward_perturbation(self, embeddings, identities, control):
                if self.pert_decoder.training:
                    self.train_batches.append(len(identities))
                return control + self.pert_decoder(embeddings, identities)

            def metric(self, inputs, task):
                self.evaluation = inputs
                return {'probe': 1.0}

        with tempfile.TemporaryDirectory() as temporary:
            model = Harness()
            with patch('bio.io.artifacts.prediction_directory', return_value=Path(temporary)):
                model.adapter_finetuning_perturbation(model.cfgs)
            self.assertEqual(model.train_batches, [65])
            self.assertEqual(len(model.pert_trainloader.dataset), 65)
            self.assertEqual(len(model.pert_validloader.dataset), 1)
            self.assertEqual(len(model.pert_testloader.dataset), 2)
            self.assertEqual(model.evaluation['truth'].shape, (2, 5))
            self.assertEqual(model.evaluation['split']['n_train'], 65)
