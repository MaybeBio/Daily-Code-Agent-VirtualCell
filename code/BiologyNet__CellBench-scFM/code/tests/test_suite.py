"""Verify task eligibility, CLI dispatch, and failure dependencies without training."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bio.runtime.suite import DATASETS, BATCHED, PERTURBATION, task_plan, run_suite, selected_configs
from bio.data.catalog import CURATED


class SuiteTests(unittest.TestCase):
    def test_suite_names_match_curated_prepared_files(self):
        root = Path(__file__).resolve().parents[1] / 'datasets'
        from bio.data.catalog import suite_datasets
        self.assertEqual(tuple(DATASETS), suite_datasets())
        self.assertNotIn('immune', DATASETS)
        self.assertNotIn('cortex', DATASETS)
        self.assertNotIn('seaad_mtg', DATASETS)
        self.assertNotIn('seaad_mtg', CURATED)
        self.assertNotIn('cortex', CURATED)
        self.assertIn('immune', CURATED)
        self.assertEqual(BATCHED, {n for n, s in CURATED.items() if s.get('batch')})
        self.assertEqual(PERTURBATION, {n for n, s in CURATED.items() if s.get('perturbation')})
        for ghost in ('liver', 'lung', 'rgc', 'celegan', 'uc_imm1', 'hPancreas', 'Zheng68k', 'pbmc12k'):
            self.assertNotIn(ghost, DATASETS)
            self.assertNotIn(ghost, CURATED)
            with self.assertRaises(ValueError):
                task_plan(ghost, {'data_cfg': {}})
        self.assertEqual(len(task_plan('immune', {'data_cfg': {}})[0]), 5)
        for name, spec in CURATED.items():
            matrix = root / spec['filename']
            self.assertTrue(matrix.is_file(), msg=name)
            self.assertTrue(matrix.with_suffix('.provenance.json').is_file(), msg=name)
        self.assertTrue(CURATED['uc_epi']['batch'] and CURATED['uc_imm']['batch'])
        self.assertTrue(CURATED['adamson']['perturbation'] and CURATED['norman']['perturbation'])


    def test_experiment_panels_are_curated(self):
        import json
        from bio.data.catalog import panel_datasets
        panels = json.loads((Path(__file__).resolve().parents[1] / 'resources/experiment_panels.json').read_text())
        for name, panel in panels.items():
            self.assertEqual(panel_datasets(name), panel['datasets'])
            self.assertEqual(len(panel['datasets']), 4)
            for dataset in panel['datasets']:
                self.assertIn(dataset, CURATED, msg=name)
        self.assertIn('uc_epi', panels['disease']['datasets'])
        self.assertNotIn('uc_imm', panels['disease']['datasets'])
        self.assertNotIn('uc_imm', DATASETS)
        self.assertIn('uc_imm', CURATED)

    def test_panel_cli_dispatches_suite_without_immune(self):
        from bio.cli import main
        from bio.runtime.suite import resolve_datasets, parse_seeds
        with patch('bio.runtime.suite.run_suite', return_value=0) as suite:
            self.assertEqual(main(['--panel', 'organ', '--dry-run']), 0)
        opt, = suite.call_args.args
        self.assertEqual(opt.task, 'all')
        self.assertEqual(opt.panel, 'organ')
        self.assertEqual(resolve_datasets(opt), ['bmmc', 'hlca', 'heart', 'kidney'])
        self.assertEqual(parse_seeds(opt.seed), [41, 42, 43])
        self.assertNotIn('immune', resolve_datasets(opt))

    def test_panel_can_run_embedding_extraction_only(self):
        from bio.cli import main
        with patch('bio.runtime.suite.run_suite', return_value=0) as suite:
            self.assertEqual(main(['--panel', 'organ', '--task', 'embedding_extraction',
                                   '--cfgs', 'configs/pca.yaml', '--seed', '41', '--dry-run']), 0)
        opt, = suite.call_args.args
        self.assertEqual(opt.task, 'embedding_extraction')
        self.assertEqual(opt.panel, 'organ')

    def test_eligibility_and_vcc_scale_guard(self):
        cfg = {'data_cfg': {}}
        self.assertNotIn('perturbation', task_plan('bmmc', cfg)[0])
        self.assertIn('correction', task_plan('bmmc', cfg)[0])
        self.assertIn('correction', task_plan('uc_epi', cfg)[0])
        self.assertEqual(task_plan('norman', cfg)[0], ['embedding_extraction', 'perturbation'])
        cfg['data_cfg']['evaluation_expression_scale'] = 'log1p_normalized'
        self.assertEqual(task_plan('norman', cfg)[0], ['embedding_extraction', 'perturbation'])
        self.assertNotIn('perturbation_vcc', task_plan('norman', cfg)[0])
        with self.assertRaises(ValueError): task_plan('unknown', cfg)

    def test_dataset_cli_dispatches_suite(self):
        from bio.cli import main
        with patch('bio.runtime.suite.run_suite', return_value=0) as suite:
            self.assertEqual(main(['--dataset', 'bmmc', '--dry-run']), 0)
        opt, = suite.call_args.args
        self.assertEqual(opt.task, 'all')
        self.assertIsNone(opt.cfgs)

    def test_failed_perturbation_skips_vcc_and_reports_failure(self):
        opt = Namespace(dataset_name='norman', cfgs='configs/pca.yaml', phase='adapter_finetuning',
                        seed=41, topk=1, gpu=None, hvg=None, dry_run=False, fail_fast=False)
        cfg = {'data_cfg': {}, 'model_cfg': {'model': 'PCA_model'}}
        with tempfile.TemporaryDirectory() as tmp, patch('bio.runtime.suite.PROJECT_ROOT', Path(tmp)), \
             patch('bio.runtime.suite.subprocess.run', side_effect=[Namespace(returncode=0), Namespace(returncode=1)]) as run:
            self.assertEqual(run_suite(opt, cfg), 1)
            self.assertEqual(run.call_count, 2)
            report = json.loads(next(Path(tmp).rglob('summary.json')).read_text())
            self.assertEqual([r['status'] for r in report['tasks']], ['passed', 'failed'])
            self.assertNotIn('perturbation_vcc', [r['task'] for r in report['tasks']])

    def test_dry_run_does_not_launch_processes(self):
        opt = Namespace(dataset_name='bmmc', cfgs='configs/pca.yaml', phase='adapter_finetuning',
                        seed=41, topk=1, gpu=None, hvg=None, dry_run=True, fail_fast=False)
        with patch('bio.runtime.suite.subprocess.run') as run:
            self.assertEqual(run_suite(opt, {'data_cfg': {}, 'model_cfg': {'model': 'PCA_model'}}), 0)
            run.assert_not_called()

    def test_default_configs_cover_all_registered_models(self):
        from bio.models import list_models
        configs = selected_configs(Namespace(cfgs=None))
        self.assertEqual([cfg['model_cfg']['model'] for _, cfg in configs], list(list_models()))
        self.assertEqual(len(configs), 16)
        for _, cfg in configs:
            self.assertEqual(len(task_plan('bmmc', cfg)[0]), 5)

    def test_failure_is_scoped_to_model_and_single_config_is_respected(self):
        opt = Namespace(dataset_name='bmmc', cfgs=None, phase='adapter_finetuning',
                        seed=41, topk=1, gpu=None, hvg=None, dry_run=False, fail_fast=False)
        configs = [(Path('configs/pca.yaml'), {'data_cfg': {}, 'model_cfg': {'model': name}})
                   for name in ('First', 'Second')]
        with tempfile.TemporaryDirectory() as tmp, patch('bio.runtime.suite.PROJECT_ROOT', Path(tmp)), \
             patch('bio.runtime.suite.selected_configs', return_value=configs), \
             patch('bio.runtime.suite.subprocess.run', side_effect=[Namespace(returncode=1)]+[Namespace(returncode=0)]*5) as run:
            self.assertEqual(run_suite(opt), 1)
            self.assertEqual(run.call_count, 6)
            report = json.loads(next(Path(tmp).rglob('summary.json')).read_text())
            self.assertEqual([r['status'] for r in report['tasks']],
                             ['failed', 'skipped', 'skipped', 'skipped', 'skipped']+['passed']*5)
            self.assertTrue((next(Path(tmp).glob('output/suites/*'))/'Second'/'bmmc_seed41_clustering.log').is_file())
        selected = selected_configs(Namespace(cfgs='configs/scgpt.yaml'))
        self.assertEqual([cfg['model_cfg']['model'] for _, cfg in selected], ['scGPT'])
