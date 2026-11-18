"""Smoke scheduling must preserve prior runs and clean up isolated backends."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('smoke_pipeline', ROOT / 'scripts/smoke_pipeline.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_model_specific_batch_size_survives_smoke_overlay(self):
        import yaml
        from bio.config import config_loader, MergeCfgsDict
        cfg = config_loader('configs/scfoundation.yaml')
        MergeCfgsDict(yaml.safe_load((ROOT / 'configs/smoke.yaml').read_text()), cfg)
        self.assertEqual(cfg['model_cfg']['inference_batch_size'], 1)

    def test_dry_run_writes_nothing_and_launches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'dry'
            with patch.object(smoke, 'run_job') as run:
                self.assertEqual(smoke.main(['--models', 'PCA_model', '--datasets', 'bmmc',
                                            '--run-dir', str(target), '--dry-run']), 0)
                run.assert_not_called()
            self.assertFalse(target.exists())

    def test_failed_embedding_skips_heads_and_resume_keeps_passed_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'run'
            argv = ['--models', 'PCA_model', 'UMAP', '--datasets', 'bmmc', '--run-dir', str(target)]
            with patch.object(smoke, 'contract', return_value='same'), \
                 patch.object(smoke, 'run_job', side_effect=[1] + [0] * 5) as run:
                self.assertEqual(smoke.main(argv), 1)
                self.assertEqual(run.call_count, 6)
            progress = target / 'progress.json'
            report = json.loads(progress.read_text())
            self.assertEqual([j['status'] for j in report['jobs']],
                             ['failed'] + ['skipped'] * 4 + ['passed'] * 5)
            with patch.object(smoke, 'contract', return_value='same'), \
                 patch.object(smoke, 'run_job', return_value=0) as run:
                self.assertEqual(smoke.main(['--resume', str(target)]), 0)
                self.assertEqual(run.call_count, 5)
            self.assertEqual(json.loads(progress.read_text())['passed'], 10)
            with patch.object(smoke, 'contract', return_value='changed'), \
                 patch.object(smoke, 'run_job') as run, self.assertRaises(SystemExit):
                smoke.main(['--resume', str(target)])
            run.assert_not_called()

    def test_fail_fast_leaves_remaining_jobs_available_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'run'
            with patch.object(smoke, 'contract', return_value='same'), \
                 patch.object(smoke, 'run_job', side_effect=subprocess.TimeoutExpired('worker', 1)):
                self.assertEqual(smoke.main(['--models', 'PCA_model', '--datasets', 'bmmc',
                                            '--run-dir', str(target), '--fail-fast']), 1)
            report = json.loads((target / 'progress.json').read_text())
            self.assertEqual([j['status'] for j in report['jobs']], ['timeout'] + ['pending'] * 4)

    def test_timeout_kills_backend_grandchild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / 'orphan-finished'
            child = f'import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()'
            parent = f'import subprocess, sys; subprocess.Popen([sys.executable, "-c", {child!r}]); import time; time.sleep(10)'
            with self.assertRaises(subprocess.TimeoutExpired):
                smoke.run_job([sys.executable, '-c', parent], root / 'worker.log', .5)
            time.sleep(1.1)
            self.assertFalse(marker.exists(), 'Timed-out backend survived its parent')

    def test_resume_restores_custom_seed_and_overlay_without_original_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overlay = root / 'custom.yaml'
            overlay.write_text('data_cfg:\n  max_cells: 48\n')
            target = root / 'run'
            with patch.object(smoke, 'contract', return_value='same'), \
                 patch.object(smoke, 'run_job', return_value=1):
                smoke.main(['--models', 'PCA_model', '--datasets', 'norman', '--seed', '42',
                            '--extra-cfg', str(overlay), '--run-dir', str(target)])
            overlay.unlink()
            with patch.object(smoke, 'contract', return_value='same') as signature, \
                 patch.object(smoke, 'run_job', return_value=0):
                self.assertEqual(smoke.main(['--resume', str(target)]), 0)
                self.assertEqual(signature.call_args.args[1:], ('data_cfg:\n  max_cells: 48\n', 42))
