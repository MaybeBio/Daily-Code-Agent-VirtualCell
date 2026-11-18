"""The production queue retains the full scope and stops for a failed repair."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_queue', ROOT / 'scripts/run_full_benchmark.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class FullQueueTests(unittest.TestCase):
    def test_complete_plan_and_explicit_user_holds(self):
        jobs = queue.make_plan([41, 42, 43], queue.HELD)
        self.assertEqual(len(jobs), 3264)
        self.assertEqual(sum(j['status'] == 'held' for j in jobs), 204)
        self.assertEqual(len({(j['model'], j['dataset'], j['seed'], j['task']) for j in jobs}), 3264)

    def test_failure_stops_and_requires_explicit_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'run'
            args = ['--run-dir', str(folder), '--models', 'PCA_model', '--datasets', 'bmmc']
            with patch.object(queue, 'fingerprint', return_value=('test', {})), \
                 patch.object(queue, 'run_job', return_value=1) as run:
                self.assertEqual(queue.main(args), 1)
                self.assertEqual(run.call_count, 1)
            saved = json.loads((folder / 'progress.json').read_text())
            self.assertEqual(len(saved['jobs']), 3264)
            self.assertEqual(saved['status_counts']['failed'], 1)
            self.assertFalse(saved['complete'])
            with patch.object(queue, 'run_job') as run:
                self.assertEqual(queue.main(args), 1)
                run.assert_not_called()
            with patch.object(queue, 'fingerprint', return_value=('test', {})), \
                 patch.object(queue, 'run_job', return_value=1) as run:
                self.assertEqual(queue.main(args + ['--retry-failed']), 1)
                self.assertEqual(run.call_count, 1)
            saved = json.loads((folder / 'progress.json').read_text())
            self.assertEqual(len(saved['jobs'][0]['attempts']), 2)

    def test_continue_records_failures_and_runs_other_models_and_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'run'
            args = ['--run-dir', str(folder), '--models', 'PCA_model', 'UMAP',
                    '--datasets', 'bmmc', 'norman', '--seeds', '41', '--continue-on-error']
            with patch.object(queue, 'fingerprint', return_value=('test', {})), \
                 patch.object(queue, 'run_job', return_value=1) as run:
                self.assertEqual(queue.main(args), 1)
                self.assertEqual(run.call_count, 4)
            report = json.loads((folder / 'progress.json').read_text())
            self.assertFalse(report['controller_active'])
            self.assertEqual(report['selected_status_counts'], {'failed': 4, 'skipped': 10})
            self.assertFalse(report['selected_complete'])
            failures = json.loads((folder / 'failures.json').read_text())
            self.assertEqual(len(failures), 14)
            self.assertEqual({j['model'] for j in failures}, {'PCA_model', 'UMAP'})
            with patch.object(queue, 'run_job') as run:
                self.assertEqual(queue.main(args), 1)
                run.assert_not_called()

    def test_held_model_is_not_executed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(queue, 'run_job') as run:
            self.assertEqual(queue.main(['--run-dir', str(Path(tmp) / 'run'),
                                         '--models', 'CellFM', '--datasets', 'bmmc']), 0)
            run.assert_not_called()

    def test_dry_run_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'run'
            self.assertEqual(queue.main(['--run-dir', str(folder), '--dry-run']), 0)
            self.assertFalse(folder.exists())
