"""One extraction failure must not prevent other datasets/models from running."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'organ_continuation', Path(__file__).resolve().parents[1] / 'scripts/continue_organ_embeddings.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class OrganContinuationTests(unittest.TestCase):
    def exercise(self, launch_error=False, previous_failure=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'output/organ_embedding_continuation/progress.json'
            if previous_failure:
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'tasks': [dict(
                    model='First', dataset='d1', status='failed', returncode=-9, log='retained.log')]}))
            completed, called = set(), []
            def inventory(dataset, model):
                return [{'shape': [10, 4]}] if (model, dataset) in completed else []
            def run(command, **kwargs):
                model = Path(command[command.index('--cfgs') + 1]).stem.title()
                dataset = command[command.index('--dataset_name') + 1]
                called.append((model, dataset))
                if (model, dataset) == ('First', 'd1'):
                    if launch_error:
                        raise OSError('Cannot launch worker')
                    return SimpleNamespace(returncode=1)
                completed.add((model, dataset))
                return SimpleNamespace(returncode=0)
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'list_models', return_value=['First', 'Second']), \
                 patch.object(queue, 'panel_datasets', return_value=['d1', 'd2']), \
                 patch.object(queue, 'inventory', side_effect=inventory), \
                 patch.object(queue.subprocess, 'run', side_effect=run), patch('sys.argv', ['queue']):
                self.assertEqual(queue.main(), 1)
            report = json.loads(path.read_text())
            self.assertEqual([t['status'] for t in report['tasks']], ['failed', 'passed', 'passed', 'passed'])
            self.assertIn('finished_at', report)
            self.assertEqual(report['failed_tasks'], 1)
            self.assertEqual(called[-1], ('Second', 'd2'))
            if previous_failure:
                self.assertNotIn(('First', 'd1'), called)
                self.assertEqual(report['tasks'][0]['log'], 'retained.log')

    def test_worker_failure_continues_other_datasets_and_models(self):
        self.exercise()

    def test_launch_error_continues_queue(self):
        self.exercise(launch_error=True)

    def test_resume_skips_prior_failure_and_preserves_evidence(self):
        self.exercise(previous_failure=True)
