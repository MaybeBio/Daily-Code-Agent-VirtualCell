"""Final acceptance cannot hide missing artifacts or undefined primary metrics."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('finish_benchmark',ROOT/'scripts/finish_benchmark.py')
finish=importlib.util.module_from_spec(spec);spec.loader.exec_module(finish)


class FinalAcceptanceTests(unittest.TestCase):
    def test_missing_metric_is_failed_acceptance_even_if_worker_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            job=dict(model='PCA_model',dataset='bmmc',seed=41,task='annotation',status='passed',latest=dict(metrics=str(root/'missing.json')))
            (root/'progress.json').write_text(json.dumps(dict(complete=True,jobs=[job])))
            with patch.object(finish,'export_panels',side_effect=FileNotFoundError('no records')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finish.finish(root),1)
            result=json.loads((root/'final_summary.json').read_text())
            self.assertFalse(result['all_planned_passed'])
            self.assertTrue(result['artifact_errors'])
            self.assertTrue((root/'task_results.csv').exists())

    def test_null_metric_is_disclosed_separately_from_execution_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metric=root/'metric.json'
            identity=dict(model='PCA_model',dataset='bmmc',seed=41,task='annotation')
            metric.write_text(json.dumps(dict(identity,metric_version=2,summary=dict(acc=1,f1_m=None,balanced_accuracy=1,precision_macro=1))))
            job=dict(identity,status='passed',latest=dict(metrics=str(metric)))
            (root/'progress.json').write_text(json.dumps(dict(complete=True,jobs=[job])))
            with patch.object(finish,'export_panels',return_value=['organ']),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finish.finish(root),0)
            result=json.loads((root/'final_summary.json').read_text())
            self.assertTrue(result['all_planned_passed'])
            self.assertFalse(result['all_primary_metrics_available'])
            self.assertEqual(result['null_primary'][0]['null_primary'],['f1_m'])
