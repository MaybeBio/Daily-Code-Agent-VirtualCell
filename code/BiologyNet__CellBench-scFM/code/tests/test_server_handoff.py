"""The handoff must fail before dispatch if resources are missing and retain full scope."""
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('server_handoff', ROOT / 'scripts/server_benchmark.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ServerHandoffTests(unittest.TestCase):
    def test_plan_covers_all_models_without_loading_or_starting_workers(self):
        with patch.object(server, 'preflight') as check, patch.object(server.subprocess, 'call') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(server.main(['plan']), 0)
            check.assert_not_called()
            run.assert_not_called()
        jobs = server.selected_jobs(server.ORDER, server.suite_datasets(), [41, 42, 43])
        self.assertEqual(len(jobs), 3264)
        self.assertEqual({j['model'] for j in jobs}, set(server.ORDER))
        self.assertEqual({j['status'] for j in jobs}, {'pending'})

    def test_missing_resources_prevent_run_and_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'not-created'
            with patch.object(server, 'preflight', return_value=dict(ready=False, errors=['missing checkpoint'])), patch.object(server.subprocess, 'call') as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(server.main(['run', '--run-dir', str(folder)]), 2)
                run.assert_not_called()
            self.assertFalse(folder.exists())

    def test_handoff_command_includes_all_models_and_failure_continuation(self):
        args = argparse.Namespace(run_dir=Path('/tmp/new-run'), models=list(server.ORDER),
                                  datasets=list(server.suite_datasets()), seeds='41', gpu=0,
                                  timeout=86400, retry_failed=False)
        cmd = server.command(args)
        self.assertIn('--continue-on-error', cmd)
        self.assertIn('--release-model', cmd)
        self.assertNotIn('--retry-failed', cmd)
        self.assertIn('CellFM', cmd)
        self.assertIn('TranscriptFormer', cmd)
        self.assertIn('scLong', cmd)

    def test_modern_uc_needs_no_legacy_sidecars_but_legacy_layout_needs_all_six(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ucepi.h5ad'
            self.assertEqual(server.uc_sidecars(path), [])
            (path.parent / 'uc_epi_cell.tsv').write_text('cell1\n')
            self.assertEqual(len(server.uc_sidecars(path)), 6)
            self.assertIn(path.parent / 'uc_epi_gene.tsv', server.uc_sidecars(path))
