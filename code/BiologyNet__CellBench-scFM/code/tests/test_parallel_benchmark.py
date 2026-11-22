"""Exercise actual scheduling overlap, cache lanes, GPU masks and cancellation."""
import contextlib
import io
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from bio.runtime.devices import cuda_selector, gpu_schedule
from scripts import run_full_benchmark as queue
from scripts import server_benchmark as server


class DeviceTests(unittest.TestCase):
    def test_gpu_budgets_and_invalid_requests(self):
        self.assertEqual(gpu_schedule(gpus='0,1'), ([0, 1], 2))
        self.assertEqual(gpu_schedule(gpus='0,1', max_workers=1), ([0, 1], 1))
        self.assertEqual(gpu_schedule(gpu=0, max_workers=2, tasks_per_gpu=2), ([0], 2))
        for options in [dict(gpus='0,0'), dict(gpus='-1'), dict(gpus=''),
                        dict(gpus='0,,1'), dict(max_workers=0), dict(tasks_per_gpu=0),
                        dict(gpus='0,1', max_workers=3), dict(gpu=0, gpus='1')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                gpu_schedule(**options)

    def test_visible_device_indices_include_uuid_and_cpu_allocations(self):
        self.assertEqual(cuda_selector(1, {'CUDA_VISIBLE_DEVICES': '2,5'}), '5')
        self.assertEqual(cuda_selector(0, {'CUDA_VISIBLE_DEVICES': 'GPU-example'}), 'GPU-example')
        self.assertEqual(cuda_selector(0, {'CUDA_VISIBLE_DEVICES': ''}), '')
        self.assertEqual(cuda_selector(0, {'CUDA_VISIBLE_DEVICES': '-1'}), '')
        self.assertEqual(cuda_selector(3, {}), '3')
        with self.assertRaises(ValueError):
            cuda_selector(2, {'CUDA_VISIBLE_DEVICES': '2,5'})

    def test_leaf_worker_selects_within_original_mask(self):
        from bio.cli import main
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,5'}), \
             patch('bio.runtime.seeds.init_seeds'), patch('bio.tasks.runner.run_model') as run:
            run.side_effect = lambda *a: self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '5')
            self.assertEqual(main(['--cfgs', 'configs/pca.yaml', '--task', 'annotation', '--seed', '41', '--gpu', '1']), 0)

    def test_suite_does_not_narrow_mask_before_dispatch(self):
        from bio.cli import main
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,5'}), patch('bio.runtime.suite.run_suite') as run:
            run.side_effect = lambda *a: int(os.environ['CUDA_VISIBLE_DEVICES'] != '2,5')
            self.assertEqual(main(['--cfgs', 'configs/pca.yaml', '--task', 'all', '--gpu', '1']), 0)

    def test_server_plan_and_command_forward_parallel_options(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, 'preflight', return_value={'ready': True}), \
             patch.object(server.subprocess, 'call', return_value=0) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(server.main(['run', '--run-dir', tmp, '--gpus', '0,1', '--max-workers', '4', '--tasks-per-gpu', '2']), 0)
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--gpus') + 1], '0,1')
            self.assertEqual(cmd[cmd.index('--max-workers') + 1], '4')
            self.assertEqual(cmd[cmd.index('--tasks-per-gpu') + 1], '2')


class ParallelQueueTests(unittest.TestCase):
    def test_overlap_gpu_limits_dependency_order_and_resume(self):
        for gpus, per_gpu in [('0,1', 1), ('0', 2)]:
            with self.subTest(gpus=gpus), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                jobs = [dict(model=m, dataset=d, seed=s, task=t, status='pending', attempts=[])
                        for m in ['PCA_model', 'UMAP'] for d in ['bmmc', 'norman']
                        for s in [41, 42] for t in ['embedding_extraction', 'annotation']]
                guard = threading.Lock()
                lanes, gpu_busy, maximum = set(), {}, [0]
                embedding_done = set()
                rendezvous = threading.Barrier(2)
                first = [0]

                def worker(cmd, log, timeout, cancel_event=None):
                    model = 'PCA_model' if 'pca.yaml' in cmd[cmd.index('--cfgs') + 1] else 'UMAP'
                    dataset, seed, task, gpu = [cmd[cmd.index(k) + 1] for k in ['--dataset_name', '--seed', '--task', '--gpu']]
                    lane = (model, dataset)
                    with guard:
                        self.assertNotIn(lane, lanes, 'Shared embedding cache lane overlapped')
                        lanes.add(lane)
                        gpu_busy[gpu] = gpu_busy.get(gpu, 0) + 1
                        self.assertLessEqual(gpu_busy[gpu], per_gpu)
                        maximum[0] = max(maximum[0], len(lanes))
                        first[0] += 1
                        synchronize = first[0] <= 2
                        if task != 'embedding_extraction':
                            self.assertIn((model, dataset, seed), embedding_done)
                    if synchronize:
                        rendezvous.wait(timeout=5)
                    time.sleep(0.01)
                    if task == 'embedding_extraction':
                        cache = folder / 'embeddings' / f'{dataset}_{model}' / 'v2' / 'test'
                        cache.mkdir(parents=True, exist_ok=True)
                        (cache / 'all.pkl').write_bytes(b'fixture')
                        (cache / 'manifest.json').write_text(json.dumps(dict(schema=queue.SCHEMA, signature='test', shape=[10, 2])))
                        log.write_text('embedding fixture\n')
                    else:
                        record = folder / f'{model}_{dataset}_{seed}_{task}.json'
                        record.write_text(json.dumps(dict(model=model, dataset=dataset, seed=int(seed), task=task,
                                                         summary=dict(acc=1, f1_m=1, balanced_accuracy=1, precision_macro=1))))
                        log.write_text(f'Metrics saved: {record}\n')
                    with guard:
                        if task == 'embedding_extraction':
                            embedding_done.add((model, dataset, seed))
                        lanes.remove(lane)
                        gpu_busy[gpu] -= 1
                    return 0

                args = ['--run-dir', tmp, '--models', 'PCA_model', 'UMAP', '--datasets', 'bmmc', 'norman',
                        '--seeds', '41,42', '--gpus', gpus, '--max-workers', '2', '--tasks-per-gpu', str(per_gpu)]
                with patch.object(queue, 'make_plan', return_value=jobs), patch.object(queue, 'fingerprint', return_value=('test', {})), \
                     patch.object(queue, 'shared_source_digest', return_value='same'), patch.object(queue, 'run_job', side_effect=worker), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(queue.main(args), 0)
                report = json.loads((folder / 'progress.json').read_text())
                self.assertTrue(report['complete'])
                self.assertEqual(maximum[0], 2)
                self.assertEqual(report['status_counts'], {'passed': 16})
                self.assertEqual({j['latest']['gpu'] for j in report['jobs']}, set(map(int, gpus.split(','))))
                with patch.object(queue, 'fingerprint', return_value=('test', {})), patch.object(queue, 'run_job') as run, \
                     patch.object(queue, 'shared_source_digest', return_value='same'), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(queue.main(args), 0)
                    run.assert_not_called()

    def test_parallel_failure_skips_only_its_dependents(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = ['--run-dir', tmp, '--models', 'PCA_model', 'UMAP', '--datasets', 'bmmc', 'norman',
                    '--seeds', '41', '--gpus', '0,1', '--max-workers', '2', '--continue-on-error']
            with patch.object(queue, 'fingerprint', return_value=('test', {})), patch.object(queue, 'run_job', return_value=1) as run:
                self.assertEqual(queue.main(args), 1)
            self.assertEqual(run.call_count, 4)
            report = json.loads((Path(tmp) / 'progress.json').read_text())
            self.assertEqual(report['selected_status_counts'], {'failed': 4, 'skipped': 10})
            self.assertFalse(report['controller_active'])

    def test_cancellation_kills_backend_grandchild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready, finished = root / 'ready', root / 'orphan'
            child = f'import time; from pathlib import Path; time.sleep(1); Path({str(finished)!r}).touch()'
            parent = f'import subprocess,sys,time; from pathlib import Path; subprocess.Popen([sys.executable,"-c",{child!r}]); Path({str(ready)!r}).touch(); time.sleep(10)'
            cancel = threading.Event()

            def stop():
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                cancel.set()

            stopper = threading.Thread(target=stop)
            stopper.start()
            try:
                with self.assertRaises(InterruptedError):
                    queue.run_job([sys.executable, '-c', parent], root / 'worker.log', 10, cancel_event=cancel)
            finally:
                cancel.set()
                stopper.join(timeout=5)
            time.sleep(1.1)
            self.assertFalse(finished.exists(), 'Cancelled backend survived its parent')

    def test_fail_fast_cancels_other_active_task(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            rendezvous = threading.Barrier(2)
            jobs = [dict(model=m, dataset='bmmc', seed=41, task='embedding_extraction', status='pending', attempts=[])
                    for m in ['PCA_model', 'UMAP']]

            def worker(cmd, log, timeout, cancel_event=None):
                rendezvous.wait(timeout=5)
                if 'pca.yaml' in cmd[cmd.index('--cfgs') + 1]:
                    return 1
                self.assertTrue(cancel_event.wait(timeout=5))
                raise InterruptedError('cancelled')

            with patch.object(queue, 'make_plan', return_value=jobs), patch.object(queue, 'fingerprint', return_value=('test', {})), \
                 patch.object(queue, 'run_job', side_effect=worker):
                self.assertEqual(queue.main(['--run-dir', tmp, '--seeds', '41', '--gpus', '0,1']), 1)
            report = json.loads((Path(tmp) / 'progress.json').read_text())
            self.assertEqual(report['status_counts'], {'failed': 1, 'interrupted': 1})
            self.assertFalse(report['controller_active'])

    def test_sigterm_cleans_up_parallel_worker_groups_and_saves_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'controller.py'
            script.write_text('''import sys, json
from pathlib import Path
sys.path.insert(0, ROOT)
from scripts import run_full_benchmark as q
q.make_plan = lambda *a: [dict(model=m, dataset='bmmc', seed=41, task='embedding_extraction', status='pending', attempts=[]) for m in ['PCA_model','UMAP']]
q.fingerprint = lambda *a: ('test', {})
q.shared_source_digest = lambda: 'same'
original = q.run_job
def worker(cmd, log, timeout, cancel_event=None):
    gpu = cmd[cmd.index('--gpu')+1]
    ready = str(Path(TMP)/('ready'+gpu))
    finished = str(Path(TMP)/('orphan'+gpu))
    child = f'import time; from pathlib import Path; time.sleep(1.5); Path({finished!r}).touch()'
    parent = f'import os, subprocess, sys, time; from pathlib import Path; subprocess.Popen([sys.executable,"-c",{child!r}]); Path({ready!r}).write_text(str(os.getpid())); time.sleep(20)'
    return original([sys.executable,'-c',parent], log, timeout, cancel_event=cancel_event)
q.run_job = worker
sys.exit(q.main(['--run-dir',str(Path(TMP)/'run'),'--seeds','41','--gpus','0,1','--continue-on-error']))
'''.replace('ROOT', repr(str(queue.ROOT))).replace('TMP', repr(tmp)))
            env = dict(os.environ)
            env.pop('CUDA_VISIBLE_DEVICES', None)
            with (root / 'controller.log').open('w') as log:
                controller = subprocess.Popen([sys.executable, str(script)], stdout=log, stderr=subprocess.STDOUT, env=env)
                try:
                    deadline = time.monotonic() + 10
                    while not all((root / f'ready{i}').exists() for i in [0, 1]) and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(all((root / f'ready{i}').exists() for i in [0, 1]), script.read_text())
                    controller.terminate()
                    self.assertEqual(controller.wait(timeout=10), 143)
                finally:
                    if controller.poll() is None:
                        controller.kill()
                        controller.wait()
                    for ready in root.glob('ready*'):
                        try:
                            os.killpg(int(ready.read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            report = json.loads((root / 'run/progress.json').read_text())
            self.assertFalse(report['controller_active'])
            self.assertEqual(report['status_counts'], {'interrupted': 2})
            time.sleep(1.6)
            self.assertFalse(list(root.glob('orphan*')))
