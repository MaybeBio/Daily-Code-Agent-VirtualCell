"""Exercise the actual self-extracting command with a tiny, offline fixture."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('offline_builder', ROOT/'scripts/build_offline_bundle.py')
builder = importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
spec = importlib.util.spec_from_file_location('offline_deployer', ROOT/'scripts/offline_deploy.py')
deployer = importlib.util.module_from_spec(spec);spec.loader.exec_module(deployer)


class OfflineDeliveryTests(unittest.TestCase):
    def test_self_extract_verify_forward_arguments_and_refuse_foreign_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';stage=Path(tmp)/'stage';payload=stage/'payload'
            (root/'scripts').mkdir(parents=True);(payload/'offline_envs').mkdir(parents=True)
            shutil.copy2(ROOT/'scripts/offline_bootstrap.sh.in',root/'scripts/offline_bootstrap.sh.in')
            (root/'deploy_and_run.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$(dirname "$0")/received_args.txt"\n')
            for directory in ('datasets','weights','public_code'):
                (root/directory).mkdir();(root/directory/'resource.bin').write_bytes(b'fixture\x00content')
            (payload/'offline_envs/manifest.json').write_text(json.dumps(dict(package_id='fixture',platform='linux-x86_64',environment_unpacked_bytes=1048576)))
            args=SimpleNamespace(stage=stage,output=Path(tmp)/'delivery.run',threads=1)
            with patch.object(builder,'ROOT',root),patch.object(builder,'code_files',return_value=['deploy_and_run.sh']),contextlib.redirect_stdout(io.StringIO()):
                builder.package(args,lambda *a,**k:None)
            with args.output.open('rb') as archive:
                self.assertEqual(archive.read(len(b'#!/usr/bin/env bash\n')), b'#!/usr/bin/env bash\n')
            destination=Path(tmp)/'installed'
            result=subprocess.run(['bash',str(args.output),'--prefix',str(destination),'--deploy-only','--gpus','0,1','--max-workers','2'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((destination/'received_args.txt').read_text(),'--deploy-only\n--gpus\n0,1\n--max-workers\n2\n')
            self.assertEqual((destination/'datasets/resource.bin').read_bytes(),b'fixture\x00content')
            self.assertTrue((destination/'.payload_extracted').exists())
            manifest = json.loads((destination/'offline_envs/manifest.json').read_text())
            self.assertEqual(manifest['pipeline_revision'], 3)
            self.assertTrue(manifest['parallel_execution']['gpu_list'])
            self.assertEqual(manifest['code_files'], ['deploy_and_run.sh'])
            delivery = json.loads(args.output.with_suffix('.run.manifest.json').read_text())
            self.assertGreaterEqual(delivery['minimum_free_bytes'], delivery['payload_bytes'] + 1048576 + 8 * 2**30)

            foreign=Path(tmp)/'foreign';foreign.mkdir();(foreign/'keep.txt').write_text('keep')
            result=subprocess.run(['bash',str(args.output),'--prefix',str(foreign)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((foreign/'keep.txt').read_text(),'keep')

    def test_delivery_code_excludes_credentials_caches_and_legacy_files(self):
        names = ['README.md', 'bio/cli.py', 'cell_eval/__init__.py', 'scripts/server_benchmark.py',
                 '.cache/huggingface/token', '.run.lock', 'w_code/old.py', 'notes.pdf',
                 '.git/config', 'bio/__pycache__/old.pyc', 'output/result.json']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            with patch.object(builder, 'ROOT', root), patch.object(builder.subprocess, 'check_output',
                    return_value=('\0'.join(names) + '\0').encode()):
                self.assertEqual(builder.code_files(), sorted(names[:4]))

    def test_refresh_removes_archived_code_from_reused_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'root'
            payload = Path(tmp) / 'payload'
            root.mkdir()
            (root / 'README.md').write_text('current')
            (payload / 'offline_envs').mkdir(parents=True)
            (payload / 'offline_envs/manifest.json').write_text(json.dumps({'code_files': ['w_code/old.py']}))
            (payload / 'w_code').mkdir()
            (payload / 'w_code/old.py').write_text('obsolete')
            with patch.object(builder, 'ROOT', root), patch.object(builder, 'code_files', return_value=['README.md']):
                self.assertEqual(builder.sync_code(payload), ['README.md'])
            self.assertFalse((payload / 'w_code/old.py').exists())
            self.assertEqual((payload / 'README.md').read_text(), 'current')

    def test_release_gate_requires_the_exact_validated_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'verification').mkdir()
            record = root / 'verification/release_validation.json'
            record.write_text(json.dumps(dict(package_id='release', all_checks_passed=True, code_sha256={'main.py': 'old'})))
            with self.assertRaisesRegex(RuntimeError, 'snapshot'):
                builder.verify_release(root, 'release', {'main.py': 'changed'})
            builder.verify_release(root, 'release', {'main.py': 'old'})

    def test_transient_memory_read_error_restores_bytes_and_digest(self):
        import errno
        import hashlib
        import io
        class TransientStream(io.BytesIO):
            failed = False
            def read(self, size=-1):
                if not self.failed:
                    self.failed = True
                    super().read(2)
                    raise OSError(errno.ENOMEM, 'transient read failure')
                return super().read(size)
        data = b'complete archive bytes'
        counts = []
        reader = builder.HashedReader(TransientStream(data), counts.append)
        with patch.object(builder.time, 'sleep'):
            result = b''.join(iter(lambda: reader.read(7), b''))
        self.assertEqual(result, data)
        self.assertEqual(reader.digest.hexdigest(), hashlib.sha256(data).hexdigest())
        self.assertEqual(sum(counts), len(data))

    def test_deployer_reuses_only_same_prefix_and_package_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'offline_envs').mkdir()
            (root/'offline_envs/manifest.json').write_text(json.dumps(dict(package_id='test')))
            state=root/'.offline_environment.json';state.write_text(json.dumps(dict(prefix=str(root.resolve()),package_id='test')))
            with patch.object(deployer.subprocess,'run') as run:
                deployer.deploy(root);run.assert_not_called()
            state.write_text(json.dumps(dict(prefix='/different/path',package_id='test')))
            with self.assertRaisesRegex(RuntimeError,'moved'):
                deployer.deploy(root)


    def test_packaged_scib_binary_runs_without_compiler_and_checks_source(self):
        import scib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            shutil.copytree(ROOT/'resources/scib',root/'resources/scib')
            package=root/'.envs/cellbench/lib/python3.10/site-packages/scib/knn_graph'
            package.mkdir(parents=True)
            source=package/'knn_graph.cpp'
            shutil.copy2(Path(scib.__file__).parent/'knn_graph/knn_graph.cpp',source)
            with patch.dict('os.environ', {'PATH':'/nonexistent'}):
                deployer.install_scib_binary(root)
                matrix=root/'graph.mtx'
                matrix.write_text('3 3 4\n1 2 1\n2 1 1\n2 3 1\n3 2 1\n')
                prefix=root/'neighbors'
                subprocess.run([str(package/'knn_graph.o'),str(matrix),str(prefix),'2','1','100'],check=True,timeout=30)
            rows=(root/'neighbors_distances_0.txt').read_text().splitlines()
            self.assertEqual([sorted(map(float,row.split(',')[1:])) for row in rows], [[1.,2.],[1.,1.],[1.,2.]])
            source.write_text('changed source')
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                deployer.install_scib_binary(root)
