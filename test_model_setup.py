"""Offline tests for setup recovery, cancellation and verified readiness."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import model_assets as models


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = b'real model bytes for checksum validation'
        self.asset = dict(group='speech', name='Speech recognition', path='speech/model.bin',
                          url='https://example.invalid/model.bin', size=len(self.content),
                          hash_type='sha256', digest=hashlib.sha256(self.content).hexdigest())
        self.manifest = self.root / 'manifest.json'
        self.write_manifest([self.asset])
        self.patch = patch.object(models, 'MANIFEST', self.manifest)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write_manifest(self, assets):
        self.manifest.write_text(json.dumps({'assets': assets}), encoding='utf-8')

    def test_success_is_verified_and_repeat_setup_is_offline(self):
        self.assertEqual(models.status(self.root)['missing'], ['Speech recognition'])
        models.ensure_models(root=self.root, opener=lambda *_a, **_k: Response(self.content))
        self.assertTrue(models.status(self.root)['ready'])
        models.ensure_models(root=self.root, opener=lambda *_a, **_k: self.fail('Unexpected download'))
        path = self.root / '.model-cache/speech/model.bin'
        path.write_bytes(b'broken')
        self.assertFalse(models.status(self.root)['ready'])
        models.ensure_models(root=self.root, opener=lambda *_a, **_k: Response(self.content))
        self.assertEqual(path.read_bytes(), self.content)

    def test_partial_download_resumes_with_range(self):
        partial = self.root / '.model-cache/speech/model.bin.part'
        partial.parent.mkdir(parents=True)
        partial.write_bytes(self.content[:10])
        def opener(request, **_kwargs):
            self.assertEqual(request.get_header('Range'), 'bytes=10-')
            return Response(self.content[10:], 206,
                            {'Content-Range': f'bytes 10-{len(self.content)-1}/{len(self.content)}'})
        models.ensure_models(root=self.root, opener=opener)
        self.assertTrue(models.status(self.root)['ready'])
        self.assertFalse(partial.exists())

    def test_server_ignoring_range_restarts_without_duplicate_bytes(self):
        partial = self.root / '.model-cache/speech/model.bin.part'
        partial.parent.mkdir(parents=True)
        partial.write_bytes(self.content[:10])
        models.ensure_models(root=self.root, opener=lambda *_a, **_k: Response(self.content))
        self.assertEqual(partial.with_suffix('').read_bytes(), self.content)

    def test_bad_checksum_never_becomes_ready(self):
        with patch.object(models.time, 'sleep', lambda _: None):
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                models.ensure_models(root=self.root, opener=lambda *_a, **_k: Response(b'x'*len(self.content)))
        self.assertFalse(models.status(self.root)['ready'])
        self.assertFalse((self.root / '.model-cache/speech/model.bin').exists())

    def test_cancel_keeps_partial_without_publishing_it(self):
        stopped = False
        def progress(message):
            nonlocal stopped
            if message.startswith('Downloading'):
                stopped = True
        with self.assertRaises(models.SetupCancelled):
            models.ensure_models(progress, lambda: stopped, root=self.root,
                                 opener=lambda *_a, **_k: Response(self.content))
        self.assertFalse(models.status(self.root)['ready'])
        self.assertTrue((self.root / '.model-cache/speech/model.bin.part').is_file())
        models.ensure_models(root=self.root, opener=lambda *_a, **_k: self.fail('Complete partial should be reused'))
        self.assertTrue(models.status(self.root)['ready'])

    def test_git_blob_hash_verifies_tokenizers(self):
        self.asset['hash_type'] = 'git_blob_sha1'
        self.asset['digest'] = hashlib.sha1(f'blob {len(self.content)}\0'.encode()+self.content).hexdigest()
        self.write_manifest([self.asset])
        models.ensure_models(root=self.root, opener=lambda *_a, **_k: Response(self.content))
        self.assertTrue(models.status(self.root)['ready'])

    def test_archive_installs_dlls_and_detects_deleted_dependency(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('llama-server.exe', b'engine')
            z.writestr('dependency.dll', b'library')
        content = archive.getvalue()
        asset = dict(self.asset, path='downloads/engine.zip', size=len(content),
                     digest=hashlib.sha256(content).hexdigest(), extract_to='engine',
                     required=['llama-server.exe', 'dependency.dll'])
        self.write_manifest([asset])
        models.ensure_models(root=self.root, opener=lambda *_a, **_k: Response(content))
        dependency = self.root / '.model-cache/engine/dependency.dll'
        self.assertTrue(models.status(self.root)['ready'])
        dependency.unlink()
        self.assertFalse(models.status(self.root)['ready'])
        models.ensure_models(root=self.root, opener=lambda *_a, **_k: self.fail('Archive should be reused'))
        self.assertEqual(dependency.read_bytes(), b'library')

    def test_archive_cannot_escape_installation(self):
        archive = self.root / 'unsafe.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('../outside.txt', b'bad')
        with self.assertRaisesRegex(ValueError, 'Unsafe path'):
            models._extract(dict(extract_to='engine', required=[]), archive,
                            self.root / '.model-cache', lambda: False)
        self.assertFalse((self.root / '.model-cache/outside.txt').exists())

    def test_invalid_receipts_and_unknown_groups_fail_safely(self):
        cache = self.root / '.model-cache'
        cache.mkdir()
        (cache / 'verified-assets.json').write_text('not valid json')
        self.assertFalse(models.status(self.root)['ready'])
        with self.assertRaisesRegex(ValueError, 'Unknown model groups'):
            models.status(self.root, ['typo'])

    def test_setup_lock_blocks_competing_install_and_releases_after_error(self):
        cache = self.root / '.model-cache'
        cache.mkdir()
        with models._setup_lock(cache):
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                with models._setup_lock(cache):
                    self.fail('Competing installer acquired the lock')
        with models._setup_lock(cache):
            pass


if __name__ == '__main__':
    unittest.main()
