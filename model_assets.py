"""Install the editor's existing local models from pinned, verified public files.

This module uses only the standard library, so setup status is available before
inference libraries are imported. ``ensure_models`` belongs on a worker thread.
It never uploads recordings or transcripts. Partial downloads resume on retry.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.error
import urllib.request
import zipfile

from app_paths import ROOT

MANIFEST = Path(__file__).with_name('model-assets.json')


class SetupCancelled(Exception):
    """The user stopped setup; completed downloads remain available."""


def _assets(groups=None):
    assets = json.loads(MANIFEST.read_text(encoding='utf-8'))['assets']
    if groups is not None:
        groups = set(groups)
        unknown = groups - {a['group'] for a in assets}
        if unknown:
            raise ValueError('Unknown model groups: ' + ', '.join(sorted(unknown)))
        assets = [a for a in assets if a['group'] in groups]
    return assets


def _destination(cache, relative):
    path = (cache / relative).resolve()
    if not path.is_relative_to(cache.resolve()):
        raise ValueError('Model path is outside the model cache')
    return path


def _fingerprint(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _receipts(cache):
    try:
        data = json.loads((cache / 'verified-assets.json').read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _installed(asset, cache, receipts):
    receipt = receipts.get(asset['path'], {})
    if not isinstance(receipt, dict) or receipt.get('digest') != asset['digest']:
        return False
    try:
        if _fingerprint(_destination(cache, asset['path'])) != receipt.get('fingerprint'):
            return False
        if asset.get('extract_to'):
            files = receipt.get('extracted', {})
            if not files or not all(name in files for name in asset['required']):
                return False
            for name, fingerprint in files.items():
                if _fingerprint(_destination(cache, asset['extract_to'] + '/' + name)) != fingerprint:
                    return False
        return True
    except (OSError, ValueError):
        return False


def status(root=ROOT, groups=None):
    """Return quick disk-only readiness; unverified or changed files need setup."""
    cache = Path(root) / '.model-cache'
    assets = _assets(groups)
    receipts = _receipts(cache)
    missing_assets = [a for a in assets if not _installed(a, cache, receipts)]
    return dict(ready=not missing_assets,
                missing=list(dict.fromkeys(a['name'] for a in missing_assets)),
                missing_groups=list(dict.fromkeys(a['group'] for a in missing_assets)),
                missing_files=[a['path'] for a in missing_assets],
                bytes_total=sum(a['size'] for a in assets),
                bytes_missing=sum(a['size'] for a in missing_assets))


def required_paths(root=ROOT, groups=None):
    """Return model files used by the selected runtime components."""
    cache = Path(root) / '.model-cache'
    paths = []
    for asset in _assets(groups):
        if asset.get('extract_to'):
            paths.extend(_destination(cache, asset['extract_to'] + '/' + name)
                         for name in asset['required'])
        else:
            paths.append(_destination(cache, asset['path']))
    return paths


def _check(cancel):
    if cancel():
        raise SetupCancelled('Model setup stopped. Downloaded files will be reused.')


def _valid(path, asset, cancel):
    try:
        if path.stat().st_size != asset['size']:
            return False
        digest = hashlib.sha256() if asset['hash_type'] == 'sha256' else hashlib.sha1()
        if asset['hash_type'] == 'git_blob_sha1':
            digest.update(f"blob {asset['size']}\0".encode())
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                _check(cancel)
                digest.update(chunk)
        return digest.hexdigest() == asset['digest']
    except OSError:
        return False


def _download(asset, path, progress, cancel, opener):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    for attempt in range(3):
        _check(cancel)
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset >= asset['size']:
                if offset == asset['size'] and _valid(partial, asset, cancel):
                    partial.replace(path)
                    return
                partial.unlink()
                offset = 0
            if shutil.disk_usage(path.parent).free < asset['size'] - offset + 10_000_000:
                raise RuntimeError(f"Not enough disk space to finish {asset['name']}. Free some space and select Retry setup.")
            headers = {'User-Agent': 'JAQQ-Editor-Model-Setup/1', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(asset['url'], headers=headers)
            with opener(request, timeout=15) as response:
                if offset and response.status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith(f'bytes {offset}-'):
                        partial.unlink()
                        raise OSError('Download server returned an incorrect resume offset')
                elif response.status == 200:
                    offset = 0
                else:
                    raise OSError(f'Unexpected download response: {response.status}')
                last_report = 0.0
                read_chunk = getattr(response, 'read1', response.read)
                with partial.open('ab' if offset else 'wb') as stream:
                    while True:
                        _check(cancel)
                        chunk = read_chunk(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        offset += len(chunk)
                        if offset > asset['size']:
                            raise OSError('Downloaded file is larger than the pinned model asset')
                        now = time.monotonic()
                        if now - last_report >= .5:
                            progress(f"Downloading {asset['name']}: {offset / 1e6:.0f} / {asset['size'] / 1e6:.0f} MB")
                            last_report = now
            progress(f"Verifying {asset['name']}…")
            if not _valid(partial, asset, cancel):
                if partial.stat().st_size >= asset['size']:
                    partial.unlink()
                raise OSError('Downloaded model is incomplete or its checksum does not match')
            _check(cancel)
            partial.replace(path)
            return
        except (OSError, urllib.error.URLError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code == 416 and partial.is_file():
                partial.unlink()
            if attempt == 2:
                raise RuntimeError(f"Could not download {asset['name']}. Check your connection and select Retry setup. {error}") from error
            progress(f"Retrying {asset['name']} ({attempt + 2}/3)…")
            for _ in range(10 * (attempt + 1)):
                _check(cancel)
                time.sleep(.1)


def _extract(asset, archive_path, cache, cancel):
    destination = _destination(cache, asset['extract_to'])
    destination.mkdir(parents=True, exist_ok=True)
    extracted = {}
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            _check(cancel)
            name = PurePosixPath(entry.filename.replace('\\', '/'))
            if name.is_absolute() or '..' in name.parts or ':' in entry.filename:
                raise ValueError('Unsafe path in model engine archive')
            if entry.is_dir():
                continue
            target = _destination(destination, str(name))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.part')
            with archive.open(entry) as source, temporary.open('wb') as output:
                while chunk := source.read(1024 * 1024):
                    _check(cancel)
                    output.write(chunk)
            temporary.replace(target)
            extracted[str(name)] = _fingerprint(target)
    if not all(name in extracted for name in asset['required']):
        raise ValueError('Downloaded engine archive is missing required runtime files')
    return extracted


@contextmanager
def _setup_lock(cache):
    stream = (cache / 'setup.lock').open('a+b')
    try:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Model setup is already running in another editor window. Wait for that setup to finish.') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
    finally:
        stream.close()


def ensure_models(progress=lambda message: None, cancel=lambda: False, *, root=ROOT,
                  groups=None, opener=urllib.request.urlopen):
    """Download, verify and install required models; safe to retry after failure.

    ``progress`` receives readable text. ``cancel`` is checked between downloads,
    during streaming/hashing, and during extraction. A socket can take at most
    its 15-second read timeout to yield to cancellation.
    """
    cache = Path(root) / '.model-cache'
    cache.mkdir(parents=True, exist_ok=True)
    assets = _assets(groups)
    with _setup_lock(cache):
        receipts = _receipts(cache)
        missing = [a for a in assets if not _installed(a, cache, receipts)]
        needed = sum(max(0, a['size'] - (_destination(cache, a['path'] + '.part').stat().st_size
                     if _destination(cache, a['path'] + '.part').is_file() else 0)) for a in missing
                     if not _destination(cache, a['path']).is_file())
        if missing and shutil.disk_usage(cache).free < needed + 100_000_000:
            raise RuntimeError(f'Model setup needs approximately {(needed + 100_000_000) / 1e9:.1f} GB of free disk space.')
        for number, asset in enumerate(assets, 1):
            _check(cancel)
            if _installed(asset, cache, receipts):
                continue
            path = _destination(cache, asset['path'])
            progress(f"Setting up {asset['name']} ({number}/{len(assets)})…")
            if not _valid(path, asset, cancel):
                _download(asset, path, progress, cancel, opener)
            receipt = dict(digest=asset['digest'], fingerprint=_fingerprint(path))
            if asset.get('extract_to'):
                progress('Installing the local editorial engine…')
                receipt['extracted'] = _extract(asset, path, cache, cancel)
            receipts[asset['path']] = receipt
            temporary = cache / 'verified-assets.json.part'
            temporary.write_text(json.dumps(receipts, indent=2), encoding='utf-8')
            temporary.replace(cache / 'verified-assets.json')
        _check(cancel)
    result = status(root, groups)
    if not result['ready']:
        raise RuntimeError('Model setup is incomplete: ' + ', '.join(result['missing']))
    progress('Local AI models are ready.')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true', help='Report installed model readiness without network access')
    parser.add_argument('--groups', nargs='+', choices=['speech', 'semantic', 'entailment', 'editor', 'retention'])
    arguments = parser.parse_args()
    if arguments.status:
        print(json.dumps(status(groups=arguments.groups), indent=2))
    else:
        ensure_models(lambda message: print(message, flush=True), groups=arguments.groups)
