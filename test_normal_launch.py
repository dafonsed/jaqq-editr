"""Persistent Windows launch integration using the ordinary startup protocol.

Run with the installed project interpreter. This creates an isolated source
checkout, opens the real pythonw GUI, waits for the bootstrap to exit, observes
the GUI's continued lifetime, and verifies a second ordinary launch reaches it.
There is no --launch-check flag, automatic quit timer, UI automation, model
download, or Resolve dependency. Only the process created by this test is ended.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parent


@unittest.skipUnless(os.name == 'nt', 'Windows pythonw integration')
class NormalLaunchTests(unittest.TestCase):
    def test_normal_window_survives_bootstrap_and_second_launch_reaches_it(self):
        python = ROOT / '.venv' / 'Scripts' / 'python.exe'
        self.assertTrue(python.with_name('pythonw.exe').is_file(),
                        'Install the project runtime before this integration check')
        report_dir = ROOT / 'analysis' / 'normal-launch-verification'
        report_dir.mkdir(parents=True, exist_ok=True)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        with tempfile.TemporaryDirectory(prefix='normal launch with spaces ') as temporary:
            checkout = Path(temporary) / 'source'
            checkout.mkdir()
            # A unique ROOT has its own singleton namespace and preferences;
            # an editor already opened by the user cannot be reached by this test.
            for source in ROOT.glob('*.py'):
                if not source.name.startswith('test_'):
                    shutil.copy2(source, checkout / source.name)
            shutil.copy2(ROOT / 'model-assets.json', checkout / 'model-assets.json')
            environment = dict(os.environ)
            for name in ('PYTHONPATH', 'PYTHONHOME', 'QT_QPA_PLATFORM',
                         'RETENTION_STARTUP_STATUS'):
                environment.pop(name, None)
            environment['PYTHONNOUSERSITE'] = '1'
            # Only runtime provisioning is replaced: the isolated source copy
            # reuses the already installed interpreter, avoiding package downloads.
            code = ('import sys; from pathlib import Path; sys.path.insert(0, '
                    + repr(str(checkout)) + '); import bootstrap_runtime as b; '
                    + 'b.ensure_runtime=lambda *args, **kwargs: Path('
                    + repr(str(python)) + '); raise SystemExit(b.main())')
            command = [str(python), '-I', '-c', code]

            def launch():
                previous = set((checkout / 'analysis').glob('startup-*.json'))
                outcome = subprocess.run(command, cwd=os.environ['WINDIR'], env=environment,
                                         capture_output=True, text=True, encoding='utf-8',
                                         errors='replace', timeout=60)
                self.assertEqual(outcome.returncode, 0, outcome.stdout + outcome.stderr)
                created = set((checkout / 'analysis').glob('startup-*.json')) - previous
                self.assertEqual(len(created), 1)
                return json.loads(created.pop().read_text(encoding='utf-8'))

            owned_handle = None
            try:
                first = launch()
                self.assertEqual(first['state'], 'ready')
                self.assertTrue(first['visible'])
                owned_pid = first['pid']
                owned_handle = kernel32.OpenProcess(0x100001, False, owned_pid)
                self.assertTrue(owned_handle, 'Could not observe the test-created GUI process')
                seconds = max(5, float(os.environ.get('RETENTION_LAUNCH_OBSERVE_SECONDS', '12')))
                deadline = time.monotonic() + seconds
                samples = 0
                while time.monotonic() < deadline:
                    self.assertEqual(kernel32.WaitForSingleObject(owned_handle, 0), 258,
                                     'GUI auto-closed after its bootstrap returned success')
                    samples += 1
                    time.sleep(.5)
                second = launch()
                self.assertEqual(second['state'], 'existing')
                self.assertEqual(second['existing_pid'], owned_pid)
                self.assertTrue(second['visible'])
                self.assertEqual(kernel32.WaitForSingleObject(owned_handle, 0), 258)
                (report_dir / 'result.json').write_text(json.dumps({
                    'normal_pythonw': True, 'launch_check_argument': False,
                    'bootstrap_exited_before_observation': True,
                    'observed_seconds': seconds, 'alive_samples': samples,
                    'singleton_acknowledged_visible_window': True,
                    'test_pid': owned_pid,
                }, indent=2), encoding='utf-8')
            finally:
                if owned_handle:
                    # This handle identifies exactly the GUI spawned above,
                    # even if its PID could later be reused by another process.
                    self.assertTrue(kernel32.TerminateProcess(owned_handle, 0))
                    self.assertEqual(kernel32.WaitForSingleObject(owned_handle, 5000), 0)
                    kernel32.CloseHandle(owned_handle)
                diagnostic = checkout / 'analysis' / 'application.log'
                if diagnostic.exists():
                    shutil.copy2(diagnostic, report_dir / 'pythonw.log')
                    # The venv redirector releases inherited log handles just
                    # after its real interpreter exits. Wait for that cleanup.
                    deadline = time.monotonic() + 5
                    while True:
                        try:
                            diagnostic.rename(diagnostic.with_name('closed-application.log'))
                            break
                        except PermissionError:
                            if time.monotonic() >= deadline:
                                raise
                            time.sleep(.05)


if __name__ == '__main__':
    unittest.main()
