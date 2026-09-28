"""Offline source-launch setup regressions; never install packages or fetch models."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import bootstrap_runtime as bootstrap


class FakeRunner:
    """Simulate child outcomes while recording executable/argument boundaries."""

    def __init__(self, root, audits=(), *, pip_exit=0, venv_exit=0):
        self.root = root
        self.audits = list(audits)
        self.pip_exit = pip_exit
        self.venv_exit = venv_exit
        self.calls = []

    def __call__(self, args, **kwargs):
        argv = [str(value) for value in args]
        self.calls.append((argv, kwargs))
        if "-m" in argv and argv[argv.index("-m") + 1] == "venv":
            output, status = "fake venv creation\n", self.venv_exit
            if status == 0:
                executable = bootstrap.runtime_python(self.root)
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.touch()
                (self.root / ".venv" / "pyvenv.cfg").write_text(
                    "home = test-base-python\n", encoding="utf-8")
        elif "-m" in argv and argv[argv.index("-m") + 1] == "pip":
            output = "simulated package-index failure\n" if self.pip_exit else "fake packages installed\n"
            status = self.pip_exit
        elif "-c" in argv:
            if not self.audits:
                raise AssertionError("Unexpected additional dependency audit")
            missing = self.audits.pop(0)
            status = int(bool(missing))
            output = json.dumps({"ok": not missing, "missing": missing}) + "\n"
        else:
            raise AssertionError("Unexpected child process: " + repr(argv))
        destination = kwargs.get("stdout")
        if hasattr(destination, "write"):
            destination.write(output)
            destination.flush()
            stdout = None
        else:
            stdout = output
        result = subprocess.CompletedProcess(argv, status, stdout=stdout, stderr="")
        if kwargs.get("check") and status:
            raise subprocess.CalledProcessError(status, argv, output=stdout, stderr="")
        return result

    def commands_for(self, module):
        return [argv for argv, _ in self.calls
                if "-m" in argv and argv[argv.index("-m") + 1] == module]


class LauncherSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cut review setup ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source checkout with spaces"
        self.root.mkdir()
        (self.root / "requirements-runtime.txt").write_text(
            "PySide6\nav\nnumpy\nfaster-whisper\nonnxruntime\nPillow\ntokenizers\n",
            encoding="utf-8")
        self.progress = []

    def existing_runtime(self):
        executable = bootstrap.runtime_python(self.root)
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
        (self.root / ".venv" / "pyvenv.cfg").write_text(
            "home = test-base-python\n", encoding="utf-8")
        return executable

    def ensure(self, runner, **kwargs):
        return bootstrap.ensure_runtime(self.root, runner=runner,
                                        progress=self.progress.append, **kwargs)

    def assert_no_models_or_global_installs(self, runner):
        self.assertFalse((self.root / ".model-cache").exists())
        for argv in runner.commands_for("pip"):
            self.assertEqual(Path(argv[0]), bootstrap.runtime_python(self.root))
            self.assertNotIn("--user", argv)
            self.assertNotIn("--target", argv)
            self.assertIn("-r", argv)
            self.assertEqual(Path(argv[argv.index("-r") + 1]),
                             self.root / "requirements-runtime.txt")
        for argv, kwargs in runner.calls:
            self.assertFalse(kwargs.get("shell", False))
            self.assertNotIn("setup_retention_models.py", " ".join(argv))

    def test_runtime_path_is_inside_source_venv(self):
        expected = self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.assertEqual(bootstrap.runtime_python(self.root), expected)

    def test_healthy_environment_does_not_install_or_recreate(self):
        executable = self.existing_runtime()
        runner = FakeRunner(self.root, audits=[[]])
        self.assertEqual(self.ensure(runner), executable)
        self.assertEqual(runner.commands_for("venv"), [])
        self.assertEqual(runner.commands_for("pip"), [])
        self.assert_no_models_or_global_installs(runner)

    def test_missing_environment_is_created_then_installed_and_rechecked(self):
        runner = FakeRunner(self.root, audits=[["PySide6", "faster_whisper"], []])
        self.assertEqual(self.ensure(runner), bootstrap.runtime_python(self.root))
        creation = runner.commands_for("venv")
        self.assertEqual(len(creation), 1)
        self.assertEqual(Path(creation[0][0]), Path(sys.executable))
        self.assertIn(str(self.root / ".venv"), creation[0])
        self.assertEqual(len(runner.commands_for("pip")), 1)
        self.assertEqual(runner.audits, [])
        self.assertTrue((self.root / "analysis" / "runtime-setup.log").is_file())
        self.assert_no_models_or_global_installs(runner)

    def test_existing_broken_environment_is_repaired_without_recreation(self):
        self.existing_runtime()
        runner = FakeRunner(self.root, audits=[["faster_whisper"], []])
        self.ensure(runner)
        self.assertEqual(runner.commands_for("venv"), [])
        self.assertEqual(len(runner.commands_for("pip")), 1)
        self.assertEqual(runner.audits, [])
        self.assert_no_models_or_global_installs(runner)

    def test_failed_install_stops_and_preserves_diagnostic_log(self):
        self.existing_runtime()
        runner = FakeRunner(self.root, audits=[["PySide6"]], pip_exit=1)
        with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
            self.ensure(runner)
        logfile = self.root / "analysis" / "runtime-setup.log"
        self.assertTrue(logfile.is_file())
        self.assertIn("simulated package-index failure", logfile.read_text(encoding="utf-8"))
        self.assertEqual(len(runner.commands_for("pip")), 1)
        self.assert_no_models_or_global_installs(runner)

    def test_successful_pip_with_broken_native_imports_is_not_ready(self):
        self.existing_runtime()
        runner = FakeRunner(self.root, audits=[["PySide6.QtCore"], ["PySide6.QtCore"]])
        with self.assertRaises(RuntimeError):
            self.ensure(runner)
        self.assertEqual(runner.audits, [])
        self.assertTrue((self.root / "analysis" / "runtime-setup.log").is_file())
        self.assert_no_models_or_global_installs(runner)

    def test_venv_failure_never_runs_pip(self):
        runner = FakeRunner(self.root, venv_exit=1)
        with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
            self.ensure(runner)
        self.assertEqual(runner.commands_for("pip"), [])
        self.assertFalse((self.root / ".model-cache").exists())

    def test_dependency_audit_uses_isolated_child_and_returns_all_failures(self):
        executable = self.existing_runtime()
        runner = FakeRunner(self.root, audits=[["PySide6.QtCore", "faster_whisper"]])
        report = bootstrap.audit_runtime(executable, runner=runner)
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing"], ["PySide6.QtCore", "faster_whisper"])
        self.assertEqual(len(runner.calls), 1)
        argv, kwargs = runner.calls[0]
        self.assertEqual(Path(argv[0]), executable)
        self.assertIn("-I", argv)
        self.assertIn("-c", argv)
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(runner.commands_for("pip"), [])
        self.assertFalse((self.root / "analysis").exists())

    def test_real_audit_detects_installed_module_that_cannot_load_native_library(self):
        # Having a package file is insufficient: native imports can still fail.
        module = self.root / "broken_native_fixture.py"
        module.write_text("raise ImportError('DLL load failed: synthetic regression fixture')\n",
                          encoding="utf-8")
        probes = {
            "healthy": "import math; assert math.sqrt(4) == 2",
            "native package": "import sys; sys.path.insert(0, " + repr(str(self.root)) +
                              "); import broken_native_fixture",
        }
        with patch.object(bootstrap, "PROBES", probes):
            report = bootstrap.audit_runtime(sys.executable)
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing"], ["native package"])
        self.assertIn("DLL load failed", report["errors"]["native package"])

    def test_check_cli_on_clean_source_is_read_only_from_unrelated_directory(self):
        script = self.root / "bootstrap_runtime.py"
        shutil.copyfile(Path(bootstrap.__file__), script)
        shutil.copyfile(Path(bootstrap.__file__).with_name('launch_status.py'),
                        self.root / 'launch_status.py')
        before = set(self.root.rglob("*"))
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        result = subprocess.run([sys.executable, "-I", "-B", str(script), "--check"],
                                cwd=self.temp.name, env=environment, capture_output=True,
                                text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["ok"])
        self.assertTrue(report["missing"])
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertFalse((self.root / ".venv").exists())
        self.assertFalse((self.root / "analysis").exists())
        self.assertFalse((self.root / ".model-cache").exists())

    def run_fixture_launcher(self, application):
        script = self.root / "launch_cut_review.py"
        shutil.copyfile(Path(bootstrap.__file__).with_name(script.name), script)
        shutil.copyfile(Path(bootstrap.__file__).with_name('launch_status.py'),
                        self.root / 'launch_status.py')
        (self.root / "automatic_review.py").write_text(application, encoding="utf-8")
        return subprocess.run([sys.executable, "-I", "-B", str(script), "--launch-check"],
                              cwd=self.temp.name, capture_output=True, text=True,
                              encoding="utf-8", timeout=20)

    def test_gui_launch_check_preserves_application_failure_exit_code(self):
        result = self.run_fixture_launcher("raise SystemExit(7)\n")
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_gui_launch_check_reports_import_failure_without_modal_dialog(self):
        result = self.run_fixture_launcher(
            "raise ModuleNotFoundError(\"No module named 'PySide6'\")\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("START CUT REVIEW.cmd", result.stderr)
        logfile = self.root / "analysis" / "launch-error.log"
        self.assertTrue(logfile.is_file())
        self.assertIn("ModuleNotFoundError", logfile.read_text(encoding="utf-8"))

    def test_normal_bootstrap_rejects_native_style_exit_and_keeps_stderr(self):
        # An immediate native-style exit bypasses Python exception handling.
        # The bootstrap must fail instead of returning success after Popen.
        (self.root / 'launch_cut_review.py').write_text(
            "import os, sys\nos.write(2, b'native startup failure fixture\\n')\nos._exit(37)\n",
            encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'exited during startup'):
            bootstrap.start_application(sys.executable, self.root, timeout=5)
        log = (self.root / 'analysis' / 'application.log').read_text(encoding='utf-8')
        self.assertIn('native startup failure fixture', log)

    def test_normal_bootstrap_requires_gui_ack_even_when_parent_stays_alive(self):
        class AliveParent:
            def poll(self):
                return None
        with self.assertRaisesRegex(RuntimeError, 'did not confirm a visible window'):
            bootstrap.wait_for_startup(AliveParent(), self.root / 'absent-status.json',
                                       self.root / 'application.log', timeout=.05)

    def test_normal_bootstrap_rejects_exit_after_ready_signal(self):
        class DeadProcess:
            def poll(self):
                return 37
        status = self.root / 'status.json'
        status.write_text(json.dumps(dict(state='ready', visible=True)), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'exited during startup'):
            bootstrap.wait_for_startup(DeadProcess(), status, self.root / 'application.log',
                                       timeout=.1, settle_seconds=0)

    def test_existing_instance_requires_visible_acknowledgement(self):
        class ExitedProcess:
            def poll(self):
                return 0
        status = self.root / 'status.json'
        status.write_text(json.dumps(dict(state='existing', visible=True, existing_pid=123)),
                          encoding='utf-8')
        report = bootstrap.wait_for_startup(ExitedProcess(), status,
                                           self.root / 'application.log', timeout=.1)
        self.assertEqual(report['existing_pid'], 123)
        status.write_text(json.dumps(dict(state='existing', visible=False)), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'exited during startup'):
            bootstrap.wait_for_startup(ExitedProcess(), status, self.root / 'application.log',
                                       timeout=.1)

    def test_managed_venv_packages_win_over_legacy_runtime_folder(self):
        import app_paths

        package = "retention_dependency_precedence_fixture"
        managed = self.root / ".venv" / "Lib" / "site-packages"
        legacy = self.root / ".runtime-deps"
        for folder in (managed, legacy):
            folder.mkdir(parents=True)
            (folder / (package + ".py")).write_text("VALUE = 'fixture'\n", encoding="utf-8")
        with patch.object(app_paths, "ROOT", self.root), \
             patch.object(sys, "prefix", str(self.root / ".venv")), \
             patch.object(sys, "base_prefix", str(self.root / "base-python")), \
             patch.object(sys, "executable", str(bootstrap.runtime_python(self.root))), \
             patch.object(sys, "path", [str(managed)] + list(sys.path)), \
             patch.object(sys, "frozen", False, create=True):
            app_paths.add_dependencies()
            spec = importlib.util.find_spec(package)
            self.assertIsNotNone(spec)
            self.assertEqual(Path(spec.origin), managed / (package + ".py"))


if __name__ == "__main__":
    unittest.main()
