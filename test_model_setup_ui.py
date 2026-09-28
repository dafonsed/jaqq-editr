"""Offline regressions for first-run setup, retry, and safe worker shutdown."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
import automatic_review as ui


class ModelSetupUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.root=patch.object(ui,'ROOT',Path(self.folder.name));self.root.start()
        self.readiness=patch.object(ui,'model_status',return_value=dict(ready=False,bytes_missing=3_800_000_000))
        self.status=self.readiness.start()
        self.window=ui.AutomaticWindow()
        self.window.backend.setCurrentIndex(self.window.backend.findData('local'))
        self.window.show();self.app.processEvents()

    def tearDown(self):
        self.window.setup_worker=None;self.window.worker=None;self.window.close()
        self.root.stop();self.readiness.stop();self.folder.cleanup()

    def test_missing_models_cannot_start_edit(self):
        self.window.source=Path('fixture.mp4')
        self.window.refresh_models()
        with patch.object(ui,'Worker') as worker:
            self.window.begin()
            worker.assert_not_called()
        self.assertFalse(self.window.create.isEnabled())
        self.assertTrue(self.window.setup_button.isVisible())
        self.assertIn('3.8 GB',self.window.status.text())

    def test_setup_failure_leaves_window_open_with_retry(self):
        with patch.object(ui,'ensure_models',side_effect=OSError('offline')):
            self.window.start_model_setup()
            for _ in range(100):
                QTest.qWait(10)
                if not self.window.setup_worker.isRunning():break
            self.app.processEvents()
        self.assertTrue(self.window.isVisible())
        self.assertTrue(self.window.setup_button.isEnabled())
        self.assertEqual(self.window.setup_button.text(),'Retry setup')
        self.assertIn('offline',self.window.status.text())
        self.assertFalse(self.window.create.isEnabled())

    def test_retry_can_finish_setup_and_enable_selected_recording(self):
        self.window.source=Path('fixture.mp4')
        with patch.object(ui,'ensure_models',return_value=dict(ready=True)):
            self.window.start_model_setup()
            for _ in range(100):
                QTest.qWait(10)
                if not self.window.setup_worker.isRunning():break
            self.app.processEvents()
        self.assertTrue(self.window.isVisible())
        self.assertTrue(self.window.create.isEnabled())
        self.assertFalse(self.window.setup_button.isVisible())

    def test_close_waits_for_download_cancellation(self):
        running=Mock(return_value=True);cancel=Mock()
        self.window.setup_worker=SimpleNamespace(isRunning=running,requestInterruption=cancel)
        self.window.close();self.app.processEvents()
        cancel.assert_called_once()
        self.assertTrue(self.window.isVisible())
        running.return_value=False
        self.window.model_setup_finished();self.app.processEvents()
        self.assertFalse(self.window.isVisible())


if __name__=='__main__':unittest.main()
