"""Offscreen proposal review and model-routing UI; no external services."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from app_paths import add_dependencies
add_dependencies()
from PySide6.QtWidgets import QApplication
from ai_review_ui import ReviewDialog
import automatic_review as ui


def proposal(folder):
    decisions=[dict(candidate_id='repair-1',start=1.,end=2.,action='REMOVE',status='accepted',
        reason='A failed opening is replaced by the complete later wording.',confidence=.97,model='gpt-6-sol'),
        dict(candidate_id='uncertain-2',start=3.,end=4.,action='NEEDS_REVIEW',status='review',
        reason='May be intentional repetition.',confidence=.4,model='gpt-6-astra')]
    return dict(source='synthetic.mp4',duration=6.,fps=30.,kept=[[0,1],[2,6]],microphone=0,
        edited_duration=5.,plan_path=str(folder/'plan.json'),ai_plan=dict(decisions=decisions,
        original_decisions=copy.deepcopy(decisions),overrides={},metrics={}))


def validated_stub(original, overrides):
    result=copy.deepcopy(original)
    result['ai_plan']['overrides']=copy.deepcopy(overrides)
    for decision in result['ai_plan']['decisions']:
        change=overrides.get(decision['candidate_id'],{})
        if decision['status']!='accepted' and change.get('enabled'):
            raise ValueError('Uncertain decision remains protected.')
        if change.get('start',1)<1 or change.get('end',2)>2:
            raise ValueError('Outside validated proposal')
        if change.get('enabled') is False:
            decision.update(action='KEEP',status='accepted')
            result['kept']=[[0,6]]
    return result


class ReviewUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        ui.configure_app(cls.app)

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.folder=Path(self.temporary.name)

    def tearDown(self):self.temporary.cleanup()

    def test_restore_reapply_undo_and_invalid_boundary_preserve_proposal(self):
        window=ReviewDialog(proposal(self.folder));window.show();self.app.processEvents()
        with patch('ai_pipeline.rebuild_plan',side_effect=validated_stub):
            window.table.selectRow(0);window.select()
            window.restore_button.click()
            self.assertEqual(window.result['kept'],[[0,6]])
            self.assertTrue(window.apply_button.isEnabled())
            window.apply_button.click()
            self.assertEqual(window.result['kept'],[[0,1],[2,6]])
            window.undo()
            self.assertEqual(window.result['kept'],[[0,6]])
            snapshot=copy.deepcopy(window.result)
            window.start.setValue(.5);window.adjust_button.click()
            self.assertEqual(window.result,snapshot)
            self.assertIn('rejected',window.status.text())
            window.table.selectRow(1);window.select()
            self.assertFalse(window.apply_button.isEnabled())
            self.assertFalse(window.adjust_button.isEnabled())
        screenshot=os.environ.get('AI_REVIEW_UI_SCREENSHOT')
        if screenshot:window.grab().save(screenshot)
        window.close()

    def test_ai_mode_uses_only_speech_model_and_requires_environment_key(self):
        with patch.object(ui,'ROOT',self.folder),patch.object(ui,'model_status',return_value=dict(ready=True,bytes_missing=0)) as status, \
             patch.dict(os.environ,{'OPENAI_API_KEY':''}):
            window=ui.AutomaticWindow();window.source=Path('fixture.mp4')
            self.assertEqual(window.backend.currentData(),'local')
            window.backend.setCurrentIndex(window.backend.findData('openai'))
            self.assertFalse(window.refresh_models()['ready'])
            self.assertIn('OPENAI_API_KEY',window.status.text())
            self.assertEqual(status.call_args.kwargs,{'groups':['speech']})
            with patch.dict(os.environ,{'OPENAI_API_KEY':'test-environment-only'}):
                self.assertTrue(window.refresh_models()['ready'])
            window.backend.setCurrentIndex(window.backend.findData('local'))
            self.assertTrue(window.refresh_models()['ready'])
            self.assertEqual(status.call_args.kwargs,{})
            window.close()

    def test_saved_openai_preference_without_key_starts_local_but_allows_explicit_choice(self):
        source=self.folder/'fixture.mp4'
        source.touch()
        (self.folder/'analysis').mkdir()
        (self.folder/'analysis'/'simple-preferences.json').write_text(json.dumps(
            dict(source=str(source),intensity='balanced',backend='openai')),encoding='utf-8')
        with patch.object(ui,'ROOT',self.folder), \
             patch.object(ui,'probe',return_value=(6.,30.,['Commentary'])), \
             patch.object(ui,'model_status',return_value=dict(ready=True,bytes_missing=0)), \
             patch.dict(os.environ,{'OPENAI_API_KEY':''}):
            window=ui.AutomaticWindow()
            self.assertEqual(window.backend.currentData(),'local')
            self.assertTrue(window.create.isEnabled())
            window.backend.setCurrentIndex(window.backend.findData('openai'))
            self.assertEqual(window.backend.currentData(),'openai')
            self.assertFalse(window.create.isEnabled())
            self.assertIn('OPENAI_API_KEY',window.status.text())
            window.close()

    def test_manual_import_is_saved_without_claiming_resolve_success(self):
        window=ReviewDialog(proposal(self.folder))
        results=[];window.exported.connect(results.append)
        window.sent(dict(window.result,folder=str(self.folder),manual_import_required=True,timeline_verified=False))
        self.assertEqual(results,[])
        self.assertIn('unverified',window.status.text())
        self.assertIn('OPEN IN RESOLVE.txt',window.status.text())
        window.close()

    def test_export_revalidates_current_plan_with_matching_render_qa(self):
        window=ReviewDialog(proposal(self.folder))
        rendered=dict(source_key='test-source',cut_signature='same-cut',review_scope='full',qa={'geometry':'PASS'})
        window.result['review_render']=rendered
        with patch('ai_pipeline.rebuild_plan',side_effect=lambda result,overrides:copy.deepcopy(result)) as rebuilding, \
             patch('ai_review_ui.SendWorker') as sending:
            window.export()
            self.assertEqual(rebuilding.call_args.args[0]['review_render'],rendered)
            self.assertEqual(sending.call_args.args[0]['review_render'],rendered)
        window.send_worker=None;window.close()

    def test_small_launcher_keeps_ai_controls_visible(self):
        with patch.object(ui,'ROOT',self.folder),patch.object(ui,'model_status',return_value=dict(ready=True,bytes_missing=0)), \
             patch.dict(os.environ,{'OPENAI_API_KEY':'test-environment-only'}):
            window=ui.AutomaticWindow();window.resize(440,740)
            window.voice.addItems(['System audio','Commentary microphone'])
            window.voice.show();window.voice_label.show();window.show();self.app.processEvents()
            for widget in (window.backend,window.intensity,window.create,window.details):
                position=widget.mapTo(window.centralWidget(),widget.rect().bottomRight())
                self.assertTrue(window.centralWidget().rect().contains(position))
            screenshot=os.environ.get('AI_LAUNCHER_UI_SCREENSHOT')
            if screenshot:window.grab().save(screenshot)
            window.close()


if __name__=='__main__':unittest.main()
