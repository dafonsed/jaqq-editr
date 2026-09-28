"""Offline Resolve fallback tests: never contact or modify a real project."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import automatic_cut as editor
from edit_manifest import cut_signature
import review_media


class ResolveFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_temp = tempfile.TemporaryDirectory(prefix='resolve fallback media ')
        cls.source = Path(cls.media_temp.name) / 'source.mp4'
        with av.open(str(cls.source), 'w') as output:
            video = output.add_stream('libx264rgb', rate=30)
            video.width = 32; video.height = 32; video.pix_fmt = 'rgb24'
            video.options = {'crf': '0', 'preset': 'ultrafast'}
            audio = output.add_stream('aac', rate=48000); audio.layout = 'mono'
            for index in range(60):
                frame = av.VideoFrame.from_ndarray(np.full((32,32,3), index, dtype=np.uint8), format='rgb24')
                frame.pts = index; frame.time_base = Fraction(1,30)
                for packet in video.encode(frame):output.mux(packet)
                values = (.1 * np.sin(2*np.pi*440*np.arange(index*1600,(index+1)*1600)/48000)).astype(np.float32)[None,:]
                frame = av.AudioFrame.from_ndarray(values, format='fltp', layout='mono')
                frame.sample_rate = 48000; frame.pts = index*1600; frame.time_base = Fraction(1,48000)
                for packet in audio.encode(frame):output.mux(packet)
            for stream in (video,audio):
                for packet in stream.encode(None):output.mux(packet)

    @classmethod
    def tearDownClass(cls):
        cls.media_temp.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='resolve fallback export ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fuscript = self.root / 'fuscript.exe'; self.fuscript.touch()
        for module, name, value in ((editor,'ROOT',self.root),
                                    (review_media,'ROOT',self.root),
                                    (editor,'FUSCRIPT',self.fuscript)):
            patcher = patch.object(module,name,value); patcher.start(); self.addCleanup(patcher.stop)
        kept = [[0,.5],[1,1.5]]
        self.plan = dict(source=str(self.source),duration=2,fps=30,kept=kept,
                         source_key=editor.key_for(self.source),cut_signature=cut_signature(kept,30),
                         analysis_run_id='test-only',edit_manifest={})

    def outcome(self, output, returncode=0):
        return subprocess.CompletedProcess(['test-only-fuscript'],returncode,stdout=output,stderr='')

    def test_missing_scripting_runtime_never_calls_resolve_and_retains_export(self):
        self.fuscript.unlink()
        with patch.object(editor.subprocess,'run') as run:
            result = editor.send(self.plan)
        run.assert_not_called()
        self.assertTrue(result['manual_import_required'])
        self.assertEqual(result['manual_import_reason'],'missing_runtime')
        self.assertIsNone(result['timeline'])
        self.assertFalse(result['timeline_verified'])
        self.assertFalse(result['render_verified'])
        self.assertFalse(result['verified'])
        self.assertEqual(result['clips'],2)
        self.assertEqual(result['edited_duration'],1)
        folder = Path(result['folder'])
        self.assertTrue((folder/'create_resolve_draft.lua').is_file())
        self.assertTrue((folder/'OPEN IN RESOLVE.txt').is_file())
        self.assertIn('dofile(',result['import_command'])
        exported = json.loads((folder/'review.json').read_text(encoding='utf-8'))
        self.assertFalse(exported['verification']['timeline_verified'])
        self.assertFalse(exported['verification']['rendered_media_verified'])
        saved = json.loads((folder/'automatic-plan.json').read_text(encoding='utf-8'))
        self.assertTrue(saved['manual_import_required'])

    def test_unavailable_api_runs_only_read_only_preflight_then_saves_manual_draft(self):
        with patch.object(editor.subprocess,'run',return_value=self.outcome('AUTO_API_UNAVAILABLE\n')) as run:
            result = editor.send(self.plan)
        self.assertEqual(run.call_count,1)
        self.assertEqual(Path(run.call_args.args[0][-1]).name,'auto-resolve-preflight.lua')
        self.assertTrue(result['manual_import_required'])
        self.assertEqual(result['manual_import_reason'],'api_unavailable')
        self.assertFalse(result['timeline_verified'])

    def test_connected_resolve_without_project_has_distinct_actionable_reason(self):
        with patch.object(editor.subprocess,'run',return_value=self.outcome('AUTO_NO_PROJECT\n')):
            with self.assertRaises(editor.ResolveUnavailable) as raised:
                editor.check_resolve()
        self.assertEqual(raised.exception.reason,'no_project')
        self.assertIn('no project',str(raised.exception))

    def test_preflight_timeout_saves_manual_draft_without_import_attempt(self):
        with patch.object(editor.subprocess,'run',side_effect=subprocess.TimeoutExpired('test-only',15)) as run:
            result = editor.send(self.plan)
        self.assertEqual(run.call_count,1)
        self.assertEqual(result['manual_import_reason'],'timeout')
        self.assertFalse(result['timeline_verified'])

    def test_import_assertion_failure_is_not_disguised_as_manual_success(self):
        with patch.object(editor.subprocess,'run',side_effect=[
                self.outcome('AUTO_READY\n'),self.outcome('assertion failed: clip duration mismatch',1)]) as run:
            with self.assertRaisesRegex(ValueError,'did not confirm the import'):
                editor.send(self.plan)
        self.assertEqual(run.call_count,2)
        self.assertEqual(Path(run.call_args.args[0][-1]).name,'create_resolve_draft.lua')
        folder = next((self.root/'exports').iterdir())
        self.assertTrue((folder/'resolve-output.txt').is_file())
        self.assertFalse((folder/'OPEN IN RESOLVE.txt').exists())
        self.assertFalse((self.root/'analysis'/'last-automatic-cut.json').exists())
        self.assertFalse(json.loads((folder/'review.json').read_text())['verification']['timeline_verified'])

    def test_import_timeout_remains_an_error_after_mutation_may_have_started(self):
        with patch.object(editor.subprocess,'run',side_effect=[
                self.outcome('AUTO_READY\n'),subprocess.TimeoutExpired('test-only-import',120)]):
            with self.assertRaises(subprocess.TimeoutExpired):
                editor.send(self.plan)
        self.assertFalse((self.root/'analysis'/'last-automatic-cut.json').exists())

    def test_nonzero_import_exit_cannot_claim_verification_from_printed_marker(self):
        with patch.object(editor.subprocess,'run',side_effect=[self.outcome('AUTO_READY\n'),
                self.outcome('CUT_REVIEW_OK RETENTION test\nAUTO_VERIFIED 2 30 1\n',37)]):
            with self.assertRaisesRegex(ValueError,'did not confirm the import'):
                editor.send(self.plan)
        self.assertFalse((self.root/'analysis'/'last-automatic-cut.json').exists())

    def test_manual_instructions_prefer_xml_when_present_and_keep_lua_alternative(self):
        (self.root/'Automatic cut.xml').write_text('<xmeml/>',encoding='utf-8')
        details=editor.manual_import_details(self.root,editor.ResolveUnavailable('Test-only unavailable'))
        self.assertEqual(details['import_xml'],str(self.root/'Automatic cut.xml'))
        self.assertIn('File > Import > Timeline',details['import_instructions'])
        self.assertIn('Workspace > Console',details['import_instructions'])

    def test_confirmed_import_sets_timeline_verified_but_never_render_verified(self):
        with patch.object(editor.subprocess,'run',side_effect=[self.outcome('AUTO_READY\n'),
                self.outcome('CUT_REVIEW_OK RETENTION test\nAUTO_VERIFIED 2 30 1\n')]):
            result = editor.send(self.plan)
        self.assertEqual(result['timeline'],'RETENTION test')
        self.assertTrue(result['timeline_verified'])
        self.assertFalse(result['manual_import_required'])
        self.assertFalse(result['render_verified'])
        self.assertFalse(result['verified'])


if __name__=='__main__':unittest.main()
