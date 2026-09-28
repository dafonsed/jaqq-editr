"""Regression checks for evidence preservation, settings routing and export claims."""
from app_paths import add_dependencies
add_dependencies()
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from speech_evidence import audit, output_locations
from automatic_cut import select_ranges, plan, send, key_for
from edit_manifest import build_manifest, cut_signature


def write_audio(path, samples):
    path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),'wb') as handle:
        handle.setparams((1,2,16000,0,'NONE','not compressed'))
        handle.writeframes(np.asarray(samples,dtype='<i2').tobytes())


class PipelineAuditTests(unittest.TestCase):
    def test_untranscribed_activity_is_retained_without_gameplay(self):
        with tempfile.TemporaryDirectory() as temp:
            audio=np.zeros(48000,dtype=np.int16);audio[16000:24000]=2000
            path=Path(temp)/'audio.wav';write_audio(path,audio)
            protected,flags=audit([],path,3)
            self.assertTrue(flags)
            kept,_,_=select_ranges([],[],3,protected=protected)
            self.assertTrue(any(a<=1 and b>=1.5 for a,b in kept))
            self.assertIn('possible omitted speech',flags[0]['reason'])

    def test_low_confidence_spans_and_zero_alignment_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'audio.wav';write_audio(path,np.zeros(64000))
            transcript=[dict(start=1,end=2,text='maybe speech',words=[
                dict(start=1,end=1,text='maybe',probability=.9),
                dict(start=1,end=2,text='speech',probability=.2)])]
            protected,flags=audit(transcript,path,4)
            self.assertEqual(protected,[[1,2]])
            self.assertEqual(flags[0]['confidence'],'uncertain')

    def test_protected_context_overrides_text_deletion(self):
        def sentence(start):
            return dict(words=[dict(start=start+i*.2,end=start+i*.2+.15,text=' '+x)
                for i,x in enumerate('This is a complete repeated sentence.'.split())])
        kept,_,removed=select_ranges([sentence(1),sentence(5)],[],10,protected=[(1,2.3)])
        self.assertEqual(removed,[])
        self.assertTrue(any(a<=1 and b>=2.3 for a,b in kept))

    def test_report_maps_deleted_and_surviving_spans(self):
        rows=output_locations([dict(source_start=1,source_end=6)],[[2,3],[5,7]])
        self.assertEqual(rows[0]['output_spans'],[
            dict(source_start=2,source_end=3,output_start=0,output_end=1),
            dict(source_start=5,source_end=6,output_start=1,output_end=2)])
        manifest=build_manifest([[2,3],[5,7]],10,30,[dict(start=3,end=5,reason='Restart gap')])
        self.assertEqual(manifest['output_frames'],90)
        self.assertEqual(manifest['removed'][1]['output_join'],1)
        self.assertEqual(manifest['verification']['render'],'NOT VERIFIED')

    def test_settings_reach_engine_and_final_audit_preserves_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.mp4';source.touch()
            model=root/'.model-cache'/'speech-small.en';model.mkdir(parents=True)
            for filename in ('model.bin','config.json','tokenizer.json'):(model/filename).touch()
            class Model:
                def __init__(self,*args,**kwargs):pass
                def transcribe(self,*args,**kwargs):
                    return iter([SimpleNamespace(start=1,end=2,text=' Hello.',words=[
                        SimpleNamespace(start=1,end=2,word=' Hello.',probability=.95)])]),None
            def extract(source,mic,start,end,path):write_audio(path,np.zeros(round((end-start)*16000)))
            def review(transcript,kept,*args,**kwargs):
                flag=[] if kwargs.get('audit_only') else [dict(source_start=1,source_end=2,
                    reason='Uncertain contextual comparison')]
                return kept,dict(script=[],changes=[],flags=flag,semantic=dict(status='tested stub'))
            with patch('automatic_cut.ROOT',root),patch('automatic_cut.probe',return_value=(4,30,['Mic'])), \
                 patch('automatic_cut.extract',side_effect=extract), \
                 patch.dict('sys.modules',{'faster_whisper':SimpleNamespace(WhisperModel=Model)}), \
                 patch('script_review.review_script',side_effect=review), \
                 patch('pause_cleanup.clean_pauses',side_effect=lambda ranges,*args,**kwargs:(ranges,[])) as pauses:
                result=plan(source,0,intensity='tight')
                self.assertEqual(result['settings']['intensity'],'tight')
                self.assertEqual(pauses.call_args.kwargs['settings']['intensity'],'tight')
                self.assertIn('Uncertain contextual',result['script_review']['flags'][0]['reason'])
                self.assertTrue(result['script_review']['flags'][0]['output_spans'])
                self.assertTrue((root/'analysis').is_dir())

    def test_changed_plan_or_source_never_reaches_export(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source.mp4';source.touch()
            result=dict(source=str(source),source_key=key_for(source),duration=5,fps=30,kept=[[1,2]],
                cut_signature=cut_signature([[1,2]],30))
            with patch('review_media.audio_sidecars') as export:
                changed=dict(result,kept=[[1,3]])
                with self.assertRaisesRegex(ValueError,'changed after review'):send(changed)
                source.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'recording changed'):send(result)
                export.assert_not_called()

    def test_semantic_failure_prevents_successful_plan(self):
        # Runtime startup behavior is explicit; no silence-only success fallback.
        from local_editor import Editor
        with tempfile.TemporaryDirectory() as temp,patch('local_editor.ROOT',Path(temp)):
            with self.assertRaisesRegex(ValueError,'model is missing'):
                with Editor():pass


if __name__=='__main__':unittest.main()
