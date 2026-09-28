"""Deterministic waveform regressions. These do not stand in for listening tests."""
import tempfile
import unittest
import wave
from pathlib import Path

from app_paths import add_dependencies
add_dependencies()
import numpy as np

from cut_integrity import protect_words, validate
from pause_cleanup import audit_pauses, bridge_word_gaps, clean_pauses, pause_settings
from speech_edges import microphone_energy, refine_edges


class PauseBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wav=Path(self.temp.name)/'microphone.wav'
        self.rate=16000
        self.transcript=[dict(words=[dict(start=1.,end=2.,text=' First.',probability=.95),
                                    dict(start=4.,end=5.,text=' Second.',probability=.95)])]

    def write(self,noise=.003,quiet_tail=False,duration=6.):
        time=np.arange(round(duration*self.rate))
        signal=np.sin(time*.2)*noise
        for a,b in ((1,2),(4,5)):
            lo,hi=round(a*self.rate),min(len(signal),round(b*self.rate))
            signal[lo:hi]=np.sin(time[lo:hi]*.2)*.15
        if quiet_tail:
            lo,hi=round(1.65*self.rate),round(2*self.rate)
            signal[lo:hi]=np.sin(time[lo:hi]*.2)*.0003
        with wave.open(str(self.wav),'wb') as f:
            f.setparams((1,2,self.rate,0,'NONE','not compressed'))
            f.writeframes((signal*32767).astype('<i2').tobytes())

    def test_room_tone_no_longer_blocks_supported_pause_cut(self):
        self.write()
        kept,cuts=clean_pauses([[0,6]],self.wav,transcript=self.transcript)
        self.assertEqual([r['pause_type'] for r in cuts],['leading','sentence','trailing'])
        self.assertAlmostEqual(sum(b-a for a,b in kept),2+.075+.45+.12,places=6)
        self.assertTrue(all(row['confidence']==.8 for row in cuts))
        self.assertAlmostEqual(cuts[1]['output_time'],2.276923076923077-.925)

    def test_intensity_and_context_have_distinct_budgets(self):
        self.write(noise=0)
        lengths=[]
        for intensity in ('natural','balanced','tight'):
            kept,_=clean_pauses([[0,6]],self.wav,transcript=self.transcript,settings=pause_settings(intensity))
            lengths.append(sum(b-a for a,b in kept))
        self.assertGreater(lengths[0],lengths[1])
        self.assertGreater(lengths[1],lengths[2])
        self.transcript[0]['words'][0]['text']=' First'
        _,cuts=clean_pauses([[0,6]],self.wav,transcript=self.transcript)
        self.assertEqual(cuts[1]['pause_type'],'hesitation')
        self.transcript[0]['words'][0]['pause_after']='topic'
        _,cuts=clean_pauses([[0,6]],self.wav,transcript=self.transcript)
        self.assertEqual(cuts[1]['pause_type'],'topic')

    def test_quiet_word_ending_is_protected(self):
        self.write(quiet_tail=True)
        kept,cuts=clean_pauses([[0,6]],self.wav,transcript=self.transcript)
        self.assertTrue(any(a<=1 and b>=2 for a,b in kept))
        self.assertEqual(cuts[1]['pause_type'],'sentence')
        kept,_=refine_edges([[.6,2.3]],self.wav,transcript=self.transcript)
        self.assertGreaterEqual(kept[0][1],2.12)

    def test_waveform_only_gap_retains_handles_on_both_sides(self):
        self.write(noise=0)
        kept,cuts=clean_pauses([[0,6]],self.wav)
        internal=next(r for r in cuts if r['pause_type']=='hesitation')
        self.assertGreater(internal['start'],2.1)
        self.assertLess(internal['end'],3.95)

    def test_visual_and_intentional_pauses_are_exempt(self):
        self.write(noise=0)
        for arguments in ({'events':[(2,4)]},
                          {'protected_pauses':[dict(start=2,end=4,reason='Let the result land')]}):
            kept,_=clean_pauses([[0,6]],self.wav,transcript=self.transcript,**arguments)
            self.assertTrue(any(a<=2 and b>=4 for a,b in kept))
            flags=audit_pauses(kept,self.wav,transcript=self.transcript,**arguments)
            self.assertTrue(any(f['status']=='protected' for f in flags))

    def test_uncertain_room_tone_gap_requires_review(self):
        self.write()
        for word in self.transcript[0]['words']:word['probability']=.1
        kept,cuts=clean_pauses([[0,6]],self.wav,transcript=self.transcript)
        self.assertEqual(kept,[[0,6]])
        self.assertFalse(cuts)
        flags=audit_pauses(kept,self.wav,transcript=self.transcript)
        self.assertTrue(any(f['start']==2 and f['end']==4 and f['status']=='review' for f in flags))

    def test_a_short_event_cannot_excuse_an_entire_long_gap(self):
        self.write()
        flags=audit_pauses([[0,6]],self.wav,events=[(2,2.2)],transcript=self.transcript)
        self.assertTrue(any(f['start']==2.2 and f['end']==4 and f['status']=='review' for f in flags))

    def test_settings_handles_survive_integrity_pass(self):
        self.write()
        settings=pause_settings('tight')
        kept,_=clean_pauses([[0,6]],self.wav,transcript=self.transcript,settings=settings)
        final,_=protect_words(kept,self.transcript,[],6,60,
                              pre_roll=settings['leading'],post_roll=settings['trailing'])
        self.assertLessEqual(sum(b-a for a,b in final)-sum(b-a for a,b in kept),4/60+.0001)
        self.assertFalse(audit_pauses(final,self.wav,transcript=self.transcript,settings=settings))

    def test_preselected_word_gap_can_use_natural_sentence_budget(self):
        self.write()
        selected=[[.925,2.12],[3.925,5.12]]
        bridged,evidence=bridge_word_gaps(selected,self.transcript)
        self.assertEqual(bridged,[[.925,5.12]])
        self.assertTrue(evidence)
        kept,_=clean_pauses(bridged,self.wav,transcript=self.transcript,settings=pause_settings('natural'))
        self.assertAlmostEqual((kept[0][1]-2)+(4-kept[1][0]),.65)

    def test_bridge_cannot_restore_rejected_or_unreliable_speech(self):
        selected=[[.925,2.12],[3.925,5.12]]
        self.assertEqual(bridge_word_gaps(selected,self.transcript,exclusions=[(2.2,3.5)])[0],selected)
        self.assertEqual(bridge_word_gaps(selected,self.transcript,protected_pauses=[(2.2,3.5)])[0],selected)
        self.transcript[0]['words'][0]['probability']=.1
        self.assertEqual(bridge_word_gaps(selected,self.transcript)[0],selected)
        self.transcript[0]['words'][0]['probability']=.95
        self.transcript[0]['words'].append(dict(start=2.5,end=3,text=' unknown'))
        self.assertEqual(bridge_word_gaps(selected,self.transcript)[0],selected)

    def test_interior_rejection_is_enforced_by_all_boundary_stages(self):
        self.write(noise=0)
        exclusion=[(1.2,1.8)]
        for kept in (protect_words([[0,6]],[],exclusion,6,30)[0],
                     refine_edges([[0,6]],self.wav,exclusions=exclusion)[0],
                     clean_pauses([[0,6]],self.wav,exclusions=exclusion)[0]):
            self.assertFalse(any(a<1.8 and b>1.2 for a,b in kept))

    def test_rejected_word_tail_does_not_return_after_quantization(self):
        transcript=[dict(words=[dict(start=2,end=2.2,text=' with...'),
                                dict(start=2.2,end=2.8,text=' Alright')])]
        kept,_=protect_words([[2.6,2.9]],transcript,[(1,2.1)],10,30000/1001)
        self.assertGreaterEqual(kept[0][0],2.2)
        self.assertLess(kept[0][0],2.2+1001/30000)

    def test_unsorted_words_and_overlapping_selection_are_normalized(self):
        transcript=[dict(words=[dict(start=1.5,end=1.9),dict(start=1,end=1.4)])]
        kept,_=protect_words([[1.6,1.8],[1.2,1.7]],transcript,[],3,30)
        self.assertEqual(len(kept),1)
        self.assertLessEqual(kept[0][0],1)
        self.assertGreaterEqual(kept[0][1],1.9)
        validate(kept,3,30)

    def test_partial_final_audio_window_is_not_lost(self):
        self.write(duration=6.003)
        energy,step,duration=microphone_energy(self.wav)
        self.assertAlmostEqual(duration,6.003)
        self.assertEqual(len(energy),601)

    def test_fractional_frame_rate_never_crosses_exclusion(self):
        fps=30000/1001
        kept,_=protect_words([[0,3]],[],[(1.013,2.014)],3,fps)
        self.assertLessEqual(kept[0][1],1.013)
        self.assertGreaterEqual(kept[1][0],2.014)
        self.assertTrue(all(abs(t*fps-round(t*fps))<1e-6 for row in kept for t in row))

    def test_invalid_numeric_settings_and_frame_rates_fail_early(self):
        self.write()
        for config in ({'leading':float('nan')},{'bogus':1},{'noise_ratio':.1}):
            with self.assertRaises(ValueError):clean_pauses([[0,6]],self.wav,settings=config)
        for fps in (0,-1,float('nan')):
            with self.assertRaises(ValueError):protect_words([[0,1]],[],[],6,fps)


if __name__=='__main__':unittest.main()
