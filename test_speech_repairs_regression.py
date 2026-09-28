"""Deterministic regressions; mocked scores test policy, not model accuracy/audio."""
import unittest
from unittest.mock import patch

from dialogue_cut import clean_dialogue
from script_review import review_script
from speech_safety import facts


def phrase(text, start=1, **extra):
    return dict(words=[dict(text=' '+word, start=start+i*.2,
        end=start+i*.2+.16, probability=.95, **extra)
        for i,word in enumerate(text.split())])


def kept_text(transcript, ranges):
    return ' '.join(w['text'].strip() for segment in transcript for w in segment['words']
        if any(a <= w['start']+.001 and b >= w['end']-.001 for a,b in ranges))


class SpeechRepairRegression(unittest.TestCase):
    def clean(self, text, expected):
        transcript=[phrase(text)]
        ranges,cuts,report=clean_dialogue(transcript,30)
        self.assertEqual(kept_text(transcript,ranges),expected)
        for word in transcript[0]['words']:
            for a,b in ranges:
                self.assertFalse(word['start']+.001<a<word['end']-.001)
                self.assertFalse(word['start']+.001<b<word['end']-.001)
        return ranges,report

    def test_entire_cutoff_start_chain(self):
        _,report=self.clean('I wa— I wa— I wanted to show you this.',
                            'I wanted to show you this.')
        self.assertEqual(len(report),2)

    def test_abandoned_opening_with_different_wording(self):
        self.clean('What you need to— what I recommend is changing this setting.',
                   'what I recommend is changing this setting.')

    def test_explicit_quantity_correction(self):
        self.clean('It costs fifteen— sorry, fifty dollars.', 'It costs fifty dollars.')

    def test_multiword_quantity_correction(self):
        self.clean('It costs one hundred and fifteen— sorry, two hundred dollars.',
                   'It costs two hundred dollars.')
        self.clean('It was negative fifteen— sorry, fifty degrees.',
                   'It was fifty degrees.')
        self.clean('It was fifteen— sorry, negative fifty degrees.',
                   'It was negative fifty degrees.')

    def test_negated_corrected_instruction(self):
        self.clean('You should enable it— actually, don’t enable it.', 'don’t enable it.')

    def test_correction_preserves_unique_context_when_uncertain(self):
        text='First save the file and enable it— actually, don’t enable it.'
        self.clean(text,text)
        transcript=[phrase(text)]
        _,report=review_script(transcript,[[.925,transcript[0]['words'][-1]['end']+.12]])
        self.assertTrue(any('self-correction' in reason for flag in report['flags'] for reason in flag['reasons']))

    def test_intentional_emphasis(self):
        for text in ('This is really, really important.', 'This is so so important.',
                     'I know that that matters.', 'Go go go!', 'No, no, no!'):
            self.clean(text,text)

    def test_quoted_and_demonstrated_mistakes(self):
        for text in ('For example I I think this is a stutter.',
                     'Say "I I wanted this" as an example.',
                     "Say 'I I wanted this' aloud.",
                     'The phrase I wa— I wanted demonstrates a repair.'):
            self.clean(text,text)

    def test_different_spelled_numbers_are_not_alternative_takes(self):
        for a,b in [('fifteen','fifty'),('fourteen','forty'),('hundred','thousand'),
                    ('fourth','fifth'),('12','13')]:
            transcript=[phrase(f'It costs {a} dollars for the plan.',1),
                        phrase(f'It costs {b} dollars for the plan.',5)]
            self.assertFalse(clean_dialogue(transcript,30)[2])

    def test_unicode_negation_and_other_contracts(self):
        self.assertNotEqual(facts('You shouldn’t enable it.'),facts('You should enable it.'))
        self.assertEqual(facts("You shouldn't enable it."),facts('You shouldn’t enable it.'))
        self.assertNotEqual(facts('It wasn’t working.'),facts('It was working.'))
        self.assertNotEqual(facts('Set it to 1.5.'),facts('Set it to 5.1.'))
        self.assertNotEqual(facts('Set it to -15.'),facts('Set it to 15.'))

    def test_near_matching_unique_detail_survives_before_semantics(self):
        transcript=[phrase('This setting makes aiming much easier in dark rooms.',1),
                    phrase('This setting makes aiming much easier in bright rooms.',5)]
        self.assertFalse(clean_dialogue(transcript,30)[2])

    def test_polarity_change_inside_one_breath_survives(self):
        text='We have 15 alright we have 14 kills right now.'
        self.clean(text,text)
        text='I think we should enable this setting I think we should not enable this setting.'
        self.clean(text,text)

    def test_adjacent_chunks_do_not_hide_cutoff_repair(self):
        original=phrase('I wa— I wanted to show you this.')['words']
        transcript=[dict(words=original[:2]),dict(words=original[2:])]
        ranges,_,changes=clean_dialogue(transcript,30)
        self.assertTrue(changes)
        self.assertEqual(kept_text(transcript,ranges),'I wanted to show you this.')

    def test_adjacent_chunks_do_not_hide_number_correction(self):
        original=phrase('It costs fifteen— sorry, fifty dollars.')['words']
        transcript=[dict(words=original[:3]),dict(words=original[3:])]
        ranges,_,_=clean_dialogue(transcript,30)
        self.assertEqual(kept_text(transcript,ranges),'It costs fifty dollars.')

    def test_speaker_changes_preserve_repetition(self):
        transcript=[phrase('The setting makes the image much brighter.',1),
                    phrase('The setting makes the image much brighter.',5)]
        transcript[0]['speaker']='A';transcript[1]['speaker']='B'
        self.assertFalse(clean_dialogue(transcript,30)[2])
        first=phrase('I');second=phrase('I think so.',1.2)
        first['speaker']='A';second['speaker']='B'
        self.assertFalse(clean_dialogue([first,second],30)[2])

    def test_low_confidence_or_overlapping_alignment_stays(self):
        transcript=[phrase('I I think this works.')]
        transcript[0]['words'][0]['probability']=.3
        self.assertFalse(clean_dialogue(transcript,30)[2])
        transcript=[phrase('I I think this works.')]
        transcript[0]['words'][0]['end']=1.3
        self.assertFalse(clean_dialogue(transcript,30)[2])

    def test_exact_complete_retake_and_failed_later_take(self):
        transcript=[phrase('The setting makes the image much brighter.',1),
                    phrase('The setting makes the image much brighter.',5)]
        ranges,_,report=clean_dialogue(transcript,30)
        self.assertEqual(len(report),1)
        self.assertEqual(kept_text(transcript,ranges),'The setting makes the image much brighter.')
        transcript[1]=phrase('The setting makes the image—',5)
        self.assertFalse(clean_dialogue(transcript,30)[2])

    def test_later_uncertain_take_does_not_replace_clear_take(self):
        transcript=[phrase('This setting makes the image much brighter.',1),
                    phrase('This setting makes the image much brighter.',5)]
        for w in transcript[1]['words']:w['probability']=.65
        self.assertFalse(clean_dialogue(transcript,30)[2])
        ranges=[(s['words'][0]['start']-.075,s['words'][-1]['end']+.12) for s in transcript]
        self.assertFalse(review_script(transcript,ranges)[1]['changes'])

    def test_long_attempt_unique_content_survives(self):
        transcript=[phrase('This setting makes aiming much easier and reduces recoil during combat.',1),
                    phrase('This setting makes aiming much easier and increases recoil during combat.',7)]
        ranges=[(s['words'][0]['start']-.075,s['words'][-1]['end']+.12) for s in transcript]
        result,report=review_script(transcript,ranges)
        self.assertEqual([list(r) for r in result],[list(r) for r in ranges])
        self.assertFalse(report['changes'])


class SemanticPolicyRegression(unittest.TestCase):
    def fixture(self):
        transcript=[phrase('This setting makes aiming much easier for beginners.',1),
                    phrase('This setting makes it easier for beginners to aim.',7)]
        return transcript,[(s['words'][0]['start']-.075,s['words'][-1]['end']+.12) for s in transcript]

    def scores(self,forward=.99,reverse=.99):
        return [dict(entailment=n,contradiction=.001,neutral=.999-n) for n in (forward,reverse)]

    def test_editor_cannot_override_missing_information_coverage(self):
        import numpy as np
        import contextual_takes
        captured=[]
        class Editor:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def verify(self,a,b,context=None):
                captured.append(context)
                return dict(keep='B',reason='mock proposal for control-flow regression')
        transcript,ranges=self.fixture()
        with patch.object(contextual_takes,'encode',return_value=np.ones((2,1))), \
             patch.object(contextual_takes,'entailment',return_value=self.scores(.3,.3)), \
             patch('local_editor.Editor',Editor):
            result,report=contextual_takes.review(transcript,ranges,editorial=True)
        self.assertEqual(result,ranges)
        self.assertFalse(report['changes'])
        self.assertEqual(set(captured[0]),{'before','between','after'})
        self.assertEqual(report['editor_pairs_checked'],1)

    def test_direction_preserves_unique_earlier_information(self):
        import numpy as np
        import contextual_takes
        transcript,ranges=self.fixture()
        with patch.object(contextual_takes,'encode',return_value=np.ones((2,1))), \
             patch.object(contextual_takes,'entailment',return_value=self.scores(.2,.99)):
            result,report=contextual_takes.review(transcript,ranges)
        self.assertEqual(len(report['changes']),1)
        self.assertEqual(report['changes'][0]['kept_source_start'],1)
        self.assertEqual(len(result),1)

    def test_spelled_quantity_blocks_even_high_similarity(self):
        import numpy as np
        import contextual_takes
        transcript=[phrase('This plan costs fifteen dollars for every extra user.',1),
                    phrase('This plan costs fifty dollars for every extra user.',7)]
        ranges=[(.925,3.2),(6.925,9.2)]
        with patch.object(contextual_takes,'encode',return_value=np.ones((2,1))), \
             patch.object(contextual_takes,'entailment') as infer:
            result,report=contextual_takes.review(transcript,ranges)
        self.assertEqual(result,ranges)
        infer.assert_not_called()
        self.assertIn('Different number',report['comparisons'][0]['blocked'])

    def test_incomplete_editor_response_is_reported(self):
        import numpy as np
        import contextual_takes
        class Editor:
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def verify(self,*args,**kwargs):return dict(keep='both',status='incomplete')
        transcript,ranges=self.fixture()
        with patch.object(contextual_takes,'encode',return_value=np.ones((2,1))), \
             patch.object(contextual_takes,'entailment',return_value=self.scores()), \
             patch('local_editor.Editor',Editor):
            result,report=contextual_takes.review(transcript,ranges,editorial=True)
        self.assertEqual(result,ranges)
        self.assertEqual(report['status'],'incomplete')
        self.assertTrue(report['warnings'])

    def test_no_candidate_coverage_is_explicit(self):
        import contextual_takes
        transcript=[phrase('One short take.')]
        result,report=contextual_takes.review(transcript,[(.925,1.8)],editorial=True)
        self.assertEqual(report['status'],'not_applicable')
        self.assertEqual(report['nli_status'],'not_run')
        self.assertEqual(report['nli_pairs_checked'],0)
        self.assertIn('no semantic comparison',report['coverage_reason'])

    def test_editor_does_not_delete_a_distinct_talking_point(self):
        import io,json
        from local_editor import Editor
        decision=dict(reason='Different subjects',same_talking_point=False,
            A_contains_useful_information_missing_from_B=True,
            B_contains_useful_information_missing_from_A=False)
        body=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(decision)))])
        editor=Editor();editor.url='http://127.0.0.1:1';editor.key='test'
        with patch.object(editor.http,'open',return_value=io.BytesIO(json.dumps(body).encode())):
            self.assertEqual(editor.choose('First subject','Different subject')['keep'],'both')

    def test_saved_preference_cannot_override_current_meaning(self):
        import json,tempfile
        from pathlib import Path
        import numpy as np
        import contextual_takes
        transcript,ranges=self.fixture()
        earlier=''.join(w['text'] for w in transcript[0]['words']).strip()
        keep=''.join(w['text'] for w in transcript[1]['words']).strip()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'semantic-preferences.json').write_text(json.dumps(dict(
                approved_alternative_takes=[dict(earlier=earlier,keep=keep)])),encoding='utf-8')
            with patch.object(contextual_takes,'ROOT',root), \
                 patch.object(contextual_takes,'encode',return_value=np.ones((2,1))), \
                 patch.object(contextual_takes,'entailment',return_value=self.scores(.3,.99)):
                result,report=contextual_takes.review(transcript,ranges)
        self.assertEqual(result,ranges)
        self.assertFalse(report['changes'])
        self.assertTrue(report['comparisons'][0]['user_approved'])
        self.assertIn('lacks current',report['comparisons'][0]['blocked'])


def write_validation_report():
    """Export inspectable synthetic annotations without claiming media validation."""
    import json
    from pathlib import Path
    examples=[
        ('cutoff_stutter','I wa— I wa— I wanted to show you this.','I wanted to show you this.'),
        ('repeated_word','I I think this works.','I think this works.'),
        ('abandoned_opening','What you need to— what I recommend is changing this setting.',
            'what I recommend is changing this setting.'),
        ('corrected_number','It costs fifteen— sorry, fifty dollars.','It costs fifty dollars.'),
        ('corrected_negation','You should enable it— actually, don’t enable it.','don’t enable it.'),
        ('intentional_emphasis','This is really, really important.','This is really, really important.'),
        ('quoted_mistake','For example I I think this is a stutter.','For example I I think this is a stutter.'),
        ('unique_context','First save the file and enable it— actually, don’t enable it.',
            'First save the file and enable it— actually, don’t enable it.'),
    ]
    annotations=[]
    fixtures=[(name,[phrase(source)],expected) for name,source,expected in examples]
    repeated='The setting makes the image much brighter.'
    fixtures.append(('full_sentence_retake',[phrase(repeated,1),phrase(repeated,5)],repeated))
    chunk_words=phrase('I wa— I wanted to show you this.')['words']
    fixtures.append(('cross_chunk_repair',[dict(words=chunk_words[:2]),dict(words=chunk_words[2:])],
        'I wanted to show you this.'))
    for name,transcript,expected in fixtures:
        ranges,_,removed=clean_dialogue(transcript,30)
        actual=kept_text(transcript,ranges)
        assert actual==expected,(name,actual,expected)
        output=[];cursor=0
        for a,b in ranges:
            output.append(dict(source_start=a,source_end=b,output_start=cursor,output_end=cursor+b-a))
            cursor+=b-a
        annotated_words=[]
        for segment in transcript:
            for word in segment['words']:
                clip=next((clip for clip in output if clip['source_start']<=word['start']+.001 and
                    clip['source_end']>=word['end']-.001),None)
                annotated_words.append(dict(text=word['text'].strip(),source_start=word['start'],
                    source_end=word['end'],retained=bool(clip),
                    output_start=clip['output_start']+word['start']-clip['source_start'] if clip else None,
                    output_end=clip['output_start']+word['end']-clip['source_start'] if clip else None))
        annotations.append(dict(case=name,status='PASS',source_text=' '.join(
            w['text'].strip() for s in transcript for w in s['words']),expected_retained_text=expected,
            actual_retained_text=actual,removed_spans=removed,kept_ranges=output,words=annotated_words,
            context_to_preserve='All words' if name in {'intentional_emphasis','quoted_mistake','unique_context'} else None))
    target=Path(__file__).resolve().parent/'analysis/verification/speech-repairs.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(dict(evidence='Synthetic word-timestamp fixtures; no audio or video',
        rendered_speech='NOT VERIFIED',asr_accuracy='NOT VERIFIED',model_accuracy='NOT VERIFIED',
        annotated_cases=len(annotations),cases=annotations),indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':
    result=unittest.main(exit=False)
    if result.result.wasSuccessful():write_validation_report()
    else:raise SystemExit(1)
