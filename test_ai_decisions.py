"""Source authorization and fail-closed regressions; no real model/audio claims."""
import copy
import tempfile
import unittest
import wave
from pathlib import Path

from app_paths import add_dependencies
add_dependencies()
import numpy as np

from ai_decisions import (build_candidates, decision_windows, indexed_words,
                          make_decisions, validate_decisions, SYSTEM_PROMPT)
from pause_cleanup import pause_settings


def phrase(text, start=1., **extra):
    return dict(words=[dict(text=' '+token, start=start+i*.25, end=start+i*.25+.15,
        probability=.98, timing_reliable=True, **extra) for i, token in enumerate(text.split())])


def proposal(candidate, words, remove=None, action='REMOVE', **extra):
    lookup = {w['source_word_id']: w for w in words}
    remove = candidate['word_ids'] if remove is None and action == 'REMOVE' else (remove or [])
    start = lookup[remove[0]]['start'] if remove else candidate['start']
    end = lookup[remove[-1]]['end'] if remove else candidate['end']
    row = dict(candidate_id=candidate['id'], action=action, remove_word_ids=remove,
        keep_word_ids=[x for x in candidate['review_word_ids'] if x not in remove],
        source_start=start, source_end=end, reason='Synthetic policy test', confidence=.99,
        required_context_word_ids=[], meaning_change=False, ambiguous=False,
        needs_audio_review=False, model='test-model')
    return dict(row, **extra)


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wav = Path(self.temp.name)/'source.wav'
        self.duration = 18.
        self.settings = pause_settings('balanced')
        self.transcript = [phrase('I I was going to show you this.')]
        self.write_audio()

    def write_audio(self):
        rate = 16000
        signal = np.zeros(round(rate*self.duration), dtype=np.int16)
        for word in indexed_words(self.transcript):
            lo, hi = round(word['start']*rate), round(word['end']*rate)
            signal[lo:hi] = (np.sin(np.arange(max(0, hi-lo))*.11)*5000).astype(np.int16)
        with wave.open(str(self.wav), 'wb') as stream:
            stream.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
            stream.writeframes(signal.tobytes())

    def candidates(self, **extra):
        self.write_audio()
        return build_candidates(self.transcript, self.wav, self.duration, 30, self.settings, **extra)

    def validate(self, rows, candidates, **extra):
        return validate_decisions(rows, candidates, self.transcript, self.duration, 30,
                                  self.wav, self.settings, **extra)


class CandidatePolicyTests(_Fixture):
    def test_candidate_generation_never_changes_source_transcript(self):
        before = copy.deepcopy(self.transcript)
        candidates = self.candidates()
        self.assertEqual(self.transcript, before)
        self.assertTrue(any(c['kind'] != 'pause' for c in candidates))
        self.assertTrue(any(c['kind'] == 'pause' for c in candidates))

    def test_stutter_and_cut_geometry_use_real_word_ids(self):
        candidates = self.candidates()
        first = indexed_words(self.transcript)[0]
        c = next(c for c in candidates if c['word_ids'] == [first['source_word_id']])
        result = self.validate([proposal(c, indexed_words(self.transcript))], [c])
        row = result['decisions'][0]
        self.assertEqual(row['status'], 'accepted')
        self.assertEqual(result['expected_retained_text'], 'I was going to show you this.')
        self.assertTrue(all(abs(t*30-round(t*30)) < 1e-6 for r in result['kept'] for t in r))
        self.assertLessEqual(row['start'], first['start'])
        self.assertGreaterEqual(row['end'], first['end'])

    def test_wrong_color_correction_preserves_prefix_and_actual_replacement(self):
        self.transcript = [phrase('Click the blue— sorry, the green button.')]
        candidates = self.candidates()
        c = next(c for c in candidates if c['kind'] == 'correction')
        words = indexed_words(self.transcript)
        row = proposal(c, words, remove=[w['source_word_id'] for w in words[1:4]])
        result = self.validate([row], [c])
        self.assertEqual(result['decisions'][0]['status'], 'accepted')
        self.assertEqual(result['expected_retained_text'], 'Click the green button.')

    def test_model_cannot_delete_required_prefix_or_unique_clause(self):
        for text in ('Click the blue— sorry, the green button.',
                     'First save the file and enable it— actually, dont enable it.'):
            self.transcript = [phrase(text)]
            candidates = self.candidates()
            c = next(c for c in candidates if c['kind'] == 'correction')
            result = self.validate([proposal(c, indexed_words(self.transcript))], [c])
            self.assertEqual(result['decisions'][0]['status'], 'review')
            self.assertFalse(result['exclusions'])

    def test_explicit_number_and_negation_corrections_keep_actual_final_words(self):
        for text, delete, expected in (
                ('It costs fifteen— sorry, fifty dollars.', (2, 4), 'It costs fifty dollars.'),
                ('You should enable it— actually, dont enable it.', (0, 5), 'dont enable it.')):
            self.transcript = [phrase(text)]
            candidates = self.candidates()
            c = next(c for c in candidates if c['kind'] == 'correction')
            words = indexed_words(self.transcript)
            row = proposal(c, words, remove=[w['source_word_id'] for w in words[slice(*delete)]])
            result = self.validate([row], [c])
            self.assertEqual(result['decisions'][0]['status'], 'accepted')
            self.assertEqual(result['expected_retained_text'], expected)

    def test_number_change_without_correction_is_review_only(self):
        self.transcript = [phrase('The plan costs fifteen dollars monthly.', 1),
                           phrase('The plan costs fifty dollars monthly.', 4)]
        candidates = self.candidates()
        c = next(c for c in candidates if c['kind'] == 'retake')
        result = self.validate([proposal(c, indexed_words(self.transcript))], [c])
        self.assertEqual(result['decisions'][0]['status'], 'review')
        self.assertFalse(result['exclusions'])

    def test_full_sentence_retake_retains_complete_later_take(self):
        text = 'This setting makes the image much brighter.'
        self.transcript = [phrase(text, 1), phrase(text, 5)]
        candidates = self.candidates()
        c = next(c for c in candidates if c['word_ids'] and len(c['word_ids']) == 7)
        words = indexed_words(self.transcript)
        row = proposal(c, words, remove=c['word_ids'], action='REPLACE_EARLIER_TAKE_WITH_LATER_TAKE')
        result = self.validate([row], [c])
        self.assertEqual(result['decisions'][0]['status'], 'accepted')
        self.assertEqual(result['expected_retained_text'], text)

    def test_replacement_conflict_cannot_erase_every_take(self):
        text = 'This setting makes the image much brighter.'
        self.transcript = [phrase(text, 1), phrase(text, 5), phrase(text, 9)]
        candidates = self.candidates()
        words = indexed_words(self.transcript)
        first = next(c for c in candidates if c['kind'] == 'retake' and c['start'] == 1 and
                     c['replacement_word_ids'][0] == 'word-000007')
        second = next(c for c in candidates if len(c['word_ids']) == 7 and c['start'] == 5)
        result = self.validate([proposal(c, words) for c in (first, second)], [first, second])
        self.assertFalse(result['exclusions'])
        self.assertTrue(all(r['status'] == 'review' for r in result['decisions']))

    def test_adjacent_pause_and_speech_cuts_keep_remaining_words_whole(self):
        candidates = self.candidates()
        words = indexed_words(self.transcript)
        speech = next(c for c in candidates if c['word_ids'] == ['word-000000'])
        pause = next(c for c in candidates if c['kind'] == 'pause' and c['start'] == 0)
        rows = [proposal(speech, words), proposal(pause, words, action='SHORTEN_PAUSE')]
        result = self.validate(rows, [speech, pause])
        for w in words[1:]:
            self.assertTrue(any(a <= w['start'] and b >= w['end'] for a, b in result['kept']))

    def test_quoted_and_multiple_speaker_repetitions_preserved(self):
        for segments in ([phrase('For example I I wanted this.')],
                         [phrase('I', speaker='A'), phrase('I think this works.', 1.25, speaker='B')]):
            self.transcript = segments
            candidates = self.candidates()
            c = next(c for c in candidates if c['kind'] == 'repetition')
            result = self.validate([proposal(c, indexed_words(self.transcript))], [c])
            self.assertFalse(result['exclusions'])

    def test_unreliable_and_overlapping_timing_preserved(self):
        candidates = self.candidates()
        c = next(c for c in candidates if c['word_ids'] == ['word-000000'])
        self.transcript[0]['words'][0]['timing_reliable'] = False
        result = self.validate([proposal(c, indexed_words(self.transcript))], [c])
        self.assertFalse(result['exclusions'])
        self.transcript[0]['words'][0].update(timing_reliable=True, end=1.3)
        result = self.validate([proposal(c, indexed_words(self.transcript))], [c])
        self.assertFalse(result['exclusions'])

    def test_uncertain_unfinished_prefix_is_retrieved_but_not_deleted(self):
        self.transcript = [phrase('What you need to... what I recommend is changing this setting.')]
        self.transcript[0]['words'][3]['probability'] = .56
        candidates = self.candidates()
        c = next(c for c in candidates if c['kind'] == 'restart')
        result = self.validate([proposal(c, indexed_words(self.transcript))], [c])
        self.assertEqual(result['decisions'][0]['status'], 'review')
        self.assertFalse(result['exclusions'])

    def test_abutting_high_energy_words_need_listening_review(self):
        self.transcript[0]['words'][0]['end'] = 38/30
        self.transcript[0]['words'][1]['start'] = 38/30
        c = next(c for c in self.candidates() if c['word_ids'] == ['word-000000'])
        result = self.validate([proposal(c, indexed_words(self.transcript))], [c])
        self.assertEqual(result['decisions'][0]['status'], 'review')
        self.assertIn('High-energy', result['decisions'][0]['validation_reason'])

    def test_rejected_shape_arbitrary_time_word_and_intended_text(self):
        candidates = self.candidates()
        c = next(c for c in candidates if c['word_ids'] == ['word-000000'])
        valid = proposal(c, indexed_words(self.transcript))
        for update in ({'source_start': 0}, {'confidence': float('nan')},
                       {'remove_word_ids': ['invented']}, {'keep_word_ids': ['invented']},
                       {'required_context_word_ids': ['unknown']}, {'source_end': float('inf')}):
            result = self.validate([dict(valid, **update)], [c])
            self.assertEqual(result['decisions'][0]['status'], 'rejected', update)
            self.assertFalse(result['exclusions'])

    def test_missing_duplicate_decisions_fail_closed(self):
        candidates = self.candidates()
        c = next(c for c in candidates if c['word_ids'] == ['word-000000'])
        row = proposal(c, indexed_words(self.transcript))
        self.assertEqual(self.validate([], [c])['decisions'][0]['status'], 'review')
        self.assertFalse(self.validate([row, row], [c])['exclusions'])

    def test_intentional_and_visual_pause_is_not_a_cut_candidate(self):
        self.transcript = [phrase('First.', 1), phrase('Second.', 6)]
        for kwargs in ({'protected': [(1.2, 6)]}, {'events': [(1.2, 6)]}):
            candidates = self.candidates(**kwargs)
            self.assertFalse(any(c['kind'] == 'pause' and c['start'] < 6 and c['end'] > 1.2 for c in candidates))

    def test_word_safe_padding_cannot_cross_a_nearby_protected_visual_interval(self):
        candidates = self.candidates(events=[(.95, .98)])
        c = next(c for c in candidates if c['word_ids'] == ['word-000000'])
        self.assertFalse(c['protected'])
        result = self.validate([proposal(c, indexed_words(self.transcript))], [c])
        self.assertFalse(result['exclusions'])
        self.assertIn('protected', result['decisions'][0]['validation_reason'])

    def test_natural_retains_filler_balanced_can_remove(self):
        self.transcript = [phrase('Um I wanted to show you.')]
        candidates = self.candidates()
        c = next(c for c in candidates if c['kind'] == 'filler')
        row = proposal(c, indexed_words(self.transcript))
        self.assertTrue(self.validate([row], [c])['exclusions'])
        self.settings = pause_settings('natural')
        self.assertFalse(self.validate([row], [c])['exclusions'])

    def test_duplicate_filler_detectors_have_one_restorable_authority(self):
        self.transcript = [phrase('I um wanted to show you this.')]
        candidates = self.candidates()
        fillers = [c for c in candidates if c['word_ids'] == ['word-000001']]
        self.assertEqual(len(fillers), 1)
        self.assertGreaterEqual(len(fillers[0]['retrieval_evidence_ids']), 2)

    def test_duplicate_validator_proposals_cannot_reenable_a_restored_cut(self):
        self.transcript = [phrase('I um wanted to show you this.')]
        c = next(c for c in self.candidates() if c['word_ids'] == ['word-000001'])
        duplicate = dict(c, id=c['id']+'-duplicate', candidate_id=c['id']+'-duplicate')
        words = indexed_words(self.transcript)
        rows = [proposal(c, words), proposal(duplicate, words)]
        result = self.validate(rows, [c, duplicate])
        self.assertEqual([r['status'] for r in result['decisions']], ['accepted', 'superseded'])
        rows[0]['enabled'] = False
        restored = self.validate(rows, [c, duplicate])
        self.assertFalse(restored['exclusions'])
        self.assertIn('um', restored['expected_retained_text'])

    def test_restore_and_inward_manual_bounds(self):
        self.transcript = [phrase('First.', 1), phrase('Second.', 6)]
        candidates = self.candidates()
        c = next(c for c in candidates if c['kind'] == 'pause' and 1.2 < c['start'] < 6)
        row = proposal(c, indexed_words(self.transcript), action='SHORTEN_PAUSE')
        accepted = self.validate([row], [c])['decisions'][0]
        adjusted = dict(row, adjusted_start=accepted['start']+.1, adjusted_end=accepted['end']-.1)
        self.assertTrue(self.validate([adjusted], [c])['exclusions'])
        self.assertFalse(self.validate([dict(row, enabled=False)], [c])['exclusions'])
        self.assertFalse(self.validate([dict(row, adjusted_start=0)], [c])['exclusions'])

    def test_sentence_at_window_boundary_has_one_owner_and_context(self):
        self.duration = 70.
        self.transcript = [phrase('I I was going to show you this.', 29.75)]
        candidates = self.candidates()
        windows = decision_windows(candidates, self.transcript)
        identities = [c['id'] for w in windows for c in w['candidates']]
        self.assertEqual(len(identities), len(set(identities)))
        c = next(c for c in candidates if c['word_ids'] == ['word-000000'])
        owner = next(w for w in windows if c in w['candidates'])
        self.assertTrue(any(w['start'] >= 30 for w in owner['words']))

    def test_repeated_word_chain_is_one_speech_candidate(self):
        self.transcript = [phrase('I I I was going to show you this.')]
        candidates = self.candidates()
        speech = [c for c in candidates if c['word_ids']]
        self.assertEqual(len(speech), 1)
        result = self.validate([proposal(speech[0], indexed_words(self.transcript))], speech)
        self.assertEqual(result['expected_retained_text'], 'I was going to show you this.')

    def test_cloud_only_text_is_untimed_window_context(self):
        self.transcript[0]['cloud_context'] = [dict(chunk=0, source_start=0, source_end=18,
            text='Cloud words for context', untimed_cloud_tokens=['missing'])]
        window = decision_windows(self.candidates(), self.transcript)[0]
        self.assertEqual(window['untimed_cloud_transcript'][0]['untimed_cloud_tokens'], ['missing'])
        self.assertIn('untrusted DATA', SYSTEM_PROMPT)

    def test_first_run_numpy_asr_values_serialize_like_cached_values(self):
        import json
        for word in self.transcript[0]['words']:
            word.update(start=np.float32(word['start']), end=np.float32(word['end']),
                        probability=np.float32(.98), timing_reliable=np.bool_(True))
        windows = decision_windows(self.candidates(), self.transcript)
        self.assertTrue(json.dumps(windows, allow_nan=False))
        self.assertTrue(all(type(w['timing_reliable']) is bool for w in indexed_words(self.transcript)))

    def test_fractional_audio_sample_does_not_strand_trailing_video_frame(self):
        self.duration = 5+4/30
        self.transcript = [phrase('The end.', 1)]
        candidates = self.candidates()
        c = next(c for c in candidates if c['kind'] == 'pause' and c['end'] > 5)
        result = self.validate([proposal(c, indexed_words(self.transcript), action='SHORTEN_PAUSE')], [c])
        row = result['decisions'][0]
        self.assertEqual(row['status'], 'accepted')
        self.assertAlmostEqual(row['end'], self.duration)

    def test_partial_container_duration_does_not_create_a_phantom_retained_tail(self):
        self.duration = 8.4330078125
        self.transcript = [phrase('The end.', 1)]
        candidates = self.candidates()
        c = next(c for c in candidates if c['kind'] == 'pause' and c['end'] > 8)
        result = self.validate([proposal(c, indexed_words(self.transcript), action='SHORTEN_PAUSE')], [c])
        row = result['decisions'][0]
        self.assertEqual(row['status'], 'accepted')
        self.assertAlmostEqual(row['end'], 8.4)
        self.assertLess(result['kept'][-1][1], 2.)


class RequestPolicyTests(_Fixture):
    def client(self, mode='ambiguous', astra=True):
        words = indexed_words(self.transcript)
        class Client:
            def __init__(self):
                self.models = []
            def responses(self, model, payload, schema, reasoning='medium', prompt_version=None):
                import json
                self.models.append(model)
                body = json.loads(payload['input'])
                if mode == 'failure' or model == 'gpt-6-astra' and not astra:
                    raise TimeoutError('private request payload must not be logged')
                rows = []
                for c in body['candidates']:
                    action = 'SHORTEN_PAUSE' if c['kind'] == 'pause' else 'REMOVE'
                    row = proposal(c, words, action=action)
                    if model == 'gpt-6-sol' and mode == 'ambiguous':
                        row.update(confidence=.7, ambiguous=True)
                    rows.append(row)
                return dict(data=dict(decisions=rows), model=model,
                    usage=dict(input_tokens=20, output_tokens=10), request_id='test-request',
                    api_call_occurred=True, cached=False)
        return Client()

    def test_only_ambiguous_speech_escalates_and_records_actual_model(self):
        candidates = self.candidates()
        client = self.client()
        result = make_decisions(candidates, self.transcript, client)
        speech = [r for r in result['decisions'] if r['action'] == 'REMOVE']
        self.assertTrue(speech)
        self.assertTrue(all(r['astra_reviewed'] and r['model'] == 'gpt-6-astra' for r in speech))
        pauses = [r for r in result['decisions'] if r['action'] == 'SHORTEN_PAUSE']
        self.assertTrue(all(not r['astra_reviewed'] for r in pauses))

    def test_failed_escalation_preserves_and_does_not_claim_astra(self):
        candidates = [c for c in self.candidates() if c['kind'] != 'pause']
        result = make_decisions(candidates, self.transcript, self.client(astra=False))
        self.assertTrue(all(r['action'] == 'NEEDS_REVIEW' and not r['astra_reviewed'] for r in result['decisions']))

    def test_failed_primary_retains_each_candidate_and_redacts_errors(self):
        candidates = self.candidates()
        result = make_decisions(candidates, self.transcript, self.client(mode='failure'))
        self.assertEqual(len(result['decisions']), len(candidates))
        self.assertTrue(all(r['action'] == 'NEEDS_REVIEW' for r in result['decisions']))
        self.assertNotIn('private request', str(result['warnings']))

    def test_api_cancellation_propagates_instead_of_becoming_fallback(self):
        from openai_editor import AICancelled
        class Client:
            def responses(self, *args, **kwargs):
                raise AICancelled()
        with self.assertRaises(AICancelled):
            make_decisions(self.candidates(), self.transcript, Client())

    def test_fixture_astra_does_not_claim_actual_review(self):
        client = self.client()
        real = client.responses
        def fixture(*args, **kwargs):
            return dict(real(*args, **kwargs), provenance='fixture')
        client.responses = fixture
        candidates = [c for c in self.candidates() if c['kind'] != 'pause']
        result = make_decisions(candidates, self.transcript, client)
        self.assertTrue(all(not r['astra_reviewed'] for r in result['decisions']))

    def test_window_schema_requires_complete_candidate_count_and_effective_model_is_logged(self):
        client = self.client(mode='clear')
        original = client.responses
        captured = []
        def respond(model, payload, schema, **kwargs):
            import json
            incoming = json.loads(payload['input'])
            array = schema['properties']['decisions']
            captured.append(array['minItems'] == array['maxItems'] == len(incoming['candidates']))
            self.assertEqual(array['items']['properties']['candidate_id']['enum'],
                             [c['id'] for c in incoming['candidates']])
            return dict(original(model, payload, schema, **kwargs), provider_model=model+'-snapshot')
        client.responses = respond
        result = make_decisions(self.candidates(), self.transcript, client)
        self.assertTrue(captured and all(captured))
        self.assertTrue(all(r['model'] == 'gpt-6-sol-snapshot' for r in result['decisions']))
        self.assertTrue(all(r['requested_model'] == 'gpt-6-sol' for r in result['decisions']))


if __name__ == '__main__':
    unittest.main()
