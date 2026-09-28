"""Deterministic text/timing alignment and chunk-boundary safety checks."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import wave

from ai_transcription import align_text, chunk_windows, transcribe_aligned
from openai_editor import AICancelled, AIError


def word(text, start, end, **extra):
    return dict(text=text, start=start, end=end, probability=.95, **extra)


class FakeTranscriber:
    def __init__(self, texts):
        self.texts = list(texts)
        self.ledger = []
        self.durations = []

    def transcribe(self, wav, **kwargs):
        with wave.open(str(wav)) as audio:
            self.durations.append(audio.getnframes() / audio.getframerate())
        value = self.texts.pop(0)
        if isinstance(value, Exception): raise value
        self.ledger.append({"model": "gpt-transcribe", "api_call_occurred": True})
        return dict(text=value, cached=False, api_call_occurred=True, request_id="req_fake")


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wav = Path(self.temp.name) / "voice.wav"

    def audio(self, duration):
        with wave.open(str(self.wav), "wb") as audio:
            audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * round(duration * 8000))

    def test_matching_words_use_identical_original_times(self):
        self.audio(3)
        source = [dict(start=.1, end=2., text="Hello, world!", words=[word(" Hello,", .1, .6), word(" world!", 1., 2.)])]
        original = copy.deepcopy(source)
        aligned, metadata = transcribe_aligned(self.wav, source, FakeTranscriber(["hello world"]))
        self.assertEqual(source, original)
        self.assertEqual([(w["start"], w["end"]) for w in aligned[0]["words"]], [(.1, .6), (1., 2.)])
        self.assertTrue(all(w["timing_reliable"] for w in aligned[0]["words"]))
        self.assertEqual(aligned[0]["words"][1]["source_word_id"], "word-000001")
        self.assertEqual(metadata["reliable_words"], 2)
        self.assertFalse(metadata["precise_cloud_word_timestamps"])

    def test_omitted_restart_is_preserved_and_uncertain(self):
        words = [word("I", 0, .2), word("was", .2, .4), word("I", .6, .8), word("was", .8, 1.), word("going", 1., 1.4)]
        proof, stats = align_text(words, "I was going", 2.)
        self.assertEqual([p["alignment_status"] for p in proof[:2]], ["local_only", "local_only"])
        self.assertFalse(proof[0]["timing_reliable"])
        self.assertTrue(proof[-1]["timing_reliable"])
        self.assertEqual(len(words), 5)
        self.assertEqual(stats["matched_tokens"], 3)

    def test_cloud_only_words_are_untimed_and_cannot_become_cuts(self):
        self.audio(3)
        source = [{"words": [word("click", .2, .5), word("button", 1., 1.5)]}]
        result, metadata = transcribe_aligned(self.wav, source, FakeTranscriber(["click the green button"]))
        self.assertEqual(len(result[0]["words"]), 2)
        self.assertEqual(metadata["chunks"][0]["untimed_cloud_tokens"], ["the", "green"])
        self.assertTrue(metadata["requires_review"])
        self.assertEqual(result[0]["cloud_context"][0]["text"], "click the green button")
        self.assertEqual(metadata["protected_alignment_regions"][0]["start"], 0)
        self.assertEqual(metadata["protected_alignment_regions"][0]["end"], 3)

    def test_low_confidence_invalid_and_prior_uncertainty_are_not_upgraded(self):
        words = [word("bad", .1, .3), word("old", .4, .6, timing_reliable=False), word("long", .7, 4.5), word("nan", float("nan"), 1.)]
        words[0]["probability"] = .2
        proof, _ = align_text(words, "bad old long nan", 5.)
        self.assertTrue(all(not p["timing_reliable"] for p in proof))
        self.assertEqual(proof[2]["alignment_status"], "invalid_local_timing")
        self.assertEqual(proof[3]["alignment_status"], "invalid_local_timing")

    def test_duplicate_phrase_with_ambiguous_alignment_is_not_trusted(self):
        words = [word("yes", .1, .3), word("yes", .5, .7)]
        proof, _ = align_text(words, "yes", 1.)
        self.assertEqual(proof[0]["alignment_status"], "ambiguous_token_match")
        self.assertFalse(proof[0]["timing_reliable"])
        self.assertFalse(proof[1]["timing_reliable"])

    def test_chunk_context_overlap_has_disjoint_ownership(self):
        windows = chunk_windows(17., 8., 1.)
        self.assertEqual([(r["owner_start"], r["owner_end"]) for r in windows], [(0., 6.), (6., 12.), (12., 17.)])
        self.assertEqual([(r["source_start"], r["source_end"]) for r in windows], [(0., 7.), (5., 13.), (11., 17.)])
        self.assertTrue(all(r["source_end"] - r["source_start"] <= 8 for r in windows))

    def test_sentence_crossing_chunk_boundary_never_duplicates_or_retimes_words(self):
        self.audio(13.)
        words = [word("first", 1., 1.5), word("crossing", 5.7, 6.3), word("middle", 8., 8.5), word("last", 12., 12.5)]
        client = FakeTranscriber(["first crossing", "crossing middle last", "last"])
        result, metadata = transcribe_aligned(self.wav, [{"words": words}], client, chunk_seconds=8., overlap_seconds=1.)
        actual = result[0]["words"]
        self.assertEqual(len(actual), 4)
        self.assertEqual([w["transcription_chunk"] for w in actual], [0, 1, 1, 2])
        self.assertEqual([(w["start"], w["end"]) for w in actual], [(w["start"], w["end"]) for w in words])
        self.assertEqual(client.durations, [7., 8., 2.])
        self.assertEqual(len(metadata["chunks"]), 3)

    def test_cancel_and_chunk_failure_preserve_caller_transcript(self):
        self.audio(2.)
        source = [{"words": [word("hello", .2, .6)]}]
        with self.assertRaises(AICancelled):
            transcribe_aligned(self.wav, source, FakeTranscriber([]), cancel=lambda: True)
        with self.assertRaises(AIError):
            transcribe_aligned(self.wav, source, FakeTranscriber([AIError("Offline.")]))
        self.assertNotIn("source_word_id", source[0]["words"][0])

    def test_successful_chunk_is_reused_after_later_network_failure(self):
        from openai_editor import OpenAIEditorClient
        from test_openai_editor import FakeOpener
        self.audio(13.)
        source = [{"words": [word("first", 1., 1.5), word("middle", 8., 8.5), word("last", 12., 12.5)]}]
        cache = Path(self.temp.name) / "api"
        initial_http = FakeOpener({"text": "first"}, urllib.error.URLError("offline"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-only-key"}):
            initial = OpenAIEditorClient(cache, opener=initial_http, max_retries=0)
            with self.assertRaises(AIError):
                transcribe_aligned(self.wav, source, initial, chunk_seconds=8., overlap_seconds=1.)
            self.assertEqual(len(initial_http.requests), 2)
            resumed_http = FakeOpener({"text": "middle last"}, {"text": "last"})
            resumed = OpenAIEditorClient(cache, opener=resumed_http, max_retries=0)
            result, metadata = transcribe_aligned(self.wav, source, resumed, chunk_seconds=8., overlap_seconds=1.)
            self.assertEqual(len(resumed_http.requests), 2)
            self.assertTrue(metadata["chunks"][0]["cached"])
            self.assertFalse(metadata["chunks"][0]["api_call_occurred"])
            self.assertEqual(len(result[0]["words"]), 3)
            self.assertEqual(resumed.video_summary()["unknown_cost_requests"], 1)


if __name__ == "__main__":
    unittest.main()
