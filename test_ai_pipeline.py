"""Integration safety for the AI orchestrator using synthetic waveform evidence.

HTTP replies are deterministic fixtures; these tests make no live-model or
listening-quality claim. The real candidate, schema and boundary validators run.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from app_paths import add_dependencies
add_dependencies()
import numpy as np

import ai_pipeline
from automatic_cut import Cancelled
from edit_manifest import cut_signature
from openai_editor import AIError, OpenAIEditorClient
from test_ai_decisions import phrase, proposal


class _Response:
    def __init__(self, data, number):
        self.data = json.dumps(data).encode()
        self.headers = {"x-request-id": f"req_pipeline_fixture_{number}"}
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, count): return self.data[:count]


class _DynamicHTTP:
    def __init__(self, text):
        self.text = text
        self.calls = []
    def open(self, request, timeout):
        self.calls.append(request.full_url)
        if request.full_url.endswith("/audio/transcriptions"):
            return _Response({"text": self.text}, len(self.calls))
        request_body = json.loads(request.data)
        window = json.loads(request_body["input"])
        rows = []
        for candidate in window["candidates"]:
            action = "SHORTEN_PAUSE" if candidate["kind"] == "pause" else "REMOVE"
            row = proposal(candidate, window["words"], action=action)
            row.pop("model")
            rows.append(row)
        data = {"status": "completed", "model": request_body["model"],
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "output": [{"type": "message", "content": [{"type": "output_text",
                    "text": json.dumps({"decisions": rows})}]}]}
        return _Response(data, len(self.calls))


class PipelineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / "recording.mp4"
        self.source.write_bytes(b"Video probe mocked; waveform and decisions are real fixtures.")
        self.duration, self.fps = 10., 30.
        self.text = "I um wanted to show you this."
        self.transcript = [phrase(self.text)]
        for segment in self.transcript:
            segment.update(start=segment["words"][0]["start"], end=segment["words"][-1]["end"], text=self.text)
        self.opener = _DynamicHTTP(self.text)
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "pipeline-fixture-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = OpenAIEditorClient(self.folder / "api", opener=self.opener)

    def extract(self, source, mic, start, end, dest):
        rate = 16000
        signal = np.zeros(round(self.duration*rate), dtype=np.int16)
        for segment in self.transcript:
            for word in segment["words"]:
                a, b = round(word["start"]*rate), round(word["end"]*rate)
                signal[a:b] = (np.sin(np.arange(b-a)*.11)*5000).astype(np.int16)
        with wave.open(str(dest), "wb") as stream:
            stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            stream.writeframes(signal.tobytes())

    def run_plan(self, client=None, tracks=None, **kwargs):
        def timing(*_):
            return copy.deepcopy(self.transcript), {"model": "fixture-source-timing", "cached": True}
        with patch.object(ai_pipeline, "ROOT", self.folder), \
             patch.object(ai_pipeline, "probe", return_value=(self.duration, self.fps, tracks or ["Commentary"])), \
             patch.object(ai_pipeline, "extract", side_effect=self.extract), \
             patch.object(ai_pipeline, "_local_timing", side_effect=timing), \
             patch("speech_evidence.audit", return_value=([], [])):
            return ai_pipeline.plan_ai(self.source, 0, client=client or self.client, **kwargs)

    def speech_cut(self, result):
        return next(d for d in result["ai_plan"]["decisions"] if d["action"] == "REMOVE" and d["status"] == "accepted")

    def test_real_transport_and_alignment_resume_without_new_http_calls(self):
        first = self.run_plan()
        count = len(self.opener.calls)
        self.assertGreaterEqual(count, 2)
        self.assertTrue(Path(first["plan_path"]).is_file())
        second_client = OpenAIEditorClient(self.folder / "api", opener=self.opener)
        second = self.run_plan(second_client)
        self.assertEqual(len(self.opener.calls), count)
        self.assertNotEqual(first["analysis_run_id"], second["analysis_run_id"])
        self.assertEqual(first["kept"], second["kept"])
        self.assertEqual(second["ai_plan"]["metrics"]["cost_usd_estimate"], 0)
        self.assertGreater(second["ai_plan"]["metrics"]["video_total"]["cost_usd_estimate"], 0)
        self.assertTrue(all(r["cached"] for r in second["ai_plan"]["calls"]))
        self.assertFalse(second["render_verified"])
        self.assertFalse(second["verified"])

    def test_no_key_fails_before_local_model_or_audio_processing(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
             patch.object(ai_pipeline, "ROOT", self.folder), \
             patch.object(ai_pipeline, "probe", return_value=(self.duration, self.fps, ["Commentary"])), \
             patch.object(ai_pipeline, "extract") as extraction, \
             patch.object(ai_pipeline, "_local_timing") as timing:
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                ai_pipeline.plan_ai(self.source, 0)
            extraction.assert_not_called()
            timing.assert_not_called()

    def test_transcription_failure_preserves_every_frame_and_remains_unverified(self):
        class Offline:
            def transcribe(self, *args, **kwargs):
                raise AIError("Transcription is unavailable.", code="network_error")
            def responses(self, *args, **kwargs):
                raise AssertionError("A failed transcript must not trigger decision calls")
        result = self.run_plan(Offline())
        self.assertEqual(result["kept"], [[0, self.duration]])
        self.assertEqual(result["ai_plan"]["transcription"]["status"], "unavailable")
        self.assertTrue(result["evidence_flags"])
        self.assertFalse(result["render_verified"])
        self.assertEqual(result["ai_plan"]["metrics"]["number_of_cuts"], 0)

    def test_cloud_only_speech_protects_the_unaligned_owner_interval(self):
        self.opener.text += " This extra sentence has no local timing."
        result = self.run_plan()
        self.assertEqual(result["kept"], [[0, self.duration]])
        self.assertTrue(result["ai_plan"]["transcription"]["requires_review"])
        self.assertIn((0., self.duration), result["protected_ranges"])

    def test_other_track_speech_cannot_be_deleted_with_microphone_silence(self):
        with patch.object(ai_pipeline, '_other_speech', return_value=[(0,self.duration)]), \
             patch('combat_detection.detect', return_value=([],{})):
            result=self.run_plan(tracks=['Commentary','Voice chat'])
        self.assertEqual(result['kept'], [[0,self.duration]])
        self.assertEqual(result['other_track_speech'][0]['track'],1)

    def test_matching_render_evidence_survives_revalidation_but_not_a_restore(self):
        result=self.run_plan()
        result['review_render']={'source_key':result['source_key'],
            'cut_signature':result['cut_signature'],'review_scope':'full','qa':{'geometry':'PASS'}}
        checked=ai_pipeline.validate_for_export(result)
        self.assertEqual(checked['review_render'],result['review_render'])
        cut=self.speech_cut(result)
        restored=ai_pipeline.rebuild_plan(result,{cut['candidate_id']:{'enabled':False}},persist=False)
        self.assertNotIn('review_render',restored)

    def test_unchanged_validated_speech_edit_rebuild_and_export_are_idempotent(self):
        result = self.run_plan()
        self.speech_cut(result)
        rebuilt = ai_pipeline.rebuild_plan(result, {}, persist=False)
        self.assertEqual(rebuilt["kept"], result["kept"])
        exported = ai_pipeline.validate_for_export(rebuilt)
        self.assertEqual(exported["cut_signature"], result["cut_signature"])
        self.assertEqual(exported["output_words"], result["output_words"])

    def test_restoring_a_cut_returns_its_words_and_invalid_adjustments_fail(self):
        result = self.run_plan()
        cut = self.speech_cut(result)
        restored = ai_pipeline.rebuild_plan(result, {cut["candidate_id"]: {"enabled": False}}, persist=False)
        self.assertIn("um", " ".join(w["text"] for w in restored["output_words"]))
        self.assertGreater(restored["edited_duration"], result["edited_duration"])
        with self.assertRaises(ValueError):
            ai_pipeline.rebuild_plan(result, {cut["candidate_id"]: {"start": 0}}, persist=False)
        with self.assertRaises(ValueError):
            ai_pipeline.rebuild_plan(result, {"forged": {"enabled": False}}, persist=False)

    def test_even_recomputed_signature_cannot_authorize_forged_cut_geometry(self):
        result = self.run_plan()
        forged = copy.deepcopy(result)
        forged["kept"] = [[7., 9.]]
        forged["cut_signature"] = cut_signature(forged["kept"], self.fps)
        with self.assertRaises(ValueError):
            ai_pipeline.validate_for_export(forged)
        with patch("review_media.audio_sidecars") as sidecars, patch("automatic_cut.export_review") as exporter:
            from automatic_cut import send
            with self.assertRaises(ValueError): send(forged)
            sidecars.assert_not_called()
            exporter.assert_not_called()

    def test_source_changes_during_analysis_are_rejected(self):
        with patch.object(ai_pipeline, "key_for", side_effect=["before", "after"]):
            with self.assertRaisesRegex(ValueError, "changed during analysis"):
                self.run_plan()

    def test_cancellation_before_extraction_never_calls_api(self):
        with self.assertRaises(Cancelled): self.run_plan(cancel=lambda: True)
        self.assertFalse(self.opener.calls)


if __name__ == "__main__":
    unittest.main()
