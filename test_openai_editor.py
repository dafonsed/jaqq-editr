"""Offline HTTP-contract and accounting regressions. No real API calls."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import wave

from openai_editor import AICancelled, AIError, OpenAIEditorClient, estimate_cost


SCHEMA = {"type": "object", "properties": {"decisions": {"type": "array", "items": {"type": "string"}}},
          "required": ["decisions"], "additionalProperties": False}


def completed(data=None, usage=True):
    response = {"status": "completed", "model": "gpt-6-sol",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(
                    data if data is not None else {"decisions": []})}]}]}
    if usage:
        response["usage"] = {"input_tokens": 1000, "output_tokens": 100,
                             "input_tokens_details": {"cached_tokens": 200}}
    return response


class FakeResponse:
    def __init__(self, value, headers=None):
        self.data = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.headers = headers or {"x-request-id": "req_test"}

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, count): return self.data[:count]


class FakeOpener:
    def __init__(self, *results):
        self.results = list(results)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)


class OpenAITransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret-not-in-logs"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def client(self, *results, **kwargs):
        self.opener = FakeOpener(*results)
        self.delays = []
        return OpenAIEditorClient(self.path / "cache", opener=self.opener,
                                  sleep=self.delays.append, **kwargs)

    def call(self, client, **kwargs):
        return client.responses("gpt-6-sol", {"instructions": "Preserve meaning.", "input": "hello"}, SCHEMA, **kwargs)

    def audio(self):
        path = self.path / "voice.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 16000)
        return path

    def test_responses_strict_format_and_cache_reuse(self):
        client = self.client(completed())
        result = self.call(client)
        request, timeout = self.opener.requests[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(body["instructions"], "Preserve meaning.")
        self.assertEqual(body["input"], "hello")
        self.assertEqual(body["reasoning"], {"effort": "medium"})
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertFalse(body["store"])
        self.assertEqual(result["data"], {"decisions": []})
        self.assertTrue(result["api_call_occurred"])
        cached = self.call(client)
        self.assertTrue(cached["cached"])
        self.assertFalse(cached["api_call_occurred"])
        self.assertEqual(cached["incurred_cost_usd_estimate"], 0)
        self.assertEqual(len(self.opener.requests), 1)
        self.assertEqual(client.summary()["cache_hits"], 1)
        self.assertAlmostEqual(client.summary()["cost_usd_estimate"], .00264)
        for file in (self.path / "cache").rglob("*.json"):
            self.assertNotIn("test-secret-not-in-logs", file.read_text())
        self.assertNotIn("test-secret-not-in-logs", json.dumps(client.ledger))

    def test_cache_identity_includes_prompt_schema_reasoning_and_model(self):
        client = self.client(*[completed() for _ in range(6)])
        self.call(client)
        self.call(client, prompt_version="next")
        self.call(client, reasoning="high")
        client.responses("gpt-6-astra", {"instructions": "Preserve meaning.", "input": "hello"}, SCHEMA)
        client.responses("gpt-6-sol", {"instructions": "Different prompt.", "input": "hello"}, SCHEMA)
        client.responses("gpt-6-sol", {"instructions": "Preserve meaning.", "input": "hello"}, dict(SCHEMA, description="new"))
        self.assertEqual(len(self.opener.requests), 6)

    def test_429_retry_after_is_bounded_and_logged_as_real_attempts(self):
        error = urllib.error.HTTPError("https://api.openai.com", 429, "secret", {"Retry-After": "999"}, io.BytesIO(b"test-secret-not-in-logs"))
        client = self.client(error, completed())
        result = self.call(client)
        self.assertAlmostEqual(sum(self.delays), 60)
        self.assertEqual(result["attempts"], 2)
        self.assertIsNone(client.summary()["cost_usd_estimate"])
        self.assertEqual(client.summary()["unknown_cost_requests"], 1)
        self.assertAlmostEqual(client.summary()["known_cost_usd_estimate"], .00264)

    def test_retry_after_date_and_invalid_value(self):
        client = self.client()
        self.assertEqual(client._retry_delay({"Retry-After": "Wed, 01 Jan 2020 00:00:00 GMT"}, 2), 0)
        self.assertEqual(client._retry_delay({"Retry-After": "invalid"}, 2), 4)
        self.assertEqual(client._retry_delay({"Retry-After": "nan"}, 2), 4)

    def test_http_errors_never_expose_provider_body_or_key(self):
        error = urllib.error.HTTPError("https://evil.example/test-secret-not-in-logs", 401,
            "test-secret-not-in-logs", {}, io.BytesIO(b"test-secret-not-in-logs"))
        client = self.client(error)
        with self.assertRaises(AIError) as raised:
            self.call(client)
        self.assertEqual(raised.exception.code, "authentication")
        self.assertNotIn("test-secret-not-in-logs", str(raised.exception))
        self.assertNotIn("test-secret-not-in-logs", json.dumps(client.ledger))
        self.assertFalse(list((self.path / "cache").glob("*.json")))

    def test_no_key_means_no_request(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            client = self.client()
            self.assertFalse(client.configured)
            with self.assertRaises(AIError) as raised:
                self.call(client)
            self.assertEqual(raised.exception.code, "missing_api_key")
            self.assertFalse(self.opener.requests)

    def test_invalid_key_header_characters_are_never_sent(self):
        for key in ("secret\r\nInjected: header", "secret\tvalue", "secret☃"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": key}):
                client = self.client()
                self.assertFalse(client.configured)
                with self.assertRaises(AIError): self.call(client)
                self.assertFalse(self.opener.requests)

    def test_cancellation_during_rate_limit_wait_accounts_attempt(self):
        error = urllib.error.HTTPError("https://api.openai.com", 429, "limit", {"Retry-After": "10"}, io.BytesIO())
        state = {"cancel": False}
        opener = FakeOpener(error)
        def sleep(_): state["cancel"] = True
        client = OpenAIEditorClient(self.path / "cache", opener=opener, sleep=sleep, cancel=lambda: state["cancel"])
        with self.assertRaises(AICancelled):
            self.call(client)
        self.assertTrue(client.ledger[-1]["api_call_occurred"])
        self.assertEqual(client.ledger[-1]["attempts"], 1)

    def test_refusal_incomplete_and_malformed_never_cache(self):
        cases = [
            ({"status": "incomplete", "output": []}, "incomplete"),
            ({"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "secret"}]}]}, "refusal"),
            ({"status": "completed", "output": None}, "invalid_response"),
            ({"status": "completed", "output": [{"type": "message", "content": [None]}]}, "invalid_response"),
            (completed(data=["not-an-object"]), "invalid_response"),
            (b"invalid-json", "invalid_response"),
        ]
        for response, code in cases:
            with self.subTest(code=code, response=response):
                client = self.client(response)
                with self.assertRaises(AIError) as raised:
                    self.call(client)
                self.assertEqual(raised.exception.code, code)
                self.assertFalse(list((self.path / "cache").glob("*.json")))

    def test_video_costs_survive_restart_without_rebilling_cached_requests(self):
        client = self.client(completed())
        self.call(client)
        original_cost = client.video_summary()["cost_usd_estimate"]
        resumed = OpenAIEditorClient(self.path / "cache", opener=FakeOpener())
        self.assertTrue(self.call(resumed)["cached"])
        self.assertEqual(resumed.summary()["cost_usd_estimate"], 0)
        self.assertEqual(resumed.video_summary()["cost_usd_estimate"], original_cost)
        self.assertEqual(resumed.video_summary()["by_model"]["gpt-6-sol"]["requests"], 1)
        self.assertTrue(resumed.video_summary()["accounting_complete"])

    def test_corrupt_accounting_does_not_break_resumption_or_claim_completeness(self):
        client = self.client(completed())
        self.call(client)
        ledger = next((self.path / "cache" / "ledger").glob("*.json"))
        row = json.loads(ledger.read_text())
        row["elapsed_seconds"] = "damaged"
        ledger.write_text(json.dumps(row))
        resumed = OpenAIEditorClient(self.path / "cache", opener=FakeOpener())
        self.assertTrue(self.call(resumed)["cached"])
        summary = resumed.video_summary()
        self.assertFalse(summary["accounting_complete"])
        self.assertEqual(summary["unreadable_ledger_events"], 1)
        self.assertIsNone(summary["cost_usd_estimate"])

    def test_unknown_usage_remains_unknown_and_failed_parse_records_usage(self):
        client = self.client(completed(usage=False))
        self.assertIsNone(self.call(client)["cost_usd_estimate"])
        self.assertIsNone(client.summary()["cost_usd_estimate"])
        self.assertIsNone(estimate_cost("gpt-6-sol", {"input_tokens": 1}))
        self.assertIsNone(estimate_cost("gpt-6-sol", {"input_tokens": True, "output_tokens": 0}))
        bad = completed(data=["wrong"])
        other = OpenAIEditorClient(self.path / "other", opener=FakeOpener(bad))
        with self.assertRaises(AIError): self.call(other)
        self.assertAlmostEqual(other.summary()["cost_usd_estimate"], .00264)

    def test_schema_violations_and_nonfinite_json_are_rejected_before_caching(self):
        for value in ({"wrong": []}, {"decisions": [4]}, {"decisions": [], "extra": True}):
            client = self.client(completed(value))
            with self.assertRaises(AIError) as raised:
                self.call(client)
            self.assertEqual(raised.exception.code, "invalid_schema")
            self.assertFalse(list((self.path / "cache").glob("*.json")))
        client = self.client(b'{"status":"completed","usage":{"input_tokens":NaN}}')
        with self.assertRaises(AIError) as raised:
            self.call(client)
        self.assertEqual(raised.exception.code, "invalid_response")

    def test_transcription_multipart_contract_and_audio_cache(self):
        client = self.client({"text": "I was I was going."})
        result = client.transcribe(self.audio(), prompt="Verbatim.")
        request, _ = self.opener.requests[0]
        body = request.data
        self.assertEqual(request.full_url, "https://api.openai.com/v1/audio/transcriptions")
        self.assertIn(b'name="model"\r\n\r\ngpt-transcribe', body)
        self.assertIn(b'name="response_format"\r\n\r\njson', body)
        self.assertIn(b'name="file"; filename="audio.wav"', body)
        self.assertNotIn(b'timestamp_granularities', body)
        self.assertAlmostEqual(result["cost_usd_estimate"], .0045 / 60)
        self.assertEqual(result["text"], "I was I was going.")
        again = client.transcribe(self.path / "voice.wav", prompt="Verbatim.")
        self.assertTrue(again["cached"])

    def test_corrupt_cache_retries_and_redirects_are_rejected(self):
        client = self.client(completed(), completed())
        self.call(client)
        cache = next((self.path / "cache").glob("*.json"))
        cache.write_text("bad cache")
        self.assertFalse(self.call(client)["cached"])
        self.assertEqual(len(self.opener.requests), 2)
        from openai_editor import _NoRedirect
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.org"))


if __name__ == "__main__":
    unittest.main()
