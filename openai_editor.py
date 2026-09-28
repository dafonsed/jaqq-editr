"""Small server-side OpenAI transport for retention editing.

Only OPENAI_API_KEY is read for authentication. Requests use a fixed HTTPS
origin, never follow redirects, and never put provider error bodies in logs.
Successful responses are content-addressed so interrupted analyses can resume.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid
import wave
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


MODELS = {"gpt-transcribe", "gpt-6-sol", "gpt-6-astra"}
RATES = {
    "as_of": "2026-09-28",
    "currency": "USD",
    "gpt-transcribe": {"audio_minute": .0045},
    "gpt-6-sol": {"input_million": 2., "cached_input_million": .2, "output_million": 10.},
    "gpt-6-astra": {"input_million": 10., "cached_input_million": 1., "output_million": 50.},
}
CACHE_VERSION = 1


class AIError(RuntimeError):
    """An intentionally sanitized error safe to show in the application."""

    def __init__(self, message, *, code="api_error", status=None, attempts=0, request_id=None):
        super().__init__(message)
        self.code, self.status, self.attempts, self.request_id = code, status, attempts, request_id


class AICancelled(AIError):
    def __init__(self):
        super().__init__("AI processing was cancelled.", code="cancelled")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _reject_constant(_):
    raise ValueError("Nonfinite JSON number")


def _loads(value):
    return json.loads(value, parse_constant=_reject_constant)


def _matches_schema(value, schema, depth=0):
    """Validate the strict JSON-schema subset used by our decision contract."""
    if depth > 32 or not isinstance(schema, dict):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
        return all(_matches_schema(item, properties[key], depth + 1)
                   for key, item in value.items() if key in properties)
    if kind == "array":
        return (isinstance(value, list) and len(value) >= schema.get("minItems", 0)
                and len(value) <= schema.get("maxItems", float("inf"))
                and all(_matches_schema(item, schema.get("items", {}), depth + 1) for item in value))
    if kind == "string":
        return (isinstance(value, str) and len(value) >= schema.get("minLength", 0)
                and len(value) <= schema.get("maxLength", float("inf")))
    if kind == "boolean":
        return isinstance(value, bool)
    if kind in ("number", "integer"):
        return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                and (kind != "integer" or isinstance(value, int))
                and value >= schema.get("minimum", -float("inf"))
                and value <= schema.get("maximum", float("inf")))
    if kind == "null":
        return value is None
    # Unknown schema constructs are never silently treated as validated.
    return False


def _safe_id(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,160}", value) else None


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _nonnegative_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _valid_ledger_row(row):
    return (isinstance(row, dict) and row.get("model") in MODELS
            and isinstance(row.get("event_id"), str)
            and _nonnegative_number(row.get("elapsed_seconds"))
            and _count(row.get("attempts")) is not None
            and all(row.get(key) is None or _nonnegative_number(row[key])
                    for key in ("cost_usd_estimate", "incurred_cost_usd_estimate")))


def estimate_cost(model, usage=None, audio_seconds=None):
    """List-price estimate, never a billing claim; unknown usage stays None."""
    if model == "gpt-transcribe":
        if not isinstance(audio_seconds, (int, float)) or not math.isfinite(audio_seconds) or audio_seconds < 0:
            return None
        return round(audio_seconds / 60 * RATES[model]["audio_minute"], 10)
    if model not in RATES or not isinstance(usage, dict):
        return None
    incoming, outgoing = _count(usage.get("input_tokens")), _count(usage.get("output_tokens"))
    details = usage.get("input_tokens_details") or {}
    cached = _count(details.get("cached_tokens", 0)) if isinstance(details, dict) else None
    if incoming is None or outgoing is None or cached is None or cached > incoming:
        return None
    rate = RATES[model]
    return round(((incoming - cached) * rate["input_million"] + cached * rate["cached_input_million"]
                  + outgoing * rate["output_million"]) / 1_000_000, 10)


def summarize_ledger(ledger):
    by_model = {}
    for model in sorted(MODELS):
        entries = [r for r in ledger if r.get("model") == model]
        charged = [r for r in entries if r.get("api_call_occurred")]
        # A retry can have unknown additional charges while the final response's
        # usage is known. Preserve that measured portion as a lower subtotal.
        known = [r["cost_usd_estimate"] for r in charged
                 if r.get("cost_usd_estimate") is not None]
        unknown = sum(r.get("incurred_cost_usd_estimate") is None for r in charged)
        by_model[model] = dict(
            requests=len(charged), cache_hits=sum(bool(r.get("cached")) for r in entries),
            successful_requests=sum(r.get("status") == "success" for r in charged),
            failed_requests=sum(r.get("status") != "success" for r in charged),
            http_attempts=sum(r.get("attempts", 0) for r in charged),
            known_cost_usd_estimate=round(sum(known), 10), unknown_cost_requests=unknown,
            cost_usd_estimate=None if unknown else round(sum(known), 10),
            elapsed_seconds=round(sum(r.get("elapsed_seconds", 0) for r in entries), 4),
        )
    unknown = sum(v["unknown_cost_requests"] for v in by_model.values())
    total = round(sum(v["known_cost_usd_estimate"] for v in by_model.values()), 10)
    return dict(by_model=by_model, rates=RATES, pricing_is_estimate=True,
                known_cost_usd_estimate=total, cost_usd_estimate=None if unknown else total,
                unknown_cost_requests=unknown,
                elapsed_seconds=round(sum(r.get("elapsed_seconds", 0) for r in ledger), 4),
                cache_hits=sum(bool(r.get("cached")) for r in ledger))


class OpenAIEditorClient:
    def __init__(self, cache_dir, *, cancel=lambda: False, timeout=60., max_retries=3,
                 opener=None, sleep=time.sleep):
        self.cache_dir = Path(cache_dir)
        self.cancel = cancel
        self.timeout = min(180., max(1., float(timeout)))
        self.max_retries = min(5, max(0, int(max_retries)))
        self._opener = opener or urllib.request.build_opener(_NoRedirect())
        self._sleep = sleep
        self._api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not self._api_key.isascii() or any(ord(char) < 33 or ord(char) > 126 for char in self._api_key):
            self._api_key = ""
        self.ledger = []
        self.run_id = uuid.uuid4().hex
        self._ledger_save_failures = 0

    @property
    def configured(self):
        return bool(self._api_key)

    def summary(self):
        return summarize_ledger(self.ledger)

    def video_summary(self):
        """Cumulative accounting for this cache directory, which must be video-scoped.

        One atomic file per invocation avoids append races between workers. Cache
        hits are recorded for observability but never counted as another charge.
        """
        events = {}
        unreadable = 0
        try:
            files = list((self.cache_dir / "ledger").glob("*.json"))
        except OSError:
            files = []
            unreadable += 1
        for path in files:
            try:
                row = _loads(path.read_text(encoding="utf-8"))
                if not _valid_ledger_row(row):
                    unreadable += 1
                    continue
                events[row["event_id"]] = row
            except (OSError, ValueError, UnicodeError, RecursionError):
                unreadable += 1
        # Include current in-memory rows even if persistence was unavailable.
        for row in self.ledger:
            events[row["event_id"]] = row
        result = summarize_ledger(list(events.values()))
        result.update(accounting_scope="source video and microphone cache directory",
                      persisted_events=len(events), unreadable_ledger_events=unreadable,
                      ledger_save_failures=self._ledger_save_failures,
                      accounting_complete=not unreadable and not self._ledger_save_failures)
        if not result["accounting_complete"]:
            result["cost_usd_estimate"] = None
        return result

    def _record(self, row):
        row = dict(row, event_id=uuid.uuid4().hex, run_id=self.run_id)
        self.ledger.append(row)
        folder = self.cache_dir / "ledger"
        temporary = folder / (row["event_id"] + ".tmp")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(_json_bytes(row))
            os.replace(temporary, folder / (row["event_id"] + ".json"))
        except OSError:
            self._ledger_save_failures += 1
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _check(self):
        if self.cancel():
            raise AICancelled()

    def _wait(self, seconds):
        # Small intervals make backoff interruptible, including Retry-After.
        remaining = max(0., min(60., seconds))
        while remaining > 0:
            self._check()
            step = min(.2, remaining)
            self._sleep(step)
            remaining -= step
        self._check()

    def _backoff(self, seconds, attempts):
        try:
            self._wait(seconds)
        except AICancelled as error:
            error.attempts = attempts
            raise

    def _retry_delay(self, headers, attempt):
        value = headers.get("Retry-After", "") if headers else ""
        try:
            delay = float(value)
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = 2. ** attempt
        return max(0., min(60., delay)) if math.isfinite(delay) else min(60., 2. ** attempt)

    def _request(self, endpoint, body, content_type):
        if not self.configured:
            raise AIError("Set OPENAI_API_KEY on the editor process to enable cloud review.", code="missing_api_key")
        for attempt in range(self.max_retries + 1):
            self._check()
            req = urllib.request.Request("https://api.openai.com/v1/" + endpoint,
                data=body, method="POST", headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": content_type,
                    "Accept": "application/json",
                    "User-Agent": "JAQQ-retention-editor/1",
                })
            try:
                with self._opener.open(req, timeout=self.timeout) as response:
                    # A fake/testing opener must obey the same size bound.
                    raw = response.read(16 * 1024 * 1024 + 1)
                    request_id = _safe_id(response.headers.get("x-request-id"))
                if len(raw) > 16 * 1024 * 1024:
                    raise AIError("The AI service returned an oversized response.", code="invalid_response",
                                  attempts=attempt + 1, request_id=request_id)
                try:
                    data = _loads(raw)
                except (ValueError, UnicodeError, RecursionError):
                    raise AIError("The AI service returned invalid JSON.", code="invalid_response",
                                  attempts=attempt + 1, request_id=request_id) from None
                if not isinstance(data, dict):
                    raise AIError("The AI service returned an unexpected response.", code="invalid_response",
                                  attempts=attempt + 1, request_id=request_id)
                return data, request_id, attempt + 1
            except urllib.error.HTTPError as error:
                status = error.code
                headers = error.headers
                request_id = _safe_id(headers.get("x-request-id")) if headers else None
                error.close()  # Never read or include provider error text.
                if status in (408, 409, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                    self._backoff(self._retry_delay(headers, attempt), attempt + 1)
                    continue
                code = "rate_limit" if status == 429 else "authentication" if status in (401, 403) else "http_error"
                raise AIError(f"AI service request failed (HTTP {status}).", code=code, status=status,
                              attempts=attempt + 1, request_id=request_id) from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt < self.max_retries:
                    self._backoff(2. ** attempt, attempt + 1)
                    continue
                raise AIError("AI service could not be reached before the timeout.", code="network_error",
                              attempts=attempt + 1) from None

    def _cached(self, key, model, prompt_version):
        try:
            value = _loads((self.cache_dir / (key + ".json")).read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError, RecursionError):
            return None
        if (not isinstance(value, dict) or value.get("cache_version") != CACHE_VERSION
                or value.get("cache_key") != key or value.get("model") != model
                or value.get("prompt_version") != prompt_version or value.get("status") != "success"
                or not isinstance(value.get("response"), dict)):
            return None
        return value

    def _save(self, key, result):
        # Cache failures do not turn a completed billable request into a retry.
        temporary = self.cache_dir / (key + "." + uuid.uuid4().hex + ".tmp")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(_json_bytes(dict(result, cache_key=key, cache_version=CACHE_VERSION)))
            os.replace(temporary, self.cache_dir / (key + ".json"))
            return True
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _run(self, *, model, prompt_version, identity, body, endpoint, content_type, parse, audio_seconds=None):
        self._check()
        if not self.configured:
            raise AIError("Set OPENAI_API_KEY on the editor process to enable cloud review.", code="missing_api_key")
        started = time.monotonic()
        key = hashlib.sha256(_json_bytes(dict(identity=identity, prompt_version=prompt_version,
                                             endpoint=endpoint, cache_version=CACHE_VERSION))).hexdigest()
        cached = self._cached(key, model, prompt_version)
        if cached is not None:
            try:
                # Revalidate cached structured content, not merely JSON syntax.
                parsed = parse(cached["response"])
            except (AIError, KeyError, TypeError, ValueError):
                cached = None
            else:
                usage = cached["response"].get("usage")
                usage = usage if isinstance(usage, dict) else None
                result = dict(cached, **parsed, cached=True, api_call_occurred=False,
                              incurred_cost_usd_estimate=0., attempts=0,
                              elapsed_seconds=time.monotonic() - started,
                              request_id=_safe_id(cached.get("request_id")),
                              usage=usage, cost_usd_estimate=estimate_cost(model, usage, audio_seconds))
                self._record(self._ledger_record(result))
                return result
        request_id = None
        attempts = 0
        received = False
        response = None
        try:
            response, request_id, attempts = self._request(endpoint, body, content_type)
            received = True
            parsed = parse(response)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
            cost = estimate_cost(model, usage, audio_seconds)
            # Retried network/5xx calls may have been billed. Their cost is unknown.
            incurred = cost if attempts == 1 else None
            result = dict(parsed, model=model, provider_model=response.get("model"), usage=usage,
                          request_id=request_id, prompt_version=prompt_version, cached=False,
                          api_call_occurred=True, status="success", attempts=attempts,
                          elapsed_seconds=time.monotonic() - started, cost_usd_estimate=cost,
                          incurred_cost_usd_estimate=incurred, audio_seconds=audio_seconds,
                          pricing_is_estimate=True, rates_as_of=RATES["as_of"], response=response)
            result["cache_saved"] = self._save(key, result)
            self._record(self._ledger_record(result))
            self._check()
            return result
        except AIError as error:
            # Cancellation after a saved success is already accounted for above.
            if not (received and self.ledger and self.ledger[-1].get("request_id") == request_id
                    and self.ledger[-1].get("status") == "success"):
                usage = response.get("usage") if isinstance(response, dict) and isinstance(response.get("usage"), dict) else None
                cost = estimate_cost(model, usage, audio_seconds) if received else None
                self._record(dict(model=model, prompt_version=prompt_version, cached=False,
                    api_call_occurred=bool(received or error.attempts), status=error.code,
                    request_id=request_id or error.request_id, usage=usage, attempts=attempts or error.attempts,
                    elapsed_seconds=time.monotonic() - started, cost_usd_estimate=cost,
                    incurred_cost_usd_estimate=cost if attempts == 1 else None, audio_seconds=audio_seconds,
                    pricing_is_estimate=True, rates_as_of=RATES["as_of"]))
            raise

    @staticmethod
    def _ledger_record(result):
        keys = ("model", "provider_model", "prompt_version", "usage", "request_id", "cached",
                "api_call_occurred", "status", "attempts", "elapsed_seconds", "cost_usd_estimate",
                "incurred_cost_usd_estimate", "audio_seconds", "pricing_is_estimate", "rates_as_of")
        return {key: result.get(key) for key in keys}

    def responses(self, model, payload, schema, reasoning="medium", *, prompt_version="retention-decisions-v1"):
        if model not in ("gpt-6-sol", "gpt-6-astra"):
            raise AIError("Unsupported editing decision model.", code="invalid_model")
        if reasoning not in ("low", "medium", "high"):
            raise AIError("Unsupported reasoning effort.", code="invalid_reasoning")
        if not isinstance(payload, dict) or not isinstance(schema, dict):
            raise AIError("Decision input and schema must be objects.", code="invalid_input")
        prompt_version = str(payload.get("prompt_version", prompt_version))
        incoming = payload.get("input", {k: v for k, v in payload.items() if k not in ("instructions", "prompt_version")})
        if not isinstance(incoming, (str, list)):
            incoming = _json_bytes(incoming).decode("utf-8")
        request = dict(model=model, input=incoming, store=False,
                       reasoning={"effort": reasoning},
                       text={"format": {"type": "json_schema", "name": "retention_decision_plan",
                                         "strict": True, "schema": schema}})
        if payload.get("instructions"):
            request["instructions"] = str(payload["instructions"])

        def parse(response):
            if response.get("status") != "completed":
                raise AIError("AI review did not complete; its proposals were not applied.", code="incomplete")
            pieces = []
            output = response.get("output", [])
            if not isinstance(output, list):
                raise AIError("Invalid structured AI response.", code="invalid_response")
            for item in output:
                if not isinstance(item, dict):
                    raise AIError("Invalid structured AI response.", code="invalid_response")
                if item.get("type") != "message":
                    continue
                contents = item.get("content", [])
                if not isinstance(contents, list):
                    raise AIError("Invalid structured AI response.", code="invalid_response")
                for content in contents:
                    if not isinstance(content, dict):
                        raise AIError("Invalid structured AI response.", code="invalid_response")
                    if content.get("type") == "refusal":
                        raise AIError("The AI service declined this review; preserve the source for review.", code="refusal")
                    if content.get("type") == "output_text":
                        pieces.append(content.get("text", ""))
            if len(pieces) != 1 or not isinstance(pieces[0], str):
                raise AIError("AI review returned no single structured decision.", code="invalid_response")
            try:
                parsed = _loads(pieces[0])
            except (ValueError, TypeError, RecursionError):
                raise AIError("AI review returned invalid structured JSON.", code="invalid_response") from None
            if not isinstance(parsed, dict):
                raise AIError("AI review returned a non-object decision.", code="invalid_response")
            if not _matches_schema(parsed, schema):
                raise AIError("AI review did not match the required decision schema.", code="invalid_schema")
            # Semantic/range/word-authority validation additionally runs in the plan validator.
            return {"data": parsed}

        return self._run(model=model, prompt_version=prompt_version, identity=request, body=_json_bytes(request),
                         endpoint="responses", content_type="application/json", parse=parse)

    def transcribe(self, wav, *, language="en", prompt="", prompt_version="retention-transcribe-v1"):
        wav = Path(wav)
        try:
            if wav.stat().st_size > 24 * 1024 * 1024:
                raise AIError("Transcription audio chunk exceeds the safe upload limit.", code="invalid_audio")
            with wave.open(str(wav), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
                if duration <= 0:
                    raise AIError("Transcription audio is empty.", code="invalid_audio")
            content = wav.read_bytes()
        except (OSError, wave.Error, ZeroDivisionError):
            raise AIError("Transcription requires a readable PCM WAV file.", code="invalid_audio") from None
        if not content or len(content) > 24 * 1024 * 1024:
            raise AIError("Transcription audio chunk exceeds the safe upload limit.", code="invalid_audio")
        fields = dict(model="gpt-transcribe", response_format="json", language=str(language))
        if prompt:
            fields["prompt"] = str(prompt)
        boundary = "jaqq" + uuid.uuid4().hex
        body = bytearray()
        for name, value in fields.items():
            body.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode("utf-8"))
        body.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
                     "Content-Type: audio/wav\r\n\r\n").encode("ascii"))
        body.extend(content)
        body.extend(f"\r\n--{boundary}--\r\n".encode("ascii"))

        def parse(response):
            if not isinstance(response.get("text"), str):
                raise AIError("The transcription service returned no transcript text.", code="invalid_response")
            return {"text": response["text"]}

        return self._run(model="gpt-transcribe", prompt_version=prompt_version,
            identity=dict(fields=fields, audio_sha256=hashlib.sha256(content).hexdigest()), body=bytes(body),
            endpoint="audio/transcriptions", content_type="multipart/form-data; boundary=" + boundary,
            parse=parse, audio_seconds=duration)


OpenAIClient = OpenAIEditorClient
