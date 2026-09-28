"""Cloud text with conservative mapping onto existing acoustic word timings.

gpt-transcribe JSON does not promise exact word timestamps. Its words are never
interpolated into the timeline. Unmatched local words remain in the transcript;
unmatched cloud words are explicitly untimed context and require review.
"""
from __future__ import annotations

import copy
from difflib import SequenceMatcher
import math
from pathlib import Path
import re
import tempfile
import wave

from openai_editor import AICancelled, AIError


PROMPT_VERSION = "retention-verbatim-transcription-v1"
TRANSCRIPTION_PROMPT = (
    "Transcribe the source verbatim, including stutters, repeated words, filler words, "
    "hesitations, false starts, self-corrections, and repeated or abandoned takes. "
    "Do not clean up, paraphrase, complete, or invent speech. Preserve quoted speech."
)


def tokens(text):
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", str(text).casefold().replace("’", "'"), re.UNICODE)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def chunk_windows(duration, chunk_seconds=480., overlap_seconds=8.):
    """Each owner interval is disjoint; audio has context on both sides."""
    if not _finite(duration) or duration <= 0:
        return []
    if not _finite(chunk_seconds) or not _finite(overlap_seconds) or not 0 <= overlap_seconds < chunk_seconds / 2:
        raise ValueError("Invalid transcription chunk/overlap duration.")
    step = chunk_seconds - 2 * overlap_seconds
    result = []
    start = 0.
    while start < duration:
        end = min(duration, start + step)
        result.append(dict(source_start=max(0., start - overlap_seconds),
                           source_end=min(duration, end + overlap_seconds),
                           owner_start=start, owner_end=end))
        start = end
    return result


def _occurrences(sequence, needle):
    if not needle:
        return 0
    return sum(sequence[i:i + len(needle)] == needle for i in range(len(sequence) - len(needle) + 1))


def align_text(local_words, cloud_text, duration):
    """Return per-word evidence and untimed text. Never manufacture timestamps."""
    local_tokens, owners = [], []
    for index, word in enumerate(local_words):
        for token in tokens(word.get("text", "")):
            local_tokens.append(token)
            owners.append(index)
    cloud_tokens = tokens(cloud_text)
    matched, ambiguous, cloud_matched = set(), set(), set()
    for block in SequenceMatcher(None, local_tokens, cloud_tokens, autojunk=False).get_matching_blocks():
        if not block.size:
            continue
        phrase = local_tokens[block.a:block.a + block.size]
        uncertain = _occurrences(local_tokens, phrase) > 1 or _occurrences(cloud_tokens, phrase) > 1
        for offset in range(block.size):
            li, ci = block.a + offset, block.b + offset
            matched.add(li)
            cloud_matched.add(ci)
            if uncertain:
                ambiguous.add(li)
    indices = {i: [] for i in range(len(local_words))}
    for i, owner in enumerate(owners):
        indices[owner].append(i)
    evidence = []
    for i, word in enumerate(local_words):
        start, end = word.get("start"), word.get("end")
        valid = _finite(start) and _finite(end) and 0 <= start < end <= duration and end - start <= 3.
        confidence = word.get("probability")
        locally_reliable = (valid and _finite(confidence) and confidence >= .55
                            and word.get("timing_reliable", True) is not False
                            and word.get("alignment_reliable", True) is not False)
        own = indices[i]
        exact = bool(own) and all(index in matched for index in own)
        repeated = any(index in ambiguous for index in own)
        status = ("invalid_local_timing" if not valid else "local_only" if not exact
                  else "ambiguous_token_match" if repeated else "exact_token_match")
        reliable = bool(locally_reliable and exact and not repeated)
        evidence.append(dict(alignment_status=status, timing_source=word.get("timing_source", "faster-whisper"),
                             timing_reliable=reliable, alignment_reliable=reliable,
                             cloud_text_matched=exact, timing_is_estimate=True))
    unaligned = [token for index, token in enumerate(cloud_tokens) if index not in cloud_matched]
    return evidence, dict(cloud_tokens=len(cloud_tokens), local_tokens=len(local_tokens),
                          matched_tokens=len(matched), untimed_cloud_tokens=unaligned,
                          token_match_ratio=len(matched) / max(1, len(local_tokens)),
                          untimed_cloud_text=" ".join(unaligned))


def transcribe_aligned(wav, local_transcript, client, progress=lambda _: None,
                       cancel=lambda: False, *, chunk_seconds=480., overlap_seconds=8.):
    """Transcribe overlapping audio and annotate a copy of the acoustic transcript.

    Failure is raised with successful chunk responses already cached for retry.
    The caller can then preserve the original source and display a fallback state.
    """
    if cancel():
        raise AICancelled()
    transcript = copy.deepcopy(local_transcript)
    words = [word for segment in transcript for word in segment.get("words", [])]
    for index, word in enumerate(words):
        word.setdefault("source_word_id", f"word-{index:06d}")
    try:
        audio = wave.open(str(wav), "rb")
    except (OSError, wave.Error):
        raise AIError("Source audio could not be opened for transcription.", code="invalid_audio") from None
    reports = []
    ledger_start = len(client.ledger) if hasattr(client, "ledger") else 0
    with audio:
        rate = audio.getframerate()
        duration = audio.getnframes() / rate
        bytes_per_second = rate * audio.getnchannels() * audio.getsampwidth()
        # Leave a generous header margin beneath the transport's 24 MiB bound.
        max_seconds = (23 * 1024 * 1024) / bytes_per_second
        seconds = min(chunk_seconds, max_seconds)
        overlap = min(overlap_seconds, seconds / 4)
        windows = chunk_windows(duration, seconds, overlap)
        if not windows:
            raise AIError("The source audio is empty.", code="invalid_audio")
        with tempfile.TemporaryDirectory(prefix="jaqq-transcribe-") as tmp:
            for index, window in enumerate(windows):
                if cancel():
                    raise AICancelled()
                progress(f"Transcribing source audio with gpt-transcribe ({index + 1}/{len(windows)})…")
                start_frame = round(window["source_start"] * rate)
                end_frame = min(audio.getnframes(), round(window["source_end"] * rate))
                audio.setpos(start_frame)
                chunk_path = Path(tmp) / "audio.wav"
                with wave.open(str(chunk_path), "wb") as chunk:
                    chunk.setparams(audio.getparams())
                    chunk.writeframes(audio.readframes(end_frame - start_frame))
                result = client.transcribe(chunk_path, prompt=TRANSCRIPTION_PROMPT, prompt_version=PROMPT_VERSION)
                if cancel():
                    raise AICancelled()
                context_words = [w for w in words if _finite(w.get("start")) and _finite(w.get("end"))
                                 and w["end"] > window["source_start"] and w["start"] < window["source_end"]]
                evidence, stats = align_text(context_words, result["text"], duration)
                for word, proof in zip(context_words, evidence):
                    midpoint = (word["start"] + word["end"]) / 2
                    if window["owner_start"] <= midpoint < window["owner_end"]:
                        word.update(proof)
                        word["transcription_chunk"] = index
                reports.append(dict(window, chunk=index, model="gpt-transcribe", text=result["text"],
                    prompt_version=PROMPT_VERSION, **stats,
                    request_id=result.get("request_id"), cached=bool(result.get("cached")),
                    api_call_occurred=bool(result.get("api_call_occurred")), usage=result.get("usage"),
                    cost_usd_estimate=result.get("cost_usd_estimate"),
                    elapsed_seconds=result.get("elapsed_seconds"),
                    timing_source="existing acoustic word boundaries; no cloud word timestamps"))
    for word in words:
        if "transcription_chunk" not in word:
            word.update(timing_reliable=False, alignment_reliable=False,
                        alignment_status="invalid_local_timing", timing_source=word.get("timing_source", "faster-whisper"))
    for segment in transcript:
        segment["transcription_model"] = "gpt-transcribe"
        segment["timing_source"] = "faster-whisper"
        segment["timing_reliable"] = bool(segment.get("words")) and all(
            w.get("timing_reliable", False) for w in segment.get("words", []))
        segment["cloud_context"] = [
            {key: report[key] for key in ("chunk", "source_start", "source_end", "owner_start", "owner_end",
                                          "text", "untimed_cloud_tokens")}
            for report in reports
            if any(_finite(w.get("start")) and _finite(w.get("end"))
                   and w["end"] > report["source_start"] and w["start"] < report["source_end"]
                   for w in segment.get("words", []))]
    uncertain = [dict(source_word_id=w["source_word_id"], start=w.get("start"), end=w.get("end"),
                      text=w.get("text", ""), status=w.get("alignment_status"))
                 for w in words if not w.get("timing_reliable")]
    metadata = dict(model="gpt-transcribe", prompt_version=PROMPT_VERSION, status="completed",
                    timing_source="faster-whisper", alignment_method="monotonic exact token matching",
                    precise_cloud_word_timestamps=False, duration=duration, chunks=reports,
                    uncertain_words=uncertain, requires_review=bool(uncertain) or any(r["untimed_cloud_tokens"] for r in reports),
                    local_words_preserved=len(words), reliable_words=sum(bool(w.get("timing_reliable")) for w in words),
                    protected_alignment_regions=[dict(start=r["owner_start"], end=r["owner_end"],
                        reason="Cloud transcription contains speech without local word timestamps", chunk=r["chunk"])
                        for r in reports if r["untimed_cloud_tokens"]],
                    ledger=client.ledger[ledger_start:] if hasattr(client, "ledger") else [])
    return transcript, metadata
