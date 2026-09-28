"""OpenAI proposal orchestration using the editor's existing timing/cut engine.

No model response reaches the exporter without source-word and waveform validation.
Successful stage artifacts survive retries. The local-only pipeline is unchanged.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from app_paths import ROOT
from automatic_cut import Cancelled, extract, key_for, probe
from cut_integrity import validate
from edit_manifest import build_manifest, cut_signature
from pause_cleanup import pause_settings
from review_core import build_frame_map, complement, map_source_span, read_transcript, save_json

VERSION = 'openai-retention-1'
TIMING_VERSION = 'whisper-source-timing-1'


def _check(cancel):
    if cancel():
        raise Cancelled()


def _speech_model():
    folder = ROOT / '.model-cache' / 'speech-small.en'
    if not all((folder / name).is_file() for name in ('model.bin', 'config.json', 'tokenizer.json')):
        raise ValueError('The local speech timing model is missing. Set up speech timing in the editor first.')
    return folder


def _local_timing(wav, cache, duration, progress, cancel):
    """Local ASR supplies acoustic timestamps; cloud text never invents them."""
    folder = _speech_model()
    identity = {name: [p.stat().st_size, p.stat().st_mtime_ns]
                for name in ('model.bin', 'config.json', 'tokenizer.json')
                for p in [folder / name]}
    stamp = hashlib.sha256(json.dumps([TIMING_VERSION, identity], sort_keys=True).encode()).hexdigest()[:16]
    target = cache / ('local-timing-' + stamp + '.json')
    existing = read_transcript(target)
    if existing is not None:
        progress('Resuming saved acoustic word timing…')
        return existing, {'version': TIMING_VERSION, 'model': str(folder), 'identity': identity,
                          'cached': True, 'seconds': 0.0}
    from faster_whisper import WhisperModel
    started = time.monotonic()
    model = WhisperModel(str(folder), device='cpu', compute_type='int8', cpu_threads=4)
    segments, _ = model.transcribe(str(wav), language='en', vad_filter=True,
        word_timestamps=True, beam_size=5, condition_on_previous_text=True,
        initial_prompt='Verbatim speech, including hesitations, repeated words, false starts and self-corrections.')
    transcript = []
    for segment in segments:
        _check(cancel)
        progress(f'Aligning source speech: {min(duration, segment.end):.0f} / {duration:.0f} seconds…')
        transcript.append(dict(start=float(segment.start), end=float(segment.end), text=segment.text,
            words=[dict(start=float(w.start), end=float(w.end), text=w.word, probability=float(w.probability),
                        timing_source='faster-whisper', timing_reliable=bool(
                            0 <= w.start < w.end <= duration and w.end-w.start < 3 and w.probability >= .60))
                   for w in segment.words or []]))
    from speech_recheck import recover
    transcript, rechecks = recover(transcript, wav, model, progress, cancel)
    save_json(cache / 'local-rechecks.json', rechecks)
    save_json(target, transcript)
    return transcript, {'version': TIMING_VERSION, 'model': str(folder), 'identity': identity,
                        'cached': False, 'seconds': round(time.monotonic()-started, 3)}


def _protected_annotations(items, duration):
    result = []
    for row in items:
        if not isinstance(row, dict) or not row.get('reason'):
            raise ValueError('Each intentional pause needs source start/end and a reason.')
        a, b = row.get('start'), row.get('end')
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                   for t in (a, b)) or not 0 <= a < b <= duration:
            raise ValueError('Intentional pause is outside the source timeline.')
        result.append(dict(row))
    return result


def _other_speech(wav, cancel):
    """Other participants on separate tracks cannot disappear with a mic cut.

    VAD is only a conservative preservation signal, not speaker identification.
    Music/noise false positives cost pacing; they never authorize deletion.
    """
    import wave
    import numpy as np
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    from review_core import merge_intervals
    spans = []
    with wave.open(str(wav)) as audio:
        rate = audio.getframerate()
        if (audio.getnchannels(), audio.getsampwidth(), rate) != (1, 2, 16000):
            raise ValueError('Other-track voice checks require source-aligned mono PCM16.')
        total = audio.getnframes()
        # Overlap prevents a speaker turn at a processing boundary being lost.
        for first in range(0, total, rate*28):
            _check(cancel)
            audio.setpos(first)
            signal = np.frombuffer(audio.readframes(rate*30), dtype='<i2').astype(np.float32)/32768
            detected = get_speech_timestamps(signal, VadOptions(min_speech_duration_ms=120,
                min_silence_duration_ms=150, speech_pad_ms=100), sampling_rate=rate)
            spans.extend((max(0,(first+r['start'])/rate-.1),
                          min(total/rate,(first+r['end'])/rate+.1)) for r in detected)
    return merge_intervals(spans, 0, total/rate)


def _output_words(transcript, kept, duration, fps):
    """Map captions against exactly the same frames as linked audio and video."""
    mapping = build_frame_map(kept, duration, fps)
    result, flags = [], []
    for segment in transcript:
        for word in segment.get('words', []):
            a, b = word.get('start'), word.get('end')
            if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (a, b)) or b <= a:
                continue
            for span in map_source_span(mapping, a, b):
                item = dict(word, source_start=a, source_end=b,
                            start=span['output_start'], end=span['output_end'])
                if span['partial']:
                    flags.append(dict(source_start=a, source_end=b,
                        reason='A retained word intersects an edit boundary; inspect the join.', status='review'))
                result.append(item)
    return result, flags


def _assemble(result, report):
    kept = report['kept']
    validate(kept, result['duration'], result['fps'])
    if not kept:
        raise ValueError('The proposed edit would remove the entire recording.')
    result['kept'] = kept
    result['ai_plan']['decisions'] = report['decisions']
    result['ai_plan']['expected_retained_word_ids'] = report.get('expected_retained_word_ids', [])
    result['cut_signature'] = cut_signature(kept, result['fps'])
    result['clips'] = len(kept)
    result['edited_duration'] = sum(round(b*result['fps'])-round(a*result['fps']) for a, b in kept)/result['fps']
    words, word_flags = _output_words(result['transcript'], kept, result['duration'], result['fps'])
    result['output_words'] = words
    result['frame_map'] = build_frame_map(kept, result['duration'], result['fps'])
    from speech_evidence import output_locations
    flags = list(result.get('source_evidence_flags', [])) + report.get('flags', []) + word_flags
    result['evidence_flags'] = output_locations(flags, kept)
    applied = [d for d in report['decisions'] if d.get('status') == 'accepted' and
               d.get('action') in ('REMOVE', 'SHORTEN_PAUSE', 'REPLACE_EARLIER_TAKE_WITH_LATER_TAKE')]
    result['edit_manifest'] = build_manifest(kept, result['duration'], result['fps'], applied,
                                           result['evidence_flags'])
    result['ai_plan']['metrics']['number_of_cuts'] = len(complement(kept, 0, result['duration']))
    result['ai_plan']['metrics']['removed_seconds'] = round(result['duration']-result['edited_duration'], 4)
    result['script_review'] = {'flags': [], 'changes': applied,
        'script': [{'source_start': w['source_start'], 'source_end': w['source_end'],
                    'timeline_start': w['start'], 'timeline_end': w['end'], 'text': w['text']}
                   for w in words]}
    return result


def plan_ai(source, mic, progress=lambda _: None, cancel=lambda: False, *,
            intensity='balanced', protected_pauses=(), client=None):
    """Analyze a recording and return a reviewable plan; never export automatically."""
    from openai_editor import OpenAIEditorClient, AIError, AICancelled
    from ai_transcription import transcribe_aligned
    from ai_decisions import build_candidates, make_decisions, validate_decisions
    source = Path(source)
    duration, fps, names = probe(source)
    if not isinstance(mic, int) or isinstance(mic, bool) or not 0 <= mic < len(names):
        raise ValueError('Choose a valid commentary track.')
    started = time.monotonic()
    identity = key_for(source)
    settings = pause_settings(intensity)
    annotations = _protected_annotations(protected_pauses, duration)
    cache = ROOT / 'analysis' / 'openai' / f'{identity}-mic{mic}'
    cache.mkdir(parents=True, exist_ok=True)
    if client is None:
        client = OpenAIEditorClient(cache / 'api', cancel=cancel)
        if not client.configured:
            raise ValueError('Set OPENAI_API_KEY in the launcher environment to use OpenAI editing, or select Local only.')
    run_id = uuid.uuid4().hex
    wav = cache / 'source-microphone.wav'
    _check(cancel)
    progress('Extracting source-aligned microphone audio…')
    extract(source, mic, 0, duration, wav)
    transcript, timing = _local_timing(wav, cache, duration, progress, cancel)
    for index, word in enumerate(sorted((w for s in transcript for w in s.get('words', [])),
                                         key=lambda w: (w['start'], w['end']))):
        word.setdefault('source_word_id', f'word-{index:06d}')
    _check(cancel)
    fallback = None
    try:
        progress('Transcribing with gpt-transcribe and checking acoustic word alignment…')
        transcript, transcription = transcribe_aligned(wav, transcript, client, progress, cancel)
    except AICancelled:
        raise Cancelled() from None
    except AIError as exc:
        # Cloud failure is visible and resumable, never permission to delete speech.
        fallback = str(exc)
        transcription = {'status': 'unavailable', 'reason': fallback, 'model': 'gpt-transcribe'}
        for segment in transcript:
            for word in segment.get('words', []):
                word['timing_reliable'] = False
                word['alignment_reliable'] = False
    from speech_evidence import audit
    protected, evidence_flags = audit(transcript, wav, duration)
    # Preserve entire phrases whenever cloud/local correspondence is uncertain.
    for segment in transcript:
        if any(w.get('alignment_reliable') is False or w.get('timing_reliable') is False
               for w in segment.get('words', [])):
            a, b = max(0, segment['start']), min(duration, segment['end'])
            if a < b:
                protected.append((a, b))
                evidence_flags.append(dict(source_start=a, source_end=b, status='review',
                    reason='Transcription/alignment disagreement; phrase retained for listening review.'))
    protected.extend((p['start'], p['end']) for p in annotations)
    for chunk in transcription.get('chunks', []):
        if chunk.get('untimed_cloud_tokens'):
            # Without an acoustic anchor, a new cloud word could be anywhere in
            # this owner interval. Silence and text alone cannot authorize loss.
            a, b = chunk['owner_start'], chunk['owner_end']
            protected.append((a, b))
            evidence_flags.append(dict(source_start=a, source_end=b, status='review',
                reason='Cloud transcription contains speech without reliable local timing; source window retained.'))
    # Use the existing visual/action detector when the recording has a separate game track.
    events = []
    other_track_evidence = []
    for other in range(len(names)):
        if other == mic:
            continue
        progress('Protecting speech on the other audio tracks…')
        other_wav = cache/f'game-{other}.wav'
        extract(source, other, 0, duration, other_wav)
        try:
            turns = _other_speech(other_wav, cancel)
            protected.extend(turns)
            other_track_evidence.append(dict(track=other, method='local Silero voice activity; preserve only', spans=turns))
        except Cancelled:
            raise
        except (OSError, ValueError, RuntimeError, ImportError):
            protected.append((0,duration))
            evidence_flags.append(dict(source_start=0,source_end=duration,status='review',
                reason=f'Other audio track {other+1} could not be checked for speech; source preserved.'))
    game = next((i for i, n in enumerate(names) if i != mic and ('game' in n.lower() or 'system' in n.lower())),
                next((i for i in range(len(names)) if i != mic), None))
    combat_checks = {'method': 'No separate game audio track'}
    if game is not None:
        game_wav = cache / f'game-{game}.wav'
        extract(source, game, 0, duration, game_wav)
        try:
            from combat_detection import detect
            events, combat_checks = detect(source, game_wav, identity+'-ai-'+str(game), duration, progress, cancel)
        except Cancelled:
            raise
        except (OSError, ValueError, RuntimeError, ImportError):
            # Unknown game context cannot be treated as expendable dead air.
            protected.append((0, duration))
            combat_checks = {'method': 'Unavailable; all source retained for visual review'}
            evidence_flags.append(dict(source_start=0, source_end=duration, status='review',
                reason='Gameplay context could not be analyzed; no automatic deletion is authorized.'))
    _check(cancel)
    candidates = build_candidates(transcript, wav, duration, fps, settings, protected, events)
    if fallback:
        proposals = {'decisions': [], 'calls': [], 'warnings': [fallback], 'prompt_version': VERSION,
                     'models': {'primary': 'gpt-6-sol', 'escalation': 'gpt-6-astra'}}
        protected.append((0, duration))
    else:
        try:
            proposals = make_decisions(candidates, transcript, client, settings=settings,
                                       progress=progress, cancel=cancel)
        except AICancelled:
            raise Cancelled() from None
    report = validate_decisions(proposals['decisions'], candidates, transcript,
                                duration, fps, wav, settings, protected)
    _check(cancel)
    if key_for(source) != identity:
        raise ValueError('The recording changed during analysis; analyze the completed recording again.')
    metrics = client.summary() if hasattr(client, 'summary') else {'api_usage': 'Test adapter; no live API measurements'}
    seconds = time.monotonic()-started
    metrics.update(processing_seconds=round(seconds, 3), source_minutes=duration/60,
                   processing_seconds_per_source_minute=round(seconds/(duration/60), 3))
    cost = metrics.get('cost_usd_estimate')
    metrics['cost_usd_per_source_minute_estimate'] = cost/(duration/60) if cost is not None else None
    if hasattr(client, 'video_summary'):
        metrics['video_total'] = client.video_summary()
    ai_plan = dict(proposals, candidates=candidates, decisions=report['decisions'],
        original_decisions=deepcopy(report['decisions']), raw_decisions=deepcopy(proposals['decisions']),
        transcription=transcription, timing=timing, metrics=metrics, overrides={},
        status='review_required', pipeline_version=VERSION,
        limitations=['Word timing is acoustic ASR alignment, not sample-exact ground truth.',
                     'Speaker labels are available only when supplied by the timing transcript.',
                     'Text and audio cues cannot establish all visual/comedic intent. Review proposed joins.'])
    result = dict(source=str(source), duration=duration, fps=fps, microphone=mic, game_audio=game,
        source_key=identity, analysis_run_id=run_id, analysis_mode='resumable', pipeline_version=VERSION,
        settings=settings, protected_pauses=annotations, protected_ranges=protected, action=events,
        raw_action=events, combat_checks=combat_checks, transcript=transcript, analysis_audio=str(wav),
        other_track_speech=other_track_evidence,
        speech_model=timing['model'], source_evidence_flags=evidence_flags, ai_plan=ai_plan,
        duplicate_takes=0, removed_duplicates=[], target_duration=None,
        timeline_verified=False, render_verified=False, verified=False,
        selection_method='Source retained until a structured AI decision passes source-word, waveform and frame validation.')
    _assemble(result, report)
    target = cache / f'plan-{run_id}.json'
    result['plan_path'] = str(target)
    save_json(target, result)
    progress('AI proposals are ready. Review cuts and source playback before exporting.')
    return result


def rebuild_plan(result, overrides, *, persist=True):
    """Restore cuts or shrink validated deletions; never widen model authority."""
    from ai_decisions import validate_decisions
    updated = deepcopy(result)
    if not isinstance(overrides, dict):
        raise ValueError('Review changes must identify existing candidates.')
    plan = updated['ai_plan']
    original = {d['candidate_id']: d for d in plan['original_decisions']}
    if set(overrides)-set(original):
        raise ValueError('Unknown cut in review changes.')
    # Revalidate original proposals against source evidence on every review edit.
    report = validate_decisions(plan['raw_decisions'], plan['candidates'], updated['transcript'],
        updated['duration'], updated['fps'], updated['analysis_audio'], updated['settings'],
        updated.get('protected_ranges', ()))
    decisions = report['decisions']
    fps = updated['fps']
    deletions = []
    for decision in decisions:
        candidate_id = decision['candidate_id']
        override = overrides.get(candidate_id, {})
        if set(override)-{'enabled', 'start', 'end'}:
            raise ValueError('Unknown review adjustment.')
        enabled = override.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError('A cut must be enabled or restored.')
        deleting = decision.get('status') == 'accepted' and decision['action'] in (
            'REMOVE', 'SHORTEN_PAUSE', 'REPLACE_EARLIER_TAKE_WITH_LATER_TAKE')
        if not enabled:
            decision.update(action='KEEP', status='accepted', reason='Restored by the user. '+decision['reason'],
                            user_restored=True, enabled=False, remove_word_ids=[])
            candidate = next(c for c in plan['candidates'] if c['id'] == candidate_id)
            decision['keep_word_ids'] = candidate['review_word_ids'][:]
            continue
        if not deleting:
            if 'start' in override or 'end' in override:
                raise ValueError('Only validated cuts can have their boundaries adjusted. Uncertain speech is retained.')
            continue
        a = override.get('start', decision['start'])
        b = override.get('end', decision['end'])
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (a,b)):
            raise ValueError('Cut boundaries must be finite numbers.')
        a, b = math.ceil(a*fps-1e-6)/fps, math.floor(b*fps+1e-6)/fps
        if not decision['start']-1e-6 <= a < b <= decision['end']+1e-6:
            raise ValueError('Adjusted boundaries must stay inside the validated cut; restore it to keep the whole span.')
        changed = a != decision['start'] or b != decision['end']
        for segment in updated['transcript'] if changed else []:
            for word in segment.get('words', []):
                x, y = word['start'], word['end']
                if (x < a < y or x < b < y):
                    raise ValueError('That boundary would split a spoken word.')
                # A partially restored word needs its complete acoustic handles too.
                previously_removed = decision['start'] <= x < y <= decision['end']
                if previously_removed and not a <= x < y <= b and ((x < b and y > a) or
                    0 <= a-y < updated['settings']['trailing']-1e-6 or
                    0 <= x-b < updated['settings']['leading']-1e-6):
                    raise ValueError('That boundary leaves insufficient space around a retained word.')
        decision.update(start=a, end=b, source_start=a, source_end=b,
                        user_adjusted=changed)
        deletions.append((a,b))
    kept = complement(deletions, 0, math.floor(updated['duration']*fps+1e-6)/fps)
    if any(b-a < max(2/fps, .08) for a,b in kept):
        raise ValueError('The adjustment would leave an unplayably short retained clip.')
    report.update(decisions=decisions, kept=kept,
        expected_retained_word_ids=[w.get('source_word_id') for s in updated['transcript'] for w in s.get('words', [])
            if not any(a <= w['start'] and w['end'] <= b for a,b in deletions)])
    plan['overrides'] = deepcopy(overrides)
    _assemble(updated, report)
    # A preview of the previous geometry cannot verify this new edit.
    updated.pop('preview', None)
    rendered = updated.get('review_render', {})
    if (rendered.get('cut_signature') != updated['cut_signature'] or
            rendered.get('source_key') != updated['source_key']):
        updated.pop('review_render', None)
    updated['render_verified'] = False
    if persist and updated.get('plan_path'):
        save_json(updated['plan_path'], updated)
    return updated


def validate_for_export(result):
    """A signed-looking JSON field alone does not authorize an AI cut."""
    checked = rebuild_plan(result, result['ai_plan'].get('overrides', {}), persist=False)
    if checked['cut_signature'] != cut_signature(result['kept'], result['fps']):
        raise ValueError('AI cut geometry does not match its validated source decisions; export stopped.')
    return checked


def write_ai_artifacts(folder, result):
    """Human-readable decisions and mapped subtitles share the exported frame map."""
    if not result.get('ai_plan'):
        return
    folder = Path(folder)
    save_json(folder / 'ai-decisions.json', result['ai_plan'])
    save_json(folder / 'output-words.json', result.get('output_words', []))
    def stamp(value):
        milliseconds = max(0, round(value*1000))
        seconds, ms = divmod(milliseconds, 1000)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f'{hours:02}:{minutes:02}:{seconds:02},{ms:03}'
    words = result.get('output_words', [])
    lines = []
    for index, word in enumerate(words, 1):
        lines.extend([str(index), f"{stamp(word['start'])} --> {stamp(word['end'])}", word['text'].strip(), ''])
    (folder / 'aligned-captions.srt').write_text('\n'.join(lines), encoding='utf-8')
