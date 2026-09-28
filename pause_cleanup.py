"""Contextual microphone pause budgets, with explicit protected spans and audit.

Energy is evidence of silence, not a verdict about speech or a visual moment.
Word spans are protected even when their recognition confidence is low.
"""
import math

import numpy as np

from review_core import complement, merge_intervals
from speech_edges import microphone_energy


def pause_settings(intensity='balanced'):
    """Return editable pause budgets in seconds; callers persist this with the plan."""
    presets = {
        'natural': (.12, .18, .40, .65, .95, .24, .08, 1.2),
        'balanced': (.075, .12, .28, .45, .70, .20, .06, .9),
        'tight': (.06, .09, .20, .30, .50, .16, .04, .7),
    }
    intensity = str(intensity).lower()
    if intensity not in presets:
        raise ValueError('Unknown pause intensity: ' + intensity)
    settings = dict(zip(('leading', 'trailing', 'hesitation', 'sentence', 'topic',
                         'restart', 'min_cut', 'audit_gap'), presets[intensity]))
    settings.update(intensity=intensity, max_noise_rms=.02, noise_ratio=1.6,
                    speech_margin=.22)
    return settings


def _settings(settings):
    result = pause_settings((settings or {}).get('intensity', 'balanced'))
    if settings:
        unknown = set(settings) - set(result)
        if unknown:
            raise ValueError('Unknown pause settings: ' + ', '.join(sorted(unknown)))
        result.update(settings)
    for key, value in result.items():
        if key != 'intensity' and (not isinstance(value, (int, float)) or
                                   not math.isfinite(value) or value < 0):
            raise ValueError('Invalid pause setting: ' + key)
    if result['noise_ratio'] < 1 or not 0 < result['speech_margin'] < 1:
        raise ValueError('Pause noise ratio/margin must separate speech from room tone')
    return result


def _spans(items):
    return [(float(x['start']), float(x['end'])) if isinstance(x, dict)
            else (float(x[0]), float(x[1])) for x in items]


def _words(transcript, exclusions):
    result = []
    for segment_index, segment in enumerate(transcript):
        for word in segment.get('words', []):
            a, b = word['start'], word['end']
            if not all(math.isfinite(v) for v in (a, b)) or not 0 < b-a < 3:
                continue
            if any(x-1e-6 <= (a+b)/2 <= y+1e-6 for x, y in exclusions):
                continue
            result.append(dict(word, segment_index=segment_index))
    return sorted(result, key=lambda w: (w['start'], w['end']))


def _context(a, b, words, exclusions):
    previous = next((w for w in reversed(words) if w['end'] <= a+.021), None)
    following = next((w for w in words if w['start'] >= b-.021), None)
    if any(x < b+.25 and y > a-.25 for x, y in exclusions):
        kind = 'restart'
    elif previous is None:
        kind = 'leading'
    elif following is None:
        kind = 'trailing'
    elif previous.get('pause_after') == 'topic':
        kind = 'topic'
    elif previous.get('text', '').rstrip().endswith(('.', '?', '!')):
        kind = 'sentence'
    else:
        kind = 'hesitation'
    return kind, previous, following


def bridge_word_gaps(ranges, transcript, *, exclusions=(), protected_pauses=()):
    """Reconsider only reliable word-free gaps for the chosen pause budget.

    Speech selection already drops inter-phrase silence. Joining safe adjacent
    speech ranges here gives each pacing mode the same source gap to evaluate.
    The waveform cleaner still decides whether that gap is actually removable.
    """
    exclusions=_spans(exclusions)
    protected=_spans(protected_pauses)
    words=_words(transcript,exclusions)
    # Include rejected words in this check: deleted text must never be bridged
    # solely because it was filtered out of the retained transcript.
    all_words=_words(transcript,())
    result=[];changes=[]
    for a,b in merge_intervals(ranges):
        if result:
            start,end=result[-1][1],a
            previous=next((w for w in reversed(words)
                           if w['end']<=start+.001 and w['end']>result[-1][0]),None)
            following=next((w for w in words if w['start']>=end-.001 and w['start']<b),None)
            safe=(previous is not None and following is not None
                  and previous.get('probability',1)>=.5 and following.get('probability',1)>=.5
                  and not any(x<end and y>start for x,y in exclusions+protected)
                  and not any(w['start']<end-1e-6 and w['end']>start+1e-6 for w in all_words))
            if safe:
                changes.append(dict(start=start,end=end,reason='Reliable word-free gap reconsidered under pacing settings'))
                result[-1][1]=b
                continue
        result.append([a,b])
    return result,changes


def _analysis(ranges, wav, events, transcript, settings, exclusions, protected_pauses):
    config = _settings(settings)
    energy, step, duration = microphone_energy(wav, .01)
    exclusions = merge_intervals(_spans(exclusions), 0, duration)
    ranges = [span for a, b in merge_intervals(ranges, 0, duration)
              for span in complement(exclusions, a, b)]
    words = _words(transcript, exclusions)
    visual = _spans(events)
    intentional = _spans(protected_pauses)
    # Tiny safety handles protect low-energy consonants and word endings. The
    # separate context budget below accounts for these handles, not adds to it.
    word_guards = [(w['start']-config['leading'], w['end']+config['trailing'])
                   for w in words]
    guards = merge_intervals(word_guards + [(a-.1, b+.1) for a, b in visual]
                             + intentional, 0, duration)
    near_silence = .0008
    adaptive = near_silence
    if len(energy) and words:
        speech = np.zeros(len(energy), dtype=bool)
        for w in words:
            if w.get('probability', 1) >= .5:
                speech[max(0, int(w['start']/step)):min(len(energy), math.ceil(w['end']/step))] = True
        floor = float(np.percentile(energy, 15))
        reference = float(np.percentile(energy[speech], 60)) if speech.any() else 0
        # A noise floor is usable only when actual recognized speech is clearly
        # louder. A uniformly quiet/noisy recording must never become all silence.
        if reference >= max(.002, floor*3):
            adaptive = max(near_silence, min(config['max_noise_rms'],
                           floor*config['noise_ratio'], reference*config['speech_margin']))
    return config, energy, step, duration, ranges, words, guards, visual, intentional, exclusions, adaptive


def clean_pauses(ranges, wav, events=(), *, transcript=(), settings=None,
                 exclusions=(), protected_pauses=()):
    """Shorten supported dead air without crossing words or editorial barriers.

    Intentional/visual pauses are explicit source spans. Missing/uncertain speech
    context permits only absolute near-silence cuts, leaving other gaps for audit.
    """
    (config, energy, step, duration, ranges, words, guards, visual, intentional,
     exclusions, adaptive) = _analysis(ranges, wav, events, transcript, settings,
                                       exclusions, protected_pauses)
    if not len(energy):
        return ranges, []
    cuts, decisions = [], []
    word_spans = [(w['start'], w['end']) for w in words]
    boundaries = np.diff(np.r_[False, energy <= adaptive, False].astype(int))
    for first, last in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
        start, end = first*step, min(duration, last*step)
        for a, b in ranges:
            for lo, hi in complement(word_spans, max(a, start), min(b, end)):
                kind, previous, following = _context(lo, hi, words, exclusions)
                adjacent = [w for w in (previous, following) if w is not None]
                reliable = bool(adjacent) and all(w.get('probability', 1) >= .5 for w in adjacent)
                local = energy[max(first, int(lo/step)):min(last, math.ceil(hi/step))]
                absolute = bool(np.max(local, initial=0) <= .0008)
                if not absolute and not reliable:
                    continue
                waveform_only = previous is None and following is None
                if waveform_only:
                    # Legacy callers without timestamps still need air on both
                    # sides of an interior acoustic gap.
                    kind = 'leading' if lo <= a+.021 else 'trailing' if hi >= b-.021 else 'hesitation'
                budget = config[kind]
                if waveform_only and kind == 'hesitation':
                    ratio = config['trailing']/max(config['leading']+config['trailing'], .001)
                    left, right = lo+budget*ratio, hi-budget*(1-ratio)
                elif kind == 'leading':
                    left, right = lo, hi-budget
                elif kind == 'trailing':
                    left, right = lo+budget, hi
                elif previous is None:
                    left, right = lo, hi-budget
                elif following is None:
                    left, right = lo+budget, hi
                else:
                    # Allocate the budget around the actual word gap. This prevents
                    # word protection from adding another pair of handles afterward.
                    gap_start, gap_end = max(lo, previous['end']), min(hi, following['start'])
                    ratio = config['trailing']/max(config['leading']+config['trailing'], .001)
                    left, right = gap_start+budget*ratio, gap_end-budget*(1-ratio)
                if right <= left:
                    continue
                for x, y in complement(guards, left, right):
                    if y-x+1e-8 < config['min_cut']:
                        continue
                    cuts.append((x, y))
                    decisions.append(dict(start=x, end=y, pause_type=kind,
                        retained_budget=budget, confidence=.98 if absolute else .8,
                        evidence='near-silence' if absolute else 'room tone below recognized speech',
                        reason=f'{kind.capitalize()} pause shortened to the {config["intensity"]} pacing budget'))
    selected = [span for a, b in ranges for span in complement(cuts, a, b)]
    for row in decisions:
        row['output_time'] = sum(max(0, min(b, row['start'])-a) for a, b in selected if a < row['start'])
    return selected, decisions


def audit_pauses(ranges, wav, events=(), *, transcript=(), settings=None,
                 exclusions=(), protected_pauses=()):
    """Explain every surviving long word-free span; never mutates the plan."""
    (config, energy, step, duration, ranges, words, guards, visual, intentional,
     exclusions, adaptive) = _analysis(ranges, wav, events, transcript, settings,
                                       exclusions, protected_pauses)
    word_spans = [(w['start'], w['end']) for w in words]
    flags = []
    output = 0
    protection = [(p, q, 'Selected visual/gameplay event') for p, q in visual]
    for item in protected_pauses:
        p,q = _spans([item])[0]
        protection.append((p,q,item.get('reason','Explicit intentional pause')
                           if isinstance(item,dict) else 'Explicit intentional pause'))
    for a, b in ranges:
        for x, y in complement(word_spans, a, b):
            if y-x < config['audit_gap']:
                continue
            points=sorted({x,y}|{max(x,min(y,t)) for p,q,_ in protection for t in (p,q)})
            for lo,hi in zip(points,points[1:]):
                if hi-lo < config['audit_gap']:
                    continue
                reasons=[reason for p,q,reason in protection if p <= lo and q >= hi]
                reason='; '.join(reasons) if reasons else 'Long gap lacks confident silence or editorial justification; listen before export'
                status='protected' if reasons else 'review'
                flags.append(dict(start=lo, end=hi, source_start=lo, source_end=hi,
                    output_time=output+lo-a, duration=hi-lo, reason=reason, status=status,
                    confidence=.5 if status == 'review' else 1.0))
        output += b-a
    return flags
