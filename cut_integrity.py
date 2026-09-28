"""Final source-coordinate and word-edge invariants, after all editorial passes."""
import math
from review_core import complement, merge_intervals


def protect_words(ranges, transcript, exclusions, duration, fps, *, pre_roll=.075, post_roll=.12):
    validate([], duration, fps)
    if not all(math.isfinite(v) and v >= 0 for v in (pre_roll, post_roll)):
        raise ValueError('Invalid speech boundary handles')
    for a,b in ranges:
        if not all(math.isfinite(v) for v in (a,b)) or b <= a:
            raise ValueError('Invalid cut boundaries; export stopped')
    words = sorted((w for s in transcript for w in s.get('words', [])
                    if all(math.isfinite(v) for v in (w['start'],w['end']))
                    and 0 <= w['start'] < w['end'] <= duration+.1
                    and w['end']-w['start'] < 3), key=lambda w:(w['start'],w['end']))
    exclusions=merge_intervals(exclusions,0,duration)
    rejected=lambda w:any(x-1e-6 <= (w['start'] + w['end']) / 2 <= y+1e-6 for x, y in exclusions)
    allowed = [w for w in words if not rejected(w)]
    # A deliberately rejected word is forbidden in full, including its quiet
    # tail. Treat all deletions as barriers, even inside a selected interval.
    barriers=merge_intervals(exclusions+[(w['start'],w['end']) for w in words if rejected(w)],0,duration)
    selected_ranges=[span for a,b in merge_intervals(ranges,0,duration)
                     for span in complement(barriers,a,b)]
    changes, result = [], []
    if selected_ranges != [list(r) for r in ranges]:
        changes.append(dict(before=[list(r) for r in ranges],after=selected_ranges,
                            reason='Enforce all editorial exclusions before protecting words'))
    for a, b in selected_ranges:
        # A waveform edge can pass the midpoint of a quiet word. Any meaningful
        # surviving audio overlap warrants protecting that word, unless the word
        # belongs to an explicit editorial deletion (filtered above).
        selected = [w for w in allowed if min(b,w['end'])-max(a,w['start'])>.015]
        left, right = a, b
        hard_left,hard_right=0,math.floor(duration*fps+1e-6)/fps
        if selected:
            left = min(left, min(w['start'] for w in selected) - pre_roll)
            right = max(right, max(w['end'] for w in selected) + post_roll)
        for x, y in barriers:
            if y <= a + 1e-6:hard_left=max(hard_left,y)
            if x >= b - 1e-6:hard_right=min(hard_right,x)
        left = max(0, math.floor(left * fps + 1e-6) / fps)
        right = min(duration, math.ceil(right * fps - 1e-6) / fps)
        left=max(left,math.ceil(hard_left*fps-1e-6)/fps)
        right=min(right,math.floor(hard_right*fps+1e-6)/fps)
        if right <= left:
            continue
        if abs(left-a) > .001 or abs(right-b) > .001:
            changes.append(dict(before=[a,b], after=[left,right], reason='Protect retained word edges and quantize to source frames'))
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    validate(result, duration, fps)
    return result, changes


def validate(ranges, duration, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source frame rate')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid source duration')
    previous = 0
    for a, b in ranges:
        if not all(math.isfinite(v) for v in (a,b)) or not 0 <= a < b <= duration + .001 or a < previous - .001:
            raise ValueError('Invalid or overlapping cut boundaries; export stopped')
        if round(b * fps) <= round(a * fps):
            raise ValueError('Empty frame range; export stopped')
        previous = b
