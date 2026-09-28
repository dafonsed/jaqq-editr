"""Keep acoustic/transcript disagreements visible instead of trusting clean ASR text.

Energy is evidence of sound, not proof of speech. Unexplained sound is retained for
review; it never authorizes a word deletion or claims to recover missing syllables.
"""
import math
import wave
import numpy as np
from review_core import merge_intervals, complement


def audit(transcript, wav, duration):
    protected, flags = [], []
    reliable = []
    for segment in transcript:
        words = segment.get('words', [])
        suspect = []
        previous = -1
        for word in words:
            a, b = word.get('start'), word.get('end')
            valid = (isinstance(a, (int, float)) and isinstance(b, (int, float)) and
                     math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration + .1)
            if not valid or b-a >= 3 or a < previous-.04 or word.get('probability', 0) < .60:
                suspect.append(word.get('text', ''))
            elif valid:
                reliable.append((max(0, a-.10), min(duration, b+.15)))
            if valid:previous = b
        if suspect or (segment.get('text', '').strip() and not words):
            a = max(0, float(segment.get('start', 0)))
            b = min(duration, float(segment.get('end', duration)))
            if math.isfinite(a) and math.isfinite(b) and b > a:
                protected.append((a, b))
                flags.append(dict(source_start=a, source_end=b, confidence='uncertain',
                    reason='Unreliable word alignment; preserve the source phrase for listening review',
                    text=segment.get('text', ''), suspect_words=suspect))
    with wave.open(str(wav), 'rb') as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError('Speech evidence requires mono PCM16 audio')
        rate = handle.getframerate()
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype='<i2').astype(np.float32)/32768
    hop = max(1, round(rate*.02))
    count = len(audio)//hop
    if count:
        rms = np.sqrt(np.mean(audio[:count*hop].reshape(-1, hop)**2, axis=1))
        # Conservative: music/noise may also exceed this floor and must be reviewed.
        floor = float(np.percentile(rms, 15))
        threshold = max(.001, min(.02, floor*3))
        active = rms > threshold
        edges = np.diff(np.r_[False, active, False].astype(int))
        sounds = [(a*hop/rate, b*hop/rate) for a, b in
                  zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b-a >= 3]
        covered = merge_intervals(reliable+protected, 0, duration)
        for a, b in sounds:
            for x, y in complement(covered, a, min(b, duration)):
                if y-x < .18:continue
                protected.append((max(0, x-.075), min(duration, y+.12)))
                flags.append(dict(source_start=x, source_end=y, confidence='uncertain',
                    reason='Audible activity is not covered by reliable word timestamps; possible omitted speech or background sound'))
    return merge_intervals(protected, 0, duration), flags


def output_locations(flags, kept):
    """Attach every surviving output span; deleted flags get an explicit empty map."""
    mapped = []
    for flag in flags:
        row = dict(flag)
        a = row.get('source_start', row.get('start', 0))
        b = row.get('source_end', row.get('end', a))
        position, locations = 0, []
        for x, y in kept:
            lo, hi = max(a, x), min(b, y)
            if hi > lo:
                locations.append(dict(source_start=lo, source_end=hi,
                    output_start=position+lo-x, output_end=position+hi-x))
            position += y-x
        row['output_spans'] = locations
        if 'timeline_start' in row:
            row['timeline_start'] = locations[0]['output_start'] if locations else sum(
                max(0,min(a,y)-x) for x,y in kept if x<a)
        row['retained_in_output'] = bool(locations)
        mapped.append(row)
    return mapped
