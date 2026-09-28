import os
import sys
from pathlib import Path

from app_paths import ROOT, add_dependencies
add_dependencies()
os.environ['HF_HOME'] = str(ROOT / '.model-cache')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import av
import numpy as np
import json
import wave

OUT = ROOT / 'analysis'
SOURCE = Path("T:/Tom Clancy's Rainbow Six Siege/Tom Clancy's Rainbow Six Siege 2026.08.21 - 22.30.24.03.mp4")

def extract(path, index, start, end, dest):
    dest = Path(dest)
    if not np.isfinite(start) or not np.isfinite(end) or not 0 <= start < end:
        raise ValueError('Audio extraction requires a finite positive source interval')
    # A file left by a failed decode is not a usable analysis cache.
    if dest.exists():
        try:
            with wave.open(str(dest), 'rb') as w:
                valid = (w.getparams()[:3] == (1, 2, 16000) and
                         w.getnframes() == round((end-start)*16000))
                if valid:
                    remaining=w.getnframes()
                    while remaining:
                        count=min(remaining,16000*30)
                        if len(w.readframes(count))!=count*2:
                            valid=False;break
                        remaining-=count
            if valid:return
        except (OSError, EOFError, wave.Error):pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(round((end-start)*16000), dtype=np.int16)
    decoded = 0
    with av.open(str(path)) as c:
        if not 0 <= index < len(c.streams.audio):
            raise ValueError('The selected microphone track does not exist')
        stream = c.streams.audio[index]
        picture = c.streams.video[0] if c.streams.video else stream
        origin = float((picture.start_time or 0) * picture.time_base)
        c.seek(max(0, int((origin + start) * av.time_base)))
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        def copy_frame(f):
            nonlocal decoded
            if f.pts is None:
                raise ValueError('Audio timestamps are missing; cannot align speech cuts to video')
            pos = round((float(f.pts * f.time_base)-origin-start)*16000)
            values = f.to_ndarray().reshape(-1)
            lo, hi = max(0, pos), min(len(data), pos + len(values))
            if hi > lo:
                data[lo:hi] = values[lo-pos:hi-pos]
                decoded += hi-lo
        for frame in c.decode(stream):
            if frame.time is not None and frame.time-origin > end + 0.2:break
            for f in resampler.resample(frame):copy_frame(f)
        for f in resampler.resample(None):copy_frame(f)
    if not decoded:
        raise ValueError('No timestamped audio decoded in the selected interval')
    temporary = dest.with_suffix(dest.suffix + '.tmp')
    with wave.open(str(temporary), 'wb') as w:
        w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        w.writeframes(data.tobytes())
    temporary.replace(dest)

if __name__ == '__main__':
    for i in range(2):
        extract(SOURCE, i, 230, 335, OUT / f'reference-audio-{i}.wav')
    from faster_whisper import WhisperModel
    model = WhisperModel('base.en', device='cpu', compute_type='int8', cpu_threads=4,
                         download_root=str(ROOT / '.model-cache'))
    for i in range(2):
        dest = OUT / f'reference-transcript-{i}.json'
        if dest.exists():
            continue
        segs, info = model.transcribe(str(OUT / f'reference-audio-{i}.wav'),
            language='en', vad_filter=True, word_timestamps=True, beam_size=3,
            condition_on_previous_text=False)
        rows = []
        for s in segs:
            row = {'start': s.start+230, 'end': s.end+230, 'text': s.text,
                   'words': [{'start': w.start+230, 'end': w.end+230,
                              'text': w.word, 'probability': w.probability} for w in s.words]}
            rows.append(row)
            print(i, round(row['start'],2), row['text'], flush=True)
        dest.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('DONE', flush=True)
