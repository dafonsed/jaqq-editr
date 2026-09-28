"""Generate openly labeled local speech fixtures without API calls/private media.

Windows System.Speech provides representative speech. Word timings must still be
measured by the real timing pipeline; text is never assigned guessed timestamps.
"""
import json
from fractions import Fraction
from pathlib import Path
import subprocess
import wave

from app_paths import ROOT, add_dependencies
add_dependencies()
import av
import numpy as np
from PIL import Image, ImageDraw


def generate(folder=None):
    folder = Path(folder or ROOT/'analysis'/'ai-evaluation')
    folder.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT/'evaluation_cases.json').read_text(encoding='utf-8'))['cases']
    config = folder/'speech-input.json'
    config.write_text(json.dumps(cases), encoding='utf-8')
    # Paths and dialogue are data files, never interpolated shell code.
    script = folder/'synthesize.ps1'
    script.write_text('''param([string]$ConfigPath, [string]$OutputFolder)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$cases = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $voice.SelectVoice('Microsoft David Desktop')
    foreach ($case in $cases) {
        for ($i = 0; $i -lt $case.parts.Count; $i++) {
            $path = Join-Path $OutputFolder ($case.id + '-part-' + $i + '.wav')
            $voice.SetOutputToWaveFile($path)
            $voice.Speak([string]$case.parts[$i])
            $voice.SetOutputToNull()
        }
    }
} finally { $voice.Dispose() }
''', encoding='utf-8')
    if not all((folder/f"{case['id']}-part-{i}.wav").is_file()
               for case in cases for i in range(len(case['parts']))):
        subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                        '-ConfigPath', str(config), '-OutputFolder', str(folder)], check=True,
                       capture_output=True, timeout=120,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    outputs = []
    for case in cases:
        parts, rate = [], None
        for index in range(len(case['parts'])):
            with wave.open(str(folder/f"{case['id']}-part-{index}.wav")) as handle:
                if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                    raise ValueError('Fixture voice must produce mono PCM16')
                if rate is not None and rate != handle.getframerate():
                    raise ValueError('Fixture audio rates changed')
                rate = handle.getframerate()
                parts.append(np.frombuffer(handle.readframes(handle.getnframes()), dtype='<i2'))
        blocks = [np.zeros(round(.25*rate), dtype='<i2')]
        source_parts, protected = [], []
        cursor = .25
        for index, part in enumerate(parts):
            source_parts.append(dict(start=cursor, end=cursor+len(part)/rate, text=case['parts'][index]))
            blocks.append(part)
            cursor += len(part)/rate
            if index < len(case['gaps']):
                gap = case['gaps'][index]
                blocks.append(np.zeros(round(gap*rate), dtype='<i2'))
                if case.get('protect_gap'):
                    protected.append(dict(start=cursor, end=cursor+gap, reason='Intentional reveal in evaluation script'))
                cursor += gap
        blocks.append(np.zeros(round(.25*rate), dtype='<i2'))
        values = np.concatenate(blocks)
        fps = 30
        frames = int(np.ceil(len(values)/rate*fps))
        duration = frames/fps
        values = np.pad(values, (0, max(0, round(duration*rate)-len(values))))
        source = folder/(case['id']+'.mp4')
        with av.open(str(source), 'w') as output:
            video = output.add_stream('libx264', rate=fps)
            video.width, video.height, video.pix_fmt = 480, 270, 'yuv420p'
            video.options = {'crf':'20', 'preset':'fast'}
            audio = output.add_stream('aac', rate=rate)
            audio.layout = 'mono'
            audio.metadata['title'] = 'Synthetic commentary'
            for index in range(frames):
                canvas = Image.new('RGB', (480,270), '#152436')
                draw = ImageDraw.Draw(canvas)
                draw.text((20,20), 'SYNTHETIC EVALUATION / '+case['id'], fill='#74dccd')
                draw.text((20,60), f'Source time {index/fps:06.2f} s / frame {index}', fill='white')
                draw.text((20,95), 'Voice: Windows David / no real user recording', fill='#acb9cb')
                for offset, part in enumerate(case['parts']):
                    draw.text((20,135+offset*28), part[:65], fill='white')
                draw.rectangle((20,235,20+round(440*index/max(1,frames-1)),242), fill='#74dccd')
                frame = av.VideoFrame.from_ndarray(np.array(canvas), format='rgb24')
                frame.pts, frame.time_base = index, Fraction(1,fps)
                for packet in video.encode(frame):
                    output.mux(packet)
                a, b = round(index*rate/fps), round((index+1)*rate/fps)
                frame = av.AudioFrame.from_ndarray(values[None,a:b].copy(), format='s16', layout='mono')
                frame.sample_rate, frame.pts, frame.time_base = rate, a, Fraction(1,rate)
                for packet in audio.encode(frame):
                    output.mux(packet)
            for stream in (video,audio):
                for packet in stream.encode(None):
                    output.mux(packet)
        outputs.append(dict(case, source=str(source), duration=duration, protected_pauses=protected,
                            source_parts=source_parts, word_timestamps='Not supplied; use acoustic timing'))
        print('Created', case['id'], round(duration,2), 'seconds', flush=True)
    (folder/'fixtures.json').write_text(json.dumps(dict(synthetic=True,cases=outputs),indent=2),encoding='utf-8')
    return outputs


if __name__ == '__main__':
    generate()
