"""Real local-model verification with a generated speech/video fixture.

This is an explicit integration check, never part of the offline unit suite. It
does not open Resolve or alter a project. The resulting MKV checks the actual
selected frame map through an independent encoder; it is not a Resolve render.
On Windows, a missing narration.wav is generated with the installed SAPI voice.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import time
import traceback
import wave

from app_paths import ROOT, add_dependencies
add_dependencies()
import av
import numpy as np
from PIL import Image, ImageDraw


NARRATION = '''<speak>Today I am testing the blue wireless mouse.
<silence msec="1000"/>This mouse feels comfortable during long gaming sessions.
<silence msec="500"/>Let me try that again.
<silence msec="600"/>This mouse feels comfortable even after several hours of playing games.
<silence msec="1100"/>The battery lasts for forty hours on a single charge.
<silence msec="500"/>Sorry, I mean fifty hours on a single charge.
<silence msec="1100"/>The textured side grip also helps me keep control during fast movements.
<silence msec="1700"/>I would recommend the blue version to a friend.</speak>'''


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def progress(message):
    print(message, flush=True)


def make_fixture(folder):
    source = folder / 'synthetic-spoken-source.mp4'
    narration = folder / 'narration.wav'
    if not narration.is_file():
        destination = str(narration.resolve()).replace("'", "''")
        speech = NARRATION.replace("'", "''")
        command = f'''$ErrorActionPreference='Stop'
$voice=New-Object -ComObject SAPI.SpVoice
$stream=New-Object -ComObject SAPI.SpFileStream
$stream.Format.Type=22
$stream.Open('{destination}',3,$false)
$voice.AudioOutputStream=$stream
$voice.Rate=0
$voice.Speak('{speech}',8) | Out-Null
$stream.Close()
'''
        subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                       check=True, timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    (folder / 'narration-source.txt').write_text(NARRATION, encoding='utf-8')
    with av.open(str(narration)) as container:
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=48000)
        chunks = [out.to_ndarray() for frame in container.decode(audio=0)
                  for out in resampler.resample(frame)]
        chunks.extend(out.to_ndarray() for out in resampler.resample(None))
    signal = np.concatenate(chunks, axis=1)
    # Explicit leading/trailing quiet and frame-aligned sample length.
    signal = np.pad(signal, ((0, 0), (24000, 24000)))
    frames = math.ceil(signal.shape[1] / 1600)
    signal = np.pad(signal, ((0, 0), (0, frames * 1600 - signal.shape[1])))
    with av.open(str(source), 'w') as container:
        video = container.add_stream('libx264rgb', rate=30)
        video.width, video.height, video.pix_fmt = 320, 180, 'rgb24'
        video.options = {'crf': '0', 'preset': 'ultrafast'}
        audio = container.add_stream('aac', rate=48000)
        audio.layout = 'mono'
        audio.metadata['name'] = 'Microphone synthetic narration'
        for index in range(frames):
            picture = Image.new('RGB', (320, 180), ((index * 11) % 128, 32, 64))
            draw = ImageDraw.Draw(picture)
            draw.text((12, 30), 'SYNTHETIC SPEECH TEST', fill='white')
            draw.text((12, 60), f'Source frame {index}', fill='white')
            draw.text((12, 85), f'Source time {index/30:.3f}s', fill='white')
            frame = av.VideoFrame.from_image(picture)
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in video.encode(frame):
                container.mux(packet)
            frame = av.AudioFrame.from_ndarray(signal[:, index*1600:(index+1)*1600],
                                               format='fltp', layout='mono')
            frame.sample_rate, frame.pts, frame.time_base = 48000, index*1600, Fraction(1, 48000)
            for packet in audio.encode(frame):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode(None):
                container.mux(packet)
    return dict(source=str(source), frames=frames, duration=frames/30,
                audio_samples=signal.shape[1], origin='Windows SAPI synthesized narration',
                human_recording=False)


def verify_models(folder):
    from model_assets import status
    ready = status()
    if not ready['ready']:
        raise RuntimeError('Models are not installed: ' + ', '.join(ready['missing']))
    from semantic_review import encode
    vectors = encode(['This mouse is comfortable.', 'The mouse feels comfortable.',
                      'The battery lasts fifty hours.'])
    assert vectors.shape == (3, 384) and np.isfinite(vectors).all()
    from contextual_takes import entailment
    meaning = entailment([('The mouse feels comfortable.', 'The mouse is comfortable.'),
                          ('The battery lasts fifty hours.', 'The battery lasts forty hours.')])
    assert len(meaning) == 2 and all(set(row) >= {'entailment', 'neutral', 'contradiction'} for row in meaning)
    # This explicit runtime check ensures the visual and game-audio models run
    # even though the narration fixture intentionally has only one audio track.
    from analyze_speech import extract
    from combat_detection import gunfire_scores, ScreenCheck
    wav = folder / 'retention-smoke.wav'
    extract(folder / 'synthetic-spoken-source.mp4', 0, 0, 3, wav)
    audio_scores = gunfire_scores(wav, folder / f'gunfire-smoke-{time.time_ns()}.json', progress, lambda: False)
    screen = ScreenCheck().classify(Image.new('RGB', (320, 180), (70, 80, 90)))
    assert audio_scores and len(screen['scores']) == 8 and np.isfinite(screen['scores']).all()
    from local_editor import Editor
    with Editor() as editor:
        started = time.monotonic()
        editorial = editor.verify('This mouse feels comfortable during long gaming sessions.',
                                  'This mouse feels comfortable even after several hours of playing games.',
                                  context={'before': 'Today I am testing the blue wireless mouse.'})
        seconds = time.monotonic() - started
    if editorial.get('status') == 'incomplete':
        raise AssertionError('Local editor returned an incomplete answer: ' + repr(editorial))
    return dict(ready=ready, embeddings_shape=list(vectors.shape), meaning=meaning,
                yamnet_windows=len(audio_scores), clip=screen, editorial=editorial,
                editorial_seconds=seconds)


def run_plan(folder):
    from automatic_cut import plan
    started = time.monotonic()
    result = plan(folder / 'synthetic-spoken-source.mp4', 0, progress)
    save(folder / 'automatic-plan.json', result)
    transcript_path = ROOT / 'analysis' / f"recovered-transcript-{result['source_key']}-run-{result['analysis_run_id']}.json"
    transcript = json.loads(transcript_path.read_text(encoding='utf-8'))
    save(folder / 'recognized-transcript.json', transcript)
    recognized = ' '.join(row['text'] for row in transcript).lower()
    # Concrete useful detail should be heard by the real Whisper model.
    assert 'textured' in recognized and 'grip' in recognized, recognized
    from script_review import readable
    selected = readable(result['script_review'])
    (folder / 'selected-script.txt').write_text(selected, encoding='utf-8')
    assert 'textured' in selected.lower() and 'grip' in selected.lower(), selected
    return dict(seconds=time.monotonic()-started, source_seconds=result['duration'],
                selected_seconds=sum(b-a for a, b in result['kept']),
                clips=len(result['kept']), analysis_run_id=result['analysis_run_id'],
                recognized_segments=len(transcript), script_review=result['script_review'],
                unique_grip_detail_preserved=True)


def verify_export(folder):
    from review_core import export_review, build_frame_map
    from review_media import audio_sidecars
    result = json.loads((folder / 'automatic-plan.json').read_text(encoding='utf-8'))
    source = result['source']
    sidecars = audio_sidecars(source, progress)
    assert sidecars and all(row['packet_copy_verified'] for row in sidecars)
    export_review(folder / 'export', source, result['duration'], result['fps'], [], {},
                  keep_ranges=result['kept'], audio=sidecars,
                  timeline_prefix='SYNTHETIC INTEGRATION CHECK - ')
    mapped = build_frame_map(result['kept'], result['duration'], result['fps'])
    with av.open(source) as container:
        pictures = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    with av.open(source) as container:
        samples = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
    with av.open(sidecars[0]['path']) as container:
        copied = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
    assert np.array_equal(samples, copied), 'Packet-preserved audio changed when decoded'
    indices = [index for row in mapped for index in range(row['source_start_frame'], row['source_end_frame'])]
    selected_audio = np.concatenate([samples[:, row['source_start_frame']*1600:row['source_end_frame']*1600]
                                     for row in mapped], axis=1)
    assert selected_audio.shape[1] == len(indices)*1600
    rendered = folder / 'independent-frame-map-output.mkv'
    with av.open(str(rendered), 'w') as container:
        video = container.add_stream('ffv1', rate=30)
        video.width, video.height, video.pix_fmt = 320, 180, 'bgr0'
        audio = container.add_stream('pcm_f32le', rate=48000)
        audio.layout = 'mono'
        for number, source_index in enumerate(indices):
            frame = av.VideoFrame.from_ndarray(pictures[source_index], format='rgb24')
            frame.pts, frame.time_base = number, Fraction(1, 30)
            for packet in video.encode(frame):
                container.mux(packet)
            frame = av.AudioFrame.from_ndarray(selected_audio[:, number*1600:(number+1)*1600], format='flt', layout='mono')
            frame.sample_rate, frame.pts, frame.time_base = 48000, number*1600, Fraction(1, 48000)
            for packet in audio.encode(frame):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode(None):
                container.mux(packet)
    with av.open(str(rendered)) as container:
        count = 0
        for number, frame in enumerate(container.decode(video=0)):
            assert np.array_equal(frame.to_ndarray(format='rgb24'), pictures[indices[number]])
            assert abs(frame.time-number/30) < .0011
            count += 1
        assert count == len(indices)
    with av.open(str(rendered)) as container:
        decoded = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
    assert np.array_equal(decoded, selected_audio)
    return dict(export_artifacts=str(folder/'export'), independent_render=str(rendered),
                selected_frames=len(indices), selected_audio_samples=selected_audio.shape[1],
                frames_exact=True, audio_samples_exact=True, audio_sidecar_packet_verified=True,
                resolve_launched=False, resolve_render_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT/'analysis/verification/synthetic-workflow')
    parser.add_argument('--stage', choices=['fixture', 'models', 'plan', 'export', 'all'], default='all')
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    report = dict(scope='Synthetic spoken media through real installed local models; no Resolve project or human recording',
                  stage=args.stage, started_at=time.time(), success=False,
                  real_recording_verified=False, resolve_render_verified=False)
    try:
        for name, operation in [('fixture', make_fixture), ('models', verify_models), ('plan', run_plan), ('export', verify_export)]:
            if args.stage in (name, 'all'):
                progress('Running ' + name)
                report[name] = operation(args.folder)
        report['success'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        progress(report['error'])
    finally:
        report['seconds'] = time.time()-report['started_at']
        save(args.folder / f'verification-{args.stage}.json', report)
    return int(not report['success'])


if __name__ == '__main__':
    raise SystemExit(main())
