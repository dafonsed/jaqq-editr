"""Bounded review renders from the editor's canonical frame map, plus measured QA.

This does not replace Resolve's production renderer. It makes source/after joins
auditionable before export. A passing decode audit is never a listening verdict.
"""
from fractions import Fraction
from pathlib import Path
import difflib
from bisect import bisect_left
import math
import re

from app_paths import add_dependencies
add_dependencies()
import av
import numpy as np
from review_core import build_frame_map, map_source_span, save_json


def map_annotations(annotations, frame_map):
    """Ripple source captions/effect intervals through all retained occurrences."""
    mapped = []
    for item in annotations:
        for span in map_source_span(frame_map, item['start'], item['end']):
            mapped.append(dict(item, start=span['output_start'], end=span['output_end'],
                               source_start=span['source_start'], source_end=span['source_end'],
                               clip_index=span['clip_index'], partial=span['partial']))
    return sorted(mapped,key=lambda item:(item['start'],item['end']))


def _caption_sidecar(path,annotations):
    def stamp(value):
        milliseconds=round(value*1000)
        seconds,ms=divmod(milliseconds,1000);minutes,seconds=divmod(seconds,60);hours,minutes=divmod(minutes,60)
        return f'{hours:02}:{minutes:02}:{seconds:02},{ms:03}'
    cues=[]
    for row in annotations:
        if row.get('text') and row['end']>row['start']:
            cues.append(f"{len(cues)+1}\n{stamp(row['start'])} --> {stamp(row['end'])}\n{row['text'].strip()}\n")
    if not cues:return None
    target=Path(path).with_suffix('.srt')
    target.write_text('\n'.join(cues),encoding='utf-8')
    return str(target)


def preview_ranges(result, start, end, context=1.5):
    """Small, chronological after-preview around a proposed cut."""
    lo, hi = max(0, start-context), min(result['duration'], end+context)
    for segment in result.get('transcript',[]):
        for word in segment.get('words',[]):
            if word['start'] < lo < word['end']:lo=max(0,word['start']-.075)
            if word['start'] < hi < word['end']:hi=min(result['duration'],word['end']+.12)
    return [[max(lo, a), min(hi, b)] for a, b in result['kept']
            if min(hi, b) > max(lo, a)]


def _audio_range(source, index, start, samples, sample_rate, channels):
    """Read by actual PTS; uncovered source audio remains measurable silence."""
    data = np.zeros((channels, samples), dtype=np.float32)
    covered = np.zeros(samples, dtype=bool)
    with av.open(str(source)) as container:
        stream = container.streams.audio[index]
        video = container.streams.video[0]
        origin = float((video.start_time or 0)*video.time_base)
        container.seek(max(0, int((origin+max(0, start-.1))*av.time_base)))
        resampler = av.AudioResampler(format='fltp', layout='mono' if channels == 1 else 'stereo', rate=sample_rate)
        def copy(frame):
            if frame.pts is None:
                raise ValueError('Review audio has missing timestamps; cannot verify synchronization.')
            pos = round((float(frame.pts*frame.time_base)-origin-start)*sample_rate)
            values = frame.to_ndarray()
            a, b = max(0, pos), min(samples, pos+values.shape[1])
            if b > a:
                data[:, a:b] = values[:, a-pos:b-pos]
                covered[a:b] = True
        for frame in container.decode(stream):
            if frame.time is not None and frame.time-origin > start+samples/sample_rate+.2:
                break
            for converted in resampler.resample(frame):
                copy(converted)
        for converted in resampler.resample(None):
            copy(converted)
    return data, int(np.count_nonzero(~covered))


def _video_range(source, first, last, fps):
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0)*stream.time_base)
        container.seek(max(0, int((origin+first/fps)*av.time_base)))
        wanted = first
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Review video has a missing presentation timestamp.')
            position = (float(frame.pts*frame.time_base)-origin)*fps
            index = round(position)
            if index < first:
                continue
            if index >= last:
                break
            if index != wanted or abs(index-position) > .15:
                raise ValueError('Review video is variable-rate or has missing frames. Conform it before editing.')
            yield frame
            wanted += 1
        if wanted != last:
            raise ValueError('Source ended before every approved review frame could be decoded.')


def _render_blocks(frame_map, fps, sample_rate, seconds=30):
    """Bound memory even when the kept plan is one multi-hour source interval."""
    block_frames=max(1,round(seconds*fps))
    for row in frame_map:
        for first in range(row['source_start_frame'],row['source_end_frame'],block_frames):
            last=min(first+block_frames,row['source_end_frame'])
            output_first=row['output_start_frame']+first-row['source_start_frame']
            # Maintain a single audio sample phase across internal decode blocks.
            sample_offset=round(output_first*sample_rate/fps)-round(row['output_start_frame']*sample_rate/fps)
            yield dict(row,source_start_frame=first,source_end_frame=last,
                source_start=row['source_start']+sample_offset/sample_rate,
                output_start_frame=output_first,output_end_frame=output_first+last-first,
                at_clip_start=first==row['source_start_frame'],at_clip_end=last==row['source_end_frame'])


def render_preview(source, kept, duration, fps, output, *, progress=lambda text: None,
                   cancel=lambda: False, max_width=960, lossless=False, annotations=(), fade_ms=0):
    """Encode exact selected picture frames and all source audio tracks.

    The preview preserves every sample position; no asynchronous seek-skipping or
    undocumented audio overlap is used. Hard joins are measured and flagged by QA.
    All tracks remain separate. Captions are supplied as a mapped sidecar artifact.
    """
    frame_map = build_frame_map(kept, duration, fps)
    if not math.isfinite(fade_ms) or not 0 <= fade_ms <= 10:
        raise ValueError('Review fades must be between zero and ten milliseconds.')
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('A review render must not overwrite the source recording.')
    if output.exists():
        raise ValueError('This review output already exists; choose a new output path.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem+'.partial'+output.suffix)
    rate, sample_rate = Fraction(fps).limit_denominator(100000), 48000
    with av.open(str(source)) as container:
        source_video = container.streams.video[0]
        width, height = source_video.width, source_video.height
        if max_width and width > max_width:
            height = round(height*max_width/width)
            width = max_width
        width, height = max(2, width//2*2), max(2, height//2*2)
        channels = [len(s.codec_context.layout.channels) for s in container.streams.audio]
        if any(c not in (1, 2) for c in channels):
            raise ValueError('Review rendering preserves mono/stereo tracks; multichannel routing needs Resolve.')
        track_names = [s.metadata.get('name', f'Audio {i+1}') for i, s in enumerate(container.streams.audio)]
    gaps, fades = [], []
    try:
        with av.open(str(temporary), 'w') as target:
            video = target.add_stream('ffv1' if lossless else 'libx264', rate=rate)
            video.width, video.height = width, height
            video.pix_fmt = 'bgr0' if lossless else 'yuv420p'
            if not lossless:
                video.options = {'crf': '18', 'preset': 'veryfast'}
            audio = []
            for count, name in zip(channels, track_names):
                stream = target.add_stream('pcm_s16le' if lossless else 'aac', rate=sample_rate)
                stream.layout = 'mono' if count == 1 else 'stereo'
                stream.metadata['title'] = name
                if not lossless:
                    stream.bit_rate = 192000
                audio.append(stream)
            for row in _render_blocks(frame_map,fps,sample_rate):
                if cancel():
                    raise InterruptedError('Review render stopped.')
                progress(f"Rendering review clip {row['clip_index']} of {len(frame_map)}…")
                audio_start = round(row['output_start_frame']*sample_rate/fps)
                audio_end = round(row['output_end_frame']*sample_rate/fps)
                buffers = []
                for index, count in enumerate(channels):
                    values, missing = _audio_range(source, index, row['source_start'],
                        audio_end-audio_start, sample_rate, count)
                    # Optional short fades only in low-energy handles. Never move
                    # a sample or shorten the picture to make room for smoothing.
                    fade_samples=min(round(fade_ms*sample_rate/1000), values.shape[1]//2)
                    for edge, active in [('start',row['at_clip_start'] and row['clip_index']>1), ('end',row['at_clip_end'] and row['clip_index']<len(frame_map))]:
                        if fade_samples and active:
                            section=values[:,:fade_samples] if edge=='start' else values[:,-fade_samples:]
                            if float(np.max(np.abs(section))) <= .05:
                                gain=np.sin(np.linspace(0, math.pi/2, fade_samples)).astype(np.float32)
                                section *= gain if edge=='start' else gain[::-1]
                                fades.append(dict(clip=row['clip_index'],track=index,edge=edge,milliseconds=fade_ms))
                    buffers.append(values)
                    if missing > sample_rate*.05:
                        gaps.append(dict(clip=row['clip_index'], track=index, missing_samples=missing))
                for local, picture in enumerate(_video_range(source, row['source_start_frame'], row['source_end_frame'], fps)):
                    if cancel():
                        raise InterruptedError('Review render stopped.')
                    position = row['output_start_frame']+local
                    picture = picture.reformat(width=width, height=height, format=video.pix_fmt)
                    picture.pts, picture.time_base = position, 1/rate
                    for packet in video.encode(picture):
                        target.mux(packet)
                    first = round(position*sample_rate/fps)
                    last = round((position+1)*sample_rate/fps)
                    for stream, count, values in zip(audio, channels, buffers):
                        samples = values[:, first-audio_start:last-audio_start]
                        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples),
                            format='fltp', layout='mono' if count == 1 else 'stereo')
                        frame.sample_rate, frame.pts, frame.time_base = sample_rate, first, Fraction(1, sample_rate)
                        for packet in stream.encode(frame):
                            target.mux(packet)
            for stream in [video, *audio]:
                for packet in stream.encode(None):
                    target.mux(packet)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    mapped = map_annotations(annotations, frame_map)
    report = dict(source=str(source), output=str(output), fps=fps, frame_map=frame_map,
        output_frames=frame_map[-1]['output_end_frame'], source_audio_gaps=gaps,
        audio_tracks=len(channels), annotations=mapped,caption_sidecar=_caption_sidecar(output,mapped),
        caption_delivery='Mapped SRT sidecar; captions are not embedded or burned into the review video.',
        rendering='Review encode using the same source/output frame map as Resolve',
        smoothing='Optional low-energy handle fades; no time overlap' if fades else 'Original audio hard joins; audition for clicks before final export.',
        applied_fades=fades,
        listening_verified=False, resolve_render_verified=False)
    save_json(output.with_suffix(output.suffix+'.render.json'), report)
    return report


def compare_render_words(expected, observed):
    """Text discrepancy evidence from a separate render transcription, not proof."""
    normalize = lambda value: re.findall(r"[\w]+(?:['’][\w]+)?", value.lower())
    before = normalize(' '.join(w.get('text', '') for w in expected))
    after = normalize(' '.join(w.get('text', '') for w in observed))
    differences = []
    for kind, a, b, x, y in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if kind != 'equal':
            differences.append(dict(kind=kind, expected=' '.join(before[a:b]), observed=' '.join(after[x:y])))
    return dict(status='REVIEW' if differences else 'TEXT_MATCH', differences=differences,
                note='Independent ASR can omit or hallucinate words; listen to every discrepancy.')


def transcribe_render(path, microphone, duration, *, progress=lambda text: None,
                      cancel=lambda: False, model=None):
    """Fresh local decode of the rendered audio for word-loss/duplication flags."""
    from app_paths import ROOT
    from analyze_speech import extract
    folder=ROOT/'.model-cache'/'speech-small.en'
    if model is None:
        if not all((folder/name).is_file() for name in ('model.bin','config.json','tokenizer.json')):
            return dict(status='NOT VERIFIED',reason='Local speech timing model is unavailable.',words=[])
        from faster_whisper import WhisperModel
        model=WhisperModel(str(folder),device='cpu',compute_type='int8',cpu_threads=4)
    wav=Path(path).with_suffix('.qa.wav')
    extract(path,microphone,0,duration,wav)
    progress('Re-listening to the rendered audio for word-loss and repeated-phrase flags…')
    segments,_=model.transcribe(str(wav),language='en',vad_filter=False,word_timestamps=True,
        beam_size=5,condition_on_previous_text=False)
    words=[]
    for segment in segments:
        if cancel():
            raise InterruptedError('Rendered speech check stopped.')
        words.extend(dict(start=w.start,end=w.end,text=w.word,probability=w.probability) for w in segment.words)
    report=dict(status='TRANSCRIBED',model=str(folder),words=words,
                limitation='Acoustic ASR discrepancy check; it does not establish naturalness or human preference.')
    save_json(Path(path).with_suffix(Path(path).suffix+'.transcript.json'),report)
    return report


def _measure_audio(path, index, duration, frame_map):
    """Streaming metrics keep full-video QA bounded to codec frames and RMS bins."""
    with av.open(str(path)) as container:
        stream=container.streams.audio[index]
        rate=stream.codec_context.sample_rate
        resampler=av.AudioResampler(format='fltp',layout=stream.codec_context.layout.name,rate=rate)
        expected_samples=round(duration*rate)
        window=max(1,round(.02*rate))
        pending=np.empty(0,dtype=np.float64)
        rms=[];joins=[];count=0;clipped=0;peak=0.;first=None;end=None;timing_gap=False
        previous=None;last_values=None;packets=0
        boundaries={round(row['output_start']*rate):row['output_start'] for row in frame_map[1:]}
        boundary_samples=sorted(boundaries)
        def measure(frame):
            nonlocal pending,count,clipped,peak,first,end,timing_gap,previous,last_values,packets
            timestamp=float(frame.pts*frame.time_base) if frame.pts is not None else None
            if packets==0:first=timestamp
            if timestamp is None or (previous is not None and abs(timestamp-previous)>.002):timing_gap=True
            end=timestamp+frame.samples/rate if timestamp is not None else None
            previous=end;packets+=1
            values=frame.to_ndarray()[:,:max(0,expected_samples-count)]
            if not values.shape[1]:return
            clipped+=int(np.count_nonzero(np.abs(values)>=.999))
            peak=max(peak,float(np.max(np.abs(values))))
            first_boundary=bisect_left(boundary_samples,count)
            last_boundary=bisect_left(boundary_samples,count+values.shape[1])
            for boundary in boundary_samples[first_boundary:last_boundary]:
                offset=boundary-count
                left=values[:,offset-1] if offset else last_values
                if left is not None:
                    jump=float(np.max(np.abs(values[:,offset]-left)))
                    if jump>.15:joins.append(dict(time=boundaries[boundary],amplitude_step=jump))
            last_values=values[:,-1].copy()
            energy=np.mean(values.astype(np.float64)**2,axis=0)
            pending=np.concatenate([pending,energy])
            complete=len(pending)//window*window
            if complete:
                rms.extend(np.sqrt(pending[:complete].reshape(-1,window).mean(axis=1)).tolist())
                pending=pending[complete:]
            count+=values.shape[1]
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):measure(converted)
        for converted in resampler.resample(None):measure(converted)
        if len(pending):rms.append(float(np.sqrt(pending.mean())))
    return dict(sample_rate=rate,start=first,end=end,decoded_samples=count,
                clipped_samples=clipped,peak=peak,rms=rms,joins=joins,timing_gap=timing_gap,packets=packets)


def audit_render(path, frame_map, fps, *, microphone=0, long_pause=.9,
                 expected_words=None, observed_words=None, progress=lambda text: None,cancel=lambda:False):
    """Decode rendered media and report timing, silence, clipping and join risks."""
    expected_frames = frame_map[-1]['output_end_frame']
    expected_duration = expected_frames/fps
    issues=[];video_count=0;video_drift=False;previous_picture=None
    joins={right['output_start_frame']:(left,right) for left,right in zip(frame_map,frame_map[1:])
           if left['source_end_frame']!=right['source_start_frame']}
    tolerance = max(.002, .1/fps)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if cancel():raise InterruptedError('Rendered timing check stopped.')
            timestamp=float(frame.pts*frame.time_base) if frame.pts is not None else None
            if timestamp is None or abs(timestamp-video_count/fps)>tolerance:video_drift=True
            # A small decoded-frame comparison exposes severe visual splices;
            # it is an audition cue, never evidence to delete source content.
            if video_count in joins or video_count+1 in joins:
                picture=frame.reformat(width=96,height=54,format='rgb24').to_ndarray()
                if video_count in joins and previous_picture is not None:
                    difference=float(np.mean(np.abs(picture.astype(float)-previous_picture.astype(float)))/255)
                    if difference>.22:
                        left,right=joins[video_count]
                        issues.append(dict(kind='visual_jump',time=video_count/fps,mean_channel_change=difference,
                            source_before=left['source_end_frame']/fps,source_after=right['source_start_frame']/fps,
                            reason='Large decoded picture change at a splice. Inspect in Resolve; a crop, B-roll or transition may help.'))
                previous_picture=picture
            video_count+=1
            if video_count % 1000==0:progress(f'Checking rendered video timing: {video_count}/{expected_frames} frames…')
        tracks = len(container.streams.audio)
    if video_count != expected_frames:
        issues.append(dict(kind='video_frame_count', expected=expected_frames, actual=video_count))
    if video_drift:
        issues.append(dict(kind='video_timestamp_drift', reason='Decoded PTS do not match the approved frame sequence'))
    audio_results = []
    for index in range(tracks):
        if cancel():raise InterruptedError('Rendered audio check stopped.')
        progress(f'Checking rendered audio track {index+1} of {tracks}…')
        measured=_measure_audio(path,index,expected_duration,frame_map)
        sample_rate=measured['sample_rate']
        if not measured['packets']:
            issues.append(dict(kind='missing_audio', track=index))
            continue
        # AAC padding is permitted up to one codec packet, never counted as speech.
        first,end=measured['start'],measured['end']
        drift = None if end is None else end-expected_duration
        if first is None or abs(first) > 1/sample_rate+.002 or drift is None or abs(drift) > 1024/sample_rate+.002:
            issues.append(dict(kind='audio_video_duration_drift', track=index, start=first, end=end, expected=expected_duration))
        if measured['timing_gap']:issues.append(dict(kind='audio_timestamp_gap',track=index))
        clipped=measured['clipped_samples']
        if clipped:
            issues.append(dict(kind='clipping', track=index, samples=clipped))
        window = max(1, round(.02*sample_rate))
        rms=np.array(measured['rms'])
        # Conservative acoustic silence threshold; room tone is not classified as speech.
        threshold = min(.006, max(.0005, float(np.percentile(rms, 15))*.7)) if len(rms) else .0005
        quiet, start = [], None
        for i, level in enumerate([*rms, threshold+1]):
            if level < threshold and start is None:
                start = i*window/sample_rate
            elif level >= threshold and start is not None:
                stop = min(i*window/sample_rate, expected_duration)
                if stop-start >= long_pause:
                    quiet.append(dict(start=start, end=stop, duration=stop-start))
                start = None
        if index == microphone:
            issues.extend(dict(kind='long_quiet_span', track=index, **span,
                reason='May be intentional; compare the approved context and listen.') for span in quiet)
        issues.extend(dict(kind='join_transient', track=index, **join) for join in measured['joins'])
        audio_results.append(dict(track=index, sample_rate=sample_rate, decoded_samples=measured['decoded_samples'],
            start=first, duration_drift=drift, peak=measured['peak'],
            clipped_samples=clipped, long_quiet_spans=quiet))
    geometry_failures = [i for i in issues if i['kind'] in ('video_frame_count', 'video_timestamp_drift',
        'audio_video_duration_drift', 'audio_timestamp_gap', 'missing_audio')]
    if tracks == 0:
        issues.append(dict(kind='missing_audio', reason='Rendered video has no audio tracks'))
        geometry_failures = issues
    word_check = compare_render_words(expected_words, observed_words) if expected_words is not None and observed_words is not None else dict(
        status='NOT VERIFIED', reason='An independent transcription of the rendered audio was not supplied.')
    report = dict(output=str(path), geometry='FAIL' if geometry_failures else 'PASS',
        video_frames=video_count, expected_frames=expected_frames, expected_duration=expected_duration,
        audio=audio_results, issues=issues, word_check=word_check,
        listening='NOT VERIFIED', visual_continuity='NOT VERIFIED', perceptual_lip_sync='NOT VERIFIED',
        human_preference='NOT MEASURED', captions='Mapped annotations require visual review; no caption OCR performed.')
    save_json(Path(path).with_suffix(Path(path).suffix+'.qa.json'), report)
    return report
