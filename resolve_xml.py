"""FCP7 XML timeline interchange using the editor's canonical frame map.

Format reference: https://developer.apple.com/library/archive/documentation/
AppleApplications/Reference/FinalCutPro_XML/Basics/Basics.html
This writes an importable timeline; it never claims that Resolve imported it.
"""
import math
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET


def rate_values(fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('XML export requires a finite positive frame rate.')
    base = round(fps)
    if abs(fps-base) < 1e-6:
        return base, False
    if abs(fps-base*1000/1001) < .0001:
        return base, True
    raise ValueError('This frame rate cannot be represented by the FCP7 XML format.')


def _add(parent, tag, text=None, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    if text is not None:
        element.text = str(text)
    return element


def _rate(parent, rate):
    element = _add(parent, 'rate')
    _add(element, 'timebase', rate[0])
    _add(element, 'ntsc', 'TRUE' if rate[1] else 'FALSE')


def _timecode(parent, rate):
    element = _add(parent, 'timecode')
    _rate(element, rate)
    _add(element, 'string', '00:00:00:00')
    _add(element, 'frame', 0)
    _add(element, 'displayformat', 'NDF')


def _probe(source, audio):
    from app_paths import add_dependencies
    add_dependencies()
    import av
    with av.open(str(source)) as container:
        video = container.streams.video[0]
        dimensions = video.codec_context.width, video.codec_context.height
        if video.sample_aspect_ratio not in (None, 1):
            raise ValueError('XML export currently requires square-pixel video.')
    tracks = []
    for item in audio:
        path = Path(item['path']).resolve()
        with av.open(str(path)) as container:
            if len(container.streams.audio) != 1:
                raise ValueError('XML audio sidecars must each contain one audio stream.')
            stream = container.streams.audio[0]
            channels = len(stream.codec_context.layout.channels)
            if channels not in (1, 2):
                raise ValueError('XML export currently supports mono or stereo source tracks.')
            tracks.append(dict(path=path, name=item.get('name', path.stem),
                               channels=channels, sample_rate=stream.codec_context.sample_rate))
    return dimensions, tracks


def _video_format(parent, dimensions, rate):
    sample = _add(parent, 'samplecharacteristics')
    _rate(sample, rate)
    _add(sample, 'width', dimensions[0])
    _add(sample, 'height', dimensions[1])
    _add(sample, 'anamorphic', 'FALSE')
    _add(sample, 'pixelaspectratio', 'square')
    _add(sample, 'fielddominance', 'none')


def _audio_format(parent, sample_rate):
    sample = _add(parent, 'samplecharacteristics')
    _add(sample, 'depth', 16)
    _add(sample, 'samplerate', sample_rate)


def write_timeline(path, source, duration, fps, frame_map, audio, *, name='Retention cut'):
    """Write a new xmeml v5 timeline with linked source video/audio channels.

    ``frame_map`` comes from review_core.build_frame_map. Source/output frame
    ends remain exclusive, including repeated or reordered source ranges.
    Each stereo sidecar is represented by two linked channel tracks, grouped as
    a stereo pair. All channels link to the corresponding picture segment.
    """
    rate = rate_values(fps)
    if not math.isfinite(duration) or duration <= 0 or not frame_map:
        raise ValueError('XML export requires a nonempty valid source/frame map.')
    source_frames = round(duration*fps)
    position = 0
    for row in frame_map:
        keys = ('source_start_frame', 'source_end_frame', 'output_start_frame', 'output_end_frame')
        if any(type(row.get(key)) is not int for key in keys):
            raise ValueError('XML frame boundaries must be integers.')
        a, b, start, end = (row[key] for key in keys)
        if not (0 <= a < b <= source_frames and start == position and end-start == b-a):
            raise ValueError('XML frame map contains a gap, overlap or invalid range.')
        position = end
    source = Path(source).resolve()
    dimensions, tracks = _probe(source, audio)
    root = ET.Element('xmeml', version='5')
    sequence = _add(root, 'sequence', id='retention-'+uuid.uuid4().hex)
    _add(sequence, 'name', name)
    _add(sequence, 'duration', position)
    _rate(sequence, rate)
    _timecode(sequence, rate)
    media = _add(sequence, 'media')
    video = _add(media, 'video')
    _video_format(_add(video, 'format'), dimensions, rate)
    video_track = _add(video, 'track')
    audio_media = _add(media, 'audio')
    _audio_format(_add(audio_media, 'format'), 48000)
    outputs = _add(audio_media, 'outputs')
    group = _add(outputs, 'group')
    _add(group, 'index', 1)
    _add(group, 'numchannels', 2)
    _add(group, 'downmix', 0)
    for channel in (1, 2):
        _add(_add(group, 'channel'), 'index', channel)
    channels = []
    for file_index, item in enumerate(tracks, 1):
        for channel in range(1, item['channels']+1):
            channels.append(dict(item, file_index=file_index, channel=channel,
                                 track_index=len(channels)+1, element=_add(audio_media, 'track')))
    defined_files = set()

    def file_ref(clip, file_id, file_path, audio_item=None):
        file = _add(clip, 'file', id=file_id)
        if file_id in defined_files:
            return
        defined_files.add(file_id)
        _add(file, 'name', file_path.name)
        _add(file, 'pathurl', file_path.as_uri())
        _rate(file, rate)
        _add(file, 'duration', source_frames)
        _timecode(file, rate)
        file_media = _add(file, 'media')
        if audio_item:
            file_audio = _add(file_media, 'audio')
            _audio_format(file_audio, audio_item['sample_rate'])
            _add(file_audio, 'channelcount', audio_item['channels'])
            _add(file_audio, 'layout', 'stereo' if audio_item['channels'] == 2 else 'mono')
        else:
            file_video = _add(file_media, 'video')
            _add(file_video, 'duration', source_frames)
            _video_format(file_video, dimensions, rate)

    def clip_item(track, clip_id, clip_name, row):
        clip = _add(track, 'clipitem', id=clip_id)
        _add(clip, 'name', clip_name)
        _add(clip, 'enabled', 'TRUE')
        _add(clip, 'duration', source_frames)
        _rate(clip, rate)
        for tag, key in [('start', 'output_start_frame'), ('end', 'output_end_frame'),
                         ('in', 'source_start_frame'), ('out', 'source_end_frame')]:
            _add(clip, tag, row[key])
        return clip

    for index, row in enumerate(frame_map, 1):
        linked = [('video', 1, f'v-{index}', None)]
        linked.extend(('audio', channel['track_index'], f"a-{channel['track_index']}-{index}",
                       channel['file_index'] if channel['channels'] == 2 else None) for channel in channels)
        clips = []
        clip = clip_item(video_track, f'v-{index}', source.name, row)
        file_ref(clip, 'source-video', source)
        source_track = _add(clip, 'sourcetrack')
        _add(source_track, 'mediatype', 'video')
        clips.append(clip)
        for channel in channels:
            clip = clip_item(channel['element'], f"a-{channel['track_index']}-{index}", channel['name'], row)
            file_ref(clip, f"source-audio-{channel['file_index']}", channel['path'], channel)
            source_track = _add(clip, 'sourcetrack')
            _add(source_track, 'mediatype', 'audio')
            _add(source_track, 'trackindex', channel['channel'])
            clips.append(clip)
        for clip in clips:
            for kind, track_index, reference, stereo_group in linked:
                link = _add(clip, 'link')
                _add(link, 'linkclipref', reference)
                _add(link, 'mediatype', kind)
                _add(link, 'trackindex', track_index)
                _add(link, 'clipindex', index)
                if stereo_group is not None:
                    _add(link, 'groupindex', stereo_group)
    for track in [video_track, *(channel['element'] for channel in channels)]:
        _add(track, 'enabled', 'TRUE')
        _add(track, 'locked', 'FALSE')
    for channel in channels:
        # Mono is sent to both outputs; stereo channels retain left/right order.
        for output in ([channel['channel']] if channel['channels'] == 2 else [1, 2]):
            _add(channel['element'], 'outputchannelindex', output)
    ET.indent(root, space='  ')
    payload = b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + ET.tostring(root, encoding='utf-8')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part-' + uuid.uuid4().hex)
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
