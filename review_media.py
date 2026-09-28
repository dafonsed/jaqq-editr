"""Preserve independent source audio streams and validate frame-based export."""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import json
import math

from app_paths import ROOT, add_dependencies
from review_core import save_json
add_dependencies()
import av


SIDECAR_VERSION = 2


def validate_video_timestamps(timestamps, fps, origin=0, time_base=0):
    """Reject VFR/gaps instead of mapping elapsed seconds to wrong source frames.

    Packet presentation times are sorted because B-frame packet decode order can
    differ from playback order. Container timestamp rounding is allowed, missing
    or duplicated presentation frames are not.
    """
    if not math.isfinite(fps) or fps <= 0 or not timestamps:
        raise ValueError('No reliable video timestamps are available for export.')
    tolerance = min(.25 / fps, max(2 * float(time_base), .02 / fps))
    frames = []
    for timestamp in timestamps:
        if timestamp is None or not math.isfinite(timestamp):
            raise ValueError('The recording has missing video timestamps; export stopped.')
        relative = timestamp - origin
        frame = round(relative * fps)
        if abs(relative - frame / fps) > tolerance:
            raise ValueError('Variable-frame-rate video needs a constant-frame-rate copy and fresh analysis before export.')
        frames.append(frame)
    frames.sort()
    if frames[0] != 0 or any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError('Variable-frame-rate or missing/duplicate video frames cannot be mapped safely; export stopped.')
    return dict(method='Every video packet presentation timestamp checked',
                constant_frame_rate=True,frames=len(frames),fps=fps,
                source_origin=origin,tolerance_seconds=tolerance)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _packet_fingerprint(path):
    digest, times, count = hashlib.sha256(), [], 0
    with av.open(str(path)) as container:
        if len(container.streams.audio) != 1:
            raise ValueError('Prepared audio does not contain exactly one source track.')
        stream = container.streams.audio[0]
        for packet in container.demux(stream):
            if packet.dts is None:
                continue
            digest.update(bytes(packet))
            times.append(float(packet.pts * packet.time_base) if packet.pts is not None else None)
            count += 1
    return digest.hexdigest(), times, count


def audio_sidecars(source, progress=lambda text: None):
    source = Path(source)
    stat = source.stat()
    identity = f'{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'
    key = hashlib.sha1(f'v{SIDECAR_VERSION}:{identity}'.encode()).hexdigest()[:14]
    folder = ROOT / 'analysis' / 'audio-for-resolve' / key
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'complete.json'
    if manifest.exists():
        try:
            cached = json.loads(manifest.read_text(encoding='utf-8'))
            items = cached['items']
            if (cached['version'] == SIDECAR_VERSION and cached['source_identity'] == identity
                    and cached['video_timing']['constant_frame_rate'] and items
                    and all(Path(x['path']).is_file() and
                            _file_hash(x['path']) == x['file_sha256'] for x in items)):
                return items
        except (OSError, ValueError, TypeError, KeyError):
            pass  # Stale, partial or corrupt caches are rebuilt, never trusted.
    outputs, items, partials = [], [], []
    source_digests, source_times = [], []
    try:
        with ExitStack() as stack:
            container = stack.enter_context(av.open(str(source)))
            if not container.streams.video or not container.streams.audio:
                raise ValueError('Choose a video recording with audio.')
            video = container.streams.video[0]
            fps = float(video.average_rate or video.base_rate or 0)
            origin = float((video.start_time or 0) * video.time_base)
            for i, stream in enumerate(container.streams.audio):
                audio_origin = float((stream.start_time or 0) * stream.time_base)
                tolerance = max(1 / (stream.codec_context.sample_rate or 48000), 1e-6)
                if abs(audio_origin - origin) > tolerance:
                    raise ValueError('Audio/video start offsets differ. Export needs an offset-aware audio conform; review playback still works.')
                if stream.codec_context.name != 'aac':
                    raise ValueError('Draft export currently supports separate AAC tracks. This audio codec requires a different export path.')
                channels = len(stream.codec_context.layout.channels)
                if channels not in (1, 2):
                    raise ValueError('Multichannel audio needs matching Resolve channel routing; export stopped to avoid dropping channels.')
                path = folder / f'audio-{i+1}.m4a'
                partial = folder / f'audio-{i+1}.partial.m4a'
                partials.append(partial)
                output = stack.enter_context(av.open(str(partial), mode='w'))
                track = output.add_stream_from_template(stream)
                outputs.append((output, track))
                source_digests.append(hashlib.sha256())
                source_times.append([])
                items.append(dict(path=str(path.resolve()),name=stream.metadata.get('name',f'Audio {i+1}'),
                    index=stream.index,source_origin=origin,sample_rate=stream.codec_context.sample_rate,
                    channels=channels,channel_layout=stream.codec_context.layout.name,
                    packet_copy_verified=False))
            mapping = {s.index: i for i, s in enumerate(container.streams.audio)}
            video_times, last = [], -1
            for packet in container.demux([video, *container.streams.audio]):
                if packet.stream.index == video.index:
                    if packet.size:
                        video_times.append(float(packet.pts * packet.time_base) if packet.pts is not None else None)
                    continue
                if packet.dts is None:
                    continue
                index = mapping[packet.stream.index]
                # Sidecar zero is the same origin used by source video/analysis.
                offset = round(origin / float(packet.time_base))
                packet.dts -= offset
                if packet.pts is not None:
                    packet.pts -= offset
                    second = int(packet.pts * packet.time_base)
                    if second // 30 > last:
                        last = second // 30
                        progress(f'Preparing and checking separate audio: {second//60}:{second%60:02}')
                source_digests[index].update(bytes(packet))
                source_times[index].append(float(packet.pts * packet.time_base) if packet.pts is not None else None)
                output, track = outputs[index]
                packet.stream = track
                output.mux(packet)
            timing = validate_video_timestamps(video_times, fps, origin, video.time_base)
        # Completed files are reopened: a successful mux call alone is not proof
        # that the container kept every packet and its presentation timestamp.
        for index, (item, partial) in enumerate(zip(items, partials)):
            digest, times, count = _packet_fingerprint(partial)
            tolerance = 1 / item['sample_rate']
            expected = source_times[index]
            if (not count or digest != source_digests[index].hexdigest() or len(times) != len(expected)
                    or any(a is None or b is None or abs(a-b) > tolerance for a,b in zip(times,expected))):
                raise ValueError('Prepared audio changed source packets or timing; export stopped.')
            item.update(packet_copy_verified=True,packet_count=count,file_sha256=_file_hash(partial),
                        video_timing=timing)
        current = source.stat()
        if (current.st_size,current.st_mtime_ns) != (stat.st_size,stat.st_mtime_ns):
            raise ValueError('The recording changed while exporting. Finish recording, then analyze it again.')
        for item, partial in zip(items, partials):
            partial.replace(item['path'])
        save_json(manifest,dict(version=SIDECAR_VERSION,source_identity=identity,
                                video_timing=timing,items=items))
        return items
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)
