"""Offline export regressions. Never connects to Resolve or touches its projects."""
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from app_paths import add_dependencies
add_dependencies()
from review_core import build_frame_map, export_review, map_source_span, save_json


class FrameMapTests(unittest.TestCase):
    def test_half_frame_uses_one_rounding_convention(self):
        with tempfile.TemporaryDirectory() as folder:
            export_review(folder, 'example.mp4', 3, 2, [], {}, keep_ranges=[[.25, 2.25]])
            data = json.loads((Path(folder)/'review.json').read_text())
            item = data['frame_map'][0]
            self.assertEqual((item['source_start_frame'],item['source_end_frame']), (0,4))
            lua = (Path(folder)/'create_resolve_draft.lua').read_text()
            self.assertIn('local frameRanges = {{0, 4, 0, 4}}', lua)
            self.assertNotIn('math.floor(r[1]', lua)
            self.assertIn('item:GetStart()==origin+r[3]', lua)
            self.assertIn('item:GetDuration()==r[4]-r[3]', lua)
            self.assertFalse(data['verification']['rendered_media_verified'])

    def test_repeated_ordered_ranges_and_partial_span_mapping(self):
        mapped = build_frame_map([[8,10],[1,4],[8,10]], 12, 60)
        self.assertEqual([(r['output_start_frame'],r['output_end_frame']) for r in mapped],
                         [(0,120),(120,300),(300,420)])
        words = map_source_span(mapped, 8.5, 9)
        self.assertEqual([r['output_start'] for r in words], [.5,5.5])
        self.assertEqual(map_source_span(mapped, 5, 6), [])
        self.assertTrue(map_source_span(mapped, 3, 5)[0]['partial'])

    def test_invalid_and_empty_plans_fail_before_artifacts(self):
        for duration,fps,ranges in [(0,60,[[0,1]]),(5,float('nan'),[[0,1]]),
                (5,60,[]),(5,60,[[1,1.001]]),(5,60,[[float('nan'),2]])]:
            with self.subTest(duration=duration,fps=fps,ranges=ranges):
                with tempfile.TemporaryDirectory() as temp:
                    folder=Path(temp)/'export'
                    with self.assertRaises(ValueError):
                        export_review(folder,'source.mp4',duration,fps,[],{},keep_ranges=ranges)
                    self.assertFalse(folder.exists())

    def test_first_run_json_creates_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'analysis'/'nested'/'data.json'
            save_json(path, {'fresh':True})
            self.assertEqual(json.loads(path.read_text()), {'fresh':True})


class MediaTests(unittest.TestCase):
    def test_reordered_packet_pts_allowed_but_vfr_rejected(self):
        from review_media import validate_video_timestamps
        valid = validate_video_timestamps([0,2/30,1/30,3/30],30)
        self.assertEqual(valid['frames'],4)
        for pts in ([0,1/30,3/30],[0,1/30,1/30],[0,.05,.1],[None,1/30]):
            with self.subTest(pts=pts), self.assertRaises(ValueError):
                validate_video_timestamps(pts,30)
        shifted=validate_video_timestamps([10,10+1/30,10+2/30],30,origin=10)
        self.assertEqual(shifted['frames'],3)

    def test_extraction_uses_nonzero_video_origin(self):
        import av
        import numpy as np
        from analyze_speech import extract
        from automatic_cut import probe
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'offset-source.mov'
            with av.open(str(source),'w') as output:
                video=output.add_stream('png',rate=30)
                video.width=32;video.height=32;video.pix_fmt='rgb24'
                audio=output.add_stream('pcm_s16le',rate=48000);audio.layout='mono'
                samples=np.arange(48000)/48000
                values=(6000*np.sin(2*np.pi*440*samples)).astype('<i2')
                for index in range(30):
                    frame=av.VideoFrame.from_ndarray(np.full((32,32,3),index,dtype=np.uint8),format='rgb24')
                    frame.pts=300+index;frame.time_base=Fraction(1,30)
                    for packet in video.encode(frame):output.mux(packet)
                    frame=av.AudioFrame.from_ndarray(values[None,index*1600:(index+1)*1600],format='s16',layout='mono')
                    frame.sample_rate=48000;frame.pts=480000+index*1600;frame.time_base=Fraction(1,48000)
                    for packet in audio.encode(frame):output.mux(packet)
                for stream in (video,audio):
                    for packet in stream.encode(None):output.mux(packet)
            with av.open(str(source)) as container:
                video=container.streams.video[0]
                self.assertEqual(float(video.start_time*video.time_base),10)
            self.assertAlmostEqual(probe(source)[0],1)
            dest=Path(temp)/'new'/'analysis.wav'
            extract(source,0,.1,.9,dest)
            with wave.open(str(dest),'rb') as handle:
                actual=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2')
            self.assertEqual(len(actual),12800)
            # Peak correlation at zero lag confirms video-relative source time.
            expected=values[4800:43200:3].astype(float)
            self.assertGreater(np.corrcoef(actual[20:-20],expected[20:-20])[0,1],.9999)

    def test_synthetic_encode_decode_all_tracks_follow_frame_map(self):
        """Validate actual media through the map; this is not a Resolve render."""
        import av
        import numpy as np
        import review_media
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(os.environ.get('AUTO_EDITOR_FIXTURE_OUTPUT',temp))
            folder.mkdir(parents=True,exist_ok=True)
            source=folder/'synthetic-source.mp4'
            with av.open(str(source),'w') as output:
                video=output.add_stream('libx264rgb',rate=30)
                video.width=64;video.height=64;video.pix_fmt='rgb24'
                video.options={'crf':'0','preset':'ultrafast'}
                audio=[]
                for _ in range(2):
                    stream=output.add_stream('aac',rate=48000)
                    stream.layout='mono';audio.append(stream)
                for index in range(90):
                    pixels=np.zeros((64,64,3),dtype=np.uint8)
                    pixels[:]=[index*2,(index*11)%256,(index*23)%256]
                    frame=av.VideoFrame.from_ndarray(pixels,format='rgb24')
                    frame.pts=index;frame.time_base=Fraction(1,30)
                    for packet in video.encode(frame):output.mux(packet)
                    for track,stream in enumerate(audio):
                        samples=np.arange(index*1600,(index+1)*1600)/48000
                        data=(.2*np.sin(2*np.pi*(440+track*220)*samples)).astype(np.float32)[None,:]
                        frame=av.AudioFrame.from_ndarray(data,format='fltp',layout='mono')
                        frame.sample_rate=48000;frame.pts=index*1600;frame.time_base=Fraction(1,48000)
                        for packet in stream.encode(frame):output.mux(packet)
                for stream in [video,*audio]:
                    for packet in stream.encode(None):output.mux(packet)
            with patch.object(review_media,'ROOT',folder):
                sidecars=review_media.audio_sidecars(source)
                self.assertEqual(len(sidecars),2)
                self.assertTrue(all(item['packet_copy_verified'] for item in sidecars))
                # An existing truncated sidecar used to satisfy the cache check.
                Path(sidecars[0]['path']).write_bytes(b'corrupt')
                repaired=review_media.audio_sidecars(source)
                self.assertTrue(repaired[0]['packet_copy_verified'])
            with av.open(str(source)) as container:
                source_frames=[frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
            source_audio=[]
            for track,item in enumerate(sidecars):
                with av.open(str(source)) as container:
                    original=np.concatenate([frame.to_ndarray() for frame in container.decode(audio=track)],axis=1)
                with av.open(item['path']) as container:
                    copied=np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)],axis=1)
                self.assertTrue(np.array_equal(original,copied))
                source_audio.append(original)
            from analyze_speech import extract
            analysis_wav=folder/'new-analysis-directory'/'microphone.wav'
            extract(source,0,.125,2.875,analysis_wav)
            with wave.open(str(analysis_wav),'rb') as handle:
                self.assertEqual(handle.getparams()[:3],(1,2,16000))
                self.assertEqual(handle.getnframes(),44000)
                extracted=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2')
            with av.open(str(source)) as container:
                resampler=av.AudioResampler(format='s16',layout='mono',rate=16000)
                reference=[]
                for frame in container.decode(audio=0):
                    reference.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
                reference.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
            # Independent resampler instances can differ by one int16 rounding
            # unit; a sample shift is orders of magnitude larger on this tone.
            self.assertLessEqual(np.max(np.abs(extracted.astype(float)-np.concatenate(reference)[2000:46000])),1)
            # Leave a valid WAV header advertising the original length, but
            # truncate sample bytes: metadata alone must not validate this cache.
            original_wav=analysis_wav.read_bytes()
            analysis_wav.write_bytes(original_wav[:100])
            extract(source,0,.125,2.875,analysis_wav)
            self.assertEqual(len(analysis_wav.read_bytes()),len(original_wav))
            # Teaser, earlier content, repeated teaser: intentionally non-monotonic.
            ranges=[[2,2.5],[.25,1.25],[2,2.5]]
            plan=build_frame_map(ranges,3,30)
            export_review(folder/'plan',source,3,30,[],{},keep_ranges=ranges,audio=sidecars)
            expected_frames=[source_frames[index] for row in plan
                for index in range(row['source_start_frame'],row['source_end_frame'])]
            expected_audio=[np.concatenate([data[:,row['source_start_frame']*1600:row['source_end_frame']*1600]
                for row in plan],axis=1) for data in source_audio]
            render=folder/'synthetic-mapped-output.mkv'
            with av.open(str(render),'w') as output:
                video=output.add_stream('ffv1',rate=30)
                video.width=64;video.height=64;video.pix_fmt='bgr0'
                audio=[]
                for _ in range(2):
                    stream=output.add_stream('pcm_f32le',rate=48000)
                    stream.layout='mono';audio.append(stream)
                for index,pixels in enumerate(expected_frames):
                    frame=av.VideoFrame.from_ndarray(pixels,format='rgb24')
                    frame.pts=index;frame.time_base=Fraction(1,30)
                    for packet in video.encode(frame):output.mux(packet)
                    for track,stream in enumerate(audio):
                        data=expected_audio[track][:,index*1600:(index+1)*1600]
                        frame=av.AudioFrame.from_ndarray(data,format='flt',layout='mono')
                        frame.sample_rate=48000;frame.pts=index*1600;frame.time_base=Fraction(1,48000)
                        for packet in stream.encode(frame):output.mux(packet)
                for stream in [video,*audio]:
                    for packet in stream.encode(None):output.mux(packet)
            with av.open(str(render)) as container:
                decoded=list(container.decode(video=0))
            self.assertEqual(len(decoded),60)
            self.assertTrue(all(np.array_equal(frame.to_ndarray(format='rgb24'),expected)
                                for frame,expected in zip(decoded,expected_frames)))
            self.assertTrue(all(abs(float(frame.time)-index/30)<.0011 for index,frame in enumerate(decoded)))
            for track in range(2):
                with av.open(str(render)) as container:
                    actual=np.concatenate([frame.to_ndarray() for frame in container.decode(audio=track)],axis=1)
                self.assertEqual(actual.shape,(1,96000))
                self.assertTrue(np.array_equal(actual,expected_audio[track]))
            save_json(folder/'synthetic-verification.json',dict(
                fixture='Synthetic changing frame colors plus two separate sine-wave tracks; no human speech',
                source=str(source),output=str(render),frame_map=plan,
                rendered_frames=60,rendered_audio_samples_per_track=96000,
                audio_tracks=2,decoded_picture_and_audio_match=True,
                sidecar_packet_and_timestamp_copy_verified=True,
                analysis_extraction_sample_alignment_and_flush_verified=True,
                limitations='Test renderer exercises the canonical map; Resolve import/render and audible speech joins remain NOT VERIFIED.'))


if __name__ == '__main__':
    unittest.main()
