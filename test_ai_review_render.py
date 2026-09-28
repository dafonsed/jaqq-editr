"""Real synthetic encode/decode review regressions; no API or Resolve access."""
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app_paths import add_dependencies
add_dependencies()
import av
import numpy as np
from review_core import build_frame_map
from review_render import render_preview, audit_render, map_annotations, compare_render_words


def make_source(folder):
    source = folder/'speech-timing-fixture.mov'
    rate = 48000
    audio_data = []
    for frequency in (220, 330):
        signal=(np.sin(np.arange(rate*3)*2*np.pi*frequency/rate)*6000).astype('<i2')
        signal[rate:round(1.8*rate)] = 0
        audio_data.append(signal)
    with av.open(str(source), 'w') as output:
        video=output.add_stream('png',rate=30)
        video.width=96;video.height=64;video.pix_fmt='rgb24'
        audio=[]
        for _ in range(2):
            stream=output.add_stream('pcm_s16le',rate=rate);stream.layout='mono';audio.append(stream)
        for index in range(90):
            pixels=np.full((64,96,3),[index*2,(index*11)%256,(index*23)%256],dtype=np.uint8)
            picture=av.VideoFrame.from_ndarray(pixels,format='rgb24')
            picture.pts=index;picture.time_base=Fraction(1,30)
            for packet in video.encode(picture):output.mux(packet)
            for stream,data in zip(audio,audio_data):
                frame=av.AudioFrame.from_ndarray(data[None,index*1600:(index+1)*1600],format='s16',layout='mono')
                frame.sample_rate=rate;frame.pts=index*1600;frame.time_base=Fraction(1,rate)
                for packet in stream.encode(frame):output.mux(packet)
        for stream in [video,*audio]:
            for packet in stream.encode(None):output.mux(packet)
    return source, audio_data


class RenderTests(unittest.TestCase):
    def test_exact_frames_and_all_audio_tracks_survive_reordered_cuts(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(os.environ.get('AI_REVIEW_FIXTURE_OUTPUT',temporary));folder.mkdir(parents=True,exist_ok=True)
            source, audio=make_source(folder)
            destination=folder/'review-output.mkv'
            if destination.exists():destination.unlink()
            kept=[[2,2.5],[.2,1.2],[2,2.5]]
            result=render_preview(source,kept,3,30,destination,lossless=True,max_width=None,
                annotations=[dict(start=2.1,end=2.3,text='same source caption')])
            self.assertEqual(result['output_frames'],60)
            self.assertEqual(len(result['annotations']),2)
            captions=Path(result['caption_sidecar']).read_text(encoding='utf-8')
            self.assertIn('00:00:00,100 --> 00:00:00,300',captions)
            self.assertIn('00:00:01,600 --> 00:00:01,800',captions)
            self.assertEqual(result['source_audio_gaps'],[])
            indices=[i for row in result['frame_map'] for i in range(row['source_start_frame'],row['source_end_frame'])]
            with av.open(str(destination)) as container:
                decoded=list(container.decode(video=0))
            self.assertEqual(len(decoded),60)
            for index,picture in zip(indices,decoded):
                self.assertTrue(np.all(picture.to_ndarray(format='rgb24')==[index*2,(index*11)%256,(index*23)%256]))
            for track,data in enumerate(audio):
                expected=np.concatenate([data[row['source_start_frame']*1600:row['source_end_frame']*1600] for row in result['frame_map']])
                with av.open(str(destination)) as container:
                    actual=np.concatenate([frame.to_ndarray().reshape(-1) for frame in container.decode(audio=track)])
                self.assertEqual(len(actual),96000)
                self.assertLessEqual(int(np.max(np.abs(actual.astype(int)-expected.astype(int)))),1)
            qa=audit_render(destination,result['frame_map'],30)
            self.assertEqual(qa['geometry'],'PASS')
            self.assertTrue(any(issue['kind']=='visual_jump' for issue in qa['issues']))
            self.assertEqual(qa['listening'],'NOT VERIFIED')
            self.assertEqual(qa['word_check']['status'],'NOT VERIFIED')
            with self.assertRaises(ValueError):render_preview(source,kept,3,30,destination)

    def test_lossy_review_decode_and_independent_qa_find_quiet_spans(self):
        with tempfile.TemporaryDirectory() as temp:
            source,_=make_source(Path(temp))
            output=Path(temp)/'review.mp4'
            result=render_preview(source,[[0,3]],3,30,output)
            qa=audit_render(output,result['frame_map'],30,long_pause=.5)
            self.assertEqual(qa['geometry'],'PASS')
            self.assertTrue(any(item['kind']=='long_quiet_span' for item in qa['issues']))
            self.assertEqual(len(qa['audio']),2)

    def test_mapping_and_text_discrepancies_do_not_claim_listening(self):
        mapped=build_frame_map([[0,1],[2,3]],3,30)
        annotations=map_annotations([dict(start=.5,end=2.5,text='partially removed')],mapped)
        self.assertEqual([(r['start'],r['end']) for r in annotations],[(.5,1),(1,1.5)])
        self.assertTrue(all(r['partial'] for r in annotations))
        report=compare_render_words([dict(text='do not enable it')],[dict(text='do enable it')])
        self.assertEqual(report['status'],'REVIEW')
        self.assertEqual(report['differences'][0]['expected'],'not')


class UiWorkerTests(unittest.TestCase):
    def test_cloud_worker_does_not_require_or_send_to_resolve(self):
        from automatic_review import Worker
        worker=Worker('fixture.mp4',0,'balanced','openai')
        with patch('ai_pipeline.plan_ai',return_value={'ai_plan':{'decisions':[]}}) as planning, \
             patch('automatic_review.check_resolve') as resolve,patch('automatic_review.send') as sending:
            worker.run()
            planning.assert_called_once()
            resolve.assert_not_called();sending.assert_not_called()


if __name__=='__main__':unittest.main()
