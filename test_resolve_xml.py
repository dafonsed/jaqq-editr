"""Interchange frame/rate/channel/link tests; never connects to Resolve."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from review_core import build_frame_map
from resolve_xml import rate_values, write_timeline


class TimelineXMLTests(unittest.TestCase):
    def metadata(self, folder):
        return ((1920, 1080), [
            dict(path=folder/'microphone.m4a', name='Voice', channels=1, sample_rate=48000),
            dict(path=folder/'game stereo.m4a', name='Game', channels=2, sample_rate=48000)])

    def test_reordered_ranges_links_and_all_audio_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            source = folder/'take & café #1.mp4'
            mapped = build_frame_map([[7, 9], [1, 3], [7, 8]], 10, 30)
            with patch('resolve_xml._probe', return_value=self.metadata(folder)):
                output = write_timeline(folder/'cut.xml', source, 10, 30, mapped, [], name='Cut & title')
            root = ET.parse(output).getroot()
            self.assertEqual((root.tag, root.get('version')), ('xmeml', '5'))
            sequence = root.find('sequence')
            self.assertEqual(sequence.findtext('name'), 'Cut & title')
            self.assertEqual(sequence.findtext('duration'), '150')
            self.assertEqual(sequence.findtext('rate/timebase'), '30')
            self.assertEqual(sequence.findtext('rate/ntsc'), 'FALSE')
            tracks = sequence.findall('media/video/track') + sequence.findall('media/audio/track')
            self.assertEqual(len(tracks), 4)
            clip_ids = {clip.get('id') for track in tracks for clip in track.findall('clipitem')}
            self.assertEqual(len(clip_ids), 12)
            expected = [(210, 270, 0, 60), (30, 90, 60, 120), (210, 240, 120, 150)]
            for track in tracks:
                clips = track.findall('clipitem')
                self.assertEqual(len(clips), 3)
                self.assertEqual([tuple(int(clip.findtext(tag)) for tag in ('in', 'out', 'start', 'end'))
                                  for clip in clips], expected)
                for index, clip in enumerate(clips, 1):
                    references = [link.findtext('linkclipref') for link in clip.findall('link')]
                    self.assertEqual(len(set(references)), 4)
                    self.assertTrue(set(references) <= clip_ids)
                    self.assertTrue(all(link.findtext('clipindex') == str(index) for link in clip.findall('link')))
                    self.assertFalse(clip.findall('filter'))
            audio_tracks = sequence.findall('media/audio/track')
            self.assertEqual([track.findtext('clipitem/sourcetrack/trackindex') for track in audio_tracks], ['1', '1', '2'])
            self.assertEqual([[e.text for e in track.findall('outputchannelindex')] for track in audio_tracks],
                             [['1', '2'], ['1'], ['2']])
            file_definitions = [file for file in sequence.findall('.//file') if file.find('pathurl') is not None]
            self.assertEqual(len(file_definitions), 3)
            self.assertEqual(file_definitions[0].findtext('pathurl'), source.as_uri())
            self.assertIn('%20', file_definitions[0].findtext('pathurl'))
            self.assertIn('%23', file_definitions[0].findtext('pathurl'))
            self.assertEqual(file_definitions[2].findtext('media/audio/channelcount'), '2')
            self.assertEqual(sequence.findtext('timecode/frame'), '0')

    def test_fractional_rate_is_ntsc_without_rounding_frame_boundaries(self):
        fps = 30000/1001
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            mapped = build_frame_map([[.51, 2.31]], 10, fps)
            with patch('resolve_xml._probe', return_value=((1280, 720), [])):
                output = write_timeline(folder/'ntsc.xml', folder/'source.mp4', 10, fps, mapped, [])
            root = ET.parse(output).getroot()
            for rate in root.findall('.//rate'):
                self.assertEqual((rate.findtext('timebase'), rate.findtext('ntsc')), ('30', 'TRUE'))
            clip = root.find('sequence/media/video/track/clipitem')
            self.assertEqual(int(clip.findtext('in')), mapped[0]['source_start_frame'])
            self.assertEqual(int(clip.findtext('out')), mapped[0]['source_end_frame'])

    def test_invalid_map_or_rate_creates_no_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.xml'
            good = build_frame_map([[0, 2]], 3, 30)
            invalid = [dict(good[0], output_start_frame=1)]
            for fps, frame_map in [(27.5, good), (30, invalid), (30, []),
                                    (30, [dict(good[0], source_end_frame=200)]),
                                    (30, [dict(good[0], source_start_frame=.5)])]:
                with self.subTest(fps=fps, frame_map=frame_map), self.assertRaises(ValueError):
                    write_timeline(path, 'nonexistent.mp4', 3, fps, frame_map, [])
                self.assertFalse(path.exists())

    def test_common_rates(self):
        for fps, expected in [(24, (24, False)), (25, (25, False)), (60, (60, False)),
                              (24000/1001, (24, True)), (29.97, (30, True)),
                              (60000/1001, (60, True))]:
            self.assertEqual(rate_values(fps), expected)

    def test_interrupted_write_is_not_advertised_as_importable_xml(self):
        from automatic_cut import manual_import_details, ResolveUnavailable
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            output = folder/'Automatic cut.xml'
            mapped = build_frame_map([[1, 3]], 10, 30)
            def interrupted_write(path, payload):
                with path.open('wb') as stream:
                    stream.write(payload[:50])
                raise OSError('disk full')
            with patch('resolve_xml._probe', return_value=((1920, 1080), [])), \
                    patch.object(Path, 'write_bytes', interrupted_write):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    write_timeline(output, folder/'source.mp4', 10, 30, mapped, [])
            self.assertFalse(output.exists())
            self.assertFalse(list(folder.glob('*.part-*')))
            details = manual_import_details(folder, ResolveUnavailable('API unavailable'))
            self.assertIsNone(details['import_xml'])
            self.assertNotIn('select Automatic cut.xml', details['import_instructions'])

    def test_interrupted_replacement_preserves_previous_valid_xml(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            output = folder/'Automatic cut.xml'
            mapped = build_frame_map([[1, 3]], 10, 30)
            with patch('resolve_xml._probe', return_value=((1920, 1080), [])):
                write_timeline(output, folder/'source.mp4', 10, 30, mapped, [], name='Prior export')
            previous = output.read_bytes()
            def interrupted_write(path, payload):
                with path.open('wb') as stream:
                    stream.write(payload[:50])
                raise OSError('disk full')
            with patch('resolve_xml._probe', return_value=((1920, 1080), [])), \
                    patch.object(Path, 'write_bytes', interrupted_write):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    write_timeline(output, folder/'source.mp4', 10, 30, mapped, [], name='New export')
            self.assertEqual(output.read_bytes(), previous)
            self.assertEqual(ET.parse(output).findtext('sequence/name'), 'Prior export')
            self.assertFalse(list(folder.glob('*.part-*')))


if __name__ == '__main__':
    unittest.main()
