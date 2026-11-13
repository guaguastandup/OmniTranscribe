import tempfile
import unittest
from pathlib import Path

from omnitranscribe.converter import SubtitleConverter


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,500
Hello world

2
00:00:04,000 --> 00:00:06,250
Second line
continues here
"""

MALFORMED_SRT = """1
00:00:01,000 --> 00:00:03,500
Good block

not-a-number
00:00:04,000 --> 00:00:06,000
Bad index is skipped
"""


class SubtitleConverterTests(unittest.TestCase):
    def setUp(self):
        self.converter = SubtitleConverter()

    def test_parse_srt(self):
        rows = self.converter.parse_srt(SAMPLE_SRT)
        self.assertEqual(len(rows), 2)
        index, start, end, text = rows[0]
        self.assertEqual(index, 1)
        self.assertAlmostEqual(start, 1.0)
        self.assertAlmostEqual(end, 3.5)
        self.assertEqual(text, "Hello world")
        # Multi-line text is preserved
        self.assertEqual(rows[1][3], "Second line\ncontinues here")

    def test_parse_srt_skips_malformed_blocks(self):
        rows = self.converter.parse_srt(MALFORMED_SRT)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][3], "Good block")

    def test_srt_to_vtt(self):
        vtt = self.converter.srt_to_vtt(SAMPLE_SRT)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:01.000 --> 00:00:03.500", vtt)
        self.assertNotIn(",", vtt.split("-->")[0])

    def test_srt_to_lrc(self):
        lrc = self.converter.srt_to_lrc(SAMPLE_SRT)
        lines = [line for line in lrc.split("\n") if line]
        self.assertIn("[00:01.00] Hello world", lines)
        # Multi-line text collapses to one LRC line
        self.assertIn("[00:04.00] Second line continues here", lines)

    def test_vtt_to_srt_roundtrip(self):
        vtt = self.converter.srt_to_vtt(SAMPLE_SRT)
        back = self.converter.vtt_to_srt(vtt)
        rows = self.converter.parse_srt(back)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][3], "Hello world")
        self.assertAlmostEqual(rows[1][2], 6.25)

    def test_convert_file_srt_to_lrc(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sample.srt"
            src.write_text(SAMPLE_SRT, encoding="utf-8")
            out = self.converter.convert_file(str(src), "lrc")
            self.assertEqual(Path(out).suffix, ".lrc")
            self.assertIn("[00:01.00]", Path(out).read_text(encoding="utf-8"))

    def test_convert_file_rejects_unknown_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sample.srt"
            src.write_text(SAMPLE_SRT, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.converter.convert_file(str(src), "docx")


if __name__ == "__main__":
    unittest.main()
