import unittest

from src.subtitle_utils import SubtitleRow, ms_to_srt, parse_srt, render_srt, srt_to_ms


class SrtHelpersTests(unittest.TestCase):
    def test_time_roundtrip(self):
        value = 3_726_543
        self.assertEqual(srt_to_ms(ms_to_srt(value)), value)

    def test_srt_roundtrip(self):
        rows = [
            SubtitleRow(1, 0, 1500, "Hello"),
            SubtitleRow(2, 1500, 3200, "Line one\nLine two"),
        ]
        parsed = parse_srt(render_srt(rows))
        self.assertEqual(
            [(r.start_ms, r.end_ms, r.text) for r in parsed],
            [(0, 1500, "Hello"), (1500, 3200, "Line one\nLine two")],
        )


if __name__ == "__main__":
    unittest.main()
