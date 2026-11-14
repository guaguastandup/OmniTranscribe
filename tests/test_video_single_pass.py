"""Regression tests for the single-pass video generation command.

ffmpeg/ffprobe are stubbed out; these tests only assert the shape of the
commands we build, so they run anywhere without media tooling.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import omnitranscribe.final_video_generator as fvg


def _make_inputs(root: Path):
    audio = root / "a.mp3"
    srt = root / "a.srt"
    image = root / "bg.png"
    for path, payload in ((audio, b"x"), (srt, b"1\n"), (image, b"x")):
        path.write_bytes(payload)
    return str(audio), str(srt), str(image)


class SinglePassCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio, self.srt, self.image = _make_inputs(Path(self.tmp.name))
        # silence font discovery output noise
        self.patcher_print = mock.patch("builtins.print")
        self.patcher_print.start()

    def tearDown(self):
        self.patcher_print.stop()
        self.tmp.cleanup()

    def _run_with_stubbed_ffmpeg(self, **kwargs):
        calls = []

        def fake_run(cmd, **_):
            calls.append(cmd)
            if cmd[0] == "ffprobe":
                return SimpleNamespace(returncode=0, stdout="12.5\n", stderr="")
            if "-h" in cmd:
                # ffmpeg -h filter=subtitles probe: report available
                return SimpleNamespace(returncode=0, stdout="Filter subtitles\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        generator = fvg.FinalVideoGenerator()
        with mock.patch.object(fvg.subprocess, "run", side_effect=fake_run):
            generator.create_video_with_subtitles(
                self.audio, self.srt, self.image,
                str(Path(self.tmp.name) / "out.mp4"), **kwargs
            )
        # drop the availability probe; keep the real encode/ffprobe calls
        return [c for c in calls if not ("-h" in c and "filter=subtitles" in c)]

    def test_single_ffmpeg_pass_with_combined_filter(self):
        calls = self._run_with_stubbed_ffmpeg()
        ffmpeg_calls = [c for c in calls if c[0] == "ffmpeg"]
        ffprobe_calls = [c for c in calls if c[0] == "ffprobe"]
        self.assertEqual(len(ffprobe_calls), 1, "duration probe exactly once")
        self.assertEqual(len(ffmpeg_calls), 1, "one encode pass, not two")
        cmd = ffmpeg_calls[0]
        vf = cmd[cmd.index("-vf") + 1]
        for stage in ("scale=", "pad=", "subtitles="):
            self.assertIn(stage, vf, f"filter chain missing {stage}")
        # audio is encoded in the same pass (no second -c:a copy pass)
        self.assertIn("aac", cmd)

    def test_max_duration_trims_audio_then_encodes_once(self):
        calls = self._run_with_stubbed_ffmpeg(max_duration=5.0)
        ffmpeg_calls = [c for c in calls if c[0] == "ffmpeg"]
        self.assertEqual(len(ffmpeg_calls), 2)  # audio clip + single encode
        clip_cmd, encode_cmd = ffmpeg_calls
        self.assertIn("-t", clip_cmd)
        self.assertIn("subtitles=", encode_cmd[encode_cmd.index("-vf") + 1])

    def test_font_size_and_position_reach_force_style(self):
        calls = self._run_with_stubbed_ffmpeg(font_size=64, subtitle_position="top")
        encode_cmd = next(c for c in calls if c[0] == "ffmpeg")
        vf = encode_cmd[encode_cmd.index("-vf") + 1]
        self.assertIn("Fontsize=64", vf)
        self.assertIn("Alignment=8", vf)


if __name__ == "__main__":
    unittest.main()
