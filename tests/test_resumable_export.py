"""Tests for resumable pipeline translation waves and subtitle exports.

Only touches the stdlib-backed modules (project_store, subtitle_utils,
converter); the translator is replaced with a fake via monkeypatching so
no API keys or SDKs are needed.
"""
import tempfile
import unittest
from pathlib import Path

import omnitranscribe.translator as translator_module
from omnitranscribe.project_store import ProjectStore
from omnitranscribe.resumable_pipeline import ResumablePipeline


class FakeTranslator:
    """Passthrough translator: returns the source SRT unchanged."""

    call_count = 0

    def __init__(self, model=None, **kwargs):
        pass

    def translate_srt(self, srt_content, **kwargs):
        type(self).call_count += 1
        return srt_content

    def save_translated_srt(self, content, output_path):
        Path(output_path).write_text(content, encoding="utf-8")


class ResumablePipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.media = self.root / "clip.mp3"
        self.media.write_bytes(b"fake-audio")
        self.store = ProjectStore(self.root / "data")
        self.project = self.store.create_project(
            self.media, translation_model="google"
        )
        self.pipeline = ResumablePipeline(self.store)

        # five subtitle segments so batch_size=2 yields three batches
        segments = [
            {"sequence": (i + 1) * 1000, "start_ms": i * 1000,
             "end_ms": i * 1000 + 900, "source_text": f"hello {i}"}
            for i in range(5)
        ]
        self.store.replace_segments_for_unit(self.project.id, 0, segments)
        self.store.renumber_segments(self.project.id)

        self._original = translator_module.UniversalTranslator
        translator_module.UniversalTranslator = FakeTranslator
        FakeTranslator.call_count = 0

    def tearDown(self):
        translator_module.UniversalTranslator = self._original
        self.tmp.cleanup()

    def test_translate_runs_batches_in_waves(self):
        checkpoint = self.pipeline.translate(
            self.project.id, batch_size=2, concurrency=2
        )
        self.assertEqual(checkpoint["status"], "complete")
        self.assertEqual(FakeTranslator.call_count, 3)  # 5 rows / batch 2
        rows = self.store.list_segments(self.project.id)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r["translation_status"] == "complete" for r in rows))
        self.assertEqual([r["translated_text"] for r in rows],
                         [f"hello {i}" for i in range(5)])

    def test_translate_failure_marks_unit_and_stage(self):
        class BrokenTranslator(FakeTranslator):
            def translate_srt(self, srt_content, **kwargs):
                raise RuntimeError("api down")

        translator_module.UniversalTranslator = BrokenTranslator
        with self.assertRaises(RuntimeError):
            self.pipeline.translate(self.project.id, batch_size=2, concurrency=1)
        checkpoint = self.store.get_checkpoint(self.project.id, "translation")
        self.assertEqual(checkpoint["status"], "paused")
        units = self.store.task_units(self.project.id, "translation")
        self.assertEqual(units[0]["status"], "failed")
        self.assertIn("api down", units[0]["last_error"])

    def test_export_vtt(self):
        self.pipeline.translate(self.project.id, batch_size=10)
        result = self.pipeline.export_media(
            self.project.id, subtitle_mode="source", output_kind="vtt"
        )
        self.assertEqual(result.suffix, ".vtt")
        content = result.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("WEBVTT"))
        self.assertIn("hello 0", content)
        self.assertIn("-->", content)

    def test_export_bilingual_requires_translations(self):
        # reset segments to untranslated to prove the export guard works
        self.store.replace_segments_for_unit(self.project.id, 0, [
            {"sequence": 1, "start_ms": 0, "end_ms": 900, "source_text": "hello 0"},
        ])
        self.store.renumber_segments(self.project.id)
        with self.assertRaises(RuntimeError):
            self.pipeline.export_media(self.project.id, subtitle_mode="bilingual",
                                       output_kind="vtt")


if __name__ == "__main__":
    unittest.main()
