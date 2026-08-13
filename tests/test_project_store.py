import tempfile
import unittest
from pathlib import Path

from src.project_store import ProjectStore


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.media = self.root / "sample.mp3"
        self.media.write_bytes(b"not-real-audio")
        self.store = ProjectStore(self.root / "data")
        self.project = self.store.create_project(self.media, translation_model="google")

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoint_resume(self):
        units = [
            {"unit_index": 0, "start_ms": 0, "end_ms": 1000},
            {"unit_index": 1, "start_ms": 1000, "end_ms": 2000},
        ]
        self.store.ensure_task_units(self.project.id, "transcription", units)
        self.store.set_task_unit_status(self.project.id, "transcription", 0, "complete")
        self.store.set_task_unit_status(self.project.id, "transcription", 1, "running")
        self.store.reset_running_units(self.project.id, "transcription")
        pending = self.store.pending_task_units(self.project.id, "transcription")
        self.assertEqual([u["unit_index"] for u in pending], [1])
        checkpoint = self.store.get_checkpoint(self.project.id, "transcription")
        self.assertEqual(checkpoint["completed_units"], 1)
        self.assertEqual(checkpoint["total_units"], 2)

    def test_checkpoint_metadata_survives_status_updates(self):
        units = [{"unit_index": 0, "start_ms": 0, "end_ms": 1000}]
        self.store.ensure_task_units(self.project.id, "transcription", units)
        self.store.refresh_checkpoint(
            self.project.id,
            "transcription",
            status="paused",
            metadata={"whisper_model": "base", "source_language": "auto"},
        )
        self.store.set_task_unit_status(self.project.id, "transcription", 0, "running")
        checkpoint = self.store.get_checkpoint(self.project.id, "transcription")
        self.assertEqual(checkpoint["metadata"]["whisper_model"], "base")

    def test_project_owns_resumable_media_copy(self):
        managed = Path(self.project.source_path)
        self.assertTrue(managed.exists())
        self.assertTrue(str(managed).startswith(str(self.store.project_dir(self.project.id))))
        self.assertEqual(self.project.original_source_path, str(self.media.resolve()))

    def test_manual_edit_marks_translation_stale_or_manual(self):
        self.store.replace_segments_for_unit(
            self.project.id,
            0,
            [{"sequence": 1, "start_ms": 0, "end_ms": 1000, "source_text": "hello"}],
        )
        self.store.save_translation(self.project.id, {1: "你好"})
        self.store.update_segment(self.project.id, 1, source_text="hello there")
        row = self.store.list_segments(self.project.id)[0]
        self.assertEqual(row["translation_status"], "stale")
        self.store.update_segment(self.project.id, 1, translated_text="你好呀")
        row = self.store.list_segments(self.project.id)[0]
        self.assertEqual(row["translation_status"], "manual")


if __name__ == "__main__":
    unittest.main()
