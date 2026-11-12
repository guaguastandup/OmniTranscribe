import json
import tempfile
import unittest
from pathlib import Path

from src.config import Config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_when_file_missing(self):
        cfg = Config(self.path)
        self.assertEqual(cfg.get("transcription.model"), "base")
        self.assertEqual(cfg.get("translation.chunk_size"), 20)
        self.assertIsNone(cfg.get("does.not.exist"))
        self.assertEqual(cfg.get("does.not.exist", "fallback"), "fallback")

    def test_set_and_persist(self):
        cfg = Config(self.path)
        cfg.set("transcription.model", "large")
        cfg.set("batch.max_workers", 4)
        self.assertEqual(cfg.get("transcription.model"), "large")
        cfg.save()

        reloaded = Config(self.path)
        self.assertEqual(reloaded.get("transcription.model"), "large")
        self.assertEqual(reloaded.get("batch.max_workers"), 4)

    def test_saved_file_merges_with_defaults(self):
        self.path.write_text(json.dumps({"translation": {"chunk_size": 5}}), encoding="utf-8")
        cfg = Config(self.path)
        self.assertEqual(cfg.get("translation.chunk_size"), 5)
        # untouched sections keep defaults
        self.assertEqual(cfg.get("transcription.model"), "base")

    def test_set_does_not_mutate_class_defaults(self):
        cfg = Config(self.path)
        cfg.set("transcription.model", "large")
        self.assertEqual(Config.DEFAULT_CONFIG["transcription"]["model"], "base")
        fresh = Config(Path(self.tmp.name) / "other.json")
        self.assertEqual(fresh.get("transcription.model"), "base")

    def test_reset_to_defaults(self):
        cfg = Config(self.path)
        cfg.set("transcription.model", "large")
        cfg.reset_to_defaults()
        self.assertEqual(cfg.get("transcription.model"), "base")

    def test_corrupt_file_falls_back_to_defaults(self):
        self.path.write_text("{not valid json", encoding="utf-8")
        cfg = Config(self.path)
        self.assertEqual(cfg.get("transcription.model"), "base")


if __name__ == "__main__":
    unittest.main()
