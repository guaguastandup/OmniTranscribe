import tempfile
import unittest
from pathlib import Path

from omnitranscribe.cache import TranscriptionCache


SRT = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"


class TranscriptionCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.media = self.root / "audio.mp3"
        self.media.write_bytes(b"fake-audio-bytes")
        self.cache = TranscriptionCache(self.root / "cache")

    def tearDown(self):
        self.tmp.cleanup()

    def test_set_then_get_roundtrip(self):
        self.cache.set(str(self.media), "tiny", SRT, "auto")
        self.assertEqual(self.cache.get(str(self.media), "tiny", "auto"), SRT)

    def test_get_miss_returns_none(self):
        self.assertIsNone(self.cache.get(str(self.media), "tiny", "auto"))

    def test_key_separates_model_and_language(self):
        self.cache.set(str(self.media), "tiny", SRT, "auto")
        self.assertIsNone(self.cache.get(str(self.media), "base", "auto"))
        self.assertIsNone(self.cache.get(str(self.media), "tiny", "ja"))

    def test_stale_entry_is_evicted_on_get(self):
        self.cache.set(str(self.media), "tiny", SRT, "auto")
        cached_file = Path(next(iter(self.cache.index.values()))["cached_file"])
        cached_file.unlink()
        self.assertIsNone(self.cache.get(str(self.media), "tiny", "auto"))
        self.assertEqual(len(self.cache.index), 0)

    def test_clear_removes_files_and_index(self):
        self.cache.set(str(self.media), "tiny", SRT, "auto")
        cached_file = Path(next(iter(self.cache.index.values()))["cached_file"])
        self.assertTrue(cached_file.exists())
        self.cache.clear()
        self.assertFalse(cached_file.exists())
        self.assertEqual(len(self.cache.index), 0)

    def test_stats_and_listing(self):
        self.cache.set(str(self.media), "tiny", SRT, "auto")
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["total_entries"], 1)
        self.assertGreater(stats["total_size_bytes"], 0)
        listed = self.cache.list_cached()
        self.assertEqual(listed[0]["filename"], "audio.mp3")
        self.assertEqual(listed[0]["model"], "tiny")


if __name__ == "__main__":
    unittest.main()
