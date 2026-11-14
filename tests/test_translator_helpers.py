"""Unit tests for translator pure helpers (no API keys or SDK required)."""
import unittest

from omnitranscribe.translator import UniversalTranslator, get_prompt_path


SRT = (
    "1\n00:00:01,000 --> 00:00:02,000\nfirst line\n\n"
    "2\n00:00:02,500 --> 00:00:03,500\nsecond line\n\n"
    "3\n00:00:04,000 --> 00:00:05,000\nthird line\n"
)


def make_translator() -> UniversalTranslator:
    """Build an instance without running __init__ (no API keys needed).

    The helpers under test are pure; only the log line in translate_srt
    touches self.model / self.service, so set harmless placeholders.
    """
    translator = UniversalTranslator.__new__(UniversalTranslator)
    translator.model = "test-model"
    translator.service = "test"
    return translator


class SrtHelperTests(unittest.TestCase):
    def setUp(self):
        self.translator = make_translator()

    def test_parse_srt_content(self):
        blocks = self.translator._parse_srt_content(SRT)
        self.assertEqual(len(blocks), 3)
        self.assertIn("first line", blocks[0])
        self.assertIn("-->", blocks[1])

    def test_split_into_chunks(self):
        chunks = self.translator._split_srt_into_chunks(SRT, 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(self.translator._parse_srt_content(chunks[0])), 2)
        self.assertEqual(len(self.translator._parse_srt_content(chunks[1])), 1)

    def test_combine_chunks_rejoins_with_blank_lines(self):
        combined = self.translator._combine_translated_chunks(["a", "", "b"])
        self.assertEqual(combined, "a\n\nb")


class ParallelTranslationTests(unittest.TestCase):
    def setUp(self):
        self.translator = make_translator()
        self.translator.load_prompt = lambda *a, **k: "system prompt"

    def _srt(self, entries: int) -> str:
        return "\n\n".join(
            f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},999\nline {i}"
            for i in range(1, entries + 1)
        )

    def test_parallel_chunks_preserve_order(self):
        import random
        import time

        seen_offsets = []

        def fake_chunk(chunk, system_prompt, max_retries, start_index, target_language):
            seen_offsets.append(start_index)
            time.sleep(random.random() * 0.02)  # force out-of-order completion
            return chunk

        self.translator._translate_chunk = fake_chunk
        result = self.translator.translate_srt(
            self._srt(30), prompt_path="ignored", chunk_size=5,
            max_concurrent_chunks=4,
        )
        rows = self.translator._parse_srt_content(result)
        self.assertEqual(len(rows), 30)
        # texts must come back in original order despite random completion
        self.assertIn("line 1", result.split("\n\n")[0])
        self.assertTrue(result.index("line 1") < result.index("line 15") < result.index("line 30"))
        # every chunk got its correct global offset
        self.assertEqual(sorted(seen_offsets), [0, 5, 10, 15, 20, 25])

    def test_serial_mode_still_works(self):
        self.translator._translate_chunk = (
            lambda chunk, sp, mr, si, tl: chunk
        )
        result = self.translator.translate_srt(
            self._srt(6), prompt_path="ignored", chunk_size=2,
            max_concurrent_chunks=1,
        )
        self.assertEqual(len(self.translator._parse_srt_content(result)), 6)


class CleanSrtContentTests(unittest.TestCase):
    def setUp(self):
        self.translator = make_translator()

    def test_strips_markdown_code_fences(self):
        dirty = "```srt\n" + SRT + "\n```"
        cleaned = self.translator._clean_srt_content(dirty)
        self.assertNotIn("```", cleaned)
        self.assertIn("first line", cleaned)

    def test_strips_json_artifacts_and_prose(self):
        dirty = (
            "翻译结果：\n{\"translated_srt\": \"...\"}\njson\n" + SRT
        )
        cleaned = self.translator._clean_srt_content(dirty)
        self.assertNotIn("translated_srt", cleaned)
        self.assertIn("first line", cleaned)

    def test_preserves_multiline_subtitle_text(self):
        dirty = "```srt\n4\n00:00:01,000 --> 00:00:02,000\nline one\nline two\n```"
        cleaned = self.translator._clean_srt_content(dirty)
        self.assertIn("line one\nline two", cleaned)


class PromptPathTests(unittest.TestCase):
    def test_project_prompt_md_resolves(self):
        path = get_prompt_path("prompt.md")
        self.assertTrue(path.endswith("prompt.md"))


if __name__ == "__main__":
    unittest.main()
