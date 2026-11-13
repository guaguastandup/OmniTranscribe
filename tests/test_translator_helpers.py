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

    The helpers under test are pure and never touch instance state.
    """
    return UniversalTranslator.__new__(UniversalTranslator)


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
