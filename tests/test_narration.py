"""Tests for narration parsing, formatting and purity checks."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from engine.narration import (
    SHORT_HEADING_RE,
    count_words,
    coverage_ratio,
    format_shorts,
    is_pure_narration,
    parse_shorts_text,
    repeated_phrases,
    sanitize_narration,
    split_into_blocks,
    split_sentences,
    word_count_to_range,
)
from tests.helpers import FIXTURE_LONG, fixture_shorts

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LEGACY_SCRIPT = REPO_ROOT / "old" / "pixel_tts.py"


def legacy_short_pattern() -> str:
    """Read the *real* pattern out of ``old/pixel_tts.py`` via AST."""
    source = LEGACY_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "parse_shorts":
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and statement.targets
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "pattern"
                ):
                    parts: list[str] = []

                    def collect(value: ast.AST) -> None:
                        if isinstance(value, ast.Constant) and isinstance(value.value, str):
                            parts.append(value.value)
                        elif isinstance(value, ast.BinOp):
                            collect(value.left)
                            collect(value.right)

                    collect(statement.value)
                    return "".join(parts)
    raise AssertionError("Could not find the pattern assignment in parse_shorts()")


class TestSentenceSplitting(unittest.TestCase):
    def test_spans_are_ordered_and_non_empty(self) -> None:
        spans = split_sentences(FIXTURE_LONG)
        self.assertGreaterEqual(len(spans), 8)
        previous_end = -1
        for start, end in spans:
            self.assertGreaterEqual(start, previous_end)
            self.assertLess(start, end)
            self.assertTrue(FIXTURE_LONG[start:end].strip())
            previous_end = end

    def test_spans_preserve_all_non_whitespace_text(self) -> None:
        spans = split_sentences(FIXTURE_LONG)
        joined = "".join(FIXTURE_LONG[start:end] for start, end in spans)
        strip = lambda text: re.sub(r"\s", "", text)  # noqa: E731
        self.assertEqual(strip(joined), strip(FIXTURE_LONG))

    def test_abbreviation_is_not_a_sentence_end(self) -> None:
        text = "Dr. Smith arrived at 9 a.m. and left. Nobody noticed."
        spans = split_sentences(text)
        self.assertEqual(len(spans), 2, msg=f"spans={spans}")
        self.assertTrue(text[spans[0][0] : spans[0][1]].endswith("left."))


class TestBlocks(unittest.TestCase):
    def test_block_offsets_point_at_the_block_text(self) -> None:
        blocks = split_into_blocks(FIXTURE_LONG, target_sentences=2, target_words=30)
        self.assertGreaterEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(FIXTURE_LONG[block.start_char : block.end_char].strip(), block.text)
            self.assertGreater(block.end_char, block.start_char)
            self.assertGreater(block.word_count, 0)

    def test_blocks_cover_all_narration(self) -> None:
        blocks = split_into_blocks(FIXTURE_LONG, target_sentences=2, target_words=30)
        self.assertAlmostEqual(coverage_ratio(FIXTURE_LONG, blocks), 1.0, places=3)

    def test_max_blocks_never_drops_text(self) -> None:
        blocks = split_into_blocks(
            FIXTURE_LONG, target_sentences=1, target_words=10, max_blocks=2
        )
        self.assertLessEqual(len(blocks), 2)
        self.assertGreaterEqual(coverage_ratio(FIXTURE_LONG, blocks), 0.99)

    def test_block_ids_are_contiguous(self) -> None:
        blocks = split_into_blocks(FIXTURE_LONG, prefix="B")
        self.assertEqual([block.block_id for block in blocks], [f"B{i:02d}" for i in range(1, len(blocks) + 1)])

    def test_empty_input(self) -> None:
        self.assertEqual(split_into_blocks(""), [])
        self.assertEqual(coverage_ratio("", []), 0.0)


class TestShortsFormat(unittest.TestCase):
    def test_round_trip(self) -> None:
        shorts = fixture_shorts(4)
        text = format_shorts(shorts)
        parsed = parse_shorts_text(text)
        self.assertEqual(len(parsed), 4)
        self.assertEqual([short.index for short in parsed], [1, 2, 3, 4])
        for parsed_short, original in zip(parsed, shorts):
            self.assertEqual(parsed_short.title, original["title"])
            self.assertEqual(parsed_short.narration, original["narration"])

    def test_heading_uses_em_dash_and_zero_padding(self) -> None:
        text = format_shorts([{"index": 1, "title": "A Title", "narration": "Hello there."}])
        self.assertIn("SHORT 01 \u2014 A Title", text)
        self.assertTrue(text.startswith("SHORT 01"))

    def test_heading_has_no_duration_metadata(self) -> None:
        text = format_shorts([{"index": 2, "title": "No Metadata", "narration": "Hello."}])
        heading = text.splitlines()[0]
        self.assertNotIn("(", heading)
        self.assertNotIn("duration", heading.lower())
        self.assertNotIn("sec", heading.lower())

    def test_legacy_regex_matches_every_heading(self) -> None:
        """The untouched Kokoro regex must match what we emit."""
        self.assertTrue(LEGACY_SCRIPT.is_file(), f"legacy script missing: {LEGACY_SCRIPT}")
        legacy = re.compile(legacy_short_pattern())
        text = format_shorts(fixture_shorts(6))
        matches = list(legacy.finditer(text))
        self.assertEqual(len(matches), 6)
        self.assertEqual([int(match.group(1)) for match in matches], [1, 2, 3, 4, 5, 6])
        self.assertEqual(matches[0].group(2), fixture_shorts(6)[0]["title"])

    def test_local_regex_equals_legacy_regex(self) -> None:
        self.assertEqual(SHORT_HEADING_RE.pattern, legacy_short_pattern())

    def test_legacy_regex_accepts_markdown_and_bold_variants(self) -> None:
        legacy = re.compile(legacy_short_pattern())
        for variant in (
            "## SHORT 01 \u2014 Title",
            "SHORT 1 - Title",
            "SHORT 02: Title",
            "=== SHORT 03 \u2014 Title ===",
        ):
            self.assertTrue(legacy.search(variant), msg=variant)

    def test_parser_ignores_heading_without_body(self) -> None:
        text = "SHORT 01 \u2014 Empty\n\nSHORT 02 \u2014 Full\n\nSome narration here."
        parsed = parse_shorts_text(text)
        self.assertEqual([short.index for short in parsed], [2])


class TestSanitize(unittest.TestCase):
    def test_removes_markdown_and_directions(self) -> None:
        raw = (
            "## SHORT 01 \u2014 Title\n\n"
            "**Bold** opening. [pause] Then a beat. (dramatic music plays)\n"
            "Narrator: the real sentence starts here.\n"
            "Here is the narration: This is the actual content."
        )
        cleaned = sanitize_narration(raw)
        self.assertNotIn("##", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("[pause]", cleaned)
        self.assertNotIn("dramatic music", cleaned)
        self.assertNotIn("Narrator:", cleaned)
        self.assertIn("This is the actual content.", cleaned)

    def test_keeps_legitimate_parentheses(self) -> None:
        cleaned = sanitize_narration("The chip (a 1993 design) was slow.")
        self.assertIn("(a 1993 design)", cleaned)

    def test_removes_urls_and_fences(self) -> None:
        cleaned = sanitize_narration(
            "```json\n{\"a\": 1}\n```\nVisit https://example.com for more."
        )
        self.assertNotIn("```", cleaned)
        self.assertNotIn("https://", cleaned)

    def test_collapses_whitespace_but_keeps_paragraphs(self) -> None:
        cleaned = sanitize_narration("One.   Two.\n\n\n\nThree.")
        self.assertEqual(cleaned, "One. Two.\n\nThree.")

    def test_empty(self) -> None:
        self.assertEqual(sanitize_narration(""), "")


class TestPurity(unittest.TestCase):
    def test_accepts_plain_narration(self) -> None:
        ok, problems = is_pure_narration(FIXTURE_LONG)
        self.assertTrue(ok, msg=problems)

    def test_rejects_markdown_heading(self) -> None:
        ok, problems = is_pure_narration("## Section One\nSome text.")
        self.assertFalse(ok)
        self.assertTrue(any("heading" in problem for problem in problems))

    def test_rejects_duration_metadata(self) -> None:
        ok, problems = is_pure_narration("Some text.\nTarget duration: 60 seconds")
        self.assertFalse(ok)
        self.assertTrue(any("duration" in problem for problem in problems))

    def test_rejects_visual_and_audio_direction(self) -> None:
        for text in (
            "Some text.\nImage prompt: a lighthouse",
            "Some text.\nCamera move: slow push",
            "Some text.\nMusic cue: sad strings",
            "Some text.\nSFX: thunder",
        ):
            ok, problems = is_pure_narration(text)
            self.assertFalse(ok, msg=text)

    def test_rejects_reasoning_tags(self) -> None:
        ok, _ = is_pure_narration("Some text. <think>hidden</think>")
        self.assertFalse(ok)

    def test_rejects_urls(self) -> None:
        ok, _ = is_pure_narration("Source: https://example.com/article")
        self.assertFalse(ok)


class TestMeasurement(unittest.TestCase):
    def test_word_count(self) -> None:
        self.assertEqual(count_words("one two three"), 3)
        self.assertEqual(count_words("don't stop"), 2)
        self.assertEqual(count_words(""), 0)

    def test_word_range(self) -> None:
        low, high = word_count_to_range(100, 0.15)
        self.assertEqual((low, high), (85, 115))

    def test_repeated_phrases_detected(self) -> None:
        sentence = "the quick brown fox jumps over the lazy dog"
        text = f"{sentence}. {sentence}. {sentence}."
        phrases = repeated_phrases(text, ngram=6, minimum_hits=3)
        self.assertTrue(phrases)
        self.assertTrue(any("quick brown" in phrase for phrase in phrases))

    def test_no_false_positive_on_unique_text(self) -> None:
        self.assertEqual(repeated_phrases(FIXTURE_LONG, ngram=8, minimum_hits=3), [])

    def test_legitimate_question_opener_not_flagged(self) -> None:
        ok, problems = is_pure_narration(
            "How does it work? Nobody agreed. That was the problem."
        )
        self.assertTrue(ok, msg=problems)


if __name__ == "__main__":
    unittest.main()
