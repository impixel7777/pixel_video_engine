"""Tests for JSON extraction, repair and validation."""

from __future__ import annotations

import unittest

from engine.json_utils import (
    JsonExtractionError,
    extract_json,
    format_validation_errors,
    strip_reasoning,
    validate_json,
)


class TestStripReasoning(unittest.TestCase):
    def test_removes_closed_block(self) -> None:
        self.assertEqual(strip_reasoning("<think>thinking</think>answer"), "answer")

    def test_removes_multiline_block(self) -> None:
        text = "<think>\nline one\nline two\n</think>\n\nFinal answer."
        self.assertEqual(strip_reasoning(text), "Final answer.")

    def test_removes_appended_block(self) -> None:
        self.assertEqual(strip_reasoning("answer<think>more</think>"), "answer")

    def test_unterminated_block_drops_the_tail(self) -> None:
        self.assertEqual(strip_reasoning("answer <think>never closed"), "answer")

    def test_plain_text_untouched(self) -> None:
        self.assertEqual(strip_reasoning("plain text"), "plain text")

    def test_empty(self) -> None:
        self.assertEqual(strip_reasoning(""), "")


class TestExtractJson(unittest.TestCase):
    def test_direct_object(self) -> None:
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_direct_array(self) -> None:
        self.assertEqual(extract_json("[1, 2, 3]"), [1, 2, 3])

    def test_fenced_block(self) -> None:
        text = 'Sure!\n```json\n{"a": [1, 2]}\n```\nHope that helps.'
        self.assertEqual(extract_json(text), {"a": [1, 2]})

    def test_object_inside_prose(self) -> None:
        text = 'Here is the result: {"a": 1, "b": {"c": 2}} — done.'
        self.assertEqual(extract_json(text), {"a": 1, "b": {"c": 2}})

    def test_braces_inside_strings(self) -> None:
        text = '{"prompt": "a {brace} inside", "n": 1}'
        self.assertEqual(extract_json(text)["prompt"], "a {brace} inside")

    def test_escaped_quotes(self) -> None:
        text = r'{"quote": "he said \"hello\""}'
        self.assertEqual(extract_json(text)["quote"], 'he said "hello"')

    def test_trailing_comma_repaired(self) -> None:
        self.assertEqual(extract_json('{"a": 1,}'), {"a": 1})

    def test_truncated_object_repaired(self) -> None:
        result = extract_json('{"a": 1, "b": [1, 2')
        self.assertEqual(result, {"a": 1, "b": [1, 2]})

    def test_truncated_string_repaired(self) -> None:
        result = extract_json('{"a": "unfinished')
        self.assertEqual(result, {"a": "unfinished"})

    def test_reasoning_is_stripped_before_extraction(self) -> None:
        text = '<think>I should return {"wrong": true}</think>{"right": true}'
        self.assertEqual(extract_json(text), {"right": True})

    def test_empty_raises(self) -> None:
        with self.assertRaises(JsonExtractionError):
            extract_json("")

    def test_no_json_raises(self) -> None:
        with self.assertRaises(JsonExtractionError):
            extract_json("this is just prose with no structure at all")

    def test_repair_can_be_disabled(self) -> None:
        with self.assertRaises(JsonExtractionError):
            extract_json('{"a": 1,}', allow_repair=False)


class TestValidate(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "required": ["name", "count"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 2},
            "count": {"type": "integer", "minimum": 1},
        },
    }

    def test_valid(self) -> None:
        self.assertEqual(validate_json({"name": "ok", "count": 2}, self.SCHEMA), [])

    def test_missing_field(self) -> None:
        errors = validate_json({"name": "ok"}, self.SCHEMA)
        self.assertTrue(any("count" in error for error in errors))

    def test_wrong_type(self) -> None:
        errors = validate_json({"name": "ok", "count": "two"}, self.SCHEMA)
        self.assertTrue(errors)

    def test_additional_property_rejected(self) -> None:
        errors = validate_json({"name": "ok", "count": 1, "extra": True}, self.SCHEMA)
        self.assertTrue(any("extra" in error for error in errors))

    def test_format_validation_errors_limits_output(self) -> None:
        rendered = format_validation_errors([f"error {index}" for index in range(20)], limit=5)
        self.assertIn("error 0", rendered)
        self.assertIn("... and 15 more", rendered)


if __name__ == "__main__":
    unittest.main()
