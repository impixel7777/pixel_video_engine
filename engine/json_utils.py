"""Robust helpers for extracting, repairing and validating JSON from LLM text.

Reasoning models frequently wrap JSON in prose, markdown fences or ``<think>``
blocks.  Nothing here guesses at semantics: it only recovers the JSON payload
and validates it against a schema.  Invalid output is surfaced, never silently
accepted.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class JsonExtractionError(ValueError):
    """Raised when no valid JSON payload can be recovered from model output."""


# --------------------------------------------------------------------------- #
# Reasoning stripping
# --------------------------------------------------------------------------- #


def strip_reasoning(text: str) -> str:
    """Remove DeepSeek-R1 style reasoning blocks from model content.

    Handles three real-world shapes:

    1. ``<think>...</think>final``      – the normal case.
    2. ``<think>`` never closed        – keep only the tail after the last
       ``</think>`` if present, otherwise drop everything before the first
       newline pair and keep nothing that looks like analysis.
    3. ``final<think>...</think>``     – reasoning appended after the answer.
    """
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub("", text)
    if _THINK_CLOSE.search(cleaned):
        # An opening tag was never closed: keep whatever follows the last close.
        cleaned = cleaned[cleaned.rfind("</think") + len("</think>") :]
    if _THINK_OPEN.search(cleaned):
        # Unterminated reasoning block: drop from the opening tag onwards.
        cleaned = cleaned[:_THINK_OPEN.search(cleaned).start()]  # type: ignore[union-attr]
    return cleaned.strip()


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


def _balanced_json_slice(text: str) -> str | None:
    """Return the first complete top-level JSON object/array found in *text*.

    String literals and escapes are respected so braces inside strings do not
    confuse the depth counter.
    """
    start = -1
    opener = ""
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            opener = char
            break
    if start < 0:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]`` outside of strings."""
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead] in " \t\r\n":
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1  # drop the comma
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _close_unbalanced(text: str) -> str:
    """Append missing closing brackets for truncated model output."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()
    repaired = text
    if in_string:
        repaired += '"'
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def extract_json(text: str, *, allow_repair: bool = True) -> Any:
    """Extract a JSON value from arbitrary model text.

    Raises :class:`JsonExtractionError` when nothing usable can be recovered.
    """
    if text is None:
        raise JsonExtractionError("Empty model response")

    candidate = strip_reasoning(str(text)).strip()
    if not candidate:
        raise JsonExtractionError("Model response was empty after stripping reasoning")

    # 1. Direct parse.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 2. Fenced block.
    for match in _FENCE.finditer(candidate):
        chunk = match.group(1).strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            sliced = _balanced_json_slice(chunk)
            if sliced:
                try:
                    return json.loads(sliced)
                except json.JSONDecodeError:
                    pass

    # 3. First balanced object/array.
    sliced = _balanced_json_slice(candidate)
    if sliced:
        try:
            return json.loads(sliced)
        except json.JSONDecodeError:
            if not allow_repair:
                raise JsonExtractionError("JSON payload was malformed beyond recovery")
            repaired = _close_unbalanced(_strip_trailing_commas(sliced))
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                raise JsonExtractionError(f"Unrepairable JSON: {exc}") from exc

    if allow_repair:
        repaired = _close_unbalanced(_strip_trailing_commas(candidate))
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise JsonExtractionError(f"No JSON found in response: {exc}") from exc

    raise JsonExtractionError("No JSON object found in model response")


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def _load_validator(schema: dict[str, Any]):
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_json(data: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty when valid)."""
    validator = _load_validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path)):
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        errors.append(f"{location}: {error.message}")
    return errors


def format_validation_errors(errors: list[str], limit: int = 12) -> str:
    """Render schema errors compactly for a repair prompt."""
    shown = errors[:limit]
    suffix = "" if len(errors) <= limit else f"\n... and {len(errors) - limit} more"
    return "\n".join(f"- {error}" for error in shown) + suffix


__all__ = [
    "JsonExtractionError",
    "extract_json",
    "strip_reasoning",
    "validate_json",
    "format_validation_errors",
]
