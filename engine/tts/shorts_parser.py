"""Canonical Shorts parser.

This is the *single* implementation of the ``SHORT 01 — Title`` contract.  The
legacy Kokoro script used the same regular expression; PIXEL AI now owns it and
no production path depends on ``old/pixel_tts.py``.

Canonical output written by the pipeline::

    SHORT 01 — The Frame That Took a Day

The parser accepts the same harmless variants the legacy script accepted
(Markdown heading markers, ``===`` rules, ``:``/``-``/en dash separators) so
existing ``Short.txt`` files keep working.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Same regular expression the legacy ``pixel_tts.py`` used.
SHORT_HEADING_PATTERN = (
    r"(?im)^\s*"
    r"(?:#{1,6}\s*)?"
    r"(?:={3,}\s*)?"
    r"SHORT\s*(\d+)"
    r"(?:\s*[:\-\u2014\u2013]\s*(.*?))?"
    r"\s*(?:={3,})?\s*$"
)
SHORT_HEADING_RE = re.compile(SHORT_HEADING_PATTERN)

#: Canonical heading written by this project (zero-padded, em dash, same line).
SHORT_HEADING_TEMPLATE = "SHORT {index:02d} \u2014 {title}"


@dataclass(frozen=True)
class ParsedShort:
    """One Short exactly as the TTS engine will consume it."""

    index: int
    title: str
    narration: str


def parse_shorts_text(text: str) -> list[ParsedShort]:
    """Parse ``Short.txt`` into Shorts, matching the legacy parser semantics."""
    matches = list(SHORT_HEADING_RE.finditer(text))
    shorts: list[ParsedShort] = []
    for index, match in enumerate(matches):
        number = int(match.group(1))
        title = (match.group(2) or f"Short {number}").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        narration = text[start:end].strip()
        if narration:
            shorts.append(ParsedShort(index=number, title=title, narration=narration))
    return shorts


__all__ = [
    "ParsedShort",
    "SHORT_HEADING_PATTERN",
    "SHORT_HEADING_RE",
    "SHORT_HEADING_TEMPLATE",
    "parse_shorts_text",
]
