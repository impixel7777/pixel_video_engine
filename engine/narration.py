"""Narration text utilities.

This module is the single authority on what counts as *narration*.  It exists
because the two files the Kokoro TTS script consumes — ``Long.txt`` and
``Short.txt`` — must contain spoken words only, and because the visual director
needs exact character anchors into that text.

Nothing here calls an AI model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from engine.tts.shorts_parser import (
    ParsedShort,
    SHORT_HEADING_PATTERN,
    SHORT_HEADING_RE,
    SHORT_HEADING_TEMPLATE,
    parse_shorts_text,
)

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "fig", "no", "approx", "dept", "est", "inc", "ltd", "co", "u.s",
    "u.k", "a.m", "p.m", "b.c", "a.d", "ca", "cf", "al",
}

#: Direction keywords that can mark a bracket as a production instruction.
_DIRECTION_WORDS = re.compile(
    r"\b(pause|music|sfx|sound effect|sound cue|cut to|scene|visual|b-?roll|"
    r"camera|fade|transition|image|shot|overlay|narration|narrator|voiceover|"
    r"v\.o|beat|silence|duration|timing|on screen|onscreen|graphic|montage)\b",
    re.IGNORECASE,
)

#: A bracket is only a cue when it *starts* with a direction word.  Merely
#: containing "camera" or "music" is not enough - "the timing was terrible" is
#: legitimate narration.
_DIRECTION_START = re.compile(
    r"^\s*(?:pause|beat|silence|music|sfx|sound effect|sound cue|sound|"
    r"cut to|fade|transition|vo|v\.o\.|voiceover|narration|narrator|camera|"
    r"b-?roll|scene|visual|image|shot|overlay|montage|on screen|"
    r"text on screen|title card|graphic)\b",
    re.IGNORECASE,
)

_CUE_VERBS = re.compile(
    r"\b(plays|playing|fades|fading|cuts|swells|rises|drops|starts|stops|"
    r"continues|hits|booms|whooshes|pauses|beats|enters|exits|appears)\b",
    re.IGNORECASE,
)

#: Words that indicate the parenthetical is ordinary prose instead of a cue.
_PROSE_MARKERS = re.compile(
    r"\b(is|was|are|were|has|had|have|i|we|you|he|she|they|the|this|that|"
    r"because|which|who|whom|whose|if|when|while|after|before)\b",
    re.IGNORECASE,
)

#: Deliberately narrow.  It only strips an opener when it names a deliverable
#: ("here is the narration:") - never a legitimate spoken opener such as
#: "Here's the thing about flash memory:".
_META_PREAMBLE = re.compile(
    r"(?is)^\s*(?:"
    r"(?:(?:okay|ok|alright|sure|certainly|absolutely|of course)\b"
    r"[\s,;:.\-\u2013\u2014]{0,4})?"
    r"(?:here(?:'s| is)|below is|the following is)\s+"
    r"(?:the|my|your|a|an)?\s*"
    r"(?:final\s+|corrected\s+|complete\s+)?"
    r"(?:narration|script|text|version|draft|content|response|answer|json|output|story)"
    r"\b[^\n]{0,60}?[:\-\u2013\u2014]\s*"
    r")"
)

_REASONING_LEFTOVER = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NarrationBlock:
    """A contiguous slice of narration used as the anchor for one scene."""

    block_id: str
    text: str
    start_char: int
    end_char: int
    sentence_count: int

    @property
    def word_count(self) -> int:
        return count_words(self.text)

    @property
    def paragraph_index(self) -> int:
        return self.start_char


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def _looks_like_parenthetical_cue(inner: str) -> bool:
    """Conservative test for ``(music plays)`` style production cues.

    Legitimate parenthetical narration is preserved even when it mentions
    timing, music or a camera.  A cue must be short, start like a direction (or
    contain a direction noun plus an explicit cue verb) and must not read like
    ordinary prose.
    """
    value = (inner or "").strip()
    if not value or len(value) > 60:
        return False
    words = value.split()
    if len(words) > 6:
        return False
    if _PROSE_MARKERS.search(value):
        return False
    if _DIRECTION_START.match(value):
        return True
    return bool(_DIRECTION_WORDS.search(value) and _CUE_VERBS.search(value))


def strip_bracketed_directions(text: str) -> str:
    """Remove production cues, but never ordinary parenthetical narration.

    ``[pause]`` and ``(dramatic music plays)`` are removed.
    ``(a 1993 design)`` and ``(the timing was terrible)`` are preserved.
    """

    def replace_square(match: re.Match[str]) -> str:
        inner = match.group(1)
        # Square brackets are production notation, but "[1]" citations and
        # "[sic]" are still readable if left alone; only drop cue-shaped ones.
        if _DIRECTION_START.match(inner or "") or (
            _DIRECTION_WORDS.search(inner or "") and len(inner) <= 30
        ):
            return " "
        return match.group(0)

    def replace_round(match: re.Match[str]) -> str:
        return " " if _looks_like_parenthetical_cue(match.group(1)) else match.group(0)

    text = re.sub(r"\[([^\[\]]{0,120})\]", replace_square, text)
    text = re.sub(r"\(([^()]{0,120})\)", replace_round, text)
    return text


def sanitize_narration(text: str) -> str:
    """Reduce model output to clean spoken narration.

    Removes production syntax, but is deliberately conservative about spoken
    language: technical terms (``TCP/IP``, ``C/C++``, ``500 km/h``, ``#1``) and
    normal parenthetical asides survive unchanged.
    """
    if not text:
        return ""

    text = _REASONING_LEFTOVER.sub("", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*={3,}\s*$", "", text)
    text = re.sub(r"(?m)^\s*-{3,}\s*$", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    text = re.sub(r"\*{1,3}|_{1,3}|~~|`", "", text)
    text = re.sub(r"(?m)^\s*[-\*\u2022]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    text = re.sub(
        r"(?m)^\s*(?:narrator|narration|voice\s*over|voiceover|v\.?o\.?|speaker|host)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = strip_bracketed_directions(text)
    text = _META_PREAMBLE.sub("", text)

    # Only a standalone ampersand is spoken as "and".  "R&D" and "AT&T" keep
    # their written form rather than becoming "R and D" / "AT and T".
    text = re.sub(r"\s&\s", " and ", text)
    # Do NOT rewrite '/' between letters: TCP/IP, C/C++ and 500 km/h must live.
    text = re.sub(r"[`~^|]", "", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_pure_narration(text: str) -> tuple[bool, list[str]]:
    """Check that a narration file contains no production metadata.

    The checks mirror :func:`sanitize_narration`: they reject metadata lines
    (``Image prompt:``, ``SFX:``, ``Duration:``) rather than normal spoken uses
    of words such as "camera", "music" or "timing".
    """
    problems: list[str] = []
    if re.search(r"```", text):
        problems.append("contains a code fence")
    if re.search(r"(?m)^\s{0,3}#{1,6}\s+", text):
        problems.append("contains a markdown heading")
    if re.search(r"(?m)^\s*[-*\u2022]\s+", text):
        problems.append("contains bullet points")
    if re.search(r"\*\*|__|~~", text):
        problems.append("contains markdown emphasis")
    if re.search(r"https?://|www\.", text):
        problems.append("contains a URL")
    if re.search(
        r"(?im)^\s*(?:target\s+duration|duration|runtime|length)\s*[:=\-\u2013\u2014]",
        text,
    ):
        problems.append("contains duration metadata")
    if re.search(
        r"(?im)^\s*(?:image|visual|video)\s+prompt\s*[:=\-\u2013\u2014]"
        r"|^\s*(?:visual\s+description|camera(?:\s+move(?:ment)?)?|shot|scene\s*\d+"
        r"|b-?roll|on[ -]screen\s+text|text\s+overlay)\s*[:=\-\u2013\u2014]",
        text,
    ):
        problems.append("contains visual direction")
    if re.search(
        r"(?im)^\s*(?:music(?:\s+cue)?|sound\s+effect|sfx|fade\s+in|fade\s+out"
        r"|audio\s+cue)\s*[:=\-\u2013\u2014]",
        text,
    ):
        problems.append("contains audio direction")
    if re.search(r"</?think\b", text):
        problems.append("contains reasoning tags")
    if re.search(r"(?im)^\s*(?:narrator|voiceover|voice\s*over|v\.o\.|speaker|host)\s*:", text):
        problems.append("contains speaker labels")
    if _META_PREAMBLE.match(text.strip()):
        problems.append("contains a model preamble")
    if re.search(r"(?im)^\s*\[(?:pause|beat|music|sfx|cut|fade)\b[^\]]*\]", text):
        problems.append("contains a bracketed stage direction")
    return (not problems, problems)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


_WORD_RE = re.compile(r"[A-Za-z0-9\u00C0-\u024F']+")


def count_words(text: str) -> int:
    """Count spoken words (numbers and contractions included)."""
    return len(_WORD_RE.findall(text))


def estimate_duration_seconds(words: int, words_per_minute: int) -> float:
    """Convert a word count into an approximate narration duration."""
    if words_per_minute <= 0:
        return 0.0
    return round(words / words_per_minute * 60.0, 2)


def word_count_to_range(words: int, tolerance: float = 0.15) -> tuple[int, int]:
    """Inclusive acceptable word range around a target."""
    low = max(1, math.floor(words * (1 - tolerance)))
    high = math.ceil(words * (1 + tolerance))
    return low, high


# --------------------------------------------------------------------------- #
# Sentences and blocks
# --------------------------------------------------------------------------- #


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans for sentences, preserving exact offsets."""
    if not text.strip():
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)

    for match in re.finditer(r"[.!?]+(?=[\"'\)\]]*(?:\s|$))", text):
        end = match.end()
        # Include a closing quote or bracket that trails the punctuation.
        while end < length and text[end] in "\"')]":
            end += 1

        prefix = text[start : match.start()]
        token = prefix.split()[-1].lower().lstrip("\"'([") if prefix.split() else ""
        if token in ABBREVIATIONS:
            continue

        chunk = text[start:end]
        if chunk.strip():
            spans.append((start, end))
        start = end
        while start < length and text[start].isspace():
            start += 1

    if start < length:
        tail = text[start:length]
        if tail.strip():
            spans.append((start, length))

    return spans


def split_into_blocks(
    text: str,
    *,
    target_sentences: int = 3,
    target_words: int = 70,
    prefix: str = "B",
    max_blocks: int | None = None,
) -> list[NarrationBlock]:
    """Group narration into scene-sized blocks with exact character anchors."""
    spans = split_sentences(text)
    if not spans:
        return []

    blocks: list[NarrationBlock] = []
    current: list[tuple[int, int]] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if not current:
            return
        start = current[0][0]
        end = current[-1][1]
        raw = text[start:end].strip()
        if raw:
            blocks.append(
                NarrationBlock(
                    block_id=f"{prefix}{len(blocks) + 1:02d}",
                    text=raw,
                    start_char=start,
                    end_char=end,
                    sentence_count=len(current),
                )
            )
        current = []
        current_words = 0

    for span in spans:
        sentence_words = count_words(text[span[0] : span[1]])
        current.append(span)
        current_words += sentence_words

        reached_sentences = len(current) >= target_sentences
        reached_words = current_words >= target_words
        if reached_sentences and reached_words:
            flush()

    flush()

    if max_blocks is not None and len(blocks) > max_blocks:
        # Merge the closest neighbours until the limit is met, never dropping text.
        while len(blocks) > max_blocks:
            merged: list[NarrationBlock] = []
            index = 0
            while index < len(blocks):
                if index + 1 < len(blocks):
                    first, second = blocks[index], blocks[index + 1]
                    merged.append(
                        NarrationBlock(
                            block_id=first.block_id,
                            text=(first.text + "\n\n" + second.text).strip(),
                            start_char=first.start_char,
                            end_char=second.end_char,
                            sentence_count=first.sentence_count + second.sentence_count,
                        )
                    )
                    index += 2
                else:
                    merged.append(blocks[index])
                    index += 1
            blocks = merged

    # Re-number so ids stay contiguous after merging.
    return [
        NarrationBlock(
            block_id=f"{prefix}{index + 1:02d}",
            text=block.text,
            start_char=block.start_char,
            end_char=block.end_char,
            sentence_count=block.sentence_count,
        )
        for index, block in enumerate(blocks)
    ]


def coverage_ratio(text: str, blocks: list[NarrationBlock]) -> float:
    """Fraction of non-whitespace narration characters covered by the blocks."""
    total = len(re.sub(r"\s", "", text))
    if total == 0:
        return 0.0
    covered = sum(len(re.sub(r"\s", "", block.text)) for block in blocks)
    return round(min(1.0, covered / total), 4)


# --------------------------------------------------------------------------- #
# Shorts
# --------------------------------------------------------------------------- #


def format_shorts(
    shorts: list[dict[str, object]],
    *,
    prefix: str = "SHORT",
    separator: str = "\u2014",
) -> str:
    """Render Short.txt in the exact format the canonical parser expects."""
    chunks: list[str] = []
    for index, short in enumerate(shorts, start=1):
        number = int(short.get("index", index) or index)
        title = str(short.get("title", "")).strip() or f"Short {number}"
        narration = sanitize_narration(str(short.get("narration", "")))
        heading = SHORT_HEADING_TEMPLATE.format(index=number, title=title)
        if prefix != "SHORT" or separator != "\u2014":
            heading = f"{prefix} {number:02d} {separator} {title}"
        chunks.append(f"{heading}\n\n{narration}")
    return "\n\n".join(chunks).strip() + "\n"


# --------------------------------------------------------------------------- #
# Repetition detection
# --------------------------------------------------------------------------- #


def repeated_phrases(text: str, ngram: int = 8, minimum_hits: int = 3) -> list[str]:
    """Find phrases of *ngram* words that appear at least *minimum_hits* times."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < ngram * minimum_hits:
        return []
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(words) - ngram + 1):
        gram = tuple(words[index : index + ngram])
        counts[gram] = counts.get(gram, 0) + 1
    repeated = [
        " ".join(gram)
        for gram, count in sorted(counts.items(), key=lambda item: -item[1])
        if count >= minimum_hits
    ]
    # Keep only maximal phrases: drop a phrase contained in a longer repeated one.
    unique: list[str] = []
    for phrase in repeated:
        if not any(phrase in other and phrase != other for other in repeated):
            unique.append(phrase)
    return unique[:10]


def sentence_openers(text: str) -> list[str]:
    """Return the first four words of each sentence, lowercased."""
    openers: list[str] = []
    for start, end in split_sentences(text):
        words = _WORD_RE.findall(text[start:end].lower())
        if words:
            openers.append(" ".join(words[:4]))
    return openers


__all__ = [
    "NarrationBlock",
    "ParsedShort",
    "SHORT_HEADING_RE",
    "SHORT_HEADING_TEMPLATE",
    "count_words",
    "coverage_ratio",
    "estimate_duration_seconds",
    "format_shorts",
    "is_pure_narration",
    "parse_shorts_text",
    "repeated_phrases",
    "sanitize_narration",
    "sentence_openers",
    "split_into_blocks",
    "split_sentences",
    "strip_bracketed_directions",
    "word_count_to_range",
]
