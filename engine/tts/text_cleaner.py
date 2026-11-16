"""Canonical text cleaning for speech synthesis.

There is exactly one narration cleaner in PIXEL AI: :func:`engine.narration.
sanitize_narration`.  This module is a thin TTS-facing adapter so the Kokoro
engine can never drift from the narration contract used by script generation,
validation and quality control.

The cleaner removes production syntax (Markdown, URLs, stage directions,
speaker labels) but preserves legitimate spoken language and technical terms
such as ``TCP/IP``, ``C/C++``, ``500 km/h`` and ``#1``.
"""

from __future__ import annotations

from engine.narration import count_words, sanitize_narration


def clean_text(text: str) -> str:
    """Return narration-only text suitable for Kokoro.

    Idempotent: cleaning an already-clean script returns it unchanged apart
    from whitespace normalisation.
    """
    cleaned = sanitize_narration(text or "")
    # Guarantee a single trailing newline without altering paragraph structure.
    return cleaned.strip() + "\n" if cleaned.strip() else ""


def text_stats(text: str) -> dict[str, int]:
    """Small deterministic statistics used by the TTS report."""
    cleaned = clean_text(text)
    return {
        "characters": len(cleaned),
        "words": count_words(cleaned),
    }


__all__ = ["clean_text", "text_stats"]
