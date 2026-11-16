"""Internal Kokoro TTS integration.

The package deliberately has no import-time dependency on Kokoro, Torch or
NumPy.  ``project_audio`` and ``kokoro_engine`` import those only when audio is
actually generated, so commands such as ``status``, ``review`` and ``validate``
stay fast and keep working on machines without the TTS stack installed.
"""

from __future__ import annotations

__all__ = [
    "ParsedShort",
    "SHORT_HEADING_PATTERN",
    "SHORT_HEADING_TEMPLATE",
    "clean_text",
    "format_shorts",
    "parse_shorts_text",
]


def __getattr__(name: str):
    """Lazily re-export the public helpers without importing heavy modules."""
    if name in {
        "ParsedShort",
        "SHORT_HEADING_PATTERN",
        "SHORT_HEADING_TEMPLATE",
        "format_shorts",
        "parse_shorts_text",
    }:
        from engine.tts import shorts_parser

        return getattr(shorts_parser, name)
    if name == "clean_text":
        from engine.tts.text_cleaner import clean_text

        return clean_text
    raise AttributeError(name)
