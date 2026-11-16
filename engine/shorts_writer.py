"""Stage 5 - Shorts Generator.

Writes the standalone Shorts and renders ``Short.txt`` in exactly the heading
format the legacy Kokoro parser expects:

    SHORT 01 \u2014 Title

The model returns JSON; Python owns the file formatting.  That guarantees the
heading is always well formed and that no metadata leaks into narration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.ai_stage import AIStage, research_digest
from engine.logging_setup import get_logger
from engine.narration import (
    count_words,
    format_shorts,
    sanitize_narration,
    sentence_openers,
    split_sentences,
    word_count_to_range,
)

log = get_logger("shorts")


@dataclass
class ShortScript:
    index: int
    title: str
    narration: str
    central_idea: str = ""

    @property
    def word_count(self) -> int:
        return count_words(self.narration)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "central_idea": self.central_idea,
            "narration": self.narration,
            "word_count": self.word_count,
        }


@dataclass
class ShortsResult:
    shorts: list[ShortScript] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return format_shorts([short.to_dict() for short in self.shorts])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "shorts": [short.to_dict() for short in self.shorts],
            "issues": list(self.issues),
        }


class ShortsWriter(AIStage):
    """Produces ``Short.txt``."""

    stage_name = "shorts_writer"

    def write(
        self,
        *,
        long_script: str,
        research: dict[str, Any],
        count: int,
        short_seconds: int,
    ) -> ShortsResult:
        target_words = self._target_words(short_seconds)
        log.info("Writing %d Shorts at about %d words each", count, target_words)

        payload = self.run_json(
            prompt_name="shorts_generation",
            schema_name="shorts_script",
            label="shorts_generation",
            temperature=0.8,
            max_tokens=min(
                2600,
                max(700, count * (target_words * 2 + 90)),
            ),
            PROFILE=self.profile_block(),
            RESEARCH_DIGEST=self.fit(
                research_digest(research), keep="tail", label="research digest"
            ),
            LONG_SCRIPT=self.fit(long_script, keep="head", label="long narration"),
            SHORTS_COUNT=count,
            SHORT_SECONDS=short_seconds,
            TARGET_WORDS=target_words,
            LANGUAGE=self.profile.language,
        )

        result = self._postprocess(payload, count, target_words)

        # One bounded repair pass when the Shorts are too similar.
        duplicates = self._duplicate_openers(result.shorts)
        if duplicates:
            log.warning("Shorts share opening phrasing: %s", ", ".join(duplicates))
            result.issues.append(
                "Shorts with near-identical openings: " + ", ".join(duplicates)
            )

        log.info(
            "Shorts ready: %s",
            ", ".join(
                f"{short.index:02d}={short.word_count}w" for short in result.shorts
            ),
        )
        return result

    # -- helpers ------------------------------------------------------------ #
    def _target_words(self, short_seconds: int) -> int:
        """Words spoken in *short_seconds*, bounded by the configured limits."""
        natural = int(short_seconds / 60 * self.pipeline.words_per_minute)
        return max(
            45,
            min(natural or self.pipeline.short_target_words, self.pipeline.short_max_words),
        )

    def _postprocess(
        self, payload: dict[str, Any], count: int, target_words: int
    ) -> ShortsResult:
        items = payload.get("shorts") or []
        if not items:
            raise ValueError("Shorts writer returned no shorts")

        shorts: list[ShortScript] = []
        issues: list[str] = []
        seen_titles: set[str] = set()

        for position, item in enumerate(items[:count], start=1):
            narration = sanitize_narration(str(item.get("narration", "")))
            if not narration:
                issues.append(f"Short {position:02d} had empty narration and was dropped")
                continue

            title = " ".join(str(item.get("title", "")).split())
            if not title:
                title = self._title_from_narration(narration)
            if title.lower() in seen_titles:
                issues.append(f"Short {position:02d} reused the title '{title}'")
            seen_titles.add(title.lower())

            shorts.append(
                ShortScript(
                    index=position,
                    title=title,
                    narration=narration,
                    central_idea=str(item.get("central_idea", "")).strip(),
                )
            )

        if len(shorts) < count:
            issues.append(
                f"Requested {count} Shorts but only {len(shorts)} usable scripts were produced"
            )

        low, high = word_count_to_range(target_words, tolerance=0.30)
        for short in shorts:
            if short.word_count < low:
                issues.append(
                    f"Short {short.index:02d} is short: {short.word_count} words "
                    f"(expected about {target_words})"
                )
            elif short.word_count > high:
                issues.append(
                    f"Short {short.index:02d} is long: {short.word_count} words "
                    f"(expected about {target_words})"
                )

        return ShortsResult(shorts=shorts, issues=issues)

    @staticmethod
    def _duplicate_openers(shorts: list[ShortScript], words: int = 5) -> list[str]:
        seen: dict[str, int] = {}
        duplicates: list[str] = []
        for short in shorts:
            spans = split_sentences(short.narration)
            if not spans:
                continue
            first = " ".join(short.narration[spans[0][0] : spans[0][1]].split())
            key = " ".join(first.lower().split()[:words])
            if key in seen:
                duplicates.append(f"Short {short.index:02d} repeats Short {seen[key]:02d}")
            else:
                seen[key] = short.index
        return duplicates

    @staticmethod
    def _title_from_narration(narration: str) -> str:
        openers = sentence_openers(narration)
        if not openers:
            return "Untitled Short"
        return " ".join(openers[0].split()[:6]).title()


__all__ = ["ShortScript", "ShortsResult", "ShortsWriter"]
