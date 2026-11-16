"""Stage 3 - Content Planner.

Turns the brief and the research guidance into a sectioned outline.  The outline
is the contract the scriptwriter must satisfy, so this stage also normalises
section ids and word budgets.
"""

from __future__ import annotations

from typing import Any

from engine.ai_stage import AIStage, brief_digest, research_digest
from engine.logging_setup import get_logger

log = get_logger("planner")

MIN_SECTIONS = 4
MAX_SECTIONS = 8


class ContentPlanner(AIStage):
    """Produces ``Outline.json``."""

    stage_name = "content_planner"

    # -- budget ------------------------------------------------------------ #
    def target_words(self, duration_minutes: float) -> int:
        words = int(duration_minutes * self.pipeline.words_per_minute)
        return max(self.pipeline.long_min_words, min(words, self.pipeline.long_max_words))

    @staticmethod
    def section_count(duration_minutes: float) -> int:
        """One main section per ~1.5 minutes, clamped to a sane range."""
        count = round(duration_minutes / 1.5)
        return max(MIN_SECTIONS, min(count, MAX_SECTIONS))

    # -- planning ---------------------------------------------------------- #
    def plan(
        self,
        *,
        brief: dict[str, Any],
        research: dict[str, Any],
        duration_minutes: float,
    ) -> dict[str, Any]:
        sections = self.section_count(duration_minutes)
        target = self.target_words(duration_minutes)

        log.info(
            "Planning %d sections, about %d spoken words (%.0f min)",
            sections,
            target,
            duration_minutes,
        )

        outline = self.run_json(
            prompt_name="content_planning",
            schema_name="video_outline",
            label="content_planning",
            temperature=0.6,
            max_tokens=min(2200, self.client.budget.prompt_tokens // 2),
            PROFILE=self.profile_block(),
            BRIEF=self.fit(brief_digest(brief), keep="head", label="brief"),
            RESEARCH_DIGEST=self.fit(
                research_digest(research), keep="head", label="research digest"
            ),
            DURATION=duration_minutes,
            TARGET_WORDS=target,
            SECTION_COUNT=sections,
            LANGUAGE=self.profile.language,
        )

        outline = self._postprocess(outline, duration_minutes, target)
        total = sum(section["estimated_words"] for section in outline["sections"])
        log.info(
            "Outline ready: %d sections, %d planned words",
            len(outline["sections"]),
            total,
        )
        return outline

    # -- post-processing --------------------------------------------------- #
    def _postprocess(
        self,
        outline: dict[str, Any],
        duration_minutes: float,
        target_words: int,
    ) -> dict[str, Any]:
        outline["schema_version"] = "1.0"
        outline["target_duration_minutes"] = duration_minutes
        outline["target_word_count"] = target_words

        sections = outline.get("sections") or []
        if len(sections) < MIN_SECTIONS:
            raise ValueError(
                f"Outline returned {len(sections)} section(s); at least {MIN_SECTIONS} "
                "are required for a long-form video."
            )
        sections = sections[:MAX_SECTIONS]

        # Renumber ids deterministically so the writer's contract is stable.
        for index, section in enumerate(sections, start=1):
            section["id"] = f"S{index}"
            section.setdefault("key_points", [])
            section.setdefault("examples", [])
            section.setdefault("research_refs", [])
            section.setdefault("visual_opportunities", [])
            section["key_points"] = [
                point for point in section["key_points"] if str(point).strip()
            ][:6]
            if not section["key_points"]:
                section["key_points"] = [section.get("goal", "Cover the section topic")]

        # Normalise the word budget so hook + sections + conclusion hit the target.
        body_share = 0.80  # hook ~8%, conclusion ~12%
        body_target = int(target_words * body_share)
        raw_total = sum(int(section.get("estimated_words", 0)) for section in sections)
        if raw_total <= 0:
            per_section = max(60, body_target // len(sections))
            for section in sections:
                section["estimated_words"] = per_section
        else:
            scale = body_target / raw_total
            for section in sections:
                section["estimated_words"] = max(
                    60, int(round(int(section.get("estimated_words", 0)) * scale))
                )

        outline["sections"] = sections
        outline.setdefault("hook", {"text": "", "purpose": ""})
        if not outline["hook"].get("text"):
            outline["hook"]["text"] = (
                f"Open on the single most surprising consequence of {outline.get('working_title', 'this topic')}."
            )
        outline["hook"].setdefault("purpose", "Create a question the viewer needs answered")
        outline.setdefault("conclusion", {"text": ""})
        outline.setdefault("emotional_progression", [])
        if not outline["emotional_progression"]:
            outline["emotional_progression"] = ["curiosity", "surprise", "understanding", "resolve"]
        return outline


__all__ = ["ContentPlanner", "MAX_SECTIONS", "MIN_SECTIONS"]
