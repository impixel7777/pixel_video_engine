"""Stage 1 - Topic Analyzer.

Turns a bare topic into a ContentBrief: what the video is about, who it is for,
what the viewer should believe afterwards, and how the story should be built.

No narration is written at this stage.
"""

from __future__ import annotations

from typing import Any

from engine.ai_stage import AIStage
from engine.logging_setup import get_logger

log = get_logger("topic")


class TopicAnalyzer(AIStage):
    """Produces ``ContentBrief.json``."""

    stage_name = "topic_analyzer"

    def analyze(
        self,
        topic: str,
        *,
        duration_minutes: int,
        shorts_count: int,
        short_seconds: int,
    ) -> dict[str, Any]:
        log.info(
            "Analysing topic '%s' for %s (%d min long, %d shorts)",
            topic,
            self.profile.name,
            duration_minutes,
            shorts_count,
        )

        brief = self.run_json(
            prompt_name="topic_analysis",
            schema_name="content_brief",
            label="topic_analysis",
            temperature=0.6,
            max_tokens=min(1800, self.client.budget.prompt_tokens // 2),
            PROFILE=self.profile_block(),
            TOPIC=topic,
            DURATION=duration_minutes,
            SHORTS_COUNT=shorts_count,
            SHORT_SECONDS=short_seconds,
            LANGUAGE=self.profile.language,
        )

        brief = self._postprocess(brief, topic)
        counts = (
            f"{len(brief.get('angles', []))} angles, "
            f"{len(brief.get('key_questions', []))} questions, "
            f"{len(brief.get('misconceptions', []))} misconceptions"
        )
        log.info("ContentBrief ready: %s", counts)
        return brief

    # -- post-processing --------------------------------------------------- #
    def _postprocess(self, brief: dict[str, Any], topic: str) -> dict[str, Any]:
        brief["schema_version"] = "1.0"
        brief["topic"] = str(brief.get("topic") or topic).strip()

        angles = brief.get("angles", []) or []
        if angles and not any(angle.get("selected") for angle in angles):
            angles[0]["selected"] = True
        elif angles:
            # Keep exactly one selection so downstream stages are unambiguous.
            selected_seen = False
            for angle in angles:
                if angle.get("selected") and not selected_seen:
                    selected_seen = True
                elif angle.get("selected"):
                    angle["selected"] = False
        brief["angles"] = angles

        # Trim runaway lists so the brief stays a brief.
        brief["key_questions"] = (brief.get("key_questions") or [])[:8]
        brief["misconceptions"] = (brief.get("misconceptions") or [])[:6]
        brief["key_terms"] = (brief.get("key_terms") or [])[:10]

        scope = brief.setdefault("scope", {"include": [], "exclude": []})
        scope.setdefault("include", [])
        scope.setdefault("exclude", [])

        return brief


__all__ = ["TopicAnalyzer"]
