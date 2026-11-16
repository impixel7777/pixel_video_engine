"""Stage 4 - Long-form Script Writer.

Generates the long narration **section by section**.  With only a few thousand
tokens of context available, writing a 10-minute script in one call is not
possible; asking for it would silently truncate the ending.

The writer keeps continuity with:

* the outline (all sections, so it knows the destination)
* a rolling summary of what has already been written
* a short research digest that never repeats the full report
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from engine.ai_stage import AIStage, brief_digest, research_digest
from engine.file_manager import utc_now_iso
from engine.logging_setup import get_logger
from engine.narration import count_words, sanitize_narration

log = get_logger("writer")


@dataclass
class ScriptPart:
    """One unit of generation: hook, intro, a main section, or the ending."""

    part_id: str
    title: str
    goal: str
    key_points: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    target_words: int = 150
    emotional_note: str = ""
    next_label: str = ""
    written: str = ""
    summary: str = ""

    @property
    def actual_words(self) -> int:
        return count_words(self.written)


@dataclass
class ScriptResult:
    """Assembled long-form script plus per-part bookkeeping."""

    text: str
    parts: list[ScriptPart]
    total_words: int
    outline_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.1",
            "generated_at": utc_now_iso(),
            "outline_hash": self.outline_hash,
            "total_words": self.total_words,
            "parts": [
                {
                    "part_id": part.part_id,
                    "title": part.title,
                    "target_words": part.target_words,
                    "actual_words": part.actual_words,
                    "goal": part.goal,
                    "summary": part.summary,
                    "written": part.written,
                }
                for part in self.parts
            ],
        }


class ScriptWriter(AIStage):
    """Produces ``Long.txt`` (narration only) and ``Long_Sections.json``."""

    stage_name = "script_writer"

    # -- planning of parts -------------------------------------------------- #
    def build_parts(self, outline: dict[str, Any], target_words: int) -> list[ScriptPart]:
        hook = outline.get("hook", {}) or {}
        conclusion = outline.get("conclusion", {}) or {}
        sections = outline.get("sections", []) or []

        hook_words = max(50, int(target_words * 0.07))
        intro_words = max(60, int(target_words * 0.10))
        ending_words = max(110, int(target_words * 0.13))

        parts: list[ScriptPart] = [
            ScriptPart(
                part_id="H1",
                title="Hook",
                goal="Win the first fifteen seconds. Create a question the viewer cannot leave unanswered.",
                key_points=[
                    str(hook.get("text", "")),
                    f"Purpose: {hook.get('purpose', 'create curiosity')}",
                ],
                target_words=hook_words,
                emotional_note="curiosity with a hint of tension",
                next_label="the introduction",
            )
        ]

        introduction = outline.get("introduction") or {}
        if introduction or sections:
            parts.append(
                ScriptPart(
                    part_id="I1",
                    title="Introduction",
                    goal=(
                        str(introduction.get("goal"))
                        if introduction.get("goal")
                        else "Explain why this matters now and promise what the viewer will understand by the end."
                    ),
                    key_points=[
                        str(point) for point in (introduction.get("key_points") or [])
                    ]
                    or ["Why this topic matters", "What the viewer will understand by the end"],
                    examples=[],
                    target_words=intro_words,
                    emotional_note="orientation, then momentum",
                    next_label=f"section {sections[0].get('id', 'S1')}" if sections else "the ending",
                )
            )

        for index, section in enumerate(sections):
            next_label = (
                f"section {sections[index + 1].get('id', 'S?')}"
                if index + 1 < len(sections)
                else "the conclusion"
            )
            parts.append(
                ScriptPart(
                    part_id=str(section.get("id", f"S{index + 1}")),
                    title=str(section.get("title", f"Section {index + 1}")),
                    goal=str(section.get("goal", "")),
                    key_points=[str(point) for point in (section.get("key_points") or [])],
                    examples=[str(item) for item in (section.get("examples") or [])],
                    target_words=int(section.get("estimated_words", 180)),
                    emotional_note=str(section.get("emotional_note", "")),
                    next_label=next_label,
                )
            )

        parts.append(
            ScriptPart(
                part_id="C1",
                title="Conclusion",
                goal=(
                    str(conclusion.get("text"))
                    if conclusion.get("text")
                    else "Resolve the opening question and leave the viewer with a changed mental model."
                ),
                key_points=[
                    f"Call back to: {hook.get('text', 'the opening image')[:160]}",
                    "State the central message in one clean sentence",
                    "End with a genuine question or a clear reason to return",
                ],
                target_words=ending_words,
                emotional_note="resolution, then a forward look",
                next_label="(end of video)",
            )
        )
        return parts

    # -- generation --------------------------------------------------------- #
    def write(
        self,
        *,
        brief: dict[str, Any],
        research: dict[str, Any],
        outline: dict[str, Any],
        target_words: int,
        resume_state: dict[str, Any] | None = None,
        on_progress: Callable[[ScriptResult], None] | None = None,
    ) -> ScriptResult:
        parts = self.build_parts(outline, target_words)
        outline_hash = hashlib.sha256(
            json.dumps(outline, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        # Resume: keep any part that was already written in an earlier attempt.
        completed: dict[str, dict[str, Any]] = {}
        if resume_state:
            for entry in resume_state.get("parts", []):
                if entry.get("written"):
                    completed[str(entry.get("part_id"))] = entry

        brief_block = self.fit(brief_digest(brief), keep="head", label="brief")
        research_block = self.fit(
            research_digest(research), keep="tail", label="research digest"
        )

        summaries: list[str] = []

        for index, part in enumerate(parts):
            if part.part_id in completed:
                part.written = str(completed[part.part_id]["written"])
                part.summary = str(completed[part.part_id].get("summary", ""))
                summaries.append(part.summary)
                log.info(
                    "Reusing existing %s (%d words)", part.part_id, part.actual_words
                )
                continue

            next_part = parts[index + 1] if index + 1 < len(parts) else None
            rollup = self._rolling_summary(part, summaries)
            log.info(
                "Writing %s '%s' (%d words target)",
                part.part_id,
                part.title,
                part.target_words,
            )

            payload = self.run_json(
                prompt_name="long_script",
                schema_name="section_script",
                label=f"script:{part.part_id}",
                temperature=0.75,
                max_tokens=self._output_budget(part),
                PROFILE=self.profile_block(),
                CENTRAL_MESSAGE=str(brief.get("central_message", "")),
                RESEARCH_DIGEST=research_block,
                SECTION_ID=part.part_id,
                SECTION_TITLE=part.title,
                SECTION_GOAL=part.goal,
                KEY_POINTS=part.key_points or ["Cover the section goal"],
                EXAMPLES=part.examples or ["(none specified - use one you can support)"],
                TARGET_WORDS=part.target_words,
                EMOTIONAL_NOTE=part.emotional_note or "(follow the channel tone)",
                PREVIOUS_SUMMARY=rollup or "(this is the opening of the video)",
                NEXT_SECTION=(
                    f"{next_part.part_id} - {next_part.title}: {next_part.goal}"
                    if next_part
                    else "(the video ends here)"
                ),
                LANGUAGE=self.profile.language,
                BRIEF_BLOCK=brief_block,
            )

            narration = sanitize_narration(str(payload.get("narration", "")))
            if not narration:
                raise ValueError(f"Section {part.part_id} produced empty narration")

            part.written = narration
            part.summary = sanitize_narration(
                str(payload.get("summary_for_continuity", ""))
            ) or self._auto_summary(narration)
            summaries.append(part.summary)

            words = part.actual_words
            deviation = (words - part.target_words) / max(1, part.target_words)
            level = log.warning if abs(deviation) > 0.35 else log.info
            level(
                "%s written: %d words (target %d, %+d%%)",
                part.part_id,
                words,
                part.target_words,
                int(deviation * 100),
            )

            # Checkpoint immediately: a crash during the next section must not
            # throw away work that is already on disk.
            if on_progress is not None:
                on_progress(self._snapshot(parts, outline_hash))

        text = "\n\n".join(part.written.strip() for part in parts if part.written.strip())
        text = text.strip() + "\n"
        total = count_words(text)
        log.info("Long script assembled: %d words across %d parts", total, len(parts))
        result = ScriptResult(
            text=text, parts=parts, total_words=total, outline_hash=outline_hash
        )
        if on_progress is not None:
            on_progress(result)
        return result

    # -- helpers ------------------------------------------------------------ #
    def _snapshot(self, parts: list[ScriptPart], outline_hash: str) -> ScriptResult:
        text = "\n\n".join(part.written.strip() for part in parts if part.written.strip())
        text = (text.strip() + "\n") if text.strip() else ""
        return ScriptResult(
            text=text,
            parts=parts,
            total_words=count_words(text),
            outline_hash=outline_hash,
        )

    def _output_budget(self, part: ScriptPart) -> int:
        """Tokens needed for this part, capped by what the context allows."""
        wanted = int(part.target_words * 2.1) + 220  # prose + JSON envelope + slack
        return max(260, min(wanted, self.client.budget.prompt_tokens // 2))

    @staticmethod
    def _rolling_summary(current: ScriptPart, summaries: list[str]) -> str:
        if not summaries:
            return ""
        # Only the tail matters for continuity; older context is in the outline.
        tail = summaries[-2:]
        joined = " ".join(summary.strip() for summary in tail if summary.strip())
        return joined[-700:]

    @staticmethod
    def _auto_summary(narration: str) -> str:
        from engine.narration import split_sentences

        spans = split_sentences(narration)
        sentences = [" ".join(narration[start:end].split()) for start, end in spans]
        return " ".join(sentences[:2])[:400]


__all__ = ["ScriptPart", "ScriptResult", "ScriptWriter"]
