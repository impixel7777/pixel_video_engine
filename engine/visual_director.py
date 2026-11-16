"""Stage 6 - Visual Director.

Converts narration into a machine-readable storyboard.

Key design decisions
--------------------
* **Python owns the anchors.**  Scene IDs, narration text, character offsets and
  durations are computed deterministically from ``Long.txt`` / ``Short.txt``.
  The model only supplies creative direction.  A model can therefore never move,
  shorten or invent narration.
* **Grouped scenes.**  Narration is grouped into blocks (several sentences each)
  before any model call, so the result is a storyboard, not one image per
  sentence.
* **Honest gaps.**  If the model omits a block, the block is retried once.  If it
  still fails, a profile-correct fallback scene is created **and flagged**, so
  the gap is visible in the quality report instead of silently disappearing.
* **Variety enforcement.**  Near-identical image prompts are detected
  deterministically and repaired once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from engine.ai_stage import AIStage, brief_digest
from engine.logging_setup import get_logger
from engine.narration import (
    NarrationBlock,
    count_words,
    coverage_ratio,
    split_into_blocks,
)

log = get_logger("visual")

FALLBACK_PURPOSE = "Cover the narration with a representative establishing image"

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class ScenePlan:
    """A scene with deterministic anchors and model-supplied direction."""

    scene_id: str
    block: NarrationBlock
    direction: dict[str, Any]

    def to_dict(self, *, words_per_minute: int) -> dict[str, Any]:
        narration = self.block.text
        excerpt = " ".join(narration.split())
        if len(excerpt) > 180:
            excerpt = excerpt[:180].rstrip() + "..."
        duration = max(2.0, round(count_words(narration) / words_per_minute * 60, 2))
        scene: dict[str, Any] = {
            "scene_id": self.scene_id,
            "block_id": self.block.block_id,
            "origin": "narration",
            "narration": narration,
            "narration_excerpt": excerpt,
            "narration_start_char": self.block.start_char,
            "narration_end_char": self.block.end_char,
            "purpose": str(self.direction.get("purpose", FALLBACK_PURPOSE)),
            "visual_description": str(
                self.direction.get("visual_description", "(no direction supplied)")
            ),
            "image_prompt": str(self.direction.get("image_prompt", "")),
            "motion": str(self.direction.get("motion", "slow push in")),
            "camera_movement": str(self.direction.get("camera_movement", "static")),
            "transition_out": str(self.direction.get("transition_out", "cut")),
            "estimated_duration_seconds": duration,
            "asset_type": str(self.direction.get("asset_type", "ai_image")),
            "visual_style": str(self.direction.get("visual_style", "")),
        }
        optional = {
            "negative_prompt": "negative_prompt",
            "video_prompt": "video_prompt",
            "on_screen_text": "on_screen_text",
        }
        for key, source in optional.items():
            value = self.direction.get(source)
            if value:
                scene[key] = str(value)
        flags = self.direction.get("uncertainty_flags") or []
        if flags:
            scene["uncertainty_flags"] = [str(flag) for flag in flags]
        if self.direction.get("_fallback"):
            scene["uncertainty_flags"] = list(scene.get("uncertainty_flags", [])) + [
                "Direction was auto-generated because the model omitted this block"
            ]
        return scene


class VisualDirector(AIStage):
    """Produces ``Long_Production.json`` and ``Shorts_Production.json``."""

    stage_name = "visual_director"

    # -- public API --------------------------------------------------------- #
    def style_summary(self) -> str:
        visuals = self.profile.visuals
        colours = ", ".join(visuals.get("primary_colors", [])) or "brand palette"
        parts = [
            str(visuals.get("style", "cinematic")),
            f"{colours} palette",
            str(visuals.get("visual_preference", "high quality")),
            str(visuals.get("motion_style", "controlled motion")),
        ]
        return "; ".join(part for part in parts if part)

    def direct_long(
        self,
        *,
        narration: str,
        brief: dict[str, Any],
        target_seconds: float,
    ) -> dict[str, Any]:
        blocks = split_into_blocks(
            narration,
            target_sentences=self.pipeline.scene_target_sentences,
            target_words=self.pipeline.scene_target_words,
            prefix="B",
            max_blocks=self.pipeline.max_scenes_long,
        )
        log.info("Directing %d narration block(s) for the long video", len(blocks))

        scenes = self._direct_blocks(blocks, prefix="L", start_index=1, brief=brief)
        scenes_dicts = [scene.to_dict(words_per_minute=self.pipeline.words_per_minute) for scene in scenes]
        scenes_dicts = self._enforce_variety(scenes_dicts, brief=brief)

        plan = self._assemble(
            scenes=scenes_dicts,
            narration=narration,
            blocks=blocks,
            fmt="long",
            aspect=self.profile.long_aspect,
            target_seconds=target_seconds,
        )
        log.info(
            "Long storyboard: %d scenes, %.0f%% narration coverage",
            len(scenes_dicts),
            plan["coverage"]["coverage_ratio"] * 100,
        )
        return plan

    def direct_shorts(
        self,
        *,
        shorts: list[dict[str, Any]],
        brief: dict[str, Any],
    ) -> dict[str, Any]:
        plans: list[dict[str, Any]] = []
        scene_counter = 1

        for short in shorts:
            narration = str(short.get("narration", ""))
            blocks = split_into_blocks(
                narration,
                target_sentences=2,
                target_words=max(24, self.pipeline.scene_target_words // 2),
                prefix="B",
                max_blocks=self.pipeline.max_scenes_per_short,
            )
            log.info(
                "Directing Short %s: %d block(s)",
                short.get("index"),
                len(blocks),
            )
            scenes = self._direct_blocks(
                blocks,
                prefix="S",
                start_index=scene_counter,
                brief=brief,
            )
            scene_counter += len(scenes)
            scene_dicts = [
                scene.to_dict(words_per_minute=self.pipeline.words_per_minute)
                for scene in scenes
            ]
            scene_dicts = self._enforce_variety(scene_dicts, brief=brief)

            plans.append(
                {
                    "short_id": f"{int(short.get('index', len(plans) + 1)):02d}",
                    "title": str(short.get("title", "")),
                    "target_duration_seconds": float(
                        self.profile.short_duration_seconds
                    ),
                    "narration_word_count": count_words(narration),
                    "central_idea": str(short.get("central_idea", "")),
                    "scenes": scene_dicts,
                    "audio_plan": self._stub_audio_plan(),
                }
            )

        return {
            "schema_version": "1.0",
            "generated_at": "",
            "project": {
                "profile_id": self.profile.id,
                "topic": str(brief.get("topic", "")),
                "slug": "",
                "aspect_ratio": self.profile.short_aspect,
                "shorts_count": len(plans),
            },
            "style_summary": self.style_summary(),
            "shorts": plans,
            "notes": [],
        }

    # -- batch direction ---------------------------------------------------- #
    def _direct_blocks(
        self,
        blocks: list[NarrationBlock],
        *,
        prefix: str,
        start_index: int,
        brief: dict[str, Any],
    ) -> list[ScenePlan]:
        if not blocks:
            return []

        batch_size = max(1, self.pipeline.visual_batch_size)
        directions: dict[str, dict[str, Any]] = {}

        for start in range(0, len(blocks), batch_size):
            batch = blocks[start : start + batch_size]
            log.info(
                "Visual batch %d/%d (%d block(s))",
                start // batch_size + 1,
                (len(blocks) + batch_size - 1) // batch_size,
                len(batch),
            )
            payload = self._call_batch(batch, brief=brief)
            for item in payload.get("scenes", []):
                block_id = str(item.get("block_id", "")).strip()
                if block_id:
                    directions[block_id] = item

        missing = [block for block in blocks if block.block_id not in directions]
        if missing:
            log.warning(
                "Model omitted %d block(s); retrying them individually", len(missing)
            )
            for block in missing:
                payload = self._call_batch([block], brief=brief)
                for item in payload.get("scenes", []):
                    if str(item.get("block_id", "")).strip() == block.block_id:
                        directions[block.block_id] = item
                        break

        scenes: list[ScenePlan] = []
        for offset, block in enumerate(blocks):
            direction = directions.get(block.block_id)
            if not direction:
                log.error(
                    "No visual direction produced for %s; using flagged fallback",
                    block.block_id,
                )
                direction = self._fallback_direction(block)
            scenes.append(
                ScenePlan(
                    scene_id=f"{prefix}{start_index + offset:03d}",
                    block=block,
                    direction=direction,
                )
            )
        return scenes

    def _call_batch(
        self, batch: list[NarrationBlock], *, brief: dict[str, Any]
    ) -> dict[str, Any]:
        rendered_blocks = "\n\n".join(
            f"[{block.block_id}] {block.text}" for block in batch
        )
        return self.run_json(
            prompt_name="visual_planning",
            schema_name="scene_direction",
            label=f"visual:{batch[0].block_id}",
            temperature=0.85,
            max_tokens=min(1800, max(600, len(batch) * 380)),
            PROFILE=self.profile_block(),
            STYLE_SUMMARY=self.style_summary(),
            BLOCKS=self.fit(rendered_blocks, keep="head", label="narration blocks"),
            BLOCK_COUNT=len(batch),
        )

    # -- variety ------------------------------------------------------------ #
    @staticmethod
    def _concept_signature(prompt: str) -> set[str]:
        words = [
            word
            for word in _WORD_RE.findall(prompt.lower())
            if len(word) > 3
            and word
            not in {
                "with", "from", "that", "this", "into", "over", "under", "very",
                "highly", "detail", "detailed", "cinematic", "image", "shot",
                "scene", "light", "lighting", "background", "foreground",
            }
        ]
        return set(words[:24])

    def _duplicate_pairs(self, scenes: list[dict[str, Any]], threshold: float = 0.65) -> list[int]:
        signatures = [self._concept_signature(str(scene.get("image_prompt", ""))) for scene in scenes]
        duplicates: list[int] = []
        for index in range(1, len(signatures)):
            previous = signatures[index - 1]
            if not previous or not signatures[index]:
                continue
            overlap = len(previous & signatures[index]) / max(1, len(previous | signatures[index]))
            if overlap >= threshold:
                duplicates.append(index)
        return duplicates

    def _enforce_variety(
        self, scenes: list[dict[str, Any]], *, brief: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not self.pipeline.variety_pass or len(scenes) < 2:
            return scenes

        rounds = max(0, self.pipeline.max_variety_repair_rounds)
        for round_index in range(rounds):
            duplicates = self._duplicate_pairs(scenes)
            if not duplicates:
                break
            log.warning(
                "Visual variety pass %d: %d repeated concept(s)",
                round_index + 1,
                len(duplicates),
            )
            for index in duplicates:
                scene = scenes[index]
                try:
                    payload = self.run_json(
                        prompt_name="visual_planning",
                        schema_name="scene_direction",
                        label=f"variety:{scene.get('scene_id')}",
                        temperature=0.95,
                        max_tokens=700,
                        PROFILE=self.profile_block(),
                        STYLE_SUMMARY=self.style_summary(),
                        BLOCKS=f"[{scene.get('block_id')}] {scene.get('narration', '')}",
                        BLOCK_COUNT=1,
                    )
                    replacement = (payload.get("scenes") or [{}])[0]
                except Exception as exc:  # noqa: BLE001 - variety is best effort
                    log.warning("Variety repair failed for %s: %s", scene.get("scene_id"), exc)
                    break

                if not replacement.get("image_prompt"):
                    break
                for key in (
                    "purpose",
                    "visual_description",
                    "image_prompt",
                    "negative_prompt",
                    "video_prompt",
                    "motion",
                    "camera_movement",
                    "transition_out",
                    "asset_type",
                    "visual_style",
                    "on_screen_text",
                ):
                    if replacement.get(key):
                        scene[key] = replacement[key]
                scene.setdefault("uncertainty_flags", []).append(
                    "Visual was re-directed to avoid repeating an earlier scene"
                )
        return scenes

    # -- assembly ----------------------------------------------------------- #
    def _assemble(
        self,
        *,
        scenes: list[dict[str, Any]],
        narration: str,
        blocks: list[NarrationBlock],
        fmt: str,
        aspect: str,
        target_seconds: float,
    ) -> dict[str, Any]:
        from engine.file_manager import utc_now_iso

        ratio = coverage_ratio(narration, blocks)
        total_duration = round(
            sum(float(scene["estimated_duration_seconds"]) for scene in scenes), 2
        )
        concepts = {
            tuple(sorted(self._concept_signature(str(scene.get("image_prompt", "")))))
            for scene in scenes
        }

        return {
            "schema_version": "1.0",
            "generated_at": utc_now_iso(),
            "project": {
                "profile_id": self.profile.id,
                "topic": "",
                "slug": "",
                "format": fmt,
                "aspect_ratio": aspect,
                "target_duration_seconds": round(target_seconds, 2),
                "narration_word_count": count_words(narration),
            },
            "style_summary": self.style_summary(),
            "scenes": scenes,
            "audio_plan": self._stub_audio_plan(),
            "coverage": {
                "narration_chars": len(narration),
                "covered_chars": sum(len(block.text) for block in blocks),
                "coverage_ratio": ratio,
                "narration_word_count": count_words(narration),
                "scene_count": len(scenes),
                "distinct_visual_concepts": len(concepts),
                "estimated_total_duration_seconds": total_duration,
            },
            "notes": [],
        }

    def _stub_audio_plan(self) -> dict[str, Any]:
        return {
            "music": {
                "style": str(self.profile.audio.get("music_style", "")),
                "overall_energy_curve": "",
                "segments": [],
            },
            "sfx": [],
            "pauses": [],
            "mix": {
                "narration_priority": str(
                    self.profile.audio.get("mixing_style", "Narration forward")
                ),
                "music_bed_db": self.pipeline.music_bed_db,
                "duck_under_narration_db": self.pipeline.narration_duck_db,
                "true_peak_db": -1.5,
            },
            "status": "pending_audio_direction",
        }

    # -- fallback ----------------------------------------------------------- #
    def _fallback_direction(self, block: NarrationBlock) -> dict[str, Any]:
        visuals = self.profile.visuals
        colours = ", ".join(visuals.get("primary_colors", [])) or "brand palette"
        excerpt = " ".join(block.text.split())[:120]
        return {
            "block_id": block.block_id,
            "purpose": FALLBACK_PURPOSE,
            "visual_description": f"Representative frame for: {excerpt}",
            "image_prompt": (
                f"{visuals.get('style', 'cinematic')} illustrative frame representing: "
                f"{excerpt}. {colours} palette, {visuals.get('visual_preference', 'high quality')}, "
                f"{visuals.get('lighting', 'directional lighting')}, no text."
            ),
            "motion": visuals.get("motion_style", "slow push in"),
            "camera_movement": "static",
            "transition_out": "cut",
            "asset_type": "ai_image",
            "visual_style": str(visuals.get("style", "cinematic")),
            "_fallback": True,
        }


__all__ = ["ScenePlan", "VisualDirector"]
