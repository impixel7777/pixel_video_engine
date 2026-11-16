"""Stage 7 - Audio Director.

Plans music, sound effects, pauses and mixing.  It does not generate audio.

Output is attached to the production plan *and* returned standalone so it can be
validated as its own document.
"""

from __future__ import annotations

from typing import Any

from engine.ai_stage import AIStage, scenes_timeline
from engine.logging_setup import get_logger

log = get_logger("audio")


class AudioDirector(AIStage):
    """Produces the ``audio_plan`` block of a production plan."""

    stage_name = "audio_director"

    def direct(
        self,
        *,
        scenes: list[dict[str, Any]],
        total_seconds: float,
        label: str = "long",
    ) -> dict[str, Any]:
        if not scenes:
            raise ValueError("Cannot plan audio without scenes")

        scene_ids = {str(scene.get("scene_id")) for scene in scenes}
        timeline = scenes_timeline(scenes)
        segments_target = max(1, round(len(scenes) / 4))
        max_segments = max(2, self.pipeline.music_segments_max)

        log.info(
            "Planning audio for %d scenes (about %d s), %d music segment(s) target",
            len(scenes),
            int(total_seconds),
            segments_target,
        )

        payload = self.run_json(
            prompt_name="audio_planning",
            schema_name="audio_direction",
            label=f"audio:{label}",
            temperature=0.6,
            max_tokens=min(1800, max(600, len(scenes) * 90)),
            PROFILE=self.profile_block(),
            SCENE_TIMELINE=self.fit(timeline, keep="head", label="scene timeline"),
            TOTAL_SECONDS=int(total_seconds),
            SCENES_PER_SEGMENT=max(2, segments_target),
            MAX_SEGMENTS=max_segments,
            MAX_SFX=min(self.pipeline.sfx_max, max(3, len(scenes) // 2)),
        )

        audio = self._postprocess(payload, scene_ids, segments_target, max_segments)
        log.info(
            "Audio plan: %d music segment(s), %d effect(s), %d pause(s)",
            len(audio["music"]["segments"]),
            len(audio["sfx"]),
            len(audio.get("pauses", [])),
        )
        return audio

    # -- post-processing ---------------------------------------------------- #
    def _postprocess(
        self,
        payload: dict[str, Any],
        scene_ids: set[str],
        segments_target: int,
        max_segments: int,
    ) -> dict[str, Any]:
        music = payload.setdefault("music", {})
        music["style"] = str(
            music.get("style") or self.profile.audio.get("music_style", "")
        )
        music["overall_energy_curve"] = str(music.get("overall_energy_curve", ""))

        segments = []
        for index, segment in enumerate(music.get("segments", [])[:max_segments], start=1):
            start_scene = str(segment.get("start_scene", "")).strip()
            if start_scene not in scene_ids:
                log.warning("Dropping music segment: unknown anchor scene %r", start_scene)
                continue
            end_scene = str(segment.get("end_scene", "")).strip()
            segment["segment_id"] = f"M{index:02d}"
            segment["start_scene"] = start_scene
            segment["end_scene"] = end_scene if end_scene in scene_ids else start_scene
            segment["intensity"] = round(
                max(0.0, min(1.0, float(segment.get("intensity", 0.4)))), 2
            )
            segment["mood"] = str(segment.get("mood", "neutral"))
            segment["description"] = str(segment.get("description", ""))
            segment.setdefault("transition", "crossfade")
            segment.setdefault("fade_in_seconds", 2.0)
            segment.setdefault("fade_out_seconds", 2.0)
            segments.append(segment)
        music["segments"] = segments

        if segments and len({segment["intensity"] for segment in segments}) == 1 and len(segments) > 1:
            log.warning("Music intensity is flat; the plan will be flagged by quality control")

        sfx = []
        for index, cue in enumerate(payload.get("sfx", [])[: self.pipeline.sfx_max], start=1):
            anchor = str(cue.get("anchor_scene", "")).strip()
            if anchor not in scene_ids:
                log.warning("Dropping sound effect: unknown anchor scene %r", anchor)
                continue
            cue["cue_id"] = f"S{index:02d}"
            cue["anchor_scene"] = anchor
            cue["gain_db"] = round(max(-30.0, min(-12.0, float(cue.get("gain_db", -20)))), 1)
            cue.setdefault("anchor_text", "")
            cue.setdefault("duration_seconds", 1.5)
            cue["description"] = str(cue.get("description", ""))
            cue["timing"] = str(cue.get("timing", "at cut"))
            sfx.append(cue)

        pauses = []
        for pause in payload.get("pauses", [])[:12]:
            anchor = str(pause.get("after_scene", "")).strip()
            if anchor not in scene_ids:
                continue
            pause["after_scene"] = anchor
            pause["seconds"] = round(max(0.0, min(4.0, float(pause.get("seconds", 0.6)))), 2)
            pauses.append(pause)

        mix = payload.setdefault("mix", {})
        mix["narration_priority"] = str(
            mix.get("narration_priority")
            or "Narration stays dominant at all times; music and effects duck under speech."
        )
        mix["music_bed_db"] = round(
            float(mix.get("music_bed_db", self.pipeline.music_bed_db)), 1
        )
        mix["duck_under_narration_db"] = round(
            float(mix.get("duck_under_narration_db", self.pipeline.narration_duck_db)), 1
        )
        mix["true_peak_db"] = round(float(mix.get("true_peak_db", -1.5)), 1)

        return {
            "music": music,
            "sfx": sfx,
            "pauses": pauses,
            "mix": mix,
            "status": "planned",
            "segments_target": segments_target,
        }

    # -- attachment --------------------------------------------------------- #
    @staticmethod
    def attach(plan: dict[str, Any], audio: dict[str, Any]) -> dict[str, Any]:
        """Put a finished audio plan into a production plan."""
        plan["audio_plan"] = audio
        return plan


__all__ = ["AudioDirector"]
