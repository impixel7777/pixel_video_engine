"""Audio/video timeline reconciliation.

The visual plan estimates scene durations from a configured words-per-minute
value.  Kokoro's real output rarely matches that estimate, so estimates are
never allowed to drive the final render once narration audio exists.

This module makes the measured TTS duration authoritative:

* scene boundaries are derived from the narration character anchors produced by
  the visual director (not from a new WPM guess);
* explicit ``AudioPlan.pauses`` are inserted into the WAV and their duration is
  attached to the scene they follow;
* ``estimated_duration_seconds`` becomes the reconciled duration used by the
  FFmpeg scene renderer, while the original estimate is preserved under
  ``timing.estimated_total_seconds``.
"""

from __future__ import annotations

from typing import Any, Iterable


def _scene_offsets(scenes: list[dict[str, Any]], total_chars: int) -> list[tuple[float, float]]:
    """Return fractional (start, end) offsets for each scene in plan order."""
    if not scenes:
        return []

    spans: list[tuple[float, float]] = []
    valid = [
        scene
        for scene in scenes
        if isinstance(scene.get("narration_start_char"), (int, float))
        and isinstance(scene.get("narration_end_char"), (int, float))
        and float(scene["narration_end_char"]) > float(scene["narration_start_char"])
    ]
    if len(valid) == len(scenes) and total_chars > 0:
        starts = sorted(
            max(0.0, min(1.0, float(scene["narration_start_char"]) / total_chars))
            for scene in scenes
        )
        # Use each scene's start as the boundary; the last scene ends at 1.0.
        for index, scene in enumerate(scenes):
            start = max(0.0, min(1.0, float(scene["narration_start_char"]) / total_chars))
            end = starts[index + 1] if index + 1 < len(starts) else 1.0
            if end <= start:
                end = min(1.0, start + 1e-6)
            spans.append((start, end))
        return spans

    # Fallback: word-derived durations, never dropping a scene.
    weights = [
        max(1.0, float(scene.get("estimated_duration_seconds") or 0))
        for scene in scenes
    ]
    total = sum(weights)
    cursor = 0.0
    for weight in weights:
        share = weight / total
        spans.append((cursor, min(1.0, cursor + share)))
        cursor += share
    if spans:
        spans[-1] = (spans[-1][0], 1.0)
    return spans


def narration_chars(plan: dict[str, Any]) -> int:
    coverage = plan.get("coverage") or {}
    value = int(coverage.get("narration_chars") or 0)
    if value > 0:
        return value
    scenes = plan.get("scenes", []) or []
    return max(
        (int(scene.get("narration_end_char") or 0) for scene in scenes),
        default=0,
    )


def scene_end_times(plan: dict[str, Any], narration_seconds: float) -> dict[str, float]:
    """Map scene ids to their end time in a plan with no inserted pauses."""
    scenes = plan.get("scenes", []) or []
    chars = narration_chars(plan)
    spans = _scene_offsets(scenes, chars)
    boundaries: dict[str, float] = {}
    for scene, (_start, end) in zip(scenes, spans):
        boundaries[str(scene.get("scene_id"))] = round(end * narration_seconds, 3)
    return boundaries


def pause_insertion_points(
    plan: dict[str, Any],
    narration_seconds: float,
    pauses: Iterable[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Convert ``pauses`` anchored to scenes into absolute WAV offsets."""
    ends = scene_end_times(plan, narration_seconds)
    points: list[tuple[float, float]] = []
    for pause in pauses or []:
        anchor = str(pause.get("after_scene", ""))
        seconds = float(pause.get("seconds", 0) or 0)
        if anchor in ends and seconds > 0:
            points.append((ends[anchor], seconds))
    return sorted(points)


def reconcile_plan(
    plan: dict[str, Any],
    actual_narration_seconds: float,
    *,
    pauses: Iterable[dict[str, Any]] = (),
    source: str = "tts",
) -> dict[str, Any]:
    """Rewrite scene durations so they sum to the measured narration duration."""
    if actual_narration_seconds <= 0:
        return plan

    scenes = plan.get("scenes", []) or []
    if not scenes:
        return plan

    pause_list = [pause for pause in (pauses or []) if float(pause.get("seconds", 0) or 0) > 0]
    pause_total = sum(float(pause.get("seconds", 0) or 0) for pause in pause_list)
    raw_seconds = max(0.1, actual_narration_seconds - pause_total)

    chars = narration_chars(plan)
    spans = _scene_offsets(scenes, chars)
    original_total = round(
        sum(float(scene.get("estimated_duration_seconds") or 0) for scene in scenes), 3
    )

    durations = [
        max(0.05, (end - start) * raw_seconds) for start, end in spans
    ]
    # Correct rounding drift on the last scene.
    drift = raw_seconds - sum(durations)
    durations[-1] = max(0.05, durations[-1] + drift)

    pauses_by_scene: dict[str, float] = {}
    for pause in pause_list:
        anchor = str(pause.get("after_scene", ""))
        pauses_by_scene[anchor] = pauses_by_scene.get(anchor, 0.0) + float(
            pause.get("seconds", 0) or 0
        )

    for scene, duration in zip(scenes, durations):
        scene_id = str(scene.get("scene_id"))
        pause_here = pauses_by_scene.get(scene_id, 0.0)
        scene["estimated_duration_seconds"] = round(duration + pause_here, 3)
        timing = dict(scene.get("timing") or {})
        timing.update(
            {
                "source": source,
                "speech_seconds": round(duration, 3),
                "pause_seconds": round(pause_here, 3),
                "actual_seconds": round(duration + pause_here, 3),
            }
        )
        scene["timing"] = timing

    reconciled_total = round(
        sum(float(scene.get("estimated_duration_seconds") or 0) for scene in scenes), 3
    )
    plan["timing"] = {
        "source": source,
        "estimated_total_seconds": original_total,
        "actual_narration_seconds": round(actual_narration_seconds, 3),
        "inserted_pause_seconds": round(pause_total, 3),
        "reconciled_total_seconds": reconciled_total,
    }
    project = plan.setdefault("project", {})
    project["actual_narration_seconds"] = round(actual_narration_seconds, 3)
    coverage = plan.setdefault("coverage", {})
    coverage["estimated_total_duration_seconds"] = reconciled_total
    return plan


__all__ = [
    "narration_chars",
    "pause_insertion_points",
    "reconcile_plan",
    "scene_end_times",
]
