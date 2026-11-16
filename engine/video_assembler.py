"""Video assembler.

Executes the deterministic FFmpeg plan produced by :mod:`engine.ffmpeg_builder`.

Guarantees
----------
* Refuses to render when any required asset is missing, and says exactly which.
* Never invents placeholder footage for a production render.
* Writes a machine-readable render report next to the output.
* Supports 16:9 long-form and 9:16 Shorts from the same code path.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import PipelineSettings
from engine import ffmpeg_builder as fb
from engine.file_manager import ProjectPaths, atomic_write_json, read_json, utc_now_iso
from engine.logging_setup import get_logger
from engine.timeline import reconcile_plan

log = get_logger("video")


@dataclass
class RenderReport:
    ok: bool
    output: Path | None = None
    missing: list[str] = field(default_factory=list)
    clips: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    mode: str = "production"

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": utc_now_iso(),
            "ok": self.ok,
            "mode": self.mode,
            "output": str(self.output) if self.output else None,
            "missing": self.missing,
            "clips": self.clips,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


class VideoAssembler:
    """Renders production plans into MP4 files."""

    def __init__(self, settings: PipelineSettings, paths: ProjectPaths) -> None:
        self.settings = settings
        self.paths = paths
        #: ``--render`` is an explicit operator instruction; it overrides the
        #: config's default-off render switch for this invocation only.
        self.allow_render = False

    # -- tools -------------------------------------------------------------- #
    def render_settings(self, fmt: str) -> fb.RenderSettings:
        width, height = (
            self.settings.long_size if fmt == "long" else self.settings.short_size
        )
        fps = self.settings.long_fps if fmt == "long" else self.settings.short_fps
        return fb.RenderSettings(
            ffmpeg=fb.require_tool(self.settings.ffmpeg_path, "ffmpeg"),
            ffprobe=fb.require_tool(self.settings.ffprobe_path, "ffprobe"),
            width=width,
            height=height,
            fps=fps,
            font_path=self.settings.video_font_path,
        )

    def tools_available(self) -> tuple[bool, str]:
        try:
            settings = self.render_settings("long")
        except fb.FFmpegNotFound as exc:
            return False, str(exc)
        return True, f"ffmpeg {settings.ffmpeg} | ffprobe {settings.ffprobe}"

    # -- public API --------------------------------------------------------- #
    def render_long(
        self, plan: dict[str, Any], *, narration: Path | None = None
    ) -> RenderReport:
        narration = narration or self.paths.tts_long_audio
        return self._render(plan, fmt="long", narration=narration)

    def render_short(
        self,
        short_plan: dict[str, Any],
        *,
        short_id: str,
        narration: Path | None = None,
    ) -> RenderReport:
        narration = narration or self.paths.tts_short_audio(int(short_id))
        return self._render(
            short_plan, fmt="short", narration=narration, label=f"short_{short_id}"
        )

    # -- core --------------------------------------------------------------- #
    def _resolve_asset(self, relative: str | None) -> Path | None:
        if not relative:
            return None
        path = Path(relative)
        if not path.is_absolute():
            path = self.paths.root / path
        return path if path.is_file() else None

    def _render(
        self,
        plan: dict[str, Any],
        *,
        fmt: str,
        narration: Path,
        label: str | None = None,
    ) -> RenderReport:
        label = label or fmt
        try:
            settings = self.render_settings(fmt)
        except fb.FFmpegNotFound as exc:
            return RenderReport(ok=False, errors=[str(exc)])

        if not self.settings.render_enabled and not self.allow_render:
            report = RenderReport(
                ok=False,
                errors=[
                    "Rendering is disabled (video.render=false in "
                    "config/pipeline_settings.json). Run with --render to override."
                ],
            )
            atomic_write_json(
                self.paths.renders / f"render_{label}.json", report.to_dict()
            )
            return report

        # The measured narration duration is authoritative.  If the plan was not
        # reconciled by the TTS stage (for example an operator supplied their own
        # WAV), reconcile a private copy before any clip is rendered.
        render_plan = copy.deepcopy(plan)
        duration = 0.0
        if narration.is_file():
            try:
                duration = fb.probe_duration(narration, settings.ffprobe)
            except RuntimeError as exc:
                report = RenderReport(ok=False, errors=[str(exc)])
                atomic_write_json(
                    self.paths.renders / f"render_{label}.json", report.to_dict()
                )
                return report
            timing = render_plan.get("timing") or {}
            reconciled = float(timing.get("actual_narration_seconds") or 0)
            if duration > 0 and abs(reconciled - duration) > 0.05:
                log.warning(
                    "%s plan timing (%.2fs) does not match narration (%.2fs); "
                    "reconciling scene durations before rendering",
                    label,
                    reconciled,
                    duration,
                )
                reconcile_plan(render_plan, duration)

        ready, missing = fb.validate_render_inputs(
            render_plan, resolve_asset=self._resolve_asset, narration_path=narration
        )
        if missing:
            log.error("Refusing to render %s: %d missing input(s)", label, len(missing))
            report = RenderReport(ok=False, missing=missing)
            atomic_write_json(
                self.paths.renders / f"render_{label}.json", report.to_dict()
            )
            return report

        render_dir = self.paths.renders / label
        clip_dir = render_dir / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)

        log.info("Rendering %d scene clip(s) for %s", len(ready), label)
        clip_paths: list[Path] = []
        errors: list[str] = []

        for index, (scene, asset) in enumerate(ready, start=1):
            scene_id = str(scene.get("scene_id", f"scene{index}"))
            clip_path = clip_dir / f"{scene_id}.mp4"
            text_path: Path | None = None
            if asset is None:
                overlay_text = (
                    scene.get("on_screen_text")
                    or scene.get("text")
                    or (scene.get("asset") or {}).get("on_screen_text")
                    or (scene.get("asset") or {}).get("text")
                    or ""
                )
                text_path = clip_dir / f"{scene_id}.txt"
                wrapped = textwrap.wrap(
                    str(overlay_text).strip(),
                    width=28 if fmt == "short" else 46,
                ) or [str(overlay_text).strip()]
                text_path.write_text("\n".join(wrapped), encoding="utf-8", newline="\n")
            try:
                args = fb.scene_clip_args(
                    asset=asset,
                    scene=scene,
                    settings=settings,
                    out_path=clip_path,
                    text_path=text_path,
                )
            except (ValueError, FileNotFoundError) as exc:
                errors.append(f"{scene_id}: {exc}")
                log.error("  [%d/%d] %s failed: %s", index, len(ready), scene_id, exc)
                continue
            outcome = self._run(args, f"clip {scene_id}")
            if outcome is None:
                clip_paths.append(clip_path)
                log.info("  [%d/%d] %s ok", index, len(ready), scene_id)
            else:
                errors.append(outcome)
                log.error("  [%d/%d] %s failed: %s", index, len(ready), scene_id, outcome)

        if errors:
            report = RenderReport(
                ok=False, clips=[str(p) for p in clip_paths], errors=errors
            )
            atomic_write_json(self.paths.renders / f"render_{label}.json", report.to_dict())
            return report

        list_path = fb.write_concat_list(clip_paths, render_dir / "clips.txt")
        silent_path = render_dir / "video_silent.mp4"
        outcome = self._run(
            fb.concat_args(list_path=list_path, out_path=silent_path, settings=settings),
            "concat",
        )
        if outcome is not None:
            report = RenderReport(ok=False, errors=[outcome], clips=[str(p) for p in clip_paths])
            atomic_write_json(self.paths.renders / f"render_{label}.json", report.to_dict())
            return report

        duration = duration or fb.probe_duration(narration, settings.ffprobe)
        audio_plan = render_plan.get("audio_plan") or {}
        music_path = self._first_asset(self.paths.music)
        sfx_pairs = self._resolve_sfx(audio_plan, render_plan)

        final_path = render_dir / f"final_{label}.mp4"
        mix = audio_plan.get("mix", {}) or {}
        segments = (audio_plan.get("music") or {}).get("segments", []) or []
        fade_in = float(
            (segments[0].get("fade_in_seconds") if segments else None)
            or self.settings.music_fade_in_seconds
        )
        fade_out = float(
            (segments[-1].get("fade_out_seconds") if segments else None)
            or self.settings.music_fade_out_seconds
        )
        outcome = self._run(
            fb.mux_args(
                video_path=silent_path,
                narration_path=narration,
                out_path=final_path,
                settings=settings,
                music_path=music_path,
                sfx=sfx_pairs,
                music_gain_db=float(mix.get("music_bed_db", self.settings.music_bed_db)),
                duck_db=float(
                    mix.get("duck_under_narration_db", self.settings.narration_duck_db)
                ),
                true_peak_db=float(mix.get("true_peak_db", -1.5)),
                fade_in=fade_in,
                fade_out=fade_out,
                total_duration=duration,
            ),
            "mux",
        )
        if outcome is not None:
            report = RenderReport(ok=False, errors=[outcome], clips=[str(p) for p in clip_paths])
            atomic_write_json(self.paths.renders / f"render_{label}.json", report.to_dict())
            return report

        report = RenderReport(
            ok=True,
            output=final_path,
            clips=[str(path) for path in clip_paths],
            duration_seconds=round(duration, 2),
        )
        atomic_write_json(self.paths.renders / f"render_{label}.json", report.to_dict())
        log.info("Rendered %s -> %s (%.1fs)", label, final_path, duration)
        return report

    # -- helpers ------------------------------------------------------------ #
    def _run(self, args: list[str], label: str) -> str | None:
        """Run ffmpeg; return an error string or ``None`` on success."""
        log.debug("ffmpeg %s: %s", label, " ".join(args))
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            return f"{label}: could not start ffmpeg ({exc})"
        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-6:]
            return f"{label}: ffmpeg exited {result.returncode}: " + " | ".join(tail)
        return None

    @staticmethod
    def _first_asset(directory: Path) -> Path | None:
        if not directory.is_dir():
            return None
        for pattern in ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg"):
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _resolve_sfx(
        self, audio_plan: dict[str, Any], plan: dict[str, Any]
    ) -> list[tuple[Path, float, float]]:
        """Map SFX cues onto real files, start times and per-cue gain."""
        cues = audio_plan.get("sfx", []) or []
        if not cues:
            return []

        starts: dict[str, float] = {}
        cursor = 0.0
        for scene in plan.get("scenes", []) or []:
            starts[str(scene.get("scene_id"))] = cursor
            cursor += float(scene.get("estimated_duration_seconds", 0))

        triples: list[tuple[Path, float, float]] = []
        for cue in cues:
            path = self.paths.sounds / f"{cue.get('cue_id')}.wav"
            if not path.is_file():
                continue
            anchor = str(cue.get("anchor_scene", ""))
            gain = float(cue.get("gain_db", 0.0) or 0.0)
            triples.append((path, starts.get(anchor, 0.0), gain))
        return triples

    # -- self test ---------------------------------------------------------- #
    def smoke_test(self) -> dict[str, Any]:
        """Prove the render path works using FFmpeg test patterns.

        This never touches production assets and is clearly labelled.  It is the
        only way to verify clip -> concat -> mux without a generation provider.
        """
        try:
            settings = self.render_settings("long")
        except fb.FFmpegNotFound as exc:
            return {"ok": False, "error": str(exc)}

        work = self.paths.renders / "smoke"
        work.mkdir(parents=True, exist_ok=True)
        patterns = ["testsrc2=size=1280x720:rate=30", "smptebars=size=1280x720:rate=30"]
        clips: list[Path] = []

        for index, pattern in enumerate(patterns, start=1):
            clip = work / f"pattern_{index}.mp4"
            args = [
                str(settings.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", pattern,
                "-t", "2",
                "-vf", settings.dimension_filter(),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p", "-r", str(settings.fps),
                str(clip),
            ]
            error = self._run(args, f"smoke pattern {index}")
            if error:
                return {"ok": False, "error": error}
            clips.append(clip)

        tone = work / "tone.wav"
        error = self._run(
            [
                str(settings.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                "-ar", "48000", "-ac", "1", str(tone),
            ],
            "smoke tone",
        )
        if error:
            return {"ok": False, "error": error}

        list_path = fb.write_concat_list(clips, work / "clips.txt")
        silent = work / "silent.mp4"
        error = self._run(
            fb.concat_args(list_path=list_path, out_path=silent, settings=settings),
            "smoke concat",
        )
        if error:
            return {"ok": False, "error": error}

        output = work / "smoke_final.mp4"
        error = self._run(
            fb.mux_args(
                video_path=silent,
                narration_path=tone,
                out_path=output,
                settings=settings,
                total_duration=4.0,
            ),
            "smoke mux",
        )
        if error:
            return {"ok": False, "error": error}

        duration = fb.probe_duration(output, settings.ffprobe)
        return {
            "ok": output.is_file() and duration > 0,
            "output": str(output),
            "duration_seconds": round(duration, 2),
            "mode": "test_pattern",
            "note": "Generated from FFmpeg test sources, not from production assets.",
        }

    # -- reporting ---------------------------------------------------------- #
    def describe(self) -> dict[str, Any]:
        reports = sorted(self.paths.renders.glob("render_*.json"))
        return {
            "render_enabled": self.settings.render_enabled,
            "renders": [read_json(path, default={}) for path in reports],
            "tools": self.tools_available()[1],
        }


__all__ = ["RenderReport", "VideoAssembler"]
