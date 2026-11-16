"""FFmpeg argument builder.

The AI never writes FFmpeg commands.  It produces structured production plans;
this module translates them into deterministic argument lists.

Rendering strategy: **one clip per scene, then concat, then mux**.  A single
giant filter graph would be faster but impossible to debug and fragile when one
scene is malformed.  Per-scene rendering means a failure names the exact scene.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from engine.logging_setup import get_logger

log = get_logger("ffmpeg")


class FFmpegNotFound(FileNotFoundError):
    pass


class MissingAssetsError(RuntimeError):
    """Raised instead of rendering a video with holes in it."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            "Cannot render: "
            + ", ".join(self.missing[:12])
            + (f" (+{len(self.missing) - 12} more)" if len(self.missing) > 12 else "")
        )


@dataclass(frozen=True)
class RenderSettings:
    ffmpeg: Path
    ffprobe: Path
    width: int
    height: int
    fps: int
    crf: int = 20
    preset: str = "medium"
    audio_bitrate: str = "192k"
    font_path: Path | None = None

    def dimension_filter(self) -> str:
        """Scale to fill, then crop to the exact frame - never letterbox."""
        return (
            f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase,"
            f"crop={self.width}:{self.height},setsar=1"
        )


# --------------------------------------------------------------------------- #
# Tool discovery / probing
# --------------------------------------------------------------------------- #


def require_tool(path: Path | None, name: str) -> Path:
    if path is None or not path.is_file():
        raise FFmpegNotFound(
            f"{name} was not found at {path}. Set video.ffmpeg_path / video.ffprobe_path "
            "in config/pipeline_settings.json."
        )
    return path


def probe_duration(path: Path, ffprobe: Path) -> float:
    """Return a media file's duration in seconds."""
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[:400]}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned no duration for {path}") from exc


# --------------------------------------------------------------------------- #
# Per-scene clips
# --------------------------------------------------------------------------- #


_MOTION_KEYWORDS = {
    "push": "push_in",
    "zoom in": "push_in",
    "dolly in": "push_in",
    "close in": "push_in",
    "pull": "pull_out",
    "zoom out": "pull_out",
    "dolly out": "pull_out",
    "pan left": "pan_left",
    "pan right": "pan_right",
    "drift": "push_in",
    "parallax": "push_in",
}


def motion_kind(scene: dict[str, Any]) -> str:
    """Map free-text motion direction onto a deterministic effect."""
    text = f"{scene.get('motion', '')} {scene.get('camera_movement', '')}".lower()
    for keyword, kind in _MOTION_KEYWORDS.items():
        if keyword in text:
            return kind
    return "static"


def _filter_path(path: Path) -> str:
    """Escape a Windows path for use inside an FFmpeg filter argument."""
    value = str(path).replace("\\", "/")
    value = value.replace(":", "\\:")
    value = value.replace("'", "\\'")
    return value


def scene_clip_args(
    *,
    asset: Path | None,
    scene: dict[str, Any],
    settings: RenderSettings,
    out_path: Path,
    text_path: Path | None = None,
) -> list[str]:
    """Build the ffmpeg arguments for a single scene clip.

    ``asset`` is ``None`` for ``text_overlay`` scenes.  Those are rendered from
    a deterministic colour source plus ``drawtext``; no external provider or
    image is required.
    """
    duration = float(scene.get("estimated_duration_seconds") or 4.0)
    duration = max(1.0, min(duration, 120.0))
    frames = max(2, int(round(duration * settings.fps)))
    motion = motion_kind(scene)
    is_video = bool(asset) and asset.suffix.lower() in {
        ".mp4", ".mov", ".mkv", ".webm", ".avi"
    }

    args: list[str] = [str(settings.ffmpeg), "-y", "-hide_banner", "-loglevel", "error"]

    if asset is None:
        if text_path is None or not text_path.is_file():
            raise ValueError(
                f"Scene {scene.get('scene_id')} is a text overlay but no text file was supplied"
            )
        if settings.font_path is None or not settings.font_path.is_file():
            raise FileNotFoundError(
                f"Text overlay font not found: {settings.font_path}. "
                "Set video.font_path in config/pipeline_settings.json."
            )
        font_size = max(24, int(settings.width / 16))
        text_filter = (
            f"drawtext=fontfile='{_filter_path(settings.font_path)}'"
            f":textfile='{_filter_path(text_path)}'"
            f":fontcolor=white:fontsize={font_size}:line_spacing={max(8, font_size // 3)}"
            ":x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.45"
            f":boxborderw={max(16, font_size // 2)}"
        )
        args += [
            "-f",
            "lavfi",
            "-t",
            f"{duration:.3f}",
            "-i",
            f"color=c=0x101820:s={settings.width}x{settings.height}:r={settings.fps}",
            "-vf",
            f"{text_filter},format=yuv420p",
        ]
    elif is_video:
        assert asset is not None
        args += ["-i", str(asset)]
        filters = [settings.dimension_filter(), f"fps={settings.fps}"]
        args += [
            "-vf",
            ",".join(filters),
            "-t",
            f"{duration:.3f}",
            "-an",
        ]
    elif motion == "static":
        assert asset is not None
        args += ["-loop", "1", "-t", f"{duration:.3f}", "-i", str(asset)]
        args += [
            "-vf",
            f"{settings.dimension_filter()},fps={settings.fps},format=yuv420p",
        ]
    else:
        assert asset is not None
        # Single frame input; zoompan generates the motion frames.
        args += ["-i", str(asset)]
        zoom_in = motion in ("push_in", "pan_left")
        zoom_expr = (
            f"min(zoom+0.0012,1.14)" if zoom_in else "if(lte(zoom,1.0),1.14,max(1.001,zoom-0.0012))"
        )
        if motion == "pan_left":
            x_expr = f"max(0,iw/2-(iw/zoom/2)-((iw/zoom)*0.10)*on/{frames})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif motion == "pan_right":
            x_expr = f"min(iw-iw/zoom,iw/2-(iw/zoom/2)+((iw/zoom)*0.10)*on/{frames})"
            y_expr = "ih/2-(ih/zoom/2)"
        else:
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        vf = (
            f"scale={settings.width * 2}:{settings.height * 2}:force_original_aspect_ratio=increase,"
            f"crop={settings.width * 2}:{settings.height * 2},setsar=1,"
            f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':"
            f"d={frames}:s={settings.width}x{settings.height}:fps={settings.fps},"
            "format=yuv420p"
        )
        args += ["-vf", vf, "-t", f"{duration:.3f}"]

    args += [
        "-c:v",
        "libx264",
        "-preset",
        settings.preset,
        "-crf",
        str(settings.crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(settings.fps),
        str(out_path),
    ]
    return args


# --------------------------------------------------------------------------- #
# Concatenation
# --------------------------------------------------------------------------- #


def write_concat_list(clip_paths: Sequence[Path], list_path: Path) -> Path:
    """Write an ffmpeg concat demuxer list with correctly escaped paths."""
    lines = []
    for clip in clip_paths:
        safe = str(clip).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{safe}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def concat_args(*, list_path: Path, out_path: Path, settings: RenderSettings) -> list[str]:
    return [
        str(settings.ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(out_path),
    ]


# --------------------------------------------------------------------------- #
# Audio mix and mux
# --------------------------------------------------------------------------- #


def mux_args(
    *,
    video_path: Path,
    narration_path: Path,
    out_path: Path,
    settings: RenderSettings,
    music_path: Path | None = None,
    sfx: Sequence[tuple[Path, float, float]] = (),
    music_gain_db: float = -22.0,
    duck_db: float = 0.0,
    true_peak_db: float = -1.5,
    fade_in: float = 2.0,
    fade_out: float = 3.0,
    total_duration: float | None = None,
) -> list[str]:
    """Build the final mux command.

    Narration is always the anchor track.  Music is attenuated to
    *music_gain_db* and, when ``duck_db`` is negative, dynamically compressed
    under the narration with ``sidechaincompress``.  Sound effects carry their
    own gain and are placed with ``adelay`` at the scene start.
    """
    args: list[str] = [str(settings.ffmpeg), "-y", "-hide_banner", "-loglevel", "error"]
    args += ["-i", str(video_path)]
    args += ["-i", str(narration_path)]
    input_count = 2

    music_index: int | None = None
    if music_path and music_path.is_file():
        args += ["-stream_loop", "-1", "-i", str(music_path)]
        music_index = input_count
        input_count += 1

    sfx_indices: list[tuple[int, float, float]] = []
    for cue in sfx:
        path, start = cue[0], cue[1]
        gain = float(cue[2]) if len(cue) > 2 else 0.0
        if path.is_file():
            args += ["-i", str(path)]
            sfx_indices.append((input_count, start, gain))
            input_count += 1

    filters: list[str] = []
    ducking = bool(music_index is not None and duck_db < 0)
    if ducking:
        filters.append(
            "[1:a]aformat=sample_fmts=fltp:sample_rates=48000,volume=1.0,"
            "asplit=2[narr][duckkey]"
        )
    else:
        filters.append(
            "[1:a]aformat=sample_fmts=fltp:sample_rates=48000,volume=1.0[narr]"
        )

    mix_inputs = ["[narr]"]
    if music_index is not None:
        fade = f"afade=t=in:st=0:d={fade_in}," if fade_in > 0 else ""
        if total_duration and fade_out > 0:
            fade += f"afade=t=out:st={max(0.0, total_duration - fade_out):.2f}:d={fade_out},"
        filters.append(
            f"[{music_index}:a]aformat=sample_fmts=fltp:sample_rates=48000,"
            f"volume={music_gain_db}dB,{fade}aresample=48000[musicraw]"
        )
        if ducking:
            ratio = max(2.0, min(20.0, 1.0 + abs(duck_db)))
            filters.append(
                "[musicraw][duckkey]sidechaincompress="
                f"threshold=0.03:ratio={ratio:.1f}:attack=20:release=350:"
                "knee=2.77:makeup=1[music]"
            )
        else:
            filters.append("[musicraw]anull[music]")
        mix_inputs.append("[music]")

    for position, (index, start, gain) in enumerate(sfx_indices):
        delay_ms = int(max(0.0, start) * 1000)
        filters.append(
            f"[{index}:a]aformat=sample_fmts=fltp:sample_rates=48000,"
            f"volume={gain}dB,adelay={delay_ms}|{delay_ms}[sfx{position}]"
        )
        mix_inputs.append(f"[sfx{position}]")

    if len(mix_inputs) > 1:
        filters.append(
            f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:"
            "duration=first:dropout_transition=0:normalize=0[mixed]"
        )
        filters.append(f"[mixed]alimiter=limit={10 ** (true_peak_db / 20):.4f}[aout]")
    else:
        filters.append(f"[narr]alimiter=limit={10 ** (true_peak_db / 20):.4f}[aout]")

    args += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        settings.audio_bitrate,
        "-movflags",
        "+faststart",
    ]
    if total_duration:
        args += ["-t", f"{total_duration:.3f}"]
    args.append(str(out_path))
    return args


def subtitles_args(
    *,
    video_path: Path,
    srt_path: Path,
    out_path: Path,
    settings: RenderSettings,
) -> list[str]:
    """Burn subtitles into a rendered video."""
    escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    return [
        str(settings.ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"subtitles='{escaped}'",
        "-c:v",
        "libx264",
        "-preset",
        settings.preset,
        "-crf",
        str(settings.crf),
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(out_path),
    ]


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def validate_render_inputs(
    plan: dict[str, Any],
    *,
    resolve_asset,
    narration_path: Path,
) -> tuple[list[tuple[dict[str, Any], Path | None]], list[str]]:
    """Return ``(ready_scenes, missing_descriptions)`` without rendering anything.

    ``text_overlay`` scenes are ready with ``asset=None``: the renderer draws
    them deterministically with FFmpeg and needs no external file.
    """
    ready: list[tuple[dict[str, Any], Path | None]] = []
    missing: list[str] = []

    if not narration_path.is_file():
        missing.append(f"narration audio: {narration_path}")

    for scene in plan.get("scenes", []) or []:
        scene_id = str(scene.get("scene_id", "?"))
        asset_info = scene.get("asset") or {}
        status = asset_info.get("status")
        asset_type = str(scene.get("asset_type") or asset_info.get("type") or "")
        overlay_text = (
            scene.get("on_screen_text")
            or scene.get("text")
            or asset_info.get("on_screen_text")
            or asset_info.get("text")
        )
        if status == "not_required" and asset_type == "text_overlay" and str(overlay_text or "").strip():
            ready.append((scene, None))
            continue
        if status == "not_required":
            missing.append(
                f"{scene_id}: {asset_type or 'asset'} scenes need a renderer or on-screen text"
            )
            continue
        if status != "ready":
            missing.append(f"{scene_id}: asset status '{status or 'missing'}'")
            continue
        path = resolve_asset(asset_info.get("path"))
        if path is None:
            missing.append(f"{scene_id}: asset file not found ({asset_info.get('path')})")
            continue
        ready.append((scene, path))

    return ready, missing


def describe_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Small summary used by ``--status`` and the doctor command."""
    scenes = plan.get("scenes", []) or []
    return {
        "format": (plan.get("project") or {}).get("format"),
        "scenes": len(scenes),
        "estimated_seconds": round(
            sum(float(scene.get("estimated_duration_seconds", 0)) for scene in scenes), 1
        ),
        "asset_types": sorted({str(scene.get("asset_type")) for scene in scenes}),
        "pending_assets": sum(
            1
            for scene in scenes
            if (scene.get("asset") or {}).get("status") in (None, "pending", "missing")
        ),
    }


__all__ = [
    "FFmpegNotFound",
    "MissingAssetsError",
    "RenderSettings",
    "concat_args",
    "describe_plan",
    "motion_kind",
    "mux_args",
    "probe_duration",
    "require_tool",
    "scene_clip_args",
    "subtitles_args",
    "validate_render_inputs",
    "write_concat_list",
]
