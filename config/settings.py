"""Central configuration for PIXEL AI.

Every tunable value lives either in ``config/model_settings.json`` or
``config/pipeline_settings.json``.  Engine modules must never hardcode model
parameters, paths or credentials.

This module exposes a small, typed, process-wide :class:`AppConfig` object so
that the rest of the code base can depend on configuration without knowing how
it is stored on disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Application root (the ``pixel-ai`` folder that contains ``main.py``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Repository root - the folder that holds ``pixel-ai/``, ``llama.cpp/``,
#: ``.venv/`` and the legacy ``old/`` Kokoro script.
REPO_ROOT: Path = PROJECT_ROOT.parent

CONFIG_DIR: Path = PROJECT_ROOT / "config"
PROFILES_DIR: Path = PROJECT_ROOT / "profiles"
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
SCHEMAS_DIR: Path = PROJECT_ROOT / "schemas"
TOOLS_DIR: Path = PROJECT_ROOT / "tools"
LEGACY_DIR: Path = REPO_ROOT / "old"

MODEL_SETTINGS_FILE: Path = CONFIG_DIR / "model_settings.json"
PIPELINE_SETTINGS_FILE: Path = CONFIG_DIR / "pipeline_settings.json"


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #


def load_json_file(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON document and return it as a dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _deep_get(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Return ``data["a"]["b"]`` for the dotted key ``"a.b"``."""
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def _resolve_path(value: str | os.PathLike[str] | None, base: Path) -> Path | None:
    """Resolve a possibly relative configured path against ``base``."""
    if value is None:
        return None
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


# --------------------------------------------------------------------------- #
# Configuration objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSettings:
    """Everything needed to talk to the local inference server."""

    raw: dict[str, Any] = field(repr=False)

    # -- connection -------------------------------------------------------- #
    @property
    def base_url(self) -> str:
        return str(self.raw.get("base_url", "http://127.0.0.1:8080/v1")).rstrip("/")

    @property
    def model(self) -> str:
        return str(self.raw.get("model", "local-model"))

    @property
    def api_key(self) -> str | None:
        """API key resolved from the environment (never stored in the repo)."""
        env_name = self.raw.get("api_key_env")
        if env_name:
            value = os.environ.get(str(env_name))
            if value:
                return value
        inline = self.raw.get("api_key")
        return str(inline) if inline else None

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rsplit('/v1', 1)[0]}/health"

    # -- request knobs ----------------------------------------------------- #
    @property
    def timeout_seconds(self) -> float:
        return float(_deep_get(self.raw, "request.timeout_seconds", 600))

    @property
    def connect_timeout_seconds(self) -> float:
        return float(_deep_get(self.raw, "request.connect_timeout_seconds", 10))

    @property
    def max_retries(self) -> int:
        return int(_deep_get(self.raw, "request.max_retries", 3))

    @property
    def retry_backoff_seconds(self) -> float:
        return float(_deep_get(self.raw, "request.retry_backoff_seconds", 4.0))

    @property
    def default_temperature(self) -> float:
        return float(_deep_get(self.raw, "request.temperature", 0.7))

    @property
    def default_max_tokens(self) -> int:
        return int(_deep_get(self.raw, "request.max_tokens", 1024))

    @property
    def top_p(self) -> float:
        return float(_deep_get(self.raw, "request.top_p", 0.95))

    @property
    def repeat_penalty(self) -> float:
        return float(_deep_get(self.raw, "request.repeat_penalty", 1.05))

    @property
    def keep_alive_seconds(self) -> int:
        return int(_deep_get(self.raw, "request.keep_alive_seconds", 120))

    # -- json behaviour ---------------------------------------------------- #
    @property
    def use_response_format(self) -> bool:
        return bool(_deep_get(self.raw, "json.use_response_format", True))

    @property
    def max_repair_attempts(self) -> int:
        return int(_deep_get(self.raw, "json.max_repair_attempts", 2))

    @property
    def strict_schema(self) -> bool:
        return bool(_deep_get(self.raw, "json.strict_schema", True))

    # -- reasoning --------------------------------------------------------- #
    @property
    def strip_think_tags(self) -> bool:
        return bool(_deep_get(self.raw, "reasoning.strip_think_tags", True))

    # -- server ------------------------------------------------------------ #
    @property
    def server_executable(self) -> Path | None:
        return _resolve_path(_deep_get(self.raw, "server.executable"), REPO_ROOT)

    @property
    def server_model_path(self) -> Path | None:
        return _resolve_path(_deep_get(self.raw, "server.model_path"), REPO_ROOT)

    @property
    def context_size(self) -> int:
        return int(_deep_get(self.raw, "server.context_size", 4096))

    @property
    def min_context_tokens(self) -> int:
        return int(_deep_get(self.raw, "reasoning.min_context_tokens", 4096))

    @property
    def server_reasoning_budget(self) -> int:
        return int(_deep_get(self.raw, "server.reasoning_budget", -1))

    @property
    def auto_start(self) -> bool:
        return bool(_deep_get(self.raw, "server.auto_start", False))

    @property
    def server_launch_command(self) -> list[str]:
        """The exact command line the doctor suggests for this machine."""
        exe = self.server_executable
        model = self.server_model_path
        return [
            str(exe) if exe else "llama-server",
            "-m",
            str(model) if model else "<model.gguf>",
            "-ngl", str(int(_deep_get(self.raw, "server.gpu_layers", 16))),
            "-t", str(int(_deep_get(self.raw, "server.cpu_threads", 6))),
            "-b", str(int(_deep_get(self.raw, "server.batch_size", 128))),
            "-ub", str(int(_deep_get(self.raw, "server.ubatch_size", 64))),
            "-c", str(self.context_size),
            "-np", str(int(_deep_get(self.raw, "server.parallel_slots", 1))),
            "-fa", str(_deep_get(self.raw, "server.flash_attention", "auto")),
            "--reasoning-budget", str(self.server_reasoning_budget),
            "--host", str(_deep_get(self.raw, "server.host", "127.0.0.1")),
            "--port", str(int(_deep_get(self.raw, "server.port", 8080))),
        ]


@dataclass(frozen=True)
class PipelineSettings:
    """Non-model knobs that drive the production pipeline."""

    raw: dict[str, Any] = field(repr=False)

    # -- paths ------------------------------------------------------------- #
    @property
    def scripts_dir(self) -> Path:
        return _resolve_path(_deep_get(self.raw, "paths.scripts_dir", "scripts"), PROJECT_ROOT) or (
            PROJECT_ROOT / "scripts"
        )

    @property
    def output_dir(self) -> Path:
        return _resolve_path(_deep_get(self.raw, "paths.output_dir", "output"), PROJECT_ROOT) or (
            PROJECT_ROOT / "output"
        )

    @property
    def logs_dir(self) -> Path:
        return _resolve_path(_deep_get(self.raw, "paths.logs_dir", "logs"), PROJECT_ROOT) or (
            PROJECT_ROOT / "logs"
        )

    @property
    def assets_subdir(self) -> str:
        return str(_deep_get(self.raw, "paths.assets_subdir", "assets"))

    # -- execution --------------------------------------------------------- #
    @property
    def require_approval_after_script(self) -> bool:
        return bool(_deep_get(self.raw, "execution.require_approval_after_script", True))

    @property
    def max_stage_retries(self) -> int:
        return int(_deep_get(self.raw, "execution.max_stage_retries", 2))

    @property
    def continue_on_stage_failure(self) -> bool:
        return bool(_deep_get(self.raw, "execution.continue_on_stage_failure", False))

    # -- script ------------------------------------------------------------ #
    @property
    def words_per_minute(self) -> int:
        return int(_deep_get(self.raw, "script.words_per_minute", 150))

    @property
    def long_min_words(self) -> int:
        return int(_deep_get(self.raw, "script.long_min_words", 900))

    @property
    def long_max_words(self) -> int:
        return int(_deep_get(self.raw, "script.long_max_words", 2200))

    @property
    def short_target_words(self) -> int:
        return int(_deep_get(self.raw, "script.short_target_words", 105))

    @property
    def short_max_words(self) -> int:
        return int(_deep_get(self.raw, "script.short_max_words", 145))

    @property
    def shorts_heading_prefix(self) -> str:
        return str(_deep_get(self.raw, "script.shorts_heading_prefix", "SHORT"))

    @property
    def shorts_heading_separator(self) -> str:
        return str(_deep_get(self.raw, "script.shorts_heading_separator", "\u2014"))

    # -- research ---------------------------------------------------------- #
    @property
    def research_provider(self) -> str:
        return str(_deep_get(self.raw, "research.provider", "none"))

    @property
    def research_max_results(self) -> int:
        return int(_deep_get(self.raw, "research.max_results_per_query", 6))

    @property
    def research_max_queries(self) -> int:
        return int(_deep_get(self.raw, "research.max_queries", 6))

    @property
    def research_summary_chars(self) -> int:
        return int(_deep_get(self.raw, "research.summary_chars", 1200))

    @property
    def research_source_chars(self) -> int:
        return int(_deep_get(self.raw, "research.source_chars", 700))

    # -- visuals ----------------------------------------------------------- #
    @property
    def scene_target_sentences(self) -> int:
        return int(_deep_get(self.raw, "visuals.scene_target_sentences", 3))

    @property
    def scene_target_words(self) -> int:
        return int(_deep_get(self.raw, "visuals.scene_target_words", 70))

    @property
    def max_scenes_long(self) -> int:
        return int(_deep_get(self.raw, "visuals.max_scenes_long", 60))

    @property
    def max_scenes_per_short(self) -> int:
        return int(_deep_get(self.raw, "visuals.max_scenes_per_short", 6))

    @property
    def visual_batch_size(self) -> int:
        return int(_deep_get(self.raw, "visuals.batch_size", 3))

    @property
    def variety_pass(self) -> bool:
        return bool(_deep_get(self.raw, "visuals.variety_pass", True))

    @property
    def max_variety_repair_rounds(self) -> int:
        return int(_deep_get(self.raw, "visuals.max_variety_repair_rounds", 2))

    # -- audio ------------------------------------------------------------- #
    @property
    def music_segments_max(self) -> int:
        return int(_deep_get(self.raw, "audio.music_segments_max", 14))

    @property
    def sfx_max(self) -> int:
        return int(_deep_get(self.raw, "audio.sfx_max", 24))

    @property
    def narration_duck_db(self) -> float:
        return float(_deep_get(self.raw, "audio.narration_duck_db", -9.0))

    @property
    def music_bed_db(self) -> float:
        return float(_deep_get(self.raw, "audio.music_bed_db", -22.0))

    @property
    def music_fade_in_seconds(self) -> float:
        return float(_deep_get(self.raw, "audio.music_fade_in_seconds", 2.0))

    @property
    def music_fade_out_seconds(self) -> float:
        return float(_deep_get(self.raw, "audio.music_fade_out_seconds", 3.0))

    # -- tts --------------------------------------------------------------- #
    @property
    def tts_enabled(self) -> bool:
        return bool(_deep_get(self.raw, "tts.enabled", True))

    @property
    def tts_voice(self) -> str:
        return str(_deep_get(self.raw, "tts.voice", "am_liam"))

    @property
    def tts_speed(self) -> float:
        return float(_deep_get(self.raw, "tts.speed", 1.25))

    @property
    def tts_device(self) -> str:
        return str(_deep_get(self.raw, "tts.device", "cpu"))

    @property
    def tts_sample_rate(self) -> int:
        return int(_deep_get(self.raw, "tts.sample_rate", 24000))

    @property
    def tts_lang_code(self) -> str:
        return str(_deep_get(self.raw, "tts.lang_code", "a"))

    # -- quality ----------------------------------------------------------- #
    @property
    def quality_ai_review(self) -> bool:
        return bool(_deep_get(self.raw, "quality.ai_review", True))

    @property
    def quality_fail_on(self) -> str:
        """Blocking threshold: ``error``, ``warning``, ``never``."""
        return str(_deep_get(self.raw, "quality.fail_on", "error")).lower()

    @property
    def quality_ai_review_can_fail(self) -> bool:
        """Whether editorial AI findings may trigger the blocking gate.

        Deterministic checks are always authoritative.  Editorial judgement is
        advisory by default so a subjective review cannot silently block an
        otherwise valid production.
        """
        return bool(_deep_get(self.raw, "quality.ai_review_can_fail", False))

    # -- assets ------------------------------------------------------------ #
    @property
    def image_provider(self) -> str:
        return str(_deep_get(self.raw, "assets.image_provider", "none"))

    @property
    def video_provider(self) -> str:
        return str(_deep_get(self.raw, "assets.video_provider", "none"))

    @property
    def sfx_provider(self) -> str:
        return str(_deep_get(self.raw, "assets.sfx_provider", "none"))

    @property
    def music_provider(self) -> str:
        return str(_deep_get(self.raw, "assets.music_provider", "none"))

    # -- video ------------------------------------------------------------- #
    @property
    def ffmpeg_path(self) -> Path | None:
        return _resolve_path(_deep_get(self.raw, "video.ffmpeg_path"), PROJECT_ROOT)

    @property
    def ffprobe_path(self) -> Path | None:
        return _resolve_path(_deep_get(self.raw, "video.ffprobe_path"), PROJECT_ROOT)

    @property
    def long_size(self) -> tuple[int, int]:
        return (
            int(_deep_get(self.raw, "video.long_width", 1920)),
            int(_deep_get(self.raw, "video.long_height", 1080)),
        )

    @property
    def short_size(self) -> tuple[int, int]:
        return (
            int(_deep_get(self.raw, "video.short_width", 1080)),
            int(_deep_get(self.raw, "video.short_height", 1920)),
        )

    @property
    def long_fps(self) -> int:
        return int(_deep_get(self.raw, "video.long_fps", 30))

    @property
    def short_fps(self) -> int:
        return int(_deep_get(self.raw, "video.short_fps", 30))

    @property
    def render_enabled(self) -> bool:
        return bool(_deep_get(self.raw, "video.render", False))

    @property
    def video_font_path(self) -> Path | None:
        raw = _deep_get(self.raw, "video.font_path", "C:/Windows/Fonts/arial.ttf")
        return _resolve_path(raw, PROJECT_ROOT) if raw else None


@dataclass(frozen=True)
class AppConfig:
    """Bundled view of both configuration files plus useful derived paths."""

    model: ModelSettings
    pipeline: PipelineSettings

    @property
    def profiles_dir(self) -> Path:
        return PROFILES_DIR

    @property
    def prompts_dir(self) -> Path:
        return PROMPTS_DIR

    @property
    def schemas_dir(self) -> Path:
        return SCHEMAS_DIR

    @property
    def tools_dir(self) -> Path:
        return TOOLS_DIR

    @property
    def legacy_dir(self) -> Path:
        return LEGACY_DIR


def _load_model_settings(path: Path | None = None) -> ModelSettings:
    target = path or MODEL_SETTINGS_FILE
    raw = load_json_file(target)
    return ModelSettings(raw=raw)


def _load_pipeline_settings(path: Path | None = None) -> PipelineSettings:
    target = path or PIPELINE_SETTINGS_FILE
    raw = load_json_file(target)
    return PipelineSettings(raw=raw)


def load_config(
    model_settings_path: Path | None = None,
    pipeline_settings_path: Path | None = None,
) -> AppConfig:
    """Load the application configuration.

    Parameters
    ----------
    model_settings_path:
        Optional override for ``config/model_settings.json``.
    pipeline_settings_path:
        Optional override for ``config/pipeline_settings.json``.
    """
    return AppConfig(
        model=_load_model_settings(model_settings_path),
        pipeline=_load_pipeline_settings(pipeline_settings_path),
    )


#: Lazily created process-wide configuration.
_CONFIG: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the process-wide :class:`AppConfig`, loading it on first use."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def reload_config() -> AppConfig:
    """Force a reload of the configuration files (used by tests and ``--reload``)."""
    global _CONFIG
    _CONFIG = load_config()
    return _CONFIG


__all__ = [
    "AppConfig",
    "ModelSettings",
    "PipelineSettings",
    "PROJECT_ROOT",
    "REPO_ROOT",
    "CONFIG_DIR",
    "PROFILES_DIR",
    "PROMPTS_DIR",
    "SCHEMAS_DIR",
    "TOOLS_DIR",
    "LEGACY_DIR",
    "get_config",
    "reload_config",
    "load_config",
    "load_json_file",
]
