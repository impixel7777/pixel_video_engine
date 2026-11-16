"""Pipeline orchestration.

Responsibilities:
    * run stages in order, with per-stage retry and explicit status
    * persist ``production_state.json`` after every stage so a crash never
      loses completed work
    * skip stages that are already complete *and* whose artefacts still exist
    * invalidate downstream stages when an upstream stage is regenerated
    * enforce the manual approval checkpoint between script and visuals
    * never hide a failed stage

The pipeline is deliberately linear.  A DAG engine would not make this project
easier to maintain.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from config.settings import AppConfig
from engine.ai_stage import brief_digest, research_digest
from engine.asset_manager import AssetManager
from engine.audio_director import AudioDirector
from engine.channel_profile import ChannelProfile
from engine.content_planner import ContentPlanner
from engine.deepseek_client import DeepSeekClient, ServerStatus
from engine.file_manager import (
    ARTIFACTS,
    FileManager,
    ProjectPaths,
    atomic_write_json,
    atomic_write_text,
    read_json,
    utc_now_iso,
)
from engine.logging_setup import get_logger
from engine.narration import count_words, parse_shorts_text
from engine.prompt_library import PromptLibrary
from engine.quality_controller import QualityController, ReviewInputs
from engine.research_engine import ResearchEngine
from engine.script_writer import ScriptResult, ScriptWriter
from engine.shorts_writer import ShortsWriter
from engine.timeline import pause_insertion_points, reconcile_plan
from engine.topic_analyzer import TopicAnalyzer
from engine.tts.project_audio import (
    engine_config,
    generate_project_audio,
    insert_pauses,
    wav_duration,
)
from engine.video_assembler import VideoAssembler
from engine.visual_director import VisualDirector

log = get_logger("pipeline")


# --------------------------------------------------------------------------- #
# Stage definitions
# --------------------------------------------------------------------------- #


class Stage(str, Enum):
    CONNECTION = "connection"
    TOPIC = "topic"
    RESEARCH = "research"
    PLAN = "plan"
    SCRIPT = "script"
    SHORTS = "shorts"
    VISUALS = "visuals"
    AUDIO = "audio"
    QUALITY = "quality"
    SAVE = "save"
    VALIDATE = "validate"
    TTS = "tts"
    ASSETS = "assets"
    VIDEO = "video"


#: Canonical order.  Approval happens after **both** narration files exist
#: (inside the SHORTS stage), and quality runs after visuals and audio so it can
#: validate them.
STAGE_ORDER: tuple[Stage, ...] = (
    Stage.CONNECTION,
    Stage.TOPIC,
    Stage.RESEARCH,
    Stage.PLAN,
    Stage.SCRIPT,
    Stage.SHORTS,
    Stage.VISUALS,
    Stage.AUDIO,
    Stage.QUALITY,
    Stage.SAVE,
    Stage.VALIDATE,
    Stage.TTS,
    Stage.ASSETS,
    Stage.VIDEO,
)

#: Stages that produce content and therefore get regenerated on retry.
RETRYABLE = {
    Stage.TOPIC,
    Stage.RESEARCH,
    Stage.PLAN,
    Stage.SCRIPT,
    Stage.SHORTS,
    Stage.VISUALS,
    Stage.AUDIO,
    Stage.QUALITY,
    Stage.TTS,
    Stage.ASSETS,
    Stage.VIDEO,
}

#: Which artefacts each stage must leave behind to count as complete.
STAGE_ARTIFACTS: dict[Stage, tuple[str, ...]] = {
    Stage.TOPIC: (ARTIFACTS["brief"],),
    Stage.RESEARCH: (ARTIFACTS["research"],),
    Stage.PLAN: (ARTIFACTS["outline"],),
    Stage.SCRIPT: (ARTIFACTS["long_script"], ARTIFACTS["script_sections"]),
    Stage.SHORTS: (ARTIFACTS["short_script"],),
    Stage.VISUALS: (ARTIFACTS["long_plan"], ARTIFACTS["shorts_plan"]),
    Stage.AUDIO: (ARTIFACTS["audio_plan"], ARTIFACTS["long_plan"], ARTIFACTS["shorts_plan"]),
    Stage.QUALITY: (ARTIFACTS["quality"],),
    Stage.SAVE: (ARTIFACTS["manifest"],),
    Stage.TTS: (ARTIFACTS["tts"], "output/Long/narration.wav"),
    Stage.ASSETS: (ARTIFACTS["assets_status"],),
}

#: Content stages invalidated when an earlier stage is recomputed.  Derived
#: metadata (manifest, quality report, TTS report, render reports) is invalidated
#: too; stale files are never left marked as current.
_ALL_CONTENT = (
    Stage.TOPIC,
    Stage.RESEARCH,
    Stage.PLAN,
    Stage.SCRIPT,
    Stage.SHORTS,
    Stage.VISUALS,
    Stage.AUDIO,
    Stage.QUALITY,
    Stage.SAVE,
    Stage.VALIDATE,
    Stage.TTS,
    Stage.ASSETS,
    Stage.VIDEO,
)
DOWNSTREAM: dict[Stage, tuple[Stage, ...]] = {
    Stage.TOPIC: _ALL_CONTENT,
    Stage.RESEARCH: _ALL_CONTENT[2:],
    Stage.PLAN: _ALL_CONTENT[3:],
    Stage.SCRIPT: _ALL_CONTENT[4:],
    Stage.SHORTS: _ALL_CONTENT[5:],
    Stage.VISUALS: (Stage.AUDIO, Stage.QUALITY, Stage.SAVE, Stage.VALIDATE, Stage.TTS, Stage.ASSETS, Stage.VIDEO),
    Stage.AUDIO: (Stage.QUALITY, Stage.SAVE, Stage.VALIDATE, Stage.TTS, Stage.VIDEO),
    Stage.QUALITY: (Stage.SAVE, Stage.VALIDATE, Stage.TTS, Stage.ASSETS, Stage.VIDEO),
    Stage.SAVE: (Stage.VALIDATE, Stage.ASSETS, Stage.VIDEO),
    Stage.VALIDATE: (Stage.TTS, Stage.ASSETS, Stage.VIDEO),
    Stage.TTS: (Stage.ASSETS, Stage.VIDEO),
    Stage.ASSETS: (Stage.VIDEO,),
}

STATE_VERSION = "1.0"


class PipelineError(RuntimeError):
    pass


class ServerNotAvailable(PipelineError):
    pass


class ApprovalRequired(PipelineError):
    """Raised when the script needs human approval before visuals."""


class QualityGateError(PipelineError):
    """Deterministic quality threshold blocked downstream production."""


@dataclass
class StageRecord:
    status: str = "pending"  # pending | running | complete | failed | skipped
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error,
            "artifacts": list(self.artifacts),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageRecord":
        return cls(
            status=str(data.get("status", "pending")),
            attempts=int(data.get("attempts", 0)),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            duration_seconds=float(data.get("duration_seconds", 0) or 0),
            error=data.get("error"),
            artifacts=list(data.get("artifacts", []) or []),
            notes=str(data.get("notes", "")),
        )


@dataclass
class PipelineResult:
    status: str
    paths: ProjectPaths
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: str | None = None
    message: str = ""
    quality_status: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "complete"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class Pipeline:
    """One production run for one topic on one channel."""

    def __init__(
        self,
        config: AppConfig,
        profile: ChannelProfile,
        topic: str,
        *,
        resume: bool = True,
        auto_approve: bool = False,
        preapprove: bool = False,
        force_stages: Iterable[str] = (),
        force_all: bool = False,
        run_assets: bool = False,
        render_video: bool = False,
        run_tts: bool | None = None,
        asset_dry_run: bool = True,
    ) -> None:
        self.config = config
        self.profile = profile
        self.topic = topic.strip()
        if not self.topic:
            raise PipelineError("Topic must not be empty")

        self.file_manager = FileManager(
            scripts_dir=config.pipeline.scripts_dir,
            assets_subdir=config.pipeline.assets_subdir,
        )
        self.paths = self.file_manager.create_project(profile.id, self.topic)
        self.resume = resume
        self.auto_approve = auto_approve
        self.preapprove = preapprove
        self.force_stages = {str(stage).lower() for stage in force_stages}
        self.force_all = force_all
        self.run_assets = run_assets
        self.render_video = render_video
        self.run_tts = config.pipeline.tts_enabled if run_tts is None else run_tts
        self.asset_dry_run = asset_dry_run

        self.prompts = PromptLibrary(config.prompts_dir)
        self.client = DeepSeekClient(config.model)
        self.state: dict[str, Any] = {}
        self._load_or_init_state()

        # Stage implementations, wired lazily so constructing a Pipeline never
        # makes a network call.
        self._handlers: dict[Stage, Callable[[], None]] = {
            Stage.CONNECTION: self._stage_connection,
            Stage.TOPIC: self._stage_topic,
            Stage.RESEARCH: self._stage_research,
            Stage.PLAN: self._stage_plan,
            Stage.SCRIPT: self._stage_script,
            Stage.SHORTS: self._stage_shorts,
            Stage.VISUALS: self._stage_visuals,
            Stage.AUDIO: self._stage_audio,
            Stage.QUALITY: self._stage_quality,
            Stage.SAVE: self._stage_save,
            Stage.VALIDATE: self._stage_validate,
            Stage.TTS: self._stage_tts,
            Stage.ASSETS: self._stage_assets,
            Stage.VIDEO: self._stage_video,
        }

    # -- state -------------------------------------------------------------- #
    def _load_or_init_state(self) -> None:
        existing = read_json(self.paths.state) if self.resume else None
        if existing:
            self.state = existing
            self.state.setdefault("stages", {})
            self.state["resumed_at"] = utc_now_iso()
            log.info("Resuming project %s/%s", self.profile.id, self.paths.slug)
        else:
            self.state = {
                "schema_version": STATE_VERSION,
                "profile_id": self.profile.id,
                "topic": self.topic,
                "slug": self.paths.slug,
                "created_at": utc_now_iso(),
                "status": "initialised",
                "stages": {},
                "approval": {
                    "script_approved": False,
                    "approved_at": None,
                    "mode": None,
                    "content_hash": None,
                },
                "artifacts": {},
                "stage_order": [stage.value for stage in STAGE_ORDER],
                "input_fingerprints": {},
            }
            log.info("New project %s/%s", self.profile.id, self.paths.slug)

        previous_fingerprints = dict(self.state.get("input_fingerprints") or {})
        current_fingerprints = self._input_fingerprints()
        if previous_fingerprints:
            self._apply_input_changes(previous_fingerprints, current_fingerprints)
        self.state["input_fingerprints"] = current_fingerprints
        self.state["profile_fingerprint"] = self._profile_fingerprint()
        self.state["stage_order"] = [stage.value for stage in STAGE_ORDER]
        self._check_approval_freshness()
        self.state["updated_at"] = utc_now_iso()
        self._write_state()

    def _profile_fingerprint(self) -> str:
        payload = json.dumps(self.profile.fingerprint(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _sha256_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _section_fingerprint(self, section: str) -> str:
        payload = json.dumps(
            self.config.pipeline.raw.get(section, {}),
            sort_keys=True,
            ensure_ascii=False,
        )
        return self._sha256_text(payload)[:16]

    def _files_fingerprint(self, directory: Path, pattern: str) -> str:
        digest = hashlib.sha256()
        for path in sorted(directory.glob(pattern)):
            if not path.is_file():
                continue
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]

    def _input_fingerprints(self) -> dict[str, str]:
        """Fingerprints of every input that can change generated artefacts."""
        model_inputs = {
            "model": self.config.model.model,
            "base_url": self.config.model.base_url,
            "temperature": self.config.model.default_temperature,
            "max_tokens": self.config.model.default_max_tokens,
            "top_p": self.config.model.top_p,
            "repeat_penalty": self.config.model.repeat_penalty,
            "context_size": self.config.model.context_size,
            "reasoning_budget": self.config.model.server_reasoning_budget,
        }
        return {
            "profile": self._profile_fingerprint(),
            "topic": self._sha256_text(" ".join(self.topic.lower().split()))[:16],
            "prompts": self._files_fingerprint(self.prompts.prompts_dir, "*.txt"),
            "schemas": self._files_fingerprint(self.config.schemas_dir, "*.json"),
            "model": self._sha256_text(
                json.dumps(model_inputs, sort_keys=True, ensure_ascii=False)
            )[:16],
            "research": self._section_fingerprint("research"),
            "script": self._section_fingerprint("script"),
            "visuals": self._section_fingerprint("visuals"),
            "audio": self._section_fingerprint("audio"),
            "tts": self._section_fingerprint("tts"),
            "assets": self._section_fingerprint("assets"),
            "video": self._section_fingerprint("video"),
        }

    def _apply_input_changes(
        self, previous: dict[str, str], current: dict[str, str]
    ) -> None:
        affected: set[Stage] = set()

        def add_from(start: Stage) -> None:
            affected.update(stage for stage in STAGE_ORDER if STAGE_ORDER.index(stage) >= STAGE_ORDER.index(start))

        for key, old_value in previous.items():
            if current.get(key) == old_value:
                continue
            log.warning(
                "Input fingerprint changed for %s (%s -> %s); invalidating dependent stages",
                key,
                old_value,
                current.get(key),
            )
            if key in ("profile", "topic", "prompts", "schemas", "model"):
                add_from(Stage.TOPIC)
            elif key == "research":
                add_from(Stage.RESEARCH)
            elif key == "script":
                add_from(Stage.PLAN)
            elif key == "visuals":
                add_from(Stage.VISUALS)
            elif key == "audio":
                add_from(Stage.AUDIO)
            elif key == "assets":
                affected.update({Stage.ASSETS, Stage.VIDEO})
            elif key == "tts":
                affected.update({Stage.TTS, Stage.VIDEO})
            elif key == "video":
                affected.add(Stage.VIDEO)

        if not affected:
            return
        for stage in STAGE_ORDER:
            if stage not in affected:
                continue
            record = self.record(stage)
            if record.status != "pending":
                record.status = "pending"
                record.error = None
                record.notes = "Invalidated because an input fingerprint changed"
                self._set_record(stage, record)
                log.info("Invalidated stage %s after input change", stage.value)
            self._purge_derived(stage)
        if any(
            stage in affected
            for stage in (Stage.TOPIC, Stage.RESEARCH, Stage.PLAN, Stage.SCRIPT, Stage.SHORTS)
        ):
            self._clear_approval("narrative inputs changed")

    def _purge_derived(self, stage: Stage) -> None:
        """Remove derived metadata that must never look current after invalidation."""
        try:
            if stage is Stage.QUALITY:
                self.paths.quality.unlink(missing_ok=True)
            elif stage is Stage.SAVE:
                self.paths.manifest.unlink(missing_ok=True)
            elif stage is Stage.VIDEO:
                for report in self.paths.renders.glob("render_*.json"):
                    report.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem race
            log.warning("Could not purge derived artefact for %s: %s", stage.value, exc)

    def _write_state(self) -> None:
        atomic_write_json(self.paths.state, self.state)

    def record(self, stage: Stage) -> StageRecord:
        raw = (self.state.get("stages") or {}).get(stage.value, {})
        return StageRecord.from_dict(raw)

    def _set_record(self, stage: Stage, record: StageRecord) -> None:
        self.state.setdefault("stages", {})[stage.value] = record.to_dict()
        self.state["updated_at"] = utc_now_iso()
        self._write_state()

    # -- artefact helpers --------------------------------------------------- #
    def _artifact_paths(self, stage: Stage) -> list[Path]:
        return [self.paths.root / name for name in STAGE_ARTIFACTS.get(stage, ())]

    def is_complete(self, stage: Stage, *, respect_force: bool = True) -> bool:
        record = self.record(stage)
        if record.status != "complete":
            return False
        if respect_force and (self.force_all or stage.value in self.force_stages):
            return False
        if stage in STAGE_ARTIFACTS:
            return all(path.is_file() for path in self._artifact_paths(stage))
        return True

    def _invalidate_downstream(self, stage: Stage) -> None:
        for downstream in DOWNSTREAM.get(stage, ()):
            record = self.record(downstream)
            if record.status != "pending":
                record.status = "pending"
                record.error = None
                record.notes = f"Invalidated because {stage.value} was regenerated"
                self._set_record(downstream, record)
                log.info("Invalidated downstream stage %s", downstream.value)
            self._purge_derived(downstream)
        if stage in (Stage.TOPIC, Stage.RESEARCH, Stage.PLAN, Stage.SCRIPT, Stage.SHORTS):
            self._clear_approval(f"{stage.value} was regenerated")

    # -- loading artefacts -------------------------------------------------- #
    def _load(self, key: str, *, required: bool = True) -> Any:
        path = self.paths.root / ARTIFACTS[key]
        data = read_json(path) if path.suffix == ".json" else None
        if path.suffix != ".json":
            if not path.is_file():
                if required:
                    raise PipelineError(f"Missing required artefact: {path.name}. Run the earlier stage first.")
                return ""
            return path.read_text(encoding="utf-8")
        if data is None:
            if required:
                raise PipelineError(
                    f"Missing required artefact: {path.name}. "
                    "Run the earlier stage first (for example: --stage plan)."
                )
            return {}
        return data

    def _require(self, *stages: Stage) -> None:
        missing = [
            stage.value
            for stage in stages
            if not self.is_complete(stage, respect_force=False)
        ]
        if missing:
            raise PipelineError(
                "This stage needs earlier stages to be complete first: "
                + ", ".join(missing)
                + ". Run them with the produce command or --stage <name>."
            )

    # -- running ------------------------------------------------------------ #
    def run(self, stages: Iterable[str] | None = None) -> PipelineResult:
        selected = self._select_stages(stages)
        result = PipelineResult(status="running", paths=self.paths)
        self.state["status"] = "running"
        self._write_state()

        for stage in selected:
            if self.is_complete(stage):
                log.info("Stage %-10s already complete - skipping", stage.value)
                result.skipped.append(stage.value)
                continue

            try:
                self._run_stage(stage)
                result.ran.append(stage.value)
            except ApprovalRequired as exc:
                self.state["status"] = "awaiting_approval"
                self._write_state()
                result.status = "awaiting_approval"
                result.message = str(exc)
                result.failed = stage.value
                return result
            except Exception as exc:  # noqa: BLE001 - every failure must be reported
                self.state["status"] = "failed"
                self._write_state()
                result.status = "failed"
                result.failed = stage.value
                result.message = str(exc)
                log.error("Stage %s failed: %s", stage.value, exc)
                if not self.config.pipeline.continue_on_stage_failure:
                    return result
                log.warning("continuing_on_stage_failure=true - moving on despite failure")

        quality = read_json(self.paths.quality, default={}) or {}
        result.quality_status = str(quality.get("overall_status", ""))
        self.state["status"] = "complete"
        self.state["client_stats"] = self.client.stats()
        self._write_state()

        result.status = "complete"
        result.message = (
            f"{len(result.ran)} stage(s) ran, {len(result.skipped)} skipped"
        )
        log.info("Production complete: %s", self.paths.root)
        log.info("Model usage: %s", self.client.stats())
        return result

    def _select_stages(self, stages: Iterable[str] | None) -> list[Stage]:
        if not stages:
            selected = list(STAGE_ORDER)
            if not self.run_assets:
                selected = [stage for stage in selected if stage is not Stage.ASSETS]
            if not self.render_video:
                selected = [stage for stage in selected if stage is not Stage.VIDEO]
            if not self.run_tts:
                selected = [stage for stage in selected if stage is not Stage.TTS]
            return selected

        wanted = []
        for name in stages:
            try:
                wanted.append(Stage(str(name).lower()))
            except ValueError as exc:
                raise PipelineError(
                    f"Unknown stage '{name}'. Valid stages: "
                    + ", ".join(stage.value for stage in STAGE_ORDER)
                ) from exc
        return [stage for stage in STAGE_ORDER if stage in wanted]

    def _run_stage(self, stage: Stage) -> None:
        # Invalidate dependants *before* generating.  If this attempt fails
        # halfway through, the old downstream artefacts must not still look
        # current.
        self._invalidate_downstream(stage)
        record = self.record(stage)
        attempts = 0
        max_attempts = max(1, self.config.pipeline.max_stage_retries + 1)
        last_error: Exception | None = None

        while attempts < max_attempts:
            attempts += 1
            record.status = "running"
            record.attempts += 1
            record.started_at = utc_now_iso()
            record.error = None
            self._set_record(stage, record)
            started = time.time()
            log.info("")
            log.info("-- Stage: %s (attempt %d/%d) --", stage.value.upper(), attempts, max_attempts)

            try:
                self._handlers[stage]()
            except ApprovalRequired:
                record.status = "pending"
                record.duration_seconds = time.time() - started
                self._set_record(stage, record)
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
                record.duration_seconds = time.time() - started
                self._set_record(stage, record)
                if stage in RETRYABLE and attempts < max_attempts:
                    log.warning(
                        "Stage %s failed (attempt %d/%d): %s - retrying",
                        stage.value,
                        attempts,
                        max_attempts,
                        exc,
                    )
                    time.sleep(min(5.0 * attempts, 20.0))
                    continue
                raise

            record.status = "complete"
            record.finished_at = utc_now_iso()
            record.duration_seconds = time.time() - started
            record.artifacts = [
                path.name for path in self._artifact_paths(stage) if path.is_file()
            ]
            self._set_record(stage, record)
            log.info("Stage %s complete in %.1fs", stage.value, record.duration_seconds)
            return

        raise PipelineError(f"Stage {stage.value} failed after {attempts} attempts: {last_error}")

    # -- stage implementations ---------------------------------------------- #
    def _stage_connection(self) -> None:
        status: ServerStatus = self.client.probe()
        if not status.online:
            raise ServerNotAvailable(self.client.startup_hint())
        detail = (
            f"Connected to {status.base_url} ({status.latency_ms} ms), "
            f"context {status.context_size} tokens"
        )
        if status.context_size < self.config.model.min_context_tokens:
            log.warning(
                "Context window is only %d tokens (recommended >= %d). Long script sections "
                "will be generated in very small chunks.",
                status.context_size,
                self.config.model.min_context_tokens,
            )
        record = self.record(Stage.CONNECTION)
        record.notes = detail
        record.artifacts = []
        self._set_record(Stage.CONNECTION, record)
        log.info("%s", detail)

    def _stage_topic(self) -> None:
        analyzer = TopicAnalyzer(
            self.client, self.prompts, self.config.schemas_dir, self.profile, self.config.pipeline
        )
        brief = analyzer.analyze(
            self.topic,
            duration_minutes=self.profile.long_duration_minutes,
            shorts_count=self.profile.shorts_count,
            short_seconds=self.profile.short_duration_seconds,
        )
        atomic_write_json(self.paths.brief, brief)

    def _stage_research(self) -> None:
        brief = self._load("brief")
        engine = ResearchEngine(
            self.client, self.prompts, self.config.schemas_dir, self.profile, self.config.pipeline
        )
        queries = engine.build_queries(self.topic, brief)
        collection = engine.collect(queries)
        report = engine.analyze(self.topic, brief, collection)
        atomic_write_json(self.paths.research, report)
        if report.get("research_status") == "unavailable":
            log.warning(
                "Research is UNAVAILABLE. The script will be hedged and figures are blocked."
            )

    def _stage_plan(self) -> None:
        brief = self._load("brief")
        research = self._load("research")
        planner = ContentPlanner(
            self.client, self.prompts, self.config.schemas_dir, self.profile, self.config.pipeline
        )
        outline = planner.plan(
            brief=brief,
            research=research,
            duration_minutes=self.profile.long_duration_minutes,
        )
        atomic_write_json(self.paths.outline, outline)

    def _stage_script(self) -> None:
        brief = self._load("brief")
        research = self._load("research")
        outline = self._load("outline")
        target_words = int(
            round(self.profile.long_duration_minutes * self.config.pipeline.words_per_minute)
        )
        writer = ScriptWriter(
            self.client, self.prompts, self.config.schemas_dir, self.profile, self.config.pipeline
        )

        forced = self.force_all or Stage.SCRIPT.value in self.force_stages
        long_exists = self.paths.long_script.is_file()
        recorded_hash = str((self.state.get("artifacts") or {}).get("long_script_sha256") or "")
        current_hash = self._sha256_file(self.paths.long_script) if long_exists else ""

        # A human may edit Long.txt between runs.  Generated section data must
        # never silently overwrite that edit.  `--force script` is the explicit
        # way to ask for regeneration.
        if long_exists and not forced and current_hash != recorded_hash:
            log.warning(
                "Long.txt differs from the generated script (human edit detected). "
                "Treating the current file as authoritative."
            )
            self._accept_external_long(current_hash)
            return

        outline_hash = self._sha256_text(
            json.dumps(outline, sort_keys=True, ensure_ascii=False)
        )
        resume_state = None
        if not forced and self.paths.sections.is_file():
            candidate = read_json(self.paths.sections, default={}) or {}
            if candidate.get("outline_hash") == outline_hash and not candidate.get("external_edit"):
                resume_state = candidate
                log.info(
                    "Resuming long script from %d persisted part(s)",
                    len([part for part in candidate.get("parts", []) if part.get("written")]),
                )
            elif candidate:
                log.warning(
                    "Ignoring Long_Sections.json because the outline changed or the file "
                    "belongs to an external edit"
                )

        def checkpoint(snapshot: ScriptResult) -> None:
            # Persist after every section.  A crash at section five keeps 1-4.
            atomic_write_json(self.paths.sections, snapshot.to_dict())

        result = writer.write(
            brief=brief,
            research=research,
            outline=outline,
            target_words=target_words,
            resume_state=resume_state,
            on_progress=checkpoint,
        )
        self.file_manager.write_text(self.paths.long_script, result.text)
        atomic_write_json(self.paths.sections, result.to_dict())

        digest = self._sha256_file(self.paths.long_script)
        self.state.setdefault("artifacts", {})["long_script_sha256"] = digest
        self.state.setdefault("script", {})
        self.state["script"].update(
            {
                "outline_hash": result.outline_hash,
                "parts": len(result.parts),
                "words": result.total_words,
                "generated_at": utc_now_iso(),
                "source": "generated",
            }
        )
        self.state["script_source"] = "generated"
        self._write_state()
        log.info("Long.txt written: %d words", result.total_words)

    def _accept_external_long(self, digest: str) -> None:
        """Treat a human-edited Long.txt as authoritative.

        Stale generated sections are replaced by an explicit marker so a later
        resume cannot reconstruct the old text from them.
        """
        marker = {
            "schema_version": "1.1",
            "generated_at": utc_now_iso(),
            "outline_hash": "",
            "external_edit": True,
            "long_sha256": digest,
            "total_words": count_words(self.paths.long_script.read_text(encoding="utf-8")),
            "parts": [],
            "notes": [
                "Long.txt was edited outside PIXEL AI. The edited file is authoritative; "
                "generated section data was discarded. Use --force script to regenerate."
            ],
        }
        atomic_write_json(self.paths.sections, marker)
        self.state.setdefault("artifacts", {})["long_script_sha256"] = digest
        self.state.setdefault("script", {})
        self.state["script"].update(
            {
                "parts": 0,
                "words": marker["total_words"],
                "generated_at": utc_now_iso(),
                "source": "human_edited",
                "long_sha256": digest,
            }
        )
        self.state["script_source"] = "human_edited"
        record = self.record(Stage.SCRIPT)
        record.notes = "Human-edited Long.txt is authoritative; generated sections discarded"
        self._set_record(Stage.SCRIPT, record)

    def _stage_shorts(self) -> None:
        _ = self._load("brief")
        research = self._load("research")
        long_text = self._load("long_script")

        forced = self.force_all or Stage.SHORTS.value in self.force_stages
        short_exists = self.paths.short_script.is_file()
        recorded_hash = str((self.state.get("artifacts") or {}).get("short_script_sha256") or "")
        current_hash = (
            self._sha256_file(self.paths.short_script) if short_exists else ""
        )

        if short_exists and not forced and current_hash != recorded_hash:
            log.warning(
                "Short.txt differs from the generated Shorts (human edit detected). "
                "Treating the current file as authoritative."
            )
            parsed = parse_shorts_text(
                self.paths.short_script.read_text(encoding="utf-8")
            )
            if len(parsed) != self.profile.shorts_count:
                raise PipelineError(
                    f"Human-edited Short.txt parses as {len(parsed)} Short(s) but "
                    f"{self.profile.shorts_count} are required."
                )
            self.state.setdefault("artifacts", {})["short_script_sha256"] = current_hash
            self.state["shorts_source"] = "human_edited"
            record = self.record(Stage.SHORTS)
            record.notes = "Human-edited Short.txt is authoritative"
            self._set_record(Stage.SHORTS, record)
        else:
            writer = ShortsWriter(
                self.client, self.prompts, self.config.schemas_dir, self.profile, self.config.pipeline
            )
            result = writer.write(
                long_script=long_text,
                research=research,
                count=self.profile.shorts_count,
                short_seconds=self.profile.short_duration_seconds,
            )
            self.file_manager.write_text(self.paths.short_script, result.text)

            parsed = parse_shorts_text(result.text)
            if len(parsed) != self.profile.shorts_count:
                raise PipelineError(
                    f"Short.txt parsed as {len(parsed)} Short(s) but {self.profile.shorts_count} "
                    "were required. The heading format is incompatible with the canonical parser."
                )
            for issue in result.issues:
                log.warning("Shorts: %s", issue)
            self.state.setdefault("artifacts", {})["short_script_sha256"] = self._sha256_file(
                self.paths.short_script
            )
            self.state["shorts_source"] = "generated"
            self._write_state()

        # Approval happens here, after *both* Long.txt and Short.txt exist.
        if self.config.pipeline.require_approval_after_script and not self._is_script_approved():
            self._request_approval()

    def _stage_visuals(self) -> None:
        self._require(Stage.SCRIPT, Stage.SHORTS)
        if self.config.pipeline.require_approval_after_script and not self._is_script_approved():
            self._request_approval()
        brief = self._load("brief")
        long_text = self._load("long_script")
        shorts_text = self._load("short_script")
        parsed = parse_shorts_text(shorts_text)
        if not parsed:
            raise PipelineError("Short.txt could not be parsed - cannot build Short storyboards")

        director = VisualDirector(
            self.client, self.prompts, self.config.schemas_dir, self.profile, self.config.pipeline
        )

        long_plan = director.direct_long(
            narration=long_text,
            brief=brief,
            target_seconds=self.profile.long_duration_minutes * 60,
        )
        long_plan["project"]["topic"] = self.topic
        long_plan["project"]["slug"] = self.paths.slug
        atomic_write_json(self.paths.long_plan, long_plan)

        shorts_payload = [
            {
                "index": short.index,
                "title": short.title,
                "narration": short.narration,
                "central_idea": "",
            }
            for short in parsed
        ]
        shorts_plan = director.direct_shorts(shorts=shorts_payload, brief=brief)
        shorts_plan["project"]["topic"] = self.topic
        shorts_plan["project"]["slug"] = self.paths.slug
        shorts_plan["generated_at"] = utc_now_iso()
        atomic_write_json(self.paths.shorts_plan, shorts_plan)
        log.info(
            "Storyboards written: %d long scene(s), %d Short(s)",
            len(long_plan["scenes"]),
            len(shorts_plan["shorts"]),
        )

    def _stage_audio(self) -> None:
        self._require(Stage.VISUALS)
        long_plan = self._load("long_plan")
        shorts_plan = self._load("shorts_plan")

        director = AudioDirector(
            self.client, self.prompts, self.config.schemas_dir, self.profile, self.config.pipeline
        )
        scenes = long_plan.get("scenes", []) or []
        total_seconds = sum(
            float(scene.get("estimated_duration_seconds", 0)) for scene in scenes
        )
        long_audio = director.direct(scenes=scenes, total_seconds=total_seconds, label="long")
        AudioDirector.attach(long_plan, long_audio)
        atomic_write_json(self.paths.long_plan, long_plan)
        atomic_write_json(self.paths.audio_plan, long_audio)

        for short in shorts_plan.get("shorts", []) or []:
            short_scenes = short.get("scenes", []) or []
            if not short_scenes:
                continue
            short_seconds = sum(
                float(scene.get("estimated_duration_seconds", 0)) for scene in short_scenes
            )
            audio = director.direct(
                scenes=short_scenes,
                total_seconds=short_seconds,
                label=f"short{short.get('short_id')}",
            )
            short["audio_plan"] = audio

        atomic_write_json(self.paths.shorts_plan, shorts_plan)
        log.info("Audio direction attached to long plan and %d Short(s)", len(shorts_plan.get("shorts", [])))

    def _stage_quality(self) -> None:
        self._require(Stage.SCRIPT, Stage.SHORTS)
        brief = self._load("brief")
        research = self._load("research")
        outline = self._load("outline")
        long_text = self._load("long_script")
        shorts_text = self._load("short_script")
        parsed = parse_shorts_text(shorts_text)
        shorts = [
            {
                "index": short.index,
                "title": short.title,
                "narration": short.narration,
                "central_idea": "",
            }
            for short in parsed
        ]

        long_plan = self._load("long_plan", required=False)
        shorts_plan = self._load("shorts_plan", required=False)
        audio_plan = self._load("audio_plan", required=False)

        controller = QualityController(
            self.client, self.prompts, self.config.schemas_dir, self.profile, self.config.pipeline
        )
        report = controller.review(
            ReviewInputs(
                brief=brief,
                research=research,
                outline=outline,
                long_script=long_text,
                shorts=shorts,
                long_plan=long_plan,
                shorts_plan=shorts_plan,
                audio_plan=audio_plan,
                paths=self.paths,
                target_minutes=self.profile.long_duration_minutes,
            )
        )
        atomic_write_json(self.paths.quality, report)

        self.state["quality"] = {
            "status": report.get("overall_status"),
            "generated_at": report.get("generated_at"),
            "summary": report.get("summary"),
        }
        self._write_state()

        overall = str(report.get("overall_status") or "").lower()
        threshold = self.config.pipeline.quality_fail_on
        if threshold not in ("error", "warning", "never"):
            log.warning("Unknown quality.fail_on=%r; using 'error'", threshold)
            threshold = "error"

        blocks = (
            threshold == "error" and overall == "fail"
        ) or (
            threshold == "warning" and overall in ("warn", "fail")
        )
        if blocks:
            raise QualityGateError(
                f"Quality gate blocked production: overall status is {overall.upper()} "
                f"with quality.fail_on={threshold}. Inspect "
                f"{self.paths.quality.name}, fix the failing content, then resume. "
                "Set quality.fail_on='never' only if the gate should be advisory."
            )
        if overall == "fail":
            log.error("Quality review returned FAIL. See %s", self.paths.quality.name)
        elif overall == "warn":
            log.warning("Quality review returned WARN. See %s", self.paths.quality.name)

    def _stage_save(self) -> None:
        tts_report = read_json(self.paths.tts_metadata, default={}) or {}
        manifest = {
            "schema_version": "1.1",
            "profile_id": self.profile.id,
            "profile_name": self.profile.name,
            "topic": self.topic,
            "slug": self.paths.slug,
            "created_at": self.state.get("created_at"),
            "updated_at": utc_now_iso(),
            "long_duration_minutes": self.profile.long_duration_minutes,
            "shorts_count": self.profile.shorts_count,
            "language": self.profile.language,
            "script_source": self.state.get("script_source", "generated"),
            "shorts_source": self.state.get("shorts_source", "generated"),
            "approval": self.state.get("approval"),
            "artifacts": {},
            "tts": {
                "engine": "internal-kokoro",
                "metadata": ARTIFACTS["tts"],
                "folder": f"{self.profile.id}/{self.paths.slug}",
                "long_audio": str(self.paths.tts_long_audio.relative_to(self.paths.root)).replace("\\", "/"),
                "shorts_audio_pattern": "output/Shorts/Short_NN/narration.wav",
                "long_duration_seconds": (tts_report.get("long") or {}).get("duration_seconds"),
                "short_durations_seconds": {
                    str(entry.get("index")): entry.get("duration_seconds")
                    for entry in tts_report.get("shorts", []) or []
                },
            },
        }

        for key in (
            "brief",
            "research",
            "outline",
            "long_script",
            "short_script",
            "long_plan",
            "shorts_plan",
            "audio_plan",
            "quality",
            "tts",
            "assets_status",
        ):
            path = self.paths.root / ARTIFACTS[key]
            if path.is_file():
                manifest["artifacts"][key] = {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                }

        long_path = self.paths.long_script
        if long_path.is_file():
            text = long_path.read_text(encoding="utf-8")
            short_count = 0
            if self.paths.short_script.is_file():
                short_count = len(
                    parse_shorts_text(self.paths.short_script.read_text(encoding="utf-8"))
                )
            manifest["metrics"] = {
                "long_words": count_words(text),
                "estimated_long_seconds": round(
                    count_words(text) / self.config.pipeline.words_per_minute * 60, 1
                ),
                "actual_long_seconds": (tts_report.get("long") or {}).get(
                    "duration_seconds"
                ),
                "shorts": short_count,
            }

        atomic_write_json(self.paths.manifest, manifest)
        log.info("Project manifest written")

    def _stage_validate(self) -> None:
        from engine.output_validator import validate_project

        result = validate_project(self.paths, self.config.schemas_dir, self.profile)
        self.state["validation"] = result.to_dict()
        self._write_state()

        for warning in result.warnings:
            log.warning("Validation: %s", warning)

        if result.failures:
            raise PipelineError(
                "Output validation failed:\n  - " + "\n  - ".join(result.failures)
            )
        log.info(
            "Validated %d artefact(s): all schemas and formats pass",
            len(result.checked),
        )

    def _stage_assets(self) -> None:
        self._require(Stage.VISUALS)
        manager = AssetManager(self.config.pipeline, self.paths, self.profile.id)

        if not manager.any_provider_configured():
            log.warning(
                "No asset provider is configured. Assets will be marked pending; "
                "nothing will be generated or faked."
            )

        long_plan = self._load("long_plan")
        summary = manager.ensure_assets(long_plan, dry_run=self.asset_dry_run)
        atomic_write_json(self.paths.long_plan, long_plan)

        shorts_plan = self._load("shorts_plan")
        for short in shorts_plan.get("shorts", []) or []:
            plan_stub = {
                "project": {
                    "format": "short",
                    "aspect_ratio": self.profile.short_aspect,
                },
                "scenes": short.get("scenes", []),
            }
            manager.ensure_assets(plan_stub, dry_run=self.asset_dry_run)
            short["scenes"] = plan_stub["scenes"]
        atomic_write_json(self.paths.shorts_plan, shorts_plan)

        # The manager merges long + Short entries into one status document.
        merged = read_json(self.paths.assets_status, default={}) or {}
        counts = merged.get("counts", {})
        log.info(
            "Assets: %d ready, %d pending, %d failed (%d not required)",
            counts.get("ready", 0),
            counts.get("pending", 0),
            counts.get("failed", 0),
            counts.get("not_required", 0) + int(merged.get("not_required", 0) or 0),
        )

    # -- TTS ---------------------------------------------------------------- #
    def _stage_tts(self) -> None:
        self._require(Stage.SCRIPT, Stage.SHORTS, Stage.VISUALS)
        if not self.run_tts:
            raise PipelineError(
                "TTS is disabled. Set tts.enabled=true in config/pipeline_settings.json "
                "or run with --tts to generate narration audio."
            )

        forced = self.force_all or Stage.TTS.value in self.force_stages
        report = generate_project_audio(
            self.paths, self.config.pipeline, force=forced
        )

        # Long-form timeline + pauses.
        long_plan = self._load("long_plan")
        long_entry = report["long"]
        self._apply_audio_to_plan(
            plan=long_plan,
            base_path=self.paths.tts_long_base,
            narration_path=self.paths.tts_long_audio,
            entry=long_entry,
            label="long",
        )
        atomic_write_json(self.paths.long_plan, long_plan)

        # Per-Short timelines + pauses.
        shorts_plan = self._load("shorts_plan")
        short_by_index = {
            int(short.get("short_id")): short
            for short in shorts_plan.get("shorts", []) or []
            if str(short.get("short_id", "")).isdigit()
        }
        for entry in report["shorts"]:
            index = int(entry["index"])
            short = short_by_index.get(index)
            if short is None:
                log.warning("TTS generated Short %02d but no matching storyboard exists", index)
                continue
            self._apply_audio_to_plan(
                plan=short,
                base_path=self.paths.tts_short_base(index),
                narration_path=self.paths.tts_short_audio(index),
                entry=entry,
                label=f"short{index:02d}",
            )
        atomic_write_json(self.paths.shorts_plan, shorts_plan)

        report["applied_at"] = utc_now_iso()
        atomic_write_json(self.paths.tts_metadata, report)
        self.state["tts"] = {
            "engine": "kokoro",
            "voice": report["config"]["voice"],
            "speed": report["config"]["speed"],
            "long_duration_seconds": report["long"].get("duration_seconds"),
            "shorts": {
                str(entry["index"]): entry.get("duration_seconds")
                for entry in report["shorts"]
            },
            "generated_at": report.get("generated_at"),
        }
        self._write_state()
        log.info(
            "TTS timeline reconciled: long %.1fs, %d Short(s)",
            float(report["long"].get("duration_seconds", 0)),
            len(report["shorts"]),
        )

    def _apply_audio_to_plan(
        self,
        *,
        plan: dict[str, Any],
        base_path: Path,
        narration_path: Path,
        entry: dict[str, Any],
        label: str,
    ) -> None:
        """Restore raw audio, insert planned pauses, reconcile scene durations."""
        if not base_path.is_file():
            if not narration_path.is_file():
                raise PipelineError(f"Missing {label} narration audio: {narration_path}")
            shutil.copy2(narration_path, base_path)
        shutil.copy2(base_path, narration_path)

        raw_duration = float(entry.get("duration_seconds") or wav_duration(base_path))
        audio_plan = plan.get("audio_plan") or {}
        pauses = audio_plan.get("pauses", []) or []
        points = pause_insertion_points(plan, raw_duration, pauses)
        final_duration = insert_pauses(narration_path, points) if points else raw_duration

        reconcile_plan(plan, final_duration, pauses=pauses)
        entry.update(
            {
                "raw_duration_seconds": round(raw_duration, 3),
                "duration_seconds": round(final_duration, 3),
                "pauses_inserted": len(points),
                "base_narration": str(base_path.relative_to(self.paths.root)).replace("\\", "/"),
            }
        )
        log.info(
            "%s narration: %.2fs raw, %d pause(s), %.2fs final",
            label,
            raw_duration,
            len(points),
            final_duration,
        )

    def _stage_video(self) -> None:
        self._require(Stage.VISUALS, Stage.TTS)
        if self.run_assets:
            self._require(Stage.ASSETS)
        assembler = VideoAssembler(self.config.pipeline, self.paths)
        assembler.allow_render = True
        long_plan = self._load("long_plan")
        long_report = assembler.render_long(long_plan)
        reports: dict[str, dict[str, Any]] = {"long": long_report.to_dict()}
        if not long_report.ok:
            self.state["renders"] = reports
            self._write_state()
            raise PipelineError(
                "Long video assembly could not run: "
                + ("; ".join(long_report.errors) if long_report.errors else "")
                + (" | missing: " + ", ".join(long_report.missing[:8]) if long_report.missing else "")
            )

        shorts_plan = self._load("shorts_plan")
        failures: list[str] = []
        for short in shorts_plan.get("shorts", []) or []:
            short_id = str(short.get("short_id", "")).strip()
            if not short_id.isdigit():
                failures.append(f"Short with invalid short_id: {short_id!r}")
                continue
            wrapper = {
                "project": {
                    "format": "short",
                    "aspect_ratio": self.profile.short_aspect,
                },
                "scenes": short.get("scenes", []),
                "audio_plan": short.get("audio_plan", {}),
            }
            report = assembler.render_short(wrapper, short_id=f"{int(short_id):02d}")
            reports[f"short_{int(short_id):02d}"] = report.to_dict()
            if not report.ok:
                failures.append(
                    f"Short {int(short_id):02d}: "
                    + ("; ".join(report.errors) if report.errors else "missing assets")
                )
        self.state["renders"] = reports
        self._write_state()
        if failures:
            raise PipelineError("Short video assembly failed: " + " | ".join(failures[:6]))
        log.info("Rendered long video and %d Short(s)", len(reports) - 1)

    # -- approval ----------------------------------------------------------- #
    def _script_hashes(self) -> tuple[str, str, str]:
        if not self.paths.long_script.is_file() or not self.paths.short_script.is_file():
            return "", "", ""
        long_hash = self._sha256_file(self.paths.long_script)
        short_hash = self._sha256_file(self.paths.short_script)
        content_hash = self._sha256_text(f"{long_hash}:{short_hash}")
        return long_hash, short_hash, content_hash

    def _is_script_approved(self) -> bool:
        approval = self.state.get("approval") or {}
        if not approval.get("script_approved"):
            return False
        _long_hash, _short_hash, content_hash = self._script_hashes()
        return bool(content_hash) and approval.get("content_hash") == content_hash

    def _clear_approval(self, reason: str) -> None:
        approval = self.state.setdefault("approval", {})
        was_approved = bool(approval.get("script_approved"))
        approval.update(
            {
                "script_approved": False,
                "approved_at": None,
                "mode": None,
                "content_hash": None,
                "long_sha256": None,
                "short_sha256": None,
                "invalidated_at": utc_now_iso(),
                "invalidation_reason": reason,
            }
        )
        if was_approved:
            log.warning("Script approval invalidated: %s", reason)

    def _check_approval_freshness(self) -> None:
        approval = self.state.get("approval") or {}
        if not approval.get("script_approved"):
            return
        _long_hash, _short_hash, content_hash = self._script_hashes()
        if content_hash and approval.get("content_hash") == content_hash:
            return
        self._clear_approval("Long.txt or Short.txt changed after approval")
        for stage in (
            Stage.VISUALS,
            Stage.AUDIO,
            Stage.QUALITY,
            Stage.SAVE,
            Stage.VALIDATE,
            Stage.TTS,
            Stage.ASSETS,
            Stage.VIDEO,
        ):
            record = self.record(stage)
            if record.status != "pending":
                record.status = "pending"
                record.notes = "Approval no longer matches the current narration"
                self._set_record(stage, record)
            self._purge_derived(stage)

    def approve_script(self, *, mode: str = "cli") -> None:
        long_hash, short_hash, content_hash = self._script_hashes()
        if not content_hash:
            raise PipelineError(
                "Cannot approve: both Long.txt and Short.txt must exist first."
            )
        self.state.setdefault("approval", {})
        self.state["approval"].update(
            {
                "script_approved": True,
                "approved_at": utc_now_iso(),
                "mode": mode,
                "content_hash": content_hash,
                "long_sha256": long_hash,
                "short_sha256": short_hash,
                "auto_bypassed": mode == "auto",
                "invalidated_at": None,
                "invalidation_reason": None,
            }
        )
        self._write_state()

    def _request_approval(self) -> None:
        if self._is_script_approved():
            return
        if not self.paths.long_script.is_file() or not self.paths.short_script.is_file():
            raise PipelineError(
                "Approval requires both Long.txt and Short.txt. Run the script and "
                "shorts stages first."
            )

        if self.auto_approve:
            self.approve_script(mode="auto")
            log.info("Script auto-approved (--auto-approve); bypass recorded in state")
            return
        if self.preapprove:
            self.approve_script(mode="cli_preapproved")
            log.info("Script pre-approved by --approve; current content hash recorded")
            return

        long_text = self._load("long_script")
        short_text = self._load("short_script", required=False)
        words = count_words(long_text)
        shorts = len(parse_shorts_text(short_text)) if short_text else 0
        _long_hash, _short_hash, content_hash = self._script_hashes()

        banner = (
            "\n"
            "==============================================================\n"
            "  SCRIPT READY FOR REVIEW  (Long + Shorts)\n"
            "==============================================================\n"
            f"  Channel : {self.profile.name} ({self.profile.id})\n"
            f"  Topic   : {self.topic}\n"
            f"  Long    : {words} words "
            f"(~{words / self.config.pipeline.words_per_minute:.1f} min at "
            f"{self.config.pipeline.words_per_minute} wpm)\n"
            f"  Shorts  : {shorts}\n"
            f"  Version : {content_hash[:12]}\n"
            f"  Folder  : {self.paths.root}\n"
            "\n"
            f"  Open the script with:\n"
            f'    notepad "{self.paths.long_script}"\n'
            f'    notepad "{self.paths.short_script}"\n'
            "\n"
            "  Editing either file after approval invalidates this approval.\n"
            "==============================================================\n"
        )
        print(banner)

        if not self.config.pipeline.require_approval_after_script:
            self.approve_script(mode="checkpoint_disabled")
            return

        try:
            answer = input("Approve this script and continue to visual production? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""

        if answer in ("y", "yes", "approve", "ok"):
            self.approve_script(mode="cli")
            log.info("Script approved - continuing")
            return

        raise ApprovalRequired(
            "Script approval was declined. Review the files, then resume with:\n"
            f'  python main.py resume --project {self.profile.id}/{self.paths.slug} --approve'
        )

    # -- introspection ------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        rows = []
        for stage in STAGE_ORDER:
            record = self.record(stage)
            rows.append(
                {
                    "stage": stage.value,
                    "status": record.status,
                    "attempts": record.attempts,
                    "duration_seconds": round(record.duration_seconds, 1),
                    "error": record.error,
                    "artifacts": record.artifacts,
                }
            )
        return {
            "project": f"{self.profile.id}/{self.paths.slug}",
            "topic": self.topic,
            "root": str(self.paths.root),
            "overall": self.state.get("status"),
            "approval": self.state.get("approval"),
            "stages": rows,
            "client": self.client.stats(),
        }

    def close(self) -> None:
        self.client.close()


__all__ = [
    "ApprovalRequired",
    "DOWNSTREAM",
    "Pipeline",
    "PipelineError",
    "PipelineResult",
    "QualityGateError",
    "STAGE_ORDER",
    "ServerNotAvailable",
    "Stage",
    "StageRecord",
]
