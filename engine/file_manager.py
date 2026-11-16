"""Filesystem layout and safe file I/O for PIXEL AI.

Every generated artefact lives inside a *project directory*::

    scripts/<profile_id>/<topic-slug>/
        Long.txt
        Short.txt
        ContentBrief.json
        ...
        assets/{images,videos,sounds,music}/

``FileManager`` owns path construction, atomic writes and state persistence so
no other module has to assemble paths by hand.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from engine.logging_setup import get_logger

log = get_logger("files")

RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

SLUG_MAX_LENGTH = 64


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #


def slugify(text: str, *, max_length: int = SLUG_MAX_LENGTH) -> str:
    """Turn a topic into a filesystem-safe, deterministic slug.

    ``"The Rise of Artificial Intelligence"`` -> ``"the-rise-of-artificial-intelligence"``
    """
    if not text or not text.strip():
        raise ValueError("Cannot build a slug from an empty string")

    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    ascii_text = re.sub(r"-{2,}", "-", ascii_text).strip("-")

    if not ascii_text:
        # Non-latin topic: fall back to a short hash so the folder is still unique.
        import hashlib

        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        ascii_text = f"topic-{digest}"

    if len(ascii_text) > max_length:
        ascii_text = ascii_text[:max_length].rstrip("-")

    if ascii_text.upper() in RESERVED_WINDOWS_NAMES:
        ascii_text = f"{ascii_text}-topic"

    return ascii_text


def utc_now_iso() -> str:
    """Timezone-aware ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically so an interrupted run never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=encoding, newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Serialise *data* as UTF-8 JSON atomically."""
    text = json.dumps(data, indent=indent, ensure_ascii=False) + "\n"
    atomic_write_text(path, text)


def read_json(path: Path, default: Any = None) -> Any:
    """Read UTF-8 JSON, returning *default* when the file is missing."""
    if not path.is_file():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        log.error("Corrupt JSON at %s: %s", path, exc)
        return default


# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #


ASSET_SUBDIRS = ("images", "videos", "sounds", "music", "overlays", "subtitles")

#: Canonical artefact filenames inside a project folder.
ARTIFACTS = {
    "brief": "ContentBrief.json",
    "research": "Research.json",
    "outline": "Outline.json",
    "long_script": "Long.txt",
    "short_script": "Short.txt",
    "long_plan": "Long_Production.json",
    "shorts_plan": "Shorts_Production.json",
    "audio_plan": "AudioPlan.json",
    "quality": "QualityReport.json",
    "state": "production_state.json",
    "assets_status": "asset_status.json",
    "script_sections": "Long_Sections.json",
    "manifest": "project.json",
    "tts": "TTS.json",
    "visual_progress": "Visual_Progress.json",
}


@dataclass
class ProjectPaths:
    """All paths belonging to one production project."""

    root: Path
    profile_id: str
    topic: str
    slug: str

    # -- core files -------------------------------------------------------- #
    @property
    def long_script(self) -> Path:
        return self.root / ARTIFACTS["long_script"]

    @property
    def short_script(self) -> Path:
        return self.root / ARTIFACTS["short_script"]

    @property
    def brief(self) -> Path:
        return self.root / ARTIFACTS["brief"]

    @property
    def research(self) -> Path:
        return self.root / ARTIFACTS["research"]

    @property
    def outline(self) -> Path:
        return self.root / ARTIFACTS["outline"]

    @property
    def long_plan(self) -> Path:
        return self.root / ARTIFACTS["long_plan"]

    @property
    def shorts_plan(self) -> Path:
        return self.root / ARTIFACTS["shorts_plan"]

    @property
    def audio_plan(self) -> Path:
        return self.root / ARTIFACTS["audio_plan"]

    @property
    def quality(self) -> Path:
        return self.root / ARTIFACTS["quality"]

    @property
    def state(self) -> Path:
        return self.root / ARTIFACTS["state"]

    @property
    def assets_status(self) -> Path:
        return self.root / ARTIFACTS["assets_status"]

    @property
    def sections(self) -> Path:
        return self.root / ARTIFACTS["script_sections"]

    @property
    def manifest(self) -> Path:
        return self.root / ARTIFACTS["manifest"]

    @property
    def tts_metadata(self) -> Path:
        return self.root / ARTIFACTS["tts"]

    @property
    def visual_progress(self) -> Path:
        return self.root / ARTIFACTS["visual_progress"]

    # -- directories ------------------------------------------------------- #
    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def images(self) -> Path:
        return self.assets / "images"

    @property
    def videos(self) -> Path:
        return self.assets / "videos"

    @property
    def sounds(self) -> Path:
        return self.assets / "sounds"

    @property
    def music(self) -> Path:
        return self.assets / "music"

    @property
    def overlays(self) -> Path:
        return self.assets / "overlays"

    @property
    def subtitles(self) -> Path:
        return self.assets / "subtitles"

    @property
    def renders(self) -> Path:
        return self.root / "renders"

    # -- Kokoro TTS output (produced by the existing legacy script) --------- #
    @property
    def tts_output(self) -> Path:
        """The ``output/`` folder the existing Kokoro script writes into."""
        return self.root / "output"

    @property
    def tts_long_audio(self) -> Path:
        return self.tts_output / "Long" / "narration.wav"

    @property
    def tts_long_base(self) -> Path:
        """Raw Kokoro output before pauses are inserted (idempotency anchor)."""
        return self.tts_output / "Long" / "narration_base.wav"

    @property
    def tts_long_cleaned(self) -> Path:
        return self.tts_output / "Long" / "cleaned_script.txt"

    def tts_short_dir(self, index: int) -> Path:
        return self.tts_output / "Shorts" / f"Short_{index:02d}"

    def tts_short_audio(self, index: int) -> Path:
        return self.tts_short_dir(index) / "narration.wav"

    def tts_short_base(self, index: int) -> Path:
        return self.tts_short_dir(index) / "narration_base.wav"

    def tts_short_cleaned(self, index: int) -> Path:
        return self.tts_short_dir(index) / "cleaned_script.txt"

    def ensure(self) -> None:
        """Create every directory the project may need."""
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.assets,
            self.images,
            self.videos,
            self.sounds,
            self.music,
            self.overlays,
            self.subtitles,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def all_json(self) -> Iterable[Path]:
        for relative in (
            ARTIFACTS["brief"],
            ARTIFACTS["research"],
            ARTIFACTS["outline"],
            ARTIFACTS["long_plan"],
            ARTIFACTS["shorts_plan"],
            ARTIFACTS["audio_plan"],
            ARTIFACTS["quality"],
            ARTIFACTS["state"],
        ):
            yield self.root / relative


@dataclass
class FileManager:
    """Owns the ``scripts/`` tree and per-project file operations."""

    scripts_dir: Path
    assets_subdir: str = "assets"
    _projects: dict[str, ProjectPaths] = field(default_factory=dict, repr=False)

    # -- discovery --------------------------------------------------------- #
    def ensure_root(self) -> None:
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    def profile_dir(self, profile_id: str) -> Path:
        return self.scripts_dir / profile_id

    def project_dir(self, profile_id: str, topic: str) -> Path:
        return self.profile_dir(profile_id) / slugify(topic)

    def project_exists(self, profile_id: str, topic: str) -> bool:
        return self.project_dir(profile_id, topic).is_dir()

    def list_projects(self, profile_id: str | None = None) -> list[ProjectPaths]:
        """Discover existing production projects."""
        found: list[ProjectPaths] = []
        if not self.scripts_dir.is_dir():
            return found
        profile_dirs = (
            [self.profile_dir(profile_id)] if profile_id else sorted(self.scripts_dir.iterdir())
        )
        for pdir in profile_dirs:
            if not pdir.is_dir():
                continue
            for tdir in sorted(pdir.iterdir()):
                if not tdir.is_dir():
                    continue
                if not (tdir / ARTIFACTS["state"]).is_file() and not (tdir / ARTIFACTS["long_script"]).is_file():
                    continue
                manifest = read_json(tdir / ARTIFACTS["manifest"], default={}) or {}
                topic = manifest.get("topic") or tdir.name.replace("-", " ").title()
                found.append(
                    ProjectPaths(
                        root=tdir,
                        profile_id=pdir.name,
                        topic=topic,
                        slug=tdir.name,
                    )
                )
        return found

    # -- creation ---------------------------------------------------------- #
    def create_project(
        self,
        profile_id: str,
        topic: str,
        *,
        allow_existing: bool = True,
    ) -> ProjectPaths:
        """Create (or reopen) the folder for one production."""
        slug = slugify(topic)
        root = self.profile_dir(profile_id) / slug

        if root.exists() and not allow_existing:
            raise FileExistsError(
                f"Project already exists: {root}\n"
                "Use --resume to continue it, or choose a different topic."
            )

        paths = ProjectPaths(root=root, profile_id=profile_id, topic=topic, slug=slug)
        existed = root.exists()
        paths.ensure()
        if not existed:
            log.info("Created project directory %s", root)
        self._projects[str(root)] = paths
        return paths

    def get_project(self, profile_id: str, topic: str) -> ProjectPaths:
        key = str(self.project_dir(profile_id, topic))
        if key in self._projects:
            return self._projects[key]
        paths = ProjectPaths(
            root=Path(key),
            profile_id=profile_id,
            topic=topic,
            slug=slugify(topic),
        )
        self._projects[key] = paths
        return paths

    def resolve(self, reference: str) -> ProjectPaths:
        """Resolve a project from a folder path, ``profile/slug`` or bare slug."""
        candidate = Path(reference)
        if candidate.is_absolute() and candidate.is_dir():
            root = candidate
            profile = root.parent.name
            return ProjectPaths(root=root, profile_id=profile, topic=root.name, slug=root.name)

        parts = reference.replace("\\", "/").strip("/").split("/")
        if len(parts) == 2:
            root = self.scripts_dir / parts[0] / parts[1]
        elif len(parts) == 1:
            matches = [p for p in self.list_projects() if p.slug == parts[0]]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous project '{reference}': "
                    + ", ".join(f"{m.profile_id}/{m.slug}" for m in matches)
                )
            raise FileNotFoundError(f"No project named '{reference}' under {self.scripts_dir}")
        else:
            root = self.scripts_dir / Path(*parts)

        if not root.is_dir():
            raise FileNotFoundError(f"Project not found: {root}")
        manifest = read_json(root / ARTIFACTS["manifest"], default={}) or {}
        return ProjectPaths(
            root=root,
            profile_id=root.parent.name,
            topic=manifest.get("topic") or root.name,
            slug=root.name,
        )

    # -- artefact helpers -------------------------------------------------- #
    @staticmethod
    def write_text(path: Path, text: str) -> Path:
        atomic_write_text(path, text)
        return path

    @staticmethod
    def write_json(path: Path, data: Any) -> Path:
        atomic_write_json(path, data)
        return path

    @staticmethod
    def read_json(path: Path, default: Any = None) -> Any:
        return read_json(path, default)

    @staticmethod
    def read_text(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")
        return path.read_text(encoding="utf-8")


__all__ = [
    "ASSET_SUBDIRS",
    "ARTIFACTS",
    "FileManager",
    "ProjectPaths",
    "atomic_write_json",
    "atomic_write_text",
    "read_json",
    "slugify",
    "utc_now_iso",
]
