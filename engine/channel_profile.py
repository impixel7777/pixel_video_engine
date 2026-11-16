"""Channel profile system.

A profile is the *identity* of a YouTube channel.  It is validated on load and
then rendered into every AI prompt, so the same engine produces a technology
video for ``pixel_tech`` and a documentary for ``pixel_history``.

Adding a channel means dropping a new ``profiles/<id>.json`` file into the
folder — no engine code changes required.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from engine.file_manager import read_json
from engine.logging_setup import get_logger

log = get_logger("profiles")

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "channel_profile.json"


class ProfileError(ValueError):
    """Raised when a channel profile is missing, invalid or ambiguous."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChannelProfile:
    """A validated, read-only view over one ``profiles/<id>.json`` file."""

    raw: dict[str, Any]
    path: Path = field(repr=False)

    # -- identity ---------------------------------------------------------- #
    @property
    def id(self) -> str:
        return str(self.raw["channel"]["id"])

    @property
    def name(self) -> str:
        return str(self.raw["channel"]["name"])

    @property
    def category(self) -> str:
        return str(self.raw["channel"].get("category", "General"))

    @property
    def language(self) -> str:
        return str(self.raw["channel"].get("language", "English"))

    @property
    def description(self) -> str:
        return str(self.raw["channel"].get("description", ""))

    @property
    def tagline(self) -> str:
        return str(self.raw["channel"].get("tagline", ""))

    # -- audience ---------------------------------------------------------- #
    @property
    def audience(self) -> dict[str, Any]:
        return dict(self.raw.get("audience", {}))

    # -- writing ----------------------------------------------------------- #
    @property
    def writing(self) -> dict[str, Any]:
        return dict(self.raw.get("writing", {}))

    @property
    def avoid(self) -> list[str]:
        return list(self.writing.get("avoid", []))

    @property
    def prefer(self) -> list[str]:
        return list(self.writing.get("prefer", []))

    # -- video ------------------------------------------------------------- #
    @property
    def video(self) -> dict[str, Any]:
        return dict(self.raw.get("video", {}))

    @property
    def long_duration_minutes(self) -> int:
        return int(self.video.get("long_duration_minutes", 10))

    @property
    def shorts_count(self) -> int:
        return int(self.video.get("shorts_count", 5))

    @property
    def short_duration_seconds(self) -> int:
        return int(self.video.get("short_duration_seconds", 45))

    @property
    def long_aspect(self) -> str:
        return str(self.video.get("aspect_ratio_long", "16:9"))

    @property
    def short_aspect(self) -> str:
        return str(self.video.get("aspect_ratio_short", "9:16"))

    # -- visuals ----------------------------------------------------------- #
    @property
    def visuals(self) -> dict[str, Any]:
        return dict(self.raw.get("visuals", {}))

    # -- audio ------------------------------------------------------------- #
    @property
    def audio(self) -> dict[str, Any]:
        return dict(self.raw.get("audio", {}))

    # -- topics ------------------------------------------------------------ #
    @property
    def topics(self) -> dict[str, Any]:
        return dict(self.raw.get("topics", {}))

    # -- prompt rendering -------------------------------------------------- #
    def to_prompt_block(self) -> str:
        """Render the profile as an instruction block for AI prompts.

        Only facts that must influence generation are included.  This is the
        single place where profile data becomes prompt text, which keeps the
        prompts consistent between stages.
        """
        writing = self.writing
        visuals = self.visuals
        audio = self.audio
        audience = self.audience
        video = self.video

        avoid = "\n".join(f"  - {item}" for item in self.avoid) or "  - (none configured)"
        prefer = "\n".join(f"  - {item}" for item in self.prefer) or "  - (none configured)"
        avoid_visuals = "\n".join(
            f"  - {item}" for item in visuals.get("avoid_visuals", [])
        ) or "  - (none configured)"
        colours = ", ".join(visuals.get("primary_colors", [])) or "not specified"

        lines = [
            f"CHANNEL IDENTITY",
            f"- Channel: {self.name} ({self.id})",
            f"- Category: {self.category}",
            f"- Language: {self.language}",
            f"- Description: {self.description}",
        ]
        if self.tagline:
            lines.append(f"- Tagline: {self.tagline}")

        lines += [
            "",
            "AUDIENCE",
            f"- Type: {audience.get('type', 'General audience')}",
            f"- Knowledge level: {audience.get('knowledge_level', 'Beginner')}",
            f"- Age group: {audience.get('age_group', 'General')}",
        ]
        if audience.get("what_they_know"):
            lines.append(f"- They already understand: {audience['what_they_know']}")
        if audience.get("what_they_want"):
            lines.append(f"- What they want from this video: {audience['what_they_want']}")

        lines += [
            "",
            "WRITING RULES (mandatory)",
            f"- Tone: {writing.get('tone', 'Engaging')}",
            f"- Style: {writing.get('style', 'Educational')}",
            f"- Complexity: {writing.get('complexity', 'Easy to understand')}",
            f"- Humour: {writing.get('humor', 'Light')}",
            f"- Narration person: {writing.get('person', 'second person')}",
            "- Never do the following:",
            avoid,
            "- Actively prefer:",
            prefer,
            "",
            "VIDEO FORMAT",
            f"- Long-form target duration: {video.get('long_duration_minutes', 10)} minutes",
            f"- Shorts to produce: {video.get('shorts_count', 5)}",
            f"- Short target duration: {video.get('short_duration_seconds', 45)} seconds",
            f"- Long aspect ratio: {video.get('aspect_ratio_long', '16:9')}",
            f"- Short aspect ratio: {video.get('aspect_ratio_short', '9:16')}",
        ]
        if video.get("hook_style"):
            lines.append(f"- Hook style: {video['hook_style']}")
        if video.get("ending_style"):
            lines.append(f"- Ending style: {video['ending_style']}")

        lines += [
            "",
            "VISUAL DIRECTION",
            f"- Visual style: {visuals.get('style', 'Cinematic')}",
            f"- Primary colours: {colours}",
            f"- Visual preference: {visuals.get('visual_preference', 'High quality')}",
            f"- Motion style: {visuals.get('motion_style', 'Controlled')}",
        ]
        for key, label in (
            ("lighting", "Lighting"),
            ("texture", "Texture"),
            ("era_reference", "Historical / era reference"),
        ):
            if visuals.get(key):
                lines.append(f"- {label}: {visuals[key]}")
        lines.append("- Never depict:")
        lines.append(avoid_visuals)

        lines += [
            "",
            "AUDIO DIRECTION",
            f"- Music style: {audio.get('music_style', 'Ambient')}",
            f"- Sound effects: {audio.get('sound_effect_style', 'Subtle')}",
            f"- Mixing: {audio.get('mixing_style', 'Narration forward')}",
        ]
        if audio.get("energy"):
            lines.append(f"- Energy curve: {audio['energy']}")
        if audio.get("notes"):
            lines.append(f"- Notes: {audio['notes']}")

        return "\n".join(line for line in lines if line is not None)

    def fingerprint(self) -> dict[str, Any]:
        """Subset used to detect when a profile changed after a project started."""
        return {
            "id": self.id,
            "name": self.name,
            "writing": self.writing,
            "visuals": self.visuals,
            "audio": self.audio,
            "video": self.video,
        }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class ProfileRegistry:
    """Discovers, validates and caches channel profiles from a directory."""

    def __init__(self, profiles_dir: Path, schema_path: Path | None = None) -> None:
        self.profiles_dir = profiles_dir
        self.schema_path = schema_path or SCHEMA_PATH
        self._cache: dict[str, ChannelProfile] = {}

    # -- schema ------------------------------------------------------------ #
    def _schema(self) -> dict[str, Any]:
        schema = read_json(self.schema_path)
        if not schema:
            raise ProfileError(f"Profile schema missing or unreadable: {self.schema_path}")
        return schema

    # -- discovery --------------------------------------------------------- #
    def available(self) -> list[str]:
        if not self.profiles_dir.is_dir():
            return []
        return sorted(
            path.stem
            for path in self.profiles_dir.glob("*.json")
            if path.is_file() and not path.name.startswith("_")
        )

    def describe_all(self) -> list[dict[str, Any]]:
        """Light listing used by ``--list-profiles``."""
        rows: list[dict[str, Any]] = []
        for profile_id in self.available():
            try:
                profile = self.load(profile_id)
            except ProfileError as exc:
                rows.append({"id": profile_id, "valid": False, "error": str(exc)})
                continue
            rows.append(
                {
                    "id": profile.id,
                    "valid": True,
                    "name": profile.name,
                    "category": profile.category,
                    "language": profile.language,
                    "long_minutes": profile.long_duration_minutes,
                    "shorts": profile.shorts_count,
                    "visual_style": profile.visuals.get("style", ""),
                    "music_style": profile.audio.get("music_style", ""),
                }
            )
        return rows

    # -- loading ----------------------------------------------------------- #
    def load(self, profile_id: str) -> ChannelProfile:
        """Load and validate a profile by id (case-insensitive)."""
        key = profile_id.strip().lower()
        if key in self._cache:
            return self._cache[key]

        matches = [name for name in self.available() if name.lower() == key]
        if not matches:
            raise ProfileError(
                f"Unknown channel profile '{profile_id}'. "
                f"Available: {', '.join(self.available()) or '(none)'}"
            )
        if len(matches) > 1:
            raise ProfileError(
                f"Ambiguous profile id '{profile_id}': {', '.join(matches)}"
            )

        path = self.profiles_dir / f"{matches[0]}.json"
        raw = read_json(path)
        if raw is None:
            raise ProfileError(f"Profile file is not valid JSON: {path}")

        errors = self.validate(raw)
        if errors:
            details = "\n".join(f"  - {error}" for error in errors)
            raise ProfileError(f"Invalid channel profile {path.name}:\n{details}")

        if str(raw["channel"]["id"]).lower() != matches[0].lower():
            raise ProfileError(
                f"Profile {path.name}: channel.id '{raw['channel']['id']}' "
                f"must match the filename '{matches[0]}'"
            )

        profile = ChannelProfile(raw=raw, path=path)
        self._cache[key] = profile
        log.debug("Loaded channel profile %s from %s", profile.id, path.name)
        return profile

    def validate(self, raw: dict[str, Any]) -> list[str]:
        """Validate a raw profile dict against the JSON schema plus semantic rules."""
        from engine.json_utils import validate_json

        errors = validate_json(raw, self._schema())
        if errors:
            return errors

        video = raw.get("video", {})
        if not 1 <= int(video.get("shorts_count", 0)) <= 20:
            errors.append("video.shorts_count must be between 1 and 20")
        if not 1 <= int(video.get("long_duration_minutes", 0)) <= 120:
            errors.append("video.long_duration_minutes must be between 1 and 120")
        if not 5 <= int(video.get("short_duration_seconds", 0)) <= 180:
            errors.append("video.short_duration_seconds must be between 5 and 180")

        writing = raw.get("writing", {})
        if not str(writing.get("tone", "")).strip():
            errors.append("writing.tone must not be empty")
        if not raw.get("visuals", {}).get("style"):
            errors.append("visuals.style must not be empty")
        if not raw.get("audio", {}).get("music_style"):
            errors.append("audio.music_style must not be empty")

        return errors

    # -- scaffolding ------------------------------------------------------- #
    def duplicate(self, source_id: str, new_id: str, new_name: str | None = None) -> Path:
        """Copy an existing profile as the starting point for a new channel."""
        source = self.load(source_id)
        safe = new_id.strip().lower()
        if not safe.replace("_", "").isalnum():
            raise ProfileError("New profile id may only contain letters, digits and underscores")
        target = self.profiles_dir / f"{safe}.json"
        if target.exists():
            raise ProfileError(f"Profile already exists: {target}")

        raw = json.loads(json.dumps(source.raw))
        raw["channel"]["id"] = safe
        raw["channel"]["name"] = new_name or safe.replace("_", " ").title()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._cache.pop(safe, None)
        return target


def load_profile(profile_id: str, profiles_dir: Path) -> ChannelProfile:
    """Convenience helper for callers that only need one profile."""
    return ProfileRegistry(profiles_dir).load(profile_id)


__all__ = [
    "ChannelProfile",
    "ProfileError",
    "ProfileRegistry",
    "load_profile",
]
