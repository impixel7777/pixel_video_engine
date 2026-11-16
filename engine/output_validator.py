"""Standalone output validation.

Used by the pipeline's ``validate`` stage and by ``python main.py validate``, so
there is exactly one definition of "the outputs are acceptable".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.channel_profile import ChannelProfile
from engine.file_manager import ProjectPaths, read_json
from engine.json_utils import validate_json
from engine.narration import is_pure_narration, parse_shorts_text

SCHEMA_FOR_ARTIFACT: dict[str, str] = {
    "ContentBrief.json": "content_brief",
    "Research.json": "research_report",
    "Outline.json": "video_outline",
    "Long_Production.json": "production_plan",
    "Shorts_Production.json": "shorts_plan",
    "QualityReport.json": "quality_report",
    "production_state.json": None,  # structural check only
}


@dataclass
class ValidationResult:
    checked: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "failures": self.failures,
            "warnings": self.warnings,
        }


def validate_project(
    paths: ProjectPaths,
    schemas_dir: Path,
    profile: ChannelProfile,
) -> ValidationResult:
    """Validate every artefact in a project folder."""
    result = ValidationResult()

    for name, schema_name in SCHEMA_FOR_ARTIFACT.items():
        path = paths.root / name
        if not path.is_file():
            result.failures.append(f"{name}: missing")
            continue
        data = read_json(path)
        if data is None:
            result.failures.append(f"{name}: invalid JSON")
            continue
        result.checked.append(name)
        if schema_name is None:
            continue
        schema_file = schemas_dir / f"{schema_name}.json"
        if not schema_file.is_file():
            result.failures.append(f"{name}: schema {schema_name}.json not found")
            continue
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        errors = validate_json(data, schema)
        for error in errors[:6]:
            result.failures.append(f"{name}: {error}")

    # AudioPlan is a sub-document of the production plan schema.
    if paths.audio_plan.is_file():
        audio = read_json(paths.audio_plan)
        result.checked.append("AudioPlan.json")
        schema_file = schemas_dir / "production_plan.json"
        if audio is not None and schema_file.is_file():
            schema = json.loads(schema_file.read_text(encoding="utf-8"))
            errors = validate_json(audio, schema["properties"]["audio_plan"])
            for error in errors[:6]:
                result.failures.append(f"AudioPlan.json: {error}")

    # Narration files.
    for name, path in (("Long.txt", paths.long_script), ("Short.txt", paths.short_script)):
        if not path.is_file():
            result.failures.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            result.failures.append(f"{name}: empty")
            continue
        result.checked.append(name)
        ok, problems = is_pure_narration(text)
        for problem in problems:
            result.failures.append(f"{name}: {problem}")

    if paths.short_script.is_file():
        text = paths.short_script.read_text(encoding="utf-8")
        parsed = parse_shorts_text(text)
        if len(parsed) != profile.shorts_count:
            result.failures.append(
                f"Short.txt: {len(parsed)} parsable Short(s), profile requires "
                f"{profile.shorts_count}"
            )
        for short in parsed:
            if not short.narration.strip():
                result.warnings.append(f"Short {short.index:02d}: empty narration")
            if not short.title.strip():
                result.warnings.append(f"Short {short.index:02d}: empty title")

    # Shorts must appear in ascending order for the legacy parser.
    if paths.short_script.is_file():
        indices = [short.index for short in parse_shorts_text(paths.short_script.read_text(encoding="utf-8"))]
        if indices != sorted(indices):
            result.warnings.append("Short.txt: Short indices are not in ascending order")

    return result


__all__ = ["SCHEMA_FOR_ARTIFACT", "ValidationResult", "validate_project"]
