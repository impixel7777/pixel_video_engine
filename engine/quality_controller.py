"""Stage 8 - Quality Controller.

Two layers, deliberately separated:

* **Deterministic checks** - regex, counting, schema validation.  Fast, repeatable
  and impossible for the model to argue with.
* **AI editorial review** - judgement calls only (continuity, clarity, accuracy
  against research, channel fit).  Optional and non-fatal: if the model cannot
  produce a review, the deterministic verdict still stands and the gap is
  recorded as ``skipped``.

Nothing here rewrites content.  It reports, so the pipeline can regenerate the
right stage instead of restarting everything.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import PipelineSettings
from engine.ai_stage import AIStage, brief_digest, research_digest
from engine.channel_profile import ChannelProfile
from engine.deepseek_client import DeepSeekClient
from engine.file_manager import ProjectPaths, utc_now_iso
from engine.json_utils import validate_json
from engine.logging_setup import get_logger
from engine.narration import (
    count_words,
    is_pure_narration,
    parse_shorts_text,
    repeated_phrases,
    sentence_openers,
    split_sentences,
    word_count_to_range,
)
from engine.prompt_library import PromptLibrary

log = get_logger("quality")

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}

GENERIC_OPENERS = (
    "in this video",
    "welcome back",
    "welcome to",
    "today we",
    "today i",
    "let's talk about",
    "lets talk about",
    "in today's",
    "have you ever wondered",
    "what's up guys",
)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail, "value": self.value}


@dataclass
class Issue:
    category: str
    severity: str
    message: str
    location: str = ""
    auto_fixable: bool = False
    issue_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.issue_id,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
            "auto_fixable": self.auto_fixable,
        }


@dataclass
class ReviewInputs:
    """Everything the controller inspects."""

    brief: dict[str, Any]
    research: dict[str, Any]
    outline: dict[str, Any]
    long_script: str
    shorts: list[dict[str, Any]]
    long_plan: dict[str, Any] | None
    shorts_plan: dict[str, Any] | None
    audio_plan: dict[str, Any] | None
    paths: ProjectPaths
    target_minutes: float


def _wav_seconds(path: Path) -> float | None:
    """Read a PCM WAV duration without importing the speech stack."""
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getframerate() <= 0:
                return None
            return handle.getnframes() / float(handle.getframerate())
    except (wave.Error, OSError):
        return None


class QualityController(AIStage):
    """Produces ``QualityReport.json``."""

    stage_name = "quality_controller"

    def __init__(
        self,
        client: DeepSeekClient,
        prompts: PromptLibrary,
        schemas_dir: Path,
        profile: ChannelProfile,
        pipeline: PipelineSettings,
    ) -> None:
        super().__init__(client, prompts, schemas_dir, profile, pipeline)
        self._issue_counter = 0

    # -- public API --------------------------------------------------------- #
    def review(self, inputs: ReviewInputs) -> dict[str, Any]:
        self._issue_counter = 0
        issues: list[Issue] = []

        categories: dict[str, dict[str, Any]] = {}
        categories["files"] = self._check_files(inputs, issues)
        categories["research"] = self._check_research(inputs, issues)
        categories["long_script"] = self._check_long_script(inputs, issues)
        categories["shorts"] = self._check_shorts(inputs, issues)
        categories["visuals"] = self._check_visuals(inputs, issues)
        categories["audio"] = self._check_audio(inputs, issues)

        ai_review: dict[str, Any] | None = None
        if self.pipeline.raw.get("quality", {}).get("ai_review", True):
            ai_review = self._ai_review(inputs, issues)

        statuses = [category["status"] for category in categories.values()]
        if "fail" in statuses:
            overall = "fail"
        elif "warn" in statuses or any(issue.severity == "warning" for issue in issues):
            overall = "warn"
        else:
            overall = "pass"

        order = {category: index for index, category in enumerate(categories)}
        issues.sort(key=lambda issue: (-SEVERITY_ORDER[issue.severity], order.get(issue.category, 99)))

        report = {
            "schema_version": "1.0",
            "generated_at": utc_now_iso(),
            "project": {
                "profile_id": self.profile.id,
                "topic": inputs.paths.topic,
                "slug": inputs.paths.slug,
            },
            "overall_status": overall,
            "summary": self._summary(overall, categories, issues, ai_review),
            "categories": categories,
            "issues": [issue.to_dict() for issue in issues],
            "ai_review": ai_review,
            "metrics": {
                "long_words": count_words(inputs.long_script),
                "short_count": len(inputs.shorts),
                "scene_count": len((inputs.long_plan or {}).get("scenes", [])),
                "issues_by_severity": {
                    severity: sum(1 for issue in issues if issue.severity == severity)
                    for severity in ("error", "warning", "info")
                },
            },
        }
        log.info(
            "Quality review: %s (%d error, %d warning, %d info)",
            overall.upper(),
            sum(1 for issue in issues if issue.severity == "error"),
            sum(1 for issue in issues if issue.severity == "warning"),
            sum(1 for issue in issues if issue.severity == "info"),
        )
        return report

    # -- helpers ------------------------------------------------------------ #
    def _issue(
        self,
        issues: list[Issue],
        *,
        category: str,
        severity: str,
        message: str,
        location: str = "",
        auto_fixable: bool = False,
    ) -> None:
        self._issue_counter += 1
        issues.append(
            Issue(
                category=category,
                severity=severity,
                message=message,
                location=location,
                auto_fixable=auto_fixable,
                issue_id=f"Q{self._issue_counter:03d}",
            )
        )

    @staticmethod
    def _category(checks: list[Check]) -> dict[str, Any]:
        statuses = {check.status for check in checks}
        if "fail" in statuses:
            status = "fail"
        elif "warn" in statuses:
            status = "warn"
        elif statuses == {"skipped"}:
            status = "skipped"
        else:
            status = "pass"
        return {"status": status, "checks": [check.to_dict() for check in checks]}

    # -- files -------------------------------------------------------------- #
    def _check_files(self, inputs: ReviewInputs, issues: list[Issue]) -> dict[str, Any]:
        checks: list[Check] = []
        paths = inputs.paths

        for label, path in (("Long.txt", paths.long_script), ("Short.txt", paths.short_script)):
            exists = path.is_file() and path.stat().st_size > 0
            checks.append(
                Check(
                    name=f"{label} present",
                    status="pass" if exists else "fail",
                    detail=f"{path.name} exists and is not empty" if exists else f"{label} missing or empty",
                    value=str(path),
                )
            )
            if not exists:
                self._issue(
                    issues,
                    category="files",
                    severity="error",
                    message=f"{label} is missing or empty",
                    location=str(path),
                )

        if paths.long_script.is_file():
            text = paths.long_script.read_text(encoding="utf-8")
            ok, problems = is_pure_narration(text)
            checks.append(
                Check(
                    name="Long.txt is narration only",
                    status="pass" if ok else "fail",
                    detail="No markdown, metadata, visual or audio directions" if ok else "; ".join(problems),
                )
            )
            for problem in problems:
                self._issue(
                    issues,
                    category="files",
                    severity="error",
                    message=f"Long.txt {problem}",
                    location="Long.txt",
                )

        if paths.short_script.is_file():
            text = paths.short_script.read_text(encoding="utf-8")
            ok, problems = is_pure_narration(text)
            checks.append(
                Check(
                    name="Short.txt is narration only",
                    status="pass" if ok else "fail",
                    detail="No markdown, metadata, visual or audio directions" if ok else "; ".join(problems),
                )
            )
            for problem in problems:
                self._issue(
                    issues,
                    category="files",
                    severity="error",
                    message=f"Short.txt {problem}",
                    location="Short.txt",
                )

            parsed = parse_shorts_text(text)
            expected = self.profile.shorts_count
            checks.append(
                Check(
                    name="Shorts heading format",
                    status="pass" if len(parsed) == expected else "fail",
                    detail=f"{len(parsed)} parsable SHORT heading(s), expected {expected}",
                    value=[short.index for short in parsed],
                )
            )
            if len(parsed) != expected:
                self._issue(
                    issues,
                    category="files",
                    severity="error",
                    message=(
                        f"Kokoro would find {len(parsed)} Shorts but the profile requires {expected}. "
                        "Check the 'SHORT NN \u2014 Title' headings."
                    ),
                    location="Short.txt",
                    auto_fixable=True,
                )

        if paths.sections.is_file() and paths.long_script.is_file():
            try:
                sections = json.loads(paths.sections.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                checks.append(
                    Check(
                        name="Long_Sections.json valid",
                        status="fail",
                        detail=str(exc),
                    )
                )
                self._issue(
                    issues,
                    category="files",
                    severity="error",
                    message=f"Long_Sections.json is unreadable: {exc}",
                    location="Long_Sections.json",
                )
            else:
                if sections.get("external_edit"):
                    checks.append(
                        Check(
                            name="Long.txt provenance",
                            status="pass",
                            detail="Long.txt was human-edited; generated section data is intentionally ignored",
                        )
                    )
                else:
                    reconstructed = "\n\n".join(
                        str(part.get("written", "")).strip()
                        for part in sections.get("parts", []) or []
                        if str(part.get("written", "")).strip()
                    ).strip()
                    long_text = paths.long_script.read_text(encoding="utf-8").strip()
                    matches = bool(reconstructed) and reconstructed == long_text
                    checks.append(
                        Check(
                            name="Long.txt matches persisted sections",
                            status="pass" if matches else "fail",
                            detail=(
                                "Long_Sections.json reconstructs Long.txt exactly"
                                if matches
                                else "Long_Sections.json does not reconstruct Long.txt"
                            ),
                        )
                    )
                    if not matches:
                        self._issue(
                            issues,
                            category="files",
                            severity="error",
                            message=(
                                "Long.txt and Long_Sections.json disagree. Resume could "
                                "silently restore stale narration."
                            ),
                            location="Long.txt",
                        )

        checks.extend(self._check_json_documents(paths, issues))
        return self._category(checks)

    def _check_json_documents(self, paths: ProjectPaths, issues: list[Issue]) -> list[Check]:
        schema_map = {
            paths.brief: "content_brief",
            paths.research: "research_report",
            paths.outline: "video_outline",
            paths.long_plan: "production_plan",
            paths.shorts_plan: "shorts_plan",
        }
        checks: list[Check] = []

        for path, schema_name in schema_map.items():
            if not path.is_file():
                checks.append(
                    Check(
                        name=f"{path.name} valid",
                        status="fail",
                        detail=f"{path.name} is missing",
                    )
                )
                self._issue(
                    issues,
                    category="files",
                    severity="error",
                    message=f"{path.name} is missing",
                    location=str(path),
                )
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except json.JSONDecodeError as exc:
                checks.append(
                    Check(name=f"{path.name} valid", status="fail", detail=f"Invalid JSON: {exc}")
                )
                self._issue(
                    issues,
                    category="files",
                    severity="error",
                    message=f"{path.name} is not valid JSON: {exc}",
                    location=str(path),
                )
                continue

            errors = validate_json(data, self.schema(schema_name))
            checks.append(
                Check(
                    name=f"{path.name} valid",
                    status="pass" if not errors else "fail",
                    detail="Matches schema" if not errors else f"{len(errors)} schema error(s)",
                    value=errors[:5],
                )
            )
            if errors:
                self._issue(
                    issues,
                    category="files",
                    severity="error",
                    message=f"{path.name} fails schema validation ({len(errors)} error(s))",
                    location=str(path),
                )

        if paths.audio_plan.is_file():
            try:
                audio = json.loads(paths.audio_plan.read_text(encoding="utf-8"))
                audio_schema = self.schema("production_plan")["properties"]["audio_plan"]
                errors = validate_json(audio, audio_schema)
                checks.append(
                    Check(
                        name="AudioPlan.json valid",
                        status="pass" if not errors else "fail",
                        detail="Matches schema" if not errors else f"{len(errors)} schema error(s)",
                        value=errors[:5],
                    )
                )
                if errors:
                    self._issue(
                        issues,
                        category="files",
                        severity="error",
                        message=f"AudioPlan.json fails schema validation ({len(errors)} error(s))",
                        location=str(paths.audio_plan),
                    )
            except (json.JSONDecodeError, OSError) as exc:
                checks.append(
                    Check(name="AudioPlan.json valid", status="fail", detail=str(exc))
                )

        return checks

    # -- research ----------------------------------------------------------- #
    def _check_research(self, inputs: ReviewInputs, issues: list[Issue]) -> dict[str, Any]:
        research = inputs.research
        checks: list[Check] = []

        status = str(research.get("research_status", "unavailable"))
        checks.append(
            Check(
                name="Research status recorded",
                status="pass",
                detail=f"research_status={status}",
                value=status,
            )
        )

        provider = research.get("provider", {}) or {}
        sources = research.get("sources", []) or []
        consistent = (status == "unavailable" and not sources) or (
            status in ("available", "partial") and bool(sources)
        )
        checks.append(
            Check(
                name="Status matches collected sources",
                status="pass" if consistent else "fail",
                detail=f"{len(sources)} source(s) with status '{status}'",
            )
        )
        if not consistent:
            self._issue(
                issues,
                category="research",
                severity="error",
                message="Research status contradicts the collected source list",
                location="Research.json",
            )

        if status == "unavailable":
            checks.append(
                Check(
                    name="External sources",
                    status="warn",
                    detail=(
                        "No research provider configured. The script must stay hedged and "
                        "avoid specific unverifiable figures."
                    ),
                )
            )
            self._issue(
                issues,
                category="research",
                severity="warning",
                message=(
                    "Research is unavailable (no provider configured). Publishing without "
                    "sources is a factual risk."
                ),
                location="Research.json",
                auto_fixable=False,
            )
        else:
            checks.append(
                Check(
                    name="External sources",
                    status="pass",
                    detail=f"{len(sources)} source(s) via {provider.get('name', 'unknown')}",
                    value=len(sources),
                )
            )

        facts = research.get("verified_facts", []) or []
        unreferenced = [
            fact for fact in facts if status != "unavailable" and not fact.get("source_ids")
        ]
        checks.append(
            Check(
                name="Verified facts carry references",
                status="pass" if not unreferenced else "warn",
                detail=(
                    "All verified facts reference a collected source"
                    if not unreferenced
                    else f"{len(unreferenced)} verified fact(s) have no source reference"
                ),
            )
        )
        for fact in unreferenced[:3]:
            self._issue(
                issues,
                category="research",
                severity="warning",
                message=f"Fact has no source reference: {str(fact.get('fact', ''))[:120]}",
                location="Research.json",
            )

        stats = research.get("statistics", []) or []
        unverified = [stat for stat in stats if not stat.get("verified")]
        checks.append(
            Check(
                name="Statistics verified",
                status="pass" if not unverified else "warn",
                detail=(
                    "All statistics are marked verified"
                    if not unverified
                    else f"{len(unverified)} of {len(stats)} statistic(s) are unverified"
                ),
            )
        )

        checks.append(
            Check(
                name="Conflicts recorded",
                status="pass" if research.get("conflicts") is not None else "warn",
                detail=f"{len(research.get('conflicts', []) or [])} conflict(s) documented",
            )
        )
        return self._category(checks)

    # -- long script -------------------------------------------------------- #
    def _check_long_script(self, inputs: ReviewInputs, issues: list[Issue]) -> dict[str, Any]:
        checks: list[Check] = []
        text = inputs.long_script
        words = count_words(text)

        target = int(inputs.target_minutes * self.pipeline.words_per_minute)
        low, high = word_count_to_range(target, tolerance=0.25)
        within = low <= words <= high
        checks.append(
            Check(
                name="Duration estimate",
                status="pass" if within else "warn",
                detail=f"{words} words ~ {words / self.pipeline.words_per_minute:.1f} min "
                f"(target {inputs.target_minutes:.0f} min, acceptable {low}-{high} words)",
                value=words,
            )
        )
        if not within:
            self._issue(
                issues,
                category="long_script",
                severity="warning",
                message=(
                    f"Long script is {words} words, outside the {low}-{high} word range for "
                    f"a {inputs.target_minutes:.0f} minute video"
                ),
                location="Long.txt",
                auto_fixable=True,
            )

        expected_parts = {"hook": bool(text.strip()), "sections": len(inputs.outline.get("sections", []))}
        checks.append(
            Check(
                name="Outline sections present",
                status="pass" if expected_parts["sections"] >= 3 else "warn",
                detail=f"{expected_parts['sections']} planned section(s) in the outline",
            )
        )

        repeated = repeated_phrases(text)
        checks.append(
            Check(
                name="Repetition",
                status="pass" if not repeated else "warn",
                detail="No phrase repeated three or more times"
                if not repeated
                else f"{len(repeated)} repeated phrase pattern(s)",
                value=repeated[:5],
            )
        )
        for phrase in repeated[:3]:
            self._issue(
                issues,
                category="long_script",
                severity="warning",
                message=f"Repeated phrase: '{phrase}'",
                location="Long.txt",
                auto_fixable=True,
            )

        spans = split_sentences(text)
        if spans:
            first = " ".join(text[spans[0][0] : spans[0][1]].split())
            hook_words = count_words(first)
            strong = hook_words <= 30 and not first.lower().startswith(GENERIC_OPENERS)
            checks.append(
                Check(
                    name="Hook",
                    status="pass" if strong else "warn",
                    detail=f"First sentence is {hook_words} words: {first[:120]}",
                )
            )
            if not strong:
                self._issue(
                    issues,
                    category="long_script",
                    severity="warning",
                    message="The opening sentence is long or generic for a YouTube hook",
                    location="Long.txt",
                    auto_fixable=True,
                )

        if inputs.research.get("research_status") == "unavailable":
            numeric_hits = [
                token
                for token in text.split()
                if "%" in token or token.startswith("$")
            ]
            checks.append(
                Check(
                    name="Numeric claims without research",
                    status="pass" if not numeric_hits else "warn",
                    detail="No bare numeric claims detected"
                    if not numeric_hits
                    else f"{len(numeric_hits)} numeric token(s) despite unavailable research",
                    value=numeric_hits[:8],
                )
            )
            if numeric_hits:
                self._issue(
                    issues,
                    category="long_script",
                    severity="warning",
                    message=(
                        "The script contains figures while research is unavailable: "
                        + ", ".join(numeric_hits[:6])
                    ),
                    location="Long.txt",
                    auto_fixable=True,
                )

        openers = sentence_openers(text)
        repeats = len(openers) - len(set(openers))
        checks.append(
            Check(
                name="Sentence opener variety",
                status="pass" if repeats <= max(2, len(openers) // 8) else "warn",
                detail=f"{repeats} repeated opener(s) across {len(openers)} sentences",
            )
        )
        return self._category(checks)

    # -- shorts ------------------------------------------------------------- #
    def _check_shorts(self, inputs: ReviewInputs, issues: list[Issue]) -> dict[str, Any]:
        checks: list[Check] = []
        shorts = inputs.shorts
        expected = self.profile.shorts_count

        checks.append(
            Check(
                name="Shorts count",
                status="pass" if len(shorts) == expected else "fail",
                detail=f"{len(shorts)} Short(s), expected {expected}",
                value=len(shorts),
            )
        )
        if len(shorts) != expected:
            self._issue(
                issues,
                category="shorts",
                severity="error",
                message=f"Produced {len(shorts)} Shorts but the profile requires {expected}",
                location="Short.txt",
                auto_fixable=True,
            )

        target_words = self.pipeline.short_target_words
        low, high = word_count_to_range(target_words, tolerance=0.35)
        for short in shorts:
            words = count_words(str(short.get("narration", "")))
            if not low <= words <= high:
                self._issue(
                    issues,
                    category="shorts",
                    severity="warning",
                    message=(
                        f"Short {int(short.get('index', 0)):02d} is {words} words; "
                        f"expected about {target_words}"
                    ),
                    location=f"Short {int(short.get('index', 0)):02d}",
                    auto_fixable=True,
                )

        generic: list[str] = []
        long_openers: list[str] = []
        for short in shorts:
            narration = str(short.get("narration", ""))
            spans = split_sentences(narration)
            if not spans:
                continue
            first = " ".join(narration[spans[0][0] : spans[0][1]].split())
            lowered = first.lower()
            if lowered.startswith(GENERIC_OPENERS):
                generic.append(f"Short {int(short.get('index', 0)):02d}")
            if count_words(first) > 28:
                long_openers.append(f"Short {int(short.get('index', 0)):02d}")

        checks.append(
            Check(
                name="Shorts hooks",
                status="pass" if not generic and not long_openers else "warn",
                detail="Every Short opens with a specific, short sentence"
                if not generic and not long_openers
                else f"generic: {generic or 'none'}; overlong: {long_openers or 'none'}",
            )
        )
        for label in generic:
            self._issue(
                issues,
                category="shorts",
                severity="warning",
                message=f"{label} opens with a generic phrase",
                location=label,
                auto_fixable=True,
            )

        overlaps = self._short_overlaps(shorts)
        checks.append(
            Check(
                name="Shorts independence",
                status="pass" if not overlaps else "warn",
                detail="No two Shorts substantially repeat each other"
                if not overlaps
                else "; ".join(overlaps[:3]),
            )
        )
        for overlap in overlaps:
            self._issue(
                issues,
                category="shorts",
                severity="warning",
                message=overlap,
                location="Short.txt",
                auto_fixable=True,
            )

        missing_ideas = [
            short
            for short in shorts
            if not str(short.get("central_idea", "")).strip()
        ]
        checks.append(
            Check(
                name="Central idea recorded",
                status="pass" if not missing_ideas else "info",
                detail=f"{len(missing_ideas)} Short(s) without a recorded central idea",
            )
        )
        return self._category(checks)

    @staticmethod
    def _short_overlaps(shorts: list[dict[str, Any]], threshold: float = 0.7) -> list[str]:
        """Flag Shorts whose word sets overlap suspiciously."""
        import re

        token_sets: list[set[str]] = []
        for short in shorts:
            tokens = set(re.findall(r"[a-z0-9']+", str(short.get("narration", "")).lower()))
            token_sets.append({token for token in tokens if len(token) > 3})
        overlaps: list[str] = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                if not token_sets[i] or not token_sets[j]:
                    continue
                ratio = len(token_sets[i] & token_sets[j]) / len(token_sets[i] | token_sets[j])
                if ratio >= threshold:
                    overlaps.append(
                        f"Short {i + 1:02d} and Short {j + 1:02d} overlap {ratio:.0%} in vocabulary"
                    )
        return overlaps

    # -- visuals ------------------------------------------------------------ #
    def _check_visuals(self, inputs: ReviewInputs, issues: list[Issue]) -> dict[str, Any]:
        checks: list[Check] = []
        plan = inputs.long_plan

        if not plan:
            return self._category(
                [Check(name="Long production plan", status="fail", detail="Plan missing")]
            )

        scenes = plan.get("scenes", []) or []
        checks.append(
            Check(
                name="Scene count",
                status="pass" if scenes else "fail",
                detail=f"{len(scenes)} scene(s)",
                value=len(scenes),
            )
        )
        if not scenes:
            self._issue(
                issues,
                category="visuals",
                severity="error",
                message="Long production plan has no scenes",
                location="Long_Production.json",
            )
            return self._category(checks)

        missing = [
            scene.get("scene_id")
            for scene in scenes
            if not str(scene.get("image_prompt", "")).strip()
            or not str(scene.get("visual_description", "")).strip()
            or not str(scene.get("purpose", "")).strip()
        ]
        checks.append(
            Check(
                name="Every scene is fully directed",
                status="pass" if not missing else "fail",
                detail="All scenes have purpose, description and image prompt"
                if not missing
                else f"{len(missing)} incomplete scene(s): {missing[:5]}",
            )
        )
        for scene_id in missing[:5]:
            self._issue(
                issues,
                category="visuals",
                severity="error",
                message=f"Scene {scene_id} is missing direction fields",
                location="Long_Production.json",
                auto_fixable=True,
            )

        coverage = float((plan.get("coverage") or {}).get("coverage_ratio", 0))
        checks.append(
            Check(
                name="Narration coverage",
                status="pass" if coverage >= 0.98 else ("warn" if coverage >= 0.9 else "fail"),
                detail=f"{coverage:.1%} of narration characters are anchored to a scene",
                value=coverage,
            )
        )
        if coverage < 0.98:
            self._issue(
                issues,
                category="visuals",
                severity="error" if coverage < 0.9 else "warning",
                message=f"Only {coverage:.1%} of narration is covered by the storyboard",
                location="Long_Production.json",
                auto_fixable=True,
            )

        distinct = int((plan.get("coverage") or {}).get("distinct_visual_concepts", 0))
        variety = distinct / max(1, len(scenes))
        checks.append(
            Check(
                name="Visual variety",
                status="pass" if variety >= 0.8 else "warn",
                detail=f"{distinct} distinct concept(s) across {len(scenes)} scene(s) ({variety:.0%})",
                value=round(variety, 3),
            )
        )
        if variety < 0.8:
            self._issue(
                issues,
                category="visuals",
                severity="warning",
                message="Several scenes share the same visual concept",
                location="Long_Production.json",
                auto_fixable=True,
            )

        style = str(self.profile.visuals.get("style", "")).lower()
        style_tokens = {token for token in style.replace(",", " ").split() if len(token) > 3}
        mismatched = []
        if style_tokens:
            for scene in scenes:
                scene_style = str(scene.get("visual_style", "")).lower()
                if not scene_style or not (style_tokens & set(scene_style.replace(",", " ").split())):
                    mismatched.append(scene.get("scene_id"))
        checks.append(
            Check(
                name="Channel visual style applied",
                status="pass" if not mismatched else "warn",
                detail="Every scene states the channel visual style"
                if not mismatched
                else f"{len(mismatched)} scene(s) do not name the channel style: {mismatched[:5]}",
            )
        )

        if inputs.shorts_plan:
            shorts = inputs.shorts_plan.get("shorts", []) or []
            empty = [short.get("short_id") for short in shorts if not short.get("scenes")]
            checks.append(
                Check(
                    name="Shorts storyboards",
                    status="pass" if shorts and not empty else "fail",
                    detail=f"{len(shorts)} Short storyboard(s), {len(empty)} empty",
                )
            )
            if empty:
                self._issue(
                    issues,
                    category="visuals",
                    severity="error",
                    message=f"Shorts without scenes: {empty}",
                    location="Shorts_Production.json",
                    auto_fixable=True,
                )

        pending = sum(
            1
            for scene in scenes
            if (scene.get("asset") or {}).get("status") in (None, "pending", "missing")
        )
        checks.append(
            Check(
                name="Asset readiness",
                status="info" if pending else "pass",
                detail=(
                    f"{pending} scene asset(s) still pending - no generation provider configured"
                    if pending
                    else "All scene assets are ready"
                ),
                value=pending,
            )
        )

        missing_files: list[str] = []
        for scene in scenes:
            asset = scene.get("asset") or {}
            if asset.get("status") != "ready":
                continue
            relative = asset.get("path")
            candidate = Path(str(relative)) if relative else None
            if candidate is not None and not candidate.is_absolute():
                candidate = inputs.paths.root / candidate
            if candidate is None or not candidate.is_file():
                missing_files.append(str(scene.get("scene_id")))
        checks.append(
            Check(
                name="Ready assets exist on disk",
                status="pass" if not missing_files else "fail",
                detail=(
                    "Every scene marked ready has a real file"
                    if not missing_files
                    else f"{len(missing_files)} scene(s) marked ready but missing files: {missing_files[:5]}"
                ),
            )
        )
        if missing_files:
            self._issue(
                issues,
                category="visuals",
                severity="error",
                message=(
                    f"{len(missing_files)} scene asset(s) are marked ready but the files "
                    f"do not exist: {missing_files[:8]}"
                ),
                location="Long_Production.json / asset_status.json",
            )
        return self._category(checks)

    # -- audio -------------------------------------------------------------- #
    def _check_audio(self, inputs: ReviewInputs, issues: list[Issue]) -> dict[str, Any]:
        checks: list[Check] = []
        audio = inputs.audio_plan

        if not audio:
            return self._category(
                [Check(name="Audio plan", status="fail", detail="Audio plan missing")]
            )

        music = audio.get("music", {}) or {}
        segments = music.get("segments", []) or []
        checks.append(
            Check(
                name="Music planned",
                status="pass" if segments else "fail",
                detail=f"{len(segments)} music segment(s), style: {music.get('style', 'unset')}",
                value=len(segments),
            )
        )
        if not segments:
            self._issue(
                issues,
                category="audio",
                severity="error",
                message="No music segments were planned",
                location="AudioPlan.json",
                auto_fixable=True,
            )

        if len(segments) > 1:
            intensities = {round(float(segment.get("intensity", 0)), 2) for segment in segments}
            checks.append(
                Check(
                    name="Music intensity varies",
                    status="pass" if len(intensities) > 1 else "warn",
                    detail=f"{len(intensities)} distinct intensity level(s): {sorted(intensities)}",
                )
            )
            if len(intensities) == 1:
                self._issue(
                    issues,
                    category="audio",
                    severity="warning",
                    message="Music intensity never changes across the video",
                    location="AudioPlan.json",
                    auto_fixable=True,
                )

        scene_ids = {
            str(scene.get("scene_id"))
            for scene in (inputs.long_plan or {}).get("scenes", []) or []
        }
        bad_anchors = [
            cue.get("cue_id")
            for cue in audio.get("sfx", []) or []
            if scene_ids and str(cue.get("anchor_scene")) not in scene_ids
        ]
        checks.append(
            Check(
                name="Sound effect anchors",
                status="pass" if not bad_anchors else "fail",
                detail="Every cue points at a real scene"
                if not bad_anchors
                else f"{len(bad_anchors)} cue(s) point at unknown scenes",
            )
        )

        loud = [
            cue.get("cue_id")
            for cue in audio.get("sfx", []) or []
            if float(cue.get("gain_db", -20)) > -12
        ]
        checks.append(
            Check(
                name="Effects do not overpower narration",
                status="pass" if not loud else "warn",
                detail="All effects sit below -12 dB"
                if not loud
                else f"{len(loud)} cue(s) louder than -12 dB: {loud[:5]}",
            )
        )

        mix = audio.get("mix", {}) or {}
        checks.append(
            Check(
                name="Narration priority stated",
                status="pass" if str(mix.get("narration_priority", "")).strip() else "warn",
                detail=str(mix.get("narration_priority", "(not set)")),
            )
        )

        narration = inputs.paths.tts_long_audio
        if narration.is_file():
            actual = _wav_seconds(narration)
            planned = sum(
                float(scene.get("estimated_duration_seconds", 0))
                for scene in (inputs.long_plan or {}).get("scenes", []) or []
            )
            tolerance = max(0.35, actual * 0.005) if actual else 0.35
            matches = actual is not None and abs(planned - actual) <= tolerance
            checks.append(
                Check(
                    name="Timeline matches real narration",
                    status="pass" if matches else "fail",
                    detail=(
                        f"Scene total {planned:.2f}s vs narration {actual:.2f}s"
                        if actual is not None
                        else "Narration WAV is unreadable"
                    ),
                    value=round(actual, 2) if actual is not None else None,
                )
            )
            if not matches:
                self._issue(
                    issues,
                    category="audio",
                    severity="error",
                    message=(
                        f"Scene durations total {planned:.2f}s but the generated narration "
                        f"is {actual if actual is not None else 'unreadable'}. Timeline "
                        "reconciliation did not run."
                    ),
                    location="Long_Production.json / output/Long/narration.wav",
                    auto_fixable=True,
                )
        else:
            checks.append(
                Check(
                    name="Timeline matches real narration",
                    status="info",
                    detail="Narration WAV not generated yet; timeline check runs after TTS",
                )
            )
        return self._category(checks)

    # -- AI review ---------------------------------------------------------- #
    def _ai_review(self, inputs: ReviewInputs, issues: list[Issue]) -> dict[str, Any] | None:
        shorts_text = "\n\n".join(
            f"Short {int(short.get('index', index + 1)):02d} - {short.get('title', '')}\n"
            f"{short.get('narration', '')}"
            for index, short in enumerate(inputs.shorts)
        )

        try:
            review = self.run_json(
                prompt_name="quality_review",
                schema_name="quality_review",
                label="quality_review",
                temperature=0.3,
                max_tokens=min(2000, max(700, self.client.budget.prompt_tokens // 2)),
                PROFILE=self.profile_block(),
                BRIEF_DIGEST=self.fit(brief_digest(inputs.brief), keep="head", label="brief"),
                RESEARCH_DIGEST=self.fit(
                    research_digest(inputs.research), keep="tail", label="research digest"
                ),
                LONG_SCRIPT=self.fit(inputs.long_script, keep="head", label="long script"),
                SHORTS=self.fit(shorts_text, keep="head", label="shorts"),
            )
        except Exception as exc:  # noqa: BLE001 - AI review is optional
            log.warning("AI editorial review unavailable: %s", exc)
            self._issue(
                issues,
                category="long_script",
                severity="info",
                message=f"AI editorial review could not be completed: {exc}",
            )
            return {"status": "skipped", "reason": str(exc)}

        for category_key, target in (
            ("research_issues", "research"),
            ("script_issues", "long_script"),
            ("shorts_issues", "shorts"),
            ("accuracy_concerns", "long_script"),
        ):
            for item in review.get(category_key, []) or []:
                severity = str(item.get("severity", "info"))
                if severity not in SEVERITY_ORDER:
                    severity = "info"
                # Editorial judgement is advisory unless explicitly allowed to
                # block.  Deterministic checks remain authoritative.
                if severity == "error" and not self.pipeline.quality_ai_review_can_fail:
                    severity = "warning"
                self._issue(
                    issues,
                    category=target,
                    severity=severity,
                    message=str(item.get("message", ""))[:400],
                    location=str(item.get("location", "AI review")),
                )

        review["status"] = "completed"
        return review

    # -- summary ------------------------------------------------------------ #
    @staticmethod
    def _summary(
        overall: str,
        categories: dict[str, dict[str, Any]],
        issues: list[Issue],
        ai_review: dict[str, Any] | None,
    ) -> str:
        counts = {
            severity: sum(1 for issue in issues if issue.severity == severity)
            for severity in ("error", "warning", "info")
        }
        failing = [name for name, category in categories.items() if category["status"] == "fail"]
        parts = [
            f"Overall status: {overall.upper()}.",
            f"{counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} note(s).",
        ]
        if failing:
            parts.append("Failing categories: " + ", ".join(failing) + ".")
        if ai_review and ai_review.get("status") == "completed":
            verdict = ai_review.get("verdict", "unrated")
            parts.append(f"Editorial verdict: {verdict}.")
            assessment = str(ai_review.get("overall_assessment", "")).strip()
            if assessment:
                parts.append(assessment)
        elif ai_review and ai_review.get("status") == "skipped":
            parts.append("AI editorial review was skipped.")
        return " ".join(parts)


__all__ = ["Check", "Issue", "QualityController", "ReviewInputs"]
