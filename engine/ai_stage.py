"""Shared plumbing for AI stages.

Every stage module in ``engine/`` subclasses :class:`AIStage`.  It provides:

* schema loading/validation from ``schemas/``
* prompt rendering from ``prompts/``
* a single place where the model is called

It deliberately does **not** try to be a general workflow engine.  Each stage
still owns its own prompt, input shaping and output post-processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import PipelineSettings
from engine.channel_profile import ChannelProfile
from engine.deepseek_client import DeepSeekClient, estimate_tokens, truncate_to_tokens
from engine.logging_setup import get_logger
from engine.prompt_library import PromptLibrary, bullets

log = get_logger("stage")

_SCHEMA_CACHE: dict[Path, dict[str, Any]] = {}


def load_schema(schemas_dir: Path, name: str) -> dict[str, Any]:
    """Load and cache a JSON schema by file stem."""
    path = schemas_dir / f"{name}.json"
    if path in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[path]
    if not path.is_file():
        raise FileNotFoundError(f"Schema not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    _SCHEMA_CACHE[path] = schema
    return schema


class AIStage:
    """Base class for one specialised DeepSeek call site."""

    stage_name: str = "ai_stage"

    def __init__(
        self,
        client: DeepSeekClient,
        prompts: PromptLibrary,
        schemas_dir: Path,
        profile: ChannelProfile,
        pipeline: PipelineSettings,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.schemas_dir = schemas_dir
        self.profile = profile
        self.pipeline = pipeline

    # -- helpers ----------------------------------------------------------- #
    def schema(self, name: str) -> dict[str, Any]:
        return load_schema(self.schemas_dir, name)

    @property
    def prompt_tokens_left(self) -> int:
        """Prompt-side token budget for the current context window."""
        return self.client.budget.prompt_tokens

    def fit(self, text: str, *, keep: str = "head", label: str = "input") -> str:
        """Truncate a variable prompt section to the remaining budget."""
        return self.client.budget.fit(text, keep=keep, label=label)

    def run_json(
        self,
        *,
        prompt_name: str,
        schema_name: str,
        label: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
        **values: Any,
    ) -> dict[str, Any]:
        """Render a prompt, call the model, and return validated JSON."""
        system = self.prompts.render(prompt_name, **values)
        target_schema = schema or self.schema(schema_name)
        result = self.client.chat_json(
            system=system,
            user="Return the JSON object now. Raw JSON only.",
            schema=target_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            label=label or f"{self.stage_name}:{prompt_name}",
            schema_name=schema_name,
        )
        log.debug(
            "%s produced %d top-level keys", self.stage_name, len(result or {})
        )
        return result

    # -- prompt-side digests ---------------------------------------------- #
    def profile_block(self) -> str:
        return self.profile.to_prompt_block()


# --------------------------------------------------------------------------- #
# Digest helpers - keep prompts small
# --------------------------------------------------------------------------- #


def brief_digest(brief: dict[str, Any], *, max_chars: int = 1400) -> str:
    """Condense a ContentBrief for downstream prompts."""
    lines = [
        f"Central message: {brief.get('central_message', '')}",
        f"Viewer promise: {brief.get('viewer_promise', '')}",
    ]
    audience = brief.get("target_audience", {}) or {}
    if audience.get("description"):
        lines.append(f"Audience: {audience['description']}")
    if brief.get("key_questions"):
        lines.append("Questions to answer:")
        lines.extend(f"  - {item}" for item in brief["key_questions"][:6])
    if brief.get("misconceptions"):
        lines.append("Misconceptions to correct:")
        for item in brief["misconceptions"][:4]:
            lines.append(f"  - {item.get('misconception', '')} -> {item.get('correction', '')}")
    selected = [angle for angle in brief.get("angles", []) if angle.get("selected")]
    if selected:
        lines.append(f"Chosen angle: {selected[0].get('title', '')} - {selected[0].get('why_interesting', '')}")
    elif brief.get("angles"):
        lines.append(f"Chosen angle: {brief['angles'][0].get('title', '')}")
    scope = brief.get("scope", {}) or {}
    if scope.get("include"):
        lines.append("In scope: " + "; ".join(scope["include"][:5]))
    if scope.get("exclude"):
        lines.append("Out of scope: " + "; ".join(scope["exclude"][:5]))
    return truncate_to_tokens("\n".join(lines), max(60, max_chars // 4))


def research_digest(research: dict[str, Any], *, max_chars: int = 1500) -> str:
    """Condense a ResearchReport into actionable writing guidance."""
    status = research.get("research_status", "unavailable")
    provider = (research.get("provider") or {}).get("name", "none")
    lines = [f"Research status: {status} (provider: {provider})"]

    guidance = research.get("guidance_for_script", {}) or {}
    if guidance.get("safe_to_state"):
        lines.append("Safe to state plainly:")
        lines.extend(f"  - {item}" for item in guidance["safe_to_state"][:8])
    if guidance.get("state_with_hedging"):
        lines.append("State with hedging:")
        lines.extend(f"  - {item}" for item in guidance["state_with_hedging"][:8])
    if guidance.get("avoid"):
        lines.append("Do not claim:")
        lines.extend(f"  - {item}" for item in guidance["avoid"][:8])

    facts = research.get("verified_facts", []) or []
    if facts:
        lines.append("Background that is well established:")
        for fact in facts[:6]:
            lines.append(f"  - {fact.get('fact', '')}")

    stats = research.get("statistics", []) or []
    if stats:
        lines.append("Figures:")
        for stat in stats[:5]:
            verified = "verified" if stat.get("verified") else "UNVERIFIED"
            lines.append(
                f"  - {stat.get('statistic', '')}: {stat.get('value', '')} "
                f"({verified}{', as of ' + stat['as_of'] if stat.get('as_of') else ''})"
            )

    if research.get("missing_evidence"):
        lines.append("Missing evidence:")
        lines.extend(f"  - {item}" for item in research["missing_evidence"][:5])

    digest = "\n".join(lines)
    # Roughly 4 characters per token; max_chars is expressed in tokens x 4.
    return truncate_to_tokens(digest, max(80, max_chars // 4))


def outline_digest(outline: dict[str, Any], *, max_chars: int = 1600) -> str:
    """Condense an outline for the scriptwriter and quality reviewer."""
    lines = [f"Working title: {outline.get('working_title', '')}", ""]
    hook = outline.get("hook", {}) or {}
    if hook.get("text"):
        lines.append(f"Hook: {hook['text']}")
    for section in outline.get("sections", []):
        lines.append(
            f"\n{section.get('id', '?')} - {section.get('title', '')} "
            f"({section.get('estimated_words', '?')} words)"
        )
        lines.append(f"  Job: {section.get('goal', '')}")
        if section.get("key_points"):
            lines.append("  Covers: " + "; ".join(section["key_points"][:5]))
        if section.get("transition_to_next"):
            lines.append(f"  Bridge: {section['transition_to_next']}")
    conclusion = outline.get("conclusion", {}) or {}
    if conclusion.get("text"):
        lines.append(f"\nConclusion: {conclusion['text']}")
    return truncate_to_tokens("\n".join(lines), max(120, max_chars // 4))


def scenes_timeline(scenes: list[dict[str, Any]], *, excerpt_chars: int = 110) -> str:
    """Compact scene timeline for the audio director."""
    lines: list[str] = []
    for scene in scenes:
        narration = " ".join(str(scene.get("narration", "")).split())
        if len(narration) > excerpt_chars:
            narration = narration[:excerpt_chars].rstrip() + "..."
        lines.append(
            f"{scene.get('scene_id', '?')} | {scene.get('purpose', '')} | "
            f"{scene.get('estimated_duration_seconds', 0)}s | {narration}"
        )
    return "\n".join(lines)


def bullet_list(items: Any, empty: str = "(none)") -> str:
    return bullets(items, empty)


__all__ = [
    "AIStage",
    "brief_digest",
    "bullet_list",
    "estimate_tokens",
    "load_schema",
    "outline_digest",
    "research_digest",
    "scenes_timeline",
    "truncate_to_tokens",
]
