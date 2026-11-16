"""Shared provider primitives.

Providers are the only place in PIXEL AI that is allowed to talk to an external
service.  Two rules apply everywhere:

* If a provider is not configured it must say so. It must never return invented
  content, invented URLs or fake file paths.
* Every result carries an explicit status so the pipeline can decide whether it
  is safe to continue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ProviderState(str, Enum):
    """How usable a provider currently is."""

    READY = "ready"
    PARTIAL = "partial"
    UNCONFIGURED = "unconfigured"
    FAILED = "failed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Research
# --------------------------------------------------------------------------- #


@dataclass
class ResearchSource:
    """One collected source. Only ever built from real provider output."""

    source_id: str
    title: str
    url: str | None
    provider: str
    extract: str
    publication_date: str | None = None
    retrieved_at: str = field(default_factory=utc_now)
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "publication_date": self.publication_date,
            "provider": self.provider,
            "extract": self.extract,
            "retrieved_at": self.retrieved_at,
            "score": self.score,
        }


@dataclass
class ResearchResult:
    """Aggregate outcome of a research collection run."""

    provider_name: str
    state: ProviderState
    sources: list[ResearchSource] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    error: str | None = None
    notes: str = ""

    @property
    def configured(self) -> bool:
        return self.state is not ProviderState.UNCONFIGURED

    @property
    def research_status(self) -> str:
        """Maps to the ``research_status`` field of Research.json."""
        if self.state is ProviderState.UNCONFIGURED or not self.sources:
            return "unavailable"
        if self.state is ProviderState.PARTIAL:
            return "partial"
        return "available"

    def to_provider_block(self) -> dict[str, Any]:
        return {
            "name": self.provider_name,
            "configured": self.configured,
            "notes": self.notes,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #


@dataclass
class AssetRequest:
    """A single asset to generate or fetch."""

    asset_id: str
    kind: str  # image | video | sfx | music
    prompt: str
    negative_prompt: str = ""
    aspect_ratio: str = "16:9"
    duration_seconds: float | None = None
    scene_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetResult:
    """Outcome of one asset request. ``path`` is only set when a real file exists."""

    asset_id: str
    state: ProviderState
    path: Path | None = None
    provider: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.state is ProviderState.READY and self.path and self.path.is_file():
            return "ready"
        if self.state is ProviderState.FAILED:
            return "failed"
        return "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "status": self.status,
            "path": str(self.path) if self.path else None,
            "provider": self.provider,
            "error": self.error,
            "metadata": self.metadata,
        }


class ProviderError(RuntimeError):
    """Raised by a provider when an external call genuinely fails."""


__all__ = [
    "AssetRequest",
    "AssetResult",
    "ProviderError",
    "ProviderState",
    "ResearchResult",
    "ResearchSource",
    "utc_now",
]
