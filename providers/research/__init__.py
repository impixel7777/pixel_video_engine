"""Research provider factory.

Adding a backend means adding one class and one entry in :data:`REGISTRY`.
No engine code changes.
"""

from __future__ import annotations

import os

from engine.logging_setup import get_logger
from providers.research.base import NullResearchProvider, ResearchProvider
from providers.research.web_search import (
    CustomHttpResearchProvider,
    SerperResearchProvider,
    TavilyResearchProvider,
)

log = get_logger("research")

REGISTRY: dict[str, type[ResearchProvider]] = {
    "none": NullResearchProvider,
    "tavily": TavilyResearchProvider,
    "serper": SerperResearchProvider,
    "custom_http": CustomHttpResearchProvider,
}


def available_providers() -> list[str]:
    return sorted(REGISTRY)


def get_research_provider(name: str | None = None) -> ResearchProvider:
    """Instantiate the configured provider.

    Falls back to :class:`NullResearchProvider` when the requested backend is
    unknown or unconfigured, and logs why.  The pipeline then records
    ``research_status=unavailable`` rather than failing the whole run.
    """
    requested = (name or os.environ.get("PIXEL_RESEARCH_PROVIDER") or "none").strip().lower()
    provider_cls = REGISTRY.get(requested)
    if provider_cls is None:
        log.warning(
            "Unknown research provider '%s'. Available: %s. Falling back to 'none'.",
            requested,
            ", ".join(available_providers()),
        )
        return NullResearchProvider()

    provider = provider_cls()
    if not provider.is_configured():
        log.warning(
            "Research provider '%s' is not configured. %s",
            requested,
            provider.configuration_hint(),
        )
        return NullResearchProvider()
    return provider


__all__ = ["REGISTRY", "available_providers", "get_research_provider"]
