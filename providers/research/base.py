"""Research provider interface.

A research provider knows how to turn search queries into :class:`ResearchSource`
objects.  It does not analyse anything — that is DeepSeek's job — and it never
invents sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from providers.base import ResearchResult, ResearchSource


class ResearchProvider(ABC):
    """Base class for every research backend."""

    #: Stable identifier used in configs and output files.
    name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        """True when credentials and endpoint are present."""

    @abstractmethod
    def configuration_hint(self) -> str:
        """Explain exactly what the operator must set to enable this provider."""

    @abstractmethod
    def search(self, queries: list[str], max_results: int) -> ResearchResult:
        """Run the queries and return real, unmodified provider output."""

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "configured": self.is_configured(),
            "hint": self.configuration_hint(),
        }


class NullResearchProvider(ResearchProvider):
    """The honest default: no research backend is configured."""

    name = "none"

    def is_configured(self) -> bool:
        return False

    def configuration_hint(self) -> str:
        return (
            "No research provider is configured. Set research.provider in "
            "config/pipeline_settings.json to 'tavily', 'serper' or 'custom_http', "
            "and put the API key in the environment variable named in "
            "providers/research/web_search.py. Until then Research.json is marked "
            "research_status=unavailable and the script may only use hedged, "
            "well-established background knowledge."
        )

    def search(self, queries: list[str], max_results: int) -> ResearchResult:
        from providers.base import ProviderState

        return ResearchResult(
            provider_name=self.name,
            state=ProviderState.UNCONFIGURED,
            sources=[],
            queries=list(queries),
            notes="No external research backend is configured.",
            error=None,
        )


__all__ = ["NullResearchProvider", "ResearchProvider", "ResearchSource"]
