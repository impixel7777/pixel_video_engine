"""HTTP search providers.

Three real backends are implemented, all credential-driven and all optional:

* ``tavily``      - Tavily search API (``TAVILY_API_KEY``)
* ``serper``      - Serper.dev Google search (``SERPER_API_KEY``)
* ``custom_http`` - any JSON search endpoint you control
                    (``PIXEL_SEARCH_ENDPOINT`` + optional ``PIXEL_SEARCH_API_KEY``)

None of them are enabled by default.  If credentials are missing the provider
reports ``unconfigured`` instead of pretending a search happened.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from engine.logging_setup import get_logger
from providers.base import ProviderState, ResearchResult, ResearchSource
from providers.research.base import ResearchProvider

log = get_logger("research")

DEFAULT_TIMEOUT = 30


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


class _HttpResearchProvider(ResearchProvider):
    """Shared retry/error plumbing for HTTP search providers."""

    env_vars: tuple[str, ...] = ()
    endpoint: str = ""

    # -- config ------------------------------------------------------------ #
    def _api_key(self) -> str | None:
        return _env(*self.env_vars)

    def is_configured(self) -> bool:
        return bool(self._api_key())

    # -- http -------------------------------------------------------------- #
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key()}",
        }
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"{self.name} returned HTTP {response.status_code}: {response.text[:400]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"{self.name} returned non-JSON response") from exc

    # -- collection -------------------------------------------------------- #
    def search(self, queries: list[str], max_results: int) -> ResearchResult:
        if not self.is_configured():
            return ResearchResult(
                provider_name=self.name,
                state=ProviderState.UNCONFIGURED,
                queries=list(queries),
                notes=self.configuration_hint(),
            )

        collected: list[ResearchSource] = []
        errors: list[str] = []
        seen_urls: set[str] = set()

        for query in queries:
            try:
                raw_results = self._query_once(query, max_results)
            except Exception as exc:  # noqa: BLE001 - provider failures are data, not crashes
                log.warning("Research query failed (%s): %s", query, exc)
                errors.append(f"{query}: {exc}")
                continue

            for entry in raw_results:
                url = entry.get("url")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                collected.append(
                    ResearchSource(
                        source_id=f"S{len(collected) + 1}",
                        title=str(entry.get("title") or "Untitled source"),
                        url=url,
                        publication_date=entry.get("publication_date"),
                        provider=self.name,
                        extract=str(entry.get("extract") or "")[:4000],
                        score=entry.get("score"),
                    )
                )

        state = ProviderState.READY
        if errors and collected:
            state = ProviderState.PARTIAL
        elif errors and not collected:
            state = ProviderState.FAILED

        return ResearchResult(
            provider_name=self.name,
            state=state,
            sources=collected,
            queries=list(queries),
            error="; ".join(errors) if errors else None,
            notes=self.configuration_hint() if not collected else "",
        )

    def _query_once(self, query: str, max_results: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class TavilyResearchProvider(_HttpResearchProvider):
    """Tavily search API: https://docs.tavily.com/"""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"
    env_vars = ("TAVILY_API_KEY",)

    def configuration_hint(self) -> str:
        return (
            "Set TAVILY_API_KEY in the environment, then set research.provider to "
            "'tavily' in config/pipeline_settings.json."
        )

    def _query_once(self, query: str, max_results: int) -> list[dict[str, Any]]:
        payload = {
            "api_key": self._api_key(),
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results,
        }
        data = self._post(payload)
        results = []
        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "extract": item.get("content"),
                    "score": item.get("score"),
                    "publication_date": item.get("published_date"),
                }
            )
        return results


class SerperResearchProvider(_HttpResearchProvider):
    """Serper.dev Google search: https://serper.dev/"""

    name = "serper"
    endpoint = "https://google.serper.dev/search"
    env_vars = ("SERPER_API_KEY",)

    def configuration_hint(self) -> str:
        return (
            "Set SERPER_API_KEY in the environment, then set research.provider to "
            "'serper' in config/pipeline_settings.json."
        )

    def _query_once(self, query: str, max_results: int) -> list[dict[str, Any]]:
        data = self._post({"q": query, "num": max_results})
        results = []
        for item in data.get("organic", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "extract": item.get("snippet"),
                    "publication_date": item.get("date"),
                }
            )
        return results


class CustomHttpResearchProvider(_HttpResearchProvider):
    """Bring-your-own search endpoint.

    Contract: POST JSON ``{"query": str, "limit": int}`` and return either a
    list of results or ``{"results": [...]}`` where each result has
    ``title``, ``url``, ``snippet``/``content``/``extract``.
    """

    name = "custom_http"
    env_vars = ("PIXEL_SEARCH_API_KEY",)

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint or _env("PIXEL_SEARCH_ENDPOINT") or ""

    def is_configured(self) -> bool:
        if not self.endpoint:
            return False
        # A self-hosted endpoint may legitimately need no key.
        return True

    def configuration_hint(self) -> str:
        return (
            "Set PIXEL_SEARCH_ENDPOINT (and optionally PIXEL_SEARCH_API_KEY) in the "
            "environment, then set research.provider to 'custom_http' in "
            "config/pipeline_settings.json."
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        response = requests.post(
            self.endpoint, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"custom_http returned HTTP {response.status_code}: {response.text[:400]}"
            )
        data = response.json()
        if isinstance(data, list):
            return {"results": data}
        return data

    def _query_once(self, query: str, max_results: int) -> list[dict[str, Any]]:
        data = self._post({"query": query, "limit": max_results})
        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url") or item.get("link"),
                    "extract": item.get("extract")
                    or item.get("content")
                    or item.get("snippet"),
                    "publication_date": item.get("publication_date")
                    or item.get("published_date"),
                    "score": item.get("score"),
                }
            )
        return results


__all__ = [
    "CustomHttpResearchProvider",
    "SerperResearchProvider",
    "TavilyResearchProvider",
]
