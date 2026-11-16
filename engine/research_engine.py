"""Stage 2 - Research Engine.

Two clearly separated halves:

1. **Collection** - delegated to a replaceable provider
   (``providers/research``).  DeepSeek has no internet access, so collection
   never happens inside the model.
2. **Analysis** - DeepSeek reads whatever the provider actually returned and
   separates verified facts from uncertain claims.

Honesty guarantees enforced in code, not just in the prompt:

* ``sources``, ``queries`` and ``provider`` always come from the provider,
  never from the model.
* ``research_status`` is computed from the provider outcome.  A model cannot
  upgrade "unavailable" to "available".
"""

from __future__ import annotations

import re
from typing import Any

from engine.ai_stage import AIStage
from engine.logging_setup import get_logger
from engine.prompt_library import bullets
from providers.base import ProviderState, ResearchResult
from providers.research import get_research_provider

log = get_logger("research")

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are",
    "was", "were", "how", "why", "what", "when", "who", "that", "this", "with",
}


class ResearchEngine(AIStage):
    """Produces ``Research.json``."""

    stage_name = "research_engine"

    # -- query building ---------------------------------------------------- #
    def build_queries(self, topic: str, brief: dict[str, Any] | None = None) -> list[str]:
        """Derive search queries deterministically from the topic and brief."""
        queries: list[str] = [topic.strip()]
        if brief:
            for question in (brief.get("key_questions") or [])[:4]:
                cleaned = re.sub(r"[?!]+$", "", str(question)).strip()
                if cleaned and cleaned.lower() not in {q.lower() for q in queries}:
                    queries.append(cleaned)
            for term in (brief.get("key_terms") or [])[:3]:
                name = str(term.get("term", "")).strip()
                if name and len(name) > 2:
                    queries.append(f"{name} {topic.split()[0] if topic.split() else ''}".strip())
        limit = max(1, self.pipeline.research_max_queries)
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for query in queries:
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(query)
        return unique[:limit]

    # -- collection -------------------------------------------------------- #
    def collect(self, queries: list[str]) -> ResearchResult:
        provider = get_research_provider(self.pipeline.research_provider)
        if not provider.is_configured():
            log.warning(
                "Research provider '%s' is not configured - research will be marked unavailable",
                provider.name,
            )
            return ResearchResult(
                provider_name=provider.name,
                state=ProviderState.UNCONFIGURED,
                sources=[],
                queries=queries,
                notes=provider.configuration_hint(),
            )

        log.info("Collecting research via '%s' (%d queries)", provider.name, len(queries))
        result = provider.search(queries, self.pipeline.research_max_results)
        log.info(
            "Research collection: %s, %d source(s)",
            result.state.value,
            len(result.sources),
        )
        return result

    # -- analysis ---------------------------------------------------------- #
    def analyze(
        self,
        topic: str,
        brief: dict[str, Any],
        collection: ResearchResult,
    ) -> dict[str, Any]:
        sources_text = self._render_sources(collection)
        digest_of_brief = str(brief.get("central_message", ""))[:400]

        report = self.run_json(
            prompt_name="research_analysis",
            schema_name="research_report",
            label="research_analysis",
            temperature=0.4,
            max_tokens=min(2000, self.client.budget.prompt_tokens // 2),
            PROFILE=self.profile_block(),
            TOPIC=topic,
            CENTRAL_MESSAGE=digest_of_brief,
            RESEARCH_STATUS=collection.research_status,
            SOURCES=self.fit(sources_text, keep="head", label="sources"),
        )

        return self._merge(report, topic, collection)

    # -- merging ----------------------------------------------------------- #
    def _merge(
        self,
        report: dict[str, Any],
        topic: str,
        collection: ResearchResult,
    ) -> dict[str, Any]:
        from engine.file_manager import utc_now_iso

        # Provider facts are authoritative; the model cannot rewrite them.
        report["schema_version"] = "1.0"
        report["topic"] = topic
        report["collected_at"] = utc_now_iso()
        report["research_status"] = collection.research_status
        report["provider"] = collection.to_provider_block()
        report["queries"] = list(collection.queries)
        report["sources"] = [source.to_dict() for source in collection.sources]

        # Drop source references that do not exist.
        valid_ids = {source.source_id for source in collection.sources}
        for key in ("verified_facts", "uncertain_claims", "statistics"):
            for item in report.get(key, []) or []:
                refs = [ref for ref in item.get("source_ids", []) if ref in valid_ids]
                item["source_ids"] = refs

        if collection.research_status == "unavailable":
            # Nothing can be verified without sources. Keep only genuinely
            # uncontroversial background in verified_facts.
            for fact in report.get("verified_facts", []) or []:
                fact["source_ids"] = []
                fact["confidence"] = "low"
                fact["notes"] = (
                    "No external source collected. Treat as general background only."
                )
            for stat in report.get("statistics", []) or []:
                stat["verified"] = False
                stat.setdefault("caveats", "No source collected; do not present as fact.")

        guidance = report.setdefault(
            "guidance_for_script",
            {"safe_to_state": [], "state_with_hedging": [], "avoid": []},
        )
        for key in ("safe_to_state", "state_with_hedging", "avoid"):
            guidance.setdefault(key, [])

        if collection.research_status == "unavailable":
            guidance["avoid"] = list(
                dict.fromkeys(
                    guidance["avoid"]
                    + [
                        "Any specific statistic, percentage or dollar figure",
                        "Any claim that requires a citation the script cannot provide",
                    ]
                )
            )

        report["missing_evidence"] = (report.get("missing_evidence") or [])[:10]
        report["conflicts"] = (report.get("conflicts") or [])[:6]
        report["uncertain_claims"] = (report.get("uncertain_claims") or [])[:14]
        report["verified_facts"] = (report.get("verified_facts") or [])[:14]
        report["statistics"] = (report.get("statistics") or [])[:10]
        return report

    # -- rendering --------------------------------------------------------- #
    @staticmethod
    def _untrusted(text: str) -> str:
        """Neutralise markup without rewriting the evidence itself."""
        value = str(text or "").replace("\x00", " ")
        # Prevent source text from closing the <sources> boundary or opening
        # pseudo-tags that a model might mistake for instructions.
        value = re.sub(r"</?\s*sources\s*>", "[tag removed]", value, flags=re.IGNORECASE)
        value = value.replace("<", "\u2039").replace(">", "\u203a")
        return " ".join(value.split())

    def _render_sources(self, collection: ResearchResult) -> str:
        if not collection.sources:
            return (
                "NONE. No external source was collected for this topic.\n"
                f"Provider: {collection.provider_name}. "
                f"Reason: {collection.notes or 'provider not configured'}."
            )

        limit = self.pipeline.research_source_chars
        chunks: list[str] = []
        for source in collection.sources:
            extract = self._untrusted(source.extract)
            if len(extract) > limit:
                extract = extract[:limit].rstrip() + "..."
            title = self._untrusted(source.title)
            url = self._untrusted(source.url or "n/a")
            published = (
                f" | published: {self._untrusted(source.publication_date)}"
                if source.publication_date
                else ""
            )
            chunks.append(
                f"[{source.source_id}] {title}{published}\n"
                f"URL: {url}\n"
                f"Extract: {extract or '(no extract returned)'}"
            )
        return "\n\n".join(chunks)


__all__ = ["ResearchEngine"]
