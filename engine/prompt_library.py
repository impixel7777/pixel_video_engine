"""External prompt templates.

Prompts live as text files under ``prompts/`` and use ``{{PLACEHOLDER}}``
markers.  Double braces were chosen deliberately: ordinary single braces appear
constantly in JSON examples inside the prompts, so ``str.format`` would break.

Rendering fails loudly when a placeholder is missing or left unfilled — a
silently half-rendered prompt is a production bug, not a warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


class PromptError(KeyError):
    """Raised when a prompt template cannot be rendered."""


@dataclass
class PromptLibrary:
    """Loads and renders prompt templates from a directory."""

    prompts_dir: Path
    _cache: dict[str, str] | None = None

    def _load(self, name: str) -> str:
        if self._cache is None:
            self._cache = {}
        if name in self._cache:
            return self._cache[name]
        path = self.prompts_dir / f"{name}.txt"
        if not path.is_file():
            raise PromptError(f"Prompt template not found: {path}")
        text = path.read_text(encoding="utf-8")
        self._cache[name] = text
        return text

    def placeholders(self, name: str) -> list[str]:
        """List the placeholders declared by a template."""
        return sorted(set(PLACEHOLDER.findall(self._load(name))))

    def render(self, name: str, **values: Any) -> str:
        """Render a template, requiring every declared placeholder to be supplied."""
        template = self._load(name)
        required = set(self.placeholders(name))
        provided = set(values)
        missing = required - provided
        if missing:
            raise PromptError(
                f"Prompt '{name}' is missing values for: {', '.join(sorted(missing))}"
            )

        def substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            value = values.get(key)
            if value is None:
                raise PromptError(f"Prompt '{name}' placeholder {key} was None")
            if isinstance(value, (list, tuple)):
                return "\n".join(f"- {item}" for item in value)
            if isinstance(value, dict):
                import json

                return json.dumps(value, indent=2, ensure_ascii=False)
            return str(value)

        rendered = PLACEHOLDER.sub(substitute, template)
        leftover = PLACEHOLDER.findall(rendered)
        if leftover:
            raise PromptError(
                f"Prompt '{name}' still contains placeholders after rendering: {leftover}"
            )
        return rendered.strip()

    def describe(self) -> list[dict[str, Any]]:
        """Introspection helper used by ``--doctor`` and tests."""
        rows: list[dict[str, Any]] = []
        if not self.prompts_dir.is_dir():
            return rows
        for path in sorted(self.prompts_dir.glob("*.txt")):
            rows.append(
                {
                    "name": path.stem,
                    "path": str(path),
                    "placeholders": self.placeholders(path.stem),
                    "chars": path.stat().st_size,
                }
            )
        return rows


def bullets(items: Any, empty: str = "(none)") -> str:
    """Render a list as compact bullet lines for prompt injection."""
    if not items:
        return empty
    if isinstance(items, str):
        return items
    return "\n".join(f"- {item}" for item in items)


__all__ = ["PromptError", "PromptLibrary", "bullets"]
