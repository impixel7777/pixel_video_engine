"""Configurable client for the local llama.cpp / DeepSeek OpenAI-compatible API.

Design notes
------------
* All connection, retry and generation parameters come from
  :mod:`config.settings` — nothing is hardcoded here.
* DeepSeek-R1 is a reasoning model.  llama.cpp may return the reasoning in a
  separate ``reasoning_content`` field **and leave ``content`` empty** while the
  model is still thinking.  That reasoning is never returned to callers as
  narration: :class:`ChatResult.content` always holds only the final answer.
* Structured output is requested as JSON, validated against a schema, and
  repaired with a bounded number of retries.  Invalid JSON is never silently
  accepted.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import requests

from config.settings import ModelSettings
from engine.json_utils import (
    JsonExtractionError,
    extract_json,
    format_validation_errors,
    strip_reasoning,
    validate_json,
)
from engine.logging_setup import get_logger

log = get_logger("deepseek")

#: Rough English character-per-token ratio. Deliberately conservative so the
#: pipeline under-fills the context window rather than overflowing it.
CHARS_PER_TOKEN = 3.4

#: Safety margin kept free at the end of the context window.
CONTEXT_SAFETY_TOKENS = 24


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DeepSeekError(RuntimeError):
    """Base class for inference errors."""


class ServerUnavailableError(DeepSeekError):
    """The llama.cpp server is not reachable."""


class ContextOverflowError(DeepSeekError):
    """A prompt does not fit in the configured context window."""


class ReasoningBudgetExhausted(DeepSeekError):
    """The model spent its entire output budget on hidden reasoning."""


class TruncatedOutputError(DeepSeekError):
    """The server stopped generation before the answer was complete.

    A truncated object must never be "repaired" into schema validity: brackets
    can be closed mechanically, but missing narration cannot be invented.
    """

    def __init__(self, message: str, *, partial: str = "", label: str = "") -> None:
        super().__init__(message)
        self.partial = partial
        self.label = label


class InvalidStructuredOutput(DeepSeekError):
    """The model did not return schema-valid JSON after all repair attempts."""

    def __init__(self, message: str, *, raw: str = "", errors: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.raw = raw
        self.errors = list(errors)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class ChatResult:
    """One completed model call."""

    content: str
    reasoning: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    attempts: int = 1
    label: str = ""

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens") or 0)

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens") or 0)


@dataclass
class ServerStatus:
    """Result of a connectivity probe."""

    online: bool
    base_url: str
    detail: str = ""
    models: list[str] = field(default_factory=list)
    context_size: int = 0
    latency_ms: int = 0


# --------------------------------------------------------------------------- #
# Token budgeting
# --------------------------------------------------------------------------- #


def estimate_tokens(text: str) -> int:
    """Estimate token count for English prose/JSON."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN) + 1)


def truncate_to_tokens(text: str, max_tokens: int, *, keep: str = "head") -> str:
    """Cut *text* so its estimated token count does not exceed *max_tokens*."""
    if max_tokens <= 0 or not text:
        return ""
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    if keep == "tail":
        return "..." + text[-max_chars:]
    return text[:max_chars].rstrip() + "..."


@dataclass
class ContextBudget:
    """Tracks how much of the fixed context window a call may use."""

    context_size: int
    reserve_output: int

    @property
    def prompt_tokens(self) -> int:
        return max(64, self.context_size - self.reserve_output - CONTEXT_SAFETY_TOKENS)

    def fits(self, prompt_text: str, max_tokens: int | None = None) -> bool:
        output = max_tokens if max_tokens is not None else self.reserve_output
        return estimate_tokens(prompt_text) + output + CONTEXT_SAFETY_TOKENS <= self.context_size

    def fit(self, text: str, *, keep: str = "head", label: str = "prompt") -> str:
        """Truncate a *single variable section* so the overall prompt fits."""
        allowed = self.prompt_tokens
        fitted = truncate_to_tokens(text, allowed, keep=keep)
        if fitted != text:
            log.warning(
                "Truncated %s from ~%d to ~%d tokens to fit the %d token context window",
                label,
                estimate_tokens(text),
                estimate_tokens(fitted),
                self.context_size,
            )
        return fitted


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class DeepSeekClient:
    """Thin, typed wrapper around ``/v1/chat/completions``."""

    def __init__(self, settings: ModelSettings, *, session: requests.Session | None = None) -> None:
        self.settings = settings
        self._session = session or requests.Session()
        self._response_format_supported: bool | None = None
        self._detected_context: int | None = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_seconds = 0.0
        self.call_count = 0

    # -- plumbing ---------------------------------------------------------- #
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.settings.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @property
    def context_size(self) -> int:
        """Effective context window: auto-detected from the server when possible."""
        return self._detected_context or self.settings.context_size

    @property
    def budget(self) -> ContextBudget:
        return ContextBudget(
            context_size=self.context_size,
            reserve_output=min(self.settings.default_max_tokens, self.context_size // 2),
        )

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "DeepSeekClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- health ------------------------------------------------------------ #
    def _detect_context(self) -> int | None:
        """Read the live context window from llama.cpp's ``/props`` endpoint."""
        root = self.settings.base_url.rsplit("/v1", 1)[0]
        try:
            response = self._session.get(
                f"{root}/props",
                headers=self._headers(),
                timeout=self.settings.connect_timeout_seconds,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
        except (requests.RequestException, ValueError):
            return None

        n_ctx = (
            payload.get("default_generation_settings", {}).get("params", {}).get("n_ctx")
            or payload.get("n_ctx")
            or payload.get("default_generation_settings", {}).get("n_ctx")
        )
        try:
            value = int(n_ctx)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def probe(self, *, timeout: float | None = None) -> ServerStatus:
        """Check that the configured server answers before a pipeline starts."""
        started = time.time()
        url = self.settings.models_url
        try:
            response = self._session.get(
                url,
                headers=self._headers(),
                timeout=timeout or self.settings.connect_timeout_seconds,
            )
        except requests.ConnectionError as exc:
            return ServerStatus(
                online=False,
                base_url=self.settings.base_url,
                detail=f"Connection refused: {exc}",
                context_size=self.context_size,
            )
        except requests.Timeout:
            return ServerStatus(
                online=False,
                base_url=self.settings.base_url,
                detail="Timed out waiting for the server",
                context_size=self.context_size,
            )
        except requests.RequestException as exc:
            return ServerStatus(
                online=False,
                base_url=self.settings.base_url,
                detail=str(exc),
                context_size=self.context_size,
            )

        latency = int((time.time() - started) * 1000)
        if response.status_code != 200:
            return ServerStatus(
                online=False,
                base_url=self.settings.base_url,
                detail=f"HTTP {response.status_code}: {response.text[:300]}",
                context_size=self.context_size,
                latency_ms=latency,
            )

        models: list[str] = []
        try:
            payload = response.json()
            for entry in payload.get("data", []) or payload.get("models", []):
                name = entry.get("id") or entry.get("name")
                if name:
                    models.append(str(name))
        except ValueError:
            pass

        self._detected_context = self._detect_context()

        return ServerStatus(
            online=True,
            base_url=self.settings.base_url,
            detail="OK",
            models=models,
            context_size=self.context_size,
            latency_ms=latency,
        )

    def startup_hint(self) -> str:
        """Human-readable instructions for starting the local server."""
        command = self.settings.server_launch_command
        pretty = "`\n    ".join(command)
        lines = [
            "The local DeepSeek server is not reachable.",
            f"  Expected endpoint : {self.settings.chat_completions_url}",
            "",
            "Start it in a separate terminal with:",
            "",
            f"  & {pretty}",
            "",
            "Note: --reasoning-budget 0 is required on this machine. Without it the",
            "model spends the entire token budget on hidden reasoning and returns",
            "an empty answer.",
        ]
        return "\n".join(lines)

    # -- core call --------------------------------------------------------- #
    def _post_once(self, payload: dict[str, Any]) -> requests.Response:
        return self._session.post(
            self.settings.chat_completions_url,
            headers=self._headers(),
            json=payload,
            timeout=(self.settings.connect_timeout_seconds, self.settings.timeout_seconds),
        )

    def _build_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        response_format: dict[str, Any] | None,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.settings.default_temperature if temperature is None else temperature,
            "max_tokens": self.settings.default_max_tokens if max_tokens is None else max_tokens,
            "top_p": self.settings.top_p,
            "repeat_penalty": self.settings.repeat_penalty,
            "stream": False,
            "cache_prompt": True,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(dict(extra))
        return payload

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        response_format: dict[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        label: str = "chat",
        retries: int | None = None,
    ) -> ChatResult:
        """Run one chat completion with retries and reasoning isolation."""
        prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
        prompt_tokens = estimate_tokens(prompt_text)
        output_tokens = self.settings.default_max_tokens if max_tokens is None else max_tokens

        if prompt_tokens + output_tokens > self.context_size:
            raise ContextOverflowError(
                f"[{label}] prompt ~{prompt_tokens} tokens + max_tokens {output_tokens} "
                f"exceeds the {self.context_size} token context window. "
                "Reduce the input or restart llama-server with a larger -c "
                "(see server.context_size in config/model_settings.json)."
            )

        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            extra=extra,
        )

        attempt_limit = (self.settings.max_retries if retries is None else retries)
        last_error: Exception | None = None
        started_all = time.time()

        for attempt in range(1, attempt_limit + 1):
            started = time.time()
            try:
                response = self._post_once(payload)
            except requests.Timeout as exc:
                last_error = DeepSeekError(
                    f"[{label}] request timed out after "
                    f"{self.settings.timeout_seconds:.0f}s (attempt {attempt}/{attempt_limit})"
                )
                log.warning("%s", last_error)
                self._sleep(attempt)
                continue
            except requests.ConnectionError as exc:
                raise ServerUnavailableError(
                    f"[{label}] cannot reach {self.settings.chat_completions_url}: {exc}"
                ) from exc
            except requests.RequestException as exc:
                last_error = DeepSeekError(f"[{label}] request failed: {exc}")
                log.warning("%s", last_error)
                self._sleep(attempt)
                continue

            # The server may reject optional fields it does not implement.
            if response.status_code == 400 and "response_format" in payload:
                log.warning("[%s] server rejected response_format, retrying without it", label)
                payload.pop("response_format", None)
                self._response_format_supported = False
                continue

            if response.status_code >= 500:
                last_error = DeepSeekError(
                    f"[{label}] server error {response.status_code}: {response.text[:300]}"
                )
                log.warning("%s", last_error)
                self._sleep(attempt)
                continue

            if response.status_code != 200:
                raise DeepSeekError(
                    f"[{label}] HTTP {response.status_code}: {response.text[:500]}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                last_error = DeepSeekError(f"[{label}] response was not JSON: {exc}")
                log.warning("%s", last_error)
                self._sleep(attempt)
                continue

            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            raw_content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""

            if self.settings.strip_think_tags:
                raw_content = strip_reasoning(raw_content)
                if reasoning:
                    reasoning = strip_reasoning(reasoning) or reasoning

            finish_reason = str(choice.get("finish_reason") or "")
            content = raw_content.strip()

            usage = data.get("usage") or {}
            elapsed = time.time() - started

            self.call_count += 1
            self.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.total_completion_tokens += int(usage.get("completion_tokens") or 0)
            self.total_seconds += elapsed

            log.info(
                "[%s] %.1fs, %s completion tokens%s",
                label,
                elapsed,
                usage.get("completion_tokens", "?"),
                f", {len(content)} chars of final answer" if content else ", NO final answer",
            )
            if reasoning and log.isEnabledFor(10):
                log.debug("[%s] reasoning (%d chars) kept out of outputs", label, len(reasoning))

            if not content:
                if finish_reason == "length":
                    raise ReasoningBudgetExhausted(
                        f"[{label}] the model used all {output_tokens} output tokens on hidden "
                        "reasoning and returned no final answer. Restart llama-server with "
                        "--reasoning-budget 0 (recommended on this machine), or raise "
                        "max_tokens for this stage."
                    )
                last_error = DeepSeekError(f"[{label}] model returned an empty answer")
                log.warning("%s", last_error)
                self._sleep(attempt)
                continue

            if finish_reason == "length":
                # A partial answer is not a valid answer.  Retry once with more
                # output room when the context allows it, otherwise fail loudly.
                max_allowed = self.context_size - prompt_tokens - CONTEXT_SAFETY_TOKENS
                if attempt < attempt_limit and max_allowed > output_tokens + 64:
                    raised = min(max_allowed, max(output_tokens * 2, output_tokens + 512))
                    log.warning(
                        "[%s] output was truncated at %d tokens; retrying with max_tokens=%d",
                        label,
                        output_tokens,
                        raised,
                    )
                    output_tokens = raised
                    payload["max_tokens"] = raised
                    continue
                raise TruncatedOutputError(
                    f"[{label}] generation stopped at finish_reason='length' after "
                    f"{usage.get('completion_tokens', '?')} completion tokens. "
                    "The partial answer was rejected; it was not repaired or accepted.",
                    partial=content,
                    label=label,
                )

            return ChatResult(
                content=content,
                reasoning=reasoning,
                finish_reason=finish_reason,
                usage=usage,
                elapsed_seconds=time.time() - started_all,
                attempts=attempt,
                label=label,
            )

        raise last_error or DeepSeekError(f"[{label}] exhausted retries without a response")

    def _sleep(self, attempt: int) -> None:
        delay = self.settings.retry_backoff_seconds * attempt
        if delay > 0:
            time.sleep(min(delay, 30.0))

    # -- structured output -------------------------------------------------- #
    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        label: str = "json",
        repair_attempts: int | None = None,
        schema_name: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request JSON, validate it, and repair it a bounded number of times."""
        repair_limit = (
            self.settings.max_repair_attempts if repair_attempts is None else repair_attempts
        )
        response_format = None
        if self.settings.use_response_format and self._response_format_supported is not False:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name or label.replace(" ", "_"),
                    "schema": schema,
                    "strict": self.settings.strict_schema,
                },
            }

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_raw = ""
        last_errors: list[str] = []

        for attempt in range(repair_limit + 1):
            result = self.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                label=f"{label}" if attempt == 0 else f"{label}:repair{attempt}",
                extra=extra,
            )
            last_raw = result.content

            try:
                data = extract_json(result.content)
            except JsonExtractionError as exc:
                last_errors = [str(exc)]
                log.warning("[%s] invalid JSON on attempt %d: %s", label, attempt + 1, exc)
                messages = self._repair_messages(
                    system,
                    user,
                    last_raw,
                    [f"Response was not valid JSON: {exc}"],
                    schema,
                    max_tokens=max_tokens,
                )
                continue

            errors = validate_json(data, schema)
            if not errors:
                return data  # type: ignore[return-value]

            last_errors = errors
            log.warning(
                "[%s] JSON failed schema validation on attempt %d (%d issue(s))",
                label,
                attempt + 1,
                len(errors),
            )
            messages = self._repair_messages(
                system, user, last_raw, errors, schema, max_tokens=max_tokens
            )

        raise InvalidStructuredOutput(
            f"[{label}] no schema-valid JSON after {repair_limit + 1} attempt(s):\n"
            + format_validation_errors(last_errors),
            raw=last_raw,
            errors=last_errors,
        )

    def _repair_messages(
        self,
        system: str,
        user: str,
        previous: str,
        errors: Sequence[str],
        schema: dict[str, Any],
        *,
        max_tokens: int | None,
    ) -> list[dict[str, str]]:
        """Build a *bounded* repair request.

        The original system + user prompts are not duplicated verbatim alongside
        the previous answer.  The previous answer is included exactly once, the
        schema is reduced to its required shape, and the total prompt is kept
        inside ``context_size - max_tokens - safety``.
        """
        output_tokens = (
            self.settings.default_max_tokens if max_tokens is None else max_tokens
        )
        allowed_prompt = max(256, self.context_size - output_tokens - CONTEXT_SAFETY_TOKENS)

        error_text = truncate_to_tokens(
            format_validation_errors(list(errors)), 220, keep="head"
        )
        schema_text = json.dumps(self._compact_schema(schema), ensure_ascii=False)
        system_short = truncate_to_tokens(system, max(64, int(allowed_prompt * 0.18)))
        user_short = truncate_to_tokens(user, max(96, int(allowed_prompt * 0.30)))
        previous_short = truncate_to_tokens(
            previous, max(64, int(allowed_prompt * 0.28)), keep="head"
        )

        def repair_text() -> str:
            return (
                "Your previous answer was rejected. Correct it and return raw JSON only.\n\n"
                f"Problems:\n{error_text}\n\n"
                "Required shape (types and required keys):\n"
                f"{schema_text}\n\n"
                "Previous invalid answer (truncated if long):\n"
                f"{previous_short}\n\n"
                "No commentary, no markdown fences, no explanation. Every required field "
                "must be present and must match the schema."
            )

        messages = [
            {"role": "system", "content": system_short},
            {"role": "user", "content": user_short},
            {"role": "user", "content": repair_text()},
        ]

        # Shrink the least critical variable parts until the request fits.
        def total_tokens() -> int:
            return sum(estimate_tokens(str(message["content"])) for message in messages)

        while total_tokens() > allowed_prompt and len(previous_short) > 64:
            previous_short = truncate_to_tokens(
                previous_short, max(32, estimate_tokens(previous_short) // 2), keep="head"
            )
            messages[2]["content"] = repair_text()
        while total_tokens() > allowed_prompt and len(user_short) > 128:
            user_short = truncate_to_tokens(
                user_short, max(64, estimate_tokens(user_short) // 2), keep="head"
            )
            messages[1]["content"] = user_short
        while total_tokens() > allowed_prompt and len(error_text) > 200:
            error_text = error_text[: len(error_text) // 2]
            messages[2]["content"] = repair_text()

        if total_tokens() > allowed_prompt:
            raise ContextOverflowError(
                f"Repair request would need ~{total_tokens()} prompt tokens but only "
                f"~{allowed_prompt} fit with max_tokens={output_tokens}. "
                "Reduce the schema/output size or restart llama-server with a larger -c."
            )

        log.info(
            "Repair request bounded to ~%d prompt tokens (context %d, output %d)",
            total_tokens(),
            self.context_size,
            output_tokens,
        )
        return messages

    @classmethod
    def _compact_schema(cls, schema: Any, depth: int = 0) -> Any:
        """Reduce a JSON schema to the information a repair actually needs."""
        if depth > 3 or not isinstance(schema, dict):
            return {}
        if "enum" in schema:
            return {"enum": schema["enum"]}
        if "const" in schema:
            return {"const": schema["const"]}

        node: dict[str, Any] = {}
        if "type" in schema:
            node["type"] = schema["type"]
        for key in ("minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength"):
            if key in schema:
                node[key] = schema[key]
        if "required" in schema:
            node["required"] = schema["required"]
        if isinstance(schema.get("properties"), dict):
            node["properties"] = {
                name: cls._compact_schema(value, depth + 1)
                for name, value in schema["properties"].items()
            }
        if isinstance(schema.get("items"), dict):
            node["items"] = cls._compact_schema(schema["items"], depth + 1)
        for key in ("oneOf", "anyOf", "allOf"):
            if isinstance(schema.get(key), list):
                node[key] = [cls._compact_schema(item, depth + 1) for item in schema[key]]
        return node

    # -- stats -------------------------------------------------------------- #
    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "seconds": round(self.total_seconds, 1),
            "average_tokens_per_second": round(
                self.total_completion_tokens / self.total_seconds, 2
            )
            if self.total_seconds
            else 0.0,
        }


__all__ = [
    "CHARS_PER_TOKEN",
    "ChatResult",
    "ContextBudget",
    "ContextOverflowError",
    "DeepSeekClient",
    "DeepSeekError",
    "InvalidStructuredOutput",
    "ReasoningBudgetExhausted",
    "ServerStatus",
    "ServerUnavailableError",
    "TruncatedOutputError",
    "estimate_tokens",
    "truncate_to_tokens",
]
