"""DeepSeek client tests: budgeting always, live server tests when reachable."""

from __future__ import annotations

import unittest
from unittest import mock

from config.settings import ModelSettings, load_config
from engine.deepseek_client import (
    ChatResult,
    ContextBudget,
    ContextOverflowError,
    DeepSeekClient,
    InvalidStructuredOutput,
    estimate_tokens,
    truncate_to_tokens,
)
from engine.json_utils import extract_json

CONFIG = load_config()
TOY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


class TestTokenBudgeting(unittest.TestCase):
    def test_estimate_tokens_is_conservative(self) -> None:
        text = "word " * 1000
        self.assertGreater(estimate_tokens(text), 250)
        self.assertLess(estimate_tokens(text), 1500)

    def test_empty_text(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)

    def test_truncate_keeps_head(self) -> None:
        text = "A" * 4000 + "TAIL"
        truncated = truncate_to_tokens(text, 100, keep="head")
        self.assertLess(len(truncated), len(text))
        self.assertNotIn("TAIL", truncated)

    def test_truncate_keeps_tail(self) -> None:
        text = "HEAD" + "A" * 4000
        truncated = truncate_to_tokens(text, 100, keep="tail")
        self.assertTrue(truncated.startswith("..."))
        self.assertTrue(truncated.endswith("A" * 10))

    def test_budget_fit(self) -> None:
        budget = ContextBudget(context_size=4096, reserve_output=512)
        self.assertGreater(budget.prompt_tokens, 0)
        self.assertLess(budget.prompt_tokens, 4096 - 512)
        self.assertTrue(budget.fits("short prompt", 100))
        self.assertFalse(budget.fits("word " * 5000, 100))


class TestContextOverflow(unittest.TestCase):
    def test_oversized_prompt_raises_before_request(self) -> None:
        client = DeepSeekClient(CONFIG.model)
        try:
            with self.assertRaises(ContextOverflowError):
                client.chat(
                    [{"role": "user", "content": "word " * 40000}],
                    max_tokens=100,
                    retries=1,
                )
        finally:
            client.close()

    def test_reported_context_matches_server_config(self) -> None:
        client = DeepSeekClient(CONFIG.model)
        try:
            self.assertGreaterEqual(client.context_size, 512)
        finally:
            client.close()


class TestRepairLoop(unittest.TestCase):
    def make_client(self) -> DeepSeekClient:
        client = DeepSeekClient(CONFIG.model)
        client._response_format_supported = False
        return client

    def test_invalid_then_valid(self) -> None:
        client = self.make_client()
        try:
            responses = [
                ChatResult(content="Here you go, hope this helps!"),
                ChatResult(content='{"ok": true}'),
            ]
            with mock.patch.object(client, "chat", side_effect=responses) as patched:
                data = client.chat_json(
                    system="Return JSON.",
                    user="Set ok to true.",
                    schema=TOY_SCHEMA,
                    max_tokens=32,
                    repair_attempts=2,
                )
            self.assertEqual(data, {"ok": True})
            self.assertEqual(patched.call_count, 2)
        finally:
            client.close()

    def test_schema_violation_then_valid(self) -> None:
        client = self.make_client()
        try:
            responses = [
                ChatResult(content='{"ok": "yes"}'),
                ChatResult(content='```json\n{"ok": false}\n```'),
            ]
            with mock.patch.object(client, "chat", side_effect=responses):
                data = client.chat_json(
                    system="Return JSON.",
                    user="Set ok.",
                    schema=TOY_SCHEMA,
                    max_tokens=32,
                    repair_attempts=2,
                )
            self.assertEqual(data, {"ok": False})
        finally:
            client.close()

    def test_exhausted_repairs_raise_with_raw(self) -> None:
        client = self.make_client()
        try:
            with mock.patch.object(client, "chat", return_value=ChatResult(content="not json")):
                with self.assertRaises(InvalidStructuredOutput) as context:
                    client.chat_json(
                        system="Return JSON.",
                        user="Anything.",
                        schema=TOY_SCHEMA,
                        max_tokens=16,
                        repair_attempts=1,
                    )
            self.assertIn("2 attempt(s)", str(context.exception))
            self.assertEqual(context.exception.raw, "not json")
        finally:
            client.close()

    def test_extra_field_is_rejected(self) -> None:
        client = self.make_client()
        try:
            responses = [
                ChatResult(content='{"ok": true, "sneaky": 1}'),
                ChatResult(content='{"ok": true}'),
            ]
            with mock.patch.object(client, "chat", side_effect=responses):
                data = client.chat_json(
                    system="Return JSON.",
                    user="Anything.",
                    schema=TOY_SCHEMA,
                    max_tokens=16,
                    repair_attempts=1,
                )
            self.assertEqual(data, {"ok": True})
            self.assertNotIn("sneaky", data)
        finally:
            client.close()

    def test_unreachable_server_raises_unavailable(self) -> None:
        import copy

        raw = copy.deepcopy(CONFIG.model.raw)
        raw["base_url"] = "http://127.0.0.1:59999/v1"
        raw.setdefault("request", {})
        raw["request"]["max_retries"] = 1
        raw["request"]["connect_timeout_seconds"] = 1
        client = DeepSeekClient(ModelSettings(raw=raw))
        try:
            status = client.probe(timeout=1)
            self.assertFalse(status.online)
            self.assertTrue(status.detail)
        finally:
            client.close()


class TestLiveServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = DeepSeekClient(CONFIG.model)
        cls.status = cls.client.probe()
        if not cls.status.online:
            cls.client.close()
            raise unittest.SkipTest(f"llama-server is offline: {cls.status.detail}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_probe_reports_context(self) -> None:
        self.assertTrue(self.status.online)
        self.assertGreaterEqual(self.status.context_size, 512)
        self.assertGreaterEqual(self.status.latency_ms, 0)

    def test_small_json_completion(self) -> None:
        data = self.client.chat_json(
            system="You return only JSON. Do not explain.",
            user='Return an object with the boolean field "ok" set to true.',
            schema=TOY_SCHEMA,
            max_tokens=48,
            temperature=0.0,
            label="test:tiny",
            repair_attempts=1,
        )
        self.assertEqual(data, {"ok": True})

    def test_reasoning_never_leaks_into_content(self) -> None:
        result = self.client.chat(
            [
                {"role": "system", "content": "You return only JSON."},
                {"role": "user", "content": 'Return {"ok": true} as raw JSON.'},
            ],
            max_tokens=48,
            temperature=0.0,
            label="test:leak",
            retries=1,
        )
        self.assertNotIn("<think", result.content.lower())
        # With --reasoning-budget 0 the server does not produce reasoning at all.
        self.assertNotIn("<think", result.reasoning.lower())
        self.assertIsNotNone(extract_json(result.content))


if __name__ == "__main__":
    unittest.main()
