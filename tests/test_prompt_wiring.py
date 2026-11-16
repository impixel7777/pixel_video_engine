"""Static wiring checks between engine call sites and prompt templates.

These catch the most common refactor mistake: renaming a placeholder in a
template (or adding one) without updating the stage that renders it.  The
runtime would fail on the first model call, which is expensive; this fails in
milliseconds.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from config.settings import load_config
from engine.prompt_library import PromptLibrary

CONFIG = load_config()
ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"
LIBRARY = PromptLibrary(CONFIG.prompts_dir)

#: Keyword arguments consumed by ``AIStage.run_json`` itself, not placeholders.
CONTROL_ARGS = {
    "prompt_name",
    "schema_name",
    "label",
    "temperature",
    "max_tokens",
    "schema",
}


def run_json_calls() -> list[tuple[Path, int, set[str], str | None]]:
    calls: list[tuple[Path, int, set[str], str | None]] = []
    for path in sorted(ENGINE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name != "run_json":
                continue
            keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
            prompt: str | None = None
            for keyword in node.keywords:
                if (
                    keyword.arg == "prompt_name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    prompt = keyword.value.value
            calls.append((path, node.lineno, keywords, prompt))
    return calls


class TestPromptWiring(unittest.TestCase):
    def test_every_run_json_call_names_a_dynamic_or_known_prompt(self) -> None:
        calls = run_json_calls()
        self.assertGreaterEqual(len(calls), 7, msg="expected one call site per AI stage")
        for path, line, _keywords, prompt in calls:
            with self.subTest(file=path.name, line=line):
                if prompt is None:
                    # Dynamic prompt names exist (e.g. a helper used by more
                    # than one stage); they are checked at runtime.
                    continue
                self.assertIn(prompt, {row["name"] for row in LIBRARY.describe()})

    def test_placeholder_sets_match_exactly(self) -> None:
        for path, line, keywords, prompt in run_json_calls():
            if prompt is None:
                continue
            with self.subTest(file=path.name, line=line, prompt=prompt):
                required = set(LIBRARY.placeholders(prompt))
                supplied = {word for word in keywords if word not in CONTROL_ARGS}
                self.assertEqual(
                    required - supplied,
                    set(),
                    msg=f"{path.name}:{line} does not supply {sorted(required - supplied)}",
                )
                self.assertEqual(
                    supplied - required,
                    set(),
                    msg=f"{path.name}:{line} supplies unused {sorted(supplied - required)}",
                )

    def test_no_orphan_templates(self) -> None:
        used = {prompt for _path, _line, _kw, prompt in run_json_calls() if prompt}
        declared = {row["name"] for row in LIBRARY.describe()}
        # report.py or future stages may add prompts; orphans are a smell, not
        # necessarily a bug, so this only fails for templates under prompts/.
        self.assertEqual(declared - used, set(), msg="unused prompt templates")


if __name__ == "__main__":
    unittest.main()
