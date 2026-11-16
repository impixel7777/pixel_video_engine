"""Contract tests: JSON schemas, prompt templates, channel profiles."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import jsonschema

from config.settings import load_config
from engine.channel_profile import ProfileError, ProfileRegistry, load_profile
from engine.prompt_library import PromptError, PromptLibrary, bullets

CONFIG = load_config()
SCHEMAS_DIR = CONFIG.schemas_dir
PROMPTS_DIR = CONFIG.prompts_dir
PROFILES_DIR = CONFIG.profiles_dir

EXPECTED_PROMPTS = {
    "topic_analysis",
    "research_analysis",
    "content_planning",
    "long_script",
    "shorts_generation",
    "visual_planning",
    "audio_planning",
    "quality_review",
}


def schema_files() -> list[Path]:
    return sorted(SCHEMAS_DIR.glob("*.json"))


class TestSchemas(unittest.TestCase):
    def test_all_schemas_exist(self) -> None:
        names = {path.stem for path in schema_files()}
        expected = {
            "channel_profile",
            "content_brief",
            "research_report",
            "video_outline",
            "scene_direction",
            "production_plan",
            "shorts_plan",
            "section_script",
            "shorts_script",
            "audio_direction",
            "quality_review",
            "quality_report",
        }
        self.assertEqual(expected - names, set(), msg="missing schema files")

    def test_every_schema_is_valid_draft_2020_12(self) -> None:
        validator_cls = jsonschema.validators.validator_for(
            {"$schema": "https://json-schema.org/draft/2020-12/schema"}
        )
        for path in schema_files():
            with self.subTest(schema=path.name):
                schema = json.loads(path.read_text(encoding="utf-8"))
                validator_cls.check_schema(schema)
                self.assertEqual(schema.get("$schema"), "https://json-schema.org/draft/2020-12/schema")
                self.assertIn("$id", schema)
                self.assertIn("title", schema)

    def test_local_refs_resolve(self) -> None:
        for path in schema_files():
            schema = json.loads(path.read_text(encoding="utf-8"))
            refs: list[str] = []

            def walk(node: object) -> None:
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key == "$ref" and isinstance(value, str):
                            refs.append(value)
                        else:
                            walk(value)
                elif isinstance(node, list):
                    for item in node:
                        walk(item)

            walk(schema)
            defs = schema.get("$defs", {})
            for ref in refs:
                with self.subTest(schema=path.name, ref=ref):
                    self.assertTrue(ref.startswith("#/$defs/"), msg=f"external ref not allowed: {ref}")
                    name = ref.split("/")[-1]
                    self.assertIn(name, defs)

    def test_ai_output_schemas_forbid_additional_properties(self) -> None:
        """Structured AI contracts must be closed so the model cannot invent fields."""
        for name in ("scene_direction", "section_script", "shorts_script", "quality_review", "audio_direction"):
            with self.subTest(schema=name):
                schema = json.loads((SCHEMAS_DIR / f"{name}.json").read_text(encoding="utf-8"))
                self.assertFalse(
                    schema.get("additionalProperties", True),
                    msg=f"{name}.json allows additional properties",
                )


class TestPrompts(unittest.TestCase):
    def setUp(self) -> None:
        self.library = PromptLibrary(PROMPTS_DIR)

    def test_expected_templates_exist(self) -> None:
        found = {row["name"] for row in self.library.describe()}
        self.assertEqual(EXPECTED_PROMPTS - found, set())

    def test_every_placeholder_is_uppercase(self) -> None:
        for row in self.library.describe():
            with self.subTest(prompt=row["name"]):
                for placeholder in row["placeholders"]:
                    self.assertRegex(placeholder, r"^[A-Z0-9_]+$")

    def test_every_template_renders(self) -> None:
        for row in self.library.describe():
            with self.subTest(prompt=row["name"]):
                values = {
                    placeholder: f"VALUE_{placeholder}" for placeholder in row["placeholders"]
                }
                rendered = self.library.render(row["name"], **values)
                self.assertNotIn("{{", rendered)
                self.assertNotIn("}}", rendered)
                for placeholder, value in values.items():
                    self.assertIn(value, rendered)

    def test_missing_placeholder_raises(self) -> None:
        with self.assertRaises(PromptError):
            self.library.render("topic_analysis")

    def test_none_value_raises(self) -> None:
        placeholders = self.library.placeholders("topic_analysis")
        values = {placeholder: "ok" for placeholder in placeholders}
        values[placeholders[0]] = None
        with self.assertRaises(PromptError):
            self.library.render("topic_analysis", **values)

    def test_missing_template_raises(self) -> None:
        with self.assertRaises(PromptError):
            self.library.render("not_a_prompt")

    def test_bullets(self) -> None:
        self.assertEqual(bullets([]), "(none)")
        self.assertEqual(bullets(["a", "b"]), "- a\n- b")
        self.assertEqual(bullets("raw"), "raw")

    def test_no_prompt_contains_thinking_instructions(self) -> None:
        """Prompts must never ask the model to expose reasoning."""
        for row in self.library.describe():
            text = Path(row["path"]).read_text(encoding="utf-8").lower()
            with self.subTest(prompt=row["name"]):
                self.assertNotIn("<think", text)
                self.assertNotIn("show your reasoning", text)


class TestProfiles(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ProfileRegistry(PROFILES_DIR)

    def test_three_profiles_available(self) -> None:
        self.assertEqual(
            sorted(self.registry.available()),
            ["pixel_history", "pixel_science", "pixel_tech"],
        )

    def test_every_profile_loads_and_is_schema_valid(self) -> None:
        for profile_id in self.registry.available():
            with self.subTest(profile=profile_id):
                profile = self.registry.load(profile_id)
                self.assertEqual(profile.id, profile_id)
                self.assertTrue(profile.name)
                self.assertGreater(profile.shorts_count, 0)
                self.assertGreater(profile.long_duration_minutes, 0)
                self.assertIn(":", profile.long_aspect)
                self.assertIn(":", profile.short_aspect)

    def test_prompt_block_contains_identity(self) -> None:
        profile = self.registry.load("pixel_tech")
        block = profile.to_prompt_block()
        self.assertIn(profile.name, block)
        self.assertIn("Technology", block)
        self.assertGreater(len(block), 200)

    def test_fingerprint_is_stable_and_hashable(self) -> None:
        profile = self.registry.load("pixel_tech")
        first = json.dumps(profile.fingerprint(), sort_keys=True)
        second = json.dumps(profile.fingerprint(), sort_keys=True)
        self.assertEqual(first, second)

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(ProfileError):
            self.registry.load("does_not_exist")

    def test_invalid_profile_is_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = json.loads((PROFILES_DIR / "pixel_tech.json").read_text(encoding="utf-8"))
            broken["channel"]["id"] = "bad_profile"
            del broken["video"]
            (root / "bad_profile.json").write_text(
                json.dumps(broken, indent=2), encoding="utf-8"
            )
            registry = ProfileRegistry(root)
            with self.assertRaises(ProfileError):
                registry.load("bad_profile")

    def test_path_traversal_style_id_is_rejected(self) -> None:
        with self.assertRaises(ProfileError):
            self.registry.load("../pixel_tech")

    def test_duplicate_creates_a_loadable_profile(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = json.loads((PROFILES_DIR / "pixel_tech.json").read_text(encoding="utf-8"))
            (root / "pixel_tech.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
            registry = ProfileRegistry(root)
            new_path = registry.duplicate("pixel_tech", "pixel_copy", "Pixel Copy")
            self.assertTrue(new_path.is_file())
            created = registry.load("pixel_copy")
            self.assertEqual(created.name, "Pixel Copy")


if __name__ == "__main__":
    unittest.main()
