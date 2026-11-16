"""Tests for project validation and the resumable pipeline state machine."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from config.settings import load_config
from engine.channel_profile import load_profile
from engine.file_manager import atomic_write_json, atomic_write_text, read_json
from engine.output_validator import validate_project
from engine.pipeline import DOWNSTREAM, Pipeline, Stage, StageRecord
from engine.narration import format_shorts
from tests.helpers import (
    TempWorkspace,
    fixture_shorts,
    make_config,
    make_project,
    seed_minimal_project,
)

CONFIG = load_config()
PROFILE = load_profile("pixel_tech", CONFIG.profiles_dir)


class TestOutputValidation(unittest.TestCase):
    def test_seeded_project_is_valid(self) -> None:
        with TempWorkspace() as root:
            _, paths = make_project(root, PROFILE.id)
            seed_minimal_project(paths)
            result = validate_project(paths, CONFIG.schemas_dir, PROFILE)
            self.assertTrue(result.ok, msg=json.dumps(result.failures, indent=2))
            self.assertIn("Long.txt", result.checked)
            self.assertIn("Short.txt", result.checked)
            self.assertIn("Long_Production.json", result.checked)

    def test_wrong_number_of_shorts_fails(self) -> None:
        with TempWorkspace() as root:
            _, paths = make_project(root, PROFILE.id)
            seed_minimal_project(paths, shorts_count=3)
            result = validate_project(paths, CONFIG.schemas_dir, PROFILE)
            self.assertFalse(result.ok)
            self.assertTrue(
                any("parsable Short" in failure for failure in result.failures),
                msg=result.failures,
            )

    def test_metadata_in_long_narration_fails(self) -> None:
        with TempWorkspace() as root:
            _, paths = make_project(root, PROFILE.id)
            seed_minimal_project(paths)
            atomic_write_text(
                paths.long_script,
                "Plain narration starts here.\n\nImage prompt: a red door\n\nMore narration.",
            )
            result = validate_project(paths, CONFIG.schemas_dir, PROFILE)
            self.assertFalse(result.ok)
            self.assertTrue(
                any("Long.txt" in failure and "visual direction" in failure for failure in result.failures),
                msg=result.failures,
            )

    def test_malformed_json_fails(self) -> None:
        with TempWorkspace() as root:
            _, paths = make_project(root, PROFILE.id)
            seed_minimal_project(paths)
            paths.outline.write_text("{not json", encoding="utf-8")
            result = validate_project(paths, CONFIG.schemas_dir, PROFILE)
            self.assertFalse(result.ok)
            self.assertTrue(any("Outline.json" in failure for failure in result.failures))

    def test_schema_violation_reports_fields(self) -> None:
        with TempWorkspace() as root:
            _, paths = make_project(root, PROFILE.id)
            seed_minimal_project(paths)
            brief = read_json(paths.brief)
            del brief["central_message"]
            atomic_write_json(paths.brief, brief)
            result = validate_project(paths, CONFIG.schemas_dir, PROFILE)
            self.assertFalse(result.ok)
            self.assertTrue(
                any("central_message" in failure for failure in result.failures),
                msg=result.failures,
            )

    def test_missing_file_fails(self) -> None:
        with TempWorkspace() as root:
            _, paths = make_project(root, PROFILE.id)
            seed_minimal_project(paths)
            paths.quality.unlink()
            result = validate_project(paths, CONFIG.schemas_dir, PROFILE)
            self.assertFalse(result.ok)
            self.assertTrue(any("QualityReport.json: missing" in failure for failure in result.failures))

    def test_bad_short_heading_is_detected(self) -> None:
        with TempWorkspace() as root:
            _, paths = make_project(root, PROFILE.id)
            seed_minimal_project(paths)
            text = paths.short_script.read_text(encoding="utf-8")
            atomic_write_text(paths.short_script, text.replace("SHORT 02", "Short #2"))
            result = validate_project(paths, CONFIG.schemas_dir, PROFILE)
            self.assertFalse(result.ok)
            self.assertTrue(any("parsable Short" in failure for failure in result.failures))


class TestPipelineState(unittest.TestCase):
    def make_pipeline(self, root: Path, **kwargs) -> Pipeline:
        config = make_config(root)
        return Pipeline(config, PROFILE, "Test Topic", **kwargs)

    def test_new_project_starts_with_nothing_complete(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root)
            for stage in (Stage.TOPIC, Stage.SCRIPT, Stage.QUALITY):
                self.assertFalse(pipeline.is_complete(stage))
            state = read_json(pipeline.paths.state)
            self.assertEqual(state["profile_id"], PROFILE.id)
            self.assertEqual(state["status"], "initialised")
            self.assertFalse(state["approval"]["script_approved"])

    def test_complete_requires_artifacts(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root)
            pipeline._set_record(Stage.TOPIC, StageRecord(status="complete"))
            self.assertFalse(pipeline.is_complete(Stage.TOPIC), "artifact missing but stage reported complete")
            atomic_write_json(pipeline.paths.brief, {"schema_version": "1.0"})
            self.assertTrue(pipeline.is_complete(Stage.TOPIC))

    def test_force_stage_overrides_completion(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root, force_stages=["topic"])
            pipeline._set_record(Stage.TOPIC, StageRecord(status="complete"))
            atomic_write_json(pipeline.paths.brief, {"schema_version": "1.0"})
            self.assertFalse(pipeline.is_complete(Stage.TOPIC))

    def test_invalidating_a_stage_resets_downstream(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root)
            for stage in DOWNSTREAM[Stage.PLAN] + (Stage.PLAN, Stage.RESEARCH, Stage.TOPIC):
                pipeline._set_record(stage, StageRecord(status="complete"))
            pipeline._invalidate_downstream(Stage.PLAN)
            for stage in DOWNSTREAM[Stage.PLAN]:
                self.assertEqual(pipeline.record(stage).status, "pending", msg=stage.value)
                self.assertIn("Invalidated", pipeline.record(stage).notes)
            self.assertEqual(pipeline.record(Stage.PLAN).status, "complete")
            self.assertEqual(pipeline.record(Stage.RESEARCH).status, "complete")

    def test_state_survives_reload(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root)
            pipeline._set_record(Stage.RESEARCH, StageRecord(status="complete", attempts=2))
            reloaded = self.make_pipeline(root, resume=True)
            record = reloaded.record(Stage.RESEARCH)
            self.assertEqual(record.status, "complete")
            self.assertEqual(record.attempts, 2)
            self.assertIn("resumed_at", reloaded.state)

    def test_no_resume_starts_fresh_state(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root)
            pipeline._set_record(Stage.RESEARCH, StageRecord(status="complete"))
            fresh = self.make_pipeline(root, resume=False)
            self.assertEqual(fresh.record(Stage.RESEARCH).status, "pending")

    def test_approval_gate(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root, auto_approve=False)
            self.assertFalse(pipeline._is_script_approved())
            pipeline.approve_script(mode="test")
            self.assertTrue(pipeline._is_script_approved())
            state = read_json(pipeline.paths.state)
            self.assertTrue(state["approval"]["script_approved"])
            self.assertEqual(state["approval"]["mode"], "test")

    def test_auto_approve_is_recorded(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root, auto_approve=True)
            self.assertFalse(pipeline._is_script_approved())
            pipeline._request_approval()
            self.assertTrue(pipeline._is_script_approved())
            state = read_json(pipeline.paths.state)
            self.assertEqual(state["approval"]["mode"], "auto")

    def test_stage_order_is_declared_in_state(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root)
            self.assertEqual(len(pipeline.state["stage_order"]), 13)
            self.assertLess(
                pipeline.state["stage_order"].index("visuals"),
                pipeline.state["stage_order"].index("quality"),
                "quality must run after visuals",
            )

    def test_status_never_claims_success(self) -> None:
        with TempWorkspace() as root:
            pipeline = self.make_pipeline(root)
            status = pipeline.status()
            self.assertIn("stages", status)
            self.assertNotEqual(status.get("status"), "complete")


if __name__ == "__main__":
    unittest.main()
