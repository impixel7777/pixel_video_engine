"""Tests for the asset manager's honest-pending behaviour."""

from __future__ import annotations

import unittest

from config.settings import load_config
from engine.asset_manager import AssetManager
from engine.file_manager import read_json
from tests.helpers import TempWorkspace, build_long_plan, make_config, make_project, seed_minimal_project

CONFIG = load_config()


class TestAssetPlanning(unittest.TestCase):
    def make_manager(self, root):
        config = make_config(root)
        _, paths = make_project(root, "pixel_tech")
        return AssetManager(config.pipeline, paths, "pixel_tech"), paths

    def test_identical_prompts_are_deduplicated(self) -> None:
        with TempWorkspace() as root:
            manager, _ = self.make_manager(root)
            plan = {
                "project": {"aspect_ratio": "16:9"},
                "scenes": [
                    {
                        "scene_id": "L001",
                        "asset_type": "ai_image",
                        "image_prompt": "A blue door",
                        "estimated_duration_seconds": 3,
                    },
                    {
                        "scene_id": "L002",
                        "asset_type": "ai_image",
                        "image_prompt": "  a   BLUE door ",
                        "estimated_duration_seconds": 5,
                    },
                    {
                        "scene_id": "L003",
                        "asset_type": "ai_image",
                        "image_prompt": "A red door",
                        "estimated_duration_seconds": 4,
                    },
                ],
            }
            planned = manager.plan_assets(plan)
            self.assertEqual(len(planned), 2)
            blue = next(asset for asset in planned if "blue" in asset.prompt.lower())
            self.assertEqual(blue.scene_ids, ["L001", "L002"])
            self.assertEqual(blue.duration_seconds, 3.0)

    def test_text_overlay_needs_no_external_asset(self) -> None:
        with TempWorkspace() as root:
            manager, _ = self.make_manager(root)
            plan = {
                "project": {"aspect_ratio": "16:9"},
                "scenes": [
                    {
                        "scene_id": "L001",
                        "asset_type": "text_overlay",
                        "image_prompt": "",
                    }
                ],
            }
            self.assertEqual(manager.plan_assets(plan), [])

    def test_empty_prompt_is_skipped(self) -> None:
        with TempWorkspace() as root:
            manager, _ = self.make_manager(root)
            plan = {
                "project": {"aspect_ratio": "16:9"},
                "scenes": [{"scene_id": "L001", "asset_type": "ai_image", "image_prompt": "   "}],
            }
            self.assertEqual(manager.plan_assets(plan), [])

    def test_unconfigured_providers_produce_pending_not_fake_files(self) -> None:
        with TempWorkspace() as root:
            manager, paths = self.make_manager(root)
            plan = build_long_plan()
            summary = manager.ensure_assets(plan, dry_run=True)

            self.assertFalse(summary["any_provider_configured"])
            self.assertEqual(summary["counts"]["ready"], 0)
            self.assertEqual(summary["counts"]["pending"], len(plan["scenes"]))
            self.assertTrue(all(asset["status"] == "pending" for asset in summary["assets"]))
            self.assertTrue(all(asset["path"] is None for asset in summary["assets"]))
            self.assertTrue(
                all("configured" in (asset.get("reason") or "").lower() for asset in summary["assets"])
            )

            for scene in plan["scenes"]:
                self.assertEqual(scene["asset"]["status"], "pending")
                self.assertIsNone(scene["asset"]["path"])

            # No generated files may be invented.
            generated = [
                path
                for path in paths.assets.rglob("*")
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".mp4", ".wav", ".mp3"}
            ]
            self.assertEqual(generated, [])

    def test_status_file_and_plan_are_written(self) -> None:
        with TempWorkspace() as root:
            manager, paths = self.make_manager(root)
            plan = build_long_plan()
            manager.ensure_assets(plan, dry_run=True)

            status = read_json(paths.assets_status)
            self.assertIsNotNone(status)
            self.assertEqual(status["profile_id"], "pixel_tech")
            self.assertIn("providers", status)
            self.assertIn("counts", status)

            plan_file = paths.assets / "plan_long.json"
            self.assertTrue(plan_file.is_file())
            stored = read_json(plan_file)
            self.assertEqual(len(stored["assets"]), len(manager.plan_assets(plan)))

    def test_second_run_reuses_status(self) -> None:
        with TempWorkspace() as root:
            manager, paths = self.make_manager(root)
            plan = build_long_plan()
            first = manager.ensure_assets(plan, dry_run=True)
            second = manager.ensure_assets(build_long_plan(), dry_run=True)
            self.assertEqual(
                [asset["asset_id"] for asset in first["assets"]],
                [asset["asset_id"] for asset in second["assets"]],
            )

    def test_resolve_path_rejects_escape(self) -> None:
        with TempWorkspace() as root:
            manager, _ = self.make_manager(root)
            self.assertIsNone(manager.resolve_path("../outside.png"))
            self.assertIsNone(manager.resolve_path(""))
            self.assertIsNone(manager.resolve_path(None))

    def test_seeded_project_assets_are_pending(self) -> None:
        with TempWorkspace() as root:
            _, paths = make_project(root, "pixel_tech")
            seed_minimal_project(paths)
            plan = read_json(paths.long_plan)
            statuses = {scene["asset"]["status"] for scene in plan["scenes"]}
            self.assertEqual(statuses, {"pending"})


if __name__ == "__main__":
    unittest.main()
