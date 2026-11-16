"""Tests for the FFmpeg argument builders and a real render smoke test."""

from __future__ import annotations

import unittest
from pathlib import Path

from config.settings import load_config
from engine import ffmpeg_builder as fb

CONFIG = load_config()


def settings() -> fb.RenderSettings:
    return fb.RenderSettings(
        ffmpeg=Path("ffmpeg.exe"),
        ffprobe=Path("ffprobe.exe"),
        width=1920,
        height=1080,
        fps=30,
    )


class TestMotionKind(unittest.TestCase):
    def test_keyword_mapping(self) -> None:
        cases = {
            "slow push in": "push_in",
            "zoom in on the face": "push_in",
            "dolly out": "pull_out",
            "pan left across the room": "pan_left",
            "pan right": "pan_right",
            "static": "static",
            "": "static",
            "a slow drift": "push_in",
        }
        for text, expected in cases.items():
            scene = {"motion": text, "camera_movement": ""}
            self.assertEqual(fb.motion_kind(scene), expected, msg=text)

    def test_camera_movement_is_considered(self) -> None:
        scene = {"motion": "none", "camera_movement": "gentle push in"}
        self.assertEqual(fb.motion_kind(scene), "push_in")


class TestRenderSettings(unittest.TestCase):
    def test_dimension_filter_never_letterboxes(self) -> None:
        text = settings().dimension_filter()
        self.assertIn("force_original_aspect_ratio=increase", text)
        self.assertIn("crop=1920:1080", text)
        self.assertIn("setsar=1", text)
        self.assertNotIn("pad=", text)


class TestSceneClipArgs(unittest.TestCase):
    def test_static_image(self) -> None:
        args = fb.scene_clip_args(
            asset=Path("frame.png"),
            scene={"motion": "static", "estimated_duration_seconds": 4.5},
            settings=settings(),
            out_path=Path("clip.mp4"),
        )
        self.assertEqual(args[0], "ffmpeg.exe")
        self.assertIn("-loop", args)
        self.assertIn("4.500", args)
        self.assertIn("libx264", args)
        self.assertIn("yuv420p", args)
        self.assertIn("fps=30", " ".join(args))
        # A still image must not be asked for a zeroth frame.
        self.assertNotIn("zoompan", " ".join(args))

    def test_motion_image_uses_zoompan(self) -> None:
        args = fb.scene_clip_args(
            asset=Path("frame.png"),
            scene={"motion": "slow push in", "estimated_duration_seconds": 3.0},
            settings=settings(),
            out_path=Path("clip.mp4"),
        )
        text = " ".join(args)
        self.assertIn("zoompan", text)
        self.assertIn("d=90", text)  # 3s * 30fps
        self.assertNotIn("-loop 1", text)

    def test_pan_direction_changes_expression(self) -> None:
        left = " ".join(
            fb.scene_clip_args(
                asset=Path("f.png"),
                scene={"motion": "pan left", "estimated_duration_seconds": 2},
                settings=settings(),
                out_path=Path("c.mp4"),
            )
        )
        right = " ".join(
            fb.scene_clip_args(
                asset=Path("f.png"),
                scene={"motion": "pan right", "estimated_duration_seconds": 2},
                settings=settings(),
                out_path=Path("c.mp4"),
            )
        )
        self.assertNotEqual(left, right)
        self.assertIn("max(0,iw/2", left)
        self.assertIn("min(iw-iw/zoom", right)

    def test_video_asset_is_trimmed_not_looped(self) -> None:
        args = fb.scene_clip_args(
            asset=Path("shot.mp4"),
            scene={"motion": "push in", "estimated_duration_seconds": 2.0},
            settings=settings(),
            out_path=Path("clip.mp4"),
        )
        self.assertIn("-an", args)
        self.assertNotIn("-loop", args)
        self.assertIn("fps=30", " ".join(args))

    def test_duration_is_clamped(self) -> None:
        scene = {"motion": "static", "estimated_duration_seconds": 900}
        args = fb.scene_clip_args(
            asset=Path("f.png"), scene=scene, settings=settings(), out_path=Path("c.mp4")
        )
        self.assertIn("120.000", args)


class TestConcatList(unittest.TestCase):
    def test_paths_are_forward_slashed_and_escaped(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            list_path = Path(tmp) / "list.txt"
            fb.write_concat_list(
                [Path(r"C:\clips\one.mp4"), Path(r"C:\clips\it's two.mp4")],
                list_path,
            )
            text = list_path.read_text(encoding="utf-8")
            self.assertIn("file 'C:/clips/one.mp4'", text)
            self.assertIn("it'\\''s two.mp4", text)
            self.assertTrue(text.endswith("\n"))

    def test_concat_args_use_the_demuxer(self) -> None:
        args = fb.concat_args(
            list_path=Path("list.txt"), out_path=Path("out.mp4"), settings=settings()
        )
        joined = " ".join(args)
        self.assertIn("-f concat", joined)
        self.assertIn("-safe 0", joined)
        self.assertIn("-c copy", joined)


class TestMuxArgs(unittest.TestCase):
    def test_narration_only_has_no_music_branch(self) -> None:
        args = fb.mux_args(
            video_path=Path("video.mp4"),
            narration_path=Path("voice.wav"),
            out_path=Path("final.mp4"),
            settings=settings(),
        )
        text = " ".join(args)
        self.assertIn("[narr]alimiter", text)
        self.assertNotIn("amix", text)
        self.assertNotIn("-stream_loop", text)
        self.assertIn("-map [aout]", text)

    def test_music_is_looped_attenuated_and_mixed(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            music = Path(tmp) / "music.mp3"
            music.write_bytes(b"")
            args = fb.mux_args(
                video_path=Path("video.mp4"),
                narration_path=Path("voice.wav"),
                out_path=Path("final.mp4"),
                settings=settings(),
                music_path=music,
                total_duration=60.0,
            )
            text = " ".join(args)
            self.assertIn("-stream_loop -1", text)
            self.assertIn("volume=-22.0dB", text)
            self.assertIn("amix=inputs=2", text)
            self.assertIn("afade=t=out:st=57.00", text)
            self.assertIn("normalize=0", text)

    def test_nonexistent_music_file_is_ignored(self) -> None:
        args = fb.mux_args(
            video_path=Path("video.mp4"),
            narration_path=Path("voice.wav"),
            out_path=Path("final.mp4"),
            settings=settings(),
            music_path=Path("does-not-exist.mp3"),
        )
        self.assertNotIn("-stream_loop", " ".join(args))

    def test_limiter_uses_true_peak(self) -> None:
        args = fb.mux_args(
            video_path=Path("v.mp4"),
            narration_path=Path("n.wav"),
            out_path=Path("o.mp4"),
            settings=settings(),
            true_peak_db=-1.5,
        )
        text = " ".join(args)
        self.assertIn("alimiter=limit=0.8414", text)


class TestValidateInputs(unittest.TestCase):
    def test_all_scenes_ready(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            voice = root / "voice.wav"
            voice.write_bytes(b"")
            image = root / "image.png"
            image.write_bytes(b"")
            plan = {
                "scenes": [
                    {
                        "scene_id": "L001",
                        "asset_type": "ai_image",
                        "asset": {"status": "ready", "path": "image.png"},
                    }
                ]
            }
            ready, missing = fb.validate_render_inputs(
                plan, resolve_asset=lambda p: root / p if p else None, narration_path=voice
            )
            self.assertEqual(missing, [])
            self.assertEqual(len(ready), 1)

    def test_missing_narration_and_pending_assets_are_reported(self) -> None:
        plan = {
            "scenes": [
                {
                    "scene_id": "L001",
                    "asset_type": "ai_image",
                    "asset": {"status": "pending", "path": None},
                },
                {
                    "scene_id": "L002",
                    "asset_type": "text_overlay",
                    "asset": {"status": "not_required", "path": None},
                },
            ]
        }
        ready, missing = fb.validate_render_inputs(
            plan, resolve_asset=lambda p: None, narration_path=Path("nope.wav")
        )
        self.assertEqual(ready, [])
        self.assertTrue(any("narration audio" in item for item in missing))
        self.assertTrue(any("L001" in item and "pending" in item for item in missing))
        self.assertTrue(any("L002" in item for item in missing))


class TestRealRender(unittest.TestCase):
    """Actually invoke FFmpeg end-to-end on synthetic media."""

    @classmethod
    def setUpClass(cls) -> None:
        ffmpeg = CONFIG.pipeline.ffmpeg_path
        ffprobe = CONFIG.pipeline.ffprobe_path
        if not (ffmpeg and ffprobe):
            raise unittest.SkipTest("FFmpeg is not configured")
        if not ffmpeg.is_file():
            raise unittest.SkipTest(f"ffmpeg not found at {ffmpeg}")

    def test_smoke_render(self) -> None:
        import tempfile

        from engine.file_manager import ProjectPaths
        from engine.video_assembler import VideoAssembler

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ProjectPaths(
                profile_id="test", topic="Smoke", slug="smoke", root=root
            )
            assembler = VideoAssembler(CONFIG.pipeline, paths)
            report = assembler.smoke_test()
            self.assertTrue(report.get("ok"), msg=report)
            for key in ("output", "duration_seconds"):
                self.assertIn(key, report)
            output = Path(report["output"])
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
