"""Shared test helpers and fixtures."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from config.settings import AppConfig, PipelineSettings, load_config
from engine.file_manager import FileManager, ProjectPaths, atomic_write_json, atomic_write_text

FIXTURE_LONG = (
    "In 1993, the fastest computer in the world could not draw a single frame of "
    "the film you are watching. Today, the phone in your pocket can do it while "
    "you wait for a bus.\n\n"
    "That is not a story about faster chips. It is a story about a decision. "
    "Engineers stopped asking how to calculate more, and started asking which "
    "calculations could be skipped entirely.\n\n"
    "The result was a machine that lied to your eyes convincingly enough to "
    "become the standard. Every game you play, every film you stream, and every "
    "map you trust now depends on that same compromise.\n\n"
    "This is how the compromise was made, why it survived, and what it quietly "
    "cost us along the way."
)

SHORT_TITLES = [
    "The Frame That Took a Day",
    "Why Cheating Became the Standard",
    "The Compromise Nobody Noticed",
    "What the Machine Cannot See",
    "The Cost of Convincing Pixels",
    "Why the Lie Still Works",
]

SHORT_BODIES = [
    "In 1993 one frame of film took a day to render. Not a scene. One frame. The machine cost more than a house and still could not finish the shot before the coffee went cold.",
    "Every image you see on a screen is a negotiation. Render the truth and you wait forever. Render a convincing lie and you watch a film tonight. The industry chose the lie and never went back.",
    "The compromise was invisible because it worked. Nobody complained that the shadows were approximations, because the shadows looked like shadows. That is how a shortcut becomes a standard.",
    "A rendered image is a guess that survived review. The machine does not know what a face is. It knows which pixels usually appear beside which other pixels, and that turns out to be enough.",
    "The bill arrived quietly. Artists learned new tools, studios cut render farms, and the look of an entire medium was set by what a chip could afford in 1995.",
    "The lie still works because your eyes are cheaper to fool than to satisfy. Understanding that is the difference between being impressed by an image and being able to explain it.",
]


def fixture_shorts(count: int = 6) -> list[dict[str, Any]]:
    return [
        {
            "index": index + 1,
            "title": SHORT_TITLES[index % len(SHORT_TITLES)],
            "narration": SHORT_BODIES[index % len(SHORT_BODIES)],
        }
        for index in range(count)
    ]


def make_config(tmp_root: Path) -> AppConfig:
    """Build an ``AppConfig`` whose scripts/output/logs point inside *tmp_root*."""
    base = load_config()
    raw = copy.deepcopy(base.pipeline.raw)
    raw.setdefault("paths", {})
    raw["paths"]["scripts_dir"] = str(tmp_root / "scripts")
    raw["paths"]["output_dir"] = str(tmp_root / "output")
    raw["paths"]["logs_dir"] = str(tmp_root / "logs")
    raw.setdefault("video", {})
    raw["video"]["render"] = True
    return AppConfig(model=base.model, pipeline=PipelineSettings(raw=raw))


def make_project(tmp_root: Path, profile_id: str = "pixel_tech") -> tuple[FileManager, ProjectPaths]:
    config = make_config(tmp_root)
    manager = FileManager(scripts_dir=config.pipeline.scripts_dir)
    paths = manager.create_project(profile_id, "Test Topic")
    return manager, paths


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def build_brief() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "topic": "Test Topic",
        "working_title": "Test Topic",
        "central_message": "A specific, testable claim about how a compromise became a standard.",
        "viewer_promise": "You will understand the mechanism, not just the outcome.",
        "target_audience": {
            "description": "Curious adults who like explanations",
            "prior_knowledge": ["basic computing"],
            "pain_points": ["explainers that skip the mechanism"],
            "desired_outcome": "be able to explain it to someone else",
        },
        "angles": [
            {
                "title": "The mechanism",
                "why_interesting": "It explains why the outcome was inevitable",
                "selected": True,
            },
            {
                "title": "The cost",
                "why_interesting": "It reframes a triumph as a trade-off",
                "selected": False,
            },
        ],
        "key_questions": [
            "How does the mechanism actually work?",
            "Why did this happen when it did?",
            "What did it cost?",
        ],
        "misconceptions": [
            {"misconception": "It is magic", "correction": "It is a deliberate engineering trade-off"}
        ],
        "storytelling": {
            "approach": "Start with a consequence, then unwind the decision behind it",
            "structure_rationale": "Chronology would bury the interesting part",
            "emotional_arc": ["curiosity", "surprise", "resolve"],
            "viewer_experience": "A growing sense of recognition",
        },
        "key_terms": [
            {"term": "rendering", "plain_explanation": "turning a description into an image"}
        ],
        "scope": {
            "include": ["the mechanism", "the trade-off"],
            "exclude": ["biographies", "company politics"],
        },
    }


def build_research() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "topic": "Test Topic",
        "collected_at": "2026-01-01T00:00:00+00:00",
        "research_status": "unavailable",
        "provider": {"name": "none", "configured": False, "notes": "", "error": None},
        "queries": ["test topic"],
        "sources": [],
        "verified_facts": [
            {
                "fact": "Background that is uncontroversial.",
                "source_ids": [],
                "confidence": "low",
                "notes": "No external source collected.",
            }
        ],
        "uncertain_claims": [
            {
                "claim": "A specific figure",
                "why_uncertain": "No source was collected",
                "verification_needed": "Find a primary source",
                "source_ids": [],
            }
        ],
        "statistics": [],
        "conflicts": [],
        "missing_evidence": ["Primary sources"],
        "outdated_risks": [],
        "internal_knowledge_notes": [],
        "guidance_for_script": {
            "safe_to_state": ["The mechanism"],
            "state_with_hedging": ["Rough timelines"],
            "avoid": ["Specific numbers"],
        },
    }


def build_outline() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "working_title": "Test Topic",
        "target_duration_minutes": 10,
        "target_word_count": 1500,
        "hook": {
            "text": "Open on a consequence that should be impossible.",
            "purpose": "create curiosity",
        },
        "introduction": {
            "goal": "Establish the question",
            "key_points": ["why it matters", "what we will answer"],
        },
        "sections": [
            {
                "id": f"S{index}",
                "title": f"Section {index}",
                "goal": "Advance the argument",
                "key_points": ["one concrete point", "one consequence"],
                "estimated_words": 250,
                "emotional_note": "curiosity",
                "transition_to_next": "which raises the next question",
            }
            for index in range(1, 5)
        ],
        "conclusion": {"text": "Resolve the opening question with the mechanism."},
        "emotional_progression": ["curiosity", "surprise", "resolve"],
    }


def build_audio_plan() -> dict[str, Any]:
    return {
        "music": {
            "style": "Ambient electronic",
            "overall_energy_curve": "calm to tense",
            "segments": [
                {
                    "segment_id": "M01",
                    "start_scene": "L001",
                    "end_scene": "L002",
                    "intensity": 0.3,
                    "mood": "calm",
                    "description": "Low pulse",
                    "transition": "crossfade",
                    "fade_in_seconds": 2.0,
                    "fade_out_seconds": 2.0,
                },
                {
                    "segment_id": "M02",
                    "start_scene": "L003",
                    "end_scene": "L003",
                    "intensity": 0.8,
                    "mood": "tense",
                    "description": "Rising pad",
                    "transition": "cut",
                    "fade_in_seconds": 0.5,
                    "fade_out_seconds": 1.0,
                },
            ],
        },
        "sfx": [],
        "pauses": [{"after_scene": "L002", "seconds": 0.8, "reason": "beat"}],
        "mix": {
            "narration_priority": "Narration forward",
            "music_bed_db": -22.0,
            "duck_under_narration_db": -9.0,
            "true_peak_db": -1.5,
        },
        "status": "planned",
    }


def build_long_plan(narration: str = FIXTURE_LONG) -> dict[str, Any]:
    from engine.narration import count_words, split_into_blocks

    blocks = split_into_blocks(narration, target_sentences=2, target_words=30)
    scenes = []
    for index, block in enumerate(blocks, start=1):
        scenes.append(
            {
                "scene_id": f"L{index:03d}",
                "block_id": block.block_id,
                "origin": "narration",
                "narration": block.text,
                "narration_excerpt": block.text[:80],
                "narration_start_char": block.start_char,
                "narration_end_char": block.end_char,
                "purpose": "Advance the argument",
                "visual_description": f"Wide establishing shot {index} of a data centre at night",
                "image_prompt": (
                    f"Cinematic wide photograph number {index} of a night-time data centre, "
                    "blue rim light, condensation on glass, 35mm lens"
                ),
                "motion": "slow push in",
                "camera_movement": "static",
                "transition_out": "cut",
                "estimated_duration_seconds": round(count_words(block.text) / 150 * 60, 2),
                "asset_type": "ai_image",
                "visual_style": "Cinematic futuristic",
                "asset": {
                    "asset_id": f"test__L{index:03d}",
                    "status": "pending",
                    "path": None,
                    "provider": "none",
                    "notes": "No provider configured",
                },
            }
        )

    total = round(sum(scene["estimated_duration_seconds"] for scene in scenes), 2)
    return {
        "schema_version": "1.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "project": {
            "profile_id": "pixel_tech",
            "topic": "Test Topic",
            "slug": "test-topic",
            "format": "long",
            "aspect_ratio": "16:9",
            "target_duration_seconds": 600,
            "narration_word_count": count_words(narration),
        },
        "style_summary": "Cinematic futuristic",
        "scenes": scenes,
        "audio_plan": build_audio_plan(),
        "coverage": {
            "narration_chars": len(narration),
            "covered_chars": len(narration),
            "coverage_ratio": 1.0,
            "narration_word_count": count_words(narration),
            "scene_count": len(scenes),
            "distinct_visual_concepts": len(scenes),
            "estimated_total_duration_seconds": total,
        },
        "notes": [],
    }


def build_shorts_plan(count: int = 6) -> dict[str, Any]:
    from engine.narration import count_words, split_into_blocks

    shorts = []
    scene_counter = 1
    for index, short in enumerate(fixture_shorts(count), start=1):
        blocks = split_into_blocks(str(short["narration"]), target_sentences=2, target_words=20)
        scenes = []
        for block in blocks:
            scenes.append(
                {
                    "scene_id": f"S{scene_counter:03d}",
                    "block_id": block.block_id,
                    "origin": "narration",
                    "narration": block.text,
                    "narration_excerpt": block.text[:80],
                    "narration_start_char": block.start_char,
                    "narration_end_char": block.end_char,
                    "purpose": "Carry the Short",
                    "visual_description": "Close-up of a hand on a keyboard",
                    "image_prompt": "Cinematic close-up of a hand on a keyboard, teal light",
                    "motion": "push in",
                    "camera_movement": "static",
                    "transition_out": "cut",
                    "estimated_duration_seconds": round(
                        count_words(block.text) / 150 * 60, 2
                    ),
                    "asset_type": "ai_image",
                    "visual_style": "Cinematic futuristic",
                }
            )
            scene_counter += 1
        shorts.append(
            {
                "short_id": f"{index:02d}",
                "title": short["title"],
                "target_duration_seconds": 45,
                "narration_word_count": count_words(str(short["narration"])),
                "central_idea": "One idea per Short",
                "scenes": scenes,
                "audio_plan": build_audio_plan(),
            }
        )

    return {
        "schema_version": "1.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "project": {
            "profile_id": "pixel_tech",
            "topic": "Test Topic",
            "slug": "test-topic",
            "aspect_ratio": "9:16",
            "shorts_count": len(shorts),
        },
        "style_summary": "Cinematic futuristic",
        "shorts": shorts,
        "notes": [],
    }


def build_quality_report(paths: ProjectPaths) -> dict[str, Any]:
    def category() -> dict[str, Any]:
        return {
            "status": "pass",
            "checks": [{"name": "fixture", "status": "pass", "detail": "Test fixture"}],
        }

    return {
        "schema_version": "1.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "project": {
            "profile_id": paths.profile_id,
            "topic": paths.topic,
            "slug": paths.slug,
        },
        "overall_status": "pass",
        "summary": "Fixture project",
        "categories": {
            "research": category(),
            "long_script": category(),
            "shorts": category(),
            "visuals": category(),
            "audio": category(),
            "files": category(),
        },
        "issues": [],
        "ai_review": None,
        "metrics": {},
    }


def build_state(paths: ProjectPaths, *, profile_id: str = "pixel_tech") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "profile_id": profile_id,
        "topic": paths.topic,
        "slug": paths.slug,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "status": "complete",
        "stages": {},
        "approval": {"script_approved": True, "approved_at": None, "mode": "test"},
        "artifacts": {},
        "stage_order": [],
    }


def seed_minimal_project(paths: ProjectPaths, *, shorts_count: int = 6) -> None:
    """Write a small but fully schema-valid set of project files."""
    from engine.narration import format_shorts, parse_shorts_text

    atomic_write_json(paths.brief, build_brief())
    atomic_write_json(paths.research, build_research())
    atomic_write_json(paths.outline, build_outline())
    atomic_write_text(paths.long_script, FIXTURE_LONG + "\n")

    shorts_text = format_shorts(fixture_shorts(shorts_count))
    atomic_write_text(paths.short_script, shorts_text)
    parsed = parse_shorts_text(shorts_text)
    assert len(parsed) == shorts_count, f"fixture produced {len(parsed)} shorts"

    atomic_write_json(paths.long_plan, build_long_plan())
    atomic_write_json(paths.shorts_plan, build_shorts_plan(shorts_count))
    atomic_write_json(paths.audio_plan, build_audio_plan())
    atomic_write_json(paths.quality, build_quality_report(paths))
    atomic_write_json(paths.sections, {"schema_version": "1.0", "sections": []})
    atomic_write_json(paths.state, build_state(paths))
    atomic_write_json(
        paths.manifest,
        {
            "schema_version": "1.0",
            "profile_id": paths.profile_id,
            "topic": paths.topic,
            "slug": paths.slug,
        },
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class TempWorkspace:
    """Context manager that yields a temporary app root."""

    def __init__(self) -> None:
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self.root: Path = Path()

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory(prefix="pixel-ai-test-")
        self.root = Path(self._tmp.name)
        return self.root

    def __exit__(self, *_exc: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
