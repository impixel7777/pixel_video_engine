"""PIXEL AI production engine.

One engine, many channel identities.  Each module owns exactly one
responsibility; the pipeline wires them together.
"""

from __future__ import annotations

__all__ = [
    "ai_stage",
    "asset_manager",
    "audio_director",
    "channel_profile",
    "content_planner",
    "deepseek_client",
    "file_manager",
    "json_utils",
    "logging_setup",
    "narration",
    "pipeline",
    "prompt_library",
    "quality_controller",
    "research_engine",
    "script_writer",
    "shorts_writer",
    "topic_analyzer",
    "video_assembler",
    "visual_director",
]
