"""Asset Manager.

Reads production plans, decides which assets are required, and asks the
configured providers for them.  Everything is provider-independent.

Behaviour when nothing is configured (the current default):
    every required asset is recorded with ``status: "pending"``, its scene keeps
    an ``asset`` block explaining why, and the run continues.  No file is
    invented, no API call is faked.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import PipelineSettings
from engine.file_manager import ProjectPaths, atomic_write_json, read_json, utc_now_iso
from engine.logging_setup import get_logger
from providers.asset_base import AssetProvider
from providers.base import AssetRequest, ProviderState
from providers.image import get_image_provider
from providers.video import get_video_provider
from providers.audio import get_audio_provider

log = get_logger("assets")

#: Which provider kind handles each scene asset type.
ASSET_PROVIDER_FOR_TYPE = {
    "ai_image": "image",
    "stock_video": "video",
    "ai_video": "video",
    "motion_graphics": "image",
    "diagram": "image",
    "text_overlay": "none",  # rendered by FFmpeg, no external asset needed
}

EXTENSIONS = {"image": ".png", "video": ".mp4", "sfx": ".wav", "music": ".mp3"}


@dataclass
class PlannedAsset:
    """One asset requirement derived from a production plan."""

    asset_id: str
    kind: str
    prompt: str
    negative_prompt: str
    aspect_ratio: str
    duration_seconds: float | None
    scene_ids: list[str]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio,
            "duration_seconds": self.duration_seconds,
            "scene_ids": self.scene_ids,
            "fingerprint": self.fingerprint,
        }


class AssetManager:
    """Turns a production plan into real files, or into honest pending entries."""

    def __init__(
        self,
        settings: PipelineSettings,
        paths: ProjectPaths,
        profile_id: str,
    ) -> None:
        self.settings = settings
        self.paths = paths
        self.profile_id = profile_id
        self.providers: dict[str, AssetProvider] = {
            "image": get_image_provider(settings.image_provider),
            "video": get_video_provider(settings.video_provider),
            "sfx": get_audio_provider("sfx", settings.sfx_provider),
            "music": get_audio_provider("music", settings.music_provider),
        }

    # -- provider status ---------------------------------------------------- #
    def provider_report(self) -> dict[str, Any]:
        return {kind: provider.describe() for kind, provider in self.providers.items()}

    def any_provider_configured(self) -> bool:
        return any(provider.is_configured() for provider in self.providers.values())

    # -- planning ----------------------------------------------------------- #
    def _fingerprint(
        self,
        kind: str,
        prompt: str,
        aspect: str,
        *,
        negative_prompt: str = "",
        style: str = "",
    ) -> str:
        """Fingerprint the *creative* input, not the scene that requested it.

        Scene ids are deliberately excluded: two scenes with the same prompt are
        an intentional dedupe.  The profile, visual style and negative prompt are
        included so changing any of them produces a new asset instead of silently
        reusing an old file.
        """
        normalised = " ".join(prompt.lower().split())
        negative = " ".join(negative_prompt.lower().split())
        payload = (
            f"{self.profile_id}|{kind}|{aspect}|{style}|{normalised}|{negative}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def plan_assets(self, production_plan: dict[str, Any]) -> list[PlannedAsset]:
        """Collect required assets, de-duplicating identical creative prompts."""
        by_fingerprint: dict[str, PlannedAsset] = {}
        project = production_plan.get("project") or {}
        aspect = str(
            project.get("aspect_ratio")
            or self.settings.raw.get("video", {}).get("long_width", "")
        )
        style = str(
            project.get("visual_style")
            or self.settings.raw.get("visuals", {}).get("style", "")
            or ""
        )

        for scene in production_plan.get("scenes", []) or []:
            asset_type = str(scene.get("asset_type", "ai_image"))
            provider_kind = ASSET_PROVIDER_FOR_TYPE.get(asset_type, "image")
            if provider_kind == "none":
                continue

            prompt = str(scene.get("image_prompt") or scene.get("video_prompt") or "").strip()
            if not prompt:
                continue

            negative = str(scene.get("negative_prompt", ""))
            fingerprint = self._fingerprint(
                provider_kind,
                prompt,
                aspect,
                negative_prompt=negative,
                style=style,
            )
            scene_id = str(scene.get("scene_id", "?"))
            if fingerprint in by_fingerprint:
                if scene_id not in by_fingerprint[fingerprint].scene_ids:
                    by_fingerprint[fingerprint].scene_ids.append(scene_id)
                continue

            by_fingerprint[fingerprint] = PlannedAsset(
                asset_id=f"{self.paths.slug}__{scene_id}__{fingerprint[:8]}",
                kind=provider_kind,
                prompt=prompt,
                negative_prompt=negative,
                aspect_ratio=aspect,
                duration_seconds=float(scene.get("estimated_duration_seconds", 0)) or None,
                scene_ids=[scene_id],
                fingerprint=fingerprint,
            )

        return list(by_fingerprint.values())

    # -- generation --------------------------------------------------------- #
    def ensure_assets(
        self,
        production_plan: dict[str, Any],
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Resolve assets for one production plan and annotate its scenes."""
        planned = self.plan_assets(production_plan)
        log.info("%d distinct asset(s) required for this plan", len(planned))

        results: dict[str, dict[str, Any]] = {}
        existing_status = read_json(self.paths.assets_status, default={}) or {}
        previous_plans: dict[str, dict[str, Any]] = {
            str(name): dict(payload)
            for name, payload in (existing_status.get("plans", {}) or {}).items()
            if isinstance(payload, dict)
        }
        cached: dict[str, dict[str, Any]] = {
            str(entry.get("asset_id")): entry
            for payload in previous_plans.values()
            for entry in payload.get("assets", []) or []
            if isinstance(entry, dict)
        }
        # Backwards compatibility with status files written before per-plan
        # tracking existed.
        for entry in existing_status.get("assets", []) or []:
            if isinstance(entry, dict) and entry.get("asset_id"):
                cached.setdefault(str(entry["asset_id"]), entry)

        for asset in planned:
            kind = asset.kind
            provider = self.providers[kind]
            target_dir = self._directory_for(kind)
            extension = EXTENSIONS.get(kind, provider.extension)
            destination = target_dir / f"{asset.asset_id}{extension}"

            # Reuse an asset that already exists on disk.
            if destination.is_file():
                results[asset.asset_id] = {
                    **asset.to_dict(),
                    "status": "ready",
                    "path": str(destination.relative_to(self.paths.root)),
                    "provider": cached.get(asset.asset_id, {}).get("provider", "cached"),
                    "error": None,
                }
                continue

            # Reuse a previous successful download path recorded in status.
            previous = cached.get(asset.asset_id)
            if previous and previous.get("status") == "ready" and previous.get("path"):
                previous_path = self.paths.root / str(previous["path"])
                if previous_path.is_file():
                    results[asset.asset_id] = dict(previous)
                    continue

            if not provider.is_configured():
                results[asset.asset_id] = {
                    **asset.to_dict(),
                    "status": "pending",
                    "path": None,
                    "provider": str(provider.name),
                    "error": None,
                    "reason": provider.configuration_hint(),
                }
                continue

            if dry_run:
                results[asset.asset_id] = {
                    **asset.to_dict(),
                    "status": "pending",
                    "path": None,
                    "provider": str(provider.name),
                    "error": None,
                    "reason": "dry run: provider configured but generation was not requested",
                }
                continue

            log.info("Requesting %s asset %s", kind, asset.asset_id)
            request = AssetRequest(
                asset_id=asset.asset_id,
                kind=kind,
                prompt=asset.prompt,
                negative_prompt=asset.negative_prompt,
                aspect_ratio=asset.aspect_ratio,
                duration_seconds=asset.duration_seconds,
            )
            result = provider.generate(request, destination)
            if result.status == "ready" and result.path and result.path.is_file():
                results[asset.asset_id] = {
                    **asset.to_dict(),
                    "status": "ready",
                    "path": str(result.path.relative_to(self.paths.root)),
                    "provider": str(provider.name),
                    "error": None,
                    "metadata": result.metadata,
                }
            else:
                results[asset.asset_id] = {
                    **asset.to_dict(),
                    "status": result.status,
                    "path": None,
                    "provider": str(provider.name),
                    "error": result.error or result.metadata.get("reason"),
                }

        # Annotate scenes so the production plan itself carries asset state.
        scene_lookup: dict[str, dict[str, Any]] = {}
        for asset_id, entry in results.items():
            for scene_id in entry.get("scene_ids", []):
                scene_lookup[scene_id] = entry

        no_asset_needed = 0
        for scene in production_plan.get("scenes", []) or []:
            asset_type = str(scene.get("asset_type", "ai_image"))
            scene_id = str(scene.get("scene_id", "?"))
            entry = scene_lookup.get(scene_id)
            if entry is None:
                if ASSET_PROVIDER_FOR_TYPE.get(asset_type) == "none":
                    no_asset_needed += 1
                    scene["asset"] = {
                        "asset_id": f"{self.paths.slug}__{scene_id}__overlay",
                        "type": asset_type,
                        "status": "not_required",
                        "path": None,
                        "provider": None,
                        "on_screen_text": str(
                            scene.get("on_screen_text") or scene.get("text") or ""
                        ),
                        "notes": f"{asset_type} is rendered directly by FFmpeg",
                    }
                continue
            scene["asset"] = {
                "asset_id": entry["asset_id"],
                "type": asset_type,
                "status": entry["status"],
                "path": entry.get("path"),
                "provider": entry.get("provider"),
                "on_screen_text": str(
                    scene.get("on_screen_text") or scene.get("text") or ""
                ),
                "notes": entry.get("reason") or entry.get("error") or "",
            }

        plan_name = (
            "long"
            if (production_plan.get("project") or {}).get("format") == "long"
            else "shorts"
        )
        previous_plans[plan_name] = {
            "generated_at": utc_now_iso(),
            "assets": list(results.values()),
            "not_required": no_asset_needed,
        }
        merged_assets: dict[str, dict[str, Any]] = {}
        for payload in previous_plans.values():
            for entry in payload.get("assets", []) or []:
                if isinstance(entry, dict) and entry.get("asset_id"):
                    merged_assets[str(entry["asset_id"])] = entry
        counts = self._counts(merged_assets)
        not_required_total = sum(
            int(payload.get("not_required", 0) or 0)
            for payload in previous_plans.values()
        )

        summary = {
            "generated_at": utc_now_iso(),
            "profile_id": self.profile_id,
            "topic": self.paths.topic,
            "providers": self.provider_report(),
            "any_provider_configured": self.any_provider_configured(),
            "assets": list(merged_assets.values()),
            "plans": previous_plans,
            "counts": counts,
            "not_required": not_required_total,
        }
        atomic_write_json(self.paths.assets_status, summary)
        atomic_write_json(self.paths.root / "assets" / f"plan_{plan_name}.json", {
            "generated_at": utc_now_iso(),
            "plan": plan_name,
            "assets": [asset.to_dict() for asset in planned],
        })

        log.info(
            "Asset status: %d ready, %d pending, %d failed, %d not required",
            counts["ready"],
            counts["pending"],
            counts["failed"],
            not_required_total,
        )
        return summary

    def _directory_for(self, kind: str) -> Path:
        directory = {
            "image": self.paths.images,
            "video": self.paths.videos,
            "sfx": self.paths.sounds,
            "music": self.paths.music,
        }.get(kind, self.paths.assets)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _counts(results: dict[str, dict[str, Any]]) -> dict[str, int]:
        counts = {"ready": 0, "pending": 0, "failed": 0, "not_required": 0}
        for entry in results.values():
            status = str(entry.get("status", "pending"))
            counts[status] = counts.get(status, 0) + 1
        return counts

    # -- resolution --------------------------------------------------------- #
    def resolve_path(self, relative: str | None) -> Path | None:
        """Turn a stored relative asset path into an absolute, validated path."""
        if not relative:
            return None
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = self.paths.root / candidate
        return candidate if candidate.is_file() else None


__all__ = ["AssetManager", "PlannedAsset", "ASSET_PROVIDER_FOR_TYPE"]
