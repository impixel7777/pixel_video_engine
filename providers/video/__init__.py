"""Video provider factory.

No video backend is enabled by default.  Add a backend by subclassing
:class:`providers.asset_base.AssetProvider` with ``kind = "video"`` and
registering it in :data:`REGISTRY`.
"""

from __future__ import annotations

from providers.asset_base import AssetProvider, NullAssetProvider

REGISTRY: dict[str, type[AssetProvider]] = {}


def get_video_provider(name: str | None = None) -> AssetProvider:
    requested = (name or "none").strip().lower()
    if requested in ("", "none", "null"):
        return NullAssetProvider("video")
    provider_cls = REGISTRY.get(requested)
    if provider_cls is None:
        return NullAssetProvider("video")
    provider = provider_cls()
    if not provider.is_configured():
        return NullAssetProvider("video")
    return provider


def available_providers() -> list[str]:
    return ["none", *sorted(REGISTRY)]


__all__ = ["REGISTRY", "available_providers", "get_video_provider"]
