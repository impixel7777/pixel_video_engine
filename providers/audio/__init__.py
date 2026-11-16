"""Audio asset provider factory (sound effects and music beds).

Narration is produced by Kokoro and is *not* handled here.  This module only
covers production assets: sound effects and background music.

Register a backend by subclassing
:class:`providers.asset_base.AssetProvider` with ``kind = "sfx"`` or
``kind = "music"`` and adding it to :data:`REGISTRY`.
"""

from __future__ import annotations

from providers.asset_base import AssetProvider, NullAssetProvider

REGISTRY: dict[str, type[AssetProvider]] = {}


def get_audio_provider(kind: str, name: str | None = None) -> AssetProvider:
    """Return a provider for ``kind`` in ``{"sfx", "music"}``."""
    if kind not in ("sfx", "music"):
        raise ValueError(f"Unsupported audio asset kind: {kind}")
    requested = (name or "none").strip().lower()
    if requested in ("", "none", "null"):
        return NullAssetProvider(kind)
    provider_cls = REGISTRY.get(requested)
    if provider_cls is None:
        return NullAssetProvider(kind)
    provider = provider_cls()
    if not provider.is_configured():
        return NullAssetProvider(kind)
    return provider


def available_providers() -> list[str]:
    return ["none", *sorted(REGISTRY)]


__all__ = ["REGISTRY", "available_providers", "get_audio_provider"]
