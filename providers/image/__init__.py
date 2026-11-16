"""Image provider factory.

No image backend is enabled by default.  A real backend plugs in here by
subclassing :class:`providers.asset_base.AssetProvider` with ``kind = "image"``.

Example of adding one (no engine changes needed)::

    # providers/image/stable_diffusion_local.py
    class StableDiffusionLocalProvider(AssetProvider):
        name = "sd_local"
        kind = "image"
        extension = ".png"
        ...

    # providers/image/__init__.py
    REGISTRY["sd_local"] = StableDiffusionLocalProvider
"""

from __future__ import annotations

from providers.asset_base import AssetProvider, NullAssetProvider

REGISTRY: dict[str, type[AssetProvider]] = {}


def get_image_provider(name: str | None = None) -> AssetProvider:
    """Return the configured image provider, or an honest placeholder."""
    requested = (name or "none").strip().lower()
    if requested in ("", "none", "null"):
        return NullAssetProvider("image")
    provider_cls = REGISTRY.get(requested)
    if provider_cls is None:
        return NullAssetProvider("image")
    provider = provider_cls()
    if not provider.is_configured():
        return NullAssetProvider("image")
    return provider


def available_providers() -> list[str]:
    return ["none", *sorted(REGISTRY)]


__all__ = ["REGISTRY", "available_providers", "get_image_provider"]
