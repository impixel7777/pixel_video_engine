"""Asset provider interface shared by image, video and audio backends.

The asset manager never talks to an API directly: it asks a provider.  A
provider that is not configured returns ``pending`` with a reason, so the
production plan is still complete and honest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from engine.logging_setup import get_logger
from providers.base import AssetRequest, AssetResult, ProviderState

log = get_logger("assets")


class AssetProvider(ABC):
    """Base class for everything that can produce a media file."""

    #: Stable identifier used in configs and status files.
    name: str = "base"
    #: One of ``image``, ``video``, ``sfx``, ``music``.
    kind: str = "asset"
    #: File extension the provider writes, e.g. ``.png``.
    extension: str = ".bin"

    @abstractmethod
    def is_configured(self) -> bool:
        """True when credentials and endpoints are available."""

    @abstractmethod
    def configuration_hint(self) -> str:
        """Tell the operator exactly what to set to enable this provider."""

    @abstractmethod
    def generate(self, request: AssetRequest, destination: Path) -> AssetResult:
        """Produce one asset at *destination*.

        Implementations must not write a file unless the asset really exists.
        """

    def generate_many(
        self, requests: Iterable[AssetRequest], output_dir: Path
    ) -> list[AssetResult]:
        results: list[AssetResult] = []
        for request in requests:
            destination = output_dir / f"{request.asset_id}{self.extension}"
            try:
                results.append(self.generate(request, destination))
            except Exception as exc:  # noqa: BLE001 - a provider failure is reportable data
                log.error("Provider %s failed for %s: %s", self.name, request.asset_id, exc)
                results.append(
                    AssetResult(
                        asset_id=request.asset_id,
                        state=ProviderState.FAILED,
                        provider=self.name,
                        error=str(exc),
                    )
                )
        return results

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "configured": self.is_configured(),
            "hint": self.configuration_hint(),
        }


class NullAssetProvider(AssetProvider):
    """Placeholder used when nothing is configured for a capability."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.name = f"none ({kind})"

    def is_configured(self) -> bool:
        return False

    def configuration_hint(self) -> str:
        return (
            f"No {self.kind} provider is configured. Set assets.{self.kind}_provider "
            "in config/pipeline_settings.json and provide its credentials as "
            "environment variables. Assets stay 'pending' until then."
        )

    def generate(self, request: AssetRequest, destination: Path) -> AssetResult:
        return AssetResult(
            asset_id=request.asset_id,
            state=ProviderState.UNCONFIGURED,
            provider="none",
            error=None,
            metadata={"reason": self.configuration_hint()},
        )


__all__ = ["AssetProvider", "NullAssetProvider"]
