"""Internal Kokoro speech engine.

Kokoro is imported lazily so the rest of PIXEL AI (and its test suite) keeps
working on machines where the optional TTS stack is not installed.  Behaviour
is preserved from the legacy script:

    voice       am_liam
    speed       1.25
    device      cpu
    sample_rate 24000
    lang_code   a

All values come from ``config/pipeline_settings.json`` (``tts`` block) and can
be overridden there; defaults in this module match the legacy contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.logging_setup import get_logger
from engine.tts.text_cleaner import clean_text

log = get_logger("tts")


class TTSError(RuntimeError):
    """Speech synthesis failed with an actionable message."""


@dataclass(frozen=True)
class TTSEngineConfig:
    voice: str = "am_liam"
    speed: float = 1.25
    device: str = "cpu"
    sample_rate: int = 24000
    lang_code: str = "a"

    def identity(self) -> dict[str, Any]:
        """Configuration fields that affect the generated waveform."""
        return {
            "voice": self.voice,
            "speed": self.speed,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "lang_code": self.lang_code,
        }


@dataclass(frozen=True)
class SynthesisResult:
    path: Path
    duration_seconds: float
    characters: int
    words: int
    config: TTSEngineConfig


class KokoroEngine:
    """One reusable Kokoro pipeline, mirroring the legacy behaviour."""

    def __init__(self, config: TTSEngineConfig) -> None:
        self.config = config
        self._pipeline: Any = None

    # -- availability ------------------------------------------------------- #
    @staticmethod
    def availability() -> tuple[bool, str]:
        """Check the optional runtime dependencies without initialising Kokoro."""
        try:
            import numpy  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"numpy is not importable: {exc}"
        try:
            import soundfile  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"soundfile is not importable: {exc}"
        try:
            import kokoro  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"kokoro is not importable: {exc}"
        return True, "Kokoro, NumPy and SoundFile are available"

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from kokoro import KPipeline
        except Exception as exc:  # noqa: BLE001
            raise TTSError(
                "Kokoro is not installed in this Python environment. "
                "Install the TTS dependencies or run with --skip-tts."
            ) from exc
        log.info(
            "Initialising Kokoro (voice=%s, speed=%s, device=%s, lang_code=%s)",
            self.config.voice,
            self.config.speed,
            self.config.device,
            self.config.lang_code,
        )
        try:
            self._pipeline = KPipeline(
                lang_code=self.config.lang_code, device=self.config.device
            )
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Kokoro initialisation failed: {exc}") from exc
        return self._pipeline

    # -- generation --------------------------------------------------------- #
    def synthesize(self, text: str, out_path: Path) -> SynthesisResult:
        """Generate one mono WAV file and return its measured duration."""
        cleaned = clean_text(text)
        if not cleaned.strip():
            raise TTSError("Script is empty after cleaning; refusing to synthesise silence")

        try:
            import numpy as np
            import soundfile as sf
        except Exception as exc:  # noqa: BLE001
            raise TTSError(
                "NumPy and SoundFile are required for Kokoro audio generation. "
                f"Import failed: {exc}"
            ) from exc

        pipeline = self._get_pipeline()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path = out_path.parent / "cleaned_script.txt"
        cleaned_path.write_text(cleaned, encoding="utf-8", newline="\n")

        log.info(
            "Synthesising %d word(s) -> %s",
            len(cleaned.split()),
            out_path.name,
        )
        chunks: list[Any] = []
        try:
            generator = pipeline(
                cleaned,
                voice=self.config.voice,
                speed=self.config.speed,
                split_pattern=r"(?<=[.!?])\s+",
            )
            for result in generator:
                audio = result.audio
                if hasattr(audio, "detach"):
                    audio = audio.detach()
                if hasattr(audio, "cpu"):
                    audio = audio.cpu()
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                chunks.append(np.asarray(audio))
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Kokoro synthesis failed: {exc}") from exc

        if not chunks:
            raise TTSError("Kokoro generated no audio")

        waveform = np.concatenate(chunks)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp.wav")
        try:
            sf.write(tmp_path, waveform, self.config.sample_rate)
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Could not write {out_path.name}: {exc}") from exc
        tmp_path.replace(out_path)

        duration = float(len(waveform)) / float(self.config.sample_rate)
        log.info(
            "Audio ready: %s (%.2f s, %d samples)",
            out_path,
            duration,
            len(waveform),
        )
        return SynthesisResult(
            path=out_path,
            duration_seconds=round(duration, 3),
            characters=len(cleaned),
            words=len(cleaned.split()),
            config=self.config,
        )


__all__ = [
    "KokoroEngine",
    "SynthesisResult",
    "TTSEngineConfig",
    "TTSError",
]
