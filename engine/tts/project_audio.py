"""Project-level TTS orchestration.

Writes the legacy-compatible layout::

    output/
        Long/cleaned_script.txt
        Long/narration.wav
        Shorts/Short_01/cleaned_script.txt
        Shorts/Short_01/narration.wav
        ...

and records the measured durations in ``TTS.json``.  No file is created unless
the speech engine actually produced it; verification (without Kokoro installed)
is a separate, deterministic operation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from config.settings import PipelineSettings
from engine.file_manager import (
    ProjectPaths,
    atomic_write_json,
    atomic_write_text,
    read_json,
    utc_now_iso,
)
from engine.logging_setup import get_logger
from engine.tts.kokoro_engine import KokoroEngine, SynthesisResult, TTSEngineConfig, TTSError
from engine.tts.shorts_parser import ParsedShort, parse_shorts_text
from engine.tts.text_cleaner import clean_text

log = get_logger("tts")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def engine_config(settings: PipelineSettings) -> TTSEngineConfig:
    """Build the speech configuration from the central config file."""
    return TTSEngineConfig(
        voice=settings.tts_voice,
        speed=settings.tts_speed,
        device=settings.tts_device,
        sample_rate=settings.tts_sample_rate,
        lang_code=settings.tts_lang_code,
    )


# --------------------------------------------------------------------------- #
# WAV helpers (stdlib only)
# --------------------------------------------------------------------------- #


def wav_duration(path: Path) -> float:
    """Measure a PCM WAV file without importing NumPy or SoundFile."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if rate <= 0:
                raise TTSError(f"Invalid sample rate in {path}")
            return round(frames / float(rate), 3)
    except (wave.Error, OSError) as exc:
        raise TTSError(f"Could not read WAV duration from {path}: {exc}") from exc


def insert_pauses(path: Path, pauses: Iterable[tuple[float, float]]) -> float:
    """Insert silence into a PCM WAV at absolute second offsets.

    ``pauses`` is an iterable of ``(after_seconds, silence_seconds)``.  The file
    is rewritten atomically; the measured final duration is returned.  This is
    used to honour ``AudioPlan.pauses`` without synthesising a second time.
    """
    points = sorted(
        (max(0.0, float(after)), max(0.0, float(seconds)))
        for after, seconds in pauses
        if seconds > 0
    )
    if not points:
        return wav_duration(path)

    try:
        with wave.open(str(path), "rb") as handle:
            params = handle.getparams()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, OSError) as exc:
        raise TTSError(f"Could not read {path} for pause insertion: {exc}") from exc

    if params.sampwidth <= 0 or params.nchannels <= 0 or params.framerate <= 0:
        raise TTSError(f"Unsupported WAV format in {path}")

    frame_size = params.sampwidth * params.nchannels
    total_frames = len(frames) // frame_size
    pieces: list[bytes] = []
    cursor = 0
    # Apply from the end so offsets stay valid.
    for after, seconds in sorted(points, key=lambda item: item[0], reverse=True):
        index = min(total_frames, max(0, int(round(after * params.framerate))))
        pieces.append(frames[index * frame_size :])
        silence_frames = max(0, int(round(seconds * params.framerate)))
        pieces.append(b"\x00" * (silence_frames * frame_size))
        frames = frames[: index * frame_size]
        total_frames = index
    pieces.append(frames)
    # Pieces were appended in reverse order.
    merged = b"".join(reversed(pieces))

    tmp = path.with_suffix(path.suffix + ".tmp.wav")
    with wave.open(str(tmp), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(merged)
    tmp.replace(path)
    return wav_duration(path)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relative(paths: ProjectPaths, path: Path) -> str:
    try:
        return str(path.relative_to(paths.root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _entry(
    paths: ProjectPaths,
    *,
    cleaned: Path,
    narration: Path,
    base_narration: Path,
    text: str,
    result: SynthesisResult | None,
    reused: bool,
    config: TTSEngineConfig,
) -> dict[str, Any]:
    cleaned_text = clean_text(text)
    payload: dict[str, Any] = {
        "script_sha256": _sha256_text(text),
        "cleaned_sha256": _sha256_text(cleaned_text),
        "cleaned_script": _relative(paths, cleaned),
        "narration": _relative(paths, narration),
        "base_narration": _relative(paths, base_narration),
        "config": config.identity(),
        "reused": reused,
    }
    if result is not None:
        payload.update(
            {
                "duration_seconds": result.duration_seconds,
                "characters": result.characters,
                "words": result.words,
            }
        )
    elif narration.is_file():
        payload["duration_seconds"] = wav_duration(narration)
        payload["characters"] = len(cleaned_text)
        payload["words"] = len(cleaned_text.split())
    return payload


def _matches_config(entry: dict[str, Any], config: TTSEngineConfig, text_sha: str) -> bool:
    if not entry or entry.get("config") != config.identity():
        return False
    if entry.get("script_sha256") != text_sha:
        return False
    return True


def _load_existing(paths: ProjectPaths) -> dict[str, Any]:
    return read_json(paths.tts_metadata, default={}) or {}


def _write_metadata(paths: ProjectPaths, report: dict[str, Any]) -> None:
    atomic_write_json(paths.tts_metadata, report)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate_project_audio(
    paths: ProjectPaths,
    settings: PipelineSettings,
    *,
    force: bool = False,
    engine: KokoroEngine | None = None,
) -> dict[str, Any]:
    """Generate long + Short narration audio and return a measured report."""
    long_path = paths.long_script
    short_path = paths.short_script
    if not long_path.is_file():
        raise TTSError(f"Long.txt is missing: {long_path}")
    if not short_path.is_file():
        raise TTSError(f"Short.txt is missing: {short_path}")

    long_text = long_path.read_text(encoding="utf-8")
    shorts_text = short_path.read_text(encoding="utf-8")
    shorts = parse_shorts_text(shorts_text)
    if not shorts:
        raise TTSError(
            "Short.txt contains no parsable Shorts. Expected headings such as "
            "'SHORT 01 \u2014 Title'."
        )

    config = engine_config(settings)
    existing = _load_existing(paths)
    existing_long = (existing.get("long") or {}) if isinstance(existing, dict) else {}
    existing_shorts = {
        int(entry.get("index", -1)): entry
        for entry in (existing.get("shorts") or [])
        if isinstance(entry, dict)
    }

    long_cleaned = paths.tts_output / "Long" / "cleaned_script.txt"
    long_narration = paths.tts_long_audio
    long_base = paths.tts_long_base
    long_cleaned_text = clean_text(long_text)
    long_sha = _sha256_text(long_text)

    reuse_long = (
        not force
        and _matches_config(existing_long, config, long_sha)
        and long_narration.is_file()
        and long_cleaned.is_file()
    )

    engine = engine if engine is not None else KokoroEngine(config)
    long_result: SynthesisResult | None = None
    if reuse_long:
        log.info("Reusing existing long narration: %s", long_narration)
    else:
        if hasattr(engine, "synthesize"):
            long_result = engine.synthesize(long_text, long_narration)
        else:  # test double
            long_result = _synthesize_with_double(engine, long_text, long_narration, config, paths)

    # Keep a raw copy so pause insertion is idempotent across resumes.
    if long_narration.is_file() and (not long_base.is_file() or not reuse_long):
        shutil.copy2(long_narration, long_base)

    long_entry = _entry(
        paths,
        cleaned=long_cleaned,
        narration=long_narration,
        base_narration=long_base,
        text=long_text,
        result=long_result,
        reused=reuse_long,
        config=config,
    )
    if reuse_long:
        # Ensure the canonical cleaned file exists even if an older run omitted it.
        atomic_write_text(long_cleaned, long_cleaned_text)

    short_entries: list[dict[str, Any]] = []
    for short in shorts:
        short_dir = paths.tts_output / "Shorts" / f"Short_{short.index:02d}"
        cleaned_path = short_dir / "cleaned_script.txt"
        narration_path = paths.tts_short_audio(short.index)
        base_path = paths.tts_short_base(short.index)
        cleaned_text = clean_text(short.narration)
        sha = _sha256_text(short.narration)
        previous = existing_shorts.get(short.index, {})
        reuse = (
            not force
            and _matches_config(previous, config, sha)
            and narration_path.is_file()
            and cleaned_path.is_file()
        )
        result: SynthesisResult | None = None
        if reuse:
            log.info("Reusing existing Short %02d narration", short.index)
        else:
            if hasattr(engine, "synthesize"):
                result = engine.synthesize(short.narration, narration_path)
            else:
                result = _synthesize_with_double(
                    engine, short.narration, narration_path, config, paths
                )
        if narration_path.is_file() and (not base_path.is_file() or not reuse):
            shutil.copy2(narration_path, base_path)
        entry = _entry(
            paths,
            cleaned=cleaned_path,
            narration=narration_path,
            base_narration=base_path,
            text=short.narration,
            result=result,
            reused=reuse,
            config=config,
        )
        entry.update({"index": short.index, "title": short.title})
        if reuse:
            atomic_write_text(cleaned_path, cleaned_text)
        short_entries.append(entry)

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "engine": "kokoro",
        "config": config.identity(),
        "layout": {
            "long": "output/Long",
            "shorts": "output/Shorts/Short_NN",
        },
        "long": long_entry,
        "shorts": short_entries,
        "notes": [
            "Durations are measured from the generated WAV files.",
            "Long.txt and Short.txt are copied through the canonical cleaner into cleaned_script.txt.",
        ],
    }
    _write_metadata(paths, report)
    log.info(
        "TTS complete: long %.1fs, %d Short(s)",
        float(long_entry.get("duration_seconds", 0)),
        len(short_entries),
    )
    return report


def _synthesize_with_double(
    engine: Any,
    text: str,
    path: Path,
    config: TTSEngineConfig,
    paths: ProjectPaths,
) -> SynthesisResult:
    """Adapter for test doubles exposing ``synthesize_text``.

    Tests can pass a small object that writes silence without Kokoro; production
    always uses :class:`KokoroEngine`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    result = engine.synthesize_text(text, path, config)
    if not isinstance(result, SynthesisResult):
        duration = wav_duration(path)
        cleaned = clean_text(text)
        return SynthesisResult(
            path=path,
            duration_seconds=duration,
            characters=len(cleaned),
            words=len(cleaned.split()),
            config=config,
        )
    return result


# --------------------------------------------------------------------------- #
# Verification (no heavy imports, no generation)
# --------------------------------------------------------------------------- #


def verify_project_audio(
    paths: ProjectPaths,
    *,
    expected_shorts: int | None = None,
) -> dict[str, Any]:
    """Check scripts, parsing and output layout without running Kokoro."""
    problems: list[str] = []
    checks: list[str] = []

    if not paths.long_script.is_file():
        problems.append(f"Long.txt is missing: {paths.long_script}")
    else:
        long_text = paths.long_script.read_text(encoding="utf-8")
        cleaned = clean_text(long_text)
        if not cleaned.strip():
            problems.append("Long.txt is empty after cleaning")
        else:
            checks.append(f"Long.txt: {len(cleaned.split())} words after cleaning")
        if paths.tts_long_audio.is_file():
            checks.append(f"Long audio: {wav_duration(paths.tts_long_audio):.2f}s")
        else:
            problems.append("Long narration WAV is missing (run generate-audio)")
        if not (paths.tts_output / "Long" / "cleaned_script.txt").is_file():
            problems.append("output/Long/cleaned_script.txt is missing")

    shorts: list[ParsedShort] = []
    if not paths.short_script.is_file():
        problems.append(f"Short.txt is missing: {paths.short_script}")
    else:
        shorts = parse_shorts_text(paths.short_script.read_text(encoding="utf-8"))
        if not shorts:
            problems.append("Short.txt contains no parsable Shorts")
        else:
            checks.append(f"Short.txt: {len(shorts)} parsable Short(s)")
        if expected_shorts is not None and len(shorts) != expected_shorts:
            problems.append(
                f"Short.txt has {len(shorts)} Short(s), profile requires {expected_shorts}"
            )
        for short in shorts:
            folder = paths.tts_output / "Shorts" / f"Short_{short.index:02d}"
            if not (folder / "cleaned_script.txt").is_file():
                problems.append(f"Short {short.index:02d}: cleaned_script.txt is missing")
            audio = folder / "narration.wav"
            if not audio.is_file():
                problems.append(f"Short {short.index:02d}: narration.wav is missing")
            else:
                checks.append(f"Short {short.index:02d}: {wav_duration(audio):.2f}s")

    return {
        "ok": not problems,
        "checks": checks,
        "problems": problems,
        "shorts": len(shorts),
    }


__all__ = [
    "engine_config",
    "generate_project_audio",
    "insert_pauses",
    "verify_project_audio",
    "wav_duration",
]
