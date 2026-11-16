"""Verify PIXEL AI output with the *untouched* legacy Kokoro parser.

This is the compatibility test that matters: it imports ``old/pixel_tts.py``
exactly as it ships and runs its real ``parse_shorts`` and ``clean_text``
functions against generated files.  If the legacy code is ever changed in a way
that breaks the format, this tool fails.

Usage
-----
    python tools/verify_tts_compat.py --project "scripts/pixel_tech/my-topic"
    python tools/verify_tts_compat.py --project "..." --generate-audio
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_ROOT.parent
LEGACY_SCRIPT = REPO_ROOT / "old" / "pixel_tts.py"


def load_legacy_module():
    """Import ``old/pixel_tts.py`` without running its entry point."""
    if not LEGACY_SCRIPT.is_file():
        raise SystemExit(f"Legacy TTS script not found: {LEGACY_SCRIPT}")
    spec = importlib.util.spec_from_file_location("legacy_pixel_tts", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit("Could not create an import spec for the legacy script")
    module = importlib.util.module_from_spec(spec)
    sys.modules["legacy_pixel_tts"] = module
    spec.loader.exec_module(module)
    return module


def count_words(text: str) -> int:
    import re

    return len(re.findall(r"[A-Za-z0-9\u00C0-\u024F']+", text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="project folder")
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="also run Kokoro to synthesise the audio (slow on CPU)",
    )
    args = parser.parse_args(argv)

    project = Path(args.project)
    if not project.is_absolute():
        project = (Path.cwd() / project).resolve()
    if not project.is_dir():
        raise SystemExit(f"Project folder not found: {project}")

    long_file = project / "Long.txt"
    short_file = project / "Short.txt"
    for path in (long_file, short_file):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    print("PIXEL AI <-> Kokoro compatibility check")
    print("=" * 70)
    print(f"Legacy script : {LEGACY_SCRIPT}")
    print(f"Project       : {project}")

    print("\nLoading the legacy module (this imports Kokoro/Torch)...")
    legacy = load_legacy_module()
    print(f"  voice     : {getattr(legacy, 'VOICE', '?')}")
    print(f"  speed     : {getattr(legacy, 'SPEED', '?')}")
    print(f"  sample    : {getattr(legacy, 'SAMPLE_RATE', '?')} Hz")
    print(f"  device    : {getattr(legacy, 'DEVICE', '?')}")

    failures: list[str] = []

    # -- Shorts ------------------------------------------------------------- #
    shorts_text = short_file.read_text(encoding="utf-8")
    parsed = legacy.parse_shorts(shorts_text)
    print(f"\n[shorts] legacy parser found {len(parsed)} Short(s)")
    for short in parsed:
        print(
            f"  {short['id']:02d}  {short['title']:<44s} "
            f"{count_words(short['script']):3d} words"
        )
    if not parsed:
        failures.append("The legacy parser found no Shorts in Short.txt")

    # -- Long narration ----------------------------------------------------- #
    long_text = long_file.read_text(encoding="utf-8")
    cleaned = legacy.clean_text(long_text)
    before = count_words(long_text)
    after = count_words(cleaned)
    ratio = after / max(1, before)
    print(f"\n[long] legacy clean_text kept {after}/{before} words ({ratio:.1%})")
    if ratio < 0.90:
        failures.append(
            f"clean_text removed {1 - ratio:.1%} of the long narration "
            "- the text may contain characters the cleaner strips"
        )
    if not cleaned.strip():
        failures.append("clean_text produced empty narration")

    # -- Shorts cleaning ---------------------------------------------------- #
    for short in parsed:
        short_clean = legacy.clean_text(short["script"])
        kept = count_words(short_clean)
        original = count_words(short["script"])
        if kept < original * 0.9:
            failures.append(
                f"Short {short['id']:02d}: clean_text removed "
                f"{original - kept} word(s)"
            )
    print(f"[shorts] clean_text checked for all {len(parsed)} Short(s)")

    # -- Expected output paths --------------------------------------------- #
    expected_long = project / "output" / "Long" / "narration.wav"
    print("\n[output paths the legacy script will write]")
    print(f"  long  : {expected_long}")
    for short in parsed:
        short_id = f"Short_{int(short['id']):02d}"
        expected_short = project / "output" / "Shorts" / short_id / "narration.wav"
        print(f"  short : {expected_short}")

    # -- Optional real synthesis ------------------------------------------- #
    if args.generate_audio:
        print("\n[audio] running the real legacy pipeline (slow)...")
        import subprocess

        folder_name = f"{project.parent.name}/{project.name}"
        result = subprocess.run(
            [sys.executable, str(LEGACY_SCRIPT), folder_name],
            cwd=str(REPO_ROOT),
            check=False,
        )
        if result.returncode != 0:
            failures.append("The legacy TTS pipeline exited with an error")
        elif not expected_long.is_file():
            failures.append(f"Expected narration file was not created: {expected_long}")
        else:
            print(f"  created: {expected_long}")

    # -- Verdict ------------------------------------------------------------ #
    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED - {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASSED - the untouched Kokoro script accepts these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
