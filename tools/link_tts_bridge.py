"""Bridge PIXEL AI projects into the untouched legacy Kokoro folder.

The legacy script resolves ``old/scripts/<folder_name>``.  Rather than editing
it, this tool creates a Windows *directory junction*:

    old/scripts/pixel_tech  ->  pixel-ai/scripts/pixel_tech

After that the legacy script sees new projects as ordinary folders:

    python old/pixel_tts.py pixel_tech/my-topic-slug

Usage
-----
    python tools/link_tts_bridge.py                 # every profile
    python tools/link_tts_bridge.py --profile pixel_tech
    python tools/link_tts_bridge.py --remove        # undo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_ROOT.parent
LEGACY_SCRIPTS = REPO_ROOT / "old" / "scripts"
SOURCE_ROOT = APP_ROOT / "scripts"


def create_junction(link: Path, target: Path) -> tuple[bool, str]:
    if link.exists():
        return True, "already exists"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode == 0:
        return True, "created"
    return False, (result.stdout or result.stderr).strip()


def remove_junction(link: Path) -> tuple[bool, str]:
    if not link.exists():
        return True, "not present"
    if not link.is_dir():
        return False, "not a directory"
    result = subprocess.run(
        ["cmd", "/c", "rmdir", str(link)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode == 0:
        return True, "removed"
    return False, (result.stdout or result.stderr).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="link only this profile")
    parser.add_argument("--remove", action="store_true", help="remove the junctions")
    args = parser.parse_args(argv)

    if not (REPO_ROOT / "old" / "pixel_tts.py").is_file():
        raise SystemExit(f"Legacy folder not found under {REPO_ROOT / 'old'}")

    LEGACY_SCRIPTS.mkdir(parents=True, exist_ok=True)

    if args.profile:
        candidates = [args.profile]
    else:
        candidates = (
            sorted(path.name for path in SOURCE_ROOT.iterdir() if path.is_dir())
            if SOURCE_ROOT.is_dir()
            else []
        )

    if not candidates:
        print(f"No generated profile folders under {SOURCE_ROOT} yet.")
        print("Run a production first, then re-run this tool.")
        return 0

    print(f"Legacy scripts folder : {LEGACY_SCRIPTS}")
    print(f"PIXEL AI scripts      : {SOURCE_ROOT}\n")

    failures = 0
    for profile_id in candidates:
        target = SOURCE_ROOT / profile_id
        link = LEGACY_SCRIPTS / profile_id

        if args.remove:
            ok, message = remove_junction(link)
        else:
            if not target.is_dir():
                print(f"  skip {profile_id}: {target} does not exist")
                continue
            ok, message = create_junction(link, target)

        status = "ok  " if ok else "FAIL"
        print(f"  {status} {link.name:<18s} {message}")
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"{failures} link(s) failed.")
        return 1

    if not args.remove:
        print("The legacy script can now read these projects directly:")
        print('  python old/pixel_tts.py pixel_tech/<topic-slug>')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
