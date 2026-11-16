"""Run the whole PIXEL AI test suite with the standard library.

    python tests/run_all.py
    python tests/run_all.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def main(argv: list[str] | None = None) -> int:
    verbosity = 2 if (argv or sys.argv[1:]) and "-v" in (argv or sys.argv[1:]) else 1
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(APP_ROOT / "tests"),
        pattern="test_*.py",
        top_level_dir=str(APP_ROOT),
    )
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print(
        f"ran {result.testsRun} test(s): "
        f"{len(result.failures)} failure(s), "
        f"{len(result.errors)} error(s), "
        f"{len(result.skipped)} skipped"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
