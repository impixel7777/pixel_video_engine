"""Logging setup for PIXEL AI.

A single configured logger tree keeps pipeline output readable on the console
while persisting a full debug log per production run under ``logs/``.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "pixel_ai"
_CONFIGURED = False


class _ConsoleFormatter(logging.Formatter):
    """Compact console formatter with stage-aware prefixes."""

    COLOURS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__("%(message)s")
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        name = record.name.split(".")[-1]
        text = f"[{name}] {record.getMessage()}"
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        if self.use_colour and level in self.COLOURS and sys.stderr.isatty():
            return f"{self.COLOURS[level]}{text}{self.RESET}"
        return text


def get_logger(name: str = "") -> logging.Logger:
    """Return a namespaced child of the PIXEL AI logger."""
    base = logging.getLogger(LOGGER_NAME)
    if not name:
        return base
    return base.getChild(name)


def configure_logging(
    logs_dir: Path | None = None,
    *,
    level: int | str = logging.INFO,
    run_name: str | None = None,
    quiet: bool = False,
) -> Path | None:
    """Configure console + file logging once per process.

    Returns the log file path when file logging was enabled.
    """
    global _CONFIGURED
    base = logging.getLogger(LOGGER_NAME)

    if _CONFIGURED:
        return getattr(base, "_pixel_log_path", None)

    base.setLevel(logging.DEBUG)
    base.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(_ConsoleFormatter())
    if quiet:
        console.setLevel(logging.WARNING)
    base.addHandler(console)

    log_path: Path | None = None
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_name = (run_name or "pixel-ai").replace(" ", "-").lower()
        log_path = logs_dir / f"{safe_name}-{stamp}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        base.addHandler(file_handler)
        base._pixel_log_path = log_path  # type: ignore[attr-defined]

    _CONFIGURED = True
    return log_path


def reset_logging() -> None:
    """Remove handlers (used by the test-suite)."""
    global _CONFIGURED
    base = logging.getLogger(LOGGER_NAME)
    for handler in list(base.handlers):
        base.removeHandler(handler)
        handler.close()
    _CONFIGURED = False


__all__ = ["LOGGER_NAME", "configure_logging", "get_logger", "reset_logging"]
