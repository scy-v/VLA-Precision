"""Small logging surface controlled by the single top-level debug flag."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping
from datetime import datetime
from io import TextIOBase
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from vla_precision.config.loader import ResolvedConfig

_DEBUG_ENABLED = False
_LOG_STREAM: TextIOBase | None = None
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_STATUS_COLORS = {
    "cyan": "96",
    "magenta": "38;5;201",
    "yellow": "93",
    "green": "92",
}


class _ConsoleFormatter(logging.Formatter):
    """Keep routine terminal messages compact while retaining warning severity."""

    _PLAIN = logging.Formatter("%(message)s")
    _IMPORTANT = logging.Formatter("%(levelname)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        formatter = self._IMPORTANT if record.levelno >= logging.WARNING else self._PLAIN
        return formatter.format(record)


class _Tee(TextIOBase):
    """Mirror Python terminal output to one run log without changing callers."""

    def __init__(self, terminal: TextIOBase, log_stream: TextIOBase):
        self.terminal = terminal
        self.log_stream = log_stream

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_stream.write(_ANSI_ESCAPE.sub("", value))
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_stream.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str | None:
        return self.terminal.encoding

    @property
    def errors(self) -> str | None:
        return self.terminal.errors

    def fileno(self) -> int:
        return self.terminal.fileno()


def status_text(label: str, message: str, *, color: str = "cyan") -> str:
    """Format one concise, visibly grouped terminal status message."""
    text = f"[{label}] {message}"
    code = _STATUS_COLORS.get(color)
    if code is None or not sys.stderr.isatty():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def log_status(logger: logging.Logger, label: str, message: str, *, color: str = "cyan") -> None:
    logger.info("%s", status_text(label, message, color=color))


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def configure_logging(
    level: str = "INFO",
    *,
    debug: bool = False,
    experiment: str = "experiment",
    role: str = "run",
    debug_root: str | Path = "./debug",
) -> Path | None:
    """Keep the terminal concise and write project debug records to one run log."""
    global _DEBUG_ENABLED, _LOG_STREAM
    _DEBUG_ENABLED = bool(debug)
    if isinstance(sys.stdout, _Tee):
        sys.stdout = sys.stdout.terminal
    if isinstance(sys.stderr, _Tee):
        sys.stderr = sys.stderr.terminal
    if _LOG_STREAM is not None:
        _LOG_STREAM.close()
        _LOG_STREAM = None
    console_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=console_level,
        format="%(message)s",
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.setLevel(console_level)
        handler.setFormatter(_ConsoleFormatter())
    # Orbax and Flax route verbose checkpoint internals through absl. Project
    # checkpoint status remains visible through the concise messages emitted
    # by the learner; third-party warnings and errors are still preserved.
    logging.getLogger("absl").setLevel(logging.WARNING)
    project_logger = logging.getLogger("vla_precision")
    for handler in project_logger.handlers:
        handler.close()
    project_logger.handlers.clear()
    project_logger.setLevel(logging.DEBUG if debug else console_level)

    if not debug:
        return None

    experiment_dir = Path(debug_root).expanduser() / _safe_filename(experiment)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = experiment_dir / f"{timestamp}_{_safe_filename(role)}.log"
    _LOG_STREAM = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, _LOG_STREAM)
    sys.stderr = _Tee(sys.stderr, _LOG_STREAM)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)
    debug_handler = logging.FileHandler(log_path, encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(lambda record: record.levelno < console_level)
    debug_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    project_logger.addHandler(debug_handler)
    project_logger.info("debug log: %s", log_path)
    return log_path


def log_startup_config(resolved: ResolvedConfig, *, role: str) -> None:
    if not resolved.config.debug:
        return
    config = resolved.config
    summary: dict[str, Any] = {
        "role": role,
        "experiment": config.experiment.name,
        "model": config.openpi.model,
        "openpi_config": config.openpi.name,
        "initialization_config": config.openpi.initialization_config_name,
        "action_horizon": config.task.action_horizon,
        "image_keys": config.task.image_keys,
        "proprio_keys": config.task.proprio_keys,
        "replay_capacity": config.buffers.replay_capacity,
        "correction_capacity": config.buffers.correction_capacity,
        "context_capacity": config.buffers.context_capacity,
        "shared_config_sha256": resolved.shared_sha256,
        "preprocess_config_sha256": resolved.preprocess_sha256,
    }
    logging.getLogger("vla_precision").info(
        "resolved startup config:\n%s",
        OmegaConf.to_yaml(OmegaConf.create(summary), sort_keys=True),
    )


def debug_record(tag: str, payload: Mapping[str, Any] | str) -> None:
    """Emit one compact debug record through the project's single log level."""
    if not _DEBUG_ENABLED:
        return
    logger = logging.getLogger("vla_precision")
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if isinstance(payload, str):
        logger.debug("%s: %s", tag, payload)
    else:
        logger.debug("%s:\n%s", tag, OmegaConf.to_yaml(OmegaConf.create(dict(payload)), sort_keys=True))
