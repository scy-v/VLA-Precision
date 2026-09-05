"""Task-completion signals independent of reward construction."""

from __future__ import annotations

import threading
from typing import Protocol


class CompletionDetector(Protocol):
    def completed(self) -> bool: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class EventCompletionDetector:
    """Expose the event semantics used by the dual-arm actor."""

    def __init__(self, event: threading.Event):
        self._event = event

    def completed(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()

    def close(self) -> None:
        return None


def build_completion_detector(
    name: str,
    *,
    event: threading.Event | None,
    double_press_interval: float,
    keyboard_enabled: bool,
) -> CompletionDetector:
    """Build the task detector without coupling reward design to an input device."""
    if name != "manual":
        raise KeyError(f"Unknown completion detector {name!r}")
    if event is not None or not keyboard_enabled:
        return EventCompletionDetector(event or threading.Event())

    from vla_precision.robotics.teleoperation.keyboard import KeyboardCompletionDetector

    return KeyboardCompletionDetector(double_press_interval)
