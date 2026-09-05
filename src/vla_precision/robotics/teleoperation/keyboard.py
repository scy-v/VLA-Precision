"""Keyboard-backed completion signal with the existing double-dot behavior."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class KeyboardEmergencyStopDetector:
    """Translate the Escape key into the environment emergency-stop signal."""

    def __init__(self, *, listener_factory: Callable[[Callable[[Any], None]], Any] | None = None):
        self._stopped = threading.Event()
        if listener_factory is None:
            from pynput import keyboard

            self._escape_key = keyboard.Key.esc
            self._listener = keyboard.Listener(on_press=self._on_press)
        else:
            self._escape_key = "escape"
            self._listener = listener_factory(self._on_press)
        self._listener.start()

    def _on_press(self, key: Any) -> None:
        if key == self._escape_key:
            self._stopped.set()

    def stopped(self) -> bool:
        return self._stopped.is_set()

    def reset(self) -> None:
        self._stopped.clear()

    def close(self) -> None:
        self._listener.stop()


class KeyboardCompletionDetector:
    def __init__(
        self,
        double_press_interval: float = 0.5,
        *,
        listener_factory: Callable[[Callable[[Any], None]], Any] | None = None,
    ):
        self._double_press_interval = float(double_press_interval)
        self._last_press: float | None = None
        self._completed = False
        if listener_factory is None:
            from pynput import keyboard

            self._dot_key = keyboard.KeyCode.from_char(".")
            self._listener = keyboard.Listener(on_press=self._on_press)
        else:
            self._dot_key = "."
            self._listener = listener_factory(self._on_press)
        self._listener.start()

    def _on_press(self, key: Any) -> None:
        if key != self._dot_key:
            return
        now = time.monotonic()
        if self._last_press is not None and now - self._last_press < self._double_press_interval:
            self._completed = True
            self._last_press = None
        else:
            self._last_press = now

    def completed(self) -> bool:
        return self._completed

    def reset(self) -> None:
        self._completed = False
        self._last_press = None

    def close(self) -> None:
        self._listener.stop()
