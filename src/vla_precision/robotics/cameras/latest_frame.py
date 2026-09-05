"""Latest-frame asynchronous camera adapter."""

from __future__ import annotations

import queue
import threading

import numpy as np

from vla_precision.robotics.cameras.base import Camera


class LatestFrameCamera:
    """Preserve the actor's async semantics: discard stale, unread frames."""

    def __init__(self, camera: Camera, *, timeout: float = 5.0):
        self.camera = camera
        self.timeout = float(timeout)
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self._enabled = True
        self._reader = threading.Thread(
            target=self._read_frames,
            name=f"acob-stream-camera-{camera.name}",
            daemon=False,
        )
        self._reader.start()

    @property
    def name(self) -> str:
        return self.camera.name

    def _read_frames(self) -> None:
        while self._enabled:
            try:
                frame = self.camera.capture()
            except Exception:  # noqa: BLE001 -- third-party camera drivers use unrelated exception types
                self._enabled = False
                return
            if not self._frames.empty():
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
            self._frames.put(frame)

    def capture(self) -> np.ndarray:
        return self._frames.get(timeout=self.timeout)

    def close(self) -> None:
        self._enabled = False
        self._reader.join(timeout=self.timeout)
        self.camera.close()
