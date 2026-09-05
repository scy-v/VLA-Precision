"""Asynchronous serial PGI gripper used by the Franka Stage-I setup."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def _open_pgi(port: str, *, initialize: bool):
    from pyDHgripper import PGE

    if initialize:
        return PGE(port=port)

    import crcmod
    import serial

    gripper = PGE.__new__(PGE)
    gripper.ser = serial.Serial(port=port, baudrate=115200)
    gripper.crc16 = crcmod.mkCrcFun(
        0x18005,
        rev=True,
        initCrc=0xFFFF,
        xorOut=0x0000,
    )
    return gripper


class FrankaPGIGripper:
    """Keep serial I/O off the action hot path, as in Franka data collection."""

    kind = "franka_pgi"
    action_dimension = 1

    def __init__(
        self,
        *,
        port: str,
        reverse: bool = False,
        initialize: bool = True,
        close_threshold: float = 0.7,
        force: int = 100,
        speed: int = 100,
        start_position: float = 1.0,
        device: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        start_worker: bool = True,
    ):
        self.port = port
        self.reverse = bool(reverse)
        self.open_before_reset = bool(initialize)
        self.close_threshold = float(close_threshold)
        self._sleep = sleep
        self._device = device or _open_pgi(port, initialize=initialize)
        if initialize:
            self._device.init_feedback()
        self._device.set_force(int(force))
        self._device.set_vel(int(speed))

        self._lock = threading.Lock()
        self._target_position = float(start_position)
        self._last_hardware_position: int | None = int(start_position > close_threshold)
        self.position: float | None = None
        self._reader_error: Exception | None = None
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None
        if start_worker:
            self._reader_thread = threading.Thread(
                target=self._read_state,
                name="vla-precision-franka-pgi-feedback",
                daemon=True,
            )
            self._reader_thread.start()
            while self.position is None:
                if self._reader_error is not None:
                    raise RuntimeError("Franka PGI feedback reader failed") from self._reader_error
                self._sleep(0.1)

    def command_chunk(self, positions: np.ndarray) -> None:
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if len(values):
            with self._lock:
                self._target_position = float(values[-1])

    def prepare_reset(self) -> None:
        """Match ``init_gripper``: only initialized hardware opens on reset."""
        if not self.open_before_reset:
            return
        with self._lock:
            self._target_position = 1.0
            self._last_hardware_position = None

    def _feedback_once(self) -> None:
        with self._lock:
            target = self._target_position
            last_hardware_position = self._last_hardware_position
        hardware_position = 0 if target <= self.close_threshold else 1
        if self.reverse:
            hardware_position = 1 - hardware_position
        if hardware_position != last_hardware_position:
            self._device.set_pos(val=1000 * hardware_position, blocking=False)
            with self._lock:
                self._last_hardware_position = hardware_position

        position = float(self._device.read_pos()) / 1000.0
        if self.reverse:
            position = 1.0 - position
        with self._lock:
            self.position = position

    def _read_state(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._feedback_once()
                self._sleep(0.01)
        except Exception as error:  # noqa: BLE001 -- preserve the hardware driver error
            self._reader_error = error

    def observations(self, robot_state: dict) -> dict[str, np.ndarray]:
        del robot_state
        if self._reader_error is not None:
            raise RuntimeError("Franka PGI feedback reader failed") from self._reader_error
        return {"gripper_pose": np.asarray([self.position], dtype=np.float32)}

    def close(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        close = getattr(self._device, "close", None)
        if close is not None:
            close()
            return
        serial_device = getattr(self._device, "ser", None)
        if serial_device is not None:
            serial_device.close()
