"""Teleoperation input boundary shared by keyboard, master arm and future VR."""

from __future__ import annotations

from typing import Protocol


class TeleoperationDevice(Protocol):
    """Device snapshot consumed by the intervention wrapper.

    A single-arm device returns six pose deltas and its gripper/stop buttons;
    dual-arm devices additionally expose ``get_action_state`` with per-arm stop
    buttons. A VR implementation can reuse this interface without touching the
    robot or camera modules.
    """

    def get_action(self) -> tuple[list[float], list[int]]: ...

    def reset_step_mode(self) -> None: ...

    def close(self) -> None: ...
