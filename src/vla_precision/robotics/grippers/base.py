"""Gripper capability boundary independent of a gripper vendor."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Gripper(Protocol):
    @property
    def action_dimension(self) -> int: ...

    def command_chunk(self, positions: np.ndarray) -> None: ...

    def observations(self, robot_state: dict) -> dict[str, np.ndarray]: ...

    def close(self) -> None: ...

