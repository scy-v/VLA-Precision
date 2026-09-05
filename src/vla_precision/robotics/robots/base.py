"""Robot capability boundary independent of a specific manufacturer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from vla_precision.robotics.grippers.base import Gripper


class Robot(Protocol):
    gripper: Gripper
    action_low: float
    action_high: float

    @property
    def action_dimension(self) -> int: ...

    @property
    def currpos(self) -> np.ndarray: ...

    def refresh_state(self) -> None: ...

    def reset(
        self,
        *,
        joint_reset: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def execute_action_chunk(self, action: np.ndarray) -> np.ndarray: ...

    def observations(self) -> dict[str, Any]: ...

    def request(self, name: str, enabled: bool) -> Any: ...

    def close(self) -> None: ...
