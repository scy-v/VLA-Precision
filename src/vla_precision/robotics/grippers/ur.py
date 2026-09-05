"""PGI gripper client hosted by the UR robot HTTP server."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

class FixedGripper:
    action_dimension = 0

    def command_chunk(self, positions: np.ndarray) -> None:
        return None

    def observations(self, robot_state: dict) -> dict[str, np.ndarray]:
        return {}

    def close(self) -> None:
        return None


class URGripper:
    """Send per-timestep PGI commands and retain the last discrete state."""

    kind = "pgi"

    def __init__(
        self,
        *,
        server_url: str,
        dual_arm: bool = False,
        left_start_position: int = 1,
        right_start_position: int = 1,
        post: Callable[..., Any] | None = None,
    ):
        if post is None:
            import requests

            post = requests.post
        self._post = post
        self.server_url = server_url.rstrip("/") + "/"
        self.dual_arm = bool(dual_arm)
        self.left_start_position = int(left_start_position)
        self.right_start_position = int(right_start_position)
        self.left_position = self.left_start_position
        self.right_position = self.right_start_position

    @property
    def action_dimension(self) -> int:
        return 2 if self.dual_arm else 1

    def command_chunk(self, positions: np.ndarray) -> None:
        values = np.asarray(positions, dtype=np.float64)
        if self.dual_arm:
            values = values.reshape((-1, 2))
            if not len(values):
                return
            left_commands = ["open_gripper" if value >= 0.5 else "close_gripper" for value in values[:, 0]]
            right_commands = ["open_gripper" if value >= 0.5 else "close_gripper" for value in values[:, 1]]
            self.left_position = int(values[-1, 0] >= 0.5)
            self.right_position = int(values[-1, 1] >= 0.5)
            self._post(
                self.server_url + "gripper_chunk",
                json={"left_commands": left_commands, "right_commands": right_commands},
            )
            return

        values = values.reshape(-1)
        if not len(values):
            return
        commands = ["open_gripper" if value >= 0.5 else "close_gripper" for value in values]
        self.left_position = int(values[-1] >= 0.5)
        self._post(self.server_url + "gripper_chunk", json={"commands": commands})

    def observations(self, robot_state: dict) -> dict[str, np.ndarray]:
        if self.dual_arm:
            return {
                "left/gripper_pose": np.asarray(robot_state["left"]["gripper_pos"]),
                "right/gripper_pose": np.asarray(robot_state["right"]["gripper_pos"]),
            }
        return {"gripper_pose": np.asarray(robot_state["gripper_pos"])}

    def close(self) -> None:
        return None
