"""Hardware-free components for learner space construction and unit tests."""

from __future__ import annotations

import numpy as np


class FakeRobot:
    action_low = -3.0
    action_high = 3.0

    def __init__(
        self,
        *,
        action_dimension: int,
        proprio_keys: tuple[str, ...],
        gripper_action_dimension: int,
        gripper_kind: str,
    ):
        self._action_dimension = int(action_dimension)
        self.proprio_keys = proprio_keys
        self.gripper = type(
            "FakeGripper",
            (),
            {"action_dimension": gripper_action_dimension, "kind": gripper_kind},
        )()
        self.server_config = {}
        self.currpos = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        self.right_currpos = self.currpos.copy()

    @property
    def action_dimension(self) -> int:
        return self._action_dimension

    def _update_state(self) -> None:
        return None

    def refresh_state(self) -> None:
        self._update_state()

    def observations(self) -> dict[str, np.ndarray]:
        result = {}
        for key in self.proprio_keys:
            field = key.split("/", 1)[-1]
            size = {"tcp_pose": 7, "tcp_vel": 6, "tcp_force": 3, "tcp_torque": 3}.get(field, 1)
            result[key] = np.zeros((size,), dtype=np.float32)
            if field == "tcp_pose":
                result[key][-1] = 1.0
        return result

    def execute_action_chunk(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        return action[None] if action.ndim == 1 else action.copy()

    def reset(
        self,
        *,
        joint_reset: bool = False,
        options: dict | None = None,
    ) -> dict[str, np.ndarray]:
        return self.observations()

    def request(self, name: str, enabled: bool):
        return {name: bool(enabled)}

    def close(self) -> None:
        return None


class FakeCamera:
    def __init__(self, name: str, shape: tuple[int, int, int] = (224, 224, 3)):
        self._name = name
        self.shape = shape

    @property
    def name(self) -> str:
        return self._name

    def capture(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.uint8)

    def close(self) -> None:
        return None
