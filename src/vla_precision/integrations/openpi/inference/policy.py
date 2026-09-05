"""Observation adapter shared by VLA-Precision evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _frame(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return array[0] if array.ndim > 0 and array.shape[0] == 1 else array


def make_policy_observation(
    observation: dict[str, Any], *, prompt: str, dual_arm: bool
) -> dict[str, Any]:
    converted = {
        "observation/state": _frame(observation["state"]).astype(np.float32, copy=False),
        "prompt": prompt,
    }
    if dual_arm:
        converted.update(
            {
                "observation/exterior_image": _frame(observation["base_0_rgb"]),
                "observation/left_wrist_image": _frame(observation["left_wrist_0_rgb"]),
                "observation/right_wrist_image": _frame(observation["right_wrist_0_rgb"]),
            }
        )
    else:
        converted.update(
            {
                "observation/image": _frame(observation["base_0_rgb"]),
                "observation/wrist_image": _frame(observation["left_wrist_0_rgb"]),
            }
        )
    return converted


class OpenPISampler:
    def __init__(self, policy: Any, *, prompt: str, dual_arm: bool, label: str):
        self.policy = policy
        self.prompt = prompt
        self.dual_arm = dual_arm
        self.label = label

    def sample(self, observation: dict[str, Any]) -> np.ndarray:
        output = self.policy.infer(
            make_policy_observation(
                observation,
                prompt=self.prompt,
                dual_arm=self.dual_arm,
            )
        )
        return np.asarray(output["actions"], dtype=np.float32)
