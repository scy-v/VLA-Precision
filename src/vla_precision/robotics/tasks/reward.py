"""Reward definitions kept separate from Gymnasium and input devices."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class RewardFunction(Protocol):
    def compute(self, *, chunk_length: int, succeeded: bool) -> np.ndarray: ...


class ChunkCompletionReward:
    """The exact chunk reward: time reward, then terminal success reward."""

    def __init__(self, *, time_reward: float, completion_reward: float):
        self.time_reward = float(time_reward)
        self.completion_reward = float(completion_reward)

    def compute(self, *, chunk_length: int, succeeded: bool) -> np.ndarray:
        rewards = np.full((chunk_length,), self.time_reward, dtype=np.float32)
        if succeeded:
            rewards[-1] = self.completion_reward
        return rewards


def build_reward_function(
    name: str,
    *,
    time_reward: float,
    completion_reward: float,
) -> RewardFunction:
    """Select a task reward independently from its completion detector."""
    if name != "chunk_completion":
        raise KeyError(f"Unknown reward function {name!r}")
    return ChunkCompletionReward(
        time_reward=time_reward,
        completion_reward=completion_reward,
    )
