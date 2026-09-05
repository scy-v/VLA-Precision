"""Gymnasium adapter for OpenPI action chunks.

The wrapped robot environment is already chunk-aware. This wrapper preserves
the reference implementation's leading one-frame observation axis and
per-action reward vector.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def stack_single_observation(observation: Any) -> Any:
    if isinstance(observation, dict):
        return {key: stack_single_observation(value) for key, value in observation.items()}
    return np.asarray(observation)[None]


def stack_space(space: gym.Space, repeat: int) -> gym.Space:
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({key: stack_space(value, repeat) for key, value in space.spaces.items()})
    raise TypeError(f"Unsupported space for action/observation stacking: {type(space).__name__}")


class ActionChunkWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action_horizon: int = 1):
        super().__init__(env)
        self.action_horizon = int(action_horizon)
        self.observation_space = stack_space(env.observation_space, 1)
        self.action_space = stack_space(env.action_space, self.action_horizon)

    def step(self, action, *args):
        action = np.asarray(action)
        if action.ndim != 2 or action.shape != self.action_space.shape:
            raise ValueError(
                f"chunk boundary mismatch: action={action.shape}, expected={self.action_space.shape}"
            )
        observation, reward, terminated, truncated, info = self.env.step(action, *args)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        if reward.shape != (action.shape[0],):
            raise ValueError(
                f"chunk boundary mismatch: reward={reward.shape}, expected=({action.shape[0]},)"
            )
        return stack_single_observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return stack_single_observation(observation), info
