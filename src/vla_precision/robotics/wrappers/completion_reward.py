"""Thin Gymnasium adapter combining completion detection and reward design."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from vla_precision.robotics.tasks.completion import CompletionDetector
from vla_precision.robotics.tasks.reward import RewardFunction


class CompletionRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, detector: CompletionDetector, reward: RewardFunction):
        super().__init__(env)
        self.detector = detector
        self.reward_function = reward

    def step(self, action):
        action = np.asarray(action)
        observation, _, env_terminated, truncated, info = self.env.step(action)
        succeeded = self.detector.completed()
        rewards = self.reward_function.compute(chunk_length=action.shape[0], succeeded=succeeded)
        terminated = bool(env_terminated or succeeded)
        info["succeed"] = bool(succeeded)
        if terminated:
            self.detector.reset()
        return observation, rewards, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.detector.reset()
        info["succeed"] = False
        return observation, info

    def close(self):
        self.detector.close()
        return self.env.close()

