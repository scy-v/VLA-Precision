"""OpenPI-facing observation adapters retained from the original actor stack."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten, flatten_space
from scipy.spatial.transform import Rotation


class QuaternionToEulerWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, *, dual_arm: bool = False):
        super().__init__(env)
        self.dual_arm = bool(dual_arm)
        keys = ("left/tcp_pose", "right/tcp_pose") if dual_arm else ("tcp_pose",)
        for key in keys:
            self.observation_space["state"][key] = gym.spaces.Box(-np.inf, np.inf, shape=(6,))

    def observation(self, observation):
        keys = ("left/tcp_pose", "right/tcp_pose") if self.dual_arm else ("tcp_pose",)
        for key in keys:
            pose = observation["state"][key]
            observation["state"][key] = np.concatenate(
                (pose[:3], Rotation.from_quat(pose[3:]).as_euler("xyz"))
            )
        return observation


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Flatten configured proprioception in its declared top-level order."""

    def __init__(self, env: gym.Env, *, proprio_keys: tuple[str, ...]):
        super().__init__(env)
        self.proprio_keys = tuple(proprio_keys)
        self.proprio_space = gym.spaces.Dict(
            [(key, env.observation_space["state"][key]) for key in self.proprio_keys]
        )
        self.observation_space = gym.spaces.Dict(
            {"state": flatten_space(self.proprio_space), **env.observation_space["images"]}
        )

    def observation(self, observation):
        return {
            "state": flatten(
                self.proprio_space,
                {key: observation["state"][key] for key in self.proprio_keys},
            ),
            **observation["images"],
        }

