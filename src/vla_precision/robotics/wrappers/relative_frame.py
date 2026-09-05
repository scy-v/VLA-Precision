"""Reset-relative frame adapter shared by single- and dual-arm UR environments."""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R


def _pose_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"RelativeFrameWrapper expects xyz+quat pose shape (7,), got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


class RelativeFrameWrapper(gym.Wrapper):
    """OpenPI-style reset-relative observation wrapper.

    This wrapper intentionally does not transform actions. OpenPI actions are
    treated as TCP-frame deltas and are converted to target poses by the robot
    server immediately before each action is executed.
    """

    def __init__(
        self,
        env,
        include_relative_pose: bool = True,
        actions_in_tcp_frame: bool = True,
        dual_arm: bool = False,
    ):
        super().__init__(env)
        self.dual_arm = bool(dual_arm)
        self.include_relative_pose = bool(include_relative_pose)
        self.actions_in_tcp_frame = bool(actions_in_tcp_frame)
        self.reference_transform = np.eye(4, dtype=np.float64)
        self.right_reference_transform = np.eye(4, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["original_state_obs"] = copy.deepcopy(obs["state"])
        if self.include_relative_pose:
            if self.dual_arm:
                self.reference_transform = _pose_to_transform(obs["state"]["left/tcp_pose"])
                self.right_reference_transform = _pose_to_transform(obs["state"]["right/tcp_pose"])
            else:
                self.reference_transform = _pose_to_transform(obs["state"]["tcp_pose"])
        return self.transform_observation(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        info["original_state_obs"] = copy.deepcopy(obs["state"])
        return self.transform_observation(obs), reward, done, truncated, info

    def execute_action_chunk(self, action):
        return self.env.execute_action_chunk(action)

    def finish_action_chunk_step(self):
        obs, reward, done, truncated, info = self.env.finish_action_chunk_step()
        info["original_state_obs"] = copy.deepcopy(obs["state"])
        return self.transform_observation(obs), reward, done, truncated, info

    def base_action_to_tcp_action(self, action: np.ndarray) -> np.ndarray:
        """Convert a base/world-frame delta action into a TCP-frame delta action."""
        action = np.asarray(action).copy()
        if not self.actions_in_tcp_frame:
            return action
        if action.shape[0] < (14 if self.dual_arm else 6):
            return action
        arm_specs = [((0, 6), np.asarray(self.unwrapped.currpos, dtype=np.float64))]
        if self.dual_arm:
            arm_specs.append(((7, 13), np.asarray(self.unwrapped.right_currpos, dtype=np.float64)))
        for (start, stop), tcp_pose in arm_specs:
            tcp_rot = R.from_quat(tcp_pose[3:])
            base_delta_pos = np.asarray(action[start : start + 3], dtype=np.float64)
            base_delta_rot = R.from_euler("xyz", np.asarray(action[start + 3 : stop], dtype=np.float64))
            tcp_delta_pos = tcp_rot.inv().apply(base_delta_pos)
            tcp_delta_rot = tcp_rot.inv() * base_delta_rot * tcp_rot
            action[start : start + 3] = tcp_delta_pos.astype(action.dtype, copy=False)
            action[start + 3 : stop] = tcp_delta_rot.as_euler("xyz").astype(action.dtype, copy=False)
        return action

    def transform_observation(self, obs):
        if not self.include_relative_pose:
            return obs
        pose_specs = [("tcp_pose", self.reference_transform)]
        if self.dual_arm:
            pose_specs = [
                ("left/tcp_pose", self.reference_transform),
                ("right/tcp_pose", self.right_reference_transform),
            ]
        for key, reference in pose_specs:
            current_transform = _pose_to_transform(obs["state"][key])
            T_rel = np.linalg.inv(reference) @ current_transform
            rel_pos = T_rel[:3, 3]
            rel_quat = R.from_matrix(T_rel[:3, :3]).as_quat()
            obs["state"][key] = np.concatenate((rel_pos, rel_quat))
        return obs
