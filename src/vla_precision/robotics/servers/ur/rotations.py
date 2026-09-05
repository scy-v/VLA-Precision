"""Pose representation conversions used by the UR controller boundary."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotvec_to_quaternion(rotvec) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_quat()


def quaternion_to_rotvec(quaternion) -> np.ndarray:
    return Rotation.from_quat(quaternion).as_rotvec()


def euler_to_quaternion(euler) -> np.ndarray:
    return Rotation.from_euler("xyz", euler).as_quat()


def euler_to_rotvec(euler) -> np.ndarray:
    return Rotation.from_euler("xyz", euler).as_rotvec()


def rotvec_pose_to_quaternion_pose(rotvec_pose) -> np.ndarray:
    pose = np.asarray(rotvec_pose)
    return np.concatenate((pose[:3], rotvec_to_quaternion(pose[3:])))


def quaternion_pose_to_rotvec_pose(quaternion_pose) -> np.ndarray:
    pose = np.asarray(quaternion_pose)
    return np.concatenate((pose[:3], quaternion_to_rotvec(pose[3:])))
