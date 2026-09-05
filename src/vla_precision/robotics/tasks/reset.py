"""Task-specific robot reset procedures with the established command ordering."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from vla_precision.robotics.robots.ur import URRobot


class ResetProcedure(Protocol):
    def reset(
        self,
        robot: URRobot,
        *,
        joint_reset: bool,
        options: dict,
    ) -> None: ...


def _quat_pose(euler_pose: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (euler_pose[:3], Rotation.from_euler("xyz", euler_pose[3:]).as_quat())
    )


def _sample_pose(
    reset_pose: np.ndarray,
    reset_range: np.ndarray,
    *,
    allow_three_value_range: bool = False,
) -> np.ndarray:
    reset_pose = reset_pose.copy()
    reset_range = np.abs(reset_range.copy())
    if allow_three_value_range and reset_range.shape == (3,):
        reset_range = np.concatenate((reset_range, [0.0, 0.0, 20.0]))
    sampled = reset_pose.copy()
    sampled[:3] += np.random.uniform(-reset_range[:3], reset_range[:3])
    euler_degrees = np.rad2deg(sampled[3:]) + np.random.uniform(
        -reset_range[3:], reset_range[3:]
    )
    return np.concatenate(
        (sampled[:3], Rotation.from_euler("xyz", euler_degrees, degrees=True).as_quat())
    )


def _sample_reset_pose(robot: URRobot, *, allow_three_value_range: bool = False) -> np.ndarray:
    arm_names = ("left", "right") if robot.dual_arm else ("left",)
    poses = [
        _sample_pose(
            robot.reset_poses[name],
            robot.reset_pose_ranges[name],
            allow_three_value_range=allow_three_value_range,
        )
        for name in arm_names
    ]
    return np.stack(poses, axis=0) if robot.dual_arm else poses[0]


def _nominal_reset_pose(robot: URRobot) -> np.ndarray:
    arm_names = ("left", "right") if robot.dual_arm else ("left",)
    poses = [_quat_pose(robot.reset_poses[name]) for name in arm_names]
    return np.stack(poses, axis=0) if robot.dual_arm else poses[0]


def _joint_reset(robot: URRobot, *, settle: bool) -> None:
    robot._post_server(
        "jointreset",
        read_timeout=60.0,
        retry_transport=True,
    )
    if settle:
        time.sleep(0.5)


def _paused_control_mode(robot: URRobot, *, task_mode: bool) -> None:
    robot.request("force_pause", True, retry_transport=True)
    robot.request("task_mode", task_mode, retry_transport=True)
    robot.request("force_pause", False, retry_transport=True)


def _set_dual_pgi_start(robot: URRobot, *, prompt_first: bool) -> None:
    if prompt_first:
        robot._wait_for_operator("Press Enter to Reset...")
    left = int(
        getattr(robot.gripper, "left_start_position", robot.gripper.left_position)
    )
    right = int(
        getattr(robot.gripper, "right_start_position", robot.gripper.right_position)
    )
    robot._post_server(
        "set_grippers",
        json={"left": left, "right": right},
        retry_transport=True,
    )
    robot.gripper.left_position = left
    robot.gripper.right_position = right
    time.sleep(0.5)


class StandardURReset:
    def reset(self, robot: URRobot, *, joint_reset: bool, options: dict) -> None:
        robot._update_state(retry_transport=True)
        robot._send_pose(robot.currpos, retry_transport=True)
        if joint_reset:
            _joint_reset(robot, settle=True)
        goal = _nominal_reset_pose(robot)
        if robot.random_reset and not robot.dual_arm:
            goal = _sample_reset_pose(robot)
        robot._interpolate_move(goal, timeout=1.0)
        robot._reapply_payload_and_zero_ft()
        if robot.wait_at_reset:
            robot._wait_for_operator("Press Enter to start the new episode...")


class CleanTestTubeReset:
    def reset(self, robot: URRobot, *, joint_reset: bool, options: dict) -> None:
        _set_dual_pgi_start(robot, prompt_first=True)
        _paused_control_mode(robot, task_mode=False)
        robot._update_state(retry_transport=True)
        robot._send_pose(robot.currpos, retry_transport=True)
        if joint_reset:
            _joint_reset(robot, settle=False)
        goal = _sample_reset_pose(robot) if robot.random_reset else _nominal_reset_pose(robot)
        robot._interpolate_move(goal, timeout=1.0)
        robot._wait_for_operator("Press Enter to start the new dual-arm episode...")

        robot.request("force_pause", True, retry_transport=True)
        try:
            robot._post_server(
                "reset_payload",
                read_timeout=10.0,
                retry_transport=True,
            )
            time.sleep(0.25)
            robot._post_server(
                "zero_ft_sensor",
                read_timeout=10.0,
                retry_transport=True,
            )
            time.sleep(0.5)
            robot.request("task_mode", True, retry_transport=True)
        finally:
            robot.request("force_pause", False, retry_transport=True)


class InsertTaskReset:
    """Reset shared by insert-rubber, diagonal-bottles and transfer-cuvette."""

    def reset(self, robot: URRobot, *, joint_reset: bool, options: dict) -> None:
        position = int(
            getattr(robot.gripper, "left_start_position", robot.gripper.left_position)
        )
        robot._post(robot.url + ("open_gripper" if position else "close_gripper"))
        robot.gripper.left_position = position
        time.sleep(0.5)
        robot._post(robot.url + "force_pause", json={"force_pause": False})
        robot._post(robot.url + "task_mode", json={"task_mode": False})

        robot._update_state()
        robot._send_pose(robot.currpos)
        time.sleep(0.3)
        robot._post(robot.url + "update_param", json=robot.options.get("precision_param", {}))
        robot._update_state()
        robot._interpolate_move(_nominal_reset_pose(robot), timeout=1.0)
        if joint_reset:
            robot._post(robot.url + "jointreset")
            time.sleep(0.5)
        goal = (
            _sample_reset_pose(robot, allow_three_value_range=True)
            if robot.random_reset
            else _nominal_reset_pose(robot)
        )
        robot._send_pose(goal)
        time.sleep(0.5)
        robot._reapply_payload_and_zero_ft(retry_transport=False)
        compliance = robot.options.get("compliance_param", {})
        robot._post(robot.url + "update_param", json=compliance)
        robot._wait_for_operator("Press Enter to start the New episode...")
        robot._post(robot.url + "task_mode", json={"task_mode": True})
        robot._post(robot.url + "update_param", json=compliance)


_PROCEDURES = {
    "standard_ur": StandardURReset,
    "clean_test_tube": CleanTestTubeReset,
    "insert_task": InsertTaskReset,
}


def build_reset_procedure(name: str) -> ResetProcedure:
    return _PROCEDURES[name]()
