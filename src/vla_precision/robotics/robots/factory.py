"""Construct interchangeable robots behind the shared capability interface."""

from __future__ import annotations

from collections.abc import Callable

from vla_precision.config.schema import RootConfig
from vla_precision.robotics.grippers.base import Gripper
from vla_precision.robotics.robots.base import Robot
from vla_precision.robotics.robots.franka import FrankaRobot
from vla_precision.robotics.robots.ur import URRobot
from vla_precision.robotics.tasks.reset import ResetProcedure

RobotFactory = Callable[..., Robot]


def _ur_robot(
    config: RootConfig,
    *,
    gripper: Gripper,
    reset_procedure: ResetProcedure,
    dual_arm: bool,
    expected_shared_sha256: str | None = None,
    strict_distributed_consistency: bool | None = True,
) -> Robot:
    return URRobot(
        config.robot,
        gripper=gripper,
        reset_procedure=reset_procedure,
        dual_arm=dual_arm,
        expected_shared_sha256=expected_shared_sha256,
        strict_distributed_consistency=strict_distributed_consistency,
    )


def _franka_robot(
    config: RootConfig,
    *,
    gripper: Gripper,
    reset_procedure: ResetProcedure,
    dual_arm: bool,
) -> Robot:
    del reset_procedure, dual_arm
    return FrankaRobot(config.robot, gripper=gripper)


def build_robot(
    config: RootConfig,
    *,
    gripper: Gripper,
    reset_procedure: ResetProcedure,
    dual_arm: bool,
    expected_shared_sha256: str | None = None,
    strict_distributed_consistency: bool | None = True,
    factories: dict[str, RobotFactory] | None = None,
) -> Robot:
    """Build a robot; vendor code remains independent of cameras and teleoperation."""
    custom_builders = factories or {}
    builders: dict[str, RobotFactory] = {
        "ur5e": _ur_robot,
        "dual_ur": _ur_robot,
        "franka": _franka_robot,
        **custom_builders,
    }
    try:
        builder = builders[config.robot.kind]
    except KeyError as error:
        raise ValueError(f"Unknown robot kind {config.robot.kind!r}; provide a RobotFactory for it") from error
    kwargs = {
        "gripper": gripper,
        "reset_procedure": reset_procedure,
        "dual_arm": dual_arm,
    }
    if builder is _ur_robot:
        kwargs["expected_shared_sha256"] = expected_shared_sha256
        kwargs["strict_distributed_consistency"] = strict_distributed_consistency
    return builder(
        config,
        **kwargs,
    )
