"""Construct interchangeable grippers behind the shared capability interface."""

from __future__ import annotations

from collections.abc import Callable

from vla_precision.config.schema import RootConfig
from vla_precision.robotics.grippers.base import Gripper
from vla_precision.robotics.grippers.franka import FrankaPGIGripper
from vla_precision.robotics.grippers.ur import FixedGripper, URGripper

GripperFactory = Callable[..., Gripper]


def _ur_gripper(
    config: RootConfig,
    *,
    dual_arm: bool,
    state_fields: tuple[str, ...],
) -> Gripper:
    del state_fields
    gripper = config.gripper
    return URGripper(
        server_url=gripper.server_url or config.robot.server_url,
        dual_arm=dual_arm,
        left_start_position=int(
            gripper.start_position if gripper.left_start_position is None else gripper.left_start_position
        ),
        right_start_position=int(
            gripper.start_position if gripper.right_start_position is None else gripper.right_start_position
        ),
    )


def _franka_gripper(
    config: RootConfig,
    *,
    dual_arm: bool,
    state_fields: tuple[str, ...],
) -> Gripper:
    del dual_arm, state_fields
    options = config.gripper.options
    return FrankaPGIGripper(
        port=config.gripper.device or "/dev/franka_pgi_gripper",
        reverse=bool(options.get("reverse", False)),
        initialize=bool(options.get("initialize", True)),
        close_threshold=float(options.get("close_threshold", 0.7)),
        force=int(options.get("force", 100)),
        speed=int(options.get("speed", 100)),
        start_position=config.gripper.start_position,
    )


def build_gripper(
    config: RootConfig,
    *,
    fixed: bool,
    dual_arm: bool,
    state_fields: tuple[str, ...],
    factories: dict[str, GripperFactory] | None = None,
) -> Gripper:
    """Build a gripper; a new vendor only needs a factory returning ``Gripper``."""
    if fixed:
        return FixedGripper()

    builders: dict[str, GripperFactory] = {
        "pgi": _ur_gripper,
        "franka_pgi": _franka_gripper,
        **(factories or {}),
    }
    kind = config.gripper.kind
    # ``pgi`` is robot-hosted for UR and serial for the established Franka setup.
    if config.robot.kind == "franka" and kind == "pgi":
        kind = "franka_pgi"
    try:
        builder = builders[kind]
    except KeyError as error:
        raise ValueError(f"Unknown gripper kind {kind!r}; provide a GripperFactory for it") from error
    return builder(config, dual_arm=dual_arm, state_fields=state_fields)
