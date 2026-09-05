"""One environment factory shared by actor, learner and the robot-side service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import gymnasium as gym

from vla_precision.config import ResolvedConfig
from vla_precision.robotics.cameras import build_cameras
from vla_precision.robotics.environments.fake import FakeCamera, FakeRobot
from vla_precision.robotics.environments.robot import RobotEnvironment
from vla_precision.robotics.grippers import GripperFactory, build_gripper
from vla_precision.robotics.robots import RobotFactory, build_robot
from vla_precision.robotics.tasks.completion import build_completion_detector
from vla_precision.robotics.tasks.reset import build_reset_procedure
from vla_precision.robotics.tasks.reward import build_reward_function
from vla_precision.robotics.teleoperation import KeyboardEmergencyStopDetector
from vla_precision.robotics.wrappers import (
    ActionChunkWrapper,
    CompletionRewardWrapper,
    FlattenObservationWrapper,
    KeyboardIntervention,
    QuaternionToEulerWrapper,
    RegraspResetWrapper,
    RelativeFrameWrapper,
)


class EnvironmentFactory(Protocol):
    def __call__(self, resolved: ResolvedConfig, *, fake_env: bool = False) -> gym.Env: ...


def _is_fixed_gripper(setup_mode: str) -> bool:
    return setup_mode.endswith("fixed-gripper")


def _start_position(side: float | None, default: float) -> int:
    return int(default if side is None else side)


def build_environment(
    resolved: ResolvedConfig,
    *,
    fake_env: bool = False,
    completion_detection: bool | None = None,
    teleoperation_enabled: bool = True,
    camera_factories=None,
    gripper_factories: dict[str, GripperFactory] | None = None,
    robot_factories: dict[str, RobotFactory] | None = None,
    emergency_stop_factory: Callable[[], object] | None = KeyboardEmergencyStopDetector,
    teleoperation_factories: dict[str, Callable] | None = None,
) -> gym.Env:
    config = resolved.config
    dual_arm = config.task.arm_mode == "dual"
    fixed_gripper = _is_fixed_gripper(config.task.setup_mode)
    gripper_action_dimension = 0 if fixed_gripper else (2 if dual_arm else 1)

    if fake_env:
        action_dimension = 14 if dual_arm else 6 + gripper_action_dimension
        robot = FakeRobot(
            action_dimension=action_dimension,
            proprio_keys=config.task.proprio_keys,
            gripper_action_dimension=gripper_action_dimension,
            gripper_kind=config.gripper.kind,
        )
        cameras = {key: FakeCamera(key) for key in config.task.image_keys}
        camera_builder = None
        emergency_stop = None
    else:
        gripper_fields = tuple(
            dict.fromkeys(
                key.split("/", 1)[-1]
                for key in config.task.proprio_keys
                if key.split("/", 1)[-1].startswith("gripper_")
            )
        )
        gripper = build_gripper(
            config,
            fixed=fixed_gripper,
            dual_arm=dual_arm,
            state_fields=gripper_fields,
            factories=gripper_factories,
        )
        robot = build_robot(
            config,
            gripper=gripper,
            reset_procedure=build_reset_procedure(config.task.reset_procedure),
            dual_arm=dual_arm,
            expected_shared_sha256=resolved.shared_sha256,
            strict_distributed_consistency=config.strict_distributed_consistency,
            factories=robot_factories,
        )
        camera_builder = lambda: build_cameras(config.cameras, factories=camera_factories)
        cameras = camera_builder()
        emergency_stop = emergency_stop_factory() if emergency_stop_factory is not None else None

    environment = RobotEnvironment(
        robot=robot,
        cameras=cameras,
        camera_builder=camera_builder,
        proprio_keys=config.task.proprio_keys,
        image_keys=config.task.image_keys,
        camera_options={name: device.options for name, device in config.cameras.devices.items()},
        max_episode_length=config.task.max_episode_length,
        transition_log_interval=config.logging.transition_log_interval,
        reconnect_delay=float(config.robot.options.get("camera_reconnect_delay", 2.0)),
        display_images=config.cameras.display_image and not fake_env,
        save_video=config.actor.environment_save_video and not fake_env,
        video_directory=Path(config.paths.output_root) / "videos",
        emergency_stop=emergency_stop,
    )
    action_reference_frame = (
        str(config.robot.options.get("reference_frame", "tcp"))
        if config.robot.kind == "franka"
        else "tcp"
    )
    env: gym.Env = RelativeFrameWrapper(
        environment,
        include_relative_pose=action_reference_frame == "tcp",
        actions_in_tcp_frame=action_reference_frame == "tcp",
        dual_arm=dual_arm,
    )

    completion_event = None
    if (
        not fake_env
        and teleoperation_enabled
        and config.teleoperation.kind not in ("none", "disabled")
    ):
        import threading

        completion_event = threading.Event() if dual_arm or config.teleoperation.kind != "keyboard" else None
        expert = None
        if config.teleoperation.kind != "keyboard":
            factories = teleoperation_factories or {}
            if config.teleoperation.kind not in factories:
                raise ValueError(
                    f"Teleoperation kind {config.teleoperation.kind!r} needs an extension implementing TeleoperationDevice"
                )
            expert = factories[config.teleoperation.kind](
                config.teleoperation,
                dual_arm,
                completion_event,
            )
        intervention = KeyboardIntervention(
            env,
            step_size_pos=config.teleoperation.step_size_position,
            step_size_rot=config.teleoperation.step_size_rotation,
            step_size_pos_alt=config.teleoperation.step_size_position_alt,
            step_size_rot_alt=config.teleoperation.step_size_rotation_alt,
            dual_arm=dual_arm,
            left_keyboard_path=config.teleoperation.left_device,
            right_keyboard_path=config.teleoperation.right_device,
            left_step_size_pos=config.teleoperation.options.get("left_step_size_position"),
            left_step_size_rot=config.teleoperation.options.get("left_step_size_rotation"),
            left_step_size_pos_alt=config.teleoperation.options.get("left_step_size_position_alt"),
            left_step_size_rot_alt=config.teleoperation.options.get("left_step_size_rotation_alt"),
            right_step_size_pos=config.teleoperation.options.get("right_step_size_position"),
            right_step_size_rot=config.teleoperation.options.get("right_step_size_rotation"),
            right_step_size_pos_alt=config.teleoperation.options.get("right_step_size_position_alt"),
            right_step_size_rot_alt=config.teleoperation.options.get("right_step_size_rotation_alt"),
            reward_keyboard_arm=str(config.teleoperation.options.get("reward_keyboard_arm", "left")),
            completion_double_press_interval=config.teleoperation.completion_double_press_interval,
            left_gripper_start_position=_start_position(
                config.gripper.left_start_position, config.gripper.start_position
            ),
            right_gripper_start_position=_start_position(
                config.gripper.right_start_position, config.gripper.start_position
            ),
            expert=expert,
            completion_event=completion_event,
        )
        completion_event = intervention.completion_event
        env = intervention

    env = QuaternionToEulerWrapper(env, dual_arm=dual_arm)
    env = FlattenObservationWrapper(env, proprio_keys=config.task.proprio_keys)
    env = ActionChunkWrapper(env, action_horizon=config.task.action_horizon)

    completion_detection = (
        config.actor.completion_detection_enabled
        if completion_detection is None
        else bool(completion_detection)
    )
    if completion_detection:
        detector = build_completion_detector(
            config.task.completion_detector,
            event=completion_event,
            double_press_interval=config.teleoperation.completion_double_press_interval,
            keyboard_enabled=not fake_env,
        )
        env = CompletionRewardWrapper(
            env,
            detector=detector,
            reward=build_reward_function(
                config.task.reward_function,
                time_reward=config.task.time_reward,
                completion_reward=config.task.completion_reward,
            ),
        )
    if not fake_env and config.task.regrasp.enabled:
        env = RegraspResetWrapper(
            env,
            regrasp_step_1=config.task.regrasp.step_1 or None,
            regrasp_step_2=config.task.regrasp.step_2 or None,
            wait_before=config.task.regrasp.wait_before,
            wait_after=config.task.regrasp.wait_after,
        )
    return env
