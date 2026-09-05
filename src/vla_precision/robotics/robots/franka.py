"""Franka ZeroRPC/Polymetis client used by the established Stage-I path."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from vla_precision.config.schema import RobotConfig
from vla_precision.robotics.grippers.base import Gripper

LOGGER = logging.getLogger(__name__)


def _transform_to_pose(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec())
    )


def _euler_pose_to_rotvec(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    return np.concatenate((pose[:3], Rotation.from_euler("xyz", pose[3:]).as_rotvec()))


class FrankaZeroRPCClient:
    """Thin typed adapter over the existing local ``franka-start-server`` RPC."""

    def __init__(self, endpoint: str, *, client_factory: Callable[[], Any] | None = None):
        if client_factory is None:
            import zerorpc

            client_factory = lambda: zerorpc.Client(timeout=120, heartbeat=20)
        self.endpoint = endpoint
        self.server = client_factory()
        self.server.connect(endpoint)

    def robot_get_joint_positions(self) -> np.ndarray:
        return np.asarray(self.server.robot_get_joint_positions(), dtype=np.float64)

    def robot_get_joint_velocities(self) -> np.ndarray:
        return np.asarray(self.server.robot_get_joint_velocities(), dtype=np.float64)

    def robot_get_ee_pose(self) -> np.ndarray:
        return np.asarray(self.server.robot_get_ee_pose(), dtype=np.float64)

    def robot_get_ee_state(self) -> dict[str, np.ndarray]:
        state = self.server.robot_get_ee_state()
        return {
            "pose": np.asarray(state["pose"], dtype=np.float64),
            "speed": np.asarray(state["speed"], dtype=np.float64),
            "wrench": np.asarray(state["wrench"], dtype=np.float64),
        }

    def robot_move_to_ee_pose(
        self,
        pose: np.ndarray,
        time_to_go: float | None = None,
        delta: bool = False,
        Kx: np.ndarray | None = None,
        Kxd: np.ndarray | None = None,
        op_space_interp: bool = True,
        blocking: bool = True,
    ) -> bool:
        return bool(
            self.server.robot_move_to_ee_pose(
                np.asarray(pose, dtype=np.float64).tolist(),
                time_to_go,
                bool(delta),
                None if Kx is None else np.asarray(Kx, dtype=np.float64).tolist(),
                None if Kxd is None else np.asarray(Kxd, dtype=np.float64).tolist(),
                bool(op_space_interp),
                bool(blocking),
            )
        )

    def robot_start_cartesian_impedance_control(
        self,
        Kx: np.ndarray | None = None,
        Kxd: np.ndarray | None = None,
    ) -> None:
        self.server.robot_start_cartesian_impedance_control(
            None if Kx is None else np.asarray(Kx, dtype=np.float64).tolist(),
            None if Kxd is None else np.asarray(Kxd, dtype=np.float64).tolist(),
        )

    def robot_update_desired_ee_pose(self, pose: np.ndarray) -> int:
        return int(self.server.robot_update_desired_ee_pose(np.asarray(pose).tolist()))

    def robot_terminate_current_policy(self) -> None:
        self.server.robot_terminate_current_policy()

    def close(self) -> None:
        self.server.close()


class FrankaRobot:
    """Stream physical TCP deltas through the current Franka impedance controller.

    The server owns Polymetis and its error clipping/filter/nullspace settings.
    This client retains the reference TCP/base-frame math, fixed-control-frame
    selection vector, idle-pose latch and controller lifecycle.
    """

    action_low = -np.inf
    action_high = np.inf

    def __init__(
        self,
        config: RobotConfig,
        *,
        gripper: Gripper,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.gripper = gripper
        self.client = client or FrankaZeroRPCClient(
            config.server_url,
            client_factory=client_factory,
        )
        self.control_hz = float(config.control_hz)
        self.reset_pose = np.asarray(config.arms.left.reset_pose, dtype=np.float64)
        self.reset_pose_range = np.asarray(config.arms.left.reset_pose_range, dtype=np.float64)
        self.random_reset = bool(config.random_reset)
        self.options = config.options
        self._sleep = sleep
        self._clock = clock

        self.pre_reset_pose = self.options.get("pre_reset_pose")
        self.reference_frame = str(self.options.get("reference_frame", "tcp"))
        if self.reference_frame not in ("base", "tcp"):
            raise ValueError("Franka reference_frame must be 'base' or 'tcp'")
        self.reset_time_to_go = float(self.options.get("reset_time_to_go", 4.0))
        self.max_translation_step = float(self.options.get("max_translation_step", 0.05))
        self.max_rotation_step = float(self.options.get("max_rotation_step", 0.2))

        impedance = self.options.get("cartesian_impedance", {})
        self.cartesian_stiffness = np.asarray(
            impedance.get("stiffness", [2000, 2000, 2000, 250, 250, 250]),
            dtype=np.float64,
        )
        self.cartesian_damping = np.asarray(
            impedance.get("damping", [89, 89, 89, 9, 9, 9]),
            dtype=np.float64,
        )
        self.select_vector = np.asarray(
            impedance.get("select_vector", [1, 1, 1, 1, 1, 1]),
            dtype=np.float64,
        )
        self.control_frame_euler_deg = np.asarray(
            impedance.get("control_frame_euler_deg", [0.0, 0.0, -45.0]),
            dtype=np.float64,
        )
        self._selection_to_base_rotation = Rotation.from_euler(
            "xyz", self.control_frame_euler_deg, degrees=True
        ).as_matrix()
        self.idle_hold_enabled = bool(
            impedance.get("hold_current_pose_on_idle", config.idle_hold_enabled)
        )
        self.idle_position_threshold = float(
            impedance.get("translation_axis_deadband", config.idle_position_threshold)
        )
        self.idle_rotation_threshold = float(
            impedance.get("rotation_axis_deadband", config.idle_rotation_threshold)
        )
        self.server_config = {
            "control_frame_euler_deg": self.control_frame_euler_deg.tolist(),
            "select_vector": self.select_vector.tolist(),
        }
        self._idle_hold_pose: np.ndarray | None = None
        self._idle_previous_motion_mask = np.zeros(6, dtype=bool)

        joints = np.asarray(self.client.robot_get_joint_positions(), dtype=np.float64)
        pose = np.asarray(self.client.robot_get_ee_pose(), dtype=np.float64)
        if joints.shape != (7,) or pose.shape != (6,):
            raise RuntimeError(
                f"Franka server returned joints={joints.shape}, pose={pose.shape}; expected (7,) and (6,)"
            )
        self._set_current_pose(pose)
        self._terminate_controller_safely()
        self._start_cartesian_controller()

    @property
    def action_dimension(self) -> int:
        return 6 + self.gripper.action_dimension

    @property
    def currpos(self) -> np.ndarray:
        return np.concatenate(
            (self._current_pose[:3], Rotation.from_rotvec(self._current_pose[3:]).as_quat())
        )

    def _set_current_pose(self, pose: np.ndarray) -> None:
        self._current_pose = np.asarray(pose, dtype=np.float64).copy()

    def _clear_idle_hold(self) -> None:
        self._idle_hold_pose = None
        self._idle_previous_motion_mask = np.zeros(6, dtype=bool)

    def _terminate_controller_safely(self) -> None:
        try:
            self.client.robot_terminate_current_policy()
        except Exception:  # The server raises when no controller is active.
            LOGGER.debug("no active Franka policy to terminate", exc_info=True)

    def _start_cartesian_controller(self) -> None:
        self.client.robot_start_cartesian_impedance_control(
            self.cartesian_stiffness,
            self.cartesian_damping,
        )
        self._clear_idle_hold()

    def refresh_state(self) -> None:
        self._set_current_pose(self.client.robot_get_ee_pose())

    def observations(self) -> dict[str, np.ndarray]:
        raw = self.client.robot_get_ee_state()
        pose = np.asarray(raw["pose"], dtype=np.float64)
        speed = np.asarray(raw["speed"], dtype=np.float64)
        wrench = np.asarray(raw["wrench"], dtype=np.float64)
        self._set_current_pose(pose)
        state = {
            "tcp_pose": self.currpos,
            "tcp_vel": speed,
            "tcp_force": wrench[:3],
            "tcp_torque": wrench[3:6],
        }
        state.update(self.gripper.observations(raw))
        return state

    def _raw_target_pose_from_action(
        self,
        action: np.ndarray,
        current_pose: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        current_pose = np.asarray(current_pose, dtype=np.float64)
        if not np.all(np.isfinite(action[:6])):
            raise ValueError(f"Franka action contains non-finite values: {action[:6].tolist()}")
        translation_norm = np.linalg.norm(action[:3])
        delta_rotation_reference = Rotation.from_euler("xyz", action[3:6]).as_matrix()
        rotation_norm = np.linalg.norm(
            Rotation.from_matrix(delta_rotation_reference).as_rotvec()
        )
        if translation_norm > self.max_translation_step:
            raise ValueError(
                f"Translation action {translation_norm:.4f} m exceeds "
                f"max_translation_step={self.max_translation_step:.4f} m"
            )
        if rotation_norm > self.max_rotation_step:
            raise ValueError(
                f"Rotation action {rotation_norm:.4f} rad exceeds "
                f"max_rotation_step={self.max_rotation_step:.4f} rad"
            )

        current_rotation = Rotation.from_rotvec(current_pose[3:]).as_matrix()
        if self.reference_frame == "tcp":
            delta_position_base = current_rotation @ action[:3]
            delta_rotation_base = (
                current_rotation @ delta_rotation_reference @ current_rotation.T
            )
        elif self.reference_frame == "base":
            delta_position_base = action[:3]
            delta_rotation_base = delta_rotation_reference
        else:
            raise ValueError(f"Unsupported Franka reference_frame {self.reference_frame!r}")

        selection_to_base = self._selection_to_base_rotation
        selected_position = selection_to_base.T @ delta_position_base
        selected_position *= self.select_vector[:3]
        delta_position_base = selection_to_base @ selected_position

        selected_rotation = selection_to_base.T @ delta_rotation_base @ selection_to_base
        selected_rotvec = Rotation.from_matrix(selected_rotation).as_rotvec()
        selected_rotvec *= self.select_vector[3:]
        delta_rotation_base = (
            selection_to_base
            @ Rotation.from_rotvec(selected_rotvec).as_matrix()
            @ selection_to_base.T
        )

        target_transform = np.eye(4, dtype=np.float64)
        target_transform[:3, :3] = delta_rotation_base @ current_rotation
        target_transform[:3, 3] = current_pose[:3] + delta_position_base
        return _transform_to_pose(target_transform)

    def _motion_delta_mask(
        self,
        current_pose: np.ndarray,
        action_target_pose: np.ndarray,
    ) -> np.ndarray:
        position_mask = (
            np.abs(action_target_pose[:3] - current_pose[:3])
            >= self.idle_position_threshold
        )
        current_rotation = Rotation.from_rotvec(current_pose[3:])
        target_rotation = Rotation.from_rotvec(action_target_pose[3:])
        rotation_active = (
            np.linalg.norm((target_rotation * current_rotation.inv()).as_rotvec())
            >= self.idle_rotation_threshold
        )
        return np.concatenate((position_mask, np.repeat(rotation_active, 3)))

    def _merge_selected_orientation(
        self,
        held_pose: np.ndarray,
        moving_pose: np.ndarray,
    ) -> np.ndarray:
        selection_to_base = Rotation.from_matrix(self._selection_to_base_rotation)
        held_rotation = Rotation.from_rotvec(np.asarray(held_pose[3:], dtype=np.float64))
        moving_rotation = Rotation.from_rotvec(np.asarray(moving_pose[3:], dtype=np.float64))
        held_euler = (selection_to_base.inv() * held_rotation).as_euler("xyz")
        moving_euler = (selection_to_base.inv() * moving_rotation).as_euler("xyz")
        merged_euler = np.where(
            self.select_vector[3:].astype(bool),
            moving_euler,
            held_euler,
        )
        return (selection_to_base * Rotation.from_euler("xyz", merged_euler)).as_rotvec()

    def _target_pose_from_action(
        self,
        action: np.ndarray,
        current_pose: np.ndarray,
    ) -> np.ndarray:
        action_target_pose = self._raw_target_pose_from_action(action, current_pose)
        if not self.idle_hold_enabled:
            return action_target_pose
        if self._idle_hold_pose is None:
            self._idle_hold_pose = current_pose.copy()

        motion_mask = self._motion_delta_mask(current_pose, action_target_pose)
        for index, (was_moving, is_moving) in enumerate(
            zip(self._idle_previous_motion_mask[:3], motion_mask[:3])
        ):
            if was_moving and not is_moving:
                self._idle_hold_pose[index] = current_pose[index]

        orientation_was_moving = self._idle_previous_motion_mask[3]
        orientation_is_moving = motion_mask[3]
        if orientation_was_moving and not orientation_is_moving:
            self._idle_hold_pose[3:] = self._merge_selected_orientation(
                self._idle_hold_pose,
                current_pose,
            )

        if not np.any(motion_mask):
            self._idle_previous_motion_mask = motion_mask
            return self._idle_hold_pose.copy()

        target_pose = self._idle_hold_pose.copy()
        for index, is_moving in enumerate(motion_mask[:3]):
            if is_moving:
                target_pose[index] = action_target_pose[index]
                self._idle_hold_pose[index] = current_pose[index]

        if orientation_is_moving:
            target_pose[3:] = self._merge_selected_orientation(
                self._idle_hold_pose,
                action_target_pose,
            )
            self._idle_hold_pose[3:] = self._merge_selected_orientation(
                self._idle_hold_pose,
                current_pose,
            )
        self._idle_previous_motion_mask = motion_mask
        return target_pose

    def execute_action_chunk(self, action: np.ndarray) -> np.ndarray:
        chunk = np.asarray(action, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None]
        if chunk.ndim != 2 or chunk.shape[-1] != self.action_dimension:
            raise ValueError(
                f"Franka action must have shape (T, {self.action_dimension}), got {chunk.shape}"
            )
        executed = []
        for row in chunk:
            started = self._clock()
            current_pose = np.asarray(self.client.robot_get_ee_pose(), dtype=np.float64)
            target_pose = self._target_pose_from_action(row, current_pose)
            self.client.robot_update_desired_ee_pose(target_pose)
            if self.gripper.action_dimension:
                self.gripper.command_chunk(row[6:7])
            executed.append(row.copy())
            self._sleep(max(0.0, 1.0 / self.control_hz - (self._clock() - started)))
        return np.asarray(executed, dtype=np.float32)

    def _sample_reset_pose(self) -> np.ndarray:
        target = _euler_pose_to_rotvec(self.reset_pose)
        if not self.random_reset:
            return target
        random_range = np.abs(self.reset_pose_range)
        if random_range.shape == (3,):
            random_range = np.concatenate((random_range, [0.0, 0.0, 20.0]))
        target[:3] += np.random.uniform(-random_range[:3], random_range[:3])
        target_euler_deg = Rotation.from_rotvec(target[3:]).as_euler("xyz", degrees=True)
        target_euler_deg += np.random.uniform(-random_range[3:], random_range[3:])
        target[3:] = Rotation.from_euler("xyz", target_euler_deg, degrees=True).as_rotvec()
        return target

    def _move_to_tcp_pose(self, pose: np.ndarray) -> None:
        self._clear_idle_hold()
        self._terminate_controller_safely()
        completed = self.client.robot_move_to_ee_pose(
            np.asarray(pose, dtype=np.float64),
            time_to_go=self.reset_time_to_go,
            blocking=True,
        )
        if not completed:
            raise RuntimeError("Franka reset trajectory did not complete")

    def reset(
        self,
        *,
        joint_reset: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        del joint_reset, options
        prepare_reset = getattr(self.gripper, "prepare_reset", None)
        if prepare_reset is not None:
            prepare_reset()
        self._clear_idle_hold()
        try:
            if self.pre_reset_pose is not None:
                self._move_to_tcp_pose(
                    _euler_pose_to_rotvec(np.asarray(self.pre_reset_pose, dtype=np.float64))
                )
            self._move_to_tcp_pose(self._sample_reset_pose())
        finally:
            self._terminate_controller_safely()
            self._start_cartesian_controller()
            current_pose = np.asarray(self.client.robot_get_ee_pose(), dtype=np.float64)
            self.client.robot_update_desired_ee_pose(current_pose)
            self._set_current_pose(current_pose)
        return self.observations()

    def request(self, name: str, enabled: bool) -> Any:
        raise NotImplementedError(
            f"Franka ZeroRPC has no UR HTTP request {name!r}={enabled!r}"
        )

    def close(self) -> None:
        try:
            self._terminate_controller_safely()
        finally:
            try:
                self.gripper.close()
            finally:
                self.client.close()
