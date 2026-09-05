#!/usr/bin/env python3
"""Standalone Franka inference entrypoint for OpenPI + VLA-Precision checkpoints.

This file is a self-contained core inference entry. Robot/camera/control logic
matches ``ur.py``; only the robot backend is replaced by the
Franka ZeroRPC/Polymetis service used by ``lerobot_franka_teleop``.
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
import argparse
import dataclasses
import yaml
import os
import time
import threading
import select
import sys
import termios
import tty
import numpy as np
import serial
import crcmod
import zerorpc
from pathlib import Path

from pyDHgripper import PGE
from typing import Dict, Any
from .utils import FpsCounter, validate_single_arm_norm_stats
from vla_precision import image_tools
from scipy.spatial.transform import Rotation as R
from .recording import Recorder
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs
def update_latest_symlink(target: Path, link_name: Path):
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    os.symlink(target, link_name)

def resolve_model_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path

def is_none_config(value) -> bool:
    return value is None or (isinstance(value, str) and value.lower() in ("none", "null"))


class FrankaZeroRPCClient:
    """Small local client for the existing Franka teleoperation server."""

    def __init__(self, host: str, port: int):
        self.endpoint = f"tcp://{host}:{int(port)}"
        self.server = zerorpc.Client(timeout=120, heartbeat=20)
        self.server.connect(self.endpoint)

    def robot_get_joint_positions(self) -> np.ndarray:
        return np.asarray(self.server.robot_get_joint_positions(), dtype=float)

    def robot_get_joint_velocities(self) -> np.ndarray:
        return np.asarray(self.server.robot_get_joint_velocities(), dtype=float)

    def robot_get_ee_pose(self) -> np.ndarray:
        return np.asarray(self.server.robot_get_ee_pose(), dtype=float)

    def robot_get_ee_state(self) -> Dict[str, np.ndarray]:
        state = self.server.robot_get_ee_state()
        return {
            "pose": np.asarray(state["pose"], dtype=float),
            "speed": np.asarray(state["speed"], dtype=float),
            "wrench": np.asarray(state["wrench"], dtype=float),
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
        return bool(self.server.robot_move_to_ee_pose(
            np.asarray(pose, dtype=float).tolist(),
            time_to_go,
            bool(delta),
            np.asarray(Kx, dtype=float).tolist() if Kx is not None else None,
            np.asarray(Kxd, dtype=float).tolist() if Kxd is not None else None,
            bool(op_space_interp),
            bool(blocking),
        ))

    def robot_start_cartesian_impedance_control(
        self,
        Kx: np.ndarray | None = None,
        Kxd: np.ndarray | None = None,
    ) -> None:
        self.server.robot_start_cartesian_impedance_control(
            np.asarray(Kx, dtype=float).tolist() if Kx is not None else None,
            np.asarray(Kxd, dtype=float).tolist() if Kxd is not None else None,
        )

    def robot_update_desired_ee_pose(
        self,
        pose: np.ndarray,
    ) -> int:
        return int(
            self.server.robot_update_desired_ee_pose(
                np.asarray(pose, dtype=float).tolist(),
            )
        )

    def robot_terminate_current_policy(self) -> None:
        self.server.robot_terminate_current_policy()

    def close(self) -> None:
        self.server.close()


class Inference:
    @staticmethod
    def _resolve_model_path(value):
        return resolve_model_path(value)

    @classmethod
    def _resolve_actor_checkpoint_dir(cls, value):
        path = cls._resolve_model_path(value)
        if path is None:
            return None
        if (path / "params").exists():
            return path
        if not path.exists():
            return path

        candidates = []
        for child in path.iterdir():
            if child.is_dir() and child.name.isdigit() and (child / "params").exists():
                candidates.append((int(child.name), child))
        if not candidates:
            return path
        step, latest = max(candidates, key=lambda item: item[0])
        logging.info("[MODEL] Resolved actor checkpoint root %s to latest step %s: %s", path, step, latest)
        return latest

    def __init__(self, config_path: Path):
        # Load YAML config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Model config. Keep OpenPI's TrainConfig/policy path, but load the
        # VLA-Precision actor checkpoint saved by core/VLA-Precision training.py.
        model = cfg["model"]
        robot = cfg["robot"]
        self.remote_policy = cfg.get("policy", {}).get("location", "local") == "server"
        self.action_horizon = int(robot["action_horizon"])
        if self.action_horizon <= 0:
            raise ValueError("robot.action_horizon must be positive.")
        self.direct_action_horizon = bool(robot.get("direct_action_horizon", True))
        self.model_config = None
        if not self.remote_policy:
            from vla_precision.integrations.openpi import configs as _config

            self.model_config = _config.get_config(model["name"])
            if self.direct_action_horizon and int(self.model_config.model.action_horizon) != self.action_horizon:
                self.model_config = dataclasses.replace(
                    self.model_config,
                    model=dataclasses.replace(self.model_config.model, action_horizon=self.action_horizon),
                )
        checkpoint_root = Path(model["checkpoint_dir"]).expanduser()
        checkpoint_step = model.get("checkpoint_step")
        if checkpoint_step not in (None, 0, "0") and checkpoint_root.name != str(checkpoint_step):
            checkpoint_root = checkpoint_root / str(checkpoint_step)
        self.checkpoint_dir = self._resolve_actor_checkpoint_dir(checkpoint_root)
        self.sample_steps = int(model.get("sample_steps", model.get("pi_sample_steps", 10)))
        self.norm_stats_path: Path | None = None
        self.norm_stats = None
        if not self.remote_policy:
            self._configure_lora_checkpoint_norm_stats()

        # Camera config
        cam = cfg["cameras"]
        self.wrist_cam_serial = cam["wrist_cam_serial"]
        self.exterior_cam_serial = cam["exterior_cam_serial"]
        self.cam_fps = cam.get("fps", 30)
        self.cam_width = cam.get("width", 640)
        self.cam_height = cam.get("height", 480)

        # Video config
        video = cfg["video"]
        self.video_fps = video.get("fps", 7)
        self.visualize = video["visualize"]
        record = cfg.get("record", {})
        self.num_episodes = record.get("num_episodes", 10)
        self.ep_timeout = record.get("episode_timeout_sec", 30)
        self.show_action_fps = record.get("show_action_fps", False)
        self.show_inference_fps = record.get("show_inference_fps", False)
        self.show_time = record.get("show_time", False)

        # Robot config
        self.server_host = robot.get("server_host", "127.0.0.1")
        self.server_port = int(robot.get("server_port", 4242))
        self.gripper_port = robot["gripper_port"]
        self.gripper_reverse = robot["gripper_reverse"]
        self.init_gripper = robot.get("init_gripper", True)
        self.fix_gripper = bool(robot.get("fix_gripper", False))
        if not self.remote_policy:
            validate_single_arm_norm_stats(
                self.norm_stats,
                fixed_gripper=self.fix_gripper,
                learned_state_dim=19,
                path=self.norm_stats_path,
            )
        self.close_threshold = robot.get("close_threshold", 0.9)
        self.target_reached_gripper_open_threshold = robot.get(
            "target_reached_gripper_open_threshold",
            self.close_threshold,
        )
        self._gripper_force = robot.get("gripper_force", 70)
        self._gripper_speed = robot.get("gripper_speed", 60)
        self.pre_reset_tcp_pose = robot.get("pre_reset_tcp_pose")
        self.init_tcp_pose = robot["init_tcp_pose"]
        self.init_pose_range = robot.get(
            "init_pose_range",
            robot.get("init_pos_range", [0.03, 0.03, 0.03, 0.0, 0.0, 20.0]),
        )
        self.target_tcp_pose = robot.get("target_tcp_pose")
        self.target_threshold = robot.get("target_threshold")
        has_target_pose = not is_none_config(self.target_tcp_pose)
        has_target_threshold = not is_none_config(self.target_threshold)
        if has_target_pose != has_target_threshold:
            raise ValueError(
                "robot.target_tcp_pose and robot.target_threshold must both be set or both be null."
            )
        self.use_target_success = has_target_pose and has_target_threshold
        self.show_target_error = robot.get("show_target_error", False)
        self.action_fps = float(robot["action_fps"])
        if not np.isfinite(self.action_fps) or self.action_fps <= 0.0:
            raise ValueError("robot.action_fps must be finite and positive.")
        self.debug = robot.get("debug", False)
        self.control_space = robot.get("control_space", "cartesian_impedance")
        if self.control_space != "cartesian_impedance":
            raise ValueError(
                "Franka inference only supports Cartesian impedance target streaming; "
                "set robot.control_space to cartesian_impedance."
            )
        self.reference_frame = robot.get("reference_frame", "tcp")
        if self.reference_frame not in ("base", "tcp"):
            raise ValueError(
                f"robot.reference_frame must be 'base' or 'tcp', got {self.reference_frame!r}."
            )
        self.reset_time_to_go = float(robot.get("reset_time_to_go", 4.0))
        self.max_translation_step = float(robot.get("max_translation_step", 0.05))
        self.max_rotation_step = float(robot.get("max_rotation_step", 0.2))
        if not np.isfinite(self.max_translation_step) or self.max_translation_step <= 0.0:
            raise ValueError("robot.max_translation_step must be finite and positive.")
        if not np.isfinite(self.max_rotation_step) or self.max_rotation_step <= 0.0:
            raise ValueError("robot.max_rotation_step must be finite and positive.")

        impedance_cfg = robot.get("cartesian_impedance", {})
        self.cartesian_stiffness = np.asarray(
            impedance_cfg.get("stiffness", [2000, 2000, 2000, 250, 250, 250]),
            dtype=float,
        )
        self.cartesian_damping = np.asarray(
            impedance_cfg.get("damping", [89, 89, 89, 9, 9, 9]),
            dtype=float,
        )
        if self.cartesian_stiffness.shape != (6,) or self.cartesian_damping.shape != (6,):
            raise ValueError("Franka Cartesian stiffness and damping must each contain 6 values.")
        if not np.all(np.isfinite(self.cartesian_stiffness)) or not np.all(
            np.isfinite(self.cartesian_damping)
        ):
            raise ValueError("Franka Cartesian stiffness and damping must be finite.")
        self.select_vector = np.asarray(
            impedance_cfg.get("select_vector", [1, 1, 1, 1, 1, 1]),
            dtype=float,
        )
        if self.select_vector.shape != (6,):
            raise ValueError("robot.cartesian_impedance.select_vector must contain 6 values.")
        if not np.all(np.isin(self.select_vector, (0.0, 1.0))):
            raise ValueError("Franka select_vector values must be 0 or 1.")
        self.control_frame_euler_deg = np.asarray(
            impedance_cfg.get("control_frame_euler_deg", [0.0, 0.0, -45.0]),
            dtype=float,
        )
        if self.control_frame_euler_deg.shape != (3,):
            raise ValueError(
                "robot.cartesian_impedance.control_frame_euler_deg must contain 3 values."
            )
        if not np.all(np.isfinite(self.control_frame_euler_deg)):
            raise ValueError(
                "robot.cartesian_impedance.control_frame_euler_deg must be finite."
            )
        self._selection_to_base_rotation = R.from_euler(
            "xyz", self.control_frame_euler_deg, degrees=True
        ).as_matrix()
        self.action_idle_hold_enabled = bool(
            impedance_cfg.get(
                "hold_current_pose_on_idle",
                robot.get("action_idle_hold_enabled", True),
            )
        )
        self.action_idle_position_threshold = float(
            impedance_cfg.get(
                "translation_axis_deadband",
                robot.get("action_idle_position_threshold", 0.0005),
            )
        )
        self.action_idle_rotation_threshold = float(
            impedance_cfg.get(
                "rotation_axis_deadband",
                robot.get("action_idle_rotation_threshold", 0.0005),
            )
        )
        if (
            not np.isfinite(self.action_idle_position_threshold)
            or not 0.0 < self.action_idle_position_threshold < self.max_translation_step
        ):
            raise ValueError(
                "robot.cartesian_impedance.translation_axis_deadband must be finite, "
                "positive, and smaller than robot.max_translation_step."
            )
        if (
            not np.isfinite(self.action_idle_rotation_threshold)
            or not 0.0 < self.action_idle_rotation_threshold < self.max_rotation_step
        ):
            raise ValueError(
                "robot.cartesian_impedance.rotation_axis_deadband must be finite, "
                "positive, and smaller than robot.max_rotation_step."
            )
        # Task config
        task = cfg["task"]
        self.task_description = task["description"]

        # time stamps
        time_str = time.strftime('%Y%m%d-%H%M%S')
        time_path = time.strftime('%Y%m%d')

        # base dir
        base_dir = Path(cfg.get("output_root", "./results")).expanduser() / cfg["experiment"]["name"] / "openpi-native"
        log_dir = base_dir / "logs"
        video_dir = base_dir / "videos" / time_path

        # create dir
        (log_dir / "all_logs").mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        # log paths
        latest_path = log_dir / "latest.yaml"
        log_path = log_dir / "all_logs" / f"log_{time_str}.yaml"

        # video paths
        wrist_video = video_dir / f"wrist_{time_str}.mp4"
        exterior_video = video_dir / f"exterior_{time_str}.mp4"

        # Recorder
        self.recorder = Recorder(log_path=log_path, video_path=[wrist_video, exterior_video], display_fps=self.video_fps, visualize=self.visualize)

        # create symlink to latest log
        update_latest_symlink(log_path, latest_path)

        # create FPS counters
        self.fps_action = FpsCounter(name="action")

        # Internal states
        self.robot = None
        self.cameras = None
        self._last_gripper_position = 1
        self._gripper_position = 1
        self._last_servoj_ts = None
        self._action_idle_hold_pose = None
        self._action_idle_prev_motion_mask = np.zeros(6, dtype=bool)
        self._episode_reference_ee_pose = None
        self._ep_start = None
        self._ep_idx = 1
        self._ep_steps = 0
        self._ep_done = False
        self._success_episodes = 0
        self._failed_episodes = 0
        self._reset_requested = threading.Event()
        self._success_requested = threading.Event()
        self._keyboard_listener_stop = threading.Event()
        self._keyboard_listener_thread = None
        self._episode_time_printer_stop = None
        self._episode_time_printer_thread = None

    # --------------------------- ROBOT --------------------------- #
    def connect_robot(self):
        """Connect to the already-running local Franka ZeroRPC server."""
        try:
            logging.info("\n===== [ROBOT] Connecting to Franka ZeroRPC server =====")
            self.robot = FrankaZeroRPCClient(self.server_host, self.server_port)
            logging.info("[ROBOT] Connected to %s", self.robot.endpoint)

            # Joint positions
            joints = self.robot.robot_get_joint_positions()
            if joints.shape == (7,):
                formatted = np.round(joints, 4).tolist()
                logging.info(f"[ROBOT] Current joint positions: {formatted}")
            else:
                raise RuntimeError(f"Expected 7 Franka joints, got {joints.shape}.")

            # The server exposes the configured Franka EE/TCP as xyz + rotation vector.
            tcp_pose = self.robot.robot_get_ee_pose()
            if tcp_pose.shape == (6,):
                formatted_pose = np.round(tcp_pose, 4).tolist()
                logging.info(f"[ROBOT] Current TCP pose: {formatted_pose}")
            else:
                raise RuntimeError(f"Expected a 6D Franka TCP pose, got {tcp_pose.shape}.")

            if not self.debug:
                self._terminate_controller_safely()
                self._start_cartesian_controller()
            logging.info(
                "[ROBOT] Cartesian stiffness=%s, damping=%s, select_vector=%s",
                self.cartesian_stiffness.tolist(),
                self.cartesian_damping.tolist(),
                self.select_vector.tolist(),
            )
            logging.info("===== [ROBOT] Franka initialized successfully =====\n")

        except Exception as e:
            logging.error("===== [ERROR] Failed to connect to Franka ZeroRPC server =====")
            logging.error(f"Exception: {e}\n")
            raise

    def _start_cartesian_controller(self):
        if self.robot is None or self.debug:
            return
        self.robot.robot_start_cartesian_impedance_control(
            self.cartesian_stiffness,
            self.cartesian_damping,
        )
        self._clear_action_idle_hold()

    def _terminate_controller_safely(self):
        if self.robot is None or self.debug:
            return
        try:
            self.robot.robot_terminate_current_policy()
        except Exception as error:
            logging.debug("No active Franka policy to terminate: %s", error)
    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        """Initialize and connect RealSense cameras."""
        try:
            logging.info("\n===== [CAM] Initializing cameras =====")

            wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.wrist_cam_serial,
                fps=self.cam_fps,
                width=self.cam_width,
                height=self.cam_height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            exterior_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.exterior_cam_serial,
                fps=self.cam_fps,
                width=self.cam_width,
                height=self.cam_height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            camera_config = {"wrist_image": wrist_cfg, "exterior_image": exterior_cfg}
            self.cameras = make_cameras_from_configs(camera_config)

            for name, cam in self.cameras.items():
                cam.connect()
                logging.info(f"[CAM] {name} connected successfully.")

            logging.info("===== [CAM] Cameras initialized successfully =====\n")

        except Exception as e:
            logging.error("[ERROR] Failed to initialize cameras.")
            logging.error(f"Exception: {e}\n")
            self.cameras = None

    # --------------------------- GRIPPER --------------------------- #
    def _create_gripper(self):
        if self.init_gripper:
            return PGE(port=self.gripper_port)

        gripper = PGE.__new__(PGE)
        gripper.ser = serial.Serial(port=self.gripper_port, baudrate=115200)
        gripper.crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)
        return gripper

    def connect_gripper(self):
        """Initialize and connect the DH gripper."""
        if self.fix_gripper:
            self.gripper = None
            logging.info("[GRIPPER] fix_gripper enabled; skipping gripper connection.")
            return
        try:
            logging.info("\n===== [GRIPPER] Initializing DH Gripper =====")
            self.gripper = self._create_gripper()
            if self.init_gripper:
                self.gripper.init_feedback()
            else:
                logging.info("[GRIPPER] Skipped init_state and init_feedback by config.")
            self.gripper.set_force(self._gripper_force)
            self.gripper.set_vel(self._gripper_speed)
            logging.info(f"[GRIPPER] Force: {self._gripper_force}, speed: {self._gripper_speed}")
            # Start gripper state reader
            self._start_gripper_state_reader()
            logging.info("[GRIPPER] DH Gripper initialized successfully.")

        except Exception as e:
            logging.error("[ERROR] Failed to initialize DH Gripper.")
            logging.error(f"Exception: {e}\n")
            self.gripper = None

    # --------------------------- GRIPPER THREAD --------------------------- #
    def _start_gripper_state_reader(self):
        threading.Thread(target=self._read_gripper_state, daemon=True).start()

    # --------------------------- GRIPPER WAIT --------------------------- #
    def wait_for_gripper_states(self):
        if self.fix_gripper:
            return
        if hasattr(self.gripper, 'position'):
            while self.gripper.position is None:
                logging.info("[GRIPPER] Waiting for gripper state to be obtained...")
                time.sleep(0.1)
        else:
            while not hasattr(self.gripper, 'position'):
                logging.info("[GRIPPER] Waiting for gripper position to be set...")
                time.sleep(0.1)

    # --------------------------- GRIPPER STATE --------------------------- #
    def _read_gripper_state(self):
        self.gripper.position = None
        while True:
            gripper_position = 0.0 if self._gripper_position <= self.close_threshold else 1.0

            if self.gripper_reverse:
                gripper_position = 1 - gripper_position

            if gripper_position != self._last_gripper_position:
                self.gripper.set_pos(val=int(1000 * gripper_position), blocking=False)
                self._last_gripper_position = gripper_position

            gripper_pos = self.gripper.read_pos() / 1000.0
            if self.gripper_reverse:
                gripper_pos = 1 - gripper_pos

            self.gripper.position = gripper_pos
            time.sleep(0.01)

    def _open_gripper_before_reset(self):
        if not self.init_gripper or getattr(self, "gripper", None) is None:
            return
        self._gripper_position = 1.0
        self._last_gripper_position = None
        logging.info("[GRIPPER] init_gripper enabled; opening gripper before episode reset.")

    # --------------------------- TCP POSE UTILS --------------------------- #
    def _pose_to_transform(self, pose: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
        transform[:3, 3] = pose[:3]
        return transform

    def _transform_to_pose(self, transform: np.ndarray) -> np.ndarray:
        return np.concatenate((
            transform[:3, 3],
            R.from_matrix(transform[:3, :3]).as_rotvec(),
        ))

    def get_ee_pose(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("Franka ZeroRPC client is not connected.")
        return self.robot.robot_get_ee_pose()

    def _pose_euler(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=float)
        return np.concatenate((pose[:3], R.from_rotvec(pose[3:]).as_euler("xyz")))

    def _relative_pose_euler(self, pose: np.ndarray) -> np.ndarray:
        if self._episode_reference_ee_pose is None:
            raise RuntimeError("Episode reference EE pose is not set.")
        reference_transform = self._pose_to_transform(self._episode_reference_ee_pose)
        current_transform = self._pose_to_transform(np.asarray(pose, dtype=float))
        relative_transform = np.linalg.inv(reference_transform) @ current_transform
        return np.concatenate((
            relative_transform[:3, 3],
            R.from_matrix(relative_transform[:3, :3]).as_euler("xyz"),
        ))

    def set_episode_reference_pose(self):
        self._episode_reference_ee_pose = self.get_ee_pose()
        logging.info(f"[STATE] Set episode reference EE pose: {np.round(self._episode_reference_ee_pose, 6).tolist()}")

    def _validate_tcp_pose(self, name: str, pose: list[float] | np.ndarray) -> np.ndarray:
        tcp_pose = np.asarray(pose, dtype=float)
        if tcp_pose.shape != (6,):
            raise ValueError(f"robot.{name} must contain 6 values, got {tcp_pose.shape}.")
        tcp_pose = tcp_pose.copy()
        tcp_pose[3:] = R.from_euler("xyz", tcp_pose[3:]).as_rotvec()
        return tcp_pose

    def _is_none_pose(self, pose) -> bool:
        return is_none_config(pose)

    def _sample_init_tcp_pose(self) -> np.ndarray:
        init_pose = self._validate_tcp_pose("init_tcp_pose", self.init_tcp_pose)
        random_range = np.abs(np.asarray(self.init_pose_range, dtype=float))
        if random_range.shape == (3,):
            random_range = np.concatenate((random_range, [0.0, 0.0, 20.0]))
        elif random_range.shape != (6,):
            raise ValueError(f"robot.init_pose_range must contain 3 or 6 values, got {random_range.shape}.")
        target_pose = init_pose.copy()
        target_pose[:3] += np.random.uniform(-random_range[:3], random_range[:3])

        init_euler = R.from_rotvec(target_pose[3:]).as_euler("xyz", degrees=True)
        delta_euler_deg = np.random.uniform(-random_range[3:], random_range[3:])
        target_pose[3:] = R.from_euler("xyz", init_euler + delta_euler_deg, degrees=True).as_rotvec()
        return target_pose

    def move_to_tcp_pose(self, tcp_pose: np.ndarray):
        tcp_pose = np.asarray(tcp_pose, dtype=float)
        self._clear_action_idle_hold()
        if not self.debug:
            if self.robot is None:
                raise RuntimeError("Franka ZeroRPC client is not connected.")
            self._terminate_controller_safely()
            completed = self.robot.robot_move_to_ee_pose(
                tcp_pose,
                time_to_go=self.reset_time_to_go,
                blocking=True,
            )
            if not completed:
                raise RuntimeError("Franka reset trajectory did not complete.")

    def reset_episode(self):
        self._clear_action_idle_hold()
        self._open_gripper_before_reset()
        try:
            if self._is_none_pose(self.pre_reset_tcp_pose):
                logging.info("[RESET] pre_reset_tcp_pose is None; skipping pre-reset TCP pose.")
            else:
                logging.info("[RESET] Moving to pre-reset TCP pose.")
                self.move_to_tcp_pose(
                    self._validate_tcp_pose("pre_reset_tcp_pose", self.pre_reset_tcp_pose)
                )

            logging.info("[RESET] Moving to randomized initial TCP pose.")
            self.move_to_tcp_pose(self._sample_init_tcp_pose())
        finally:
            if not self.debug and self.robot is not None:
                self._terminate_controller_safely()
                self._start_cartesian_controller()
                current_pose = self.robot.robot_get_ee_pose()
                self.robot.robot_update_desired_ee_pose(current_pose)
        self.set_episode_reference_pose()

    def _has_success_gripper_state(self) -> bool:
        if self.fix_gripper:
            return True
        if not self.init_gripper:
            return True

        gripper_position = getattr(getattr(self, "gripper", None), "position", None)
        if gripper_position is None:
            if self.show_target_error:
                logging.info("[STATE] Target pose reached, waiting for gripper position.")
            return False

        gripper_position = float(gripper_position)
        gripper_open = gripper_position > self.target_reached_gripper_open_threshold
        if not gripper_open:
            logging.info(
                "[STATE] Target pose reached, but gripper is not open yet: "
                f"position={gripper_position:.3f}, threshold>{self.target_reached_gripper_open_threshold:.3f}"
            )
        return gripper_open

    def has_reached_target(self) -> bool:
        target_pose = self._validate_tcp_pose("target_tcp_pose", self.target_tcp_pose)
        threshold = np.asarray(self.target_threshold, dtype=float)
        if threshold.size == 1:
            threshold = np.repeat(threshold.item(), 6)
        if threshold.shape != (6,):
            raise ValueError(f"robot.target_threshold must contain 1 or 6 values, got {threshold.shape}.")

        current_pose = self.get_ee_pose()
        position_error = current_pose[:3] - target_pose[:3]
        rotation_error = (R.from_rotvec(target_pose[3:]) * R.from_rotvec(current_pose[3:]).inv()).as_euler("xyz")
        pose_error = np.abs(np.concatenate((position_error, rotation_error)))
        if self.show_target_error:
            logging.info(f"[STATE] Target error [x, y, z, rx, ry, rz]: {np.round(pose_error, 5).tolist()}")
        reached = bool(np.all(pose_error <= threshold))
        if reached and not self._has_success_gripper_state():
            return False
        if reached:
            logging.info(f"[STATE] Target reached, pose error: {np.round(pose_error, 5).tolist()}")
        return reached

    def _clear_action_idle_hold(self):
        self._action_idle_hold_pose = None
        self._action_idle_prev_motion_mask = np.zeros(6, dtype=bool)

    def _raw_target_pose_from_action(
        self,
        action: np.ndarray,
        current_tcp_pose: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 6:
            raise ValueError(f"action must have at least 6 values, got {action.shape}")
        if not np.all(np.isfinite(action[:6])):
            raise ValueError(f"Franka action contains non-finite values: {action[:6].tolist()}")
        current_tcp_pose = np.asarray(current_tcp_pose, dtype=float)
        if current_tcp_pose.shape != (6,):
            raise ValueError(f"current TCP pose must contain 6 values, got {current_tcp_pose.shape}")

        translation_norm = np.linalg.norm(action[:3])
        delta_rotation_reference = R.from_euler("xyz", action[3:6]).as_matrix()
        rotation_norm = np.linalg.norm(R.from_matrix(delta_rotation_reference).as_rotvec())
        if translation_norm > self.max_translation_step:
            raise ValueError(
                f"Translation action {translation_norm:.4f} m exceeds "
                f"max_translation_step={self.max_translation_step:.4f} m."
            )
        if rotation_norm > self.max_rotation_step:
            raise ValueError(
                f"Rotation action {rotation_norm:.4f} rad exceeds "
                f"max_rotation_step={self.max_rotation_step:.4f} rad."
            )

        current_rotation = R.from_rotvec(current_tcp_pose[3:]).as_matrix()

        if self.reference_frame == "base":
            delta_position_base = action[:3]
            delta_rotation_base = delta_rotation_reference
        elif self.reference_frame == "tcp":
            delta_position_base = current_rotation @ action[:3]
            delta_rotation_base = (
                current_rotation @ delta_rotation_reference @ current_rotation.T
            )
        else:
            raise ValueError(f"Unsupported reference_frame: {self.reference_frame}")

        # Match Franka teleoperation: select_vector is expressed in the fixed
        # control frame, not by zeroing raw TCP-frame action numbers.
        selection_to_base = self._selection_to_base_rotation
        delta_position_selection = selection_to_base.T @ delta_position_base
        delta_position_selection *= self.select_vector[:3]
        delta_position_base = selection_to_base @ delta_position_selection

        delta_rotation_selection = (
            selection_to_base.T @ delta_rotation_base @ selection_to_base
        )
        delta_rotation_vector_selection = R.from_matrix(
            delta_rotation_selection
        ).as_rotvec()
        delta_rotation_vector_selection *= self.select_vector[3:]
        delta_rotation_base = (
            selection_to_base
            @ R.from_rotvec(delta_rotation_vector_selection).as_matrix()
            @ selection_to_base.T
        )

        target_transform = np.eye(4)
        target_transform[:3, :3] = delta_rotation_base @ current_rotation
        target_transform[:3, 3] = current_tcp_pose[:3] + delta_position_base
        return self._transform_to_pose(target_transform)

    def _motion_delta_mask(
        self,
        current_pose: np.ndarray,
        action_target_pose: np.ndarray,
    ) -> np.ndarray:
        position_mask = (
            np.abs(action_target_pose[:3] - current_pose[:3])
            >= self.action_idle_position_threshold
        )
        current_rotation = R.from_rotvec(current_pose[3:])
        target_rotation = R.from_rotvec(action_target_pose[3:])
        rotation_active = (
            np.linalg.norm((target_rotation * current_rotation.inv()).as_rotvec())
            >= self.action_idle_rotation_threshold
        )
        return np.concatenate((position_mask, np.repeat(rotation_active, 3)))

    def _merge_selected_orientation(
        self,
        held_pose: np.ndarray,
        moving_pose: np.ndarray,
    ) -> np.ndarray:
        selection_to_base = R.from_matrix(self._selection_to_base_rotation)
        held_rotation = R.from_rotvec(np.asarray(held_pose[3:], dtype=float))
        moving_rotation = R.from_rotvec(np.asarray(moving_pose[3:], dtype=float))
        held_euler = (selection_to_base.inv() * held_rotation).as_euler("xyz")
        moving_euler = (selection_to_base.inv() * moving_rotation).as_euler("xyz")
        merged_euler = np.where(
            self.select_vector[3:].astype(bool),
            moving_euler,
            held_euler,
        )
        return (selection_to_base * R.from_euler("xyz", merged_euler)).as_rotvec()

    def _target_pose_from_action(
        self,
        action: np.ndarray,
        current_tcp_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        current_tcp_pose = (
            self.get_ee_pose()
            if current_tcp_pose is None
            else np.asarray(current_tcp_pose, dtype=float)
        )
        action_target_pose = self._raw_target_pose_from_action(action, current_tcp_pose)
        if not self.action_idle_hold_enabled:
            return action_target_pose

        if self._action_idle_hold_pose is None:
            self._action_idle_hold_pose = current_tcp_pose.copy()

        motion_mask = self._motion_delta_mask(current_tcp_pose, action_target_pose)
        for index, (was_moving, is_moving) in enumerate(
            zip(self._action_idle_prev_motion_mask[:3], motion_mask[:3])
        ):
            if was_moving and not is_moving:
                self._action_idle_hold_pose[index] = current_tcp_pose[index]

        orientation_was_moving = self._action_idle_prev_motion_mask[3]
        orientation_is_moving = motion_mask[3]
        if orientation_was_moving and not orientation_is_moving:
            self._action_idle_hold_pose[3:] = self._merge_selected_orientation(
                self._action_idle_hold_pose,
                current_tcp_pose,
            )

        if not np.any(motion_mask):
            self._action_idle_prev_motion_mask = motion_mask
            return self._action_idle_hold_pose.copy()

        target_pose = self._action_idle_hold_pose.copy()
        for index, is_moving in enumerate(motion_mask[:3]):
            if is_moving:
                target_pose[index] = action_target_pose[index]
                self._action_idle_hold_pose[index] = current_tcp_pose[index]

        if orientation_is_moving:
            target_pose[3:] = self._merge_selected_orientation(
                self._action_idle_hold_pose,
                action_target_pose,
            )
            self._action_idle_hold_pose[3:] = self._merge_selected_orientation(
                self._action_idle_hold_pose,
                current_tcp_pose,
            )

        self._action_idle_prev_motion_mask = motion_mask
        return target_pose

    def _send_cartesian_action(self, action: np.ndarray):
        if self.robot is None:
            raise RuntimeError("Franka ZeroRPC client is not connected.")
        current_pose = self.robot.robot_get_ee_pose()
        target_pose = self._target_pose_from_action(action, current_pose)
        self.robot.robot_update_desired_ee_pose(target_pose)

    def stop_robot_control(self):
        self._terminate_controller_safely()

    def _find_lora_checkpoint_norm_stats_path(self) -> Path:
        if self.checkpoint_dir is None:
            raise ValueError("VLA-Precision LoRA actor checkpoint directory is required to load norm_stats.json")

        assets_root = self.checkpoint_dir / "assets"
        if not assets_root.exists():
            raise FileNotFoundError(f"VLA-Precision LoRA actor checkpoint assets not found: {assets_root}")

        norm_stats_paths = sorted(assets_root.rglob("norm_stats.json"))
        if not norm_stats_paths:
            raise FileNotFoundError(f"No norm_stats.json found under VLA-Precision LoRA actor checkpoint assets: {assets_root}")
        if len(norm_stats_paths) > 1:
            logging.warning(
                "[MODEL] Multiple norm_stats.json files found under VLA-Precision LoRA actor checkpoint assets; "
                f"using first sorted path: {norm_stats_paths[0]}"
            )
        return norm_stats_paths[0]

    def _configure_lora_checkpoint_norm_stats(self) -> None:
        self.norm_stats_path = self._find_lora_checkpoint_norm_stats_path()
        assets_root = self.checkpoint_dir / "assets"
        asset_id = self.norm_stats_path.parent.relative_to(assets_root).as_posix()
        data_assets = dataclasses.replace(
            self.model_config.data.assets,
            assets_dir=str(assets_root),
            asset_id=asset_id,
        )
        self.model_config = dataclasses.replace(
            self.model_config,
            data=dataclasses.replace(self.model_config.data, assets=data_assets),
        )
        data_config = self.model_config.data.create(self.model_config.assets_dirs, self.model_config.model)
        self.norm_stats = data_config.norm_stats
        logging.info(f"[MODEL] Using OpenPI norm stats from VLA-Precision actor checkpoint: {self.norm_stats_path}")

    # --------------------------- OBS TRANSFER --------------------------- #
    def _transfer_obs_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Build the same 19D policy state used by the UR inference path."""

        state_parts = [
            np.asarray(obs["tcp_pose"], dtype=np.float32),
            np.asarray(obs["tcp_speed"], dtype=np.float32),
            np.asarray(obs["tcp_force"], dtype=np.float32),
        ]
        if not self.fix_gripper:
            state_parts.append(np.asarray([obs["gripper_position"]], dtype=np.float32))
        state = np.concatenate(state_parts)

        policy_obs = {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

        return policy_obs

    # --------------------------- OBS STATE --------------------------- #
    def get_obs_state(self) -> Dict[str, Any]:
        """Return current TCP-reference observation from robot."""
        obs = {}

        if self.robot:
            ee_state = self.robot.robot_get_ee_state()
            ee_pose = ee_state["pose"]
            if self.reference_frame == "base":
                obs["tcp_pose"] = self._pose_euler(ee_pose)
            elif self.reference_frame == "tcp":
                obs["tcp_pose"] = self._relative_pose_euler(ee_pose)
            else:
                raise ValueError(f"Unsupported reference_frame: {self.reference_frame}")
            obs["tcp_speed"] = ee_state["speed"]
            obs["tcp_force"] = ee_state["wrench"]

        if self.cameras:
            for name, cam in self.cameras.items():
                obs[name] = cam.read()

        if self.task_description:
            obs["prompt"] = self.task_description

        if not self.fix_gripper and self.gripper:
            obs["gripper_position"] = self.gripper.position

        return self._transfer_obs_state(obs)

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray, block: bool = False):
        """Execute model actions as TCP-frame end-effector deltas."""
        if self.robot is None:
            logging.error("[ERROR] Robot controller not connected. Cannot execute actions.")
            return

        if block:
            logging.info("[STATE] Moving robot to TCP pose...")
            self.move_to_tcp_pose(actions)
            logging.info("[STATE] Robot reached TCP pose.")
            return np.asarray(actions)
        action_chunk = np.asarray(actions)
        if action_chunk.ndim != 2:
            raise ValueError(
                f"Model action chunk must have shape (T, A), got {action_chunk.shape}."
            )
        if action_chunk.shape[0] == 0:
            raise ValueError("Model returned an empty action chunk.")
        self.fps_action.reset()
        executed_actions = []
        for action in action_chunk[:self.action_horizon]:
            start_time = time.perf_counter()
            action = np.asarray(action, dtype=float)
            expected_action_dim = 6 if self.fix_gripper else 7
            if action.ndim != 1 or action.size < expected_action_dim:
                raise ValueError(
                    f"Each model action must contain at least {expected_action_dim} values, "
                    f"got {action.shape}."
                )
            action = action[:expected_action_dim]

            if not self.debug:
                self._send_cartesian_action(action)

            if not self.fix_gripper:
                self._gripper_position = float(action[6])
            executed_actions.append(np.asarray(action, dtype=float).copy())

            elapsed = time.perf_counter() - start_time
            to_sleep = 1.0 / self.action_fps - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            self.fps_action.update(show=self.show_action_fps)

        return np.stack(executed_actions, axis=0)


    # --------------------------- KEYBOARD EPISODE CONTROL --------------------------- #
    def _keyboard_episode_control_listener(self):
        """Request reset on double Enter, or success on double Space within 0.5 seconds."""
        last_enter_time = None
        last_space_time = None
        old_termios = None
        if sys.stdin.isatty():
            old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        try:
            while not self._keyboard_listener_stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                char = sys.stdin.read(1)
                now = time.perf_counter()
                if char in ("\n", "\r"):
                    if last_enter_time is not None and now - last_enter_time <= 0.5:
                        self._reset_requested.set()
                        last_enter_time = None
                        logging.info("[RESET] Double Enter detected. Reset requested; current episode marked failed.")
                    else:
                        last_enter_time = now
                    continue
                if char != " ":
                    continue
                if last_space_time is not None and now - last_space_time <= 0.5:
                    self._success_requested.set()
                    last_space_time = None
                    logging.info("[EPISODE] Double Space detected. Success requested.")
                else:
                    last_space_time = now
        finally:
            if old_termios is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_termios)

    def _start_keyboard_episode_control_listener(self):
        self._reset_requested.clear()
        self._success_requested.clear()
        self._keyboard_listener_stop.clear()
        self._keyboard_listener_thread = threading.Thread(target=self._keyboard_episode_control_listener, daemon=True)
        self._keyboard_listener_thread.start()
        if self.use_target_success:
            logging.info("[EPISODE] Press Enter twice within 0.5s to fail/reset, or Space twice within 0.5s to request target-checked success.")
        else:
            logging.info("[EPISODE] Press Enter twice within 0.5s to fail/reset, or Space twice within 0.5s to mark success.")

    def _stop_keyboard_episode_control_listener(self):
        self._keyboard_listener_stop.set()
        if self._keyboard_listener_thread is not None:
            self._keyboard_listener_thread.join(timeout=0.2)
            self._keyboard_listener_thread = None

    def _wait_for_next_episode_start(self):
        input(f"Press Enter to start episode {self._ep_idx}...")

    # --------------------------- EPISODE TIME DISPLAY --------------------------- #
    def _print_episode_time_every_second(
        self,
        episode_index: int,
        started_at: float,
        stop_event: threading.Event,
    ):
        next_elapsed_sec = 1
        while True:
            wait_sec = started_at + next_elapsed_sec - time.perf_counter()
            if stop_event.wait(max(wait_sec, 0.0)):
                return
            elapsed_sec = int(time.perf_counter() - started_at)
            if elapsed_sec <= 0:
                next_elapsed_sec = 1
                continue
            print(f"[TIME] Episode {episode_index} elapsed: {elapsed_sec}s", flush=True)
            next_elapsed_sec = elapsed_sec + 1

    def _start_episode_time_printer(self):
        if not self.show_time or self._ep_start is None:
            return

        self._stop_episode_time_printer()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._print_episode_time_every_second,
            args=(self._ep_idx, self._ep_start, stop_event),
            daemon=True,
        )
        self._episode_time_printer_stop = stop_event
        self._episode_time_printer_thread = thread
        thread.start()

    def _stop_episode_time_printer(self):
        stop_event = self._episode_time_printer_stop
        thread = self._episode_time_printer_thread
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        self._episode_time_printer_stop = None
        self._episode_time_printer_thread = None

    # --------------------------- PIPELINE --------------------------- #
    def _create_openpi_policy(self):
        from openpi.policies import policy_config as _policy_config

        if self.checkpoint_dir is None or not (self.checkpoint_dir / "params").exists():
            raise FileNotFoundError(f"VLA-Precision LoRA actor checkpoint params not found: {self.checkpoint_dir / 'params'}")

        logging.info("[MODEL] Loading VLA-Precision policy through OpenPI policy_config.create_trained_policy")
        logging.info(f"[MODEL] OpenPI config: {self.model_config.name}")
        logging.info(f"[MODEL] model.action_horizon: {self.model_config.model.action_horizon}")
        logging.info(f"[MODEL] direct_action_horizon: {self.direct_action_horizon}; execute action_horizon: {self.action_horizon}")
        logging.info(f"[MODEL] action_expert_variant: {getattr(self.model_config.model, 'action_expert_variant', None)}")
        logging.info(f"[MODEL] RL LoRA actor checkpoint: {self.checkpoint_dir}")
        logging.info(f"[MODEL] norm stats checkpoint asset: {self.norm_stats_path}")
        logging.info(f"[MODEL] diffusion sample steps: {self.sample_steps}")
        logging.info(
            "[MODEL] VLA-Precision training actor checkpoints are OpenPI checkpoint dirs "
            "containing params/ and assets/ after full-base initialization and LoRA training."
        )

        return _policy_config.create_trained_policy(
            self.model_config,
            self.checkpoint_dir,
            sample_kwargs={"num_steps": self.sample_steps},
        )

    def _prepare_inference(self):
        logging.info("========== Starting Inference Pipeline ==========")
        self.connect_robot()
        self.connect_cameras()
        self.connect_gripper()
        self.wait_for_gripper_states()
        self.reset_episode()

        obs = self.get_obs_state()
        logging.info(f"[STATE] Observation state: {obs.keys()}")
        obs_values = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim <= 1:
                obs_values[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                obs_values[key] = {
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "min": float(np.min(value)),
                    "max": float(np.max(value)),
                }
            else:
                obs_values[key] = value
        logging.info(f"[STATE] Observation values: {obs_values}")
        policy = self._create_openpi_policy()

        logging.info("Warming up the VLA-Precision OpenPI model")
        start = time.time()
        policy.infer(obs)
        logging.info(f"Model warmup completed, took {time.time() - start:.2f}s")

        if self.fix_gripper:
            logging.info("[GRIPPER] fix_gripper enabled; skipping manual pre-inference close.")
        elif not self.init_gripper:
            input("Press Enter to close the gripper...")
            self._gripper_position = 0.0
            logging.info("[GRIPPER] Gripper close target set.")
        else:
            logging.info("[GRIPPER] init_gripper enabled; skipping manual pre-inference close.")
        input("Press Enter to continue inference...")

        # Model warmup and the manual confirmation can take an arbitrary amount
        # of time. Refresh the Cartesian policy immediately before inference so
        # the first target update is never sent to a stale/missing controller.
        if not self.debug:
            self._terminate_controller_safely()
            self._start_cartesian_controller()
            current_pose = self.robot.robot_get_ee_pose()
            self.robot.robot_update_desired_ee_pose(current_pose)
            self._clear_action_idle_hold()
            logging.info("[ROBOT] Cartesian controller refreshed before inference.")

        return policy

    def _start_episode(self):
        self._stop_episode_time_printer()
        self._ep_start = time.perf_counter()
        self._ep_steps = 0
        self._ep_done = False
        self._start_episode_time_printer()

    def _submit_step(self, result, obs, infer_idx: int):
        action_dim = 6 if self.fix_gripper else 7
        robot_actions = np.asarray(result["actions"])[..., :action_dim]
        executed_actions = self.execute_actions(robot_actions)
        self._ep_steps += 1
        self.recorder.submit_actions(
            executed_actions,
            infer_idx,
            obs["prompt"],
            state=obs["observation/state"],
            episode_index=self._ep_idx,
            episode_action_count=self._ep_steps,
            action_horizon=self.action_horizon,
        )
        self.recorder.submit_obs(obs)

    def _submit_episode(self, status: str, completed: bool, **extra):
        duration = time.perf_counter() - self._ep_start
        self.recorder.submit_episode_result(
            episode_index=self._ep_idx,
            duration_sec=duration,
            status=status,
            action_batches=self._ep_steps,
            completed=completed,
            **extra,
        )
        if completed:
            self._success_episodes += 1
        else:
            self._failed_episodes += 1

        total_ended = self._success_episodes + self._failed_episodes
        success_rate = self._success_episodes / total_ended
        failure_rate = self._failed_episodes / total_ended
        logging.info(
            f"[EPISODE] Success/failure ratio after episode {self._ep_idx}: "
            f"success {self._success_episodes}/{total_ended} ({success_rate:.1%}), "
            f"failure {self._failed_episodes}/{total_ended} ({failure_rate:.1%})"
        )
        self._ep_done = True
        self._stop_episode_time_printer()

    def _finish_episode_if_needed(self) -> bool:
        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self._success_requested.clear()
            logging.info("[RESET] Restarting current episode from initial pose.")
            self._submit_episode("keyboard_reset", completed=False, reason="enter")
        elif self._success_requested.is_set():
            self._success_requested.clear()
            if not self.use_target_success:
                self._submit_episode("manual_success", completed=True, reason="double_space")
            elif self.has_reached_target():
                self._submit_episode("target_reached", completed=True, reason="double_space")
            else:
                logging.info("[EPISODE] Double Space ignored because target pose threshold is not satisfied.")
                return False
        elif time.perf_counter() - self._ep_start >= self.ep_timeout:
            logging.info(f"[RESET] Episode timeout ({self.ep_timeout}s). Restarting current episode.")
            self._submit_episode("timeout", completed=False, reason="episode_timeout")
        elif self.use_target_success and self.has_reached_target():
            self._submit_episode("target_reached", completed=True)
        else:
            return False

        if self._ep_idx >= self.num_episodes:
            self._ep_idx += 1
            return True

        self._ep_idx += 1
        self._stop_keyboard_episode_control_listener()
        self.reset_episode()
        self._wait_for_next_episode_start()
        self._start_keyboard_episode_control_listener()
        self._start_episode()
        return True

    def _close_episode_on_exit(self):
        if self._ep_steps > 0 and not self._ep_done:
            duration = time.perf_counter() - self._ep_start
            logging.info(
                f"[EPISODE] Episode {self._ep_idx} interrupted: "
                f"{self.recorder._format_duration(duration)}"
            )

    def _reset_before_exit(self):
        if self.robot is None:
            return
        try:
            logging.info("[RESET] Resetting robot before exit.")
            self.reset_episode()
        except Exception as e:
            logging.error(f"[ERROR] Failed to reset robot before exit: {e}")

    def _ask_save_video(self):
        try:
            ans = input("Save recorded videos before exiting? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                logging.info("[INFO] Saving recorded videos before exiting...")
                self.recorder.save_video()
        except Exception as e:
            logging.error(f"[ERROR] Failed to save videos: {e}")

    def run(self):
        """Main pipeline: connect robot, cameras, and run inference."""
        try:
            policy = self._prepare_inference()
            infer_idx = 1
            self._start_keyboard_episode_control_listener()
            self._start_episode()
            logging.info("========== Starting Inference Loop ==========")
            while self._ep_idx <= self.num_episodes:
                loop_start = time.perf_counter()
                obs_start = time.perf_counter()
                obs = self.get_obs_state()
                obs_elapsed = time.perf_counter() - obs_start
                infer_start = time.perf_counter()
                result = policy.infer(obs)
                infer_elapsed = time.perf_counter() - infer_start
                submit_start = time.perf_counter()
                self._submit_step(result, obs, infer_idx)
                submit_elapsed = time.perf_counter() - submit_start
                self._finish_episode_if_needed()
                if self.show_inference_fps:
                    loop_elapsed = time.perf_counter() - loop_start
                    logging.info(
                        f"[STATE] Inference loop rate: {1 / loop_elapsed:.1f} HZ "
                        f"(loop={loop_elapsed:.4f}s, get_obs={obs_elapsed:.4f}s, "
                        f"infer={infer_elapsed:.4f}s, submit_step={submit_elapsed:.4f}s)"
                    )
                infer_idx += 1
            logging.info(f"[INFO] Finished {self.num_episodes} inference episodes.")
        except KeyboardInterrupt:
            logging.info("[INFO] KeyboardInterrupt detected. Saving recorded videos before exiting...")
        except Exception as e:
            logging.error(f"[ERROR] Inference loop encountered an error: {e}")
        finally:
            cleanup_steps = (
                ("episode keyboard listener", self._stop_keyboard_episode_control_listener),
                ("episode close", self._close_episode_on_exit),
                ("episode timer", self._stop_episode_time_printer),
            )
            for name, cleanup in cleanup_steps:
                try:
                    cleanup()
                except Exception as error:
                    logging.error("[ERROR] Failed to clean up %s: %s", name, error)
            self._reset_before_exit()
            try:
                self.recorder.submit_episode_summary(
                    action_horizon=self.action_horizon,
                    description=self.task_description,
                )
            except Exception as error:
                logging.error("[ERROR] Failed to save episode summary: %s", error)
            try:
                self.stop_robot_control()
            except Exception as error:
                logging.error("[ERROR] Failed to stop Franka controller: %s", error)
            if self.robot is not None:
                try:
                    self.robot.close()
                except Exception as error:
                    logging.error("[ERROR] Failed to close Franka ZeroRPC client: %s", error)
                finally:
                    self.robot = None

        self._ask_save_video()

# --------------------------- MAIN --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run standalone Franka inference with an OpenPI VLA-Precision checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the VLA-Precision Franka inference YAML config.",
    )
    args = parser.parse_args()
    inference = Inference(args.config)
    inference.run()

# --------------------------- ENTRY POINT --------------------------- #
if __name__ == "__main__":
    main()
