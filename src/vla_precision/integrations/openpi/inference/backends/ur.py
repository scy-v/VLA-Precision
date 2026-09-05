#!/usr/bin/env python3
"""Standalone UR5e inference entrypoint for OpenPI + VLA-Precision LoRA checkpoints.

This file is a self-contained core inference entry. Robot/camera/control logic
matches the OpenPI UR inference flow, while model loading restores the VLA-Precision
LoRA actor checkpoint saved by ``core/VLA-Precision training.py``.
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
from pathlib import Path

from pyDHgripper import PGE
from typing import Dict, Any
from .utils import FpsCounter, validate_single_arm_norm_stats
from vla_precision import image_tools
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation as R
from .recording import Recorder
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs
PAYLOAD_SETTLE_TIME_S = 0.25
FT_ZERO_SETTLE_TIME_S = 0.50

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
        self.robot_ip = robot["ip"]
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
        self.action_fps = robot["action_fps"]
        self.debug = robot.get("debug", False)
        self.control_space = robot.get("control_space", "position")
        self.reference_frame = robot.get("reference_frame", "tcp")
        self.filter_zero_output = robot.get("filter_zero_output", True)
        self.action_idle_hold_enabled = bool(
            robot.get("action_idle_hold_enabled", robot.get("ACTION_IDLE_HOLD_ENABLED", True))
        )
        legacy_idle_threshold = robot.get(
            "action_idle_hold_threshold", robot.get("ACTION_IDLE_HOLD_THRESHOLD")
        )
        self.action_idle_position_threshold = float(
            robot.get(
                "action_idle_position_threshold",
                robot.get(
                    "ACTION_IDLE_POSITION_THRESHOLD",
                    0.0001 if legacy_idle_threshold is None else legacy_idle_threshold,
                ),
            )
        )
        self.action_idle_rotation_threshold = float(
            robot.get(
                "action_idle_rotation_threshold",
                robot.get(
                    "ACTION_IDLE_ROTATION_THRESHOLD",
                    0.001 if legacy_idle_threshold is None else legacy_idle_threshold,
                ),
            )
        )
        self.payload_mass = robot.get("payload", {}).get("mass", 1.601)
        self.payload_cog = robot.get("payload", {}).get("cog", [0.011, -0.002, 0.052])

        position_cfg = robot.get("position_mode", {})
        self._velocity = position_cfg.get("speed", 0.5)
        self._acceleration = position_cfg.get("acceleration", 0.5)
        self._servo_time = position_cfg.get("servo_time", 1.0 / self.action_fps)
        self._lookahead_time = position_cfg.get("lookahead_time", 0.1)
        self._gain = position_cfg.get("gain", 300)

        force_cfg = robot.get("force_mode", {})
        self.kp = force_cfg.get("kp", 2000)
        self.kd = force_cfg.get("kd", 200)
        self.kp_rot = force_cfg.get("kp_rot", 4000)
        self.kd_rot = force_cfg.get("kd_rot", 800)
        self.rtde_freq = force_cfg.get("rtde_freq", 125)
        self.select_vector = force_cfg.get("select_vector", [1, 1, 1, 1, 1, 1])
        self.force_limit = force_cfg.get("force_limit", [2, 2, 2, 2, 2, 2])
        self.pos_delta = force_cfg.get("pos_delta", 0.2)
        self.vel_delta = force_cfg.get("vel_delta", 0.2)
        self.gain_scale = force_cfg.get("gain_scale", 1.5)
        self.control_frame_euler_deg = self._validate_control_frame_euler_deg(
            force_cfg.get("control_frame_euler_deg", [0.0, 0.0, 0.0])
        )
        self.control_to_base_rotation = R.from_euler(
            "xyz", self.control_frame_euler_deg, degrees=True
        ).as_matrix()
        self.use_control_frame = not np.allclose(
            self.control_frame_euler_deg, [0.0, 0.0, 0.0]
        )
        self.task_frame = self._make_force_task_frame()
        self.force_type = 2
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
        self.rtde_r = None
        self.rtde_c = None
        self.cameras = None
        self._last_gripper_position = 1
        self._gripper_position = 1
        self._last_servoj_ts = None
        self._zero_force_lock_active = False
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
        """Connect to UR5e robot and print current state."""
        try:
            logging.info("\n===== [ROBOT] Connecting to UR5e robot =====")
            self.rtde_r = RTDEReceiveInterface(self.robot_ip)
            self.rtde_c = RTDEControlInterface(self.robot_ip)
            self.rtde_c.setPayload(self.payload_mass, self.payload_cog)
            if self.control_space == "force":
                self.rtde_c.forceModeSetGainScaling(self.gain_scale)
                logging.info(
                    "[ROBOT] control frame Euler xyz(deg)=%s, task_frame=%s, select_vector=%s",
                    self.control_frame_euler_deg.tolist(),
                    np.round(self.task_frame, 6).tolist(),
                    self.select_vector,
                )

            # Joint positions
            joints = self.rtde_r.getActualQ()
            if joints and len(joints) == 6:
                formatted = [round(j, 4) for j in joints]
                logging.info(f"[ROBOT] Current joint positions: {formatted}")
            else:
                logging.info("[ERROR] Failed to read joint positions.")

            # TCP pose
            tcp_pose = self.rtde_r.getActualTCPPose()
            if tcp_pose and len(tcp_pose) == 6:
                formatted_pose = [round(p, 4) for p in tcp_pose]
                logging.info(f"[ROBOT] Current TCP pose: {formatted_pose}")
                logging.info(
                    f"[ROBOT] Translation (m): x={formatted_pose[0]}, y={formatted_pose[1]}, z={formatted_pose[2]}"
                )
                logging.info(
                    f"[ROBOT] Rotation (rad): rx={formatted_pose[3]}, ry={formatted_pose[4]}, rz={formatted_pose[5]}"
                )
                logging.info("===== [ROBOT] UR5e initialized successfully =====\n")
            else:
                logging.info("[ERROR] Failed to read TCP pose.")

        except Exception as e:
            logging.error("===== [ERROR] Failed to connect to UR5e robot =====")
            logging.error(f"Exception: {e}\n")
            exit(1)
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

    def _get_current_tcp_offset(self) -> np.ndarray:
        return np.asarray(self.rtde_c.getTCPOffset(), dtype=float)

    def tcp_to_ee_pose(self, tcp_pose: np.ndarray, tcp_offset: np.ndarray) -> np.ndarray:
        tcp_transform = self._pose_to_transform(np.asarray(tcp_pose, dtype=float))
        offset_transform = self._pose_to_transform(np.asarray(tcp_offset, dtype=float))
        ee_transform = tcp_transform @ np.linalg.inv(offset_transform)
        return self._transform_to_pose(ee_transform)

    def ee_to_tcp_pose(self, ee_pose: np.ndarray, tcp_offset: np.ndarray) -> np.ndarray:
        ee_transform = self._pose_to_transform(np.asarray(ee_pose, dtype=float))
        offset_transform = self._pose_to_transform(np.asarray(tcp_offset, dtype=float))
        tcp_transform = ee_transform @ offset_transform
        return self._transform_to_pose(tcp_transform)

    def get_ee_pose(self) -> np.ndarray:
        tcp_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
        tcp_offset = self._get_current_tcp_offset()
        return self.tcp_to_ee_pose(tcp_pose, tcp_offset)

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
            self.stop_force()
            self.rtde_c.servoStop()
            self.rtde_c.moveL(tcp_pose.tolist(), self._velocity, self._acceleration)

    def _reapply_payload_and_zero_ft_sensor(self):
        """Reapply gravity compensation, then zero the F/T sensor."""
        payload_result = self.rtde_c.setPayload(self.payload_mass, self.payload_cog)
        if isinstance(payload_result, (bool, np.bool_)) and not bool(payload_result):
            raise RuntimeError("RTDE setPayload returned False after episode reset")
        time.sleep(PAYLOAD_SETTLE_TIME_S)
        zero_result = self.rtde_c.zeroFtSensor()
        if isinstance(zero_result, (bool, np.bool_)) and not bool(zero_result):
            raise RuntimeError("RTDE zeroFtSensor returned False after episode reset")
        time.sleep(FT_ZERO_SETTLE_TIME_S)
        logging.info(
            "[RESET] Reapplied payload and zeroed F/T sensor: payload_mass=%s "
            "payload_cog=%s settle_time=%.2f s",
            self.payload_mass,
            self.payload_cog,
            PAYLOAD_SETTLE_TIME_S + FT_ZERO_SETTLE_TIME_S,
        )

    def reset_episode(self):
        self._clear_action_idle_hold()
        self._open_gripper_before_reset()

        if self._is_none_pose(self.pre_reset_tcp_pose):
            logging.info("[RESET] pre_reset_tcp_pose is None; skipping pre-reset TCP pose.")
        else:
            logging.info("[RESET] Moving to pre-reset TCP pose.")
            self.move_to_tcp_pose(self._validate_tcp_pose("pre_reset_tcp_pose", self.pre_reset_tcp_pose))

        logging.info("[RESET] Moving to randomized initial TCP pose.")
        self.move_to_tcp_pose(self._sample_init_tcp_pose())
        self._reapply_payload_and_zero_ft_sensor()
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

        current_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
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

    def _action_motion_mask(self, action: np.ndarray) -> np.ndarray:
        """Return per-axis translation activity and one shared rotation state."""
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 6:
            raise ValueError(f"action must have at least 6 values, got {action.shape}")
        if not self.action_idle_hold_enabled:
            return np.ones(6, dtype=bool)
        position_threshold = float(self.action_idle_position_threshold)
        rotation_threshold = float(self.action_idle_rotation_threshold)
        translation_mask = (
            np.ones(3, dtype=bool)
            if position_threshold < 0
            else np.abs(action[:3]) > position_threshold
        )
        rotation_active = (
            True
            if rotation_threshold < 0
            else np.linalg.norm(action[3:6]) > rotation_threshold
        )
        return np.concatenate((translation_mask, np.repeat(rotation_active, 3)))

    @staticmethod
    def _base_motion_delta_from_action(
        action: np.ndarray,
        current_rotation: np.ndarray,
        reference_frame: str,
    ) -> np.ndarray:
        """Express a 6D Cartesian action in the base frame used by hold anchors."""
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 6:
            raise ValueError(f"action must have at least 6 values, got {action.shape}")
        current_rot = R.from_matrix(np.asarray(current_rotation, dtype=float))
        delta_rot = R.from_euler("xyz", action[3:6])
        if reference_frame == "base":
            delta_pos_base = action[:3]
            delta_rot_base = delta_rot
        elif reference_frame == "tcp":
            delta_pos_base = current_rot.apply(action[:3])
            delta_rot_base = current_rot * delta_rot * current_rot.inv()
        else:
            raise ValueError(f"Unsupported reference_frame: {reference_frame}")
        return np.concatenate((delta_pos_base, delta_rot_base.as_rotvec()))

    def _clear_action_idle_hold(self):
        self._action_idle_hold_pose = None
        self._action_idle_prev_motion_mask = np.zeros(6, dtype=bool)

    def _target_pose_from_action(self, action: np.ndarray, current_tcp_pose: np.ndarray | None = None) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 6:
            raise ValueError(f"action must have at least 6 values, got {action.shape}")

        if current_tcp_pose is None:
            current_tcp_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
        else:
            current_tcp_pose = np.asarray(current_tcp_pose, dtype=float)
        if current_tcp_pose.shape != (6,):
            raise ValueError(f"current TCP pose must contain 6 values, got {current_tcp_pose.shape}")

        tcp_offset = self._get_current_tcp_offset()
        current_ee_pose = self.tcp_to_ee_pose(current_tcp_pose, tcp_offset)
        current_position = current_ee_pose[:3]
        current_rotation = R.from_rotvec(current_ee_pose[3:]).as_matrix()

        delta_position = np.asarray(action[:3], dtype=float)
        delta_rotation = R.from_euler("xyz", np.asarray(action[3:6], dtype=float)).as_matrix()

        if self.reference_frame == "base":
            target_position = current_position + delta_position
            target_rotation = delta_rotation @ current_rotation
        elif self.reference_frame == "tcp":
            target_position = current_position + current_rotation @ delta_position
            target_rotation = current_rotation @ delta_rotation
        else:
            raise ValueError(f"Unsupported reference_frame: {self.reference_frame}")

        target_ee_transform = np.eye(4)
        target_ee_transform[:3, :3] = target_rotation
        target_ee_transform[:3, 3] = target_position
        target_ee_pose = self._transform_to_pose(target_ee_transform)
        action_target_pose = self.ee_to_tcp_pose(target_ee_pose, tcp_offset)

        if not self.action_idle_hold_enabled or (
            self.action_idle_position_threshold < 0
            and self.action_idle_rotation_threshold < 0
        ):
            self._clear_action_idle_hold()
            return action_target_pose

        if self._action_idle_hold_pose is None:
            self._action_idle_hold_pose = current_tcp_pose.copy()

        base_motion_delta = self._base_motion_delta_from_action(
            action,
            current_rotation,
            self.reference_frame,
        )
        translation_mask = (
            np.ones(3, dtype=bool)
            if self.action_idle_position_threshold < 0
            else np.abs(base_motion_delta[:3]) > self.action_idle_position_threshold
        )
        rotation_active = (
            True
            if self.action_idle_rotation_threshold < 0
            else np.linalg.norm(base_motion_delta[3:]) > self.action_idle_rotation_threshold
        )
        motion_mask = np.concatenate((translation_mask, np.repeat(rotation_active, 3)))
        stopped_mask = self._action_idle_prev_motion_mask & ~motion_mask
        self._action_idle_hold_pose[stopped_mask] = current_tcp_pose[stopped_mask]

        target_pose = self._action_idle_hold_pose.copy()
        target_pose[motion_mask] = action_target_pose[motion_mask]
        # While an axis is moving, keep its future hold baseline aligned with
        # the actual pose. On the first idle sample it is latched once above.
        self._action_idle_hold_pose[motion_mask] = current_tcp_pose[motion_mask]
        self._action_idle_prev_motion_mask = motion_mask.copy()
        return target_pose

    def _calculate_force(self, target_pose: np.ndarray, current_pose: np.ndarray, current_vel: np.ndarray) -> np.ndarray:
        diff_p = np.clip(target_pose[:3] - current_pose[:3], -self.pos_delta, self.pos_delta)
        diff_d = np.clip(-current_vel[:3], -self.vel_delta, self.vel_delta)
        force_pos = self.kp * diff_p + self.kd * diff_d

        target_rot = R.from_rotvec(target_pose[3:]).as_matrix()
        current_rot = R.from_rotvec(current_pose[3:]).as_matrix()
        rot_err = R.from_matrix(target_rot @ current_rot.T).as_rotvec()
        torque = (self.kp_rot * rot_err - self.kd_rot * current_vel[3:]) / self.rtde_freq
        return self._wrench_base_to_task(np.concatenate((force_pos, torque)))

    @staticmethod
    def _validate_control_frame_euler_deg(euler_deg) -> np.ndarray:
        euler_deg = np.asarray(euler_deg, dtype=float)
        if euler_deg.shape != (3,) or not np.all(np.isfinite(euler_deg)):
            raise ValueError(
                "robot.force_mode.control_frame_euler_deg must contain three finite degree values."
            )
        return euler_deg

    def _make_force_task_frame(self) -> list[float]:
        if not self.use_control_frame:
            return [0.0] * 6
        rotvec = R.from_euler(
            "xyz", self.control_frame_euler_deg, degrees=True
        ).as_rotvec()
        return [0.0, 0.0, 0.0, *rotvec.tolist()]

    def _wrench_base_to_task(self, wrench_base: np.ndarray) -> np.ndarray:
        wrench_base = np.asarray(wrench_base, dtype=float)
        if wrench_base.shape != (6,):
            raise ValueError(f"Base-frame wrench must be 6D, got {wrench_base.shape}.")
        if not self.use_control_frame:
            return wrench_base
        base_to_control = self.control_to_base_rotation.T
        return np.concatenate((
            base_to_control @ wrench_base[:3],
            base_to_control @ wrench_base[3:],
        ))

    def _is_zero_output_action(self, action: np.ndarray) -> bool:
        return (
            self.filter_zero_output
            and np.any(self._action_motion_mask(action))
            and np.allclose(action[:6], 0.0, atol=1e-3)
        )

    def _calculate_ft_target(self, action: np.ndarray) -> np.ndarray:
        current_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)
        current_vel = np.asarray(self.rtde_r.getActualTCPSpeed(), dtype=float)
        target_pose = self._target_pose_from_action(action, current_tcp_pose=current_pose)
        return self._calculate_force(target_pose, current_pose, current_vel)

    def _send_force_action(self, action: np.ndarray):
        if self._is_zero_output_action(action):
            if self._zero_force_lock_active:
                return
            if self.show_target_error:
                logging.info(f"[ACTION] Zero output detected. Sending zero forceMode once. action={action.tolist()}")
            self.rtde_c.forceMode(
                self.task_frame,
                [0, 0, 0, 0, 0, 0],
                np.zeros(6),
                self.force_type,
                self.force_limit,
            )
            self._zero_force_lock_active = True
            return

        self._zero_force_lock_active = False
        ft_target = self._calculate_ft_target(action)
        self.rtde_c.forceMode(
            self.task_frame,
            self.select_vector,
            ft_target,
            self.force_type,
            self.force_limit,
        )

    def stop_force(self):
        if self.rtde_c is not None and self.control_space == "force":
            self.rtde_c.forceMode(
                self.task_frame,
                [0, 0, 0, 0, 0, 0],
                np.zeros(6),
                self.force_type,
                self.force_limit,
            )
            if hasattr(self.rtde_c, "forceModeStop"):
                self.rtde_c.forceModeStop()

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
        """Transfer raw TCP-reference observation state to UR5e policy format."""

        state_parts = [
            np.asarray(obs["tcp_pose"], dtype=np.float32),
            np.asarray(obs["tcp_speed"], dtype=np.float32),
            np.asarray(obs["tcp_force"], dtype=np.float32),
        ]
        if not self.fix_gripper:
            state_parts.append(np.asarray([obs["gripper_position"]], dtype=np.float32))
        state = np.concatenate(state_parts)

        ur5e_obs = {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

        return ur5e_obs

    # --------------------------- OBS STATE --------------------------- #
    def get_obs_state(self) -> Dict[str, Any]:
        """Return current TCP-reference observation from robot."""
        obs = {}

        if self.rtde_r:
            ee_pose = self.get_ee_pose()
            if self.reference_frame == "base":
                obs["tcp_pose"] = self._pose_euler(ee_pose)
            elif self.reference_frame == "tcp":
                obs["tcp_pose"] = self._relative_pose_euler(ee_pose)
            else:
                raise ValueError(f"Unsupported reference_frame: {self.reference_frame}")
            obs["tcp_speed"] = self.rtde_r.getActualTCPSpeed()
            obs["tcp_force"] = self.rtde_r.getActualTCPForce()

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
        if self.rtde_c is None:
            logging.error("[ERROR] Robot controller not connected. Cannot execute actions.")
            return

        if block:
            logging.info("[STATE] Moving robot to TCP pose...")
            self.move_to_tcp_pose(actions)
            logging.info("[STATE] Robot reached TCP pose.")
            return
        self.fps_action.reset()
        for action in np.asarray(actions)[:self.action_horizon]:
            start_time = time.perf_counter()
            action = np.asarray(action, dtype=float)
            expected_action_dim = 6 if self.fix_gripper else 7
            if action.ndim != 1 or action.size < expected_action_dim:
                raise ValueError(
                    f"Each model action must contain at least {expected_action_dim} values, got {action.shape}."
                )
            action = action[:expected_action_dim]

            if not self.debug:
                if self.control_space == "position":
                    target_pose = self._target_pose_from_action(action)
                    # t_start = self.rtde_c.initPeriod()
                    self.rtde_c.servoL(
                        target_pose.tolist(),
                        self._velocity,
                        self._acceleration,
                        self._servo_time,
                        self._lookahead_time,
                        self._gain,
                    )
                    # self.rtde_c.waitPeriod(t_start)
                elif self.control_space == "force":
                    # t_start = self.rtde_c.initPeriod()
                    self._send_force_action(action)
                    # self.rtde_c.waitPeriod(t_start)
                else:
                    raise ValueError(f"Unsupported control_space: {self.control_space}")

            if not self.fix_gripper:
                self._gripper_position = float(action[6])

            elapsed = time.perf_counter() - start_time
            to_sleep = 1.0 / self.action_fps - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            self.fps_action.update(show=self.show_action_fps)


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
        return policy

    def _start_episode(self):
        self._stop_episode_time_printer()
        self._ep_start = time.perf_counter()
        self._ep_steps = 0
        self._ep_done = False
        self._zero_force_lock_active = False
        self._start_episode_time_printer()

    def _submit_step(self, result, obs, infer_idx: int):
        action_dim = 6 if self.fix_gripper else 7
        robot_actions = np.asarray(result["actions"])[..., :action_dim]
        self.execute_actions(robot_actions)
        self._ep_steps += 1
        self.recorder.submit_actions(
            robot_actions[:self.action_horizon],
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
        if self.rtde_r is None or self.rtde_c is None:
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
            self._stop_keyboard_episode_control_listener()
            self._close_episode_on_exit()
            self._stop_episode_time_printer()
            self._reset_before_exit()
            self.recorder.submit_episode_summary(
                action_horizon=self.action_horizon,
                description=self.task_description,
            )
            self.stop_force()

        self._ask_save_video()

# --------------------------- MAIN --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run standalone UR5e inference with an OpenPI VLA-Precision LoRA checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "ur.yaml",
        help="Path to the VLA-Precision UR inference YAML config.",
    )
    args = parser.parse_args()
    inference = Inference(args.config)
    inference.run()

# --------------------------- ENTRY POINT --------------------------- #
if __name__ == "__main__":
    main()
