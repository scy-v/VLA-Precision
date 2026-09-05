#!/usr/bin/env python3
"""Dual-UR5e inference entrypoint for OpenPI + VLA-Precision checkpoints.

The policy state is ``left[19] + right[19]`` and every action is
``left[6D + gripper] + right[6D + gripper]``.  Episode control, logging,
success/failure accounting and model loading reuse the proven single-arm
entrypoint; all hardware-facing behavior is implemented per arm here.
"""

import argparse
import dataclasses
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict

import crcmod
import numpy as np
import serial
import yaml
from scipy.spatial.transform import Rotation as R

from .ur import Inference as SingleArmInference
from .ur import is_none_config, update_latest_symlink
from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from vla_precision import image_tools
from pyDHgripper import PGE
from .recording import Recorder
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from .utils import FpsCounter


logging.basicConfig(level=logging.INFO, format="%(message)s")
ARMS = ("left", "right")
PAYLOAD_SETTLE_TIME_S = 0.25
FT_ZERO_SETTLE_TIME_S = 0.50


class DualInference(SingleArmInference):
    """Run one policy over two independent UR5e controller/gripper stacks."""

    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model = cfg["model"]
        robot = cfg["robot"]
        self.remote_policy = cfg.get("policy", {}).get("location", "local") == "server"
        arm_configs = robot.get("arms", {})
        missing_arms = [arm for arm in ARMS if arm not in arm_configs]
        if missing_arms:
            raise KeyError(f"robot.arms is missing: {', '.join(missing_arms)}")

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

        self.arms = {
            arm: self._parse_arm_config(arm, arm_configs[arm])
            for arm in ARMS
        }
        if self.remote_policy:
            for arm in ARMS:
                if self.arms[arm].fix_gripper:
                    self.arms[arm].fixed_gripper_position = float(
                        arm_configs[arm].get("fixed_gripper_position", 0.0)
                    )
        else:
            self._configure_fixed_gripper_positions()
        action_fps_values = [float(self.arms[arm].action_fps) for arm in ARMS]
        self.action_fps = min(action_fps_values)
        if not np.allclose(action_fps_values, action_fps_values[0]):
            logging.warning(
                "[ACTION] left/right action_fps differ (%s); synchronized action chunks run at the slower %.3f Hz.",
                action_fps_values,
                self.action_fps,
            )

        cam = cfg["cameras"]
        self.left_wrist_cam_serial = cam["left_wrist_cam_serial"]
        self.right_wrist_cam_serial = cam["right_wrist_cam_serial"]
        self.exterior_cam_serial = cam["exterior_cam_serial"]
        self.cam_fps = cam.get("fps", 30)
        self.cam_width = cam.get("width", 640)
        self.cam_height = cam.get("height", 480)

        video = cfg["video"]
        self.video_fps = video.get("fps", 7)
        self.visualize = video["visualize"]
        record = cfg.get("record", {})
        self.num_episodes = record.get("num_episodes", 10)
        self.ep_timeout = record.get("episode_timeout_sec", 30)
        self.show_action_fps = record.get("show_action_fps", False)
        self.show_inference_fps = record.get("show_inference_fps", False)
        self.show_time = record.get("show_time", False)

        self.task_description = cfg["task"]["description"]
        self.use_target_success = any(self.arms[arm].target_success_enabled for arm in ARMS)

        time_str = time.strftime("%Y%m%d-%H%M%S")
        time_path = time.strftime("%Y%m%d")
        base_dir = Path(cfg.get("output_root", "./results")).expanduser() / cfg["experiment"]["name"] / "openpi-native"
        log_dir = base_dir / "logs"
        video_dir = base_dir / "videos" / time_path
        (log_dir / "all_logs").mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "all_logs" / f"log_dual_{time_str}.yaml"
        update_latest_symlink(log_path, log_dir / "latest_dual.yaml")
        self.recorder = Recorder(
            log_path=log_path,
            video_path=[
                video_dir / f"left_wrist_{time_str}.mp4",
                video_dir / f"exterior_{time_str}.mp4",
                video_dir / f"right_wrist_{time_str}.mp4",
            ],
            display_fps=self.video_fps,
            visualize=self.visualize,
        )

        self.fps_action = FpsCounter(name="dual_action")
        self.cameras = None
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
        self._arm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-ur")

    def _parse_arm_config(self, arm: str, cfg: dict) -> SimpleNamespace:
        gripper = cfg.get("gripper", {})
        payload = cfg.get("payload", {})
        position = cfg.get("position_mode", {})
        force = cfg.get("force_mode", {})
        action_fps = float(cfg["action_fps"])
        control_frame_euler_deg = self._validate_control_frame_euler_deg(
            force.get("control_frame_euler_deg", [0.0, 0.0, 0.0])
        )
        control_to_base_rotation = R.from_euler(
            "xyz", control_frame_euler_deg, degrees=True
        ).as_matrix()
        use_control_frame = not np.allclose(
            control_frame_euler_deg, [0.0, 0.0, 0.0]
        )
        task_frame_rotvec = R.from_euler(
            "xyz", control_frame_euler_deg, degrees=True
        ).as_rotvec()

        target_pose = cfg.get("target_tcp_pose")
        target_threshold = cfg.get("target_threshold")
        has_target_pose = not is_none_config(target_pose)
        has_target_threshold = not is_none_config(target_threshold)
        if has_target_pose != has_target_threshold:
            raise ValueError(
                f"robot.arms.{arm}.target_tcp_pose and target_threshold must both be set or both be null."
            )

        gripper_min_position = int(gripper.get("min_position", 0))
        gripper_max_position = int(gripper.get("max_position", 1000))
        if not 0 <= gripper_min_position < gripper_max_position <= 1000:
            raise ValueError(
                f"robot.arms.{arm}.gripper position range must satisfy "
                "0 <= min_position < max_position <= 1000, got "
                f"[{gripper_min_position}, {gripper_max_position}]."
            )

        legacy_idle_threshold = cfg.get("action_idle_hold_threshold")

        return SimpleNamespace(
            name=arm,
            ip=cfg["ip"],
            init_gripper=bool(gripper.get("init", True)),
            fix_gripper=bool(gripper.get("fix_observation", False)),
            gripper_port=gripper["port"],
            gripper_reverse=bool(gripper.get("reverse", False)),
            close_threshold=float(gripper.get("close_threshold", 0.7)),
            target_reached_gripper_open_threshold=float(
                gripper.get("target_reached_open_threshold", gripper.get("close_threshold", 0.7))
            ),
            gripper_force=int(gripper.get("force", 70)),
            gripper_speed=int(gripper.get("speed", 60)),
            gripper_min_position=gripper_min_position,
            gripper_max_position=gripper_max_position,
            pre_reset_tcp_pose=cfg.get("pre_reset_tcp_pose"),
            init_tcp_pose=cfg["init_tcp_pose"],
            init_pose_range=cfg.get("init_pose_range", [0.03, 0.03, 0.03, 0.0, 0.0, 20.0]),
            target_tcp_pose=target_pose,
            target_threshold=target_threshold,
            target_success_enabled=has_target_pose and has_target_threshold,
            show_target_error=bool(cfg.get("show_target_error", False)),
            action_fps=action_fps,
            debug=bool(cfg.get("debug", False)),
            control_space=cfg.get("control_space", "position"),
            reference_frame=cfg.get("reference_frame", "tcp"),
            filter_zero_output=bool(cfg.get("filter_zero_output", True)),
            action_idle_hold_enabled=bool(cfg.get("action_idle_hold_enabled", True)),
            action_idle_position_threshold=float(
                cfg.get(
                    "action_idle_position_threshold",
                    0.0001 if legacy_idle_threshold is None else legacy_idle_threshold,
                )
            ),
            action_idle_rotation_threshold=float(
                cfg.get(
                    "action_idle_rotation_threshold",
                    0.001 if legacy_idle_threshold is None else legacy_idle_threshold,
                )
            ),
            payload_mass=float(payload.get("mass", 1.601)),
            payload_cog=payload.get("cog", [0.011, -0.002, 0.052]),
            velocity=float(position.get("speed", 0.5)),
            acceleration=float(position.get("acceleration", 0.5)),
            servo_time=float(position.get("servo_time", 1.0 / action_fps)),
            lookahead_time=float(position.get("lookahead_time", 0.1)),
            gain=int(position.get("gain", 300)),
            kp=float(force.get("kp", 2000)),
            kd=float(force.get("kd", 200)),
            kp_rot=float(force.get("kp_rot", 4000)),
            kd_rot=float(force.get("kd_rot", 800)),
            rtde_freq=float(force.get("rtde_freq", 125)),
            select_vector=force.get("select_vector", [1, 1, 1, 1, 1, 1]),
            force_limit=force.get("force_limit", [2, 2, 2, 2, 2, 2]),
            pos_delta=np.asarray(force.get("pos_delta", 0.2), dtype=float),
            vel_delta=np.asarray(force.get("vel_delta", 0.2), dtype=float),
            gain_scale=float(force.get("gain_scale", 1.5)),
            control_frame_euler_deg=control_frame_euler_deg,
            control_to_base_rotation=control_to_base_rotation,
            use_control_frame=use_control_frame,
            task_frame=(
                [0.0, 0.0, 0.0, *task_frame_rotvec.tolist()]
                if use_control_frame
                else [0.0] * 6
            ),
            force_type=int(force.get("force_type", 2)),
            rtde_r=None,
            rtde_c=None,
            gripper=None,
            target_gripper_position=1.0,
            last_gripper_position=1.0,
            fixed_gripper_position=None,
            episode_reference_ee_pose=None,
            idle_hold_pose=None,
            idle_prev_motion_mask=np.zeros(6, dtype=bool),
            zero_force_lock_active=False,
        )

    def _configure_fixed_gripper_positions(self):
        state_stats = self.norm_stats.get("state") or self.norm_stats.get("observation/state")
        if state_stats is None:
            raise KeyError(f"Dual-arm state norm stats not found in {self.norm_stats_path}")
        state_mean = state_stats.get("mean") if isinstance(state_stats, dict) else getattr(state_stats, "mean", None)
        state_mean = np.asarray(state_mean, dtype=float) if state_mean is not None else np.array([])
        action_stats = self.norm_stats.get("actions") or self.norm_stats.get("action")
        action_mean = (
            action_stats.get("mean")
            if isinstance(action_stats, dict)
            else getattr(action_stats, "mean", None)
        )
        action_mean = np.asarray(action_mean, dtype=float) if action_mean is not None else np.array([])
        if state_mean.size != 38 or action_mean.size != 14:
            raise ValueError(
                "DualInference requires a dual-arm checkpoint with 38D state and 14D action norm stats; "
                f"got state={state_mean.size}, action={action_mean.size} in {self.norm_stats_path}."
            )
        for arm, index in (("left", 18), ("right", 37)):
            if self.arms[arm].fix_gripper:
                self.arms[arm].fixed_gripper_position = float(state_mean[index])
                logging.info(
                    "[STATE][%s] fix_observation enabled; norm-stat gripper mean=%.6f",
                    arm,
                    self.arms[arm].fixed_gripper_position,
                )

    # --------------------------- ROBOTS --------------------------- #
    def connect_robot(self):
        logging.info("\n===== [ROBOT] Connecting to dual UR5e robots =====")
        for arm in ARMS:
            a = self.arms[arm]
            try:
                a.rtde_r = RTDEReceiveInterface(a.ip)
                a.rtde_c = RTDEControlInterface(a.ip)
                a.rtde_c.setPayload(a.payload_mass, a.payload_cog)
                if a.control_space == "force":
                    a.rtde_c.forceModeSetGainScaling(a.gain_scale)
                    logging.info(
                        "[ROBOT][%s] control frame Euler xyz(deg)=%s, task_frame=%s, select_vector=%s",
                        arm,
                        a.control_frame_euler_deg.tolist(),
                        np.round(a.task_frame, 6).tolist(),
                        a.select_vector,
                    )
                joints = np.asarray(a.rtde_r.getActualQ(), dtype=float)
                tcp_pose = np.asarray(a.rtde_r.getActualTCPPose(), dtype=float)
                logging.info("[ROBOT][%s] joints: %s", arm, np.round(joints, 4).tolist())
                logging.info("[ROBOT][%s] TCP pose: %s", arm, np.round(tcp_pose, 4).tolist())
            except Exception as exc:
                raise RuntimeError(f"Failed to connect {arm} UR5e at {a.ip}: {exc}") from exc
        logging.info("===== [ROBOT] Both UR5e robots initialized successfully =====\n")

    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        logging.info("\n===== [CAM] Initializing three RealSense cameras =====")
        serials = {
            "left_wrist_image": self.left_wrist_cam_serial,
            "right_wrist_image": self.right_wrist_cam_serial,
            "exterior_image": self.exterior_cam_serial,
        }
        camera_config = {
            name: RealSenseCameraConfig(
                serial_number_or_name=serial_number,
                fps=self.cam_fps,
                width=self.cam_width,
                height=self.cam_height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )
            for name, serial_number in serials.items()
        }
        self.cameras = make_cameras_from_configs(camera_config)
        for name, camera in self.cameras.items():
            camera.connect()
            logging.info("[CAM] %s connected successfully.", name)
        logging.info("===== [CAM] Three cameras initialized successfully =====\n")

    # --------------------------- GRIPPERS --------------------------- #
    @staticmethod
    def _create_arm_gripper(a: SimpleNamespace):
        if a.init_gripper:
            return PGE(port=a.gripper_port)
        gripper = PGE.__new__(PGE)
        gripper.ser = serial.Serial(port=a.gripper_port, baudrate=115200)
        gripper.crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)
        return gripper

    def connect_gripper(self):
        logging.info("\n===== [GRIPPER] Initializing two DH grippers =====")
        for arm in ARMS:
            a = self.arms[arm]
            try:
                a.gripper = self._create_arm_gripper(a)
                if a.init_gripper:
                    a.gripper.init_feedback()
                a.gripper.set_force(a.gripper_force)
                a.gripper.set_vel(a.gripper_speed)
                threading.Thread(target=self._read_gripper_state, args=(arm,), daemon=True).start()
                logging.info(
                    "[GRIPPER][%s] port=%s force=%s speed=%s reverse=%s position_range=[%s, %s]",
                    arm,
                    a.gripper_port,
                    a.gripper_force,
                    a.gripper_speed,
                    a.gripper_reverse,
                    a.gripper_min_position,
                    a.gripper_max_position,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize {arm} gripper: {exc}") from exc

    def _read_gripper_state(self, arm: str):
        a = self.arms[arm]
        a.gripper.position = None
        while True:
            command = 0.0 if a.target_gripper_position <= a.close_threshold else 1.0
            if a.gripper_reverse:
                command = 1.0 - command
            if command != a.last_gripper_position:
                command = float(np.clip(command, 0.0, 1.0))
                device_position = int(
                    round(
                        a.gripper_min_position
                        + command * (a.gripper_max_position - a.gripper_min_position)
                    )
                )
                a.gripper.set_pos(val=device_position, blocking=False)
                a.last_gripper_position = command
            position = a.gripper.read_pos() / 1000.0
            a.gripper.position = 1.0 - position if a.gripper_reverse else position
            time.sleep(0.01)

    def wait_for_gripper_states(self):
        while any(getattr(self.arms[arm].gripper, "position", None) is None for arm in ARMS):
            logging.info("[GRIPPER] Waiting for both gripper states...")
            time.sleep(0.1)

    def _open_grippers_before_reset(self):
        for arm in ARMS:
            a = self.arms[arm]
            if a.init_gripper and a.gripper is not None:
                a.target_gripper_position = 1.0
                a.last_gripper_position = None
                logging.info("[GRIPPER][%s] opening before reset.", arm)

    # --------------------------- POSE / RESET --------------------------- #
    def _get_current_tcp_offset(self, arm: str) -> np.ndarray:
        return np.asarray(self.arms[arm].rtde_c.getTCPOffset(), dtype=float)

    def _get_ee_pose(self, arm: str, tcp_pose: np.ndarray | None = None) -> np.ndarray:
        if tcp_pose is None:
            tcp_pose = np.asarray(self.arms[arm].rtde_r.getActualTCPPose(), dtype=float)
        return self.tcp_to_ee_pose(tcp_pose, self._get_current_tcp_offset(arm))

    def _relative_pose_euler_for_arm(self, arm: str, pose: np.ndarray) -> np.ndarray:
        reference = self.arms[arm].episode_reference_ee_pose
        if reference is None:
            raise RuntimeError(f"{arm} episode reference EE pose is not set.")
        relative = np.linalg.inv(self._pose_to_transform(reference)) @ self._pose_to_transform(pose)
        return np.concatenate((relative[:3, 3], R.from_matrix(relative[:3, :3]).as_euler("xyz")))

    def _validate_arm_tcp_pose(self, arm: str, name: str, pose) -> np.ndarray:
        tcp_pose = np.asarray(pose, dtype=float)
        if tcp_pose.shape != (6,):
            raise ValueError(f"robot.arms.{arm}.{name} must contain 6 values, got {tcp_pose.shape}.")
        tcp_pose = tcp_pose.copy()
        tcp_pose[3:] = R.from_euler("xyz", tcp_pose[3:]).as_rotvec()
        return tcp_pose

    def _sample_init_tcp_pose(self, arm: str) -> np.ndarray:
        a = self.arms[arm]
        target = self._validate_arm_tcp_pose(arm, "init_tcp_pose", a.init_tcp_pose)
        random_range = np.abs(np.asarray(a.init_pose_range, dtype=float))
        if random_range.shape == (3,):
            random_range = np.concatenate((random_range, [0.0, 0.0, 20.0]))
        if random_range.shape != (6,):
            raise ValueError(f"robot.arms.{arm}.init_pose_range must contain 3 or 6 values.")
        target[:3] += np.random.uniform(-random_range[:3], random_range[:3])
        initial_euler_deg = R.from_rotvec(target[3:]).as_euler("xyz", degrees=True)
        target[3:] = R.from_euler(
            "xyz",
            initial_euler_deg + np.random.uniform(-random_range[3:], random_range[3:]),
            degrees=True,
        ).as_rotvec()
        return target

    def _clear_action_idle_hold(self, arm: str):
        a = self.arms[arm]
        a.idle_hold_pose = None
        a.idle_prev_motion_mask = np.zeros(6, dtype=bool)

    def _stop_force_arm(self, arm: str):
        a = self.arms[arm]
        if a.rtde_c is None or a.control_space != "force":
            return
        a.rtde_c.forceMode(a.task_frame, [0] * 6, np.zeros(6), a.force_type, a.force_limit)
        if hasattr(a.rtde_c, "forceModeStop"):
            a.rtde_c.forceModeStop()

    def move_to_tcp_pose(self, arm: str, tcp_pose: np.ndarray):
        a = self.arms[arm]
        self._clear_action_idle_hold(arm)
        if not a.debug:
            self._stop_force_arm(arm)
            a.rtde_c.servoStop()
            a.rtde_c.moveL(np.asarray(tcp_pose, dtype=float).tolist(), a.velocity, a.acceleration)

    def _run_parallel(self, function):
        futures = {self._arm_executor.submit(function, arm): arm for arm in ARMS}
        for future, arm in futures.items():
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"{arm} arm operation failed: {exc}") from exc

    def _reset_payload(self, arm: str):
        """Reapply one arm's payload, then zero that arm's F/T sensor."""
        a = self.arms[arm]
        if a.debug:
            return
        result = a.rtde_c.setPayload(a.payload_mass, a.payload_cog)
        if isinstance(result, (bool, np.bool_)) and not bool(result):
            raise RuntimeError(f"{arm} RTDE setPayload returned False")
        time.sleep(PAYLOAD_SETTLE_TIME_S)
        zero_result = a.rtde_c.zeroFtSensor()
        if isinstance(zero_result, (bool, np.bool_)) and not bool(zero_result):
            raise RuntimeError(f"{arm} RTDE zeroFtSensor returned False")
        time.sleep(FT_ZERO_SETTLE_TIME_S)
        logging.info(
            "[RESET][%s] payload reapplied and F/T sensor zeroed: "
            "payload_mass=%s payload_cog=%s settle_time=%.2f s",
            arm,
            a.payload_mass,
            a.payload_cog,
            PAYLOAD_SETTLE_TIME_S + FT_ZERO_SETTLE_TIME_S,
        )

    def reset_episode(self):
        for arm in ARMS:
            self._clear_action_idle_hold(arm)
        self._open_grippers_before_reset()

        def pre_reset(arm: str):
            pose = self.arms[arm].pre_reset_tcp_pose
            if is_none_config(pose):
                logging.info("[RESET][%s] pre_reset_tcp_pose is null; skipping.", arm)
                return
            logging.info("[RESET][%s] moving to pre-reset pose.", arm)
            self.move_to_tcp_pose(arm, self._validate_arm_tcp_pose(arm, "pre_reset_tcp_pose", pose))

        self._run_parallel(pre_reset)
        logging.info("[RESET] Moving both arms to randomized initial TCP poses.")
        self._run_parallel(lambda arm: self.move_to_tcp_pose(arm, self._sample_init_tcp_pose(arm)))
        # Reapply gravity compensation and then zero F/T after both arms have
        # reached their reset poses, before the new episode starts.
        self._run_parallel(self._reset_payload)
        for arm in ARMS:
            self.arms[arm].episode_reference_ee_pose = self._get_ee_pose(arm)
            logging.info(
                "[STATE][%s] episode reference EE pose: %s",
                arm,
                np.round(self.arms[arm].episode_reference_ee_pose, 6).tolist(),
            )

    # --------------------------- SUCCESS --------------------------- #
    def _has_success_gripper_state(self, arm: str) -> bool:
        a = self.arms[arm]
        if not a.init_gripper:
            return True
        position = getattr(a.gripper, "position", None)
        if position is None:
            return False
        is_open = float(position) > a.target_reached_gripper_open_threshold
        if not is_open:
            logging.info(
                "[STATE][%s] pose reached but gripper %.3f <= open threshold %.3f",
                arm,
                float(position),
                a.target_reached_gripper_open_threshold,
            )
        return is_open

    def _arm_has_reached_target(self, arm: str) -> bool:
        a = self.arms[arm]
        if not a.target_success_enabled:
            return True
        target = self._validate_arm_tcp_pose(arm, "target_tcp_pose", a.target_tcp_pose)
        threshold = np.asarray(a.target_threshold, dtype=float)
        if threshold.size == 1:
            threshold = np.repeat(threshold.item(), 6)
        if threshold.shape != (6,):
            raise ValueError(f"robot.arms.{arm}.target_threshold must contain 1 or 6 values.")
        current = np.asarray(a.rtde_r.getActualTCPPose(), dtype=float)
        position_error = current[:3] - target[:3]
        rotation_error = (R.from_rotvec(target[3:]) * R.from_rotvec(current[3:]).inv()).as_euler("xyz")
        pose_error = np.abs(np.concatenate((position_error, rotation_error)))
        if a.show_target_error:
            logging.info("[STATE][%s] target error: %s", arm, np.round(pose_error, 5).tolist())
        reached = bool(np.all(pose_error <= threshold))
        return reached and self._has_success_gripper_state(arm)

    def has_reached_target(self) -> bool:
        results = {
            arm: self._arm_has_reached_target(arm)
            for arm in ARMS
            if self.arms[arm].target_success_enabled
        }
        reached = bool(results) and all(results.values())
        if reached:
            logging.info("[STATE] Dual-arm target reached: %s", results)
        return reached

    # --------------------------- OBSERVATION --------------------------- #
    def _transfer_obs_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        per_arm_states = []
        for arm in ARMS:
            per_arm_states.append(np.concatenate((
                np.asarray(obs[f"{arm}_tcp_pose"], dtype=np.float32),
                np.asarray(obs[f"{arm}_tcp_speed"], dtype=np.float32),
                np.asarray(obs[f"{arm}_tcp_force"], dtype=np.float32),
                np.asarray([obs[f"{arm}_gripper_position"]], dtype=np.float32),
            )))
        state = np.concatenate(per_arm_states)
        if state.shape != (38,):
            raise ValueError(f"Dual UR policy state must be 38D (left19 + right19), got {state.shape}.")
        return {
            "observation/state": state,
            "observation/exterior_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/left_wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["left_wrist_image"], 224, 224)
            ),
            "observation/right_wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["right_wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

    def get_obs_state(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {}
        for arm in ARMS:
            a = self.arms[arm]
            current_tcp_pose = np.asarray(a.rtde_r.getActualTCPPose(), dtype=float)
            ee_pose = self._get_ee_pose(arm, current_tcp_pose)
            if a.reference_frame == "base":
                obs[f"{arm}_tcp_pose"] = self._pose_euler(ee_pose)
            elif a.reference_frame == "tcp":
                obs[f"{arm}_tcp_pose"] = self._relative_pose_euler_for_arm(arm, ee_pose)
            else:
                raise ValueError(f"Unsupported {arm} reference_frame: {a.reference_frame}")
            obs[f"{arm}_tcp_speed"] = a.rtde_r.getActualTCPSpeed()
            obs[f"{arm}_tcp_force"] = a.rtde_r.getActualTCPForce()
            gripper_position = a.gripper.position
            if a.fix_gripper:
                gripper_position = a.fixed_gripper_position
            obs[f"{arm}_gripper_position"] = gripper_position

        for name, camera in self.cameras.items():
            obs[name] = camera.read()
        obs["prompt"] = self.task_description
        return self._transfer_obs_state(obs)

    # --------------------------- ACTIONS --------------------------- #
    def _action_motion_mask(self, arm: str, action: np.ndarray) -> np.ndarray:
        a = self.arms[arm]
        if not a.action_idle_hold_enabled:
            return np.ones(6, dtype=bool)
        action = np.asarray(action, dtype=float)
        translation_mask = (
            np.ones(3, dtype=bool)
            if a.action_idle_position_threshold < 0
            else np.abs(action[:3]) > a.action_idle_position_threshold
        )
        rotation_active = (
            True
            if a.action_idle_rotation_threshold < 0
            else np.linalg.norm(action[3:6]) > a.action_idle_rotation_threshold
        )
        return np.concatenate((translation_mask, np.repeat(rotation_active, 3)))

    def _target_pose_from_action(
        self,
        arm: str,
        action: np.ndarray,
        current_tcp_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        a = self.arms[arm]
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 6:
            raise ValueError(f"{arm} action must have at least 6 values, got {action.shape}")
        if current_tcp_pose is None:
            current_tcp_pose = np.asarray(a.rtde_r.getActualTCPPose(), dtype=float)
        else:
            current_tcp_pose = np.asarray(current_tcp_pose, dtype=float)

        tcp_offset = self._get_current_tcp_offset(arm)
        current_ee_pose = self.tcp_to_ee_pose(current_tcp_pose, tcp_offset)
        current_rotation = R.from_rotvec(current_ee_pose[3:]).as_matrix()
        delta_position = action[:3]
        delta_rotation = R.from_euler("xyz", action[3:6]).as_matrix()
        if a.reference_frame == "base":
            target_position = current_ee_pose[:3] + delta_position
            target_rotation = delta_rotation @ current_rotation
        elif a.reference_frame == "tcp":
            target_position = current_ee_pose[:3] + current_rotation @ delta_position
            target_rotation = current_rotation @ delta_rotation
        else:
            raise ValueError(f"Unsupported {arm} reference_frame: {a.reference_frame}")
        target_transform = np.eye(4)
        target_transform[:3, :3] = target_rotation
        target_transform[:3, 3] = target_position
        action_target = self.ee_to_tcp_pose(self._transform_to_pose(target_transform), tcp_offset)

        if not a.action_idle_hold_enabled or (
            a.action_idle_position_threshold < 0
            and a.action_idle_rotation_threshold < 0
        ):
            self._clear_action_idle_hold(arm)
            return action_target
        if a.idle_hold_pose is None:
            a.idle_hold_pose = current_tcp_pose.copy()
        # Hold anchors and action_target are represented in the robot base frame.
        # Translation remains per-axis, while orientation is always held or
        # updated as a complete rotation rather than as three rotvec components.
        base_motion_delta = self._base_motion_delta_from_action(
            action,
            current_rotation,
            a.reference_frame,
        )
        translation_mask = (
            np.ones(3, dtype=bool)
            if a.action_idle_position_threshold < 0
            else np.abs(base_motion_delta[:3]) > a.action_idle_position_threshold
        )
        rotation_active = (
            True
            if a.action_idle_rotation_threshold < 0
            else np.linalg.norm(base_motion_delta[3:]) > a.action_idle_rotation_threshold
        )
        motion_mask = np.concatenate((translation_mask, np.repeat(rotation_active, 3)))
        stopped_mask = a.idle_prev_motion_mask & ~motion_mask
        a.idle_hold_pose[stopped_mask] = current_tcp_pose[stopped_mask]
        target = a.idle_hold_pose.copy()
        target[motion_mask] = action_target[motion_mask]
        a.idle_hold_pose[motion_mask] = current_tcp_pose[motion_mask]
        a.idle_prev_motion_mask = motion_mask.copy()
        return target

    def _calculate_force(
        self,
        arm: str,
        target_pose: np.ndarray,
        current_pose: np.ndarray,
        current_vel: np.ndarray,
    ) -> np.ndarray:
        a = self.arms[arm]
        diff_p = np.clip(target_pose[:3] - current_pose[:3], -a.pos_delta, a.pos_delta)
        diff_d = np.clip(-current_vel[:3], -a.vel_delta, a.vel_delta)
        force_pos = a.kp * diff_p + a.kd * diff_d
        target_rot = R.from_rotvec(target_pose[3:]).as_matrix()
        current_rot = R.from_rotvec(current_pose[3:]).as_matrix()
        rot_err = R.from_matrix(target_rot @ current_rot.T).as_rotvec()
        torque = (a.kp_rot * rot_err - a.kd_rot * current_vel[3:]) / a.rtde_freq
        return self._wrench_base_to_task(arm, np.concatenate((force_pos, torque)))

    def _wrench_base_to_task(self, arm: str, wrench_base: np.ndarray) -> np.ndarray:
        a = self.arms[arm]
        wrench_base = np.asarray(wrench_base, dtype=float)
        if wrench_base.shape != (6,):
            raise ValueError(f"{arm} base-frame wrench must be 6D, got {wrench_base.shape}.")
        if not a.use_control_frame:
            return wrench_base
        base_to_control = a.control_to_base_rotation.T
        return np.concatenate((
            base_to_control @ wrench_base[:3],
            base_to_control @ wrench_base[3:],
        ))

    def _calculate_ft_target(self, arm: str, action: np.ndarray) -> np.ndarray:
        """Read the current pose once and reuse it for target and force."""
        a = self.arms[arm]
        current_pose = np.asarray(a.rtde_r.getActualTCPPose(), dtype=float)
        current_vel = np.asarray(a.rtde_r.getActualTCPSpeed(), dtype=float)
        target_pose = self._target_pose_from_action(arm, action, current_tcp_pose=current_pose)
        return self._calculate_force(arm, target_pose, current_pose, current_vel)

    def _send_force_action(self, arm: str, action: np.ndarray):
        a = self.arms[arm]
        zero_output = (
            a.filter_zero_output
            and np.any(self._action_motion_mask(arm, action))
            and np.allclose(action[:6], 0.0, atol=1e-3)
        )
        if zero_output:
            if not a.zero_force_lock_active:
                a.rtde_c.forceMode(a.task_frame, [0] * 6, np.zeros(6), a.force_type, a.force_limit)
                a.zero_force_lock_active = True
            return
        a.zero_force_lock_active = False
        a.rtde_c.forceMode(
            a.task_frame,
            a.select_vector,
            self._calculate_ft_target(arm, action),
            a.force_type,
            a.force_limit,
        )

    def _execute_arm_action(self, arm: str, action: np.ndarray):
        a = self.arms[arm]
        if not a.debug:
            if a.control_space == "position":
                target = self._target_pose_from_action(arm, action)
                a.rtde_c.servoL(
                    target.tolist(),
                    a.velocity,
                    a.acceleration,
                    a.servo_time,
                    a.lookahead_time,
                    a.gain,
                )
            elif a.control_space == "force":
                self._send_force_action(arm, action)
            else:
                raise ValueError(f"Unsupported {arm} control_space: {a.control_space}")
        a.target_gripper_position = float(action[6])

    def execute_actions(self, actions: np.ndarray, block: bool = False):
        if block:
            raise ValueError("DualInference block actions are unsupported; reset uses per-arm TCP poses.")
        if any(self.arms[arm].rtde_c is None for arm in ARMS):
            raise RuntimeError("Both robot controllers must be connected before action execution.")
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] < 14:
            raise ValueError(f"Dual policy actions must have shape (T, 14+), got {actions.shape}.")

        self.fps_action.reset()
        for action in actions[:self.action_horizon]:
            started = time.perf_counter()
            arm_actions = {"left": action[:7].copy(), "right": action[7:14].copy()}
            self._run_parallel(lambda arm: self._execute_arm_action(arm, arm_actions[arm]))
            sleep_time = 1.0 / self.action_fps - (time.perf_counter() - started)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.fps_action.update(show=self.show_action_fps)

    def stop_force(self):
        self._run_parallel(self._stop_force_arm)

    # --------------------------- PIPELINE OVERRIDES --------------------------- #
    def _prepare_inference(self):
        logging.info("========== Starting Dual-Arm Inference Pipeline ==========")
        self.connect_robot()
        self.connect_cameras()
        self.connect_gripper()
        self.wait_for_gripper_states()
        self.reset_episode()
        obs = self.get_obs_state()
        logging.info("[STATE] Dual observation keys: %s", obs.keys())
        logging.info("[STATE] observation/state shape: %s", obs["observation/state"].shape)
        policy = self._create_openpi_policy()
        logging.info("Warming up the dual-arm VLA-Precision OpenPI model")
        started = time.time()
        policy.infer(obs)
        logging.info("Model warmup completed, took %.2fs", time.time() - started)
        manual_close_arms = [arm for arm in ARMS if not self.arms[arm].init_gripper]
        if manual_close_arms:
            input(f"Press Enter to close uninitialized grippers ({', '.join(manual_close_arms)})...")
            for arm in manual_close_arms:
                self.arms[arm].target_gripper_position = 0.0
        input("Press Enter to continue inference...")
        return policy

    def _start_episode(self):
        super()._start_episode()
        for arm in ARMS:
            self.arms[arm].zero_force_lock_active = False

    def _reset_before_exit(self):
        if any(self.arms[arm].rtde_r is None or self.arms[arm].rtde_c is None for arm in ARMS):
            return
        try:
            logging.info("[RESET] Resetting both robots before exit.")
            self.reset_episode()
        except Exception as exc:
            logging.error("[ERROR] Failed to reset dual robots before exit: %s", exc)

    def run(self):
        try:
            super().run()
        finally:
            self._arm_executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run dual UR5e inference with an OpenPI VLA-Precision checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "dual_ur.yaml",
        help="Path to the dual-UR VLA-Precision inference YAML config.",
    )
    args = parser.parse_args()
    DualInference(args.config).run()


if __name__ == "__main__":
    main()
