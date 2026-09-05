"""UR RTDE force controller, gripper queue and HTTP service.

This is a behavior-preserving migration of the paper implementation's physical
UR service.  Hardware libraries are imported only when the service starts so
configuration and contract tests remain runnable without a robot installation.
"""

from __future__ import annotations

import logging
import socket
import time
from collections import deque
from dataclasses import dataclass
from threading import Event, Lock, RLock, Thread
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from vla_precision.config import ResolvedConfig
from vla_precision.config.schema import (
    RootConfig,
    URArmServerConfig,
    URGripperSwitchConfig,
    URLowLevelGripperConfig,
)
from vla_precision.robotics.servers.ur.rotations import (
    euler_to_quaternion,
    euler_to_rotvec,
    rotvec_pose_to_quaternion_pose,
)
from vla_precision.runtime_identity import runtime_code_identity

LOGGER = logging.getLogger(__name__)
_DEFAULT_RTDE_LOCK = RLock()


@dataclass(frozen=True)
class _RuntimeGripperSwitch:
    enabled: bool
    min_hold_sec: float
    timeout_sec: float
    stable_window: int
    stable_threshold: float
    poll_sec: float


@dataclass(frozen=True)
class _RuntimeGripper:
    init: bool
    kind: str
    port: str
    force: int
    speed: int
    min_position: int
    max_position: int


@dataclass(frozen=True)
class _RuntimeArm:
    name: str
    robot_ip: str
    init_pose: tuple[float, ...]
    base_pose: tuple[float, ...]
    control_frame_euler_deg: tuple[float, ...]
    selection_vector: tuple[int, ...]
    force_type: int
    force_limits: tuple[float, ...]
    kp: float
    kd: float
    kp_rot: float
    kd_rot: float
    pos_delta: float
    vel_delta: float
    rot_delta: float
    reset_pos_delta: float
    reset_rot_delta: float
    force_gain_scale: float | None
    error_delta: float
    payload_mass: float
    payload_cog: tuple[float, ...]
    gripper: _RuntimeGripper


@dataclass(frozen=True)
class _RuntimeServer:
    dual_arm: bool
    action_fps: float
    debug: bool
    strict_distributed_consistency: bool | None
    init_velocity: float
    init_acceleration: float
    rt_receive_priority: int
    rt_control_priority: int
    dashboard_port: int
    ur_cap_port: int
    bind_host: str
    bind_port: int
    shared_sha256: str
    calibration_verified: bool
    gripper_switch: _RuntimeGripperSwitch
    left_arm: _RuntimeArm
    right_arm: _RuntimeArm | None


def ur_server_contract(runtime: _RuntimeServer) -> dict[str, Any]:
    """Return the distributed facts every environment client must match."""
    return {
        "shared_sha256": runtime.shared_sha256,
        "runtime_code_identity": runtime_code_identity().to_dict(),
        "strict_distributed_consistency": runtime.strict_distributed_consistency,
    }


def split_ur_action_chunk(
    actions: Any,
    *,
    dual_arm: bool,
    gripper_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate an HTTP action payload and preserve the left-then-right order."""
    chunk = np.asarray(actions, dtype=np.float64)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    expected_dim = 14 if dual_arm else (7 if gripper_enabled else 6)
    if chunk.ndim != 2 or chunk.shape[1] != expected_dim:
        raise ValueError(
            f"expected action chunk shape (T, {expected_dim}), got {chunk.shape}"
        )
    if dual_arm:
        return chunk, chunk[:, :7], chunk[:, 7:14]
    return chunk, chunk, None


def _runtime_gripper(
    root: RootConfig,
    low_level: URLowLevelGripperConfig,
    *,
    side: str,
    enabled: bool,
) -> _RuntimeGripper:
    device = (
        root.gripper.left_device or root.gripper.device or ""
        if side == "left"
        else root.gripper.right_device or ""
    )
    return _RuntimeGripper(
        init=enabled,
        kind="pgi",
        port=device,
        force=low_level.force,
        speed=low_level.speed,
        min_position=low_level.min_position,
        max_position=low_level.max_position,
    )


def _runtime_arm(
    root: RootConfig,
    config: URArmServerConfig,
    *,
    name: str,
    gripper: _RuntimeGripper,
) -> _RuntimeArm:
    locators = root.robot_server.locators
    robot_ip = locators.left_robot_ip if name == "left" else locators.right_robot_ip
    init_pose = getattr(root.robot.arms, name).reset_pose
    return _RuntimeArm(
        name=name,
        robot_ip=robot_ip,
        init_pose=tuple(init_pose),
        base_pose=config.base_pose,
        control_frame_euler_deg=config.control_frame_euler_deg,
        selection_vector=config.selection_vector,
        force_type=config.force_type,
        force_limits=config.force_limits,
        kp=config.kp,
        kd=config.kd,
        kp_rot=config.kp_rotation,
        kd_rot=config.kd_rotation,
        pos_delta=config.position_error_clip,
        vel_delta=config.velocity_error_clip,
        rot_delta=config.rotation_error_clip_degrees,
        reset_pos_delta=config.reset_position_error_clip,
        reset_rot_delta=config.reset_rotation_error_clip_degrees,
        force_gain_scale=config.force_gain_scale,
        error_delta=config.controller_error_threshold,
        payload_mass=config.payload_mass,
        payload_cog=config.payload_cog,
        gripper=gripper,
    )


def build_ur_server_runtime(resolved: ResolvedConfig) -> _RuntimeServer:
    """Derive the low-level runtime from the same immutable task configuration."""
    root = resolved.config
    server = root.robot_server
    dual_arm = root.task.arm_mode == "dual"
    gripper_enabled = not root.task.setup_mode.endswith("fixed-gripper")
    left_gripper = _runtime_gripper(
        root,
        root.gripper.left,
        side="left",
        enabled=gripper_enabled,
    )
    right_gripper = _runtime_gripper(
        root,
        root.gripper.right,
        side="right",
        enabled=dual_arm and gripper_enabled,
    )
    switch: URGripperSwitchConfig = server.gripper_switch
    return _RuntimeServer(
        dual_arm=dual_arm,
        action_fps=server.action_fps,
        debug=root.debug,
        strict_distributed_consistency=root.strict_distributed_consistency,
        init_velocity=server.init_velocity,
        init_acceleration=server.init_acceleration,
        rt_receive_priority=server.rt_receive_priority,
        rt_control_priority=server.rt_control_priority,
        dashboard_port=server.locators.dashboard_port,
        ur_cap_port=server.locators.ur_cap_port,
        bind_host=server.locators.bind_host,
        bind_port=server.locators.bind_port,
        shared_sha256=resolved.shared_sha256,
        calibration_verified=server.calibration_verified,
        gripper_switch=_RuntimeGripperSwitch(
            enabled=switch.enabled,
            min_hold_sec=switch.min_hold_seconds,
            timeout_sec=switch.timeout_seconds,
            stable_window=switch.stable_window,
            stable_threshold=switch.stable_threshold,
            poll_sec=switch.poll_seconds,
        ),
        left_arm=_runtime_arm(
            root,
            server.left_arm,
            name="left",
            gripper=left_gripper,
        ),
        right_arm=(
            _runtime_arm(
                root,
                server.right_arm,
                name="right",
                gripper=right_gripper,
            )
            if dual_arm
            else None
        ),
    )

def _load_gripper_server_classes():
    """Import the serial driver only when an enabled PGI device needs it."""
    from vla_precision.robotics.servers.ur.pgi import (
        LeftPGIGripperServer,
        RightPGIGripperServer,
    )

    return LeftPGIGripperServer, RightPGIGripperServer


def _load_rtde_interfaces():
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    return RTDEControlInterface, RTDEReceiveInterface

def set_program_mode(mode, robot_ip, dashboard_port=29999):
    # UR controller dashboard port.
    ur_port = int(dashboard_port)

    # The dashboard interface expects a newline-terminated command.
    command = f"{mode}\n"

    # Open the dashboard connection.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((robot_ip, ur_port))

    # Send the command and close the connection.
    sock.send(command.encode("utf-8"))
    sock.close()

class URArmController:
    def __init__(
            self,
            robot_ip,
            config,
            rtde_frequency = 100.0 ,
            rt_receive_priority = 90 , 
            rt_control_priority = 85 , 
            flags=None, 
            ur_cap_port = 50002 ,
            dashboard_port = 29999,
            arm_name="left",
            base_pose=None,
            init_pose=None,
            init_velocity=None,
            init_acceleration=None,
            gripper_switch_config=None,
            suppress_actions_during_gripper_hold=False,
            rtde_lock=None,
    ):
        self.robot_ip = robot_ip
        self.arm_name = str(arm_name)
        self.rtde_frequency = rtde_frequency
        self._stop_event = Event()
        self._close_lock = Lock()
        self._closed = False
        self._worker_threads: list[Thread] = []
        self.rt_receive_priority = rt_receive_priority
        self.rt_control_priority = rt_control_priority
        RTDEControl, RTDEReceive = _load_rtde_interfaces()
        if flags is None:
            flags = RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT
        self.flags = flags
        self.ur_cap_port = ur_cap_port
        self.gripper_switch_config = gripper_switch_config
        self.suppress_actions_during_gripper_hold = bool(suppress_actions_during_gripper_hold)
        self._init_velocity = float(init_velocity)
        self._init_acceleration = float(init_acceleration)
        control_euler_deg = np.asarray(
            config.control_frame_euler_deg,
            dtype=np.float64,
        )
        if control_euler_deg.shape != (3,) or not np.all(np.isfinite(control_euler_deg)):
            raise ValueError(
                f"{self.arm_name} control_frame_euler_deg must contain three finite values, "
                f"got {control_euler_deg.tolist()}"
            )
        self.control_frame_euler_deg = control_euler_deg
        self.control_to_base_rotation = R.from_euler("xyz", control_euler_deg, degrees=True).as_matrix()
        control_rotvec = R.from_matrix(self.control_to_base_rotation).as_rotvec()
        self.task_frame = [0.0, 0.0, 0.0, *control_rotvec.tolist()]
        self.selection_vector = config.selection_vector
        self.force_type = config.force_type
        self.limits = config.force_limits
        self.kd = config.kd
        self.kp = config.kp
        self.gain_scale = config.force_gain_scale
        self.pos_delta = config.pos_delta
        self.vel_delta = config.vel_delta
        self.rot_delta = config.rot_delta
        self.reset_pos_delta = config.reset_pos_delta
        self.reset_rot_delta = config.reset_rot_delta
        self.Kp_rot = config.kp_rot
        self.Kd_rot = config.kd_rot
        self.delta = config.error_delta
        self.curr_force_lowpass = np.zeros((6,), dtype=np.float32)
        self.curr_pose = np.zeros(7)
        self.curr_basedpose = np.zeros(7)
        self.curr_vel = np.zeros(6)
        self.curr_q = np.zeros(6)
        self.curr_qd = np.zeros(6)
        self.curr_force = np.zeros(6)
        self.rtde_lock = _DEFAULT_RTDE_LOCK if rtde_lock is None else rtde_lock
        self._state_lock = Lock()
        self._target_lock = Lock()
        self._gripper_hold_lock = Lock()
        self._idle_hold_lock = Lock()
        self._gripper_hold_active = False
        self._gripper_hold_pose = None
        self._action_idle_hold_pose = None
        self._action_idle_prev_motion_mask = np.zeros(6, dtype=bool)
        self.idle_action_hold_enabled = True
        self.idle_action_position_threshold = 0.0001
        self.idle_action_rotation_threshold = 0.001
        self.force_pause = False
        self.task_mode = False
        self.mass = config.payload_mass
        self.cx, self.cy, self.cz = config.payload_cog
        set_program_mode("stop", self.robot_ip, dashboard_port)
        time.sleep(1)
        with self.rtde_lock:
            self.rtde_r = RTDEReceive(robot_ip, rtde_frequency, [], True, False, rt_receive_priority)
            self.rtde_c = RTDEControl(robot_ip, rtde_frequency, flags, ur_cap_port, rt_control_priority)

        if self.gain_scale is not None:
            self.rtde_c.forceModeSetGainScaling(self.gain_scale)
        self.rtde_c.setPayload(self.mass, [self.cx, self.cy, self.cz])
        
        self.left_basepose = np.asarray(
            [0, 0, 0, 0, 0, 0, 1] if base_pose is None else base_pose,
            dtype=np.float64,
        )
        if self.left_basepose.shape != (7,):
            raise ValueError(f"{self.arm_name} base_pose must have 7 values, got {self.left_basepose.shape}")

        if init_pose is None:
            raise ValueError(f"{self.arm_name} init_pose must be passed by the server entrypoint")
        pose_vals = [float(x) for x in init_pose]
        if len(pose_vals) != 6:
            raise ValueError(f"{self.arm_name} init_pose must contain 6 values, got {len(pose_vals)}")

        # init_pose is xyz + euler(xyz) -> store raw euler pose and precompute rotvec-based pose for moveL
        self._init_pose_euler = pose_vals
        euler = np.array(pose_vals[3:], dtype=float)

        # moveL needs xyz + rotvec
        rotvec = euler_to_rotvec(euler)
        self._init_pose_raw = [pose_vals[0], pose_vals[1], pose_vals[2], rotvec[0], rotvec[1], rotvec[2]]

        # internal controller uses xyz + quat
        quat = euler_to_quaternion(euler)  # [x,y,z,w]
        local_init_quat_pose = np.array(
            [pose_vals[0], pose_vals[1], pose_vals[2], quat[0], quat[1], quat[2], quat[3]],
            dtype=np.float64,
        )
        self.left_target_pos = self.compose_poses(self.left_basepose, local_init_quat_pose)
        self.left_target_actions = deque()
        self._action_chunk_id = 0
        self._action_chunk_consumed = 0
        self._action_chunk_total = 0
        self._last_consumed_action_meta = None
        self._last_action_chunk_force_done = {
            "id": 0,
            "consumed": 0,
            "total": 0,
            "force_done_time": None,
            "move_elapsed": None,
        }

        try:
            update_left_thread = Thread(target=self._update_state_loop, daemon=True)
            self._worker_threads.append(update_left_thread)
            update_left_thread.start()
            # perform robot reset (moveL to init pose) before starting RTDE control
            self.robot_reset()

            # Start the RTDE control thread.
            left_rtde_thread = Thread(target=self.run_left_rtde_control, daemon=True)
            # right_rtde_thread = Thread(target=run_right_rtde_control, args=(right_controller,), daemon=True)
            self._worker_threads.append(left_rtde_thread)
            left_rtde_thread.start()
            # right_rtde_thread.start()
        except BaseException:
            self.close()
            raise

    def reset_payload(self):
        """Reapply this arm's configured payload to the UR controller."""
        mass = float(self.mass)
        cog = [float(self.cx), float(self.cy), float(self.cz)]
        with self.rtde_lock:
            result = self.rtde_c.setPayload(mass, cog)
        if isinstance(result, (bool, np.bool_)) and not bool(result):
            raise RuntimeError(f"{self.arm_name} RTDE setPayload returned False")
        return {"mass": mass, "cog": cog}

    def zero_ft_sensor(self):
        """Zero this arm's built-in force/torque sensor."""
        with self.rtde_lock:
            result = self.rtde_c.zeroFtSensor()
        if isinstance(result, (bool, np.bool_)) and not bool(result):
            raise RuntimeError(f"{self.arm_name} RTDE zeroFtSensor returned False")
        return {"zeroed": True}

    def get_server_config(self):
        return {
            "arm_name": self.arm_name,
            "base_pose": self.left_basepose.tolist(),
            "init_pose": list(self._init_pose_euler),
            "task_frame": list(self.task_frame),
            "control_frame_euler_deg": self.control_frame_euler_deg.tolist(),
            "select_vector": list(self.selection_vector),
            "selection_vector": list(self.selection_vector),
            # forceMode's select vector is expressed in the control task frame;
            # it is not a component-wise mask for TCP-local policy deltas.
            "delta_action_mask": [1, 1, 1, 1, 1, 1],
            "force_type": int(self.force_type),
            "limits": list(self.limits),
            "kp": float(self.kp),
            "kd": float(self.kd),
            "kp_rot": float(self.Kp_rot),
            "kd_rot": float(self.Kd_rot),
            "pos_delta": float(self.pos_delta),
            "vel_delta": float(self.vel_delta),
            "rot_delta": float(self.rot_delta),
            "reset_pos_delta": float(self.reset_pos_delta),
            "reset_rot_delta": float(self.reset_rot_delta),
            "force_gain_scale": None if self.gain_scale is None else float(self.gain_scale),
            "error_delta": float(self.delta),
            "idle_hold_enabled": bool(self.idle_action_hold_enabled),
            "idle_position_threshold": float(self.idle_action_position_threshold),
            "idle_rotation_threshold": float(self.idle_action_rotation_threshold),
            "payload_mass": float(self.mass),
            "payload_cog": [float(self.cx), float(self.cy), float(self.cz)],
        }

    def mask_delta_actions(self, actions):
        """Keep policy TCP deltas intact; forceMode applies selection in task-frame axes."""
        actions = np.asarray(actions, dtype=np.float64)
        return actions.copy()

    def _wrench_base_to_task(self, wrench_base):
        wrench_base = np.asarray(wrench_base, dtype=np.float64)
        if wrench_base.shape != (6,):
            raise ValueError(f"{self.arm_name} base-frame wrench must be 6D, got {wrench_base.shape}")
        base_to_control = self.control_to_base_rotation.T
        return np.concatenate(
            (base_to_control @ wrench_base[:3], base_to_control @ wrench_base[3:])
        )
    

    def reset_joint(self):
        with self.rtde_lock:
            self.rtde_c.moveJ(self.rest_joint)

    def robot_reset(self):
        """Reset robot by moving linearly to the init pose using moveL.

        Reset pose, velocity, and acceleration are supplied by the entrypoint.
        Once motion is complete, reapply payload compensation and zero F/T.
        """
        with self.rtde_lock:
            try:
                LOGGER.info(
                    "Resetting %s robot to init pose (rotvec) %s with v=%s, a=%s",
                    self.arm_name,
                    self._init_pose_raw,
                    self._init_velocity,
                    self._init_acceleration,
                )
                # use the original rotvec-based pose for moveL (x,y,z,rx,ry,rz)
                self.rtde_c.moveL(self._init_pose_raw, self._init_velocity, self._init_acceleration)
            except Exception as e:
                LOGGER.error(f"robot_reset moveL failed: {e}")
                return
        self.reset_payload()
        self.zero_ft_sensor()
        LOGGER.info("Reset %s payload and F/T sensor after robot reset", self.arm_name)

    def get_pos(self):
        with self.rtde_lock:
            self.curr_pose = rotvec_pose_to_quaternion_pose(self.rtde_r.getActualTCPPose())
            self.curr_basedpose = self.compose_poses(self.left_basepose, self.curr_pose)
        return(self.curr_pose)
    
    def get_vel(self):
        with self.rtde_lock:
            self.curr_vel = self.rtde_r.getActualTCPSpeed()
        return(self.curr_vel)
    
    def get_q(self):
        with self.rtde_lock:
            self.curr_q = self.rtde_r.getActualQ()
        return(self.curr_q)
    
    def get_qd(self):
        with self.rtde_lock:
            self.curr_qd = self.rtde_r.getActualQd()
        return(self.curr_qd)
    
    def get_force(self):
        with self.rtde_lock:
            self.curr_force = self.rtde_r.getActualTCPForce()
        return(self.curr_force)
    
    def request(self, name, param: bool):
        import requests

        requests.post(self.url + name, json={name: param})

    def compose_poses(self, Q1, Q2):
        # Split positions and quaternions.
        pos1, quat1 = Q1[:3], Q1[3:]
        pos2, quat2 = Q2[:3], Q2[3:]
        
        # Compose rotations in the established Q1-then-Q2 order.
        rot1 = R.from_quat(quat1)
        rot2 = R.from_quat(quat2)
        rot_AC = rot1 * rot2
        quat_AC = rot_AC.as_quat()
        
        # Compose positions in Q1's frame.
        pos_AC = pos1 + rot1.apply(pos2)
        
        return np.concatenate([pos_AC, quat_AC])
    
    
    def decompose_poses(self, Q_AC, Q1):
        # Split the composed pose and Q1 into positions and quaternions.
        pos_AC, quat_AC = Q_AC[:3], Q_AC[3:]
        pos1, quat1 = Q1[:3], Q1[3:]
        
        # Recover Q2's rotation from the composed rotation.
        rot_AC = R.from_quat(quat_AC)
        rot1 = R.from_quat(quat1)
        rot2 = rot1.inv() * rot_AC  
        quat2 = rot2.as_quat()
        
        # Recover Q2's position in Q1's frame.
        pos2 = rot1.inv().apply(pos_AC - pos1)
        
        return np.concatenate([pos2, quat2])
    

    def _calculate_force_running(self, target_pos, current_pose=None, current_vel=None):
        kp, kd = self.kp, self.kd
        target = np.asarray(target_pos, dtype=np.float64)
        curr = np.asarray(self.curr_pose if current_pose is None else current_pose, dtype=np.float64)
        curr_vel = np.asarray(self.curr_vel if current_vel is None else current_vel, dtype=np.float64)

        # ---- position error (clip) ----
        pos_err = target[:3] - curr[:3]
        pos_clip = np.array(self.pos_delta)
        diff_p = np.clip(pos_err, -pos_clip, pos_clip)

        vel_clip = np.asarray(self.vel_delta, dtype=np.float64)
        diff_d = np.clip(-curr_vel[:3], -vel_clip, vel_clip)

        force = kp * diff_p + kd * diff_d

        # ---- rotation error (angle clip) ----
        rotvec = (R.from_quat(target[3:]) * R.from_quat(curr[3:]).inv()).as_rotvec()

        angle = np.linalg.norm(rotvec)
        max_angle = np.deg2rad(self.rot_delta)  # ° → rad
        if angle > max_angle and angle > 1e-8:
            rotvec = rotvec / angle * max_angle

        rot_vel = R.from_rotvec(curr_vel[3:]).inv().as_rotvec()

        torque = (self.Kp_rot * rotvec + self.Kd_rot * rot_vel) / self.rtde_frequency

        return self._wrench_base_to_task(np.concatenate((force, torque)))

    def _calculate_force_reset(self, target_pos, current_pose=None, current_vel=None):
        kp, kd = self.kp, self.kd
        target = np.asarray(target_pos, dtype=np.float64)
        curr = np.asarray(self.curr_pose if current_pose is None else current_pose, dtype=np.float64)
        curr_vel = np.asarray(self.curr_vel if current_vel is None else current_vel, dtype=np.float64)

        # ---- position error (clip) ----
        pos_err = target[:3] - curr[:3]
        pos_clip = np.array(self.reset_pos_delta)
        diff_p = np.clip(pos_err, -pos_clip, pos_clip)

        vel_clip = np.asarray(self.vel_delta, dtype=np.float64)
        diff_d = np.clip(-curr_vel[:3], -vel_clip, vel_clip)

        force = kp * diff_p + kd * diff_d

        # ---- rotation error (angle clip) ----
        rotvec = (R.from_quat(target[3:]) * R.from_quat(curr[3:]).inv()).as_rotvec()

        angle = np.linalg.norm(rotvec)
        max_angle = np.deg2rad(self.reset_rot_delta)  # ° → rad
        if angle > max_angle and angle > 1e-8:
            rotvec = rotvec / angle * max_angle

        rot_vel = R.from_rotvec(curr_vel[3:]).inv().as_rotvec()

        torque = (self.Kp_rot * rotvec + self.Kd_rot * rot_vel) / self.rtde_frequency

        return self._wrench_base_to_task(np.concatenate((force, torque)))

    def move(self, pose, current_pose=None, current_vel=None):
        pos = self.decompose_poses(pose, self.left_basepose)
        if self.task_mode:
            force = self._calculate_force_running(pos, current_pose, current_vel)
        else:
            force = self._calculate_force_reset(pos, current_pose, current_vel)
        with self.rtde_lock:
            if self._stop_event.is_set():
                return
            if self.force_pause:
                self.rtde_c.forceMode(self.task_frame,[0, 0, 0, 0, 0, 0],np.array([0,0,0,0,0,0]),self.force_type,self.limits)
            else:
                self.rtde_c.forceMode(self.task_frame,self.selection_vector,force,self.force_type,self.limits)
            sleep_seconds = max(0.0, 1.0 / self.rtde_frequency)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            # self.rtde_c.forceModeStop()
            # self.rtde_c.stopScript()
        # with self.rtde_lock:
        #         self.rtde_c.moveL(pos)

    def close(self) -> None:
        """Leave UR force mode and close RTDE connections exactly once."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        LOGGER.info("stopping %s UR controller and leaving force mode", self.arm_name)
        self._stop_event.set()
        with self.rtde_lock:
            try:
                self.rtde_c.forceModeStop()
            except Exception:
                LOGGER.exception("failed to stop %s UR force mode", self.arm_name)
            try:
                self.rtde_c.stopScript()
            except Exception:
                LOGGER.exception("failed to stop %s UR RTDE control script", self.arm_name)

        for thread in self._worker_threads:
            thread.join(timeout=1.0)

        with self.rtde_lock:
            for name, interface in (("control", self.rtde_c), ("receive", self.rtde_r)):
                try:
                    interface.disconnect()
                except Exception:
                    LOGGER.exception("failed to disconnect %s UR RTDE %s interface", self.arm_name, name)

    @staticmethod
    def _normalize_quat(quat):
        quat = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    def _base_motion_delta_from_tcp_action(self, action, current_rot):
        """Express a TCP-local 6D delta in the base frame used by hold anchors."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] < 6:
            raise ValueError(f"action must have at least 6 values, got {action.shape}")
        delta_pos_base = current_rot.apply(action[:3])
        delta_rot_tcp = R.from_euler("xyz", action[3:6])
        delta_rot_base = current_rot * delta_rot_tcp * current_rot.inv()
        return np.concatenate((delta_pos_base, delta_rot_base.as_rotvec()))

    def _base_motion_mask(self, base_motion_delta):
        """Return per-axis translation activity and one shared rotation state.

        Hold anchors are base-frame poses, so translation is tested per base
        axis.  Orientation is deliberately atomic: rotvec components are a
        coupled representation and must never be independently latched.
        """
        base_motion_delta = np.asarray(base_motion_delta, dtype=np.float64)
        if base_motion_delta.shape != (6,):
            raise ValueError(f"base motion delta must have shape (6,), got {base_motion_delta.shape}")
        if not self.idle_action_hold_enabled:
            return np.ones(6, dtype=bool)
        translation_mask = (
            np.ones(3, dtype=bool)
            if self.idle_action_position_threshold < 0
            else np.abs(base_motion_delta[:3]) > self.idle_action_position_threshold
        )
        rotation_active = (
            True
            if self.idle_action_rotation_threshold < 0
            else np.linalg.norm(base_motion_delta[3:]) > self.idle_action_rotation_threshold
        )
        return np.concatenate((translation_mask, np.repeat(rotation_active, 3)))

    def _clear_action_idle_hold(self):
        with self._idle_hold_lock:
            self._action_idle_hold_pose = None
            self._action_idle_prev_motion_mask = np.zeros(6, dtype=bool)

    @staticmethod
    def _quat_pose_to_rotvec_pose(pose):
        pose = np.asarray(pose, dtype=np.float64)
        return np.concatenate((pose[:3], R.from_quat(pose[3:]).as_rotvec()))

    @staticmethod
    def _rotvec_pose_to_quat_pose(pose):
        pose = np.asarray(pose, dtype=np.float64)
        return np.concatenate((pose[:3], R.from_rotvec(pose[3:]).as_quat()))

    def _target_pose_from_tcp_action(self, action, current_basedpose=None):
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] < 6:
            raise ValueError(f"action must have at least 6 values, got {action.shape}")

        if current_basedpose is None:
            with self._state_lock:
                current = np.asarray(self.curr_basedpose, dtype=np.float64).copy()
        else:
            current = np.asarray(current_basedpose, dtype=np.float64).copy()
        current_rot = R.from_quat(self._normalize_quat(current[3:]))
        delta_pos = action[:3]
        delta_rot = R.from_euler("xyz", action[3:6])
        target_pos = current[:3] + current_rot.apply(delta_pos)
        target_quat = (current_rot * delta_rot).as_quat()
        action_target = np.concatenate((target_pos, target_quat))

        if not self.idle_action_hold_enabled or (
            self.idle_action_position_threshold < 0
            and self.idle_action_rotation_threshold < 0
        ):
            self._clear_action_idle_hold()
            return action_target

        current_pose6 = self._quat_pose_to_rotvec_pose(current)
        action_target6 = self._quat_pose_to_rotvec_pose(action_target)
        # Hold anchors and action_target6 are expressed in the robot base frame.
        # Translation is held per base axis. Rotation is held as one complete
        # orientation, because independently splicing base rotvec components
        # does not represent independent physical rotation axes.
        base_motion_delta = self._base_motion_delta_from_tcp_action(action, current_rot)
        motion_mask = self._base_motion_mask(base_motion_delta)
        with self._idle_hold_lock:
            if self._action_idle_hold_pose is None:
                self._action_idle_hold_pose = current_pose6.copy()
            stopped_mask = self._action_idle_prev_motion_mask & ~motion_mask
            self._action_idle_hold_pose[stopped_mask] = current_pose6[stopped_mask]
            target_pose6 = self._action_idle_hold_pose.copy()
            target_pose6[motion_mask] = action_target6[motion_mask]
            self._action_idle_hold_pose[motion_mask] = current_pose6[motion_mask]
            self._action_idle_prev_motion_mask = motion_mask.copy()
        return self._rotvec_pose_to_quat_pose(target_pose6)

    def set_target_pose(self, pose):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.ndim == 1 and pose.shape[0] == 7:
            target = pose.copy()
        elif pose.ndim == 2 and pose.shape[1] == 7:
            target = pose.copy()
        else:
            raise ValueError(f"target pose must have shape (7,) or (T, 7), got {pose.shape}")
        with self._target_lock:
            self.left_target_actions.clear()
            self._clear_action_idle_hold()
            self.left_target_pos = target
            self._action_chunk_total = 0
            self._last_consumed_action_meta = None
    def set_target_actions(
        self,
        actions,
        idle_hold_enabled=None,
        idle_position_threshold=None,
        idle_rotation_threshold=None,
    ):
        if idle_hold_enabled is not None:
            self.idle_action_hold_enabled = bool(idle_hold_enabled)
            if not self.idle_action_hold_enabled:
                self._clear_action_idle_hold()
        if idle_position_threshold is not None:
            self.idle_action_position_threshold = float(idle_position_threshold)
        if idle_rotation_threshold is not None:
            self.idle_action_rotation_threshold = float(idle_rotation_threshold)
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1 and actions.shape[0] >= 6:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] < 6:
            raise ValueError(f"action arr must have shape (A,) or (T, A>=6), got {actions.shape}")
        actions = self.mask_delta_actions(actions)
        with self._target_lock:
            self.left_target_actions.clear()
            for action in actions:
                self.left_target_actions.append(action.copy())
            self._action_chunk_id += 1
            self._action_chunk_consumed = 0
            self._action_chunk_total = len(self.left_target_actions)
            self._last_consumed_action_meta = None
            chunk_id = self._action_chunk_id
        return chunk_id

    def _current_tcp_pose_for_hold(self):
        with self._state_lock:
            current = np.asarray(self.curr_basedpose, dtype=np.float64).copy()
        if current.shape[0] != 7 or np.linalg.norm(current[3:]) < 1e-8:
            with self._target_lock:
                target = np.asarray(self.left_target_pos, dtype=np.float64).copy()
            current = target[0].copy() if target.ndim == 2 else target.copy()
        current = np.asarray(current, dtype=np.float64).copy()
        current[3:] = self._normalize_quat(current[3:])
        return current

    def begin_gripper_switch_hold(self, command=None):
        del command
        if not self.gripper_switch_config.enabled:
            return False
        hold_pose = self._current_tcp_pose_for_hold()
        # The next model action must establish fresh per-axis hold anchors at
        # the post-gripper actual pose, not reuse anchors from before the hold.
        self._clear_action_idle_hold()
        with self._gripper_hold_lock:
            self._gripper_hold_active = True
            self._gripper_hold_pose = hold_pose
        with self._target_lock:
            self.left_target_pos = hold_pose.copy()
        return True

    def end_gripper_switch_hold(self, reason="done"):
        del reason
        with self._gripper_hold_lock:
            if not self._gripper_hold_active:
                return False
            self._gripper_hold_active = False
            self._gripper_hold_pose = None
        current = self._current_tcp_pose_for_hold()
        with self._target_lock:
            self.left_target_pos = current.copy()
        return True

    def get_gripper_switch_hold_pose(self):
        with self._gripper_hold_lock:
            if not self._gripper_hold_active or self._gripper_hold_pose is None:
                return None
            return np.asarray(self._gripper_hold_pose, dtype=np.float64).copy()

    def is_gripper_switch_hold_active(self):
        with self._gripper_hold_lock:
            return bool(self._gripper_hold_active)

    def _consume_action_locked(self):
        action = self.left_target_actions.popleft() if self.left_target_actions else None
        if action is None:
            self._last_consumed_action_meta = None
            return None

        self._action_chunk_consumed += 1
        self._last_consumed_action_meta = {
            "id": self._action_chunk_id,
            "consumed": self._action_chunk_consumed,
            "remaining": len(self.left_target_actions),
            "total": self._action_chunk_total,
            "consume_time": time.time(),
        }
        return np.asarray(action, dtype=np.float64)

    def consume_target_pose(self, current_basedpose=None):
        with self._target_lock:
            action = self._consume_action_locked()
        if action is not None:
            target = self._target_pose_from_tcp_action(action, current_basedpose=current_basedpose)
            with self._target_lock:
                self.left_target_pos = target.copy()
            return target

        with self._target_lock:
            target = np.asarray(self.left_target_pos, dtype=np.float64)
            if target.ndim == 2:
                current = target[0].copy()
                self.left_target_pos = target[1:].copy() if target.shape[0] > 1 else current.copy()
                return current
            current = target.copy()
            return current

    def consume_target_pose_during_gripper_hold(self, hold_pose):
        """Discard one stale pose row while this arm holds its TCP pose."""
        hold_pose = np.asarray(hold_pose, dtype=np.float64).copy()
        with self._target_lock:
            action = self._consume_action_locked()
            self.left_target_pos = hold_pose.copy()
        return hold_pose, action is not None

    def _control_state_snapshot(self):
        """Atomically snapshot the state used by one target-to-force cycle."""
        with self._state_lock:
            return (
                np.asarray(self.curr_pose, dtype=np.float64).copy(),
                np.asarray(self.curr_basedpose, dtype=np.float64).copy(),
                np.asarray(self.curr_vel, dtype=np.float64).copy(),
            )
    
    def mark_action_move_done(self, meta, move_elapsed):
        if not meta or int(meta.get("remaining", 0)) != 0:
            return
        done_time = time.time()
        info = {
            "id": int(meta.get("id", 0)),
            "consumed": int(meta.get("consumed", 0)),
            "total": int(meta.get("total", 0)),
            "force_done_time": done_time,
            "move_elapsed": float(move_elapsed),
            "consume_time": float(meta.get("consume_time", done_time)),
        }
        with self._target_lock:
            if info["id"] < int(self._last_action_chunk_force_done.get("id", 0)):
                return
            self._last_action_chunk_force_done = info

    def get_action_status(self):
        with self._target_lock:
            return {
                "current_id": int(self._action_chunk_id),
                "consumed": int(self._action_chunk_consumed),
                "total": int(self._action_chunk_total),
                "queued": int(len(self.left_target_actions)),
                "last_force_done": dict(self._last_action_chunk_force_done),
            }

    def moveJ(self, joints):
        with self.rtde_lock:
            self.rtde_c.forceModeStop()
            self.rtde_c.moveJ(joints)

    def get_state(self):
        with self._state_lock:
            return{
                "pose": self.curr_basedpose.tolist().copy(),
                "vel": self.curr_vel.tolist().copy(),
                "q": self.curr_q.tolist().copy(),
                "dq": self.curr_qd.tolist().copy(),
                "force": self.curr_force[:3].tolist().copy(),
                "torque": self.curr_force[3:].tolist().copy()
            }
        
    def _update_state_loop(self):
        while not self._stop_event.is_set():
            try:
                with self.rtde_lock:
                    curr_pose_new = rotvec_pose_to_quaternion_pose(self.rtde_r.getActualTCPPose())
                    curr_basedpose_new = self.compose_poses(self.left_basepose, curr_pose_new)
                    curr_vel_new = self.rtde_r.getActualTCPSpeed()
                    curr_q_new = self.rtde_r.getActualQ()
                    curr_qd_new = self.rtde_r.getActualQd()
                    curr_force_new = self.rtde_r.getActualTCPForce()
                with self._state_lock:
                    self.curr_pose = curr_pose_new
                    self.curr_basedpose = curr_basedpose_new
                    self.curr_vel = curr_vel_new
                    self.curr_q = curr_q_new
                    self.curr_qd = curr_qd_new
                    self.curr_force = curr_force_new
                time.sleep(1.0 / self.rtde_frequency)

            except Exception as e:
                LOGGER.error("state update failed: %s", e)
                time.sleep(0.1)
    
    def jacobian1(self):
        # Keep the behavior-anchoring Robotics Toolbox calculation and UR5e
        # standard-DH parameters exactly.  The import is lazy so non-hardware
        # configuration tooling does not require the kinematics dependency.
        from roboticstoolbox import DHRobot, RevoluteDH

        robot = DHRobot(
            [
                RevoluteDH(d=0.1625, a=0, alpha=np.pi / 2),
                RevoluteDH(d=0, a=-0.425, alpha=0),
                RevoluteDH(d=0, a=-0.3922, alpha=0),
                RevoluteDH(d=0.1333, a=0, alpha=np.pi / 2),
                RevoluteDH(d=0.0997, a=0, alpha=-np.pi / 2),
                RevoluteDH(d=0.0996, a=0, alpha=0),
            ]
        )
        return robot.jacob0(self.curr_q)


    def run_left_rtde_control(self):
        """Run the RTDE control thread."""
        LOGGER.info("started %s UR RTDE control loop", self.arm_name)
        while not self._stop_event.is_set():
            try:
                current_pose, current_basedpose, current_vel = self._control_state_snapshot()
                target = self.get_gripper_switch_hold_pose()
                if target is None:
                    target = self.consume_target_pose(current_basedpose=current_basedpose)
                elif self.suppress_actions_during_gripper_hold:
                    target, _ = self.consume_target_pose_during_gripper_hold(target)
                else:
                    with self._target_lock:
                        self._last_consumed_action_meta = None
                with self._target_lock:
                    meta = dict(self._last_consumed_action_meta) if self._last_consumed_action_meta else None
                    self._last_consumed_action_meta = None
                move_start = time.time()
                self.move(target, current_pose=current_pose, current_vel=current_vel)
                self.mark_action_move_done(meta, time.time() - move_start)
            except Exception as e:
                break


def _init_gripper_command_queue(left_gripper):
    left_gripper._command_lock = Lock()
    left_gripper._pending_commands = deque()
    left_gripper._last_gripper_command = None
    left_gripper._switch_hold_controller = None


def _canonical_gripper_command(command):
    if command in ("hold", None):
        return None
    if command in ("open_gripper", "close_gripper", "reset_gripper"):
        return command
    if isinstance(command, dict) and command.get("type") == "move_gripper":
        return {"type": "move_gripper", "position": int(np.clip(int(command["position"]), 0, 1000))}
    raise ValueError(f"unknown gripper command: {command}")


def _gripper_command_key(command):
    if isinstance(command, dict) and command.get("type") == "move_gripper":
        return (command["type"], command["position"])
    return command


def _gripper_command_target_norm(command):
    if command in ("open_gripper", "reset_gripper"):
        return 1.0
    if command == "close_gripper":
        return 0.0
    if isinstance(command, dict) and command.get("type") == "move_gripper":
        return float(np.clip(int(command["position"]), 0, 1000)) / 1000.0
    if isinstance(command, tuple) and len(command) == 2 and command[0] == "move_gripper":
        return float(np.clip(int(command[1]), 0, 1000)) / 1000.0
    return None


def _record_executed_gripper_command(left_gripper, command):
    command_key = _gripper_command_key(command)
    with left_gripper._command_lock:
        left_gripper._last_gripper_command = command_key
        return left_gripper._last_gripper_command


def _normalized_gripper_position(position):
    try:
        value = float(position)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    if value > 1.0:
        value = value / 1000.0
    return float(np.clip(value, 0.0, 1.0))


def clear_gripper_command_queue(left_gripper, reason):
    del reason
    with left_gripper._command_lock:
        dropped = list(left_gripper._pending_commands)
        left_gripper._pending_commands.clear()
    return dropped


def _should_hold_for_gripper_command(left_gripper, controller, command):
    switch_config = controller.gripper_switch_config
    if not switch_config.enabled:
        return False
    target = _gripper_command_target_norm(command)
    if target is None:
        return False
    with left_gripper._command_lock:
        previous_target = _gripper_command_target_norm(left_gripper._last_gripper_command)
    if previous_target is not None and abs(previous_target - target) > 0.5:
        return True
    current = _normalized_gripper_position(getattr(left_gripper, "position", None))
    trigger_threshold = max(0.05, float(switch_config.stable_threshold) * 2.0)
    return current is not None and abs(current - target) > trigger_threshold


def _wait_for_gripper_switch_stable(left_gripper, controller, command):
    switch_config = controller.gripper_switch_config
    min_hold = max(0.0, float(switch_config.min_hold_sec))
    timeout = max(min_hold, float(switch_config.timeout_sec))
    stable_window = max(1, int(switch_config.stable_window))
    stable_threshold = max(0.0, float(switch_config.stable_threshold))
    poll_sec = max(0.001, float(switch_config.poll_sec))
    target = _gripper_command_target_norm(command)
    trigger_threshold = max(0.05, stable_threshold * 2.0)
    movement_threshold = max(0.01, stable_threshold * 2.0)
    initial_pos = None
    observed_motion = False
    positions = deque(maxlen=stable_window)
    start = time.time()
    reason = "timeout"

    while True:
        now = time.time()
        elapsed = now - start
        try:
            left_gripper.get_gripose()
            pos = _normalized_gripper_position(getattr(left_gripper, "position", None))
            if pos is not None:
                if initial_pos is None:
                    initial_pos = pos
                elif abs(pos - initial_pos) >= movement_threshold:
                    observed_motion = True
                positions.append(pos)
        except Exception as exc:
            reason = f"read_error:{type(exc).__name__}"
            LOGGER.warning("gripper switch hold position read failed: %s", exc)
            break

        initial_close_to_target = initial_pos is not None and target is not None and abs(initial_pos - target) <= trigger_threshold
        can_release_on_stable = observed_motion or initial_close_to_target or target is None
        if elapsed >= min_hold and len(positions) >= stable_window and can_release_on_stable:
            if max(positions) - min(positions) <= stable_threshold:
                reason = "stable"
                break
        if elapsed >= timeout:
            reason = "timeout"
            break
        time.sleep(poll_sec)

    controller.end_gripper_switch_hold(reason=reason)
    clear_gripper_command_queue(left_gripper, reason=f"switch_end:{reason}")


def _gripper_switch_hold_active(left_gripper):
    controller = getattr(left_gripper, "_switch_hold_controller", None)
    if controller is None:
        return False
    try:
        return bool(controller.is_gripper_switch_hold_active())
    except Exception:
        return False


def enqueue_gripper_commands(left_gripper, commands):
    queued = []
    hold_active = _gripper_switch_hold_active(left_gripper)
    with left_gripper._command_lock:
        if hold_active:
            left_gripper._pending_commands.clear()
            for raw_command in commands:
                _canonical_gripper_command(raw_command)
        else:
            last_key = (
                _gripper_command_key(left_gripper._pending_commands[-1])
                if left_gripper._pending_commands
                else left_gripper._last_gripper_command
            )
            for raw_command in commands:
                command = _canonical_gripper_command(raw_command)
                if command is None:
                    continue
                command_key = _gripper_command_key(command)
                if command_key == last_key:
                    continue
                left_gripper._pending_commands.append(command)
                queued.append(command)
                last_key = command_key
    return queued


def pop_gripper_command(left_gripper):
    with left_gripper._command_lock:
        if not left_gripper._pending_commands:
            return None
        command = left_gripper._pending_commands.popleft()
    return command


def execute_gripper_command(left_gripper, command, controller=None):
    should_hold = controller is not None and _should_hold_for_gripper_command(left_gripper, controller, command)
    hold_started = False
    if should_hold:
        hold_started = controller.begin_gripper_switch_hold(command=command)
    try:
        if command == "open_gripper":
            left_gripper.open()
        elif command == "close_gripper":
            left_gripper.close()
        elif command == "reset_gripper":
            left_gripper.reset_gripper()
        elif isinstance(command, dict) and command.get("type") == "move_gripper":
            left_gripper.set_position(command["position"], blocking=False)
        else:
            raise ValueError(f"unknown gripper command: {command}")

        _record_executed_gripper_command(left_gripper, command)

        if hold_started:
            _wait_for_gripper_switch_stable(left_gripper, controller, command)
    finally:
        if hold_started:
            if controller.end_gripper_switch_hold(reason="command_done"):
                clear_gripper_command_queue(left_gripper, reason="switch_finally:command_done")


def run_leftgrip_control(left_gripper):
    while True:
        try:
            command = pop_gripper_command(left_gripper)
            if command is not None:
                execute_gripper_command(left_gripper, command, getattr(left_gripper, "_switch_hold_controller", None))
                continue
            left_gripper.get_gripose()
            time.sleep(0.02)
        except Exception:
            LOGGER.exception("UR gripper control loop stopped")
            break
        
def _start_gripper_control(gripper_config, controller, gripper_server_cls, *, arm_name):
    """Construct and start one enabled gripper, or return None without loading hardware."""
    if not bool(gripper_config.init):
        LOGGER.info("%s gripper disabled; driver, serial port, and control thread will not be started", arm_name)
        return None
    gripper_kind = str(getattr(gripper_config, "kind", "pgi"))
    if gripper_kind != "pgi":
        raise ValueError(f"Unsupported gripper kind {gripper_kind!r}; expected 'pgi'")
    LOGGER.info("Starting %s PGI gripper server: gripper_port=%s", arm_name, gripper_config.port)
    gripper = gripper_server_cls(
        init_gripper=True,
        gripper_port=gripper_config.port,
        gripper_force=gripper_config.force,
        gripper_speed=gripper_config.speed,
        min_position=gripper_config.min_position,
        max_position=gripper_config.max_position,
    )
    gripper._gripper_kind = gripper_kind
    _init_gripper_command_queue(gripper)
    gripper._switch_hold_controller = controller
    Thread(target=run_leftgrip_control, args=(gripper,), daemon=True).start()
    return gripper


def wait_until_reached(controller, target, threshold=0.005):
    target = np.array(target[:3]) 

    while True:
        with controller._state_lock:
            curr = np.array(controller.curr_basedpose[:3])

        error = np.linalg.norm(curr - target)
        if error < threshold:
            break
        time.sleep(0.02)

def euler_pose_to_quat_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError("pose must be a 6D xyz+euler pose")
    quat = R.from_euler("xyz", pose[3:]).as_quat()
    return np.concatenate([pose[:3], quat])

def _run_server(server_config, controllers):
    """Run a single- or dual-arm server from an entrypoint-owned config."""
    from flask import Flask, jsonify, request

    dual_arm = bool(server_config.dual_arm)
    webapp = Flask(__name__)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    def validate_arm_config(arm_config):
        prefix = str(arm_config.name)
        selection = [int(x) for x in arm_config.selection_vector]
        control = [float(x) for x in arm_config.control_frame_euler_deg]
        limits = [float(x) for x in arm_config.force_limits]
        cog = [float(x) for x in arm_config.payload_cog]
        init_pose = [float(x) for x in arm_config.init_pose]
        base_pose = [float(x) for x in arm_config.base_pose]
        default_min_position = 80 if prefix == "left" else 0
        gripper_kind = str(getattr(arm_config.gripper, "kind", "pgi"))
        gripper_min_position = int(getattr(arm_config.gripper, "min_position", default_min_position))
        gripper_max_position = int(getattr(arm_config.gripper, "max_position", 1000))
        if len(selection) != 6 or any(value not in (0, 1) for value in selection):
            raise ValueError(
                f"robot_server.{prefix}_arm.selection_vector must contain six 0/1 values, got {selection}"
            )
        if len(control) != 3 or not np.all(np.isfinite(control)):
            raise ValueError(f"--{prefix}_control_frame_euler_deg must contain three finite values")
        if len(limits) != 6 or not np.all(np.isfinite(limits)):
            raise ValueError(f"--{prefix}_force_limits must contain six finite values")
        if len(cog) != 3 or not np.all(np.isfinite(cog)):
            raise ValueError(f"--{prefix}_payload_cog must contain three finite values")
        if len(init_pose) != 6 or not np.all(np.isfinite(init_pose)):
            raise ValueError(f"--{prefix}_init_pose must contain six finite values")
        if len(base_pose) != 7 or not np.all(np.isfinite(base_pose)):
            raise ValueError(f"--{prefix}_base_pose must contain seven finite values")
        if gripper_kind != "pgi":
            raise ValueError(f"--gripper_type must be 'pgi', got {gripper_kind!r}")
        if not 0 <= gripper_min_position < gripper_max_position <= 1000:
            raise ValueError(
                f"--{prefix}_gripper_min_position/--{prefix}_gripper_max_position must satisfy "
                f"0 <= min < max <= 1000, got [{gripper_min_position}, {gripper_max_position}]"
            )
        return arm_config

    left_config = validate_arm_config(server_config.left_arm)
    right_config = None
    if dual_arm:
        if server_config.right_arm is None:
            raise ValueError("dual-arm server config must include right_arm")
        right_config = validate_arm_config(server_config.right_arm)
    if server_config.debug:
        LOGGER.debug(
            "UR server startup: dual_arm=%s bind=%s:%d left_select=%s right_select=%s ",
            dual_arm,
            server_config.bind_host,
            server_config.bind_port,
            left_config.selection_vector,
            right_config.selection_vector if right_config is not None else None,
        )
    left_controller = URArmController(
        robot_ip=left_config.robot_ip,
        config=left_config,
        rtde_frequency=server_config.action_fps,
        rt_receive_priority=server_config.rt_receive_priority,
        rt_control_priority=server_config.rt_control_priority,
        ur_cap_port=server_config.ur_cap_port,
        dashboard_port=server_config.dashboard_port,
        arm_name="left",
        base_pose=left_config.base_pose,
        init_pose=left_config.init_pose,
        init_velocity=server_config.init_velocity,
        init_acceleration=server_config.init_acceleration,
        suppress_actions_during_gripper_hold=dual_arm,
        gripper_switch_config=server_config.gripper_switch,
    )
    controllers.append(left_controller)
    right_controller = None
    if dual_arm:
        right_controller = URArmController(
            robot_ip=right_config.robot_ip,
            config=right_config,
            rtde_frequency=server_config.action_fps,
            rt_receive_priority=server_config.rt_receive_priority,
            rt_control_priority=server_config.rt_control_priority,
            ur_cap_port=server_config.ur_cap_port,
            dashboard_port=server_config.dashboard_port,
            arm_name="right",
            base_pose=right_config.base_pose,
            init_pose=right_config.init_pose,
            init_velocity=server_config.init_velocity,
            init_acceleration=server_config.init_acceleration,
            gripper_switch_config=server_config.gripper_switch,
            suppress_actions_during_gripper_hold=True,
            rtde_lock=RLock(),
        )
        controllers.append(right_controller)

    left_gripper_config = left_config.gripper
    right_gripper_config = right_config.gripper if dual_arm else None
    left_gripper_server_cls = None
    right_gripper_server_cls = None
    if bool(left_gripper_config.init):
        left_gripper_server_cls, _ = _load_gripper_server_classes()
    if dual_arm and bool(right_gripper_config.init):
        _, right_gripper_server_cls = _load_gripper_server_classes()

    left_gripper = _start_gripper_control(
        left_gripper_config,
        left_controller,
        left_gripper_server_cls,
        arm_name="left",
    )
    right_gripper = None
    if dual_arm:
        right_gripper = _start_gripper_control(
            right_gripper_config,
            right_controller,
            right_gripper_server_cls,
            arm_name="right",
        )

    def controller_state(controller, gripper):
        state = {
            "pose": np.asarray(controller.curr_basedpose).tolist(),
            "vel": np.asarray(controller.curr_vel).tolist(),
            "force": np.asarray(controller.curr_force[:3]).tolist(),
            "torque": np.asarray(controller.curr_force[3:]).tolist(),
            "q": np.asarray(controller.curr_q).tolist(),
            "dq": np.asarray(controller.curr_qd).tolist(),
            "jacobian": np.asarray(controller.jacobian1()).tolist(),
        }
        if gripper is not None:
            state["gripper_pos"] = gripper.position
        return state

    def gripper_runtime_config(gripper_config, gripper):
        config = {
            "enabled": gripper is not None,
            "type": str(getattr(gripper_config, "kind", "pgi")),
        }
        if gripper is not None:
            config.update(
                port=gripper_config.port,
                force=int(gripper_config.force),
                speed=int(gripper_config.speed),
                min_position=int(gripper_config.min_position),
                max_position=int(gripper_config.max_position),
            )
        return config

    def gripper_unavailable(side="left"):
        return jsonify({
            "error": f"{side} gripper is disabled; start the server with the corresponding init_gripper flag to enable it",
            "gripper_enabled": False,
        }), 409

    def require_grippers(*, include_right=False):
        if left_gripper is None:
            return gripper_unavailable("left")
        if include_right and right_gripper is None:
            return gripper_unavailable("right")
        return None

    @webapp.route("/action_status", methods=["POST"])
    def action_status():
        left_status = left_controller.get_action_status()
        if not dual_arm:
            return jsonify(left_status)
        right_status = right_controller.get_action_status()
        left_done = left_status["last_force_done"]
        right_done = right_status["last_force_done"]
        combined_done_id = min(int(left_done.get("id", 0)), int(right_done.get("id", 0)))
        return jsonify({
            "current_id": min(int(left_status["current_id"]), int(right_status["current_id"])),
            "consumed": min(int(left_status["consumed"]), int(right_status["consumed"])),
            "total": min(int(left_status["total"]), int(right_status["total"])),
            "queued": max(int(left_status["queued"]), int(right_status["queued"])),
            "last_force_done": {"id": combined_done_id, "force_done_time": min(left_done.get("force_done_time") or 0, right_done.get("force_done_time") or 0)},
            "left": left_status,
            "right": right_status,
        })

    @webapp.route("/server_config", methods=["POST"])
    def get_server_config_route():
        config = left_controller.get_server_config()
        left_runtime_config = left_controller.get_server_config()
        left_runtime_config["gripper"] = gripper_runtime_config(left_config.gripper, left_gripper)
        config["left"] = left_runtime_config
        config["gripper"] = dict(left_runtime_config["gripper"])
        if dual_arm:
            config = dict(config)
            right_runtime_config = right_controller.get_server_config()
            right_runtime_config["gripper"] = gripper_runtime_config(right_config.gripper, right_gripper)
            config["right"] = right_runtime_config
            config["dual_arm"] = True
            config["left_task_frame"] = list(left_controller.task_frame)
            config["right_task_frame"] = list(right_controller.task_frame)
            config["left_control_frame_euler_deg"] = left_controller.control_frame_euler_deg.tolist()
            config["right_control_frame_euler_deg"] = right_controller.control_frame_euler_deg.tolist()
            config["left_select_vector"] = list(left_controller.selection_vector)
            config["right_select_vector"] = list(right_controller.selection_vector)
            config["select_vector"] = list(left_controller.selection_vector) + list(right_controller.selection_vector)
            config["selection_vector"] = config["select_vector"]
            config["delta_action_mask"] = [1] * 12
        else:
            config["dual_arm"] = False
        config["server_debug"] = bool(server_config.debug)
        config.update(ur_server_contract(server_config))
        return jsonify(config)

    @webapp.route("/reset_payload", methods=["POST"])
    def reset_payload():
        """Reapply the startup payload configuration to every active arm."""
        try:
            payloads = {"left": left_controller.reset_payload()}
            if dual_arm:
                payloads["right"] = right_controller.reset_payload()
        except Exception as exc:
            LOGGER.exception("Failed to reset UR payload")
            return jsonify({"success": False, "error": f"{type(exc).__name__}: {exc}"}), 500
        return jsonify({"success": True, "dual_arm": dual_arm, **payloads})

    @webapp.route("/zero_ft_sensor", methods=["POST"])
    def zero_ft_sensor():
        """Zero the built-in force/torque sensor on every active arm."""
        try:
            results = {"left": left_controller.zero_ft_sensor()}
            if dual_arm:
                results["right"] = right_controller.zero_ft_sensor()
        except Exception as exc:
            LOGGER.exception("Failed to zero UR F/T sensor")
            return jsonify({"success": False, "error": f"{type(exc).__name__}: {exc}"}), 500
        return jsonify({"success": True, "dual_arm": dual_arm, **results})

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        xyz = left_controller.curr_basedpose[:3]
        r = R.from_quat(left_controller.curr_basedpose[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})
    
    @webapp.route("/getpos_quat", methods=["POST"])
    def get_pose_quat():
        xyz = left_controller.curr_basedpose[:3]
        quat = left_controller.curr_basedpose[3:]
        return jsonify({"pose": np.concatenate([xyz, quat]).tolist()})

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": np.array(left_controller.curr_basedpose).tolist()})
    
    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(left_controller.curr_vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(left_controller.curr_force[:3]).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(left_controller.curr_force[3:]).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(left_controller.curr_q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(left_controller.curr_qd).tolist()})
    
    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.array(left_controller.jacobian1()).tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        unavailable = require_grippers()
        if unavailable is not None:
            return unavailable
        return jsonify({"gripper": left_gripper.position})

    @webapp.route("/get_gripper_state", methods=["POST"])
    def get_gripper_state():
        unavailable = require_grippers()
        if unavailable is not None:
            return unavailable
        return jsonify({"gripper_state": left_gripper.get_state()})
    
    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        unavailable = require_grippers(include_right=dual_arm)
        if unavailable is not None:
            return unavailable
        queued = enqueue_gripper_commands(left_gripper, ["open_gripper"])
        right_queued = enqueue_gripper_commands(right_gripper, ["open_gripper"]) if dual_arm else []
        return jsonify({"queued": queued, "right_queued": right_queued})

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        unavailable = require_grippers(include_right=dual_arm)
        if unavailable is not None:
            return unavailable
        queued = enqueue_gripper_commands(left_gripper, ["close_gripper"])
        right_queued = enqueue_gripper_commands(right_gripper, ["close_gripper"]) if dual_arm else []
        return jsonify({"queued": queued, "right_queued": right_queued})

    @webapp.route("/set_grippers", methods=["POST"])
    def set_grippers():
        unavailable = require_grippers(include_right=dual_arm)
        if unavailable is not None:
            return unavailable
        data = request.get_json(force=True) or {}
        left_value = int(data.get("left", 1))
        right_value = int(data.get("right", left_value))
        if left_value not in (0, 1) or right_value not in (0, 1):
            return jsonify({"error": "left/right gripper values must be 0 (close) or 1 (open)"}), 400
        left_command = "open_gripper" if left_value else "close_gripper"
        left_queued = enqueue_gripper_commands(left_gripper, [left_command])
        right_queued = []
        if dual_arm:
            right_command = "open_gripper" if right_value else "close_gripper"
            right_queued = enqueue_gripper_commands(right_gripper, [right_command])
        return jsonify({"left_queued": left_queued, "right_queued": right_queued})

    @webapp.route("/open_forced", methods=["POST"])
    def open_forced():
        unavailable = require_grippers()
        if unavailable is not None:
            return unavailable
        queued = enqueue_gripper_commands(left_gripper, ["open_gripper"])
        return jsonify({"queued": queued})

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        unavailable = require_grippers()
        if unavailable is not None:
            return unavailable
        gripper_pos = request.json
        pos = left_gripper.clamp_position(gripper_pos["gripper_pos"])
        queued = enqueue_gripper_commands(left_gripper, [{"type": "move_gripper", "position": pos}])
        return jsonify({"queued": queued})

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        unavailable = require_grippers()
        if unavailable is not None:
            return unavailable
        queued = enqueue_gripper_commands(left_gripper, ["reset_gripper"])
        return jsonify({"queued": queued})

    @webapp.route("/gripper_chunk", methods=["POST"])
    def gripper_chunk():
        unavailable = require_grippers(include_right=dual_arm)
        if unavailable is not None:
            return unavailable
        data = request.get_json(force=True) or {}
        if dual_arm:
            left_commands = data.get("left_commands", [])
            right_commands = data.get("right_commands", [])
            if not isinstance(left_commands, list) or not isinstance(right_commands, list):
                return jsonify({"error": "left_commands and right_commands must be lists"}), 400
            if len(left_commands) != len(right_commands):
                return jsonify({"error": "left/right gripper command chunks must have equal length"}), 400
            try:
                left_queued = enqueue_gripper_commands(left_gripper, left_commands)
                right_queued = enqueue_gripper_commands(right_gripper, right_commands)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            return jsonify({"left_queued": left_queued, "right_queued": right_queued})
        commands = data.get("commands", [])
        if not isinstance(commands, list):
            return jsonify({"error": "commands must be a list"}), 400
        try:
            queued = enqueue_gripper_commands(left_gripper, commands)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"queued": queued})

    @webapp.route("/regrasp", methods=["POST"])
    def regrasp():
        data = request.get_json(force=True) or {}
        regrasp_step_1 = data.get("regrasp_step_1")
        regrasp_step_2 = data.get("regrasp_step_2")

        for name, regrasp_step in (
            ("regrasp_step_1", regrasp_step_1),
            ("regrasp_step_2", regrasp_step_2),
        ):
            if regrasp_step is None:
                continue
            try:
                regrasp_arr = np.asarray(regrasp_step, dtype=np.float64)
                if dual_arm and regrasp_arr.shape == (2, 6):
                    left_pose = euler_pose_to_quat_pose(regrasp_arr[0])
                    right_pose = euler_pose_to_quat_pose(regrasp_arr[1])
                    left_controller.set_target_pose(left_pose)
                    right_controller.set_target_pose(right_pose)
                    left_wait = Thread(target=wait_until_reached, args=(left_controller, left_pose, 0.02))
                    right_wait = Thread(target=wait_until_reached, args=(right_controller, right_pose, 0.02))
                    left_wait.start()
                    right_wait.start()
                    left_wait.join()
                    right_wait.join()
                    continue
                regrasp_pose = euler_pose_to_quat_pose(regrasp_arr)
            except ValueError as exc:
                return jsonify({"error": f"{name} must be shape (6,), dual shape (2, 6), or None: {exc}"}), 400
            left_controller.set_target_pose(regrasp_pose)
            wait_until_reached(left_controller, regrasp_pose, threshold=0.02)

        return "regrasp"
    
    @webapp.route("/force_pause", methods=["POST"])
    def force_pause():
        left_controller.force_pause = request.json["force_pause"]
        if dual_arm:
            right_controller.force_pause = request.json["force_pause"]
        return jsonify({"success": left_controller.force_pause})

    @webapp.route("/task_mode", methods=["POST"])
    def task_mode():
        left_controller.task_mode = request.json["task_mode"]
        if dual_arm:
            right_controller.task_mode = request.json["task_mode"]
        return jsonify({"success": left_controller.task_mode})

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        if dual_arm:
            left_thread = Thread(target=left_controller.robot_reset)
            right_thread = Thread(target=right_controller.robot_reset)
            left_thread.start()
            right_thread.start()
            left_thread.join()
            right_thread.join()
        else:
            left_controller.robot_reset()
        return "Reset Joint"
    
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        if dual_arm:
            return jsonify({
                "left": controller_state(left_controller, left_gripper),
                "right": controller_state(right_controller, right_gripper),
            })
        return jsonify(controller_state(left_controller, left_gripper))
    
    @webapp.route("/pose", methods=["POST"])
    def pose():
        data = request.get_json(force=True) or {}
        pos = np.array(data["arr"], dtype=np.float64)
        if dual_arm:
            if pos.shape != (2, 7):
                return jsonify({"error": f"dual-arm pose arr must have shape (2, 7), got {pos.shape}"}), 400
            left_controller.set_target_pose(pos[0])
            right_controller.set_target_pose(pos[1])
        elif pos.ndim == 1:
            left_controller.set_target_pose(pos)
        elif pos.ndim == 2:
            left_controller.set_target_pose(pos)
        else:
            return jsonify({"error": f"pose arr must be 1D or 2D, got shape {pos.shape}"}), 400
        return "Moved"

    @webapp.route("/action", methods=["POST"])
    def action():
        data = request.get_json(force=True) or {}
        try:
            actions, left_actions, right_actions = split_ur_action_chunk(
                data["arr"],
                dual_arm=dual_arm,
                gripper_enabled=left_gripper is not None,
            )
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        if dual_arm:
            left_actions = left_controller.mask_delta_actions(left_actions)
            right_actions = right_controller.mask_delta_actions(right_actions)
            actions = np.concatenate([left_actions, right_actions], axis=-1)
        else:
            left_actions = left_controller.mask_delta_actions(left_actions)
            actions = left_actions
        idle_position_threshold = data.get("idle_position_threshold", 0.0001)
        idle_rotation_threshold = data.get("idle_rotation_threshold", 0.001)
        try:
            chunk_id = left_controller.set_target_actions(
                left_actions,
                idle_hold_enabled=data.get("idle_hold_enabled", True),
                idle_position_threshold=idle_position_threshold,
                idle_rotation_threshold=idle_rotation_threshold,
            )
            if dual_arm:
                right_chunk_id = right_controller.set_target_actions(
                    right_actions,
                    idle_hold_enabled=data.get("idle_hold_enabled", True),
                    idle_position_threshold=idle_position_threshold,
                    idle_rotation_threshold=idle_rotation_threshold,
                )
                if int(right_chunk_id) != int(chunk_id):
                    raise RuntimeError(f"left/right action chunk ids diverged: {chunk_id} != {right_chunk_id}")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({
            "queued": int(actions.shape[0] if actions.ndim == 2 else 1),
            "action_chunk_id": int(chunk_id),
            "select_vector": list(left_controller.selection_vector) + (list(right_controller.selection_vector) if dual_arm else []),
        })
    

    
    webapp.run(
        host=server_config.bind_host,
        port=server_config.bind_port,
        debug=False,
        threaded=True,
    )


def _validate_hardware_boundary(root: RootConfig, runtime: _RuntimeServer) -> None:
    """Validate only values that could make this hardware launch unsafe."""
    if not runtime.calibration_verified:
        raise RuntimeError(
            "robot_server.calibration_verified is false. Verify this task's UR reset pose, "
            "force controller, selection vector, payload and gripper calibration before "
            "starting physical hardware."
        )
    if root.robot.kind not in ("ur5e", "dual_ur"):
        raise ValueError("serve robot supports only the UR service; Franka uses its external ZeroRPC server")
    if not runtime.left_arm.robot_ip:
        raise ValueError("robot_server.locators.left_robot_ip is required")
    if runtime.dual_arm and (runtime.right_arm is None or not runtime.right_arm.robot_ip):
        raise ValueError("robot_server.locators.right_robot_ip is required for a dual-arm task")
    for arm in (runtime.left_arm, runtime.right_arm if runtime.dual_arm else None):
        if arm is None:
            continue
        if len(arm.init_pose) != 6:
            pose_path = f"robot.arms.{arm.name}.reset_pose"
            raise ValueError(f"{pose_path} must contain six xyz+Euler values")
        if arm.gripper.init and arm.gripper.kind == "pgi" and not arm.gripper.port:
            device_path = "gripper.left_device (or gripper.device)" if arm.name == "left" else "gripper.right_device"
            raise ValueError(f"{device_path} is required for an enabled PGI gripper")


def run_robot_server(resolved: ResolvedConfig) -> None:
    """Launch the task-configured UR service; this is the sole hardware gate."""
    runtime = build_ur_server_runtime(resolved)
    _validate_hardware_boundary(resolved.config, runtime)
    LOGGER.info(
        "starting %s UR robot server on %s:%d shared_sha256=%s",
        "dual-arm" if runtime.dual_arm else "single-arm",
        runtime.bind_host,
        runtime.bind_port,
        runtime.shared_sha256,
    )
    controllers = []
    try:
        _run_server(runtime, controllers)
    finally:
        for controller in reversed(controllers):
            controller.close()
