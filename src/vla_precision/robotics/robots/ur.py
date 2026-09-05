"""HTTP client for the existing single- and dual-UR robot servers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from vla_precision.config.schema import RobotConfig
from vla_precision.robotics.grippers.base import Gripper
from vla_precision.robotics.tasks.reset import ResetProcedure
from vla_precision.runtime_identity import compatible_runtime_identity, consistency_mode, runtime_code_identity

LOGGER = logging.getLogger(__name__)

PAYLOAD_SETTLE_TIME_S = 0.25
FT_ZERO_SETTLE_TIME_S = 0.50
HTTP_CONNECT_TIMEOUT_S = 2.0
HTTP_FAST_READ_TIMEOUT_S = 5.0
HTTP_RESET_READ_TIMEOUT_S = 60.0
HTTP_SENSOR_READ_TIMEOUT_S = 10.0
HTTP_RETRY_INTERVAL_S = 1.0


def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(euler, dtype=np.float64)).as_quat()


class URRobot:
    """Robot-only capability extracted from the prior mixed UR Gym environment."""

    def __init__(
        self,
        config: RobotConfig,
        *,
        gripper: Gripper,
        reset_procedure: ResetProcedure,
        dual_arm: bool = False,
        expected_shared_sha256: str | None = None,
        strict_distributed_consistency: bool | None = True,
        post: Callable[..., Any] | None = None,
        wait_for_operator: Callable[[str], Any] = input,
    ):
        if post is None:
            import requests

            post = requests.post
            self._transport_errors = (requests.ConnectionError, requests.Timeout)
            self._http_error = requests.HTTPError
        else:
            self._transport_errors = (ConnectionError, TimeoutError)
            self._http_error = RuntimeError
        self._post = post
        self._wait_for_operator = wait_for_operator
        self.url = config.server_url.rstrip("/") + "/"
        self.dual_arm = bool(dual_arm)
        self.expected_shared_sha256 = expected_shared_sha256
        self.strict_distributed_consistency = strict_distributed_consistency
        self.expected_runtime_code_identity = runtime_code_identity().to_dict()
        self.gripper = gripper
        self.reset_procedure = reset_procedure
        self.control_hz = float(config.control_hz)
        self.random_reset = bool(config.random_reset)
        arm_names = ("left", "right") if self.dual_arm else ("left",)
        self.reset_poses = {
            name: np.asarray(getattr(config.arms, name).reset_pose, dtype=np.float64)
            for name in arm_names
        }
        self.reset_pose_ranges = {
            name: np.asarray(getattr(config.arms, name).reset_pose_range, dtype=np.float64)
            for name in arm_names
        }
        self.idle_hold_enabled = bool(config.idle_hold_enabled)
        self.idle_position_threshold = float(config.idle_position_threshold)
        self.idle_rotation_threshold = float(config.idle_rotation_threshold)
        self.options = config.options
        options = self.options
        self.action_status_poll = float(options.get("action_chunk_status_poll", 0.002))
        self.action_done_timeout = float(options.get("action_chunk_done_timeout", 1.0))
        self.wait_at_reset = bool(options.get("wait_for_operator", True))
        self.delta_action_mask = np.ones((12 if self.dual_arm else 6,), dtype=np.float64)
        self.delta_action_mask_source = "default"
        self.server_config: dict[str, Any] = {}
        self.last_action_chunk_id: int | None = None
        self._load_server_configuration(config.delta_action_mask)
        self._update_state()

    @property
    def action_dimension(self) -> int:
        if self.dual_arm:
            return 14
        return 6 + self.gripper.action_dimension

    action_low = -3.0
    action_high = 3.0

    @property
    def currpos(self) -> np.ndarray:
        return self._left_state["pose"]

    @property
    def right_currpos(self) -> np.ndarray:
        return self._right_state["pose"]

    def _post_server(
        self,
        endpoint: str,
        *,
        json: Any = None,
        read_timeout: float = HTTP_FAST_READ_TIMEOUT_S,
        retry_transport: bool = False,
    ):
        attempts = 0
        while True:
            try:
                response = self._post(
                    self.url + endpoint.lstrip("/"),
                    json=json,
                    timeout=(HTTP_CONNECT_TIMEOUT_S, float(read_timeout)),
                )
            except self._transport_errors as exc:
                if not retry_transport:
                    raise RuntimeError(
                        f"UR server transport failed during {endpoint}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from None
                attempts += 1
                if attempts == 1 or attempts % 10 == 0:
                    LOGGER.warning("UR server unavailable during %s; retry=%d", endpoint, attempts)
                time.sleep(HTTP_RETRY_INTERVAL_S)
                continue
            try:
                response.raise_for_status()
            except self._http_error as exc:
                status_code = getattr(response, "status_code", "unknown")
                response_text = str(getattr(response, "text", "")).strip()
                if len(response_text) > 2_000:
                    response_text = response_text[:2_000] + "...<truncated>"
                detail = response_text or str(exc)
                raise RuntimeError(
                    f"UR server operation {endpoint!r} failed with HTTP "
                    f"{status_code}: {detail}"
                ) from None
            return response

    def _set_delta_action_mask(self, values, source: str) -> None:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        expected = len(self.delta_action_mask)
        if vector.size < expected:
            raise ValueError(
                f"delta_action_mask needs {expected} values, got {vector.tolist()}"
            )
        self.delta_action_mask = (vector[:expected] != 0.0).astype(np.float64)
        self.delta_action_mask_source = source

    def _check_server_configuration(self, payload: dict) -> None:
        strict = consistency_mode(
            getattr(self, "strict_distributed_consistency", True),
            payload.get("strict_distributed_consistency"),
        )
        expected_hash = getattr(self, "expected_shared_sha256", None)
        if strict is not None and expected_hash is not None and payload.get("shared_sha256") != expected_hash:
            raise RuntimeError(
                "UR robot-server configuration mismatch: "
                f"environment={expected_hash}, robot_server={payload.get('shared_sha256')!r}"
            )
        expected_identity = getattr(
            self,
            "expected_runtime_code_identity",
            runtime_code_identity().to_dict(),
        )
        actual_identity = payload.get("runtime_code_identity")
        if strict is not None and not compatible_runtime_identity(expected_identity, actual_identity or {}, strict=strict):
            raise RuntimeError(
                "UR robot-server runtime code mismatch: "
                f"environment={expected_identity!r}, robot_server={actual_identity!r}"
            )
        server_dual_arm = bool(payload.get("dual_arm", False))
        if server_dual_arm != self.dual_arm:
            raise RuntimeError(
                f"robot config arm_mode={'dual' if self.dual_arm else 'single'} does not match "
                f"server dual_arm={server_dual_arm}"
            )
        if self.dual_arm:
            return

        server_gripper = (payload.get("left", {}) or {}).get(
            "gripper",
            payload.get("gripper", {}),
        ) or {}
        enabled = server_gripper.get("enabled")
        expected_enabled = self.gripper.action_dimension != 0
        if enabled is not None and bool(enabled) != expected_enabled:
            raise RuntimeError(
                f"robot gripper enabled={expected_enabled} does not match "
                f"server gripper.enabled={bool(enabled)}"
            )
        server_kind = server_gripper.get("type")
        gripper_kind = getattr(self.gripper, "kind", None)
        if not expected_enabled or server_kind is None:
            return
        if str(server_kind) != gripper_kind:
            raise RuntimeError(
                f"robot gripper kind={gripper_kind!r} does not match "
                f"server gripper.type={server_kind!r}"
            )

    def _load_server_configuration(self, configured_delta_mask=()) -> None:
        if configured_delta_mask:
            self._set_delta_action_mask(configured_delta_mask, "config:delta_action_mask")
        response = self._post_server("server_config")
        payload = dict(response.json())
        self.server_config = payload
        self._check_server_configuration(payload)
        delta_mask = payload.get("delta_action_mask")
        if delta_mask is not None:
            self._set_delta_action_mask(
                delta_mask,
                "server:server_config:delta_action_mask",
            )

    def _update_state(self, *, retry_transport: bool = False) -> None:
        state = self._post_server("getstate", retry_transport=retry_transport).json()
        self._raw_state = state
        self._left_state = state["left"] if self.dual_arm else state
        self._right_state = state["right"] if self.dual_arm else None

    def refresh_state(self) -> None:
        self._update_state()

    def observations(self) -> dict[str, np.ndarray]:
        self._update_state()

        def arm_observation(state: dict) -> dict[str, np.ndarray]:
            return {
                "tcp_pose": np.asarray(state["pose"]),
                "tcp_vel": np.asarray(state["vel"]),
                "tcp_force": np.asarray(state["force"]),
                "tcp_torque": np.asarray(state["torque"]),
            }

        left = arm_observation(self._left_state)
        if self.dual_arm:
            state = {
                **{f"left/{key}": value for key, value in left.items()},
                **{f"right/{key}": value for key, value in arm_observation(self._right_state).items()},
            }
        else:
            state = left
        state.update(self.gripper.observations(self._raw_state))
        return state

    def _mask_action(self, action: np.ndarray) -> np.ndarray:
        masked = np.asarray(action, dtype=np.float64).copy()
        if self.dual_arm:
            masked[..., :6] *= self.delta_action_mask[:6]
            masked[..., 7:13] *= self.delta_action_mask[6:12]
        else:
            masked[..., :6] *= self.delta_action_mask
        return masked

    def execute_action_chunk(self, action: np.ndarray) -> np.ndarray:
        chunk = np.asarray(action, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None]
        if chunk.ndim != 2 or chunk.shape[-1] != self.action_dimension:
            raise ValueError(f"UR action must have shape (T, {self.action_dimension}), got {chunk.shape}")
        chunk = self._mask_action(np.clip(chunk, -3.0, 3.0))
        replay_chunk = chunk.astype(np.float32, copy=True)
        hardware_chunk = chunk.copy()
        if self.gripper.action_dimension:
            gripper_actions = hardware_chunk[:, (6, 13)] if self.dual_arm else hardware_chunk[:, 6]
            self.gripper.command_chunk(gripper_actions)
        response = self._post(self.url + "action", json={
            "arr": hardware_chunk.astype(np.float32).tolist(),
            "idle_hold_enabled": self.idle_hold_enabled,
            "idle_position_threshold": self.idle_position_threshold,
            "idle_rotation_threshold": self.idle_rotation_threshold,
        })
        try:
            self.last_action_chunk_id = response.json().get("action_chunk_id")
        except (AttributeError, TypeError, ValueError):
            self.last_action_chunk_id = None
        self._wait_for_action_chunk(self.last_action_chunk_id)
        return replay_chunk

    def _action_finished(self, status: Any, expected_id: int | None) -> bool:
        if expected_id is None or not isinstance(status, dict) or "error" in status:
            return False
        last_done = status.get("last_force_done", {})
        return last_done.get("force_done_time") is not None and int(last_done.get("id")) >= int(expected_id)

    def _wait_for_action_chunk(self, expected_id: int | None) -> None:
        if expected_id is None:
            return
        deadline = time.time() + self.action_done_timeout
        while True:
            try:
                status = self._post(self.url + "action_status", timeout=0.2).json()
            except (*self._transport_errors, AttributeError, TypeError, ValueError) as exc:
                status = {"error": str(exc)}
            if self._action_finished(status, expected_id) or time.time() >= deadline:
                return
            time.sleep(self.action_status_poll)

    def _send_pose(self, pose: np.ndarray, *, retry_transport: bool = False) -> None:
        target = np.asarray(pose, dtype=np.float32)
        if self.dual_arm and target.shape == (7,):
            target = np.stack((target, self.right_currpos.astype(np.float32)), axis=0)
        self._post_server("pose", json={"arr": target.tolist()}, retry_transport=retry_transport)

    def _interpolate_move(self, goal: np.ndarray, timeout: float) -> None:
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape == (6,):
            goal = np.concatenate((goal[:3], _euler_to_quat(goal[3:])))
        self._update_state(retry_transport=True)
        current = np.stack((self.currpos, self.right_currpos), axis=0) if self.dual_arm else self.currpos
        for pose in np.linspace(current, goal, int(timeout * self.control_hz)):
            self._send_pose(pose, retry_transport=True)
            time.sleep(1.0 / self.control_hz)
        time.sleep(2.0)
        self._update_state(retry_transport=True)

    def _reapply_payload_and_zero_ft(self, *, retry_transport: bool = True) -> None:
        self._post_server(
            "reset_payload",
            read_timeout=HTTP_SENSOR_READ_TIMEOUT_S,
            retry_transport=retry_transport,
        )
        time.sleep(PAYLOAD_SETTLE_TIME_S)
        self._post_server(
            "zero_ft_sensor",
            read_timeout=HTTP_SENSOR_READ_TIMEOUT_S,
            retry_transport=retry_transport,
        )
        time.sleep(FT_ZERO_SETTLE_TIME_S)

    def recover(self) -> None:
        """Retain the reset recovery hook; the current UR server needs no command here."""

    def reset(
        self,
        *,
        joint_reset: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        self.recover()
        self.reset_procedure.reset(
            self,
            joint_reset=joint_reset,
            options=options or {},
        )
        self.recover()
        return self.observations()

    def request(
        self,
        name: str,
        enabled: bool,
        *,
        retry_transport: bool = False,
        read_timeout: float = HTTP_FAST_READ_TIMEOUT_S,
    ) -> Any:
        response = self._post_server(
            name,
            json={name: bool(enabled)},
            read_timeout=read_timeout,
            retry_transport=retry_transport,
        )
        try:
            return response.json()
        except ValueError:
            return response.text

    def command(
        self,
        endpoint: str,
        payload: dict | None = None,
        *,
        retry_transport: bool = False,
    ) -> Any:
        """Send a task-level command without exposing HTTP details to a wrapper."""
        response = self._post_server(
            endpoint,
            json=payload,
            retry_transport=retry_transport,
        )
        try:
            return response.json()
        except ValueError:
            return response.text

    def close(self) -> None:
        self.gripper.close()
