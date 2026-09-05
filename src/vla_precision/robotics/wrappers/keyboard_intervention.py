"""Gym adapter for the existing chunk-aware keyboard intervention sequence."""

from __future__ import annotations

import threading

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from vla_precision.robotics.teleoperation.dual_keyboard import DualKeyboardExpert
from vla_precision.robotics.teleoperation.single_keyboard import SingleKeyboardExpert


class KeyboardIntervention(gym.ActionWrapper):
    """Keyboard intervention wrapper for OpenPI action chunks.

    A policy action is shape ``(T, A)``. If the keyboard expert is inactive at
    the chunk boundary, the policy chunk is executed normally. If the expert is
    active, this wrapper skips the policy chunk and executes human actions
    one-by-one in realtime for up to ``T`` steps, then returns a full
    ``intervene_action`` chunk for replay.
    """

    def __init__(
        self,
        env,
        action_indices=None,
        step_size_pos: float = 0.048,
        step_size_rot: float = 0.05,
        step_size_pos_alt: float | None = None,
        step_size_rot_alt: float | None = None,
        dual_arm: bool = False,
        left_keyboard_path: str | None = None,
        right_keyboard_path: str | None = None,
        left_step_size_pos: float | None = None,
        left_step_size_rot: float | None = None,
        left_step_size_pos_alt: float | None = None,
        left_step_size_rot_alt: float | None = None,
        right_step_size_pos: float | None = None,
        right_step_size_rot: float | None = None,
        right_step_size_pos_alt: float | None = None,
        right_step_size_rot_alt: float | None = None,
        reward_keyboard_arm: str = "left",
        completion_double_press_interval: float = 0.5,
        left_gripper_start_position: int = 1,
        right_gripper_start_position: int = 1,
        expert=None,
        completion_event: threading.Event | None = None,
    ):
        super().__init__(env)
        self.dual_arm = bool(dual_arm)
        self.gripper_enabled = not bool(getattr(self.unwrapped, "fixed_gripper", False))
        self.gripper_indices = (6, 13) if self.dual_arm else ((6,) if self.gripper_enabled else ())
        self.step_size_pos = float(step_size_pos)
        self.step_size_rot = float(step_size_rot)
        self.step_size_pos_alt = self.step_size_pos if step_size_pos_alt is None else float(step_size_pos_alt)
        self.step_size_rot_alt = self.step_size_rot if step_size_rot_alt is None else float(step_size_rot_alt)
        self.completion_event = completion_event or (threading.Event() if self.dual_arm else None)
        self.reward_keyboard_arm = reward_keyboard_arm
        if expert is not None:
            self.expert = expert
        elif self.dual_arm:
            self.expert = DualKeyboardExpert(
                left_keyboard_path=left_keyboard_path,
                right_keyboard_path=right_keyboard_path,
                left_step_size_pos=self.step_size_pos if left_step_size_pos is None else left_step_size_pos,
                left_step_size_rot=self.step_size_rot if left_step_size_rot is None else left_step_size_rot,
                left_step_size_pos_alt=(
                    self.step_size_pos_alt if left_step_size_pos_alt is None else left_step_size_pos_alt
                ),
                left_step_size_rot_alt=(
                    self.step_size_rot_alt if left_step_size_rot_alt is None else left_step_size_rot_alt
                ),
                right_step_size_pos=self.step_size_pos if right_step_size_pos is None else right_step_size_pos,
                right_step_size_rot=self.step_size_rot if right_step_size_rot is None else right_step_size_rot,
                right_step_size_pos_alt=(
                    self.step_size_pos_alt if right_step_size_pos_alt is None else right_step_size_pos_alt
                ),
                right_step_size_rot_alt=(
                    self.step_size_rot_alt if right_step_size_rot_alt is None else right_step_size_rot_alt
                ),
                completion_event=self.completion_event,
                reward_keyboard_arm=reward_keyboard_arm,
                completion_double_press_interval=completion_double_press_interval,
            )
        else:
            self.expert = SingleKeyboardExpert(
                step_size_pos=self.step_size_pos,
                step_size_rot=self.step_size_rot,
                step_size_pos_alt=self.step_size_pos_alt,
                step_size_rot_alt=self.step_size_rot_alt,
            )
        self.close_gripper_pressed = False
        self.open_gripper_pressed = False
        self.stop_pressed = False
        self.left_close_gripper_pressed = False
        self.left_open_gripper_pressed = False
        self.left_stop_pressed = False
        self.right_close_gripper_pressed = False
        self.right_open_gripper_pressed = False
        self.right_stop_pressed = False
        self.action_indices = action_indices
        self.base_action_to_tcp_action = getattr(env, "base_action_to_tcp_action", None)
        self._configure_keyboard_frames()
        self.left_gripper_start_position = int(left_gripper_start_position)
        self.right_gripper_start_position = int(right_gripper_start_position)
        self.gripper_state = self.left_gripper_start_position
        self.right_gripper_state = self.right_gripper_start_position
        if self.gripper_state not in (0, 1):
            raise ValueError(
                "LEFT_GRIPPER_START_POS (or legacy GRIPPER_START_POS) must be one of "
                "(0, 1)"
            )
        if self.right_gripper_state not in (0, 1):
            raise ValueError("RIGHT_GRIPPER_START_POS must be 0 or 1")

    def _configure_keyboard_frames(self):
        """Use the active robot-server task frame for manual keyboard deltas."""
        server_config = dict(getattr(self.unwrapped, "robot_server_config", {}) or {})
        if self.dual_arm:
            control_eulers = (
                server_config.get("left_control_frame_euler_deg", [0.0, 0.0, 0.0]),
                server_config.get("right_control_frame_euler_deg", [0.0, 0.0, 0.0]),
            )
            select_vectors = (
                server_config.get("left_select_vector", [1.0] * 6),
                server_config.get("right_select_vector", [1.0] * 6),
            )
        else:
            control_eulers = (server_config.get("control_frame_euler_deg", [0.0, 0.0, 0.0]),)
            select_vectors = (server_config.get("select_vector", [1.0] * 6),)

        self.keyboard_control_eulers_deg = []
        self.keyboard_control_to_base_rotations = []
        self.keyboard_select_vectors = []
        for arm_index, (control_euler, select_vector) in enumerate(zip(control_eulers, select_vectors)):
            control_euler = np.asarray(control_euler, dtype=np.float64).reshape(-1)
            select_vector = np.asarray(select_vector, dtype=np.float64).reshape(-1)
            arm = "left" if arm_index == 0 else "right"
            if control_euler.shape != (3,) or not np.all(np.isfinite(control_euler)):
                raise ValueError(f"{arm} control_frame_euler_deg must contain three finite values")
            if select_vector.shape != (6,):
                raise ValueError(f"{arm} select_vector must contain six values")
            if not np.all(np.isin(select_vector, (0.0, 1.0))):
                raise ValueError(f"{arm} select_vector values must be 0 or 1")
            self.keyboard_control_eulers_deg.append(control_euler)
            self.keyboard_control_to_base_rotations.append(R.from_euler("xyz", control_euler, degrees=True))
            self.keyboard_select_vectors.append(select_vector)

    def _control_action_to_base_action(self, action: np.ndarray) -> np.ndarray:
        """Mask a keyboard delta in control axes, then rotate it into each UR base."""
        action = np.asarray(action).copy()
        arm_slices = ((0, 6), (7, 13)) if self.dual_arm else ((0, 6),)
        for arm_index, (start, stop) in enumerate(arm_slices):
            control_delta = np.asarray(action[start:stop], dtype=np.float64)
            control_delta *= self.keyboard_select_vectors[arm_index]
            control_to_base = self.keyboard_control_to_base_rotations[arm_index]
            base_delta_pos = control_to_base.apply(control_delta[:3])
            base_delta_rot = control_to_base * R.from_euler("xyz", control_delta[3:]) * control_to_base.inv()
            action[start : start + 3] = base_delta_pos.astype(action.dtype, copy=False)
            action[start + 3 : stop] = base_delta_rot.as_euler("xyz").astype(action.dtype, copy=False)
        return action

    def _chunk_gripper_changed(self, policy_action, intervention_action) -> bool:
        policy_arr = np.asarray(policy_action)
        intervention_arr = np.asarray(intervention_action)
        if policy_arr.ndim < 2 or intervention_arr.ndim < 2:
            return False
        if policy_arr.shape != intervention_arr.shape or policy_arr.shape[-1] < 1:
            return False
        policy_state = np.rint(policy_arr[:, self.gripper_indices]).astype(np.int64)
        intervention_state = np.rint(intervention_arr[:, self.gripper_indices]).astype(np.int64)
        return bool(np.any(policy_state != intervention_state))

    def _policy_gripper_state(self, value: float) -> int:
        return 1 if float(value) >= 0.5 else 0

    def _single_expert_action(self, reference_action: np.ndarray):
        if self.dual_arm and hasattr(self.expert, "get_action_state"):
            expert_a, buttons, stop_buttons = self.expert.get_action_state()
        else:
            expert_a, buttons = self.expert.get_action()
            stop_buttons = (False, False)
        expert_a = np.asarray(expert_a, dtype=np.float32)
        if self.dual_arm:
            left_close, left_open, right_close, right_open = tuple(buttons)
            left_stop, right_stop = tuple(stop_buttons)
            self.left_stop_pressed = bool(left_stop)
            self.right_stop_pressed = bool(right_stop)
            # Stop has priority over commands from the same keyboard and
            # preserves that arm's current gripper state.
            self.left_close_gripper_pressed = bool(left_close and not self.left_stop_pressed)
            self.left_open_gripper_pressed = bool(left_open and not self.left_stop_pressed)
            self.right_close_gripper_pressed = bool(right_close and not self.right_stop_pressed)
            self.right_open_gripper_pressed = bool(right_open and not self.right_stop_pressed)
            self.close_gripper_pressed = bool(
                self.left_close_gripper_pressed or self.right_close_gripper_pressed
            )
            self.open_gripper_pressed = bool(
                self.left_open_gripper_pressed or self.right_open_gripper_pressed
            )
            self.stop_pressed = bool(self.left_stop_pressed or self.right_stop_pressed)
            if self.left_stop_pressed:
                expert_a[:6] = 0.0
            if self.right_stop_pressed:
                expert_a[6:12] = 0.0
        else:
            l_pressed, o_pressed, stop_pressed = tuple(buttons)
            self.close_gripper_pressed = bool(l_pressed)
            self.open_gripper_pressed = bool(o_pressed)
            self.stop_pressed = bool(stop_pressed)
            self.left_stop_pressed = bool(self.stop_pressed)
            self.right_stop_pressed = False
            self.left_close_gripper_pressed = bool(self.close_gripper_pressed)
            self.left_open_gripper_pressed = bool(self.open_gripper_pressed)
            self.right_close_gripper_pressed = False
            self.right_open_gripper_pressed = False
        if not self.gripper_enabled:
            self.close_gripper_pressed = False
            self.open_gripper_pressed = False
            self.left_close_gripper_pressed = False
            self.left_open_gripper_pressed = False
            self.right_close_gripper_pressed = False
            self.right_open_gripper_pressed = False
        intervened = False
        if self.dual_arm:
            left_pose_pressed = bool(np.linalg.norm(expert_a[:6]) > 0.001)
            right_pose_pressed = bool(np.linalg.norm(expert_a[6:12]) > 0.001)
        else:
            left_pose_pressed = bool(np.linalg.norm(expert_a) > 0.001)
            right_pose_pressed = False
        left_gripper_pressed = bool(
            self.left_close_gripper_pressed
            or self.left_open_gripper_pressed
        )
        right_gripper_pressed = bool(self.right_close_gripper_pressed or self.right_open_gripper_pressed)
        left_intervened = bool(left_pose_pressed or left_gripper_pressed or self.left_stop_pressed)
        right_intervened = bool(right_pose_pressed or right_gripper_pressed or self.right_stop_pressed)
        pose_pressed = bool(left_pose_pressed or right_pose_pressed)
        gripper_pressed = bool(
            self.close_gripper_pressed
            or self.open_gripper_pressed
        )
        intervention_meta = {
            "pose_pressed": pose_pressed,
            "gripper_pressed": gripper_pressed,
            "stop_pressed": bool(self.stop_pressed),
            "left_stop_pressed": bool(self.left_stop_pressed),
            "right_stop_pressed": bool(self.right_stop_pressed),
            "left_pose_pressed": left_pose_pressed,
            "right_pose_pressed": right_pose_pressed,
            "left_gripper_pressed": left_gripper_pressed,
            "right_gripper_pressed": right_gripper_pressed,
            "left_intervened": left_intervened,
            "right_intervened": right_intervened,
        }

        if self.stop_pressed and not self.dual_arm:
            intervened = True
            expert_a = np.zeros_like(expert_a)

        if self.dual_arm and (self.left_stop_pressed or self.right_stop_pressed):
            intervened = True

        if pose_pressed:
            intervened = True

        if self.gripper_enabled:
            if self.dual_arm:
                if self.left_close_gripper_pressed:
                    self.gripper_state = 0
                    intervened = True
                elif self.left_open_gripper_pressed:
                    self.gripper_state = 1
                    intervened = True
                if self.right_close_gripper_pressed:
                    self.right_gripper_state = 0
                    intervened = True
                elif self.right_open_gripper_pressed:
                    self.right_gripper_state = 1
                    intervened = True
                expert_a = np.concatenate(
                    (
                        np.asarray(expert_a[:6], dtype=np.float32),
                        np.asarray([self.gripper_state], dtype=np.float32),
                        np.asarray(expert_a[6:12], dtype=np.float32),
                        np.asarray([self.right_gripper_state], dtype=np.float32),
                    ),
                    axis=0,
                )
            else:
                if self.close_gripper_pressed:
                    self.gripper_state = 0
                    intervened = True
                elif self.open_gripper_pressed:
                    self.gripper_state = 1
                    intervened = True
                gripper_action = np.asarray([self.gripper_state], dtype=np.float32)
                expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if intervened:
            expert_a = self._control_action_to_base_action(expert_a)
        if intervened:
            try:
                if self.base_action_to_tcp_action is None:
                    raise AttributeError("KeyboardIntervention requires an environment with base_action_to_tcp_action")
                self.unwrapped._update_currpos()
                expert_a = self.base_action_to_tcp_action(expert_a)
            except Exception as exc:
                raise RuntimeError(
                    "Keyboard delta frame conversion failed; refusing to send an action in the wrong frame"
                ) from exc

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        # A dual-arm keyboard only owns its corresponding 7D action segment.
        # Preserve the policy segment for an arm whose keyboard is idle; if
        # both keyboards are active, both segments are replaced in the same
        # joint 14D timestep.
        if self.dual_arm and intervened:
            merged_expert_a = np.asarray(reference_action).copy()
            if left_intervened:
                merged_expert_a[:7] = expert_a[:7]
            if right_intervened:
                merged_expert_a[7:14] = expert_a[7:14]
            expert_a = merged_expert_a

        if not intervened:
            return reference_action, False, intervention_meta
        return expert_a.astype(reference_action.dtype, copy=False), True, intervention_meta

    def step(self, action):
        action = np.asarray(action)
        if action.ndim != 2:
            raise ValueError(f"KeyboardIntervention expects action chunk shape (T, A), got {action.shape}")

        first_expert_action, intervened, first_meta = self._single_expert_action(action[0])
        chunk_pose_pressed = bool(first_meta["pose_pressed"])
        chunk_gripper_pressed = bool(first_meta["gripper_pressed"])
        chunk_stop_pressed = bool(first_meta["stop_pressed"])
        chunk_left_pose_pressed = bool(first_meta["left_pose_pressed"])
        chunk_right_pose_pressed = bool(first_meta["right_pose_pressed"])
        chunk_left_gripper_pressed = bool(first_meta["left_gripper_pressed"])
        chunk_right_gripper_pressed = bool(first_meta["right_gripper_pressed"])
        chunk_left_stop_pressed = bool(first_meta["left_stop_pressed"])
        chunk_right_stop_pressed = bool(first_meta["right_stop_pressed"])
        if not intervened:
            obs, rew, done, truncated, info = self.env.step(action)
            if self.gripper_enabled:
                self.gripper_state = self._policy_gripper_state(action[-1, 6])
                if self.dual_arm:
                    self.right_gripper_state = 1 if action[-1, 13] >= 0.5 else 0
            info["close_gripper_pressed"] = self.close_gripper_pressed
            info["open_gripper_pressed"] = self.open_gripper_pressed
            info["pose_intervention_pressed"] = False
            info["gripper_intervention_pressed"] = False
            info["stop_intervention_pressed"] = False
            info["gripper_action_changed_env"] = False
            info["left_intervention_steps_executed"] = 0
            info["right_intervention_steps_executed"] = 0
            info["left_pose_intervention_pressed"] = False
            info["right_pose_intervention_pressed"] = False
            info["left_gripper_intervention_pressed"] = False
            info["right_gripper_intervention_pressed"] = False
            info["left_stop_intervention_pressed"] = False
            info["right_stop_intervention_pressed"] = False
            info["left_close_gripper_pressed"] = self.left_close_gripper_pressed
            info["left_open_gripper_pressed"] = self.left_open_gripper_pressed
            info["right_close_gripper_pressed"] = self.right_close_gripper_pressed
            info["right_open_gripper_pressed"] = self.right_open_gripper_pressed
            info["left"] = self.close_gripper_pressed
            info["right"] = self.open_gripper_pressed
            return obs, rew, done, truncated, info

        self.env.execute_action_chunk(first_expert_action[None, :])
        human_actions = [first_expert_action]
        human_control_steps = 1
        left_intervention_steps = int(first_meta["left_intervened"])
        right_intervention_steps = int(first_meta["right_intervened"])
        left_ever_intervened = bool(first_meta["left_intervened"])
        right_ever_intervened = bool(first_meta["right_intervened"])
        if self.dual_arm:
            if not left_ever_intervened:
                self.gripper_state = self._policy_gripper_state(action[0, 6])
            if not right_ever_intervened:
                self.right_gripper_state = 1 if action[0, 13] >= 0.5 else 0
        still_collecting_human = True
        for step_idx in range(1, action.shape[0]):
            if self.dual_arm:
                expert_action, currently_intervening, expert_meta = self._single_expert_action(action[step_idx])
                chunk_pose_pressed = chunk_pose_pressed or bool(expert_meta["pose_pressed"])
                chunk_gripper_pressed = chunk_gripper_pressed or bool(expert_meta["gripper_pressed"])
                chunk_stop_pressed = chunk_stop_pressed or bool(expert_meta["stop_pressed"])
                chunk_left_pose_pressed = chunk_left_pose_pressed or bool(expert_meta["left_pose_pressed"])
                chunk_right_pose_pressed = chunk_right_pose_pressed or bool(expert_meta["right_pose_pressed"])
                chunk_left_gripper_pressed = chunk_left_gripper_pressed or bool(expert_meta["left_gripper_pressed"])
                chunk_right_gripper_pressed = chunk_right_gripper_pressed or bool(expert_meta["right_gripper_pressed"])
                chunk_left_stop_pressed = chunk_left_stop_pressed or bool(expert_meta["left_stop_pressed"])
                chunk_right_stop_pressed = chunk_right_stop_pressed or bool(expert_meta["right_stop_pressed"])

                left_current = bool(expert_meta["left_intervened"])
                right_current = bool(expert_meta["right_intervened"])
                left_intervention_steps += int(left_current)
                right_intervention_steps += int(right_current)
                if currently_intervening:
                    human_control_steps += 1

                action_to_execute = np.asarray(expert_action).copy()
                if left_current:
                    left_ever_intervened = True
                elif left_ever_intervened:
                    action_to_execute[:6] = 0.0
                    action_to_execute[6] = self.gripper_state
                else:
                    self.gripper_state = self._policy_gripper_state(action[step_idx, 6])

                if right_current:
                    right_ever_intervened = True
                elif right_ever_intervened:
                    action_to_execute[7:13] = 0.0
                    action_to_execute[13] = self.right_gripper_state
                else:
                    self.right_gripper_state = 1 if action[step_idx, 13] >= 0.5 else 0
            elif still_collecting_human:
                expert_action, still_intervening, expert_meta = self._single_expert_action(action[step_idx])
                chunk_pose_pressed = chunk_pose_pressed or bool(expert_meta["pose_pressed"])
                chunk_gripper_pressed = chunk_gripper_pressed or bool(expert_meta["gripper_pressed"])
                chunk_stop_pressed = chunk_stop_pressed or bool(expert_meta["stop_pressed"])
                chunk_left_pose_pressed = chunk_left_pose_pressed or bool(expert_meta["left_pose_pressed"])
                chunk_right_pose_pressed = chunk_right_pose_pressed or bool(expert_meta["right_pose_pressed"])
                chunk_left_gripper_pressed = chunk_left_gripper_pressed or bool(expert_meta["left_gripper_pressed"])
                chunk_right_gripper_pressed = chunk_right_gripper_pressed or bool(expert_meta["right_gripper_pressed"])
                chunk_left_stop_pressed = chunk_left_stop_pressed or bool(expert_meta["left_stop_pressed"])
                chunk_right_stop_pressed = chunk_right_stop_pressed or bool(expert_meta["right_stop_pressed"])
                if still_intervening:
                    action_to_execute = expert_action
                    human_control_steps += 1
                    left_intervention_steps += int(expert_meta["left_intervened"])
                    right_intervention_steps += int(expert_meta["right_intervened"])
                else:
                    still_collecting_human = False
                    action_to_execute = np.zeros_like(human_actions[-1])
                    if self.gripper_enabled:
                        action_to_execute[list(self.gripper_indices)] = human_actions[-1][list(self.gripper_indices)]
            else:
                action_to_execute = np.zeros_like(human_actions[-1])
                if self.gripper_enabled:
                    action_to_execute[list(self.gripper_indices)] = human_actions[-1][list(self.gripper_indices)]

            self.env.execute_action_chunk(action_to_execute[None, :])
            human_actions.append(action_to_execute)

        intervention_chunk = np.stack(human_actions, axis=0)
        gripper_action_changed_env = self._chunk_gripper_changed(action, intervention_chunk)

        obs, rew, done, truncated, info = self.env.finish_action_chunk_step()
        rew = np.asarray(rew, dtype=np.float32).reshape(-1)
        if rew.shape[0] != action.shape[0]:
            padded_rew = np.zeros((action.shape[0],), dtype=np.float32)
            padded_rew[: min(rew.shape[0], action.shape[0])] = rew[: action.shape[0]]
            rew = padded_rew
        executed_action = info.get("executed_action")
        if executed_action is not None:
            executed_action = np.asarray(executed_action, dtype=intervention_chunk.dtype)
            if executed_action.shape == intervention_chunk.shape:
                intervention_chunk = executed_action
        info["intervene_action"] = intervention_chunk
        info["intervention_steps_executed"] = human_control_steps
        info["left_intervention_steps_executed"] = left_intervention_steps
        info["right_intervention_steps_executed"] = right_intervention_steps
        info["steps_executed"] = action.shape[0]
        info["close_gripper_pressed"] = self.close_gripper_pressed
        info["open_gripper_pressed"] = self.open_gripper_pressed
        info["pose_intervention_pressed"] = chunk_pose_pressed
        info["gripper_intervention_pressed"] = chunk_gripper_pressed
        info["stop_intervention_pressed"] = chunk_stop_pressed
        info["gripper_action_changed_env"] = gripper_action_changed_env
        info["left_pose_intervention_pressed"] = chunk_left_pose_pressed
        info["right_pose_intervention_pressed"] = chunk_right_pose_pressed
        info["left_gripper_intervention_pressed"] = chunk_left_gripper_pressed
        info["right_gripper_intervention_pressed"] = chunk_right_gripper_pressed
        info["left_stop_intervention_pressed"] = chunk_left_stop_pressed
        info["right_stop_intervention_pressed"] = chunk_right_stop_pressed
        info["left_close_gripper_pressed"] = self.left_close_gripper_pressed
        info["left_open_gripper_pressed"] = self.left_open_gripper_pressed
        info["right_close_gripper_pressed"] = self.right_close_gripper_pressed
        info["right_open_gripper_pressed"] = self.right_open_gripper_pressed
        info["left"] = self.close_gripper_pressed
        info["right"] = self.open_gripper_pressed
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self.gripper_state = self.left_gripper_start_position
        self.right_gripper_state = self.right_gripper_start_position
        if hasattr(self.expert, "reset_step_mode"):
            self.expert.reset_step_mode()
        return self.env.reset(**kwargs)

    def close(self):
        if hasattr(self.expert, "close"):
            self.expert.close()
        return self.env.close()
