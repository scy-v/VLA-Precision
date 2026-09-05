"""Insert-task regrasp adapter kept outside the robot implementation."""

from __future__ import annotations

import threading

import gymnasium as gym
import numpy as np


class RegraspResetWrapper(gym.Wrapper):
    """Optionally run the established regrasp sequence before an episode reset."""

    def __init__(
        self,
        env: gym.Env,
        *,
        regrasp_step_1=None,
        regrasp_step_2=None,
        wait_before: bool = False,
        wait_after: bool = False,
        manual_adjustment_event: threading.Event | None = None,
    ):
        super().__init__(env)
        self.regrasp_step_1 = (
            None if regrasp_step_1 is None else np.asarray(regrasp_step_1, dtype=np.float32)
        )
        self.regrasp_step_2 = (
            None if regrasp_step_2 is None else np.asarray(regrasp_step_2, dtype=np.float32)
        )
        self.wait_before = bool(wait_before)
        self.wait_after = bool(wait_after)
        self._manual_adjustment_event = manual_adjustment_event or threading.Event()
        self._listener = None
        if manual_adjustment_event is None:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.KeyCode.from_char("m"):
                    self._manual_adjustment_event.set()

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()

    @property
    def robot_environment(self):
        return self.unwrapped

    def _enter_reset_mode(self) -> None:
        self.robot_environment.request("force_pause", False)
        self.robot_environment.request("task_mode", False)

    def _wait_for_manual_adjustment(self) -> None:
        self._manual_adjustment_event.clear()
        zero_action = np.zeros(self.action_space.shape, dtype=np.float32)
        while not self._manual_adjustment_event.is_set():
            self.step(zero_action)

    def _regrasp(self) -> None:
        if self.regrasp_step_1 is None and self.regrasp_step_2 is None:
            return
        if self.wait_before:
            self._wait_for_manual_adjustment()
        self.robot_environment.robot.command(
            "regrasp",
            {
                "regrasp_step_1": (
                    None if self.regrasp_step_1 is None else self.regrasp_step_1.tolist()
                ),
                "regrasp_step_2": (
                    None if self.regrasp_step_2 is None else self.regrasp_step_2.tolist()
                ),
            },
        )
        if self.wait_after:
            self._wait_for_manual_adjustment()

    def reset(self, *, seed=None, options=None):
        reset_options = options or {}
        if reset_options.get("regrasp_before_reset", True):
            self._enter_reset_mode()
            self._regrasp()
        return self.env.reset(seed=seed, options=options)

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        self.env.close()
