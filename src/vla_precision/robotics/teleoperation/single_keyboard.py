"""Single-keyboard teleoperation used by the real-time intervention path."""

from __future__ import annotations

import multiprocessing


class SingleKeyboardExpert:
    def __init__(
        self,
        step_size_pos: float = 0.048,
        step_size_rot: float = 0.05,
        step_size_pos_alt: float | None = None,
        step_size_rot_alt: float | None = None,
    ):
        """Produce the existing 6-DoF delta and gripper/stop button state."""
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        # l, o, [: close, open, and stop.
        self.latest_data["buttons"] = [0, 0, 0]

        self.step_size_pos = float(step_size_pos)
        self.step_size_rot = float(step_size_rot)
        self.step_size_pos_alt = self.step_size_pos if step_size_pos_alt is None else float(step_size_pos_alt)
        self.step_size_rot_alt = self.step_size_rot if step_size_rot_alt is None else float(step_size_rot_alt)
        self.latest_data["step_mode"] = 0
        self.latest_data["step_size_pos"] = self.step_size_pos
        self.latest_data["step_size_rot"] = self.step_size_rot
        self.latest_data["reset_step_mode_request"] = 0
        self.process = multiprocessing.Process(target=self._read_keyboard)
        self.process.daemon = True
        self.process.start()

    def _read_keyboard(self):
        from pynput import keyboard

        key_mapping = {
            "z": (-1, 0.0),
            "s": (0, 1.0),
            "w": (0, -1.0),
            "d": (1, 1.0),
            "a": (1, -1.0),
            "q": (2, 1.0),
            "e": (2, -1.0),
            "r": (3, 1.0),
            "t": (3, -1.0),
            "g": (4, 1.0),
            "f": (4, -1.0),
            "b": (5, 1.0),
            "v": (5, -1.0),
            "l": ("button", 0, 1.0),
            "o": ("button", 1, 1.0),
            "[": ("button", 2, 1.0),
        }
        key_state = {key: False for key in key_mapping}
        step_mode = 0
        toggle_down = False
        reset_step_mode_seen = int(self.latest_data.get("reset_step_mode_request", 0))

        def key_to_char(key):
            key_char = getattr(key, "char", None)
            if key_char:
                return key_char
            if key == keyboard.KeyCode.from_char("/"):
                return "/"
            key_text = str(key)
            if key_text in {"'/'", "/"}:
                return "/"
            return None

        def maybe_reset_step_mode():
            nonlocal step_mode, toggle_down, reset_step_mode_seen
            request = int(self.latest_data.get("reset_step_mode_request", 0))
            if request == reset_step_mode_seen:
                return
            reset_step_mode_seen = request
            step_mode = 0
            toggle_down = False
            step_size_pos, step_size_rot = self._active_step_sizes(step_mode)
            self.latest_data["step_mode"] = step_mode
            self.latest_data["step_size_pos"] = step_size_pos
            self.latest_data["step_size_rot"] = step_size_rot
            self._update_action(key_mapping, key_state, step_mode=step_mode)

        def on_press(key):
            nonlocal step_mode, toggle_down
            maybe_reset_step_mode()
            key_char = key_to_char(key)
            if key_char is None:
                return
            if key_char == "/":
                if toggle_down:
                    return
                toggle_down = True
                step_mode = 1 - step_mode
                step_size_pos, step_size_rot = self._active_step_sizes(step_mode)
                self.latest_data["step_mode"] = step_mode
                self.latest_data["step_size_pos"] = step_size_pos
                self.latest_data["step_size_rot"] = step_size_rot
                self._update_action(key_mapping, key_state, step_mode=step_mode)
                return
            if key_char in key_mapping:
                key_state[key_char] = True
                self._update_action(key_mapping, key_state, step_mode=step_mode)

        def on_release(key):
            nonlocal toggle_down
            maybe_reset_step_mode()
            key_char = key_to_char(key)
            if key_char is None:
                return
            if key_char == "/":
                toggle_down = False
                return
            if key_char in key_mapping:
                key_state[key_char] = False
                self._update_action(key_mapping, key_state, step_mode=step_mode)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        listener.join()

    def _active_step_sizes(self, step_mode: int):
        if int(step_mode) == 1:
            return self.step_size_pos_alt, self.step_size_rot_alt
        return self.step_size_pos, self.step_size_rot

    def reset_step_mode(self):
        self.latest_data["step_mode"] = 0
        self.latest_data["step_size_pos"] = self.step_size_pos
        self.latest_data["step_size_rot"] = self.step_size_rot
        self.latest_data["reset_step_mode_request"] = int(self.latest_data.get("reset_step_mode_request", 0)) + 1

    def _update_action(self, key_mapping, key_state, *, step_mode: int = 0):
        action = [0.0] * 6
        buttons = [0, 0, 0]
        step_size_pos, step_size_rot = self._active_step_sizes(step_mode)

        for key, value in key_state.items():
            if value:
                mapping = key_mapping[key]
                if mapping[0] == "button":
                    buttons[mapping[1]] = mapping[2]
                elif mapping[0] == -1:
                    pass
                elif mapping[0] in (3, 4, 5):
                    action[mapping[0]] = mapping[1] * step_size_rot
                else:
                    action[mapping[0]] = mapping[1] * step_size_pos

        self.latest_data["action"] = action
        self.latest_data["buttons"] = buttons

    def get_action(self) -> tuple[list[float], list[int]]:
        buttons = self.latest_data["buttons"]
        return self.latest_data["action"], buttons

    def get_buttons(self) -> list[int]:
        return self.latest_data["buttons"]

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.2)
        self.manager.shutdown()
