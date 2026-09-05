"""Two-device keyboard teleoperation for dual-arm ACoB-Stream actors."""

from __future__ import annotations

import logging
import threading
import time
from typing import ClassVar

import numpy as np

try:
    from evdev import InputDevice, categorize, ecodes
except ImportError:
    InputDevice = None
    categorize = None
    ecodes = None


LOGGER = logging.getLogger(__name__)


class DualKeyboardExpert:
    """Read two physical keyboards and produce left-then-right 12-DoF deltas.

    Both keyboards use the same key layout. Device identity, rather than a
    different set of keys, decides which arm is controlled. This mirrors the
    dual-UR data-collection teleoperator.
    """

    ARMS = ("left", "right")
    KEY_MAPPING: ClassVar[dict] = {
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
        "l": ("close", 1.0),
        "o": ("open", 1.0),
        "[": ("stop", 1.0),
    }
    EVDEV_KEY_TO_CHAR: ClassVar[dict[str, str]] = {
        "KEY_W": "w",
        "KEY_S": "s",
        "KEY_A": "a",
        "KEY_D": "d",
        "KEY_Q": "q",
        "KEY_E": "e",
        "KEY_R": "r",
        "KEY_T": "t",
        "KEY_F": "f",
        "KEY_G": "g",
        "KEY_V": "v",
        "KEY_B": "b",
        "KEY_O": "o",
        "KEY_L": "l",
        "KEY_LEFTBRACE": "[",
        "KEY_SLASH": "/",
        "KEY_DOT": ".",
    }

    def __init__(
        self,
        *,
        left_keyboard_path: str | None,
        right_keyboard_path: str | None,
        left_step_size_pos: float = 0.048,
        left_step_size_rot: float = 0.05,
        left_step_size_pos_alt: float | None = None,
        left_step_size_rot_alt: float | None = None,
        right_step_size_pos: float = 0.048,
        right_step_size_rot: float = 0.05,
        right_step_size_pos_alt: float | None = None,
        right_step_size_rot_alt: float | None = None,
        completion_event: threading.Event | None = None,
        reward_keyboard_arm: str = "left",
        completion_double_press_interval: float = 0.5,
        reconnect_delay: float = 1.0,
        start_listeners: bool = True,
        input_device_factory=None,
    ):
        self.keyboard_paths = {
            "left": left_keyboard_path,
            "right": right_keyboard_path,
        }
        self.default_step_sizes = {
            "left": (float(left_step_size_pos), float(left_step_size_rot)),
            "right": (float(right_step_size_pos), float(right_step_size_rot)),
        }
        self.alternate_step_sizes = {
            "left": (
                float(left_step_size_pos if left_step_size_pos_alt is None else left_step_size_pos_alt),
                float(left_step_size_rot if left_step_size_rot_alt is None else left_step_size_rot_alt),
            ),
            "right": (
                float(right_step_size_pos if right_step_size_pos_alt is None else right_step_size_pos_alt),
                float(right_step_size_rot if right_step_size_rot_alt is None else right_step_size_rot_alt),
            ),
        }
        for arm in self.ARMS:
            for mode, sizes in (
                ("default", self.default_step_sizes[arm]),
                ("alternate", self.alternate_step_sizes[arm]),
            ):
                if not all(np.isfinite(value) and value > 0 for value in sizes):
                    raise ValueError(f"{arm} {mode} position/rotation step sizes must be finite and positive")
        if reward_keyboard_arm not in self.ARMS:
            raise ValueError("reward_keyboard_arm must be 'left' or 'right'")
        if completion_double_press_interval <= 0:
            raise ValueError("completion_double_press_interval must be positive")
        self.reward_keyboard_arm = reward_keyboard_arm
        self.completion_event = completion_event
        if reconnect_delay < 0:
            raise ValueError("reconnect_delay must be non-negative")
        self.reconnect_delay = float(reconnect_delay)
        self.completion_double_press_interval = float(completion_double_press_interval)
        self._lock = threading.RLock()
        self._pressed_keys = {arm: set() for arm in self.ARMS}
        self._step_modes = {arm: 0 for arm in self.ARMS}
        self._last_completion_press = None
        self._stop_event = threading.Event()
        self._devices = {}
        self._reader_threads = []
        self._input_device_factory = input_device_factory or InputDevice

        if start_listeners:
            self._start_listeners()

    def _validate_device_paths(self):
        if self._input_device_factory is None or categorize is None or ecodes is None:
            raise RuntimeError(
                "evdev is required for dual physical keyboard control. "
                "Install it with `uv sync --group stage2 --group real-robot` "
                "(or `pip install evdev`)."
            )
        if not self.keyboard_paths["left"] or not self.keyboard_paths["right"]:
            raise ValueError(
                "DualKeyboardExpert requires both left_keyboard_path and right_keyboard_path; "
                "use stable /dev/input/by-id/*-event-kbd paths."
            )
        if self.keyboard_paths["left"] == self.keyboard_paths["right"]:
            raise ValueError("left_keyboard_path and right_keyboard_path must identify different devices")

    def _start_listeners(self):
        self._validate_device_paths()
        try:
            for arm in self.ARMS:
                path = self.keyboard_paths[arm]
                try:
                    device = self._input_device_factory(path)
                except PermissionError as exc:
                    raise PermissionError(
                        f"Cannot read {arm} keyboard device {path}; grant access to /dev/input or join the input group."
                    ) from exc
                self._devices[arm] = device
                thread = threading.Thread(
                    target=self._read_device_events,
                    args=(arm, device),
                    name=f"acob-stream-{arm}-keyboard",
                    daemon=True,
                )
                thread.start()
                self._reader_threads.append(thread)
        except Exception:
            self.close()
            raise

    def _read_device_events(self, arm, device):
        while not self._stop_event.is_set():
            try:
                for event in device.read_loop():
                    if self._stop_event.is_set():
                        return
                    if event.type != ecodes.EV_KEY:
                        continue
                    key_event = categorize(event)
                    key_name = key_event.keycode
                    if isinstance(key_name, list):
                        key_name = key_name[0]
                    key_char = self.EVDEV_KEY_TO_CHAR.get(key_name)
                    if key_char is None:
                        continue
                    if key_event.keystate == key_event.key_down:
                        self._handle_key_event(arm, key_char, True, key_down=True)
                    elif key_event.keystate == key_event.key_up:
                        self._handle_key_event(arm, key_char, False, key_down=False)
                    elif key_event.keystate == key_event.key_hold:
                        self._handle_key_event(arm, key_char, True, key_down=False)
                return
            except OSError as exc:
                if self._stop_event.is_set():
                    return
                with self._lock:
                    # A release event may be lost when USB disappears. Never
                    # preserve a held key across reconnection, because doing so
                    # could keep commanding robot motion without an operator.
                    self._pressed_keys[arm].clear()
                message = (
                    f"[DualKeyboardExpert] {arm} keyboard disconnected: "
                    f"{type(exc).__name__}: {exc}; reconnecting"
                )
                print(message, flush=True)
                try:
                    device.close()
                except Exception:  # evdev and extension device backends vary.
                    LOGGER.exception("failed to close disconnected %s keyboard", arm)
                device = self._reconnect_device_until_ready(arm)
                if device is None:
                    return

    def _reconnect_device_until_ready(self, arm):
        path = self.keyboard_paths[arm]
        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            if self.reconnect_delay > 0 and self._stop_event.wait(self.reconnect_delay):
                return None
            try:
                device = self._input_device_factory(path)
            except (OSError, PermissionError) as exc:
                message = (
                    f"[DualKeyboardExpert] {arm} keyboard reconnect failed "
                    f"attempt={attempt} path={path}: {type(exc).__name__}: {exc}"
                )
                print(message, flush=True)
                continue
            self._devices[arm] = device
            message = (
                f"[DualKeyboardExpert] {arm} keyboard reconnect succeeded "
                f"attempt={attempt} path={getattr(device, 'path', path)} "
                f"name={getattr(device, 'name', 'unknown')}"
            )
            print(message, flush=True)
            return device
        return None

    def _handle_key_event(
        self,
        arm: str,
        key_char: str,
        is_pressed: bool,
        *,
        key_down: bool | None = None,
        now: float | None = None,
    ):
        """Apply one decoded device event. Public-by-convention for hardware-free tests."""
        if arm not in self.ARMS:
            raise ValueError(f"Unknown arm {arm!r}")
        with self._lock:
            was_pressed = key_char in self._pressed_keys[arm]
            new_key_down = bool(is_pressed and not was_pressed) if key_down is None else bool(key_down)

            if key_char == "/":
                if new_key_down:
                    self._step_modes[arm] = 1 - self._step_modes[arm]
                if is_pressed:
                    self._pressed_keys[arm].add(key_char)
                else:
                    self._pressed_keys[arm].discard(key_char)
                return

            if key_char == ".":
                if new_key_down and arm == self.reward_keyboard_arm:
                    timestamp = time.monotonic() if now is None else float(now)
                    if (
                        self._last_completion_press is not None
                        and timestamp - self._last_completion_press < self.completion_double_press_interval
                    ):
                        if self.completion_event is not None:
                            self.completion_event.set()
                        self._last_completion_press = None
                    else:
                        self._last_completion_press = timestamp
                if is_pressed:
                    self._pressed_keys[arm].add(key_char)
                else:
                    self._pressed_keys[arm].discard(key_char)
                return

            if key_char not in self.KEY_MAPPING:
                return
            if is_pressed:
                self._pressed_keys[arm].add(key_char)
            else:
                self._pressed_keys[arm].discard(key_char)

    def _active_step_sizes(self, arm: str):
        if self._step_modes[arm] == 1:
            return self.alternate_step_sizes[arm]
        return self.default_step_sizes[arm]

    @property
    def step_modes(self):
        with self._lock:
            return dict(self._step_modes)

    def get_action(self) -> tuple[list[float], list[int]]:
        action, buttons, _ = self.get_action_state()
        return action, buttons

    def get_action_state(self):
        """Return one atomic action, gripper-button, and per-arm stop snapshot."""
        action = np.zeros((12,), dtype=np.float32)
        buttons = [0, 0, 0, 0]  # left close/open, right close/open
        stop_buttons = [0, 0]  # left stop, right stop
        with self._lock:
            pressed = {arm: set(keys) for arm, keys in self._pressed_keys.items()}
            step_sizes = {arm: self._active_step_sizes(arm) for arm in self.ARMS}

        for arm_index, arm in enumerate(self.ARMS):
            action_offset = 6 * arm_index
            button_offset = 2 * arm_index
            step_size_pos, step_size_rot = step_sizes[arm]
            stop_pressed = "[" in pressed[arm]
            stop_buttons[arm_index] = int(stop_pressed)
            for key_char in pressed[arm]:
                mapping = self.KEY_MAPPING.get(key_char)
                if mapping is None:
                    continue
                # Stop wins over all commands from the same keyboard. The
                # other physical keyboard and arm remain independent.
                if mapping[0] == "stop" or stop_pressed:
                    continue
                if mapping[0] == "close":
                    buttons[button_offset] = mapping[1]
                elif mapping[0] == "open":
                    buttons[button_offset + 1] = mapping[1]
                else:
                    dim, direction = mapping
                    scale = step_size_rot if dim >= 3 else step_size_pos
                    action[action_offset + dim] += direction * scale
        return action.tolist(), buttons, stop_buttons

    def get_buttons(self) -> list[int]:
        return self.get_action()[1]

    def reset_step_mode(self):
        with self._lock:
            self._step_modes = {arm: 0 for arm in self.ARMS}
            self._pressed_keys = {arm: set() for arm in self.ARMS}
            self._last_completion_press = None
        if self.completion_event is not None:
            self.completion_event.clear()

    def close(self):
        self._stop_event.set()
        for device in list(self._devices.values()):
            try:
                device.close()
            except Exception:  # Close every pluggable input backend.
                LOGGER.exception("failed to close keyboard device")
        for thread in self._reader_threads:
            if thread is not threading.current_thread():
                thread.join(timeout=0.2)
        self._devices.clear()
        self._reader_threads.clear()
