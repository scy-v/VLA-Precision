"""Gym environment that composes a robot, cameras, gripper and detectors."""

from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from vla_precision import image_tools
from vla_precision.robotics.cameras.base import Camera
from vla_precision.robotics.cameras.factory import close_cameras
from vla_precision.robotics.robots.base import Robot

LOGGER = logging.getLogger(__name__)
TRANSITION_LOGGER = logging.getLogger("transition")


def _state_space(proprio_keys: tuple[str, ...]) -> gym.spaces.Dict:
    spaces: dict[str, gym.Space] = {}
    for key in proprio_keys:
        field = key.split("/", 1)[-1]
        size = {"tcp_pose": 7, "tcp_vel": 6, "tcp_force": 3, "tcp_torque": 3}.get(field, 1)
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
    return gym.spaces.Dict(spaces)


class _ImageDisplayer(threading.Thread):
    def __init__(self, frames: queue.Queue, name: str):
        super().__init__(name=f"acob-stream-display-{name}", daemon=True)
        self.frames = frames
        self.window_name = name

    def run(self) -> None:
        import cv2

        while True:
            images = self.frames.get()
            if images is None:
                return
            frame = np.concatenate(
                [cv2.resize(image, (128, 128)) for key, image in images.items() if "full" not in key],
                axis=1,
            )
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)


class RobotEnvironment(gym.Env):
    """Keep hardware execution in components while retaining the original step sequence."""

    def __init__(
        self,
        *,
        robot: Robot,
        cameras: dict[str, Camera],
        camera_builder: Callable[[], dict[str, Camera]] | None,
        proprio_keys: tuple[str, ...],
        image_keys: tuple[str, ...],
        camera_options: dict[str, dict],
        max_episode_length: int,
        transition_log_interval: int = 10,
        reconnect_delay: float = 2.0,
        display_images: bool = False,
        save_video: bool = False,
        video_directory: Path = Path("videos"),
        emergency_stop=None,
    ):
        self.robot = robot
        self.cameras = cameras
        self.camera_builder = camera_builder
        self.camera_options = camera_options
        self.max_episode_length = int(max_episode_length)
        self.transition_log_interval = int(transition_log_interval)
        self.reconnect_delay = float(reconnect_delay)
        self.display_images = bool(display_images)
        self.save_video = bool(save_video)
        self.video_directory = Path(video_directory)
        self.emergency_stop = emergency_stop
        self.fixed_gripper = robot.gripper.action_dimension == 0
        self.gripper_type = getattr(robot.gripper, "kind", "fixed" if self.fixed_gripper else "unknown")
        self.robot_server_config = getattr(robot, "server_config", {})
        self.curr_path_length = 0
        self._pending_action_steps = 0
        self._executed_actions: list[np.ndarray] = []
        self.recording_frames: list[dict[str, np.ndarray]] = []

        self.action_space = gym.spaces.Box(
            float(getattr(robot, "action_low", -3.0)),
            float(getattr(robot, "action_high", 3.0)),
            shape=(robot.action_dimension,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict({
            "state": _state_space(proprio_keys),
            "images": gym.spaces.Dict({
                key: gym.spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8)
                for key in image_keys
            }),
        })

        self._display_queue = None
        self._displayer = None
        if self.display_images:
            self._display_queue = queue.Queue()
            self._displayer = _ImageDisplayer(self._display_queue, type(robot).__name__)
            self._displayer.start()

    @property
    def currpos(self):
        return self.robot.currpos

    @property
    def right_currpos(self):
        return self.robot.right_currpos

    def _update_currpos(self) -> None:
        self.robot.refresh_state()

    def request(self, name: str, param: bool):
        return self.robot.request(name, param)

    def _capture_images(self) -> dict[str, np.ndarray]:
        reconnect_cycle = 0
        while True:
            images = {}
            display = {}
            full_resolution = {}
            try:
                for name, camera in self.cameras.items():
                    bgr = camera.capture()
                    crop = self.camera_options.get(name, {}).get("crop")
                    if crop:
                        top, bottom, left, right = map(int, crop)
                        bgr = bgr[top:bottom, left:right]
                    # Every migrated task used OpenPI's ``resize_with_pad``
                    # before the final (normally no-op) OpenCV resize.  Keep
                    # that aspect-preserving, zero-padded image contract: a
                    # direct resize changes the policy input distribution.
                    resized = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(bgr, 224, 224)
                    )
                    images[name] = resized[..., ::-1]
                    display[name] = resized
                    display[name + "_full"] = resized
                    full_resolution[name] = copy.deepcopy(resized)
                break
            except (queue.Empty, RuntimeError, OSError):
                if self.camera_builder is None:
                    raise
                reconnect_cycle += 1
                close_cameras(self.cameras)
                while True:
                    if self.reconnect_delay > 0:
                        time.sleep(self.reconnect_delay)
                    try:
                        self.cameras = self.camera_builder()
                    except Exception:  # noqa: BLE001, S112 -- reconnect every pluggable camera driver
                        continue
                    break
        if self.save_video:
            self.recording_frames.append(full_resolution)
        if self._display_queue is not None:
            self._display_queue.put(display)
        return images

    def _observation(self, state: dict | None = None) -> dict:
        return {
            "state": self.robot.observations() if state is None else state,
            "images": self._capture_images(),
        }

    def execute_action_chunk(self, action: np.ndarray) -> int:
        executed = self.robot.execute_action_chunk(action)
        self._executed_actions.append(executed)
        steps = len(executed)
        self._pending_action_steps += steps
        return steps

    def finish_action_chunk_step(self):
        observation = self._observation()
        steps = self._pending_action_steps or 1
        self._pending_action_steps = 0
        self.curr_path_length += 1
        if (
            self.transition_log_interval > 0
            and (
                self.curr_path_length % self.transition_log_interval == 0
                or self.curr_path_length >= self.max_episode_length
            )
        ):
            episode_limit = (
                str(int(self.max_episode_length))
                if np.isfinite(self.max_episode_length)
                else "inf"
            )
            TRANSITION_LOGGER.info(
                "%d/%s",
                self.curr_path_length,
                episode_limit,
            )
        stopped = bool(self.emergency_stop and self.emergency_stop.stopped())
        terminated = self.curr_path_length >= self.max_episode_length or stopped
        rewards = np.zeros((steps,), dtype=np.float32)
        info = {
            "succeed": False,
            "steps_executed": steps,
            "transition_index": self.curr_path_length,
        }
        if self._executed_actions:
            info["executed_action"] = np.concatenate(self._executed_actions, axis=0)
            self._executed_actions.clear()
        mask = getattr(self.robot, "delta_action_mask", None)
        if mask is not None:
            info["delta_action_mask"] = np.asarray(mask, dtype=np.float32)
        return observation, rewards, terminated, False, info

    def step(self, action):
        self.execute_action_chunk(action)
        return self.finish_action_chunk_step()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.save_video:
            self._save_video_recording()
        joint_reset = bool((options or {}).get("joint_reset", False))
        state = self.robot.reset(joint_reset=joint_reset, options=options or {})
        self.curr_path_length = 0
        self._pending_action_steps = 0
        self._executed_actions.clear()
        if self.emergency_stop:
            self.emergency_stop.reset()
        return self._observation(state), {"succeed": False}

    def _save_video_recording(self) -> None:
        if not self.recording_frames:
            return
        try:
            import cv2

            self.video_directory.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
            for camera_name in self.recording_frames[0]:
                first = self.recording_frames[0][camera_name]
                height, width = first.shape[:2]
                writer = cv2.VideoWriter(
                    str(self.video_directory / f"{camera_name}_{timestamp}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (width, height),
                )
                for frames in self.recording_frames:
                    writer.write(frames[camera_name])
                writer.release()
            self.recording_frames.clear()
        except Exception:  # Recording failure never blocks a hardware reset.
            LOGGER.exception("failed to save episode video")

    def close(self):
        close_cameras(self.cameras)
        self.robot.close()
        if self.emergency_stop:
            self.emergency_stop.close()
        if self._display_queue is not None:
            import cv2

            self._display_queue.put(None)
            cv2.destroyAllWindows()
            self._displayer.join()
