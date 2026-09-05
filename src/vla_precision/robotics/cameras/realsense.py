"""Intel RealSense implementation of the camera capability boundary."""

from __future__ import annotations

import numpy as np


class RealSenseCamera:
    """Color/depth capture with the established stream and alignment settings."""

    def __init__(
        self,
        *,
        name: str,
        serial_number: str,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        depth: bool = False,
        exposure: int = 40_000,
    ):
        import pyrealsense2 as rs

        self._rs = rs
        self._name = name
        self.serial_number = serial_number
        self.depth = bool(depth)
        available = [
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().devices
        ]
        if serial_number not in available:
            raise RuntimeError(
                f"RealSense {name!r} serial={serial_number} is not available; detected_serials={available}"
            )

        self.pipe = rs.pipeline()
        self.stream = rs.config()
        self.stream.enable_device(serial_number)
        self.stream.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if self.depth:
            self.stream.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._started = False
        self._closed = False
        try:
            profile = self.pipe.start(self.stream)
            self._started = True
            profile.get_device().query_sensors()[0].set_option(rs.option.exposure, exposure)
            self.align = rs.align(rs.stream.color)
        except Exception:
            self.close()
            raise

    @property
    def name(self) -> str:
        return self._name

    def capture(self) -> np.ndarray:
        frames = self.align.process(self.pipe.wait_for_frames())
        color_frame = frames.get_color_frame()
        if not color_frame.is_video_frame():
            raise RuntimeError(f"RealSense {self.name!r} returned no color frame")
        image = np.asarray(color_frame.get_data())
        if not self.depth:
            return image
        depth_frame = frames.get_depth_frame()
        if not depth_frame.is_depth_frame():
            raise RuntimeError(f"RealSense {self.name!r} returned no depth frame")
        depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
        return np.concatenate((image, depth), axis=-1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._started:
                self.pipe.stop()
        finally:
            self._started = False
            self.stream.disable_all_streams()
