"""Construct cameras from the single top-level configuration."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable

from vla_precision.config.schema import CameraConfig, CameraDeviceConfig
from vla_precision.robotics.cameras.base import Camera
from vla_precision.robotics.cameras.latest_frame import LatestFrameCamera
from vla_precision.robotics.cameras.realsense import RealSenseCamera

CameraFactory = Callable[[str, CameraDeviceConfig], Camera]
LOGGER = logging.getLogger(__name__)


def _realsense(name: str, config: CameraDeviceConfig) -> Camera:
    options = config.options
    return RealSenseCamera(
        name=name,
        serial_number=config.serial_number,
        width=config.width,
        height=config.height,
        fps=config.fps,
        depth=bool(options.get("depth", False)),
        exposure=config.exposure if config.exposure is not None else 40_000,
    )


def build_cameras(
    config: CameraConfig,
    *,
    factories: dict[str, CameraFactory] | None = None,
) -> OrderedDict[str, Camera]:
    """Build enabled devices; callers can inject a new driver without changing the environment."""
    driver_factories = {"realsense": _realsense, **(factories or {})}
    cameras: OrderedDict[str, Camera] = OrderedDict()
    try:
        for name, device in config.devices.items():
            if not device.enabled:
                continue
            camera = driver_factories[device.driver](name, device)
            cameras[name] = LatestFrameCamera(camera) if config.capture_mode == "async" else camera
    except Exception:
        close_cameras(cameras)
        raise
    return cameras


def close_cameras(cameras: dict[str, Camera]) -> None:
    for name, camera in list(cameras.items()):
        try:
            camera.close()
        except Exception:  # One driver must not strand the remaining devices.
            LOGGER.exception("failed to close camera %s", name)
