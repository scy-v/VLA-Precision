"""Keep the paper's LeRobot revision usable with the pinned OpenPI snapshot."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def install_lerobot_import_compat() -> None:
    """Expose the modern LeRobot dataset module at OpenPI's legacy import path."""
    legacy_name = "lerobot.common.datasets.lerobot_dataset"
    try:
        importlib.import_module(legacy_name)
        return
    except ModuleNotFoundError as error:
        if error.name not in {"lerobot.common", "lerobot.common.datasets", legacy_name}:
            raise

    lerobot = importlib.import_module("lerobot")
    modern = importlib.import_module("lerobot.datasets.lerobot_dataset")

    common = ModuleType("lerobot.common")
    common.__path__ = []
    datasets = ModuleType("lerobot.common.datasets")
    datasets.__path__ = []
    datasets.lerobot_dataset = modern
    common.datasets = datasets
    lerobot.common = common

    sys.modules["lerobot.common"] = common
    sys.modules["lerobot.common.datasets"] = datasets
    sys.modules[legacy_name] = modern
