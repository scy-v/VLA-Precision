"""Configuration for direct OpenPI inference on physical hardware."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    """Choose local policy execution or OpenPI's native WebSocket transport."""

    location: str = "local"
    host: str = "127.0.0.1"
    bind_host: str = "0.0.0.0"
    port: int = 8000
    cuda_visible_devices: str = "0"


@dataclass(frozen=True)
class NativeInferenceConfig:
    """One self-contained hardware inference YAML."""

    path: Path
    raw: dict[str, Any]
    experiment_name: str
    backend: str
    policy: PolicyRuntimeConfig


def load_native_inference_config(path: str | Path) -> NativeInferenceConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    experiment = raw.get("experiment", {})
    robot = raw.get("robot", {})
    policy_data = raw.get("policy", {})
    policy = PolicyRuntimeConfig(
        location=str(policy_data.get("location", "local")),
        host=str(policy_data.get("host", "127.0.0.1")),
        bind_host=str(policy_data.get("bind_host", "0.0.0.0")),
        port=int(policy_data.get("port", 8000)),
        cuda_visible_devices=str(policy_data.get("cuda_visible_devices", "0")),
    )
    if policy.location not in {"local", "server"}:
        raise ValueError("policy.location must be 'local' or 'server'")

    backend = str(robot.get("kind", "ur"))
    supported = {"ur", "dual_ur", "franka"}
    if backend not in supported:
        raise ValueError(f"robot.kind must be one of {sorted(supported)}, got {backend!r}")
    experiment_name = str(experiment.get("name", "")).strip()
    if not experiment_name:
        raise ValueError("experiment.name is required")
    if "task" not in raw or "model" not in raw or "cameras" not in raw:
        raise ValueError("task, model and cameras are required")

    return NativeInferenceConfig(
        path=config_path,
        raw=raw,
        experiment_name=experiment_name,
        backend=backend,
        policy=policy,
    )
