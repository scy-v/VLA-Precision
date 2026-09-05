"""Direct hardware OpenPI inference for VLA-Precision."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

from vla_precision.integrations.openpi.inference.config import NativeInferenceConfig

LOGGER = logging.getLogger(__name__)


def _backend_class(name: str):
    if name == "ur":
        from .backends.ur import Inference

        return Inference
    if name == "dual_ur":
        from .backends.dual_ur import DualInference

        return DualInference
    if name == "franka":
        from .backends.franka import Inference

        return Inference
    raise ValueError(f"Unsupported native inference backend: {name}")


def _remote_policy(config: NativeInferenceConfig):
    from openpi_client import websocket_client_policy

    LOGGER.info(
        "OpenPI policy client: ws://%s:%d",
        config.policy.host,
        config.policy.port,
    )
    return websocket_client_policy.WebsocketClientPolicy(
        host=config.policy.host,
        port=config.policy.port,
    )


def _use_remote_policy(engine: Any, config: NativeInferenceConfig) -> None:
    def create_policy(_self):
        return _remote_policy(config)

    engine._create_openpi_policy = MethodType(create_policy, engine)


def _attach_result_writer(engine: Any, config: NativeInferenceConfig) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    path = (
        Path(config.raw.get("output_root", "./results")).expanduser()
        / config.experiment_name
        / "openpi-native"
        / f"{timestamp}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    original = engine.recorder._store_episode_result

    def write() -> None:
        episodes = list(engine.recorder.episode_results)
        successes = sum(bool(item.get("completed")) for item in episodes)
        payload = {
            "experiment": config.experiment_name,
            "evaluation": "openpi-native",
            "policy_location": config.policy.location,
            "episodes": episodes,
            "summary": {
                "episode_count": len(episodes),
                "success_count": successes,
                "success_rate": successes / len(episodes) if episodes else 0.0,
            },
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def store(_recorder, result):
        original(result)
        write()

    engine.recorder._store_episode_result = MethodType(store, engine.recorder)
    write()
    return path


def _checkpoint_directory(model: dict[str, Any]) -> Path:
    root = Path(model["checkpoint_dir"]).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    step = model.get("checkpoint_step")
    if step not in (None, 0, "0"):
        return root if root.name == str(step) else root / str(step)
    if (root / "params").exists():
        return root
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and child.name.isdigit() and (child / "params").exists()
    ]
    if not candidates:
        return root
    return max(candidates, key=lambda child: int(child.name))


def _create_local_policy(config: NativeInferenceConfig):
    """Create a policy without importing any robot-side hardware package."""
    from vla_precision.integrations.openpi import install_lerobot_import_compat

    install_lerobot_import_compat()
    from openpi.policies import policy_config

    from vla_precision.integrations.openpi import configs

    model = config.raw["model"]
    robot = config.raw["robot"]
    train_config = configs.get_config(str(model["name"]))
    action_horizon = int(robot["action_horizon"])
    if bool(robot.get("direct_action_horizon", True)):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, action_horizon=action_horizon),
        )
    checkpoint = _checkpoint_directory(model)
    assets_root = checkpoint / "assets"
    norm_stats = sorted(assets_root.rglob("norm_stats.json"))
    if norm_stats:
        asset_id = norm_stats[0].parent.relative_to(assets_root).as_posix()
        assets = dataclasses.replace(
            train_config.data.assets,
            assets_dir=str(assets_root),
            asset_id=asset_id,
        )
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, assets=assets),
        )
    LOGGER.info("OpenPI checkpoint: %s", checkpoint)
    return policy_config.create_trained_policy(
        train_config,
        checkpoint,
        sample_kwargs={"num_steps": int(model.get("sample_steps", 10))},
    )


def run_native_inference(config: NativeInferenceConfig) -> None:
    """Run robot, gripper, cameras and policy client in one local process."""
    if config.policy.location == "local":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.policy.cuda_visible_devices
        from vla_precision.integrations.openpi import install_lerobot_import_compat

        install_lerobot_import_compat()
    backend = _backend_class(config.backend)
    engine = backend(config.path)
    if config.policy.location == "server":
        _use_remote_policy(engine, config)
    result_path = _attach_result_writer(engine, config)
    LOGGER.info(
        "OpenPI inference: experiment=%s backend=%s policy=%s",
        config.experiment_name,
        config.backend,
        config.policy.location,
    )
    LOGGER.info("OpenPI evaluation result: %s", result_path)
    engine.run()


def serve_native_policy(config: NativeInferenceConfig) -> None:
    """Load the configured checkpoint and serve it through OpenPI WebSocket."""
    os.environ["CUDA_VISIBLE_DEVICES"] = config.policy.cuda_visible_devices
    policy = _create_local_policy(config)

    from openpi.serving import websocket_policy_server

    LOGGER.info(
        "OpenPI policy server: experiment=%s bind=%s:%d",
        config.experiment_name,
        config.policy.bind_host,
        config.policy.port,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=config.policy.bind_host,
        port=config.policy.port,
        metadata=policy.metadata,
    )
    server.serve_forever()
