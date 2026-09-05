"""Load task, deployment and command-line values into one immutable config."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import tyro
from omegaconf import DictConfig, ListConfig, OmegaConf

from vla_precision.config.schema import CameraDeviceConfig, RootConfig, Stage1Config

_SHARED_SECTIONS = (
    "schema_version",
    "experiment",
    "task",
    "robot_server",
    "openpi",
    "acob",
    "stream",
    "buffers",
    "actor",
    "learner",
    "checkpoint",
)

_SHARED_FIELDS = {
    "data": (
        "lerobot_repo_id",
        "state_key",
        "state_indices",
        "action_key",
        "action_indices",
        "extra_delta_transform",
        "prompt_from_task",
        "image_key_map",
    ),
    "robot": (
        "kind",
        "control_hz",
        "arms",
        "random_reset",
        "delta_action_mask",
        "idle_hold_enabled",
        "idle_position_threshold",
        "idle_rotation_threshold",
        "options",
    ),
    "gripper": (
        "kind",
        "start_position",
        "left_start_position",
        "right_start_position",
        "left",
        "right",
        "options",
    ),
    "teleoperation": (
        "kind",
        "completion_double_press_interval",
        "step_size_position",
        "step_size_position_alt",
        "step_size_rotation",
        "step_size_rotation_alt",
        "options",
    ),
}

_SHARED_PREPROCESS_FIELDS = ("num_episodes", "chunk_mode", "bc_only")


@dataclass(frozen=True)
class ResolvedConfig:
    config: RootConfig
    shared_sha256: str
    preprocess_sha256: str

    def save(self, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.structured(self.config), destination)
        return destination


@dataclass(frozen=True)
class ResolvedStage1Config:
    config: Stage1Config
    config_sha256: str

    def save(self, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.structured(self.config), destination)
        return destination


def _shared_hash(config: RootConfig) -> str:
    root = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    shared = {name: root[name] for name in _SHARED_SECTIONS}
    # UR controller gains, force selections, payloads, reset poses and gripper
    # state machines are shared experiment semantics. Network/device locators
    # belong to the common deployment config and do not alter training semantics.
    shared["robot_server"].pop("locators")
    shared["actor"].pop("cuda_visible_devices")
    shared["learner"].pop("cuda_visible_devices")
    shared.update(
        {
            section: {name: root[section][name] for name in names}
            for section, names in _SHARED_FIELDS.items()
        }
    )
    shared["data"]["preprocess"] = {
        name: root["data"]["preprocess"][name] for name in _SHARED_PREPROCESS_FIELDS
    }
    payload = OmegaConf.to_yaml(OmegaConf.create(shared), resolve=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _preprocess_hash(config: RootConfig) -> str:
    """Hash only values that change materialized Stage-II cache contents."""
    root = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    payload = {
        "experiment": {name: root["experiment"][name] for name in ("name", "seed")},
        "task": {
            name: root["task"][name]
            for name in (
                "instruction",
                "arm_mode",
                "setup_mode",
                "image_keys",
                "proprio_keys",
                "action_horizon",
                "reward_function",
                "time_reward",
                "completion_reward",
            )
        },
        "data": {
            name: root["data"][name]
            for name in (
                "lerobot_repo_id",
                "state_key",
                "state_indices",
                "action_key",
                "action_indices",
                "extra_delta_transform",
                "prompt_from_task",
                "image_key_map",
            )
        },
        "openpi": {
            name: root["openpi"][name]
            for name in (
                "model",
                "name",
                "initialization_config_name",
                "action_dim",
                "action_horizon",
                "max_token_len",
                "initialization_checkpoint",
                "initialization_checkpoint_step",
            )
        },
        "acob": {
            name: root["acob"][name]
            for name in ("discount", "reward_scale", "reward_bias", "dense_reward")
        },
    }
    payload["data"]["preprocess"] = {
        name: root["data"]["preprocess"][name] for name in ("num_episodes", "chunk_mode")
    }
    serialized = OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _unlock_merge_template(node) -> None:
    """Temporarily unlock every frozen dataclass container for OmegaConf merge."""
    OmegaConf.set_readonly(node, False)
    if isinstance(node, DictConfig):
        for child in node.values():
            if isinstance(child, (DictConfig, ListConfig)):
                _unlock_merge_template(child)
    elif isinstance(node, ListConfig):
        for child in node:
            if isinstance(child, (DictConfig, ListConfig)):
                _unlock_merge_template(child)


def _freeze_container_values(value):
    """Restore tuple annotations after YAML list decoding and rebuild dataclasses."""
    if is_dataclass(value):
        return type(value)(
            **{item.name: _freeze_container_values(getattr(value, item.name)) for item in fields(value)}
        )
    if isinstance(value, list):
        return tuple(_freeze_container_values(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze_container_values(item) for key, item in value.items()}
    return value


def _seed_camera_device_schema(schema: DictConfig, layers: Sequence[DictConfig]) -> None:
    """Create typed dynamic camera entries before merging frozen configs."""
    for layer in layers:
        for name in OmegaConf.select(layer, "cameras.devices", default={}) or {}:
            if name not in schema.cameras.devices:
                device = OmegaConf.structured(CameraDeviceConfig)
                _unlock_merge_template(device)
                schema.cameras.devices[name] = device
    _unlock_merge_template(schema)


def _validate_config(config: RootConfig) -> None:
    """Check only invariants that protect model/data compatibility or hardware."""
    if config.schema_version != 1:
        raise ValueError(f"Unsupported schema_version={config.schema_version}")

    branch_modes = (
        ("task.arm_mode", config.task.arm_mode, {"single", "dual"}),
        ("cameras.capture_mode", config.cameras.capture_mode, {"sync", "async"}),
        (
            "data.preprocess.chunk_mode",
            config.data.preprocess.chunk_mode,
            {"overlap", "non_overlap"},
        ),
    )
    for name, value, choices in branch_modes:
        if value not in choices:
            raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")

    expected_prefix = "dual-arm-" if config.task.arm_mode == "dual" else "single-arm-"
    if not config.task.setup_mode.startswith(expected_prefix):
        raise ValueError(
            f"task.arm_mode={config.task.arm_mode!r} does not match "
            f"task.setup_mode={config.task.setup_mode!r}"
        )

    if config.task.action_horizon < 1:
        raise ValueError("task.action_horizon must be positive")
    if (
        config.openpi.action_horizon is not None
        and config.openpi.action_horizon < config.task.action_horizon
    ):
        raise ValueError("openpi.action_horizon cannot be shorter than task.action_horizon")
    if not config.task.image_keys or not config.task.proprio_keys:
        raise ValueError("task.image_keys and task.proprio_keys must be configured")
    if config.learner.batch_size < 2:
        raise ValueError("learner.batch_size must be at least two for replay/correction sampling")


def load_config(
    task: str | Path,
    *,
    deployment: str | Path | None = None,
    cli_args: Sequence[str] = (),
) -> ResolvedConfig:
    """Resolve defaults < task YAML < deployment YAML < Tyro overrides."""
    schema = OmegaConf.structured(RootConfig)
    # Frozen dataclasses make the structured template read-only. Temporarily
    # unlock only the merge container; ``to_object`` below returns frozen
    # dataclass instances again.
    _unlock_merge_template(schema)
    value_layers = [OmegaConf.load(Path(task))]
    if deployment is not None:
        value_layers.append(OmegaConf.load(Path(deployment)))

    # ``dict[str, CameraDeviceConfig]`` has dynamic keys. OmegaConf creates new
    # frozen dataclass nodes as read-only during merge, so seed only those typed
    # entries before applying task/deployment values.
    _seed_camera_device_schema(schema, value_layers)

    merged = OmegaConf.merge(schema, *value_layers)
    OmegaConf.resolve(merged)
    config = _freeze_container_values(OmegaConf.to_object(merged))
    if cli_args:
        config = tyro.cli(RootConfig, default=config, args=list(cli_args))

    _validate_config(config)
    return ResolvedConfig(
        config=config,
        shared_sha256=_shared_hash(config),
        preprocess_sha256=_preprocess_hash(config),
    )


def load_stage1_config(
    experiment: str | Path,
    *,
    cli_args: Sequence[str] = (),
) -> ResolvedStage1Config:
    """Resolve one self-contained Stage-I config."""
    schema = OmegaConf.structured(Stage1Config)
    _unlock_merge_template(schema)
    layers = [OmegaConf.load(Path(experiment))]
    merged = OmegaConf.merge(schema, *layers)
    OmegaConf.resolve(merged)
    config = _freeze_container_values(OmegaConf.to_object(merged))
    if cli_args:
        config = tyro.cli(Stage1Config, default=config, args=list(cli_args))
    if config.schema_version != 1:
        raise ValueError(f"Unsupported Stage-I schema_version={config.schema_version}")
    if config.openpi.model.action_dim < 1 or config.openpi.model.action_horizon < 1:
        raise ValueError("Stage-I action_dim and action_horizon must be positive")
    if config.openpi.resume and config.openpi.overwrite:
        raise ValueError("Stage-I resume and overwrite cannot both be enabled")
    payload = OmegaConf.to_yaml(OmegaConf.structured(config), resolve=True, sort_keys=True)
    return ResolvedStage1Config(
        config=config,
        config_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
