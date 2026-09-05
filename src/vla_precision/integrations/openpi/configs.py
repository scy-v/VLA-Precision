"""Project-owned OpenPI configs, declared in the native OpenPI style."""

from __future__ import annotations

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import dataclasses
import difflib

from flax import nnx
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import config as openpi_config
from openpi.training import optimizer, weight_loaders

from vla_precision.config.schema import RootConfig, Stage1Config
from vla_precision.integrations.openpi.checkpoints import (
    resolve_openpi_assets,
    resolve_openpi_checkpoint,
)
from vla_precision.integrations.openpi.data_configs import (
    LeRobotDualUR5eDataConfig,
    LeRobotFrankaDataConfig,
    LeRobotUR5eDataConfig,
    make_robot_data_config_template,
)

OPENPI_UPSTREAM_COMMIT = "2d70d966582e711128ad8358d8dbf23d2cc3d658"

_ACTION_EXPERT_ONLY_FREEZE_FILTER = nnx.Not(
    nnx.All(
        nnx_utils.PathRegex(".*llm.*_1.*"),
        nnx_utils.PathRegex(".*lora.*"),
    )
)


# User-facing configuration registry. Like OpenPI's own config.py, each
# supported configuration is a concrete TrainConfig near the top of this file.
# Adding a robot/config means adding a DataConfigFactory and one
# explicit TrainConfig here; no secondary stage/model/platform profile exists.
_CONFIGS = [
    openpi_config.TrainConfig(
        name="pi0_full_finetune_ur5e",
        model=pi0_config.Pi0Config(),
        data=make_robot_data_config_template(LeRobotUR5eDataConfig),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
    ),
    openpi_config.TrainConfig(
        name="pi05_full_finetune_ur5e",
        model=pi0_config.Pi0Config(pi05=True),
        data=make_robot_data_config_template(LeRobotUR5eDataConfig),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    openpi_config.TrainConfig(
        name="pi0_full_finetune_dual_ur",
        model=pi0_config.Pi0Config(),
        data=make_robot_data_config_template(LeRobotDualUR5eDataConfig, dual=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
    ),
    openpi_config.TrainConfig(
        name="pi05_full_finetune_dual_ur",
        model=pi0_config.Pi0Config(pi05=True),
        data=make_robot_data_config_template(LeRobotDualUR5eDataConfig, dual=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    openpi_config.TrainConfig(
        name="pi0_full_finetune_franka",
        model=pi0_config.Pi0Config(),
        data=make_robot_data_config_template(LeRobotFrankaDataConfig),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
    ),
    openpi_config.TrainConfig(
        name="pi05_full_finetune_franka",
        model=pi0_config.Pi0Config(pi05=True),
        data=make_robot_data_config_template(LeRobotFrankaDataConfig),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
    openpi_config.TrainConfig(
        name="pi0_acob_ur5e",
        model=pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora"),
        data=make_robot_data_config_template(LeRobotUR5eDataConfig),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        freeze_filter=_ACTION_EXPERT_ONLY_FREEZE_FILTER,
        ema_decay=None,
    ),
    openpi_config.TrainConfig(
        name="pi05_acob_ur5e",
        model=pi0_config.Pi0Config(pi05=True, action_expert_variant="gemma_300m_lora"),
        data=make_robot_data_config_template(LeRobotUR5eDataConfig),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=_ACTION_EXPERT_ONLY_FREEZE_FILTER,
        ema_decay=None,
    ),
    openpi_config.TrainConfig(
        name="pi0_acob_dual_ur",
        model=pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora"),
        data=make_robot_data_config_template(LeRobotDualUR5eDataConfig, dual=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        freeze_filter=_ACTION_EXPERT_ONLY_FREEZE_FILTER,
        ema_decay=None,
    ),
    openpi_config.TrainConfig(
        name="pi05_acob_dual_ur",
        model=pi0_config.Pi0Config(pi05=True, action_expert_variant="gemma_300m_lora"),
        data=make_robot_data_config_template(LeRobotDualUR5eDataConfig, dual=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=_ACTION_EXPERT_ONLY_FREEZE_FILTER,
        ema_decay=None,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("OpenPI extension config names must be unique")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def action_expert_only_freeze_filter() -> nnx.filterlib.Filter:
    """Freeze every parameter except LoRA leaves in the action expert LLM."""
    return _ACTION_EXPERT_ONLY_FREEZE_FILTER


def get_config(name: str) -> openpi_config.TrainConfig:
    try:
        return _CONFIGS_DICT[name]
    except KeyError as error:
        closest = difflib.get_close_matches(name, _CONFIGS_DICT, n=1, cutoff=0.0)
        hint = f" Did you mean {closest[0]!r}?" if closest else ""
        raise ValueError(f"Unknown VLA-Precision OpenPI config {name!r}.{hint}") from error


def supported_configs() -> tuple[str, ...]:
    return tuple(_CONFIGS_DICT)


def _overlay_data(base, data):
    base_config = dataclasses.replace(
        base.base_config or openpi_config.DataConfig(),
        prompt_from_task=data.prompt_from_task,
        action_sequence_keys=(data.action_key,),
    )
    return dataclasses.replace(
        base,
        repo_id=data.lerobot_repo_id,
        base_config=base_config,
        state_key=data.state_key,
        action_key=data.action_key,
        image_key_map=dict(data.image_key_map),
        extra_delta_transform=data.extra_delta_transform,
    )


def _lr(value):
    return optimizer.CosineDecaySchedule(
        warmup_steps=value.warmup_steps,
        peak_lr=value.peak_lr,
        decay_steps=value.decay_steps,
        decay_lr=value.decay_lr,
    )


def _optimizer(value):
    return optimizer.AdamW(
        b1=value.b1,
        b2=value.b2,
        eps=value.eps,
        weight_decay=value.weight_decay,
        clip_gradient_norm=value.clip_gradient_norm,
    )


def _stage2_base_configs(
    config: RootConfig,
) -> tuple[openpi_config.TrainConfig, openpi_config.TrainConfig]:
    actor = get_config(config.openpi.name)
    initialization = get_config(config.openpi.initialization_config_name)
    if "_acob_" not in actor.name:
        raise ValueError(f"Stage-II config must be an ACoB config, got {actor.name!r}")
    if "_full_finetune_" not in initialization.name:
        raise ValueError(
            "Stage-II initialization_config_name must select a full-finetune config"
        )
    pi05 = config.openpi.model == "pi05"
    if config.openpi.model not in {"pi0", "pi05"}:
        raise ValueError("openpi.model must be 'pi0' or 'pi05'")
    if actor.model.pi05 is not pi05 or initialization.model.pi05 is not pi05:
        raise ValueError("Stage-II OpenPI config names do not match openpi.model")
    if type(actor.data) is not type(initialization.data):
        raise ValueError("Stage-II actor and initialization configs must use the same robot data config")
    return actor, initialization


def build_stage1_train_config(config: Stage1Config) -> openpi_config.TrainConfig:
    base = get_config(config.openpi.name)
    if "_full_finetune_" not in base.name:
        raise ValueError(f"Stage-I config must be a full-finetune config, got {base.name!r}")
    model = dataclasses.replace(
        base.model,
        action_dim=config.openpi.model.action_dim,
        action_horizon=config.openpi.model.action_horizon,
        max_token_len=config.openpi.model.max_token_len,
    )
    return dataclasses.replace(
        base,
        project_name=config.openpi.project_name,
        exp_name=config.openpi.exp_name,
        model=model,
        data=_overlay_data(base.data, config.data),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            config.openpi.initialization_checkpoint
        ),
        lr_schedule=_lr(config.openpi.lr_schedule),
        optimizer=_optimizer(config.openpi.optimizer),
        ema_decay=config.openpi.ema_decay,
        assets_base_dir=config.paths.openpi_assets_root,
        checkpoint_base_dir=config.paths.checkpoint_root,
        seed=config.openpi.seed,
        batch_size=config.openpi.batch_size,
        num_workers=config.openpi.num_workers,
        num_train_steps=config.openpi.num_train_steps,
        log_interval=config.openpi.log_interval,
        save_interval=config.openpi.save_interval,
        keep_period=config.openpi.keep_period,
        overwrite=config.openpi.overwrite,
        resume=config.openpi.resume,
        wandb_enabled=config.openpi.wandb_enabled,
        fsdp_devices=config.openpi.fsdp_devices,
    )


def build_stage2_train_config(config: RootConfig) -> openpi_config.TrainConfig:
    base, _ = _stage2_base_configs(config)
    action_horizon = config.openpi.action_horizon or config.task.action_horizon
    model = dataclasses.replace(
        base.model,
        action_dim=config.openpi.action_dim,
        action_horizon=action_horizon,
        max_token_len=config.openpi.max_token_len,
    )
    data = _overlay_data(base.data, config.data)
    initialization = config.openpi.initialization_checkpoint
    if initialization:
        initialization_step = config.openpi.initialization_checkpoint_step
        checkpoint = resolve_openpi_checkpoint(
            initialization,
            requested_step=initialization_step,
        )
        weight_loader = weight_loaders.CheckpointWeightLoader(checkpoint.params_path)
        assets = resolve_openpi_assets(
            initialization,
            default_asset_id=config.data.lerobot_repo_id,
            requested_step=initialization_step,
        )
        if assets is not None:
            data = dataclasses.replace(
                data,
                assets=openpi_config.AssetsConfig(
                    assets_dir=assets.directory,
                    asset_id=assets.asset_id,
                ),
            )
    else:
        weight_loader = base.weight_loader
    return dataclasses.replace(
        base,
        project_name=config.logging.wandb.project,
        exp_name=config.experiment.name,
        model=model,
        data=data,
        weight_loader=weight_loader,
        lr_schedule=_lr(config.openpi.lr_schedule),
        optimizer=_optimizer(config.openpi.optimizer),
        assets_base_dir=config.paths.openpi_assets_root,
        checkpoint_base_dir=config.paths.checkpoint_root,
        seed=config.experiment.seed,
        batch_size=config.learner.batch_size,
        num_workers=config.openpi.num_workers,
        num_train_steps=config.stream.max_steps,
        save_interval=config.checkpoint.save_interval,
        keep_period=config.checkpoint.keep_period,
        overwrite=config.checkpoint.overwrite,
        resume=config.checkpoint.resume,
        wandb_enabled=config.logging.wandb.enabled,
        fsdp_devices=config.openpi.fsdp_devices,
    )


def build_stage2_initialization_train_config(
    config: RootConfig,
) -> openpi_config.TrainConfig:
    """Build the full-model config used to preprocess/evaluate Stage-I weights."""
    _, base = _stage2_base_configs(config)
    action_horizon = config.openpi.action_horizon or config.task.action_horizon
    model = dataclasses.replace(
        base.model,
        action_dim=config.openpi.action_dim,
        action_horizon=action_horizon,
        max_token_len=config.openpi.max_token_len,
    )
    data = _overlay_data(base.data, config.data)
    initialization = config.openpi.initialization_checkpoint
    weight_loader = base.weight_loader
    if initialization:
        initialization_step = config.openpi.initialization_checkpoint_step
        checkpoint = resolve_openpi_checkpoint(
            initialization,
            requested_step=initialization_step,
        )
        weight_loader = weight_loaders.CheckpointWeightLoader(checkpoint.params_path)
        assets = resolve_openpi_assets(
            initialization,
            default_asset_id=config.data.lerobot_repo_id,
            requested_step=initialization_step,
        )
        if assets is not None:
            data = dataclasses.replace(
                data,
                assets=openpi_config.AssetsConfig(
                    assets_dir=assets.directory,
                    asset_id=assets.asset_id,
                ),
            )
    return dataclasses.replace(
        base,
        project_name=config.logging.wandb.project,
        exp_name=config.experiment.name,
        model=model,
        data=data,
        weight_loader=weight_loader,
        assets_base_dir=config.paths.openpi_assets_root,
        checkpoint_base_dir=config.paths.checkpoint_root,
        seed=config.experiment.seed,
        wandb_enabled=config.logging.wandb.enabled,
        fsdp_devices=config.openpi.fsdp_devices,
    )
