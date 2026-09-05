"""Config-derived data layout shared by preprocessing, actor, and learner."""

from __future__ import annotations

from pathlib import Path

from vla_precision.config.schema import RootConfig, Stage1Config


def source_lerobot_root(config: RootConfig | Stage1Config) -> Path | None:
    """Resolve the dataset directory shared by Stage I and Stage II."""
    value = config.data.lerobot_root
    if value is None or value == "":
        return None

    root = Path(value).expanduser()
    if (root / "meta" / "info.json").exists():
        return root

    candidates = [root / config.data.lerobot_repo_id]
    if "/" in config.data.lerobot_repo_id:
        candidates.append(root / Path(config.data.lerobot_repo_id))
    for candidate in candidates:
        if (candidate / "meta" / "info.json").exists():
            return candidate
    return root


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def experiment_data_dir(config: RootConfig) -> Path:
    arm_suffix = "_dual_arm" if config.task.arm_mode == "dual" else ""
    name = f"{_safe_name(config.experiment.name)}_action_horizon_{config.task.action_horizon}{arm_suffix}"
    return Path(config.paths.train_data_root).expanduser() / name


def preprocess_transition_dir(config: RootConfig) -> Path:
    return experiment_data_dir(config) / "preprocess" / "transition"


def preprocess_context_dir(config: RootConfig) -> Path:
    return experiment_data_dir(config) / "preprocess" / "context"


def training_transition_dir(config: RootConfig) -> Path:
    return experiment_data_dir(config) / "training" / "transition"


def training_context_dir(config: RootConfig) -> Path:
    return experiment_data_dir(config) / "training" / "context"


def stage2_checkpoint_dir(config: RootConfig) -> Path:
    return Path(config.paths.checkpoint_root).expanduser() / "stage2" / config.experiment.name


def stage2_actor_checkpoint_dir(config: RootConfig) -> Path:
    return stage2_checkpoint_dir(config) / "actor"


def initial_buffer_dir(config: RootConfig) -> Path:
    # Critic warmup deliberately owns a separate reusable fill cache. This
    # prevents a run without warmup from silently reusing the different
    # correction/context barrier collected for a warmup run (and vice versa).
    stage = "initial_buffer_warmup" if config.stream.critic_warmup_steps > 0 else "initial_buffer"
    return experiment_data_dir(config) / stage


def initial_replay_path(config: RootConfig) -> Path:
    return initial_buffer_dir(config) / "transition" / "replay_buffer" / "transitions.pkl"


def initial_correction_path(config: RootConfig) -> Path:
    return initial_buffer_dir(config) / "transition" / "correction_buffer" / "transitions.pkl"


def initial_context_dir(config: RootConfig) -> Path:
    return initial_buffer_dir(config) / "context"
