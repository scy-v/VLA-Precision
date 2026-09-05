"""Load materialized Stage-II preprocessing data without hot-path transforms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vla_precision.acob_stream.buffers.context import ContextSource
from vla_precision.config.loader import ResolvedConfig
from vla_precision.data.paths import preprocess_transition_dir

_REQUIRED_CORE_FIELDS = (
    "source_repo_id",
    "dual_arm",
    "action_horizon",
    "preprocess_chunk_mode",
    "image_keys",
    "state_key",
    "state_indices",
    "action_key",
    "action_indices",
    "image_key_map",
    "time_reward",
    "completion_reward",
    "openpi_profile",
)


def _core_cache_inputs(resolved: ResolvedConfig) -> dict:
    config = resolved.config
    return {
        "source_repo_id": config.data.lerobot_repo_id,
        "dual_arm": config.task.arm_mode == "dual",
        "action_horizon": config.task.action_horizon,
        "preprocess_chunk_mode": config.data.preprocess.chunk_mode,
        "image_keys": list(config.task.image_keys),
        "state_key": config.data.state_key,
        "state_indices": list(config.data.state_indices),
        "action_key": config.data.action_key,
        "action_indices": list(config.data.action_indices),
        "image_key_map": dict(config.data.image_key_map),
        "time_reward": config.task.time_reward,
        "completion_reward": config.task.completion_reward,
        "openpi_profile": config.openpi.initialization_config_name,
        "openpi_model": config.openpi.model,
        "initialization_checkpoint_step": config.openpi.initialization_checkpoint_step,
        "discount": config.acob.discount,
        "reward_scale": config.acob.reward_scale,
        "reward_bias": config.acob.reward_bias,
        "dense_reward": config.acob.dense_reward,
    }


def load_cache_metadata(resolved: ResolvedConfig) -> tuple[Path, dict]:
    directory = preprocess_transition_dir(resolved.config)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Preprocessed train cache is missing: {metadata_path}. "
            "Run `python main.py --stage stage2 --mode preprocess ...` first."
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != "vla_precision_train_cache":
        raise ValueError(f"Unsupported train cache format at {metadata_path}: {metadata.get('format')!r}")

    expected = _core_cache_inputs(resolved)
    missing = [name for name in _REQUIRED_CORE_FIELDS if name not in metadata]
    if missing:
        raise ValueError(f"Preprocessed cache metadata is missing core fields at {metadata_path}: {missing}")
    mismatches = {
        name: (metadata[name], value)
        for name, value in expected.items()
        if name in metadata and metadata[name] != value
    }
    if mismatches:
        details = "; ".join(
            f"{name}: cache={cache_value!r}, runtime={runtime_value!r}"
            for name, (cache_value, runtime_value) in mismatches.items()
        )
        raise ValueError(
            f"Preprocessed cache does not match current core preprocessing inputs at {metadata_path}; {details}"
        )
    return directory, metadata


def load_preprocessed_transitions(resolved: ResolvedConfig) -> tuple[list[dict], dict]:
    """Materialize insert-ready ReplayBuffer rows from precomputed arrays."""
    directory, metadata = load_cache_metadata(resolved)
    image_keys = tuple(metadata["image_keys"])
    images = {key: np.load(directory / f"{key}.npy", mmap_mode="r") for key in image_keys}
    state = np.load(directory / "state.npy", mmap_mode="r")
    actions = np.load(directory / "actions.npy", mmap_mode="r")
    rewards = np.load(directory / "rewards.npy", mmap_mode="r")
    masks = np.load(directory / "masks.npy", mmap_mode="r")
    dones = np.load(directory / "dones.npy", mmap_mode="r")
    returns = np.load(directory / "mc_returns.npy", mmap_mode="r")
    episode_indices = np.load(directory / "episode_index.npy", mmap_mode="r")
    frame_indices = np.load(directory / "frame_index.npy", mmap_mode="r")
    context_ids = np.load(directory / "context_ids.npy", mmap_mode="r")
    next_context_ids = np.load(directory / "next_context_ids.npy", mmap_mode="r")

    frame_lookup = {
        (int(episode_indices[index]), int(frame_indices[index])): index
        for index in range(len(frame_indices))
    }
    action_horizon = int(metadata["action_horizon"])
    transitions: list[dict] = []
    for index in range(len(frame_indices)):
        episode = int(episode_indices[index])
        frame = int(frame_indices[index])
        next_index = frame_lookup.get((episode, frame + action_horizon), index)
        observation = {key: np.asarray(images[key][index]) for key in image_keys}
        observation["state"] = np.asarray(state[index])
        next_observation = {key: np.asarray(images[key][next_index]) for key in image_keys}
        next_observation["state"] = np.asarray(state[next_index])
        transitions.append(
            {
                "observations": observation,
                "actions": np.asarray(actions[index], dtype=np.float32),
                "next_observations": next_observation,
                "rewards": np.asarray(rewards[index], dtype=np.float32),
                "masks": float(masks[index]),
                "dones": bool(dones[index]),
                "mc_returns": np.float32(returns[index]),
                "context_id": np.int64(context_ids[index]),
                "next_context_id": np.int64(next_context_ids[index]),
                "context_source": np.int16(ContextSource.PREPROCESS),
                "next_context_source": np.int16(ContextSource.PREPROCESS),
                "episode_index": episode,
                "frame_index": frame,
            }
        )
    return transitions, metadata
