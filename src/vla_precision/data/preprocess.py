"""Build train-ready ACoB demo transitions and their ContextBuffer.

This is a config-driven port of the paper implementation.  State and action
indices are materialized once on the Hugging Face dataset before any frame is
sampled; action chunking, returns, image decoding, and OpenPI prefix encoding
retain their original ordering.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

from vla_precision import image_tools
from vla_precision.acob_stream.buffers.context import ContextBuffer
from vla_precision.config.loader import ResolvedConfig
from vla_precision.config.schema import RootConfig
from vla_precision.data.indexing import materialize_lerobot_indices
from vla_precision.data.paths import preprocess_context_dir, preprocess_transition_dir, source_lerobot_root
from vla_precision.integrations.openpi.adapter import OpenPIObservationAdapter
from vla_precision.integrations.openpi.configs import (
    build_stage2_initialization_train_config,
)
from vla_precision.integrations.openpi.context import make_encode_context_fn

TRAIN_IMAGE_SIZE = 224
OPENPI_IMAGE_TOKENS = 256


class OpenPIInferenceState(NamedTuple):
    params: Any
    model_def: Any


def _import_lerobot_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    return LeRobotDataset, LeRobotDatasetMetadata


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def train_cache_dir(config: RootConfig) -> Path:
    return preprocess_transition_dir(config)


def _path_contains_files(path: Path) -> bool:
    return path.exists() and any(child.is_file() or child.is_symlink() for child in path.rglob("*"))


def prepare_cache_dir(path: Path, *, overwrite: bool) -> Path:
    path = path.expanduser()
    if _path_contains_files(path):
        if not overwrite:
            raise FileExistsError(f"Preprocess output already contains files: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_frame_lookup(episode_indices, frame_indices) -> dict[tuple[int, int], int]:
    return {
        (int(episode_indices[idx]), int(frame_indices[idx])): idx
        for idx in range(len(frame_indices))
    }


def _fixed_gripper(config: RootConfig) -> bool:
    return config.task.setup_mode == "single-arm-fixed-gripper"


def _state_dim(config: RootConfig) -> int:
    if config.data.state_indices:
        return len(config.data.state_indices)
    widths = {"tcp_pose": 6, "tcp_vel": 6, "tcp_force": 3, "tcp_torque": 3, "gripper_pose": 1}
    return sum(widths[key.split("/", 1)[-1]] for key in config.task.proprio_keys)


def _learned_state_dim(config: RootConfig) -> int | None:
    if config.task.arm_mode == "dual" or _fixed_gripper(config):
        return None
    return _state_dim(config)


def _dynamic_openpi_batch(adapter, observations: dict[str, np.ndarray]):
    from openpi.models import model as _model

    batch_size = int(next(iter(observations.values())).shape[0])
    obs_list = []
    for idx in range(batch_size):
        single = {key: np.asarray(value[idx]) for key, value in observations.items()}
        obs_list.append(adapter.observation(single))
    return _model.Observation(
        images={key: jnp.concatenate([obs.images[key] for obs in obs_list], axis=0) for key in obs_list[0].images},
        image_masks={key: jnp.concatenate([obs.image_masks[key] for obs in obs_list], axis=0) for key in obs_list[0].image_masks},
        state=jnp.concatenate([obs.state for obs in obs_list], axis=0),
        tokenized_prompt=jnp.concatenate([obs.tokenized_prompt for obs in obs_list], axis=0),
        tokenized_prompt_mask=jnp.concatenate([obs.tokenized_prompt_mask for obs in obs_list], axis=0),
    )


def _pad_observations_to_batch(observations: dict[str, np.ndarray], batch_size: int) -> dict[str, np.ndarray]:
    current_size = int(next(iter(observations.values())).shape[0])
    if current_size == batch_size:
        return observations
    if current_size <= 0 or current_size > batch_size:
        raise ValueError(f"Cannot pad observation batch of size {current_size} to {batch_size}.")
    pad_count = batch_size - current_size
    return {
        key: np.concatenate([value, np.repeat(value[-1:], pad_count, axis=0)], axis=0)
        for key, value in observations.items()
    }


def initialize_inference_state(train_config, rng) -> OpenPIInferenceState:
    """Initialize OpenPI params for preprocess-time inference without optimizer state."""
    from flax import nnx, traverse_util
    from openpi.shared import nnx_utils

    def init(init_rng, partial_params=None):
        init_rng, model_rng = jax.random.split(init_rng)
        model = train_config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            train_config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )
        return OpenPIInferenceState(params=params, model_def=nnx.graphdef(model))

    state_shape = jax.eval_shape(init, rng)
    loaded_params = train_config.weight_loader.load(state_shape.params.to_pure_dict())
    flat_loaded = traverse_util.flatten_dict(loaded_params)
    partial = traverse_util.unflatten_dict(
        {key: value for key, value in flat_loaded.items() if not isinstance(value, jax.ShapeDtypeStruct)}
    )
    return jax.jit(init)(rng, partial)


def _prefix_lengths_from_openpi_obs(pi_obs, *, actual_count: int) -> np.ndarray:
    lengths = np.zeros((actual_count,), dtype=np.int32)
    for mask in pi_obs.image_masks.values():
        image_mask = np.asarray(jax.device_get(mask))[:actual_count].astype(bool)
        lengths += image_mask.astype(np.int32) * OPENPI_IMAGE_TOKENS
    if pi_obs.tokenized_prompt_mask is not None:
        prompt_mask = np.asarray(jax.device_get(pi_obs.tokenized_prompt_mask))[:actual_count].astype(bool)
        lengths += prompt_mask.sum(axis=-1).astype(np.int32)
    return lengths


def _preprocess_train_config(config: RootConfig):
    return build_stage2_initialization_train_config(config)


def build_openpi_context_buffer(
    cache_dir: Path,
    image_keys: list[str],
    config: RootConfig,
    *,
    batch_size: int,
) -> dict[str, Any]:
    import openpi.models.gemma as _gemma
    import openpi.training.sharding as pi_sharding

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"data.preprocess.context_batch_size must be positive, got {batch_size}")
    device_count = jax.device_count()
    if batch_size % device_count != 0:
        raise ValueError(
            f"context_batch_size={batch_size} must be divisible by jax.device_count()={device_count} "
            "so the batch can be sharded data-parallel across visible devices."
        )

    train_config = _preprocess_train_config(config)
    mesh = pi_sharding.make_mesh(train_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(pi_sharding.DATA_AXIS))
    rng = jax.random.PRNGKey(config.experiment.seed)
    pi_state = initialize_inference_state(train_config, rng)
    adapter = OpenPIObservationAdapter(
        train_config,
        config.task.instruction,
        image_keys=image_keys,
        fixed_gripper=_fixed_gripper(config),
        dual_arm=config.task.arm_mode == "dual",
        learned_state_dim=_learned_state_dim(config),
        action_horizon=config.task.action_horizon,
        debug_fn=None,
    )
    max_prefix_tokens = len(adapter.model_image_keys) * 256 + int(train_config.model.max_token_len)
    paligemma_config = _gemma.get_config(train_config.model.paligemma_variant)
    embedding_shape = (2, paligemma_config.depth, max_prefix_tokens, paligemma_config.num_kv_heads, paligemma_config.head_dim)
    embedding_mask_shape = (max_prefix_tokens,)
    encode_fn = make_encode_context_fn(pi_state)

    images = {key: np.load(cache_dir / f"{key}.npy", mmap_mode="r") for key in image_keys}
    state = np.load(cache_dir / "state.npy", mmap_mode="r")
    episode_indices = np.load(cache_dir / "episode_index.npy", mmap_mode="r")
    frame_indices = np.load(cache_dir / "frame_index.npy", mmap_mode="r")
    num_frames = len(frame_indices)

    prefix_scan_start = time.time()
    prefix_lengths = np.empty((num_frames,), dtype=np.int32)
    progress = tqdm(total=num_frames, desc="Scanning OpenPI prefix lengths", unit="frame", dynamic_ncols=True)
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        actual_count = end - start
        observations = {key: np.asarray(images[key][start:end]) for key in image_keys}
        observations["state"] = np.asarray(state[start:end])
        observations = _pad_observations_to_batch(observations, batch_size)
        pi_obs = _dynamic_openpi_batch(adapter, observations)
        prefix_lengths[start:end] = _prefix_lengths_from_openpi_obs(pi_obs, actual_count=actual_count)
        progress.update(actual_count)
    progress.close()
    prefix_scan_elapsed = time.time() - prefix_scan_start
    stored_prefix_tokens = int(prefix_lengths.max())
    min_prefix_tokens = int(prefix_lengths.min())
    mean_prefix_tokens = float(prefix_lengths.mean())
    if stored_prefix_tokens <= 0 or stored_prefix_tokens > max_prefix_tokens:
        raise ValueError(f"Invalid compact OpenPI prefix length {stored_prefix_tokens}; max_prefix_tokens={max_prefix_tokens}")
    storage_embedding_shape = (
        2,
        paligemma_config.depth,
        stored_prefix_tokens,
        paligemma_config.num_kv_heads,
        paligemma_config.head_dim,
    )
    storage_embedding_mask_shape = (stored_prefix_tokens,)

    default_transition_dir = preprocess_transition_dir(config)
    context_dir = preprocess_context_dir(config) if cache_dir == default_transition_dir else cache_dir / "context"
    context_store = ContextBuffer.create(
        context_dir,
        capacity=num_frames,
        embedding_shape=embedding_shape,
        embedding_mask_shape=embedding_mask_shape,
        storage_embedding_shape=storage_embedding_shape,
        storage_embedding_mask_shape=storage_embedding_mask_shape,
        overwrite=True,
        shard_capacity=config.buffers.context_shard_size,
        metadata={
            "experiment": config.experiment.name,
            "openpi_profile": config.openpi.initialization_config_name,
            "max_prefix_tokens": max_prefix_tokens,
            "stored_prefix_tokens": stored_prefix_tokens,
            "prefix_length_min": min_prefix_tokens,
            "prefix_length_mean": mean_prefix_tokens,
        },
    )
    context_ids = open_memmap(cache_dir / "context_ids.npy", np.int64, (num_frames,))
    next_context_ids = open_memmap(cache_dir / "next_context_ids.npy", np.int64, (num_frames,))

    avg_record_mib = context_store.record_bytes / (1024 ** 2)
    total_record_bytes = context_store.record_bytes * num_frames
    total_record_gib = total_record_bytes / (1024 ** 3)
    print(
        "Building compact OpenPI context buffer: "
        f"frames={num_frames}, logical_tokens={max_prefix_tokens}, stored_tokens={stored_prefix_tokens}, "
        f"shape={embedding_shape}, storage_shape={storage_embedding_shape}, batch_size={batch_size}, "
        f"context_dir={context_dir}"
    )
    print(
        "OpenPI prefix length scan: "
        f"min={min_prefix_tokens}, max={stored_prefix_tokens}, mean={mean_prefix_tokens:.2f}, "
        f"elapsed={prefix_scan_elapsed:.2f}s"
    )
    print(
        "OpenPI context compact storage estimate: "
        f"count={num_frames}, avg_per_context={avg_record_mib:.2f} MiB, total={total_record_gib:.2f} GiB"
    )
    start_time = time.time()
    adapter_elapsed = 0.0
    encode_elapsed = 0.0
    write_elapsed = 0.0
    progress = tqdm(total=num_frames, desc="Caching compact OpenPI context", unit="frame", dynamic_ncols=True)
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        actual_count = end - start
        observations = {key: np.asarray(images[key][start:end]) for key in image_keys}
        observations["state"] = np.asarray(state[start:end])
        observations = _pad_observations_to_batch(observations, batch_size)
        adapter_start = time.time()
        pi_obs = _dynamic_openpi_batch(adapter, observations)
        pi_obs = jax.device_put(pi_obs, data_sharding)
        adapter_elapsed += time.time() - adapter_start

        encode_start = time.time()
        with pi_sharding.set_mesh(mesh):
            batch_embeddings, batch_masks, batch_offsets = encode_fn(pi_state.params, pi_obs)
        jax.block_until_ready((batch_embeddings, batch_masks, batch_offsets))
        encode_elapsed += time.time() - encode_start

        write_start = time.time()
        ids = np.arange(start, end, dtype=np.int64)
        context_store.put_batch(
            ids,
            jax.device_get(batch_embeddings)[:actual_count],
            jax.device_get(batch_masks)[:actual_count],
            jax.device_get(batch_offsets)[:actual_count],
        )
        context_ids[start:end] = ids
        write_elapsed += time.time() - write_start
        progress.update(actual_count)

    progress.close()
    flush_start = time.time()
    context_store.flush()
    context_ids.flush()
    flush_elapsed = time.time() - flush_start

    frame_lookup = _cache_frame_lookup(episode_indices, frame_indices)
    action_horizon = config.task.action_horizon
    for idx in range(num_frames):
        ep = int(episode_indices[idx])
        frame_idx = int(frame_indices[idx])
        next_context_ids[idx] = frame_lookup.get((ep, frame_idx + action_horizon), idx)
    next_context_ids.flush()

    total_elapsed = prefix_scan_elapsed + (time.time() - start_time)
    print(
        "OpenPI compact context buffer done in "
        f"{total_elapsed:.2f}s "
        f"(prefix_scan={prefix_scan_elapsed:.2f}s, adapter_batch={adapter_elapsed:.2f}s, "
        f"openpi_encode={encode_elapsed:.2f}s, write_context={write_elapsed:.2f}s, flush={flush_elapsed:.2f}s)"
    )
    print(
        "OpenPI context buffer summary: "
        f"count={num_frames}, logical_tokens={max_prefix_tokens}, stored_tokens={stored_prefix_tokens}, "
        f"avg_per_context={avg_record_mib:.2f} MiB, "
        f"payload_per_context={context_store.kv_record_bytes / (1024 ** 2):.2f} MiB, "
        f"mask_offset_per_context={(context_store.mask_record_bytes + context_store.offset_record_bytes) / 1024:.2f} KiB, "
        f"dir={context_dir}"
    )

    return {
        "context_source": "openpi_prefix",
        "context_dtype": "bfloat16_bits_uint16",
        "context_shape": list(embedding_shape),
        "context_mask_shape": list(embedding_mask_shape),
        "context_storage_shape": list(storage_embedding_shape),
        "context_storage_mask_shape": list(storage_embedding_mask_shape),
        "openpi_profile": config.openpi.initialization_config_name,
        "context_count": num_frames,
        "context_avg_bytes": context_store.record_bytes,
        "context_total_bytes": total_record_bytes,
        "context_max_tokens": max_prefix_tokens,
        "context_stored_tokens": stored_prefix_tokens,
        "context_prefix_length_min": min_prefix_tokens,
        "context_prefix_length_mean": mean_prefix_tokens,
        "context_compact_valid_tokens": True,
    }


def image_to_train_hwc_uint8(value) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    arr = image_tools.convert_to_uint8(arr)
    arr = image_tools.resize_with_pad(arr, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE)
    return image_tools.convert_to_uint8(arr)


def chunk_return(rewards, discount: float) -> float:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    powers = discount ** np.arange(rewards.shape[0], dtype=np.float32)
    return float(np.sum(rewards * powers))


def add_chunk_mc_returns_to_trajectory(
    trajectory: list[dict],
    discount: float,
    reward_scale: float,
    reward_bias: float,
    time_reward: float,
    *,
    is_sparse_reward: bool = True,
    return_stride: int = 1,
) -> list[dict]:
    if not trajectory:
        return trajectory

    chunk_returns = [chunk_return(t["rewards"], discount) * reward_scale + reward_bias for t in trajectory]
    reward_horizon = np.asarray(trajectory[0]["rewards"]).reshape(-1).shape[0]
    gamma_chunk = discount ** reward_horizon
    if is_sparse_reward:
        time_reward = chunk_return(np.full((reward_horizon,), time_reward, dtype=np.float32), discount) * reward_scale + reward_bias

    mc_returns = [0.0] * len(trajectory)
    return_stride = max(1, int(return_stride))
    for i in range(len(trajectory) - 1, -1, -1):
        next_i = i + return_stride
        next_return = mc_returns[next_i] if next_i < len(trajectory) else 0.0
        mc_returns[i] = chunk_returns[i] + gamma_chunk * next_return * (1 - trajectory[i]["dones"])

    if is_sparse_reward and np.allclose(np.asarray(chunk_returns, dtype=np.float32), time_reward):
        mc_returns = [float(time_reward / (1 - gamma_chunk))] * len(trajectory)

    for transition, mc_return in zip(trajectory, mc_returns):
        transition["mc_returns"] = np.float32(mc_return)
    return trajectory


def add_next_context_to_trajectory(trajectory: list[dict]) -> list[dict]:
    for i, transition in enumerate(trajectory):
        source = transition if i == len(trajectory) - 1 else trajectory[i + 1]
        if "context_id" in source:
            transition["next_context_id"] = source["context_id"]
            if "context_source" in source:
                transition["next_context_source"] = source["context_source"]
            elif "context_source" in transition:
                transition["next_context_source"] = transition["context_source"]
        if "context_embeddings" in source:
            transition["next_context_embeddings"] = source["context_embeddings"]
            transition["next_context_masks"] = source["context_masks"]
            transition["next_context_offsets"] = source["context_offsets"]
    return trajectory


def episode_indices(dataset) -> list[int]:
    dataset._ensure_hf_dataset_loaded()
    eps = set()
    for idx in range(len(dataset.hf_dataset)):
        row = dataset.hf_dataset[idx]
        eps.add(int(to_numpy(row["episode_index"]).reshape(-1)[0]))
    return sorted(eps)


def parse_num_episodes(value) -> int | None:
    text = str(value).strip().lower()
    if text in ("", "all", "-1"):
        return None
    num_episodes = int(text)
    if num_episodes <= 0:
        return None
    return num_episodes


def select_episode_indices(all_episodes: list[int], num_episodes, seed: int) -> list[int]:
    requested = parse_num_episodes(num_episodes)
    if requested is None or requested >= len(all_episodes):
        return list(all_episodes)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(all_episodes), size=requested, replace=False).astype(int).tolist())


def make_chunked_rows_from_frame_df(
    frame_df: pd.DataFrame,
    config: RootConfig,
    task_mapping: dict[int, int],
    new_episode_index: int,
    start_index: int,
    *,
    overlap: bool,
) -> list[dict]:
    action_horizon = config.task.action_horizon
    rows = []
    n = len(frame_df)
    actions = [np.asarray(action, dtype=np.float32) for action in frame_df[config.data.action_key].tolist()]
    rewards_per_step = [config.task.time_reward] * n

    start_indices = list(range(n) if overlap else range(0, n, action_horizon))
    for row_offset, i in enumerate(start_indices):
        frame = frame_df.iloc[i]
        end = min(i + action_horizon, n)
        chunk_actions = [action.copy() for action in actions[i:end]]
        chunk_rewards = list(rewards_per_step[i:end])
        while len(chunk_actions) < action_horizon:
            chunk_actions.append(chunk_actions[-1].copy())
            chunk_rewards.append(config.task.time_reward)

        done = float(row_offset == len(start_indices) - 1)
        if done:
            chunk_rewards[-1] = config.task.completion_reward

        state = np.asarray(frame[config.data.state_key], dtype=np.float32)

        if "index" not in frame.index:
            raise KeyError("Source LeRobot parquet frame is missing global index column required for image lookup")

        rows.append(
            {
                "source_index": int(frame["index"]),
                "source_episode_index": int(frame["episode_index"]),
                "action": np.stack(chunk_actions, axis=0).astype(np.float32),
                "state": state.astype(np.float32),
                "timestamp": np.float32(frame["timestamp"]),
                "frame_index": int(frame["frame_index"]),
                "episode_index": int(new_episode_index),
                "index": int(start_index + row_offset),
                "task_index": int(task_mapping[int(frame["task_index"])]),
                "rewards": np.asarray(chunk_rewards, dtype=np.float32),
                "masks": np.asarray([1.0 - done], dtype=np.float32),
                "dones": np.asarray([done], dtype=np.float32),
            }
        )

    add_chunk_mc_returns_to_trajectory(
        rows,
        config.acob.discount,
        config.acob.reward_scale,
        config.acob.reward_bias,
        config.task.time_reward,
        is_sparse_reward=not config.acob.dense_reward,
        return_stride=action_horizon if overlap else 1,
    )
    for row in rows:
        row["mc_returns"] = np.asarray([row["mc_returns"]], dtype=np.float32)
    return rows


def load_selected_episode_dataframes(dataset, selected: list[int]) -> dict[int, pd.DataFrame]:
    dataset._ensure_hf_dataset_loaded()
    if not selected:
        return {}

    # Match OpenPI's normal LeRobot read path: select rows from the already loaded
    # Hugging Face dataset instead of resolving parquet files through per-episode
    # data/file_index metadata. Video decoding remains a separate parallel stage.
    hf_dataset = dataset.hf_dataset.with_format(None)
    episode_indices = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    selected_set = {int(ep) for ep in selected}
    row_indices = np.flatnonzero(np.isin(episode_indices, list(selected_set)))
    selected_df = hf_dataset.select(row_indices.tolist()).to_pandas()

    episode_frames = {
        int(ep): ep_df.copy()
        for ep, ep_df in selected_df.groupby("episode_index", sort=False)
        if int(ep) in selected_set
    }
    result: dict[int, pd.DataFrame] = {}
    for ep in selected:
        if ep not in episode_frames:
            raise ValueError(f"Selected episode {ep} has no frames")
        ep_df = episode_frames[ep]
        ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
        result[ep] = ep_df
    return result


def load_source_obs_images(dataset, source_index: int, image_key_map: dict[str, str]) -> dict[str, np.ndarray]:
    frame = dataset[int(source_index)]
    return {dst_key: image_to_train_hwc_uint8(frame[src_key])[None] for dst_key, src_key in image_key_map.items()}


_EPISODE_WORKER_DS = None
_EPISODE_WORKER_IMAGE_KEY_MAP = None
_EPISODE_WORKER_CACHE_DIR = None


def init_episode_image_worker(repo_id: str, root: str | None, image_key_map: dict[str, str], cache_dir: str) -> None:
    global _EPISODE_WORKER_DS, _EPISODE_WORKER_IMAGE_KEY_MAP, _EPISODE_WORKER_CACHE_DIR
    LeRobotDataset, _ = _import_lerobot_dataset()
    _EPISODE_WORKER_DS = LeRobotDataset(repo_id, root=None if root is None else Path(root))
    _EPISODE_WORKER_DS._ensure_hf_dataset_loaded()
    _EPISODE_WORKER_IMAGE_KEY_MAP = image_key_map
    _EPISODE_WORKER_CACHE_DIR = Path(cache_dir)


def _decode_episode_camera_frames(source_episode: int, src_key: str, timestamps: list[float]) -> np.ndarray:
    from lerobot.datasets.video_utils import decode_video_frames

    if _EPISODE_WORKER_DS is None:
        raise RuntimeError("LeRobot episode image worker was not initialized")
    ep = _EPISODE_WORKER_DS.meta.episodes[int(source_episode)]
    from_timestamp = float(ep[f"videos/{src_key}/from_timestamp"])
    shifted_timestamps = [from_timestamp + float(ts) for ts in timestamps]
    video_path = _EPISODE_WORKER_DS.root / _EPISODE_WORKER_DS.meta.get_video_file_path(int(source_episode), src_key)
    frames = decode_video_frames(
        video_path,
        shifted_timestamps,
        _EPISODE_WORKER_DS.tolerance_s,
        backend=_EPISODE_WORKER_DS.video_backend,
    )
    arr = to_numpy(frames)
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
        arr = np.moveaxis(arr, 1, -1)
    arr = image_tools.convert_to_uint8(arr)
    arr = image_tools.resize_with_pad(arr, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE)
    return image_tools.convert_to_uint8(arr)


def write_episode_images_process(task: dict) -> tuple[int, int]:
    if _EPISODE_WORKER_CACHE_DIR is None or _EPISODE_WORKER_IMAGE_KEY_MAP is None:
        raise RuntimeError("Episode image worker was not initialized")
    row_indices = np.asarray(task["row_indices"], dtype=np.int64)
    timestamps = [float(ts) for ts in task["timestamps"]]
    source_episode = int(task["source_episode_index"])
    for dst_key, src_key in _EPISODE_WORKER_IMAGE_KEY_MAP.items():
        images = _decode_episode_camera_frames(source_episode, src_key, timestamps)
        mmap = np.load(_EPISODE_WORKER_CACHE_DIR / f"{dst_key}.npy", mmap_mode="r+")
        mmap[row_indices] = images[:, None]
        mmap.flush()
    return source_episode, len(row_indices)


def episode_image_tasks(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        grouped[int(row["source_episode_index"])].append((row_idx, row))
    tasks = []
    for source_episode, indexed_rows in sorted(grouped.items()):
        indexed_rows.sort(key=lambda item: int(item[1]["frame_index"]))
        tasks.append(
            {
                "source_episode_index": int(source_episode),
                "row_indices": [int(row_idx) for row_idx, _row in indexed_rows],
                "timestamps": [float(_row["timestamp"]) for _row_idx, _row in indexed_rows],
            }
        )
    return tasks


def open_memmap(path: Path, dtype, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def flush_cache_arrays(arrays: dict) -> None:
    for arr in arrays["images"].values():
        arr.flush()
    for key in ["state", "actions", "rewards", "masks", "dones", "mc_returns", "episode_index", "frame_index"]:
        arrays[key].flush()



def build_chunked_cache_rows(src, selected: list[int], config: RootConfig) -> list[dict]:
    episode_dfs = load_selected_episode_dataframes(src, selected)
    rows: list[dict] = []
    global_index = 0
    for new_ep, old_ep in enumerate(selected):
        ep_df = episode_dfs[old_ep]
        task_mapping = {int(task_idx): int(task_idx) for task_idx in ep_df["task_index"].unique().tolist()}
        ep_rows = make_chunked_rows_from_frame_df(
            ep_df,
            config,
            task_mapping,
            new_ep,
            global_index,
            overlap=config.data.preprocess.chunk_mode == "overlap",
        )
        rows.extend(ep_rows)
        global_index += len(ep_rows)
    return rows


def preprocess_train_cache_from_lerobot(
    resolved: ResolvedConfig,
    cache_dir: Path | None = None,
) -> Path:
    config = resolved.config
    cache_dir = prepare_cache_dir(
        cache_dir or train_cache_dir(config),
        overwrite=config.data.preprocess.overwrite,
    )
    LeRobotDataset, LeRobotDatasetMetadata = _import_lerobot_dataset()
    image_key_map = dict(config.data.image_key_map)
    src_root = source_lerobot_root(config)
    src_meta = LeRobotDatasetMetadata(config.data.lerobot_repo_id, root=src_root)
    src = LeRobotDataset(config.data.lerobot_repo_id, root=src_root)
    data_metadata = materialize_lerobot_indices(
        src,
        config.data,
        metadata_path=cache_dir / "index_materialization.json",
        config_sha256=resolved.preprocess_sha256,
    )

    all_episodes = episode_indices(src)
    selected = select_episode_indices(
        all_episodes,
        config.data.preprocess.num_episodes,
        config.experiment.seed,
    )
    if not selected:
        raise ValueError("No episodes selected for preprocessing")

    transition_start = time.time()
    rows = build_chunked_cache_rows(src, selected, config)
    if not rows:
        raise ValueError("Selected episodes produced no chunked transitions")
    state_dim = int(np.asarray(rows[0]["state"]).shape[-1])
    action_dim = int(np.asarray(rows[0]["action"]).shape[-1])
    if config.task.arm_mode == "dual":
        expected_images = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
        if set(image_key_map) != expected_images:
            raise ValueError(
                "Dual-arm preprocessing requires base/left-wrist/right-wrist image mappings; "
                f"got {sorted(image_key_map)}"
            )
        if state_dim != 38:
            raise ValueError(
                f"Dual-arm preprocessing requires the ordered 38-D state, got {state_dim}. "
                "Configure data.state_indices in left-then-right order."
            )
        if action_dim != 14:
            raise ValueError(
                f"Dual-arm preprocessing requires 14-D actions "
                f"([left 6D+gripper, right 6D+gripper]), got {action_dim}."
            )
    else:
        fixed_gripper = _fixed_gripper(config)
        expected_state_dim = _state_dim(config)
        expected_action_dim = 6 if fixed_gripper else 7
        if state_dim != expected_state_dim or action_dim != expected_action_dim:
            raise ValueError(
                f"Single-arm fixed_gripper={fixed_gripper} requires {expected_state_dim}-D state "
                f"and {expected_action_dim}-D action data, got state={state_dim}, action={action_dim}. "
                "Fixed-gripper datasets must omit gripper state/action fields; learned-gripper "
                "datasets must include them."
            )

    print(f"Building train-ready VLA-Precision cache: {cache_dir}")
    print(
        f"Dataset={config.data.lerobot_repo_id}, root={src_root}, episodes={len(selected)}, "
        f"transitions={len(rows)}, workers={config.data.preprocess.workers}"
    )

    first_images = load_source_obs_images(src, int(rows[0]["source_index"]), image_key_map)
    image_keys = list(image_key_map.keys())
    arrays = {
        "image_keys": image_keys,
        "images": {
            key: open_memmap(cache_dir / f"{key}.npy", first_images[key].dtype, (len(rows), *first_images[key].shape))
            for key in image_keys
        },
        "state": open_memmap(cache_dir / "state.npy", np.float32, (len(rows), 1, state_dim)),
        "actions": open_memmap(cache_dir / "actions.npy", np.float32, (len(rows), *np.asarray(rows[0]["action"]).shape)),
        "rewards": open_memmap(cache_dir / "rewards.npy", np.float32, (len(rows), *np.asarray(rows[0]["rewards"]).shape)),
        "masks": open_memmap(cache_dir / "masks.npy", np.float32, (len(rows),)),
        "dones": open_memmap(cache_dir / "dones.npy", np.bool_, (len(rows),)),
        "mc_returns": open_memmap(cache_dir / "mc_returns.npy", np.float32, (len(rows),)),
        "episode_index": open_memmap(cache_dir / "episode_index.npy", np.int64, (len(rows),)),
        "frame_index": open_memmap(cache_dir / "frame_index.npy", np.int64, (len(rows),)),
    }
    for row_idx, row in enumerate(rows):
        arrays["state"][row_idx] = np.asarray(row["state"], dtype=np.float32)[None]
        arrays["actions"][row_idx] = np.asarray(row["action"], dtype=np.float32)
        arrays["rewards"][row_idx] = np.asarray(row["rewards"], dtype=np.float32)
        arrays["masks"][row_idx] = float(np.asarray(row["masks"]).reshape(-1)[0])
        arrays["dones"][row_idx] = bool(np.asarray(row["dones"]).reshape(-1)[0] > 0.5)
        arrays["mc_returns"][row_idx] = float(np.asarray(row["mc_returns"]).reshape(-1)[0])
        arrays["episode_index"][row_idx] = int(row["episode_index"])
        arrays["frame_index"][row_idx] = int(row["frame_index"])
    print(f"Cached non-image train fields: {len(rows)}/{len(rows)}")

    tasks = episode_image_tasks(rows)
    workers = max(1, config.data.preprocess.workers)
    print(f"Caching train images by episode: episodes={len(tasks)}, workers={workers}")
    completed = 0
    if workers == 1 or len(tasks) <= 1:
        init_episode_image_worker(
            config.data.lerobot_repo_id,
            None if src_root is None else str(src_root),
            image_key_map,
            str(cache_dir),
        )
        for task in tasks:
            try:
                source_episode, count = write_episode_images_process(task)
            except Exception as exc:
                raise RuntimeError(f"Failed to cache images for source episode={task['source_episode_index']}") from exc
            completed += count
            print(f"Cached train image frames: {completed}/{len(rows)} (episode {source_episode})")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=ctx,
            initializer=init_episode_image_worker,
            initargs=(
                config.data.lerobot_repo_id,
                None if src_root is None else str(src_root),
                image_key_map,
                str(cache_dir),
            ),
        ) as executor:
            future_to_episode = {executor.submit(write_episode_images_process, task): int(task["source_episode_index"]) for task in tasks}
            for future in as_completed(future_to_episode):
                source_episode = future_to_episode[future]
                try:
                    _source_episode, count = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to cache images for source episode={source_episode}") from exc
                completed += count
                print(f"Cached train image frames: {completed}/{len(rows)} (episode {_source_episode})")

    flush_cache_arrays(arrays)
    transition_elapsed = time.time() - transition_start
    print(
        "VLA-Precision transition cache generation done in "
        f"{transition_elapsed:.2f}s (transitions={len(rows)}, dir={cache_dir})"
    )

    context_start = time.time()
    context_metadata = build_openpi_context_buffer(
        cache_dir,
        image_keys,
        config,
        batch_size=config.data.preprocess.context_batch_size,
    )
    context_elapsed = time.time() - context_start
    avg_context_mib = float(context_metadata["context_avg_bytes"]) / (1024 ** 2)
    total_context_gib = float(context_metadata["context_total_bytes"]) / (1024 ** 3)
    print(
        "VLA-Precision context buffer generation done in "
        f"{context_elapsed:.2f}s (transitions={len(rows)})"
    )
    print(
        "VLA-Precision preprocess timing summary: "
        f"transition_generation={transition_elapsed:.2f}s, context_generation={context_elapsed:.2f}s, "
        f"total={transition_elapsed + context_elapsed:.2f}s"
    )
    print(
        "VLA-Precision context buffer size summary: "
        f"total={total_context_gib:.2f} GiB, avg_per_transition={avg_context_mib:.2f} MiB, "
        f"transitions={len(rows)}"
    )
    metadata = {
        "format": "vla_precision_train_cache",
        "format_version": 1,
        "experiment": config.experiment.name,
        "preprocess_sha256": resolved.preprocess_sha256,
        "dual_arm": config.task.arm_mode == "dual",
        "source_repo_id": config.data.lerobot_repo_id,
        "source_root": None if src_root is None else str(src_root),
        "source_fps": getattr(src_meta, "fps", None),
        "selected_episodes": [int(ep) for ep in selected],
        "num_frames": len(rows),
        "action_horizon": config.task.action_horizon,
        "preprocess_chunk_mode": config.data.preprocess.chunk_mode,
        "image_size": TRAIN_IMAGE_SIZE,
        "image_keys": image_keys,
        "state_key": config.data.state_key,
        "state_indices": list(config.data.state_indices),
        "action_key": config.data.action_key,
        "action_indices": list(config.data.action_indices),
        "data_schema_sha256": data_metadata.schema_sha256,
        "image_key_map": image_key_map,
        "time_reward": config.task.time_reward,
        "completion_reward": config.task.completion_reward,
        "openpi_model": config.openpi.model,
        "initialization_checkpoint_step": config.openpi.initialization_checkpoint_step,
        "discount": config.acob.discount,
        "reward_scale": config.acob.reward_scale,
        "reward_bias": config.acob.reward_bias,
        "dense_reward": config.acob.dense_reward,
        "transition_generation_seconds": transition_elapsed,
        "context_generation_seconds": context_elapsed,
        "preprocess_total_seconds": transition_elapsed + context_elapsed,
        **context_metadata,
    }
    tmp_meta = cache_dir / "metadata.json.tmp"
    tmp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_meta.replace(cache_dir / "metadata.json")
    return cache_dir


def run_preprocess(resolved: ResolvedConfig) -> Path:
    start = time.time()
    output_root = preprocess_train_cache_from_lerobot(resolved)
    elapsed = time.time() - start
    print(f"Preprocess finished in {elapsed:.2f}s. Train-ready VLA-Precision cache: {output_root}")
    return output_root
