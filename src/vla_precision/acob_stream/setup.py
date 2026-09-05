"""Assemble real actor/learner runtimes from the single RootConfig."""

from __future__ import annotations

import dataclasses
import logging
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np

from vla_precision.acob.factory import create_agent
from vla_precision.acob_stream.actor import ActorRuntime
from vla_precision.acob_stream.buffers.context import ContextBuffer, ContextSource
from vla_precision.acob_stream.buffers.correction import CorrectionBuffer
from vla_precision.acob_stream.buffers.replay import ReplayBuffer
from vla_precision.acob_stream.checkpoints import (
    DataConfigProvider,
    checkpoint_layout,
    initialize_fresh_lora,
    prepare_agent,
    prepare_fresh_run_layout,
    save_agent,
)
from vla_precision.acob_stream.communication import (
    STATS_REQUEST,
    PyroEnvironmentClient,
    create_actor_trainer_client,
    create_learner_trainer_server,
)
from vla_precision.acob_stream.initial_buffers import load_initial_buffers, save_initial_buffers
from vla_precision.acob_stream.jax_runtime import configure_parallel_sharding
from vla_precision.acob_stream.learner import LearnerRuntime
from vla_precision.acob_stream.metrics import TrainingMetricsTracker, learner_payload
from vla_precision.acob_stream.preprocessed_data import load_preprocessed_transitions
from vla_precision.acob_stream.trajectories import TransitionShardWriter, load_transition_shards
from vla_precision.config.loader import ResolvedConfig
from vla_precision.data.paths import (
    preprocess_context_dir,
    preprocess_transition_dir,
    training_context_dir,
)
from vla_precision.integrations.openpi.adapter import b64_to_numpy
from vla_precision.integrations.openpi.configs import build_stage2_train_config

LOGGER = logging.getLogger(__name__)


def _online_context_dir(resolved: ResolvedConfig) -> Path:
    return training_context_dir(resolved.config)


def _online_context_store(agent, resolved: ResolvedConfig) -> ContextBuffer:
    cfg = resolved.config
    logical_shape = tuple(agent.algo_config["context_embedding_shape"])
    logical_mask_shape = tuple(agent.algo_config["context_mask_shape"])
    image_tokens = len(agent.adapter.source_image_keys) * 256
    stored_tokens = min(logical_shape[2], image_tokens + int(agent.pi_train_config.model.max_token_len))
    storage_shape = (*logical_shape[:2], stored_tokens, *logical_shape[3:])
    capacity = max(
        cfg.buffers.context_capacity,
        cfg.buffers.replay_capacity,
        cfg.stream.max_steps,
        1,
    )
    store = ContextBuffer.open_or_create(
        _online_context_dir(resolved),
        capacity=capacity,
        embedding_shape=logical_shape,
        embedding_mask_shape=logical_mask_shape,
        storage_embedding_shape=storage_shape,
        storage_embedding_mask_shape=(stored_tokens,),
        shard_capacity=cfg.buffers.context_shard_size,
        overwrite=False,
        metadata={
            "stage": "online_training",
            "experiment": cfg.experiment.name,
            "openpi_profile": cfg.openpi.name,
            "context_source": int(ContextSource.TRAINING),
        },
        mode="r+",
    )
    if not cfg.checkpoint.resume and store.has_written_records():
        raise FileExistsError(
            f"Fresh Stage-II training cannot reuse an existing training Context Buffer: {store.root}. "
            "Set checkpoint.resume=true to restore the run, or remove this training/context directory."
        )
    return store


def _device_put_agent(agent, sharding):
    import jax
    import jax.numpy as jnp

    return jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding)


def _observation_action_spaces(cache_directory: Path, metadata: dict):
    state = np.load(cache_directory / "state.npy", mmap_mode="r")
    actions = np.load(cache_directory / "actions.npy", mmap_mode="r")
    spaces: dict[str, gym.Space] = {
        "state": gym.spaces.Box(-np.inf, np.inf, shape=state.shape[1:], dtype=np.float32)
    }
    for key in metadata["image_keys"]:
        image = np.load(cache_directory / f"{key}.npy", mmap_mode="r")
        spaces[key] = gym.spaces.Box(0, 255, shape=image.shape[1:], dtype=image.dtype)
    observation_space = gym.spaces.Dict(spaces)
    action_space = gym.spaces.Box(-3.0, 3.0, shape=actions.shape[1:], dtype=np.float32)
    return observation_space, action_space


def _target_context_storage_shape(agent, metadata: dict):
    logical_shape = tuple(agent.algo_config["context_embedding_shape"])
    online_tokens = min(
        logical_shape[2],
        len(agent.adapter.source_image_keys) * 256 + int(agent.pi_train_config.model.max_token_len),
    )
    preprocess_shape = tuple(metadata.get("context_storage_shape", logical_shape))
    target_tokens = min(logical_shape[2], max(online_tokens, int(preprocess_shape[2])))
    return (*logical_shape[:2], target_tokens, *logical_shape[3:]), (target_tokens,)


class _MetricLogger:
    def __init__(self, resolved: ResolvedConfig, replay_buffer, correction_buffer):
        self._run = None
        self._lock = threading.Lock()
        self._step = -1
        self._replay_buffer = replay_buffer
        self._correction_buffer = correction_buffer
        self._tracker = TrainingMetricsTracker(
            rolling_window=resolved.config.logging.training_metrics_window
        )
        if resolved.config.logging.wandb.enabled:
            import wandb

            self._run = wandb.init(
                project=resolved.config.logging.wandb.project,
                entity=resolved.config.logging.wandb.entity,
                name=resolved.config.experiment.name,
                config=dataclasses.asdict(resolved.config),
            )

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def _next_step(self, preferred: int | None = None) -> int:
        with self._lock:
            candidate = self._step + 1 if preferred is None else int(preferred)
            self._step = max(self._step + 1, candidate)
            return self._step

    def actor_request(self, request_type: str, payload: dict) -> dict:
        if request_type != STATS_REQUEST:
            raise ValueError(f"Unsupported trainer request: {request_type}")
        if self._run is not None:
            self._tracker.add_actor_payload(payload)
            self._run.log(payload, step=self._next_step())
        return {}

    def start(self) -> None:
        self._tracker.start()

    def learner_update(self, step: int, info, timings: dict[str, float]) -> None:
        if self._run is None:
            return
        payload = learner_payload(
            info,
            timings,
            self._replay_buffer,
            self._correction_buffer,
        )
        self._tracker.add_learner_payload(payload, learner_step=step)
        self._run.log(payload, step=self._next_step(step))


def build_actor_runtime(resolved: ResolvedConfig) -> ActorRuntime:
    cfg = resolved.config
    train_config = build_stage2_train_config(cfg)
    mesh = configure_parallel_sharding(train_config)
    environment = PyroEnvironmentClient.connect(resolved)
    sample_observation = b64_to_numpy(environment.env_observation_space_sample())
    sample_action = b64_to_numpy(environment.env_action_space_sample())
    agent = create_agent(
        cfg,
        sample_observation=sample_observation,
        sample_action=sample_action,
        train_config=train_config,
    )
    layout = checkpoint_layout(cfg)
    if cfg.checkpoint.resume:
        agent, _manager, _step = prepare_agent(agent, cfg, layout)
    else:
        prepare_fresh_run_layout(cfg, layout)
        agent = initialize_fresh_lora(agent)
    agent = _device_put_agent(agent, mesh.replicated_sharding)

    from agentlace.data.data_store import QueuedDataStore

    replay_queue = QueuedDataStore(50_000)
    correction_queue = QueuedDataStore(50_000)
    trainer_client = create_actor_trainer_client(resolved, replay_queue, correction_queue)
    writer = TransitionShardWriter(layout.replay_transitions, layout.correction_transitions)
    import jax

    return ActorRuntime(
        resolved=resolved,
        agent=agent,
        environment=environment,
        replay_store=replay_queue,
        correction_store=correction_queue,
        trainer_client=trainer_client,
        sampling_rng=jax.device_put(jax.random.PRNGKey(cfg.experiment.seed), mesh.replicated_sharding),
        context_buffer=_online_context_store(agent, resolved),
        replicated_sharding=mesh.replicated_sharding,
        transition_writer=writer,
        start_step=writer.next_step if cfg.checkpoint.resume else 0,
    )


def build_learner_runtime(resolved: ResolvedConfig) -> LearnerRuntime:
    cfg = resolved.config
    transitions, metadata = load_preprocessed_transitions(resolved)
    cache_directory = preprocess_transition_dir(cfg)
    observation_space, action_space = _observation_action_spaces(cache_directory, metadata)
    train_config = build_stage2_train_config(cfg)
    mesh = configure_parallel_sharding(train_config)
    agent = create_agent(
        cfg,
        sample_observation=observation_space.sample(),
        sample_action=action_space.sample(),
        train_config=train_config,
    )
    layout = checkpoint_layout(cfg)
    agent, manager, resumed_step = prepare_agent(agent, cfg, layout)
    agent = _device_put_agent(agent, mesh.replicated_sharding)

    online_context = _online_context_store(agent, resolved)
    context_sources = {
        int(ContextSource.PREPROCESS): preprocess_context_dir(cfg),
        int(ContextSource.TRAINING): online_context,
    }
    storage_shape, storage_mask_shape = _target_context_storage_shape(agent, metadata)
    buffer_kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
        "image_keys": cfg.task.image_keys,
        "context_shape": tuple(agent.algo_config["context_embedding_shape"]),
        "context_mask_shape": tuple(agent.algo_config["context_mask_shape"]),
        "context_dtype": agent.algo_config["context_dtype"],
        "include_context": True,
        "context_buffers": context_sources,
        "default_context_source": int(ContextSource.TRAINING),
        "storage_context_shape": storage_shape,
        "storage_context_mask_shape": storage_mask_shape,
        "include_mc_returns": True,
    }
    replay_buffer = ReplayBuffer(capacity=cfg.buffers.replay_capacity, **buffer_kwargs)
    correction_buffer = CorrectionBuffer(capacity=cfg.buffers.correction_capacity, **buffer_kwargs)
    for transition in transitions:
        replay_buffer.insert(transition)
    if cfg.checkpoint.resume:
        for transition in load_transition_shards(layout.replay_transitions):
            replay_buffer.insert(transition)
        for transition in load_transition_shards(layout.correction_transitions):
            correction_buffer.insert(transition)
    LOGGER.info(
        "loaded buffers: replay=%d correction=%d preprocess=%d",
        len(replay_buffer),
        correction_buffer.sampleable_size(),
        len(transitions),
    )

    metrics = _MetricLogger(resolved, replay_buffer, correction_buffer)
    server = create_learner_trainer_server(resolved, metrics.actor_request)
    provider = DataConfigProvider(train_config)
    return LearnerRuntime(
        resolved=resolved,
        agent=agent,
        replay_buffer=replay_buffer,
        correction_buffer=correction_buffer,
        trainer_server=server,
        mesh=mesh,
        checkpoint_saver=lambda current, step: save_agent(current, manager, provider, layout, step),
        checkpoint_waiter=manager.wait_until_finished,
        metrics_sink=metrics.learner_update if metrics.enabled else None,
        metrics_starter=metrics.start if metrics.enabled else None,
        initial_buffer_loader=lambda replay, correction, minimum: load_initial_buffers(
            cfg,
            replay,
            correction,
            minimum,
        ),
        initial_buffer_saver=lambda replay, correction, minimum: save_initial_buffers(
            cfg,
            replay,
            correction,
            minimum,
            online_context,
        ),
        resumed_step=resumed_step,
    )
