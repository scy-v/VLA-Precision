"""Robot-side ACoB-Stream actor loop.

The control, intervention, context-write, and episode-commit order is ported
from the paper implementation.  Construction is dependency-injected so the
same code runs against Pyro hardware and deterministic unit-test doubles.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from vla_precision.acob_stream.buffers.context import ContextSource
from vla_precision.acob_stream.communication import replace_actor_parameters
from vla_precision.acob_stream.interfaces import (
    ActorAgent,
    ActorParameterReplacer,
    ActorTrainerClient,
    ContextWriter,
    EnvironmentClient,
    TransitionShardWriter,
    TransitionStore,
)
from vla_precision.acob_stream.jax_runtime import JaxActorBackend
from vla_precision.acob_stream.metrics import Timer, actor_episode_payload, actor_timing_payload
from vla_precision.acob_stream.trajectories import (
    add_chunk_mc_returns,
    attach_next_context,
    commit_episode,
    decode_trajectory,
    scalar_bool,
)
from vla_precision.config.loader import ResolvedConfig
from vla_precision.integrations.openpi.adapter import b64_to_numpy
from vla_precision.logging import log_status

LOGGER = logging.getLogger(__name__)


@dataclass
class ActorRuntime:
    resolved: ResolvedConfig
    agent: ActorAgent
    environment: EnvironmentClient
    replay_store: TransitionStore
    correction_store: TransitionStore
    trainer_client: ActorTrainerClient
    sampling_rng: Any
    context_buffer: ContextWriter | None = None
    replicated_sharding: Any | None = None
    backend: Any = field(default_factory=JaxActorBackend)
    replace_parameters: ActorParameterReplacer = replace_actor_parameters
    transition_writer: TransitionShardWriter | None = None
    start_step: int = 0
    compile_warmup: bool = True


def _info_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    array = np.asarray(value).reshape(-1)
    return default if array.size == 0 else bool(array[0])


def _chunk_gripper_changed(policy_actions: np.ndarray, correction_actions: np.ndarray, *, dual_arm: bool) -> bool:
    policy = np.asarray(policy_actions)
    correction = np.asarray(correction_actions)
    indices = (6, 13) if dual_arm else (6,)
    if policy.ndim < 2 or correction.ndim < 2 or policy.shape != correction.shape:
        return False
    if policy.shape[-1] <= max(indices):
        return False
    policy_state = np.rint(policy[:, indices]).astype(np.int64)
    correction_state = np.rint(correction[:, indices]).astype(np.int64)
    return bool(np.any(policy_state != correction_state))


def _effective_intervention(
    policy_actions: np.ndarray,
    correction_actions: np.ndarray,
    metadata: dict,
    *,
    dual_arm: bool,
) -> tuple[bool, bool]:
    gripper_changed = _chunk_gripper_changed(policy_actions, correction_actions, dual_arm=dual_arm)
    if not metadata["present"]:
        return True, gripper_changed
    if metadata["pose_pressed"] or metadata["stop_pressed"]:
        return True, gripper_changed
    if metadata["gripper_pressed"]:
        return gripper_changed, gripper_changed
    return True, gripper_changed


def _delta_action_mask(value, *, dual_arm: bool) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value, dtype=np.float32).reshape(-1)
    expected = 12 if dual_arm else 6
    if mask.shape[0] < expected:
        return None
    return (mask[:expected] != 0.0).astype(np.float32)


def _apply_delta_action_mask(actions: np.ndarray, mask: np.ndarray | None, *, dual_arm: bool) -> np.ndarray:
    result = np.asarray(actions).copy()
    if mask is not None and dual_arm and result.shape[-1] >= 14:
        result[..., :6] *= mask[:6].astype(result.dtype, copy=False)
        result[..., 7:13] *= mask[6:12].astype(result.dtype, copy=False)
    elif mask is not None and result.shape[-1] >= 6:
        result[..., :6] *= mask[:6].astype(result.dtype, copy=False)
    return result


def _returned_action(value, *, reference: np.ndarray) -> np.ndarray | None:
    if value is None:
        return None
    action = np.asarray(value, dtype=reference.dtype)
    if action.ndim == 1:
        action = action[None, :]
    if action.shape != reference.shape:
        LOGGER.warning(
            "ignored environment action shape=%s; expected=%s",
            action.shape,
            reference.shape,
        )
        return None
    return action


def _actor_context(agent: ActorAgent, observation, backend) -> tuple[np.ndarray, np.ndarray, np.int32]:
    embeddings, masks, offsets = agent.encode_context(observation)
    embeddings, masks, offsets = backend.device_get((embeddings, masks, offsets))
    return (
        np.asarray(embeddings[0]),
        np.asarray(masks[0], dtype=bool),
        np.int32(np.asarray(offsets[0])),
    )


class ActorLoop:
    def __init__(self, runtime: ActorRuntime):
        self.runtime = runtime
        self.agent = runtime.agent
        self._agent_lock = threading.Lock()
        self._network_updates = 0

    def _install_parameter_receiver(self) -> None:
        def update(parameters) -> None:
            started = time.time()
            with self._agent_lock:
                current = self.agent
            updated = self.runtime.replace_parameters(current, parameters)
            if self.runtime.replicated_sharding is not None:
                updated = self.runtime.backend.replicate_agent(updated, self.runtime.replicated_sharding)
            with self._agent_lock:
                self.agent = updated
                self._network_updates += 1
            log_status(
                LOGGER,
                "Parameter sync",
                f"Actor parameters received: update {self._network_updates}, {time.time() - started:.2f} s",
                color="yellow",
            )

        self.runtime.trainer_client.recv_network_callback(update)

    def run(self):
        cfg = self.runtime.resolved.config
        dual_arm = cfg.task.arm_mode == "dual"
        self._install_parameter_receiver()

        observation, _ = self.runtime.environment.env_reset(options={"regrasp_before_reset": False})
        if self.runtime.compile_warmup:
            compile_started = time.time()
            log_status(LOGGER, "JIT compilation", "Actor inference started", color="magenta")
            with self._agent_lock:
                local_agent = self.agent
            self.runtime.sampling_rng = self.runtime.backend.compile_warmup(
                local_agent,
                b64_to_numpy(observation),
                self.runtime.sampling_rng,
            )
            log_status(
                LOGGER,
                "JIT compilation",
                f"Actor inference finished: {time.time() - compile_started:.2f} s",
                color="magenta",
            )

        trajectory: list[dict] = []
        pending_replay: list[dict] = []
        pending_corrections: list[dict] = []
        episode_index = 0
        episode_started = time.time()
        intervention_active = False
        intervention_count = 0
        intervention_steps = 0
        left_intervention_steps = 0
        right_intervention_steps = 0
        episode_action_steps = 0
        episode_return = 0.0
        timer = Timer()

        stop_step = cfg.stream.max_steps
        progress = tqdm(
            range(self.runtime.start_step, stop_step),
            desc="actor",
            dynamic_ncols=True,
        )
        for step in progress:
            timer.tick("total")
            with timer.context("sample_actions"):
                decoded_observation = b64_to_numpy(observation)
                with self._agent_lock:
                    local_agent = self.agent
                if step < cfg.stream.random_steps:
                    actions = b64_to_numpy(self.runtime.environment.env_action_space_sample())
                    context_embeddings, context_masks, context_offsets = _actor_context(
                        local_agent,
                        decoded_observation,
                        self.runtime.backend,
                    )
                else:
                    self.runtime.sampling_rng, key = self.runtime.backend.split_rng(self.runtime.sampling_rng)
                    actions, context_embeddings, context_masks, context_offsets = local_agent.sample_actions(
                        observations=decoded_observation,
                        seed=key,
                    )
                actions = np.array(self.runtime.backend.device_get(actions), copy=True)

            with timer.context("step_env"):
                policy_actions = actions if actions.ndim == 2 else actions[None, :]
                replay_actions = policy_actions.copy()
                intervention_bad_actions = policy_actions.copy()
                next_observation, reward, done, truncated, info = self.runtime.environment.env_step(
                    policy_actions.tolist()
                )
                reward = b64_to_numpy(reward)
                done = scalar_bool(b64_to_numpy(done))
                truncated = scalar_bool(b64_to_numpy(truncated))
                info = b64_to_numpy(info)

                delta_mask = _delta_action_mask(info.pop("delta_action_mask", None), dual_arm=dual_arm)
                executed_action = _returned_action(info.pop("executed_action", None), reference=policy_actions)
                if delta_mask is not None:
                    policy_actions = _apply_delta_action_mask(policy_actions, delta_mask, dual_arm=dual_arm)
                    replay_actions = policy_actions.copy()
                    intervention_bad_actions = policy_actions.copy()

                metadata_keys = (
                    "pose_intervention_pressed",
                    "gripper_intervention_pressed",
                    "stop_intervention_pressed",
                )
                intervention_metadata = {
                    "present": any(key in info for key in metadata_keys),
                    "pose_pressed": _info_bool(info.pop("pose_intervention_pressed", None)),
                    "gripper_pressed": _info_bool(info.pop("gripper_intervention_pressed", None)),
                    "stop_pressed": _info_bool(info.pop("stop_intervention_pressed", None)),
                    "env_gripper_changed": _info_bool(info.pop("gripper_action_changed_env", None)),
                }
                current_left_steps = int(info.pop("left_intervention_steps_executed", 0))
                current_right_steps = int(info.pop("right_intervention_steps_executed", 0))
                raw_intervened = "intervene_action" in info
                intervened = False
                gripper_changed = False
                if raw_intervened:
                    correction = np.asarray(info.pop("intervene_action"), dtype=policy_actions.dtype)
                    replay_actions = correction if correction.ndim == 2 else correction[None, :]
                    replay_actions = _apply_delta_action_mask(replay_actions, delta_mask, dual_arm=dual_arm)
                    if executed_action is not None:
                        replay_actions = executed_action.astype(replay_actions.dtype, copy=False)
                    intervened, gripper_changed = _effective_intervention(
                        policy_actions,
                        replay_actions,
                        intervention_metadata,
                        dual_arm=dual_arm,
                    )
                    current_intervention_steps = int(info.pop("intervention_steps_executed", 1))
                    intervention_steps += current_intervention_steps
                    left_intervention_steps += current_left_steps
                    right_intervention_steps += current_right_steps
                    if not intervention_active:
                        intervention_count += 1
                    intervention_active = True
                else:
                    if executed_action is not None:
                        replay_actions = executed_action.astype(replay_actions.dtype, copy=False)
                    intervention_active = False

                steps_executed = int(np.asarray(info.pop("steps_executed", replay_actions.shape[0])).reshape(-1)[0])
                episode_action_steps += steps_executed
                reward_chunk = np.asarray(reward, dtype=np.float32).reshape(-1)
                if reward_chunk.shape[0] != replay_actions.shape[0]:
                    raise ValueError(
                        f"environment returned reward shape {reward_chunk.shape} for action chunk {replay_actions.shape}"
                    )
                if intervention_bad_actions.shape != replay_actions.shape:
                    raise ValueError(
                        "policy and replay action chunks must match for intervention preference: "
                        f"{intervention_bad_actions.shape} != {replay_actions.shape}"
                    )
                episode_return += float(reward_chunk.sum())

                if self.runtime.context_buffer is not None:
                    context_id = np.int64(step)
                    self.runtime.context_buffer.put_batch(
                        [context_id],
                        np.asarray(context_embeddings)[None],
                        np.asarray(context_masks, dtype=bool)[None],
                        np.asarray([context_offsets], dtype=np.int32),
                    )
                    context_payload = {
                        "context_id": context_id,
                        "context_source": np.int16(ContextSource.TRAINING),
                    }
                else:
                    context_payload = {
                        "context_embeddings": context_embeddings,
                        "context_masks": context_masks,
                        "context_offsets": context_offsets,
                    }

                trajectory.append(
                    {
                        "observations": observation,
                        "actions": replay_actions,
                        "intervention_bad_actions": intervention_bad_actions,
                        "next_observations": next_observation,
                        "rewards": reward_chunk,
                        "masks": 1.0 - done,
                        "dones": done,
                        "intervened": intervened,
                        "raw_intervened": raw_intervened,
                        "intervention_metadata_present": intervention_metadata["present"],
                        "pose_intervention_pressed": intervention_metadata["pose_pressed"],
                        "gripper_intervention_pressed": intervention_metadata["gripper_pressed"],
                        "stop_intervention_pressed": intervention_metadata["stop_pressed"],
                        "gripper_action_changed": gripper_changed,
                        "env_gripper_action_changed": intervention_metadata["env_gripper_changed"],
                        **context_payload,
                    }
                )
                observation = next_observation

                if done or truncated:
                    episode_index += 1
                    duration = time.time() - episode_started
                    self.runtime.environment.env_request(name="force_pause", param=True)
                    trajectory = decode_trajectory(trajectory, b64_to_numpy)
                    add_chunk_mc_returns(
                        trajectory,
                        discount=cfg.acob.discount,
                        reward_scale=cfg.acob.reward_scale,
                        reward_bias=cfg.acob.reward_bias,
                        time_reward=cfg.task.time_reward,
                        # The paper implementation always applies sparse-return
                        # handling to online episodes. ``dense_reward`` only
                        # selects the offline demonstration preprocessing path.
                        sparse_reward=True,
                    )
                    attach_next_context(trajectory)
                    if self.runtime.context_buffer is not None:
                        self.runtime.context_buffer.flush()

                    succeeded = scalar_bool(info.get("succeed", False))
                    replay_copy, correction_copy = commit_episode(
                        trajectory,
                        replay_store=self.runtime.replay_store,
                        correction_store=self.runtime.correction_store,
                        succeeded=succeeded,
                        intervention_count=intervention_count,
                    )
                    pending_replay.extend(replay_copy)
                    pending_corrections.extend(correction_copy)

                    episode = info.setdefault("episode", {})
                    episode.update(
                        {
                            "intervention_count": intervention_count,
                            "intervention_steps": intervention_steps,
                            "left_intervention_steps": left_intervention_steps,
                            "right_intervention_steps": right_intervention_steps,
                            "trajectory_len": len(trajectory),
                            "action_steps": episode_action_steps,
                            "duration_seconds": float(duration),
                            "return_sum": float(episode_return),
                            "intervention_step_ratio": float(intervention_steps / max(episode_action_steps, 1)),
                            "intervention_count_per_transition": float(intervention_count / max(len(trajectory), 1)),
                            "succeed": int(succeeded),
                            "no_intervention_succeed": int(succeeded and intervention_count == 0),
                            "timeout": int(not succeeded),
                            "done": int(done),
                            "truncated": int(truncated),
                            "index": episode_index,
                            "total_steps": step,
                        }
                    )
                    self.runtime.trainer_client.request("send-stats", actor_episode_payload(info))
                    intervention_active = False
                    intervention_count = 0
                    intervention_steps = 0
                    left_intervention_steps = 0
                    right_intervention_steps = 0
                    episode_action_steps = 0
                    episode_return = 0.0
                    # This remains one explicit network transfer per completed episode.
                    self.runtime.trainer_client.update()

                    trajectory = []
                    observation, _ = self.runtime.environment.env_reset(options={"regrasp_before_reset": True})
                    episode_started = time.time()

            if (
                self.runtime.transition_writer is not None
                and cfg.stream.buffer_save_interval > 0
                and step > 0
                and step % cfg.stream.buffer_save_interval == 0
            ):
                self.runtime.transition_writer(step, pending_replay, pending_corrections)
                pending_replay = []
                pending_corrections = []

            timer.tock("total")
            timing = timer.get_latest_times()
            total_seconds = timing.get("total", 0.0)
            inference_seconds = timing.get("sample_actions", 0.0)
            progress.set_postfix(
                {
                    "loop": f"{1.0 / total_seconds:.2f} Hz" if total_seconds > 0 else "--",
                    "infer": f"{1.0 / inference_seconds:.2f} Hz" if inference_seconds > 0 else "--",
                    "env": f"{timing.get('step_env', 0.0):.2f} s",
                },
                refresh=True,
            )
            if cfg.stream.log_interval > 0 and step % cfg.stream.log_interval == 0:
                self.runtime.trainer_client.request("send-stats", actor_timing_payload(timer.get_average_times()))

        return self.agent


def run_actor(resolved: ResolvedConfig, *, runtime: ActorRuntime | None = None):
    """Run the actor selected by the unified top-level command.

    Runtime assembly is deliberately separate from the loop so robot, camera,
    and teleoperation implementations remain replaceable.
    """
    if runtime is None:
        from vla_precision.acob_stream.setup import build_actor_runtime

        runtime = build_actor_runtime(resolved)
    LOGGER.info("starting ACoB-Stream actor")
    return ActorLoop(runtime).run()
