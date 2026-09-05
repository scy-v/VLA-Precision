"""Episode finalization shared by online collection and offline preprocessing."""

from __future__ import annotations

import os
import pickle
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from vla_precision.acob_stream.interfaces import TransitionStore


def chunk_return(rewards, discount: float) -> float:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    powers = discount ** np.arange(rewards.shape[0], dtype=np.float32)
    return float(np.sum(rewards * powers))


def add_chunk_mc_returns(
    trajectory: list[dict],
    *,
    discount: float,
    reward_scale: float,
    reward_bias: float,
    time_reward: float,
    sparse_reward: bool,
    return_stride: int = 1,
) -> list[dict]:
    """Preserve the paper implementation's chunk-level Monte-Carlo returns."""
    if not trajectory:
        return trajectory

    chunk_returns = [chunk_return(row["rewards"], discount) * reward_scale + reward_bias for row in trajectory]
    reward_horizon = np.asarray(trajectory[0]["rewards"]).reshape(-1).shape[0]
    gamma_chunk = discount**reward_horizon
    if sparse_reward:
        time_reward = (
            chunk_return(np.full((reward_horizon,), time_reward, dtype=np.float32), discount) * reward_scale
            + reward_bias
        )

    mc_returns = [0.0] * len(trajectory)
    return_stride = max(1, int(return_stride))
    for index in range(len(trajectory) - 1, -1, -1):
        next_index = index + return_stride
        next_return = mc_returns[next_index] if next_index < len(trajectory) else 0.0
        mc_returns[index] = chunk_returns[index] + gamma_chunk * next_return * (1 - trajectory[index]["dones"])

    if sparse_reward and np.allclose(np.asarray(chunk_returns, dtype=np.float32), time_reward):
        mc_returns = [float(time_reward / (1 - gamma_chunk))] * len(trajectory)

    for transition, mc_return in zip(trajectory, mc_returns, strict=True):
        transition["mc_returns"] = np.float32(mc_return)
    return trajectory


def attach_next_context(trajectory: list[dict]) -> list[dict]:
    """Point each row at the next row's context; terminal rows reuse their own.

    Reusing the last current context is intentional and matches the original
    implementation.  The terminal mask prevents a bootstrap through it.
    """
    for index, transition in enumerate(trajectory):
        source = transition if index == len(trajectory) - 1 else trajectory[index + 1]
        if "context_id" in source:
            transition["next_context_id"] = source["context_id"]
            transition["next_context_source"] = source.get(
                "context_source",
                transition.get("context_source"),
            )
        if "context_embeddings" in source:
            transition["next_context_embeddings"] = source["context_embeddings"]
            transition["next_context_masks"] = source["context_masks"]
            transition["next_context_offsets"] = source["context_offsets"]
    return trajectory


def commit_episode(
    trajectory: list[dict],
    *,
    replay_store: TransitionStore,
    correction_store: TransitionStore,
    succeeded: bool,
    intervention_count: int,
) -> tuple[list[dict], list[dict]]:
    """Commit a completed episode to the two online buffers.

    Every transition belongs to ReplayBuffer.  Only effective human
    corrections belong to CorrectionBuffer.  Copies returned by this function
    are suitable for optional checkpoint shards and cannot alias replay memory.
    """
    replay_copy: list[dict] = []
    correction_copy: list[dict] = []
    no_intervention_succeeded = bool(succeeded and intervention_count == 0)
    for transition in trajectory:
        transition["episode_succeed"] = bool(succeeded)
        transition["episode_no_intervention_succeed"] = no_intervention_succeeded
        replay_store.insert(transition)
        replay_copy.append(deepcopy(transition))
        if bool(transition["intervened"]):
            correction_store.insert(transition)
            correction_copy.append(deepcopy(transition))
    return replay_copy, correction_copy


def decode_trajectory(trajectory: list[dict], decoder) -> list[dict]:
    """Decode Pyro array payloads only once, at episode finalization."""
    return [decoder(transition) for transition in trajectory]


def scalar_bool(value: Any) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


_NUMBER = re.compile(r"(\d+)")


def _natural_key(path: Path):
    return tuple((1, int(part)) if part.isdigit() else (0, part) for part in _NUMBER.split(path.name))


def transition_shards(directory: Path) -> tuple[Path, ...]:
    directory = Path(directory)
    return tuple(sorted(directory.glob("transitions_*.pkl"), key=_natural_key))


def last_transition_step(directory: Path) -> int | None:
    shards = transition_shards(directory)
    if not shards:
        return None
    match = re.fullmatch(r"transitions_(\d+)\.pkl", shards[-1].name)
    return int(match.group(1)) if match else None


def _write_atomic(path: Path, transitions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(list(transitions), stream)
    os.replace(temporary, path)


class TransitionShardWriter:
    def __init__(self, replay_directory: Path, correction_directory: Path):
        self.replay_directory = Path(replay_directory)
        self.correction_directory = Path(correction_directory)

    @property
    def next_step(self) -> int:
        last = last_transition_step(self.replay_directory)
        return 0 if last is None else last + 1

    def __call__(self, step: int, replay: list[dict], correction: list[dict]) -> None:
        _write_atomic(self.replay_directory / f"transitions_{int(step)}.pkl", replay)
        _write_atomic(self.correction_directory / f"transitions_{int(step)}.pkl", correction)


def load_transition_shards(directory: Path) -> list[dict]:
    transitions: list[dict] = []
    for path in transition_shards(directory):
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        transitions.extend(payload.get("transitions", []) if isinstance(payload, dict) else payload or [])
    return transitions
