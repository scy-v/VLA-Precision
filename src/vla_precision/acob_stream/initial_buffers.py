"""Save and restore initial online buffers, including their ContextBuffer rows."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from vla_precision.acob_stream.buffers.context import ContextBuffer, ContextSource
from vla_precision.config.schema import RootConfig
from vla_precision.data.paths import (
    initial_buffer_dir,
    initial_context_dir,
    initial_correction_path,
    initial_replay_path,
)

LOGGER = logging.getLogger(__name__)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return list(payload.get("transitions", []) if isinstance(payload, dict) else payload or [])


def _write(path: Path, transitions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(list(transitions), stream)
    os.replace(temporary, path)


def _source(transition: dict, key: str) -> int:
    if key in transition:
        return int(np.asarray(transition[key]).reshape(-1)[0])
    if key == "next_context_source" and "context_source" in transition:
        return int(np.asarray(transition["context_source"]).reshape(-1)[0])
    return int(ContextSource.TRAINING)


def _without_preprocess(transitions: list[dict]) -> list[dict]:
    return [
        transition
        for transition in transitions
        if all(
            _source(transition, source_key) != int(ContextSource.PREPROCESS)
            for source_key in ("context_source", "next_context_source")
            if source_key in transition
        )
    ]


def _context_ids(transitions: list[dict], source: ContextSource) -> np.ndarray:
    ids: set[int] = set()
    for transition in transitions:
        for id_key, source_key in (
            ("context_id", "context_source"),
            ("next_context_id", "next_context_source"),
        ):
            if id_key in transition and _source(transition, source_key) == int(source):
                ids.update(int(value) for value in np.asarray(transition[id_key]).reshape(-1))
    return np.asarray(sorted(ids), dtype=np.int64)


def _remap(transitions: list[dict], source: ContextSource, target: ContextSource) -> list[dict]:
    result = []
    for transition in transitions:
        row = transition.copy()
        for key in ("context_source", "next_context_source"):
            if key in row and _source(row, key) == int(source):
                row[key] = np.int16(target)
        result.append(row)
    return result


def _register_initial(replay_buffer, correction_buffer, path: Path) -> None:
    replay_buffer.register_context_buffer(int(ContextSource.INITIAL), path)
    correction_buffer.register_context_buffer(int(ContextSource.INITIAL), path)


def _discard_incomplete_cache(config: RootConfig, context_path: Path) -> None:
    """Remove one unusable initial cache so the fill barrier can be rebuilt."""
    root = initial_buffer_dir(config)
    LOGGER.warning(
        "ignoring incomplete initial buffer cache without Context metadata: %s",
        context_path,
    )
    shutil.rmtree(root)


def load_initial_buffers(config: RootConfig, replay_buffer, correction_buffer, minimum: int) -> None:
    replay = _read(initial_replay_path(config))
    correction = _read(initial_correction_path(config))
    if not replay and not correction:
        return
    replay = _remap(_without_preprocess(replay), ContextSource.TRAINING, ContextSource.INITIAL)
    correction = _remap(_without_preprocess(correction), ContextSource.TRAINING, ContextSource.INITIAL)
    referenced = _context_ids(replay + correction, ContextSource.INITIAL)
    context_path = initial_context_dir(config)
    if referenced.size:
        if not (context_path / "metadata.json").exists():
            _discard_incomplete_cache(config, context_path)
            return
        _register_initial(replay_buffer, correction_buffer, context_path)
    for transition in replay:
        replay_buffer.insert(transition)
    for transition in correction:
        correction_buffer.insert(transition)
    if correction_buffer.sampleable_size() < minimum:
        return


def save_initial_buffers(
    config: RootConfig,
    replay_buffer,
    correction_buffer,
    minimum: int,
    training_context: ContextBuffer,
) -> None:
    """Persist the first fill barrier once; subsequent fresh runs reuse it."""
    if initial_correction_path(config).exists():
        return
    replay = _without_preprocess(replay_buffer.export_transitions())
    correction = _without_preprocess(correction_buffer.export_transitions())
    if len(correction) < int(minimum):
        raise ValueError(f"Initial correction cache has {len(correction)} rows; requires {minimum}")

    ids = _context_ids(replay + correction, ContextSource.TRAINING)
    if ids.size:
        initial = ContextBuffer.open_or_create(
            initial_context_dir(config),
            capacity=int(ids.max()) + 1,
            embedding_shape=training_context.logical_embedding_shape,
            embedding_mask_shape=training_context.logical_mask_shape,
            storage_embedding_shape=training_context.embedding_shape,
            storage_embedding_mask_shape=training_context.mask_shape,
            shard_capacity=training_context.shard_capacity,
            overwrite=False,
            metadata={
                "stage": "initial_buffer",
                "experiment": config.experiment.name,
                "context_source": int(ContextSource.INITIAL),
            },
            mode="r+",
        )
        bits, masks, offsets = training_context.get_batch_bits(ids)
        initial.put_batch_bits(ids, bits, masks, offsets)
        initial.flush()

    replay = _remap(replay, ContextSource.TRAINING, ContextSource.INITIAL)
    correction = _remap(correction, ContextSource.TRAINING, ContextSource.INITIAL)
    _write(initial_replay_path(config), replay)
    _write(initial_correction_path(config), correction)
