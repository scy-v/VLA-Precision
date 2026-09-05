"""Replay-buffer helpers for ACoB."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import gymnasium as gym
import jax
import ml_dtypes
import numpy as np
from flax.core import frozen_dict

from vla_precision.acob_stream.buffers._data_store import MemoryEfficientReplayBufferDataStore
from vla_precision.acob_stream.buffers._dataset import _sample
from vla_precision.acob_stream.buffers._memory_replay import MemoryEfficientReplayBuffer
from vla_precision.acob_stream.buffers._replay_base import ReplayBuffer as ArrayReplayBuffer


class ReplayBuffer(MemoryEfficientReplayBufferDataStore):
    """Memory-efficient replay buffer with optional ContextBuffer hydration."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        *,
        context_dim: int | None = 384,
        context_shape=None,
        context_mask_shape=None,
        context_dtype=np.float32,
        include_context: bool = False,
        context_buffers: dict[int, object] | None = None,
        context_buffer: object | None = None,
        default_context_source: int = 0,
        storage_context_shape=None,
        storage_context_mask_shape=None,
        detect_pixel_sequence_boundaries: bool = False,
        **kwargs,
    ):
        if context_buffers is None and context_buffer is not None:
            context_buffers = {int(default_context_source): context_buffer}
        self._context_enabled = bool(include_context and context_buffers)
        self._context_specs = {int(k): v for k, v in (context_buffers or {}).items()}
        self._context_opened = {}
        self._context_lock = threading.Lock()
        self._default_context_source = int(default_context_source)
        self._detect_pixel_sequence_boundaries = bool(detect_pixel_sequence_boundaries)
        self._previous_source_next_context_ref: tuple[int, int] | None = None
        self._previous_source_next_pixels: dict[str, np.ndarray] | None = None
        self._have_previous_source_transition = False
        self.last_context_hydration_seconds = 0.0

        dtype = ml_dtypes.bfloat16 if str(context_dtype) == "bfloat16" else np.dtype(context_dtype)
        logical_shape = tuple(context_shape) if context_shape is not None else (int(context_dim),)
        logical_mask_shape = tuple(context_mask_shape) if context_mask_shape is not None else (logical_shape[2] if len(logical_shape) >= 3 else logical_shape[0],)
        shape = tuple(storage_context_shape) if storage_context_shape is not None else logical_shape
        mask_shape = tuple(storage_context_mask_shape) if storage_context_mask_shape is not None else logical_mask_shape
        self._context_dtype = dtype
        self._context_embedding_transport_dtype = np.uint16
        self._context_embedding_logical_shape = logical_shape
        self._context_embedding_logical_mask_shape = logical_mask_shape
        self._context_shape = shape
        self._context_mask_shape = mask_shape

        super().__init__(
            observation_space,
            action_space,
            capacity,
            image_keys=image_keys,
            include_context=include_context and not self._context_enabled,
            **kwargs,
        )
        if include_context and self._context_enabled:
            self.dataset_dict["context_id"] = np.empty((capacity,), dtype=np.int64)
            self.dataset_dict["next_context_id"] = np.empty((capacity,), dtype=np.int64)
            self.dataset_dict["context_source"] = np.empty((capacity,), dtype=np.int16)
            self.dataset_dict["next_context_source"] = np.empty((capacity,), dtype=np.int16)
        elif include_context:
            self.dataset_dict["context_embeddings"] = np.empty((capacity, *shape), dtype=dtype)
            self.dataset_dict["next_context_embeddings"] = np.empty((capacity, *shape), dtype=dtype)
            self.dataset_dict["context_masks"] = np.empty((capacity, *mask_shape), dtype=bool)
            self.dataset_dict["next_context_masks"] = np.empty((capacity, *mask_shape), dtype=bool)
            self.dataset_dict["context_offsets"] = np.empty((capacity,), dtype=np.int32)
            self.dataset_dict["next_context_offsets"] = np.empty((capacity,), dtype=np.int32)

        self.dataset_dict["intervened"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["raw_intervened"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["intervention_metadata_present"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["pose_intervention_pressed"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["gripper_intervention_pressed"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["stop_intervention_pressed"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["gripper_action_changed"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["env_gripper_action_changed"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["episode_succeed"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["episode_no_intervention_succeed"] = np.empty((capacity,), dtype=bool)
        self.dataset_dict["intervention_bad_actions"] = np.empty_like(self.dataset_dict["actions"])
        self.last_outcome_sample_counts = {"success": 0, "failure": 0, "uniform": 0}
        self.last_replay_pool_counts = {
            "recent_pool": 0,
            "older_excluded": 0,
        }

    @property
    def context_enabled(self) -> bool:
        return self._context_enabled

    @property
    def context_record_bytes(self) -> int:
        if not self._context_enabled:
            return 0
        embedding_bytes = int(np.prod(self._context_shape)) * np.dtype(np.uint16).itemsize
        mask_bytes = int(np.prod(self._context_mask_shape)) * np.dtype(np.bool_).itemsize
        return embedding_bytes + mask_bytes + np.dtype(np.int32).itemsize

    def register_context_buffer(self, source: int, context_buffer) -> None:
        if not self._context_enabled:
            raise ValueError("Cannot register a ContextBuffer when context hydration is disabled.")
        source = int(source)
        with self._context_lock:
            self._context_specs[source] = context_buffer
            self._context_opened.pop(source, None)

    def sampleable_size(self) -> int:
        correct_index = getattr(self, "_is_correct_index", None)
        if correct_index is None:
            return len(self)
        return int(np.count_nonzero(correct_index[: len(self)]))

    def _valid_indices_chronological(self) -> np.ndarray:
        size = len(self)
        if size <= 0:
            return np.empty((0,), dtype=np.int64)
        if size < self._capacity:
            order = np.arange(size, dtype=np.int64)
        else:
            order = np.concatenate(
                (
                    np.arange(self._insert_index, size, dtype=np.int64),
                    np.arange(0, self._insert_index, dtype=np.int64),
                )
            )
        correct_index = getattr(self, "_is_correct_index", None)
        if correct_index is None:
            return order
        return order[np.asarray(correct_index[order], dtype=bool)]

    @staticmethod
    def _row_from_batch_tree(value, row: int):
        if isinstance(value, dict):
            return {key: ReplayBuffer._row_from_batch_tree(item, row) for key, item in value.items()}
        if hasattr(value, "unfreeze"):
            return ReplayBuffer._row_from_batch_tree(value.unfreeze(), row)
        return np.array(np.asarray(value)[row], copy=True)

    def export_transitions(self, limit: int | None = None) -> list[dict]:
        """Return insert-ready transitions from valid sampleable rows.

        MemoryEfficientReplayBuffer stores image frames unstacked internally. This
        method exports through the indexed sampling path so pixel observations are
        reconstructed into the same stacked transition format accepted by insert().
        """
        with self._lock:
            indices = self._valid_indices_chronological()
            if limit is not None:
                indices = indices[: max(0, int(limit))]
            if indices.size == 0:
                return []
            batch = self._sample_by_index(
                batch_size=int(indices.size),
                indx=indices,
                pack_obs=False,
            )
            batch = batch.unfreeze() if hasattr(batch, "unfreeze") else dict(batch)
            return [self._row_from_batch_tree(batch, row) for row in range(int(indices.size))]

    def _context_buffer_for(self, source: int):
        source = int(source)
        with self._context_lock:
            if source in self._context_opened:
                return self._context_opened[source]
            if source not in self._context_specs:
                raise KeyError(f"No ContextBuffer configured for context_source={source}")
            spec = self._context_specs[source]
            if hasattr(spec, "get_batch"):
                store = spec
            else:
                from vla_precision.acob_stream.buffers.context import ContextBuffer

                store = ContextBuffer(Path(spec).expanduser(), mode="r")
            if tuple(store.logical_embedding_shape) != tuple(self._context_embedding_logical_shape):
                raise ValueError(
                    f"ContextBuffer {store.root} logical shape {store.logical_embedding_shape} does not match replay buffer logical shape {self._context_embedding_logical_shape}"
                )
            if tuple(store.logical_mask_shape) != tuple(self._context_embedding_logical_mask_shape):
                raise ValueError(
                    f"ContextBuffer {store.root} logical mask shape {store.logical_mask_shape} does not match replay buffer logical mask shape {self._context_embedding_logical_mask_shape}"
                )
            if tuple(store.embedding_shape[:2] + store.embedding_shape[3:]) != tuple(self._context_shape[:2] + self._context_shape[3:]):
                raise ValueError(
                    f"ContextBuffer {store.root} storage shape {store.embedding_shape} is incompatible with replay buffer target shape {self._context_shape}"
                )
            if int(store.embedding_shape[2]) > int(self._context_shape[2]):
                raise ValueError(
                    f"ContextBuffer {store.root} stored tokens {store.embedding_shape[2]} exceed replay buffer target tokens {self._context_shape[2]}"
                )
            self._context_opened[source] = store
            return store

    def _fetch_context_batch(self, ids, sources, *, context_embeddings_out=None, masks_out=None, offsets_out=None):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        sources = np.asarray(sources, dtype=np.int16).reshape(-1)
        if ids.shape[0] != sources.shape[0]:
            raise ValueError(f"context ids shape {ids.shape} does not match sources shape {sources.shape}")
        expected_context_shape = (ids.shape[0], *self._context_shape)
        expected_mask_shape = (ids.shape[0], *self._context_mask_shape)
        context_embeddings = context_embeddings_out if context_embeddings_out is not None else np.empty(expected_context_shape, dtype=self._context_embedding_transport_dtype)
        masks = masks_out if masks_out is not None else np.empty(expected_mask_shape, dtype=bool)
        offsets = offsets_out if offsets_out is not None else np.empty((ids.shape[0],), dtype=np.int32)
        if context_embeddings.shape != expected_context_shape:
            raise ValueError(f"context_embeddings_out shape {context_embeddings.shape} does not match expected {expected_context_shape}")
        if masks.shape != expected_mask_shape:
            raise ValueError(f"masks_out shape {masks.shape} does not match expected {expected_mask_shape}")
        if offsets.shape != (ids.shape[0],):
            raise ValueError(f"offsets_out shape {offsets.shape} does not match expected {(ids.shape[0],)}")
        for source in np.unique(sources):
            row_mask = sources == source
            row_indices = np.flatnonzero(row_mask)
            context_ids_for_rows = ids[row_mask]
            order = np.argsort(context_ids_for_rows)
            store = self._context_buffer_for(int(source))
            target_rows = row_indices[order]
            if hasattr(store, "get_batch_bits_into"):
                store.get_batch_bits_into(
                    context_ids_for_rows[order],
                    context_embeddings,
                    masks,
                    offsets,
                    output_rows=target_rows,
                )
                continue
            if hasattr(store, "get_batch_bits"):
                batch_context_embeddings, batch_masks, batch_offsets = store.get_batch_bits(
                    context_ids_for_rows[order],
                    pad_to_embedding_shape=self._context_shape,
                    pad_to_mask_shape=self._context_mask_shape,
                )
            else:
                batch_context_embeddings, batch_masks, batch_offsets = store.get_batch(
                    context_ids_for_rows[order],
                    pad_to_embedding_shape=self._context_shape,
                    pad_to_mask_shape=self._context_mask_shape,
                )
                batch_context_embeddings = np.asarray(batch_context_embeddings, dtype=ml_dtypes.bfloat16).view(np.uint16)
            context_embeddings[target_rows] = np.asarray(batch_context_embeddings, dtype=self._context_embedding_transport_dtype)
            masks[target_rows] = np.asarray(batch_masks, dtype=bool)
            offsets[target_rows] = np.asarray(batch_offsets, dtype=np.int32)
        return context_embeddings, masks, offsets

    @staticmethod
    def _chunk_gripper_changed(policy_actions, intervention_actions) -> bool:
        try:
            policy_arr = np.asarray(policy_actions)
            intervention_arr = np.asarray(intervention_actions)
        except Exception:  # noqa: BLE001 - malformed external actions are treated as unchanged
            return False
        if policy_arr.ndim < 2 or intervention_arr.ndim < 2:
            return False
        if policy_arr.shape != intervention_arr.shape or policy_arr.shape[-1] < 1:
            return False
        gripper_indices = (6, 13) if policy_arr.shape[-1] == 14 else (policy_arr.shape[-1] - 1,)
        policy_open = policy_arr[:, gripper_indices] >= 0.5
        intervention_open = intervention_arr[:, gripper_indices] >= 0.5
        return bool(np.any(policy_open != intervention_open))

    @staticmethod
    def _context_ref(data_dict, *, next_observation: bool) -> tuple[int, int] | None:
        id_key = "next_context_id" if next_observation else "context_id"
        store_key = "next_context_source" if next_observation else "context_source"
        if id_key not in data_dict or store_key not in data_dict:
            return None
        context_id = int(np.asarray(data_dict[id_key]).reshape(-1)[0])
        source = int(np.asarray(data_dict[store_key]).reshape(-1)[0])
        return source, context_id

    def _observation_pixels(self, observations) -> dict[str, np.ndarray] | None:
        if not isinstance(observations, Mapping):
            return None
        pixels = {}
        for pixel_key in self.pixel_keys:
            if pixel_key not in observations:
                return None
            pixels[pixel_key] = np.array(np.asarray(observations[pixel_key]), copy=True)
        return pixels

    @staticmethod
    def _pixel_observations_equal(
        previous_next_pixels: dict[str, np.ndarray] | None,
        current_pixels: dict[str, np.ndarray] | None,
    ) -> bool:
        if previous_next_pixels is None or current_pixels is None:
            return False
        if previous_next_pixels.keys() != current_pixels.keys():
            return False
        return all(
            np.array_equal(previous_next_pixels[key], current_pixels[key])
            for key in previous_next_pixels
        )

    def _source_transition_is_contiguous(
        self,
        *,
        current_context_ref: tuple[int, int] | None,
        current_pixels: dict[str, np.ndarray] | None,
    ) -> bool:
        if not self._have_previous_source_transition:
            return False
        if self._previous_source_next_context_ref is not None and current_context_ref is not None:
            return self._previous_source_next_context_ref == current_context_ref
        return self._pixel_observations_equal(self._previous_source_next_pixels, current_pixels)

    def _padding_row_for_sequence_start(self, data_dict):
        """Build an unstacked, non-sampleable row used only to reach ring index zero."""
        padding = data_dict.copy()
        padding["observations"] = data_dict["observations"].copy()
        padding["next_observations"] = data_dict["next_observations"].copy()
        for pixel_key in self.pixel_keys:
            current_stack = np.asarray(padding["observations"][pixel_key])
            padding["observations"][pixel_key] = current_stack[-1]
            padding["next_observations"].pop(pixel_key, None)
        return padding

    def _pad_ring_before_sequence_start(self, data_dict) -> None:
        """Keep a new pixel context and its valid row on the same side of ring wrap."""
        if self._insert_index == 0:
            return
        required_rows = int(self._num_stack) + 1
        remaining_rows = int(self._capacity) - int(self._insert_index)
        if remaining_rows >= required_rows:
            return
        padding = self._padding_row_for_sequence_start(data_dict)
        for _ in range(remaining_rows):
            self._is_correct_index[self._insert_index] = False
            ArrayReplayBuffer.insert(self, padding)

    def insert(self, data_dict):
        data_dict = data_dict.copy()
        has_intervened_key = "intervened" in data_dict
        has_episode_succeed_key = "episode_succeed" in data_dict
        data_dict.setdefault("intervened", False)
        data_dict.setdefault("episode_succeed", True)
        if "episode_no_intervention_succeed" not in data_dict:
            data_dict["episode_no_intervention_succeed"] = (
                data_dict["episode_succeed"]
                if not has_intervened_key and not has_episode_succeed_key
                else False
            )
        data_dict.setdefault("intervention_bad_actions", data_dict["actions"])
        data_dict.setdefault("raw_intervened", data_dict["intervened"])
        data_dict.setdefault("intervention_metadata_present", False)
        data_dict.setdefault("pose_intervention_pressed", False)
        data_dict.setdefault("gripper_intervention_pressed", False)
        data_dict.setdefault("stop_intervention_pressed", False)
        data_dict.setdefault("gripper_action_changed", False)
        data_dict.setdefault("env_gripper_action_changed", data_dict["gripper_action_changed"])
        data_dict["intervened"] = bool(np.asarray(data_dict["intervened"]).reshape(-1)[0])
        data_dict["raw_intervened"] = bool(np.asarray(data_dict["raw_intervened"]).reshape(-1)[0])
        data_dict["intervention_metadata_present"] = bool(np.asarray(data_dict["intervention_metadata_present"]).reshape(-1)[0])
        data_dict["pose_intervention_pressed"] = bool(np.asarray(data_dict["pose_intervention_pressed"]).reshape(-1)[0])
        data_dict["gripper_intervention_pressed"] = bool(np.asarray(data_dict["gripper_intervention_pressed"]).reshape(-1)[0])
        data_dict["stop_intervention_pressed"] = bool(np.asarray(data_dict["stop_intervention_pressed"]).reshape(-1)[0])
        data_dict["gripper_action_changed"] = bool(np.asarray(data_dict["gripper_action_changed"]).reshape(-1)[0])
        data_dict["env_gripper_action_changed"] = bool(np.asarray(data_dict["env_gripper_action_changed"]).reshape(-1)[0])
        if data_dict["raw_intervened"] or data_dict["intervened"]:
            recomputed_gripper_changed = self._chunk_gripper_changed(
                data_dict["intervention_bad_actions"],
                data_dict["actions"],
            )
            data_dict["gripper_action_changed"] = recomputed_gripper_changed
            if data_dict["intervention_metadata_present"]:
                data_dict["intervened"] = bool(
                    data_dict["pose_intervention_pressed"]
                    or data_dict["stop_intervention_pressed"]
                    or (data_dict["gripper_intervention_pressed"] and data_dict["gripper_action_changed"])
                )
        data_dict["episode_succeed"] = bool(np.asarray(data_dict["episode_succeed"]).reshape(-1)[0])
        data_dict["episode_no_intervention_succeed"] = bool(
            np.asarray(data_dict["episode_no_intervention_succeed"]).reshape(-1)[0]
        )
        if self._context_enabled:
            if "context_id" not in data_dict or "next_context_id" not in data_dict:
                raise KeyError("ACoB replay transitions require context_id and next_context_id")
            data_dict.setdefault("context_source", np.int16(self._default_context_source))
            data_dict.setdefault("next_context_source", np.int16(data_dict["context_source"]))

        if not self._detect_pixel_sequence_boundaries:
            return super().insert(data_dict)

        current_context_ref = self._context_ref(data_dict, next_observation=False)
        next_context_ref = self._context_ref(data_dict, next_observation=True)
        current_pixels = (
            None
            if current_context_ref is not None
            else self._observation_pixels(data_dict.get("observations"))
        )
        next_pixels = (
            None
            if next_context_ref is not None
            else self._observation_pixels(data_dict.get("next_observations"))
        )
        transition_done = bool(np.asarray(data_dict["dones"]).reshape(-1)[0])

        # MemoryEfficientReplayBuffer assumes every inserted source transition is
        # adjacent to the previous one and reconstructs current RGB from the
        # previous stored next RGB. Correction buffers contain only intervened
        # transitions, so a new intervention segment must seed its own current
        # RGB context. This is a storage boundary only: never change done/mask.
        with self._lock:
            if self._have_previous_source_transition and not self._source_transition_is_contiguous(
                current_context_ref=current_context_ref,
                current_pixels=current_pixels,
            ):
                self._first = True
            if self._first:
                self._pad_ring_before_sequence_start(data_dict)

            # Call the unlocked grandparent implementation because this method
            # already owns MemoryEfficientReplayBufferDataStore._lock.
            MemoryEfficientReplayBuffer.insert(self, data_dict)

            if transition_done:
                self._have_previous_source_transition = False
                self._previous_source_next_context_ref = None
                self._previous_source_next_pixels = None
            else:
                self._have_previous_source_transition = True
                self._previous_source_next_context_ref = next_context_ref
                self._previous_source_next_pixels = next_pixels

    def _outcome_sample_indices(self, batch_size: int, success_ratio: float | None):
        self.last_outcome_sample_counts = {"success": 0, "failure": 0, "uniform": int(batch_size)}
        if success_ratio is None:
            return None
        success_ratio = float(success_ratio)
        if not np.isfinite(success_ratio):
            raise ValueError(f"online_replay_success_ratio must be finite, got {success_ratio}.")
        if success_ratio <= 0.0:
            return None
        if len(self) <= 0 or "episode_succeed" not in self.dataset_dict:
            return None
        valid = np.flatnonzero(self._is_correct_index[: len(self)])
        if valid.size == 0:
            return None
        outcomes = np.asarray(self.dataset_dict["episode_succeed"][: len(self)], dtype=bool)
        success_indices = valid[outcomes[valid]]
        failure_indices = valid[~outcomes[valid]]
        if success_indices.size == 0:
            raise ValueError(
                "replay_buffer has no successful transitions available for outcome-balanced sampling. "
                "Successful replay transitions are required to fill the non-correction replay batch."
            )
        target_success_count = int(np.ceil(batch_size * success_ratio / (1.0 + success_ratio)))
        target_success_count = min(batch_size - 1, max(1, target_success_count)) if batch_size > 1 else batch_size
        target_failure_count = batch_size - target_success_count
        failure_count = min(int(failure_indices.size), target_failure_count)
        success_count = batch_size - failure_count
        if success_indices.size < success_count:
            raise ValueError(
                f"replay_buffer has only {success_indices.size} successful transitions, "
                f"but {success_count} are required to fill a replay batch of {batch_size} "
                f"with online_replay_success_ratio={success_ratio}."
            )
        rng = self.np_random
        success_sample = rng.choice(success_indices, size=success_count, replace=False)
        failure_sample = rng.choice(failure_indices, size=failure_count, replace=False) if failure_count > 0 else np.empty((0,), dtype=np.int64)
        indices = np.concatenate([success_sample, failure_sample]).astype(np.int64, copy=False)
        rng.shuffle(indices)
        self.last_outcome_sample_counts = {"success": int(success_count), "failure": int(failure_count), "uniform": 0}
        return indices

    def _recent_sample_indices(
        self,
        batch_size: int,
        *,
        recent_window: int,
        success_ratio: float | None,
    ):
        """Uniformly sample from the most recently inserted valid replay rows.

        Preprocessed demos are inserted before restored/online transitions, so
        they naturally behave as the oldest replay data. They remain eligible
        until enough newer rows push them outside ``recent_window``. ContextBuffer
        identity is deliberately irrelevant to replay eligibility.
        """
        recent_window = int(recent_window)
        self.last_replay_pool_counts = {
            "recent_pool": 0,
            "older_excluded": 0,
        }
        if recent_window <= 0:
            return None
        if success_ratio is not None and float(success_ratio) > 0.0:
            raise ValueError(
                "online_replay_recent_window cannot currently be combined with "
                "online_replay_success_ratio > 0; use one replay sampling policy at a time."
            )
        if batch_size <= 0:
            return np.empty((0,), dtype=np.int64)

        self.last_outcome_sample_counts = {"success": 0, "failure": 0, "uniform": int(batch_size)}
        with self._lock:
            valid = self._valid_indices_chronological()
            recent = valid[-recent_window:]
            older_excluded = max(0, int(valid.size - recent.size))
            if recent.size == 0:
                raise ValueError("Replay buffer has no sampleable rows in the recent window.")
            rng = self.np_random
            indices = rng.choice(
                recent,
                size=batch_size,
                replace=bool(recent.size < batch_size),
            ).astype(np.int64, copy=False)
            rng.shuffle(indices)

        self.last_replay_pool_counts = {
            "recent_pool": int(recent.size),
            "older_excluded": int(older_excluded),
        }
        return indices

    def latest_replay_sampling_stats(self) -> dict[str, int]:
        return dict(self.last_replay_pool_counts)

    def _sample_by_index(self, batch_size: int, keys=None, indx=None, pack_obs: bool = False):
        if indx is None:
            return super().sample(batch_size=batch_size, keys=keys, indx=None, pack_obs=pack_obs)
        indx = np.asarray(indx, dtype=np.int64)
        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys
        keys = list(keys)
        keys.remove("observations")
        batch = ArrayReplayBuffer.sample(self, batch_size, keys, indx)
        batch = batch.unfreeze()

        obs_keys = list(self.dataset_dict["observations"].keys())
        for pixel_key in self.pixel_keys:
            obs_keys.remove(pixel_key)

        batch["observations"] = {}
        for key in obs_keys:
            batch["observations"][key] = _sample(self.dataset_dict["observations"][key], indx)

        for pixel_key in self.pixel_keys:
            obs_pixels = self.dataset_dict["observations"][pixel_key]
            obs_pixels = np.lib.stride_tricks.sliding_window_view(obs_pixels, self._num_stack + 1, axis=0)
            obs_pixels = obs_pixels[indx - self._num_stack]
            obs_pixels = obs_pixels.transpose((0, 4, 1, 2, 3))

            if pack_obs:
                batch["observations"][pixel_key] = obs_pixels
            else:
                batch["observations"][pixel_key] = obs_pixels[:, :-1, ...]
                if "next_observations" in keys:
                    batch["next_observations"][pixel_key] = obs_pixels[:, 1:, ...]

        return frozen_dict.freeze(batch)

    def hydrate_context_batch(self, batch, *, include_current: bool = True, include_next: bool = True):
        if not self._context_enabled:
            return batch
        if not include_current and not include_next:
            return batch
        start = time.time()
        mutable = batch.unfreeze() if hasattr(batch, "unfreeze") else dict(batch)
        ids = np.asarray(mutable.pop("context_id"), dtype=np.int64)
        next_ids = np.asarray(mutable.pop("next_context_id"), dtype=np.int64)
        # Context sources are also the persistent data-source labels used by the
        # preprocess BC-only RL mask. Keep them in the hydrated learner batch.
        sources = np.asarray(mutable["context_source"], dtype=np.int16)
        next_sources = np.asarray(mutable["next_context_source"], dtype=np.int16)
        if include_current:
            context_embeddings, masks, offsets = self._fetch_context_batch(ids, sources)
            mutable["context_embeddings"] = context_embeddings
            mutable["context_masks"] = masks
            mutable["context_offsets"] = offsets
        if include_next:
            next_context_embeddings, next_masks, next_offsets = self._fetch_context_batch(next_ids, next_sources)
            mutable["next_context_embeddings"] = next_context_embeddings
            mutable["next_context_masks"] = next_masks
            mutable["next_context_offsets"] = next_offsets
        self.last_context_hydration_seconds = time.time() - start
        return frozen_dict.freeze(mutable)

    def sample(
        self,
        batch_size: int,
        keys=None,
        indx=None,
        pack_obs: bool = False,
        hydrate_context: bool = True,
        hydrate_current_context: bool = True,
        hydrate_next_context: bool = True,
        success_ratio: float | None = None,
        recent_online_window: int = 0,
    ):
        if self._context_enabled and keys is not None:
            keys = list(keys)
            required_keys = []
            if hydrate_current_context or not hydrate_context:
                required_keys.extend(("context_id", "context_source"))
            if hydrate_next_context or not hydrate_context:
                required_keys.extend(("next_context_id", "next_context_source"))
            for required_key in required_keys:
                if required_key not in keys:
                    keys.append(required_key)
        if indx is None:
            indx = self._recent_sample_indices(
                batch_size,
                recent_window=recent_online_window,
                success_ratio=success_ratio,
            )
        if indx is None:
            indx = self._outcome_sample_indices(batch_size, success_ratio)
        batch = self._sample_by_index(batch_size=batch_size, keys=keys, indx=indx, pack_obs=pack_obs)
        if not self._context_enabled or not hydrate_context:
            return batch
        return self.hydrate_context_batch(
            batch,
            include_current=hydrate_current_context,
            include_next=hydrate_next_context,
        )


def _as_mutable_mapping(batch):
    return batch.unfreeze() if hasattr(batch, "unfreeze") else batch


def _concat_batches_cpu(left, right, *, axis=0):
    left = _as_mutable_mapping(left)
    right = _as_mutable_mapping(right)
    batch = {}
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, dict) or hasattr(left_value, "unfreeze"):
            batch[key] = _concat_batches_cpu(left_value, right_value, axis=axis)
        else:
            batch[key] = np.concatenate((np.asarray(left_value), np.asarray(right_value)), axis=axis)
    return batch


def online_batch_split_counts(batch_size: int, correction_ratio: float = 1.0) -> tuple[int, int]:
    batch_size = int(batch_size)
    correction_ratio = float(correction_ratio)
    if batch_size < 2:
        raise ValueError(f"Online batch_size must be at least 2, got {batch_size}.")
    if not np.isfinite(correction_ratio) or correction_ratio <= 0.0:
        raise ValueError(f"online_correction_ratio must be finite and positive, got {correction_ratio}.")
    correction_batch = int(np.ceil(batch_size * correction_ratio / (1.0 + correction_ratio)))
    correction_batch = min(batch_size - 1, max(1, correction_batch))
    replay_batch = batch_size - correction_batch
    return replay_batch, correction_batch


def make_prefetched_online_batch_iterator(
    replay_buffer,
    correction_buffer,
    *,
    batch_size: int,
    device,
    queue_size: int = 16,
    num_workers: int = 4,
    cta_ratio: int = 2,
    correction_ratio: float = 1.0,
    replay_success_ratio: float | None = None,
    replay_recent_window: int = 0,
    prefetch_to_device: bool = False,
    device_queue_size: int = 1,
    prefetch_device=None,
    active_kinds: tuple[str, ...] = ("critic", "full"),
    max_inflight_per_kind: int = 2,
):
    replay_batch_size, correction_batch_size = online_batch_split_counts(batch_size, correction_ratio)
    queue_size = max(1, int(queue_size))
    device_queue_size = max(1, int(device_queue_size))
    cpu_queues = {
        "critic": queue.Queue(maxsize=queue_size),
        "full": queue.Queue(maxsize=queue_size),
    }
    device_queues = {
        "critic": queue.Queue(maxsize=device_queue_size),
        "full": queue.Queue(maxsize=device_queue_size),
    }
    stop_event = threading.Event()
    sentinel = object()
    num_workers = max(1, int(num_workers))
    cta_ratio = max(1, int(cta_ratio))
    active_kinds = tuple(dict.fromkeys(active_kinds))
    max_inflight_per_kind = int(max_inflight_per_kind)
    if max_inflight_per_kind <= 0:
        raise ValueError(f"max_inflight_per_kind must be positive, got {max_inflight_per_kind}.")
    inflight_slots = {
        kind: threading.BoundedSemaphore(max_inflight_per_kind) for kind in cpu_queues
    }
    inflight_counts = {kind: 0 for kind in cpu_queues}
    inflight_peaks = {kind: 0 for kind in cpu_queues}
    inflight_lock = threading.Lock()
    invalid_kinds = sorted(set(active_kinds) - set(cpu_queues))
    if invalid_kinds:
        raise ValueError(f"Unknown online batch kind(s): {invalid_kinds}")
    if not active_kinds:
        raise ValueError("active_kinds must contain at least one online batch kind.")
    if active_kinds == ("critic",):
        worker_kinds = ["critic"] * num_workers
    elif active_kinds == ("full",):
        worker_kinds = ["full"] * num_workers
    elif num_workers == 1:
        worker_kinds = ["critic", "full"]
    else:
        critic_fraction = (cta_ratio - 1) / cta_ratio if cta_ratio > 1 else 0.5
        critic_workers = round(num_workers * critic_fraction)
        critic_workers = min(max(1, critic_workers), num_workers - 1)
        full_workers = max(1, num_workers - critic_workers)
        worker_kinds = (["critic"] * critic_workers) + (["full"] * full_workers)
    last_stats = {}
    use_fast_context = bool(
        getattr(replay_buffer, "context_enabled", False)
        and getattr(correction_buffer, "context_enabled", False)
        and hasattr(replay_buffer, "hydrate_context_batch")
    )
    if prefetch_to_device and prefetch_device is None:
        raise ValueError("prefetch_to_device=True requires an explicit separate prefetch_device.")
    device_put_lock = threading.Lock()
    stage_device = prefetch_device if prefetch_device is not None else device
    staged_to_train_device = bool(prefetch_to_device and prefetch_device is not None and prefetch_device != device)

    def sample_cpu_batch(kind: str):
        if kind not in cpu_queues:
            raise ValueError(f"Unknown online batch kind: {kind}")
        hydrate_current = kind == "full"
        hydrate_next = True
        start = time.time()
        replay_batch = replay_buffer.sample(
            batch_size=replay_batch_size,
            pack_obs=True,
            hydrate_context=not use_fast_context,
            success_ratio=replay_success_ratio,
            recent_online_window=replay_recent_window,
        )
        replay_elapsed = time.time() - start
        replay_context_elapsed = float(getattr(replay_buffer, "last_context_hydration_seconds", 0.0))
        replay_outcome_counts = getattr(replay_buffer, "last_outcome_sample_counts", {})
        replay_pool_counts = (
            replay_buffer.latest_replay_sampling_stats()
            if hasattr(replay_buffer, "latest_replay_sampling_stats")
            else {}
        )
        correction_start = time.time()
        correction_batch = correction_buffer.sample(batch_size=correction_batch_size, pack_obs=True, hydrate_context=not use_fast_context)
        correction_elapsed = time.time() - correction_start
        correction_context_elapsed = float(getattr(correction_buffer, "last_context_hydration_seconds", 0.0))
        concat_start = time.time()
        batch = frozen_dict.freeze(_concat_batches_cpu(replay_batch, correction_batch, axis=0))
        concat_elapsed = time.time() - concat_start
        hydrate_elapsed = 0.0
        if use_fast_context:
            hydrate_start = time.time()
            batch = replay_buffer.hydrate_context_batch(
                batch,
                include_current=hydrate_current,
                include_next=hydrate_next,
            )
            hydrate_elapsed = time.time() - hydrate_start
            replay_context_elapsed = hydrate_elapsed
            correction_context_elapsed = 0.0
        return batch, {
            "prefetch_fast_context": float(use_fast_context),
            "prefetch_to_device": float(prefetch_to_device),
            "prefetch_stage_to_train_device": float(staged_to_train_device),
            "prefetch_critic_only_context": float(kind == "critic"),
            "prefetch_hydrate_current_context": float(hydrate_current),
            "prefetch_hydrate_next_context": float(hydrate_next),
            "prefetch_replay_batch_size": float(replay_batch_size),
            "prefetch_replay_success_batch_size": float(replay_outcome_counts.get("success", 0)),
            "prefetch_replay_failure_batch_size": float(replay_outcome_counts.get("failure", 0)),
            "prefetch_replay_uniform_batch_size": float(replay_outcome_counts.get("uniform", replay_batch_size)),
            "prefetch_replay_recent_pool_size": float(replay_pool_counts.get("recent_pool", 0)),
            "prefetch_replay_older_excluded": float(replay_pool_counts.get("older_excluded", 0)),
            "prefetch_correction_batch_size": float(correction_batch_size),
            "prefetch_replay_sample": replay_elapsed,
            "prefetch_replay_context": replay_context_elapsed,
            "prefetch_correction_sample": correction_elapsed,
            "prefetch_correction_context": correction_context_elapsed,
            "prefetch_cpu_concat": concat_elapsed,
            "prefetch_final_context": hydrate_elapsed,
            "prefetch_cpu_total": time.time() - start,
        }

    def acquire_inflight(kind: str) -> bool:
        while not stop_event.is_set():
            if inflight_slots[kind].acquire(timeout=0.1):
                with inflight_lock:
                    inflight_counts[kind] += 1
                    inflight_peaks[kind] = max(inflight_peaks[kind], inflight_counts[kind])
                return True
        return False

    def release_inflight(kind: str):
        with inflight_lock:
            inflight_counts[kind] -= 1
        inflight_slots[kind].release()

    def cpu_worker(worker_id: int, kind: str):
        while not stop_event.is_set():
            if not acquire_inflight(kind):
                break
            item = None
            enqueued = False
            try:
                try:
                    item = sample_cpu_batch(kind)
                except BaseException as exc:  # noqa: BLE001 - propagate worker failure to consumer
                    item = exc
                target_queue = cpu_queues[kind]
                while not stop_event.is_set():
                    try:
                        target_queue.put(item, timeout=0.1)
                        enqueued = True
                        break
                    except queue.Full:
                        continue
            finally:
                release_inflight(kind)
            if enqueued and isinstance(item, BaseException):
                break

    def move_to_device(item):
        if isinstance(item, BaseException):
            return item
        batch, stats = item
        put_start = time.time()
        with device_put_lock:
            batch = jax.device_put(batch, device=stage_device)
            batch = jax.block_until_ready(batch)
        stage_device_put = time.time() - put_start
        return batch, stats | {"prefetch_stage_device_put": stage_device_put, "prefetch_device_prefetched": 1.0}

    def device_worker(kind: str):
        source_queue = cpu_queues[kind]
        target_queue = device_queues[kind]
        while not stop_event.is_set():
            try:
                item = source_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is sentinel:
                break
            if not isinstance(item, BaseException):
                while not stop_event.is_set() and target_queue.full():
                    time.sleep(0.05)
            if stop_event.is_set():
                break
            item = move_to_device(item)
            while not stop_event.is_set():
                try:
                    target_queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if isinstance(item, BaseException):
                break

    cpu_threads = [
        threading.Thread(
            target=cpu_worker,
            args=(worker_id, kind),
            name=f"acob-stream-cpu-prefetch-{kind}-{worker_id}",
            daemon=True,
        )
        for worker_id, kind in enumerate(worker_kinds)
    ]
    for thread in cpu_threads:
        thread.start()

    device_threads = []
    if prefetch_to_device:
        device_threads = [
            threading.Thread(target=device_worker, args=(kind,), name=f"acob-stream-device-prefetch-{kind}", daemon=True)
            for kind in active_kinds
        ]
        for thread in device_threads:
            thread.start()

    def next_batch(kind: str = "full"):
        if kind not in cpu_queues:
            raise ValueError(f"Unknown online batch kind: {kind}")
        if kind not in active_kinds:
            raise ValueError(f"Online batch kind {kind!r} is not active for this prefetch iterator: {active_kinds}")
        wait_start = time.time()
        source_queue = device_queues[kind] if prefetch_to_device else cpu_queues[kind]
        try:
            item = source_queue.get(timeout=120.0)
        except queue.Empty as exc:
            queue_sizes = {
                "prefetch_cpu_queue_size": float(cpu_queues[kind].qsize()),
                "prefetch_device_queue_size": float(device_queues[kind].qsize()),
            }
            for queue_kind, cpu_queue in cpu_queues.items():
                queue_sizes[f"prefetch_{queue_kind}_cpu_queue_size"] = float(cpu_queue.qsize())
                queue_sizes[f"prefetch_{queue_kind}_device_queue_size"] = float(device_queues[queue_kind].qsize())
            cpu_alive = [thread.is_alive() for thread in cpu_threads]
            device_alive = [thread.is_alive() for thread in device_threads]
            raise TimeoutError(
                "Timed out waiting for ACoB online prefetch batch. "
                f"kind={kind}, prefetch_to_device={prefetch_to_device}, "
                f"queue_sizes={queue_sizes}, cpu_threads_alive={cpu_alive}, "
                f"device_threads_alive={device_alive}, last_stats={dict(last_stats)}"
            ) from exc
        queue_wait = time.time() - wait_start
        if item is sentinel:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        if prefetch_to_device:
            batch, stats = item
            if staged_to_train_device:
                put_start = time.time()
                batch = jax.device_put(batch, device=device)
                batch = jax.block_until_ready(batch)
                device_put = time.time() - put_start
            else:
                device_put = 0.0
            consume_total = queue_wait + device_put
        else:
            batch, stats = item
            put_start = time.time()
            batch = jax.device_put(batch, device=device)
            batch = jax.block_until_ready(batch)
            device_put = time.time() - put_start
            stats = stats | {"prefetch_device_prefetched": 0.0}
            consume_total = queue_wait + device_put
        queue_sizes = {
            "prefetch_cpu_queue_size": float(cpu_queues[kind].qsize()),
            "prefetch_device_queue_size": float(device_queues[kind].qsize()),
        }
        for queue_kind, cpu_queue in cpu_queues.items():
            queue_sizes[f"prefetch_{queue_kind}_cpu_queue_size"] = float(cpu_queue.qsize())
            queue_sizes[f"prefetch_{queue_kind}_device_queue_size"] = float(device_queues[queue_kind].qsize())
        with inflight_lock:
            inflight_stats = {
                f"prefetch_{queue_kind}_inflight": float(inflight_counts[queue_kind])
                for queue_kind in cpu_queues
            } | {
                f"prefetch_{queue_kind}_inflight_peak": float(inflight_peaks[queue_kind])
                for queue_kind in cpu_queues
            }
        last_stats.clear()
        last_stats.update(
            stats
            | queue_sizes
            | inflight_stats
            | {
                "prefetch_queue_wait": queue_wait,
                "prefetch_device_put": device_put,
                "prefetch_consume_total": consume_total,
            }
        )
        return batch

    def latest_stats():
        return dict(last_stats)

    def close():
        stop_event.set()
        for batch_queue in list(cpu_queues.values()) + list(device_queues.values()):
            for _ in range(len(cpu_threads) + len(device_threads) + 1):
                try:
                    batch_queue.put_nowait(sentinel)
                except queue.Full:
                    pass
        for thread in cpu_threads + device_threads:
            thread.join(timeout=1.0)

    return next_batch, latest_stats, close
