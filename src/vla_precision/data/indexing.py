"""Materialize configured state/action indices before dataset sampling.

``DataConfig.state_indices`` and ``DataConfig.action_indices`` are zero-based.
This differs from the old environment-variable interface, whose values were
one-based and were converted with ``index - 1``.  The resolved YAML values are
used directly here: for example ``[1, 0]`` swaps the first two elements.

Both columns are rewritten by one Hugging Face ``Dataset.map`` call.  Nothing
in this module wraps ``__getitem__`` or performs indexing in the sampling hot
path.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

INDEX_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexMaterializationMetadata:
    """A reproducible description of the materialized dataset columns."""

    format: str
    format_version: int
    index_base: int
    state_key: str
    state_indices: tuple[int, ...]
    action_key: str
    action_indices: tuple[int, ...]
    schema_sha256: str
    source_fingerprint: str | None = None
    materialized_fingerprint: str | None = None
    config_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state_indices"] = list(self.state_indices)
        payload["action_indices"] = list(self.action_indices)
        return payload

    def write(self, destination: str | Path) -> Path:
        """Atomically write the indexing record next to the prepared data."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)
        return destination


@dataclass(frozen=True)
class IndexMaterializationResult:
    hf_dataset: Any
    metadata: IndexMaterializationMetadata


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def _features(dataset: Any) -> Any:
    return _jsonable(getattr(dataset, "features", None))


def _schema_hash(
    *,
    state_key: str,
    state_indices: tuple[int, ...],
    action_key: str,
    action_indices: tuple[int, ...],
    source_features: Any,
    materialized_features: Any,
) -> str:
    schema = {
        "format": "vla_precision.index_materialization",
        "format_version": INDEX_SCHEMA_VERSION,
        "index_base": 0,
        "state": {"key": state_key, "indices": list(state_indices)},
        "action": {"key": action_key, "indices": list(action_indices)},
        "source_features": source_features,
        "materialized_features": materialized_features,
    }
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _take_last_axis(value: Any, indices: tuple[int, ...]) -> list[Any]:
    return np.take(np.asarray(value), indices, axis=-1).tolist()


def _feature_names(dataset: Any, key: str) -> tuple[str, ...]:
    names = getattr(getattr(dataset, "meta", None), "names", {})
    selected = names.get(key) if isinstance(names, dict) else None
    if not isinstance(selected, (list, tuple)):
        return ()
    return tuple(str(name) for name in selected)


def _selected_feature_names(
    key: str,
    indices: tuple[int, ...],
    names: tuple[str, ...],
) -> list[str]:
    if not indices:
        return list(names) if names else ["all source dimensions"]
    return [names[index] if index < len(names) else f"{key}[{index}]" for index in indices]


def materialize_indices(
    hf_dataset: Any,
    *,
    state_key: str,
    state_indices: Sequence[int] = (),
    action_key: str,
    action_indices: Sequence[int] = (),
    metadata_path: str | Path | None = None,
    config_sha256: str | None = None,
) -> IndexMaterializationResult:
    """Materialize zero-based index selection with at most one ``map`` call.

    Empty index lists keep their corresponding columns unchanged.  Duplicate
    and reordered indices are intentional and supported.  The dataset itself
    supplies the useful boundary failure if a key or index does not exist.
    """
    state_indices = tuple(int(index) for index in state_indices)
    action_indices = tuple(int(index) for index in action_indices)
    if any(index < 0 for index in (*state_indices, *action_indices)):
        raise ValueError("state_indices and action_indices use zero-based, non-negative values")

    source_features = _features(hf_dataset)
    source_fingerprint = getattr(hf_dataset, "_fingerprint", None)

    materialized = hf_dataset
    if state_indices or action_indices:

        def select_indices(example: dict[str, Any]) -> dict[str, Any]:
            if state_indices:
                example[state_key] = _take_last_axis(example[state_key], state_indices)
            if action_indices:
                example[action_key] = _take_last_axis(example[action_key], action_indices)
            return example

        materialized = hf_dataset.map(select_indices, desc="Materializing state/action indices")

    metadata = IndexMaterializationMetadata(
        format="vla_precision.index_materialization",
        format_version=INDEX_SCHEMA_VERSION,
        index_base=0,
        state_key=state_key,
        state_indices=state_indices,
        action_key=action_key,
        action_indices=action_indices,
        schema_sha256=_schema_hash(
            state_key=state_key,
            state_indices=state_indices,
            action_key=action_key,
            action_indices=action_indices,
            source_features=source_features,
            materialized_features=_features(materialized),
        ),
        source_fingerprint=None if source_fingerprint is None else str(source_fingerprint),
        materialized_fingerprint=(
            None if getattr(materialized, "_fingerprint", None) is None else str(materialized._fingerprint)
        ),
        config_sha256=config_sha256,
    )
    if metadata_path is not None:
        metadata.write(metadata_path)
    return IndexMaterializationResult(hf_dataset=materialized, metadata=metadata)


def materialize_lerobot_indices(
    dataset: Any,
    data_config: Any,
    *,
    metadata_path: str | Path | None = None,
    config_sha256: str | None = None,
) -> IndexMaterializationMetadata:
    """Apply top-level ``DataConfig`` indices to a LeRobot dataset in place.

    Call this immediately after constructing ``LeRobotDataset`` and before
    constructing an OpenPI sampler/data loader.
    """
    ensure_loaded = getattr(dataset, "_ensure_hf_dataset_loaded", None)
    if ensure_loaded is not None:
        ensure_loaded()
    state_indices = tuple(int(index) for index in data_config.state_indices)
    action_indices = tuple(int(index) for index in data_config.action_indices)
    state_names = _feature_names(dataset, data_config.state_key)
    action_names = _feature_names(dataset, data_config.action_key)
    result = materialize_indices(
        dataset.hf_dataset,
        state_key=data_config.state_key,
        state_indices=state_indices,
        action_key=data_config.action_key,
        action_indices=action_indices,
        metadata_path=metadata_path,
        config_sha256=config_sha256,
    )
    dataset.hf_dataset = result.hf_dataset
    logger.info("State reorder indices (zero-based): %s", list(state_indices))
    logger.info(
        "State reorder names: %s",
        _selected_feature_names(data_config.state_key, state_indices, state_names),
    )
    logger.info("Action reorder indices (zero-based): %s", list(action_indices))
    logger.info(
        "Action reorder names: %s",
        _selected_feature_names(data_config.action_key, action_indices, action_names),
    )
    return result.metadata
