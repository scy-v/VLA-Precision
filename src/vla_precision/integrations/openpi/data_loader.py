"""OpenPI loader construction with all data columns prepared before sampling."""

from __future__ import annotations

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import logging
import os
from pathlib import Path

import jax
import numpy as np
from openpi import transforms
from openpi.training import data_loader as openpi_data_loader

from vla_precision.config.schema import RootConfig, Stage1Config
from vla_precision.data.indexing import materialize_lerobot_indices
from vla_precision.data.paths import source_lerobot_root

logger = logging.getLogger(__name__)


class TransformedDataset:
    """OpenPI-equivalent transform wrapper owned by the spawn-safe extension."""

    def __init__(self, dataset, transform_fns):
        self._dataset = dataset
        self._transform = transforms.compose(transform_fns)

    def __getitem__(self, index):
        return self._transform(self._dataset[index])

    def __len__(self):
        return len(self._dataset)


def _collate_fn(items):
    """Match OpenPI's NumPy collation without importing its loader in workers."""
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0),
        *items,
    )


def _worker_init_fn(worker_id: int) -> None:
    """Match OpenPI's worker-side JAX allocator setup."""
    del worker_id
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class TorchDataLoader(openpi_data_loader.TorchDataLoader):
    """Keep OpenPI's loader behavior with extension-owned spawn callbacks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_loader.collate_fn = _collate_fn
        self.torch_loader.worker_init_fn = _worker_init_fn


def transform_dataset(dataset, data_config, *, skip_norm_stats: bool = False):
    """Apply OpenPI's transform order with a spawn-safe dataset wrapper."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found. Run Stage-I norm-stats first.")
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def _lerobot_dataset(dataset):
    """Reach the LeRobot owner before OpenPI's optional prompt wrapper."""
    current = dataset
    while not hasattr(current, "hf_dataset") and hasattr(current, "_dataset"):
        current = current._dataset
    return current


def create_torch_dataset(
    data_config,
    action_horizon: int,
    model_config,
    root_config: RootConfig | Stage1Config,
):
    """Mirror OpenPI's LeRobot constructor while honoring the top-level root."""
    if data_config.repo_id == "fake" or root_config.data.lerobot_root is None:
        return openpi_data_loader.create_torch_dataset(
            data_config,
            action_horizon,
            model_config,
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    root = source_lerobot_root(root_config)
    metadata = LeRobotDatasetMetadata(data_config.repo_id, root=root)
    dataset = LeRobotDataset(
        data_config.repo_id,
        root=root,
        delta_timestamps={
            key: [step / metadata.fps for step in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset,
            [transforms.PromptFromLeRobotTask(metadata.tasks)],
        )
    return dataset


def create_data_loader(
    train_config,
    root_config: RootConfig | Stage1Config,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
):
    """Create the standard OpenPI loader after one eager HF ``map`` pass."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.rlds_data_dir is not None or data_config.repo_id == "fake":
        return openpi_data_loader.create_data_loader(
            train_config,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework="jax",
        )

    dataset = create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
        root_config,
    )
    experiment_name = (
        root_config.openpi.exp_name
        if isinstance(root_config, Stage1Config)
        else root_config.experiment.name
    )
    metadata_path = (
        Path(root_config.paths.cache_root)
        / "index_materialization"
        / experiment_name
        / "metadata.json"
    )
    metadata = materialize_lerobot_indices(
        _lerobot_dataset(dataset),
        root_config.data,
        metadata_path=metadata_path,
    )
    logger.info("materialized OpenPI training columns: schema_sha256=%s", metadata.schema_sha256)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = train_config.batch_size // jax.process_count()
    loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        sampler=None,
        num_batches=num_batches,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
        framework="jax",
    )
    return openpi_data_loader.DataLoaderImpl(data_config, loader)
