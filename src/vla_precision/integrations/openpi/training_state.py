"""The narrow TrainState construction boundary between ACoB and OpenPI internals."""

from __future__ import annotations

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import jax
import jax.numpy as jnp
from flax import nnx, traverse_util
from openpi.shared import nnx_utils
from openpi.training import config as openpi_config
from openpi.training import optimizer as openpi_optimizer
from openpi.training import utils as training_utils

from vla_precision.logging import debug_record


def initialize_train_state(
    train_config: openpi_config.TrainConfig,
    rng: jax.Array,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> training_utils.TrainState:
    """Initialize OpenPI exactly as Stage I/II training does."""
    debug_record(
        "openpi.init_state",
        (
            f"resume={resume}, checkpoint_dir={train_config.checkpoint_dir}, "
            f"model={train_config.model}, batch_size={train_config.batch_size}, "
            f"save_interval={train_config.save_interval}, weight_loader={train_config.weight_loader}"
        ),
    )
    optimizer = openpi_optimizer.create_optimizer(
        train_config.optimizer,
        train_config.lr_schedule,
        weight_decay_mask=None,
    )

    def initialize(init_rng, partial_params=None):
        _, model_rng = jax.random.split(init_rng)
        model = train_config.model.create(model_rng)
        if partial_params is not None:
            graph_definition, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graph_definition, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            train_config.freeze_filter,
            lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16)),
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=optimizer,
            opt_state=optimizer.init(params.filter(train_config.trainable_filter)),
            ema_decay=train_config.ema_decay,
            ema_params=None if train_config.ema_decay is None else params,
        )

    state_shape = jax.eval_shape(initialize, rng)
    if resume:
        # Orbax restores directly into this shape target. Constructing a full
        # random policy first would add a redundant model allocation/compile.
        return state_shape

    params_shape = state_shape.params.to_pure_dict()
    loaded_params = train_config.weight_loader.load(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)
    partial = traverse_util.unflatten_dict(
        {key: value for key, value in flat_loaded.items() if not isinstance(value, jax.ShapeDtypeStruct)}
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return jax.jit(initialize, in_shardings=replicated, out_shardings=replicated)(rng, partial)
