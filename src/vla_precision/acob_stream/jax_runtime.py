"""JAX devices, sharding and compiled backends shared by the actor and learner."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from vla_precision.logging import log_status

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeMesh:
    mesh: Any
    data_sharding: Any
    replicated_sharding: Any
    devices: tuple[Any, ...]


def configure_parallel_sharding(pi_train_config, *, devices=None) -> RuntimeMesh:
    """Build the same OpenPI data/FSDP mesh on either role."""
    import jax
    import openpi.training.sharding as pi_sharding

    selected = tuple(jax.local_devices() if devices is None else devices)
    fsdp_devices = int(pi_train_config.fsdp_devices)
    if len(selected) % fsdp_devices != 0:
        raise ValueError(
            f"device count {len(selected)} must be divisible by OpenPI fsdp_devices={fsdp_devices}"
        )
    mesh_shape = (len(selected) // fsdp_devices, fsdp_devices)
    device_mesh = np.asarray(selected, dtype=object).reshape(mesh_shape)
    mesh = jax.sharding.Mesh(device_mesh, (pi_sharding.BATCH_AXIS, pi_sharding.FSDP_AXIS))
    return RuntimeMesh(
        mesh=mesh,
        data_sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(pi_sharding.DATA_AXIS)),
        replicated_sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        devices=selected,
    )


class JaxActorBackend:
    """Small JAX surface kept injectable for hardware-free actor tests."""

    def split_rng(self, rng):
        import jax

        return jax.random.split(rng)

    def device_get(self, value):
        import jax

        return jax.device_get(value)

    def replicate_agent(self, agent, sharding):
        import jax
        import jax.numpy as jnp

        replicated = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding)
        jax.block_until_ready(replicated.state)
        return replicated

    def compile_warmup(self, agent, observation, rng):
        import jax

        rng, key = jax.random.split(rng)
        actions, _, _, _ = agent.sample_actions(observations=observation, seed=key)
        jax.block_until_ready(actions)
        return rng


class JaxLearnerBackend:
    """Mesh and synchronization operations used by the learner loop."""

    def __init__(self):
        self._compiled_updates: dict[tuple[int, frozenset[str]], Any] = {}

    def update(self, runtime: RuntimeMesh, agent, batch, *, networks):
        import jax
        import openpi.training.sharding as pi_sharding

        networks = frozenset(networks)
        cache_key = (id(runtime.mesh), networks)
        update_fn = self._compiled_updates.get(cache_key)
        if update_fn is None:
            def update_fn(current_agent, current_batch):
                return current_agent._update_impl(
                    current_batch,
                    networks_to_update=networks,
                )

            # Keep model/optimizer state replicated and shard only the batch.
            # Explicit output sharding prevents Critic-only and Joint updates
            # from feeding different inferred layouts into one another.
            update_fn = jax.jit(
                update_fn,
                in_shardings=(runtime.replicated_sharding, runtime.data_sharding),
                out_shardings=(runtime.replicated_sharding, runtime.replicated_sharding),
            )
            self._compiled_updates[cache_key] = update_fn

        with pi_sharding.set_mesh(runtime.mesh):
            return update_fn(agent, batch)

    def block_until_ready(self, value):
        import jax

        return jax.block_until_ready(value)


def compile_learner_updates(
    *,
    runtime: RuntimeMesh,
    backend: JaxLearnerBackend,
    agent,
    next_critic_batch,
    next_full_batch,
    critic_networks: frozenset[str],
    all_networks: frozenset[str],
    full_steps: int,
    critic_updates_per_step: int,
):
    """Compile the Critic and Joint update graphs.

    The backend fixes agent input/output sharding, so one call per static graph
    is sufficient. The original agent is returned; compilation never advances
    logical training state.
    """
    temporary_agent = agent

    def run_update(*, batch, networks, label: str):
        nonlocal temporary_agent
        started = time.time()
        log_status(LOGGER, "JIT compilation", f"{label} started", color="magenta")
        temporary_agent, info = backend.update(
            runtime,
            temporary_agent,
            batch,
            networks=networks,
        )
        backend.block_until_ready((temporary_agent.state, info))
        log_status(
            LOGGER,
            "JIT compilation",
            f"{label} finished: {time.time() - started:.2f} s",
            color="magenta",
        )

    def run_full_step(*, label: str):
        for _ in range(max(0, int(critic_updates_per_step) - 1)):
            run_update(
                batch=next_critic_batch(),
                networks=critic_networks,
                label=f"{label} critic",
            )
        run_update(
            batch=next_full_batch(),
            networks=all_networks,
            label=f"{label} joint",
        )

    # Additional warmup steps reuse the same two static graphs.
    for warmup_index in range(max(1, int(full_steps))):
        run_full_step(
            label=f"Online warmup {warmup_index + 1}/{max(1, int(full_steps))}",
        )
    return agent
