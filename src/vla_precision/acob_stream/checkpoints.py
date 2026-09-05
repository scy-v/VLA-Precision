"""Synchronized OpenPI-actor and ACoB-critic checkpoints."""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from flax.training import checkpoints as flax_checkpoints
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.training import checkpoints as openpi_checkpoints
from openpi.training import sharding as openpi_sharding
from openpi.training import utils as training_utils

from vla_precision.config.schema import RootConfig
from vla_precision.data.paths import (
    stage2_checkpoint_dir,
    training_transition_dir,
)
from vla_precision.integrations.openpi.context import make_encode_context_fn, make_sample_actions_fn


class DataConfigProvider:
    def __init__(self, train_config):
        self.train_config = train_config

    def data_config(self):
        return self.train_config.data.create(self.train_config.assets_dirs, self.train_config.model)


@dataclass(frozen=True)
class CheckpointLayout:
    """All Stage-II artifacts for one experiment."""

    root: Path
    actor: Path
    critic: Path
    replay_transitions: Path
    correction_transitions: Path


def checkpoint_layout(config: RootConfig) -> CheckpointLayout:
    # Orbax/TensorStore requires absolute paths for array checkpoints. Resolve
    # once at the layout boundary so actor, critic, and transition artifacts
    # all refer to the same locations regardless of the process working directory.
    root = stage2_checkpoint_dir(config).resolve()
    transitions = training_transition_dir(config).resolve()
    return CheckpointLayout(
        root=root,
        actor=(root / "actor"),
        critic=root / "critic",
        replay_transitions=transitions / "replay_buffer",
        correction_transitions=transitions / "correction_buffer",
    )


def ensure_layout(layout: CheckpointLayout) -> None:
    for path in (
        layout.actor,
        layout.critic,
        layout.replay_transitions,
        layout.correction_transitions,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_fresh_layout(layout: CheckpointLayout, *, overwrite: bool = False) -> None:
    """Refuse to mix a fresh run with existing Stage-II artifacts."""
    roots = (layout.root, layout.replay_transitions, layout.correction_transitions)
    files = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    ]
    if files:
        if overwrite:
            for root in roots:
                if root.exists():
                    shutil.rmtree(root)
            return
        preview = ", ".join(str(path) for path in files[:6])
        raise FileExistsError(
            f"Stage-II checkpoint directory is not empty: {layout.root}. "
            f"Use checkpoint.resume=true or a different experiment/output root. Existing: {preview}"
        )


def prepare_fresh_run_layout(config: RootConfig, layout: CheckpointLayout) -> None:
    """Apply the same fresh-run artifact policy for actor and learner."""
    if config.checkpoint.resume:
        return
    ensure_fresh_layout(layout, overwrite=config.checkpoint.overwrite)
    ensure_layout(layout)


def create_checkpoint_manager(train_config, layout: CheckpointLayout):
    manager, _ = openpi_checkpoints.initialize_checkpoint_dir(
        layout.actor,
        keep_period=train_config.keep_period,
        overwrite=train_config.overwrite,
        resume=False,
    )
    return manager


def open_checkpoint_manager(train_config, layout: CheckpointLayout):
    manager, resuming = openpi_checkpoints.initialize_checkpoint_dir(
        layout.actor,
        keep_period=train_config.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming:
        raise FileNotFoundError(f"Cannot resume: actor checkpoint is empty or missing: {layout.actor}")
    return manager


def checkpoint_steps(manager) -> tuple[int, ...]:
    if manager is None:
        return ()
    return tuple(sorted(int(step) for step in manager.all_steps()))


def latest_checkpoint_step(manager) -> int | None:
    steps = checkpoint_steps(manager)
    return steps[-1] if steps else None


def resolve_resume_step(manager, requested_step: int = 0) -> int:
    steps = checkpoint_steps(manager)
    if not steps:
        raise FileNotFoundError("Cannot resume: actor checkpoint is empty")
    if requested_step == 0:
        return steps[-1]
    if requested_step not in steps:
        raise FileNotFoundError(f"Actor checkpoint step {requested_step} is unavailable; found {list(steps)}")
    return requested_step


def _split_checkpoint_params(state: training_utils.TrainState):
    with at.disable_typechecking():
        if state.ema_params is not None:
            return dataclasses.replace(state, ema_params=None), state.ema_params
        return dataclasses.replace(state, params={}), state.params


def _merge_checkpoint_params(state: training_utils.TrainState, params: dict):
    with at.disable_typechecking():
        if state.params:
            return dataclasses.replace(state, ema_params=params["params"])
        return dataclasses.replace(state, params=params["params"])


def _restore_args_like(target, sharding):
    def restore_arg(value):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                dtype=value.dtype,
                sharding=sharding,
                global_shape=value.shape,
            )
        return ocp.RestoreArgs()

    return jax.tree_util.tree_map(restore_arg, target)


def restore_openpi_state(manager, state, sharding, *, step: int):
    train_state, params = _split_checkpoint_params(state)
    train_state_item = {
        "step": train_state.step,
        "params": train_state.params,
        "model_def": train_state.model_def,
        "opt_state": train_state.opt_state,
        "ema_params": train_state.ema_params,
    }
    with at.disable_typechecking():
        restored = manager.restore(
            step,
            args=ocp.args.Composite(
                train_state=ocp.args.PyTreeRestore(
                    item=train_state_item,
                    restore_args=_restore_args_like(train_state_item, sharding),
                ),
                params=ocp.args.PyTreeRestore(
                    item={"params": params},
                    restore_args=_restore_args_like({"params": params}, sharding),
                ),
            ),
        )
        restored_state = dataclasses.replace(
            train_state,
            step=restored["train_state"]["step"],
            params=restored["train_state"]["params"],
            opt_state=restored["train_state"]["opt_state"],
            ema_params=restored["train_state"]["ema_params"],
        )
        return _merge_checkpoint_params(restored_state, restored["params"])


def restore_actor(agent, manager, *, step: int):
    mesh = openpi_sharding.make_mesh(agent.pi_train_config.fsdp_devices)
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    pi_state = restore_openpi_state(manager, agent.state.pi_state, sharding, step=step)
    return agent.replace(
        state=agent.state.replace(pi_state=pi_state),
        pi_sample_actions=agent.pi_sample_actions or make_sample_actions_fn(pi_state),
        pi_encode_context=agent.pi_encode_context or make_encode_context_fn(pi_state),
    )


def _critic_steps(directory: Path) -> tuple[int, ...]:
    if not directory.exists():
        return ()
    return tuple(sorted(int(path.name) for path in directory.iterdir() if path.name.isdigit()))


def restore_critic(agent, layout: CheckpointLayout, *, step: int):
    checkpoint = layout.critic / str(step)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Critic checkpoint at actor step {step} is missing; available={list(_critic_steps(layout.critic))}"
        )
    critic_state = flax_checkpoints.restore_checkpoint(
        str(layout.critic),
        agent.state.critic_state,
        step=step,
        prefix="",
    )
    return agent.replace(state=agent.state.replace(critic_state=critic_state))


def _zero_all_lora(params):
    return nnx_utils.state_map(
        params,
        nnx_utils.PathRegex(".*lora.*"),
        lambda variable: variable.replace(jnp.zeros_like(variable.value)),
    )


def _zero_lora_b(params):
    return nnx_utils.state_map(
        params,
        nnx_utils.PathRegex(".*lora_b.*"),
        lambda variable: variable.replace(jnp.zeros_like(variable.value)),
    )


def initialize_fresh_lora(agent):
    """Current actor: random A / zero B; frozen reference: zero LoRA."""
    lora = agent.state.pi_state.params.filter(nnx_utils.PathRegex(".*lora.*")).flat_state()
    if not lora:
        return agent
    actor_params = _zero_lora_b(agent.state.pi_state.params)
    reference_params = _zero_all_lora(actor_params)
    pi_state = dataclasses.replace(agent.state.pi_state, params=actor_params)
    return agent.replace(
        state=agent.state.replace(pi_state=pi_state, ref_pi_params=reference_params)
    )


def initialize_resumed_reference(agent, *, full_weight_loader=None):
    """Rebuild the frozen reference policy from Stage-I weights.

    The online actor checkpoint contains the evolving Stage-II LoRA leaves.
    The ACoB reference policy must instead remain the Stage-I policy with all
    LoRA leaves zeroed.  When no explicit Stage-I checkpoint was configured,
    the restored actor's frozen base is the equivalent fallback used by the
    paper implementation.
    """
    if full_weight_loader is None:
        reference_params = agent.state.pi_state.params
    else:
        loaded_params = full_weight_loader.load(agent.state.pi_state.params.to_pure_dict())
        model = nnx.merge(agent.state.pi_state.model_def, agent.state.pi_state.params)
        graph_definition, state = nnx.split(model)
        state.replace_by_pure_dict(loaded_params)
        reference_params = nnx.state(nnx.merge(graph_definition, state))
    return agent.replace(
        state=agent.state.replace(ref_pi_params=_zero_all_lora(reference_params))
    )


def prepare_agent(agent, config: RootConfig, layout: CheckpointLayout):
    """Initialize a fresh run or atomically restore actor and critic at one step."""
    if config.checkpoint.resume:
        manager = open_checkpoint_manager(agent.pi_train_config, layout)
        step = resolve_resume_step(manager, config.checkpoint.resume_step)
        agent = restore_actor(agent, manager, step=step)
        agent = initialize_resumed_reference(
            agent,
            full_weight_loader=(
                agent.pi_train_config.weight_loader
                if config.openpi.initialization_checkpoint
                else None
            ),
        )
        agent = restore_critic(agent, layout, step=step)
        return agent, manager, step

    prepare_fresh_run_layout(config, layout)
    manager = create_checkpoint_manager(agent.pi_train_config, layout)
    return initialize_fresh_lora(agent), manager, None


def _state_for_checkpoint(agent):
    mesh = openpi_sharding.make_mesh(agent.pi_train_config.fsdp_devices)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = agent.state.pi_state
    return dataclasses.replace(
        state,
        step=jax.device_put(state.step, replicated),
        params=jax.device_put(state.params, replicated),
        opt_state=jax.device_put(state.opt_state, replicated),
        ema_params=None if state.ema_params is None else jax.device_put(state.ema_params, replicated),
    )


def save_agent(agent, manager, provider: DataConfigProvider, layout: CheckpointLayout, step: int) -> None:
    """Save actor and critic under the same learner step."""
    step = int(step)
    if step in checkpoint_steps(manager):
        manager.wait_until_finished()
        manager.delete(step)
    critic_step = layout.critic / str(step)
    if critic_step.exists():
        shutil.rmtree(critic_step) if critic_step.is_dir() else critic_step.unlink()

    openpi_checkpoints.save_state(manager, _state_for_checkpoint(agent), provider, step)
    layout.critic.mkdir(parents=True, exist_ok=True)
    flax_checkpoints.save_checkpoint(
        str(layout.critic),
        agent.state.critic_state,
        step=step,
        prefix="",
        keep=100,
        overwrite=True,
    )
