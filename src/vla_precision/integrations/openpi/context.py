"""OpenPI prefix-context encoding and cached action sampling.

The ContextBuffer stores the key/value output of ``embed_prefix``. Both π0 and
π0.5 use that same prefix representation; their different state/time handling
remains inside OpenPI's native ``embed_suffix`` implementation.
"""

from __future__ import annotations

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models import model as openpi_model
from openpi.models.pi0 import make_attn_mask
from openpi.training import utils as training_utils


def embeddings_as_bfloat16(embeddings: jax.Array) -> jax.Array:
    embeddings = jnp.asarray(embeddings)
    if embeddings.dtype == jnp.uint16:
        embeddings = jax.lax.bitcast_convert_type(embeddings, jnp.bfloat16)
    else:
        embeddings = embeddings.astype(jnp.bfloat16)
    return jax.lax.stop_gradient(embeddings)


def encode_context(model: openpi_model.BaseModel, observation: openpi_model.Observation):
    """Encode one OpenPI observation into ContextBuffer fields."""
    observation = openpi_model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, (cache_k, cache_v) = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attention_mask,
        positions=prefix_positions,
    )
    embeddings = jnp.transpose(jnp.stack([cache_k, cache_v], axis=0), (2, 0, 1, 3, 4, 5))
    offsets = jnp.sum(prefix_mask, axis=-1).astype(jnp.int32)
    return (
        jax.lax.stop_gradient(embeddings),
        jax.lax.stop_gradient(prefix_mask),
        jax.lax.stop_gradient(offsets),
    )


def velocity_from_context(
    model: openpi_model.BaseModel,
    observation: openpi_model.Observation,
    embeddings: jax.Array,
    masks: jax.Array,
    offsets: jax.Array,
    noisy_actions: jax.Array,
    time: jax.Array,
) -> jax.Array:
    """Run only the OpenPI suffix against a previously encoded prefix."""
    embeddings = embeddings_as_bfloat16(embeddings)
    prefix_cache = (jnp.swapaxes(embeddings[:, 0], 0, 1), jnp.swapaxes(embeddings[:, 1], 0, 1))
    prefix_mask = jnp.asarray(masks, dtype=jnp.bool_)
    prefix_offsets = jnp.asarray(offsets, dtype=jnp.int32).reshape((-1,))

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, noisy_actions, time
    )
    suffix_attention_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    suffix_to_prefix_mask = jnp.broadcast_to(
        prefix_mask[:, None, :],
        (prefix_mask.shape[0], suffix_tokens.shape[1], prefix_mask.shape[1]),
    )
    attention_mask = jnp.concatenate([suffix_to_prefix_mask, suffix_attention_mask], axis=-1)
    suffix_positions = prefix_offsets[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    _, suffix_output = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=attention_mask,
        positions=suffix_positions,
        kv_cache=prefix_cache,
        adarms_cond=[None, adarms_cond],
    )[0]
    return model.action_out_proj(suffix_output[:, -model.action_horizon :])


def sample_actions_from_context_scan(
    model: openpi_model.BaseModel,
    observation: openpi_model.Observation,
    embeddings: jax.Array,
    masks: jax.Array,
    offsets: jax.Array,
    rng: jax.Array,
    *,
    num_steps: int,
) -> jax.Array:
    """Cached sampler used inside ACoB learner updates (fixed-length scan)."""
    batch_size = embeddings.shape[0]
    noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))
    dt = -1.0 / num_steps

    def step(carry, _):
        actions, time = carry
        velocity = velocity_from_context(
            model,
            observation,
            embeddings,
            masks,
            offsets,
            actions,
            jnp.broadcast_to(time, batch_size),
        )
        return (actions + dt * velocity, time + dt), None

    (actions, _), _ = jax.lax.scan(
        step,
        (noise, jnp.asarray(1.0, dtype=noise.dtype)),
        None,
        length=num_steps,
    )
    return actions


def sample_actions_from_context(
    model: openpi_model.BaseModel,
    observation: openpi_model.Observation,
    embeddings: jax.Array,
    masks: jax.Array,
    offsets: jax.Array,
    rng: jax.Array,
    *,
    num_steps: int,
) -> jax.Array:
    """Cached sampler used by the actor, preserving OpenPI's while-loop path."""
    batch_size = embeddings.shape[0]
    noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))
    dt = -1.0 / num_steps

    def step(carry):
        actions, time = carry
        velocity = velocity_from_context(
            model,
            observation,
            embeddings,
            masks,
            offsets,
            actions,
            jnp.broadcast_to(time, batch_size),
        )
        return actions + dt * velocity, time + dt

    def condition(carry):
        _, time = carry
        return time >= -dt / 2

    actions, _ = jax.lax.while_loop(condition, step, (noise, 1.0))
    return actions


def make_encode_context_fn(pi_state: training_utils.TrainState):
    model_definition = pi_state.model_def

    @jax.jit
    def encode(params, observation):
        model = nnx.merge(model_definition, params)
        model.eval()
        return encode_context(model, observation)

    return encode


def make_sample_actions_fn(pi_state: training_utils.TrainState):
    model_definition = pi_state.model_def

    @functools.partial(jax.jit, static_argnames=("num_steps",))
    def sample(params, rng, observation, *, num_steps):
        model = nnx.merge(model_definition, params)
        model.eval()
        embeddings, masks, offsets = encode_context(model, observation)
        actions = sample_actions_from_context(
            model,
            observation,
            embeddings,
            masks,
            offsets,
            rng,
            num_steps=num_steps,
        )
        return actions, embeddings, masks, offsets

    return sample
