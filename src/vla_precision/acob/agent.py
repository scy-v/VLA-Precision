"""ACoB agent, loss functions and state construction.

The numerical update path is migrated from the paper implementation. OpenPI
specific conversion and cached-context operations are kept behind the OpenPI
integration package; generic RL building blocks live with ACoB.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.gemma as _gemma
import openpi.models.model as _model
import openpi.training.config as _pi_config
import openpi.training.sharding as pi_sharding
import openpi.training.utils as pi_training_utils
import optax
from flax import nnx

from vla_precision.acob.augmentation import make_batch_augmentation
from vla_precision.acob.networks import MLP, ValueAdvantageCritic, ensemblize
from vla_precision.acob.pretrained_resnet import (
    DEFAULT_CRITIC_RESNET10_PARAMS_PATH,
    load_pretrained_resnet10_critic_params,
)
from vla_precision.acob.training import (
    JaxRLTrainState,
    ModuleDict,
    make_optimizer,
    nonpytree_field,
    unpack_memory_efficient_batch,
)
from vla_precision.acob.types import Batch, Data, Params, PRNGKey
from vla_precision.acob.vision import EncodingWrapper
from vla_precision.acob_stream.buffers import ContextSource
from vla_precision.integrations.openpi.adapter import OpenPIObservationAdapter
from vla_precision.integrations.openpi.context import (
    make_encode_context_fn,
    make_sample_actions_fn,
    sample_actions_from_context_scan,
    velocity_from_context,
)
from vla_precision.integrations.openpi.training_state import initialize_train_state
from vla_precision.logging import debug_record


def _normalize_actor_weights(flow_weight: float, imp_weight: float, ref_weight: float) -> tuple[float, float, float]:
    weights = np.asarray([flow_weight, imp_weight, ref_weight], dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"Actor loss weights must be finite, got {weights.tolist()}")
    if np.any(weights < 0.0):
        raise ValueError(f"Actor loss weights must be non-negative, got {weights.tolist()}")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("At least one actor loss weight must be positive.")
    weights = weights / total
    return tuple(float(weight) for weight in weights)


def _validate_nonnegative_weight(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def _preprocess_rl_sample_weights(batch, *, preprocess_bc_only: bool, dtype=jnp.float32) -> jax.Array:
    """Return one RL weight per row, masking only preprocess KV-store rows."""
    batch_size = int(batch["rewards"].shape[0])
    if not preprocess_bc_only:
        return jnp.ones((batch_size,), dtype=dtype)
    if "context_source" not in batch:
        raise KeyError(
            "preprocess_bc_only=True requires context_source in every learner batch; data-source masking cannot be applied."
        )
    store_ids = jnp.asarray(batch["context_source"]).reshape(batch_size, -1)[:, 0]
    return (store_ids != int(ContextSource.PREPROCESS)).astype(dtype)


def _weighted_sample_mean(values, sample_weights, *, batch_axis: int) -> jax.Array:
    """Mean over all elements while assigning one weight to each batch row."""
    values = jnp.asarray(values)
    sample_weights = jnp.asarray(sample_weights, dtype=values.dtype).reshape(-1)
    batch_axis = batch_axis % values.ndim
    if values.shape[batch_axis] != sample_weights.shape[0]:
        raise ValueError(
            f"Batch axis size {values.shape[batch_axis]} does not match sample weights {sample_weights.shape[0]}."
        )
    broadcast_shape = [1] * values.ndim
    broadcast_shape[batch_axis] = sample_weights.shape[0]
    broadcast_weights = sample_weights.reshape(broadcast_shape)
    elements_per_sample = values.size // sample_weights.shape[0]
    denominator = jnp.maximum(jnp.sum(sample_weights) * elements_per_sample, 1.0)
    return jnp.sum(values * broadcast_weights) / denominator


def _weighted_sample_std(values, sample_weights, *, batch_axis: int) -> jax.Array:
    mean = _weighted_sample_mean(values, sample_weights, batch_axis=batch_axis)
    variance = _weighted_sample_mean(
        jnp.square(jnp.asarray(values) - mean),
        sample_weights,
        batch_axis=batch_axis,
    )
    return jnp.sqrt(jnp.maximum(variance, 0.0))


def _validate_ablation_flags(
    *,
    ablate_critic_pref: bool,
    ablate_actor_bc: bool,
    ablate_actor_advantage: bool,
) -> tuple[bool, bool, bool]:
    flags = {
        "ablate_critic_pref": bool(ablate_critic_pref),
        "ablate_actor_bc": bool(ablate_actor_bc),
        "ablate_actor_advantage": bool(ablate_actor_advantage),
    }
    enabled = [name for name, value in flags.items() if value]
    if len(enabled) > 1:
        raise ValueError(f"At most one ACoB ablation flag can be True, got {enabled}")
    return (
        flags["ablate_critic_pref"],
        flags["ablate_actor_bc"],
        flags["ablate_actor_advantage"],
    )


def _paired_pessimistic_ensemble_delta(
    sampled_ensemble: jax.Array,
    reference_ensemble: jax.Array,
    bad_ensemble: jax.Array,
    bad_baseline_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Subtract baselines within each critic before ensemble aggregation."""
    mask = jnp.asarray(bad_baseline_mask)[None, :] > 0.0
    baseline_ensemble = jnp.where(
        mask,
        jnp.maximum(reference_ensemble, bad_ensemble),
        reference_ensemble,
    )
    delta_ensemble = sampled_ensemble - baseline_ensemble
    return delta_ensemble.min(axis=0), delta_ensemble.mean(axis=0), delta_ensemble


def _paired_pessimistic_ensemble_margin(
    good_ensemble: jax.Array,
    bad_ensemble: jax.Array,
) -> jax.Array:
    """Subtract paired critic outputs before taking the pessimistic margin."""
    return (good_ensemble - bad_ensemble).min(axis=0)


class ACoBState(flax.struct.PyTreeNode):
    critic_state: JaxRLTrainState
    pi_state: pi_training_utils.TrainState
    ref_pi_params: Any

    @property
    def params(self):
        return {"critic": self.critic_state.params, "pi": self.pi_state.params}

    @property
    def target_params(self):
        return {"critic": self.critic_state.target_params, "pi": self.pi_state.params}

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


class ACoBAgent(flax.struct.PyTreeNode):
    state: ACoBState
    algo_config: dict = nonpytree_field()
    adapter: OpenPIObservationAdapter = nonpytree_field()
    pi_train_config: _pi_config.TrainConfig = nonpytree_field()
    pi_sample_actions: Any = nonpytree_field(default=None)
    pi_encode_context: Any = nonpytree_field(default=None)

    def _pi_model(self, params=None):
        return nnx.merge(self.state.pi_state.model_def, params or self.state.pi_state.params)

    def _ref_pi_model(self):
        return nnx.merge(self.state.pi_state.model_def, self.state.ref_pi_params)

    def forward_critic_components(
        self, observations: Data, actions: jax.Array, rng: PRNGKey, *, grad_params=None, train=True
    ):
        return self.state.critic_state.apply_fn(
            {"params": grad_params or self.state.critic_state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
            return_components=True,
        )

    def forward_critic(self, observations: Data, actions: jax.Array, rng: PRNGKey, *, grad_params=None, train=True):
        return self.forward_critic_components(
            observations,
            actions,
            rng,
            grad_params=grad_params,
            train=train,
        )["q"]

    def forward_target_critic(self, observations: Data, actions: jax.Array, rng: PRNGKey):
        return self.forward_critic(observations, actions, rng, grad_params=self.state.critic_state.target_params)

    def _flatten_critic_action_chunk(self, action_chunk: jax.Array) -> jnp.ndarray:
        action_chunk = jnp.asarray(action_chunk, dtype=jnp.float32)
        single_dim = int(self.algo_config["single_action_dim"])
        critic_dim = int(self.algo_config.get("critic_action_dim", self.algo_config["action_dim"]))
        if action_chunk.shape[-1] == critic_dim:
            return action_chunk
        if action_chunk.ndim >= 3:
            body_indices = tuple(self.algo_config.get("critic_action_indices", tuple(range(single_dim))))
            if int(action_chunk.shape[-1]) != len(body_indices):
                action_chunk = action_chunk[..., jnp.asarray(body_indices)]
            return action_chunk.reshape(*action_chunk.shape[:-2], -1)
        return action_chunk[..., :critic_dim]

    def _env_actions_to_critic_actions(self, actions: jax.Array, observations: Data) -> jnp.ndarray:
        # Replay/env actions are stored in robot action space.  Normalize them
        # with the same OpenPI action transform used by the flow-matching loss
        # before feeding them to the critic.
        normalized_chunk = self.adapter.action_chunk(actions, observations)
        return self._flatten_critic_action_chunk(normalized_chunk)

    def _current_context_from_batch(self, batch):
        return batch["context_embeddings"], batch["context_masks"], batch["context_offsets"]

    def _next_context_from_batch(self, batch):
        return batch["next_context_embeddings"], batch["next_context_masks"], batch["next_context_offsets"]

    def _pi_velocity_from_context(
        self,
        model: _model.BaseModel,
        pi_obs: _model.Observation,
        embeddings: jnp.ndarray,
        context_masks: jnp.ndarray,
        context_offsets: jnp.ndarray,
        x_t: jnp.ndarray,
        time: jnp.ndarray,
    ) -> jnp.ndarray:
        return velocity_from_context(
            model,
            pi_obs,
            embeddings,
            context_masks,
            context_offsets,
            x_t,
            time,
        )

    def _sample_pi_actions_from_context(
        self,
        model: _model.BaseModel,
        pi_obs: _model.Observation,
        embeddings: jnp.ndarray,
        context_masks: jnp.ndarray,
        context_offsets: jnp.ndarray,
        rng: PRNGKey,
    ) -> jnp.ndarray:
        return sample_actions_from_context_scan(
            model,
            pi_obs,
            embeddings,
            context_masks,
            context_offsets,
            rng,
            num_steps=int(self.algo_config["pi_sample_steps"]),
        )

    def _pi_flow_loss_from_context(
        self,
        model: _model.BaseModel,
        pi_obs: _model.Observation,
        embeddings: jnp.ndarray,
        context_masks: jnp.ndarray,
        context_offsets: jnp.ndarray,
        action_chunk: _model.Actions,
        rng: PRNGKey,
        *,
        loss_horizon: int | None = None,
    ) -> jnp.ndarray:
        _, noise_rng, time_rng = jax.random.split(rng, 3)
        batch_shape = action_chunk.shape[:-2]
        noise = jax.random.normal(noise_rng, action_chunk.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * action_chunk
        u_t = noise - action_chunk
        v_t = self._pi_velocity_from_context(model, pi_obs, embeddings, context_masks, context_offsets, x_t, time)
        per_step_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if loss_horizon is not None:
            per_step_loss = per_step_loss[..., :loss_horizon]
        return jnp.mean(per_step_loss, axis=-1)

    def _action_chunk_for_model_horizon(self, action_chunk: jax.Array, model_action_horizon: int) -> jnp.ndarray:
        current_horizon = int(action_chunk.shape[-2])
        if current_horizon == model_action_horizon:
            return action_chunk
        if current_horizon > model_action_horizon:
            return action_chunk[..., :model_action_horizon, :]
        pad_count = model_action_horizon - current_horizon
        pad = jnp.repeat(action_chunk[..., -1:, :], pad_count, axis=-2)
        return jnp.concatenate([action_chunk, pad], axis=-2)

    def _model_action_chunk_to_critic_actions(
        self, action_chunk: jax.Array, observations: Data | None = None
    ) -> jnp.ndarray:
        del observations
        action_chunk = action_chunk[..., : self.algo_config["action_horizon"], :]
        return self._flatten_critic_action_chunk(action_chunk)

    def _safe_critic_components(self, observations: Data, actions: jax.Array, rng: PRNGKey):
        outputs = self.forward_critic_components(observations, actions, rng=rng)
        q_values = outputs["q"]
        advantage_values = outputs["advantage"]
        return {
            "q_min": q_values.min(axis=0),
            "q_ensemble": q_values,
            "advantage_min": advantage_values.min(axis=0),
            "advantage_ensemble": advantage_values,
        }

    def _sample_vla_cons_actions(self, batch, rng: PRNGKey):
        if not self.algo_config.get("use_context", False):
            raise NotImplementedError("ACoB VLA conservative critic loss requires cached OpenPI KV embeddings.")
        pi_obs = self.adapter.batch(batch["observations"])
        current_context_embeddings, current_masks, current_offsets = self._current_context_from_batch(batch)

        model = self._pi_model(self.state.pi_state.params)
        model.eval()
        lora_action_chunk = self._sample_pi_actions_from_context(
            model,
            pi_obs,
            current_context_embeddings,
            current_masks,
            current_offsets,
            rng,
        )
        lora_actions = self._model_action_chunk_to_critic_actions(lora_action_chunk, batch["observations"])

        ref_model = self._ref_pi_model()
        ref_model.eval()
        ref_action_chunk = self._sample_pi_actions_from_context(
            ref_model,
            pi_obs,
            current_context_embeddings,
            current_masks,
            current_offsets,
            rng,
        )
        ref_actions = self._model_action_chunk_to_critic_actions(ref_action_chunk, batch["observations"])
        return {
            "lora": jax.lax.stop_gradient(lora_actions),
            "ref": jax.lax.stop_gradient(ref_actions),
        }

    def _compute_next_actions(self, batch, rng):
        if self.algo_config.get("use_context", False):
            pi_obs = self.adapter.batch(batch["next_observations"])
            model = self._pi_model(self.state.pi_state.params)
            model.eval()
            next_context_embeddings, next_masks, next_offsets = self._next_context_from_batch(batch)
            actions = self._sample_pi_actions_from_context(
                model,
                pi_obs,
                next_context_embeddings,
                next_masks,
                next_offsets,
                rng,
            )
        else:
            raise NotImplementedError("ACoB learner requires cached OpenPI KV embeddings for next-action sampling.")
        return self._model_action_chunk_to_critic_actions(actions)

    def _vla_conservative_critic_loss(
        self,
        batch,
        params: Params,
        rng: PRNGKey,
        *,
        predicted_batch_qs: jax.Array,
        lora_actions: jax.Array,
        ref_actions: jax.Array,
        rl_sample_weights: jax.Array,
    ):
        lora_actions = jax.lax.stop_gradient(lora_actions)
        ref_actions = jax.lax.stop_gradient(ref_actions)
        cons_actions = jnp.stack([lora_actions, ref_actions], axis=1)
        cons_qs = self.forward_critic(batch["observations"], cons_actions, rng, grad_params=params)
        q_lora = cons_qs[:, :, 0]
        q_ref = cons_qs[:, :, 1]
        baseline = jax.lax.stop_gradient(jnp.maximum(predicted_batch_qs, q_ref))
        delta_q = q_lora - baseline
        violation = jax.nn.relu(delta_q)
        loss = _weighted_sample_mean(jnp.square(violation), rl_sample_weights, batch_axis=1)
        return loss, {
            "vla_cons_loss": loss,
            "vla_cons_delta_q": _weighted_sample_mean(delta_q, rl_sample_weights, batch_axis=1),
            "vla_cons_violation_rate": _weighted_sample_mean(
                (delta_q > 0.0).astype(delta_q.dtype),
                rl_sample_weights,
                batch_axis=1,
            ),
        }

    def _intervention_preference_critic_loss(
        self,
        batch,
        params: Params,
        rng: PRNGKey,
        *,
        predicted_good_qs: jax.Array,
        predicted_good_advantages: jax.Array,
        rl_sample_weights: jax.Array,
    ):
        if "intervention_bad_actions" not in batch or "intervened" not in batch:
            zero = jnp.asarray(0.0, dtype=predicted_good_qs.dtype)
            return zero, {
                "intervention_pref_loss": zero,
                "intervention_pref_margin": zero,
                "intervention_pref_accuracy": zero,
                "intervention_pref_margin_satisfied_rate": zero,
                "intervention_pref_target_margin": jnp.asarray(
                    self.algo_config.get("critic_intervention_pref_margin", 0.0), dtype=predicted_good_qs.dtype
                ),
                "intervention_pref_pair_count": zero,
                "intervention_pref_raw_pair_count": zero,
                "intervention_action_l2": zero,
                "intervention_action_diff_rate": zero,
                "intervention_q_good": zero,
                "intervention_q_bad": zero,
                "intervention_adv_good": zero,
                "intervention_adv_bad": zero,
            }
        good_actions = self._env_actions_to_critic_actions(batch["actions"], batch["observations"])
        bad_actions = self._env_actions_to_critic_actions(batch["intervention_bad_actions"], batch["observations"])
        bad_outputs = self.forward_critic_components(batch["observations"], bad_actions, rng, grad_params=params)
        bad_qs = bad_outputs["q"]
        bad_advantages = bad_outputs["advantage"]
        mask = jnp.asarray(batch["intervened"], dtype=predicted_good_qs.dtype)
        rl_sample_weights = jnp.asarray(rl_sample_weights, dtype=mask.dtype)
        raw_pair_count = jnp.sum(mask * rl_sample_weights)
        action_l2 = jnp.linalg.norm(good_actions - bad_actions, axis=-1)
        valid_mask = mask * rl_sample_weights * (action_l2 > 1e-6).astype(mask.dtype)
        pair_count = jnp.sum(valid_mask)
        denom = jnp.maximum(pair_count, 1.0)
        margin_advantages = predicted_good_advantages - bad_advantages
        target_margin = jnp.asarray(
            self.algo_config.get("critic_intervention_pref_margin", 0.0), dtype=margin_advantages.dtype
        )
        violation = jax.nn.relu(target_margin - margin_advantages)
        per_sample_loss = jnp.mean(jnp.square(violation), axis=0)
        loss = jnp.sum(per_sample_loss * valid_mask) / denom

        good_min_advantage = jnp.min(predicted_good_advantages, axis=0)
        bad_min_advantage = jnp.min(bad_advantages, axis=0)
        min_advantage_margin = _paired_pessimistic_ensemble_margin(
            predicted_good_advantages,
            bad_advantages,
        )
        pref_accuracy = jnp.sum((min_advantage_margin > 0.0).astype(mask.dtype) * valid_mask) / denom
        good_min_q = jnp.min(predicted_good_qs, axis=0)
        bad_min_q = jnp.min(bad_qs, axis=0)
        raw_denom = jnp.maximum(raw_pair_count, 1.0)
        return loss, {
            "intervention_pref_loss": loss,
            "intervention_pref_margin": jnp.sum(min_advantage_margin * valid_mask) / denom,
            "intervention_pref_accuracy": pref_accuracy,
            "intervention_pref_margin_satisfied_rate": jnp.sum(
                (min_advantage_margin >= target_margin).astype(mask.dtype) * valid_mask
            )
            / denom,
            "intervention_pref_target_margin": target_margin,
            "intervention_pref_pair_count": pair_count,
            "intervention_pref_raw_pair_count": raw_pair_count,
            "intervention_action_l2": jnp.sum(action_l2 * mask * rl_sample_weights) / raw_denom,
            "intervention_action_diff_rate": jnp.sum((action_l2 > 1e-6).astype(mask.dtype) * mask * rl_sample_weights)
            / raw_denom,
            "intervention_q_good": jnp.sum(good_min_q * valid_mask) / denom,
            "intervention_q_bad": jnp.sum(bad_min_q * valid_mask) / denom,
            "intervention_adv_good": jnp.sum(good_min_advantage * valid_mask) / denom,
            "intervention_adv_bad": jnp.sum(bad_min_advantage * valid_mask) / denom,
        }

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey, *, vla_cons_actions=None):
        batch_size = batch["rewards"].shape[0]
        rl_sample_weights = _preprocess_rl_sample_weights(
            batch,
            preprocess_bc_only=bool(self.algo_config.get("preprocess_bc_only", False)),
            dtype=jnp.float32,
        )
        actions = self._env_actions_to_critic_actions(batch["actions"], batch["observations"])
        rng, next_action_rng = jax.random.split(rng)
        next_actions = self._compute_next_actions(batch, next_action_rng)
        target_next_qs = self.forward_target_critic(batch["next_observations"], next_actions, rng=rng)
        if self.algo_config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            target_next_qs = target_next_qs[
                jax.random.randint(
                    subsample_key,
                    (self.algo_config["critic_subsample_size"],),
                    0,
                    self.algo_config["critic_ensemble_size"],
                )
            ]
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))
        rewards = batch["rewards"]
        if rewards.ndim == 1:
            chunk_return = rewards
            bootstrap_discount = self.algo_config["discount"]
        else:
            discount_powers = self.algo_config["discount"] ** jnp.arange(rewards.shape[-1], dtype=rewards.dtype)
            chunk_return = jnp.sum(rewards * discount_powers, axis=-1)
            bootstrap_discount = self.algo_config["discount"] ** rewards.shape[-1]
        target_q = chunk_return + bootstrap_discount * batch["masks"] * target_next_min_q

        predicted_outputs = self.forward_critic_components(batch["observations"], actions, rng=rng, grad_params=params)
        predicted_qs = predicted_outputs["q"]
        predicted_advantages = predicted_outputs["advantage"]
        predicted_values = predicted_outputs["value"]
        chex.assert_shape(predicted_qs, (self.algo_config["critic_ensemble_size"], batch_size))
        target_qs = target_q[None].repeat(self.algo_config["critic_ensemble_size"], axis=0)
        td_loss = _weighted_sample_mean(
            jnp.square(predicted_qs - target_qs),
            rl_sample_weights,
            batch_axis=1,
        )
        critic_loss = td_loss
        info = {
            "critic_loss": critic_loss,
            "td_loss": td_loss,
            "predicted_qs": _weighted_sample_mean(predicted_qs, rl_sample_weights, batch_axis=1),
            "predicted_values": _weighted_sample_mean(predicted_values, rl_sample_weights, batch_axis=1),
            "predicted_advantages": _weighted_sample_mean(predicted_advantages, rl_sample_weights, batch_axis=1),
            "target_qs": _weighted_sample_mean(target_qs, rl_sample_weights, batch_axis=1),
            "batch_reward_mean": _weighted_sample_mean(rewards, rl_sample_weights, batch_axis=0),
            "chunk_reward": _weighted_sample_mean(chunk_return, rl_sample_weights, batch_axis=0),
            "critic_rl_sample_weight_mean": jnp.mean(rl_sample_weights),
            "critic_preprocess_sample_rate": 1.0 - jnp.mean(rl_sample_weights),
            "ablate_critic_pref": jnp.asarray(self.algo_config.get("ablate_critic_pref", False), dtype=td_loss.dtype),
        }
        if vla_cons_actions is not None and self.algo_config["critic_vla_cons_weight"] > 0.0:
            cons_loss, cons_info = self._vla_conservative_critic_loss(
                batch,
                params,
                rng,
                predicted_batch_qs=predicted_qs,
                lora_actions=vla_cons_actions["lora"],
                ref_actions=vla_cons_actions["ref"],
                rl_sample_weights=rl_sample_weights,
            )
            cons_weight = (
                self.algo_config["critic_vla_cons_weight"] * self.algo_config["critic_vla_cons_frequency_scale"]
            )
            critic_loss = critic_loss + cons_weight * cons_loss
            info.update(cons_info)
            info.update(
                {
                    "critic_loss": critic_loss,
                    "vla_cons_effective_weight": cons_weight,
                }
            )
        pref_weight = self.algo_config.get("critic_intervention_pref_weight", 0.0)
        if self.algo_config.get("ablate_critic_pref", False):
            pref_weight = 0.0
        if pref_weight > 0.0:
            pref_loss, pref_info = self._intervention_preference_critic_loss(
                batch,
                params,
                rng,
                predicted_good_qs=predicted_qs,
                predicted_good_advantages=predicted_advantages,
                rl_sample_weights=rl_sample_weights,
            )
            critic_loss = critic_loss + pref_weight * pref_loss
            info.update(pref_info)
            info.update(
                {
                    "critic_loss": critic_loss,
                    "intervention_pref_weight": pref_weight,
                }
            )
        return critic_loss, info

    def policy_loss_fn(self, batch, model: _model.BaseModel, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        pi_obs = self.adapter.batch(batch["observations"])
        action_chunk = self.adapter.action_chunk(batch["actions"], batch["observations"])
        rng, flow_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        ablate_actor_bc = bool(self.algo_config.get("ablate_actor_bc", False))
        if self.algo_config.get("use_context", False):
            current_context_embeddings, current_masks, current_offsets = self._current_context_from_batch(batch)
            if ablate_actor_bc:
                bc_loss_per_sample = jnp.zeros((batch_size,), dtype=action_chunk.dtype)
                success_bc_weights = jnp.zeros_like(bc_loss_per_sample)
                intervention_bc_weights = jnp.zeros_like(bc_loss_per_sample)
                bc_weights = jnp.zeros_like(bc_loss_per_sample)
                bc_loss = jnp.asarray(0.0, dtype=action_chunk.dtype)
            else:
                model.train()
                flow_action_chunk = self._action_chunk_for_model_horizon(action_chunk, int(model.action_horizon))
                bc_loss_per_sample = self._pi_flow_loss_from_context(
                    model,
                    pi_obs,
                    current_context_embeddings,
                    current_masks,
                    current_offsets,
                    flow_action_chunk,
                    flow_rng,
                    loss_horizon=int(self.algo_config["action_horizon"]),
                )
                success_bc_weights = jnp.ones_like(bc_loss_per_sample)
                if "episode_succeed" in batch:
                    success_bc_weights = jnp.asarray(batch["episode_succeed"], dtype=bc_loss_per_sample.dtype)
                intervention_bc_weights = jnp.zeros_like(bc_loss_per_sample)
                if "intervened" in batch:
                    intervention_bc_weights = jnp.asarray(batch["intervened"], dtype=bc_loss_per_sample.dtype)
                bc_weights = jnp.maximum(success_bc_weights, intervention_bc_weights)
                bc_loss = jnp.sum(bc_loss_per_sample * bc_weights) / jnp.maximum(jnp.sum(bc_weights), 1.0)
        else:
            raise NotImplementedError("ACoB actor loss requires cached OpenPI KV embeddings.")

        model.eval()
        sampled_action_chunk = self._sample_pi_actions_from_context(
            model,
            pi_obs,
            current_context_embeddings,
            current_masks,
            current_offsets,
            sample_rng,
        )
        q_actions = self._model_action_chunk_to_critic_actions(sampled_action_chunk, batch["observations"])

        ref_model = self._ref_pi_model()
        ref_model.eval()
        ref_action_chunk = self._sample_pi_actions_from_context(
            ref_model,
            pi_obs,
            current_context_embeddings,
            current_masks,
            current_offsets,
            sample_rng,
        )
        ref_q_actions = jax.lax.stop_gradient(
            self._model_action_chunk_to_critic_actions(ref_action_chunk, batch["observations"])
        )
        batch_q_actions = jax.lax.stop_gradient(
            self._env_actions_to_critic_actions(batch["actions"], batch["observations"])
        )

        sampled_critic = self._safe_critic_components(batch["observations"], q_actions, rng=critic_rng)
        ref_critic = self._safe_critic_components(batch["observations"], ref_q_actions, rng=critic_rng)
        batch_critic = self._safe_critic_components(batch["observations"], batch_q_actions, rng=critic_rng)
        q_sampled_actions = sampled_critic["q_min"]
        q_sampled_ensemble = sampled_critic["q_ensemble"]
        advantage_sampled_actions = sampled_critic["advantage_min"]
        advantage_sampled_ensemble = sampled_critic["advantage_ensemble"]
        q_ref_actions = jax.lax.stop_gradient(ref_critic["q_min"])
        q_ref_ensemble = jax.lax.stop_gradient(ref_critic["q_ensemble"])
        advantage_ref_actions = jax.lax.stop_gradient(ref_critic["advantage_min"])
        advantage_ref_ensemble = jax.lax.stop_gradient(ref_critic["advantage_ensemble"])
        q_batch_actions = jax.lax.stop_gradient(batch_critic["q_min"])
        advantage_batch_actions = jax.lax.stop_gradient(batch_critic["advantage_min"])

        q_bad_actions = q_ref_actions
        q_bad_ensemble = jax.lax.stop_gradient(q_ref_ensemble)
        advantage_bad_actions = advantage_ref_actions
        advantage_bad_ensemble = jax.lax.stop_gradient(advantage_ref_ensemble)
        imp_bad_baseline_mask = jnp.zeros_like(q_ref_actions)
        if "intervention_bad_actions" in batch and "intervened" in batch:
            bad_q_actions = jax.lax.stop_gradient(
                self._env_actions_to_critic_actions(batch["intervention_bad_actions"], batch["observations"])
            )
            bad_critic = self._safe_critic_components(batch["observations"], bad_q_actions, rng=critic_rng)
            q_bad_actions = jax.lax.stop_gradient(bad_critic["q_min"])
            q_bad_ensemble = jax.lax.stop_gradient(bad_critic["q_ensemble"])
            advantage_bad_actions = jax.lax.stop_gradient(bad_critic["advantage_min"])
            advantage_bad_ensemble = jax.lax.stop_gradient(bad_critic["advantage_ensemble"])
            intervention_mask = jnp.asarray(batch["intervened"], dtype=q_ref_actions.dtype)
            bad_action_l2 = jnp.linalg.norm(
                jax.lax.stop_gradient(bad_q_actions - batch_q_actions).reshape(batch_size, -1),
                axis=-1,
            )
            imp_bad_baseline_mask = intervention_mask * (bad_action_l2 > 1e-6).astype(intervention_mask.dtype)

        local_delta_q, _, _ = _paired_pessimistic_ensemble_delta(
            q_sampled_ensemble,
            q_ref_ensemble,
            q_bad_ensemble,
            imp_bad_baseline_mask,
        )
        local_delta_advantage, _, _ = _paired_pessimistic_ensemble_delta(
            advantage_sampled_ensemble,
            advantage_ref_ensemble,
            advantage_bad_ensemble,
            imp_bad_baseline_mask,
        )
        actor_imp_margin = jnp.asarray(self.algo_config.get("actor_imp_margin", 0.0), dtype=local_delta_advantage.dtype)
        ablate_critic_pref = bool(self.algo_config.get("ablate_critic_pref", False))
        imp_delta_for_loss = local_delta_q if ablate_critic_pref else local_delta_advantage
        rl_sample_weights = _preprocess_rl_sample_weights(
            batch,
            preprocess_bc_only=bool(self.algo_config.get("preprocess_bc_only", False)),
            dtype=imp_delta_for_loss.dtype,
        )
        imp_q_scale = jnp.asarray(self.algo_config.get("actor_imp_tau", 0.05), dtype=imp_delta_for_loss.dtype)
        ablate_actor_advantage = bool(self.algo_config.get("ablate_actor_advantage", False))
        if ablate_actor_advantage:
            imp_loss_per_sample = -q_sampled_actions
        elif self.algo_config.get("actor_imp_direct_delta", False):
            imp_loss_per_sample = -imp_delta_for_loss
        else:
            imp_loss_per_sample = imp_q_scale * jax.nn.softplus((actor_imp_margin - imp_delta_for_loss) / imp_q_scale)
        imp_loss = _weighted_sample_mean(imp_loss_per_sample, rl_sample_weights, batch_axis=0)

        if self.algo_config["actor_ref_weight"] > 0.0:
            norm_action_horizon = self.algo_config["action_horizon"]
            ref_indices = jnp.asarray(
                self.algo_config.get("body_action_indices", tuple(range(self.algo_config["single_action_dim"])))
            )
            sampled_norm_actions = sampled_action_chunk[:, :norm_action_horizon, ref_indices]
            ref_norm_actions = jax.lax.stop_gradient(ref_action_chunk[:, :norm_action_horizon, ref_indices])
            ref_loss_per_sample = jnp.mean(jnp.square(sampled_norm_actions - ref_norm_actions), axis=(-2, -1))
            ref_weights = jnp.ones_like(ref_loss_per_sample)
            ref_loss = ref_loss_per_sample.mean()
        else:
            ref_weights = jnp.zeros_like(bc_loss_per_sample)
            ref_loss = jnp.asarray(0.0, dtype=bc_loss.dtype)

        flow_term = self.algo_config["actor_flow_weight"] * bc_loss
        imp_term = self.algo_config["actor_imp_weight"] * imp_loss
        ref_term = self.algo_config["actor_ref_weight"] * ref_loss
        actor_loss = flow_term + imp_term + ref_term
        q_value = _weighted_sample_mean(q_sampled_actions, rl_sample_weights, batch_axis=0)
        q_ref_value = _weighted_sample_mean(q_ref_actions, rl_sample_weights, batch_axis=0)
        q_batch_value = _weighted_sample_mean(q_batch_actions, rl_sample_weights, batch_axis=0)
        q_bad_value = _weighted_sample_mean(q_bad_actions, rl_sample_weights, batch_axis=0)
        advantage_value = _weighted_sample_mean(advantage_sampled_actions, rl_sample_weights, batch_axis=0)
        advantage_ref_value = _weighted_sample_mean(advantage_ref_actions, rl_sample_weights, batch_axis=0)
        advantage_batch_value = _weighted_sample_mean(advantage_batch_actions, rl_sample_weights, batch_axis=0)
        advantage_bad_value = _weighted_sample_mean(advantage_bad_actions, rl_sample_weights, batch_axis=0)
        sample_to_ref_l2 = jnp.mean(
            jnp.linalg.norm(
                jax.lax.stop_gradient(q_actions - ref_q_actions).reshape(batch_size, -1),
                axis=-1,
            )
        )
        info = {
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "imp_loss": imp_loss,
            "ref_loss": ref_loss,
            "q_value": q_value,
            "q_ref_action": q_ref_value,
            "q_batch_action": q_batch_value,
            "q_bad_action": q_bad_value,
            "adv_value": advantage_value,
            "adv_ref_action": advantage_ref_value,
            "adv_batch_action": advantage_batch_value,
            "adv_bad_action": advantage_bad_value,
            "local_delta_q": _weighted_sample_mean(local_delta_q, rl_sample_weights, batch_axis=0),
            "local_delta_advantage": _weighted_sample_mean(local_delta_advantage, rl_sample_weights, batch_axis=0),
            "imp_margin_satisfied_rate": _weighted_sample_mean(
                (local_delta_advantage >= actor_imp_margin).astype(rl_sample_weights.dtype),
                rl_sample_weights,
                batch_axis=0,
            ),
            "imp_under_baseline_rate": _weighted_sample_mean(
                (local_delta_advantage < 0.0).astype(rl_sample_weights.dtype),
                rl_sample_weights,
                batch_axis=0,
            ),
            "sample_to_ref_l2": sample_to_ref_l2,
            "ref_to_flow_ratio": jnp.abs(ref_term) / (jnp.abs(flow_term) + 1e-6),
            "actor_rl_sample_weight_mean": jnp.mean(rl_sample_weights),
            "actor_preprocess_sample_rate": 1.0 - jnp.mean(rl_sample_weights),
            "imp_margin": actor_imp_margin,
            "imp_tau": imp_q_scale,
            "imp_direct_delta": jnp.asarray(self.algo_config.get("actor_imp_direct_delta", False), dtype=bc_loss.dtype),
            "imp_uses_q_delta": jnp.asarray(ablate_critic_pref, dtype=bc_loss.dtype),
            "ablate_actor_bc": jnp.asarray(ablate_actor_bc, dtype=bc_loss.dtype),
            "ablate_actor_advantage": jnp.asarray(ablate_actor_advantage, dtype=bc_loss.dtype),
            "ref_weight_mean": jnp.mean(ref_weights),
            "_vla_cons_actions": {
                "lora": jax.lax.stop_gradient(q_actions),
                "ref": jax.lax.stop_gradient(ref_q_actions),
            },
        }
        return actor_loss, info

    def _prepare_batch(self, batch):
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        if self.algo_config["image_keys"][0] not in batch["next_observations"]:
            batch = unpack_memory_efficient_batch(batch)
        rng, aug_rng = jax.random.split(self.state.critic_state.rng)
        if self.algo_config["augmentation_function"] is not None:
            batch = self.algo_config["augmentation_function"](batch, aug_rng)
        reward_bias = self.algo_config["reward_bias"]
        if batch["rewards"].ndim > 1:
            bias = jnp.zeros_like(batch["rewards"]).at[..., -1].set(reward_bias)
            batch = batch.copy(add_or_replace={"rewards": batch["rewards"] + bias})
        else:
            batch = batch.copy(add_or_replace={"rewards": batch["rewards"] + reward_bias})
        return batch, rng

    def _update_impl(self, batch, *, networks_to_update: frozenset[str]):
        batch, rng = self._prepare_batch(batch)

        new_critic_state = self.state.critic_state.replace(rng=rng)
        new_pi_state = self.state.pi_state
        info = {}

        vla_cons_actions = None
        pending_actor_update = None
        if "actor" in networks_to_update:
            pi_rng, _ = jax.random.split(new_critic_state.rng)
            model = nnx.merge(new_pi_state.model_def, new_pi_state.params)
            diff_state = nnx.DiffState(0, self.pi_train_config.trainable_filter)
            (actor_loss, actor_info), grads = nnx.value_and_grad(
                functools.partial(self.policy_loss_fn, batch),
                argnums=diff_state,
                has_aux=True,
            )(model, pi_rng)
            del actor_loss
            vla_cons_actions = actor_info.pop("_vla_cons_actions", None)
            pending_actor_update = (model, grads, actor_info)
        elif "vla_cons" in networks_to_update:
            pi_rng, _ = jax.random.split(new_critic_state.rng)
            vla_cons_actions = self._sample_vla_cons_actions(batch, pi_rng)

        if "critic" in networks_to_update:
            critic_loss_fn = functools.partial(self.critic_loss_fn, batch)
            if vla_cons_actions is not None and self.algo_config["critic_vla_cons_weight"] > 0.0:
                critic_loss_fn = functools.partial(self.critic_loss_fn, batch, vla_cons_actions=vla_cons_actions)
            (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
                new_critic_state.params, rng
            )
            del critic_loss
            critic_updates, critic_opt_state = new_critic_state.txs["critic"].update(
                critic_grads, new_critic_state.opt_states["critic"], new_critic_state.params
            )
            critic_params = optax.apply_updates(new_critic_state.params, critic_updates)
            new_critic_state = new_critic_state.replace(
                step=new_critic_state.step + 1,
                params=critic_params,
                opt_states={**new_critic_state.opt_states, "critic": critic_opt_state},
            )
            new_critic_state = new_critic_state.target_update(self.algo_config["soft_target_update_rate"])
            info.update(critic_info)

        if pending_actor_update is not None:
            model, grads, actor_info = pending_actor_update
            params = new_pi_state.params.filter(self.pi_train_config.trainable_filter)
            updates, new_opt_state = new_pi_state.tx.update(grads, new_pi_state.opt_state, params)
            updated_params = optax.apply_updates(params, updates)
            nnx.update(model, updated_params)
            new_pi_state = dataclasses.replace(
                new_pi_state,
                step=new_pi_state.step + 1,
                params=nnx.state(model),
                opt_state=new_opt_state,
            )
            if new_pi_state.ema_decay is not None:
                new_pi_state = dataclasses.replace(
                    new_pi_state,
                    ema_params=jax.tree.map(
                        lambda old, new: new_pi_state.ema_decay * old + (1 - new_pi_state.ema_decay) * new,
                        new_pi_state.ema_params,
                        new_pi_state.params,
                    ),
                )
            info.update(actor_info)

        for name, opt_state in new_critic_state.opt_states.items():
            if hasattr(opt_state, "hyperparams") and "learning_rate" in opt_state.hyperparams:
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]
        return self.replace(state=self.state.replace(critic_state=new_critic_state, pi_state=new_pi_state)), info

    @functools.partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update_ql(
        self,
        batch: Batch,
        *,
        pmap_axis: str | None = None,
        networks_to_update: frozenset[str] = frozenset({"actor", "critic"}),
        **kwargs,
    ):
        del pmap_axis, kwargs
        return self._update_impl(batch, networks_to_update=networks_to_update)

    def sample_actions(self, observations: Data, *, seed: PRNGKey | None = None):
        obs = self.adapter.observation(observations)
        sample_actions_fn = self.pi_sample_actions
        if sample_actions_fn is None:
            sample_actions_fn = make_sample_actions_fn(self.state.pi_state)
        action_chunk, embeddings, context_masks, context_offsets = sample_actions_fn(
            self.state.pi_state.params,
            seed,
            obs,
            num_steps=self.algo_config["pi_sample_steps"],
        )
        normalized_action = action_chunk[0, : self.algo_config["action_horizon"], :]
        action = self.adapter.policy_output_actions(normalized_action, obs)
        action = action[..., : self.algo_config["env_action_dim"]]
        return (
            action,
            np.asarray(jax.device_get(embeddings[0])),
            np.asarray(jax.device_get(context_masks[0]), dtype=bool),
            np.int32(np.asarray(jax.device_get(context_offsets[0]))),
        )

    def encode_context(self, observations: Data):
        encode_fn = self.pi_encode_context
        if encode_fn is None:
            encode_fn = make_encode_context_fn(self.state.pi_state)
        obs = self.adapter.observation(observations)
        return encode_fn(self.state.pi_state.params, obs)


def _create_acob_agent(
    seed,
    sample_obs,
    sample_action,
    task_desc,
    pi_train_config,
    image_keys,
    encoder_type,
    discount,
    fix_gripper,
    actor_flow_weight,
    actor_imp_weight,
    actor_ref_weight,
    critic_vla_cons_weight,
    critic_vla_cons_frequency_scale,
    *,
    actor_imp_margin: float = 0.02,
    actor_imp_tau: float = 0.05,
    actor_imp_direct_delta: bool = False,
    critic_intervention_pref_weight: float = 50.0,
    critic_intervention_pref_margin: float = 0.05,
    ablate_critic_pref: bool = False,
    ablate_actor_bc: bool = False,
    ablate_actor_advantage: bool = False,
    resume_pi: bool,
    action_horizon: int,
    pi_sample_steps: int,
    debug_enabled: bool,
    critic_resnet10_params_path: str = DEFAULT_CRITIC_RESNET10_PARAMS_PATH,
    dual_arm: bool = False,
):
    ablate_critic_pref, ablate_actor_bc, ablate_actor_advantage = _validate_ablation_flags(
        ablate_critic_pref=ablate_critic_pref,
        ablate_actor_bc=ablate_actor_bc,
        ablate_actor_advantage=ablate_actor_advantage,
    )
    actor_flow_weight, actor_imp_weight, actor_ref_weight = _normalize_actor_weights(
        0.0 if ablate_actor_bc else actor_flow_weight,
        actor_imp_weight,
        actor_ref_weight,
    )
    actor_imp_margin = _validate_nonnegative_weight("actor_imp_margin", actor_imp_margin)
    actor_imp_tau = _validate_nonnegative_weight("actor_imp_tau", actor_imp_tau)
    if actor_imp_tau <= 0.0:
        raise ValueError(f"actor_imp_tau must be positive, got {actor_imp_tau}")
    actor_imp_direct_delta = bool(actor_imp_direct_delta)
    critic_vla_cons_weight = _validate_nonnegative_weight("critic_vla_cons_weight", critic_vla_cons_weight)
    critic_vla_cons_frequency_scale = _validate_nonnegative_weight(
        "critic_vla_cons_frequency_scale",
        critic_vla_cons_frequency_scale,
    )
    critic_intervention_pref_weight = _validate_nonnegative_weight(
        "critic_intervention_pref_weight",
        critic_intervention_pref_weight,
    )
    critic_intervention_pref_margin = _validate_nonnegative_weight(
        "critic_intervention_pref_margin",
        critic_intervention_pref_margin,
    )
    rng = jax.random.PRNGKey(seed)
    rng, pi_rng, critic_rng = jax.random.split(rng, 3)
    mesh = pi_sharding.make_mesh(pi_train_config.fsdp_devices)
    pi_state = initialize_train_state(pi_train_config, pi_rng, mesh, resume=resume_pi)

    if encoder_type == "resnet":
        from vla_precision.acob.vision import resnetv1_configs

        encoders = {
            image_key: resnetv1_configs["resnetv1-10"](
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    elif encoder_type == "resnet-pretrained":
        from vla_precision.acob.vision import PreTrainedResNetEncoder, resnetv1_configs

        pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](pre_pooling=True, name="pretrained_encoder")
        encoders = {
            image_key: PreTrainedResNetEncoder(
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                pretrained_encoder=pretrained_encoder,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    else:
        raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

    critic_encoder = EncodingWrapper(encoder=encoders, use_proprio=True, enable_stacking=True, image_keys=image_keys)
    value_backbone = ensemblize(
        functools.partial(
            MLP, activations=jax.nn.tanh, use_layer_norm=True, hidden_dims=[256, 256], activate_final=True
        ),
        2,
    )(name="critic_value_ensemble")
    advantage_backbone = ensemblize(
        functools.partial(
            MLP, activations=jax.nn.tanh, use_layer_norm=True, hidden_dims=[256, 256], activate_final=True
        ),
        2,
    )(name="critic_advantage_ensemble")
    critic_def = ValueAdvantageCritic(
        encoder=critic_encoder,
        value_network=value_backbone,
        advantage_network=advantage_backbone,
        name="critic",
    )
    critic_model_def = ModuleDict({"critic": critic_def})
    sample_action = jnp.asarray(sample_action, dtype=jnp.float32)
    if sample_action.ndim == 1:
        sample_action = sample_action[None, :]
    env_action_horizon = int(sample_action.shape[-2])
    if env_action_horizon != int(action_horizon):
        raise ValueError(
            f"sample_action horizon {env_action_horizon} does not match task.action_horizon {action_horizon}"
        )
    env_action_dim = int(sample_action.shape[-1])
    expected_env_action_dim = 14 if dual_arm else (6 if fix_gripper else 7)
    if env_action_dim != expected_env_action_dim:
        raise ValueError(
            f"dual_arm={dual_arm}, fixed_gripper={fix_gripper} requires environment action dim "
            f"{expected_env_action_dim}, got {env_action_dim} from {sample_action.shape}."
        )
    if dual_arm:
        gripper_action_indices = (6, 13)
    else:
        gripper_action_indices = () if fix_gripper else (6,)
    body_action_indices = tuple(idx for idx in range(env_action_dim) if idx not in gripper_action_indices)
    critic_action_indices = body_action_indices if fix_gripper else tuple(range(env_action_dim))
    single_action_dim = len(critic_action_indices)
    action_for_critic = sample_action[..., jnp.asarray(critic_action_indices)]
    action_for_critic = action_for_critic.reshape(-1)
    critic_params = critic_model_def.init({"params": critic_rng}, critic=[sample_obs, action_for_critic])["params"]
    if encoder_type == "resnet-pretrained":
        critic_params = load_pretrained_resnet10_critic_params(
            critic_params,
            critic_resnet10_params_path,
        )
    critic_state = JaxRLTrainState.create(
        apply_fn=critic_model_def.apply,
        params=critic_params,
        txs={"critic": make_optimizer(learning_rate=3e-4)},
        target_params=critic_params,
        rng=rng,
    )
    adapter = OpenPIObservationAdapter(
        pi_train_config,
        task_desc,
        image_keys=image_keys,
        fixed_gripper=bool(fix_gripper),
        dual_arm=bool(dual_arm),
        learned_state_dim=None if dual_arm or fix_gripper else int(np.asarray(sample_obs["state"]).shape[-1]),
        action_horizon=env_action_horizon,
        debug_fn=debug_record if debug_enabled else None,
    )
    use_context = True
    paligemma_config = _gemma.get_config(pi_train_config.model.paligemma_variant)
    max_prefix_tokens = len(adapter.model_image_keys) * 256 + int(pi_train_config.model.max_token_len)
    context_embedding_shape = (
        2,
        paligemma_config.depth,
        max_prefix_tokens,
        paligemma_config.num_kv_heads,
        paligemma_config.head_dim,
    )
    action_dim = env_action_horizon * single_action_dim
    if debug_enabled:
        debug_record(
            "agent.init",
            {
                "seed": seed,
                "encoder_type": encoder_type,
                "image_keys": tuple(image_keys),
                "fix_gripper": fix_gripper,
                "resume_pi": resume_pi,
                "use_context": use_context,
                "context_embedding_shape": context_embedding_shape,
                "context_max_tokens": max_prefix_tokens,
                "weights": {
                    "actor_flow_weight": actor_flow_weight,
                    "actor_imp_weight": actor_imp_weight,
                    "actor_imp_margin": actor_imp_margin,
                    "actor_imp_tau": actor_imp_tau,
                    "actor_imp_direct_delta": actor_imp_direct_delta,
                    "actor_ref_weight": actor_ref_weight,
                    "critic_vla_cons_weight": critic_vla_cons_weight,
                    "critic_vla_cons_frequency_scale": critic_vla_cons_frequency_scale,
                    "critic_intervention_pref_weight": critic_intervention_pref_weight,
                    "critic_intervention_pref_margin": critic_intervention_pref_margin,
                    "ablate_critic_pref": ablate_critic_pref,
                    "ablate_actor_bc": ablate_actor_bc,
                    "ablate_actor_advantage": ablate_actor_advantage,
                },
                "dims": {
                    "env_action_dim": env_action_dim,
                    "single_action_dim": single_action_dim,
                    "action_horizon": env_action_horizon,
                    "critic_action_dim": action_dim,
                    "critic_action_space": "openpi_normalized_action_chunk",
                    "pi_action_dim": pi_train_config.model.action_dim,
                    "pi_action_horizon": pi_train_config.model.action_horizon,
                    "pi_sample_steps": pi_sample_steps,
                },
            },
        )
    return ACoBAgent(
        state=ACoBState(critic_state=critic_state, pi_state=pi_state, ref_pi_params=pi_state.params),
        adapter=adapter,
        pi_train_config=pi_train_config,
        pi_sample_actions=make_sample_actions_fn(pi_state),
        pi_encode_context=make_encode_context_fn(pi_state),
        algo_config=dict(  # noqa: C408 - mirrors the algorithm's named hyperparameter block
            critic_ensemble_size=2,
            critic_subsample_size=None,
            discount=discount,
            fix_gripper=fix_gripper,
            dual_arm=bool(dual_arm),
            gripper_action_indices=gripper_action_indices,
            body_action_indices=body_action_indices,
            critic_action_indices=critic_action_indices,
            soft_target_update_rate=0.005,
            actor_flow_weight=actor_flow_weight,
            actor_imp_weight=actor_imp_weight,
            actor_imp_margin=actor_imp_margin,
            actor_imp_tau=actor_imp_tau,
            actor_imp_direct_delta=actor_imp_direct_delta,
            actor_ref_weight=actor_ref_weight,
            critic_vla_cons_weight=critic_vla_cons_weight,
            critic_vla_cons_frequency_scale=critic_vla_cons_frequency_scale,
            critic_intervention_pref_weight=critic_intervention_pref_weight,
            critic_intervention_pref_margin=critic_intervention_pref_margin,
            ablate_critic_pref=ablate_critic_pref,
            ablate_actor_bc=ablate_actor_bc,
            ablate_actor_advantage=ablate_actor_advantage,
            action_dim=action_dim,
            critic_action_dim=action_dim,
            single_action_dim=single_action_dim,
            env_action_dim=env_action_dim,
            action_horizon=env_action_horizon,
            image_keys=tuple(image_keys),
            reward_bias=0.0,
            augmentation_function=make_batch_augmentation(image_keys),
            pi_sample_steps=pi_sample_steps,
            use_context=use_context,
            context_embedding_shape=context_embedding_shape,
            context_mask_shape=(max_prefix_tokens,),
            context_dtype="bfloat16",
            context_max_tokens=max_prefix_tokens,
        ),
    )
