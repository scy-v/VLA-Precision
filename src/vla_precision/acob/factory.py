"""Construct the ACoB agent from the single top-level configuration."""

from __future__ import annotations

from typing import Any

from vla_precision.acob.agent import ACoBAgent, _create_acob_agent
from vla_precision.config.schema import RootConfig
from vla_precision.integrations.openpi.configs import build_stage2_train_config


def create_agent(
    config: RootConfig,
    *,
    sample_observation: Any,
    sample_action: Any,
    train_config: Any | None = None,
) -> ACoBAgent:
    """Build Stage-II ACoB without a second config or experiment mapping."""
    if train_config is None:
        train_config = build_stage2_train_config(config)

    agent = _create_acob_agent(
        seed=config.experiment.seed,
        sample_obs=sample_observation,
        sample_action=sample_action,
        task_desc=config.task.instruction,
        pi_train_config=train_config,
        image_keys=config.task.image_keys,
        encoder_type=config.acob.critic_encoder,
        discount=config.acob.discount,
        fix_gripper=config.task.setup_mode == "single-arm-fixed-gripper",
        actor_flow_weight=config.acob.actor_flow_weight,
        actor_imp_weight=config.acob.actor_improvement_weight,
        actor_imp_margin=config.acob.actor_improvement_margin,
        actor_imp_tau=config.acob.actor_improvement_temperature,
        actor_imp_direct_delta=config.acob.actor_improvement_direct_delta,
        actor_ref_weight=config.acob.actor_reference_weight,
        critic_vla_cons_weight=config.acob.critic_vla_conservative_weight,
        critic_vla_cons_frequency_scale=max(1, config.stream.critic_updates_per_step),
        critic_intervention_pref_weight=config.acob.critic_intervention_preference_weight,
        critic_intervention_pref_margin=config.acob.critic_intervention_preference_margin,
        ablate_critic_pref=config.acob.ablate_critic_preference,
        ablate_actor_bc=config.acob.ablate_actor_bc,
        ablate_actor_advantage=config.acob.ablate_actor_advantage,
        resume_pi=config.checkpoint.resume,
        action_horizon=config.task.action_horizon,
        pi_sample_steps=config.openpi.sample_steps,
        debug_enabled=config.debug,
        critic_resnet10_params_path=config.paths.critic_resnet10_params_path,
        dual_arm=config.task.arm_mode == "dual",
    )
    return agent.replace(
        algo_config={
            **agent.algo_config,
            "preprocess_bc_only": config.data.preprocess.bc_only,
        }
    )
