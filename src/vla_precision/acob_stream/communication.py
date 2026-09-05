"""Pyro environment and AgentLace trainer links for ACoB-Stream."""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any

from vla_precision.config.loader import ResolvedConfig
from vla_precision.runtime_identity import (
    agentlace_contract_version,
    compatible_runtime_identity,
    consistency_mode,
    runtime_code_identity,
)

REPLAY_STORE_NAME = "replay"
CORRECTION_STORE_NAME = "correction"
STATS_REQUEST = "send-stats"
ENVIRONMENT_OBJECT_ID = "environment"


def actor_parameter_payload(agent):
    """Publish only OpenPI leaves trainable by the Stage-II profile."""
    return agent.state.pi_state.params.filter(agent.pi_train_config.trainable_filter)


def replace_actor_parameters(agent, parameters):
    """Merge a learner payload without touching critic or reference parameters."""
    from flax import nnx

    from vla_precision.integrations.openpi.context import make_encode_context_fn, make_sample_actions_fn

    model = nnx.merge(agent.state.pi_state.model_def, agent.state.pi_state.params)
    nnx.update(model, parameters)
    pi_state = dataclasses.replace(agent.state.pi_state, params=nnx.state(model))
    return agent.replace(
        state=agent.state.replace(pi_state=pi_state),
        pi_sample_actions=agent.pi_sample_actions or make_sample_actions_fn(pi_state),
        pi_encode_context=agent.pi_encode_context or make_encode_context_fn(pi_state),
    )


def _wait_for_environment_manager(proxy_factory, uri: str, *, retry_interval: float = 1.0):
    """Wait for the robot-side manager before attempting configuration RPCs."""
    while True:
        proxy = proxy_factory(uri)
        try:
            proxy._pyroBind()
            return proxy
        except Exception as error:  # noqa: BLE001 -- preserve indefinite wait for the robot process
            release = getattr(proxy, "_pyroRelease", None)
            if release is not None:
                release()
            logging.getLogger(__name__).info(
                "waiting for environment manager at %s: %s",
                uri,
                error,
            )
            time.sleep(retry_interval)


def make_trainer_config(resolved: ResolvedConfig):
    """Create one identical AgentLace contract on actor and learner hosts."""
    from agentlace.trainer import TrainerConfig

    config = resolved.config.network
    return TrainerConfig(
        port_number=config.trainer_port,
        broadcast_port=config.data_port,
        request_types=[STATS_REQUEST],
        version=agentlace_contract_version(
            resolved.shared_sha256,
            strict=resolved.config.strict_distributed_consistency,
        ),
    )


def create_actor_trainer_client(resolved: ResolvedConfig, replay_store, correction_store):
    from agentlace.trainer import TrainerClient

    return TrainerClient(
        REPLAY_STORE_NAME,
        resolved.config.network.learner_host,
        make_trainer_config(resolved),
        data_stores={
            REPLAY_STORE_NAME: replay_store,
            CORRECTION_STORE_NAME: correction_store,
        },
        wait_for_server=resolved.config.actor.wait_for_learner,
        timeout_ms=3000,
    )


def create_learner_trainer_server(resolved: ResolvedConfig, request_callback):
    from agentlace.trainer import TrainerServer

    return TrainerServer(make_trainer_config(resolved), request_callback=request_callback)


@dataclass
class PyroEnvironmentClient:
    """Proxy to a session created by the fixed environment manager.

    The environment process loads the same task/deployment/overrides itself. The
    manager handshake prevents an actor from controlling a service launched
    with a different shared experiment configuration or code snapshot. It then
    rebuilds and registers the real environment, whose returned URI becomes
    ``proxy``; no Python class or mapping key crosses the network.
    """

    proxy: Any

    @classmethod
    def connect(
        cls,
        resolved: ResolvedConfig,
        *,
        ignore_episode_length: bool = False,
        allow_intervention: bool = True,
        proxy_factory=None,
    ) -> PyroEnvironmentClient:
        if proxy_factory is None:
            import Pyro5.api

            proxy_factory = Pyro5.api.Proxy

        network = resolved.config.network
        manager_uri = f"PYRO:{ENVIRONMENT_OBJECT_ID}@{network.environment_host}:{network.environment_port}"
        manager = _wait_for_environment_manager(proxy_factory, manager_uri)
        try:
            expected = {
                "experiment_name": resolved.config.experiment.name,
                "shared_sha256": resolved.shared_sha256,
                "runtime_code_identity": runtime_code_identity().to_dict(),
            }
            strict = resolved.config.strict_distributed_consistency
            reply = manager.handshake(
                expected["experiment_name"],
                expected["shared_sha256"],
                expected["runtime_code_identity"],
                strict,
            )
            effective_strict = consistency_mode(
                strict,
                reply.get("strict_distributed_consistency"),
            )
            reply_matches = effective_strict is None or (
                reply.get("experiment_name") == expected["experiment_name"]
                and reply.get("shared_sha256") == expected["shared_sha256"]
                and compatible_runtime_identity(
                    expected["runtime_code_identity"],
                    reply.get("runtime_code_identity", {}),
                    strict=effective_strict,
                )
            )
            if not reply_matches:
                raise RuntimeError(
                    f"environment configuration/runtime handshake mismatch: expected={expected}, received={reply}"
                )
            environment_uri = manager.get_environment(
                expected["experiment_name"],
                expected["shared_sha256"],
                expected["runtime_code_identity"],
                strict,
                ignore_episode_length=ignore_episode_length,
                allow_intervention=allow_intervention,
            )
            if environment_uri is None:
                raise RuntimeError("environment service returned no session URI")
            proxy = proxy_factory(environment_uri)
            proxy._pyroBind()
        finally:
            release = getattr(manager, "_pyroRelease", None)
            if release is not None:
                release()
        logging.getLogger(__name__).info(
            "connected to environment session at %s via %s",
            environment_uri,
            manager_uri,
        )
        return cls(proxy=proxy)

    def env_reset(self, *, options):
        return self.proxy.env_reset(options=options)

    def env_step(self, actions: list):
        return self.proxy.env_step(actions)

    def env_action_space_sample(self):
        return self.proxy.env_action_space_sample()

    def env_observation_space_sample(self):
        return self.proxy.env_observation_space_sample()

    def env_request(self, *, name: str, param: Any):
        return self.proxy.env_request(name=name, param=param)
