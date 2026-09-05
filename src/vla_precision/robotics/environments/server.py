"""Pyro manager for the robot-side ACoB-Stream environment."""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable
from io import BytesIO
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from vla_precision.config import ResolvedConfig
from vla_precision.robotics.environments.factory import build_environment
from vla_precision.runtime_identity import (
    RuntimeCodeIdentity,
    compatible_runtime_identity,
    consistency_mode,
    runtime_code_identity,
)

LOGGER = logging.getLogger(__name__)


def numpy_to_b64(value):
    """Encode NumPy leaves for Pyro's default serializer."""
    if isinstance(value, np.ndarray):
        buffer = BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: numpy_to_b64(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numpy_to_b64(item) for item in value]
    if isinstance(value, tuple):
        return tuple(numpy_to_b64(item) for item in value)
    return value


def disable_internal_episode_length(env: gym.Env) -> tuple[str, ...]:
    """Disable step-count truncation throughout an evaluation wrapper tree."""
    pending = [env]
    visited: set[int] = set()
    changed: list[str] = []
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        current_values = vars(current)
        if "max_episode_length" in current_values:
            current.max_episode_length = float("inf")
            changed.append(type(current).__name__)
        for attribute in ("env", "env_left", "env_right"):
            child = current_values.get(attribute)
            if child is not None:
                pending.append(child)
    return tuple(changed)


class EnvironmentServiceProtocol(Protocol):
    def handshake(
        self,
        experiment_name: str,
        shared_sha256: str,
        code_identity: dict[str, str],
        strict_distributed_consistency: bool | None = True,
    ) -> dict[str, Any]: ...

    def get_environment(
        self,
        experiment_name: str,
        shared_sha256: str,
        code_identity: dict[str, str],
        strict_distributed_consistency: bool | None = True,
        ignore_episode_length: bool = False,
        allow_intervention: bool = True,
    ): ...


class RemoteEnvironment(RecordEpisodeStatistics):
    """The per-session environment object registered under a generated URI."""

    def env_reset(self, options=None):
        observation, info = self.reset(options=options)
        return numpy_to_b64(observation), numpy_to_b64(info)

    def env_action_space_sample(self):
        return numpy_to_b64(self.action_space.sample())

    def env_observation_space_sample(self):
        return numpy_to_b64(self.observation_space.sample())

    def env_step(self, actions):
        values = self.step(np.asarray(actions))
        return tuple(numpy_to_b64(value) for value in values)

    def env_request(self, name: str, param: Any):
        env: Any = self
        while env is not None:
            request = getattr(env, "request", None)
            if request is not None:
                request(name=name, param=param)
                return True
            env = getattr(env, "env", None)
        raise AttributeError(f"environment wrapper chain has no request method for {name!r}")


class EnvironmentService:
    """Singleton manager that rebuilds one server-local environment per client.

    The experiment identity, shared configuration hash, and lightweight runtime
    code identity cross the wire. The environment class and all of its values
    come from the server's own copy of the same resolved task/deployment configuration.
    """

    def __init__(
        self,
        resolved: ResolvedConfig,
        daemon,
        *,
        environment_factory: Callable[..., gym.Env] = build_environment,
        close_delay_seconds: float = 0.2,
        code_identity: RuntimeCodeIdentity | None = None,
    ):
        self.resolved = resolved
        self.daemon = daemon
        self.environment_factory = environment_factory
        self.close_delay_seconds = float(close_delay_seconds)
        self.code_identity = code_identity or runtime_code_identity()
        self.env: RemoteEnvironment | None = None

    @property
    def experiment_name(self) -> str:
        return self.resolved.config.experiment.name

    @property
    def shared_sha256(self) -> str:
        return self.resolved.shared_sha256

    def handshake(
        self,
        experiment_name: str,
        shared_sha256: str,
        code_identity: dict[str, str],
        strict_distributed_consistency: bool | None = True,
    ) -> dict[str, Any]:
        strict = consistency_mode(
            self.resolved.config.strict_distributed_consistency,
            strict_distributed_consistency,
        )
        remote = {
            "experiment_name": self.experiment_name,
            "shared_sha256": self.shared_sha256,
            "runtime_code_identity": self.code_identity.to_dict(),
            "strict_distributed_consistency": strict,
        }
        requested = {
            "experiment_name": experiment_name,
            "shared_sha256": shared_sha256,
            "runtime_code_identity": code_identity,
        }
        matches = strict is None or (
            requested["experiment_name"] == remote["experiment_name"]
            and requested["shared_sha256"] == remote["shared_sha256"]
            and compatible_runtime_identity(
                requested["runtime_code_identity"],
                remote["runtime_code_identity"],
                strict=strict,
            )
        )
        if not matches:
            raise RuntimeError(
                "environment configuration mismatch or runtime code mismatch: "
                f"actor={requested}, server={remote}"
            )
        return remote

    def _close_current_environment(self) -> None:
        old_environment = self.env
        self.env = None
        if old_environment is None:
            return

        object_id = getattr(old_environment, "_pyroId", None)
        try:
            old_environment.close()
            if self.close_delay_seconds > 0:
                time.sleep(self.close_delay_seconds)
            LOGGER.info("closed previous environment object_id=%s", object_id)
        except Exception:  # Release the Pyro object even when hardware close fails.
            LOGGER.exception("failed to close previous environment object_id=%s", object_id)

        try:
            self.daemon.unregister(old_environment)
        except Exception:  # A stale registration must not block rebuilding hardware.
            LOGGER.exception("failed to unregister previous environment object_id=%s", object_id)

    def get_environment(
        self,
        experiment_name: str,
        shared_sha256: str,
        code_identity: dict[str, str],
        strict_distributed_consistency: bool | None = True,
        ignore_episode_length: bool = False,
        allow_intervention: bool = True,
    ):
        """Close any previous session, rebuild locally, and return its Pyro URI."""
        self.handshake(
            experiment_name,
            shared_sha256,
            code_identity,
            strict_distributed_consistency,
        )
        self._close_current_environment()

        environment = self.environment_factory(
            self.resolved,
            fake_env=False,
            teleoperation_enabled=allow_intervention,
        )
        if ignore_episode_length:
            changed = disable_internal_episode_length(environment)
            LOGGER.info(
                "disabled internal episode length for evaluation: %s",
                ", ".join(changed) if changed else "no max_episode_length attribute found",
            )
        self.env = RemoteEnvironment(environment)
        return self.daemon.register(self.env)

    def close(self) -> None:
        self._close_current_environment()


def run_environment_server(resolved: ResolvedConfig) -> None:
    """Serve the singleton manager; hardware is acquired by ``get_environment``."""
    import Pyro5
    import Pyro5.api

    Pyro5.config.SOCK_NODELAY = True
    Pyro5.api.expose(EnvironmentService)
    Pyro5.api.expose(RemoteEnvironment)
    daemon = Pyro5.api.Daemon(
        host=resolved.config.network.environment_bind_host,
        port=resolved.config.network.environment_port,
        nathost=resolved.config.network.environment_host,
        natport=resolved.config.network.environment_port,
    )
    service = EnvironmentService(resolved, daemon)
    daemon.register(service, objectId="environment")
    try:
        daemon.requestLoop()
    finally:
        service.close()
        daemon.close()
