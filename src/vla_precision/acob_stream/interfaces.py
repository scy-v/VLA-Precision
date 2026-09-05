"""Narrow runtime interfaces used by both ACoB-Stream roles.

The protocols keep the online algorithm independent of a particular robot,
Pyro proxy, AgentLace implementation, checkpoint manager, or logger.  Both
roles still consume the same :class:`RootConfig`; ``role`` only selects which
runtime is assembled at the top-level command.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol


class EnvironmentClient(Protocol):
    """A per-session environment proxy returned by the fixed Pyro manager."""

    def env_reset(self, *, options: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]: ...

    def env_step(self, actions: list) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]: ...

    def env_action_space_sample(self) -> Any: ...

    def env_request(self, *, name: str, param: Any) -> Any: ...


class TransitionStore(Protocol):
    def insert(self, transition: Mapping[str, Any]) -> None: ...

    def __len__(self) -> int: ...


class ContextWriter(Protocol):
    def put_batch(self, ids, embeddings, masks, offsets) -> None: ...

    def flush(self) -> None: ...


class ActorTrainerClient(Protocol):
    def recv_network_callback(self, callback: Callable[[Any], None]) -> None: ...

    def update(self) -> bool: ...

    def request(self, type: str, payload: dict) -> dict | None: ...


class LearnerTrainerServer(Protocol):
    def register_data_store(self, name: str, data_store: TransitionStore) -> None: ...

    def start(self, threaded: bool = False) -> None: ...

    def publish_network(self, payload: Any) -> None: ...

    def stop(self) -> None: ...


class ActorAgent(Protocol):
    algo_config: Mapping[str, Any]

    def sample_actions(self, observations: Any, *, seed: Any): ...

    def encode_context(self, observations: Any): ...


class LearnerAgent(Protocol):
    state: Any

    def update_ql(
        self,
        batch: Any,
        *,
        networks_to_update: frozenset[str],
    ) -> tuple[Any, Mapping[str, Any]]: ...


ActorParameterReplacer = Callable[[ActorAgent, Any], ActorAgent]
ActorParameterPublisher = Callable[[LearnerAgent], Any]
CheckpointSaver = Callable[[LearnerAgent, int], None]
LearnerMetricsSink = Callable[[int, Mapping[str, Any], Mapping[str, float]], None]
ActorStatsSink = Callable[[dict], None]
TransitionShardWriter = Callable[[int, list[dict], list[dict]], None]
