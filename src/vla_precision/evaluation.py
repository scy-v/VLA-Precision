"""Checkpoint evaluation through the same Pyro robot environment as ACoB-Stream.

Both evaluators use OpenPI's native policy loader.  A Stage-I checkpoint loads
the full policy, while an ACoB checkpoint loads the Stage-II action-expert
actor parameters saved by the learner.  The rollout loop itself is shared so
reset, action-chunk, completion and episode accounting cannot drift.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from vla_precision.config import ResolvedConfig
from vla_precision.integrations.openpi.checkpoints import (
    OpenPICheckpoint,
    resolve_openpi_checkpoint,
)
from vla_precision.integrations.openpi.inference.policy import OpenPISampler
from vla_precision.integrations.openpi.inference.policy import (
    make_policy_observation as _make_policy_observation,
)
from vla_precision.serialization import b64_to_numpy

EvaluationMode = Literal["acob", "vla"]
LOGGER = logging.getLogger(__name__)
OpenPIPolicySampler = OpenPISampler


def openpi_policy_observation(
    observation: dict[str, Any], *, prompt: str, dual_arm: bool
) -> dict[str, Any]:
    """Backward-compatible public helper backed by the OpenPI extension."""
    return _make_policy_observation(observation, prompt=prompt, dual_arm=dual_arm)


class EvaluationSampler(Protocol):
    label: str

    def sample(self, observation: dict[str, Any]) -> np.ndarray: ...


class EvaluationEnvironment(Protocol):
    def env_reset(self, *, options: dict[str, Any]): ...

    def env_step(self, actions: list): ...

    def env_action_space_sample(self): ...


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    success: bool
    steps: int
    duration_seconds: float
    reward_sum: float
    terminated: bool
    truncated: bool
    timed_out: bool


@dataclass(frozen=True)
class EvaluationResult:
    mode: EvaluationMode
    checkpoint: str
    checkpoint_step: int | None
    episodes: tuple[EpisodeResult, ...]
    result_path: Path

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(int(episode.success) for episode in self.episodes) / len(self.episodes)


def prepare_environment_actions(policy_actions: Any, action_space_sample: Any) -> np.ndarray:
    """Trim OpenPI padding only after checking the real Pyro action boundary."""
    actions = np.asarray(policy_actions, dtype=np.float32)
    expected = np.asarray(action_space_sample)
    if actions.ndim == 1:
        actions = actions[None]
    if actions.ndim != 2 or expected.ndim != 2:
        raise ValueError(
            f"evaluation action chunks must be rank two: policy={actions.shape}, environment={expected.shape}"
        )
    horizon, action_dimension = expected.shape
    if actions.shape[0] < horizon or actions.shape[1] < action_dimension:
        raise ValueError(
            f"policy action chunk {actions.shape} is smaller than environment action space {expected.shape}"
        )
    return actions[:horizon, :action_dimension]


def completion_succeeded(info: Any) -> bool:
    """Read only the explicit task-completion signal from environment info."""
    if isinstance(info, dict) and "succeed" in info:
        value = np.asarray(info["succeed"]).reshape(-1)
        return bool(value[0]) if value.size else False
    return False


def resolve_checkpoint(
    resolved: ResolvedConfig,
    *,
    mode: EvaluationMode,
    train_config: Any,
) -> OpenPICheckpoint:
    """Resolve either a Stage-I checkpoint or the saved Stage-II actor."""
    config = resolved.config
    requested_step_value = int(config.evaluation.checkpoint_step)
    requested_step = None if requested_step_value == 0 else requested_step_value
    if mode == "acob":
        from vla_precision.data.paths import stage2_actor_checkpoint_dir

        source: str | Path = stage2_actor_checkpoint_dir(config)
    else:
        source = config.openpi.initialization_checkpoint or train_config.checkpoint_dir
    return resolve_openpi_checkpoint(source, requested_step=requested_step)


def build_sampler(
    resolved: ResolvedConfig,
    *,
    mode: EvaluationMode,
    policy_factory=None,
    train_config_builder=None,
) -> tuple[OpenPIPolicySampler, OpenPICheckpoint, str]:
    """Load a directly executable OpenPI policy for the selected checkpoint."""
    if train_config_builder is None:
        from vla_precision.integrations.openpi.configs import (
            build_stage2_initialization_train_config,
            build_stage2_train_config,
        )

        train_config_builder = (
            build_stage2_train_config
            if mode == "acob"
            else build_stage2_initialization_train_config
        )
    if policy_factory is None:
        # Importing our extension first installs the modern-LeRobot bridge while
        # keeping the vendored OpenPI source untouched.
        from vla_precision.integrations.openpi import install_lerobot_import_compat

        install_lerobot_import_compat()
        from openpi.policies import policy_config

        policy_factory = policy_config.create_trained_policy

    train_config = train_config_builder(resolved.config)
    if mode == "vla" and int(train_config.model.action_horizon) != resolved.config.task.action_horizon:
        # The established direct-VLA evaluator samples only the chunk that the
        # environment will execute. ACoB keeps its training-time model horizon
        # and selects the executed prefix after sampling.
        train_config = replace(
            train_config,
            model=replace(
                train_config.model,
                action_horizon=resolved.config.task.action_horizon,
            ),
        )
    checkpoint = resolve_checkpoint(resolved, mode=mode, train_config=train_config)
    policy = policy_factory(
        train_config,
        checkpoint.directory,
        sample_kwargs={"num_steps": resolved.config.openpi.sample_steps},
        default_prompt=resolved.config.task.instruction,
    )
    if mode == "acob" and hasattr(policy, "_rng"):
        # Match the ACoB actor/evaluator sequence: the policy splits the
        # configured experiment key immediately before each sample.
        import jax

        policy._rng = jax.random.PRNGKey(resolved.config.experiment.seed)
    sampler = OpenPIPolicySampler(
        policy,
        prompt=resolved.config.task.instruction,
        dual_arm=resolved.config.task.arm_mode == "dual",
        label=f"{mode}:{train_config.name}",
    )
    return sampler, checkpoint, train_config.name


def _result_path(resolved: ResolvedConfig, *, mode: EvaluationMode) -> Path:
    run_time = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return (
        Path(resolved.config.paths.output_root).expanduser()
        / resolved.config.experiment.name
        / mode
        / f"{run_time}.json"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_result(
    resolved: ResolvedConfig,
    *,
    path: Path,
    mode: EvaluationMode,
    checkpoint: OpenPICheckpoint,
    profile: str,
    episodes: list[EpisodeResult],
) -> Path:
    successes = [episode for episode in episodes if episode.success]
    payload = {
        "experiment": resolved.config.experiment.name,
        "mode": mode,
        "model": resolved.config.openpi.model,
        "profile": profile,
        "checkpoint": str(checkpoint.directory),
        "checkpoint_step": checkpoint.step,
        "episodes": [asdict(episode) for episode in episodes],
        "summary": {
            "episode_count": len(episodes),
            "success_count": len(successes),
            "success_rate": len(successes) / len(episodes) if episodes else 0.0,
            "average_episode_steps": (
                float(np.mean([episode.steps for episode in episodes])) if episodes else 0.0
            ),
            "average_duration_seconds": (
                float(np.mean([episode.duration_seconds for episode in episodes]))
                if episodes
                else 0.0
            ),
            "average_success_duration_seconds": (
                float(np.mean([episode.duration_seconds for episode in successes]))
                if successes
                else 0.0
            ),
        },
    }
    _atomic_write_json(path, payload)
    return path


def evaluate_rollouts(
    resolved: ResolvedConfig,
    *,
    mode: EvaluationMode,
    environment: EvaluationEnvironment,
    sampler: EvaluationSampler,
    checkpoint: OpenPICheckpoint,
    profile: str,
    monotonic=time.monotonic,
) -> EvaluationResult:
    """Run rollouts until explicit completion or wall timeout.

    This is the reference Pyro-evaluator boundary: raw Gym termination and
    truncation values are recorded for diagnosis but never decide task
    completion.
    """
    config = resolved.config
    requested_episodes = int(config.evaluation.episodes)
    timeout = float(config.evaluation.episode_timeout_seconds)
    progress_interval = float(config.evaluation.progress_interval_seconds)
    if requested_episodes < 1:
        raise ValueError("evaluation.episodes must be positive")
    if timeout < 0.0:
        raise ValueError("evaluation.episode_timeout_seconds cannot be negative")

    action_space_sample = b64_to_numpy(environment.env_action_space_sample())
    episodes: list[EpisodeResult] = []
    result_path = _result_path(resolved, mode=mode)
    _write_result(
        resolved,
        path=result_path,
        mode=mode,
        checkpoint=checkpoint,
        profile=profile,
        episodes=episodes,
    )
    for episode_index in range(requested_episodes):
        reset_options = {
            "regrasp_before_reset": bool(
                episode_index > 0 and config.evaluation.regrasp_before_reset
            )
        }
        observation, _ = environment.env_reset(options=reset_options)
        observation = b64_to_numpy(observation)

        # Match the established evaluator: compile/sample once after each reset
        # and start episode timing only after this discarded warm-up action.
        sampler.sample(observation)
        started = monotonic()
        next_progress = progress_interval
        steps = 0
        episode_reward = 0.0
        success = terminated = truncated = timed_out = False

        while True:
            elapsed = monotonic() - started
            if timeout > 0.0 and elapsed >= timeout:
                timed_out = True
                break

            actions = prepare_environment_actions(
                sampler.sample(observation),
                action_space_sample,
            )
            next_observation, reward, terminated_value, truncated_value, info = (
                environment.env_step(actions.astype(float).tolist())
            )
            observation = b64_to_numpy(next_observation)
            reward = np.asarray(b64_to_numpy(reward), dtype=np.float32).reshape(-1)
            raw_terminated = bool(
                np.asarray(b64_to_numpy(terminated_value)).reshape(-1)[0]
            )
            raw_truncated = bool(
                np.asarray(b64_to_numpy(truncated_value)).reshape(-1)[0]
            )
            terminated = bool(terminated or raw_terminated)
            truncated = bool(truncated or raw_truncated)
            info = b64_to_numpy(info)
            success = completion_succeeded(info)
            steps += 1
            episode_reward += float(reward.sum())
            elapsed = monotonic() - started
            timed_out = bool(timeout > 0.0 and elapsed >= timeout)
            if progress_interval > 0.0 and elapsed >= next_progress:
                if timeout > 0.0:
                    LOGGER.info(
                        "evaluation episode=%d elapsed=%.0fs/%.0fs",
                        episode_index + 1,
                        elapsed,
                        timeout,
                    )
                else:
                    LOGGER.info(
                        "evaluation episode=%d elapsed=%.0fs",
                        episode_index + 1,
                        elapsed,
                    )
                next_progress = (
                    int(elapsed // progress_interval) + 1
                ) * progress_interval
            if success or timed_out:
                break

        duration = timeout if timed_out and timeout > 0.0 else elapsed
        result = EpisodeResult(
            episode=episode_index + 1,
            success=bool(success and not timed_out),
            steps=steps,
            duration_seconds=round(float(duration), 3),
            reward_sum=round(episode_reward, 6),
            terminated=bool(terminated),
            truncated=bool(truncated),
            timed_out=bool(timed_out),
        )
        episodes.append(result)
        _write_result(
            resolved,
            path=result_path,
            mode=mode,
            checkpoint=checkpoint,
            profile=profile,
            episodes=episodes,
        )
        success_count = sum(int(item.success) for item in episodes)
        LOGGER.info(
            "evaluation episode=%d/%d success=%s steps=%d duration=%.3fs "
            "success_rate=%d/%d (%.1f%%)",
            result.episode,
            requested_episodes,
            result.success,
            result.steps,
            result.duration_seconds,
            success_count,
            len(episodes),
            100.0 * success_count / len(episodes),
        )

    result = EvaluationResult(
        mode=mode,
        checkpoint=str(checkpoint.directory),
        checkpoint_step=checkpoint.step,
        episodes=tuple(episodes),
        result_path=result_path,
    )
    LOGGER.info(
        "evaluation complete mode=%s success_rate=%.3f result=%s",
        mode,
        result.success_rate,
        result_path,
    )
    return result


def run_evaluation(
    resolved: ResolvedConfig,
    *,
    mode: EvaluationMode,
    environment: EvaluationEnvironment | None = None,
    sampler: EvaluationSampler | None = None,
    checkpoint: OpenPICheckpoint | None = None,
    profile: str | None = None,
) -> EvaluationResult:
    """Runtime used by ``python main.py --stage ... --mode evaluate``."""
    if environment is None:
        from vla_precision.acob_stream.communication import PyroEnvironmentClient

        environment = PyroEnvironmentClient.connect(
            resolved,
            ignore_episode_length=True,
            allow_intervention=resolved.config.evaluation.allow_intervention,
        )
    if sampler is None:
        sampler, loaded_checkpoint, loaded_profile = build_sampler(resolved, mode=mode)
        checkpoint = loaded_checkpoint
        profile = loaded_profile
    if checkpoint is None or profile is None:
        raise ValueError("injected evaluation runtimes must provide checkpoint and profile metadata")
    return evaluate_rollouts(
        resolved,
        mode=mode,
        environment=environment,
        sampler=sampler,
        checkpoint=checkpoint,
        profile=profile,
    )
