"""Paper-facing metrics with no separate debug telemetry surface."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

import numpy as np


class _TimerContextManager:
    def __init__(self, timer: Timer, key: str):
        self.timer = timer
        self.key = key

    def __enter__(self):
        self.timer.tick(self.key)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.timer.tock(self.key)


class Timer:
    """Collect average wall-clock durations for named loop sections."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}
        self.latest_times = {}

    def tick(self, key):
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        elapsed = time.time() - self.start_times[key]
        self.counts[key] += 1
        self.times[key] += elapsed
        self.latest_times[key] = elapsed
        del self.start_times[key]

    def context(self, key):
        return _TimerContextManager(self, key)

    def get_average_times(self, reset=True):
        result = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return result

    def get_latest_times(self):
        return dict(self.latest_times)

_ACTOR_UPDATE_KEYS = {
    "actor_loss": "loss",
    "bc_loss": "bc_loss",
    "imp_loss": "imp_loss",
    "ref_loss": "ref_loss",
    "local_delta_advantage": "local_delta_advantage",
    "local_delta_q": "local_delta_q",
    "imp_margin_satisfied_rate": "imp_margin_satisfied_rate",
    "imp_under_baseline_rate": "imp_under_baseline_rate",
    "sample_to_ref_l2": "sample_to_ref_l2",
    "ref_to_flow_ratio": "ref_to_flow_ratio",
}

_CRITIC_UPDATE_KEYS = {
    "critic_loss": "loss",
    "td_loss": "td_loss",
    "predicted_qs": "predicted_qs",
    "target_qs": "target_qs",
    "intervention_pref_loss": "intervention_pref_loss",
    "intervention_pref_margin": "intervention_pref_margin",
    "intervention_pref_accuracy": "intervention_pref_accuracy",
    "intervention_pref_margin_satisfied_rate": "intervention_pref_margin_satisfied_rate",
    "intervention_adv_good": "intervention_adv_good",
    "intervention_adv_bad": "intervention_adv_bad",
}

_EPISODE_KEYS = {
    "return_sum": "return_sum",
    "l": "length",
    "t": "duration_seconds",
    "duration_seconds": "duration_seconds",
    "succeed": "succeed",
    "no_intervention_succeed": "no_intervention_succeed",
    "timeout": "timeout",
    "index": "index",
    "total_steps": "total_steps",
    "intervention_count": "intervention_count",
    "intervention_steps": "intervention_steps",
    "left_intervention_steps": "left_intervention_steps",
    "right_intervention_steps": "right_intervention_steps",
    "trajectory_len": "trajectory_len",
    "action_steps": "action_steps",
    "intervention_step_ratio": "intervention_step_ratio",
    "intervention_count_per_transition": "intervention_count_per_transition",
}


def _scalar(value):
    try:
        import jax

        value = jax.device_get(value)
    except (ImportError, TypeError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return value
    if array.shape == ():
        return array.item()
    return value


def actor_episode_payload(info: dict) -> dict:
    """Keep the original paper-facing ``environment.episode`` wire payload."""
    episode = dict(info.get("episode", {}))
    if "no_intervention_succeed" not in episode and "succeed" in episode and "intervention_count" in episode:
        succeed = bool(np.asarray(episode["succeed"]).reshape(-1)[0])
        intervention_count = int(np.asarray(episode["intervention_count"]).reshape(-1)[0])
        episode["no_intervention_succeed"] = int(succeed and intervention_count == 0)
    selected = {target: _scalar(episode[source]) for source, target in _EPISODE_KEYS.items() if source in episode}
    if "return_sum" not in selected and "r" in episode:
        selected["return_sum"] = float(np.asarray(episode["r"], dtype=np.float64).sum())
    return {"environment": {"episode": selected}}


def actor_timing_payload(timer_stats: dict) -> dict:
    core_keys = ("total", "sample_actions", "step_env")
    timing = {key: _scalar(timer_stats[key]) for key in core_keys if key in timer_stats}
    total = float(timing.get("total", 0.0))
    inference = float(timing.get("sample_actions", 0.0))
    if total > 0:
        timing["loop_hz"] = 1.0 / total
    if inference > 0:
        timing["inference_hz"] = 1.0 / inference
    return {"actor_timing": timing}


def learner_payload(update_info, timings, replay_buffer, correction_buffer) -> dict:
    """Partition the stable ACoB losses and operational metrics for W&B."""
    payload: dict[str, dict] = {"actor": {}, "critic": {}, "optimizer": {}}
    for key, value in dict(update_info).items():
        if key in _ACTOR_UPDATE_KEYS:
            payload["actor"][_ACTOR_UPDATE_KEYS[key]] = _scalar(value)
        elif key in _CRITIC_UPDATE_KEYS:
            payload["critic"][_CRITIC_UPDATE_KEYS[key]] = _scalar(value)
        elif key.endswith("_lr") or key in {"learning_rate", "lr"}:
            payload["optimizer"][key] = _scalar(value)
    payload = {key: value for key, value in payload.items() if value}

    correction_size = (
        correction_buffer.sampleable_size()
        if hasattr(correction_buffer, "sampleable_size")
        else len(correction_buffer)
    )
    payload["buffer"] = {
        "replay_size": len(replay_buffer),
        "correction_size": correction_size,
    }
    if hasattr(replay_buffer, "latest_replay_sampling_stats"):
        for key, value in replay_buffer.latest_replay_sampling_stats().items():
            payload["buffer"][f"replay_{key}"] = int(value)
    # Prefetch and update timings are operational metrics, not debug traces.
    payload["learner_timing"] = {key: _scalar(value) for key, value in timings.items()}
    return payload


def _finite_float(value, default: float = 0.0) -> float:
    try:
        array = np.asarray(_scalar(value), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return float(default)
    if array.size == 0 or not np.isfinite(array[0]):
        return float(default)
    return float(array[0])


def _nonnegative_int(value, default: int = 0) -> int:
    return max(0, round(_finite_float(value, float(default))))


class TrainingMetricsTracker:
    """Track the rolling success/intervention metrics reported in the paper."""

    def __init__(self, *, rolling_window: int = 20, time_fn=None):
        self.rolling_window = max(1, int(rolling_window))
        self._time_fn = time_fn or time.time
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._episode_count = 0
        self._cumulative_intervention_transitions = 0
        self._cumulative_intervention_action_steps = 0
        self._autonomous_success = deque(maxlen=self.rolling_window)
        self._assisted_success = deque(maxlen=self.rolling_window)
        self._transition_intervention_rate = deque(maxlen=self.rolling_window)
        self._action_intervention_rate = deque(maxlen=self.rolling_window)
        self._episode_duration = deque(maxlen=self.rolling_window)

    def start(self) -> None:
        with self._lock:
            if self._start_time is None:
                self._start_time = float(self._time_fn())

    def _elapsed(self) -> float | None:
        if self._start_time is None:
            return None
        return max(0.0, float(self._time_fn()) - self._start_time)

    def add_actor_payload(self, payload: dict) -> dict:
        environment = payload.get("environment")
        episode = environment.get("episode") if isinstance(environment, dict) else None
        if not isinstance(episode, dict):
            return payload
        with self._lock:
            elapsed = self._elapsed()
            if elapsed is None:
                return payload
            assisted = int(bool(_finite_float(episode.get("succeed", 0))))
            intervention_transitions = _nonnegative_int(episode.get("intervention_count", 0))
            autonomous = int(
                bool(
                    _finite_float(
                        episode.get(
                            "no_intervention_succeed",
                            int(assisted and intervention_transitions == 0),
                        )
                    )
                )
            )
            intervention_actions = _nonnegative_int(episode.get("intervention_steps", 0))
            episode_transitions = _nonnegative_int(
                episode.get("trajectory_len", episode.get("length", 0))
            )
            episode_actions = _nonnegative_int(
                episode.get("action_steps", episode_transitions)
            )
            transition_rate = _finite_float(
                episode.get(
                    "intervention_count_per_transition",
                    intervention_transitions / max(episode_transitions, 1),
                )
            )
            action_rate = _finite_float(
                episode.get(
                    "intervention_step_ratio",
                    intervention_actions / max(episode_actions, 1),
                )
            )
            duration = episode.get("duration_seconds")

            self._episode_count += 1
            self._cumulative_intervention_transitions += intervention_transitions
            self._cumulative_intervention_action_steps += intervention_actions
            self._autonomous_success.append(autonomous)
            self._assisted_success.append(assisted)
            self._transition_intervention_rate.append(transition_rate)
            self._action_intervention_rate.append(action_rate)
            if duration is not None:
                self._episode_duration.append(_finite_float(duration))

            raw = payload.setdefault("train_metric_raw", {})
            raw.update(
                {
                    "elapsed_sec": elapsed,
                    "num_episodes": self._episode_count,
                    "episode_idx": _nonnegative_int(
                        episode.get("index", self._episode_count)
                    ),
                    "actor_step": _nonnegative_int(episode.get("total_steps", 0)),
                    "auto_success": autonomous,
                    "assist_success": assisted,
                    "interv_rate_trans": transition_rate,
                    "interv_rate_action": action_rate,
                    "eff_success_trans": assisted - transition_rate,
                    "eff_success_action": assisted - action_rate,
                    "interv_trans": intervention_transitions,
                    "interv_actions": intervention_actions,
                    "ep_trans": episode_transitions,
                    "ep_actions": episode_actions,
                    "rolling_window": self.rolling_window,
                    "rolling_num_episodes": len(self._autonomous_success),
                    "rolling_is_full_window": int(
                        len(self._autonomous_success) >= self.rolling_window
                    ),
                }
            )
            if duration is not None:
                raw["duration_seconds"] = _finite_float(duration)

            assisted_rate = float(np.mean(self._assisted_success))
            transition_rolling = float(np.mean(self._transition_intervention_rate))
            action_rolling = float(np.mean(self._action_intervention_rate))
            result = payload.setdefault("train_metric_result", {})
            result.update(
                {
                    "elapsed_sec": elapsed,
                    "cum_interv_trans": self._cumulative_intervention_transitions,
                    "cum_interv_actions": self._cumulative_intervention_action_steps,
                    "auto_success_rate": float(np.mean(self._autonomous_success)),
                    "assist_success_rate": assisted_rate,
                    "interv_rate_trans": transition_rolling,
                    "interv_rate_action": action_rolling,
                    "eff_success_trans": assisted_rate - transition_rolling,
                    "eff_success_action": assisted_rate - action_rolling,
                }
            )
            if self._episode_duration:
                result["duration_seconds"] = float(np.mean(self._episode_duration))
        return payload

    def add_learner_payload(self, payload: dict, *, learner_step: int) -> dict:
        with self._lock:
            elapsed = self._elapsed()
            if elapsed is None:
                return payload
            payload.setdefault("train_metric_result", {}).update(
                {"elapsed_sec": elapsed, "learner_step": int(learner_step)}
            )
        return payload
