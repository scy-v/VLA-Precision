"""Helpers shared by the native OpenPI inference backends."""

import time
import logging
from pathlib import Path

import numpy as np


def _norm_stat_dim(norm_stats, keys: tuple[str, ...]) -> int | None:
    if norm_stats is None:
        return None
    for key in keys:
        stats = norm_stats.get(key)
        if stats is None:
            continue
        mean = stats.get("mean") if isinstance(stats, dict) else getattr(stats, "mean", None)
        if mean is not None:
            return int(np.asarray(mean).reshape(-1).size)
    return None


def validate_single_arm_norm_stats(
    norm_stats,
    *,
    fixed_gripper: bool,
    learned_state_dim: int,
    fixed_state_dim: int = 18,
    path: Path | str | None = None,
) -> None:
    """Reject checkpoints whose norm stats do not match inference observations/actions."""
    expected_state_dim = int(fixed_state_dim) if fixed_gripper else int(learned_state_dim)
    expected_action_dim = 6 if fixed_gripper else 7
    state_dim = _norm_stat_dim(norm_stats, ("state", "observation/state"))
    action_dim = _norm_stat_dim(norm_stats, ("actions", "action"))
    if state_dim != expected_state_dim or action_dim != expected_action_dim:
        raise ValueError(
            f"Single-arm fixed_gripper={fixed_gripper} inference requires {expected_state_dim}-D "
            f"state and {expected_action_dim}-D action norm stats, got state={state_dim}, "
            f"action={action_dim} in {path}. Fixed-gripper checkpoints must be trained from "
            "datasets that omit gripper state and action fields."
        )


class FpsCounter:
    """Utility class to measure the frequency of a loop"""

    def __init__(self, name: str, log_interval: float = 1.0):
        self.name = name
        self.last_time = time.perf_counter()

    def reset(self):
        """Reset the timer to the current time."""
        self.last_time = time.perf_counter()
        
    def update(self, show: bool = True):
        """Compute and log the frequency when show is enabled."""
        if not show:
            return
        now = time.perf_counter()
        fps = 1.0 / (now - self.last_time)
        self.last_time = now
        logging.info(f"[DEBUG] {self.name} loop actual frequency: {fps:.2f} Hz")

