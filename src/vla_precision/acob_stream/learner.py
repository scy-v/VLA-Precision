"""Server-side ACoB-Stream learner loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tqdm.auto import tqdm

from vla_precision.acob_stream.buffers.replay import (
    make_prefetched_online_batch_iterator,
    online_batch_split_counts,
)
from vla_precision.acob_stream.communication import (
    CORRECTION_STORE_NAME,
    REPLAY_STORE_NAME,
    actor_parameter_payload,
)
from vla_precision.acob_stream.interfaces import (
    ActorParameterPublisher,
    CheckpointSaver,
    LearnerAgent,
    LearnerMetricsSink,
    LearnerTrainerServer,
)
from vla_precision.acob_stream.jax_runtime import (
    JaxLearnerBackend,
    RuntimeMesh,
    compile_learner_updates,
)
from vla_precision.acob_stream.metrics import Timer
from vla_precision.config.loader import ResolvedConfig
from vla_precision.logging import log_status

LOGGER = logging.getLogger(__name__)
PrefetchFactory = Callable[[tuple[str, ...]], tuple[Callable, Callable, Callable]]
InitialBufferLoader = Callable[[Any, Any, int], None]
InitialBufferSaver = Callable[[Any, Any, int], None]


@dataclass
class LearnerRuntime:
    resolved: ResolvedConfig
    agent: LearnerAgent
    replay_buffer: Any
    correction_buffer: Any
    trainer_server: LearnerTrainerServer
    mesh: RuntimeMesh
    backend: Any = field(default_factory=JaxLearnerBackend)
    parameter_payload: ActorParameterPublisher = actor_parameter_payload
    prefetch_factory: PrefetchFactory | None = None
    checkpoint_saver: CheckpointSaver | None = None
    checkpoint_waiter: Callable[[], None] | None = None
    metrics_sink: LearnerMetricsSink | None = None
    metrics_starter: Callable[[], None] | None = None
    initial_buffer_loader: InitialBufferLoader | None = None
    initial_buffer_saver: InitialBufferSaver | None = None
    resumed_step: int | None = None
    sleep: Callable[[float], None] = time.sleep
    compile_warmup: bool = True


class LearnerLoop:
    def __init__(self, runtime: LearnerRuntime):
        self.runtime = runtime
        self.agent = runtime.agent

    def _correction_size(self) -> int:
        buffer = self.runtime.correction_buffer
        return int(buffer.sampleable_size() if hasattr(buffer, "sampleable_size") else len(buffer))

    def _publish(self, step: int | str) -> None:
        self.agent = self.runtime.backend.block_until_ready(self.agent)
        self.runtime.trainer_server.publish_network(self.runtime.parameter_payload(self.agent))
        if step == "init":
            message = "Initial Actor parameters published"
        elif step == "ready":
            message = "Buffer-ready Actor parameters published"
        else:
            message = f"Actor parameters published: learner step {step}"
        log_status(LOGGER, "Parameter sync", message, color="yellow")

    def _prefetch(self, active_kinds: tuple[str, ...]):
        if self.runtime.prefetch_factory is not None:
            return self.runtime.prefetch_factory(active_kinds)
        cfg = self.runtime.resolved.config
        return make_prefetched_online_batch_iterator(
            self.runtime.replay_buffer,
            self.runtime.correction_buffer,
            batch_size=cfg.learner.batch_size,
            device=self.runtime.mesh.data_sharding,
            queue_size=cfg.learner.prefetch_queue_size,
            num_workers=cfg.learner.prefetch_workers,
            max_inflight_per_kind=cfg.learner.prefetch_max_inflight_per_kind,
            cta_ratio=cfg.stream.critic_updates_per_step,
            correction_ratio=cfg.stream.correction_to_replay_ratio,
            replay_success_ratio=cfg.buffers.replay_success_ratio,
            replay_recent_window=cfg.buffers.replay_recent_window,
            prefetch_to_device=cfg.learner.prefetch_to_device,
            prefetch_device=self.runtime.mesh.data_sharding if cfg.learner.prefetch_to_device else None,
            active_kinds=active_kinds,
        )

    def _emit_metrics(self, step: int, info, timings: dict[str, float]) -> None:
        if self.runtime.metrics_sink is not None:
            self.runtime.metrics_sink(step, info, timings)

    def run(self):
        cfg = self.runtime.resolved.config
        start_step = 0 if self.runtime.resumed_step is None else int(self.runtime.resumed_step) + 1
        warmup_steps = 0 if self.runtime.resumed_step is not None else max(0, cfg.stream.critic_warmup_steps)
        critic_networks = frozenset({"critic"})
        all_networks = frozenset({"critic", "actor"})

        self.runtime.trainer_server.register_data_store(REPLAY_STORE_NAME, self.runtime.replay_buffer)
        self.runtime.trainer_server.register_data_store(CORRECTION_STORE_NAME, self.runtime.correction_buffer)
        self.runtime.trainer_server.start(threaded=True)

        # Publish once immediately so a waiting actor can initialize while the
        # learner waits for enough online data.
        self._publish("init")
        _, correction_batch_size = online_batch_split_counts(
            cfg.learner.batch_size,
            cfg.stream.correction_to_replay_ratio,
        )
        correction_min_size = correction_batch_size
        if warmup_steps > 0:
            correction_min_size = max(correction_min_size, 2 * correction_batch_size)

        if self.runtime.initial_buffer_loader is not None and (
            len(self.runtime.replay_buffer) < cfg.stream.training_starts
            or self._correction_size() < correction_min_size
        ):
            self.runtime.initial_buffer_loader(
                self.runtime.replay_buffer,
                self.runtime.correction_buffer,
                correction_min_size,
            )

        log_status(
            LOGGER,
            "Buffer status",
            (
                f"Replay {len(self.runtime.replay_buffer)}/{cfg.stream.training_starts} | "
                f"Correction {self._correction_size()}/{correction_min_size}"
            ),
            color="cyan",
        )
        with tqdm(
            total=cfg.stream.training_starts,
            initial=min(len(self.runtime.replay_buffer), cfg.stream.training_starts),
            desc="Replay buffer",
            dynamic_ncols=True,
        ) as progress:
            while len(self.runtime.replay_buffer) < cfg.stream.training_starts:
                size = min(len(self.runtime.replay_buffer), cfg.stream.training_starts)
                progress.update(size - progress.n)
                self.runtime.sleep(1.0)
            progress.update(cfg.stream.training_starts - progress.n)
        with tqdm(
            total=correction_min_size,
            initial=min(self._correction_size(), correction_min_size),
            desc="Correction buffer",
            dynamic_ncols=True,
        ) as progress:
            while self._correction_size() < correction_min_size:
                size = min(self._correction_size(), correction_min_size)
                progress.update(size - progress.n)
                self.runtime.sleep(1.0)
            progress.update(correction_min_size - progress.n)

        log_status(
            LOGGER,
            "Buffer status",
            (
                f"Replay ready {len(self.runtime.replay_buffer)}/{cfg.stream.training_starts} | "
                f"Correction ready {self._correction_size()}/{correction_min_size}"
            ),
            color="green",
        )

        if self.runtime.initial_buffer_saver is not None and correction_min_size > 0:
            self.runtime.initial_buffer_saver(
                self.runtime.replay_buffer,
                self.runtime.correction_buffer,
                correction_min_size,
            )

        # Republish after the fill barrier; this is the actor's definitive
        # starting parameter set for online collection.
        self._publish("ready")

        next_batch, latest_stats, close_prefetch = self._prefetch(("critic", "full"))
        try:
            if self.runtime.compile_warmup:
                compile_started = time.time()
                log_status(LOGGER, "JIT compilation", "Learner updates started", color="magenta")
                self.agent = compile_learner_updates(
                    runtime=self.runtime.mesh,
                    backend=self.runtime.backend,
                    agent=self.agent,
                    next_critic_batch=lambda: next_batch(kind="critic"),
                    next_full_batch=lambda: next_batch(kind="full"),
                    critic_networks=critic_networks,
                    all_networks=all_networks,
                    full_steps=cfg.learner.warmup_steps,
                    critic_updates_per_step=cfg.stream.critic_updates_per_step,
                )
                log_status(
                    LOGGER,
                    "JIT compilation",
                    f"Learner updates finished: {time.time() - compile_started:.2f} s",
                    color="magenta",
                )

            if self.runtime.metrics_starter is not None:
                self.runtime.metrics_starter()

            online_start = start_step
            timer = Timer()
            if warmup_steps > 0:
                log_status(
                    LOGGER,
                    "Critic warmup",
                    f"Started: {warmup_steps} learner steps",
                    color="cyan",
                )
                close_prefetch()
                next_batch, latest_stats, close_prefetch = self._prefetch(("critic",))
                warmup_stop = min(cfg.stream.max_steps, start_step + warmup_steps)
                for step in tqdm(
                    range(start_step, warmup_stop),
                    desc="Critic warmup",
                    dynamic_ncols=True,
                ):
                    step_started = time.time()
                    last_info = {}
                    critic_sample_seconds = 0.0
                    critic_update_seconds = 0.0
                    critic_prefetch_stats: dict[str, float] = {}
                    for _ in range(cfg.stream.critic_updates_per_step):
                        sample_started = time.time()
                        with timer.context("critic_warmup_sample"):
                            batch = next_batch(kind="critic")
                        critic_sample_seconds += time.time() - sample_started
                        for name, value in latest_stats().items():
                            critic_prefetch_stats[name] = (
                                critic_prefetch_stats.get(name, 0.0) + float(value)
                            )
                        update_started = time.time()
                        with timer.context("critic_warmup_update"):
                            self.agent, last_info = self.runtime.backend.update(
                                self.runtime.mesh,
                                self.agent,
                                batch,
                                networks=critic_networks,
                            )
                        critic_update_seconds += time.time() - update_started
                    timings = {
                        "sample_critic_batch": critic_sample_seconds,
                        "sample_actor_batch": 0.0,
                        "critic_update": critic_update_seconds,
                        "actor_update": 0.0,
                        "publish": 0.0,
                        "save": 0.0,
                        "total": time.time() - step_started,
                        **{
                            f"sample_critic_{name}": value
                            for name, value in critic_prefetch_stats.items()
                        },
                    }
                    if cfg.stream.log_interval > 0 and step % cfg.stream.log_interval == 0:
                        self._emit_metrics(step, last_info, timings)
                online_start = warmup_stop
                log_status(
                    LOGGER,
                    "Critic warmup",
                    f"Finished: {warmup_steps} learner steps",
                    color="green",
                )
                close_prefetch()
                next_batch, latest_stats, close_prefetch = self._prefetch(("critic", "full"))

            online_stop = cfg.stream.max_steps
            log_status(
                LOGGER,
                "Online training",
                f"Started: learner step {online_start}",
                color="cyan",
            )

            for step in tqdm(
                range(online_start, online_stop),
                desc="Online training",
                dynamic_ncols=True,
            ):
                step_started = time.time()
                critic_sample_seconds = 0.0
                critic_update_seconds = 0.0
                critic_prefetch_stats: dict[str, float] = {}
                for _ in range(max(0, cfg.stream.critic_updates_per_step - 1)):
                    started = time.time()
                    batch = next_batch(kind="critic")
                    critic_sample_seconds += time.time() - started
                    for name, value in latest_stats().items():
                        critic_prefetch_stats[name] = (
                            critic_prefetch_stats.get(name, 0.0) + float(value)
                        )
                    started = time.time()
                    self.agent, _ = self.runtime.backend.update(
                        self.runtime.mesh,
                        self.agent,
                        batch,
                        networks=critic_networks,
                    )
                    critic_update_seconds += time.time() - started

                started = time.time()
                batch = next_batch(kind="full")
                full_sample_seconds = time.time() - started
                actor_prefetch_stats = latest_stats()
                started = time.time()
                self.agent, update_info = self.runtime.backend.update(
                    self.runtime.mesh,
                    self.agent,
                    batch,
                    networks=all_networks,
                )
                actor_update_seconds = time.time() - started

                started = time.time()
                if cfg.stream.publish_interval > 0 and step > 0 and step % cfg.stream.publish_interval == 0:
                    self._publish(step)
                publish_seconds = time.time() - started

                started = time.time()
                if (
                    self.runtime.checkpoint_saver is not None
                    and cfg.checkpoint.save_interval > 0
                    and step > 0
                    and step % cfg.checkpoint.save_interval == 0
                ):
                    log_status(LOGGER, "Checkpoint save", f"Started: learner step {step}", color="yellow")
                    self.runtime.checkpoint_saver(self.agent, step)
                    log_status(
                        LOGGER,
                        "Checkpoint save",
                        f"Actor submitted, Critic saved: learner step {step}",
                        color="yellow",
                    )
                save_seconds = time.time() - started
                timings = {
                    "sample_critic_batch": critic_sample_seconds,
                    "sample_actor_batch": full_sample_seconds,
                    "critic_update": critic_update_seconds,
                    "actor_update": actor_update_seconds,
                    "publish": publish_seconds,
                    "save": save_seconds,
                    "total": time.time() - step_started,
                    **{
                        f"sample_critic_{name}": value
                        for name, value in critic_prefetch_stats.items()
                    },
                    **{
                        f"sample_actor_{name}": value
                        for name, value in actor_prefetch_stats.items()
                    },
                }
                if cfg.stream.log_interval > 0 and step % cfg.stream.log_interval == 0:
                    self._emit_metrics(step, update_info, timings)
        finally:
            close_prefetch()
            self.runtime.trainer_server.stop()

        if self.runtime.checkpoint_waiter is not None:
            self.runtime.checkpoint_waiter()
        return self.agent


def run_learner(resolved: ResolvedConfig, *, runtime: LearnerRuntime | None = None):
    if runtime is None:
        from vla_precision.acob_stream.setup import build_learner_runtime

        runtime = build_learner_runtime(resolved)
    LOGGER.info("starting ACoB-Stream learner")
    return LearnerLoop(runtime).run()
