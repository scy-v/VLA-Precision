"""The single public configuration schema for VLA-Precision.

YAML contains values that users may change. Python-only OpenPI objects are
constructed later by ``vla_precision.integrations.openpi`` from these values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42


@dataclass(frozen=True)
class RegraspConfig:
    """Optional pre-reset regrasp sequence used by the three insert tasks."""

    enabled: bool = False
    step_1: tuple[float, ...] = ()
    step_2: tuple[float, ...] = ()
    wait_before: bool = False
    wait_after: bool = False


@dataclass(frozen=True)
class TaskConfig:
    instruction: str = ""
    reset_procedure: str = "standard_ur"
    completion_detector: str = "manual"
    reward_function: str = "chunk_completion"
    regrasp: RegraspConfig = field(default_factory=RegraspConfig)
    arm_mode: str = "single"
    setup_mode: str = "single-arm-learned-gripper"
    image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")
    proprio_keys: tuple[str, ...] = (
        "tcp_pose",
        "tcp_vel",
        "tcp_force",
        "tcp_torque",
        "gripper_pose",
    )
    action_horizon: int = 3
    completion_reward: float = 10.0
    time_reward: float = -0.01
    max_episode_length: int = 120


@dataclass(frozen=True)
class RobotArmConfig:
    """One arm's nominal reset pose and independent randomization range."""

    reset_pose: tuple[float, ...] = ()
    reset_pose_range: tuple[float, ...] = ()


@dataclass(frozen=True)
class RobotArmsConfig:
    left: RobotArmConfig = field(default_factory=RobotArmConfig)
    right: RobotArmConfig = field(default_factory=RobotArmConfig)


@dataclass(frozen=True)
class RobotConfig:
    kind: str = "ur5e"
    server_url: str = "http://127.0.0.1:5001/"
    control_hz: float = 10.0
    arms: RobotArmsConfig = field(default_factory=RobotArmsConfig)
    random_reset: bool = True
    delta_action_mask: tuple[float, ...] = ()
    idle_hold_enabled: bool = True
    idle_position_threshold: float = 0.0001
    idle_rotation_threshold: float = 0.001
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class URGripperSwitchConfig:
    """TCP hold behavior while a low-level gripper changes state."""

    enabled: bool = True
    min_hold_seconds: float = 0.15
    timeout_seconds: float = 0.8
    stable_window: int = 8
    stable_threshold: float = 0.005
    poll_seconds: float = 0.02


@dataclass(frozen=True)
class URLowLevelGripperConfig:
    """Low-level PGI gripper control parameters."""

    force: int = 78
    speed: int = 70
    min_position: int = 80
    max_position: int = 1000


@dataclass(frozen=True)
class URArmServerConfig:
    """One arm's force controller, reset, payload and end-effector facts."""

    base_pose: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    control_frame_euler_deg: tuple[float, ...] = (0.0, 0.0, 0.0)
    selection_vector: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    force_type: int = 2
    force_limits: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
    kp: float = 1200.0
    kd: float = 600.0
    kp_rotation: float = 2000.0
    kd_rotation: float = 400.0
    position_error_clip: float = 0.048
    velocity_error_clip: float = 4.0
    rotation_error_clip_degrees: float = 20.0
    reset_position_error_clip: float = 0.06
    reset_rotation_error_clip_degrees: float = 20.0
    force_gain_scale: float | None = 1.5
    controller_error_threshold: float = 0.3
    payload_mass: float = 0.0
    payload_cog: tuple[float, ...] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class URServerLocatorConfig:
    """Machine-local addresses and device locators excluded from shared hashes."""

    bind_host: str = "127.0.0.1"
    bind_port: int = 5001
    left_robot_ip: str = ""
    right_robot_ip: str = ""
    dashboard_port: int = 29999
    ur_cap_port: int = 50002


@dataclass(frozen=True)
class URRobotServerConfig:
    """Typed configuration for the single/dual UR Flask + RTDE service."""

    calibration_verified: bool = False
    action_fps: float = 100.0
    init_velocity: float = 0.2
    init_acceleration: float = 0.2
    rt_receive_priority: int = 90
    rt_control_priority: int = 85
    gripper_switch: URGripperSwitchConfig = field(default_factory=URGripperSwitchConfig)
    left_arm: URArmServerConfig = field(default_factory=URArmServerConfig)
    right_arm: URArmServerConfig = field(default_factory=URArmServerConfig)
    locators: URServerLocatorConfig = field(default_factory=URServerLocatorConfig)


@dataclass(frozen=True)
class CameraDeviceConfig:
    driver: str = "realsense"
    serial_number: str = ""
    width: int = 640
    height: int = 480
    fps: int = 15
    exposure: int | None = None
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraConfig:
    capture_mode: str = "sync"
    display_image: bool = True
    devices: dict[str, CameraDeviceConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class GripperConfig:
    kind: str = "pgi"
    server_url: str | None = None
    device: str | None = None
    left_device: str | None = None
    right_device: str | None = None
    start_position: float = 1.0
    left_start_position: float | None = None
    right_start_position: float | None = None
    left: URLowLevelGripperConfig = field(default_factory=URLowLevelGripperConfig)
    right: URLowLevelGripperConfig = field(default_factory=URLowLevelGripperConfig)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeleoperationConfig:
    kind: str = "keyboard"
    device: str | None = None
    left_device: str | None = None
    right_device: str | None = None
    completion_double_press_interval: float = 0.5
    step_size_position: float = 0.04
    step_size_position_alt: float = 0.015
    step_size_rotation: float = 0.05
    step_size_rotation_alt: float = 0.045
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessConfig:
    num_episodes: str = "all"
    chunk_mode: str = "non_overlap"
    bc_only: bool = True
    context_batch_size: int = 64
    workers: int = 4
    overwrite: bool = False


@dataclass(frozen=True)
class Stage1DataConfig:
    """Dataset fields shared by OpenPI normalization and training."""

    lerobot_repo_id: str = ""
    lerobot_root: str | None = None
    state_key: str = "observation.state"
    state_indices: tuple[int, ...] = ()
    action_key: str = "action"
    action_indices: tuple[int, ...] = ()
    extra_delta_transform: bool = False
    prompt_from_task: bool = True
    image_key_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DataConfig(Stage1DataConfig):
    """Stage-II dataset fields, including offline cache materialization."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


@dataclass(frozen=True)
class OpenPIModelOverrides:
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200


@dataclass(frozen=True)
class OpenPILRScheduleConfig:
    warmup_steps: int = 10_000
    peak_lr: float = 5.0e-5
    decay_steps: int = 1_000_000
    decay_lr: float = 5.0e-5


@dataclass(frozen=True)
class OpenPIOptimizerConfig:
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1.0e-8
    weight_decay: float = 1.0e-10
    clip_gradient_norm: float = 1.0


@dataclass(frozen=True)
class Stage1OpenPIConfig:
    name: str = "pi05_full_finetune_ur5e"
    exp_name: str = "experiment"
    initialization_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    model: OpenPIModelOverrides = field(default_factory=OpenPIModelOverrides)
    batch_size: int = 64
    num_train_steps: int = 30_000
    resume: bool = False
    overwrite: bool = False
    fsdp_devices: int = 1
    project_name: str = "vla-precision"
    num_workers: int = 2
    lr_schedule: OpenPILRScheduleConfig = field(default_factory=OpenPILRScheduleConfig)
    optimizer: OpenPIOptimizerConfig = field(default_factory=OpenPIOptimizerConfig)
    ema_decay: float | None = 0.999
    log_interval: int = 100
    save_interval: int = 10_000
    keep_period: int | None = 5_000
    wandb_enabled: bool = True
    seed: int = 42


@dataclass(frozen=True)
class Stage1PathConfig:
    checkpoint_root: str = "./checkpoints/stage1"
    cache_root: str = "./.cache"
    openpi_assets_root: str = "./assets"


@dataclass(frozen=True)
class Stage1Config:
    schema_version: int = 1
    cuda_visible_devices: str = "0,1,2,3"
    xla_memory_fraction: float = 0.9
    debug: bool = False
    openpi: Stage1OpenPIConfig = field(default_factory=Stage1OpenPIConfig)
    data: Stage1DataConfig = field(default_factory=Stage1DataConfig)
    paths: Stage1PathConfig = field(default_factory=Stage1PathConfig)


@dataclass(frozen=True)
class Stage2OpenPIConfig:
    model: str = "pi05"
    name: str = "pi05_acob_ur5e"
    initialization_config_name: str = "pi05_full_finetune_ur5e"
    action_dim: int = 32
    action_horizon: int | None = None
    max_token_len: int = 200
    num_workers: int = 2
    fsdp_devices: int = 1
    sample_steps: int = 10
    initialization_checkpoint: str | None = None
    initialization_checkpoint_step: int | None = None
    lr_schedule: OpenPILRScheduleConfig = field(
        default_factory=lambda: OpenPILRScheduleConfig(
            warmup_steps=1_000,
            peak_lr=1.0e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        )
    )
    optimizer: OpenPIOptimizerConfig = field(default_factory=OpenPIOptimizerConfig)


@dataclass(frozen=True)
class ACoBConfig:
    critic_encoder: str = "resnet-pretrained"
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    dense_reward: bool = False
    actor_flow_weight: float = 0.25
    actor_improvement_weight: float = 0.5
    actor_reference_weight: float = 0.25
    actor_improvement_margin: float = 0.0
    actor_improvement_temperature: float = 0.05
    actor_improvement_direct_delta: bool = False
    critic_vla_conservative_weight: float = 0.0
    critic_intervention_preference_weight: float = 50.0
    critic_intervention_preference_margin: float = 0.05
    ablate_critic_preference: bool = False
    ablate_actor_bc: bool = False
    ablate_actor_advantage: bool = False


@dataclass(frozen=True)
class ACoBStreamConfig:
    critic_updates_per_step: int = 2
    correction_to_replay_ratio: float = 1.0
    critic_warmup_steps: int = 100
    publish_interval: int = 10
    buffer_save_interval: int = 1_000
    log_interval: int = 10
    random_steps: int = 0
    training_starts: int = 100
    max_steps: int = 1_000_000


@dataclass(frozen=True)
class BufferConfig:
    replay_capacity: int = 200_000
    correction_capacity: int = 200_000
    replay_recent_window: int = 10_600
    replay_success_ratio: float = 0.0
    context_capacity: int = 200_000
    context_shard_size: int = 4096


@dataclass(frozen=True)
class ActorRuntimeConfig:
    cuda_visible_devices: str = "0"
    environment_save_video: bool = False
    completion_detection_enabled: bool = True
    wait_for_learner: bool = True


@dataclass(frozen=True)
class PreprocessRuntimeConfig:
    cuda_visible_devices: str = "0,1,2,3"


@dataclass(frozen=True)
class LearnerRuntimeConfig:
    cuda_visible_devices: str = "0,1"
    batch_size: int = 128
    warmup_steps: int = 1
    prefetch_workers: int = 6
    prefetch_queue_size: int = 2
    prefetch_max_inflight_per_kind: int = 2
    prefetch_to_device: bool = True


@dataclass(frozen=True)
class CheckpointConfig:
    save_interval: int = 250
    keep_period: int = 250
    resume: bool = False
    overwrite: bool = False
    resume_step: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    """Checkpoint-rollout settings shared by baseline and ACoB evaluation."""

    checkpoint_step: int = 0
    episodes: int = 10
    episode_timeout_seconds: float = 60.0
    progress_interval_seconds: float = 5.0
    allow_intervention: bool = False
    regrasp_before_reset: bool = True


@dataclass(frozen=True)
class WandBConfig:
    enabled: bool = True
    project: str = "vla-precision"
    entity: str | None = None


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    transition_log_interval: int = 10
    training_metrics_window: int = 20
    wandb: WandBConfig = field(default_factory=WandBConfig)


@dataclass(frozen=True)
class NetworkConfig:
    learner_host: str = "127.0.0.1"
    trainer_port: int = 3335
    data_port: int = 3336
    environment_host: str = "127.0.0.1"
    environment_bind_host: str = "0.0.0.0"
    environment_port: int = 50001


@dataclass(frozen=True)
class PathConfig:
    train_data_root: str = "./data/train"
    checkpoint_root: str = "./checkpoints"
    cache_root: str = "./.cache"
    output_root: str = "./results"
    openpi_assets_root: str = "./assets"
    critic_resnet10_params_path: str = "./assets/resnet10_params.pkl"


@dataclass(frozen=True)
class RootConfig:
    schema_version: int = 1
    debug: bool = False
    strict_distributed_consistency: bool | None = True
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    robot_server: URRobotServerConfig = field(default_factory=URRobotServerConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    teleoperation: TeleoperationConfig = field(default_factory=TeleoperationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    openpi: Stage2OpenPIConfig = field(default_factory=Stage2OpenPIConfig)
    acob: ACoBConfig = field(default_factory=ACoBConfig)
    stream: ACoBStreamConfig = field(default_factory=ACoBStreamConfig)
    buffers: BufferConfig = field(default_factory=BufferConfig)
    actor: ActorRuntimeConfig = field(default_factory=ActorRuntimeConfig)
    preprocess: PreprocessRuntimeConfig = field(default_factory=PreprocessRuntimeConfig)
    learner: LearnerRuntimeConfig = field(default_factory=LearnerRuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    paths: PathConfig = field(default_factory=PathConfig)
