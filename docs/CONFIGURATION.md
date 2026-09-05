# Configuration Reference

VLA-Precision uses three user-facing YAML types:

| YAML type | Location | Purpose |
|---|---|---|
| Stage I task | `configs/stage1/<task>.yaml` | OpenPI normalization and full-parameter fine-tuning |
| Stage II task | `configs/stage2/tasks/<task>.yaml` | Task semantics, ACoB, ACoB-Stream, preprocessing, training, and evaluation |
| Stage II deployment | `configs/stage2/deployments/<deployment>.yaml` | Machine, GPU, network, camera, robot, gripper, and path locators |

A Stage I file is self-contained. Stage II resolves values in this order:

```text
schema defaults < task YAML < deployment YAML < command-line overrides
```

Keep task-dependent values in the task file and machine-dependent values in the deployment file. Use the same task/deployment pair for every process participating in one run. Relative paths are resolved from the directory in which the command is launched.

## Stage I task YAML

### Runtime

| Parameter | Meaning |
|---|---|
| `schema_version` | Configuration format version; currently `1`. |
| `cuda_visible_devices` | GPUs exposed to normalization and full-parameter training, in CUDA list syntax. |
| `xla_memory_fraction` | Maximum fraction of each visible GPU memory reserved by XLA. |
| `debug` | Writes concise resolved-configuration diagnostics; it does not change training behavior. |

### OpenPI run and model

| Parameter | Meaning |
|---|---|
| `openpi.name` | Project OpenPI `TrainConfig` profile, such as `pi05_full_finetune_ur5e`. |
| `openpi.exp_name` | Experiment name used in the checkpoint hierarchy and W&B run. |
| `openpi.initialization_checkpoint` | PI0/PI0.5 parameter source loaded before fine-tuning. |
| `openpi.model.action_dim` | Padded action dimension expected by the OpenPI model. |
| `openpi.model.action_horizon` | Number of actions predicted by one OpenPI query. |
| `openpi.model.max_token_len` | Maximum language-token sequence length. |
| `openpi.batch_size` | Global training batch size. |
| `openpi.num_train_steps` | Total optimizer steps. |
| `openpi.resume` | Continue the existing experiment directory. Cannot be enabled with `overwrite`. |
| `openpi.overwrite` | Replace the existing experiment directory. Cannot be enabled with `resume`. |
| `openpi.fsdp_devices` | Number of visible devices used by OpenPI FSDP sharding. |
| `openpi.project_name` | W&B project name. |
| `openpi.num_workers` | OpenPI data-loader worker count. |
| `openpi.ema_decay` | Exponential moving-average decay; `null` disables EMA. |
| `openpi.log_interval` | Optimizer steps between metric logs. |
| `openpi.save_interval` | Optimizer steps between checkpoint saves. |
| `openpi.keep_period` | Preserve every Nth checkpoint while intermediate checkpoints are pruned; `null` disables periodic preservation. |
| `openpi.wandb_enabled` | Enables W&B logging. |
| `openpi.seed` | OpenPI training seed. |

### Learning-rate schedule and optimizer

| Parameter | Meaning |
|---|---|
| `openpi.lr_schedule.warmup_steps` | Linear learning-rate warmup length. |
| `openpi.lr_schedule.peak_lr` | Peak learning rate reached after warmup. |
| `openpi.lr_schedule.decay_steps` | Cosine-decay duration. |
| `openpi.lr_schedule.decay_lr` | Learning rate at the end of decay. |
| `openpi.optimizer.b1` | AdamW first-moment coefficient. |
| `openpi.optimizer.b2` | AdamW second-moment coefficient. |
| `openpi.optimizer.eps` | AdamW numerical-stability term. |
| `openpi.optimizer.weight_decay` | AdamW weight decay. |
| `openpi.optimizer.clip_gradient_norm` | Global gradient-norm clipping threshold. |

### Dataset

Indices are zero-based. State and action columns are materialized once before sampling; they are not remapped inside the sampling loop.

| Parameter | Meaning |
|---|---|
| `data.lerobot_repo_id` | Hugging Face LeRobot dataset identifier. It also identifies the normalization assets. |
| `data.lerobot_root` | Optional local LeRobot dataset root; `null` uses the normal cache. |
| `data.state_key` | Source column containing the full state vector. |
| `data.state_indices` | Ordered source-state columns retained for the policy. |
| `data.action_key` | Source column containing the full action vector. |
| `data.action_indices` | Ordered source-action columns retained for the policy. |
| `data.extra_delta_transform` | Applies the robot DataConfig's additional absolute/delta action transform. |
| `data.prompt_from_task` | Uses each sample's task field as the language prompt. |
| `data.image_key_map` | Maps policy image keys to LeRobot dataset columns. |

### Paths

| Parameter | Meaning |
|---|---|
| `paths.checkpoint_root` | Stage I checkpoint root. OpenPI appends the config name, experiment name, and step. |
| `paths.cache_root` | Local dataset/transformation cache root. |
| `paths.openpi_assets_root` | OpenPI normalization-assets root. |

## Stage II task YAML

### Global and experiment identity

| Parameter | Meaning |
|---|---|
| `schema_version` | Configuration format version; currently `1`. |
| `debug` | Writes concise role-specific debug logs without changing the algorithm. |
| `strict_distributed_consistency` | `true`: compare shared configuration, complete project source/package identity, protocol revision, and OpenPI commit. `false`: compare shared configuration, protocol revision, and OpenPI commit, but not complete source/package identity. `null`: disable distributed configuration/code-identity alignment checks. Runtime shape contracts and hardware-safety checks remain active in every mode. The default is `true`. |
| `experiment.name` | Stable experiment identity used by checkpoints, logs, results, and distributed handshakes. |
| `experiment.seed` | Shared random seed. |

### Task semantics

| Parameter | Meaning |
|---|---|
| `task.instruction` | Language instruction sent to the VLA. |
| `task.reset_procedure` | Registered task-reset implementation. |
| `task.completion_detector` | Registered task-completion detector. |
| `task.reward_function` | Registered reward function. |
| `task.regrasp.enabled` | Runs the optional regrasp sequence before normal reset. |
| `task.regrasp.step_1` | First regrasp action vector; an empty list skips it. |
| `task.regrasp.step_2` | Second regrasp action vector; an empty list skips it. |
| `task.regrasp.wait_before` | Waits for operator confirmation before regrasp. |
| `task.regrasp.wait_after` | Waits for operator confirmation after regrasp. |
| `task.arm_mode` | `single` or `dual`; selects one- or two-arm observation/action structure. |
| `task.setup_mode` | Hardware layout, for example `single-arm-learned-gripper` or `dual-arm-learned-gripper`. |
| `task.image_keys` | Ordered policy image keys supplied by the camera layer. |
| `task.proprio_keys` | Ordered proprioceptive fields flattened into the policy state. |
| `task.action_horizon` | Environment actions executed per online policy query. |
| `task.completion_reward` | Terminal success reward. |
| `task.time_reward` | Per-action nonterminal reward. |
| `task.max_episode_length` | Maximum number of environment transitions in one episode. |

### High-level robot behavior

| Parameter | Meaning |
|---|---|
| `robot.kind` | Registered robot implementation (`ur5e`, `dual_ur`, or `franka`). |
| `robot.control_hz` | High-level action-chunk execution/control frequency. |
| `robot.arms.left.reset_pose` | Nominal left/single-arm TCP reset pose `[x, y, z, rx, ry, rz]`. |
| `robot.arms.left.reset_pose_range` | Left/single-arm per-axis random reset range. |
| `robot.arms.right.reset_pose` | Nominal right-arm TCP reset pose for dual-arm tasks. |
| `robot.arms.right.reset_pose_range` | Right-arm per-axis random reset range. |
| `robot.random_reset` | Independently samples each configured arm inside its `reset_pose_range`. |
| `robot.delta_action_mask` | Enables/disables action dimensions in the robot command. |
| `robot.idle_hold_enabled` | Holds the current pose when an action is effectively zero. |
| `robot.idle_position_threshold` | Translation threshold used by idle-hold detection. |
| `robot.idle_rotation_threshold` | Rotation threshold used by idle-hold detection. |
| `robot.options` | Robot-specific extension values. Current examples include `action_chunk_status_poll`, `action_chunk_done_timeout`, `wait_for_operator`, `camera_reconnect_delay`, `precision_param`, `compliance_param`, and Franka controller settings. |

`robot.server_url` is a deployment value and is described below.

### Low-level UR robot server

| Parameter | Meaning |
|---|---|
| `robot_server.calibration_verified` | Explicit acknowledgement that poses, frames, force selections, payloads, and devices were checked for this hardware. |
| `robot_server.action_fps` | Low-level RTDE action-loop frequency. |
| `robot_server.init_velocity` | UR reset/start motion velocity. |
| `robot_server.init_acceleration` | UR reset/start motion acceleration. |
| `robot_server.rt_receive_priority` | RTDE receive-thread real-time priority. |
| `robot_server.rt_control_priority` | RTDE control-thread real-time priority. |
| `robot_server.gripper_switch.enabled` | Holds the TCP while the low-level gripper changes state. |
| `robot_server.gripper_switch.min_hold_seconds` | Minimum TCP hold duration after a gripper switch. |
| `robot_server.gripper_switch.timeout_seconds` | Maximum wait for gripper stabilization. |
| `robot_server.gripper_switch.stable_window` | Number of samples used to judge gripper stability. |
| `robot_server.gripper_switch.stable_threshold` | Maximum variation considered stable. |
| `robot_server.gripper_switch.poll_seconds` | Poll interval while waiting for stability. |

The following fields exist under both `robot_server.left_arm` and `robot_server.right_arm`; a single-arm task normally defines only `left_arm`.

| Arm parameter | Meaning |
|---|---|
| `robot_server.<side>.base_pose` | Arm base pose `[x, y, z, qx, qy, qz, qw]` in the shared workspace frame. |
| `robot_server.<side>.control_frame_euler_deg` | Force-control frame Euler rotation in degrees. |
| `robot_server.<side>.selection_vector` | Six-axis force/position selection vector. |
| `robot_server.<side>.force_type` | UR force-mode type. |
| `robot_server.<side>.force_limits` | Six-axis force-mode limits. |
| `robot_server.<side>.kp`, `robot_server.<side>.kd` | Translational proportional and derivative gains. |
| `robot_server.<side>.kp_rotation`, `robot_server.<side>.kd_rotation` | Rotational proportional and derivative gains. |
| `robot_server.<side>.position_error_clip` | Translational controller-error clipping threshold. |
| `robot_server.<side>.velocity_error_clip` | Velocity controller-error clipping threshold. |
| `robot_server.<side>.rotation_error_clip_degrees` | Rotational controller-error clipping threshold in degrees. |
| `robot_server.<side>.reset_position_error_clip` | Translational error clip used during reset. |
| `robot_server.<side>.reset_rotation_error_clip_degrees` | Rotational error clip used during reset. |
| `robot_server.<side>.force_gain_scale` | Optional multiplier applied to force-control gains; `null` disables the multiplier. |
| `robot_server.<side>.controller_error_threshold` | Controller-failure/error threshold. |
| `robot_server.<side>.payload_mass` | Tool payload mass in kilograms. |
| `robot_server.<side>.payload_cog` | Tool payload center of gravity `[x, y, z]` in meters. |

The server's IPs and ports belong in the deployment YAML.

### Gripper behavior

| Parameter | Meaning |
|---|---|
| `gripper.kind` | Registered gripper implementation, such as `pgi` or `fixed`. |
| `gripper.start_position` | Default logical gripper state at episode start. |
| `gripper.left_start_position` | Optional left-arm override of `start_position`. |
| `gripper.right_start_position` | Optional right-arm override of `start_position`. |
| `gripper.options` | Implementation-specific values; current uses include Franka PGI options (`reverse`, `initialize`, `close_threshold`, `force`, `speed`). |

The following low-level fields may appear under `gripper.left` and `gripper.right`:

| Side parameter | Meaning |
|---|---|
| `gripper.<side>.force` | Gripper force command. |
| `gripper.<side>.speed` | Gripper speed command. |
| `gripper.<side>.min_position` | Device-space minimum position. |
| `gripper.<side>.max_position` | Device-space maximum position. |

Physical devices belong in the deployment YAML.

### Teleoperation behavior

| Parameter | Meaning |
|---|---|
| `teleoperation.kind` | Teleoperation implementation, for example `keyboard`; `none`/`disabled` disables intervention. |
| `teleoperation.completion_double_press_interval` | Maximum interval between completion-key presses. |
| `teleoperation.step_size_position` | Default Cartesian translation increment. |
| `teleoperation.step_size_position_alt` | Alternate translation increment selected by the mode key. |
| `teleoperation.step_size_rotation` | Default Cartesian rotation increment. |
| `teleoperation.step_size_rotation_alt` | Alternate rotation increment selected by the mode key. |
| `teleoperation.options` | Device-specific values. Dual keyboard uses per-arm step sizes and `reward_keyboard_arm`. |

Physical input-device paths belong in the deployment YAML.

### Offline dataset and preprocessing

The meanings of `data.lerobot_repo_id`, `state_key`, `state_indices`, `action_key`, `action_indices`, `extra_delta_transform`, `prompt_from_task`, and `image_key_map` are the same as in Stage I. Stage I and Stage II values must describe the same model-facing data layout.

| Parameter | Meaning |
|---|---|
| `data.preprocess.num_episodes` | Episode selection (`"all"` or a numeric string). |
| `data.preprocess.chunk_mode` | `overlap` creates every valid action window; `non_overlap` advances by one full chunk. |
| `data.preprocess.bc_only` | Marks preprocessed demonstrations as behavior-cloning data. |
| `data.preprocess.context_batch_size` | Batch size used to materialize OpenPI context/KV data. |
| `data.preprocess.workers` | CPU preprocessing worker count. |
| `data.preprocess.overwrite` | Replaces an existing preprocessed cache. |

`data.lerobot_root` is a machine/deployment value.

### OpenPI Stage II profile

| Parameter | Meaning |
|---|---|
| `openpi.model` | Model family: `pi0` or `pi05`. |
| `openpi.name` | Action-expert-only ACoB `TrainConfig` profile. |
| `openpi.initialization_config_name` | Matching full-finetune `TrainConfig` used to interpret Stage I weights and data transforms. |
| `openpi.action_dim` | Padded model action dimension. |
| `openpi.action_horizon` | Explicit model horizon; `null` follows `task.action_horizon`. |
| `openpi.max_token_len` | Maximum language-token length. |
| `openpi.num_workers` | OpenPI data worker count. |
| `openpi.fsdp_devices` | Number of devices used by OpenPI FSDP sharding. |
| `openpi.sample_steps` | Flow-matching sampling/inference steps. |
| `openpi.initialization_checkpoint` | Stage I OpenPI checkpoint-manager directory used to initialize Stage II. |
| `openpi.initialization_checkpoint_step` | Exact Stage I step; `null` selects the greatest complete step. |

`openpi.lr_schedule.*` and `openpi.optimizer.*` have the same meanings as their Stage I counterparts and configure online Actor optimization.

### ACoB objective

| Parameter | Meaning |
|---|---|
| `acob.critic_encoder` | Critic visual encoder selection. `resnet-pretrained` loads the configured ResNet-10 parameters. |
| `acob.discount` | Discount factor. |
| `acob.reward_scale` | Multiplicative reward transformation. |
| `acob.reward_bias` | Additive reward transformation. |
| `acob.dense_reward` | Enables the dense-reward return path. |
| `acob.actor_flow_weight` | Flow-matching term weight. |
| `acob.actor_improvement_weight` | Critic-guided improvement term weight. |
| `acob.actor_reference_weight` | Reference/correction regularization term weight. |
| `acob.actor_improvement_margin` | Margin used by the Actor improvement objective. |
| `acob.actor_improvement_temperature` | Temperature used by Actor improvement weighting. |
| `acob.actor_improvement_direct_delta` | Uses the direct-delta variant of the Actor improvement objective. |
| `acob.critic_vla_conservative_weight` | Conservative VLA regularizer weight for the Critic. |
| `acob.critic_intervention_preference_weight` | Weight ranking corrections above interrupted actions. |
| `acob.critic_intervention_preference_margin` | Margin for the intervention preference objective. |
| `acob.ablate_critic_preference` | Disables the Critic intervention-preference component. |
| `acob.ablate_actor_bc` | Disables the Actor behavior-cloning/reference component. |
| `acob.ablate_actor_advantage` | Disables the Actor critic-guided improvement component. |

### ACoB-Stream schedule

| Parameter | Meaning |
|---|---|
| `stream.critic_updates_per_step` | Critic updates per online Learner step. |
| `stream.correction_to_replay_ratio` | Correction samples divided by Replay samples in each batch. |
| `stream.critic_warmup_steps` | Critic-only updates before joint Actor-Critic updates. |
| `stream.publish_interval` | Learner steps between Actor-weight publications. |
| `stream.buffer_save_interval` | Newly collected transitions per persistent transition shard/save. |
| `stream.log_interval` | Learner steps between metric logging. |
| `stream.random_steps` | Initial collection steps using the random-action path. |
| `stream.training_starts` | Required transition count before online updates start. |
| `stream.max_steps` | Maximum online Learner steps. |

### Buffers

| Parameter | Meaning |
|---|---|
| `buffers.replay_capacity` | Maximum Replay Buffer transitions. |
| `buffers.correction_capacity` | Maximum Correction Buffer transitions. |
| `buffers.replay_recent_window` | Restricts Replay sampling to the most recent N transitions; `0` uses the whole buffer. |
| `buffers.replay_success_ratio` | Requested successful-transition proportion in Replay sampling. |
| `buffers.context_capacity` | Maximum Context Buffer entries containing OpenPI context/KV tensors. |
| `buffers.context_shard_size` | Context entries stored per on-disk shard; the default is `4096`, matching the established online-training layout. |

### Actor, Learner, and preprocessing runtimes

| Parameter | Meaning |
|---|---|
| `actor.environment_save_video` | Records robot-side environment video. |
| `actor.completion_detection_enabled` | Enables completion detection and completion reward. |
| `actor.wait_for_learner` | Waits for initial Learner parameters before collection. |
| `learner.batch_size` | Total online training batch size. |
| `learner.warmup_steps` | JIT warmup iterations before timed/normal training. |
| `learner.prefetch_workers` | Host-side asynchronous batch-construction workers. |
| `learner.prefetch_queue_size` | Ready-batch queue depth. |
| `learner.prefetch_max_inflight_per_kind` | Maximum concurrent requests for each sample kind. |
| `learner.prefetch_to_device` | Places prefetched batches on accelerator devices before update. |

The three roles' GPU selectors (`actor.cuda_visible_devices`, `learner.cuda_visible_devices`, and `preprocess.cuda_visible_devices`) belong in the deployment YAML.

### Checkpoints and evaluation

| Parameter | Meaning |
|---|---|
| `checkpoint.save_interval` | Learner steps between synchronized Stage II checkpoint saves. |
| `checkpoint.keep_period` | Preserve every Nth checkpoint while intermediate saves are pruned. |
| `checkpoint.resume` | Restores an existing synchronized run. Actor and Learner read the same value. |
| `checkpoint.overwrite` | Replaces an existing Stage II checkpoint run. |
| `checkpoint.resume_step` | Exact synchronized Stage II step; `0` selects the latest. |
| `evaluation.checkpoint_step` | Stage I or Stage II checkpoint step to evaluate; `0` selects the latest. |
| `evaluation.episodes` | Number of evaluation episodes. |
| `evaluation.episode_timeout_seconds` | Wall-clock timeout for one episode. |
| `evaluation.progress_interval_seconds` | Seconds between elapsed-time updates; `0` disables them. |
| `evaluation.allow_intervention` | Allows human action overrides only in VLA-Precision evaluation. |
| `evaluation.regrasp_before_reset` | Enables configured regrasp behavior during evaluation resets. |

### Logging

| Parameter | Meaning |
|---|---|
| `logging.level` | Python logging level. |
| `logging.transition_log_interval` | Robot-side transition progress interval; `0` disables it. |
| `logging.training_metrics_window` | Window used to smooth displayed/logged training metrics. |
| `logging.wandb.enabled` | Enables W&B for Stage II. |
| `logging.wandb.project` | W&B project name. |
| `logging.wandb.entity` | Optional W&B entity/team; `null` uses the active W&B account. |

## Stage II deployment YAML

Deployment files contain values tied to a computer, physical setup, or filesystem. They are merged over the selected Stage II task.

### Compute and local data

| Parameter | Meaning |
|---|---|
| `actor.cuda_visible_devices` | GPU list exposed to the online Actor. |
| `learner.cuda_visible_devices` | GPU list exposed to the online Learner. |
| `preprocess.cuda_visible_devices` | GPU list exposed while building offline Replay and Context caches. |
| `data.lerobot_root` | Optional machine-local LeRobot dataset root; `null` uses the normal cache. |

### Robot and UR service locators

| Parameter | Meaning |
|---|---|
| `robot.server_url` | Robot client's low-level service endpoint. |
| `robot_server.locators.bind_host` | Interface on which the UR HTTP service listens. Use loopback unless remote hardware access is intentionally required. |
| `robot_server.locators.bind_port` | UR HTTP service port. |
| `robot_server.locators.left_robot_ip` | Left/single UR controller IP. |
| `robot_server.locators.right_robot_ip` | Right UR controller IP; empty for a single-arm setup. |
| `robot_server.locators.dashboard_port` | UR dashboard service port. |
| `robot_server.locators.ur_cap_port` | URCap/external-control port. |

### Gripper and teleoperation locators

| Parameter | Meaning |
|---|---|
| `gripper.server_url` | Optional gripper service endpoint; `null` follows `robot.server_url`. |
| `gripper.device` | Generic/Franka serial device path. |
| `gripper.left_device` | Left/single UR gripper device path. |
| `gripper.right_device` | Right UR gripper device path. |
| `teleoperation.device` | Generic single-device locator for a custom teleoperator. |
| `teleoperation.left_device` | Left keyboard/device locator. |
| `teleoperation.right_device` | Right keyboard/device locator. |

### Cameras

| Parameter | Meaning |
|---|---|
| `cameras.capture_mode` | `sync` captures on demand; `async` serves the latest frame from a capture thread. |
| `cameras.display_image` | Shows the robot-side camera preview. |
| `cameras.devices.<image_key>.driver` | Registered camera driver. |
| `cameras.devices.<image_key>.serial_number` | Physical camera serial number or driver locator. |
| `cameras.devices.<image_key>.width` | Capture width in pixels. |
| `cameras.devices.<image_key>.height` | Capture height in pixels. |
| `cameras.devices.<image_key>.fps` | Capture frame rate. |
| `cameras.devices.<image_key>.exposure` | Manual exposure; `null` lets the driver use its default. |
| `cameras.devices.<image_key>.enabled` | Includes or skips this device. |
| `cameras.devices.<image_key>.options` | Driver-specific values. RealSense currently supports `depth`; the environment also accepts an image `crop`. |

Each enabled device key must match an entry in `task.image_keys` and `data.image_key_map`.

### Distributed networking

| Parameter | Meaning |
|---|---|
| `network.learner_host` | Address used by the Actor to reach the Learner. |
| `network.trainer_port` | AgentLace request/training port. |
| `network.data_port` | AgentLace parameter-publication/data port. |
| `network.environment_host` | Address used by the remote agent (Actor or evaluator) to reach the robot-agent Pyro bridge. Do not use `0.0.0.0` as a client destination. |
| `network.environment_bind_host` | Interface on which the robot-agent Pyro bridge listens. |
| `network.environment_port` | Robot-agent Pyro bridge port. |

### Runtime paths

| Parameter | Meaning |
|---|---|
| `paths.train_data_root` | Preprocessed transitions, saved online transitions, and Context Buffer shards. |
| `paths.checkpoint_root` | Stage II Actor/Critic checkpoint root. |
| `paths.cache_root` | Local dataset/transformation cache root. |
| `paths.output_root` | Evaluation results and optional videos. |
| `paths.openpi_assets_root` | OpenPI normalization-assets root. |
| `paths.critic_resnet10_params_path` | ResNet-10 parameter file; it is downloaded automatically when required. |

Large roots can remain repository-relative and be redirected with filesystem symlinks.
