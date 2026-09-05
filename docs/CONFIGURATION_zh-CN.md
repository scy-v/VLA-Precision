# 配置参数说明

VLA-Precision 提供三类用户配置：

| YAML 类型 | 位置 | 用途 |
|---|---|---|
| Stage I 任务 | `configs/stage1/<task>.yaml` | OpenPI normalization 与全参数微调 |
| Stage II 任务 | `configs/stage2/tasks/<task>.yaml` | 任务语义、ACoB、ACoB-Stream、预处理、训练与评估 |
| Stage II 部署 | `configs/stage2/deployments/<deployment>.yaml` | 机器、GPU、网络、相机、机器人、夹爪和路径定位信息 |

Stage I 配置是独立完整的。Stage II 按以下顺序合并配置：

```text
代码默认值 < 任务 YAML < 部署 YAML < 命令行覆盖参数
```

与任务有关的参数放在任务配置中，与具体机器和硬件有关的参数放在部署配置中。同一次运行的所有进程应使用同一组任务和部署配置。相对路径以执行命令时所在的目录为基准。

## Stage I 任务 YAML

### 运行环境

| 参数 | 作用 |
|---|---|
| `schema_version` | 配置格式版本，目前为 `1`。 |
| `cuda_visible_devices` | normalization 和全参数训练可见的 GPU，格式与 CUDA 设备列表一致。 |
| `xla_memory_fraction` | XLA 最多占用每张可见 GPU 显存的比例。 |
| `debug` | 输出简洁的最终配置诊断信息，不改变训练行为。 |

### OpenPI 实验与模型

| 参数 | 作用 |
|---|---|
| `openpi.name` | 本项目注册的 OpenPI `TrainConfig`，例如 `pi05_full_finetune_ur5e`。 |
| `openpi.exp_name` | checkpoint 目录层级和 W&B run 使用的实验名称。 |
| `openpi.initialization_checkpoint` | 全参数微调前加载的 PI0/PI0.5 参数地址。 |
| `openpi.model.action_dim` | OpenPI 模型使用的补齐后动作维度。 |
| `openpi.model.action_horizon` | OpenPI 单次推理预测的动作数量。 |
| `openpi.model.max_token_len` | 最大语言 token 序列长度。 |
| `openpi.batch_size` | 全局训练 batch size。 |
| `openpi.num_train_steps` | 优化器总更新步数。 |
| `openpi.resume` | 从已有实验目录继续训练，不能与 `overwrite` 同时开启。 |
| `openpi.overwrite` | 覆盖已有实验目录，不能与 `resume` 同时开启。 |
| `openpi.fsdp_devices` | OpenPI FSDP 使用的可见设备数量。 |
| `openpi.project_name` | W&B project 名称。 |
| `openpi.num_workers` | OpenPI dataloader worker 数量。 |
| `openpi.ema_decay` | 参数指数移动平均衰减率；`null` 表示关闭 EMA。 |
| `openpi.log_interval` | 每隔多少优化器步记录一次指标。 |
| `openpi.save_interval` | 每隔多少优化器步保存一次 checkpoint。 |
| `openpi.keep_period` | 中间 checkpoint 被清理时，永久保留每 N 步的 checkpoint；`null` 表示不做周期保留。 |
| `openpi.wandb_enabled` | 是否启用 W&B。 |
| `openpi.seed` | OpenPI 训练随机种子。 |

### 学习率与优化器

| 参数 | 作用 |
|---|---|
| `openpi.lr_schedule.warmup_steps` | 学习率线性 warmup 步数。 |
| `openpi.lr_schedule.peak_lr` | warmup 结束时的最大学习率。 |
| `openpi.lr_schedule.decay_steps` | 余弦衰减步数。 |
| `openpi.lr_schedule.decay_lr` | 衰减结束时的学习率。 |
| `openpi.optimizer.b1` | AdamW 一阶动量系数。 |
| `openpi.optimizer.b2` | AdamW 二阶动量系数。 |
| `openpi.optimizer.eps` | AdamW 数值稳定项。 |
| `openpi.optimizer.weight_decay` | AdamW 权重衰减。 |
| `openpi.optimizer.clip_gradient_norm` | 全局梯度范数裁剪阈值。 |

### 数据集

所有 indices 均从 0 开始。state 和 action 列会在采样前一次性完成映射，不会在采样循环中重复处理。

| 参数 | 作用 |
|---|---|
| `data.lerobot_repo_id` | Hugging Face LeRobot 数据集标识，同时用于标识 normalization assets。 |
| `data.lerobot_root` | 可选的本地 LeRobot 数据集根目录；`null` 使用默认缓存。 |
| `data.state_key` | 原始完整 state 向量所在的数据列。 |
| `data.state_indices` | 按给定顺序保留的原始 state 列。 |
| `data.action_key` | 原始完整 action 向量所在的数据列。 |
| `data.action_indices` | 按给定顺序保留的原始 action 列。 |
| `data.extra_delta_transform` | 是否执行对应机器人 DataConfig 的额外绝对动作/增量动作转换。 |
| `data.prompt_from_task` | 是否使用每条样本的 task 字段作为语言指令。 |
| `data.image_key_map` | policy 图像键到 LeRobot 数据列的映射。 |

### 路径

| 参数 | 作用 |
|---|---|
| `paths.checkpoint_root` | Stage I checkpoint 根目录，OpenPI 会继续添加 config 名称、实验名称和 step。 |
| `paths.cache_root` | 本地数据集与转换缓存目录。 |
| `paths.openpi_assets_root` | OpenPI normalization assets 根目录。 |

## Stage II 任务 YAML

### 全局配置与实验标识

| 参数 | 作用 |
|---|---|
| `schema_version` | 配置格式版本，目前为 `1`。 |
| `debug` | 写入简洁的角色专属 debug 日志，不改变算法。 |
| `strict_distributed_consistency` | `true`：比较共享配置、完整项目源码/包身份、通信协议版本与 OpenPI commit；`false`：比较共享配置、通信协议版本与 OpenPI commit，但不比较完整源码/包身份；`null`：关闭分布式配置和代码身份对齐检查。三种模式都会保留运行时 shape 契约与硬件安全检查。默认值为 `true`。 |
| `experiment.name` | checkpoint、日志、结果和分布式握手共同使用的稳定实验名称。 |
| `experiment.seed` | 共享随机种子。 |

### 任务语义

| 参数 | 作用 |
|---|---|
| `task.instruction` | 发送给 VLA 的语言指令。 |
| `task.reset_procedure` | 已注册的任务 reset 实现名称。 |
| `task.completion_detector` | 已注册的任务完成 detector 名称。 |
| `task.reward_function` | 已注册的 reward 函数名称。 |
| `task.regrasp.enabled` | 是否在正常 reset 前执行可选的 regrasp 流程。 |
| `task.regrasp.step_1` | 第一段 regrasp 动作向量；空列表表示跳过。 |
| `task.regrasp.step_2` | 第二段 regrasp 动作向量；空列表表示跳过。 |
| `task.regrasp.wait_before` | regrasp 前是否等待操作者确认。 |
| `task.regrasp.wait_after` | regrasp 后是否等待操作者确认。 |
| `task.arm_mode` | `single` 或 `dual`，决定单臂或双臂 observation/action 结构。 |
| `task.setup_mode` | 硬件形式，例如 `single-arm-learned-gripper` 或 `dual-arm-learned-gripper`。 |
| `task.image_keys` | 相机层提供给 policy 的有序图像键。 |
| `task.proprio_keys` | 按顺序拼接为 policy state 的本体状态字段。 |
| `task.action_horizon` | 每次在线 policy 推理后执行的环境动作数量。 |
| `task.completion_reward` | 任务成功时的终止奖励。 |
| `task.time_reward` | 每个非终止动作的奖励。 |
| `task.max_episode_length` | 单个 episode 最多包含的环境 transition 数量。 |

### 机器人高层行为

| 参数 | 作用 |
|---|---|
| `robot.kind` | 已注册的机器人实现（`ur5e`、`dual_ur` 或 `franka`）。 |
| `robot.control_hz` | 高层 action chunk 执行/控制频率。 |
| `robot.arms.left.reset_pose` | 单臂或左臂的标准 TCP reset 位姿 `[x, y, z, rx, ry, rz]`。 |
| `robot.arms.left.reset_pose_range` | 单臂或左臂 reset 位姿各维度的随机范围。 |
| `robot.arms.right.reset_pose` | 双臂任务中右臂的标准 TCP reset 位姿。 |
| `robot.arms.right.reset_pose_range` | 右臂 reset 位姿各维度的随机范围。 |
| `robot.random_reset` | 是否在各机械臂自己的 `reset_pose_range` 内独立随机采样。 |
| `robot.delta_action_mask` | 是否启用机器人命令中的各个动作维度。 |
| `robot.idle_hold_enabled` | 动作接近零时是否保持当前位置。 |
| `robot.idle_position_threshold` | 判断 idle hold 的平移阈值。 |
| `robot.idle_rotation_threshold` | 判断 idle hold 的旋转阈值。 |
| `robot.options` | 机器人实现专用扩展参数。当前示例包括 `action_chunk_status_poll`、`action_chunk_done_timeout`、`wait_for_operator`、`camera_reconnect_delay`、`precision_param`、`compliance_param` 和 Franka 控制器参数。 |

`robot.server_url` 属于部署参数，在后文说明。

### UR 底层机器人服务

| 参数 | 作用 |
|---|---|
| `robot_server.calibration_verified` | 明确确认当前硬件的位姿、坐标系、力控选择、负载和设备均已检查。 |
| `robot_server.action_fps` | 底层 RTDE 动作循环频率。 |
| `robot_server.init_velocity` | UR reset/启动运动速度。 |
| `robot_server.init_acceleration` | UR reset/启动运动加速度。 |
| `robot_server.rt_receive_priority` | RTDE 接收线程的实时优先级。 |
| `robot_server.rt_control_priority` | RTDE 控制线程的实时优先级。 |
| `robot_server.gripper_switch.enabled` | 底层夹爪切换状态时是否保持 TCP。 |
| `robot_server.gripper_switch.min_hold_seconds` | 夹爪切换后保持 TCP 的最短时间。 |
| `robot_server.gripper_switch.timeout_seconds` | 等待夹爪稳定的最长时间。 |
| `robot_server.gripper_switch.stable_window` | 判断夹爪稳定所使用的连续样本数。 |
| `robot_server.gripper_switch.stable_threshold` | 被视为稳定的最大变化量。 |
| `robot_server.gripper_switch.poll_seconds` | 等待稳定时的轮询间隔。 |

下列参数同时适用于 `robot_server.left_arm` 和 `robot_server.right_arm`；单臂任务通常只配置 `left_arm`。

| 单臂参数 | 作用 |
|---|---|
| `robot_server.<side>.base_pose` | 共享工作空间中的机械臂基座位姿 `[x, y, z, qx, qy, qz, qw]`。 |
| `robot_server.<side>.control_frame_euler_deg` | 力控坐标系的欧拉角旋转，单位为度。 |
| `robot_server.<side>.selection_vector` | 六轴力控/位置控制选择向量。 |
| `robot_server.<side>.force_type` | UR force mode 类型。 |
| `robot_server.<side>.force_limits` | 六轴 force mode 限制。 |
| `robot_server.<side>.kp`, `robot_server.<side>.kd` | 平移比例和微分增益。 |
| `robot_server.<side>.kp_rotation`, `robot_server.<side>.kd_rotation` | 旋转比例和微分增益。 |
| `robot_server.<side>.position_error_clip` | 平移控制误差裁剪阈值。 |
| `robot_server.<side>.velocity_error_clip` | 速度控制误差裁剪阈值。 |
| `robot_server.<side>.rotation_error_clip_degrees` | 旋转控制误差裁剪阈值，单位为度。 |
| `robot_server.<side>.reset_position_error_clip` | reset 期间的平移误差裁剪阈值。 |
| `robot_server.<side>.reset_rotation_error_clip_degrees` | reset 期间的旋转误差裁剪阈值。 |
| `robot_server.<side>.force_gain_scale` | 可选的力控增益倍率；`null` 表示不使用倍率。 |
| `robot_server.<side>.controller_error_threshold` | 控制器失败/误差阈值。 |
| `robot_server.<side>.payload_mass` | 工具负载质量，单位为千克。 |
| `robot_server.<side>.payload_cog` | 工具负载质心 `[x, y, z]`，单位为米。 |

服务 IP 和端口放在部署 YAML 中。

### 夹爪行为

| 参数 | 作用 |
|---|---|
| `gripper.kind` | 已注册的夹爪实现，例如 `pgi` 或 `fixed`。 |
| `gripper.start_position` | episode 开始时的默认逻辑夹爪状态。 |
| `gripper.left_start_position` | 左臂对 `start_position` 的可选覆盖值。 |
| `gripper.right_start_position` | 右臂对 `start_position` 的可选覆盖值。 |
| `gripper.options` | 实现专用参数；目前包括 Franka PGI 的 `reverse`、`initialize`、`close_threshold`、`force` 和 `speed`。 |

以下底层字段可分别配置在 `gripper.left` 和 `gripper.right` 中：

| 单侧参数 | 作用 |
|---|---|
| `gripper.<side>.force` | 夹爪力命令。 |
| `gripper.<side>.speed` | 夹爪速度命令。 |
| `gripper.<side>.min_position` | 设备坐标中的最小位置。 |
| `gripper.<side>.max_position` | 设备坐标中的最大位置。 |

物理设备放在部署 YAML 中。

### 遥操作行为

| 参数 | 作用 |
|---|---|
| `teleoperation.kind` | 遥操作实现，例如 `keyboard`；`none`/`disabled` 表示关闭人工干预。 |
| `teleoperation.completion_double_press_interval` | 两次任务完成按键之间允许的最长间隔。 |
| `teleoperation.step_size_position` | 默认笛卡尔平移增量。 |
| `teleoperation.step_size_position_alt` | 通过模式按键切换的另一组平移增量。 |
| `teleoperation.step_size_rotation` | 默认笛卡尔旋转增量。 |
| `teleoperation.step_size_rotation_alt` | 通过模式按键切换的另一组旋转增量。 |
| `teleoperation.options` | 设备专用参数；双键盘使用左右臂各自的步长和 `reward_keyboard_arm`。 |

物理输入设备路径放在部署 YAML 中。

### 离线数据与预处理

`data.lerobot_repo_id`、`state_key`、`state_indices`、`action_key`、`action_indices`、`extra_delta_transform`、`prompt_from_task` 和 `image_key_map` 与 Stage I 含义相同。Stage I 与 Stage II 必须描述相同的模型输入输出数据布局。

| 参数 | 作用 |
|---|---|
| `data.preprocess.num_episodes` | episode 选择范围，可为 `"all"` 或表示数量的字符串。 |
| `data.preprocess.chunk_mode` | `overlap` 生成所有有效动作窗口；`non_overlap` 每次前进一个完整 chunk。 |
| `data.preprocess.bc_only` | 将预处理 demonstration 标记为仅用于行为克隆的数据。 |
| `data.preprocess.context_batch_size` | 生成 OpenPI context/KV 数据时的 batch size。 |
| `data.preprocess.workers` | CPU 预处理 worker 数量。 |
| `data.preprocess.overwrite` | 是否覆盖已有的预处理缓存。 |

`data.lerobot_root` 属于机器/部署参数。

### OpenPI Stage II 配置

| 参数 | 作用 |
|---|---|
| `openpi.model` | 模型系列：`pi0` 或 `pi05`。 |
| `openpi.name` | 仅训练 action expert 的 ACoB `TrainConfig`。 |
| `openpi.initialization_config_name` | 与之匹配的全参数微调 `TrainConfig`，用于解释 Stage I 权重和数据转换。 |
| `openpi.action_dim` | 模型使用的补齐后动作维度。 |
| `openpi.action_horizon` | 显式模型 horizon；`null` 表示跟随 `task.action_horizon`。 |
| `openpi.max_token_len` | 最大语言 token 长度。 |
| `openpi.num_workers` | OpenPI 数据 worker 数量。 |
| `openpi.fsdp_devices` | OpenPI FSDP 使用的设备数量。 |
| `openpi.sample_steps` | flow matching 采样/推理步数。 |
| `openpi.initialization_checkpoint` | 初始化 Stage II 的 Stage I OpenPI checkpoint-manager 目录。 |
| `openpi.initialization_checkpoint_step` | 指定 Stage I step；`null` 自动选择最大的完整 step。 |

`openpi.lr_schedule.*` 和 `openpi.optimizer.*` 与 Stage I 中同名参数含义一致，用于在线 Actor 优化。

### ACoB 目标函数

| 参数 | 作用 |
|---|---|
| `acob.critic_encoder` | Critic 视觉编码器；`resnet-pretrained` 会加载配置的 ResNet-10 参数。 |
| `acob.discount` | 折扣因子。 |
| `acob.reward_scale` | reward 乘法变换系数。 |
| `acob.reward_bias` | reward 加法变换偏置。 |
| `acob.dense_reward` | 是否使用 dense reward return 分支。 |
| `acob.actor_flow_weight` | flow matching 损失权重。 |
| `acob.actor_improvement_weight` | Critic 引导改进损失权重。 |
| `acob.actor_reference_weight` | reference/correction 正则项权重。 |
| `acob.actor_improvement_margin` | Actor 改进目标使用的 margin。 |
| `acob.actor_improvement_temperature` | Actor 改进加权使用的 temperature。 |
| `acob.actor_improvement_direct_delta` | 是否使用 Actor 改进目标的 direct-delta 形式。 |
| `acob.critic_vla_conservative_weight` | Critic conservative VLA 正则项权重。 |
| `acob.critic_intervention_preference_weight` | 让 correction 排在被干预动作之上的目标权重。 |
| `acob.critic_intervention_preference_margin` | intervention preference 目标的 margin。 |
| `acob.ablate_critic_preference` | 是否关闭 Critic intervention preference 部分。 |
| `acob.ablate_actor_bc` | 是否关闭 Actor 行为克隆/reference 部分。 |
| `acob.ablate_actor_advantage` | 是否关闭 Actor 的 Critic 引导改进部分。 |

### ACoB-Stream 调度

| 参数 | 作用 |
|---|---|
| `stream.critic_updates_per_step` | 每个在线 Learner step 执行的 Critic 更新次数。 |
| `stream.correction_to_replay_ratio` | 单个 batch 中 Correction 样本数与 Replay 样本数之比。 |
| `stream.critic_warmup_steps` | Actor-Critic 联合更新前执行的 Critic-only 更新数。 |
| `stream.publish_interval` | 每隔多少 Learner step 发布一次 Actor 权重。 |
| `stream.buffer_save_interval` | 每收集多少条新 transition 写入一个持久化 shard/存盘。 |
| `stream.log_interval` | 每隔多少 Learner step 记录一次指标。 |
| `stream.random_steps` | 初始使用随机动作采集的步数。 |
| `stream.training_starts` | 开始在线更新前要求的 transition 数量。 |
| `stream.max_steps` | 最大在线 Learner 更新步数。 |

### Buffer

| 参数 | 作用 |
|---|---|
| `buffers.replay_capacity` | Replay Buffer 最大 transition 数量。 |
| `buffers.correction_capacity` | Correction Buffer 最大 transition 数量。 |
| `buffers.replay_recent_window` | Replay 只从最近 N 条 transition 中采样；`0` 表示使用整个 buffer。 |
| `buffers.replay_success_ratio` | Replay 采样时期望的成功 transition 比例。 |
| `buffers.context_capacity` | 保存 OpenPI context/KV tensor 的 Context Buffer 最大条目数。 |
| `buffers.context_shard_size` | 每个磁盘 shard 保存的 context 条目数；默认 `4096`，与原在线训练布局一致。 |

### Actor、Learner 与预处理运行参数

| 参数 | 作用 |
|---|---|
| `actor.environment_save_video` | 是否录制机器人侧环境视频。 |
| `actor.completion_detection_enabled` | 是否启用任务完成检测和完成奖励。 |
| `actor.wait_for_learner` | 采集前是否等待第一份 Learner 参数。 |
| `learner.batch_size` | 在线训练总 batch size。 |
| `learner.warmup_steps` | 正常训练/计时前执行的 JIT warmup 次数。 |
| `learner.prefetch_workers` | 主机端异步构造 batch 的 worker 数量。 |
| `learner.prefetch_queue_size` | 已准备 batch 的队列深度。 |
| `learner.prefetch_max_inflight_per_kind` | 每种采样类型允许同时处理的最大请求数。 |
| `learner.prefetch_to_device` | 是否在更新前将预取 batch 放到加速器。 |

三种角色的 GPU 参数 `actor.cuda_visible_devices`、`learner.cuda_visible_devices` 和 `preprocess.cuda_visible_devices` 均放在部署 YAML 中。

### Checkpoint 与评估

| 参数 | 作用 |
|---|---|
| `checkpoint.save_interval` | 每隔多少 Learner step 保存一次同步 Stage II checkpoint。 |
| `checkpoint.keep_period` | 中间 checkpoint 被清理时，永久保留每 N 步的 checkpoint。 |
| `checkpoint.resume` | 恢复已有的同步训练；Actor 和 Learner 读取同一个值。 |
| `checkpoint.overwrite` | 覆盖已有 Stage II checkpoint 实验。 |
| `checkpoint.resume_step` | 指定同步 Stage II step；`0` 自动选择最新 step。 |
| `evaluation.checkpoint_step` | 需要评估的 Stage I 或 Stage II checkpoint step；`0` 自动选择最新 step。 |
| `evaluation.episodes` | 评估 episode 数量。 |
| `evaluation.episode_timeout_seconds` | 单个 episode 的真实时间上限。 |
| `evaluation.progress_interval_seconds` | 每隔多少秒更新一次已用时间；`0` 表示关闭。 |
| `evaluation.allow_intervention` | 是否允许在 VLA-Precision 评估中用人工动作覆盖 policy 动作。 |
| `evaluation.regrasp_before_reset` | 评估 reset 时是否启用已配置的 regrasp。 |

### 日志

| 参数 | 作用 |
|---|---|
| `logging.level` | Python 日志级别。 |
| `logging.transition_log_interval` | 机器人侧 transition 进度打印间隔；`0` 表示关闭。 |
| `logging.training_metrics_window` | 训练指标显示/记录时的平滑窗口。 |
| `logging.wandb.enabled` | 是否为 Stage II 启用 W&B。 |
| `logging.wandb.project` | W&B project 名称。 |
| `logging.wandb.entity` | 可选的 W&B entity/team；`null` 使用当前登录账号。 |

## Stage II 部署 YAML

部署配置保存与具体计算机、实体硬件或文件系统绑定的参数，并覆盖所选 Stage II 任务配置中的同名值。

### 计算资源与本地数据

| 参数 | 作用 |
|---|---|
| `actor.cuda_visible_devices` | 在线 Actor 可见的 GPU 列表。 |
| `learner.cuda_visible_devices` | 在线 Learner 可见的 GPU 列表。 |
| `preprocess.cuda_visible_devices` | 构建离线 Replay 和 Context 缓存时可见的 GPU 列表。 |
| `data.lerobot_root` | 可选的机器本地 LeRobot 数据集根目录；`null` 使用默认缓存。 |

### 机器人与 UR 服务定位参数

| 参数 | 作用 |
|---|---|
| `robot.server_url` | 机器人客户端访问的底层服务地址。 |
| `robot_server.locators.bind_host` | UR HTTP 服务监听的网卡地址；除非确实需要远程硬件访问，否则应使用 loopback。 |
| `robot_server.locators.bind_port` | UR HTTP 服务端口。 |
| `robot_server.locators.left_robot_ip` | 左臂/单臂 UR 控制器 IP。 |
| `robot_server.locators.right_robot_ip` | 右臂 UR 控制器 IP；单臂配置为空字符串。 |
| `robot_server.locators.dashboard_port` | UR dashboard 服务端口。 |
| `robot_server.locators.ur_cap_port` | URCap/external-control 端口。 |

### 夹爪与遥操作设备定位参数

| 参数 | 作用 |
|---|---|
| `gripper.server_url` | 可选的夹爪服务地址；`null` 跟随 `robot.server_url`。 |
| `gripper.device` | 通用/Franka 串口设备路径。 |
| `gripper.left_device` | 左臂/单臂 UR 夹爪设备路径。 |
| `gripper.right_device` | 右臂 UR 夹爪设备路径。 |
| `teleoperation.device` | 自定义单设备遥操作器的通用设备路径。 |
| `teleoperation.left_device` | 左键盘/设备路径。 |
| `teleoperation.right_device` | 右键盘/设备路径。 |

### 相机

| 参数 | 作用 |
|---|---|
| `cameras.capture_mode` | `sync` 按需采集；`async` 通过采集线程持续提供最新帧。 |
| `cameras.display_image` | 是否显示机器人侧相机预览。 |
| `cameras.devices.<image_key>.driver` | 已注册的相机驱动名称。 |
| `cameras.devices.<image_key>.serial_number` | 实体相机序列号或对应驱动使用的设备定位信息。 |
| `cameras.devices.<image_key>.width` | 采集宽度，单位为像素。 |
| `cameras.devices.<image_key>.height` | 采集高度，单位为像素。 |
| `cameras.devices.<image_key>.fps` | 相机采集帧率。 |
| `cameras.devices.<image_key>.exposure` | 手动曝光值；`null` 使用驱动默认值。 |
| `cameras.devices.<image_key>.enabled` | 是否启用当前设备。 |
| `cameras.devices.<image_key>.options` | 驱动专用参数；RealSense 当前支持 `depth`，环境层还支持图像 `crop`。 |

每个启用的设备键都应与 `task.image_keys` 和 `data.image_key_map` 中的键对应。

### 分布式网络

| 参数 | 作用 |
|---|---|
| `network.learner_host` | Actor 访问 Learner 使用的地址。 |
| `network.trainer_port` | AgentLace 请求/训练端口。 |
| `network.data_port` | AgentLace 参数发布/数据端口。 |
| `network.environment_host` | 远程智能体（Actor或评估器）访问机器人—智能体 Pyro 通信桥的地址；客户端不能连接 `0.0.0.0`。 |
| `network.environment_bind_host` | 机器人—智能体 Pyro 通信桥监听的网卡地址。 |
| `network.environment_port` | 机器人—智能体 Pyro 通信桥端口。 |

### 运行路径

| 参数 | 作用 |
|---|---|
| `paths.train_data_root` | 预处理 transition、在线持久化 transition 和 Context Buffer shard 的根目录。 |
| `paths.checkpoint_root` | Stage II Actor/Critic checkpoint 根目录。 |
| `paths.cache_root` | 本地数据集与转换缓存目录。 |
| `paths.output_root` | 评估结果和可选视频目录。 |
| `paths.openpi_assets_root` | OpenPI normalization assets 根目录。 |
| `paths.critic_resnet10_params_path` | ResNet-10 参数文件；首次需要时自动下载。 |

大容量目录可以继续使用仓库内相对路径，并通过文件系统软连接重定向到其他磁盘。
