# VLA-Precision 代码结构

本文面向需要阅读或修改源码的开发者，说明仓库中各目录、源码文件及其主要调用关系。安装、训练与评估命令见根目录 `README_zh.md`；YAML 参数见 `CONFIGURATION_zh-CN.md`；添加新硬件或任务见 `EXTENDING_zh-CN.md`。

## 1. 总体分层

```text
main.py
└── vla_precision.cli
    ├── config/                 配置模型、加载与命令行覆盖
    ├── integrations/openpi/    OpenPI 配置、训练、模型和原生推理扩展
    ├── data/                   数据索引物化与 Stage II 预处理
    ├── acob/                   ACoB 算法、Critic 和 JAX 训练状态
    ├── acob_stream/            Actor/Learner 在线循环、通信和三个 Buffer
    ├── robotics/               机器人、相机、夹爪、遥操作和环境
    └── evaluation.py           Stage I/II 统一环境评估
```

核心依赖方向如下：

```text
robotics ──提供环境──> acob_stream <──使用算法── acob
                              │
                              └──使用模型── integrations/openpi

config 和 data 为以上模块提供统一配置与预处理数据。
```

`acob/` 只关心算法更新；`acob_stream/` 负责在线系统如何运行；`integrations/openpi/` 隔离第三方 VLA 接口；`robotics/` 隔离具体硬件。新增设备时不应把厂商 SDK 写入 ACoB 或 OpenPI 模块。

## 2. 仓库根目录

| 路径 | 作用 | 主要使用者 |
| --- | --- | --- |
| `main.py` | 用户启动入口，将参数交给 `vla_precision.cli.main`。 | 所有命令 |
| `README.md` | 英文安装、训练与评估指南。 | 用户 |
| `README_zh.md` | 中文安装、训练与评估指南。 | 用户 |
| `.gitignore` | 排除虚拟环境、缓存、训练产物、日志和本地设备文件。 | Git |
| `pyproject.toml` | 包元数据、Stage I/Stage II/真机依赖组、固定第三方版本和工具配置。 | `uv`、Hatch、pytest、Ruff |
| `uv.lock` | 可复现的完整依赖锁文件。 | `uv sync --frozen` |
| `configs/` | 用户可编辑的 Stage I、Stage II 任务和部署配置。 | CLI 配置加载器 |
| `docs/` | 中英文代码结构、配置和扩展说明。 | 用户、开发者 |
| `src/vla_precision/` | 可安装的项目源码包。 | 所有运行模式 |
| `tests/unit/` | 不连接真机的行为与边界回归测试；不参与训练和推理。 | 开发者、CI |

## 3. 配置文件 `configs/`

| 路径 | 作用 | 加载位置 |
| --- | --- | --- |
| `configs/stage1/<task>.yaml` | 一个 Stage I 全参微调任务的完整配置。目录中每个任务文件具有相同结构。 | `config.loader.load_stage1_config` |
| `configs/stage2/tasks/<task>.yaml` | Stage II 的任务、模型、算法、Buffer、训练和评估参数。 | `config.loader.load_config` |
| `configs/stage2/deployments/single_ur.yaml` | 单臂 UR 的网络、GPU、设备和本地路径配置。 | `config.loader.load_config` |
| `configs/stage2/deployments/dual_ur.yaml` | 双臂 UR 的网络、GPU、设备和本地路径配置。 | `config.loader.load_config` |
| `configs/stage2/deployments/example.yaml` | 新部署配置的最小参考模板。 | 用户复制后由 `load_config` 加载 |

Stage I 只加载一个任务 YAML。Stage II 将任务 YAML 与一个部署 YAML 合并，再应用命令行覆盖。每一项参数的含义和覆盖顺序见配置说明文档。

## 4. 包级公共文件 `src/vla_precision/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `__init__.py` | 包说明与版本号。 | Python 包系统、运行身份检查 |
| `cli.py` | 解析公开的 `--stage/--mode/--role` 接口，加载配置，并调度训练、评估或服务入口。 | 根目录 `main.py` |
| `evaluation.py` | 通过同一 Pyro 环境评估 Stage I 全参模型或 Stage II ACoB 模型；逐回合写入结果。 | `cli.py --mode evaluate` |
| `logging.py` | 控制台颜色、debug 文件、终端转存和关键状态日志。 | CLI、Actor、Learner、服务 |
| `runtime_identity.py` | 生成并比较分布式进程的协议、OpenPI 版本和源码身份。 | Pyro 环境握手 |
| `image_tools.py` | 图像转 `uint8` 及保持长宽比的补边缩放。 | 数据预处理、机器人观测 |
| `serialization.py` | 递归地将 Base64 编码的 `.npy` 载荷解码为 NumPy 数据。 | 机器人 HTTP/Pyro 边界 |

各子包中的 `__init__.py` 只声明包边界或导出稳定的公共对象，不包含独立运行流程。
其中 `integrations/__init__.py` 声明第三方集成边界，`robotics/__init__.py` 声明可组合硬件边界。

## 5. 配置实现 `config/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `schema.py` | 定义 Stage I `Stage1Config`、Stage II `RootConfig` 及其所有分组 dataclass。 | `loader.py`、全项目类型接口 |
| `loader.py` | 合并默认值、任务 YAML、部署 YAML 和 CLI 覆盖；执行核心边界检查并生成配置哈希。 | `cli.py` |
| `__init__.py` | 导出配置类型和加载函数。 | `cli.py`、各运行模块 |

## 6. 数据准备 `data/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `indexing.py` | 在 DataLoader 创建前，将 `state_indices` 和 `action_indices` 一次性物化到 Hugging Face/LeRobot 数据集。 | Stage I loader、Stage II preprocess |
| `preprocess.py` | 将 LeRobot episode 转成可直接训练的 action chunk、transition 和 OpenPI Context Buffer；支持多进程图像解码和多设备 context 编码。 | `cli.py --stage stage2 --mode preprocess` |
| `paths.py` | 从统一配置计算 LeRobot 根目录、预处理数据、在线数据、初始 Buffer 和 checkpoint 路径。 | preprocess、Actor、Learner、评估 |
| `__init__.py` | 说明数据层“不在采样热路径重做 indices”的约束。 | 包导入 |

## 7. ACoB 算法 `acob/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `agent.py` | `ACoBState`、`ACoBAgent`、Actor/Critic loss、动作采样、critic-only 与 joint update。 | `factory.py`、Learner、Actor |
| `factory.py` | 根据 `RootConfig`、OpenPI TrainConfig 和观测/动作空间构建 ACoB Agent。 | `acob_stream.setup` |
| `networks.py` | Value/Advantage Critic、MLP、时间编码和 ensemble 网络组件。 | `agent.py` |
| `vision.py` | ResNet 编码器、预训练编码器、FiLM 和空间特征模块。 | `agent.py`、`networks.py` |
| `augmentation.py` | 随机裁剪、颜色、模糊等 Critic 图像增强。 | `agent.py` |
| `training.py` | Optimizer、batch sharding、模块容器及 `JaxRLTrainState`。 | `agent.py`、OpenPI 训练边界 |
| `pretrained_resnet.py` | 下载、校验并装载 Critic 的预训练 ResNet-10 权重。 | `agent.py`（由 `factory.py` 触发构建） |
| `types.py` | ACoB 使用的数组、batch 和 PyTree 类型别名。 | ACoB 内部模块 |
| `__init__.py` | 延迟导出 Agent，避免仅使用轻量模块时加载完整训练依赖。 | 外部导入 |

## 8. 在线框架 `acob_stream/`

### 8.1 主运行文件

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `actor.py` | 在线采集与推理主循环：接收参数、编码当前 context、推理 action chunk、处理远程环境返回的人工干预并提交 episode。 | `cli.py --role actor` |
| `learner.py` | 服务器侧主循环：注册 Buffer、warmup Critic、执行 critic/joint update、发布参数并保存 checkpoint。 | `cli.py --role learner` |
| `setup.py` | 从同一 `ResolvedConfig` 组装真实 Actor/Learner runtime、Agent、Buffer、通信和日志对象。 | `run_actor`、`run_learner` |
| `jax_runtime.py` | JAX mesh、sharding、Actor 推理 backend、Learner 更新 backend 和 JIT 预编译。 | `setup.py`、主循环 |
| `communication.py` | Pyro 环境客户端、AgentLace Trainer client/server，以及 Actor 参数序列化与替换。 | `setup.py`、Actor、Learner |
| `interfaces.py` | Environment、Buffer、Agent 和通信对象的最小 Protocol，便于测试和替换实现。 | `actor.py`、`learner.py` |
| `checkpoints.py` | 在同一 step 保存/恢复 OpenPI Actor（含 LoRA）、Actor/Critic 优化器和 Critic 状态；恢复时重建 reference policy。 | `setup.py`、Learner |
| `metrics.py` | Timer、Actor/Learner 指标整理和 W&B 训练指标追踪。 | Actor、Learner |
| `trajectories.py` | 计算 chunk return、绑定 next context、提交 episode，并原子读写 transition shard。 | Actor、preprocess、resume |
| `preprocessed_data.py` | 检查并加载 Stage II preprocess 生成的 metadata 与 transition。 | `setup.py`、Learner |
| `initial_buffers.py` | 保存/恢复最初收集的 Replay、Correction 与对应 Context，并重映射 context source。 | Learner 启动流程 |
| `__init__.py` | 声明 ACoB-Stream 包。 | 包导入 |

### 8.2 三个 Buffer `acob_stream/buffers/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `replay.py` | Replay Buffer 公开实现、context hydration、Replay/Correction 混合采样和后台预取。 | Learner、AgentLace 数据接收 |
| `correction.py` | 继承 Replay Buffer 的稀疏人工修正 Buffer，保持独立的公开概念。 | Actor、Learner |
| `context.py` | mmap/shard 支持的 Context Buffer，持久化 OpenPI prefix KV、mask、有效长度和存储元数据；`ContextSource` 区分 transition 引用的 preprocess、training 与 initial 存储。 | preprocess、Actor、Learner sampler |
| `_dataset.py` | 嵌套 NumPy dataset 的长度、选取与采样基础。 | 底层 Replay Buffer |
| `_replay_base.py` | 环形写入、索引和 iterator 的基础 Replay Buffer。 | `_memory_replay.py` |
| `_memory_replay.py` | 面向图像观测的内存高效存储实现。 | `_data_store.py` |
| `_data_store.py` | 将 Replay Buffer 适配成 AgentLace DataStore，并提供线程安全插入。 | `replay.py`、通信层 |
| `__init__.py` | 延迟导出 Replay/Correction，直接导出 Context 类型。 | `setup.py`、外部导入 |

## 9. OpenPI 扩展 `integrations/openpi/`

OpenPI 由 `pyproject.toml` 固定到指定 Git commit。本目录只保存项目扩展和适配，不修改上游源码。

### 9.1 训练、配置与数据

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `configs.py` | 以 OpenPI 原生风格声明项目 TrainConfig，并把顶层 YAML 值覆盖到 Stage I/II 配置对象。 | norm stats、Stage I 训练、Stage II setup、评估 |
| `data_configs.py` | 定义 UR5e、双 UR5e 和 Franka 的 OpenPI `DataConfigFactory`。 | `configs.py` |
| `policies/ur5e.py` | 单臂 UR 的 OpenPI 输入/输出 transforms 和示例 observation。 | 对应 DataConfig、Policy |
| `policies/dual_ur.py` | 双臂 UR 的 OpenPI 输入/输出 transforms。 | 对应 DataConfig、Policy |
| `policies/franka.py` | Franka 的 OpenPI 输入/输出 transforms。 | 对应 DataConfig、Policy |
| `policies/__init__.py` | 导出三类机器人 policy transforms。 | `data_configs.py` |
| `data_loader.py` | 在采样前完成 indices 物化，并构建兼容 OpenPI 的 dataset 和 Torch DataLoader。 | Stage I 训练、norm stats |
| `train.py` | Stage I 全参微调：初始化状态、JIT train step、W&B、checkpoint 和训练循环。 | `cli.py --stage stage1 --mode train` |
| `norm_stats.py` | 按 OpenPI 数据变换计算并保存 normalization statistics。 | `cli.py --mode norm-stats` |
| `training_state.py` | 在 OpenPI 内部 API 与 ACoB 之间集中构建 TrainState。 | `acob.factory`、Stage I 训练 |
| `checkpoints.py` | 解析 Stage I/II OpenPI checkpoint 及 assets 目录，处理 step 选择。 | setup、评估、原生推理 |
| `adapter.py` | 在机器人 observation/action 与 OpenPI transform tree 之间转换。 | preprocess、Actor、评估 |
| `context.py` | 编码 OpenPI prefix context，并从缓存 context 进行 action sampling。 | preprocess、ACoB Agent |
| `lerobot_compat.py` | 为固定 OpenPI 与固定 LeRobot 版本安装导入名称兼容层。 | OpenPI 扩展模块导入时 |
| `__init__.py` | 保持 OpenPI 包边界轻量，并安装 LeRobot 导入兼容。 | 所有 OpenPI 扩展 |

### 9.2 原生 OpenPI 推理 `integrations/openpi/inference/`

这条路径不经过 ACoB-Stream 或 Pyro，用于直接评估 Stage I 模型；policy 可在机器人电脑本地运行，也可连接 OpenPI WebSocket policy server。

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `config.py` | 读取原生推理 YAML，生成 policy 和硬件运行配置。 | `runtime.py` |
| `runtime.py` | 选择硬件 backend，创建本地/远程 policy，启动推理或 WebSocket policy server。 | CLI 的原生推理模式 |
| `policy.py` | 构造 OpenPI observation，并提供统一 sampler。 | 原生推理、统一评估 |
| `configs/ur.yaml` | 单臂 UR 原生推理预设。 | `config.py` |
| `configs/dual_ur.yaml` | 双臂 UR 原生推理预设。 | `config.py` |
| `backends/ur.py` | 单臂 UR 的相机、控制、episode 和人工操作推理流程。 | `runtime.py` |
| `backends/dual_ur.py` | 双臂 UR 推理流程。 | `runtime.py` |
| `backends/franka.py` | Franka 推理流程。 | `runtime.py` |
| `backends/recording.py` | 后台保存视频与 episode 记录。 | 各硬件 backend |
| `backends/plots.py` | 从 episode 日志生成轨迹与状态图。 | 记录分析工具 |
| `backends/utils.py` | normalization 维度检查和 FPS 统计等共享工具。 | 各硬件 backend |
| `__init__.py`、`backends/__init__.py` | 导出公开入口并声明原生推理边界。 | 包导入 |

## 10. 机器人系统 `robotics/`

### 10.1 Robot `robotics/robots/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `base.py` | 定义厂商无关的 `Robot` Protocol。 | Environment、扩展实现 |
| `factory.py` | 按配置构建 UR 或 Franka Robot。 | Environment factory |
| `ur.py` | 通过 HTTP 调用独立 UR server，读取状态并发送 action chunk。 | `RobotEnvironment` |
| `franka.py` | 通过 ZeroRPC/Polymetis 调用 Franka 并完成状态/动作转换。 | `RobotEnvironment` |
| `__init__.py` | 导出 Robot 接口、factory 和实现。 | robotics 内部 |

### 10.2 Camera `robotics/cameras/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `base.py` | 定义厂商无关的 `Camera` Protocol。 | Environment、扩展实现 |
| `factory.py` | 从部署配置创建和关闭相机集合。 | Environment factory |
| `realsense.py` | Intel RealSense 的帧采集实现。 | Camera factory |
| `latest_frame.py` | 异步读取底层相机，只向控制循环提供最新帧。 | Camera factory、Environment |
| `__init__.py` | 导出 Camera 接口与实现。 | robotics 内部 |

### 10.3 Gripper `robotics/grippers/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `base.py` | 定义厂商无关的 `Gripper` Protocol。 | Environment、扩展实现 |
| `factory.py` | 根据机器人和 setup mode 构建固定夹爪、UR 夹爪或 Franka PGI。 | Environment factory |
| `ur.py` | 通过 UR HTTP server 操作 PGI，或提供固定夹爪实现。 | `RobotEnvironment` |
| `franka.py` | Franka 使用的异步串口 PGI 实现。 | `RobotEnvironment` |
| `__init__.py` | 导出 Gripper 接口、factory 和实现。 | robotics 内部 |

### 10.4 Teleoperation `robotics/teleoperation/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `base.py` | 定义键盘、VR、同构遥操作可共同实现的 `TeleoperationDevice` Protocol。 | intervention wrapper、未来设备扩展 |
| `single_keyboard.py` | 单臂键盘动作输入和状态。 | Keyboard intervention |
| `dual_keyboard.py` | 双设备/双臂键盘动作输入和线程同步。 | Keyboard intervention |
| `keyboard.py` | 键盘急停与任务完成事件 detector。 | Environment factory、任务 detector |
| `__init__.py` | 导出遥操作公共接口和键盘实现。 | robotics 内部 |

### 10.5 Task logic `robotics/tasks/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `completion.py` | 独立的任务完成 detector 接口及事件实现。 | Completion reward wrapper |
| `reward.py` | 独立的 RewardFunction 接口和 chunk completion reward。 | Completion reward wrapper |
| `reset.py` | 各任务 reset procedure，保留机器人、夹爪和提示的既有执行顺序。 | Environment factory、regrasp wrapper |
| `__init__.py` | 导出 completion、reward 和 reset 构建入口。 | Environment factory |

### 10.6 Gymnasium Wrapper `robotics/wrappers/`

Wrapper 负责组合处理顺序；具体完成判断、reward 和遥操作输入仍位于各自模块。

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `action_chunk.py` | 将 OpenPI action chunk 适配为逐动作环境执行，并组织 chunk observation/reward。 | Environment factory |
| `keyboard_intervention.py` | 将遥操作输入应用到 action chunk，处理接管、急停和实际执行动作。 | Environment factory |
| `relative_frame.py` | 在 reset-relative 坐标系与机器人绝对坐标系之间转换。 | Environment factory |
| `observations.py` | 四元数/欧拉角及扁平 observation 适配。 | Environment factory |
| `completion_reward.py` | 将独立 completion detector 和 reward function 接入 Gymnasium step。 | Environment factory |
| `regrasp.py` | 在特定任务 reset 前后组合 regrasp 流程。 | Environment factory |
| `__init__.py` | 导出 Wrapper 类型。 | Environment factory |

### 10.7 Environment `robotics/environments/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `robot.py` | 组合 Robot、Camera、Gripper 和 detector 的基础 Gymnasium 真机环境。 | `factory.py` |
| `factory.py` | 按固定顺序构建基础环境及 Wrapper stack。 | Pyro environment server、测试 |
| `server.py` | 通过 Pyro 桥接本地机器人系统与远程智能体（Actor或评估器），处理 reset、step 和 episode 统计。 | `cli.py --mode robot-agent-bridge`、Actor |
| `fake.py` | 无硬件 Robot/Camera，用于 Learner 构建空间和单元测试。 | setup、tests |
| `__init__.py` | 导出 Environment factory 和基础环境。 | robotics 内部 |

### 10.8 低层 UR 服务 `robotics/servers/ur/`

| 文件 | 作用 | 主要调用者 |
| --- | --- | --- |
| `server.py` | UR RTDE force-mode 控制、状态读取、动作队列、夹爪线程、HTTP 路由和安全退出。 | `cli.py --mode serve-robot`、`robots/ur.py` |
| `pgi.py` | UR 服务内使用的 PGI 串口夹爪驱动。 | `server.py` |
| `rotations.py` | rotvec、quaternion 和 Euler pose 转换。 | UR controller |
| `__init__.py` | 导出 `run_robot_server`。 | CLI |

`robotics/servers/__init__.py` 声明独立低层服务包。低层服务与 Gym environment 分开启动，避免在算法进程中直接持有硬件 SDK。

## 11. 单元测试 `tests/unit/`

这些文件只用于开发验证，不会被正常训练、评估或服务启动导入。

| 文件 | 覆盖范围 |
| --- | --- |
| `test_main.py`、`test_cli.py` | 根入口、stage/mode 规范化和调度。 |
| `test_config.py`、`test_task_presets.py` | 配置合并、覆盖、哈希和任务配置可加载性。 |
| `test_data_indexing.py`、`test_lerobot_root.py`、`test_preprocess.py`、`test_preprocessed_data.py` | indices 物化、数据根目录和 preprocess 产物。 |
| `test_acob_algorithm_parity.py`、`test_acob_factory.py`、`test_resnet_asset.py` | ACoB 数值行为、Agent 构建和预训练视觉权重。 |
| `test_replay_buffers.py`、`test_context_buffer.py`、`test_initial_buffers.py` | Replay、Correction、Context 及初始 Buffer 持久化。 |
| `test_stream_actor.py`、`test_stream_learner.py`、`test_stream_communication.py`、`test_stream_metrics.py`、`test_stream_trajectories.py` | Actor/Learner 主循环、通信、指标和 episode/transition。 |
| `test_acob_checkpoints.py` | Stage II Actor/Critic checkpoint 保存和恢复。 |
| `test_openpi_extension.py`、`test_openpi_training_state.py`、`test_openpi_train.py`、`test_openpi_checkpoints.py`、`test_openpi_spawn.py` | OpenPI 扩展、TrainState、训练、checkpoint 和进程导入边界。 |
| `test_openpi_native_inference.py`、`test_evaluation.py` | 原生 OpenPI 推理与统一评估。 |
| `test_robotics_components.py`、`test_environment_wrappers.py`、`test_ur_robot_server.py` | 机器人组件、Wrapper 顺序与 UR 服务控制契约。 |
| `test_image_tools.py`、`test_logging.py`、`test_runtime_identity.py` | 图像工具、日志和分布式运行身份。 |

## 12. 主要调用链

### Stage I normalization 与全参微调

```text
main.py -> cli.py -> load_stage1_config
  ├── norm-stats -> integrations/openpi/norm_stats.py
  └── train      -> integrations/openpi/train.py
                       ├── configs.py + data_configs.py
                       ├── data_loader.py
                       └── OpenPI model/checkpoint API
```

### Stage II 预处理

```text
main.py -> cli.py -> load_config
  -> data/preprocess.py
       ├── data/indexing.py
       ├── integrations/openpi/adapter.py + context.py
       └── preprocess transitions + Context Buffer
```

### Stage II 在线训练

```text
机器人电脑                          GPU 服务器
serve-robot                        learner.py
    │                                  │
robot-agent-bridge <────── Pyro ──── actor.py
    │                                  │
robotics wrapper stack           AgentLace Buffer/parameters
                                       │
                                  acob/agent.py
```

Actor 与 Learner 使用同一任务和部署配置，但只读取各自职责所需字段。Pyro 负责 Actor 与机器人环境交互，AgentLace 负责 Actor/Learner 间的 transition、correction 和参数传输。

### 评估

```text
统一环境评估：cli.py -> evaluation.py -> OpenPI sampler -> Pyro environment
原生 OpenPI：cli.py -> inference/runtime.py -> hardware backend
                                      └── local policy 或 WebSocket policy
```

## 13. 运行时生成目录

| 目录 | 内容 | 是否源码 |
| --- | --- | --- |
| `assets/` | normalization statistics、自动下载的 ResNet 权重及 OpenPI assets。 | 否 |
| `checkpoints/` | Stage I 和 Stage II checkpoint；可软连接到大容量磁盘。 | 否 |
| `train_data/` | preprocess cache、transition shard 与 Context Buffer；可软连接到大容量磁盘。 | 否 |
| `.cache/` | Hugging Face、JAX/OpenPI 等项目级缓存。 | 否 |
| `debug/` | 开启唯一 `debug` 参数后生成的分角色日志。 | 否 |
| `results/` | 逐回合追加的评估结果。 | 否 |
| `wandb/` | W&B 本地运行记录。 | 否 |
| `.venv/` | `uv` 管理的 Python 环境。 | 否 |

这些目录不定义算法行为，不应从其中导入 Python 源码。
