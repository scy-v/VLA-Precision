# VLA-Precision Code Structure

This document is for developers who need to read or modify the source code. It describes the repository directories, source files, and their primary call paths. See the root `README.md` for installation, training, and evaluation commands; `CONFIGURATION.md` for YAML fields; and `EXTENDING.md` for adding hardware or tasks.

## 1. Architectural layers

```text
main.py
└── vla_precision.cli
    ├── config/                 schemas, loading, and CLI overrides
    ├── integrations/openpi/    OpenPI configs, training, models, and native inference
    ├── data/                   index materialization and Stage II preprocessing
    ├── acob/                   ACoB algorithm, critic, and JAX training state
    ├── acob_stream/            online actor/learner loops, communication, and buffers
    ├── robotics/               robots, cameras, grippers, teleoperation, and environments
    └── evaluation.py           shared Stage I/II environment evaluation
```

The main dependency direction is:

```text
robotics ──provides environment──> acob_stream <──uses algorithm── acob
                                          │
                                          └──uses model── integrations/openpi

config and data provide unified configuration and preprocessed data.
```

`acob/` owns algorithm updates, `acob_stream/` owns the online system, `integrations/openpi/` isolates the third-party VLA boundary, and `robotics/` isolates hardware. Vendor SDK code should not be added to ACoB or OpenPI modules.

## 2. Repository root

| Path | Purpose | Primary consumer |
| --- | --- | --- |
| `main.py` | User entrypoint; forwards arguments to `vla_precision.cli.main`. | All commands |
| `README.md` | English installation, training, and evaluation guide. | Users |
| `README_zh.md` | Chinese installation, training, and evaluation guide. | Users |
| `.gitignore` | Excludes virtual environments, caches, training artifacts, logs, and local device files. | Git |
| `pyproject.toml` | Package metadata, Stage I/Stage II/real-robot dependency groups, pinned sources, and tool settings. | `uv`, Hatch, pytest, Ruff |
| `uv.lock` | Fully resolved dependency lockfile. | `uv sync --frozen` |
| `configs/` | User-editable Stage I, Stage II task, and deployment configurations. | CLI configuration loader |
| `docs/` | English and Chinese code-structure, configuration, and extension guides. | Users and developers |
| `src/vla_precision/` | Installable project source package. | Every runtime mode |
| `tests/unit/` | Hardware-free behavior and boundary regression tests; never part of training or inference. | Developers and CI |

## 3. Configuration files `configs/`

| Path | Purpose | Loaded by |
| --- | --- | --- |
| `configs/stage1/<task>.yaml` | Complete configuration for one Stage I full-parameter fine-tuning task. Every task file in this directory follows the same schema. | `config.loader.load_stage1_config` |
| `configs/stage2/tasks/<task>.yaml` | Stage II task, model, algorithm, buffer, training, and evaluation values. | `config.loader.load_config` |
| `configs/stage2/deployments/single_ur.yaml` | Network, GPU, device, and local-path values for one UR arm. | `config.loader.load_config` |
| `configs/stage2/deployments/dual_ur.yaml` | Network, GPU, device, and local-path values for two UR arms. | `config.loader.load_config` |
| `configs/stage2/deployments/example.yaml` | Minimal reference for creating a deployment configuration. | Copied by users, then loaded by `load_config` |

Stage I loads one self-contained task YAML. Stage II merges one task YAML with one deployment YAML, then applies CLI overrides. See the configuration guide for every field and the merge order.

## 4. Package-level files `src/vla_precision/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `__init__.py` | Package description and version. | Python packaging and runtime identity |
| `cli.py` | Parses the public `--stage/--mode/--role` interface, loads configuration, and dispatches training, evaluation, or services. | Root `main.py` |
| `evaluation.py` | Evaluates a Stage I full model or Stage II ACoB model through the same Pyro environment and writes results after every episode. | `cli.py --mode evaluate` |
| `logging.py` | Console colors, debug files, terminal teeing, and key status messages. | CLI, actor, learner, services |
| `runtime_identity.py` | Builds and compares protocol, OpenPI revision, and source identities for distributed processes. | Pyro environment handshake |
| `image_tools.py` | Converts images to `uint8` and performs aspect-preserving padded resize. | Preprocessing and robot observations |
| `serialization.py` | Recursively decodes Base64-encoded `.npy` payloads into NumPy data. | Robot HTTP/Pyro boundaries |

Each subpackage `__init__.py` only declares the package boundary or exports stable public objects; it does not define a separate runtime flow.
In particular, `integrations/__init__.py` declares third-party integration boundaries and `robotics/__init__.py` declares the composable hardware boundary.

## 5. Configuration implementation `config/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `schema.py` | Defines Stage I `Stage1Config`, Stage II `RootConfig`, and all grouped configuration dataclasses. | `loader.py` and project-wide type interfaces |
| `loader.py` | Merges defaults, task YAML, deployment YAML, and CLI overrides; performs core boundary checks and computes hashes. | `cli.py` |
| `__init__.py` | Exports configuration types and loaders. | CLI and runtime modules |

## 6. Data preparation `data/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `indexing.py` | Materializes `state_indices` and `action_indices` into the Hugging Face/LeRobot dataset before a DataLoader is created. | Stage I loader and Stage II preprocess |
| `preprocess.py` | Converts LeRobot episodes into train-ready action chunks, transitions, and OpenPI context; supports parallel image decoding and multi-device context encoding. | `cli.py --stage stage2 --mode preprocess` |
| `paths.py` | Derives LeRobot roots, preprocessed data, online data, initial buffers, and checkpoint paths from the unified config. | Preprocess, actor, learner, evaluation |
| `__init__.py` | Documents the rule that index transforms are not repeated in the sampling hot path. | Package imports |

## 7. ACoB algorithm `acob/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `agent.py` | `ACoBState`, `ACoBAgent`, actor/critic losses, action sampling, and critic-only/joint updates. | `factory.py`, learner, actor |
| `factory.py` | Builds an ACoB agent from `RootConfig`, an OpenPI TrainConfig, and observation/action spaces. | `acob_stream.setup` |
| `networks.py` | Value/Advantage critic, MLP, time embeddings, and ensemble components. | `agent.py` |
| `vision.py` | ResNet encoders, pretrained encoders, FiLM, and spatial feature modules. | `agent.py`, `networks.py` |
| `augmentation.py` | Random crop, color, blur, and related critic image augmentations. | `agent.py` |
| `training.py` | Optimizers, batch sharding, module container, and `JaxRLTrainState`. | `agent.py` and the OpenPI training boundary |
| `pretrained_resnet.py` | Downloads, verifies, and loads pretrained ResNet-10 critic weights. | `agent.py` (construction starts in `factory.py`) |
| `types.py` | Array, batch, and PyTree type aliases used by ACoB. | ACoB internals |
| `__init__.py` | Lazily exports the agent to avoid loading full training dependencies for lightweight imports. | External imports |

## 8. Online framework `acob_stream/`

### 8.1 Runtime files

| File | Purpose | Primary callers |
| --- | --- | --- |
| `actor.py` | Online collection and inference loop: receives parameters, encodes context, infers action chunks, handles intervention returned by the remote environment, and commits episodes. | `cli.py --role actor` |
| `learner.py` | Server-side loop: registers buffers, warms up the critic, performs critic/joint updates, publishes parameters, and saves checkpoints. | `cli.py --role learner` |
| `setup.py` | Builds real actor/learner runtimes, agents, buffers, communication objects, and loggers from one `ResolvedConfig`. | `run_actor`, `run_learner` |
| `jax_runtime.py` | JAX mesh and sharding, actor inference backend, learner update backend, and JIT precompilation. | `setup.py` and runtime loops |
| `communication.py` | Pyro environment client, AgentLace trainer client/server, and actor-parameter serialization/replacement. | `setup.py`, actor, learner |
| `interfaces.py` | Minimal Protocols for environments, stores, agents, and communication objects. | `actor.py`, `learner.py` |
| `checkpoints.py` | Saves/restores the OpenPI actor (including LoRA), actor/critic optimizers, and critic at one step; reconstructs the reference policy on resume. | `setup.py`, learner |
| `metrics.py` | Timers, actor/learner metric shaping, and W&B training metric tracking. | Actor and learner |
| `trajectories.py` | Computes chunk returns, attaches next context, commits episodes, and atomically reads/writes transition shards. | Actor, preprocess, resume |
| `preprocessed_data.py` | Validates and loads Stage II preprocess metadata and transitions. | `setup.py`, learner |
| `initial_buffers.py` | Saves/restores initial replay and correction data with referenced context, remapping context sources. | Learner startup |
| `__init__.py` | Declares the ACoB-Stream package. | Package imports |

### 8.2 Three buffers `acob_stream/buffers/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `replay.py` | Public replay buffer, context hydration, mixed replay/correction sampling, and background prefetch. | Learner and AgentLace data ingestion |
| `correction.py` | Sparse human-correction buffer with an independent public concept on top of replay storage. | Actor and learner |
| `context.py` | mmap/sharded context buffer that persists OpenPI prefix KV values, masks, valid lengths, and storage metadata; `ContextSource` distinguishes preprocess, training, and initial references in transitions. | Preprocess, actor, learner sampler |
| `_dataset.py` | Nested NumPy dataset length, selection, and sampling primitives. | Low-level replay buffer |
| `_replay_base.py` | Base ring insertion, indexing, and iteration. | `_memory_replay.py` |
| `_memory_replay.py` | Memory-efficient storage for image observations. | `_data_store.py` |
| `_data_store.py` | Adapts replay storage to an AgentLace DataStore and provides thread-safe insertion. | `replay.py`, communication layer |
| `__init__.py` | Lazily exports replay/correction and directly exports context types. | `setup.py`, external imports |

## 9. OpenPI extension `integrations/openpi/`

OpenPI is pinned to a Git commit in `pyproject.toml`. This directory contains project-owned extensions and adapters; it does not modify upstream sources.

### 9.1 Training, configuration, and data

| File | Purpose | Primary callers |
| --- | --- | --- |
| `configs.py` | Declares project TrainConfigs in native OpenPI style and overlays top-level YAML values for Stage I/II. | Norm stats, Stage I training, Stage II setup, evaluation |
| `data_configs.py` | Defines OpenPI `DataConfigFactory` classes for UR5e, dual UR5e, and Franka. | `configs.py` |
| `policies/ur5e.py` | OpenPI input/output transforms and example observation for one UR arm. | Matching DataConfig and policy |
| `policies/dual_ur.py` | OpenPI input/output transforms for two UR arms. | Matching DataConfig and policy |
| `policies/franka.py` | OpenPI input/output transforms for Franka. | Matching DataConfig and policy |
| `policies/__init__.py` | Exports the three robot transform sets. | `data_configs.py` |
| `data_loader.py` | Materializes indices before sampling and builds OpenPI-compatible datasets and Torch DataLoaders. | Stage I training and norm stats |
| `train.py` | Stage I full fine-tuning: state initialization, JIT train step, W&B, checkpoints, and loop. | `cli.py --stage stage1 --mode train` |
| `norm_stats.py` | Computes and saves normalization statistics through OpenPI transforms. | `cli.py --mode norm-stats` |
| `training_state.py` | Central TrainState construction boundary between OpenPI internals and ACoB. | `acob.factory`, Stage I training |
| `checkpoints.py` | Resolves Stage I/II OpenPI checkpoint and asset directories, including step selection. | Setup, evaluation, native inference |
| `adapter.py` | Converts between robot observation/action structures and the OpenPI transform tree. | Preprocess, actor, evaluation |
| `context.py` | Encodes OpenPI prefix context and samples actions from cached context. | Preprocess and ACoB agent |
| `lerobot_compat.py` | Installs import-name compatibility for the pinned OpenPI and LeRobot revisions. | OpenPI extension imports |
| `__init__.py` | Keeps the boundary lightweight and installs LeRobot import compatibility. | All OpenPI extension modules |

### 9.2 Native OpenPI inference `integrations/openpi/inference/`

This path bypasses ACoB-Stream and Pyro for direct Stage I evaluation. The policy can run locally on the robot computer or through an OpenPI WebSocket policy server.

| File | Purpose | Primary callers |
| --- | --- | --- |
| `config.py` | Loads native-inference YAML into policy and hardware runtime configuration. | `runtime.py` |
| `runtime.py` | Selects a hardware backend, creates a local/remote policy, and starts inference or the WebSocket server. | Native-inference CLI modes |
| `policy.py` | Builds OpenPI observations and exposes a common sampler. | Native inference and shared evaluation |
| `configs/ur.yaml` | Native inference preset for one UR arm. | `config.py` |
| `configs/dual_ur.yaml` | Native inference preset for two UR arms. | `config.py` |
| `backends/ur.py` | Camera, control, episode, and operator flow for one UR arm. | `runtime.py` |
| `backends/dual_ur.py` | Dual-UR inference flow. | `runtime.py` |
| `backends/franka.py` | Franka inference flow. | `runtime.py` |
| `backends/recording.py` | Background video and episode recording. | Hardware backends |
| `backends/plots.py` | Generates trajectory and state plots from episode logs. | Recording analysis |
| `backends/utils.py` | Shared normalization-dimension validation and FPS utilities. | Hardware backends |
| `__init__.py`, `backends/__init__.py` | Export public entrypoints and declare the native-inference boundary. | Package imports |

## 10. Robotics system `robotics/`

### 10.1 Robot `robotics/robots/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `base.py` | Vendor-independent `Robot` Protocol. | Environment and new implementations |
| `factory.py` | Builds a UR or Franka robot from configuration. | Environment factory |
| `ur.py` | Calls the separate UR server over HTTP, reads state, and sends action chunks. | `RobotEnvironment` |
| `franka.py` | Calls Franka through ZeroRPC/Polymetis and converts state/actions. | `RobotEnvironment` |
| `__init__.py` | Exports the Robot interface, factory, and implementations. | Robotics internals |

### 10.2 Camera `robotics/cameras/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `base.py` | Vendor-independent `Camera` Protocol. | Environment and new implementations |
| `factory.py` | Creates and closes the configured camera collection. | Environment factory |
| `realsense.py` | Intel RealSense frame acquisition. | Camera factory |
| `latest_frame.py` | Reads a camera asynchronously and exposes only its latest frame to the control loop. | Camera factory and environment |
| `__init__.py` | Exports the Camera interface and implementations. | Robotics internals |

### 10.3 Gripper `robotics/grippers/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `base.py` | Vendor-independent `Gripper` Protocol. | Environment and new implementations |
| `factory.py` | Builds a fixed gripper, UR-hosted gripper, or Franka PGI according to setup mode. | Environment factory |
| `ur.py` | Operates PGI through the UR HTTP server and provides a fixed-gripper implementation. | `RobotEnvironment` |
| `franka.py` | Asynchronous serial PGI implementation used with Franka. | `RobotEnvironment` |
| `__init__.py` | Exports the Gripper interface, factory, and implementations. | Robotics internals |

### 10.4 Teleoperation `robotics/teleoperation/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `base.py` | Shared `TeleoperationDevice` Protocol for keyboards, VR, and future master devices. | Intervention wrapper and extensions |
| `single_keyboard.py` | Single-arm keyboard action input and state. | Keyboard intervention |
| `dual_keyboard.py` | Dual-device/dual-arm keyboard input and thread synchronization. | Keyboard intervention |
| `keyboard.py` | Keyboard emergency-stop and task-completion event detectors. | Environment factory and task detectors |
| `__init__.py` | Exports the shared teleoperation interface and keyboard implementations. | Robotics internals |

### 10.5 Task logic `robotics/tasks/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `completion.py` | Independent task-completion detector interface and event implementation. | Completion/reward wrapper |
| `reward.py` | Independent RewardFunction interface and chunk-completion reward. | Completion/reward wrapper |
| `reset.py` | Per-task reset procedures preserving established robot, gripper, and prompt ordering. | Environment factory and regrasp wrapper |
| `__init__.py` | Exports completion, reward, and reset builders. | Environment factory |

### 10.6 Gymnasium wrappers `robotics/wrappers/`

Wrappers define composition order; completion detection, reward design, and teleoperation input remain independent modules.

| File | Purpose | Primary callers |
| --- | --- | --- |
| `action_chunk.py` | Adapts an OpenPI action chunk to per-action environment execution and groups chunk observations/rewards. | Environment factory |
| `keyboard_intervention.py` | Applies teleoperation to action chunks and handles takeover, emergency stop, and executed actions. | Environment factory |
| `relative_frame.py` | Converts between reset-relative and absolute robot frames. | Environment factory |
| `observations.py` | Quaternion/Euler and flattened-observation adapters. | Environment factory |
| `completion_reward.py` | Connects independent completion detectors and reward functions to Gymnasium steps. | Environment factory |
| `regrasp.py` | Composes task-specific regrasp behavior around reset. | Environment factory |
| `__init__.py` | Exports wrapper types. | Environment factory |

### 10.7 Environment `robotics/environments/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `robot.py` | Base real-robot Gymnasium environment composing robot, cameras, gripper, and detectors. | `factory.py` |
| `factory.py` | Builds the base environment and wrapper stack in a fixed order. | Pyro environment server and tests |
| `server.py` | Bridges the local robot system to a remote agent (Actor or evaluator) through Pyro and handles reset, step, and episode statistics. | `cli.py --mode robot-agent-bridge`, actor |
| `fake.py` | Hardware-free robot/camera used for learner space construction and unit tests. | Setup and tests |
| `__init__.py` | Exports the environment factory and base environment. | Robotics internals |

### 10.8 Low-level UR service `robotics/servers/ur/`

| File | Purpose | Primary callers |
| --- | --- | --- |
| `server.py` | UR RTDE force-mode control, state acquisition, action queues, gripper threads, HTTP routes, and safe shutdown. | `cli.py --mode serve-robot`, `robots/ur.py` |
| `pgi.py` | PGI serial gripper driver hosted by the UR service. | `server.py` |
| `rotations.py` | Rotvec, quaternion, and Euler pose conversion. | UR controller |
| `__init__.py` | Exports `run_robot_server`. | CLI |

`robotics/servers/__init__.py` declares the low-level service package. The low-level service is launched separately from the Gym environment so algorithm processes do not own hardware SDK objects directly.

## 11. Unit tests `tests/unit/`

These files are development checks and are not imported by normal training, evaluation, or service processes.

| File | Coverage |
| --- | --- |
| `test_main.py`, `test_cli.py` | Root entrypoint, stage/mode normalization, and dispatch. |
| `test_config.py`, `test_task_presets.py` | Configuration merge, overrides, hashes, and task-config loadability. |
| `test_data_indexing.py`, `test_lerobot_root.py`, `test_preprocess.py`, `test_preprocessed_data.py` | Index materialization, dataset roots, and preprocess artifacts. |
| `test_acob_algorithm_parity.py`, `test_acob_factory.py`, `test_resnet_asset.py` | ACoB numerical behavior, agent construction, and pretrained visual weights. |
| `test_replay_buffers.py`, `test_context_buffer.py`, `test_initial_buffers.py` | Replay, correction, context, and initial-buffer persistence. |
| `test_stream_actor.py`, `test_stream_learner.py`, `test_stream_communication.py`, `test_stream_metrics.py`, `test_stream_trajectories.py` | Actor/learner loops, communication, metrics, and episode/transition handling. |
| `test_acob_checkpoints.py` | Stage II actor/critic checkpoint save and restore. |
| `test_openpi_extension.py`, `test_openpi_training_state.py`, `test_openpi_train.py`, `test_openpi_checkpoints.py`, `test_openpi_spawn.py` | OpenPI extension, TrainState, training, checkpoints, and process-import boundaries. |
| `test_openpi_native_inference.py`, `test_evaluation.py` | Native OpenPI inference and shared evaluation. |
| `test_robotics_components.py`, `test_environment_wrappers.py`, `test_ur_robot_server.py` | Robotics components, wrapper order, and UR service control contracts. |
| `test_image_tools.py`, `test_logging.py`, `test_runtime_identity.py` | Image helpers, logging, and distributed runtime identity. |

## 12. Main call paths

### Stage I normalization and full fine-tuning

```text
main.py -> cli.py -> load_stage1_config
  ├── norm-stats -> integrations/openpi/norm_stats.py
  └── train      -> integrations/openpi/train.py
                       ├── configs.py + data_configs.py
                       ├── data_loader.py
                       └── OpenPI model/checkpoint API
```

### Stage II preprocessing

```text
main.py -> cli.py -> load_config
  -> data/preprocess.py
       ├── data/indexing.py
       ├── integrations/openpi/adapter.py + context.py
       └── preprocessed transitions + Context Buffer
```

### Stage II online training

```text
Robot computer                         GPU server
serve-robot                           learner.py
    │                                     │
robot-agent-bridge <────── Pyro ──── actor.py
    │                                     │
robotics wrapper stack              AgentLace buffers/parameters
                                          │
                                     acob/agent.py
```

Actor and learner load the same task and deployment configurations but consume only fields relevant to their roles. Pyro carries actor-to-environment interaction; AgentLace carries transitions, corrections, and parameters between actor and learner.

### Evaluation

```text
Shared environment: cli.py -> evaluation.py -> OpenPI sampler -> Pyro environment
Native OpenPI:       cli.py -> inference/runtime.py -> hardware backend
                                               └── local or WebSocket policy
```

## 13. Runtime-generated directories

| Directory | Contents | Source code? |
| --- | --- | --- |
| `assets/` | Normalization statistics, downloaded ResNet weights, and OpenPI assets. | No |
| `checkpoints/` | Stage I and Stage II checkpoints; may be symlinked to large storage. | No |
| `train_data/` | Preprocess cache, transition shards, and context buffers; may be symlinked to large storage. | No |
| `.cache/` | Project-local Hugging Face, JAX/OpenPI, and related caches. | No |
| `debug/` | Per-role logs created when the single `debug` flag is enabled. | No |
| `results/` | Evaluation results appended after each episode. | No |
| `wandb/` | Local W&B run data. | No |
| `.venv/` | Python environment managed by `uv`. | No |

These directories do not define algorithm behavior and must not be used as Python source locations.
