# VLA-Precision 二次开发

机器人、相机、夹爪和遥操作通过独立接口组合。新增设备时，只实现对应接口并在 factory 中注册，不需要修改 ACoB 或 ACoB-Stream。下面均为最小结构示例，硬件 SDK 调用应放在具体实现类内部。

## 通用流程

1. 从最接近的现有实现开始，新文件放在同类组件目录中。
2. 实现公共接口，并在对应 factory 中注册简短名称。
3. 多进程必须一致的行为放在任务 YAML；IP、序列号和设备路径放在部署 YAML。
4. 先单独测试组件，再使用 fake environment 或低速真机确认。

## 新增机器人

在 `robotics/robots/<robot>.py` 中实现 `Robot`。机器人只负责机器人能力，不应自行创建相机、夹爪或遥操作设备。

```python
import numpy as np


class MyRobot:
    action_low = -1.0
    action_high = 1.0

    def __init__(self, config, *, gripper, reset_procedure):
        self.config = config
        self.gripper = gripper
        self.reset_procedure = reset_procedure
        self._state = np.zeros(6, dtype=np.float32)

    @property
    def action_dimension(self) -> int:
        return 6 + self.gripper.action_dimension

    @property
    def currpos(self) -> np.ndarray:
        return self._state.copy()

    def refresh_state(self) -> None:
        self._state = read_pose_from_sdk()

    def reset(self, *, joint_reset=False, options=None):
        self.reset_procedure.reset(self, joint_reset=joint_reset, options=options or {})
        return self.observations()

    def execute_action_chunk(self, action: np.ndarray) -> np.ndarray:
        return send_chunk_with_sdk(action)

    def observations(self) -> dict:
        return {"tcp_pose": self.currpos, **self.gripper.observations({})}

    def request(self, name: str, enabled: bool):
        return send_named_request(name, enabled)

    def close(self) -> None:
        stop_control_safely()
        self.gripper.close()
```

在 `robotics/robots/factory.py` 中增加构造函数和注册项：

```python
def _my_robot(config, *, gripper, reset_procedure, dual_arm):
    del dual_arm
    return MyRobot(config.robot, gripper=gripper, reset_procedure=reset_procedure)


builders = {
    "ur5e": _ur_robot,
    "franka": _franka_robot,
    "my_robot": _my_robot,
}
```

任务 YAML 使用 `robot.kind: my_robot`。若需要新的 reset 顺序，在 `robotics/tasks/reset.py` 中实现并注册：

```python
class MyReset:
    def reset(self, robot, *, joint_reset: bool, options: dict) -> None:
        robot.move_home(joint_reset=joint_reset)
        robot.refresh_state()


_PROCEDURES["my_reset"] = MyReset
```

## 新增相机

在 `robotics/cameras/<camera>.py` 中实现 `Camera`：

```python
class USBCamera:
    def __init__(self, name: str, device: str):
        self._name = name
        self._capture = open_camera(device)

    @property
    def name(self) -> str:
        return self._name

    def capture(self):
        return self._capture.read_rgb()

    def close(self) -> None:
        self._capture.close()
```

在 `robotics/cameras/factory.py` 中注册：

```python
def _usb(name, config):
    return USBCamera(name, device=str(config.options["device"]))


driver_factories = {"realsense": _realsense, "usb": _usb}
```

部署 YAML 使用同一个逻辑图像键：

```yaml
cameras:
  devices:
    base_0_rgb:
      driver: usb
      options:
        device: /dev/video0
```

`task.image_keys`、`cameras.devices` 和 `data.image_key_map` 中的键必须一致。`capture_mode: async` 会自动包装为最新帧采集。

## 新增夹爪

在 `robotics/grippers/<gripper>.py` 中实现 `Gripper`：

```python
import numpy as np


class ElectricGripper:
    def __init__(self, port: str):
        self._device = open_gripper(port)

    @property
    def action_dimension(self) -> int:
        return 1

    def command_chunk(self, positions: np.ndarray) -> None:
        for position in positions.reshape(-1):
            self._device.move(float(position))

    def observations(self, robot_state: dict) -> dict[str, np.ndarray]:
        del robot_state
        return {"gripper_pose": np.asarray([self._device.position()], dtype=np.float32)}

    def close(self) -> None:
        self._device.close()
```

在 `robotics/grippers/factory.py` 中注册：

```python
def _electric(config, *, dual_arm: bool, state_fields: tuple[str, ...]):
    del dual_arm, state_fields
    return ElectricGripper(config.gripper.device)


builders = {"pgi": _ur_gripper, "electric": _electric}
```

部署 YAML 中设置：

```yaml
gripper:
  kind: electric
  device: /dev/ttyUSB0
```

模型需要夹爪状态时，将对应键加入 `task.proprio_keys`，并同步更新 OpenPI state/action 映射。

## 新增遥操作设备

键盘、VR 和同构主从遥操作共享 `TeleoperationDevice`。设备只返回动作和按键状态，机器人执行、干预记录和 Correction Buffer 写入继续复用现有 wrapper。

```python
class VRTeleoperation:
    def __init__(self, config, dual_arm: bool, completion_event):
        self._client = connect_vr(config.device)
        self._completion_event = completion_event

    def get_action(self) -> tuple[list[float], list[int]]:
        pose_delta, buttons = self._client.read()
        if buttons.completed:
            self._completion_event.set()
        return pose_delta.tolist(), buttons.as_list()

    def reset_step_mode(self) -> None:
        self._client.reset_origin()

    def close(self) -> None:
        self._client.close()
```

在 `robotics/environments/factory.py` 的遥操作 factory 映射中注册：

```python
def _vr(config, dual_arm, completion_event):
    return VRTeleoperation(config, dual_arm, completion_event)


teleoperation_factories = {"vr": _vr}
```

任务 YAML 选择 `teleoperation.kind: vr`，实体设备路径放在部署 YAML。双臂设备可以额外提供与双键盘相同的左右臂停止状态。

## 新增任务完成检测与 Reward

Detector 只回答任务是否完成，Reward 只根据完成结果生成 reward，二者由 `CompletionRewardWrapper` 组合。

```python
class DigitalCompletionDetector:
    def __init__(self, input_pin):
        self._input_pin = input_pin

    def completed(self) -> bool:
        return bool(self._input_pin.read())

    def reset(self) -> None:
        self._input_pin.clear()

    def close(self) -> None:
        self._input_pin.close()


# Add this branch in build_completion_detector().
if name == "digital_input":
    return DigitalCompletionDetector(open_input_pin())
```

```python
import numpy as np


class BinaryCompletionReward:
    def compute(self, *, chunk_length: int, succeeded: bool) -> np.ndarray:
        reward = np.zeros(chunk_length, dtype=np.float32)
        if succeeded:
            reward[-1] = 1.0
        return reward


# Add this branch in build_reward_function().
if name == "binary_completion":
    return BinaryCompletionReward()
```

任务 YAML 选择：

```yaml
task:
  completion_detector: digital_input
  reward_function: binary_completion
```

不要把 detector 专用判断写进 `robotics/wrappers/completion_reward.py`。

## 扩展 OpenPI

项目扩展位于 `integrations/openpi`，定义方式与 OpenPI 原生配置一致。先在 `data_configs.py` 中增加 `DataConfigFactory`：

```python
import dataclasses
from openpi.training import config as openpi_config


@dataclasses.dataclass(frozen=True)
class MyRobotDataConfig(openpi_config.DataConfigFactory):
    state_key: str = "observation.state"
    action_key: str = "action"

    def create(self, assets_dirs, model_config):
        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            data_transforms=make_my_robot_transforms(model_config),
        )
```

再在 `configs.py` 顶部的 `_CONFIGS` 中增加原生 `TrainConfig`：

```python
openpi_config.TrainConfig(
    name="pi05_full_finetune_my_robot",
    model=pi0_config.Pi0Config(pi05=True),
    data=make_robot_data_config_template(MyRobotDataConfig),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
)
```

若 observation/action 布局不同，还应在 `integrations/openpi/policies/` 中实现输入输出转换。YAML 通过 `openpi.name` 选择配置；数据集、indices、batch size 和路径继续由顶层 YAML 提供。不要修改安装后的 OpenPI 上游源码。

## 新增任务与部署配置

新任务复制最接近的 Stage I、Stage II YAML，硬件拓扑变化时再新增部署 YAML：

```yaml
# configs/stage2/tasks/my_task.yaml
experiment:
  name: my_task
task:
  instruction: "place the object in the tray"
  image_keys: [base_0_rgb, left_wrist_0_rgb]
  proprio_keys: [tcp_pose, tcp_vel, gripper_pose]
openpi:
  name: pi05_acob_my_robot
  initialization_config_name: pi05_full_finetune_my_robot
```

```yaml
# configs/stage2/deployments/my_robot.yaml
robot:
  server_url: http://127.0.0.1:5001/
cameras:
  devices:
    base_0_rgb:
      driver: usb
      options:
        device: /dev/video0
```

有序的 `state_indices`、`action_indices`、`image_key_map`、`task.proprio_keys` 和 `task.image_keys` 必须描述同一套模型输入输出。随后依次运行 normalization、Stage I、Stage II preprocess 和短时 rollout。

所有 YAML 字段参见 [CONFIGURATION_zh-CN.md](CONFIGURATION_zh-CN.md)。
