# Extending VLA-Precision

Robots, cameras, grippers, and teleoperation devices are composed through independent interfaces. A new component implements its interface and registers a factory; it does not require changes to ACoB or ACoB-Stream. The examples below show only the minimum structure. Keep hardware SDK calls inside the concrete implementation.

## General workflow

1. Start from the closest existing implementation and place the new file beside components of the same kind.
2. Implement the common interface and register a short factory name.
3. Put behavior that must agree across processes in the task YAML; put IPs, serial numbers, and device paths in the deployment YAML.
4. Test the component alone, then use a fake environment or a low-speed hardware run.

## Adding a robot

Implement `Robot` in `robotics/robots/<robot>.py`. A robot owns robot capabilities only; it must not construct cameras, grippers, or teleoperation devices.

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

Add a constructor and registry entry in `robotics/robots/factory.py`:

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

Select it with `robot.kind: my_robot`. If it needs a new reset sequence, implement and register it in `robotics/tasks/reset.py`:

```python
class MyReset:
    def reset(self, robot, *, joint_reset: bool, options: dict) -> None:
        robot.move_home(joint_reset=joint_reset)
        robot.refresh_state()


_PROCEDURES["my_reset"] = MyReset
```

## Adding a camera

Implement `Camera` in `robotics/cameras/<camera>.py`:

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

Register the driver in `robotics/cameras/factory.py`:

```python
def _usb(name, config):
    return USBCamera(name, device=str(config.options["device"]))


driver_factories = {"realsense": _realsense, "usb": _usb}
```

Use the same logical image key in the deployment YAML:

```yaml
cameras:
  devices:
    base_0_rgb:
      driver: usb
      options:
        device: /dev/video0
```

The keys in `task.image_keys`, `cameras.devices`, and `data.image_key_map` must agree. `capture_mode: async` automatically wraps the camera with latest-frame capture.

## Adding a gripper

Implement `Gripper` in `robotics/grippers/<gripper>.py`:

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

Register its factory in `robotics/grippers/factory.py`:

```python
def _electric(config, *, dual_arm: bool, state_fields: tuple[str, ...]):
    del dual_arm, state_fields
    return ElectricGripper(config.gripper.device)


builders = {"pgi": _ur_gripper, "electric": _electric}
```

Select it in the deployment YAML:

```yaml
gripper:
  kind: electric
  device: /dev/ttyUSB0
```

If the model consumes gripper state, add its key to `task.proprio_keys` and update the OpenPI state/action mapping.

## Adding a teleoperation device

Keyboard, VR, and master-arm teleoperation share `TeleoperationDevice`. The device returns actions and button state only; the existing wrappers retain robot execution, intervention recording, and Correction Buffer insertion.

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

Register it in the teleoperation factory mapping in `robotics/environments/factory.py`:

```python
def _vr(config, dual_arm, completion_event):
    return VRTeleoperation(config, dual_arm, completion_event)


teleoperation_factories = {"vr": _vr}
```

Select `teleoperation.kind: vr` in the task YAML and keep its physical path in the deployment YAML. A dual-arm implementation may also expose the per-arm stop state used by the dual keyboard.

## Adding completion detection and reward design

A detector answers only whether the task completed. A reward function converts that result to rewards. `CompletionRewardWrapper` composes them.

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

Select both implementations in the task YAML:

```yaml
task:
  completion_detector: digital_input
  reward_function: binary_completion
```

Do not place detector-specific behavior in `robotics/wrappers/completion_reward.py`.

## Extending OpenPI

Project-owned OpenPI extensions live in `integrations/openpi` and follow native OpenPI configuration style. First add a `DataConfigFactory` in `data_configs.py`:

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

Then add a native `TrainConfig` to `_CONFIGS` near the top of `configs.py`:

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

If the observation/action layout is new, also implement input/output transforms under `integrations/openpi/policies/`. YAML selects the profile through `openpi.name`; datasets, indices, batch sizes, and paths remain top-level YAML values. Do not edit the installed upstream OpenPI source.

## Adding a task and deployment

Copy the closest Stage I and Stage II task YAMLs. Add a deployment only when the hardware topology changes:

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

The ordered `state_indices`, `action_indices`, `image_key_map`, `task.proprio_keys`, and `task.image_keys` must describe the same model-facing layout. Then run normalization, Stage I, Stage II preprocessing, and a short rollout in that order.

See [CONFIGURATION.md](CONFIGURATION.md) for every YAML field.
