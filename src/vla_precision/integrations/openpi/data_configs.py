"""VLA-Precision robot data factories written in the native OpenPI style."""

from __future__ import annotations

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import dataclasses
import pathlib

from openpi import transforms
from openpi.models import model as openpi_model
from openpi.training import config as openpi_config
from typing_extensions import override

from vla_precision.integrations.openpi.policies import dual_ur, franka, ur5e


def make_robot_data_config_template(factory, *, dual: bool = False):
    """Create the robot-specific OpenPI DataConfig template selected by TrainConfig."""
    return factory(
        repo_id="",
        base_config=openpi_config.DataConfig(
            prompt_from_task=False,
            action_sequence_keys=("action",),
        ),
        image_key_map=(
            {
                "base_0_rgb": "observation.images.exterior_image",
                "left_wrist_0_rgb": "observation.images.left_wrist_image",
                "right_wrist_0_rgb": "observation.images.right_wrist_image",
            }
            if dual
            else {
                "base_0_rgb": "observation.images.exterior_image",
                "left_wrist_0_rgb": "observation.images.wrist_image",
            }
        ),
        extra_delta_transform=False,
    )


@dataclasses.dataclass(frozen=True)
class LeRobotUR5eDataConfig(openpi_config.DataConfigFactory):
    extra_delta_transform: bool = True
    state_key: str = "observation.state"
    action_key: str = "action"
    image_key_map: dict[str, str] | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: openpi_model.BaseModelConfig,
    ) -> openpi_config.DataConfig:
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": (self.image_key_map or {}).get(
                            "base_0_rgb", "observation.images.exterior_image"
                        ),
                        "observation/wrist_image": (self.image_key_map or {}).get(
                            "left_wrist_0_rgb", "observation.images.wrist_image"
                        ),
                        "observation/state": self.state_key,
                        "actions": self.action_key,
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[ur5e.UR5eInputs(model_type=model_config.model_type)],
            outputs=[ur5e.UR5eOutputs()],
        )
        if self.extra_delta_transform:
            mask = transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDualUR5eDataConfig(openpi_config.DataConfigFactory):
    extra_delta_transform: bool = True
    state_key: str = "observation.state"
    action_key: str = "action"
    image_key_map: dict[str, str] | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: openpi_model.BaseModelConfig,
    ) -> openpi_config.DataConfig:
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/exterior_image": (self.image_key_map or {}).get(
                            "base_0_rgb", "observation.images.exterior_image"
                        ),
                        "observation/left_wrist_image": (self.image_key_map or {}).get(
                            "left_wrist_0_rgb", "observation.images.left_wrist_image"
                        ),
                        "observation/right_wrist_image": (self.image_key_map or {}).get(
                            "right_wrist_0_rgb", "observation.images.right_wrist_image"
                        ),
                        "observation/state": self.state_key,
                        "actions": self.action_key,
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[dual_ur.DualURInputs(model_type=model_config.model_type)],
            outputs=[dual_ur.DualUROutputs()],
        )
        if self.extra_delta_transform:
            mask = transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDataConfig(openpi_config.DataConfigFactory):
    extra_delta_transform: bool = True
    state_key: str = "observation.state"
    action_key: str = "action"
    image_key_map: dict[str, str] | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: openpi_model.BaseModelConfig,
    ) -> openpi_config.DataConfig:
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": (self.image_key_map or {}).get(
                            "base_0_rgb", "observation.images.exterior_image"
                        ),
                        "observation/wrist_image": (self.image_key_map or {}).get(
                            "left_wrist_0_rgb", "observation.images.wrist_image"
                        ),
                        "observation/state": self.state_key,
                        "actions": self.action_key,
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[franka.FrankaInputs(model_type=model_config.model_type)],
            outputs=[franka.FrankaOutputs()],
        )
        if self.extra_delta_transform:
            mask = transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
        )
