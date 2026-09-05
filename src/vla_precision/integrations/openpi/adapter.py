"""ACoB observation/action adapter for the OpenPI backend.

Robot observations should already use the camera names produced by the selected
OpenPI data transform. OpenPI-owned key names are discovered from TrainConfig at
initialization and cached, so the learner hot path does not repeatedly run
Python transforms.
"""

from __future__ import annotations

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import copy
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.training.config as _pi_config
import openpi.transforms as pi_transforms

from vla_precision.serialization import b64_to_numpy

DebugFn = Callable[[str, dict[str, Any]], None]
_Path = tuple[str, ...]


# Config-driven raw example discovery for prompt tokenization.
def _camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def make_prompt_tokenize_example(data_config: _pi_config.DataConfig, prompt: str) -> dict:
    """Create the raw example associated with the configured input transform."""
    for transform in data_config.data_transforms.inputs:
        class_name = type(transform).__name__
        if not class_name.endswith("Inputs"):
            continue
        module = sys.modules.get(type(transform).__module__)
        if module is None:
            continue
        policy_name = _camel_to_snake(class_name.removesuffix("Inputs"))
        make_example = getattr(module, f"make_{policy_name}_example", None)
        if callable(make_example):
            example = make_example()
            example["prompt"] = prompt
            return example

    raise ValueError(
        "Could not infer a raw OpenPI example from data_config.data_transforms.inputs. "
        "Expected a policy input transform such as UR5eInputs with a matching make_ur5e_example()."
    )


# Nested-dict helpers used only during initialization and actor conversion.
def _flatten_leaves(tree: Mapping[str, Any], prefix: _Path = ()) -> dict[_Path, Any]:
    leaves = {}
    for key, value in tree.items():
        path = (*prefix, str(key))
        if isinstance(value, Mapping):
            leaves.update(_flatten_leaves(value, path))
        else:
            leaves[path] = value
    return leaves


def _set_path(tree: dict, path: _Path, value: Any) -> None:
    cursor = tree
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


class OpenPIObservationAdapter:
    """Convert ACoB-Stream observations into OpenPI model inputs."""

    def __init__(
        self,
        train_config: _pi_config.TrainConfig,
        task_desc: str,
        *,
        image_keys: Iterable[str],
        fixed_gripper: bool = False,
        dual_arm: bool = False,
        learned_state_dim: int | None = None,
        action_horizon: int | None = None,
        debug_fn: DebugFn | None = None,
    ):
        self.train_config = train_config
        self.task_desc = task_desc
        self.debug_fn = debug_fn
        self.fixed_gripper = bool(fixed_gripper)
        self.dual_arm = bool(dual_arm)
        self.learned_state_dim = None if learned_state_dim is None else int(learned_state_dim)
        self.action_horizon = int(action_horizon if action_horizon is not None else train_config.model.action_horizon)

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = data_config.norm_stats
        self.norm_stats = norm_stats
        self.use_quantile_norm = data_config.use_quantile_norm
        self._build_transforms(data_config, norm_stats)
        self._discover_config_schema(data_config, tuple(image_keys), norm_stats)
        self._cache_prompt_tokens()
        self._debug_init(norm_stats is not None)

    # Initialization pipeline.
    def _build_transforms(self, data_config: _pi_config.DataConfig, norm_stats) -> None:
        """Construct OpenPI's configured input/output transforms once."""
        self.data_input_transform = pi_transforms.compose(data_config.data_transforms.inputs)
        self.data_action_input_transforms = tuple(data_config.data_transforms.inputs)
        self.data_action_output_transforms = tuple(data_config.data_transforms.outputs)
        input_transforms = list(data_config.data_transforms.inputs)
        if norm_stats is not None:
            input_transforms.append(pi_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
        input_transforms.extend(data_config.model_transforms.inputs)
        self.input_transform = pi_transforms.compose(input_transforms)

        output_transforms = list(data_config.model_transforms.outputs)
        if norm_stats is not None:
            output_transforms.append(pi_transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
        output_transforms.extend(data_config.data_transforms.outputs)
        output_transforms.extend(data_config.repack_transforms.outputs)
        self.output_transform = pi_transforms.compose(output_transforms)

    def _discover_config_schema(
        self, data_config: _pi_config.DataConfig, image_keys: tuple[str, ...], norm_stats
    ) -> None:
        """Infer OpenPI raw/model keys from the selected config."""
        self.raw_template = make_prompt_tokenize_example(data_config, self.task_desc)
        self.repack_output_keys = self._repack_output_keys(data_config)

        data_transformed = self.data_input_transform(copy.deepcopy(self.raw_template))
        self.image_key, self.image_mask_key, self.state_key = self._discover_model_fields(data_transformed)
        self.prompt_key = self._discover_prompt_key(self.raw_template)
        self.raw_action_key = data_config.action_sequence_keys[0]
        self.action_key = "actions" if norm_stats is not None and "actions" in norm_stats else self.raw_action_key

        self.model_image_keys = tuple(data_transformed[self.image_key].keys())
        self.source_image_keys = self._validate_source_image_keys(image_keys)
        self.padding_image_keys = tuple(key for key in self.model_image_keys if key not in self.source_image_keys)

        self.raw_image_paths = self._infer_raw_image_paths(self.raw_template, self.source_image_keys)
        self.raw_state_path = self._infer_raw_state_path(self.raw_template)
        self.state_input_dim = self._norm_dim(norm_stats, self.state_key)
        self.action_norm_dim = self._norm_dim(norm_stats, self.action_key)
        self._validate_arm_schema()

    def _validate_arm_schema(self) -> None:
        """Require norm stats to match the selected physical state/action schema."""
        if not self.dual_arm:
            if len(self.source_image_keys) != 2:
                raise ValueError(
                    "Single-arm mode requires exactly two real image inputs "
                    f"(base and wrist), got {self.source_image_keys}."
                )
            expected_state_dim = 18 if self.fixed_gripper else (self.learned_state_dim or 19)
            expected_action_dim = 6 if self.fixed_gripper else 7
            if self.state_input_dim is not None and self.state_input_dim != expected_state_dim:
                raise ValueError(
                    f"Single-arm fixed_gripper={self.fixed_gripper} requires {expected_state_dim}-D "
                    f"state normalization statistics, got {self.state_input_dim}. "
                    "Fixed-gripper datasets must omit gripper state instead of filling it from norm stats."
                )
            if self.action_norm_dim is not None and self.action_norm_dim != expected_action_dim:
                raise ValueError(
                    f"Single-arm fixed_gripper={self.fixed_gripper} requires {expected_action_dim}-D "
                    f"action normalization statistics, got {self.action_norm_dim}. "
                    "Fixed-gripper datasets must omit gripper actions."
                )
            return
        if len(self.source_image_keys) != 3:
            raise ValueError(
                "dual_arm=True requires exactly three real image inputs "
                f"(base, left wrist, right wrist), got {self.source_image_keys}."
            )
        if self.state_input_dim is not None and self.state_input_dim != 38:
            raise ValueError(
                "dual_arm=True requires 38-D normalization statistics for the ordered "
                "left/right robot state, but the selected OpenPI assets contain "
                f"{self.state_input_dim}-D state statistics. Generate/select dual-arm norm stats."
            )
        if self.action_norm_dim is not None and self.action_norm_dim != 14:
            raise ValueError(
                "dual_arm=True requires 14-D action normalization statistics "
                f"([left 6D+gripper, right 6D+gripper]), got {self.action_norm_dim}."
            )

    def _cache_prompt_tokens(self) -> None:
        """Tokenize the fixed task prompt once for reuse in learner batches."""
        tokenized = self.input_transform(copy.deepcopy(self.raw_template))
        self.tokenized_prompt = np.asarray(tokenized["tokenized_prompt"])
        self.tokenized_prompt_mask = np.asarray(tokenized["tokenized_prompt_mask"])
        self.token_ar_mask = np.asarray(tokenized["token_ar_mask"]) if "token_ar_mask" in tokenized else None
        self.token_loss_mask = np.asarray(tokenized["token_loss_mask"]) if "token_loss_mask" in tokenized else None

    def _debug_init(self, use_norm_stats: bool) -> None:
        if self.debug_fn is None:
            return
        self.debug_fn(
            "adapter.init",
            {
                "source_image_keys": self.source_image_keys,
                "padding_image_keys": self.padding_image_keys,
                "model_image_keys": self.model_image_keys,
                "raw_image_paths": {key: "/".join(path) for key, path in self.raw_image_paths.items()},
                "raw_state_path": "/".join(self.raw_state_path),
                "repack_output_keys": self.repack_output_keys,
                "state_key": self.state_key,
                "state_input_dim": self.state_input_dim,
                "raw_action_key": self.raw_action_key,
                "action_norm_key": self.action_key,
                "action_norm_dim": self.action_norm_dim,
                "model_action_dim": self.train_config.model.action_dim,
                "model_action_horizon": self.train_config.model.action_horizon,
                "use_norm_stats": use_norm_stats,
                "use_quantile_norm": self.use_quantile_norm,
                "norm_stat_keys": tuple(self.norm_stats.keys()) if self.norm_stats is not None else (),
                "fixed_gripper": self.fixed_gripper,
                "data_action_input_transforms": tuple(type(t).__name__ for t in self.data_action_input_transforms),
                "data_action_output_transforms": tuple(type(t).__name__ for t in self.data_action_output_transforms),
            },
        )

    # Config/schema discovery utilities. These run only during initialization.
    @staticmethod
    def _repack_output_keys(data_config: _pi_config.DataConfig) -> tuple[str, ...]:
        for transform in data_config.repack_transforms.inputs:
            structure = getattr(transform, "structure", None)
            if isinstance(structure, Mapping):
                return tuple("/".join(path) for path in _flatten_leaves(structure))
        return ()

    @staticmethod
    def _discover_model_fields(data: Mapping[str, Any]) -> tuple[str, str, str]:
        image_key = None
        for key, value in data.items():
            if isinstance(value, Mapping) and value:
                leaves = [np.asarray(v) for v in value.values()]
                if all(leaf.ndim >= 3 for leaf in leaves):
                    image_key = key
                    break
        if image_key is None:
            raise KeyError("Could not infer OpenPI image dict key from data transform output.")

        image_names = set(data[image_key])
        mask_key = None
        for key, value in data.items():
            if key == image_key or not isinstance(value, Mapping) or set(value) != image_names:
                continue
            leaves = [np.asarray(v) for v in value.values()]
            if all(leaf.dtype == np.bool_ or leaf.dtype == bool for leaf in leaves):
                mask_key = key
                break
        if mask_key is None:
            raise KeyError("Could not infer OpenPI image mask dict key from data transform output.")

        state_key = None
        for key, value in data.items():
            if key in (image_key, mask_key):
                continue
            array = np.asarray(value)
            if array.ndim <= 1 and np.issubdtype(array.dtype, np.number):
                state_key = key
                break
        if state_key is None:
            raise KeyError("Could not infer OpenPI state key from data transform output.")
        return image_key, mask_key, state_key

    @staticmethod
    def _discover_prompt_key(raw_template: Mapping[str, Any]) -> str:
        for key, value in raw_template.items():
            if isinstance(value, str | bytes):
                return key
        return "prompt"

    def _validate_source_image_keys(self, image_keys: tuple[str, ...]) -> tuple[str, ...]:
        unknown = tuple(key for key in image_keys if key not in self.model_image_keys)
        if unknown:
            raise ValueError(f"ACoB-Stream image_keys must match OpenPI config image keys; unknown keys: {unknown}.")
        if not image_keys:
            raise ValueError(f"ACoB-Stream image_keys is empty; OpenPI model image keys are {self.model_image_keys}.")
        return image_keys

    def _infer_raw_image_paths(self, raw_template: dict, image_keys: Iterable[str]) -> dict[str, _Path]:
        leaves = _flatten_leaves(raw_template)
        image_paths = {key: None for key in image_keys}
        for path, value in leaves.items():
            value = np.asarray(value)
            if value.ndim < 3:
                continue
            probe = copy.deepcopy(raw_template)
            marker = np.full_like(value, 17 if np.issubdtype(value.dtype, np.integer) else 0.25)
            _set_path(probe, path, marker)
            transformed = self.data_input_transform(probe)
            for image_key in image_paths:
                image = np.asarray(transformed[self.image_key][image_key])
                expected = 17 if np.issubdtype(image.dtype, np.integer) else 0.25
                if np.allclose(image, expected):
                    image_paths[image_key] = path

        missing = [key for key, path in image_paths.items() if path is None]
        if missing:
            raise ValueError(f"Could not infer raw image paths for OpenPI image keys: {missing}.")
        return image_paths

    def _infer_raw_state_path(self, raw_template: dict) -> _Path:
        leaves = _flatten_leaves(raw_template)
        for path, value in leaves.items():
            value = np.asarray(value)
            if value.ndim > 1 or not np.issubdtype(value.dtype, np.number):
                continue
            probe = copy.deepcopy(raw_template)
            marker = np.full_like(value, 0.123, dtype=np.float32)
            _set_path(probe, path, marker)
            transformed = self.data_input_transform(probe)
            state = np.asarray(transformed[self.state_key])
            if state.shape[-1] >= marker.shape[-1] and np.allclose(state[..., : marker.shape[-1]], marker, atol=1e-5):
                return path
        raise ValueError("Could not infer raw state path from OpenPI data transform example.")

    # Tensor shape/value helpers. ACoB-Stream stores observations with a short time axis.
    @staticmethod
    def _last_time_np(value):
        value = np.asarray(value)
        if value.ndim >= 2 and value.shape[0] <= 8:
            value = value[-1]
        return value

    @staticmethod
    def _last_time_jax(value):
        value = jnp.asarray(value)
        if value.ndim >= 3 and value.shape[1] <= 8:
            value = value[:, -1]
        return value

    @staticmethod
    def _norm_dim(norm_stats, key: str) -> int | None:
        if norm_stats is None or key not in norm_stats:
            return None
        stats = norm_stats[key]
        return int(np.asarray(stats.mean).shape[-1])

    # State/image conversion into raw or final OpenPI shapes.
    def _state_for_openpi_np(self, value):
        state = self._last_time_np(value)
        return state[..., : self.state_input_dim] if self.state_input_dim is not None else state

    def _state_for_openpi_jax(self, value):
        state = self._last_time_jax(value)
        if self.state_input_dim is not None:
            state = state[..., : self.state_input_dim]
        state = self.normalize_state(state)
        return self._pad_last_dim(state, self.train_config.model.action_dim)

    @staticmethod
    def _frame_for_openpi_np(value):
        image = OpenPIObservationAdapter._last_time_np(value)
        if np.issubdtype(image.dtype, np.floating):
            if image.min(initial=0) < -0.1:
                return image.astype(np.float32)
            scale = 255.0 if image.max(initial=0) <= 1.5 else 1.0
            return (image * scale).clip(0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _frame_for_openpi_jax(value):
        image = OpenPIObservationAdapter._last_time_jax(value)
        if image.dtype == jnp.uint8:
            return image
        image = image.astype(jnp.float32)
        min_value = jnp.min(image)
        max_value = jnp.max(image)
        scaled_from_unit = image * 2.0 - 1.0
        scaled_from_uint8 = image / 255.0 * 2.0 - 1.0
        return jnp.where(min_value < -0.1, image, jnp.where(max_value <= 1.5, scaled_from_unit, scaled_from_uint8))

    @staticmethod
    def _pad_last_dim(value, target_dim: int):
        value = jnp.asarray(value)
        if value.shape[-1] >= target_dim:
            return value
        pad_width = [(0, 0)] * value.ndim
        pad_width[-1] = (0, target_dim - value.shape[-1])
        return jnp.pad(value, pad_width)

    def _stats_for_key(self, key: str):
        if self.norm_stats is None or key not in self.norm_stats:
            return None
        return self.norm_stats[key]

    def _normalize_value(self, value, key: str):
        stats = self._stats_for_key(key)
        value = jnp.asarray(value, dtype=jnp.float32)
        if stats is None:
            return value
        if self.use_quantile_norm:
            q01 = jnp.asarray(stats.q01, dtype=value.dtype)[..., : value.shape[-1]]
            q99 = jnp.asarray(stats.q99, dtype=value.dtype)[..., : value.shape[-1]]
            return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        mean = jnp.asarray(stats.mean, dtype=value.dtype)[..., : value.shape[-1]]
        std = jnp.asarray(stats.std, dtype=value.dtype)[..., : value.shape[-1]]
        return (value - mean) / (std + 1e-6)

    def _unnormalize_value(self, value, key: str):
        stats = self._stats_for_key(key)
        value = jnp.asarray(value, dtype=jnp.float32)
        if stats is None:
            return value
        target_dim = value.shape[-1]
        if self.use_quantile_norm:
            q01 = jnp.asarray(stats.q01, dtype=value.dtype)
            q99 = jnp.asarray(stats.q99, dtype=value.dtype)
            stat_dim = q01.shape[-1]
            if stat_dim < target_dim:
                restored = (value[..., :stat_dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
                return jnp.concatenate([restored, value[..., stat_dim:]], axis=-1)
            q01 = q01[..., :target_dim]
            q99 = q99[..., :target_dim]
            return (value + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        mean = jnp.asarray(stats.mean, dtype=value.dtype)
        std = jnp.asarray(stats.std, dtype=value.dtype)
        stat_dim = mean.shape[-1]
        if stat_dim < target_dim:
            restored = value[..., :stat_dim] * (std + 1e-6) + mean
            return jnp.concatenate([restored, value[..., stat_dim:]], axis=-1)
        mean = mean[..., :target_dim]
        std = std[..., :target_dim]
        return value * (std + 1e-6) + mean

    def normalize_state(self, state):
        return self._normalize_value(state, self.state_key)

    def normalize_action(self, actions):
        return self._normalize_value(actions, self.action_key)

    def unnormalize_action(self, actions):
        return self._unnormalize_value(actions, self.action_key)

    def _raw_state_for_action_transform(self, observations: Mapping[str, Any]):
        if self.state_key not in observations:
            raise KeyError(f"Robot observations are missing required OpenPI state key: {self.state_key}")
        state = self._last_time_jax(observations[self.state_key])
        if self.state_input_dim is not None:
            state = state[..., : self.state_input_dim]
        return state

    @staticmethod
    def _apply_delta_transform(actions, state, transform):
        mask = getattr(transform, "mask", None)
        if mask is None:
            return actions
        mask = jnp.asarray(mask, dtype=bool)
        dims = mask.shape[-1]
        state_delta = jnp.where(mask, state[..., :dims], 0)
        return actions.at[..., :dims].add(-state_delta[..., None, :])

    @staticmethod
    def _apply_absolute_transform(actions, state, transform):
        mask = getattr(transform, "mask", None)
        if mask is None:
            return actions
        mask = jnp.asarray(mask, dtype=bool)
        dims = mask.shape[-1]
        state_delta = jnp.where(mask, state[..., :dims], 0)
        if actions.ndim == state_delta.ndim:
            return actions.at[..., :dims].add(state_delta)
        return actions.at[..., :dims].add(state_delta[..., None, :])

    def _apply_config_action_inputs(self, actions, observations: Mapping[str, Any]):
        state = self._raw_state_for_action_transform(observations)
        for transform in self.data_action_input_transforms:
            if type(transform).__name__ == "DeltaActions":
                actions = self._apply_delta_transform(actions, state, transform)
        return actions

    def _apply_config_action_outputs(self, actions, observations: Mapping[str, Any]):
        state = self._raw_state_for_action_transform(observations)
        actions = self.unnormalize_action(actions)
        for transform in self.data_action_output_transforms:
            if type(transform).__name__ == "AbsoluteActions":
                actions = self._apply_absolute_transform(actions, state, transform)
        return actions

    def model_actions_to_env_actions(self, actions, observations: Mapping[str, Any]):
        return self._apply_config_action_outputs(actions, observations)

    def policy_output_actions(self, actions, observation: _model.Observation):
        outputs = {
            "state": np.asarray(jax.device_get(observation.state[0])),
            "actions": np.asarray(jax.device_get(actions)),
        }
        transformed = self.output_transform(outputs)
        return jnp.asarray(transformed["actions"], dtype=jnp.float32)

    # Actor path: one robot observation -> OpenPI raw dict -> OpenPI transforms.
    def _raw_single(self, obs: dict, actions: np.ndarray | None = None) -> dict:
        raw = copy.deepcopy(self.raw_template)
        for image_key, raw_path in self.raw_image_paths.items():
            if image_key not in obs:
                raise KeyError(f"Robot observation is missing required OpenPI image key: {image_key}")
            _set_path(raw, raw_path, self._frame_for_openpi_np(obs[image_key]))
        if self.state_key not in obs:
            raise KeyError(f"Robot observation is missing required OpenPI state key: {self.state_key}")
        _set_path(raw, self.raw_state_path, self._state_for_openpi_np(obs[self.state_key]))
        raw[self.prompt_key] = self.task_desc
        if actions is not None:
            raw[self.raw_action_key] = actions
        return raw

    def observation(self, obs: dict) -> _model.Observation:
        obs = b64_to_numpy(obs)
        raw = self._raw_single(obs)
        transformed = self.input_transform(raw)
        batched = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None, ...], transformed)
        observation = _model.Observation.from_dict(batched)
        return observation

    # Learner path: JIT-friendly batch repack directly into model-ready fields.
    def batch(self, observations: dict) -> _model.Observation:
        for image_key in self.source_image_keys:
            if image_key not in observations:
                raise KeyError(f"ACoB batch observations are missing required OpenPI image key: {image_key}")
        if self.state_key not in observations:
            raise KeyError(f"ACoB batch observations are missing required OpenPI state key: {self.state_key}")

        images = {key: self._frame_for_openpi_jax(observations[key]) for key in self.source_image_keys}
        first_image = next(iter(images.values()))
        images.update({key: jnp.zeros_like(first_image) for key in self.padding_image_keys})
        state = self._state_for_openpi_jax(observations[self.state_key])
        batch_size = state.shape[0]

        openpi_batch = {
            self.image_key: {key: images[key] for key in self.model_image_keys},
            self.image_mask_key: {
                key: jnp.full((batch_size,), key in self.source_image_keys, dtype=jnp.bool_)
                for key in self.model_image_keys
            },
            self.state_key: state,
            "tokenized_prompt": jnp.broadcast_to(
                jnp.asarray(self.tokenized_prompt), (batch_size, *self.tokenized_prompt.shape)
            ),
            "tokenized_prompt_mask": jnp.broadcast_to(
                jnp.asarray(self.tokenized_prompt_mask), (batch_size, *self.tokenized_prompt_mask.shape)
            ),
        }
        if self.token_ar_mask is not None:
            openpi_batch["token_ar_mask"] = jnp.broadcast_to(
                jnp.asarray(self.token_ar_mask), (batch_size, *self.token_ar_mask.shape)
            )
        if self.token_loss_mask is not None:
            openpi_batch["token_loss_mask"] = jnp.broadcast_to(
                jnp.asarray(self.token_loss_mask), (batch_size, *self.token_loss_mask.shape)
            )

        observation = _model.Observation.from_dict(openpi_batch)
        return observation

    def _batch_size_from_observations(self, observations: Mapping[str, Any]) -> int:
        if self.state_key not in observations:
            return 1
        return int(jnp.asarray(observations[self.state_key]).shape[0])

    # Training target path: env action -> OpenPI configured action horizon/action dimension.
    def action_chunk(self, actions: jnp.ndarray, observations: Mapping[str, Any]) -> jnp.ndarray:
        actions = jnp.asarray(actions, dtype=jnp.float32)
        if actions.ndim == 1:
            actions = actions[None, None, :]
        elif actions.ndim == 2:
            actions = (
                actions[None, ...]
                if actions.shape[0] != self._batch_size_from_observations(observations)
                else actions[:, None, :]
            )
        elif actions.ndim != 3:
            raise ValueError(f"Expected actions with shape (A,), (C, A), or (B, C, A), got {actions.shape}")

        target_horizon = int(self.action_horizon)
        if int(actions.shape[1]) != target_horizon:
            raise ValueError(
                f"Expected env action chunk horizon {target_horizon}, got {actions.shape[1]} for actions shape {actions.shape}. "
                f"OpenPI model.action_horizon is {self.train_config.model.action_horizon}; ACoB may use a longer model horizon for timing/debug while executing only the env horizon."
            )
        action_chunk = actions

        action_chunk = self._apply_config_action_inputs(action_chunk, observations)
        action_chunk = self.normalize_action(action_chunk)
        if int(action_chunk.shape[-1]) > int(self.train_config.model.action_dim):
            raise ValueError(
                f"Normalized action dimension {action_chunk.shape[-1]} exceeds OpenPI "
                f"model.action_dim={self.train_config.model.action_dim}; refusing to crop action targets."
            )
        action_chunk = self._pad_last_dim(action_chunk, self.train_config.model.action_dim)
        return action_chunk
