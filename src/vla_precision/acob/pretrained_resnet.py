"""Download, verify and load the frozen ACoB critic ResNet-10 parameters."""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CRITIC_RESNET10_PARAMS_PATH = "./assets/resnet10_params.pkl"
RESNET10_PARAMS_URL = "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
RESNET10_PARAMS_SHA256 = "175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b"

_EXPECTED_BACKBONE_LEAF_COUNT = 36
_EXPECTED_BACKBONE_PARAM_COUNT = 4_905_792
LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_resnet10_params(params_path: str = DEFAULT_CRITIC_RESNET10_PARAMS_PATH) -> Path:
    """Download the public SERL ResNet-10 weights once and verify the pinned asset."""
    resolved_path = Path(params_path or DEFAULT_CRITIC_RESNET10_PARAMS_PATH).expanduser().resolve()
    if resolved_path.is_file():
        checksum = _sha256(resolved_path)
        if checksum != RESNET10_PARAMS_SHA256:
            raise ValueError(
                f"Critic ResNet-10 checksum mismatch at {resolved_path}: "
                f"expected {RESNET10_PARAMS_SHA256}, got {checksum}"
            )
        return resolved_path

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_suffix(resolved_path.suffix + ".download")
    LOGGER.info("downloading critic ResNet-10 weights to %s", resolved_path)
    try:
        with urllib.request.urlopen(RESNET10_PARAMS_URL, timeout=120) as response:
            with temporary_path.open("wb") as file:
                shutil.copyfileobj(response, file)
        checksum = _sha256(temporary_path)
        if checksum != RESNET10_PARAMS_SHA256:
            raise ValueError(
                f"Downloaded critic ResNet-10 checksum mismatch: expected {RESNET10_PARAMS_SHA256}, got {checksum}"
            )
        temporary_path.replace(resolved_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    LOGGER.info("downloaded critic ResNet-10 weights: %s", resolved_path)
    return resolved_path


def _load_and_validate_source(critic_params: Any, params_path: str):
    import pickle

    import numpy as np
    from flax import traverse_util

    resolved_path = ensure_resnet10_params(params_path)

    with resolved_path.open("rb") as file:
        source = pickle.load(file)
    if not isinstance(source, dict):
        raise TypeError(
            f"Expected a dict in critic ResNet-10 weights {resolved_path}, got {type(source).__name__}."
        )

    source_flat = traverse_util.flatten_dict(source)
    source_backbone_flat = {
        key: value for key, value in source_flat.items() if not key or key[0] != "output_head"
    }
    source_param_count = sum(int(np.asarray(value).size) for value in source_backbone_flat.values())
    if len(source_backbone_flat) != _EXPECTED_BACKBONE_LEAF_COUNT:
        raise ValueError(
            f"Unexpected ResNet-10 backbone leaf count in {resolved_path}: "
            f"expected {_EXPECTED_BACKBONE_LEAF_COUNT}, got {len(source_backbone_flat)}."
        )
    if source_param_count != _EXPECTED_BACKBONE_PARAM_COUNT:
        raise ValueError(
            f"Unexpected ResNet-10 backbone parameter count in {resolved_path}: "
            f"expected {_EXPECTED_BACKBONE_PARAM_COUNT}, got {source_param_count}."
        )

    critic_flat = traverse_util.flatten_dict(critic_params)
    prefixes = {
        key[: key.index("pretrained_encoder") + 1]
        for key in critic_flat
        if "pretrained_encoder" in key
    }
    if len(prefixes) != 1:
        raise ValueError(
            "Expected exactly one shared critic pretrained_encoder parameter subtree, "
            f"found {len(prefixes)}: {sorted(prefixes)}"
        )
    prefix = next(iter(prefixes))
    target_backbone_flat = {
        key[len(prefix) :]: value
        for key, value in critic_flat.items()
        if key[: len(prefix)] == prefix
    }

    source_keys = set(source_backbone_flat)
    target_keys = set(target_backbone_flat)
    if source_keys != target_keys:
        missing = sorted(source_keys - target_keys)
        unexpected = sorted(target_keys - source_keys)
        raise ValueError(
            "Critic ResNet-10 parameter tree does not match the pretrained asset: "
            f"missing_target_leaves={missing}, unexpected_target_leaves={unexpected}, path={resolved_path}"
        )

    for key in sorted(source_keys):
        source_shape = tuple(np.asarray(source_backbone_flat[key]).shape)
        target_shape = tuple(target_backbone_flat[key].shape)
        if source_shape != target_shape:
            leaf_name = "/".join(str(part) for part in key)
            raise ValueError(
                f"Critic ResNet-10 shape mismatch for {leaf_name}: "
                f"expected target shape {target_shape}, pretrained shape {source_shape}, path={resolved_path}"
            )

    return resolved_path, critic_flat, prefix, target_backbone_flat, source_backbone_flat


def load_pretrained_resnet10_critic_params(critic_params: Any, params_path: str):
    """Replace the single shared frozen critic backbone with pretrained ResNet-10 weights."""
    import jax.numpy as jnp
    from flax import traverse_util
    from flax.core import FrozenDict, freeze, unfreeze

    was_frozen = isinstance(critic_params, FrozenDict)
    mutable_params = unfreeze(critic_params) if was_frozen else critic_params
    resolved_path, critic_flat, prefix, target_flat, source_flat = _load_and_validate_source(
        mutable_params,
        params_path,
    )

    loaded_flat = dict(critic_flat)
    for key, source_value in source_flat.items():
        target_value = target_flat[key]
        loaded_flat[prefix + key] = jnp.asarray(source_value, dtype=target_value.dtype)
    loaded_params = traverse_util.unflatten_dict(loaded_flat)
    if was_frozen:
        loaded_params = freeze(loaded_params)

    message = (
        f"Loaded critic pretrained ResNet-10: path={resolved_path}, "
        f"leaves={len(source_flat)}, params={_EXPECTED_BACKBONE_PARAM_COUNT}"
    )
    print(message, flush=True)
    return loaded_params


def verify_pretrained_resnet10_critic_params(critic_params: Any, params_path: str, *, label: str) -> None:
    """Fail if a restored critic carries a random/legacy frozen ResNet backbone."""
    import jax
    import numpy as np

    resolved_path, _, _, target_flat, source_flat = _load_and_validate_source(critic_params, params_path)
    for key in sorted(source_flat):
        current = np.asarray(jax.device_get(target_flat[key]))
        expected = np.asarray(source_flat[key]).astype(current.dtype, copy=False)
        if not np.allclose(current, expected, rtol=1e-6, atol=1e-7):
            leaf_name = "/".join(str(part) for part in key)
            max_abs_diff = float(np.max(np.abs(current.astype(np.float64) - expected.astype(np.float64))))
            raise RuntimeError(
                f"{label} does not contain the configured pretrained critic ResNet-10 weights: "
                f"first_mismatch={leaf_name}, max_abs_diff={max_abs_diff:.9g}, path={resolved_path}. "
                "Do not resume this legacy random-backbone critic; start a fresh critic or regenerate warm-up/pretrain checkpoints."
            )
    print(
        f"Verified critic pretrained ResNet-10: label={label}, path={resolved_path}, "
        f"leaves={len(source_flat)}",
        flush=True,
    )
