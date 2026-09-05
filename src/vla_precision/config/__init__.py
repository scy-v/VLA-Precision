"""Typed, top-level configuration for every VLA-Precision entrypoint."""

from vla_precision.config.loader import (
    ResolvedConfig,
    ResolvedStage1Config,
    load_config,
    load_stage1_config,
)
from vla_precision.config.schema import RootConfig, Stage1Config

__all__ = [
    "ResolvedConfig",
    "ResolvedStage1Config",
    "RootConfig",
    "Stage1Config",
    "load_config",
    "load_stage1_config",
]
