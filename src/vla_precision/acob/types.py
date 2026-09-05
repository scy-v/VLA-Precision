"""Array/tree aliases used by the ACoB implementation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import flax
import jax.numpy as jnp
import numpy as np

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any
InfoDict = dict[str, float]
Array = np.ndarray | jnp.ndarray
Data = Array | dict[str, "Data"]
Batch = dict[str, Data]
ModuleMethod = str | Callable | None
