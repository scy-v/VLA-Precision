"""Camera capability boundary independent of a camera vendor."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Camera(Protocol):
    @property
    def name(self) -> str: ...

    def capture(self) -> np.ndarray: ...

    def close(self) -> None: ...
