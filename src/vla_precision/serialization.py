"""Lightweight serialization helpers shared by robot and inference processes."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO

import numpy as np


def b64_to_numpy(value):
    """Decode Pyro/AgentLace base64 NumPy payloads while preserving normal text."""
    if isinstance(value, str):
        try:
            return np.load(BytesIO(base64.b64decode(value)), allow_pickle=False)
        except (binascii.Error, EOFError, OSError, TypeError, ValueError):
            return value
    if isinstance(value, dict):
        return {key: b64_to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [b64_to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(b64_to_numpy(item) for item in value)
    return value
