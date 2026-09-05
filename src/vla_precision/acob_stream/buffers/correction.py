"""Sparse human-correction buffer."""

from __future__ import annotations

from vla_precision.acob_stream.buffers.replay import ReplayBuffer


class CorrectionBuffer(ReplayBuffer):
    """Replay storage whose sparse intervention segments seed their own pixels.

    Human corrections are not necessarily adjacent in the source trajectory.
    The inherited insertion path therefore preserves the paper implementation's
    discontinuity padding while leaving reward/done/mask values untouched.
    """

    def __init__(self, *args, **kwargs):
        kwargs["detect_pixel_sequence_boundaries"] = True
        super().__init__(*args, **kwargs)
