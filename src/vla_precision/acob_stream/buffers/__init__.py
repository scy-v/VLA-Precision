"""The replay, correction, and context buffers used by ACoB-Stream."""

from vla_precision.acob_stream.buffers.context import ContextBuffer, ContextSource

__all__ = ["ContextBuffer", "ContextSource", "CorrectionBuffer", "ReplayBuffer"]


def __getattr__(name: str):
    """Keep the lightweight ContextBuffer usable without training extras."""
    if name == "ReplayBuffer":
        from vla_precision.acob_stream.buffers.replay import ReplayBuffer

        return ReplayBuffer
    if name == "CorrectionBuffer":
        from vla_precision.acob_stream.buffers.correction import CorrectionBuffer

        return CorrectionBuffer
    raise AttributeError(name)
