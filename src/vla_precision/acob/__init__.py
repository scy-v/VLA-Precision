"""ACoB algorithm package with lazy optional-training imports."""

from __future__ import annotations

__all__ = ["ACoBAgent", "ACoBState", "create_agent"]


def __getattr__(name: str):
    if name in {"ACoBAgent", "ACoBState"}:
        from vla_precision.acob.agent import ACoBAgent, ACoBState

        return {"ACoBAgent": ACoBAgent, "ACoBState": ACoBState}[name]
    if name == "create_agent":
        from vla_precision.acob.factory import create_agent

        return create_agent
    raise AttributeError(name)
