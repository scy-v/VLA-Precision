"""OpenPI integration boundary.

Keep package import lightweight so the robot-side WebSocket client does not
need the full OpenPI/JAX model environment.
"""

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

__all__ = ["install_lerobot_import_compat"]
