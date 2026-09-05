"""Deterministic identity for code participating in distributed execution.

This module is intentionally limited to the standard library and the small
``vla_precision`` package initializer.  Robot-side processes can therefore
verify code compatibility without importing OpenPI, JAX, or training code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from vla_precision import __version__

ACOB_STREAM_PROTOCOL_REVISION = "2"
OPENPI_PINNED_COMMIT = "2d70d966582e711128ad8358d8dbf23d2cc3d658"


@dataclass(frozen=True)
class RuntimeCodeIdentity:
    """Code facts that must agree between communicating processes."""

    package_version: str
    acob_stream_protocol_revision: str
    openpi_commit: str
    source_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def compatibility_dict(self, *, strict: bool | None) -> dict[str, str]:
        """Return the identity fields required by the selected check mode."""
        if strict is None:
            return {}
        if strict:
            return self.to_dict()
        return {
            "acob_stream_protocol_revision": self.acob_stream_protocol_revision,
            "openpi_commit": self.openpi_commit,
        }


def compatible_runtime_identity(
    left: RuntimeCodeIdentity | dict[str, str],
    right: RuntimeCodeIdentity | dict[str, str],
    *,
    strict: bool | None,
) -> bool:
    """Compare complete sources in strict mode and protocol facts otherwise."""
    if strict is None:
        return True
    left_values = left.to_dict() if isinstance(left, RuntimeCodeIdentity) else left
    right_values = right.to_dict() if isinstance(right, RuntimeCodeIdentity) else right
    fields = (
        ("package_version", "acob_stream_protocol_revision", "openpi_commit", "source_sha256")
        if strict
        else ("acob_stream_protocol_revision", "openpi_commit")
    )
    return all(left_values.get(field) == right_values.get(field) for field in fields)


def consistency_mode(left: bool | None, right: bool | None) -> bool | None:
    """Use the strongest consistency level requested by either endpoint."""
    if left is True or right is True:
        return True
    if left is False or right is False:
        return False
    return None


def hash_package_sources(package_root: str | Path) -> str:
    """Hash sorted relative Python source names and bytes below a package root."""
    root = Path(package_root)
    digest = hashlib.sha256()
    for source_path in sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()):
        relative_name = source_path.relative_to(root).as_posix().encode("utf-8")
        contents = source_path.read_bytes()
        # Length prefixes make the name/content stream unambiguous without
        # changing the source bytes included in the identity.
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def runtime_code_identity() -> RuntimeCodeIdentity:
    """Return the process identity, computing the package source hash once."""
    package_root = Path(__file__).resolve().parent
    return RuntimeCodeIdentity(
        package_version=__version__,
        acob_stream_protocol_revision=ACOB_STREAM_PROTOCOL_REVISION,
        openpi_commit=OPENPI_PINNED_COMMIT,
        source_sha256=hash_package_sources(package_root),
    )


def agentlace_contract_version(
    shared_config_sha256: str,
    code_identity: RuntimeCodeIdentity | None = None,
    *,
    strict: bool | None = True,
) -> str:
    """Encode the shared run and code identity into AgentLace's version field."""
    if strict is None:
        return json.dumps({"distributed_consistency": None}, sort_keys=True, separators=(",", ":"))
    identity = code_identity or runtime_code_identity()
    metadata = {
        "runtime_code_identity": identity.compatibility_dict(strict=strict),
        "shared_config_sha256": shared_config_sha256,
    }
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))
