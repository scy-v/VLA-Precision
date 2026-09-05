"""Dataset preparation utilities.

Index selection is materialized into the Hugging Face dataset before a
sampler or data loader is constructed.  Sampling code therefore reads the
already-resolved state and action columns without applying index transforms.
"""

from vla_precision.data.indexing import (
    INDEX_SCHEMA_VERSION,
    IndexMaterializationMetadata,
    IndexMaterializationResult,
    materialize_indices,
    materialize_lerobot_indices,
)

__all__ = [
    "INDEX_SCHEMA_VERSION",
    "IndexMaterializationMetadata",
    "IndexMaterializationResult",
    "materialize_indices",
    "materialize_lerobot_indices",
]
