"""Disk-backed OpenPI prefix state for the ACoB-Stream context buffer.

The context buffer stores BF16 KV values as their raw uint16 bit patterns. It
supports a single mmap for small datasets and lazily-created mmap shards for
long-running online collection.
"""

from __future__ import annotations

import json
import shutil
from enum import IntEnum
from pathlib import Path

import ml_dtypes
import numpy as np


class ContextSource(IntEnum):
    """Origin of a context row referenced by a replay transition."""

    PREPROCESS = 0
    TRAINING = 1
    INITIAL = 2


def _path_contains_files(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    if path.is_file() or path.is_symlink():
        return True
    return any(child.is_file() or child.is_symlink() for child in path.rglob("*"))


def bf16_to_uint16_bits(value) -> np.ndarray:
    arr = np.asarray(value, dtype=ml_dtypes.bfloat16)
    return arr.view(np.uint16)


def uint16_bits_to_bf16(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.uint16)
    return arr.view(ml_dtypes.bfloat16)


def _validate_storage_shapes(
    logical_embedding_shape, logical_mask_shape, storage_embedding_shape, storage_mask_shape
) -> None:
    if len(logical_embedding_shape) != 5 or len(storage_embedding_shape) != 5:
        raise ValueError(
            f"OpenPI KV embedding shapes must be rank-5, got {logical_embedding_shape} and {storage_embedding_shape}"
        )
    if len(logical_mask_shape) != 1 or len(storage_mask_shape) != 1:
        raise ValueError(f"OpenPI KV mask shapes must be rank-1, got {logical_mask_shape} and {storage_mask_shape}")
    if (
        logical_embedding_shape[:2] != storage_embedding_shape[:2]
        or logical_embedding_shape[3:] != storage_embedding_shape[3:]
    ):
        raise ValueError(
            f"Storage KV shape {storage_embedding_shape} must match logical shape {logical_embedding_shape} except token axis"
        )
    if storage_embedding_shape[2] > logical_embedding_shape[2]:
        raise ValueError(
            f"Storage token count {storage_embedding_shape[2]} cannot exceed logical token count {logical_embedding_shape[2]}"
        )
    if storage_mask_shape[0] != storage_embedding_shape[2]:
        raise ValueError(
            f"Storage mask tokens {storage_mask_shape[0]} must match storage KV tokens {storage_embedding_shape[2]}"
        )
    if logical_mask_shape[0] != logical_embedding_shape[2]:
        raise ValueError(
            f"Logical mask tokens {logical_mask_shape[0]} must match logical KV tokens {logical_embedding_shape[2]}"
        )


class ContextBuffer:
    def __init__(self, root: Path, *, mode: str = "r"):
        self.root = Path(root).expanduser()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Context buffer metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        self.capacity = int(self.metadata["capacity"])
        self.embedding_shape = tuple(int(x) for x in self.metadata["embedding_shape"])
        self.mask_shape = tuple(int(x) for x in self.metadata["embedding_mask_shape"])
        self.logical_embedding_shape = tuple(
            int(x) for x in self.metadata.get("logical_embedding_shape", self.embedding_shape)
        )
        self.logical_mask_shape = tuple(
            int(x) for x in self.metadata.get("logical_embedding_mask_shape", self.mask_shape)
        )
        self.mode = mode
        self._mmap_mode = "r+" if mode != "r" else "r"
        self.storage_backend = self.metadata.get("storage_backend", "single_npy")
        self.shard_capacity = None
        self._shards = {}
        if self.storage_backend == "sharded_npy":
            self.shard_capacity = int(self.metadata["shard_capacity"])
            self.kv_bits = self.masks = self.offsets = None
        else:
            self.kv_bits = np.load(self.root / "kv_bf16_bits.npy", mmap_mode=self._mmap_mode)
            self.masks = np.load(self.root / "kv_masks.npy", mmap_mode=self._mmap_mode)
            self.offsets = np.load(self.root / "kv_offsets.npy", mmap_mode=self._mmap_mode)

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        capacity: int,
        embedding_shape: tuple[int, ...],
        embedding_mask_shape: tuple[int, ...],
        overwrite: bool,
        metadata: dict | None = None,
        storage_embedding_shape: tuple[int, ...] | None = None,
        storage_embedding_mask_shape: tuple[int, ...] | None = None,
        shard_capacity: int | None = None,
    ) -> ContextBuffer:
        root = Path(root).expanduser()
        if root.exists():
            if not overwrite and _path_contains_files(root):
                raise FileExistsError(f"Context buffer directory already contains files: {root}")
            if overwrite:
                shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        capacity = int(capacity)
        logical_embedding_shape = tuple(int(x) for x in embedding_shape)
        logical_embedding_mask_shape = tuple(int(x) for x in embedding_mask_shape)
        storage_embedding_shape = tuple(int(x) for x in (storage_embedding_shape or logical_embedding_shape))
        storage_embedding_mask_shape = tuple(
            int(x) for x in (storage_embedding_mask_shape or logical_embedding_mask_shape)
        )
        _validate_storage_shapes(
            logical_embedding_shape, logical_embedding_mask_shape, storage_embedding_shape, storage_embedding_mask_shape
        )
        if capacity <= 0:
            raise ValueError(f"Context buffer capacity must be positive, got {capacity}")
        shard_capacity = None if shard_capacity is None else int(shard_capacity)
        if shard_capacity is not None and shard_capacity <= 0:
            raise ValueError(f"Context buffer shard capacity must be positive, got {shard_capacity}")
        if shard_capacity is None:
            np.lib.format.open_memmap(
                root / "kv_bf16_bits.npy", mode="w+", dtype=np.uint16, shape=(capacity, *storage_embedding_shape)
            )
            np.lib.format.open_memmap(
                root / "kv_masks.npy", mode="w+", dtype=np.bool_, shape=(capacity, *storage_embedding_mask_shape)
            )
            np.lib.format.open_memmap(root / "kv_offsets.npy", mode="w+", dtype=np.int32, shape=(capacity,))
        payload = {
            "format": "vla_precision_context_buffer",
            "format_version": 3 if shard_capacity is not None else 2,
            "capacity": capacity,
            "embedding_dtype": "bfloat16_bits_uint16",
            "storage_backend": "sharded_npy" if shard_capacity is not None else "single_npy",
            **({"shard_capacity": shard_capacity} if shard_capacity is not None else {}),
            "embedding_shape": list(storage_embedding_shape),
            "embedding_mask_shape": list(storage_embedding_mask_shape),
            "logical_embedding_shape": list(logical_embedding_shape),
            "logical_embedding_mask_shape": list(logical_embedding_mask_shape),
            "compact_valid_tokens": storage_embedding_shape[2] < logical_embedding_shape[2],
            "stored_prefix_tokens": int(storage_embedding_shape[2]),
            "logical_prefix_tokens": int(logical_embedding_shape[2]),
        }
        if metadata:
            payload.update(metadata)
        (root / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        return cls(root, mode="r+")

    @classmethod
    def open_or_create(
        cls,
        root: Path,
        *,
        capacity: int,
        embedding_shape: tuple[int, ...],
        embedding_mask_shape: tuple[int, ...],
        overwrite: bool = False,
        metadata: dict | None = None,
        mode: str = "r+",
        storage_embedding_shape: tuple[int, ...] | None = None,
        storage_embedding_mask_shape: tuple[int, ...] | None = None,
        shard_capacity: int | None = None,
    ) -> ContextBuffer:
        root = Path(root).expanduser()
        metadata_path = root / "metadata.json"
        expected_logical_shape = tuple(int(x) for x in embedding_shape)
        expected_logical_mask_shape = tuple(int(x) for x in embedding_mask_shape)
        expected_storage_shape = tuple(int(x) for x in (storage_embedding_shape or expected_logical_shape))
        expected_storage_mask_shape = tuple(
            int(x) for x in (storage_embedding_mask_shape or expected_logical_mask_shape)
        )
        _validate_storage_shapes(
            expected_logical_shape, expected_logical_mask_shape, expected_storage_shape, expected_storage_mask_shape
        )
        if metadata_path.exists():
            store = cls(root, mode=mode)
            if store.capacity < int(capacity):
                raise ValueError(f"Existing context buffer capacity {store.capacity} < requested {capacity}: {root}")
            if store.logical_embedding_shape != expected_logical_shape:
                raise ValueError(
                    f"Existing context buffer logical shape {store.logical_embedding_shape} "
                    f"!= requested {expected_logical_shape}: {root}"
                )
            if store.logical_mask_shape != expected_logical_mask_shape:
                raise ValueError(
                    f"Existing context buffer logical mask shape {store.logical_mask_shape} "
                    f"!= requested {expected_logical_mask_shape}: {root}"
                )
            if (
                store.embedding_shape[:2] != expected_storage_shape[:2]
                or store.embedding_shape[3:] != expected_storage_shape[3:]
            ):
                raise ValueError(
                    f"Existing context buffer storage shape {store.embedding_shape} is incompatible with {expected_storage_shape}: {root}"
                )
            if store.embedding_shape[2] < expected_storage_shape[2]:
                raise ValueError(
                    f"Existing context buffer stored tokens {store.embedding_shape[2]} "
                    f"< requested {expected_storage_shape[2]}: {root}"
                )
            return store

        if _path_contains_files(root) and not overwrite:
            raise FileExistsError(f"Context buffer directory contains files but has no metadata.json: {root}")
        return cls.create(
            root,
            capacity=capacity,
            embedding_shape=expected_logical_shape,
            embedding_mask_shape=expected_logical_mask_shape,
            overwrite=True if root.exists() else overwrite,
            metadata=metadata,
            storage_embedding_shape=expected_storage_shape,
            storage_embedding_mask_shape=expected_storage_mask_shape,
            shard_capacity=shard_capacity,
        )

    @property
    def kv_record_bytes(self) -> int:
        return int(np.prod(self.embedding_shape)) * np.dtype(np.uint16).itemsize

    @property
    def mask_record_bytes(self) -> int:
        return int(np.prod(self.mask_shape)) * np.dtype(np.bool_).itemsize

    @property
    def offset_record_bytes(self) -> int:
        return np.dtype(np.int32).itemsize

    @property
    def record_bytes(self) -> int:
        return self.kv_record_bytes + self.mask_record_bytes + self.offset_record_bytes

    def has_written_records(self) -> bool:
        """Return whether at least one context row has been populated."""
        if self.shard_capacity is None:
            return bool(np.any(self.offsets != 0))
        for path in self.root.glob("kv_offsets_*.npy"):
            offsets = np.load(path, mmap_mode="r")
            if np.any(offsets != 0):
                return True
        return False

    def _index_for_ids(self, ids: np.ndarray):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return ids
        start = int(ids[0])
        if np.array_equal(ids, np.arange(start, start + ids.size, dtype=np.int64)):
            return slice(start, start + ids.size)
        return ids

    def _validated_flat_ids(self, ids) -> np.ndarray:
        flat_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if flat_ids.size and (int(flat_ids.min()) < 0 or int(flat_ids.max()) >= self.capacity):
            raise IndexError(
                f"Context ids must be in [0, {self.capacity}), got min={int(flat_ids.min())}, max={int(flat_ids.max())}"
            )
        return flat_ids

    def _shard_paths(self, shard_index: int) -> tuple[Path, Path, Path]:
        suffix = f"{int(shard_index):06d}.npy"
        return (
            self.root / f"kv_bf16_bits_{suffix}",
            self.root / f"kv_masks_{suffix}",
            self.root / f"kv_offsets_{suffix}",
        )

    def _shard_row_count(self, shard_index: int) -> int:
        start = int(shard_index) * int(self.shard_capacity)
        if start < 0 or start >= self.capacity:
            raise IndexError(f"Context shard index {shard_index} is outside capacity {self.capacity}")
        return min(int(self.shard_capacity), self.capacity - start)

    def _open_shard(self, shard_index: int, *, create: bool):
        shard_index = int(shard_index)
        if shard_index in self._shards:
            return self._shards[shard_index]
        paths = self._shard_paths(shard_index)
        exists = tuple(path.exists() for path in paths)
        if all(exists):
            arrays = tuple(np.load(path, mmap_mode=self._mmap_mode) for path in paths)
        elif any(exists):
            raise FileNotFoundError(
                f"Incomplete context shard {shard_index} at {self.root}; "
                f"files={[(path.name, present) for path, present in zip(paths, exists)]}"
            )
        elif not create:
            raise FileNotFoundError(f"Context shard {shard_index} has not been created at {self.root}")
        else:
            if self.mode == "r":
                raise PermissionError(f"Cannot create context shard {shard_index} from read-only buffer {self.root}")
            rows = self._shard_row_count(shard_index)
            arrays = (
                np.lib.format.open_memmap(paths[0], mode="w+", dtype=np.uint16, shape=(rows, *self.embedding_shape)),
                np.lib.format.open_memmap(paths[1], mode="w+", dtype=np.bool_, shape=(rows, *self.mask_shape)),
                np.lib.format.open_memmap(paths[2], mode="w+", dtype=np.int32, shape=(rows,)),
            )
        self._shards[shard_index] = arrays
        return arrays

    def _get_sharded_batch_bits(self, ids):
        ids_array = np.asarray(ids, dtype=np.int64)
        flat_ids = self._validated_flat_ids(ids_array)
        embeddings = np.empty((flat_ids.size, *self.embedding_shape), dtype=np.uint16)
        masks = np.empty((flat_ids.size, *self.mask_shape), dtype=np.bool_)
        offsets = np.empty((flat_ids.size,), dtype=np.int32)
        if flat_ids.size:
            shard_ids = flat_ids // int(self.shard_capacity)
            for shard_index in np.unique(shard_ids):
                rows = np.flatnonzero(shard_ids == shard_index)
                local_ids = flat_ids[rows] - int(shard_index) * int(self.shard_capacity)
                shard_bits, shard_masks, shard_offsets = self._open_shard(int(shard_index), create=False)
                embeddings[rows] = np.asarray(shard_bits[local_ids], dtype=np.uint16)
                masks[rows] = np.asarray(shard_masks[local_ids], dtype=np.bool_)
                offsets[rows] = np.asarray(shard_offsets[local_ids], dtype=np.int32)
        prefix = tuple(ids_array.shape)
        return (
            embeddings.reshape(*prefix, *self.embedding_shape),
            masks.reshape(*prefix, *self.mask_shape),
            offsets.reshape(prefix),
        )

    def _compact_for_storage(self, embeddings, masks, offsets):
        embeddings = np.asarray(embeddings)
        masks = np.asarray(masks, dtype=np.bool_)
        offsets = np.asarray(offsets, dtype=np.int32).reshape(-1)
        if tuple(embeddings.shape[1:]) == self.embedding_shape and tuple(masks.shape[1:]) == self.mask_shape:
            return embeddings, masks, offsets
        if tuple(embeddings.shape[1:]) != self.logical_embedding_shape:
            raise ValueError(
                f"KV embeddings shape {embeddings.shape[1:]} does not match logical shape {self.logical_embedding_shape}"
            )
        if tuple(masks.shape[1:]) != self.logical_mask_shape:
            raise ValueError(
                f"KV masks shape {masks.shape[1:]} does not match logical mask shape {self.logical_mask_shape}"
            )

        batch_size = int(embeddings.shape[0])
        stored_tokens = int(self.embedding_shape[2])
        compact_embeddings = np.zeros((batch_size, *self.embedding_shape), dtype=ml_dtypes.bfloat16)
        compact_masks = np.zeros((batch_size, *self.mask_shape), dtype=np.bool_)
        for row in range(batch_size):
            valid_positions = np.flatnonzero(masks[row])
            count = int(valid_positions.shape[0])
            if count != int(offsets[row]):
                raise ValueError(f"KV mask true count {count} != offset {int(offsets[row])} for batch row {row}")
            if count > stored_tokens:
                raise ValueError(
                    f"KV offset {count} exceeds stored prefix tokens {stored_tokens}; rebuild the KV cache with a larger storage shape"
                )
            compact_embeddings[row, :, :, :count, :, :] = np.take(embeddings[row], valid_positions, axis=2)
            compact_masks[row, :count] = True
        return compact_embeddings, compact_masks, offsets

    def put_batch(self, ids, embeddings, masks, offsets) -> None:
        ids = self._validated_flat_ids(ids)
        embeddings, masks, offsets = self._compact_for_storage(embeddings, masks, offsets)
        if embeddings.shape[0] != ids.size or masks.shape[0] != ids.size or offsets.shape[0] != ids.size:
            raise ValueError(
                f"Context batch rows must match ids={ids.size}, got "
                f"embeddings={embeddings.shape[0]}, masks={masks.shape[0]}, offsets={offsets.shape[0]}"
            )
        self.put_batch_bits(ids, bf16_to_uint16_bits(embeddings), masks, offsets)

    def put_batch_bits(self, ids, embedding_bits, masks, offsets) -> None:
        """Write already-encoded BF16 uint16 bit patterns without numeric conversion."""
        ids = self._validated_flat_ids(ids)
        embedding_bits = np.asarray(embedding_bits, dtype=np.uint16)
        masks = np.asarray(masks, dtype=np.bool_)
        offsets = np.asarray(offsets, dtype=np.int32).reshape(-1)
        if tuple(embedding_bits.shape[1:]) != self.embedding_shape:
            raise ValueError(f"KV bit storage shape {embedding_bits.shape[1:]} does not match {self.embedding_shape}")
        if tuple(masks.shape[1:]) != self.mask_shape:
            raise ValueError(f"KV bit mask shape {masks.shape[1:]} does not match {self.mask_shape}")
        if embedding_bits.shape[0] != ids.size or masks.shape[0] != ids.size or offsets.shape[0] != ids.size:
            raise ValueError(
                f"Context bit batch rows must match ids={ids.size}, got "
                f"embeddings={embedding_bits.shape[0]}, masks={masks.shape[0]}, offsets={offsets.shape[0]}"
            )
        if self.shard_capacity is None:
            index = self._index_for_ids(ids)
            self.kv_bits[index] = embedding_bits
            self.masks[index] = masks
            self.offsets[index] = offsets
            return
        shard_ids = ids // int(self.shard_capacity)
        for shard_index in np.unique(shard_ids):
            rows = np.flatnonzero(shard_ids == shard_index)
            local_ids = ids[rows] - int(shard_index) * int(self.shard_capacity)
            shard_bits, shard_masks, shard_offsets = self._open_shard(int(shard_index), create=True)
            shard_bits[local_ids] = embedding_bits[rows]
            shard_masks[local_ids] = masks[rows]
            shard_offsets[local_ids] = offsets[rows]

    def get_batch_bits_into(
        self,
        ids,
        embeddings_out,
        masks_out,
        offsets_out,
        *,
        output_rows=None,
    ):
        """Copy selected BF16-bit KV rows directly into caller-owned buffers.

        Unlike ``get_batch_bits(..., pad_to_*)``, this path never materializes a
        batch-sized source array or a second padded array. It copies each mmap
        row directly into its final destination and zero-pads only that row when
        the on-disk token dimension is compact.
        """
        flat_ids = self._validated_flat_ids(ids)
        embeddings_out = np.asarray(embeddings_out)
        masks_out = np.asarray(masks_out)
        offsets_out = np.asarray(offsets_out)
        if embeddings_out.dtype != np.dtype(np.uint16):
            raise TypeError(f"embeddings_out must have dtype uint16, got {embeddings_out.dtype}")
        if masks_out.dtype != np.dtype(np.bool_):
            raise TypeError(f"masks_out must have dtype bool, got {masks_out.dtype}")
        if offsets_out.dtype != np.dtype(np.int32):
            raise TypeError(f"offsets_out must have dtype int32, got {offsets_out.dtype}")
        if embeddings_out.ndim != 1 + len(self.logical_embedding_shape):
            raise ValueError(f"embeddings_out has invalid shape {embeddings_out.shape}")
        if masks_out.ndim != 1 + len(self.logical_mask_shape):
            raise ValueError(f"masks_out has invalid shape {masks_out.shape}")
        if offsets_out.ndim != 1:
            raise ValueError(f"offsets_out has invalid shape {offsets_out.shape}")
        target_embedding_shape = tuple(int(x) for x in embeddings_out.shape[1:])
        target_mask_shape = tuple(int(x) for x in masks_out.shape[1:])
        _validate_storage_shapes(
            target_embedding_shape,
            target_mask_shape,
            self.embedding_shape,
            self.mask_shape,
        )
        if output_rows is None:
            output_rows = np.arange(flat_ids.size, dtype=np.int64)
        else:
            output_rows = np.asarray(output_rows, dtype=np.int64).reshape(-1)
        if output_rows.size != flat_ids.size:
            raise ValueError(f"output_rows length {output_rows.size} does not match ids length {flat_ids.size}")
        if output_rows.size and (
            int(output_rows.min()) < 0
            or int(output_rows.max()) >= int(embeddings_out.shape[0])
            or int(output_rows.max()) >= int(masks_out.shape[0])
            or int(output_rows.max()) >= int(offsets_out.shape[0])
        ):
            raise IndexError(f"output_rows are outside destination buffers: {output_rows}")

        stored_tokens = int(self.embedding_shape[2])
        needs_padding = self.embedding_shape != target_embedding_shape
        for source_id, destination_row in zip(flat_ids.tolist(), output_rows.tolist(), strict=True):
            if self.shard_capacity is None:
                source_bits = self.kv_bits[source_id]
                source_masks = self.masks[source_id]
                source_offset = self.offsets[source_id]
            else:
                shard_index = int(source_id) // int(self.shard_capacity)
                local_id = int(source_id) - shard_index * int(self.shard_capacity)
                shard_bits, shard_masks, shard_offsets = self._open_shard(shard_index, create=False)
                source_bits = shard_bits[local_id]
                source_masks = shard_masks[local_id]
                source_offset = shard_offsets[local_id]
            if needs_padding:
                embeddings_out[destination_row].fill(0)
                masks_out[destination_row].fill(False)
                embeddings_out[destination_row, :, :, :stored_tokens, :, :] = source_bits
                masks_out[destination_row, :stored_tokens] = source_masks
            else:
                embeddings_out[destination_row] = source_bits
                masks_out[destination_row] = source_masks
            offsets_out[destination_row] = source_offset
        return embeddings_out, masks_out, offsets_out

    def get_batch(self, ids, *, pad_to_embedding_shape=None, pad_to_mask_shape=None):
        embedding_bits, masks, offsets = self.get_batch_bits(ids)
        embeddings = uint16_bits_to_bf16(embedding_bits)
        if pad_to_embedding_shape is not None or pad_to_mask_shape is not None:
            target_embedding_shape = tuple(int(x) for x in (pad_to_embedding_shape or self.logical_embedding_shape))
            target_mask_shape = tuple(int(x) for x in (pad_to_mask_shape or self.logical_mask_shape))
            embeddings, masks = self._pad_batch(embeddings, masks, target_embedding_shape, target_mask_shape)
        return embeddings, masks, offsets

    def get_batch_bits(self, ids, *, pad_to_embedding_shape=None, pad_to_mask_shape=None):
        ids = np.asarray(ids, dtype=np.int64)
        if self.shard_capacity is None:
            self._validated_flat_ids(ids)
            embeddings = np.asarray(self.kv_bits[ids], dtype=np.uint16)
            masks = np.asarray(self.masks[ids], dtype=np.bool_)
            offsets = np.asarray(self.offsets[ids], dtype=np.int32)
        else:
            embeddings, masks, offsets = self._get_sharded_batch_bits(ids)
        if pad_to_embedding_shape is not None or pad_to_mask_shape is not None:
            target_embedding_shape = tuple(int(x) for x in (pad_to_embedding_shape or self.logical_embedding_shape))
            target_mask_shape = tuple(int(x) for x in (pad_to_mask_shape or self.logical_mask_shape))
            embeddings, masks = self._pad_batch(embeddings, masks, target_embedding_shape, target_mask_shape)
        return embeddings, masks, offsets

    def _pad_batch(self, embeddings, masks, target_embedding_shape, target_mask_shape):
        if tuple(embeddings.shape[1:]) == target_embedding_shape and tuple(masks.shape[1:]) == target_mask_shape:
            return embeddings, masks
        _validate_storage_shapes(target_embedding_shape, target_mask_shape, self.embedding_shape, self.mask_shape)
        if self.embedding_shape[2] > target_embedding_shape[2]:
            raise ValueError(
                f"Cannot pad storage tokens {self.embedding_shape[2]} into smaller target {target_embedding_shape[2]}"
            )
        padded_embeddings = np.zeros((embeddings.shape[0], *target_embedding_shape), dtype=embeddings.dtype)
        padded_masks = np.zeros((masks.shape[0], *target_mask_shape), dtype=np.bool_)
        stored_tokens = int(self.embedding_shape[2])
        padded_embeddings[:, :, :, :stored_tokens, :, :] = embeddings
        padded_masks[:, :stored_tokens] = masks
        return padded_embeddings, padded_masks

    def flush(self) -> None:
        if self.shard_capacity is None:
            arrays = (self.kv_bits, self.masks, self.offsets)
        else:
            arrays = (array for shard in self._shards.values() for array in shard)
        for array in arrays:
            array.flush()
