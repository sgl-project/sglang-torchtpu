# SPDX-License-Identifier: Apache-2.0
"""Exact TPU TopK primitives for vocab-parallel token sampling.

Sharded reduction communicates only local TopK logits and global token IDs
either through an exact native Pallas direct-HBM ring reduction (TP8/TP16) or
standard PyTorch collective gathers.
"""
import functools
import math
import os
import threading
from collections.abc import Callable
from typing import NamedTuple

import torch

_TPU_TOP_K_OP: Callable | None = None
_TPU_SHARDED_TOP_K_OP: Callable | None = None
_PALLAS_WORLD_SIZES = frozenset((8, 16))

# Pallas Direct-HBM TopK Constants
MAX_DISTRIBUTED_TOP_K = 64
SUPPORTED_WORLD_SIZES = frozenset((8, 16))

_DMA_WIDTH = 128
_LANES = 128
_SUBLANES = 8
_TOKEN_BLOCK = 16
_VOCAB_ALIGNMENT = 128

_OPS: dict[int, Callable] = {}
_OPS_LOCK = threading.Lock()


def supports_distributed_tpu_top_k(lm_head, top_k: int) -> bool:
    """Return whether an LM head has a usable vocab-sharded TPU layout.
    Replicated or incomplete heads must retain the full-logit sampling path."""

    required_layout = (
        "num_embeddings",
        "num_embeddings_per_partition",
        "shard_indices",
    )
    return (
        getattr(lm_head, "enable_tp", False)
        and all(hasattr(lm_head, attr) for attr in required_layout)
        and 0 < top_k <= lm_head.num_embeddings
        and top_k <= lm_head.num_embeddings_per_partition
    )


class TpuVocabShard(NamedTuple):
    """Describe valid base and added tokens inside one padded LM-head shard.
    Device kernels use this metadata to mask padding and reconstruct global IDs.

    A NamedTuple rather than the usual msgspec.Struct: it is constructed
    inside the torch.compile'd forward (fullgraph, so no graph break is
    possible there), and Dynamo cannot instantiate msgspec's C-level
    ``__new__`` in-graph while NamedTuples are traced natively."""

    width: int
    global_vocab_size: int
    org_vocab_start: int
    org_vocab_count: int
    org_vocab_padded_count: int
    added_vocab_start: int
    added_vocab_count: int
    org_vocab_size: int = 0

    @classmethod
    def from_lm_head(cls, lm_head) -> "TpuVocabShard":
        """Build validated host metadata from an LM head's partition indices.
        Invalid padded layouts fail before they can produce device token IDs."""

        if not getattr(lm_head, "enable_tp", False):
            raise ValueError("distributed TPU TopK requires a vocab-sharded LM head")
        indices = lm_head.shard_indices
        org_vocab_size = getattr(lm_head, "org_vocab_size", None)
        if org_vocab_size is None or org_vocab_size <= 0:
            if hasattr(indices, "added_vocab_start_index") and hasattr(
                indices, "num_org_elements_padded"
            ):
                rank = getattr(lm_head, "rank", None)
                if rank is None:
                    import torch.distributed as dist

                    if dist.is_initialized():
                        rank = dist.get_rank()
                    else:
                        rank = int(os.environ.get("RANK", "0"))
                added_width = (
                    lm_head.num_embeddings_per_partition
                    - indices.num_org_elements_padded
                )
                org_vocab_size = indices.added_vocab_start_index - rank * added_width
            else:
                org_vocab_size = lm_head.num_embeddings

        shard = cls(
            width=lm_head.num_embeddings_per_partition,
            global_vocab_size=lm_head.num_embeddings,
            org_vocab_start=indices.org_vocab_start_index,
            org_vocab_count=indices.num_org_elements,
            org_vocab_padded_count=indices.num_org_elements_padded,
            added_vocab_start=indices.added_vocab_start_index,
            added_vocab_count=indices.num_added_elements,
            org_vocab_size=org_vocab_size,
        )
        if shard.org_vocab_padded_count > shard.width:
            raise ValueError(
                "base-vocabulary padded width exceeds LM-head shard width: "
                f"{shard.org_vocab_padded_count} > {shard.width}"
            )
        if shard.org_vocab_padded_count + shard.added_vocab_count > shard.width:
            raise ValueError("added vocabulary exceeds LM-head shard width")
        if not 0 < shard.get_org_vocab_size() <= shard.global_vocab_size:
            raise ValueError("original vocabulary must fit the global vocabulary")
        return shard

    def get_org_vocab_size(self) -> int:
        if self.org_vocab_size > 0:
            return self.org_vocab_size
        if self.added_vocab_count > 0 and self.org_vocab_padded_count > 0:
            rank = int(os.environ.get("RANK", "0"))
            added_width = self.width - self.org_vocab_padded_count
            return self.added_vocab_start - rank * added_width
        return self.global_vocab_size


def _validate_tpu_sharded_top_k(
    values: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
) -> None:
    if values.device.type != "tpu":
        raise ValueError(
            f"tpu_sharded_top_k requires a TPU tensor, got {values.device}"
        )
    if values.ndim != 2:
        raise ValueError(
            f"tpu_sharded_top_k expects a 2D tensor, got shape {values.shape}"
        )
    if values.shape[-1] != shard.width:
        raise ValueError(
            "vocab shard width does not match local logits: "
            f"{shard.width} != {values.shape[-1]}"
        )
    if not 0 < k <= shard.width or k > shard.global_vocab_size:
        raise ValueError(
            "k must fit both the local padded shard and global vocabulary, "
            f"got k={k}, shard_width={shard.width}, "
            f"global_vocab_size={shard.global_vocab_size}"
        )


def _is_canonical_pallas_shard(shard: TpuVocabShard, rank: int) -> bool:
    """Whether rank-based Pallas ID reconstruction matches this shard."""

    org_vocab_size = shard.get_org_vocab_size()
    org_start = min(rank * shard.org_vocab_padded_count, org_vocab_size)
    org_end = min(
        (rank + 1) * shard.org_vocab_padded_count,
        org_vocab_size,
    )
    added_width = shard.width - shard.org_vocab_padded_count
    added_start = min(
        org_vocab_size + rank * added_width,
        shard.global_vocab_size,
    )
    added_end = min(
        org_vocab_size + (rank + 1) * added_width,
        shard.global_vocab_size,
    )
    return (
        added_width >= 0
        and shard.org_vocab_start == org_start
        and shard.org_vocab_count == org_end - org_start
        and (shard.added_vocab_count == 0 or shard.added_vocab_start == added_start)
        and shard.added_vocab_count == added_end - added_start
    )


def supports_pallas_distributed_tpu_top_k(
    shard: TpuVocabShard,
    k: int,
    *,
    group_world_size: int,
    model_rank: int | None,
) -> bool:
    """Return whether static metadata guarantees the direct-HBM path.

    Multi-host callers use this before skipping the full-vocabulary gather.
    That prevents a later Pallas miss from reaching the native collective,
    which is not supported across TPU v7 hosts.
    """

    return (
        supports_pallas_distributed_tpu_top_k_topology(group_world_size)
        and isinstance(model_rank, int)
        and 0 <= model_rank < group_world_size
        and shard.width >= 64
        and 0 < k <= 64
        and _is_canonical_pallas_shard(shard, model_rank)
    )


def supports_pallas_distributed_tpu_top_k_topology(group_world_size: int) -> bool:
    """Return whether a process group can use the full-world Pallas ring."""
    from sglang.srt.environ import envs

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    disable_pallas = getattr(
        envs, "SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K", None
    )
    is_disabled = disable_pallas.get() if disable_pallas is not None else False
    return (
        not is_disabled
        and group_world_size == world_size
        and world_size in _PALLAS_WORLD_SIZES
    )


def _get_tpu_top_k_op() -> Callable:
    """Create the exported native-XLA TopK custom op once per process.
    Registration remains lazy so non-TPU sampling does not import JAX."""

    global _TPU_TOP_K_OP
    if _TPU_TOP_K_OP is not None:
        return _TPU_TOP_K_OP

    import jax

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(values: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
        # torch_tpu's aten::topk lowering is currently a full StableHLO
        # sort plus slice. lax.top_k exports XLA's native exact TopK op.
        return jax.lax.top_k(values, k)

    op = cpu_export_jax_op("sglang_tpu_pallas::exact_top_k", _wrapped)

    def _fake(values: torch.Tensor, k: int):
        shape = (*values.shape[:-1], k)
        return (
            values.new_empty(shape),
            values.new_empty(shape, dtype=torch.int32),
        )

    op.register_fake(_fake)
    _TPU_TOP_K_OP = op
    return op


def tpu_top_k(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exact descending TopK values from a two-dimensional TPU tensor.
    Indices are int32 positions in the input's final dimension."""

    if values.device.type != "tpu":
        raise ValueError(f"tpu_top_k requires a TPU tensor, got {values.device}")
    if values.ndim != 2:
        raise ValueError(f"tpu_top_k expects a 2D tensor, got shape {values.shape}")
    if not 0 < k <= values.shape[-1]:
        raise ValueError(f"k must be in [1, {values.shape[-1]}], got {k}")
    return _get_tpu_top_k_op()(values, k)


def _get_tpu_sharded_top_k_op() -> Callable:
    """Create the fused padding-mask, local TopK, and ID-mapping custom op.
    Fusion avoids materializing a vocabulary-sized global-ID tensor."""

    global _TPU_SHARDED_TOP_K_OP
    if _TPU_SHARDED_TOP_K_OP is not None:
        return _TPU_SHARDED_TOP_K_OP

    import jax
    import jax.numpy as jnp

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(
        values: jax.Array,
        k: int,
        org_vocab_start: int,
        org_vocab_count: int,
        org_vocab_padded_count: int,
        added_vocab_start: int,
        added_vocab_count: int,
    ) -> tuple[jax.Array, jax.Array]:
        positions = jnp.arange(values.shape[-1], dtype=jnp.int32)
        added_positions = positions - org_vocab_padded_count
        valid = (positions < org_vocab_count) | (
            (added_positions >= 0) & (added_positions < added_vocab_count)
        )
        masked_values = jnp.where(valid, values, -jnp.inf)
        top_values, top_positions = jax.lax.top_k(masked_values, k)
        token_ids = jnp.where(
            top_positions < org_vocab_count,
            org_vocab_start + top_positions,
            added_vocab_start + top_positions - org_vocab_padded_count,
        )
        return top_values, token_ids.astype(jnp.int32)

    op = cpu_export_jax_op("sglang_tpu_pallas::exact_sharded_top_k", _wrapped)

    def _fake(
        values: torch.Tensor,
        k: int,
        org_vocab_start: int,
        org_vocab_count: int,
        org_vocab_padded_count: int,
        added_vocab_start: int,
        added_vocab_count: int,
    ):
        del (
            org_vocab_start,
            org_vocab_count,
            org_vocab_padded_count,
            added_vocab_start,
            added_vocab_count,
        )
        shape = (*values.shape[:-1], k)
        return (
            values.new_empty(shape),
            values.new_empty(shape, dtype=torch.int32),
        )

    op.register_fake(_fake)
    _TPU_SHARDED_TOP_K_OP = op
    return op


def tpu_sharded_top_k(
    values: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one shard's exact TopK while excluding padded columns.
    Returned int32 IDs already refer to the unpadded global vocabulary."""

    _validate_tpu_sharded_top_k(values, shard, k)
    return _get_tpu_sharded_top_k_op()(
        values,
        k,
        shard.org_vocab_start,
        shard.org_vocab_count,
        shard.org_vocab_padded_count,
        shard.added_vocab_start,
        shard.added_vocab_count,
    )


# ---------------------------------------------------------------------------
# Bitonic Top64 and Compressed-Transpose Helper Primitives
# ---------------------------------------------------------------------------


def _ceil_div(dividend: int, divisor: int) -> int:
    return -(-dividend // divisor)


def _ceil_multiple(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


def _log2_ceil(value: int) -> int:
    return 0 if value == 0 else math.ceil(math.log2(value))


def _pad_min(array, target_shape):
    import jax.numpy as jnp

    padding = tuple(
        (0, target_shape[axis] - array.shape[axis]) for axis in range(array.ndim)
    )
    if all(width == (0, 0) for width in padding):
        return array
    minimum = (
        -jnp.inf
        if jnp.issubdtype(array.dtype, jnp.floating)
        else jnp.iinfo(array.dtype).min
    )
    return jnp.pad(array, padding, constant_values=minimum)


def _iota_tile(axis: int, shape: tuple[int, int] = (_SUBLANES, _LANES)):
    import jax.numpy as jnp

    if axis == 0:
        values = jnp.arange(shape[0], dtype=jnp.int32).reshape(shape[0], 1)
    else:
        values = jnp.arange(shape[1], dtype=jnp.int32).reshape(1, shape[1])
    return jnp.broadcast_to(values, shape)


def _split_tiles(array, shape: tuple[int, int] = (_SUBLANES, _LANES)):
    import jax.numpy as jnp

    tile_rows, tile_columns = shape
    row_groups = array.shape[0] // tile_rows
    column_groups = array.shape[1] // tile_columns
    assert array.shape[0] % tile_rows == 0
    assert array.shape[1] % tile_columns == 0
    tiles = []
    for row_strip in jnp.split(array, row_groups, axis=0):
        tiles.extend(jnp.split(row_strip, column_groups, axis=1))
    return tiles


def _to_compressed_transpose(array):
    import jax.numpy as jnp

    rows, columns = array.shape
    assert rows <= _LANES and _LANES % rows == 0
    assert columns >= _LANES
    row_groups = jnp.split(array, _LANES // rows, axis=1)
    return jnp.concatenate(row_groups, axis=0).T


def _from_compressed_transpose(tiles, rows: int):
    import jax.numpy as jnp

    array = jnp.concatenate(tiles, axis=0)
    original_columns = array.shape[0]
    if original_columns < _LANES:
        array = jnp.pad(array, ((0, _LANES - original_columns), (0, 0)))
    array = array.T
    row_groups = jnp.split(array, array.shape[0] // rows, axis=0)
    array = jnp.concatenate(row_groups, axis=1)
    return array[:, :original_columns]


def _resplit(tiles, target_rows: int):
    import jax.numpy as jnp

    tiles = list(tiles)
    rows = tiles[0].shape[0]
    if rows == target_rows:
        return tiles
    if rows > target_rows:
        result = []
        for tile in tiles:
            result.extend(jnp.split(tile, rows // target_rows, axis=0))
        return result
    group_count = target_rows // rows
    return [
        jnp.concatenate(tiles[index : index + group_count], axis=0)
        for index in range(0, len(tiles), group_count)
    ]


def _compare_and_swap(
    left_values,
    left_indices,
    right_values,
    right_indices,
    descending: bool = True,
    right_half=None,
):
    import jax.numpy as jnp

    if isinstance(descending, bool):
        condition = (
            left_values < right_values if descending else left_values > right_values
        )
    else:
        condition = (descending & (left_values < right_values)) | (
            (~descending) & (left_values > right_values)
        )

    if right_half is None:
        new_left_values = jnp.where(condition, right_values, left_values)
        new_left_indices = jnp.where(condition, right_indices, left_indices)
        new_right_values = jnp.where(condition, left_values, right_values)
        new_right_indices = jnp.where(condition, left_indices, right_indices)
        return (
            new_left_values,
            new_left_indices,
            new_right_values,
            new_right_indices,
        )

    condition = jnp.bitwise_xor(condition, right_half)
    output_values = jnp.where(condition, right_values, left_values)
    output_indices = jnp.where(condition, right_indices, left_indices)
    return output_values, output_indices


def _pair_start(index: int, separation: int) -> int:
    return (index % separation) + 2 * separation * (index // separation)


def _bit_indicator(bit: int, values):
    return (((values >> bit) & 1) != 0)


def _is_descending(
    stage: int,
    tile_start: int,
    tile_offset,
    sort_offset: int = 0,
    full_size: int = 0,
):
    if stage == 0:
        return True
    if stage <= _log2_ceil(_SUBLANES):
        return _bit_indicator(stage, tile_offset + sort_offset)
    if stage < _log2_ceil(full_size):
        return _bit_indicator(stage, tile_start + sort_offset)
    return _bit_indicator(stage, tile_start + tile_offset + sort_offset)


def _bitonic_substage(
    value_tiles,
    index_tiles,
    *,
    substage: int,
    batch_size: int,
    stage: int | None = None,
    sort_offset: int = 0,
    full_size: int | None = None,
    max_reduce: bool = False,
):
    import jax.numpy as jnp

    assert max_reduce or stage is not None
    separation = 1 << substage
    if full_size is None:
        full_size = len(value_tiles) * value_tiles[0].shape[0]

    value_tiles = _resplit(value_tiles, _SUBLANES)
    index_tiles = _resplit(index_tiles, _SUBLANES)
    if separation < _SUBLANES or separation >= full_size:
        axis = int(separation >= full_size)
        intra_tile_separation = (
            separation if axis == 0 else separation * batch_size // full_size
        )
        tile_offset = _iota_tile(0) + _iota_tile(1) // batch_size * full_size
        right_half = _bit_indicator(_log2_ceil(intra_tile_separation), _iota_tile(axis))
        permutation = jnp.bitwise_xor(_iota_tile(axis), intra_tile_separation)
        output_values = [None] * len(value_tiles)
        output_indices = [None] * len(index_tiles)
        for tile_index, (values, indices) in enumerate(
            zip(value_tiles, index_tiles, strict=True)
        ):
            permuted_values = jnp.take_along_axis(values, permutation, axis=axis)
            permuted_indices = jnp.take_along_axis(indices, permutation, axis=axis)
            descending = (
                True
                if max_reduce
                else _is_descending(
                    stage,
                    tile_index * _SUBLANES,
                    tile_offset,
                    sort_offset,
                    full_size,
                )
            )
            output_values[tile_index], output_indices[tile_index] = _compare_and_swap(
                values,
                indices,
                permuted_values,
                permuted_indices,
                descending=descending,
                right_half=right_half,
            )
    else:
        tile_shape = value_tiles[0].shape
        tile_separation = separation // tile_shape[0]
        tile_offset = (
            _iota_tile(0, tile_shape)
            + _iota_tile(1, tile_shape) // batch_size * full_size
        )
        output_values = [None] * len(value_tiles)
        output_indices = [None] * len(index_tiles)
        for pair_index in range(len(value_tiles) // 2):
            left = _pair_start(pair_index, tile_separation)
            right = left + tile_separation
            descending = (
                True
                if max_reduce
                else _is_descending(
                    stage,
                    left * tile_shape[0],
                    tile_offset,
                    sort_offset,
                    full_size,
                )
            )
            swapped = _compare_and_swap(
                value_tiles[left],
                index_tiles[left],
                value_tiles[right],
                index_tiles[right],
                descending=descending,
            )
            output_values[left], output_indices[left] = swapped[:2]
            if not max_reduce:
                output_values[right], output_indices[right] = swapped[2:]

    if max_reduce:
        output_values = [value for value in output_values if value is not None]
        output_indices = [index for index in output_indices if index is not None]
    assert all(value is not None for value in output_values)
    return output_values, output_indices


def _topk_padded_shape(rows: int, columns: int) -> tuple[int, int]:
    if rows >= _LANES:
        return _ceil_multiple(rows, _LANES), _ceil_multiple(
            columns, MAX_DISTRIBUTED_TOP_K
        )
    candidate_rows = (
        1 << exponent
        for exponent in range(_log2_ceil(_SUBLANES), _log2_ceil(_LANES) + 1)
        if 1 << exponent >= rows
    )
    shapes = [
        (
            padded_rows,
            _ceil_multiple(
                _ceil_multiple(columns, _LANES * _LANES // padded_rows),
                MAX_DISTRIBUTED_TOP_K * _LANES // padded_rows,
            ),
        )
        for padded_rows in candidate_rows
    ]
    return min(shapes, key=lambda shape: (shape[0] * shape[1], -shape[0]))


def _bitonic_top64(values, indices):
    """Exact descending Top64 using a TPU compressed-transpose bitonic merge."""

    rows, columns = values.shape
    padded_shape = _topk_padded_shape(rows, columns)
    values = _pad_min(values, padded_shape)
    indices = _pad_min(indices, padded_shape)
    batch_size = values.shape[0]
    value_tiles = _split_tiles(_to_compressed_transpose(values))
    index_tiles = _split_tiles(_to_compressed_transpose(indices))
    tile_count = len(value_tiles)

    merge_count = _log2_ceil(columns) - _log2_ceil(MAX_DISTRIBUTED_TOP_K)
    sublane_merges = _log2_ceil(_ceil_div(_SUBLANES, MAX_DISTRIBUTED_TOP_K))
    lane_merges = _log2_ceil(_ceil_div(columns, tile_count * _SUBLANES))
    tile_merges = merge_count - sublane_merges - lane_merges

    def substage(value_tiles, index_tiles, **kwargs):
        return _bitonic_substage(
            value_tiles, index_tiles, batch_size=batch_size, **kwargs
        )

    def reduce_stage(value_tiles, index_tiles, stage):
        for substage_index in reversed(range(_log2_ceil(MAX_DISTRIBUTED_TOP_K))):
            value_tiles, index_tiles = substage(
                value_tiles,
                index_tiles,
                substage=substage_index,
                stage=stage,
            )
        return substage(
            value_tiles,
            index_tiles,
            substage=stage,
            max_reduce=True,
        )

    for stage in range(1, _log2_ceil(MAX_DISTRIBUTED_TOP_K)):
        for substage_index in reversed(range(stage)):
            value_tiles, index_tiles = substage(
                value_tiles,
                index_tiles,
                substage=substage_index,
                stage=stage,
            )

    reduction_stage = _log2_ceil(MAX_DISTRIBUTED_TOP_K)
    for _ in range(tile_merges):
        group_tiles = 2 * _ceil_div(MAX_DISTRIBUTED_TOP_K, _SUBLANES)
        remainder = len(value_tiles) % group_tiles
        remainder_values = value_tiles[-remainder:] if remainder else []
        remainder_indices = index_tiles[-remainder:] if remainder else []
        if remainder:
            value_tiles = value_tiles[:-remainder]
            index_tiles = index_tiles[:-remainder]
        value_tiles, index_tiles = reduce_stage(
            value_tiles, index_tiles, reduction_stage
        )
        value_tiles.extend(remainder_values)
        index_tiles.extend(remainder_indices)

    for lane_stage in reversed(range(lane_merges)):
        value_tiles, index_tiles = reduce_stage(
            value_tiles, index_tiles, reduction_stage + lane_stage
        )
    for sublane_stage in reversed(range(sublane_merges)):
        value_tiles, index_tiles = reduce_stage(
            value_tiles, index_tiles, reduction_stage + sublane_stage
        )
    for substage_index in reversed(range(reduction_stage)):
        value_tiles, index_tiles = substage(
            value_tiles,
            index_tiles,
            substage=substage_index,
            stage=reduction_stage,
            sort_offset=MAX_DISTRIBUTED_TOP_K,
        )

    top_values = _from_compressed_transpose(value_tiles, batch_size)
    top_indices = _from_compressed_transpose(index_tiles, batch_size)
    return (
        top_values[:rows, :MAX_DISTRIBUTED_TOP_K],
        top_indices[:rows, :MAX_DISTRIBUTED_TOP_K],
    )


# ---------------------------------------------------------------------------
# Pallas Direct-HBM Distributed Top-K Reduction Kernel
# ---------------------------------------------------------------------------


def _distributed_topk_kernel(
    rank_ref,
    tp_rank_to_device_id_ref,
    logits_ref,
    candidate_values_ref,
    candidate_indices_ref,
    send_sems_ref,
    local_sems_ref,
    recv_sem_ref,
    value_buffer,
    index_buffer,
    *,
    world_size: int,
    num_tiles: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
) -> None:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    @jax.named_scope("sync_barrier_signal")
    def sync_barrier_signal():
        barrier_sem = pltpu.get_barrier_semaphore()
        for i in range(world_size):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id={"model": jnp.int32(i)},
                device_id_type=pltpu.DeviceIdType.MESH,
            )

    @jax.named_scope("sync_barrier_wait")
    def sync_barrier_wait():
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_wait(barrier_sem, world_size)

    def sync_barrier():
        sync_barrier_signal()
        sync_barrier_wait()

    # Step 1: Entry Optical Barrier ensuring all cores have initialized Pallas buffers
    @pl.when(pl.program_id(0) == 0)
    def _():
        sync_barrier()

    token_start = pl.program_id(0) * _TOKEN_BLOCK
    vocab_rank = rank_ref[0].astype(jnp.int32)
    aligned_width = logits_ref.shape[-1]
    added_width = local_width - org_vocab_padded_count

    positions = lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1)
    org_count = jnp.clip(
        org_vocab_size - vocab_rank * org_vocab_padded_count,
        0,
        org_vocab_padded_count,
    )
    added_count = jnp.clip(
        global_vocab_size - (org_vocab_size + vocab_rank * added_width),
        0,
        added_width,
    )
    added_positions = positions - org_vocab_padded_count
    valid = (positions < org_count) | (
        (added_positions >= 0) & (added_positions < added_count)
    )
    logits = jnp.where(valid, logits_ref[...], -jnp.inf)
    local_indices = positions
    local_values, local_indices = _bitonic_top64(
        logits.astype(jnp.float32), local_indices
    )
    local_values = jnp.pad(
        local_values.astype(logits_ref.dtype),
        ((0, 0), (0, _DMA_WIDTH - MAX_DISTRIBUTED_TOP_K)),
        constant_values=-jnp.inf,
    )
    local_indices = jnp.pad(
        local_indices + vocab_rank * aligned_width,
        ((0, 0), (0, _DMA_WIDTH - MAX_DISTRIBUTED_TOP_K)),
    )

    value_buffer[...] = local_values
    index_buffer[...] = local_indices

    # Direct All-to-All Distribution from local VMEM to peer HBM buffers (exact fused_moe_rs pattern)
    for dest_rank in range(world_size):
        is_local = dest_rank == vocab_rank

        @pl.when(is_local)
        def _():
            pltpu.make_async_copy(
                src_ref=value_buffer,
                dst_ref=candidate_values_ref.at[
                    vocab_rank, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                sem=local_sems_ref.at[0],
            ).start()
            pltpu.make_async_copy(
                src_ref=index_buffer,
                dst_ref=candidate_indices_ref.at[
                    vocab_rank, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                sem=local_sems_ref.at[1],
            ).start()

        @pl.when(~is_local)
        def _():
            target_device_id = tp_rank_to_device_id_ref[dest_rank]
            pltpu.make_async_remote_copy(
                src_ref=value_buffer,
                dst_ref=candidate_values_ref.at[
                    vocab_rank, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                send_sem=send_sems_ref.at[2 * dest_rank],
                recv_sem=recv_sem_ref.at[0],
                device_id={"model": target_device_id},
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            pltpu.make_async_remote_copy(
                src_ref=index_buffer,
                dst_ref=candidate_indices_ref.at[
                    vocab_rank, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                send_sem=send_sems_ref.at[2 * dest_rank + 1],
                recv_sem=recv_sem_ref.at[0],
                device_id={"model": target_device_id},
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()

    # Consuming wait on count=1 for local and outgoing remote DMA sends to finish before reusing VMEM buffers
    pl.semaphore_wait(local_sems_ref.at[0], 1)
    pl.semaphore_wait(local_sems_ref.at[1], 1)
    for dest_rank in range(world_size):
        is_local = dest_rank == vocab_rank

        @pl.when(~is_local)
        def _():
            pl.semaphore_wait(send_sems_ref.at[2 * dest_rank], 1)
            pl.semaphore_wait(send_sems_ref.at[2 * dest_rank + 1], 1)

    # Step 3: Exit Optical Barrier & Recv Drain (exact fused_moe_rs pattern)
    @pl.when(pl.program_id(0) == (num_tiles - 1))
    def _():
        sync_barrier()
        total_recv = 2 * (world_size - 1) * num_tiles
        @pl.when(total_recv > 0)
        def _():
            pl.semaphore_wait(recv_sem_ref.at[0], total_recv)


def _ring_candidates(
    local_logits,
    rank_input,
    tp_rank_to_device_id,
    *,
    world_size: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, local_width = local_logits.shape
    padded_tokens = _ceil_multiple(tokens, _TOKEN_BLOCK)
    num_tiles = padded_tokens // _TOKEN_BLOCK
    aligned_width = _ceil_multiple(local_width, _VOCAB_ALIGNMENT)
    local_logits = jnp.pad(
        local_logits,
        ((0, padded_tokens - tokens), (0, aligned_width - local_width)),
        constant_values=-jnp.inf,
    )
    kernel = functools.partial(
        _distributed_topk_kernel,
        world_size=world_size,
        num_tiles=num_tiles,
        local_width=local_width,
        org_vocab_size=org_vocab_size,
        global_vocab_size=global_vocab_size,
        org_vocab_padded_count=org_vocab_padded_count,
    )
    candidate_shape = (world_size, padded_tokens, _DMA_WIDTH)
    output_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    )
    scratch_shapes = (
        pltpu.SemaphoreType.REGULAR((2 * world_size,)),  # send_sems_ref
        pltpu.SemaphoreType.REGULAR((2,)),               # local_sems_ref
        pltpu.SemaphoreType.REGULAR((1,)),               # recv_sem_ref
        pltpu.VMEM((_TOKEN_BLOCK, _DMA_WIDTH), local_logits.dtype),
        pltpu.VMEM((_TOKEN_BLOCK, _DMA_WIDTH), jnp.int32),
    )
    compiler_params = pltpu.CompilerParams(
        disable_bounds_checks=True,
        collective_id=0,
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(candidate_shape, local_logits.dtype),
            jax.ShapeDtypeStruct(candidate_shape, jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(padded_tokens // _TOKEN_BLOCK,),
            in_specs=(
                pl.BlockSpec(
                    (_TOKEN_BLOCK, aligned_width),
                    lambda tile, *_: (tile, 0),
                    memory_space=pltpu.VMEM,
                ),
            ),
            out_specs=output_specs,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=compiler_params,
        name=f"distributed_topk_tp{world_size}",
    )(rank_input, tp_rank_to_device_id, local_logits)


@functools.cache
def _get_mesh(world_size: int):
    import numpy as np
    from jax.sharding import Mesh

    from sglang.srt.hardware_backend.tpu.pallas_export import (
        get_torchtpu_export_devices,
    )

    return Mesh(np.asarray(get_torchtpu_export_devices(world_size)), ("model",))


def _build_pallas_op(world_size: int) -> Callable:
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh(world_size)

    def _local(
        local_logits: jax.Array,
        rank_input: jax.Array,
        tp_rank_to_device_id: jax.Array,
        org_vocab_size: int,
        global_vocab_size: int,
        org_vocab_padded_count: int,
        k: int,
    ) -> tuple[jax.Array, jax.Array]:
        tokens, local_width = local_logits.shape
        aligned_width = _ceil_multiple(local_width, _VOCAB_ALIGNMENT)
        candidate_values, candidate_indices = _ring_candidates(
            local_logits,
            rank_input,
            tp_rank_to_device_id,
            world_size=world_size,
            org_vocab_size=org_vocab_size,
            global_vocab_size=global_vocab_size,
            org_vocab_padded_count=org_vocab_padded_count,
        )
        candidate_values = jnp.transpose(
            candidate_values[:, :tokens, :MAX_DISTRIBUTED_TOP_K], (1, 0, 2)
        ).reshape(tokens, world_size * MAX_DISTRIBUTED_TOP_K)
        candidate_indices = jnp.transpose(
            candidate_indices[:, :tokens, :MAX_DISTRIBUTED_TOP_K], (1, 0, 2)
        ).reshape(tokens, world_size * MAX_DISTRIBUTED_TOP_K)

        values, candidate_positions = jax.lax.top_k(candidate_values, k)
        packed_ids = jnp.take_along_axis(
            candidate_indices, candidate_positions, axis=1
        )
        source_rank = packed_ids // aligned_width
        local_position = packed_ids % aligned_width
        added_width = local_width - org_vocab_padded_count
        token_ids = jnp.where(
            local_position < org_vocab_padded_count,
            source_rank * org_vocab_padded_count + local_position,
            org_vocab_size
            + source_rank * added_width
            + local_position
            - org_vocab_padded_count,
        )
        return values, token_ids.astype(jnp.int32)

    sharded = jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(None, "model"), P("model"), P(), None, None, None, None),
        out_specs=(P(), P()),
        check_vma=False,
    )

    def _wrapped(
        local_logits: jax.Array,
        rank_input: jax.Array,
        tp_rank_to_device_id: jax.Array,
        org_vocab_size: int,
        global_vocab_size: int,
        org_vocab_padded_count: int,
        k: int,
    ) -> tuple[jax.Array, jax.Array]:
        return sharded(
            local_logits,
            rank_input,
            tp_rank_to_device_id,
            org_vocab_size,
            global_vocab_size,
            org_vocab_padded_count,
            k,
        )

    op = cpu_export_jax_op(
        f"sglang_tpu_pallas::distributed_top_k_tp{world_size}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(
            P(None, "model"),
            P("model"),
            P(),
            None,
            None,
            None,
            None,
        ),
        output_partition_specs=(P(), P()),
    )

    def _fake(
        local_logits: torch.Tensor,
        rank_input: torch.Tensor,
        tp_rank_to_device_id: torch.Tensor,
        org_vocab_size: int,
        global_vocab_size: int,
        org_vocab_padded_count: int,
        k: int,
    ):
        del (
            rank_input,
            tp_rank_to_device_id,
            org_vocab_size,
            global_vocab_size,
            org_vocab_padded_count,
        )
        shape = (*local_logits.shape[:-1], k)
        return (
            local_logits.new_empty(shape, dtype=local_logits.dtype),
            local_logits.new_empty(shape, dtype=torch.int32),
        )

    op.register_fake(_fake)
    return op


def get_pallas_distributed_top_k_op(world_size: int) -> Callable:
    """Return the lazily registered custom op for one supported topology."""

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Pallas distributed TopK does not support TP{world_size}")
    op = _OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _OPS.get(world_size)
        if op is None:
            op = _build_pallas_op(world_size)
            _OPS[world_size] = op
    return op


def register_pallas_distributed_top_k_op(world_size: int) -> None:
    """Register the current TP8/TP16 topology before full-graph tracing."""

    if world_size in SUPPORTED_WORLD_SIZES:
        get_pallas_distributed_top_k_op(world_size)


def pallas_distributed_top_k(
    local_logits: torch.Tensor,
    *,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    k: int,
    world_size: int,
    model_rank: int,
    tp_rank_to_device_id: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return replicated exact global TopK values and unpadded token IDs."""

    from sglang.srt.hardware_backend.tpu.topology import (
        get_tp_rank_to_device_id_mapping,
    )

    rank_input = local_logits.new_tensor((model_rank,), dtype=torch.int32)
    if tp_rank_to_device_id is None or tp_rank_to_device_id.numel() != world_size:
        _, mapping = get_tp_rank_to_device_id_mapping()
        tp_rank_to_device_id = local_logits.new_tensor(mapping, dtype=torch.int32)

    values, token_ids = get_pallas_distributed_top_k_op(world_size)(
        local_logits,
        rank_input,
        tp_rank_to_device_id,
        org_vocab_size,
        global_vocab_size,
        org_vocab_padded_count,
        k,
    )
    return values.float(), token_ids


def register_tpu_top_k_ops() -> None:
    """Register local and global TopK ops before full-graph tracing begins.
    Eager registration keeps custom-op schemas visible to torch.compile."""

    _get_tpu_top_k_op()
    _get_tpu_sharded_top_k_op()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size in _PALLAS_WORLD_SIZES:
        from sglang.srt.environ import envs

        disable_pallas = getattr(
            envs, "SGLANG_TPU_DISABLE_PALLAS_DISTRIBUTED_TOP_K", None
        )
        is_disabled = disable_pallas.get() if disable_pallas is not None else False
        if not is_disabled:
            register_pallas_distributed_top_k_op(world_size)


def distributed_tpu_top_k_collective(
    local_logits: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
    *,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge shard-local TopK candidates with Torch collectives."""

    sharded_top_k_logits, sharded_top_k_token_ids = tpu_sharded_top_k(
        local_logits, shard, k
    )

    if group.world_size > 1:
        if shard.global_vocab_size <= 2**24:
            # Token IDs are exact in fp32 for every practical serving
            # vocabulary. Packing values and IDs halves collective latency at
            # the cost of a tiny payload increase over bf16 + int32.
            packed_sharded_top_k = torch.stack(
                (sharded_top_k_logits.float(), sharded_top_k_token_ids.float()),
                dim=-1,
            )
            gathered_sharded_top_k = group.all_gather(packed_sharded_top_k, dim=-2)
            gathered_sharded_top_k_logits = gathered_sharded_top_k[..., 0]
            gathered_sharded_top_k_token_ids = gathered_sharded_top_k[..., 1].to(
                torch.int32
            )
        else:
            # Preserve exact IDs for unusually large vocabularies.
            gathered_sharded_top_k_logits = group.all_gather(
                sharded_top_k_logits, dim=-1
            )
            gathered_sharded_top_k_token_ids = group.all_gather(
                sharded_top_k_token_ids, dim=-1
            )
    else:
        gathered_sharded_top_k_logits = sharded_top_k_logits
        gathered_sharded_top_k_token_ids = sharded_top_k_token_ids

    global_top_k_logits, global_top_k_positions = tpu_top_k(
        gathered_sharded_top_k_logits, k
    )
    global_top_k_token_ids = torch.gather(
        gathered_sharded_top_k_token_ids,
        dim=1,
        index=global_top_k_positions.to(torch.int64),
    )
    return global_top_k_logits.float(), global_top_k_token_ids.to(torch.int32)


def distributed_tpu_top_k(
    local_logits: torch.Tensor,
    shard: TpuVocabShard,
    k: int,
    *,
    group,
    require_pallas: bool = False,
    tp_rank_to_device_id: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge exact sharded TopK logits into a replicated global TopK."""
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    model_rank = getattr(group, "rank_in_group", None)
    if model_rank is None and hasattr(group, "rank"):
        model_rank = group.rank
    if model_rank is None and dist.is_initialized():
        model_rank = dist.get_rank()
    if model_rank is None:
        model_rank = int(os.environ.get("RANK", "0"))

    group_world_size = getattr(group, "world_size", None)
    if group_world_size is None and hasattr(group, "size"):
        group_world_size = group.size()
    if group_world_size is None and dist.is_initialized():
        group_world_size = dist.get_world_size()
    if group_world_size is None:
        group_world_size = world_size

    if (
        supports_pallas_distributed_tpu_top_k(
            shard,
            k,
            group_world_size=group_world_size,
            model_rank=model_rank,
        )
        and local_logits.shape[0] > 0
        and local_logits.dtype in (torch.bfloat16, torch.float32)
    ):
        return pallas_distributed_top_k(
            local_logits,
            org_vocab_size=shard.get_org_vocab_size(),
            global_vocab_size=shard.global_vocab_size,
            org_vocab_padded_count=shard.org_vocab_padded_count,
            k=k,
            world_size=world_size,
            model_rank=model_rank,
            tp_rank_to_device_id=tp_rank_to_device_id,
        )

    if require_pallas:
        raise RuntimeError(
            "multi-host distributed TPU TopK lost Pallas eligibility; "
            "select the full-vocabulary gather before computing local logits"
        )

    return distributed_tpu_top_k_collective(
        local_logits,
        shard,
        k,
        group=group,
    )
