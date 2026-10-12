# SPDX-License-Identifier: Apache-2.0
"""Pure JAX/Pallas TopK reduction for TP8 and TP16 TPU meshes.

The Pallas kernel selects each shard's exact Top64 and exchanges only those
candidates through a bidirectional direct-HBM ring. A small JAX TopK merges the
source-rank-ordered candidates and reconstructs unpadded global token IDs.
"""

import functools
import math

MAX_DISTRIBUTED_TOP_K = 64
SUPPORTED_WORLD_SIZES = frozenset((8, 16))

_DMA_WIDTH = 128
_LANES = 128
_SUBLANES = 8
_TOKEN_BLOCK = 32
_VOCAB_ALIGNMENT = 128

# The local reduction below is a handwritten specialization of the
# compressed-transpose bitonic network in PyTorch Helion's Apache-2.0 Pallas
# TopK lowering, itself derived from Tallax:
# https://github.com/pytorch/helion/blob/2be976894bb434724726cd1064c7eee57210faa6/
# helion/_compiler/pallas/topk_impl.py. Keeping the network here avoids a Helion
# runtime dependency and checked-in generated kernels while preserving exact
# Top64.


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
    combine = target_rows // rows
    return [
        jnp.concatenate(tiles[index * combine : (index + 1) * combine], axis=0)
        for index in range(len(tiles) // combine)
    ]


def _bit_indicator(bit: int, index):
    return (index & (1 << bit)) > 0


def _compare_and_swap(
    left_values,
    left_indices,
    right_values,
    right_indices,
    *,
    descending,
    right_half=None,
):
    import jax.numpy as jnp

    if right_half is not None:
        compare_left = jnp.where(right_half, right_values, left_values)
        compare_right = jnp.where(right_half, left_values, right_values)
    else:
        compare_left, compare_right = left_values, right_values

    if isinstance(descending, bool) and descending:
        take_left = compare_left > compare_right
    else:
        take_left = compare_right > compare_left
    if not isinstance(descending, bool):
        take_left = take_left ^ descending.astype(bool)

    if right_half is not None:
        return (
            jnp.where(take_left, left_values, right_values),
            jnp.where(take_left, left_indices, right_indices),
        )
    return (
        jnp.where(take_left, left_values, right_values),
        jnp.where(take_left, left_indices, right_indices),
        jnp.where(take_left, right_values, left_values),
        jnp.where(take_left, right_indices, left_indices),
    )


def _pair_start(index: int, separation: int) -> int:
    return (index // separation) * 2 * separation + index % separation


def _is_descending(
    stage: int,
    tile_start: int,
    tile_offset,
    sort_offset: int,
    full_size: int,
):
    if stage < _log2_ceil(_SUBLANES) or stage >= _log2_ceil(full_size):
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


def _start_and_wait(copies) -> None:
    for copy in copies:
        copy.start()
    for copy in copies:
        copy.wait()


def _distributed_topk_kernel(
    logits_ref,
    candidate_values_ref,
    candidate_indices_ref,
    send_semaphores,
    receive_semaphores,
    value_buffer_0,
    index_buffer_0,
    value_buffer_1,
    index_buffer_1,
    value_buffer_2,
    index_buffer_2,
    *,
    world_size: int,
    local_width: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    token_start = pl.program_id(0) * _TOKEN_BLOCK
    # The native mesh follows logical ranks for both routing and vocabulary shards.
    mesh_rank = lax.axis_index("model").astype(jnp.int32)
    aligned_width = logits_ref.shape[-1]
    added_width = local_width - org_vocab_padded_count

    positions = lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1)
    org_count = jnp.clip(
        org_vocab_size - mesh_rank * org_vocab_padded_count,
        0,
        org_vocab_padded_count,
    )
    added_start = org_vocab_size + mesh_rank * added_width
    added_count = jnp.clip(global_vocab_size - added_start, 0, added_width)
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
        local_indices + mesh_rank * aligned_width,
        ((0, 0), (0, _DMA_WIDTH - MAX_DISTRIBUTED_TOP_K)),
    )

    value_buffers = (value_buffer_0, value_buffer_1, value_buffer_2)
    index_buffers = (index_buffer_0, index_buffer_1, index_buffer_2)
    for value_buffer, index_buffer in zip(value_buffers, index_buffers, strict=True):
        value_buffer[...] = local_values
        index_buffer[...] = local_indices

    clockwise_peer = jnp.mod(mesh_rank + 1, world_size)
    counterclockwise_peer = jnp.mod(mesh_rank - 1, world_size)
    peers = (mesh_rank, clockwise_peer, counterclockwise_peer)
    seed_copies = []
    for copy_index, (value_buffer, index_buffer, peer) in enumerate(
        zip(value_buffers, index_buffers, peers, strict=True)
    ):
        seed_copies.extend(
            (
                pltpu.make_async_remote_copy(
                    value_buffer,
                    candidate_values_ref.at[
                        mesh_rank, pl.ds(token_start, _TOKEN_BLOCK), :
                    ],
                    send_semaphores.at[2 * copy_index],
                    receive_semaphores.at[2 * copy_index],
                    device_id=(peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ),
                pltpu.make_async_remote_copy(
                    index_buffer,
                    candidate_indices_ref.at[
                        mesh_rank, pl.ds(token_start, _TOKEN_BLOCK), :
                    ],
                    send_semaphores.at[2 * copy_index + 1],
                    receive_semaphores.at[2 * copy_index + 1],
                    device_id=(peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ),
            )
        )
    _start_and_wait(seed_copies)

    half_world = world_size // 2
    for step in range(half_world - 2):
        clockwise_source = jnp.mod(mesh_rank - step - 1, world_size)
        counterclockwise_source = jnp.mod(mesh_rank + step + 1, world_size)
        forward_copies = (
            pltpu.make_async_remote_copy(
                candidate_values_ref.at[
                    clockwise_source, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                candidate_values_ref.at[
                    clockwise_source, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                send_semaphores.at[6],
                receive_semaphores.at[6],
                device_id=(clockwise_peer,),
                device_id_type=pl.DeviceIdType.MESH,
            ),
            pltpu.make_async_remote_copy(
                candidate_indices_ref.at[
                    clockwise_source, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                candidate_indices_ref.at[
                    clockwise_source, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                send_semaphores.at[7],
                receive_semaphores.at[7],
                device_id=(clockwise_peer,),
                device_id_type=pl.DeviceIdType.MESH,
            ),
            pltpu.make_async_remote_copy(
                candidate_values_ref.at[
                    counterclockwise_source, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                candidate_values_ref.at[
                    counterclockwise_source, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                send_semaphores.at[8],
                receive_semaphores.at[8],
                device_id=(counterclockwise_peer,),
                device_id_type=pl.DeviceIdType.MESH,
            ),
            pltpu.make_async_remote_copy(
                candidate_indices_ref.at[
                    counterclockwise_source, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                candidate_indices_ref.at[
                    counterclockwise_source, pl.ds(token_start, _TOKEN_BLOCK), :
                ],
                send_semaphores.at[9],
                receive_semaphores.at[9],
                device_id=(counterclockwise_peer,),
                device_id_type=pl.DeviceIdType.MESH,
            ),
        )
        _start_and_wait(forward_copies)

    opposite_source = jnp.mod(mesh_rank - half_world + 1, world_size)
    final_copies = (
        pltpu.make_async_remote_copy(
            candidate_values_ref.at[
                opposite_source, pl.ds(token_start, _TOKEN_BLOCK), :
            ],
            candidate_values_ref.at[
                opposite_source, pl.ds(token_start, _TOKEN_BLOCK), :
            ],
            send_semaphores.at[10],
            receive_semaphores.at[10],
            device_id=(clockwise_peer,),
            device_id_type=pl.DeviceIdType.MESH,
        ),
        pltpu.make_async_remote_copy(
            candidate_indices_ref.at[
                opposite_source, pl.ds(token_start, _TOKEN_BLOCK), :
            ],
            candidate_indices_ref.at[
                opposite_source, pl.ds(token_start, _TOKEN_BLOCK), :
            ],
            send_semaphores.at[11],
            receive_semaphores.at[11],
            device_id=(clockwise_peer,),
            device_id_type=pl.DeviceIdType.MESH,
        ),
    )
    _start_and_wait(final_copies)


def _ring_candidates(
    local_logits,
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
    aligned_width = _ceil_multiple(local_width, _VOCAB_ALIGNMENT)
    local_logits = jnp.pad(
        local_logits,
        ((0, padded_tokens - tokens), (0, aligned_width - local_width)),
        constant_values=-jnp.inf,
    )
    kernel = functools.partial(
        _distributed_topk_kernel,
        world_size=world_size,
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
        pltpu.SemaphoreType.DMA((12,)),
        pltpu.SemaphoreType.DMA((12,)),
        pltpu.VMEM((_TOKEN_BLOCK, _DMA_WIDTH), local_logits.dtype),
        pltpu.VMEM((_TOKEN_BLOCK, _DMA_WIDTH), jnp.int32),
        pltpu.VMEM((_TOKEN_BLOCK, _DMA_WIDTH), local_logits.dtype),
        pltpu.VMEM((_TOKEN_BLOCK, _DMA_WIDTH), jnp.int32),
        pltpu.VMEM((_TOKEN_BLOCK, _DMA_WIDTH), local_logits.dtype),
        pltpu.VMEM((_TOKEN_BLOCK, _DMA_WIDTH), jnp.int32),
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(candidate_shape, local_logits.dtype),
            jax.ShapeDtypeStruct(candidate_shape, jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(padded_tokens // _TOKEN_BLOCK,),
            in_specs=(
                pl.BlockSpec(
                    (_TOKEN_BLOCK, aligned_width),
                    lambda tile: (tile, 0),
                    memory_space=pltpu.VMEM,
                ),
            ),
            out_specs=output_specs,
            scratch_shapes=scratch_shapes,
        ),
        name=f"distributed_topk_ring_tp{world_size}",
    )(local_logits)


def distributed_top_k(
    local_logits,
    *,
    world_size: int,
    org_vocab_size: int,
    global_vocab_size: int,
    org_vocab_padded_count: int,
    k: int,
):
    """Return exact global TopK values and unpadded token IDs on every rank."""

    import jax
    import jax.numpy as jnp

    tokens, local_width = local_logits.shape
    aligned_width = _ceil_multiple(local_width, _VOCAB_ALIGNMENT)
    candidate_values, candidate_indices = _ring_candidates(
        local_logits,
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
    packed_ids = jnp.take_along_axis(candidate_indices, candidate_positions, axis=1)
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
