# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA over ragged per-row candidate counts (decode on a split latent cache).

On a split latent cache (kv_shard.py) a rank holds ~1/shards of each row's top-k,
compacted to a prefix of the row (kernel._own_shard_entries). The per-query kernel
(kernel.py) gathers and attends every slot of every row. This path splits each
row's live prefix into blocks of ``block`` slots, gathers only the live blocks
(one SparseCore gather per chunk of rows, packed back to back) and attends them.
Gather bytes and attention work scale with the live count.

Each row's blocks land in one VMEM tile and get one softmax over a bucket of
blocks: the per-query kernel's math on a narrower width (the masked slots of the
wider bucket add exact zeros). Without rope lanes the output is bit-exact with it;
with rope the score dot's K-chunk order follows the width, so a score can differ
by one f32 rounding.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.mla_sparse import dsa_gather, kv_cache_utils
from sglang_tpu.attention.pallas.mla_sparse.kernel import (
    DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE,
    DEFAULT_VMEM_LIMIT_BYTES,
    prepare_gather_indices,
    prepare_q_inputs,
)

# Gather subchunks per subcore (multiples of 4 = the gather's streams): 4 spreads a
# packed live prefix of a few thousand rows over all subcores. Read at trace time.
GATHER_SUBCHUNKS = 4
_F32_MIN = float(jnp.finfo(jnp.float32).min)
# Rows whose block DMAs are in flight ahead of the row being attended: one row's
# compute (~0.3 us) is shorter than a DMA's latency.
ROW_PREFETCH = 7
# Softmax widths in blocks, plus the row's full width: one static code path each.
BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16)
# Rows per schedule, gather and kernel call: the packed gather buffer holds every
# slot of a chunk's rows, as the per-query path's gather chunk does.
CHUNK_ROWS = DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE
# Rows of the kernel's table (i32[_ROW_TABLE_ROWS, T]).
_R_TOKEN, _R_SRC, _R_N, _R_LEN, _ROW_TABLE_ROWS = range(5)


def _live_blocks(kv_lens, token_end, block):
    live = jnp.arange(kv_lens.shape[0], dtype=jnp.int32) < token_end
    return jnp.where(live, (kv_lens + block - 1) // block, 0).astype(jnp.int32)


def _row_of(ends, entries):
    """The row each flat entry falls in (rows end at ``ends``, inclusive cumsum)."""
    row = jnp.sum((ends[None, :] <= entries[:, None]).astype(jnp.int32), axis=1)
    return jnp.minimum(row, ends.shape[0] - 1)


def _row_start(ends, entries):
    """First entry of each entry's row: the largest row end at or before it. Not
    ``(ends - counts)[row]``: XLA TPU ran that scalar gather as a 36 us loop at 512."""
    done = ends[None, :] <= entries[:, None]
    return jnp.max(jnp.where(done, ends[None, :], 0), axis=1)


def ragged_schedule(
    kv_lens: jax.Array,  # i32[T]: live slots per row
    token_end: jax.Array,  # i32: rows at or past it are padding
    *,
    block: int,
    width: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Each row's live blocks back to back, in row order.

    Returns ``(block_row i32[C], block_index i32[C], blocks i32[T], total i32)``
    for capacity ``C = T * width // block``: entry ``b < total`` is block
    ``block_index[b]`` of row ``block_row[b]``; entries past ``total`` are clamped
    in-range filler.
    """
    num_rows = kv_lens.shape[0]
    per_row = width // block
    capacity = num_rows * per_row
    blocks = _live_blocks(kv_lens, token_end, block)
    ends = jnp.cumsum(blocks, dtype=jnp.int32)
    b = jnp.arange(capacity, dtype=jnp.int32)
    block_row = _row_of(ends, b)
    block_index = jnp.clip(b - _row_start(ends, b), 0, per_row - 1)
    return block_row, block_index, blocks, ends[-1]


def row_table(
    kv_lens: jax.Array, blocks: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rows with a live block, in row order: ``(table i32[_ROW_TABLE_ROWS, T],
    attended bool[T], rows i32)``; per row its token, first packed block
    (ragged_schedule order), block count and live slots."""
    num_rows = kv_lens.shape[0]
    starts = jnp.cumsum(blocks, dtype=jnp.int32) - blocks
    attended = blocks > 0
    dest = jnp.where(attended, jnp.cumsum(attended, dtype=jnp.int32) - 1, num_rows)
    cols = jnp.stack(
        [jnp.arange(num_rows, dtype=jnp.int32), starts, blocks, kv_lens]
    ).astype(jnp.int32)
    table = jnp.zeros((_ROW_TABLE_ROWS, num_rows), jnp.int32)
    table = table.at[:4, dest].set(cols, mode="drop")
    return table, attended, jnp.sum(attended.astype(jnp.int32))


def _bucket_kernel(
    table_ref,  # i32[_ROW_TABLE_ROWS, T]
    rows_ref,  # i32[1]: rows with a live block
    q_hbm_ref,
    nope_hbm_ref,
    rope_hbm_ref,
    o_hbm_ref,
    lse_hbm_ref,
    nbuf,  # u8[S, max_blocks * block * WORD_BYTES, 128]; S = ROW_PREFETCH + 1 slots
    rbuf,  # u8[S, max_blocks * block, 128]
    q_ref,  # [S, H, D]
    o_ref,  # [S, H, D]
    lse_ref,  # f32[S, H, 1]
    sems,  # DMA[5, S]: kv nope, kv rope, q, o, lse
    *,
    block: int,
    buckets: tuple[int, ...],
    sm_scale: float,
    k_scale: float,
):
    _, num_heads, head_dim = q_hbm_ref.shape
    nope_dim = kv_cache_utils.WORD_BYTES * kv_cache_utils.TILE_LANE_BYTES
    rows_b4 = block * kv_cache_utils.WORD_BYTES
    q_dtype = q_hbm_ref.dtype
    num_live = rows_ref[0]
    num_slots = nbuf.shape[0]

    def block_copies(src, slot, j):
        return (
            pltpu.make_async_copy(nope_hbm_ref.at[src], nbuf.at[slot, pl.ds(j * rows_b4, rows_b4)], sems.at[0, slot]),
            pltpu.make_async_copy(rope_hbm_ref.at[src], rbuf.at[slot, pl.ds(j * block, block)], sems.at[1, slot]),
        )  # fmt: skip

    def q_copy(row, slot):
        return pltpu.make_async_copy(
            q_hbm_ref.at[row], q_ref.at[slot], sems.at[2, slot]
        )

    def out_copies(row, slot):
        return (
            pltpu.make_async_copy(o_ref.at[slot], o_hbm_ref.at[row], sems.at[3, slot]),
            pltpu.make_async_copy(lse_ref.at[slot], lse_hbm_ref.at[row], sems.at[4, slot]),
        )  # fmt: skip

    def issue(k, slot):
        src0 = table_ref[_R_SRC, k]

        def one(j, carry):
            for cp in block_copies(src0 + j, slot, j):
                cp.start()
            return carry

        lax.fori_loop(0, table_ref[_R_N, k], one, 0)
        q_copy(table_ref[_R_TOKEN, k], slot).start()

    def wait(k, slot):
        def one(j, carry):
            for cp in block_copies(0, slot, 0):  # waits count bytes
                cp.wait()
            return carry

        lax.fori_loop(0, table_ref[_R_N, k], one, 0)
        q_copy(0, slot).wait()

    def attend(k, slot, width):
        # The per-query kernel's unpack, masking and softmax (kernel._attention_kernel).
        kv_len = table_ref[_R_LEN, k]
        nope = nbuf[slot, pl.ds(0, width * kv_cache_utils.WORD_BYTES)].reshape(
            width, nope_dim
        )
        nope = pltpu.bitcast(nope, jnp.float8_e4m3fn)
        rope = pltpu.bitcast(rbuf[slot, pl.ds(0, width)], jnp.float8_e4m3fn)
        kv = jnp.concatenate([nope, rope], axis=-1)
        if kv.shape[-1] < head_dim:
            kv = jnp.pad(kv, ((0, 0), (0, head_dim - kv.shape[-1])))
        kv = jnp.where(lax.broadcasted_iota(jnp.int32, kv.shape, 0) < kv_len, kv, 0)
        q = q_ref[slot]
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        if k_scale != 1.0:
            s *= k_scale
        s = jnp.where(lax.broadcasted_iota(jnp.int32, s.shape, 1) < kv_len, s, _F32_MIN)
        m = jnp.max(s, axis=1, keepdims=True)
        p = jnp.exp(s - m)
        l_sum = jnp.sum(p, axis=1, keepdims=True)
        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
        if k_scale != 1.0:
            pv *= k_scale
        o_ref[slot] = (pv * pltpu.reciprocal(l_sum, approx=True)).astype(q_dtype)
        lse_ref[slot] = m + jnp.log(l_sum)

    def body(k, carry):
        slot = lax.rem(k, num_slots)
        ahead = k + num_slots - 1

        @pl.when(ahead < num_live)
        def _prefetch():
            issue(ahead, lax.rem(ahead, num_slots))

        wait(k, slot)

        @pl.when(k >= num_slots)
        def _slot_free():
            for cp in out_copies(0, slot):
                cp.wait()

        blocks = table_ref[_R_N, k]
        lower = 0
        for bucket in buckets:
            pl.when((blocks > lower) & (blocks <= bucket))(
                functools.partial(attend, k, slot, bucket * block)
            )
            lower = bucket
        for cp in out_copies(table_ref[_R_TOKEN, k], slot):
            cp.start()
        return carry

    @pl.when(num_live > 0)
    def _run():
        for j in range(num_slots - 1):
            pl.when(j < num_live)(functools.partial(issue, j, j))
        lax.fori_loop(0, num_live, body, 0)
        # Each used slot still has its last row's output copies in flight.
        for j in range(num_slots):

            @pl.when(j < num_live)
            def _drain():
                for cp in out_copies(0, j):
                    cp.wait()


def bucket_attention_kernel(
    q: jax.Array,  # [T, H, D], D a multiple of 128
    nope: jax.Array,  # gathered u8[C, block * WORD_BYTES, 128]
    rope: jax.Array,  # gathered u8[C, block, 128]
    table: jax.Array,  # row_table
    rows: jax.Array,  # i32
    *,
    block: int,
    max_blocks: int,
    sm_scale: float,
    k_scale: float,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array]:
    """``(out [T, H, D], lse f32[T, H, 1])``: one softmax per table row over a
    bucket of its blocks; rows with no block are left unwritten (the caller masks
    them)."""
    num_rows, num_heads, head_dim = q.shape
    buckets = tuple(b for b in BUCKETS if b < max_blocks) + (max_blocks,)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    slots = ROW_PREFETCH + 1
    scratch = [
        pltpu.VMEM((slots, max_blocks * nope.shape[1], nope.shape[2]), nope.dtype),
        pltpu.VMEM((slots, max_blocks * rope.shape[1], rope.shape[2]), rope.dtype),
        pltpu.VMEM((slots, num_heads, head_dim), q.dtype),
        pltpu.VMEM((slots, num_heads, head_dim), q.dtype),
        pltpu.VMEM((slots, num_heads, 1), jnp.float32),
        pltpu.SemaphoreType.DMA((5, slots)),
    ]
    name = f"MLA-ragged-rows-b{block}-m{max_blocks}-s{slots}"
    return pl.pallas_call(
        functools.partial(
            _bucket_kernel,
            block=block,
            buckets=buckets,
            sm_scale=sm_scale,
            k_scale=k_scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[hbm] * 3,
            out_specs=[hbm, hbm],
            scratch_shapes=scratch,
        ),
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((num_rows, num_heads, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=name,
    )(table, jnp.reshape(rows, (1,)).astype(jnp.int32), q, nope, rope)


def gather_ragged_blocks(
    cache_nope: jax.Array,  # u8[pages, page_size, 4, 128]
    cache_rope: jax.Array,  # u8[pages, page_size // 4, 4, 128]
    physical_indices: jax.Array,  # i32[T, width]; live prefix per row
    block_row: jax.Array,
    block_index: jax.Array,
    total: jax.Array,
    *,
    block: int,
    gather_rope: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """The scheduled blocks' rows, packed: (u8[C, block * 4, 128], u8[C, block, 128])."""
    num_rows, width = physical_indices.shape
    per_row = width // block
    capacity = block_row.shape[0]
    rows = physical_indices.reshape(num_rows * per_row, block)[
        block_row * per_row + block_index
    ]
    nope, rope = dsa_gather.dsa_gather(
        cache_nope,
        cache_rope,
        rows.reshape(-1),
        num_valid_indices=total * block,
        num_row_subchunks=GATHER_SUBCHUNKS,
        gather_rope=gather_rope,
    )
    return (
        nope.reshape(capacity, block * kv_cache_utils.WORD_BYTES, -1),
        rope.reshape(capacity, block, -1),
    )


def _attend_chunk(q, kv_lens, physical, token_end, cache_nope, cache_rope, *,
                  block, sm_scale, k_scale, gather_rope):  # fmt: skip
    """One chunk of rows: schedule, gather and attend its live blocks."""
    width = physical.shape[1]
    block_row, block_index, blocks, total = ragged_schedule(
        kv_lens, token_end, block=block, width=width
    )
    nope, rope = gather_ragged_blocks(
        cache_nope,
        cache_rope,
        physical,
        block_row,
        block_index,
        total,
        block=block,
        gather_rope=gather_rope,
    )
    table, attended, rows = row_table(kv_lens, blocks)
    out, lse = bucket_attention_kernel(
        q,
        nope,
        rope,
        table,
        rows,
        block=block,
        max_blocks=width // block,
        sm_scale=sm_scale,
        k_scale=k_scale,
    )
    out = jnp.where(attended[:, None, None], out, 0)
    lse = jnp.where(attended[:, None], lse[..., 0], _F32_MIN)
    return out, lse


def sparse_attention_ragged(
    q: jax.Array,  # [T, H, head_dim]
    cache_nope: jax.Array,  # u8[pages, page_size, 4, 128]; this rank's share
    cache_rope: jax.Array,  # u8[pages, page_size // 4, 4, 128]
    topk_indices: jax.Array,  # i32[T, width]; sequence positions, -1 padded
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float,
    k_scale: float,
    block: int,
    kv_shards: tuple[int, int, int],
    gather_rope: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """``sparse_ragged_paged_attention(..., return_lse=True, kv_shards=...)`` for a
    decode-sized batch, gathering and attending only each row's live blocks.

    Returns ``(out [T, H, head_dim], lse f32[T, H])``. A row with no live entry
    gets out 0 and lse at the f32 floor, as the per-query kernel gives it.
    """
    num_rows, num_heads, head_dim = q.shape
    width = topk_indices.shape[1]
    if width % block:
        raise ValueError(f"row width {width} is not a multiple of block {block}")
    token_end = cu_q_lens[distribution[2]]
    kv_lens, physical = prepare_gather_indices(
        topk_indices,
        page_indices,
        cu_q_lens,
        page_size=cache_nope.shape[1],
        total_num_pages=cache_nope.shape[0],
        kv_shards=kv_shards,
    )
    # Materialize: without it XLA fused the [T, width] row count into each of the
    # schedule's T * width / block entries (a 36 us fusion at T = 64).
    kv_lens, physical = jax.lax.optimization_barrier(
        (kv_lens.astype(jnp.int32), physical)
    )
    qp = prepare_q_inputs(q)
    outs, lses = [], []
    for start in range(0, num_rows, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, num_rows)
        out, lse = _attend_chunk(
            qp[start:end], kv_lens[start:end], physical[start:end],
            jnp.clip(token_end - start, 0, end - start), cache_nope, cache_rope,
            block=block, sm_scale=sm_scale, k_scale=k_scale, gather_rope=gather_rope,
        )  # fmt: skip
        outs.append(out)
        lses.append(lse)
    out = outs[0] if len(outs) == 1 else jnp.concatenate(outs)
    lse = lses[0] if len(lses) == 1 else jnp.concatenate(lses)
    return out[:, :num_heads, :head_dim].astype(q.dtype), lse[:, :num_heads]
