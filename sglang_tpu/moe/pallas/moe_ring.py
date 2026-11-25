# SPDX-License-Identifier: Apache-2.0
"""MoE ring kernel: the routed-expert part of an MoE layer for small batches.

Where it sits among the MoE paths (`--moe-runner-backend`):

  pallas_tpu_fused_decode:
    one kernel for the whole MoE block: routing, routed experts, shared expert
    and the cross-rank all-reduce (fused_moe_decode, moe_tp_decode).
  pallas_tpu (default), per fused-MoE call:
    EP = 1 and the cost model below prefers it: this kernel;
    otherwise: gmm_v2.

At EP = 1 (every expert is on this rank, as a 1/TP slice of its width) a small
batch touches many experts with only a few rows each, so the time goes to
reading each touched expert's weights from HBM, not to the matmuls. gmm_v2
reads them one tile at a time. This kernel:

  * keeps `slots` expert-sized buffers in VMEM and always has the next
    `slots - 1` experts' weights being copied in while it computes the current
    one, so the copies overlap each other and the compute;
  * reads each touched expert once and runs it only on the rows routed to it,
    in tiles of 8, 16 or 32 rows depending on how many rows the expert has;
  * adds each expert's weighted output into an f32 [T, H] buffer that stays in
    VMEM, so nothing goes back to HBM between the two matmuls.

Before the kernel, XLA groups the (token, expert) pairs by expert and counts
the rows per expert (`_route`).

Each (token, expert) output is bit-identical to gmm_v2's: the same input
quantization (per row, per 512 columns), bf16 partial sums and clamped SiLU.
Only the final f32 sum runs in expert order rather than top-k order.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.gmm_v2 import _FP32_MOE_SCALES
from sglang_tpu.moe.pallas.moe_tp_decode.all_reduce import tiled_bytes
from sglang_tpu.moe.pallas.moe_tp_decode.kernel import (
    _Q_BLOCK,
    VMEM_FRACTION,
    VMEM_HEADROOM,
)

_FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
# Ring depth: slots - 1 experts in flight while one computes.
_SLOTS = 4
# DMAs per expert's w1 / w2 (split along their leading weight dim).
_W1_CHUNKS = 2
_W2_CHUNKS = 1
# Row-tile sizes. The kernel runs one expert on a fixed-size batch of rows (a
# "row tile"): it gathers the rows routed to that expert into a small buffer and
# runs both matmuls on it. The buffer shape must be static, so the size is picked
# from this list by how many rows the expert has:
#   1-8 rows: one 8-row tile; 9-16 rows: one 16-row tile;
#   more: ceil(rows / 32) tiles of 32 rows.
# A tile's unused rows repeat its last real row and are not added to the output.
# Small tiles matter because a small batch gives most experts only a few rows, so a
# single large tile would mostly compute padding; the largest size bounds the
# gather buffers' VMEM. The kernel has one loop per size.
_TILE_ROWS = (8, 16, 32)
# Bytes of one VMEM tile: [8, 128] f32, [16, 128] bf16 or [32, 128] fp8.
_VMEM_TILE_BYTES = 4096


def _vmem_need(
    tokens: int,
    hidden: int,
    n: int,
    slots: int,
    tile_rows: tuple[int, ...],
    info: pltpu.TpuInfo,
) -> int:
    """VMEM bytes of every buffer the kernel holds, padded to Mosaic tiles.

    Args:
      tokens: T, rows of x.
      hidden: H, model width.
      n: expert width on this rank (w1 is [H, 2n], w2 is [n, H]).
      slots: weight ring depth.
      tile_rows: the row-tile sizes; the gather buffers hold the largest.
      info: the TPU's lane and sublane counts.
    """
    b = functools.partial(tiled_bytes, info=info)
    nq = hidden // _Q_BLOCK
    lanes = info.num_lanes
    ring = slots * (
        b((hidden, 2 * n), jnp.float8_e4m3fn)  # w1
        + b((n, hidden), jnp.float8_e4m3fn)  # w2
        + b((1, 2 * n), jnp.float32)  # w1 scale
        + b((1, hidden), jnp.float32)  # w2 scale
    )
    rows = (
        2 * b((tokens, hidden), jnp.bfloat16)  # x in, out
        + 2 * b((tokens, hidden), jnp.float32)  # quantized x, accumulator
        + b((nq, tokens, lanes), jnp.float32)  # x block scales
        + b((max(tile_rows), hidden), jnp.float32)  # one tile's gathered rows
        + b((nq, max(tile_rows), lanes), jnp.float32)  # and their block scales
    )
    return ring + rows


def _vmem_limit(
    tokens: int,
    hidden: int,
    n: int,
    slots: int,
    tile_rows: tuple[int, ...],
    info: pltpu.TpuInfo,
) -> int | None:
    """The kernel's VMEM limit in bytes, by moe_tp_decode's rule: VMEM_HEADROOM
    times the planned bytes, or None when that passes VMEM_FRACTION of the
    capacity."""
    limit = int(_vmem_need(tokens, hidden, n, slots, tile_rows, info) * VMEM_HEADROOM)
    if limit > int(info.vmem_capacity_bytes * VMEM_FRACTION):
        return None
    return min(limit, info.vmem_capacity_bytes)


def fit_moe_ring(
    tokens: int,
    hidden: int,
    two_n: int,
    *,
    slots: int = _SLOTS,
    tile_rows: tuple[int, ...] = _TILE_ROWS,
) -> bool:
    """Whether the kernel can run these shapes: aligned, bit-compatible with
    gmm_v2, and within VMEM. There is no token cap of its own; the VMEM limit
    bounds the tokens (the row buffers grow with tokens * hidden)."""
    info = pltpu.get_tpu_info()
    n = two_n // 2
    # _down quantizes its lhs as one block, which is gmm_v2's rule only for
    # n <= _Q_BLOCK; wider experts would not match gmm_v2.
    if hidden % _Q_BLOCK or n % info.num_lanes or n > _Q_BLOCK or tokens < 1:
        return False
    return _vmem_limit(tokens, hidden, n, slots, tile_rows, info) is not None


def _route(
    topk_ids: jax.Array, topk_weights: jax.Array, num_experts: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Groups the (token, expert) pairs by expert, in XLA, before the kernel.

    Every token has k pairs, one per expert it was routed to. The kernel wants
    them listed expert by expert, so that one expert's rows are contiguous.

    Args:
      topk_ids: int [T, k]. Expert id of each pair. Ids outside [0, E) are
        padding: they are listed after every real pair and never read.
      topk_weights: [T, k]. Routing weight of each pair.
      num_experts: E.

    Returns:
      counts: int32 [E]. Number of pairs of each expert.
      offsets: int32 [E]. Where each expert's pairs start in the list.
      tok: int32 [T * k]. Token of each pair, in list order (by expert, then by
        the pair's original position).
      wt: f32 [T * k]. Routing weight of each pair, in list order.
    """
    tokens, k = topk_ids.shape
    p = tokens * k
    ids = topk_ids.reshape(p).astype(jnp.int32)
    valid = (ids >= 0) & (ids < num_experts)
    key = jnp.where(valid, ids, num_experts)
    counts = jnp.sum(
        (key[:, None] == jnp.arange(num_experts, dtype=jnp.int32)[None, :]).astype(
            jnp.int32
        ),
        axis=0,
    )
    offsets = jnp.cumsum(counts) - counts
    # Each slot's row in (expert, slot) order from pairwise compares, then a
    # one-hot select per row: no sort or gather over the <= 2048 slots.
    slot = jnp.arange(p, dtype=jnp.int32)
    before = (key[None, :] < key[:, None]) | (
        (key[None, :] == key[:, None]) & (slot[None, :] < slot[:, None])
    )
    row = jnp.sum(before, axis=1, dtype=jnp.int32)
    hit = row[None, :] == slot[:, None]
    tok = jnp.sum(jnp.where(hit, (slot // k)[None, :], 0), axis=1, dtype=jnp.int32)
    weights = topk_weights.reshape(p).astype(jnp.float32)
    wt = jnp.sum(jnp.where(hit, weights[None, :], 0.0), axis=1)
    return counts.astype(jnp.int32), offsets.astype(jnp.int32), tok, wt


def _gate_up(
    xq: jax.Array,
    xs: list[jax.Array],
    w_ref: jax.Array,
    s_row: jax.Array,
    *,
    limit: float | None,
) -> jax.Array:
    """First matmul of one expert on one row tile, then the activation.

    Computes x @ w1 the way gmm_v2 does, so the result is bit-identical. The H
    input columns are split into blocks of _Q_BLOCK (512). Each block's fp8 x
    fp8 product is rounded to bf16, multiplied by that block's x scale and by
    the weight's per-column scale, and the blocks are summed in bf16. The 2n
    outputs are a gate half and an up half; the result is
    silu(min(gate, limit)) * clip(up, -limit, limit).

    Args:
      xq: f32 [M, H]. The tile's rows, already quantized to fp8 values.
      xs: H / 512 arrays, f32 [M, lanes]. Each row's scale for each block,
        repeated across one lane group.
      w_ref: VMEM fp8 [H, 2n]. The expert's gate/up weights.
      s_row: f32 [1, 2n]. Their per-column scales.
      limit: SiLU clamp limit, or None for no clamp.

    Returns:
      bf16 [M, n].
    """
    m, h = xq.shape
    two_n = w_ref.shape[1]
    n = two_n // 2
    lanes = xs[0].shape[1]
    sdt = jnp.float32 if _FP32_MOE_SCALES else jnp.bfloat16
    acc = jnp.zeros((m, two_n), jnp.bfloat16)
    for b in range(h // _Q_BLOCK):
        part = jax.lax.dot_general(
            xq[:, b * _Q_BLOCK : (b + 1) * _Q_BLOCK].astype(jnp.float8_e4m3fn),
            w_ref[pl.ds(b * _Q_BLOCK, _Q_BLOCK), :],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        sc = jnp.concatenate([xs[b]] * (two_n // lanes), axis=1)
        acc = acc + (part.astype(sdt) * sc.astype(sdt) * s_row.astype(sdt)).astype(
            jnp.bfloat16
        )
    gate, up = acc[:, :n], acc[:, n:]
    if limit is not None:
        gate = jnp.minimum(gate, limit)
        up = jnp.clip(up, -limit, limit)
    return (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(
        jnp.bfloat16
    )


def _down(mid: jax.Array, w_ref: jax.Array, s_row: jax.Array) -> jax.Array:
    """Second matmul of one expert on one row tile: mid @ w2.

    gmm_v2 quantizes a matmul's input to fp8 with one scale per row per 512
    columns (its lhs_quant_block_size). Here each row gets a single scale over
    all n columns, which is the same thing only while n <= 512. fit_moe_ring
    refuses wider experts, so this is never called with n > 512.

    Args:
      mid: bf16 [M, n]. Output of _gate_up.
      w_ref: VMEM fp8 [n, H]. The expert's down weights.
      s_row: f32 [1, H]. Their per-column scales.

    Returns:
      bf16 [M, H].
    """
    sdt = jnp.float32 if _FP32_MOE_SCALES else jnp.bfloat16
    scale = jnp.max(jnp.abs(mid), axis=1, keepdims=True) / _FP8_MAX
    inv = jnp.where(scale == 0, 0, 1 / scale)
    part = jax.lax.dot_general(
        (mid * inv).astype(jnp.float8_e4m3fn),
        w_ref[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    y = (part.astype(sdt) * scale.astype(sdt) * s_row.astype(sdt)).astype(jnp.bfloat16)
    return jnp.zeros_like(y) + y


def _kernel(
    counts_ref: jax.Array,  # SMEM int32 [E]. Rows routed to each expert.
    offsets_ref: jax.Array,  # SMEM int32 [E]. Each expert's first pair in the list.
    tok_ref: jax.Array,  # SMEM int32 [T * k]. Token of each pair, grouped by expert.
    wt_ref: jax.Array,  # SMEM f32 [T * k]. Routing weight of each pair.
    x_ref: jax.Array,  # VMEM bf16 [T, H]. Input rows.
    w1_hbm: jax.Array,  # HBM fp8 [E, H, 2n]. Gate/up weights.
    s1_hbm: jax.Array,  # HBM f32 [E, 1, 1, 2n]. Their per-column scales.
    w2_hbm: jax.Array,  # HBM fp8 [E, n, H]. Down weights.
    s2_hbm: jax.Array,  # HBM f32 [E, 1, 1, H]. Their per-column scales.
    o_ref: jax.Array,  # VMEM [T, H], x's dtype. Output rows.
    w1_buf: jax.Array,  # VMEM fp8 [slots, H, 2n]. Ring: w1 of the experts in flight.
    s1_buf: jax.Array,  # VMEM f32 [slots, 1, 2n]. Ring: their w1 scales.
    w2_buf: jax.Array,  # VMEM fp8 [slots, n, H]. Ring: their w2.
    s2_buf: jax.Array,  # VMEM f32 [slots, 1, H]. Ring: their w2 scales.
    sems: jax.Array,  # DMA semaphores [slots, w1_chunks + w2_chunks + 2], one per copy.
    xq_ref: jax.Array,  # VMEM f32 [T, H]. x quantized to fp8 values.
    xs_ref: jax.Array,  # VMEM f32 [nq, T, lanes]. x's scale per row per block.
    acc_ref: jax.Array,  # VMEM f32 [T, H]. Sum of weight * expert output.
    xg_ref: jax.Array,  # VMEM f32 [R, H]. One row tile gathered from xq_ref.
    xsg_ref: jax.Array,  # VMEM f32 [nq, R, lanes]. That tile's block scales.
    active_ref: jax.Array,  # SMEM int32 [E + 1]. Ids of the hit experts, by tile size.
    *,
    limit: float | None,  # SiLU clamp limit, or None.
    slots: int,  # Weight ring depth.
    w1_chunks: int,  # DMAs per expert's w1.
    w2_chunks: int,  # DMAs per expert's w2.
    tile_rows: tuple[int, ...],  # Row-tile sizes, see _TILE_ROWS.
):
    """Kernel body. In the shapes: T tokens, k experts per token, E experts, H
    model width, n expert width, nq = H / 512 quantization blocks, R =
    max(tile_rows)."""
    t, h = x_ref.shape
    num_experts, _, two_n = w1_hbm.shape
    n = two_n // 2
    nq = h // _Q_BLOCK
    lanes = xs_ref.shape[-1]
    h1 = h // w1_chunks
    n2 = n // w2_chunks

    def weight_copies(i: int | jax.Array, slot: int | jax.Array) -> list:
        """The DMAs that copy the i-th hit expert's weights and scales from HBM
        into ring slot `slot`: w1 in w1_chunks pieces, its scale, w2 in
        w2_chunks pieces, its scale. Each has its own semaphore."""
        e = active_ref[i]
        out = []
        for c in range(w1_chunks):
            out.append(
                pltpu.make_async_copy(
                    w1_hbm.at[e, pl.ds(c * h1, h1), :],
                    w1_buf.at[slot, pl.ds(c * h1, h1), :],
                    sems.at[slot, c],
                )
            )
        out.append(
            pltpu.make_async_copy(
                s1_hbm.at[e, 0], s1_buf.at[slot], sems.at[slot, w1_chunks]
            )
        )
        for c in range(w2_chunks):
            out.append(
                pltpu.make_async_copy(
                    w2_hbm.at[e, pl.ds(c * n2, n2), :],
                    w2_buf.at[slot, pl.ds(c * n2, n2), :],
                    sems.at[slot, w1_chunks + 1 + c],
                )
            )
        out.append(
            pltpu.make_async_copy(
                s2_hbm.at[e, 0],
                s2_buf.at[slot],
                sems.at[slot, w1_chunks + 1 + w2_chunks],
            )
        )
        return out

    def start_dma(lo: jax.Array | int, hi: jax.Array | int) -> None:
        """Starts the weight copies of hit experts lo..hi-1 that belong to the
        first slots - 1 experts (the ones in flight before the loop starts)."""
        for s in range(slots - 1):

            @pl.when((s >= lo) & (s < hi))
            def _(s=s):
                for cp in weight_copies(s, s):
                    cp.start()

    # Hit experts grouped by pair-count class: class k takes counts in
    # (tile_rows[k-1], tile_rows[k]], the last class everything above. The first
    # class's DMAs start before the other classes are compacted.
    ends = []
    count = jnp.int32(0)
    for k, hi in enumerate(tile_rows):
        lo = tile_rows[k - 1] if k else 0
        last = k == len(tile_rows) - 1

        def compact(e, count, lo=lo, hi=hi, last=last):
            # Branch-free: a later hit overwrites a miss's entry.
            c = counts_ref[e]
            active_ref[count] = e
            hit = (c > lo) if last else ((c > lo) & (c <= hi))
            return count + hit.astype(jnp.int32)

        count = jax.lax.fori_loop(0, num_experts, compact, count)
        ends.append(count)
        if k == 0:
            start_dma(0, count)
    n_u = count
    start_dma(ends[0], n_u)

    # x quantized once, as gmm_v2 quantizes each lhs block (bf16 scale and
    # inverse, no clip), while the first experts' weights are in flight.
    for b in range(nq):
        blk = x_ref[:, b * _Q_BLOCK : (b + 1) * _Q_BLOCK]
        scale = jnp.max(jnp.abs(blk), axis=1, keepdims=True) / _FP8_MAX
        inv = jnp.where(scale == 0, 0, 1 / scale)
        xq_ref[:, b * _Q_BLOCK : (b + 1) * _Q_BLOCK] = (
            (blk * inv).astype(jnp.float8_e4m3fn).astype(jnp.float32)
        )
        xs_ref[b] = jnp.broadcast_to(scale.astype(jnp.float32), (t, lanes))
    acc_ref[...] = jnp.zeros_like(acc_ref)

    def run_expert_tile(
        slot: jax.Array, base: jax.Array, rem: jax.Array, tm: int
    ) -> None:
        """Runs one row tile of the expert in ring slot `slot`: gathers pairs
        base .. base + min(rem, tm) - 1 of the pair list into a tm-row buffer,
        runs gate/up, the activation and down on it, and adds each real row's
        output times its routing weight into acc_ref. `rem` is how many of the
        expert's pairs are left from `base`; rows past it repeat the last pair
        and are not added."""
        for j in range(tm):
            tj = tok_ref[base + jnp.minimum(j, rem - 1)]
            xg_ref[pl.ds(j, 1), :] = xq_ref[pl.ds(tj, 1), :]
            for b in range(nq):
                xsg_ref[b, pl.ds(j, 1), :] = xs_ref[b, pl.ds(tj, 1), :]
        hmid = _gate_up(
            xg_ref[0:tm, :],
            [xsg_ref[b, 0:tm, :] for b in range(nq)],
            w1_buf.at[slot],
            s1_buf[slot],
            limit=limit,
        )
        y = _down(hmid, w2_buf.at[slot], s2_buf[slot]).astype(jnp.float32)
        for j in range(tm):

            @pl.when(j < rem)
            def _(j=j):
                p = base + j
                tj = tok_ref[p]
                acc_ref[pl.ds(tj, 1), :] += y[j : j + 1, :] * wt_ref[p]

    def ring_loop(lo: jax.Array, hi: jax.Array, tm: int, multi: bool) -> None:
        """Runs hit experts lo..hi-1, all of one tile size tm. Per expert: start
        the copies of the expert slots - 1 ahead, wait for this expert's copies,
        then run its row tiles (several when `multi`, else exactly one)."""

        def body(i, carry):
            slot = i % slots

            @pl.when(i + slots - 1 < n_u)
            def _():
                nxt = i + slots - 1
                for cp in weight_copies(nxt, nxt % slots):
                    cp.start()

            for cp in weight_copies(i, slot):
                cp.wait()
            e = active_ref[i]
            off, cnt = offsets_ref[e], counts_ref[e]
            if multi:

                def more(r, c2):
                    run_expert_tile(slot, off + r * tm, cnt - r * tm, tm)
                    return c2

                jax.lax.fori_loop(0, pl.cdiv(cnt, tm), more, 0)
            else:
                run_expert_tile(slot, off, cnt, tm)
            return carry

        jax.lax.fori_loop(lo, hi, body, 0)

    lo = jnp.int32(0)
    for k, tm in enumerate(tile_rows):
        ring_loop(lo, ends[k], tm, multi=k == len(tile_rows) - 1)
        lo = ends[k]
    o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def moe_ring(
    x: jax.Array,
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array,
    w2: jax.Array,
    w2_scale: jax.Array,
    clamp_limit: float | None,
    *,
    slots: int = _SLOTS,
    w1_chunks: int = _W1_CHUNKS,
    w2_chunks: int = _W2_CHUNKS,
    tile_rows: tuple[int, ...] = _TILE_ROWS,
) -> jax.Array:
    """The routed-expert output of one MoE call: sum over each token's k experts
    of weight * expert(x).

    Args:
      x: bf16 [T, H]. Input rows.
      topk_ids: int [T, k]. Experts of each token; ids outside [0, E) add nothing.
      topk_weights: [T, k]. Their routing weights.
      w1: fp8 [E, H, 2n]. Gate/up weights.
      w1_scale: f32 [E, 1, 1, 2n]. Their per-column scales.
      w2: fp8 [E, n, H]. Down weights.
      w2_scale: f32 [E, 1, 1, H]. Their per-column scales.
      clamp_limit: SiLU clamp limit, or None.
      slots: weight ring depth; slots - 1 experts are copied ahead.
      w1_chunks: DMAs per expert's w1 (split along H).
      w2_chunks: DMAs per expert's w2 (split along n).
      tile_rows: row-tile sizes, see _TILE_ROWS.

    Returns:
      [T, H], x's dtype.
    """
    tokens, hidden = x.shape
    num_experts, _, two_n = w1.shape
    n = two_n // 2
    k = topk_ids.shape[1]
    assert slots >= 2 and hidden % w1_chunks == 0 and n % w2_chunks == 0
    assert list(tile_rows) == sorted(set(tile_rows)) and min(tile_rows) >= 1
    info = pltpu.get_tpu_info()
    counts, offsets, tok, wt = _route(topk_ids, topk_weights, num_experts)
    s1 = w1_scale.astype(jnp.float32)
    s2 = w2_scale.astype(jnp.float32)
    nq = hidden // _Q_BLOCK
    lanes = info.num_lanes
    limit = _vmem_limit(tokens, hidden, n, slots, tile_rows, info)
    assert limit is not None, "moe_ring does not fit VMEM; check fit_moe_ring first"
    nsem = w1_chunks + w2_chunks + 2
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            limit=clamp_limit,
            slots=slots,
            w1_chunks=w1_chunks,
            w2_chunks=w2_chunks,
            tile_rows=tuple(tile_rows),
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[smem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((slots, hidden, two_n), jnp.float8_e4m3fn),
                pltpu.VMEM((slots, 1, two_n), jnp.float32),
                pltpu.VMEM((slots, n, hidden), jnp.float8_e4m3fn),
                pltpu.VMEM((slots, 1, hidden), jnp.float32),
                pltpu.SemaphoreType.DMA((slots, nsem)),
                pltpu.VMEM((tokens, hidden), jnp.float32),
                pltpu.VMEM((nq, tokens, lanes), jnp.float32),
                pltpu.VMEM((tokens, hidden), jnp.float32),
                pltpu.VMEM((max(tile_rows), hidden), jnp.float32),
                pltpu.VMEM((nq, max(tile_rows), lanes), jnp.float32),
                # One past E: a compaction pass after all E hits stores a miss there.
                pltpu.SMEM((num_experts + 1,), jnp.int32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=limit),
        name=f"moe_ring_{tokens}x{k}_{num_experts}x{hidden}x{two_n}_s{slots}",
    )(counts, offsets, tok, wt, x, w1, s1, w2, s2)


# Cost model: TensorCore cycles per call on tpu7x, for this kernel and for the
# gmm_v2 path it replaces, from static shapes. Weight sizes are counted in VMEM
# tiles (the weights are whole tiles: fit_moe_ring requires lane-aligned
# widths). The constants are fitted to timed calls over 11 weight shapes
# (hidden 2048-8192, expert width 128-768, 1-512 tokens, routing spread over all
# experts and over 16), converted to cycles at the 1.9 GHz clock XLA's DMA
# latency model assumes. The fit reproduces the timings to 8% rms (21% worst)
# for the ring and 10% rms (24% worst) for gmm_v2. The per-token terms count
# rows, not row tiles: rounding the rows up to whole 16-row tiles fitted worse.


def _expert_weight_tiles(hidden: int, n: int) -> float:
    """VMEM tiles of one expert's fp8 weights on this rank: w1 [hidden, 2n] +
    w2 [n, hidden]."""
    return 3 * hidden * n / _VMEM_TILE_BYTES


def _rows_per_expert_pmf(tokens: int, topk: int, num_experts: int) -> list[float]:
    """P(an expert is routed c of the tokens), c = 0..tokens, when every token
    picks topk distinct experts uniformly: what the trace-time choice assumes,
    since the real routing is only known at run time."""
    p = min(1.0, topk / num_experts)
    return [
        math.comb(tokens, c) * p**c * (1.0 - p) ** (tokens - c)
        for c in range(tokens + 1)
    ]


def ring_cost_cycles(
    tokens: int,
    topk: int,
    num_experts: int,
    hidden: int,
    n: int,
    tile_rows: tuple[int, ...] = _TILE_ROWS,
) -> float:
    """Modelled cycles of one moe_ring call. Each hit expert costs the longer of
    its weight copy and its row tiles, which overlap."""
    fixed_cycles = 14_140
    cycles_per_element = 0.0813  # per token per hidden column: load, quantize, store
    cycles_per_pair_sq = 0.0038  # _route compares all (token, expert) pairs pairwise
    expert_cycles = 51  # per distinct expert
    copy_tiles_per_cycle = 0.449  # weights into VMEM, slots - 1 experts in flight
    row_tile_cycles = 988  # one row tile through one expert ...
    row_tile_cycles_per_row_weight_tile = 0.049  # ... plus per row per weight tile

    weight_tiles = _expert_weight_tiles(hidden, n)
    copy = weight_tiles / copy_tiles_per_cycle
    per_expert = 0.0
    for c, prob in enumerate(_rows_per_expert_pmf(tokens, topk, num_experts)):
        if c == 0:
            continue
        rows = next((r for r in tile_rows if c <= r), tile_rows[-1])
        row_tiles = pl.cdiv(c, rows)
        one_tile = (
            row_tile_cycles + row_tile_cycles_per_row_weight_tile * rows * weight_tiles
        )
        per_expert += prob * (expert_cycles + max(copy, row_tiles * one_tile))
    pairs = tokens * topk
    return (
        fixed_cycles
        + cycles_per_element * tokens * hidden
        + cycles_per_pair_sq * pairs * pairs
        + num_experts * per_expert
    )


def gmm_cost_cycles(
    tokens: int, topk: int, num_experts: int, hidden: int, n: int
) -> float:
    """Modelled cycles of the gmm_v2 path for the same call."""
    fixed_cycles = 29_640
    expert_cycles = 1_786  # per distinct expert
    cycles_per_weight_tile = 1.87  # per VMEM tile of a distinct expert's weights
    cycles_per_element = 0.1545  # per token per hidden column
    cycles_per_element_width = 8.52e-6  # ... plus per column of expert width

    hit = num_experts * (1.0 - _rows_per_expert_pmf(tokens, topk, num_experts)[0])
    return (
        fixed_cycles
        + hit
        * (expert_cycles + cycles_per_weight_tile * _expert_weight_tiles(hidden, n))
        + tokens * hidden * (cycles_per_element + cycles_per_element_width * n)
    )


def prefers_ring(tokens: int, topk: int, num_experts: int, hidden: int, n: int) -> bool:
    """Whether the model puts this call's moe_ring cycles under its gmm_v2 cycles."""
    return ring_cost_cycles(tokens, topk, num_experts, hidden, n) < gmm_cost_cycles(
        tokens, topk, num_experts, hidden, n
    )


def maybe_moe_ring(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    *,
    activation_name: str,
    no_combine: bool,
    use_group_offset: bool,
    w_quant_bits: int,
) -> jax.Array | None:
    """The ring kernel's output for a fused-MoE call it covers, else None.

    Covers bf16 tokens with every expert local, unbiased FP8 experts with
    per-channel scales and the clamped SiLU, where the kernel fits VMEM
    (fit_moe_ring) and the cost model puts it under gmm_v2 (prefers_ring)."""
    if not activation_name.startswith("silu_and_mul_clamp:"):
        return None
    clamp_limit = float(activation_name.split(":", 1)[1])
    tokens, hidden = hidden_states.shape
    num_experts, w1_k, two_n = w1.shape
    if (
        no_combine
        or use_group_offset
        or w_quant_bits
        or w1_bias is not None
        or w2_bias is not None
        or w1_scale is None
        or w2_scale is None
        or hidden_states.dtype != jnp.bfloat16
        or w1.dtype != jnp.float8_e4m3fn
        or w2.dtype != jnp.float8_e4m3fn
        or w1_k != hidden
        or w2.shape != (num_experts, two_n // 2, hidden)
        or w1_scale.shape != (num_experts, 1, 1, two_n)
        or w2_scale.shape != (num_experts, 1, 1, hidden)
        or not fit_moe_ring(tokens, hidden, two_n)
        or not prefers_ring(tokens, topk_ids.shape[1], num_experts, hidden, two_n // 2)
    ):
        return None
    return moe_ring(
        hidden_states,
        topk_ids,
        topk_weights,
        w1,
        w1_scale,
        w2,
        w2_scale,
        clamp_limit,
    )
