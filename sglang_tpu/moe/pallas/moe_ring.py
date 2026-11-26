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
from collections.abc import Sequence

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
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    num_experts: int,
    tile_rows: tuple[int, ...] = _TILE_ROWS,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Groups the (token, expert) pairs by expert and lists the hit experts by
    tile size, in XLA, before the kernel.

    Every token has k pairs, one per expert it was routed to. The kernel wants
    them listed expert by expert, so that one expert's rows are contiguous, and
    it runs the hit experts one tile size at a time (see _TILE_ROWS).

    Args:
      topk_ids: int [T, k]. Expert id of each pair. Ids outside [0, E) are
        padding: they are listed after every real pair and never read.
      topk_weights: [T, k]. Routing weight of each pair.
      num_experts: E.
      tile_rows: row-tile sizes, see _TILE_ROWS.

    Returns:
      counts: int32 [E]. Number of pairs of each expert.
      offsets: int32 [E]. Where each expert's pairs start in the list.
      tok: int32 [T * k]. Token of each pair, in list order (by expert, then by
        the pair's original position).
      wt: f32 [T * k]. Routing weight of each pair, in list order.
      active: int32 [E]. Ids of the hit experts: first those whose pair count
        takes the smallest tile size, then the next size, and so on; by id
        within a size. Entries past the last hit expert are never read.
      ends: int32 [len(tile_rows)]. Where each tile size's experts end in
        `active`; the last entry is the number of hit experts.
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
    # Tile-size class of each expert: 0 = not hit, c = uses tile_rows[c - 1]
    # (counts above the largest size stay in the last class).
    lows = jnp.asarray((0,) + tuple(tile_rows[:-1]), jnp.int32)
    size_class = jnp.sum(counts[:, None] > lows[None, :], axis=1, dtype=jnp.int32)
    classes = jnp.arange(1, len(tile_rows) + 1, dtype=jnp.int32)
    ends = jnp.cumsum(
        jnp.sum(size_class[:, None] == classes[None, :], axis=0, dtype=jnp.int32)
    )
    # The hit experts in (class, id) order, by the same compare-and-select as
    # the pairs above; experts that were not hit sort last.
    expert = jnp.arange(num_experts, dtype=jnp.int32)
    order = jnp.where(size_class > 0, size_class, len(tile_rows) + 1) * num_experts
    order = order + expert
    place = jnp.sum(order[None, :] < order[:, None], axis=1, dtype=jnp.int32)
    active = jnp.sum(
        jnp.where(place[None, :] == expert[:, None], expert[None, :], 0),
        axis=1,
        dtype=jnp.int32,
    )
    return (
        counts.astype(jnp.int32),
        offsets.astype(jnp.int32),
        tok,
        wt,
        active,
        ends.astype(jnp.int32),
    )


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
    active_ref: jax.Array,  # SMEM int32 [E]. Ids of the hit experts, by tile size.
    ends_ref: jax.Array,  # SMEM int32 [len(tile_rows)]. Where each size ends in it.
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

    # active_ref lists the hit experts one tile size after another; ends[k] is
    # where size k's experts end, and the last one is the number of hit experts.
    ends = [ends_ref[k] for k in range(len(tile_rows))]
    n_u = ends[-1]

    # The first slots - 1 hit experts' copies start before the loop.
    for s in range(slots - 1):

        @pl.when(s < n_u)
        def _(s=s):
            for cp in weight_copies(s, s):
                cp.start()

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
    counts, offsets, tok, wt, active, ends = _route(
        topk_ids, topk_weights, num_experts, tuple(tile_rows)
    )
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
            num_scalar_prefetch=5,
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
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=limit),
        name=f"moe_ring_{tokens}x{k}_{num_experts}x{hidden}x{two_n}_s{slots}",
    )(counts, offsets, tok, active, ends, wt, x, w1, s1, w2, s2)


# Cost model: TensorCore cycles of one call on tpu7x, for this kernel and for the
# gmm_v2 path it replaces, as functions of the model's shape: E experts, H
# hidden, n expert width on this rank, T tokens, k experts per token. Only the
# difference between the two matters (prefers_ring).
#
# Nothing here is fitted to timings. There are two kinds of numbers:
#
#   * Compiled-code cycle counts. The compiler's schedule of each kernel (the
#     LLO dump) has one bundle of instructions per cycle, so the cycles of one
#     pass of a loop can be read off it; a term is that count times how often
#     the code runs the loop. The counts were read for 7 weight shapes (E 96 to
#     288, H 2048 to 8192, n 128 to 512), 2 to 256 tokens and k = 8 and 16; a
#     coefficient below is the least-squares line through those counts.
#   * DMA times, from the model XLA uses for this chip.
#
# The counts belong to this file's kernel, gmm_v2 and the compiler they were
# dumped from: re-derive them after changing _kernel, _route, _TILE_ROWS or
# _SLOTS, or when gmm_v2 or the compiler changes. PR #2468's description links
# the write-up with the dump commands, the scripts and the comparison with
# timed calls (the ring within 7%, gmm_v2 from 11% under to 33% over). Not
# covered by any dump: other chips, k other than 8 and 16, more than 288
# experts, H above 8192.

# XLA's DMA model for tpu7x (kGhostfish in dma_latency_model.cc): a copy takes
# a latency plus its bytes at a bandwidth.
_HBM_BYTES_PER_CYCLE = 1688
_HBM_TO_VMEM_LATENCY = 692
_VMEM_TO_HBM_LATENCY = 594
_SMEM_BYTES_PER_CYCLE = 32
_VMEM_TO_SMEM_LATENCY = 578
_HBM_TO_SMEM_LATENCY = 722
_STRIDED_BYTES_PER_CYCLE = 343  # dma.general, which gmm_v2 moves its row tiles with
_STRIDED_LATENCY = 2796

# moe_ring, cycles read from its LLO.
# One row tile through one expert (gather, gate/up, activation, down, add), by
# tile size: (constant, per H, per H * n).
_RING_TILE_CYCLES = {
    8: (933, 0.0498, 0.000722),
    16: (1235, 0.1165, 0.000726),
    32: (1703, 0.2699, 0.000621),
}
_RING_MULTI_TILE_EXPERT_CYCLES = 181  # per expert of the last size: its tile loop
_RING_ROW_ADD_CYCLES = (9, 0.00586)  # (constant, per H): skipped for a padding row
_RING_PROLOGUE_CYCLES = (514, 0.00149)  # (constant, per T * H): quantize x, zero acc
_RING_EPILOGUE_CYCLES = (171, 0.00035)  # (constant, per T * H): cast the output
# _route and the XLA code around the kernel: (constant, per E, per (T * k)^2).
# The last one is the pairwise compare; it is the largest dumped size's value
# and overstates smaller sizes (by 7k cycles at T * k = 2048).
_RING_ROUTE_CYCLES = (3703, 3.86, 0.00563)
_RING_SMEM_COPIES = 6  # counts, offsets, tok, active, ends, wt

# gmm_v2, cycles read from its LLO. It runs two kernels (gate/up, then down);
# each first scans all E groups, then runs one step per (expert, row tile).
_GMM_SCAN_CYCLES = (37, 17)  # (per expert, per step), in each kernel
_GMM_KERNEL_CYCLES = (150, 135)  # straight-line code of the gate/up and down kernels
# One step: (constant, per H * n, per tile row, per H * tile row).
_GMM_GATE_UP_STEP_CYCLES = (578, 0.000544, 2.27, 0.000198)
_GMM_DOWN_STEP_CYCLES = (687, 0.000124, -0.21, 0.00111)
# A step starts the next step's weight copy 110 cycles in and waits for its own
# 240 cycles in, so a copy has one step plus this many cycles.
_GMM_WEIGHT_COPY_LEAD_CYCLES = 130
_GMM_MAX_TILE_ROWS = 128
# The XLA code around the two kernels (sort, gather, unsort): a constant, plus
# per T * k * H up to 256 pairs and from 512 pairs, where it switches to a
# gather over all rows. The weakest numbers here: those fusions have loops
# whose trip counts depend on the data, so their static count is rough.
_GMM_XLA_CYCLES = 3387
_GMM_XLA_CYCLES_PER_PAIR_H = 0.0016
_GMM_XLA_GATHER_PAIRS = 512
_GMM_XLA_GATHER_CYCLES_PER_PAIR_H = 0.00677


def _expected_experts_with_rows(
    tokens: int, topk: int, num_experts: int
) -> list[float]:
    """Expected number of experts routed c of the tokens, c = 0..tokens, when
    every token picks topk distinct experts uniformly: what the trace-time
    choice assumes, since the real routing is only known at run time."""
    p = min(1.0, topk / num_experts)
    return [
        num_experts * math.comb(tokens, c) * p**c * (1.0 - p) ** (tokens - c)
        for c in range(tokens + 1)
    ]


def _ring_cycles(
    experts_with_rows: Sequence[float],
    tokens: int,
    topk: int,
    num_experts: int,
    hidden: int,
    n: int,
) -> float:
    """Cycles of one moe_ring call in which experts_with_rows[c] experts have c
    rows each."""
    pairs = tokens * topk
    # XLA before the kernel: _route, while both scale arrays and x are copied
    # into VMEM.
    route = (
        _RING_ROUTE_CYCLES[0]
        + _RING_ROUTE_CYCLES[1] * num_experts
        + _RING_ROUTE_CYCLES[2] * pairs * pairs
    )
    scale_bytes = 4 * num_experts * (2 * n + hidden) + 2 * tokens * hidden
    scales = _HBM_TO_VMEM_LATENCY + scale_bytes / _HBM_BYTES_PER_CYCLE
    # The routing arrays go to SMEM, counted one copy after another.
    smem = (
        _RING_SMEM_COPIES * _VMEM_TO_SMEM_LATENCY
        + 4 * (3 * num_experts + 2 * pairs) / _SMEM_BYTES_PER_CYCLE
    )
    prologue = _RING_PROLOGUE_CYCLES[0] + _RING_PROLOGUE_CYCLES[1] * tokens * hidden
    # One expert's weights. The copies in flight share the bandwidth, so in the
    # loop an expert costs the longer of its copy and its row tiles; the first
    # expert's copy also pays the latency, less the prologue it overlaps.
    copy = 3 * hidden * n / _HBM_BYTES_PER_CYCLE
    first_copy = max(0.0, _HBM_TO_VMEM_LATENCY + copy - prologue)
    row_add = _RING_ROW_ADD_CYCLES[0] + _RING_ROW_ADD_CYCLES[1] * hidden
    experts = 0.0
    for rows_routed, count in enumerate(experts_with_rows):
        if rows_routed == 0 or count == 0:
            continue
        rows = next((r for r in _TILE_ROWS if rows_routed <= r), _TILE_ROWS[-1])
        tiles = pl.cdiv(rows_routed, rows)
        const, per_h, per_hn = _RING_TILE_CYCLES[rows]
        run = tiles * (const + per_h * hidden + per_hn * hidden * n)
        run -= (tiles * rows - rows_routed) * row_add
        if rows == _TILE_ROWS[-1]:
            run += _RING_MULTI_TILE_EXPERT_CYCLES
        experts += count * max(run, copy)
    output = (
        _RING_EPILOGUE_CYCLES[0]
        + _RING_EPILOGUE_CYCLES[1] * tokens * hidden
        + _VMEM_TO_HBM_LATENCY
        + 2 * tokens * hidden / _HBM_BYTES_PER_CYCLE
    )
    return max(route, scales) + smem + prologue + first_copy + experts + output


def _gmm_cycles(
    experts_hit: float, tokens: int, topk: int, num_experts: int, hidden: int, n: int
) -> float:
    """Cycles of the gmm_v2 path for one call that hits experts_hit experts."""
    pairs = tokens * topk
    # gmm_v2's row tile: the pairs rounded up to a power of two, 16 to 128 rows.
    tile_rows = min(_GMM_MAX_TILE_ROWS, max(16, 1 << (pairs - 1).bit_length()))
    tiles = pl.cdiv(pairs, tile_rows)
    steps = experts_hit + tiles - 1
    total = _GMM_XLA_CYCLES + pairs * hidden * (
        _GMM_XLA_CYCLES_PER_PAIR_H
        if pairs < _GMM_XLA_GATHER_PAIRS
        else _GMM_XLA_GATHER_CYCLES_PER_PAIR_H
    )
    # Per kernel: its step cycles, its fixed code, the bytes of one step's
    # weights, and of one row tile in and out.
    kernels = (
        (
            _GMM_GATE_UP_STEP_CYCLES,
            _GMM_KERNEL_CYCLES[0],
            2 * hidden * n,
            2 * tile_rows * hidden,
            2 * tile_rows * n,
        ),
        (
            _GMM_DOWN_STEP_CYCLES,
            _GMM_KERNEL_CYCLES[1],
            hidden * n,
            2 * tile_rows * n,
            2 * tile_rows * hidden,
        ),
    )
    for step_cycles, fixed, weight_bytes, in_bytes, out_bytes in kernels:
        const, per_hn, per_row, per_h_row = step_cycles
        step = (
            const
            + per_hn * hidden * n
            + per_row * tile_rows
            + per_h_row * hidden * tile_rows
        )
        scan = (
            _HBM_TO_SMEM_LATENCY
            + 4 * num_experts / _SMEM_BYTES_PER_CYCLE
            + _GMM_SCAN_CYCLES[0] * num_experts
            + _GMM_SCAN_CYCLES[1] * steps
        )
        # The next step's weights are copied during the current step.
        weight_copy = _HBM_TO_VMEM_LATENCY + weight_bytes / _HBM_BYTES_PER_CYCLE
        weight_wait = weight_copy + steps * max(
            0.0, weight_copy - step - _GMM_WEIGHT_COPY_LEAD_CYCLES
        )
        # The first row tile in and the last one out are waited for in full. A
        # later tile comes in during the step before it; a tile goes out while
        # the next tile's steps run, which is not counted.
        row_in = _STRIDED_LATENCY + in_bytes / _STRIDED_BYTES_PER_CYCLE
        row_out = _STRIDED_LATENCY + out_bytes / _STRIDED_BYTES_PER_CYCLE
        row_wait = row_in + row_out + (tiles - 1) * max(0.0, row_in - step)
        total += fixed + scan + steps * step + weight_wait + row_wait
    return total


def ring_cost_cycles(
    tokens: int, topk: int, num_experts: int, hidden: int, n: int
) -> float:
    """Modelled cycles of one moe_ring call under uniform routing."""
    return _ring_cycles(
        _expected_experts_with_rows(tokens, topk, num_experts),
        tokens,
        topk,
        num_experts,
        hidden,
        n,
    )


def gmm_cost_cycles(
    tokens: int, topk: int, num_experts: int, hidden: int, n: int
) -> float:
    """Modelled cycles of the gmm_v2 path for the same call."""
    missed = _expected_experts_with_rows(tokens, topk, num_experts)[0]
    return _gmm_cycles(num_experts - missed, tokens, topk, num_experts, hidden, n)


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
