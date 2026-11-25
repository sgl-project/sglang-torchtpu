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
# Rows per compute tile by expert pair count: up to 8 pairs one 8-row tile, up
# to 16 one 16-row tile, more in 32-row tiles.
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


def _route(topk_ids, topk_weights, num_experts):
    """Expert-major pairs: per expert its pair count and first pair, and per
    pair (sorted by expert, then slot) its token and f32 routing weight. Ids
    outside [0, E) are never referenced."""
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


def _gate_up(xq, xs, w_ref, s_row, *, limit):
    """gmm_v2's fused GMM1 on pre-quantized rows: per 512-column block an fp8
    dot rounded to bf16, scaled by the x block scale then the weight scale and
    summed in bf16; then fused_moe's clamped SiLU. xq [M, H] f32 (fp8 values),
    xs list of [M, lanes] f32 block scales, w_ref [H, 2n] fp8. Returns [M, n] bf16."""
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


def _down(h, w_ref, s_row):
    """gmm_v2's GMM2 for K <= 512: one quantization block. h [M, n] bf16,
    w_ref [n, H] fp8. Returns [M, H] bf16."""
    sdt = jnp.float32 if _FP32_MOE_SCALES else jnp.bfloat16
    scale = jnp.max(jnp.abs(h), axis=1, keepdims=True) / _FP8_MAX
    inv = jnp.where(scale == 0, 0, 1 / scale)
    part = jax.lax.dot_general(
        (h * inv).astype(jnp.float8_e4m3fn),
        w_ref[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    y = (part.astype(sdt) * scale.astype(sdt) * s_row.astype(sdt)).astype(jnp.bfloat16)
    return jnp.zeros_like(y) + y


def _kernel(
    counts_ref: jax.Array,
    offsets_ref: jax.Array,
    tok_ref: jax.Array,
    wt_ref: jax.Array,
    x_ref: jax.Array,
    w1_hbm: jax.Array,
    s1_hbm: jax.Array,
    w2_hbm: jax.Array,
    s2_hbm: jax.Array,
    o_ref: jax.Array,
    w1_buf: jax.Array,
    s1_buf: jax.Array,
    w2_buf: jax.Array,
    s2_buf: jax.Array,
    sems: jax.Array,
    xq_ref: jax.Array,
    xs_ref: jax.Array,
    acc_ref: jax.Array,
    xg_ref: jax.Array,
    xsg_ref: jax.Array,
    active_ref: jax.Array,
    *,
    limit: float | None,
    slots: int,
    w1_chunks: int,
    w2_chunks: int,
    tile_rows: tuple[int, ...],
):
    """Kernel body. T tokens, k experts per token, E experts, H model width,
    n expert width, nq = H / 512 quantization blocks, R = max(tile_rows).

    Args:
      counts_ref: SMEM int32 [E]. Rows routed to each expert.
      offsets_ref: SMEM int32 [E]. Each expert's first entry in the pair list.
      tok_ref: SMEM int32 [T * k]. Token of each pair; pairs grouped by expert.
      wt_ref: SMEM f32 [T * k]. Routing weight of each pair.
      x_ref: VMEM bf16 [T, H]. Input rows.
      w1_hbm: HBM fp8 [E, H, 2n]. Gate/up weights.
      s1_hbm: HBM f32 [E, 1, 1, 2n]. Their per-column scales.
      w2_hbm: HBM fp8 [E, n, H]. Down weights.
      s2_hbm: HBM f32 [E, 1, 1, H]. Their per-column scales.
      o_ref: VMEM [T, H], x's dtype. Output rows.
      w1_buf: VMEM fp8 [slots, H, 2n]. Weight ring: w1 of the experts in flight.
      s1_buf: VMEM f32 [slots, 1, 2n].
      w2_buf: VMEM fp8 [slots, n, H].
      s2_buf: VMEM f32 [slots, 1, H].
      sems: DMA semaphores [slots, w1_chunks + w2_chunks + 2], one per copy.
      xq_ref: VMEM f32 [T, H]. x quantized to fp8 values.
      xs_ref: VMEM f32 [nq, T, lanes]. x's scale per row per block.
      acc_ref: VMEM f32 [T, H]. Sum of weight * expert output.
      xg_ref: VMEM f32 [R, H]. One tile's rows gathered from xq_ref.
      xsg_ref: VMEM f32 [nq, R, lanes]. Those rows' block scales.
      active_ref: SMEM int32 [E + 1]. Ids of the hit experts, by tile class.
      limit: SiLU clamp limit, or None.
      slots: weight ring depth.
      w1_chunks: DMAs per expert's w1.
      w2_chunks: DMAs per expert's w2.
      tile_rows: row-tile sizes by expert row count.
    """
    t, h = x_ref.shape
    num_experts, _, two_n = w1_hbm.shape
    n = two_n // 2
    nq = h // _Q_BLOCK
    lanes = xs_ref.shape[-1]
    h1 = h // w1_chunks
    n2 = n // w2_chunks

    def copies(i, slot):
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

    def start(lo, hi):
        for s in range(slots - 1):

            @pl.when((s >= lo) & (s < hi))
            def _(s=s):
                for cp in copies(s, s):
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
            start(0, count)
    n_u = count
    start(ends[0], n_u)

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

    def tile(slot, base, rem, tm):
        """Rows [base, base + min(rem, tm)) of the pair list through one expert."""
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

    def ring_loop(lo, hi, tm, multi):
        def body(i, carry):
            slot = i % slots

            @pl.when(i + slots - 1 < n_u)
            def _():
                nxt = i + slots - 1
                for cp in copies(nxt, nxt % slots):
                    cp.start()

            for cp in copies(i, slot):
                cp.wait()
            e = active_ref[i]
            off, cnt = offsets_ref[e], counts_ref[e]
            if multi:

                def more(r, c2):
                    tile(slot, off + r * tm, cnt - r * tm, tm)
                    return c2

                jax.lax.fori_loop(0, pl.cdiv(cnt, tm), more, 0)
            else:
                tile(slot, off, cnt, tm)
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
    """x [T, H] bf16; ids / weights [T, k]; w1 [E, H, 2n] / w2 [E, n, H] FP8
    with per-channel f32 scales [E, 1, 1, 2n] / [E, 1, 1, H]. Returns [T, H].
    Ids outside [0, E) contribute nothing."""
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
# gmm_v2 path it replaces, from static shapes. The constants are fitted to timed
# calls over 11 weight shapes (hidden 2048-8192, expert width 128-768, 1-512
# tokens, routing spread over all experts and over 16), converted to cycles at
# the 1.9 GHz clock XLA's DMA latency model assumes. The fit reproduces the
# timings to 8% rms (21% worst) for the ring and 10% rms (24% worst) for gmm_v2.
_RING_FIXED_CYCLES = 14_140
_RING_CYCLES_PER_ELEMENT = 0.0813  # per token per hidden column: load, quantize, store
_RING_CYCLES_PER_PAIR_SQ = 0.0038  # routing compares all (token, expert) pairs pairwise
_RING_EXPERT_CYCLES = 51  # per distinct expert
_RING_COPY_BYTES_PER_CYCLE = 1_838  # weights into VMEM, slots - 1 experts in flight
_RING_TILE_CYCLES = 988  # one row tile through one expert ...
_RING_TILE_CYCLES_PER_ROW_BYTE = 1.196e-5  # ... plus per row per weight byte
_GMM_FIXED_CYCLES = 29_640
_GMM_EXPERT_CYCLES = 1_786  # per distinct expert
_GMM_CYCLES_PER_BYTE = 4.566e-4  # per weight byte of a distinct expert
_GMM_CYCLES_PER_ELEMENT = 0.1545  # per token per hidden column
_GMM_CYCLES_PER_ELEMENT_WIDTH = 8.52e-6  # ... plus per column of expert width


def _expert_weight_bytes(hidden: int, n: int) -> int:
    """FP8 bytes of one expert on this rank: w1 [hidden, 2n] + w2 [n, hidden]."""
    return 3 * hidden * n


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
    weight_bytes = _expert_weight_bytes(hidden, n)
    copy = weight_bytes / _RING_COPY_BYTES_PER_CYCLE
    per_expert = 0.0
    for c, prob in enumerate(_rows_per_expert_pmf(tokens, topk, num_experts)):
        if c == 0:
            continue
        rows = next((r for r in tile_rows if c <= r), tile_rows[-1])
        tiles = pl.cdiv(c, rows)
        tile = _RING_TILE_CYCLES + _RING_TILE_CYCLES_PER_ROW_BYTE * rows * weight_bytes
        per_expert += prob * (_RING_EXPERT_CYCLES + max(copy, tiles * tile))
    pairs = tokens * topk
    return (
        _RING_FIXED_CYCLES
        + _RING_CYCLES_PER_ELEMENT * tokens * hidden
        + _RING_CYCLES_PER_PAIR_SQ * pairs * pairs
        + num_experts * per_expert
    )


def gmm_cost_cycles(
    tokens: int, topk: int, num_experts: int, hidden: int, n: int
) -> float:
    """Modelled cycles of the gmm_v2 path for the same call."""
    hit = num_experts * (1.0 - _rows_per_expert_pmf(tokens, topk, num_experts)[0])
    return (
        _GMM_FIXED_CYCLES
        + hit
        * (_GMM_EXPERT_CYCLES + _GMM_CYCLES_PER_BYTE * _expert_weight_bytes(hidden, n))
        + tokens
        * hidden
        * (_GMM_CYCLES_PER_ELEMENT + _GMM_CYCLES_PER_ELEMENT_WIDTH * n)
    )


def prefers_ring(tokens: int, topk: int, num_experts: int, hidden: int, n: int) -> bool:
    """Whether the model puts this call's moe_ring cycles under its gmm_v2 cycles."""
    return ring_cost_cycles(tokens, topk, num_experts, hidden, n) < gmm_cost_cycles(
        tokens, topk, num_experts, hidden, n
    )


def maybe_moe_ring(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    w1_bias,
    w2_bias,
    *,
    activation_name,
    no_combine,
    use_group_offset,
    w_quant_bits,
):
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
