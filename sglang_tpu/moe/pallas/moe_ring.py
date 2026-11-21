# SPDX-License-Identifier: Apache-2.0
"""Decode fused MoE that streams each distinct expert's weights once through a
multi-slot VMEM ring.

At decode row counts with every expert local (EP = 1) the cost is the weight
bytes of the distinct experts the batch touches. gmm_v2 fetches them one tile
DMA at a time (about one stream of HBM bandwidth); here the kernel keeps
``slots - 1`` experts in flight, each split into several DMAs, and computes
each expert for its routed rows only:

  * XLA builds an expert-major pair list (one sort of the routed slots) and
    per-expert counts and offsets;
  * the kernel compacts the hit experts, starts their weight DMAs, quantizes
    x once as gmm_v2 quantizes its lhs (per row per 512 columns, bf16 scales);
  * per expert: gather its rows (an 8-, 16- or 32-row tile by its pair count,
    one loop per count class), gate/up, clamped SiLU and down exactly as
    gmm_v2 + fused_moe compute them, then add weight * output into an f32
    [T, H] accumulator.

Each (token, expert) output matches gmm_v2's bit for bit; the f32 combine
sums in expert order rather than top-k order.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.gmm_v2 import _FP32_MOE_SCALES

FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)
# gmm_v2's lhs quantization block.
_Q_BLOCK = 512
# Ring depth: slots - 1 experts in flight while one computes.
_SLOTS = 4
# DMAs per expert's w1 / w2 (split along their leading weight dim).
_W1_CHUNKS = 2
_W2_CHUNKS = 1
# Rows per compute tile by expert pair count: up to 8 pairs one 8-row tile, up
# to 16 one 16-row tile, more in 32-row tiles (measured per-tile cost 0.9 / 1.1 / 1.4 us).
_TILE_ROWS = (8, 16, 32)
_VMEM_HEADROOM = 1.25
_VMEM_FRACTION = 0.85


def _tiled_bytes(shape, dtype, info):
    itemsize = jnp.dtype(dtype).itemsize
    sublanes = info.num_sublanes * max(1, 4 // itemsize)
    *lead, rows, cols = (1, *shape) if len(shape) == 1 else shape
    rows = pl.cdiv(rows, sublanes) * sublanes
    cols = pl.cdiv(cols, info.num_lanes) * info.num_lanes
    out = rows * cols * itemsize
    for d in lead:
        out *= d
    return out


def _vmem_need(t, h, n, slots, tile_rows, info):
    b = functools.partial(_tiled_bytes, info=info)
    nq = h // _Q_BLOCK
    lanes = info.num_lanes
    ring = slots * (
        b((h, 2 * n), FP8)
        + b((n, h), FP8)
        + b((1, 2 * n), jnp.float32)
        + b((1, h), jnp.float32)
    )
    rows = (
        2 * b((t, h), jnp.bfloat16)  # x in, out
        + 2 * b((t, h), jnp.float32)  # quantized x, accumulator
        + b((nq, t, lanes), jnp.float32)  # x block scales
        + b((max(tile_rows), h), jnp.float32)
        + b((nq, max(tile_rows), lanes), jnp.float32)
    )
    return ring + rows


def supports(tokens, hidden, two_n, *, slots=_SLOTS, tile_rows=_TILE_ROWS) -> bool:
    info = pltpu.get_tpu_info()
    n = two_n // 2
    if hidden % _Q_BLOCK or n % info.num_lanes or tokens < 1:
        return False
    need = _vmem_need(tokens, hidden, n, slots, tile_rows, info)
    return need * _VMEM_HEADROOM <= info.vmem_capacity_bytes * _VMEM_FRACTION


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
    shift = max(1, (p - 1).bit_length())
    srt = jnp.sort((key << shift) | jnp.arange(p, dtype=jnp.int32))
    slot = srt & ((1 << shift) - 1)
    tok = slot // k
    wt = topk_weights.reshape(p).astype(jnp.float32)[slot]
    return (
        counts.astype(jnp.int32),
        offsets.astype(jnp.int32),
        tok.astype(jnp.int32),
        wt,
    )


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
            xq[:, b * _Q_BLOCK : (b + 1) * _Q_BLOCK].astype(FP8),
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
    scale = jnp.max(jnp.abs(h), axis=1, keepdims=True) / FP8_MAX
    inv = jnp.where(scale == 0, 0, 1 / scale)
    part = jax.lax.dot_general(
        (h * inv).astype(FP8),
        w_ref[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    y = (part.astype(sdt) * scale.astype(sdt) * s_row.astype(sdt)).astype(jnp.bfloat16)
    return jnp.zeros_like(y) + y


def _kernel(
    counts_ref,
    offsets_ref,
    tok_ref,
    wt_ref,
    x_ref,
    w1_hbm,
    s1_hbm,
    w2_hbm,
    s2_hbm,
    o_ref,
    w1_buf,
    s1_buf,
    w2_buf,
    s2_buf,
    sems,
    xq_ref,
    xs_ref,
    acc_ref,
    xg_ref,
    xsg_ref,
    active_ref,
    *,
    limit,
    slots,
    w1_chunks,
    w2_chunks,
    tile_rows,
):
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
        scale = jnp.max(jnp.abs(blk), axis=1, keepdims=True) / FP8_MAX
        inv = jnp.where(scale == 0, 0, 1 / scale)
        xq_ref[:, b * _Q_BLOCK : (b + 1) * _Q_BLOCK] = (
            (blk * inv).astype(FP8).astype(jnp.float32)
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
    need = _vmem_need(tokens, hidden, n, slots, tile_rows, info)
    limit = min(int(need * _VMEM_HEADROOM) + (8 << 20), info.vmem_capacity_bytes)
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
                pltpu.VMEM((slots, hidden, two_n), FP8),
                pltpu.VMEM((slots, 1, two_n), jnp.float32),
                pltpu.VMEM((slots, n, hidden), FP8),
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
    min_tokens,
    max_tokens,
):
    """The ring kernel's output for a fused-MoE call it covers, else None.

    Covers min_tokens..max_tokens bf16 tokens with every expert local, unbiased
    FP8 experts with per-channel scales and the clamped SiLU."""
    if not activation_name.startswith("silu_and_mul_clamp:"):
        return None
    clamp_limit = float(activation_name.split(":", 1)[1])
    tokens, hidden = hidden_states.shape
    num_experts, w1_k, two_n = w1.shape
    if (
        not min_tokens <= tokens <= max_tokens
        or no_combine
        or use_group_offset
        or w_quant_bits
        or w1_bias is not None
        or w2_bias is not None
        or w1_scale is None
        or w2_scale is None
        or hidden_states.dtype != jnp.bfloat16
        or w1.dtype != FP8
        or w2.dtype != FP8
        or w1_k != hidden
        or w2.shape != (num_experts, two_n // 2, hidden)
        or w1_scale.shape != (num_experts, 1, 1, two_n)
        or w2_scale.shape != (num_experts, 1, 1, hidden)
        or not supports(tokens, hidden, two_n)
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
