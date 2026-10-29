# SPDX-License-Identifier: Apache-2.0
"""Band-skip variant of the dense vision flash attention (single-step path).

The dense kernel (``flash_attention.py``, ``block_k == kv_len``) computes, for every
query block, scores against the whole bucket, masks them with ``segment_ids``, and
normalises over the full width. NaViT's masks are block-diagonal (one segment per
image/clip on the full-attention layers, one per 32x32 tile on the sparse layers,
pad tokens in their own segment), so most key positions of a query block are masked
and contribute exactly ``0.0`` to the softmax: ``s + MASK`` rounds to ``MASK`` and
``exp(MASK - m)`` underflows, because every row also sees its own key.

This kernel finds, per query block, a band of key chunks that contains every key
sharing a segment with the block (computed in the kernel from the segment-id blocks
the dense kernel already receives; ``band_tables`` is the reference) and only
computes inside it. It keeps the dense kernel's bits:

* scores, scale, mask, row max, exp and the divide are the same per-element ops;
* the row sum adds the 128-lane slices of ``p`` in key order and then reduces the
  lanes, which is how the dense kernel's full-width sum is evaluated; masked
  slices only add exact zeros there;
* the PV product accumulates 256-key MXU passes in key order, as the dense
  kernel's single dot does; masked passes only add exact zeros there.

Query blocks whose band covers most of the bucket run the dense kernel's own body.
Outputs (and ``l`` / ``m``) were measured byte-identical to the dense kernel on the
served NaViT layouts; shapes outside ``_CONFIGS`` use the dense kernel unchanged.
Band widths up to the dense threshold can run as straight-line bodies of static width
(``static_band``): one ``pl.when`` per width, each the dense body's ops over the band's
keys (dynamic start). The row max, the row sum (128-lane slices in key order) and the
PV dot (K passes in key order) see the band loop's values in the band loop's order, so
the bits are the band loop's; one region per width lets the scheduler pack the MXU, VPU
and EUP phases that the loop runs apart.

The dense body can run as a software pipeline (``dense_pipeline``) over units (head,
128-row block): per 512-key tile, the scores of one unit, the exps of the previous unit
and the PV of the one before run in the same step, each element with the dense body's
ops and the row max / row sum / PV accumulated as the band loop does.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.kernels.flash_attention import (
    DEFAULT_MASK_VALUE,
    MIN_BLOCK_SIZE,
    NUM_LANES,
    NUM_SUBLANES,
    TRANS_B_DIM_NUMBERS,
    SegmentIds,
    _fwd_cost_estimate,
)

# Keys per band chunk (band granularity).
CHUNK = 512
# Keys per PV sub-product: the MXU contraction depth the dense kernel's dot
# accumulates in.
PV_K = 256
# A query block whose band spans more than this percentage of the bucket runs the
# dense body (cheaper than the band loop there).
DENSE_ABOVE_PERCENT = 75


class _Config(NamedTuple):
    heads_per_step: int  # heads per grid step (independent work per loop step)
    chunks_per_step: int  # band chunks per loop step
    dense_fallback: bool  # dense body for wide bands (needs its VMEM temporaries)
    single_buffer_kv: bool  # K/V blocks change once per head group
    static_band: bool = False  # band widths up to the dense threshold as straight-line bodies
    dense_pipeline: bool = False  # dense body as a pipeline over (head, 128-row) units


# (kv_len, block_q) -> config, each measured to fit the default scoped VMEM and to
# be byte-identical to the dense kernel. Other shapes keep the dense kernel.
_CONFIGS = {
    (3072, 256): _Config(2, 2, True, False, static_band=True, dense_pipeline=True),
    (4096, 256): _Config(2, 2, True, False, static_band=True, dense_pipeline=True),
    (6144, 256): _Config(2, 2, True, False, static_band=True, dense_pipeline=True),
    (8192, 256): _Config(2, 4, True, True, dense_pipeline=True),
    (12288, 256): _Config(1, 4, True, True),
    (16384, 128): _Config(1, 4, True, True),
}


def band_skip_config(
    num_heads: int, q_len: int, kv_len: int, block_q: int
) -> _Config | None:
    """The kernel config for this shape, or None to use the dense kernel."""
    cfg = _CONFIGS.get((kv_len, block_q))
    if cfg is None or q_len != kv_len or num_heads % cfg.heads_per_step:
        return None
    return cfg


def band_tables(seg_q, seg_kv, block_q: int, chunk: int = CHUNK):
    """Reference for the band the kernel computes per query block, int32 [B * nq] x2.

    Chunk c is in block i's band when the [min, max] of the chunk's kv ids meets the
    [min, max] of the block's q ids; the band runs from the first to the last such
    chunk. Every key sharing a segment with the block lies in it, for any id layout
    (its id is inside both ranges). A block with no such chunk gets the whole width.
    """
    B, q_len = seg_q.shape
    kv_len = seg_kv.shape[1]
    nq, nc = q_len // block_q, kv_len // chunk
    qb = seg_q.reshape(B, nq, block_q)
    kc = seg_kv.reshape(B, nc, chunk)
    qmin, qmax = qb.min(-1), qb.max(-1)
    cmin, cmax = kc.min(-1), kc.max(-1)
    meets = (cmin[:, None, :] <= qmax[:, :, None]) & (
        cmax[:, None, :] >= qmin[:, :, None]
    )
    idx = jnp.arange(nc, dtype=jnp.int32)
    lo = jnp.min(jnp.where(meets, idx, nc), axis=-1)
    hi = jnp.max(jnp.where(meets, idx, -1), axis=-1)
    empty = hi < lo
    lo = jnp.where(empty, 0, lo)
    n = jnp.where(empty, nc, hi - lo + 1)
    return lo.reshape(-1).astype(jnp.int32), n.reshape(-1).astype(jnp.int32)


def _chunk_ranges(ks_ref, cmin_ref, cmax_ref, kv_len):
    """[min, max] kv id of every key chunk, into SMEM."""
    for c in range(kv_len // CHUNK):
        ks = ks_ref[0, :, c * CHUNK : (c + 1) * CHUNK]
        cmin_ref[c] = jnp.min(ks)
        cmax_ref[c] = jnp.max(ks)


def _band(qs_ref, cmin_ref, cmax_ref, kv_len):
    """``band_tables`` for this query block, in the kernel: (first chunk, count).

    Computed from the segment-id blocks the dense kernel already receives, so the
    custom call's operands stay the dense kernel's (extra operands were measured to
    change XLA's scheduling of the surrounding compiled program, and its bits)."""
    nck = kv_len // CHUNK
    qs = qs_ref[0]  # [block_q, NUM_LANES], each row's id on every lane
    qmin = jnp.min(qs)
    qmax = jnp.max(qs)

    def step(c, lh):
        meets = jnp.logical_and(cmin_ref[c] <= qmax, cmax_ref[c] >= qmin)
        return (
            jnp.minimum(lh[0], jnp.where(meets, c, nck)),
            jnp.maximum(lh[1], jnp.where(meets, c, -1)),
        )

    lo, hi = lax.fori_loop(0, nck, step, (jnp.int32(nck), jnp.int32(-1)))
    empty = hi < lo
    return jnp.where(empty, 0, lo), jnp.where(empty, nck, hi - lo + 1)


def _dense_body(
    h,
    q_ref,
    k_ref,
    v_ref,
    qs_ref,
    ks_ref,
    o_ref,
    l_ref,
    m_ref,
    *,
    sm_scale,
    mask_value,
    kv_len,
):
    """``_flash_attention_kernel_single_batch_single_step`` for head ``h`` of the block."""
    q = q_ref[0, h]
    k = k_ref[0, h]
    s = jax.lax.dot_general(
        q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
    )
    if sm_scale != 1.0:
        s *= sm_scale
    q_segment_ids = jnp.tile(qs_ref[0], (1, kv_len // NUM_LANES))
    kv_segment_ids = ks_ref[0, :1]
    mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)
    s = s + jnp.where(mask, 0.0, mask_value)
    m = jnp.max(s, axis=1)[:, None]
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=1)[:, None]
    p /= l
    if m_ref is not None:
        m_ref[0, h] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
    if l_ref is not None:
        l_ref[0, h] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))
    v = v_ref[0, h]
    o_ref[0, h] = jax.lax.dot(
        p.astype(v.dtype), v, preferred_element_type=jnp.float32
    ).astype(o_ref.dtype)


def _static_band_widths(cfg: _Config, kv_len: int) -> tuple[int, ...]:
    """Band widths (in chunks) run as straight-line bodies: every width the band path
    takes (1 .. the dense threshold); wider bands run the dense body."""
    if not cfg.static_band:
        return ()
    assert cfg.dense_fallback, "static_band needs dense_fallback for the wide bands"
    return tuple(range(1, DENSE_ABOVE_PERCENT * (kv_len // CHUNK) // 100 + 1))


def _band_static(
    width,
    lo,
    q_ref,
    k_ref,
    v_ref,
    qs_ref,
    ks_ref,
    o_ref,
    l_ref,
    m_ref,
    *,
    sm_scale,
    mask_value,
):
    """The band of ``width`` chunks from chunk ``lo``, for every head of the step, as
    one straight-line body: ``_dense_body``'s ops over the band's keys."""
    W = width * CHUNK
    st = pl.multiple_of(lo * CHUNK, CHUNK)
    qseg = jnp.tile(qs_ref[0], (1, W // NUM_LANES))
    ks = ks_ref[0, pl.ds(0, 1), pl.ds(st, W)]
    mask = jnp.equal(qseg, ks).astype(jnp.bool_)
    for h in range(q_ref.shape[1]):
        k = k_ref[0, h, pl.ds(st, W), :]
        s = jax.lax.dot_general(
            q_ref[0, h], k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
        )
        if sm_scale != 1.0:
            s *= sm_scale
        s = s + jnp.where(mask, 0.0, mask_value)
        m = jnp.max(s, axis=1)[:, None]
        p = jnp.exp(s - m)
        l = jnp.sum(p, axis=1)[:, None]
        p = p / l
        if m_ref is not None:
            m_ref[0, h] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
        if l_ref is not None:
            l_ref[0, h] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))
        v = v_ref[0, h, pl.ds(st, W), :]
        o_ref[0, h] = jax.lax.dot(
            p.astype(v.dtype), v, preferred_element_type=jnp.float32
        ).astype(o_ref.dtype)


# Dense pipeline: rows per unit, keys per step, scratch slots when it owns the scratch.
PIPE_ROWS = 128
PIPE_TILE = 512
PIPE_SLOTS = 3


def _dense_pipeline(
    q_ref,
    k_ref,
    v_ref,
    qs_ref,
    ks_ref,
    o_ref,
    l_ref,
    m_ref,
    p_ref,
    *,
    sm_scale,
    mask_value,
    ring,
):
    """``_dense_body`` for every head of the step as a software pipeline over units
    (head, PIPE_ROWS-row block). Step s runs, per PIPE_TILE-key tile, the scores of
    unit s (QK, scale, mask, store, running max), the exps of unit s-1 (exp, store,
    running sum) and the PV of unit s-2 (divide, cast, PV_K-key dots). The row max and
    the row sum (128-lane slices in key order) and the PV (PV_K-key dots in key order)
    accumulate as the band loop does. ``ring``: p_ref is [PIPE_SLOTS, PIPE_ROWS, kv]
    (unit i uses slot i % PIPE_SLOTS), else [heads, block_q, kv]."""
    H, block_q, head_dim = q_ref.shape[1], q_ref.shape[2], q_ref.shape[3]
    kv_len = k_ref.shape[2]
    R = PIPE_ROWS
    units = [(h, rb) for rb in range(block_q // R) for h in range(H)]
    lanes = PIPE_TILE // NUM_LANES

    def rows(u):
        return pl.ds(u[1] * R, R)

    def scratch(i, u, st):
        if ring:
            return (i % PIPE_SLOTS, slice(None), pl.ds(st, PIPE_TILE))
        return (u[0], rows(u), pl.ds(st, PIPE_TILE))

    def scores(i, u, st, mx):
        qseg = jnp.tile(qs_ref[0, rows(u), :], (1, lanes))
        ks = ks_ref[0, pl.ds(0, 1), pl.ds(st, PIPE_TILE)]
        mask = jnp.equal(qseg, ks).astype(jnp.bool_)
        k = k_ref[0, u[0], pl.ds(st, PIPE_TILE), :]
        s = lax.dot_general(
            q_ref[0, u[0], rows(u), :],
            k,
            TRANS_B_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        if sm_scale != 1.0:
            s *= sm_scale
        s = s + jnp.where(mask, 0.0, mask_value)
        p_ref[scratch(i, u, st)] = s
        for j in range(lanes):
            mx = jnp.maximum(mx, s[:, j * NUM_LANES : (j + 1) * NUM_LANES])
        return mx

    def exps(i, u, st, m, acc):
        p = jnp.exp(p_ref[scratch(i, u, st)] - m)
        p_ref[scratch(i, u, st)] = p
        for j in range(lanes):
            acc = acc + p[:, j * NUM_LANES : (j + 1) * NUM_LANES]
        return acc

    def pv(i, u, st, l, o):
        p = p_ref[scratch(i, u, st)] / l
        for j in range(PIPE_TILE // PV_K):
            vj = v_ref[0, u[0], pl.ds(st + j * PV_K, PV_K), :]
            o = o + jax.lax.dot(
                p[:, j * PV_K : (j + 1) * PV_K].astype(vj.dtype),
                vj,
                preferred_element_type=jnp.float32,
            )
        return o

    m_of, l_of = {}, {}
    n_units = len(units)
    for step in range(n_units + 2):
        a, b, c = step, step - 1, step - 2
        mx = jnp.full((R, NUM_LANES), -jnp.inf, jnp.float32)
        acc = jnp.zeros((R, NUM_LANES), jnp.float32)
        o = jnp.zeros((R, head_dim), jnp.float32)
        for t in range(kv_len // PIPE_TILE):
            st = t * PIPE_TILE
            if a < n_units:
                mx = scores(a, units[a], st, mx)
            if 0 <= b < n_units:
                acc = exps(b, units[b], st, m_of[b], acc)
            if 0 <= c < n_units:
                o = pv(c, units[c], st, l_of[c], o)
        if a < n_units:
            m_of[a] = jnp.max(mx, axis=1)[:, None]
        if 0 <= b < n_units:
            l_of[b] = jnp.sum(acc, axis=1)[:, None]
        if 0 <= c < n_units:
            h = units[c][0]
            if m_ref is not None:
                m_ref[0, h, rows(units[c]), :] = lax.broadcast_in_dim(
                    m_of[c], (R, m_ref.shape[3]), range(2)
                )
            if l_ref is not None:
                l_ref[0, h, rows(units[c]), :] = lax.broadcast_in_dim(
                    l_of[c], (R, l_ref.shape[3]), range(2)
                )
            o_ref[0, h, rows(units[c]), :] = o.astype(o_ref.dtype)


def _band_skip_kernel(
    q_ref,
    k_ref,
    v_ref,
    qs_ref,
    ks_ref,
    o_ref,
    l_ref,
    m_ref,
    p_ref,  # scratch [heads_per_step, block_q, kv_len] f32
    cmin_ref,
    cmax_ref,  # scratch [kv_len // CHUNK] int32 (SMEM): kv id range per chunk
    lo_tab_ref,
    n_tab_ref,  # scratch [q_len // block_q] int32 (SMEM): band per query block
    *,
    sm_scale,
    kv_len,
    cfg,
    mask_value,
):
    H, U = cfg.heads_per_step, cfg.chunks_per_step
    # The band depends on the segment ids only: computed while the first head group
    # runs (the grid runs in order, see dimension_semantics), read back for the rest.
    i = pl.program_id(2)
    first_heads = pl.program_id(1) == 0
    pl.when(jnp.logical_and(first_heads, i == 0))(
        lambda: _chunk_ranges(ks_ref, cmin_ref, cmax_ref, kv_len)
    )

    @pl.when(first_heads)
    def _():
        lo_i, n_i = _band(qs_ref, cmin_ref, cmax_ref, kv_len)
        lo_tab_ref[i] = lo_i
        n_tab_ref[i] = n_i

    lo = lo_tab_ref[i]
    n = n_tab_ref[i]
    nck = kv_len // CHUNK
    block_q, head_dim = q_ref.shape[2], q_ref.shape[3]

    def band():
        qseg = jnp.tile(qs_ref[0], (1, CHUNK // NUM_LANES))
        qh = [q_ref[0, h] for h in range(H)]

        def start(i):
            return pl.multiple_of((lo + i) * CHUNK, CHUNK)

        def over_band(body, init):
            # Band chunks in key order, U per loop step, then the remainder.
            def step(i, c):
                for u in range(U):
                    c = body(start(i * U + u), c)
                return c

            c = lax.fori_loop(0, n // U, step, init)
            for u in range(U - 1):
                i = (n // U) * U + u
                c = lax.cond(i < n, lambda c, i=i: body(start(i), c), lambda c: c, c)
            return c

        def scores(st, mx):
            ks = ks_ref[0, pl.ds(0, 1), pl.ds(st, CHUNK)]
            mask = jnp.equal(qseg, ks).astype(jnp.bool_)
            out = []
            for h in range(H):
                k = k_ref[0, h, pl.ds(st, CHUNK), :]
                s = lax.dot_general(
                    qh[h], k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
                )
                if sm_scale != 1.0:
                    s *= sm_scale
                s = s + jnp.where(mask, 0.0, mask_value)
                p_ref[h, :, pl.ds(st, CHUNK)] = s
                x = mx[h]
                for j in range(CHUNK // NUM_LANES):
                    x = jnp.maximum(x, s[:, j * NUM_LANES : (j + 1) * NUM_LANES])
                out.append(x)
            return tuple(out)

        mx = over_band(
            scores,
            tuple(
                jnp.full((block_q, NUM_LANES), -jnp.inf, jnp.float32) for _ in range(H)
            ),
        )
        m = [jnp.max(x, axis=1)[:, None] for x in mx]

        def exps(st, acc):
            out = []
            for h in range(H):
                p = jnp.exp(p_ref[h, :, pl.ds(st, CHUNK)] - m[h])
                p_ref[h, :, pl.ds(st, CHUNK)] = p
                a = acc[h]
                for j in range(CHUNK // NUM_LANES):
                    a = a + p[:, j * NUM_LANES : (j + 1) * NUM_LANES]
                out.append(a)
            return tuple(out)

        acc = over_band(
            exps, tuple(jnp.zeros((block_q, NUM_LANES), jnp.float32) for _ in range(H))
        )
        l = [jnp.sum(a, axis=1)[:, None] for a in acc]

        def pv(st, o):
            out = []
            for h in range(H):
                p = p_ref[h, :, pl.ds(st, CHUNK)] / l[h]
                x = o[h]
                for j in range(CHUNK // PV_K):
                    vj = v_ref[0, h, pl.ds(st + j * PV_K, PV_K), :]
                    x = x + jax.lax.dot(
                        p[:, j * PV_K : (j + 1) * PV_K].astype(vj.dtype),
                        vj,
                        preferred_element_type=jnp.float32,
                    )
                out.append(x)
            return tuple(out)

        o = over_band(
            pv, tuple(jnp.zeros((block_q, head_dim), jnp.float32) for _ in range(H))
        )
        for h in range(H):
            if m_ref is not None:
                m_ref[0, h] = lax.broadcast_in_dim(m[h], m_ref.shape[2:], range(2))
            if l_ref is not None:
                l_ref[0, h] = lax.broadcast_in_dim(l[h], l_ref.shape[2:], range(2))
            o_ref[0, h] = o[h].astype(o_ref.dtype)

    if not cfg.dense_fallback:
        band()
        return
    use_band = n * 100 <= DENSE_ABOVE_PERCENT * nck
    widths = _static_band_widths(cfg, kv_len)
    if widths:
        # One straight-line body per band width the band path takes (n >= 1).
        for w in widths:
            pl.when(n == w)(
                functools.partial(
                    _band_static,
                    w,
                    lo,
                    q_ref,
                    k_ref,
                    v_ref,
                    qs_ref,
                    ks_ref,
                    o_ref,
                    l_ref,
                    m_ref,
                    sm_scale=sm_scale,
                    mask_value=mask_value,
                )
            )
    else:
        pl.when(use_band)(band)

    @pl.when(jnp.logical_not(use_band))
    def _():
        if cfg.dense_pipeline:
            _dense_pipeline(
                q_ref,
                k_ref,
                v_ref,
                qs_ref,
                ks_ref,
                o_ref,
                l_ref,
                m_ref,
                p_ref,
                sm_scale=sm_scale,
                mask_value=mask_value,
                ring=cfg.static_band,
            )
            return
        for h in range(H):
            _dense_body(
                h,
                q_ref,
                k_ref,
                v_ref,
                qs_ref,
                ks_ref,
                o_ref,
                l_ref,
                m_ref,
                sm_scale=sm_scale,
                mask_value=mask_value,
                kv_len=kv_len,
            )


def _p_scratch_shape(cfg: _Config, block_q: int, kv_len: int) -> tuple[int, ...]:
    """The f32 scratch: [heads, block_q, kv] for the band loop (the dense pipeline
    shares it); [PIPE_SLOTS, PIPE_ROWS, kv] for the dense pipeline alone; one tile
    when nothing uses it (static band widths, dense fallback without pipeline)."""
    if cfg.static_band:
        if cfg.dense_pipeline:
            return (PIPE_SLOTS, PIPE_ROWS, kv_len)
        return (1, NUM_SUBLANES, NUM_LANES)
    return (cfg.heads_per_step, block_q, kv_len)


@functools.partial(jax.jit, static_argnames=["sm_scale", "block_q", "save_residuals"])
def band_skip_flash_attention(
    q,  # [batch_size, num_heads, seq_len, head_dim]
    k,  # [batch_size, num_heads, seq_len, head_dim]
    v,  # [batch_size, num_heads, seq_len, head_dim]
    segment_ids: SegmentIds,
    *,
    sm_scale: float,
    block_q: int,
    save_residuals: bool = False,
):
    """Non-causal segment-masked attention, byte-identical to ``flash_attention``
    with ``block_sizes=BlockSizes(block_q, kv_len, kv_len, 1)`` (the dense
    single-step path) for shapes in ``_CONFIGS``; raises for other shapes."""
    B, H, q_len, D = q.shape
    kv_len = k.shape[2]
    cfg = band_skip_config(H, q_len, kv_len, block_q)
    if cfg is None:
        raise ValueError(
            f"no band-skip config for heads={H} q_len={q_len} kv_len={kv_len} block_q={block_q}"
        )
    nq = q_len // block_q
    HB = cfg.heads_per_step
    q_segment_ids = lax.broadcast_in_dim(segment_ids.q, (B, q_len, NUM_LANES), (0, 1))
    kv_segment_ids = lax.broadcast_in_dim(
        segment_ids.kv, (B, NUM_SUBLANES, kv_len), (0, 2)
    )

    def q_map(b, h, i, j):
        return (b, h, i, 0)

    def kv_map(b, h, i, j):
        return (b, h, 0, 0)

    def q_seg_map(b, h, i, j):
        return (b, i, 0)

    def kv_seg_map(b, h, i, j):
        return (b, 0, 0)

    kv_buf = dict(pipeline_mode=pl.Buffered(1)) if cfg.single_buffer_kv else {}
    in_specs = [
        pl.BlockSpec((1, HB, block_q, D), q_map),
        pl.BlockSpec((1, HB, kv_len, D), kv_map, **kv_buf),
        pl.BlockSpec((1, HB, kv_len, D), kv_map, **kv_buf),
        pl.BlockSpec((1, block_q, NUM_LANES), q_seg_map),
        pl.BlockSpec((1, NUM_SUBLANES, kv_len), kv_seg_map),
    ]
    out_shape = [jax.ShapeDtypeStruct(q.shape, q.dtype)]
    out_specs = [pl.BlockSpec((1, HB, block_q, D), q_map)]
    if save_residuals:
        lm = jax.ShapeDtypeStruct((B, H, q_len, MIN_BLOCK_SIZE), jnp.float32)
        out_shape += [lm, lm]
        out_specs += [pl.BlockSpec((1, HB, block_q, MIN_BLOCK_SIZE), q_map)] * 2
    else:
        out_shape += [None, None]
        out_specs += [None, None]
    kernel = functools.partial(
        _band_skip_kernel,
        sm_scale=sm_scale,
        kv_len=kv_len,
        cfg=cfg,
        mask_value=DEFAULT_MASK_VALUE,
    )
    o, *aux = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(B, H // HB, nq, 1),
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=[
                pltpu.VMEM(_p_scratch_shape(cfg, block_q, kv_len), jnp.float32),
                pltpu.SMEM((kv_len // CHUNK,), jnp.int32),
                pltpu.SMEM((kv_len // CHUNK,), jnp.int32),
                pltpu.SMEM((nq,), jnp.int32),
                pltpu.SMEM((nq,), jnp.int32),
            ],
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary", "arbitrary"),
            vmem_limit_bytes=None,
        ),
        # The dense kernel's estimate, so XLA schedules the surrounding program as
        # it does around the dense kernel.
        cost_estimate=_fwd_cost_estimate(
            q,
            k,
            v,
            None,
            segment_ids,
            causal=False,
            sm_scale=sm_scale,
            kernel_inputs_specs=(q, k, v, None, q_segment_ids, kv_segment_ids),
            kernel_outputs_specs=out_shape,
        ),
    )(q, k, v, q_segment_ids, kv_segment_ids)
    if save_residuals:
        l, m = (x[..., 0] for x in aux[-2:])
        return o, l, m
    return o
