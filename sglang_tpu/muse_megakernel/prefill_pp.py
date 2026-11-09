# SPDX-License-Identifier: Apache-2.0
"""Muse Spark prefill, pipeline-parallel: one stage's layers whole on one device, no collectives.

A stage holds its layers whole (every head, all experts, the full pooler and unpooler) on one
device, so nothing in it talks to another device; the residual stream moves between stages
outside. Per layer slot, three calls:

  attn_in    (fused Pallas) qkvg projection, qk norms, rope -> q, k, v, gate
  attention  sglang_tpu's batched RPA prefill kernel, as the P/D prefill serves it: the
             chunk's K/V into the slot's fp8 SEQ_ALONG_LANE pool, attention over the pool
             (cached context and the causal chunk), each head's output RMS-normalized
  post_attn  (fused Pallas) gated o projection and residual, router top-k, expert projection;
             each routed expert once over its rows (gathered rows, weights streamed per
             intermediate tile, normalized outputs scattered to their (token, slot) rows);
             weighted slot sum, unpooler, gated residual

Every stage has the same slot layout (cfg.slots layers of dense weights, global attention at
cfg.global_slots; expert slabs whose rows hold the stage's layers back to back), so one program
serves every stage. A per-stage slot table (SMEM) says which slots hold a layer, its expert
format and its slab offsets.
"""

import os

import jax
import jax.numpy as jnp
import msgspec
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.kernel import (
    _fp4_dot,
    _fp8_dot,
    _heads_rms,
    _rms,
    _topk_iter,
)

LANE = 128
PHASES = ("attn_in", "attn_out", "moe", "post_moe")
FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32

# slot table: a header, then NF int32 fields per slot
HDR = 8
H_STAGE, H_LAYERS = 0, 1
NF = 8
F_VALID, F_GLOBAL, F_FP8, F_WOFF, F_SOFF, F_LAYER = range(6)
# fp8 experts carry one channelwise scale row, padded to a bf16 tile
FP8_SCALE_ROWS = 16

# rows of the per-layer vector slab [slots, V_ROWS, vec_width] f32
V = dict(
    pre_attn_w=0,
    attn_alpha=1,
    attn_beta=2,
    pre_ffn_w=3,
    post_ffn_w=4,
    ffn_alpha=5,
    ffn_beta=6,
    pre_expert_w=7,
    expert_bias=8,
    post_expert_w=9,
    wqkvg_s=10,
    wo_s=11,
    wpool_s=12,
    wunpool_s=13,
)
V_ROWS = 16


class PPConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Static description of one stage program (every stage runs the same one)."""

    hidden: int
    head_dim: int
    q_heads: int
    kv_heads: int
    n_experts: int
    top_k: int
    expert_hidden: int
    inter: int
    rms_eps: float
    post_eps: float
    scale_query_by: float
    attn_scale: float
    router_mult: float
    route_scale: float
    router_eps: float
    window: int
    # token rows of the chunk
    rows: int
    # layer slots per stage
    slots: int
    page: int
    pages_per_seq: int
    # pool pages of a global / sliding-window slot
    global_pages: int
    swa_pages: int
    # u32 word rows per expert in the slabs (all of a stage's layers)
    slab_rows: int
    max_position: int
    # slots of the global attention layers (the rest are sliding-window)
    global_slots: tuple = (2, 6)
    # dense row tile, weight column tile, MoE row block and intermediate tile
    tm: int = 256
    tn: int = 512
    rb: int = 256
    ti: int = 512
    # MoE matmul row counts: a block runs the smallest that holds its rows
    moe_row_buckets: tuple = (128, 192)

    @property
    def q_dim(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def qkvg_dim(self) -> int:
        return 2 * self.q_dim + 2 * self.kv_dim

    @property
    def vec_width(self) -> int:
        return max(self.qkvg_dim, self.hidden)

    def validate(self) -> None:
        """Raises ValueError for a geometry the stage kernel does not tile."""
        H, He, T = self.hidden, self.expert_hidden, self.rows
        checks = [
            (self.tn == self.ti, "the weight buffer is shared: tn == ti"),
            (
                self.rb == self.tm and H >= 2 * He,
                "MoE rows reuse the dense row buffers",
            ),
            (self.tm % (8 * self.top_k) == 0 and self.tm % 16 == 0, "row tile"),
            (
                T % self.tm == 0 and T % self.page == 0,
                "chunk rows",
            ),
            (
                self.qkvg_dim % self.tn == 0 and H % self.tn == 0 and He % self.tn == 0,
                "tn",
            ),
            (self.q_dim % self.tn == 0 and self.kv_dim % self.tn == 0, "head tiles"),
            (self.tn % (2 * self.head_dim) == 0, "head pairs per column tile"),
            (
                self.inter == He,
                "expert slabs share row offsets: inter == expert_hidden",
            ),
            (self.inter % self.ti == 0 and self.ti % 128 == 0, "intermediate tile"),
            (self.page == 256 and self.head_dim % 4 == 0, "fp8 SAL pages of 256"),
            (self.n_experts <= 2 * LANE and self.n_experts % LANE == 0, "router lanes"),
            (self.top_k <= 16 and (T * self.top_k) % LANE == 0, "routing table"),
            (self.q_heads % self.kv_heads == 0, "GQA"),
            (all(r % 32 == 0 for r in self.moe_row_buckets), "MoE row buckets"),
        ]
        bad = [why for ok, why in checks if not ok]
        if bad:
            raise ValueError(f"PPConfig {self}: {'; '.join(bad)}")


def expert_rows(kind: str, k: int) -> tuple[int, int]:
    """(u32 word rows, scale rows) one layer's [K, N] expert matrix takes in a slab."""
    if kind == "fp4":
        return k // 8, k // 16
    if kind == "fp8":
        return k // 4, FP8_SCALE_ROWS
    raise ValueError(f"expert kind {kind}")


def stage_layers(n_layers: int, split: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Consecutive layer runs of the given sizes."""
    if sum(split) != n_layers:
        raise ValueError(f"split {split} does not cover {n_layers} layers")
    out, lo = [], 0
    for n in split:
        out.append(tuple(range(lo, lo + n)))
        lo += n
    return out


def slot_table(
    c: PPConfig, stage: int, layers: tuple[int, ...], kinds: tuple, global_layers: tuple
) -> np.ndarray:
    """The stage's SMEM slot table [HDR + slots * NF] int32. Its layers take consecutive slots
    from the one that puts its global layers on cfg.global_slots.

    Raises:
      ValueError: The layers do not fit the stage's slots or slab rows, or their global layers
        cannot sit on cfg.global_slots.
    """
    globals_at = [i for i, layer in enumerate(layers) if layer in global_layers]
    first = c.global_slots[0] - globals_at[0] if globals_at else 0
    placed = [first + i for i in globals_at]
    if first < 0 or first + len(layers) > c.slots or placed != list(c.global_slots):
        raise ValueError(
            f"stage {stage}: layers {layers} (global {[layers[i] for i in globals_at]}) do not "
            f"fit {c.slots} slots with global attention at {c.global_slots}"
        )
    t = np.zeros(HDR + c.slots * NF, np.int32)
    t[H_STAGE], t[H_LAYERS] = stage, len(layers)
    woff = soff = 0
    for i, layer in enumerate(layers):
        s = first + i
        f = t[HDR + s * NF : HDR + (s + 1) * NF]
        wr, sr = expert_rows(kinds[layer], c.expert_hidden)
        f[F_VALID], f[F_GLOBAL] = 1, int(layer in global_layers)
        f[F_FP8], f[F_WOFF], f[F_SOFF], f[F_LAYER] = (
            int(kinds[layer] == "fp8"),
            woff,
            soff,
            layer,
        )
        woff, soff = woff + wr, soff + sr
    if woff > c.slab_rows or soff > c.slab_rows // 2:
        raise ValueError(
            f"stage {stage}: expert slabs need {woff} word / {soff} scale rows"
        )
    return t


def weight_shapes(c: PPConfig) -> dict[str, tuple[tuple[int, ...], jnp.dtype]]:
    """One stage's weights (per device). Dense weights are column-tile major, each tile a
    contiguous [K, tn] block; expert slabs are [E, tile, slab rows, ti]: w13 tiles are the
    gate's then the up projection's intermediate tiles, w2 tiles its output column pieces."""
    S, H, He, E, tn, ti = c.slots, c.hidden, c.expert_hidden, c.n_experts, c.tn, c.ti
    R = c.slab_rows
    return {
        "wqkvg": ((S, c.qkvg_dim // tn, H, tn), FP8),
        "wo": ((S, H // tn, c.q_dim, tn), FP8),
        "wpool": ((S, He // tn, H, tn), FP8),
        "wunpool": ((S, H // tn, He, tn), FP8),
        "wrouter": ((S, H, E), BF16),
        "vecs": ((S, V_ROWS, c.vec_width), F32),
        "w13": ((E, 2 * c.inter // ti, R, ti), jnp.int32),
        "s13": ((E, 2 * c.inter // ti, R // 2, ti), BF16),
        "w2": ((E, He // ti, R, ti), jnp.int32),
        "s2": ((E, He // ti, R // 2, ti), BF16),
    }


WEIGHT_ORDER = (
    "wqkvg",
    "wo",
    "wpool",
    "wunpool",
    "wrouter",
    "vecs",
    "w13",
    "s13",
    "w2",
    "s2",
)


def pool_shape(c: PPConfig, pages: int) -> tuple[int, ...]:
    return (pages, 2 * c.kv_heads, c.head_dim // 4, 4, c.page)


def _moe_rows(c: PPConfig) -> tuple[int, int]:
    """(word rows, scale rows) of one MoE step's weights in the shared buffers."""
    He, ti, npc = c.expert_hidden, c.ti, c.expert_hidden // c.ti
    w4, s4 = expert_rows("fp4", He)
    w8, _ = expert_rows("fp8", He)
    words = max(2 * w4 + npc * (ti // 8), 2 * w8 + npc * (ti // 4))
    scales = max(2 * s4 + npc * (ti // 16), (2 + npc) * FP8_SCALE_ROWS)
    return words, scales


def _bf(x):
    return x.astype(BF16).astype(F32)


def _mm(a_ref, w_ref, k: int, kc: int = 512):
    """a [M, k] bf16 @ w [k, N] (bf16 or fp8 refs), f32, contracted in kc slices."""
    acc = None
    for k0 in range(0, k, kc):
        part = jnp.dot(
            a_ref[:, pl.ds(k0, kc)], w_ref[pl.ds(k0, kc), :], preferred_element_type=F32
        )
        acc = part if acc is None else acc + part
    return acc


def _rope_lanes(x, cos, sin):
    """Interleaved rotary on [M, n * hd] with lane-expanded cos / sin of the same shape."""
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    even = (lane % 2) == 0
    w = x.shape[1]
    partner = jnp.where(
        even, pltpu.roll(x, shift=w - 1, axis=1), pltpu.roll(x, shift=1, axis=1)
    )
    return x * cos + jnp.where(even, -1.0, 1.0) * partner * sin


def skipped_phases() -> frozenset[str]:
    """MUSE_PP_SKIP=a,b: timing ablation only (wrong numerics), those phases do not run."""
    skip = frozenset(p for p in os.environ.get("MUSE_PP_SKIP", "").split(",") if p)
    if not skip <= set(PHASES):
        raise ValueError(f"MUSE_PP_SKIP={sorted(skip)}: phases are {PHASES}")
    return skip


def stage_calls(c: PPConfig):
    """The stage's two fused calls, for the slot in ctl[2] (ctl [3] i32 = [cached tokens, active
    stage or -1, slot]); a device whose stage is not active, or a slot without a layer, returns
    at once:
      attn_in(ctl, slot table, x [T, H] f32, rope [max_position, 2 hd] f32, *weights) ->
        (q [T, q_dim] bf16, k, v [T, kv_dim] bf16, gate [T, q_dim] bf16)
      post_attn(ctl, slot table, x, o [T, q_dim] bf16, gate, *weights) -> (x, scratch...),
        x updated in place."""
    c.validate()
    skip = skipped_phases()
    T, H, He, E, K = c.rows, c.hidden, c.expert_hidden, c.n_experts, c.top_k
    hd, nkv = c.head_dim, c.kv_heads
    qd, kvd, nq = c.q_dim, c.kv_dim, c.qkvg_dim
    tm, tn, rb, ti = c.tm, c.tn, c.rb, c.ti
    n_rt = T // tm
    sub = 16  # rows per vector pass over a [tm, H] tile
    n_qt, n_kt = qd // tn, kvd // tn
    ntc = nq // tn
    nti, npc = c.inter // ti, He // ti
    w4, s4 = expert_rows("fp4", He)
    w8, _ = expert_rows("fp8", He)
    moe_w, moe_s = _moe_rows(c)
    wr_rows = max(H, qd, He, 4 * moe_w)
    row_buckets = tuple(sorted({min(rb, r) for r in c.moe_row_buckets} | {rb}))
    tok_per_grp = tm // K
    slots_n = T * K
    S = c.slots

    def body(R, phases):
        ctl, slots, x_in = R["ctl"], R["slots"], R["x_in"]
        rope, o_h, g_in = R.get("rope"), R.get("o_h"), R.get("g_in")
        wqkvg, wo, wpool, wunpool = R["wqkvg"], R["wo"], R["wpool"], R["wunpool"]
        wrouter, vecs, w13, s13, w2, s2 = (R[n] for n in WEIGHT_ORDER[4:])
        q_h, k_h, v_h, g_h = R.get("q_h"), R.get("k_h"), R.get("v_h"), R.get("g_h")
        xe_h, ids_h, tw_h, y_h = (
            R.get("xe_h"),
            R.get("ids_h"),
            R.get("tw_h"),
            R.get("y_h"),
        )
        x_out = x_in  # the stream is aliased in place
        xbuf, accbuf, abuf, gbuf, pbuf, wbuf, sbuf = (R[n] for n in SCRATCH[:7])
        vec_v, wr_v, rope_v, rope8, otile, idp_v, tw_v = (R[n] for n in SCRATCH[7:14])
        ids_s, cnt_s, off_s, fill_s, lst_s, ite_s, itb_s, misc_s = (
            R[n] for n in SCRATCH[14:22]
        )
        sem_x, sem_w, sem_o, sem_m, sem_g, sem_y = (R[n] for n in SCRATCH[22:])

        def vrow(name, n, off=0):
            return vec_v[pl.ds(V[name], 1), pl.ds(off, n)]

        def rows(i, n):
            return pl.ds(pl.multiple_of(i * n, n), n)

        def wv(b):
            return wbuf.at[b].bitcast(jnp.int32)

        def attn_in(sl, is_g, ctx):
            total = n_rt * ntc

            def wcp(idx, b):
                return pltpu.make_async_copy(
                    wqkvg.at[sl, idx % ntc], wbuf.at[b, pl.ds(0, H)], sem_w.at[b]
                )

            def out_cp(b, dst):
                return pltpu.make_async_copy(otile.at[b], dst, sem_o.at[b])

            def emit(acc, dst):
                """A [tm, tn] output tile through the double-buffered otile."""
                n = misc_s[4]
                b = n % 2

                @pl.when(n >= 2)
                def _():
                    out_cp(b, q_h.at[pl.ds(0, tm), pl.ds(0, tn)]).wait()

                otile[b] = acc.astype(BF16)
                out_cp(b, dst).start()
                misc_s[4] = n + 1

            misc_s[4] = 0
            wcp(0, 0).start()

            def row_tile(i, carry):
                r0 = pl.multiple_of(i * tm, tm)
                cx = pltpu.make_async_copy(x_in.at[pl.ds(r0, tm)], xbuf, sem_x.at[0])
                cr = pltpu.make_async_copy(
                    rope.at[pl.ds(ctx + r0, tm)], rope_v, sem_x.at[1]
                )
                cx.start()
                cr.start()
                cx.wait()
                cr.wait()
                rp = rope_v[...]
                rope8[0] = jnp.concatenate([rp[:, :hd]] * (tn // hd), axis=1)
                rope8[1] = jnp.concatenate([rp[:, hd:]] * (tn // hd), axis=1)

                def norm(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    abuf[rr, :] = (
                        _rms(xbuf[rr, :], c.rms_eps) * vrow("pre_attn_w", H)
                    ).astype(BF16)
                    return cc

                jax.lax.fori_loop(0, tm // sub, norm, 0)

                def tile(jt):
                    idx = i * ntc + jt
                    b = idx % 2
                    wcp(idx, b).wait()

                    @pl.when(idx + 1 < total)
                    def _():
                        wcp(idx + 1, 1 - b).start()

                    return _mm(abuf, wbuf.at[b], H) * vrow(
                        "wqkvg_s", tn, pl.multiple_of(jt * tn, tn)
                    )

                def roped(x):
                    return jnp.where(is_g != 0, x, _rope_lanes(x, rope8[0], rope8[1]))

                def q_tile(jt, cc):
                    q = _heads_rms(
                        tile(jt), tn // hd, hd, c.rms_eps, gain=c.scale_query_by
                    )
                    emit(roped(q), q_h.at[pl.ds(r0, tm), rows(jt, tn)])
                    return cc

                jax.lax.fori_loop(0, n_qt, q_tile, 0)
                for kt in range(n_kt):
                    k = roped(_heads_rms(tile(n_qt + kt), tn // hd, hd, c.rms_eps))
                    emit(k, k_h.at[pl.ds(r0, tm), rows(kt, tn)])
                for kt in range(n_kt):
                    emit(tile(n_qt + n_kt + kt), v_h.at[pl.ds(r0, tm), rows(kt, tn)])

                def g_tile(jt, cc):
                    emit(
                        tile(jt), g_h.at[pl.ds(r0, tm), rows(jt - n_qt - 2 * n_kt, tn)]
                    )
                    return cc

                jax.lax.fori_loop(n_qt + 2 * n_kt, ntc, g_tile, 0)
                return carry

            jax.lax.fori_loop(0, n_rt, row_tile, 0)
            for b in range(2):

                @pl.when(misc_s[4] > b)
                def _(b=b):
                    out_cp(b, q_h.at[pl.ds(0, tm), pl.ds(0, tn)]).wait()

        def attn_out(sl):
            n_wo, n_wp = H // tn, He // tn
            per = n_wo + n_wp
            total = n_rt * per

            def wcp(idx, b, wait):
                j = idx % per

                @pl.when(j < n_wo)
                def _():
                    cp = pltpu.make_async_copy(
                        wo.at[sl, j], wbuf.at[b, pl.ds(0, qd)], sem_w.at[b]
                    )
                    cp.wait() if wait else cp.start()

                @pl.when(j >= n_wo)
                def _():
                    cp = pltpu.make_async_copy(
                        wpool.at[sl, j - n_wo], wbuf.at[b, pl.ds(0, H)], sem_w.at[b]
                    )
                    cp.wait() if wait else cp.start()

            def take(idx):
                b = idx % 2
                wcp(idx, b, True)

                @pl.when(idx + 1 < total)
                def _():
                    wcp(idx + 1, 1 - b, False)

                return b

            cwr = pltpu.make_async_copy(wrouter.at[sl], wr_v, sem_m.at[5])
            cwr.start()
            wcp(0, 0, False)

            def row_tile(i, carry):
                r0 = pl.multiple_of(i * tm, tm)
                cx = pltpu.make_async_copy(x_out.at[pl.ds(r0, tm)], xbuf, sem_x.at[0])
                ca = pltpu.make_async_copy(o_h.at[pl.ds(r0, tm)], abuf, sem_x.at[1])
                cg = pltpu.make_async_copy(g_in.at[pl.ds(r0, tm)], gbuf, sem_x.at[2])
                for cp in (cx, ca, cg):
                    cp.start()
                for cp in (cx, ca, cg):
                    cp.wait()

                def gated(r, cc):
                    # the attention kernel stores each head's output RMS-normalized
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    g = jax.nn.sigmoid(gbuf[rr, :].astype(F32))
                    abuf[rr, :] = (g * abuf[rr, :].astype(F32)).astype(BF16)
                    return cc

                jax.lax.fori_loop(0, tm // sub, gated, 0)

                def wo_tile(j, cc):
                    b = take(i * per + j)
                    accbuf[:, rows(j, tn)] = _mm(abuf, wbuf.at[b], qd)
                    return cc

                jax.lax.fori_loop(0, n_wo, wo_tile, 0)

                def resid(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    ao = _bf(accbuf[rr, :] * vrow("wo_s", H))
                    x = vrow("attn_alpha", H) * xbuf[rr, :] + vrow(
                        "attn_beta", H
                    ) * _bf(_rms(ao, c.post_eps))
                    xbuf[rr, :] = x
                    abuf[rr, :] = (_rms(x, c.rms_eps) * vrow("pre_ffn_w", H)).astype(
                        BF16
                    )
                    return cc

                jax.lax.fori_loop(0, tm // sub, resid, 0)
                cxo = pltpu.make_async_copy(xbuf, x_out.at[pl.ds(r0, tm)], sem_x.at[0])
                cxo.start()

                @pl.when(i == 0)
                def _():
                    cwr.wait()

                logits = _mm(abuf, wr_v, H)
                sc = jax.nn.sigmoid(logits * c.router_mult)
                ids, picks = _topk_iter(sc + vrow("expert_bias", E), K)
                tws = [
                    jnp.sum(jnp.where(pk, sc, 0.0), axis=1, keepdims=True)
                    for pk in picks
                ]
                denom = sum(tws) + c.router_eps
                lane = jax.lax.broadcasted_iota(jnp.int32, (tm, LANE), 1)
                ids_t = jnp.zeros((tm, LANE), jnp.int32)
                tw_t = jnp.zeros((tm, LANE), F32)
                for k in range(K):
                    ids_t = jnp.where(lane == k, ids[k], ids_t)
                    tw_t = jnp.where(lane == k, tws[k] / denom * c.route_scale, tw_t)
                tw_v[...] = tw_t
                ctw = pltpu.make_async_copy(tw_v, tw_h.at[pl.ds(r0, tm)], sem_m.at[6])
                ctw.start()
                # (token, slot) ids packed lane-dense: row j holds tokens 16j.. of the tile
                per_row = LANE // K
                rep = ids_t
                s = K
                while s < LANE:
                    rep = rep | pltpu.roll(rep, shift=s, axis=1)
                    s *= 2
                row = jax.lax.broadcasted_iota(jnp.int32, (tm, LANE), 0)
                sel = jnp.where(row % per_row == lane // K, rep, 0)
                idp_v[...] = jnp.sum(sel.reshape(tm // per_row, per_row, LANE), axis=1)
                cid = pltpu.make_async_copy(
                    idp_v,
                    ids_h.at[pl.ds(i * (tm // per_row), tm // per_row)],
                    sem_m.at[7],
                )
                cid.start()

                def wp_tile(j, cc):
                    b = take(i * per + n_wo + j)
                    pbuf[:, rows(j, tn)] = _bf(
                        _mm(abuf, wbuf.at[b], H)
                        * vrow("wpool_s", tn, pl.multiple_of(j * tn, tn))
                    )
                    return cc

                jax.lax.fori_loop(0, n_wp, wp_tile, 0)

                def xe_norm(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    pbuf[rr, :] = _bf(
                        _rms(pbuf[rr, :], c.rms_eps) * vrow("pre_expert_w", He)
                    )
                    return cc

                jax.lax.fori_loop(0, tm // sub, xe_norm, 0)
                cxe = pltpu.make_async_copy(pbuf, xe_h.at[pl.ds(r0, tm)], sem_x.at[1])
                cxe.start()
                for cp in (cxo, ctw, cid, cxe):
                    cp.wait()
                return carry

            jax.lax.fori_loop(0, n_rt, row_tile, 0)

        def moe(is_f8, woff, soff):
            cp = pltpu.make_async_copy(ids_h, ids_s, sem_m.at[0])
            cp.start()
            cp.wait()

            def zero(e, cc):
                cnt_s[e] = 0
                return cc

            jax.lax.fori_loop(0, E, zero, 0)

            def expert_of(s):
                return ids_s[s // LANE, s % LANE]

            def count(s, cc):
                e = expert_of(s)
                cnt_s[e] = cnt_s[e] + 1
                return cc

            jax.lax.fori_loop(0, slots_n, count, 0)

            def offsets(e, run):
                off_s[e] = run
                fill_s[e] = run
                return run + cnt_s[e]

            jax.lax.fori_loop(0, E, offsets, jnp.int32(0))

            def place(s, cc):
                e = expert_of(s)
                f = fill_s[e]
                lst_s[f] = s
                fill_s[e] = f + 1
                return cc

            jax.lax.fori_loop(0, slots_n, place, 0)

            def items(e, n):
                nb = (cnt_s[e] + rb - 1) // rb

                def blk(bb, cc):
                    ite_s[n + bb] = e
                    itb_s[n + bb] = bb
                    return cc

                jax.lax.fori_loop(0, nb, blk, 0)
                return n + nb

            n_items = jax.lax.fori_loop(0, E, items, jnp.int32(0))
            n_steps = n_items * nti

            def w_copies(idx, b, fp8):
                item, j = idx // nti, idx % nti
                e = ite_s[item]
                if fp8:
                    pw, wsz, sz2 = w8, ti // 4, FP8_SCALE_ROWS
                    gs = [
                        (s13.at[e, j, pl.ds(soff, sz2)], sbuf.at[b, pl.ds(0, sz2)]),
                        (
                            s13.at[e, nti + j, pl.ds(soff, sz2)],
                            sbuf.at[b, pl.ds(sz2, sz2)],
                        ),
                    ]
                    gs += [
                        (
                            s2.at[e, cc, pl.ds(soff, sz2)],
                            sbuf.at[b, pl.ds((2 + cc) * sz2, sz2)],
                        )
                        for cc in range(npc)
                    ]
                else:
                    pw, wsz = w4, ti // 8
                    gs = [
                        (s13.at[e, j, pl.ds(soff, s4)], sbuf.at[b, pl.ds(0, s4)]),
                        (
                            s13.at[e, nti + j, pl.ds(soff, s4)],
                            sbuf.at[b, pl.ds(s4, s4)],
                        ),
                    ]
                    gs += [
                        (
                            s2.at[
                                e,
                                cc,
                                pl.ds(
                                    pl.multiple_of(soff + j * (ti // 16), 16), ti // 16
                                ),
                            ],
                            sbuf.at[b, pl.ds(2 * s4 + cc * (ti // 16), ti // 16)],
                        )
                        for cc in range(npc)
                    ]
                ws = [
                    (w13.at[e, j, pl.ds(woff, pw)], wv(b).at[pl.ds(0, pw)]),
                    (w13.at[e, nti + j, pl.ds(woff, pw)], wv(b).at[pl.ds(pw, pw)]),
                ]
                ws += [
                    (
                        w2.at[e, cc, pl.ds(pl.multiple_of(woff + j * wsz, 16), wsz)],
                        wv(b).at[pl.ds(2 * pw + cc * wsz, wsz)],
                    )
                    for cc in range(npc)
                ]
                return [pltpu.make_async_copy(a, d, sem_w.at[b]) for a, d in ws + gs]

            def w_issue(idx, b, wait):
                for fp8 in (False, True):

                    @pl.when((is_f8 != 0) == fp8)
                    def _(fp8=fp8):
                        for cp in w_copies(idx, b, fp8):
                            cp.wait() if wait else cp.start()

            def xg(gb):
                return xbuf.at[:, pl.ds(gb * He, He)]

            def ystage(yb):
                return accbuf.at[:, pl.ds(yb * He, He)]

            def block(item):
                e, bb = ite_s[item], itb_s[item]
                return off_s[e] + bb * rb, jnp.minimum(rb, cnt_s[e] - bb * rb)

            def gather(item, gb):
                base, n = block(item)

                def one(r, cc):
                    tok = lst_s[base + r] // K
                    pltpu.make_async_copy(
                        xe_h.at[pl.ds(tok, 1)], xg(gb).at[pl.ds(r, 1)], sem_g.at[gb]
                    ).start()
                    return cc

                jax.lax.fori_loop(0, n, one, 0)
                misc_s[gb] = n

            def gather_wait(gb):
                def one(r, cc):
                    pltpu.make_async_copy(
                        xe_h.at[pl.ds(0, 1)], xg(gb).at[pl.ds(0, 1)], sem_g.at[gb]
                    ).wait()
                    return cc

                jax.lax.fori_loop(0, misc_s[gb], one, 0)

            def scatter(item, yb):
                base, n = block(item)

                def one(r, cc):
                    s = lst_s[base + r]
                    pltpu.make_async_copy(
                        ystage(yb).at[pl.ds(r, 1)],
                        y_h.at[pl.ds((s % K) * T + s // K, 1)],
                        sem_y.at[yb],
                    ).start()
                    return cc

                jax.lax.fori_loop(0, n, one, 0)
                misc_s[2 + yb] = n

            def scatter_wait(yb):
                def one(r, cc):
                    pltpu.make_async_copy(
                        ystage(yb).at[pl.ds(0, 1)], y_h.at[pl.ds(0, 1)], sem_y.at[yb]
                    ).wait()
                    return cc

                jax.lax.fori_loop(0, misc_s[2 + yb], one, 0)
                misc_s[2 + yb] = 0

            for i in range(4):
                misc_s[i] = 0
            w_issue(0, 0, False)
            gather(0, 0)

            def step(idx, carry):
                item, j = idx // nti, idx % nti
                b, gb = idx % 2, item % 2

                @pl.when(j == 0)
                def _():
                    gather_wait(gb)
                    pbuf[...] = jnp.zeros(pbuf.shape, F32)

                    @pl.when(item + 1 < n_items)
                    def _():
                        gather(item + 1, 1 - gb)

                w_issue(idx, b, True)

                @pl.when(idx + 1 < n_steps)
                def _():
                    w_issue(idx + 1, 1 - b, False)

                n_rows = block(item)[1]
                x_rows = xg(gb)
                w = wv(b)

                def expert_tile(R):
                    """This step's matmuls over the block's first R rows."""

                    def lhs(c8, n8, _fp8_lhs):
                        return x_rows[pl.ds(0, R), pl.ds(c8 * 8, n8 * 8)].astype(BF16)

                    @pl.when(is_f8 == 0)
                    def _():
                        g = _fp4_dot(
                            None,
                            w.at[pl.ds(0, w4)],
                            sbuf.at[b, pl.ds(0, s4)],
                            He,
                            ti,
                            32,
                            fast_bf16=True,
                            lhs_chunk_fn=lhs,
                            nrows=R,
                        )
                        u = _fp4_dot(
                            None,
                            w.at[pl.ds(w4, w4)],
                            sbuf.at[b, pl.ds(s4, s4)],
                            He,
                            ti,
                            32,
                            fast_bf16=True,
                            lhs_chunk_fn=lhs,
                            nrows=R,
                        )
                        hdn = (g * jax.nn.sigmoid(g) * u).astype(BF16)
                        for cc in range(npc):
                            wp = w.at[pl.ds(2 * w4 + cc * (ti // 8), ti // 8)]
                            sp = sbuf.at[b, pl.ds(2 * s4 + cc * (ti // 16), ti // 16)]
                            pbuf[pl.ds(0, R), rows(cc, ti)] += _fp4_dot(
                                hdn, wp, sp, ti, ti, 32, fast_bf16=True
                            )

                    @pl.when(is_f8 != 0)
                    def _():
                        def srow(r):
                            return sbuf[b, pl.ds(r * FP8_SCALE_ROWS, 1), :].astype(F32)

                        g = _fp8_dot(
                            None,
                            w.at[pl.ds(0, w8)],
                            srow(0),
                            He,
                            ti,
                            64,
                            lhs_chunk_fn=lhs,
                            nrows=R,
                        )
                        u = _fp8_dot(
                            None,
                            w.at[pl.ds(w8, w8)],
                            srow(1),
                            He,
                            ti,
                            64,
                            lhs_chunk_fn=lhs,
                            nrows=R,
                        )
                        hdn = (g * jax.nn.sigmoid(g) * u).astype(BF16)
                        for cc in range(npc):
                            wp = w.at[pl.ds(2 * w8 + cc * (ti // 4), ti // 4)]
                            pbuf[pl.ds(0, R), rows(cc, ti)] += _fp8_dot(
                                hdn, wp, srow(2 + cc), ti, ti, 64
                            )

                # the smallest row bucket holding the block's rows: MXU work follows them
                lo = 0
                for R in row_buckets:

                    @pl.when((n_rows > lo) & (n_rows <= R))
                    def _(R=R):
                        expert_tile(R)

                    lo = R

                @pl.when(j == nti - 1)
                def _():
                    yb = item % 2
                    scatter_wait(yb)
                    ys = ystage(yb)

                    def fin(r, cc):
                        rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                        y = _bf(pbuf[rr, :])
                        ys[rr, :] = _bf(_rms(y * vrow("post_expert_w", He), c.post_eps))
                        return cc

                    jax.lax.fori_loop(0, rb // sub, fin, 0)
                    scatter(item, yb)

                return carry

            jax.lax.fori_loop(0, n_steps, step, 0)
            scatter_wait(0)
            scatter_wait(1)

        def post_moe(sl):
            n_wu = H // tn
            total = n_rt * n_wu
            n_grp = tm // tok_per_grp

            def wcp(idx, b):
                return pltpu.make_async_copy(
                    wunpool.at[sl, idx % n_wu], wbuf.at[b, pl.ds(0, He)], sem_w.at[b]
                )

            def ycp(i, gi, yb):
                """The K slot rows of a group of tokens: slot k's into staging rows k * G."""
                t0 = i * tm + gi * tok_per_grp
                return [
                    pltpu.make_async_copy(
                        y_h.at[pl.ds(k * T + t0, tok_per_grp)],
                        accbuf.at[
                            pl.ds(k * tok_per_grp, tok_per_grp), pl.ds(yb * He, He)
                        ],
                        sem_g.at[yb],
                    )
                    for k in range(K)
                ]

            wcp(0, 0).start()

            def row_tile(i, carry):
                r0 = pl.multiple_of(i * tm, tm)
                cx = pltpu.make_async_copy(x_out.at[pl.ds(r0, tm)], xbuf, sem_x.at[0])
                ct = pltpu.make_async_copy(tw_h.at[pl.ds(r0, tm)], tw_v, sem_x.at[1])
                cx.start()
                ct.start()
                for cp in ycp(i, 0, 0):
                    cp.start()
                ct.wait()

                def comb(gi, cc):
                    yb = gi % 2
                    for cp in ycp(i, gi, yb):
                        cp.wait()

                    @pl.when(gi + 1 < n_grp)
                    def _():
                        for cp in ycp(i, gi + 1, 1 - yb):
                            cp.start()

                    stage = accbuf.at[:, pl.ds(yb * He, He)]
                    t0 = pl.multiple_of(gi * tok_per_grp, tok_per_grp)
                    for l0 in range(0, He, 512):
                        u = None
                        for k in range(K):
                            wk = tw_v[pl.ds(t0, tok_per_grp), pl.ds(k, 1)]
                            yk = stage[
                                pl.ds(k * tok_per_grp, tok_per_grp), pl.ds(l0, 512)
                            ]
                            u = wk * yk if u is None else u + wk * yk
                        abuf[pl.ds(t0, tok_per_grp), pl.ds(l0, 512)] = u.astype(BF16)
                    return cc

                jax.lax.fori_loop(0, n_grp, comb, 0)

                def wu_tile(j, cc):
                    idx = i * n_wu + j
                    b = idx % 2
                    wcp(idx, b).wait()

                    @pl.when(idx + 1 < total)
                    def _():
                        wcp(idx + 1, 1 - b).start()

                    f = _mm(abuf.at[:, pl.ds(0, He)], wbuf.at[b, pl.ds(0, He)], He)
                    accbuf[:, rows(j, tn)] = _bf(
                        f * vrow("wunpool_s", tn, pl.multiple_of(j * tn, tn))
                    )
                    return cc

                jax.lax.fori_loop(0, n_wu, wu_tile, 0)
                cx.wait()

                def resid(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    f = accbuf[rr, :] * vrow("post_ffn_w", H)
                    xbuf[rr, :] = vrow("ffn_alpha", H) * xbuf[rr, :] + vrow(
                        "ffn_beta", H
                    ) * _bf(_rms(f, c.post_eps))
                    return cc

                jax.lax.fori_loop(0, tm // sub, resid, 0)
                cxo = pltpu.make_async_copy(xbuf, x_out.at[pl.ds(r0, tm)], sem_x.at[0])
                cxo.start()
                cxo.wait()
                return carry

            jax.lax.fori_loop(0, n_rt, row_tile, 0)

        sl = ctl[2]
        base = HDR + sl * NF
        active = (ctl[1] < 0) | (ctl[1] == slots[H_STAGE])

        @pl.when(active & (slots[base + F_VALID] != 0))
        def _():
            is_g, is_f8 = slots[base + F_GLOBAL], slots[base + F_FP8]
            cv = pltpu.make_async_copy(vecs.at[sl], vec_v, sem_m.at[0])
            cv.start()
            cv.wait()
            ctx = ctl[0]
            # slab offsets are whole layers: multiples of an fp8 scale tile (16 rows)
            woff = pl.multiple_of(slots[base + F_WOFF], 16)
            soff = pl.multiple_of(slots[base + F_SOFF], 16)
            for name, run in (
                ("attn_in", lambda: attn_in(sl, is_g, ctx)),
                ("attn_out", lambda: attn_out(sl)),
                ("moe", lambda: moe(is_f8, woff, soff)),
                ("post_moe", lambda: post_moe(sl)),
            ):
                if name in phases and name not in skip:
                    run()

    vmem, smem_t = pltpu.VMEM, pltpu.SMEM
    dma = pltpu.SemaphoreType.DMA
    scratch = [
        vmem((tm, H), F32),  # xbuf: stream tile; MoE gathered rows (two halves)
        vmem((tm, H), F32),  # accbuf: projection outputs; MoE / combine staging
        vmem((tm, H), BF16),  # abuf: matmul inputs
        vmem((tm, qd), BF16),  # gbuf: attention gate
        vmem((tm, He), F32),  # pbuf: pooled rows; MoE output accumulator
        vmem((2, wr_rows, tn), FP8),  # wbuf: weight tiles (MoE words as int32)
        vmem((2, moe_s, ti), BF16),  # sbuf: MoE scales
        vmem((V_ROWS, c.vec_width), F32),
        vmem((H, E), BF16),
        vmem((tm, 2 * hd), F32),
        vmem((2, tm, tn), F32),  # rope8: lane-expanded cos / sin of a row tile
        vmem((2, tm, tn), BF16),  # otile
        vmem((tm // (LANE // K), LANE), jnp.int32),  # idp_v
        vmem((tm, LANE), F32),  # tw_v
        smem_t((slots_n // LANE, LANE), jnp.int32),  # ids_s
        smem_t((E,), jnp.int32),
        smem_t((E,), jnp.int32),
        smem_t((E,), jnp.int32),
        smem_t((slots_n,), jnp.int32),  # lst_s
        smem_t((E + slots_n // rb + 8,), jnp.int32),  # ite_s
        smem_t((E + slots_n // rb + 8,), jnp.int32),  # itb_s
        smem_t((8,), jnp.int32),  # misc_s
        dma((3,)),
        dma((2,)),
        dma((2,)),
        dma((8,)),
        dma((2,)),
        dma((2,)),
    ]
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    sds = jax.ShapeDtypeStruct
    params = pltpu.CompilerParams(vmem_limit_bytes=64 * 2**20 - 2**19)
    tag = f"t{T}" + "".join(f"_no_{p}" for p in sorted(skip))

    def build(in_names, outs, phases, aliases, name):
        names = in_names + list(outs) + list(SCRATCH)

        def kernel(*refs):
            body(dict(zip(names, refs)), phases)

        n_smem = 2
        return pl.pallas_call(
            kernel,
            out_shape=tuple(outs.values()),
            in_specs=[smem] * n_smem + [hbm] * (len(in_names) - n_smem),
            out_specs=(hbm,) * len(outs),
            scratch_shapes=scratch,
            input_output_aliases=aliases,
            compiler_params=params,
            name=name,
        )

    attn_in_call = build(
        ["ctl", "slots", "x_in", "rope", *WEIGHT_ORDER],
        {
            "q_h": sds((T, qd), BF16),
            "k_h": sds((T, kvd), BF16),
            "v_h": sds((T, kvd), BF16),
            "g_h": sds((T, qd), BF16),
        },
        ("attn_in",),
        {},
        f"muse_prefill_pp_attn_in_{tag}",
    )
    post_call = build(
        ["ctl", "slots", "x_in", "o_h", "g_in", *WEIGHT_ORDER],
        {
            "x_out": sds((T, H), F32),
            "xe_h": sds((T, He), F32),
            "ids_h": sds((slots_n // LANE, LANE), jnp.int32),
            "tw_h": sds((T, LANE), F32),
            "y_h": sds((slots_n, He), F32),
        },
        ("attn_out", "moe", "post_moe"),
        {2: 0},
        f"muse_prefill_pp_post_attn_{tag}",
    )
    return attn_in_call, post_call


SCRATCH = (
    "xbuf", "accbuf", "abuf", "gbuf", "pbuf", "wbuf", "sbuf", "vec_v", "wr_v", "rope_v",
    "rope8", "otile", "idp_v", "tw_v", "ids_s", "cnt_s", "off_s", "fill_s", "lst_s", "ite_s",
    "itb_s", "misc_s", "sem_x", "sem_w", "sem_o", "sem_m", "sem_g", "sem_y",
)  # fmt: skip


def stage_forward(
    c: PPConfig,
    ctl,
    slots,
    x,
    rope,
    weights,
    pools,
    kv_lens,
    page_tables,
    cu_q,
    dist,
    scratch: bool = False,
):
    """One stage: per slot the fused attention-input kernel, the batched RPA prefill attention
    (which also writes the chunk's K/V into the slot's pool) and the fused attention-output,
    MoE and post-MoE kernel. ctl [2] i32 = [cached tokens, active stage or -1]; pools: one per
    slot (global slots' as wide as the page table); page_tables: (global, sliding window).
    Returns (x, pools), and with scratch each slot's post_attn intermediates (expert input,
    packed ids, routing weights, slot outputs)."""
    from sglang_tpu.attention.pallas.batched_rpa import configs
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import ragged_paged_attention

    attn_in_call, post_call = stage_calls(c)
    T = c.rows
    pools, posts = list(pools), []
    for s in range(c.slots):
        ctl_s = jnp.concatenate([ctl, jnp.full((1,), s, jnp.int32)])
        is_g = s in c.global_slots
        with jax.named_scope(f"slot{s}/attn_in"):
            q, k, v, g = attn_in_call(ctl_s, slots, x, rope, *weights)
        with jax.named_scope(f"slot{s}/attention"):
            o, pools[s] = ragged_paged_attention(
                q.reshape(T, c.q_heads, c.head_dim),
                k.astype(FP8).reshape(T, c.kv_heads, c.head_dim),
                v.astype(FP8).reshape(T, c.kv_heads, c.head_dim),
                pools[s],
                kv_lens,
                page_tables[0 if is_g else 1],
                cu_q,
                dist,
                sm_scale=c.attn_scale,
                sliding_window=None if is_g else c.window,
                k_scale=1.0,
                v_scale=1.0,
                kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
                out_head_norm_eps=c.rms_eps,
            )
        with jax.named_scope(f"slot{s}/post_attn"):
            x, *rest = post_call(ctl_s, slots, x, o.reshape(T, c.q_dim), g, *weights)
        posts.append(rest)
    return (x, pools, posts) if scratch else (x, pools)
