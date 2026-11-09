# SPDX-License-Identifier: Apache-2.0
"""Muse Spark prefill, pipeline-parallel: one stage's layers whole on one device, no collectives.

A stage holds its layers whole (every head, all experts, the full pooler and unpooler) on one
device, so nothing in it talks to another device; the residual stream moves between stages
outside. Per layer slot:

  attn_in    (fused Pallas) qkvg projection, qk norms, rope -> q, k, v, gate
  attention  sglang_tpu's batched RPA prefill kernel, as the P/D prefill serves it: the
             chunk's K/V into the slot's fp8 SEQ_ALONG_LANE pool, attention over the pool
             (cached context and the causal chunk), each head's output RMS-normalized
  route      (fused Pallas) gated o projection and residual, router top-k, expert projection
  experts    sglang_tpu's fused MoE (sort, gather, gmm_v2 x2, un-permute), as served:
             NVFP4 W4A16, or fp8 W8A8 on the stage's fp8 layer
  combine    (fused Pallas) post-expert norm per slot, weighted slot sum, unpooler, gated
             residual

Every stage has the same slot layout (cfg.slots layers of dense weights, global attention at
cfg.global_slots, NVFP4 experts per slot and one fp8 expert set), so one program serves every
stage; a per-stage slot table says which slots hold a layer and which take the fp8 experts.
"""

import os

import jax
import jax.numpy as jnp
import msgspec
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.kernel import _heads_rms, _rms, _topk_iter

LANE = 128
PHASES = ("attn_in", "attention", "route", "experts", "combine")
FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32

# slot table: a header, then NF int32 fields per slot
HDR = 8
H_STAGE, H_LAYERS = 0, 1
NF = 4
F_VALID, F_GLOBAL, F_FP8, F_LAYER = range(4)

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
    max_position: int
    # slots of the global attention layers (the rest are sliding-window)
    global_slots: tuple = (2, 6)
    # dense row tile and weight column tile
    tm: int = 256
    tn: int = 512

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
        """Raises ValueError for a geometry the stage kernels do not tile."""
        H, He, T = self.hidden, self.expert_hidden, self.rows
        checks = [
            (H >= 2 * He and self.q_dim >= 2 * He, "combine staging halves"),
            (
                self.tm % self.top_k == 0 and (self.tm // self.top_k) % 8 == 0,
                "row tile",
            ),
            (T % self.tm == 0 and T % self.page == 0, "chunk rows"),
            (
                self.qkvg_dim % self.tn == 0 and H % self.tn == 0 and He % self.tn == 0,
                "tn",
            ),
            (self.q_dim % self.tn == 0 and self.kv_dim % self.tn == 0, "head tiles"),
            (self.n_experts <= 2 * LANE and self.n_experts % LANE == 0, "router lanes"),
            (self.top_k <= LANE, "routing lanes"),
            (self.q_heads % self.kv_heads == 0, "GQA"),
        ]
        bad = [why for ok, why in checks if not ok]
        if bad:
            raise ValueError(f"PPConfig {self}: {'; '.join(bad)}")


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
    """The stage's slot table [HDR + slots * NF] int32. Its layers take consecutive slots
    from the one that puts its global layers on cfg.global_slots.

    Raises:
      ValueError: The layers do not fit the stage's slots, their global layers cannot sit on
        cfg.global_slots, or more than one of them has fp8 experts.
    """
    globals_at = [i for i, layer in enumerate(layers) if layer in global_layers]
    first = c.global_slots[0] - globals_at[0] if globals_at else 0
    placed = [first + i for i in globals_at]
    if first < 0 or first + len(layers) > c.slots or placed != list(c.global_slots):
        raise ValueError(
            f"stage {stage}: layers {layers} (global {[layers[i] for i in globals_at]}) do not "
            f"fit {c.slots} slots with global attention at {c.global_slots}"
        )
    if sum(kinds[layer] == "fp8" for layer in layers) > 1:
        raise ValueError(f"stage {stage}: one fp8 expert set per stage")
    t = np.zeros(HDR + c.slots * NF, np.int32)
    t[H_STAGE], t[H_LAYERS] = stage, len(layers)
    for i, layer in enumerate(layers):
        f = t[HDR + (first + i) * NF : HDR + (first + i + 1) * NF]
        f[F_VALID], f[F_GLOBAL] = 1, int(layer in global_layers)
        f[F_FP8], f[F_LAYER] = int(kinds[layer] == "fp8"), layer
    return t


def weight_shapes(c: PPConfig) -> dict[str, tuple[tuple[int, ...], jnp.dtype]]:
    """One stage's dense weights (per device), column-tile major: each [K, tn] tile is one
    contiguous block."""
    S, H, He, E, tn = c.slots, c.hidden, c.expert_hidden, c.n_experts, c.tn
    return {
        "wqkvg": ((S, c.qkvg_dim // tn, H, tn), FP8),
        "wo": ((S, H // tn, c.q_dim, tn), FP8),
        "wpool": ((S, He // tn, H, tn), FP8),
        "wunpool": ((S, H // tn, He, tn), FP8),
        "wrouter": ((S, H, E), BF16),
        "vecs": ((S, V_ROWS, c.vec_width), F32),
    }


WEIGHT_ORDER = ("wqkvg", "wo", "wpool", "wunpool", "wrouter", "vecs")


def expert_shapes(c: PPConfig, kind: str) -> list[tuple[tuple[int, ...], jnp.dtype]]:
    """(w13, w13_scale, w2, w2_scale) of one layer in the fused MoE's serving layouts: NVFP4 as
    e2m1 codes packed two per byte along K with bf16 block scales (16 along K) paired into
    int32 words; fp8 [E, K, N] with f32 channel scales."""
    E, He, I = c.n_experts, c.expert_hidden, c.inter
    if kind == "fp4":
        return [
            ((E, He // 2, 2 * I), jnp.uint8),
            ((E, He // 32, 2 * I), jnp.int32),
            ((E, I // 2, He), jnp.uint8),
            ((E, I // 32, He), jnp.int32),
        ]
    if kind == "fp8":
        return [
            ((E, He, 2 * I), FP8),
            ((E, 1, 1, 2 * I), F32),
            ((E, I, He), FP8),
            ((E, 1, 1, He), F32),
        ]
    raise ValueError(f"expert kind {kind}")


def pool_shape(c: PPConfig, pages: int) -> tuple[int, ...]:
    return (pages, 2 * c.kv_heads, c.head_dim // 4, 4, c.page)


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


SCRATCH = (
    "xbuf", "accbuf", "abuf", "gbuf", "pbuf", "wbuf", "vec_v", "wr_v", "rope_v", "rope8",
    "otile", "ids_v", "tw_v", "misc_s", "sem_x", "sem_w", "sem_o", "sem_m",
)  # fmt: skip


def stage_calls(c: PPConfig):
    """The stage's fused calls, for the slot in ctl[2] (ctl [3] i32 = [cached tokens, active
    stage or -1, slot]); a device whose stage is not active, or a slot without a layer, returns
    at once (outputs unwritten):
      attn_in(ctl, slot table, x [T, H] f32, rope [max_position, 2 hd] f32, *weights) ->
        (q [T, q_dim] bf16, k, v [T, kv_dim] bf16, gate [T, q_dim] bf16)
      route(ctl, slot table, x, o [T, q_dim] bf16, gate, *weights) ->
        (x, expert input [T, He] bf16, ids [T, 128] i32, weights [T, 128] f32), lanes past
        top_k zero, x updated in place
      combine(ctl, slot table, x, y [T * top_k, He] bf16 (token-major slots), weights,
        *weights) -> (x,), x updated in place"""
    c.validate()
    skip = skipped_phases()
    T, H, He, E, K = c.rows, c.hidden, c.expert_hidden, c.n_experts, c.top_k
    hd = c.head_dim
    qd, kvd, nq = c.q_dim, c.kv_dim, c.qkvg_dim
    tm, tn = c.tm, c.tn
    n_rt = T // tm
    sub = 16  # rows per vector pass over a [tm, H] tile
    n_qt, n_kt = qd // tn, kvd // tn
    ntc = nq // tn
    G = tm // K  # tokens per combine group: their K slot rows fill a tile
    wr_rows = max(H, qd, He)

    def body(R, phases):
        ctl, slots, x_in = R["ctl"], R["slots"], R["x_in"]
        wqkvg, wo, wpool, wunpool, wrouter, vecs = (R[n] for n in WEIGHT_ORDER)
        xbuf, accbuf, abuf, gbuf, pbuf, wbuf = (R[n] for n in SCRATCH[:6])
        vec_v, wr_v, rope_v, rope8, otile, ids_v, tw_v = (R[n] for n in SCRATCH[6:13])
        misc_s, sem_x, sem_w, sem_o, sem_m = (R[n] for n in SCRATCH[13:])
        x_out = x_in  # the stream is aliased in place

        def vrow(name, n, off=0):
            return vec_v[pl.ds(V[name], 1), pl.ds(off, n)]

        def rows(i, n):
            return pl.ds(pl.multiple_of(i * n, n), n)

        def attn_in(sl, is_g, ctx):
            rope, q_h, k_h, v_h, g_h = (
                R[n] for n in ("rope", "q_h", "k_h", "v_h", "g_h")
            )
            total = n_rt * ntc

            def wcp(idx, b):
                return pltpu.make_async_copy(
                    wqkvg.at[sl, idx % ntc], wbuf.at[b, pl.ds(0, H)], sem_w.at[b]
                )

            def out_cp(b, dst):
                return pltpu.make_async_copy(otile.at[b], dst, sem_o.at[b])

            def emit(acc, dst):
                """A [tm, tn] output tile through the double-buffered otile."""
                n = misc_s[0]
                b = n % 2

                @pl.when(n >= 2)
                def _():
                    out_cp(b, q_h.at[pl.ds(0, tm), pl.ds(0, tn)]).wait()

                otile[b] = acc.astype(BF16)
                out_cp(b, dst).start()
                misc_s[0] = n + 1

            misc_s[0] = 0
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

                @pl.when(misc_s[0] > b)
                def _(b=b):
                    out_cp(b, q_h.at[pl.ds(0, tm), pl.ds(0, tn)]).wait()

        def route(sl):
            o_h, g_in, xe_h, ids_h, tw_h = (
                R[n] for n in ("o_h", "g_in", "xe_h", "ids_h", "tw_h")
            )
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

            cwr = pltpu.make_async_copy(wrouter.at[sl], wr_v, sem_m.at[1])
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
                ids_v[...] = ids_t
                tw_v[...] = tw_t
                cid = pltpu.make_async_copy(ids_v, ids_h.at[pl.ds(r0, tm)], sem_m.at[2])
                ctw = pltpu.make_async_copy(tw_v, tw_h.at[pl.ds(r0, tm)], sem_m.at[3])
                cid.start()
                ctw.start()

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
                    gbuf[rr, pl.ds(0, He)] = (
                        _rms(pbuf[rr, :], c.rms_eps) * vrow("pre_expert_w", He)
                    ).astype(BF16)
                    return cc

                jax.lax.fori_loop(0, tm // sub, xe_norm, 0)
                cxe = pltpu.make_async_copy(
                    gbuf.at[:, pl.ds(0, He)], xe_h.at[pl.ds(r0, tm)], sem_x.at[1]
                )
                cxe.start()
                for cp in (cxo, cid, ctw, cxe):
                    cp.wait()
                return carry

            jax.lax.fori_loop(0, n_rt, row_tile, 0)

        def combine(sl):
            y_h, tw_in = R["y_h"], R["tw_in"]
            n_wu = H // tn
            total = n_rt * n_wu
            n_grp = tm // G

            def wcp(idx, b):
                return pltpu.make_async_copy(
                    wunpool.at[sl, idx % n_wu], wbuf.at[b, pl.ds(0, He)], sem_w.at[b]
                )

            def ycp(i, gi, yb):
                """A group's K slot rows per token (token-major) into a staging half."""
                return pltpu.make_async_copy(
                    y_h.at[pl.ds((i * tm + gi * G) * K, tm)],
                    gbuf.at[:, pl.ds(yb * He, He)],
                    sem_m.at[4 + yb],
                )

            wcp(0, 0).start()

            def row_tile(i, carry):
                r0 = pl.multiple_of(i * tm, tm)
                cx = pltpu.make_async_copy(x_out.at[pl.ds(r0, tm)], xbuf, sem_x.at[0])
                ct = pltpu.make_async_copy(tw_in.at[pl.ds(r0, tm)], tw_v, sem_x.at[1])
                cx.start()
                ct.start()
                ycp(i, 0, 0).start()
                ct.wait()

                def group(gi, cc):
                    yb = gi % 2
                    ycp(i, gi, yb).wait()

                    @pl.when(gi + 1 < n_grp)
                    def _():
                        ycp(i, gi + 1, 1 - yb).start()

                    stage = gbuf.at[:, pl.ds(yb * He, He)]

                    def tokens(r, c2):
                        # 8 tokens' K slot rows (an f32 row tile): normalize, weight, sum the slots
                        nt = 8
                        t0 = pl.multiple_of(gi * G + r * nt, nt)
                        y = stage[pl.ds(pl.multiple_of(r * nt * K, nt * K), nt * K), :]
                        yn = _bf(
                            _rms(y.astype(F32) * vrow("post_expert_w", He), c.post_eps)
                        )
                        tw = tw_v[pl.ds(t0, nt), :]
                        tw_rows = jnp.broadcast_to(
                            tw[:, None, :], (nt, K, LANE)
                        ).reshape(nt * K, LANE)
                        slot = (
                            jax.lax.broadcasted_iota(jnp.int32, (nt * K, LANE), 0) % K
                        )
                        lane = jax.lax.broadcasted_iota(jnp.int32, (nt * K, LANE), 1)
                        w = jnp.sum(
                            jnp.where(lane == slot, tw_rows, 0.0), axis=1, keepdims=True
                        )
                        accbuf[pl.ds(t0, nt), pl.ds(0, He)] = jnp.sum(
                            (yn * w).reshape(nt, K, He), axis=1
                        )
                        return c2

                    jax.lax.fori_loop(0, G // 8, tokens, 0)
                    return cc

                jax.lax.fori_loop(0, n_grp, group, 0)

                def to_bf16(r, cc):
                    rr = pl.ds(pl.multiple_of(r * sub, sub), sub)
                    abuf[rr, pl.ds(0, He)] = accbuf[rr, pl.ds(0, He)].astype(BF16)
                    return cc

                jax.lax.fori_loop(0, tm // sub, to_bf16, 0)

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
            is_g = slots[base + F_GLOBAL]
            cv = pltpu.make_async_copy(vecs.at[sl], vec_v, sem_m.at[0])
            cv.start()
            cv.wait()
            for name, run in (
                ("attn_in", lambda: attn_in(sl, is_g, ctl[0])),
                ("route", lambda: route(sl)),
                ("combine", lambda: combine(sl)),
            ):
                if name in phases and name not in skip:
                    run()

    vmem, smem_t = pltpu.VMEM, pltpu.SMEM
    dma = pltpu.SemaphoreType.DMA
    scratch = [
        vmem((tm, H), F32),  # xbuf: stream tile
        vmem((tm, H), F32),  # accbuf: projection outputs; the combined slots
        vmem((tm, H), BF16),  # abuf: matmul inputs
        vmem((tm, qd), BF16),  # gbuf: attention gate; expert input; slot staging halves
        vmem((tm, He), F32),  # pbuf: pooled rows
        vmem((2, wr_rows, tn), FP8),  # wbuf: weight tiles
        vmem((V_ROWS, c.vec_width), F32),
        vmem((H, E), BF16),
        vmem((tm, 2 * hd), F32),
        vmem((2, tm, tn), F32),  # rope8: lane-expanded cos / sin of a row tile
        vmem((2, tm, tn), BF16),  # otile
        vmem((tm, LANE), jnp.int32),  # ids_v
        vmem((tm, LANE), F32),  # tw_v
        smem_t((8,), jnp.int32),  # misc_s
        dma((3,)),
        dma((2,)),
        dma((2,)),
        dma((8,)),
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

        return pl.pallas_call(
            kernel,
            out_shape=tuple(outs.values()),
            in_specs=[smem] * 2 + [hbm] * (len(in_names) - 2),
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
    route_call = build(
        ["ctl", "slots", "x_in", "o_h", "g_in", *WEIGHT_ORDER],
        {
            "x_out": sds((T, H), F32),
            "xe_h": sds((T, He), BF16),
            "ids_h": sds((T, LANE), jnp.int32),
            "tw_h": sds((T, LANE), F32),
        },
        ("route",),
        {2: 0},
        f"muse_prefill_pp_route_{tag}",
    )
    combine_call = build(
        ["ctl", "slots", "x_in", "y_h", "tw_in", *WEIGHT_ORDER],
        {"x_out": sds((T, H), F32)},
        ("combine",),
        {2: 0},
        f"muse_prefill_pp_combine_{tag}",
    )
    return attn_in_call, route_call, combine_call


def experts(c: PPConfig, xe, ids, tw, ex):
    """sglang_tpu's fused MoE as served (no combine): [T, top_k, He] bf16 slot outputs."""
    from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

    w13, s13, w2, s2 = ex
    return _fused_moe_jax(
        xe, w13, w2, tw[:, : c.top_k].astype(BF16), ids[:, : c.top_k], s13, s2, None, None,
        c.top_k, "silu", no_combine=True, use_sc_gather=1,
    )  # fmt: skip


def stage_forward(
    c: PPConfig,
    ctl,
    slots,
    x,
    rope,
    weights,
    experts_fp4,
    experts_fp8,
    pools,
    kv_lens,
    page_tables,
    cu_q,
    dist,
    scratch: bool = False,
):
    """One stage. Per slot: attn_in, the batched RPA prefill attention (which also writes the
    chunk's K/V into the slot's pool), route, the fused MoE (the slot's NVFP4 experts, or the
    stage's fp8 set where the slot table says so), combine. ctl [2] i32 = [cached tokens,
    active stage or -1]; experts_fp4: per slot (w13, w13_scale, w2, w2_scale); pools: one per
    slot (global slots' as wide as the page table); page_tables: (global, sliding window).
    Returns (x, pools), and with scratch each slot's (ids, routing weights)."""
    from sglang_tpu.attention.pallas.batched_rpa import configs
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import ragged_paged_attention

    skip = skipped_phases()
    attn_in_call, route_call, combine_call = stage_calls(c)
    T, K = c.rows, c.top_k
    pools, routes = list(pools), []
    for s in range(c.slots):
        ctl_s = jnp.concatenate([ctl, jnp.full((1,), s, jnp.int32)])
        is_g = s in c.global_slots
        with jax.named_scope(f"slot{s}/attn_in"):
            q, k, v, g = attn_in_call(ctl_s, slots, x, rope, *weights)
        with jax.named_scope(f"slot{s}/attention"):
            if "attention" in skip:
                o = jnp.zeros((T, c.q_dim), BF16)
            else:
                o, pools[s] = ragged_paged_attention(
                    q.reshape(T, c.q_heads, c.head_dim),
                    k.astype(FP8).reshape(T, c.kv_heads, c.head_dim),
                    v.astype(FP8).reshape(T, c.kv_heads, c.head_dim),
                    pools[s], kv_lens, page_tables[0 if is_g else 1], cu_q, dist,
                    sm_scale=c.attn_scale, sliding_window=None if is_g else c.window,
                    k_scale=1.0, v_scale=1.0, kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
                    out_head_norm_eps=c.rms_eps,
                )  # fmt: skip
        with jax.named_scope(f"slot{s}/route"):
            x, xe, ids, tw = route_call(
                ctl_s, slots, x, o.reshape(T, c.q_dim), g, *weights
            )
        with jax.named_scope(f"slot{s}/experts"):
            if "experts" in skip:
                y = jnp.zeros((T, K, c.expert_hidden), BF16)
            else:
                is_fp8 = slots[HDR + s * NF + F_FP8] != 0
                y = jax.lax.cond(
                    is_fp8,
                    lambda: experts(c, xe, ids, tw, experts_fp8),
                    lambda: experts(c, xe, ids, tw, experts_fp4[s]),
                )
        with jax.named_scope(f"slot{s}/combine"):
            (x,) = combine_call(
                ctl_s, slots, x, y.reshape(T * K, c.expert_hidden), tw, *weights
            )
        routes.append((ids, tw))
    return (x, pools, routes) if scratch else (x, pools)
