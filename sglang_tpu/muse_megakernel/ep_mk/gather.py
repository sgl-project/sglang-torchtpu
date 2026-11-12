# SPDX-License-Identifier: Apache-2.0
"""The EP layout's small-row arm: every rank holds every row (no sequence parallelism, no
count exchange, no ragged all-to-all). Per layer:

  o-proj partials --RS by column--> o tile --AG--> every row's post-attention residual
  pool, own He/tp columns --AG--> pre-expert norm of every row
  32 local experts over the (row, slot) pairs routed to them; per-slot post-expert norm and
  router weight applied here (a slot's whole expert output lives on one rank)
  u partials (f32) --RS by column--> u (bf16) --AG--> unpool, own H/tp columns --AG--> FFN
  residual of every row, then the next layer's pre-attention input.

The dense matmuls read only this rank's columns of the replicated wpool / wunpool.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.envs import dev_int
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_megakernel.ep_mk import route
from sglang_tpu.muse_megakernel.ep_mk.config import (
    EPPlan,
    ep_buffers,
    gather_plan,
    item_scratch,
)
from sglang_tpu.muse_megakernel.ep_mk.dense import stream_matmul, stream_prefetch
from sglang_tpu.muse_megakernel.ep_mk.experts import item_sizes, sized
from sglang_tpu.muse_megakernel.ep_mk.layer import _phase, expert_loop
from sglang_tpu.muse_megakernel.mk.config import MKConfig
from sglang_tpu.muse_megakernel.mk.consts import LANE, SMALL
from sglang_tpu.muse_megakernel.mk.hooks import EPLayerStep, EPOProjStep
from sglang_tpu.muse_megakernel.mk.numerics import _rms
from sglang_tpu.muse_megakernel.mk.packing import _pack_bf16x2, _unpack_bf16x2

f32, bf16 = jnp.float32, jnp.bfloat16

# collective phases: one receive / send semaphore each
PH_ORS, PH_OAG, PH_PAG, PH_URS, PH_UAG, PH_FAG = range(6)
_PH_NAMES = ("o_rs", "o_ag", "pool_ag", "u_rs", "u_ag", "f_ag")
_TP_UNUSED = {
    "route_v",
    "route_s",
    "ids_s",
    "ystage",
    "ystage_sems",
    "yrow",
    "yrow_sems",
}


def _row(smr, name, n):
    return smr[SMALL[name] : SMALL[name] + 1, :n]


def _lane_pad(n: int) -> int:
    return -(-n // LANE) * LANE


class EPGatherExperts:
    """EPHook: the EP layout's experts over every row, rows replicated on every rank."""

    def __init__(self, cfg: MKConfig, item_rows: int = 32):
        self.plan = gather_plan(cfg, item_rows)

    def scratch_plan(self, cfg: MKConfig, plan: list) -> list:
        return [e for e in plan if e[0] not in _TP_UNUSED] + gather_scratch(
            cfg, self.plan
        )

    def o_proj(self, step: EPOProjStep) -> None:
        _o_proj(step)

    def layer_moe(self, step: EPLayerStep) -> None:
        c, S, p = step.c, step.S, self.plan
        n_items = []

        def routing() -> None:  # under the pooled columns' all-gather
            with _phase("router"):
                step.router()
                route.own_topk(c, p, S, step.smr, (0, c.rows, 0))
                _places(c, p, S, step.n_live, step.me)
            with _phase("items"):
                n_items.append(_build_items(c, p, S, step.n_live))

        _pool(step, routing)
        S["eg_u"][...] = jnp.zeros(S["eg_u"].shape, f32)
        with _phase("experts"):
            expert_loop(
                step, p, n_items[0], lambda: None, x_src="ep_xo", emit=_emit(step, p)
            )
        # the unpool's first weight tiles stream in under the u reduce-scatter / all-gather
        stream_prefetch(
            S,
            step.W["wunpool"],
            step.layer,
            c.expert_hidden,
            c.hidden_local,
            step.me * c.hidden_local,
            c.hidden_local,
        )
        with _phase("u_rs"):
            _u_reduce(step)
        with _phase("unpool"):
            _unpool(step)
        nsmr = None if step.next_smalls is None else step.next_smalls()
        with _phase("residual"):
            _ffn_residual(step, nsmr)

    def pre_attn(self, c: MKConfig, S: dict) -> None:
        """layer_moe left the next layer's pre-attention input in S["lhs"]."""


def gather_scratch(c: MKConfig, p: EPPlan) -> list:
    R, K, tp, He, Hl = c.rows, c.top_k, c.tp, c.expert_hidden, c.hidden_local
    vmem, smem = pltpu.VMEM, pltpu.SMEM
    u32, i32 = jnp.uint32, jnp.int32
    dma = pltpu.SemaphoreType.DMA
    rp = _lane_pad(R)
    return item_scratch(c, p) + [
        # o-proj: column tiles received from every rank, the two send buffers, the summed
        # tiles of every owner (bf16 pairs)
        ("eg_ors", vmem((tp, R, Hl // 2), u32)),
        ("eg_osend", vmem((2, R, Hl // 2), u32)),
        ("eg_oag", vmem((tp, R, Hl // 2), u32)),
        # pooled columns of every owner (bf16 pairs), the pre-expert-normed expert input
        ("eg_pag", vmem((tp, R, He // tp // 2), u32)),
        ("ep_xo", vmem((R, He // 2), u32)),
        ("ep_ids", vmem((R, LANE), i32)),
        ("ep_wsel", vmem((R, LANE), f32)),
        # [slot k, row]: the slot's place among this rank's routed slots (-1: elsewhere), its
        # router weight; [0 / 1, local expert]: slots, first place
        ("eg_pos_v", vmem((8, rp), i32)),
        ("eg_pos_s", smem((8, rp), i32)),
        ("eg_w_v", vmem((8, rp), f32)),
        ("eg_w_s", smem((8, rp), f32)),
        ("eg_tab_v", vmem((8, LANE), i32)),
        ("eg_tab_s", smem((8, LANE), i32)),
        ("ep_items", smem((1 + 3 * p.max_items,), i32)),
        (
            "ep_rows",
            smem((R * K + 1,), i32),
        ),  # places by expert: row | slot << ROW_BITS
        # this rank's weighted slot sums, every owner's u partials, u of every owner (bf16
        # pairs), the unpool columns of every owner (bf16 pairs)
        ("eg_u", vmem((R, He), f32)),
        ("eg_urs", vmem((tp, R, He // tp), f32)),
        ("eg_uag", vmem((tp, R, He // tp // 2), u32)),
        ("eg_fag", vmem((tp, R, Hl // 2), u32)),
        ("eg_sc", vmem((8, Hl), f32)),  # own columns' pool / unpool scales
        ("ep_w_sems", dma((p.nslot, 4))),
        ("ep_dn_sems", dma((p.nslot,))),
        ("ep_x_sems", dma((2,))),
        ("ep_misc_sems", dma((4,))),
        ("eg_rsems", dma((6,))),
        ("eg_psems", dma((6, tp))),  # pair all-gathers: per phase, per source block
        ("eg_ssems", dma((6,))),
        ("eg_osent", dma((2,))),
        ("eg_lsems", dma((2,))),
    ]


def _wait(tmpl, n: int) -> None:
    for _ in range(n):
        tmpl.wait()


# all-gathers of steps from this many rows relay through the partner core: measured 2-4%
# faster steps from 48 rows, 1.6-2.4% slower at 16 (latency-bound, one more hop)
PAIR_AG_MIN_ROWS = 48


def _pair_ag(c: MKConfig) -> bool:
    """All-gathers relay through the partner core (4 chips x 2 cores: half the ICI bytes);
    SGLANG_TPU_EP_GATHER_PAIR_MIN_ROWS overrides the row bound (A/B, read at trace time)."""
    lo = dev_int(tpu_envs.SGLANG_TPU_EP_GATHER_PAIR_MIN_ROWS, PAIR_AG_MIN_ROWS)
    return c.tp == 8 and c.rows >= lo


def _all_gather(
    c: MKConfig, S: dict, me, buf, ph: int, overlap: Callable[[], None] | None = None
) -> None:
    """buf[me] (filled here) into every rank's buf[me]; overlap() runs once the own block is
    on its way."""
    rsem, ssem = S["eg_rsems"].at[ph], S["eg_ssems"].at[ph]
    if _pair_ag(c):
        _all_gather_pair(S, me, buf, ph, overlap)
        return
    with _phase(_PH_NAMES[ph]):
        for d in range(1, c.tp):
            pltpu.make_async_remote_copy(
                src_ref=buf.at[me],
                dst_ref=buf.at[me],
                send_sem=ssem,
                recv_sem=rsem,
                device_id=((me + d) % c.tp,),
            ).start()
    if overlap is not None:
        overlap()
    with _phase(_PH_NAMES[ph] + "_wait"):
        _wait(pltpu.make_async_copy(buf.at[0], buf.at[0], rsem), c.tp - 1)
        _wait(pltpu.make_async_copy(buf.at[0], buf.at[0], ssem), c.tp - 1)


def _all_gather_pair(
    S: dict, me, buf, ph: int, overlap: Callable[[], None] | None
) -> None:
    """Own block to the partner core and to the same core of the 3 other chips; each block
    from another chip is relayed to the partner on arrival."""
    core, chip, partner = me % 2, me // 2, jnp.bitwise_xor(me, 1)
    psem, ssem = S["eg_psems"].at[ph], S["eg_ssems"].at[ph]

    def send(k, peer):
        pltpu.make_async_remote_copy(
            src_ref=buf.at[k],
            dst_ref=buf.at[k],
            send_sem=ssem,
            recv_sem=psem.at[k],
            device_id=(peer,),
        ).start()

    def arrived(k):
        pltpu.make_async_copy(buf.at[0], buf.at[0], psem.at[k]).wait()

    far = [2 * jnp.bitwise_xor(chip, d) + core for d in (1, 2, 3)]
    with _phase(_PH_NAMES[ph]):
        send(me, partner)
        for k in far:
            send(me, k)
    if overlap is not None:
        overlap()
    with _phase(_PH_NAMES[ph] + "_wait"):
        for k in far:
            arrived(k)
            send(k, partner)
        arrived(partner)
        for k in far:
            arrived(jnp.bitwise_xor(k, 1))
        _wait(pltpu.make_async_copy(buf.at[0], buf.at[0], ssem), 7)


def _local_copy(S: dict, src, dst) -> None:
    cp = pltpu.make_async_copy(src, dst, S["eg_lsems"].at[0])
    cp.start()
    cp.wait()


def _stream_tiles(c: MKConfig, S: dict, fn) -> None:
    """fn(ref, t) on each row tile t of S["h_stream"] ([RT, H] f32), written back."""
    RT, hs = c.row_tile, S["h_stream"]
    if c.tiled_vmem_pads:
        for t in range(c.rows // RT):
            fn(hs.at[pl.ds(t * RT, RT)], t)
        return
    stage = S["sin"].at[0]
    for t in range(c.rows // RT):
        _local_copy(S, hs.at[pl.ds(t * RT, RT)], stage)
        fn(stage, t)
        _local_copy(S, stage, hs.at[pl.ds(t * RT, RT)])


def _blocks(buf, rows, n: int):
    """[rows, n * W] f32: the bf16-pair blocks buf[0..n) side by side."""
    return jnp.concatenate([_unpack_bf16x2(buf[d, rows]) for d in range(n)], axis=1)


def _o_proj(step: EPOProjStep) -> None:
    """Every row: o = sum over ranks of the o-proj partials (column tile m summed on rank m,
    then all-gathered), h_stream = attn_alpha * stream + attn_beta * norm(o) and S["lhs"] =
    the pre-FFN-normed stream."""
    c, S, smr, me = step.c, step.S, step.smr, step.me
    tp, H, Hl, RT = c.tp, c.hidden, c.hidden_local, c.row_tile
    recv, send = S["eg_ors"], S["eg_osend"]
    rsem = S["eg_rsems"].at[PH_ORS]

    def sent(b):
        return pltpu.make_async_copy(send.at[0], recv.at[0], S["eg_osent"].at[b])

    def ship(ni, col):
        # one copy per tile completes on eg_osent[b]: the remote send, or the local copy
        b = ni % 2
        if ni >= 2:
            sent(b).wait()
        send[b] = _pack_bf16x2(S["acc"][:, pl.ds(0, Hl)])

        @pl.when(col == me)
        def _():
            pltpu.make_async_copy(send.at[b], recv.at[me], S["eg_osent"].at[b]).start()

        @pl.when(col != me)
        def _():
            pltpu.make_async_remote_copy(
                src_ref=send.at[b],
                dst_ref=recv.at[me],
                send_sem=S["eg_osent"].at[b],
                recv_sem=rsem,
                device_id=(col,),
            ).start()

    with _phase("o_proj_mv"):
        step.mv(ship)
    n_tiles = H // Hl
    with _phase("o_rs"):
        for ni in range(max(0, n_tiles - 2), n_tiles):
            sent(ni % 2).wait()
        _wait(pltpu.make_async_copy(recv.at[0], recv.at[0], rsem), tp - 1)
    oag = S["eg_oag"]
    for t in range(c.rows // RT):
        rows = pl.ds(t * RT, RT)
        o = sum(_unpack_bf16x2(recv[src, rows]) for src in range(tp))
        oag.at[me][rows, :] = _pack_bf16x2(o)
    _all_gather(c, S, me, oag, PH_OAG)

    a_alpha, a_beta = _row(smr, "attn_alpha", H), _row(smr, "attn_beta", H)
    pre_ffn = _row(smr, "pre_ffn_w", H)

    def residual(hs, t):
        rows = pl.ds(t * RT, RT)
        a = _blocks(oag, rows, tp)  # [RT, H], bf16 values
        s_a = jax.lax.rsqrt(
            jnp.mean(jnp.square(a), axis=-1, keepdims=True) + c.post_eps
        )
        s = a_alpha * hs[...] + a_beta * (a * s_a).astype(bf16).astype(f32)
        hs[...] = s
        S["lhs"][rows, :] = _rms(s, c.rms_eps, pre_ffn).astype(bf16)

    _stream_tiles(c, S, residual)


def _places(c: MKConfig, p: EPPlan, S: dict, n_live, me) -> None:
    """eg_pos / eg_w [k, row]: each live slot routed to this rank's experts, its place in
    ep_rows (slots by local expert, rows ascending) or -1, and its router weight; eg_tab
    rows 0 / 1: slots per local expert and the first place of each. Small exact integers
    through f32 matmuls of one-hots against prefix masks (route.send_places' method)."""
    K, R, el = c.top_k, c.rows, p.experts
    i32 = jnp.int32
    ids = S["ep_ids"][:, pl.ds(0, K)] - me * el
    wsel = S["ep_wsel"][:, pl.ds(0, K)]
    r_col = jax.lax.broadcasted_iota(i32, (R, 1), 0)
    lane = jax.lax.broadcasted_iota(i32, (R, LANE), 1)
    ok = (lane < el) & (r_col < n_live)
    onehots = [((ids[:, k : k + 1] == lane) & ok).astype(f32) for k in range(K)]
    hit = sum(onehots)  # [R, LANE]: a row picks a local expert at most once
    count = jnp.sum(hit, axis=0, keepdims=True)  # [1, LANE]
    l_r = jax.lax.broadcasted_iota(i32, (LANE, LANE), 0)
    l_c = jax.lax.broadcasted_iota(i32, (LANE, LANE), 1)
    base = route._dot(count, (l_r < l_c).astype(f32))  # [1, LANE]
    tri = (jax.lax.broadcasted_iota(i32, (R, R), 1) < r_col).astype(f32)
    place = base + route._dot(tri, hit)  # [R, LANE]
    pos = jnp.concatenate(
        [
            jnp.where(
                jnp.sum(oh, axis=1, keepdims=True) > 0,
                jnp.sum(oh * place, axis=1, keepdims=True),
                -1.0,
            )
            for oh in onehots
        ],
        axis=1,
    ).astype(i32)  # [R, K]
    rp = _lane_pad(R)

    def by_slot(x, dt):  # [R, K] -> [8, rp]: slot k on row k
        xt = jnp.transpose(jnp.concatenate([x, jnp.zeros((R, LANE - K), dt)], axis=1))[
            :8
        ]
        if rp > R:
            xt = jnp.concatenate([xt, jnp.zeros((8, rp - R), dt)], axis=1)
        return xt

    S["eg_pos_v"][...] = by_slot(pos, i32)
    S["eg_w_v"][...] = by_slot(wsel, f32)
    S["eg_tab_v"][...] = jnp.concatenate(
        [count.astype(i32), base.astype(i32), jnp.zeros((6, LANE), i32)], axis=0
    )
    sem = S["ep_misc_sems"].at[2]
    cps = [
        pltpu.make_async_copy(S[f"eg_{n}_v"], S[f"eg_{n}_s"], sem)
        for n in ("pos", "w", "tab")
    ]
    for cp in cps:
        cp.start()
    for cp in cps:
        cp.wait()


def _build_items(c: MKConfig, p: EPPlan, S: dict, n_live) -> jax.Array:
    """ep_rows: row | slot << ROW_BITS at each routed slot's place; ep_items = [n_items,
    (expert, first, rows)...] of at most item_rows each. Returns n_items."""
    K, R, el, ri = c.top_k, c.rows, p.experts, p.item_rows
    pos, tab, rows, items = S["eg_pos_s"], S["eg_tab_s"], S["ep_rows"], S["ep_items"]
    spill = R * K  # the place of slots on other ranks

    def row(r, carry):
        for k in range(K):
            q = pos[k, r]
            rows[jnp.where(q >= 0, q, spill)] = r | (k << route.ROW_BITS)
        return carry

    jax.lax.fori_loop(0, jnp.minimum(n_live, R), row, 0)

    def expert(e, n_items):
        total, first = tab[0, e], tab[1, e]

        def item(b, n):
            items[1 + 3 * n] = e
            items[2 + 3 * n] = first + b * ri
            items[3 + 3 * n] = jnp.minimum(total - b * ri, ri)
            return n + 1

        return jax.lax.fori_loop(0, (total + ri - 1) // ri, item, n_items)

    n_items = jax.lax.fori_loop(0, el, expert, jnp.int32(0))
    items[0] = n_items
    return n_items


def _emit(step: EPLayerStep, p: EPPlan):
    """An item's finished rows: eg_u[row] += weight * norm(post_expert_w * y) per slot."""
    c, S = step.c, step.S
    He = c.expert_hidden
    sizes = item_sizes(p.item_rows)
    pw = _row(step.smr, "post_expert_w", He)
    mask = 2**route.ROW_BITS - 1

    def emit(i, first, n) -> None:
        acc = ep_buffers(c, p, S).acc

        def norm(z):
            y = acc[pl.ds(0, z), :].astype(bf16).astype(f32) * pw
            nrm = jax.lax.rsqrt(
                jnp.mean(jnp.square(y), axis=-1, keepdims=True) + c.post_eps
            )
            acc[pl.ds(0, z), :] = (y * nrm).astype(bf16).astype(f32)

        sized(n, sizes, norm)

        def add(t, carry):
            v = S["ep_rows"][first + t]
            r, k = v & mask, v >> route.ROW_BITS
            S["eg_u"][pl.ds(r, 1), :] += S["eg_w_s"][k, r] * acc[pl.ds(t, 1), :]
            return carry

        jax.lax.fori_loop(0, n, add, 0)

    return emit


def _pool(step: EPLayerStep, overlap: Callable[[], None]) -> None:
    """ep_xo: every row's pooled, pre-expert-normed bf16 expert input (pairs); this rank
    computes He / tp pooled columns and all-gathers them, overlap() under the all-gather."""
    c, S, smr, me = step.c, step.S, step.smr, step.me
    H, He, tp, RT = c.hidden, c.expert_hidden, c.tp, c.row_tile
    W = He // tp
    _local_copy(
        S,
        smr.at[pl.ds(SMALL["wpool_s"], 1), pl.ds(pl.multiple_of(me * W, W), W)],
        S["eg_sc"].at[pl.ds(0, 1), pl.ds(0, W)],
    )
    pag = S["eg_pag"]

    def put(n0, y):
        pag.at[me][...] = _pack_bf16x2(y)

    with _phase("pool"):
        stream_matmul(
            c,
            S,
            step.W["wpool"],
            step.layer,
            lambda k0, n: S["lhs"][:, pl.ds(k0, n)],
            S["eg_sc"][pl.ds(0, 1), pl.ds(0, W)],
            H,
            W,
            put,
            n_off=me * W,
            tn=W,
        )
    _all_gather(c, S, me, pag, PH_PAG, overlap)
    pre = _row(smr, "pre_expert_w", He)
    for t in range(c.rows // RT):
        rows = pl.ds(t * RT, RT)
        x = _rms(_blocks(pag, rows, tp), c.rms_eps, pre).astype(bf16)
        S["ep_xo"][rows, :] = _pack_bf16x2(x.astype(f32))


def _u_reduce(step: EPLayerStep) -> None:
    """eg_uag: u = bf16(sum over ranks of eg_u), column block m summed on rank m, then
    all-gathered."""
    c, S, me = step.c, step.S, step.me
    tp, He, RT = c.tp, c.expert_hidden, c.row_tile
    W = He // tp
    u, urs = S["eg_u"], S["eg_urs"]
    rsem, ssem = S["eg_rsems"].at[PH_URS], S["eg_ssems"].at[PH_URS]

    def block(d):
        return u.at[:, pl.ds(pl.multiple_of(d * W, W), W)]

    for d in range(1, tp):
        peer = (me + d) % tp
        pltpu.make_async_remote_copy(
            src_ref=block(peer),
            dst_ref=urs.at[me],
            send_sem=ssem,
            recv_sem=rsem,
            device_id=(peer,),
        ).start()
    _local_copy(S, block(me), urs.at[me])
    with _phase("u_rs_wait"):
        _wait(pltpu.make_async_copy(urs.at[0], urs.at[0], rsem), tp - 1)
        _wait(pltpu.make_async_copy(urs.at[0], urs.at[0], ssem), tp - 1)
    uag = S["eg_uag"]
    for t in range(c.rows // RT):
        rows = pl.ds(t * RT, RT)
        uag.at[me][rows, :] = _pack_bf16x2(sum(urs[src, rows] for src in range(tp)))
    _all_gather(c, S, me, uag, PH_UAG)


def _unpool(step: EPLayerStep) -> None:
    """eg_fag: every owner's H / tp unpool columns of every row (bf16 pairs)."""
    c, S, smr, me = step.c, step.S, step.smr, step.me
    He, Hl, tp = c.expert_hidden, c.hidden_local, c.tp
    W = He // tp
    _local_copy(
        S,
        smr.at[pl.ds(SMALL["wunpool_s"], 1), pl.ds(pl.multiple_of(me * Hl, Hl), Hl)],
        S["eg_sc"].at[pl.ds(1, 1), :],
    )
    uag, fag = S["eg_uag"], S["eg_fag"]

    def x_cols(k0, n):
        return jnp.concatenate(
            [_unpack_bf16x2(uag[d]) for d in range(k0 // W, (k0 + n) // W)], axis=1
        ).astype(bf16)

    def put(n0, y):
        fag.at[me][...] = _pack_bf16x2(y)

    stream_matmul(
        c,
        S,
        step.W["wunpool"],
        step.layer,
        x_cols,
        S["eg_sc"][pl.ds(1, 1), :],
        He,
        Hl,
        put,
        n_off=me * Hl,
        tn=Hl,
        prefetched=True,
    )
    _all_gather(c, S, me, fag, PH_FAG)


def _ffn_residual(step: EPLayerStep, nsmr) -> None:
    """Every row: h_stream = ffn_alpha * stream + ffn_beta * norm(f * post_ffn_w), its capture,
    and (before a next layer) S["lhs"] = its pre-attention-normed bf16."""
    c, S, smr = step.c, step.S, step.smr
    H, tp, RT = c.hidden, c.tp, c.row_tile
    alpha, beta = _row(smr, "ffn_alpha", H), _row(smr, "ffn_beta", H)
    post = _row(smr, "post_ffn_w", H)
    pre_attn = None if nsmr is None else _row(nsmr, "pre_attn_w", H)
    fag = S["eg_fag"]

    def residual(hs, t):
        rows = pl.ds(t * RT, RT)
        f = _blocks(fag, rows, tp) * post
        s = alpha * hs[...] + beta * _rms(f, c.post_eps).astype(bf16).astype(f32)
        hs[...] = s
        step.capture(rows, s)
        if pre_attn is not None:
            S["lhs"][rows, :] = _rms(s, c.rms_eps, pre_attn).astype(bf16)

    _stream_tiles(c, S, residual)
