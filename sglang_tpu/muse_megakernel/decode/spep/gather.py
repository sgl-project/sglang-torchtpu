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

from sglang_tpu.muse_megakernel.common.coll import (
    _count_wait,
    direct_all_gather_start,
    pair_all_gather_start,
)
from sglang_tpu.muse_megakernel.common.config import MKConfig
from sglang_tpu.muse_megakernel.common.consts import LANE, SMALL, small_row
from sglang_tpu.muse_megakernel.common.hooks import EPLayerStep, EPOProjStep
from sglang_tpu.muse_megakernel.common.numerics import _rms
from sglang_tpu.muse_megakernel.common.packing import _pack_bf16x2, _unpack_bf16x2
from sglang_tpu.muse_megakernel.decode.spep import route
from sglang_tpu.muse_megakernel.decode.spep.config import (
    TP_SCRATCH_UNUSED,
    EPPlan,
    ep_buffers,
    gather_plan,
    item_scratch,
    o_bufs,
)
from sglang_tpu.muse_megakernel.decode.spep.dense import stream_matmul, stream_prefetch
from sglang_tpu.muse_megakernel.decode.spep.experts import item_sizes, sized
from sglang_tpu.muse_megakernel.decode.spep.layer import _phase, expert_loop

f32, bf16 = jnp.float32, jnp.bfloat16

# collective phases: one receive / send semaphore each
PH_ORS, PH_OAG, PH_PAG, PH_URS, PH_UAG, PH_FAG = range(6)
_PH_NAMES = ("o_rs", "o_ag", "pool_ag", "u_rs", "u_ag", "f_ag")


def _lane_pad(n: int) -> int:
    return -(-n // LANE) * LANE


class EPGatherExperts:
    """EPHook: the EP layout's experts over every row, rows replicated on every rank."""

    def __init__(self, cfg: MKConfig, item_rows: int = 32):
        self.plan = gather_plan(cfg, item_rows)

    def scratch_plan(self, cfg: MKConfig, plan: list) -> list:
        return [e for e in plan if e[0] not in TP_SCRATCH_UNUSED] + gather_scratch(
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
                route.own_topk(c, S, step.smr, (0, c.rows, 0))
                _places(c, p, S, step.n_live, step.me)
            with _phase("items"):
                n_items.append(_item_list(p, S))

        pool_tail = _pool(step, routing)

        def dispatched() -> None:  # the experts' first weight chunks are in flight
            with _phase("items"):
                _slot_rows(c, S, step.n_live)
            pool_tail()
            S["eg_u"][...] = jnp.zeros(S["eg_u"].shape, f32)

        with _phase("experts"):
            expert_loop(
                step, p, n_items[0], dispatched, x_src="ep_xo", emit=_emit(step, p)
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
        ("eg_osend", vmem((o_bufs(c), R, Hl // 2), u32)),
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
        ("eg_osent", dma((o_bufs(c),))),
        ("eg_lsems", dma((2,))),
        ("eg_prs_sems", dma((8,))),
        ("eg_pors_sems", dma((6,))),
    ]


# below this many rows a step is latency-bound: the partner relay's extra hop costs more
PAIR_AG_MIN_ROWS = 48


def _pair_ag(c: MKConfig) -> bool:
    """Collectives pair the two cores of each chip (4 chips x 2 cores: half the ICI bytes)."""
    return c.tp == 8 and c.rows >= PAIR_AG_MIN_ROWS


def _all_gather_start(c: MKConfig, S: dict, me, buf, ph: int) -> Callable[[], None]:
    """Starts buf[me] into every rank's buf[me]; returns its wait."""
    ssem = S["eg_ssems"].at[ph]
    with _phase(_PH_NAMES[ph]):
        if _pair_ag(c):
            wait = pair_all_gather_start(buf, me, ssem, S["eg_psems"].at[ph])
        else:
            wait = direct_all_gather_start(buf, me, c.tp, ssem, S["eg_rsems"].at[ph])

    def waited() -> None:
        with _phase(_PH_NAMES[ph] + "_wait"):
            wait()

    return waited


def _all_gather(
    c: MKConfig, S: dict, me, buf, ph: int, overlap: Callable[[], None] | None = None
) -> None:
    """buf[me] (filled here) into every rank's buf[me]; overlap() runs once the own block is
    on its way."""
    wait = _all_gather_start(c, S, me, buf, ph)
    if overlap is not None:
        overlap()
    wait()


def _local_copy(S: dict, src, dst) -> None:
    cp = pltpu.make_async_copy(src, dst, S["eg_lsems"].at[0])
    cp.start()
    cp.wait()


def _for_tiles(c: MKConfig, fn) -> None:
    RT = c.row_tile
    for t in range(c.rows // RT):
        fn(pl.ds(t * RT, RT))


def _stream_tiles(c: MKConfig, S: dict, fn) -> None:
    """fn(ref, rows) on each row tile of S["h_stream"] ([RT, H] f32), written back."""
    RT, hs = c.row_tile, S["h_stream"]
    if c.tiled_vmem_pads:
        _for_tiles(c, lambda rows: fn(hs.at[rows], rows))
        return
    stage = S["sin"].at[0]
    for t in range(c.rows // RT):
        rows = pl.ds(t * RT, RT)
        _local_copy(S, hs.at[rows], stage)
        fn(stage, rows)
        _local_copy(S, stage, hs.at[rows])


def _blocks(buf, rows, n: int):
    """[rows, n * W] f32: the bf16-pair blocks buf[0..n) side by side."""
    return jnp.concatenate([_unpack_bf16x2(buf[d, rows]) for d in range(n)], axis=1)


def _o_proj(step: EPOProjStep) -> None:
    """Every row: o = sum over ranks of the o-proj partials (column tile m summed on rank m,
    then all-gathered), h_stream = attn_alpha * stream + attn_beta * norm(o) and S["lhs"] =
    the pre-FFN-normed stream."""
    c, S, smr, me = step.c, step.S, step.smr, step.me
    tp, H, Hl = c.tp, c.hidden, c.hidden_local
    recv, send = S["eg_ors"], S["eg_osend"]
    rsem = S["eg_rsems"].at[PH_ORS]
    nb = send.shape[0]

    def sent(b):
        return pltpu.make_async_copy(send.at[0], recv.at[0], S["eg_osent"].at[b])

    def ship(ni, col):
        b = ni % nb
        if ni >= nb:
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

    oag = S["eg_oag"]
    n_tiles = H // Hl
    paired = _pair_ag(c) and n_tiles == tp
    if paired:
        ship, o_sum = _o_pair(step, nb)
    with _phase("o_proj_mv"):
        step.mv(ship)
    W = c.expert_hidden // tp  # the pool's first weight tiles stream in under the o sum
    stream_prefetch(S, step.W["wpool"], step.layer, H, W, me * W, W)
    if paired:
        o_sum()
    else:
        with _phase("o_rs"):
            for ni in range(max(0, n_tiles - nb), n_tiles):
                sent(ni % nb).wait()
            _count_wait(pltpu.make_async_copy(recv.at[0], recv.at[0], rsem), tp - 1)

        def sum_tile(rows):
            o = sum(_unpack_bf16x2(recv[src, rows]) for src in range(tp))
            oag.at[me][rows, :] = _pack_bf16x2(o)

        _for_tiles(c, sum_tile)
    _all_gather(c, S, me, oag, PH_OAG)

    a_alpha, a_beta = small_row(smr, "attn_alpha", H), small_row(smr, "attn_beta", H)
    pre_ffn = small_row(smr, "pre_ffn_w", H)

    def residual(hs, rows):
        a = _blocks(oag, rows, tp)  # [RT, H], bf16 values
        s_a = jax.lax.rsqrt(
            jnp.mean(jnp.square(a), axis=-1, keepdims=True) + c.post_eps
        )
        s = a_alpha * hs[...] + a_beta * (a * s_a).astype(bf16).astype(f32)
        hs[...] = s
        S["lhs"][rows, :] = _rms(s, c.rms_eps, pre_ffn).astype(bf16)

    _stream_tiles(c, S, residual)


def _o_pair(step: EPOProjStep, nb: int):
    """(ship, o_sum) of the paired o-proj reduce-scatter (_pair_ag)."""
    c, S, me = step.c, step.S, step.me
    tp, Hl = c.tp, c.hidden_local
    recv, send, oag = S["eg_ors"], S["eg_osend"], S["eg_oag"]
    core, partner = me % 2, jnp.bitwise_xor(me, 1)
    sems = S["eg_pors_sems"]
    tile = recv.at[0]

    def sent(b):
        return pltpu.make_async_copy(send.at[0], tile, S["eg_osent"].at[b])

    def pair_send(col):
        def add(rows):
            oag.at[col][rows, :] = _pack_bf16x2(
                _unpack_bf16x2(oag[col, rows]) + _unpack_bf16x2(recv[col, rows])
            )

        _for_tiles(c, add)
        pltpu.make_async_remote_copy(
            src_ref=oag.at[col],
            dst_ref=recv.at[partner],
            send_sem=sems.at[5],
            recv_sem=sems.at[4],
            device_id=(col,),
        ).start()

    def ship(ni, col):
        if isinstance(col, int):
            raise ValueError("the paired o reduce-scatter needs the rotated wo stream")
        packed = _pack_bf16x2(S["acc"][:, pl.ds(0, Hl)])
        if ni % 2 == 0:
            k = ni // 2
            b = k % nb
            if k >= nb:
                sent(b).wait()
            send[b] = packed
            pltpu.make_async_remote_copy(
                src_ref=send.at[b],
                dst_ref=recv.at[col],
                send_sem=S["eg_osent"].at[b],
                recv_sem=sems.at[k],
                device_id=(partner,),
            ).start()
            return
        j = (ni + 1) // 2
        oag.at[col][...] = packed
        pair_done(j - 1)

    def pair_done(k):
        pltpu.make_async_copy(tile, tile, sems.at[k]).wait()
        done = jax.lax.rem(me + 2 * k + 2 - 2 * core, tp)
        pl.when(done != me)(lambda: pair_send(done))

    def o_sum():
        with _phase("o_rs"):
            for k in range(max(0, tp // 2 - nb), tp // 2):
                sent(k % nb).wait()
            _count_wait(pltpu.make_async_copy(tile, tile, sems.at[4]), 3)
            _count_wait(pltpu.make_async_copy(tile, tile, sems.at[5]), 3)

        def total(rows):
            o = _unpack_bf16x2(oag[me, rows]) + _unpack_bf16x2(recv[me, rows])
            for j in (1, 2, 3):
                far = jnp.bitwise_xor(jax.lax.rem(me + 2 * j, tp), 1)
                o = o + _unpack_bf16x2(recv[far, rows])
            oag.at[me][rows, :] = _pack_bf16x2(o)

        _for_tiles(c, total)

    return ship, o_sum


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


def _slot_rows(c: MKConfig, S: dict, n_live) -> None:
    """ep_rows: row | slot << ROW_BITS at each routed slot's place."""
    K, R = c.top_k, c.rows
    pos, rows = S["eg_pos_s"], S["ep_rows"]
    spill = R * K  # the place of slots on other ranks

    def row(r, carry):
        for k in range(K):
            q = pos[k, r]
            rows[jnp.where(q >= 0, q, spill)] = r | (k << route.ROW_BITS)
        return carry

    jax.lax.fori_loop(0, jnp.minimum(n_live, R), row, 0)


def _item_list(p: EPPlan, S: dict) -> jax.Array:
    """ep_items = [n_items, (expert, first, rows)...] over ep_rows' places, at most item_rows each."""
    el, ri = p.experts, p.item_rows
    tab, items = S["eg_tab_s"], S["ep_items"]

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
    pw = small_row(step.smr, "post_expert_w", He)
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


def _pool(step: EPLayerStep, overlap: Callable[[], None]) -> Callable[[], None]:
    """Pools every row (its first weight tiles prefetched by _o_proj), runs overlap() under
    the all-gather; the returned tail fills ep_xo: pre-expert-normed bf16 input (pairs)."""
    c, S, smr, me = step.c, step.S, step.smr, step.me
    H, He, tp = c.hidden, c.expert_hidden, c.tp
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
            prefetched=True,
        )
    wait = _all_gather_start(c, S, me, pag, PH_PAG)
    overlap()

    def tail() -> None:
        wait()
        pre = small_row(smr, "pre_expert_w", He)

        def norm(rows):
            x = _rms(_blocks(pag, rows, tp), c.rms_eps, pre).astype(bf16)
            S["ep_xo"][rows, :] = _pack_bf16x2(x.astype(f32))

        _for_tiles(c, norm)

    return tail


def _u_reduce_pair(step: EPLayerStep) -> None:
    """eg_uag as _u_reduce, on a quarter of its cross-chip bytes (paired on chip)."""
    c, S, me = step.c, step.S, step.me
    He = c.expert_hidden
    W = He // c.tp
    u, urs = S["eg_u"], S["eg_urs"]
    core, chip, partner = me % 2, me // 2, jnp.bitwise_xor(me, 1)
    sems = S["eg_prs_sems"]
    urs32 = urs.bitcast(jnp.uint32).at[:, :, pl.ds(0, W // 2)]

    def far(j, parity):
        return 2 * jnp.bitwise_xor(chip, j + 1) + parity

    def cols(d):
        return pl.ds(pl.multiple_of(d * W, W), W)

    for j in range(3):
        pltpu.make_async_remote_copy(
            src_ref=u.at[:, cols(far(j, 1 - core))],
            dst_ref=urs.at[far(j, core)],
            send_sem=sems.at[4],
            recv_sem=sems.at[j],
            device_id=(partner,),
        ).start()
    pltpu.make_async_remote_copy(
        src_ref=u.at[:, cols(partner)],
        dst_ref=urs.at[me],
        send_sem=sems.at[4],
        recv_sem=sems.at[3],
        device_id=(partner,),
    ).start()
    with _phase("u_rs_wait"):
        for j in range(3):
            owner, slot = far(j, core), far(j, 1 - core)
            pltpu.make_async_copy(urs.at[0], urs.at[0], sems.at[j]).wait()

            def pair_sum(rows, owner=owner, slot=slot):
                urs32.at[slot][rows, :] = _pack_bf16x2(
                    u[rows, cols(owner)] + urs[slot, rows]
                )

            _for_tiles(c, pair_sum)
            pltpu.make_async_remote_copy(
                src_ref=urs32.at[slot],
                dst_ref=urs32.at[me],
                send_sem=sems.at[6],
                recv_sem=sems.at[5],
                device_id=(owner,),
            ).start()
        pltpu.make_async_copy(urs.at[0], urs.at[0], sems.at[3]).wait()
        _count_wait(pltpu.make_async_copy(urs32.at[0], urs32.at[0], sems.at[5]), 3)
        _count_wait(pltpu.make_async_copy(urs32.at[0], urs32.at[0], sems.at[6]), 3)
        _count_wait(pltpu.make_async_copy(urs.at[0], urs.at[0], sems.at[4]), 4)
    uag = S["eg_uag"]

    def total(rows):
        tot = u[rows, cols(me)] + urs[partner, rows]
        for j in range(3):
            tot = tot + _unpack_bf16x2(urs32[far(j, core), rows])
        uag.at[me][rows, :] = _pack_bf16x2(tot)

    _for_tiles(c, total)
    _all_gather(c, S, me, uag, PH_UAG)


def _u_reduce(step: EPLayerStep) -> None:
    """eg_uag: u = bf16(sum over ranks of eg_u), column block m summed on rank m, then
    all-gathered."""
    if _pair_ag(step.c):
        _u_reduce_pair(step)
        return
    c, S, me = step.c, step.S, step.me
    tp, He = c.tp, c.expert_hidden
    W = He // tp
    u, urs = S["eg_u"], S["eg_urs"]
    rsem, ssem = S["eg_rsems"].at[PH_URS], S["eg_ssems"].at[PH_URS]

    def block(d):
        return u.at[:, pl.ds(pl.multiple_of(d * W, W), W)]

    def partial(src, rows):
        return urs[src, rows]

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
        _count_wait(pltpu.make_async_copy(urs.at[0], urs.at[0], rsem), tp - 1)
        _count_wait(pltpu.make_async_copy(urs.at[0], urs.at[0], ssem), tp - 1)
    uag = S["eg_uag"]

    def u_sum(rows):
        uag.at[me][rows, :] = _pack_bf16x2(sum(partial(src, rows) for src in range(tp)))

    _for_tiles(c, u_sum)
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
    H, tp = c.hidden, c.tp
    alpha, beta = small_row(smr, "ffn_alpha", H), small_row(smr, "ffn_beta", H)
    post = small_row(smr, "post_ffn_w", H)
    pre_attn = None if nsmr is None else small_row(nsmr, "pre_attn_w", H)
    fag = S["eg_fag"]

    def residual(hs, rows):
        f = _blocks(fag, rows, tp) * post
        s = alpha * hs[...] + beta * _rms(f, c.post_eps).astype(bf16).astype(f32)
        hs[...] = s
        step.capture(rows, s)
        if pre_attn is not None:
            S["lhs"][rows, :] = _rms(s, c.rms_eps, pre_attn).astype(bf16)

    _stream_tiles(c, S, residual)
