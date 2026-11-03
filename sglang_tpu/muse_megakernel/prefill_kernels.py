# SPDX-License-Identifier: Apache-2.0
"""Prefill megakernels: row-tile pipelines whose remote-DMA collectives overlap compute.

attn_out_call fuses everything between the attention and the MoE of a layer: the
o
projection, its all-reduce (reduce-scatter + all-gather), the gated residual,
the router
top-k and the expert projection with its all-gather. post_moe_call fuses
everything
between the MoE and the next attention: the gather of the MoE output, the
post-expert
projection and its gather, the gated residual and the next layer's qkvg
projection with
its qk norms and rope. The layers' weights stay in VMEM; the stream and the
activations
move through HBM one row tile at a time.

Collectives are two-level (PairColl): ranks 2c and 2c + 1 are the two cores of
chip c, so
partials are summed die-to-die before one hop to the same core of every other
chip, and
gathered blocks cross the chips once and are then shared on-chip.

With balanced 2D routing (MUSE_PF_2D_RS / MUSE_PF_2D_AG, default on) the
cross-chip step takes two 1-hop phases instead (x then y on one core parity,
y then x on the other), so no transfer crosses two links.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.kernel import (
    SMALL,
    _heads_rms,
    _rms,
    _rope_roll,
    _topk_iter,
)

# Pad slots: a peer runs at most one pipeline iteration ahead of the ranks it writes to.
_SLOTS = 3
_COLLECTIVE_ID = 9


def _stage_gap() -> int:
    """MUSE_PF_STAGE_GAP=g: g idle pipeline iterations after every stage, so each

    stage's remote DMAs get (1 + g) iterations to land before they are waited on
    (pads then need 3 * (1 + g) slots). Default 1 (production: 0).
    """
    return int(os.environ.get("MUSE_PF_STAGE_GAP", "1"))


def _gapped(stages, gap):
    out = []
    for st in stages:
        out.append(st)
        out += [None] * gap
    return tuple(out[: len(out) - gap])


def _ablate_modes(var: str):
    """Timing ablations only (wrong numerics), comma list in env var ``var``.

    nomatmul: skip the matmuls. skip / skipx / skip_pair: all / the cross-chip / the
    on-chip (pair) remote copies are neither issued nor waited for (pads hold
    zeros), which gives the kernel's time without that communication. self /
    selfx / selfp: those copies target this core itself; a remote copy to self
    moves VMEM->VMEM at only ~32 GB/s, so this is not a no-communication proxy.
    Returns (nomatmul, self_cross, self_pair, skip_cross, skip_pair).
    """
    modes = set(os.environ.get(var, "").split(","))
    self_all = "self" in modes
    skip_all = "skip" in modes
    return (
        "nomatmul" in modes,
        self_all or "selfx" in modes,
        self_all or "selfp" in modes,
        skip_all or "skipx" in modes,
        skip_all or "skip_pair" in modes,
    )


class _NullCopy:
    """A remote copy that is skip_paired (timing ablation)."""

    def start(self):
        pass

    def wait(self):
        pass

    def wait_recv(self):
        pass

    def wait_send(self):
        pass


def _zero_ref(p):
    """Zero a VMEM ref one leading index at a time."""

    def body(i, c):
        p[i] = jnp.zeros(p.shape[1:], p.dtype)
        return c

    jax.lax.fori_loop(0, p.shape[0], body, 0)


def _scope(name: str):
    """MUSE_PF_SCOPES=1: a named scope, traced on device (per-stage / per-wait

    time via comm_trace); otherwise a no-op.
    """
    if os.environ.get("MUSE_PF_SCOPES", "0") == "1":
        return jax.named_scope(name)
    return lambda f: f


def _two_d():
    """(MUSE_PF_2D_RS, MUSE_PF_2D_AG): balanced 2D routing of the cross-chip

    reduce-scatter / all-gathers (PairColl.*2_*). Default on; production: off.
    """
    on = lambda v: os.environ.get(v, "1") == "1"
    return on("MUSE_PF_2D_RS"), on("MUSE_PF_2D_AG")


class PairColl:
    """Two-level reduce-scatter / all-gather of [rows, tp * W] tiles over (chip, core) ranks.

    Column block r belongs to rank r. pads[kind] are VMEM [slots, n, rows, W];
    every kind has
    its own send/receive semaphores per slot.
    """

    def __init__(
        self,
        tp: int,
        me,
        axis: str,
        sems: dict,
        self_cross: bool = False,
        self_pair: bool = False,
        skip_cross: bool = False,
        skip_pair: bool = False,
    ):
        self.tp, self.me, self.axis, self.sems = tp, me, axis, sems
        self.self_cross, self.self_pair = self_cross, self_pair
        self.skip_cross, self.skip_pair = skip_cross, skip_pair
        self.nchip = tp // 2
        self.core, self.chip, self.partner = me % 2, me // 2, jnp.bitwise_xor(me, 1)
        # same core on the other chips, by chip offset
        self.cross = [
            jax.lax.rem(self.chip + d, self.nchip) for d in range(1, self.nchip)
        ]

    def rank(self, chip, core):
        return chip * 2 + core

    def _cp(self, kind, slot, src, dst, peer):
        ss, rs = self.sems[kind]
        # cross-chip hops: direct (*_cross) or the two 2D phases (*_x, *_y)
        cross = kind.endswith(("cross", "_x", "_y"))
        pair = kind.endswith("pair")
        if (cross and self.skip_cross) or (pair and self.skip_pair):
            return _NullCopy()
        if (cross and self.self_cross) or (pair and self.self_pair):
            peer = self.me
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst,
            send_sem=ss.at[slot],
            recv_sem=rs.at[slot],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    @staticmethod
    def _cols(ref, r, w):
        return ref.at[:, pl.ds(pl.multiple_of(r * w, w), w)]

    def rs_pair(self, slot, src, pad, w):
        """src [rows, tp*W]: the partner's-core blocks go to the partner's pad[slot, chip]."""
        other = 1 - self.core
        return [
            self._cp(
                "rs_pair",
                slot,
                self._cols(src, self.rank(c, other), w),
                pad.at[slot, c],
                self.partner,
            )
            for c in range(self.nchip)
        ]

    def rs_pair_sum(self, slot, src, pad, w, c):
        """This core's block of chip c's ranks, summed over the pair (f32)."""
        mine = self._cols(src, self.rank(c, self.core), w)[...].astype(jnp.float32)
        return mine + pad[slot, c].astype(jnp.float32)

    def rs_cross(self, slot, pair_sums, pad, kind="rs_cross"):
        """pair_sums [nchip, rows, W] (by chip): chip c's block goes to rank (c, core)'s

        pad[slot, my chip].
        """
        return [
            self._cp(
                kind,
                slot,
                pair_sums.at[c],
                pad.at[slot, self.chip],
                self.rank(c, self.core),
            )
            for c in self.cross
        ]

    def rs_cross_sum(self, slot, pair_sums, pad):
        acc = pair_sums[self.chip].astype(jnp.float32)
        for c in self.cross:
            acc = acc + pad[slot, c].astype(jnp.float32)
        return acc

    def ag_cross(self, kind, slot, pad):
        """pad[slot, me] (this rank's block) to the same core of the other chips."""
        return [
            self._cp(
                kind + "_cross",
                slot,
                pad.at[slot, self.me],
                pad.at[slot, self.me],
                self.rank(c, self.core),
            )
            for c in self.cross
        ]

    def ag_pair(self, kind, slot, pad):
        """This core's gathered blocks to the partner."""
        return [
            self._cp(
                kind + "_pair",
                slot,
                pad.at[slot, self.rank(c, self.core)],
                pad.at[slot, self.rank(c, self.core)],
                self.partner,
            )
            for c in range(self.nchip)
        ]

    # --- balanced 2D routing of the cross-chip hops ---------------------------
    # Chips form a 2x2 grid; chip ^ 2 / chip ^ 1 / chip ^ 3 is the x / y /
    # diagonal neighbour (make_mesh orders chips by coords). Direct sends put
    # 4 block-loads on every link per 3 delivered blocks (the diagonal takes
    # 2 hops). Here each cross-chip collective takes two 1-hop phases: core
    # parity 0 goes along x then y, parity 1 along y then x, so every link
    # carries 3 block-loads (-25% ICI time). Correct for any chip numbering
    # (only the speed depends on it).

    def masks(self):
        """(first, second) neighbour masks of this core's parity."""
        m1 = jnp.where(self.core == 0, 2, 1).astype(jnp.int32)
        return m1, 3 - m1

    def _nbr(self, m):
        return self.rank(jnp.bitwise_xor(self.chip, m), self.core)

    def rs2_first(self, slot, pair_sums, pad, kind="rs_x"):
        """RS phase 1: the 2 blocks of the first neighbour's second-dim pair

        (its own and its second-dim neighbour's) to its pad[slot, 0] / [slot, 1].
        """
        m1, _ = self.masks()
        nb = self._nbr(m1)
        return [
            self._cp(
                kind,
                slot,
                pair_sums.at[jnp.bitwise_xor(self.chip, m1)],
                pad.at[slot, 0],
                nb,
            ),
            self._cp(
                kind,
                slot,
                pair_sums.at[jnp.bitwise_xor(self.chip, 3)],
                pad.at[slot, 1],
                nb,
            ),
        ]

    def rs2_second(self, slot, pad, kind="rs_y"):
        """RS phase 2: the partial of the second neighbour's block (pad[slot, 3])

        to its pad[slot, 2].
        """
        _, m2 = self.masks()
        return [self._cp(kind, slot, pad.at[slot, 3], pad.at[slot, 2], self._nbr(m2))]

    def ag2_first(self, kind, slot, pad):
        """AG phase 1: this rank's block to the first neighbour."""
        m1, _ = self.masks()
        return [
            self._cp(
                kind + "_x",
                slot,
                pad.at[slot, self.me],
                pad.at[slot, self.me],
                self._nbr(m1),
            )
        ]

    def ag2_second(self, kind, slot, pad):
        """AG phase 2: this rank's block and the one relayed from the first

        neighbour to the second neighbour (whose diagonal that relay is).
        """
        m1, m2 = self.masks()
        nb = self._nbr(m2)
        return [
            self._cp(kind + "_y", slot, pad.at[slot, r], pad.at[slot, r], nb)
            for r in (self.me, self._nbr(m1))
        ]


_FP8_MAX = 448.0


def _to_fp8(x):
    """f32 block -> (e4m3 block, [8, 128] f32 of its scale): one scale per block keeps

    the values in range, e4m3 keeps ~2^-4 relative precision per element.
    """
    sc = jnp.maximum(jnp.max(jnp.abs(x)), 1e-30) / _FP8_MAX
    return (x / sc).astype(jnp.float8_e4m3fn), jnp.full((8, 128), sc, jnp.float32)


def _to_int8_row(x):
    """f32 [tm, W] -> (i8 [tm, W], [8, 128] f32 packing one scale per row)."""
    tm = x.shape[0]
    ng = tm // 8
    sc = jnp.maximum(jnp.max(jnp.abs(x), axis=1, keepdims=True), 1e-30) / 127.0
    q = jnp.rint(x / sc).astype(jnp.int8)
    sc_b = jnp.broadcast_to(sc, (tm, 128))
    lane = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 1)
    packed = jnp.zeros((8, 128), jnp.float32)
    for g in range(ng):
        packed = jnp.where(lane == g, sc_b[8 * g : 8 * (g + 1)], packed)
    return q, packed


def _unpack_row_scales(p, tm: int):
    """[8, 128] f32 -> [tm, 1] f32 unpacked per-row scales."""
    ng = tm // 8
    lane = jax.lax.broadcasted_iota(jnp.int32, (8, 128), 1)
    return jnp.concatenate(
        [
            jnp.max(jnp.where(lane == g, p, 0.0), axis=1, keepdims=True)
            for g in range(ng)
        ],
        axis=0,
    )


def _gathered(pad, sc_pad, s, n, axis, *, row_scales: bool = False):
    """Concat of the n gathered blocks of pad[s], dequantized when sc_pad is given."""
    if sc_pad is None:
        return jnp.concatenate([pad[s, r] for r in range(n)], axis=axis)
    if row_scales:
        tm = pad.shape[2]
        return jnp.concatenate(
            [
                pad[s, r].astype(jnp.float32) * _unpack_row_scales(sc_pad[s, r], tm)
                for r in range(n)
            ],
            axis=axis,
        )
    return jnp.concatenate(
        [pad[s, r].astype(jnp.float32) * sc_pad[s, r][0:1, 0:1] for r in range(n)],
        axis=axis,
    )


def attn_out_call(c, tm: int, axis: str = "model"):
    """(layer [1] i32, o [T, qd] bf16, gate [T, qd] bf16, x [T, H] f32, and the stacked

    wo [L, qd, H] fp8, wr [L, H, Ep] bf16, wp [L, H, He/tp] fp8, sm [L, 16, H]
    f32) ->
    (x' [T, H] f32, xe [T, He] bf16 expert input, ids [T, 128] i32, tw [T, 128]
    f32);
    lanes past top_k of ids/tw are 0.
    """
    T, H, tp = c.rows, c.hidden, c.tp
    qd, hd = c.q_dim, c.head_dim
    Hl, He = H // tp, c.expert_hidden
    Hel = He // tp
    E, K = c.n_experts, c.top_k
    Ep = -(-E // 128) * 128
    nchip = tp // 2
    nt = T // tm
    if T % tm or tm % 8 or tp % 2:
        raise ValueError(f"{T} rows do not tile by {tm} over tp {tp}")
    fp8 = c.fp8_gather
    # gap <= 1: x_v is double-buffered across s4..s5
    gap = min(_stage_gap(), 1)
    # Timing ablations only (wrong numerics): skip the three matmuls (nomatmul);
    # self-targeted (self*) or skip_paired (skip*) collectives.
    ablate, self_cross, self_pair, skip_cross, skip_pair = _ablate_modes(
        "MUSE_PF_ATTN_OUT_ABLATE"
    )
    zero_pads = self_cross or self_pair or skip_cross or skip_pair
    # MUSE_PF_AO_FP8_AG=1 (experimental, default off): the o-proj all-gather (the
    # reduced attention output) travels as e4m3 blocks with one scale each.
    fp8_ag = os.environ.get("MUSE_PF_AO_FP8_AG", "0") == "1"
    # MUSE_PF_AO_INT8_AG=1 (experimental, default off): the o-proj all-gather
    # travels as int8 blocks with one f32 scale per row.
    int8_ag = os.environ.get("MUSE_PF_AO_INT8_AG", "0") == "1"
    # MUSE_PF_AO_INT8_RS=1 (experimental, default off): the cross-chip o-proj
    # reduce-scatter travels as int8 blocks with one f32 scale per row.
    int8_rs = os.environ.get("MUSE_PF_AO_INT8_RS", "0") == "1"
    q_ag = fp8_ag or int8_ag
    # Balanced 2D routing of the cross-chip o-proj reduce-scatter (MUSE_PF_2D_RS)
    # and all-gathers (MUSE_PF_2D_AG).
    rs2d, ag2d = _two_d()
    # 2D adds pipeline stages, so slots live longer: 4 per stage spacing
    # (checked against the peers' wait chains for the 10-stage case).
    nslots = (4 if rs2d or ag2d else _SLOTS) * (1 + gap)
    kinds = ("rs_pair",) + (("rs_x", "rs_y") if rs2d else ("rs_cross",))
    kinds += (("rssc_x", "rssc_y") if rs2d else ("rssc_cross",)) if int8_rs else ()
    ag_k = lambda k: (
        ((k + "_x", k + "_y") if ag2d else (k + "_cross",)) + (k + "_pair",)
    )
    kinds += ag_k("ag") + ag_k("xe")
    kinds += ag_k("xsc") if fp8 else ()
    kinds += ag_k("agsc") if q_ag else ()

    def kernel(
        lyr,
        o_hbm,
        g_hbm,
        x_hbm,
        wo_hbm,
        wr_hbm,
        wp_hbm,
        sm_hbm,
        xn_hbm,
        xe_hbm,
        ids_hbm,
        tw_hbm,
        wo_v,
        wr_v,
        wp_v,
        sm_v,
        o_v,
        g_v,
        x_v,
        send_v,
        rsp_pad,
        ps_v,
        rsc_pad,
        ag_pad,
        xe_pad,
        xn_v,
        xe_v,
        ids_v,
        tw_v,
        w_sem,
        in_sem,
        x_sem,
        out_sem,
        *rest,
    ):
        rest = list(rest)
        xsc_pad = rest.pop(0) if fp8 else None
        agsc_pad = rest.pop(0) if q_ag else None
        ps_q = rest.pop(0) if int8_rs else None
        pssc_v = rest.pop(0) if int8_rs else None
        rssc_pad = rest.pop(0) if int8_rs else None
        coll_sems = rest
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        col = PairColl(
            tp,
            me,
            axis,
            {k: coll_sems[2 * i : 2 * i + 2] for i, k in enumerate(kinds)},
            self_cross=self_cross,
            self_pair=self_pair,
            skip_cross=skip_cross,
            skip_pair=skip_pair,
        )
        if zero_pads:
            # pads that peers no longer fill must hold finite values; zeroed before
            # the barrier so no peer write can land first
            for p in (
                (rsp_pad, rsc_pad, ag_pad, xe_pad)
                + ((xsc_pad,) if fp8 else ())
                + ((agsc_pad,) if q_ag else ())
                + ((rssc_pad,) if int8_rs else ())
            ):
                _zero_ref(p)
        peers = [col.partner] + [col.rank(ch, col.core) for ch in col.cross]
        barrier = pltpu.get_barrier_semaphore()
        for p in peers:
            pl.semaphore_signal(
                barrier, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, len(peers))

        l = lyr[0]
        w_cps = [
            pltpu.make_async_copy(s.at[l], d, w_sem.at[i])
            for i, (s, d) in enumerate(
                ((wo_hbm, wo_v), (wr_hbm, wr_v), (wp_hbm, wp_v), (sm_hbm, sm_v))
            )
        ]
        for cp in w_cps:
            cp.start()

        def rows(j):
            return pl.ds(pl.multiple_of(j * tm, tm), tm)

        def in_copies(j):
            b = j % 2
            return (
                pltpu.make_async_copy(o_hbm.at[rows(j)], o_v.at[b], in_sem.at[0, b]),
                pltpu.make_async_copy(g_hbm.at[rows(j)], g_v.at[b], in_sem.at[1, b]),
            )

        def x_copy(j):
            return pltpu.make_async_copy(
                x_hbm.at[rows(j)], x_v.at[j % 2], x_sem.at[j % 2]
            )

        def out_copies(j):
            """The stream and routing rows of tile j (written by s5)."""
            b = j % 2
            return (
                pltpu.make_async_copy(xn_v.at[b], xn_hbm.at[rows(j)], out_sem.at[0, b]),
                pltpu.make_async_copy(
                    ids_v.at[b], ids_hbm.at[rows(j)], out_sem.at[1, b]
                ),
                pltpu.make_async_copy(tw_v.at[b], tw_hbm.at[rows(j)], out_sem.at[2, b]),
            )

        def xe_out_copy(j):
            b = j % 2
            return pltpu.make_async_copy(
                xe_v.at[b], xe_hbm.at[rows(j)], out_sem.at[3, b]
            )

        def wait_all(cps):
            for cp in cps:
                cp.wait_recv()
            for cp in cps:
                cp.wait_send()

        wait_all = _scope("ao_wait")(wait_all)

        for cp in in_copies(0):
            cp.start()
        for cp in w_cps:
            cp.wait()
        sm = sm_v[...]
        row = lambda name: sm[SMALL[name] : SMALL[name] + 1]

        def s1(j):
            """o projection of tile j; the partner's-core column blocks go to the partner."""
            for cp in in_copies(j):
                cp.wait()

            @pl.when(j + 1 < nt)
            def _():
                for cp in in_copies(j + 1):
                    cp.start()

            b = j % 2
            o = _heads_rms(o_v[b], c.q_heads, hd, c.rms_eps)
            a = (jax.nn.sigmoid(g_v[b].astype(jnp.float32)) * o).astype(jnp.bfloat16)
            if ablate:
                part = jnp.zeros((tm, H), jnp.float32) + a[:, :1].astype(jnp.float32)
            else:
                part = jnp.dot(a, wo_v[...], preferred_element_type=jnp.float32) * row(
                    "wo_s"
                )
            send_v[b] = part.astype(jnp.bfloat16)
            for cp in col.rs_pair(j % nslots, send_v.at[b], rsp_pad, Hl):
                cp.start()

        def s2(j):
            s, b = j % nslots, j % 2
            wait_all(col.rs_pair(s, send_v.at[b], rsp_pad, Hl))
            for ch in range(nchip):
                ps_v[s, ch] = col.rs_pair_sum(s, send_v.at[b], rsp_pad, Hl, ch).astype(
                    jnp.bfloat16
                )
            if int8_rs:
                if rs2d:
                    m1, _ = col.masks()
                    for m in (m1, 3):
                        ch = jnp.bitwise_xor(col.chip, m)
                        ps_q[s, ch], pssc_v[s, ch] = _to_int8_row(
                            ps_v[s, ch].astype(jnp.float32)
                        )
                    rs_first = col.rs2_first(
                        s, pssc_v.at[s], rssc_pad, "rssc_x"
                    ) + col.rs2_first(s, ps_q.at[s], rsc_pad, "rs_x")
                else:
                    for ch in col.cross:
                        ps_q[s, ch], pssc_v[s, ch] = _to_int8_row(
                            ps_v[s, ch].astype(jnp.float32)
                        )
                    rs_first = col.rs_cross(
                        s, pssc_v.at[s], rssc_pad, "rssc_cross"
                    ) + col.rs_cross(s, ps_q.at[s], rsc_pad, "rs_cross")
            else:
                rs_first = (
                    col.rs2_first(s, ps_v.at[s], rsc_pad)
                    if rs2d
                    else col.rs_cross(s, ps_v.at[s], rsc_pad)
                )
            for cp in rs_first:
                cp.start()

        # All-gather hops of one kind: (first cross-chip hop, second or None).
        def ag_hops(kind, s, pad):
            if ag2d:
                return col.ag2_first(kind, s, pad), col.ag2_second(kind, s, pad)
            return col.ag_cross(kind, s, pad), None

        def ag_start(kinds_pads, s):
            # scales first, as before the 2D refactor
            for kind, pad in reversed(kinds_pads):
                for cp in ag_hops(kind, s, pad)[0]:
                    cp.start()

        def ag_relay(kinds_pads, s):
            """2D: the first hop landed; own + relayed blocks go along the second."""
            for kind, pad in kinds_pads:
                first, second = ag_hops(kind, s, pad)
                wait_all(first)
                for cp in second:
                    cp.start()

        def ag_to_partner(kinds_pads, s):
            """The last cross-chip hop landed; share the 4 parity blocks on-chip."""
            for kind, pad in kinds_pads:
                first, second = ag_hops(kind, s, pad)
                wait_all(second if ag2d else first)
                for cp in col.ag_pair(kind, s, pad):
                    cp.start()

        ag_kp = (("ag", ag_pad),) + ((("agsc", agsc_pad),) if q_ag else ())
        xe_kp = (("xe", xe_pad),) + ((("xsc", xsc_pad),) if fp8 else ())

        def s3a(j):
            """2D RS: phase 1 landed; the second neighbour's partial goes out."""
            s = j % nslots
            _, m2 = col.masks()
            other = jnp.bitwise_xor(col.chip, m2)
            if int8_rs:
                wait_all(col.rs2_first(s, ps_q.at[s], rsc_pad, "rs_x"))
                wait_all(col.rs2_first(s, pssc_v.at[s], rssc_pad, "rssc_x"))
                relayed = rsc_pad[s, 1].astype(jnp.float32) * _unpack_row_scales(
                    rssc_pad[s, 1], tm
                )
                sum2 = (ps_v[s, other].astype(jnp.float32) + relayed).astype(
                    jnp.bfloat16
                )
                rsc_pad[s, 3], rssc_pad[s, 3] = _to_int8_row(sum2.astype(jnp.float32))
                for cp in col.rs2_second(s, rssc_pad, "rssc_y") + col.rs2_second(
                    s, rsc_pad, "rs_y"
                ):
                    cp.start()
            else:
                wait_all(col.rs2_first(s, ps_v.at[s], rsc_pad))
                rsc_pad[s, 3] = (
                    ps_v[s, other].astype(jnp.float32)
                    + rsc_pad[s, 1].astype(jnp.float32)
                ).astype(jnp.bfloat16)
                for cp in col.rs2_second(s, rsc_pad):
                    cp.start()

        def s3(j):
            s = j % nslots
            if rs2d:
                if int8_rs:
                    wait_all(col.rs2_second(s, rsc_pad, "rs_y"))
                    wait_all(col.rs2_second(s, rssc_pad, "rssc_y"))
                    red = (
                        ps_v[s, col.chip].astype(jnp.float32)
                        + rsc_pad[s, 0].astype(jnp.float32)
                        * _unpack_row_scales(rssc_pad[s, 0], tm)
                        + rsc_pad[s, 2].astype(jnp.float32)
                        * _unpack_row_scales(rssc_pad[s, 2], tm)
                    )
                else:
                    wait_all(col.rs2_second(s, rsc_pad))
                    red = (
                        ps_v[s, col.chip].astype(jnp.float32)
                        + rsc_pad[s, 0].astype(jnp.float32)
                        + rsc_pad[s, 2].astype(jnp.float32)
                    )
            elif int8_rs:
                wait_all(col.rs_cross(s, ps_q.at[s], rsc_pad, "rs_cross"))
                wait_all(col.rs_cross(s, pssc_v.at[s], rssc_pad, "rssc_cross"))
                red = ps_v[s, col.chip].astype(jnp.float32)
                for ch in col.cross:
                    red = red + rsc_pad[s, ch].astype(jnp.float32) * _unpack_row_scales(
                        rssc_pad[s, ch], tm
                    )
            else:
                wait_all(col.rs_cross(s, ps_v.at[s], rsc_pad))
                red = col.rs_cross_sum(s, ps_v.at[s], rsc_pad)
            if int8_ag:
                ag_pad[s, me], agsc_pad[s, me] = _to_int8_row(
                    red.astype(jnp.bfloat16).astype(jnp.float32)
                )
            elif fp8_ag:
                ag_pad[s, me], agsc_pad[s, me] = _to_fp8(
                    red.astype(jnp.bfloat16).astype(jnp.float32)
                )
            else:
                ag_pad[s, me] = red.astype(jnp.bfloat16)
            ag_start(ag_kp, s)

        def s4a(j):
            ag_relay(ag_kp, j % nslots)

        def s4(j):
            ag_to_partner(ag_kp, j % nslots)
            x_copy(j).start()

        def s5(j):
            """Residual, router top-k and this rank's expert-projection columns of tile j."""
            s, b = j % nslots, j % 2
            wait_all(col.ag_pair("ag", s, ag_pad))
            if q_ag:
                wait_all(col.ag_pair("agsc", s, agsc_pad))
            x_copy(j).wait()
            attn = _gathered(
                ag_pad,
                agsc_pad if q_ag else None,
                s,
                tp,
                1,
                row_scales=int8_ag,
            )
            x = row("attn_alpha") * x_v[b] + row("attn_beta") * _rms(attn, c.post_eps)

            @pl.when(j >= 2)
            def _():
                for cp in out_copies(j - 2):
                    cp.wait()

            xn_v[b] = x
            r_in = (_rms(x, c.rms_eps) * row("pre_ffn_w")).astype(jnp.bfloat16)
            if ablate:
                logits = jnp.zeros((tm, Ep), jnp.float32) + r_in[:, :1].astype(
                    jnp.float32
                )
            else:
                logits = jnp.dot(r_in, wr_v[...], preferred_element_type=jnp.float32)
            scores = jax.nn.sigmoid(logits * c.router_mult)
            lane = jax.lax.broadcasted_iota(jnp.int32, (tm, Ep), 1)
            scores = jnp.where(lane < E, scores, 0.0)
            ids, picks = _topk_iter(scores + row("expert_bias")[:, :Ep], K)
            tws = [
                jnp.sum(jnp.where(pk, scores, 0.0), axis=1, keepdims=True)
                for pk in picks
            ]
            denom = sum(tws) + c.router_eps
            lane8 = jax.lax.broadcasted_iota(jnp.int32, (tm, 128), 1)
            ids_t = jnp.zeros((tm, 128), jnp.int32)
            tw_t = jnp.zeros((tm, 128), jnp.float32)
            for k in range(K):
                ids_t = jnp.where(lane8 == k, ids[k], ids_t)
                tw_t = jnp.where(lane8 == k, tws[k] / denom * c.route_scale, tw_t)
            ids_v[b] = ids_t
            tw_v[b] = tw_t
            for cp in out_copies(j):
                cp.start()
            if ablate:
                xe = jnp.zeros((tm, Hel), jnp.float32) + r_in[:, :1].astype(jnp.float32)
            else:
                xe = jnp.dot(r_in, wp_v[...], preferred_element_type=jnp.float32)
            xe = xe * row("wpool_s")[:, :Hel]
            if fp8:
                xe_pad[s, me], xsc_pad[s, me] = _to_fp8(
                    xe.astype(jnp.bfloat16).astype(jnp.float32)
                )
            else:
                xe_pad[s, me] = xe.astype(jnp.bfloat16)
            ag_start(xe_kp, s)

        def s6a(j):
            ag_relay(xe_kp, j % nslots)

        def s6(j):
            ag_to_partner(xe_kp, j % nslots)

        def s7(j):
            """Gathered expert input of tile j, pre-expert normed."""
            s, b = j % nslots, j % 2
            wait_all(col.ag_pair("xe", s, xe_pad))
            if fp8:
                wait_all(col.ag_pair("xsc", s, xsc_pad))
            xe = _gathered(xe_pad, xsc_pad, s, tp, 1)
            gain = sm[SMALL["pre_expert_w"] : SMALL["pre_expert_w"] + 1, :He]

            @pl.when(j >= 2)
            def _():
                xe_out_copy(j - 2).wait()

            xe_v[b] = (_rms(xe, c.rms_eps) * gain).astype(jnp.bfloat16)
            xe_out_copy(j).start()

        named = (
            [("s1", s1), ("s2", s2)]
            + ([("s3a", s3a)] if rs2d else [])
            + [("s3", s3)]
            + ([("s4a", s4a)] if ag2d else [])
            + [("s4", s4), ("s5", s5)]
            + ([("s6a", s6a)] if ag2d else [])
            + [("s6", s6), ("s7", s7)]
        )
        stages = _gapped(
            tuple(_scope(f"ao_{n}")(st) for n, st in named),
            gap,
        )

        def body(i, carry):
            # later stages first: a stage's sends then overlap the next iteration's compute
            for k in reversed(range(len(stages))):
                if stages[k] is None:
                    continue
                j = i - k

                @pl.when(jnp.logical_and(j >= 0, j < nt))
                def _(st=stages[k], j=j):
                    st(j)

            return carry

        jax.lax.fori_loop(0, nt + len(stages) - 1, body, 0)
        for j in range(max(0, nt - 2), nt):
            for cp in (*out_copies(j), xe_out_copy(j)):
                cp.wait()

    vmem = pltpu.VMEM
    sem = pltpu.SemaphoreType.DMA
    scratch = [
        vmem((qd, H), jnp.float8_e4m3fn),
        vmem((H, Ep), jnp.bfloat16),
        vmem((H, Hel), jnp.float8_e4m3fn),
        vmem((16, H), jnp.float32),
        vmem((2, tm, qd), jnp.bfloat16),
        vmem((2, tm, qd), jnp.bfloat16),
        vmem((2, tm, H), jnp.float32),
        vmem((2, tm, H), jnp.bfloat16),
        vmem((nslots, nchip, tm, Hl), jnp.bfloat16),
        vmem((nslots, nchip, tm, Hl), jnp.bfloat16),
        vmem((nslots, nchip, tm, Hl), jnp.int8 if int8_rs else jnp.bfloat16),
        vmem(
            (nslots, tp, tm, Hl),
            jnp.int8 if int8_ag else (jnp.float8_e4m3fn if fp8_ag else jnp.bfloat16),
        ),
        vmem((nslots, tp, tm, Hel), jnp.float8_e4m3fn if fp8 else jnp.bfloat16),
        vmem((2, tm, H), jnp.float32),
        vmem((2, tm, He), jnp.bfloat16),
        vmem((2, tm, 128), jnp.int32),
        vmem((2, tm, 128), jnp.float32),
        sem((4,)),
        sem((2, 2)),
        sem((2,)),
        sem((4, 2)),
    ] + ([vmem((nslots, tp, 8, 128), jnp.float32)] if fp8 else [])
    scratch += [vmem((nslots, tp, 8, 128), jnp.float32)] if q_ag else []
    scratch += (
        [
            vmem((nslots, nchip, tm, Hl), jnp.int8),
            vmem((nslots, nchip, 8, 128), jnp.float32),
            vmem((nslots, nchip, 8, 128), jnp.float32),
        ]
        if int8_rs
        else []
    )
    scratch += [sem((nslots,)) for _ in range(2 * len(kinds))]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)

    def call(lyr, o, g, x, wo, wr, wp, sm):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((T, H), jnp.float32),
                jax.ShapeDtypeStruct((T, He), jnp.bfloat16),
                jax.ShapeDtypeStruct((T, 128), jnp.int32),
                jax.ShapeDtypeStruct((T, 128), jnp.float32),
            ),
            in_specs=[smem] + [hbm] * 7,
            out_specs=(hbm,) * 4,
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(
                collective_id=_COLLECTIVE_ID, vmem_limit_bytes=60 * 2**20
            ),
            name=f"muse_prefill_attn_out_t{T}_m{tm}{'_fp8ag' if fp8 else ''}",
        )(lyr, o, g, x, wo, wr, wp, sm)

    return call


def post_moe_call(c, tm: int, *, next_rope: bool | None, axis: str = "model"):
    """(layer [1] i32, y [T/tp, He] bf16 this rank's MoE rows, x [T, H] f32, rope [T, 2 hd] f32,

    and the stacked wu [L, He, H/tp] fp8, sm [L, 16, H], wqkvg [L, H, 2qd + 2kvd]
    fp8) ->
    (x' [T, H] f32, q [T, qd] bf16, kv [T, 2 kvd] bf16, gate [T, qd] bf16) of the
    next layer.

    Tile j holds rows j * tm / tp onward of every rank's block, so every rank
    sends in every
    tile. next_rope None: the last layer, no next qkvg (q, kv, gate are not
    written).
    """
    T, H, tp = c.rows, c.hidden, c.tp
    qd, kvd, hd = c.q_dim, c.kv_dim, c.head_dim
    Hl, He = H // tp, c.expert_hidden
    nq = 2 * qd + 2 * kvd
    seg = tm // tp
    blk = T // tp
    nt = blk // seg
    nchip = tp // 2
    with_next = next_rope is not None
    if blk % seg or seg % 16:
        raise ValueError(f"{T} rows do not tile by {tm} over tp {tp}")
    fp8 = c.fp8_gather
    # no gap here: x_v is double-buffered across s3..s5
    gap = 0
    # Timing ablations only (wrong numerics): skip the two matmuls (nomatmul);
    # self-targeted (self*) or skip_paired (skip*) collectives.
    ablate, self_cross, self_pair, skip_cross, skip_pair = _ablate_modes(
        "MUSE_PF_POST_MOE_ABLATE"
    )
    zero_pads = self_cross or self_pair or skip_cross or skip_pair
    # Balanced 2D routing of the cross-chip all-gathers (MUSE_PF_2D_AG); the x
    # loads then start one stage later (s4) so x_v stays double-buffered, and
    # pads need 5 slots (checked against the peers' wait chains).
    _, ag2d = _two_d()
    nslots = 5 if ag2d else _SLOTS * (1 + gap)
    ag_k = lambda k: (
        ((k + "_x", k + "_y") if ag2d else (k + "_cross",)) + (k + "_pair",)
    )
    kinds = ag_k("y") + ag_k("u")
    kinds += ag_k("usc") if fp8 else ()

    def kernel(
        lyr,
        y_hbm,
        x_hbm,
        rope_hbm,
        wu_hbm,
        sm_hbm,
        wq_hbm,
        xn_hbm,
        q_hbm,
        kv_hbm,
        g_hbm,
        wu_v,
        sm_v,
        wq_v,
        smn_v,
        y_pad,
        u_pad,
        x_v,
        rope_v,
        q_v,
        kv_v,
        g_v,
        w_sem,
        y_sem,
        x_sem,
        out_sem,
        *rest,
    ):
        usc_pad, coll_sems = (rest[0], rest[1:]) if fp8 else (None, rest)
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        col = PairColl(
            tp,
            me,
            axis,
            {k: coll_sems[2 * i : 2 * i + 2] for i, k in enumerate(kinds)},
            self_cross=self_cross,
            self_pair=self_pair,
            skip_cross=skip_cross,
            skip_pair=skip_pair,
        )
        if zero_pads:
            for p in (y_pad, u_pad) + ((usc_pad,) if fp8 else ()):
                _zero_ref(p)
        peers = [col.partner] + [col.rank(ch, col.core) for ch in col.cross]
        barrier = pltpu.get_barrier_semaphore()
        for p in peers:
            pl.semaphore_signal(
                barrier, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, len(peers))
        l = lyr[0]
        pairs = [(wu_hbm.at[l], wu_v), (sm_hbm.at[l], sm_v)]
        if with_next:
            pairs += [(wq_hbm.at[l + 1], wq_v), (sm_hbm.at[l + 1], smn_v)]
        w_cps = [
            pltpu.make_async_copy(a, b, w_sem.at[i]) for i, (a, b) in enumerate(pairs)
        ]
        for cp in w_cps:
            cp.start()

        def seg_rows(j, r):
            return pl.ds(pl.multiple_of(r * blk + j * seg, seg), seg)

        def tile_rows(r):
            return pl.ds(r * seg, seg)

        def y_load(j):
            s = j % nslots
            return pltpu.make_async_copy(
                y_hbm.at[pl.ds(pl.multiple_of(j * seg, seg), seg)],
                y_pad.at[s, me],
                y_sem.at[s % 2],
            )

        def x_loads(j):
            b = j % 2
            cps = [
                pltpu.make_async_copy(
                    x_hbm.at[seg_rows(j, r)], x_v.at[b, tile_rows(r)], x_sem.at[0, b]
                )
                for r in range(tp)
            ]
            if with_next and next_rope:
                cps += [
                    pltpu.make_async_copy(
                        rope_hbm.at[seg_rows(j, r)],
                        rope_v.at[b, tile_rows(r)],
                        x_sem.at[1, b],
                    )
                    for r in range(tp)
                ]
            return cps

        def out_writes(j):
            b = j % 2
            cps = [
                pltpu.make_async_copy(
                    x_v.at[b, tile_rows(r)],
                    xn_hbm.at[seg_rows(j, r)],
                    out_sem.at[0, b],
                )
                for r in range(tp)
            ]
            if with_next:
                for k, (v, h) in enumerate(
                    ((q_v, q_hbm), (kv_v, kv_hbm), (g_v, g_hbm))
                ):
                    cps += [
                        pltpu.make_async_copy(
                            v.at[b, tile_rows(r)],
                            h.at[seg_rows(j, r)],
                            out_sem.at[1 + k, b],
                        )
                        for r in range(tp)
                    ]
            return cps

        def wait_all(cps):
            for cp in cps:
                cp.wait_recv()
            for cp in cps:
                cp.wait_send()

        wait_all = _scope("pm_wait")(wait_all)

        y_load(0).start()
        for cp in w_cps:
            cp.wait()
        sm = sm_v[...]
        row = lambda m, name: m[SMALL[name] : SMALL[name] + 1]

        def ag_hops(kind, s, pad):
            if ag2d:
                return col.ag2_first(kind, s, pad), col.ag2_second(kind, s, pad)
            return col.ag_cross(kind, s, pad), None

        def ag_start(kinds_pads, s):
            # scales first, as before the 2D refactor
            for kind, pad in reversed(kinds_pads):
                for cp in ag_hops(kind, s, pad)[0]:
                    cp.start()

        def ag_relay(kinds_pads, s):
            for kind, pad in kinds_pads:
                first, second = ag_hops(kind, s, pad)
                wait_all(first)
                for cp in second:
                    cp.start()

        def ag_to_partner(kinds_pads, s):
            for kind, pad in kinds_pads:
                first, second = ag_hops(kind, s, pad)
                wait_all(second if ag2d else first)
                for cp in col.ag_pair(kind, s, pad):
                    cp.start()

        y_kp = (("y", y_pad),)
        u_kp = (("u", u_pad),) + ((("usc", usc_pad),) if fp8 else ())

        def start_x_loads(j):
            @pl.when(j >= 2)
            def _():
                for cp in out_writes(j - 2):
                    cp.wait()

            for cp in x_loads(j):
                cp.start()

        def s1(j):
            s = j % nslots
            y_load(j).wait()

            @pl.when(j + 1 < nt)
            def _():
                y_load(j + 1).start()

            ag_start(y_kp, s)

        def s2a(j):
            ag_relay(y_kp, j % nslots)

        def s2(j):
            ag_to_partner(y_kp, j % nslots)

        def s3(j):
            """Post-expert projection of tile j: this rank's output columns."""
            s, b = j % nslots, j % 2
            wait_all(col.ag_pair("y", s, y_pad))
            y = jnp.concatenate([y_pad[s, r] for r in range(tp)], axis=0)
            if ablate:
                u = jnp.zeros((y.shape[0], Hl), jnp.float32) + y[:, :1].astype(
                    jnp.float32
                )
            else:
                u = jnp.dot(y, wu_v[...], preferred_element_type=jnp.float32)
            u = (u * row(sm, "wunpool_s")[:, :Hl]).astype(jnp.bfloat16)
            if fp8:
                u_pad[s, me], usc_pad[s, me] = _to_fp8(u.astype(jnp.float32))
            else:
                u_pad[s, me] = u
            ag_start(u_kp, s)
            if not ag2d:
                start_x_loads(j)

        def s4a(j):
            ag_relay(u_kp, j % nslots)

        def s4(j):
            ag_to_partner(u_kp, j % nslots)
            if ag2d:
                start_x_loads(j)

        def s5(j):
            """Residual of tile j, then the next layer's attention inputs."""
            s, b = j % nslots, j % 2
            wait_all(col.ag_pair("u", s, u_pad))
            if fp8:
                wait_all(col.ag_pair("usc", s, usc_pad))
            for cp in x_loads(j):
                cp.wait()
            u = _gathered(u_pad, usc_pad, s, tp, 1)
            u = _rms(u.astype(jnp.float32) * row(sm, "post_ffn_w"), c.post_eps)
            x = row(sm, "ffn_alpha") * x_v[b] + row(sm, "ffn_beta") * u
            x_v[b] = x
            if with_next:
                smn = smn_v[...]
                a_in = (_rms(x, c.rms_eps) * row(smn, "pre_attn_w")).astype(
                    jnp.bfloat16
                )
                if ablate:
                    qkvg = jnp.zeros((a_in.shape[0], nq), jnp.float32) + a_in[
                        :, :1
                    ].astype(jnp.float32)
                else:
                    qkvg = jnp.dot(a_in, wq_v[...], preferred_element_type=jnp.float32)
                qkvg = qkvg * row(smn, "wqkvg_s")[:, :nq]
                q = _heads_rms(
                    qkvg[:, :qd], c.q_heads, hd, c.rms_eps, gain=c.scale_query_by
                )
                k = _heads_rms(qkvg[:, qd : qd + kvd], c.kv_heads, hd, c.rms_eps)
                if next_rope:
                    rp = rope_v[b]
                    q = _rope_roll(q, c.q_heads, hd, rp[:, :hd], rp[:, hd:])
                    k = _rope_roll(k, c.kv_heads, hd, rp[:, :hd], rp[:, hd:])
                q_v[b] = q.astype(jnp.bfloat16)
                kv_v[b] = jnp.concatenate(
                    [k, qkvg[:, qd + kvd : qd + 2 * kvd]], axis=1
                ).astype(jnp.bfloat16)
                g_v[b] = qkvg[:, qd + 2 * kvd :].astype(jnp.bfloat16)
            for cp in out_writes(j):
                cp.start()

        named = (
            [("s1", s1)]
            + ([("s2a", s2a)] if ag2d else [])
            + [("s2", s2), ("s3", s3)]
            + ([("s4a", s4a)] if ag2d else [])
            + [("s4", s4), ("s5", s5)]
        )
        stages = _gapped(
            tuple(_scope(f"pm_{n}")(st) for n, st in named),
            gap,
        )

        def body(i, carry):
            for k in reversed(range(len(stages))):
                if stages[k] is None:
                    continue
                j = i - k

                @pl.when(jnp.logical_and(j >= 0, j < nt))
                def _(st=stages[k], j=j):
                    st(j)

            return carry

        jax.lax.fori_loop(0, nt + len(stages) - 1, body, 0)
        for j in range(max(0, nt - 2), nt):
            for cp in out_writes(j):
                cp.wait()

    vmem = pltpu.VMEM
    sem = pltpu.SemaphoreType.DMA
    wq_shape = (H, nq) if with_next else (8, 128)
    scratch = [
        vmem((He, Hl), jnp.float8_e4m3fn),
        vmem((16, H), jnp.float32),
        vmem(wq_shape, jnp.float8_e4m3fn),
        vmem((16, H), jnp.float32),
        vmem((nslots, tp, seg, He), jnp.bfloat16),
        vmem((nslots, tp, tm, Hl), jnp.float8_e4m3fn if fp8 else jnp.bfloat16),
        vmem((2, tm, H), jnp.float32),
        vmem((2, tm, 2 * hd), jnp.float32),
        vmem((2, tm, qd), jnp.bfloat16),
        vmem((2, tm, 2 * kvd), jnp.bfloat16),
        vmem((2, tm, qd), jnp.bfloat16),
        sem((4,)),
        sem((2,)),
        sem((2, 2)),
        sem((4, 2)),
    ] + ([vmem((nslots, tp, 8, 128), jnp.float32)] if fp8 else [])
    scratch += [sem((nslots,)) for _ in range(2 * len(kinds))]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)

    def call(lyr, y, x, rope, wu, sm, wq):
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((T, H), jnp.float32),
                jax.ShapeDtypeStruct((T, qd), jnp.bfloat16),
                jax.ShapeDtypeStruct((T, 2 * kvd), jnp.bfloat16),
                jax.ShapeDtypeStruct((T, qd), jnp.bfloat16),
            ),
            in_specs=[smem] + [hbm] * 6,
            out_specs=(hbm,) * 4,
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(
                collective_id=_COLLECTIVE_ID + 1, vmem_limit_bytes=60 * 2**20
            ),
            name=(
                f"muse_prefill_post_moe_t{T}_m{tm}_r{next_rope}{'_fp8ag' if fp8 else ''}"
            ),
        )(lyr, y, x, rope, wu, sm, wq)

    return call
