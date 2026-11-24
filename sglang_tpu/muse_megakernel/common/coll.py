# SPDX-License-Identifier: Apache-2.0
"""In-kernel remote-DMA collectives of the Muse megakernels.

At tp 8 ranks 2c and 2c + 1 are the two cores of chip c, and the 4 chips form a 2x2 grid
(chip ^ 1 / ^ 2 one ICI hop away, chip ^ 3 two). Three routes, one per message regime:

  direct  every rank sends to every other rank: 1 step, ~8 blocks of ICI link traffic per
          rank (diagonal sends cross 2 links). Latency-bound (small) steps.
  pair    the same core of the 3 other chips, then the partner core over die-to-die: 2 steps,
          half the ICI bytes. Coll.all_gather_start waits for every chip before relaying;
          Coll.all_gather_blocks_* and pair_all_gather_start relay each block on arrival.
  grid    two 1-hop cross-chip phases with relay, then the partner (GridColl): 3 steps, the
          fewest link bytes (3 blocks), no 2-hop routes; its reduce-scatter sums the pair
          first. Bandwidth-bound (prefill) steps, pipelined over row tiles (run_pipeline).

Coll serves direct and pair (decode), GridColl grid (prefill).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.common.consts import LANE

_FP8_MAX = 448.0


class Topo:
    """Rank arithmetic of tp 8 as 4 chips x 2 cores."""

    def __init__(self, me):
        self.me = me
        self.core, self.chip, self.partner = me % 2, me // 2, jnp.bitwise_xor(me, 1)

    @staticmethod
    def rank(chip, core):
        return chip * 2 + core

    def far(self, d):
        """This core's rank on chip ^ d (d in 1, 2, 3)."""
        return 2 * jnp.bitwise_xor(self.chip, d) + self.core

    def far_all(self) -> list:
        return [self.far(d) for d in (1, 2, 3)]


# ----------------------------------------------------------------------------- direct / pair
class Coll:
    """One-hop collectives over the mesh axis as remote DMAs into peers' VMEM pads.

    Every collective kind has its own pad and its own send/receive semaphores: a
    peer
    one round ahead then cannot satisfy this round's receive wait with its next
    round's
    bytes. A kind's pad is reused only after at least one other rendezvous round,
    so a
    single buffer per kind suffices. ``slot`` is the sender's position in the
    gathered
    result (its TP rank for column-parallel outputs, its mesh index otherwise).
    """

    def __init__(
        self,
        tp,
        me,
        axis,
        pads,
        ssems,
        rsems,
        order="ring",
        pair_ag=0,
        pair_ssems=None,
        pair_rsems=None,
        can_pair_fold=False,
    ):
        self.tp, self.me, self.axis = tp, me, axis
        self.pads, self.ssems, self.rsems = pads, ssems, rsems
        self.kinds = {k: i for i, k in enumerate(pads)}
        self.order = order if tp & (tp - 1) == 0 else "ring"
        self.pair_ag = pair_ag if (tp == 8 and pair_ssems is not None) else 0
        self.pair_ssems = pair_ssems
        self.pair_rsems = pair_rsems
        self.can_pair_fold = bool(self.pair_ag and can_pair_fold)
        self.topo = Topo(me)

    def _peer(self, d):
        if self.order == "xor":
            return jnp.bitwise_xor(self.me, d)
        return jax.lax.rem(self.me + d, self.tp)

    def _copy(self, kind, src, dst, peer):
        i = self.kinds[kind]
        return pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst,
            send_sem=self.ssems.at[i],
            recv_sem=self.rsems.at[i],
            device_id=(peer,),
        )

    def all_gather(self, kind, src_ref, slot, live=None):
        """``src_ref`` from every device into ``pad[slot_of_sender]``; returns the pad ref.

        ``live(m)``: whether device m's part is needed at all; parts of dead devices
        are
        neither sent nor waited for, and their pad slots hold stale data.
        """
        pad = self.pads[kind]
        if live is None or self.pair_ag >= 1:
            return self.all_gather_start(kind, src_ref, slot)()

        def local():
            pad[slot] = src_ref[...]

        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]

        @pl.when(live(self.me))
        def _():
            for c in cps:
                c.start()
            local()
            for c in cps:
                c.wait_send()

        pl.when(jnp.logical_not(live(self.me)))(local)
        for d, c in zip(range(1, self.tp), cps):

            @pl.when(live(self._peer(d)))
            def _(c=c):
                c.wait_recv()

        return pad

    def gather_few(self, kind, src_ref, slot):
        """``src_ref [F, W]`` from every device into ``pad[slot_of_sender]`` of the pad ``[tp, F, W]``."""
        pad = self.pads[kind]
        pad[slot] = src_ref[...]
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()
        for c in cps:
            c.wait()
        return pad

    def all_gather_start(self, kind, src_ref, slot):
        """Issue ``all_gather``; returns the function that waits for it and returns the pad."""
        pad = self.pads[kind]
        if self.pair_ag >= 1:
            ik = self.kinds[kind]
            core, partner = self.topo.core, self.topo.partner
            pad[self.me] = src_ref[...]
            cross_cps = [
                self._copy(kind, src_ref, pad.at[self.me], self.topo.far(d))
                for d in (1, 2, 3)
            ]
            for cp in cross_cps:
                cp.start()

            def wait_pair():
                for cp in cross_cps:
                    cp.wait()
                pair_cps = [
                    pltpu.make_async_remote_copy(
                        src_ref=pad.at[2 * c_k + core],
                        dst_ref=pad.at[2 * c_k + core],
                        send_sem=self.pair_ssems.at[ik],
                        recv_sem=self.pair_rsems.at[ik],
                        device_id=(partner,),
                    )
                    for c_k in range(4)
                ]
                for cp in pair_cps:
                    cp.start()
                for cp in pair_cps:
                    cp.wait()
                return pad

            return wait_pair

        def local():
            pad[slot] = src_ref[...]

        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()
        local()

        def wait():
            for c in cps:
                c.wait()
            return pad

        return wait

    def _block_copy_to(self, kind, src_ref, dst_slot, rsems, peer):
        return pltpu.make_async_remote_copy(
            src_ref=src_ref,
            dst_ref=self.pads[kind].at[dst_slot],
            send_sem=self.ssems.at[self.kinds[kind]],
            recv_sem=rsems.at[dst_slot],
            device_id=(peer,),
        )

    def all_gather_blocks_start(self, kind, src_ref, slot, rsems):
        """all_gather whose sends signal the receiver's rsems[slot], so a receiver can wait

        for one sender's block (all_gather_blocks_wait).
        """
        pad = self.pads[kind]
        if self.can_pair_fold:
            ik = self.kinds[kind]
            partner = self.topo.partner
            pad[self.me] = src_ref[...]
            pltpu.make_async_remote_copy(
                src_ref=src_ref,
                dst_ref=pad.at[self.me],
                send_sem=self.pair_ssems.at[ik],
                recv_sem=rsems.at[self.me],
                device_id=(partner,),
            ).start()
            for d in (1, 2, 3):
                peer = self.topo.far(d)
                self._block_copy_to(kind, src_ref, self.me, rsems, peer).start()
            return

        def local():
            pad[slot] = src_ref[...]

        for d in range(1, self.tp):
            self._block_copy_to(kind, src_ref, slot, rsems, self._peer(d)).start()
        local()

    def all_gather_blocks_wait(self, kind, src_ref, slot, rsems):
        """(wait_block(k), finish()) of an all_gather_blocks_start: the first waits

        for pad slot k (own slot: no-op), the second for this device's sends.
        """
        if self.can_pair_fold:
            ik = self.kinds[kind]
            core, chip, partner = self.topo.core, self.topo.chip, self.topo.partner
            pad = self.pads[kind]

            def wait_block_pair(k):
                @pl.when(k % 2 == 0)
                def _():
                    pg = k // 2

                    @pl.when(pg == chip)
                    def _():
                        k_other = 2 * chip + (1 - core)
                        pltpu.make_async_remote_copy(
                            src_ref=src_ref,
                            dst_ref=pad.at[k_other],
                            send_sem=self.pair_ssems.at[ik],
                            recv_sem=rsems.at[k_other],
                            device_id=(self.me,),
                        ).wait_recv()
                        pltpu.make_async_remote_copy(
                            src_ref=src_ref,
                            dst_ref=pad.at[self.me],
                            send_sem=self.pair_ssems.at[ik],
                            recv_sem=rsems.at[self.me],
                            device_id=(partner,),
                        ).wait_send()

                    @pl.when(pg != chip)
                    def _():
                        k_self = 2 * pg + core
                        k_other = 2 * pg + (1 - core)
                        self._block_copy_to(
                            kind, src_ref, k_self, rsems, self.me
                        ).wait_recv()
                        relay = pltpu.make_async_remote_copy(
                            src_ref=pad.at[k_self],
                            dst_ref=pad.at[k_self],
                            send_sem=self.pair_ssems.at[ik],
                            recv_sem=rsems.at[k_self],
                            device_id=(partner,),
                        )
                        relay.start()
                        pltpu.make_async_remote_copy(
                            src_ref=pad.at[k_other],
                            dst_ref=pad.at[k_other],
                            send_sem=self.pair_ssems.at[ik],
                            recv_sem=rsems.at[k_other],
                            device_id=(self.me,),
                        ).wait_recv()
                        relay.wait_send()

            def finish_pair():
                for d in (1, 2, 3):
                    peer = self.topo.far(d)
                    self._block_copy_to(kind, src_ref, self.me, rsems, peer).wait_send()

            return wait_block_pair, finish_pair

        def wait_block(k):
            pl.when(k != slot)(
                self._block_copy_to(kind, src_ref, k, rsems, self.me).wait_recv
            )

        def finish():
            for d in range(1, self.tp):
                self._block_copy_to(
                    kind, src_ref, slot, rsems, self._peer(d)
                ).wait_send()

        return wait_block, finish

    def block_copy(self, kind, send_ref, owner, rows_per):
        """Rows ``[owner*rows_per, ...)`` of ``send_ref`` into ``owner``'s pad slot ``me``."""
        return self._copy(
            kind,
            send_ref.at[pl.ds(pl.multiple_of(owner * rows_per, rows_per), rows_per)],
            self.pads[kind].at[self.me],
            owner,
        )

    def reduce_scatter_rows(self, kind, send_ref, rows_per, n_live, started=False):
        """bf16-packed ``send_ref [tp*rows_per, ...]``: device m receives everyone's rows

        ``[m*rows_per, ...)``; returns the pad ``[tp, rows_per, ...]`` (own slot
        filled locally).
        Devices whose rows are all past ``n_live`` neither receive nor get sent to.
        ``started``: every live owner's ``block_copy`` was already started; only
        wait.
        """
        pad = self.pads[kind]
        my_live = self.me * rows_per < n_live

        def local():
            @pl.when(my_live)
            def _():
                pad[self.me] = send_ref[
                    pl.ds(pl.multiple_of(self.me * rows_per, rows_per), rows_per)
                ]

        cps = []
        for d in range(1, self.tp):
            peer = self._peer(d)
            cps.append((peer, self.block_copy(kind, send_ref, peer, rows_per)))
        if not started:
            for peer, cp in cps:
                pl.when(peer * rows_per < n_live)(cp.start)
        local()
        for peer, cp in cps:
            pl.when(peer * rows_per < n_live)(cp.wait_send)

        @pl.when(my_live)
        def _():
            for _, cp in cps:
                cp.wait_recv()

        return pad

    def _cols_copy(self, kind, src_ref, width, m):
        c0 = m * width if isinstance(m, int) else pl.multiple_of(m * width, LANE)
        return self._copy(
            kind, src_ref.at[:, pl.ds(c0, width)], self.pads[kind].at[self.me], m
        )

    def reduce_scatter_cols(self, kind, src_ref, width):
        """``src_ref [R, tp*width]`` f32: device m receives everyone's columns ``[m*width, ...)``."""
        pad = self.pads[kind]

        def local():
            pad[self.me] = src_ref[
                :, pl.ds(pl.multiple_of(self.me * width, LANE), width)
            ]

        cps = [
            self._cols_copy(kind, src_ref, width, self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()
        local()
        for c in cps:
            c.wait()
        return pad

    def reduce_scatter_cols_streamed(self, kind, src_ref, width):
        """``reduce_scatter_cols`` issued one column block at a time: ``send(m)`` once

        ``src_ref[:, m*width:...]`` is written; ``finish()`` waits and returns the
        pad.
        """
        pad = self.pads[kind]

        def send(m):
            cp = self._cols_copy(kind, src_ref, width, m)
            pl.when(self.me != m)(cp.start)

            @pl.when(self.me == m)
            def _():
                c0 = (
                    m * width if isinstance(m, int) else pl.multiple_of(m * width, LANE)
                )
                pad[self.me] = src_ref[:, pl.ds(c0, width)]

        def finish():
            # every block is the same size: these descriptors count the same bytes
            cps = [
                self._cols_copy(kind, src_ref, width, self._peer(d))
                for d in range(1, self.tp)
            ]
            for cp in cps:
                cp.wait_send()
            for cp in cps:
                cp.wait_recv()
            return pad

        return send, finish


# ----------------------------------------------------------------------------- counted all-gathers
def _count_wait(tmpl, n: int) -> None:
    for _ in range(n):
        tmpl.wait()


def direct_all_gather_start(buf, me, tp: int, ssem, rsem):
    """direct: buf[me] into every rank's buf[me], counted on one (send, receive) semaphore
    pair; returns the wait."""
    for d in range(1, tp):
        pltpu.make_async_remote_copy(
            src_ref=buf.at[me],
            dst_ref=buf.at[me],
            send_sem=ssem,
            recv_sem=rsem,
            device_id=((me + d) % tp,),
        ).start()

    def wait():
        _count_wait(pltpu.make_async_copy(buf.at[0], buf.at[0], rsem), tp - 1)
        _count_wait(pltpu.make_async_copy(buf.at[0], buf.at[0], ssem), tp - 1)

    return wait


def pair_all_gather_start(buf, me, ssem, psem):
    """pair (tp 8): own block to the partner and to the same core of the 3 other chips; the
    wait relays each other chip's block to the partner on arrival. psem[k] counts block k."""
    topo = Topo(me)

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

    far = topo.far_all()
    send(me, topo.partner)
    for k in far:
        send(me, k)

    def wait():
        for k in far:
            arrived(k)
            send(k, topo.partner)
        arrived(topo.partner)
        for k in far:
            arrived(jnp.bitwise_xor(k, 1))
        _count_wait(pltpu.make_async_copy(buf.at[0], buf.at[0], ssem), 7)

    return wait


# ----------------------------------------------------------------------------- grid
def ag_kinds(kind: str) -> tuple:
    """Semaphore kinds of one grid all-gather: the two cross-chip phases, then the pair share."""
    return (kind + "_x", kind + "_y", kind + "_pair")


RS_KINDS = ("rs_pair", "rs_x", "rs_y")


class GridColl:
    """grid reduce-scatter / all-gather of [rows, tp * W] tiles (tp 8).

    Column block r belongs to rank r. sems[kind] holds a (send, receive) semaphore array with
    one entry per pad slot. Core parity 0 goes along x then y, parity 1 along y then x, so both
    links carry traffic in both phases.
    """

    def __init__(self, tp: int, me, sems: dict):
        if tp != 8:
            raise ValueError(f"GridColl routes over a 2x2 chip grid (tp 8), got {tp}")
        self.tp, self.me, self.sems = tp, me, sems
        self.nchip = tp // 2
        self.topo = Topo(me)
        self.core, self.chip, self.partner = (
            self.topo.core,
            self.topo.chip,
            self.topo.partner,
        )
        self.rank = Topo.rank

    def peers(self) -> list:
        """The partner core and the same core of every other chip."""
        return [self.partner] + [
            self.rank(jax.lax.rem(self.chip + d, self.nchip), self.core)
            for d in range(1, self.nchip)
        ]

    def barrier(self) -> None:
        sem = pltpu.get_barrier_semaphore()
        peers = self.peers()
        for p in peers:
            pl.semaphore_signal(
                sem, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(sem, len(peers))

    def _cp(self, kind, slot, src, dst, peer):
        ss, rs = self.sems[kind]
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

    # ------------------------------------------------------------------ reduce-scatter
    def rs_pair(self, slot, src, pad, w):
        """src [rows, tp * W]: the partner core's blocks go to the partner's pad[slot, chip]."""
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

    # Chips form a 2x2 grid: chip ^ 2 / chip ^ 1 / chip ^ 3 is the x / y / diagonal neighbour
    # (make_mesh orders chips by coords; any numbering is correct, only slower).
    def masks(self):
        """(first, second) neighbour masks of this core's parity."""
        m1 = jnp.where(self.core == 0, 2, 1).astype(jnp.int32)
        return m1, 3 - m1

    def _nbr(self, m):
        return self.rank(jnp.bitwise_xor(self.chip, m), self.core)

    def rs2_first(self, slot, pair_sums, pad):
        """Phase 1: the first neighbour's block and the diagonal chip's (which it relays) to
        its pad[slot, 0] / pad[slot, 1]."""
        m1, _ = self.masks()
        nb = self._nbr(m1)
        return [
            self._cp(
                "rs_x",
                slot,
                pair_sums.at[jnp.bitwise_xor(self.chip, m)],
                pad.at[slot, i],
                nb,
            )
            for i, m in enumerate((m1, 3))
        ]

    def rs2_second(self, slot, pad):
        """Phase 2: the second neighbour's block plus the relayed partial (pad[slot, 3]) to
        its pad[slot, 2]."""
        _, m2 = self.masks()
        return [self._cp("rs_y", slot, pad.at[slot, 3], pad.at[slot, 2], self._nbr(m2))]

    # ------------------------------------------------------------------ all-gather
    def ag2_first(self, kind, slot, pad):
        """Phase 1: this rank's block to the first neighbour."""
        m1, _ = self.masks()
        own = pad.at[slot, self.me]
        return [self._cp(kind + "_x", slot, own, own, self._nbr(m1))]

    def ag2_second(self, kind, slot, pad):
        """Phase 2: this rank's block and the first neighbour's to the second neighbour."""
        m1, m2 = self.masks()
        nb = self._nbr(m2)
        return [
            self._cp(kind + "_y", slot, pad.at[slot, r], pad.at[slot, r], nb)
            for r in (self.me, self._nbr(m1))
        ]

    def ag_pair(self, kind, slot, pad):
        """This core's gathered blocks (one per chip) to the partner."""
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

    def ag_start(self, kinds_pads, slot) -> None:
        # scale pads (listed last) go first
        for kind, pad in reversed(kinds_pads):
            for cp in self.ag2_first(kind, slot, pad):
                cp.start()

    def ag_relay(self, kinds_pads, slot) -> None:
        """Phase 1 landed: own and relayed blocks go along the second dimension."""
        for kind, pad in kinds_pads:
            wait_all(self.ag2_first(kind, slot, pad))
            for cp in self.ag2_second(kind, slot, pad):
                cp.start()

    def ag_to_partner(self, kinds_pads, slot) -> None:
        """Phase 2 landed: share this core's parity blocks on-chip."""
        for kind, pad in kinds_pads:
            wait_all(self.ag2_second(kind, slot, pad))
            for cp in self.ag_pair(kind, slot, pad):
                cp.start()

    def ag_wait(self, kinds_pads, slot) -> None:
        for kind, pad in kinds_pads:
            wait_all(self.ag_pair(kind, slot, pad))


def wait_all(cps) -> None:
    for cp in cps:
        cp.wait_recv()
    for cp in cps:
        cp.wait_send()


def to_fp8(x):
    """f32 block -> (e4m3 block, [8, 128] f32 holding its one scale)."""
    sc = jnp.maximum(jnp.max(jnp.abs(x)), 1e-30) / _FP8_MAX
    return (x / sc).astype(jnp.float8_e4m3fn), jnp.full((8, 128), sc, jnp.float32)


def gathered(pad, sc_pad, slot, n: int):
    """The n gathered blocks of pad[slot] side by side, dequantized when sc_pad is given."""
    if sc_pad is None:
        return jnp.concatenate([pad[slot, r] for r in range(n)], axis=1)
    return jnp.concatenate(
        [
            pad[slot, r].astype(jnp.float32) * sc_pad[slot, r][0:1, 0:1]
            for r in range(n)
        ],
        axis=1,
    )


def run_pipeline(stages: tuple, nt: int, gap: int = 0) -> None:
    """Software pipeline over nt row tiles: iteration i runs stage k on tile i - k.

    gap idle iterations follow every stage, so its remote DMAs get 1 + gap iterations to land
    before the next stage waits on them. Later stages run first in an iteration, so a stage's
    sends overlap the next stage's compute.
    """
    spaced = []
    for st in stages:
        spaced += [st] + [None] * gap
    spaced = spaced[: len(spaced) - gap]

    def body(i, carry):
        for k in reversed(range(len(spaced))):
            if spaced[k] is None:
                continue
            j = i - k

            @pl.when(jnp.logical_and(j >= 0, j < nt))
            def _(st=spaced[k], j=j):
                st(j)

        return carry

    jax.lax.fori_loop(0, nt + len(spaced) - 1, body, 0)
