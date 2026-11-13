# SPDX-License-Identifier: Apache-2.0
"""In-kernel collectives: one-hop remote-DMA all-gathers and reduce-scatters."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.mk.consts import LANE


# ----------------------------------------------------------------------------- collectives
class _Coll:
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
        local_last=False,
        pair_ag=0,
        pair_ssems=None,
        pair_rsems=None,
        can_pair_fold=False,
    ):
        self.tp, self.me, self.axis = tp, me, axis
        self.pads, self.ssems, self.rsems = pads, ssems, rsems
        self.kinds = {k: i for i, k in enumerate(pads)}
        self.order = order if tp & (tp - 1) == 0 else "ring"
        self.local_last = local_last
        self.pair_ag = pair_ag if (tp == 8 and pair_ssems is not None) else 0
        self.pair_ssems = pair_ssems
        self.pair_rsems = pair_rsems
        self.can_pair_fold = bool(self.pair_ag and can_pair_fold)

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

    def _local(self, fn, issued):
        """The round's own-part copy: before its remote copies, or after them with

        local_last (``issued`` False / True).
        """
        if self.local_last == issued:
            fn()

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

        self._local(local, False)
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]

        @pl.when(live(self.me))
        def _():
            for c in cps:
                c.start()
            self._local(local, True)
            for c in cps:
                c.wait_send()

        if self.local_last:
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
            core = self.me % 2
            chip = self.me // 2
            partner = jnp.bitwise_xor(self.me, 1)
            pad[self.me] = src_ref[...]
            cross_cps = [
                self._copy(
                    kind,
                    src_ref,
                    pad.at[self.me],
                    2 * jnp.bitwise_xor(chip, d) + core,
                )
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

        self._local(local, False)
        cps = [
            self._copy(kind, src_ref, pad.at[slot], self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()
        self._local(local, True)

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
            core = self.me % 2
            chip = self.me // 2
            partner = jnp.bitwise_xor(self.me, 1)
            pad[self.me] = src_ref[...]
            pltpu.make_async_remote_copy(
                src_ref=src_ref,
                dst_ref=pad.at[self.me],
                send_sem=self.pair_ssems.at[ik],
                recv_sem=rsems.at[self.me],
                device_id=(partner,),
            ).start()
            for d in (1, 2, 3):
                peer = 2 * jnp.bitwise_xor(chip, d) + core
                self._block_copy_to(kind, src_ref, self.me, rsems, peer).start()
            return

        def local():
            pad[slot] = src_ref[...]

        self._local(local, False)
        for d in range(1, self.tp):
            self._block_copy_to(kind, src_ref, slot, rsems, self._peer(d)).start()
        self._local(local, True)

    def all_gather_blocks_wait(self, kind, src_ref, slot, rsems):
        """(wait_block(k), finish()) of an all_gather_blocks_start: the first waits

        for pad slot k (own slot: no-op), the second for this device's sends.
        """
        if self.can_pair_fold:
            ik = self.kinds[kind]
            core = self.me % 2
            chip = self.me // 2
            partner = jnp.bitwise_xor(self.me, 1)
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
                    peer = 2 * jnp.bitwise_xor(chip, d) + core
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

        self._local(local, False)
        cps = []
        for d in range(1, self.tp):
            peer = self._peer(d)
            cps.append((peer, self.block_copy(kind, send_ref, peer, rows_per)))
        if not started:
            for peer, cp in cps:
                pl.when(peer * rows_per < n_live)(cp.start)
        self._local(local, True)
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

        self._local(local, False)
        cps = [
            self._cols_copy(kind, src_ref, width, self._peer(d))
            for d in range(1, self.tp)
        ]
        for c in cps:
            c.start()
        self._local(local, True)
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
