# SPDX-License-Identifier: Apache-2.0
"""Two-level remote-DMA collectives and the row-tile pipeline of the prefill kernels.

Ranks 2c and 2c + 1 are the two cores of chip c. A reduce-scatter sums the pair die-to-die
first; cross-chip hops then take two 1-hop phases over the 2x2 chip grid (core parity 0
along x then y, parity 1 along y then x), so no transfer crosses two links.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_FP8_MAX = 448.0


def ag_kinds(kind: str) -> tuple:
    """Semaphore kinds of one all-gather: the two cross-chip phases, then the pair share."""
    return (kind + "_x", kind + "_y", kind + "_pair")


RS_KINDS = ("rs_pair", "rs_x", "rs_y")


class PairColl:
    """Reduce-scatter / all-gather of [rows, tp * W] tiles over (chip, core) ranks.

    Column block r belongs to rank r. sems[kind] holds a (send, receive) semaphore array with
    one entry per pad slot.
    """

    def __init__(self, tp: int, me, sems: dict):
        if tp != 8:
            raise ValueError(f"PairColl routes over a 2x2 chip grid (tp 8), got {tp}")
        self.tp, self.me, self.sems = tp, me, sems
        self.nchip = tp // 2
        self.core, self.chip, self.partner = me % 2, me // 2, jnp.bitwise_xor(me, 1)

    def rank(self, chip, core):
        return chip * 2 + core

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
