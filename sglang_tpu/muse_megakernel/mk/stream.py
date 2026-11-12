# SPDX-License-Identifier: Apache-2.0
"""The dense weight stream (ring of K-major tiles)."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.mk.consts import LANE, SMALL
from sglang_tpu.muse_megakernel.mk.packing import _quant_fp8_rows
from sglang_tpu.muse_megakernel.mk.plan import _PACK
from sglang_tpu.muse_megakernel.mk.trace import _scope


class _WeightStream:
    """Every dense weight of a layer as one static tile sequence through one VMEM ring.

    The sequence is identical for every layer, so tile t of any layer lives in
    slot
    ``t % slots`` and the issue window runs ``depth`` tiles ahead across matvec
    and layer
    boundaries: attention, collectives and the expert phase do not leave the DMA
    engine idle
    while there is dense weight left to fetch. ``depth <= slots - 1``: the slot a
    new tile lands
    in held the tile consumed just before it. Slots are u32 words, viewed as the
    tile's dtype.
    """

    def __init__(
        self, W, buf, sems, tiles, depth, col_rot=None, k_rot=None, fp8_lhs=False
    ):
        self.W, self.buf, self.sems = W, buf, sems
        self.fp8_lhs = fp8_lhs
        # name -> traced shift: column tile ni of that weight streams physical column
        # (ni + shift) % n_cols (a per-device order)
        self.col_rot = col_rot or {}
        # name -> (traced own block, blocks per group G, tiles per group s_per): logical K tile
        # j * s_per + s streams the physical tile of group (own // G) ^ j (fold arrival order)
        self.k_rot = k_rot or {}
        # LAZY: while hold is set, refills of tiles >= hold go to self.held (issued by flush)
        self.hold, self.held = None, []
        self.slots = buf.shape[0]
        self.tiles, self.ranges = tiles, {}
        for t, (name, _, ni, tk, tn, dt) in enumerate(tiles):
            t0 = self.ranges[name][0] if name in self.ranges else t
            self.ranges[name] = (t0, t + 1, tk, tn, ni + 1, dt)
        self.T = len(self.tiles)
        if self.T % self.slots:
            raise ValueError(
                f"{self.T} dense tiles per layer do not divide into {self.slots} ring"
                " slots"
            )
        self.depth = min(depth, self.slots - 1, self.T)

    def _copy(self, l, t):
        name, ki, ni, tk, tn, dt = self.tiles[t]
        pack = _PACK[jnp.dtype(dt)]
        s_ = t % self.slots
        col = self.phys_col(name, ni)
        c0 = ni * tn if isinstance(col, int) else pl.multiple_of(col * tn, LANE)
        pk = self.phys_k(name, ki)
        r0 = (
            pk * (tk // pack)
            if isinstance(pk, int)
            else pl.multiple_of(pk * (tk // pack), 8)
        )
        src = (
            self.W[name].bitcast(jnp.uint32).at[l, pl.ds(r0, tk // pack), pl.ds(c0, tn)]
        )
        return pltpu.make_async_copy(
            src,
            self.buf.at[s_, pl.ds(0, tk // pack), pl.ds(0, tn)],
            self.sems.at[s_],
        )

    def phys_k(self, name, ki):
        if name not in self.k_rot:
            return ki
        own, G, s_per = self.k_rot[name]
        return jnp.bitwise_xor(own // G, ki // s_per) * s_per + ki % s_per

    def refill(self, l, t, next_l):
        """Issue the tile depth past tile t, which was just consumed."""
        u = t + self.depth
        if self.hold is not None and u >= self.hold:
            self.held.append((l, t, next_l))
            return
        if u < self.T:
            self._copy(l, u).start()
        elif next_l is not None:
            cp = self._copy(next_l, u - self.T)
            if isinstance(next_l, int):
                cp.start()
            else:
                # a rolled loop's last layer has no next layer
                pl.when(next_l < self.W["smalls"].shape[0])(cp.start)

    def flush(self):
        """Issue the held refills."""
        held, self.held, self.hold = self.held, [], None
        for l, t, next_l in held:
            self.refill(l, t, next_l)

    def mv_fold(self, l, names, next_l, G, bw, wait_block, lhs_block):
        """Matvecs of the fold group names over arriving K blocks of width bw.

        Group j holds the G physical blocks k = ((own // G) ^ j) * G + i; per block,
        wait_block(k) waits for it and lhs_block(k) gives the bf16 LHS of its K rows
        (any row
        count). Returns name -> unscaled f32 [rows, N].
        """
        own = self.k_rot[names[0]][0]
        geo = {n: self.ranges[n] for n in names}
        acc = {n: [None] * geo[n][4] for n in names}
        t, t_end = min(geo[n][0] for n in names), max(geo[n][1] for n in names)

        def group_of(u):
            return self.tiles[u][1] // (G * bw // self.tiles[u][3])

        while t < t_end:
            # one group: the consecutive tiles of its K blocks (dense_tiles order)
            j, gt = group_of(t), []
            while t < t_end and group_of(t) == j:
                gt.append(t)
                t += 1
            with _scope("ring_wait"):
                for u in gt:
                    self._copy(l, u).wait()
            pg = jnp.bitwise_xor(own // G, j)
            for i in range(G):
                k = pg * G + i
                wait_block(k)
                lhs = lhs_block(k)
                for u in gt:
                    name, ki, ni, tk, tn, dt = self.tiles[u]
                    bpt = tk // bw
                    s = ki % (G // bpt)
                    if i // bpt != s:
                        continue
                    r0 = (i % bpt) * bw
                    w = self.buf.at[u % self.slots].bitcast(dt)[
                        pl.ds(r0, bw), pl.ds(0, tn)
                    ]
                    if (
                        self.fp8_lhs
                        and jnp.dtype(dt) == jnp.dtype(jnp.float8_e4m3fn)
                        and lhs.shape[0] >= 32
                    ):
                        xf_d, xsc_d = _quant_fp8_rows(lhs)
                        part = (
                            jnp.dot(xf_d, w, preferred_element_type=jnp.float32) * xsc_d
                        )
                    else:
                        part = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
                    acc[name][ni] = (
                        part if acc[name][ni] is None else acc[name][ni] + part
                    )
            for u in gt:
                self.refill(l, u, next_l)
        return {
            n: a[0] if len(a) == 1 else jnp.concatenate(a, axis=1)
            for n, a in acc.items()
        }

    def phys_col(self, name, ni):
        if name not in self.col_rot:
            return ni
        return jax.lax.rem(ni + self.col_rot[name], self.ranges[name][4])

    def prologue(self, l):
        for t in range(self.depth):
            self._copy(l, t).start()

    def mv(self, x, l, name, next_l, sm=None, on_col=None, sm_ref=None):
        """``x [R, K]`` @ ``W_l[name]`` -> f32 ``[R, N]``; fp8 weights take their scale row from ``sm``.

        ``on_col(ni, col)`` gets each finished (scaled) column tile as soon as its
        last K tile is in.
        """
        assert name not in self.k_rot, "folded weights stream through mv_fold"
        t0, t1, tk, tn, nn, dt = self.ranges[name]
        nk = (t1 - t0) // nn
        is_fp8 = jnp.dtype(dt) == jnp.dtype(jnp.float8_e4m3fn)
        xb = x.astype(jnp.bfloat16)
        cols = [None] * nn
        for t in range(t0, t1):
            with _scope("ring_wait"):
                self._copy(l, t).wait()
            _, ki, ni, _, _, _ = self.tiles[t]
            w = self.buf.at[t % self.slots].bitcast(dt)[pl.ds(0, tk), pl.ds(0, tn)]
            lhs_s = xb[:, ki * tk : (ki + 1) * tk]
            if self.fp8_lhs and is_fp8 and xb.shape[0] >= 32:
                xf_d, xsc_d = _quant_fp8_rows(lhs_s)
                part = jnp.dot(xf_d, w, preferred_element_type=jnp.float32) * xsc_d
            else:
                part = jnp.dot(lhs_s, w, preferred_element_type=jnp.float32)
            cols[ni] = part if cols[ni] is None else cols[ni] + part
            self.refill(l, t, next_l)
            if on_col is not None and ki == nk - 1:
                col = self.phys_col(name, ni)
                if is_fp8 and isinstance(col, int):
                    r = SMALL[name + "_s"]
                    cols[ni] = cols[ni] * sm[r : r + 1, ni * tn : (ni + 1) * tn]
                elif (
                    is_fp8
                ):  # a rotated column's scales: a dynamic lane window of the ref
                    r = SMALL[name + "_s"]
                    cols[ni] = (
                        cols[ni]
                        * sm_ref[r : r + 1, pl.ds(pl.multiple_of(col * tn, LANE), tn)]
                    )
                on_col(col, cols[ni])
        out = cols[0] if nn == 1 else jnp.concatenate(cols, axis=1)
        if is_fp8 and on_col is None:
            r = SMALL[name + "_s"]
            out = out * sm[r : r + 1, : out.shape[1]]
        return out
