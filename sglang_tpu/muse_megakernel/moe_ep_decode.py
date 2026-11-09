"""Expert-parallel MoE for decode: one rank's whole experts over the slots an all-to-all
delivered to it (muse_megakernel/kernel_ep.py's layout).

Each rank holds E / tp whole NVFP4 experts (intermediate I = 4096). A decode step brings it
a few slots per expert, so the expert loop is weight-bound: every picked expert's weights
stream from HBM once, in I-chunks (the gate and up columns of a chunk and the matching down
rows), while 8-slot tiles of its rows go through the decode megakernel's block-diagonal fp8
dot (kernel._fp4_blockdiag_dot: e2m1 codes straight to the MXU, no VPU dequant, the scales
applied after the dot). The bf16 output of a slot is silu(gate) * up @ w2, before routing
weights and the post-expert norm.

The block-diagonal dot is kernel._fp4_blockdiag_dot's math, restructured for whole experts:
each slot row is spread over 16 LHS rows, one per 16-wide K group, so a few slots fill the
MXU and the block scales apply after the dot. Here an LHS is built once per K chunk and
reused by every column tile (and by gate and up), and the partials sum in VMEM.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.kernel import FP4_GROUP, FP4_PACK

TILE_ROWS = 8  # slots per block-diagonal dot: [8 * 16, 256] LHS
BD_KC = 256
BD_N_TILE = 256  # [128, 256] f32 accumulators: 32 vregs
ARMS = (1, 2, 4, 8)  # slots per tile's dot, as the megakernel's compact buckets


@dataclasses.dataclass(frozen=True)
class EPMoEConfig:
    slots: int  # M: slot capacity of the received buffer
    experts: int = 32  # whole experts on this rank
    hidden: int = 4096  # the pooled expert input / output width
    inter: int = 4096  # an expert's whole intermediate
    chunk: int = 512  # intermediate columns per weight DMA
    item_rows: int = 64  # rows of one work item; a busier expert takes several items
    depth: int = 3  # weight-chunk buffers in flight

    @property
    def max_items(self) -> int:
        return self.experts + -(-self.slots // self.item_rows)

    @property
    def rows(self) -> int:
        """The kernel's row buffer: each expert's slots start on an 8-row tile, plus one
        item of overrun."""
        return self.slots + (TILE_ROWS - 1) * self.experts + self.item_rows

    @property
    def n_chunks(self) -> int:
        return self.inter // self.chunk

    def validate(self) -> None:
        assert self.inter % self.chunk == 0 and self.chunk % BD_KC == 0
        assert self.item_rows % TILE_ROWS == 0 and self.hidden % BD_KC == 0


def _bd_lhs(xs: jax.Array) -> jax.Array:
    """[A, kc] bf16 -> [A * 16, kc]: row a * 16 + g keeps K group g of slot a."""
    A, kc = xs.shape
    rows = A * FP4_GROUP
    keep = (
        jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 0) % FP4_GROUP
        == jax.lax.broadcasted_iota(jnp.int32, (rows, kc), 1) // FP4_GROUP
    )
    spread = jnp.broadcast_to(xs[:, None, :], (A, FP4_GROUP, kc)).reshape(rows, kc)
    return jnp.where(keep, spread, jnp.zeros_like(spread))


def _bd_partial(lhs: jax.Array, w_ref, s_ref, acc_ref, k0, n_tile: int) -> None:
    """acc_ref[:A*16] += lhs @ fp8(e2m1 words w_ref[k0/8 :+ kc/8]) * their block scales."""
    rows, kc = lhs.shape
    A, N = rows // FP4_GROUP, w_ref.shape[-1]
    kw, kg = k0 // FP4_PACK, k0 // FP4_GROUP
    if not isinstance(k0, int):  # a traced chunk start: its alignment, for Mosaic
        kw, kg = pl.multiple_of(kw, kc // FP4_PACK), pl.multiple_of(kg, kc // FP4_GROUP)
    for n0 in range(0, N, n_tile):
        cols = pl.ds(n0, n_tile)
        w = pltpu.bitcast(
            w_ref[pl.ds(kw, kc // FP4_PACK), cols], jnp.float4_e2m1fn
        ).astype(jnp.float8_e4m3fn)
        sc = s_ref[pl.ds(kg, kc // FP4_GROUP), cols].astype(jnp.float32)
        res = jnp.dot(lhs, w, preferred_element_type=jnp.float32)
        acc_ref[pl.ds(0, rows), cols] += res * jnp.tile(sc, (A, 1))


def _group_sum(acc_ref, A: int, N: int) -> jax.Array:
    """[A, N]: each slot's 16 group rows summed."""
    return jnp.sum(
        acc_ref[pl.ds(0, A * FP4_GROUP), pl.ds(0, N)].reshape(A, FP4_GROUP, N), axis=1
    )


def layout(
    c: EPMoEConfig, local_e: jax.Array, n_recv: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """(items, rows): the kernel's SMEM table [n_items, (expert, first row, rows) per
    item...] and each slot's row in the kernel's buffer, where every expert's slots start on
    an 8-row tile (HBM is tiled (8, 128): DMAs start on a tile).

    Args:
      local_e: [slots] each received slot's local expert, sorted ascending over the first
        n_recv slots (the rest are padding; their rows land past the experts').
    """
    valid = jnp.arange(c.slots) < n_recv
    cnt = jnp.bincount(jnp.where(valid, local_e, c.experts), length=c.experts + 1)[
        : c.experts
    ]
    off = jnp.cumsum(cnt) - cnt
    tiles = -(-cnt // TILE_ROWS) * TILE_ROWS
    aoff = jnp.cumsum(tiles) - tiles
    e_of = jnp.clip(local_e, 0, c.experts - 1)
    slot = jnp.arange(c.slots)
    rows = jnp.where(
        valid, aoff[e_of] + slot - off[e_of], jnp.sum(tiles) + slot - n_recv
    ).astype(jnp.int32)
    nb = -(-cnt // c.item_rows)
    base = jnp.cumsum(nb) - nb
    q = jnp.arange(c.max_items)
    e = jnp.clip(jnp.searchsorted(base + nb, q, side="right"), 0, c.experts - 1)
    b = q - base[e]
    r0 = aoff[e] + b * c.item_rows
    n = jnp.clip(cnt[e] - b * c.item_rows, 0, c.item_rows)
    table = jnp.stack([e, r0, n], axis=1).astype(jnp.int32).reshape(-1)
    return jnp.concatenate([jnp.sum(nb).astype(jnp.int32)[None], table]), rows


def ep_moe_call(c: EPMoEConfig):
    """fn(items, x, w13, s13, w2, s2) -> y.

    items: layout()'s. x: [rows, hidden] f32, each slot at its layout() row.
    w13 [E, hidden/8, 2*inter] u32 (8 e2m1 codes per word along K; gate columns then up),
    s13 [E, hidden/16, 2*inter] bf16, w2 [E, inter/8, hidden] u32, s2 [E, inter/16, hidden]
    bf16. y: [rows, hidden] f32 at the same rows; the others are garbage.
    """
    c.validate()
    He, I, CI, D = c.hidden, c.inter, c.chunk, c.depth
    NC, RI = c.n_chunks, c.item_rows

    def kernel(
        items,
        x_hbm,
        w13_hbm,
        s13_hbm,
        w2_hbm,
        s2_hbm,
        y_hbm,
        xbuf,
        acc,
        wg,
        wu,
        sg,
        su,
        w2b,
        s2b,
        ybuf,
        part_g,
        part_u,
        part_y,
        wsems,
        xsem,
        ysem,
    ):
        total = items[0] * NC

        def item(k):
            i = k // NC
            return items[1 + 3 * i], items[2 + 3 * i], items[3 + 3 * i], k % NC

        def chunk_copies(k, slot):
            e, _, _, j = item(k)
            g0, u0 = pl.multiple_of(j * CI, CI), pl.multiple_of(I + j * CI, CI)
            k2 = pl.multiple_of(j * (CI // FP4_PACK), CI // FP4_PACK)
            k2s = pl.multiple_of(j * (CI // FP4_GROUP), CI // FP4_GROUP)
            return [
                pltpu.make_async_copy(
                    w13_hbm.at[e, :, pl.ds(g0, CI)], wg.at[slot], wsems.at[slot, 0]
                ),
                pltpu.make_async_copy(
                    w13_hbm.at[e, :, pl.ds(u0, CI)], wu.at[slot], wsems.at[slot, 1]
                ),
                pltpu.make_async_copy(
                    s13_hbm.at[e, :, pl.ds(g0, CI)], sg.at[slot], wsems.at[slot, 2]
                ),
                pltpu.make_async_copy(
                    s13_hbm.at[e, :, pl.ds(u0, CI)], su.at[slot], wsems.at[slot, 3]
                ),
                pltpu.make_async_copy(
                    w2_hbm.at[e, pl.ds(k2, CI // FP4_PACK)],
                    w2b.at[slot],
                    wsems.at[slot, 4],
                ),
                pltpu.make_async_copy(
                    s2_hbm.at[e, pl.ds(k2s, CI // FP4_GROUP)],
                    s2b.at[slot],
                    wsems.at[slot, 5],
                ),
            ]

        def start(k):
            @pl.when(k < total)
            def _():
                for cp in chunk_copies(k, k % D):
                    cp.start()

        for p in range(D - 1):  # the first chunks in flight
            start(jnp.int32(p))

        def body(k, carry):
            slot = k % D
            _, r0, n, j = item(k)
            r0 = pl.multiple_of(r0, TILE_ROWS)  # layout() starts every expert on a tile
            ntiles = (n + TILE_ROWS - 1) // TILE_ROWS

            @pl.when(j == 0)
            def _():  # the item's rows, a fresh accumulator
                cp = pltpu.make_async_copy(x_hbm.at[pl.ds(r0, RI)], xbuf, xsem.at[0])
                cp.start()
                cp.wait()
                acc[...] = jnp.zeros_like(acc)

            start(k + D - 1)
            for cp in chunk_copies(k, slot):
                cp.wait()

            def arm(A):
                """The tile's first A slots."""

                def run(t):
                    rows = pl.ds(pl.multiple_of(t * TILE_ROWS, TILE_ROWS), A)
                    zg = jnp.zeros((A * FP4_GROUP, CI), jnp.float32)
                    part_g[pl.ds(0, A * FP4_GROUP), :] = zg
                    part_u[pl.ds(0, A * FP4_GROUP), :] = zg

                    def gate_up(ci, carry3):
                        k0 = pl.multiple_of(ci * BD_KC, BD_KC)
                        lhs = _bd_lhs(xbuf[rows, pl.ds(k0, BD_KC)].astype(jnp.bfloat16))
                        _bd_partial(
                            lhs, wg.at[slot], sg.at[slot], part_g, k0, BD_N_TILE
                        )
                        _bd_partial(
                            lhs, wu.at[slot], su.at[slot], part_u, k0, BD_N_TILE
                        )
                        return carry3

                    jax.lax.fori_loop(0, He // BD_KC, gate_up, 0)
                    g, u = _group_sum(part_g, A, CI), _group_sum(part_u, A, CI)
                    h = (g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16)
                    part_y[pl.ds(0, A * FP4_GROUP), :] = jnp.zeros(
                        (A * FP4_GROUP, He), jnp.float32
                    )
                    for k0 in range(0, CI, BD_KC):
                        _bd_partial(
                            _bd_lhs(h[:, k0 : k0 + BD_KC]),
                            w2b.at[slot],
                            s2b.at[slot],
                            part_y,
                            k0,
                            BD_N_TILE,
                        )
                    acc[rows, :] += _group_sum(part_y, A, He)

                return run

            arms = [arm(A) for A in ARMS]

            def tile(t, carry2):
                left = jnp.minimum(n - t * TILE_ROWS, TILE_ROWS)
                which = sum((left > A).astype(jnp.int32) for A in ARMS[:-1])
                jax.lax.switch(which, arms, t)
                return carry2

            jax.lax.fori_loop(0, ntiles, tile, 0)

            @pl.when(j == NC - 1)
            def _():
                # 8-row tiles may run into the next item's rows; it writes them after.
                def store(t, carry2):
                    rows = pl.ds(pl.multiple_of(t * TILE_ROWS, TILE_ROWS), TILE_ROWS)
                    ybuf[...] = acc[rows, :]
                    cp = pltpu.make_async_copy(
                        ybuf,
                        y_hbm.at[
                            pl.ds(
                                pl.multiple_of(r0 + t * TILE_ROWS, TILE_ROWS), TILE_ROWS
                            )
                        ],
                        ysem.at[0],
                    )
                    cp.start()
                    cp.wait()
                    return carry2

                jax.lax.fori_loop(0, ntiles, store, 0)

            return carry

        jax.lax.fori_loop(0, total, body, 0)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pltpu.VMEM
    scratch = [
        vmem((RI, He), jnp.float32),  # xbuf
        vmem((RI, He), jnp.float32),  # acc
        vmem((D, He // FP4_PACK, CI), jnp.uint32),  # wg
        vmem((D, He // FP4_PACK, CI), jnp.uint32),  # wu
        vmem((D, He // FP4_GROUP, CI), jnp.bfloat16),  # sg
        vmem((D, He // FP4_GROUP, CI), jnp.bfloat16),  # su
        vmem((D, CI // FP4_PACK, He), jnp.uint32),  # w2b
        vmem((D, CI // FP4_GROUP, He), jnp.bfloat16),  # s2b
        vmem((TILE_ROWS, He), jnp.float32),  # ybuf
        vmem((TILE_ROWS * FP4_GROUP, CI), jnp.float32),  # part_g: gate partials
        vmem((TILE_ROWS * FP4_GROUP, CI), jnp.float32),  # part_u: up partials
        vmem((TILE_ROWS * FP4_GROUP, He), jnp.float32),  # part_y: down partials
        pltpu.SemaphoreType.DMA((D, 6)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((1,)),
    ]

    def call(items, x, w13, s13, w2, s2):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((c.rows, He), jnp.float32),
            in_specs=[smem] + [hbm] * 5,
            out_specs=hbm,
            scratch_shapes=scratch,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 2**20),
            name=f"muse_ep_moe_decode_m{c.slots}_e{c.experts}",
        )(items, x, w13, s13, w2, s2)

    return call


def reference(
    c: EPMoEConfig,
    local_e: jax.Array,
    n_recv: jax.Array,
    x: jax.Array,
    w13: jax.Array,
    s13: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
) -> jax.Array:
    """XLA math of ep_moe_call for the first n_recv slots (f32; others zero)."""
    from sglang_tpu.muse_megakernel import kernel_ep

    def deq(words, scale):
        return kernel_ep.dequant(words.view(jnp.int32), scale, "fp4")

    out = jnp.zeros((c.slots, c.hidden), jnp.float32)
    xs = x[: c.slots]
    valid = jnp.arange(c.slots) < n_recv
    for e in range(c.experts):
        mask = valid & (local_e == e)
        gu = jnp.dot(
            xs.astype(jnp.bfloat16).astype(jnp.float32),
            deq(w13[e], s13[e]),
            precision=jax.lax.Precision.HIGHEST,
        )
        g, u = gu[:, : c.inter], gu[:, c.inter :]
        h = (g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16).astype(jnp.float32)
        y = jnp.dot(h, deq(w2[e], s2[e]), precision=jax.lax.Precision.HIGHEST)
        out = jnp.where(mask[:, None], y, out)
    return out
