"""Expert-parallel MoE for decode: one rank's whole experts over the slots an all-to-all
delivered to it (muse_megakernel/kernel_ep.py's layout).

Each rank holds E / tp whole NVFP4 experts (intermediate I = 4096). A decode step brings it
a few slots per expert, so the expert loop is weight-bound: every picked expert's weights
stream from HBM once, in I-chunks (the gate and up columns of a chunk and the matching down
rows). A work item of at most 4 slots goes through the decode megakernel's block-diagonal
fp8 dot (kernel._fp4_blockdiag_dot: e2m1 codes straight to the MXU, the scales applied after
the dot, so VPU work grows with the slots); a busier item dequantizes each weight block to
bf16 once for all its rows. Weights and x come in kperm order (kernel_weights, kernel_x) so
a dequantized vreg takes its block scales without a broadcast. The bf16 output of a slot is
silu(gate) * up @ w2, before routing weights and the post-expert norm.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.muse_megakernel.ep_mk.experts import (
    BD_KC,
    DENSE_ARMS,
    ChunkRefs,
    chunk_compute,
    fp4_weights_kperm,
    item_sizes,
)
from sglang_tpu.muse_megakernel.ep_mk.experts import sized as sized_switch
from sglang_tpu.muse_megakernel.ep_mk.experts import (
    x_kperm,
)
from sglang_tpu.muse_megakernel.mk.consts import FP4_GROUP, FP4_PACK

TILE_ROWS = 8  # every expert's rows start on an 8-row tile
kernel_weights = fp4_weights_kperm  # ep_moe_call's weight layout from the natural one
kernel_x = x_kperm  # ep_moe_call's x layout: hidden columns in kperm order


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
        assert self.item_rows in DENSE_ARMS and self.hidden % BD_KC == 0


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

    items: layout()'s. x: [rows, hidden] f32, each slot at its layout() row, columns as
    kernel_x(). w13, s13, w2, s2 as kernel_weights() of: w13 [E, hidden/8, 2*inter] u32 (8 e2m1 codes per word along K; gate columns then up),
    s13 [E, hidden/16, 2*inter] bf16, w2 [E, inter/8, hidden] u32, s2 [E, inter/16, hidden]
    bf16. y: [rows, hidden] f32 at the same rows; the others are garbage.
    """
    c.validate()
    He, I, CI, D = c.hidden, c.inter, c.chunk, c.depth
    NC, RI = c.n_chunks, c.item_rows
    sizes = item_sizes(RI)  # rows an item loads and stores

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
        wsems,
        xsem,
        ysem,
    ):
        n_items = items[0]
        total = n_items * NC

        def item(k):
            i = k // NC
            return items[1 + 3 * i], items[2 + 3 * i], items[3 + 3 * i], k % NC

        def sized(n, fn):
            sized_switch(n, sizes, fn)

        def x_copy(i, b, z):
            r0 = pl.multiple_of(items[2 + 3 * i], TILE_ROWS)
            return pltpu.make_async_copy(
                x_hbm.at[pl.ds(r0, z)], xbuf.at[b, pl.ds(0, z)], xsem.at[b]
            )

        def y_copy(i, b, z):
            r0 = pl.multiple_of(items[2 + 3 * i], TILE_ROWS)
            return pltpu.make_async_copy(
                acc.at[b, pl.ds(0, z)], y_hbm.at[pl.ds(r0, z)], ysem.at[b]
            )

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

        @pl.when(n_items > 0)
        def _():
            sized(items[3], lambda z: x_copy(0, 0, z).start())

        def body(k, carry):
            slot = k % D
            _, _, n, j = item(k)
            i = k // NC
            b = i % 2  # x and accumulator buffers alternate between items
            xb, ab = xbuf.at[b], acc.at[b]

            @pl.when(j == 0)
            def _():  # the item's rows (prefetched), the next item's, a fresh accumulator
                sized(n, lambda z: x_copy(i, b, z).wait())

                @pl.when(i + 1 < n_items)
                def _():
                    sized(items[6 + 3 * i], lambda z: x_copy(i + 1, 1 - b, z).start())

                def zero(z):
                    ab[pl.ds(0, z), :] = jnp.zeros((z, He), jnp.float32)

                sized(n, zero)

            start(k + D - 1)
            for cp in chunk_copies(k, slot):
                cp.wait()

            chunk_compute(
                "fp4",
                n,
                lambda R: xb[pl.ds(0, R), :],
                ab,
                ChunkRefs(
                    wg.at[slot],
                    wu.at[slot],
                    sg.at[slot],
                    su.at[slot],
                    w2b.at[slot],
                    s2b.at[slot],
                ),
                He,
                CI,
                RI,
            )

            @pl.when(j == NC - 1)
            def _():
                # A store may run into the next item's rows: the previous one lands first.
                @pl.when(i > 0)
                def _():
                    sized(items[3 * i], lambda z: y_copy(i - 1, 1 - b, z).wait())

                sized(n, lambda z: y_copy(i, b, z).start())

            return carry

        jax.lax.fori_loop(0, total, body, 0)

        @pl.when(n_items > 0)
        def _():
            last = n_items - 1
            sized(items[3 + 3 * last], lambda z: y_copy(last, last % 2, z).wait())

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pltpu.VMEM
    scratch = [
        vmem((2, RI, He), jnp.float32),  # xbuf
        vmem((2, RI, He), jnp.float32),  # acc
        vmem((D, He // FP4_PACK, CI), jnp.uint32),  # wg
        vmem((D, He // FP4_PACK, CI), jnp.uint32),  # wu
        vmem((D, He // FP4_GROUP, CI), jnp.bfloat16),  # sg
        vmem((D, He // FP4_GROUP, CI), jnp.bfloat16),  # su
        vmem((D, CI // FP4_PACK, He), jnp.uint32),  # w2b
        vmem((D, CI // FP4_GROUP, He), jnp.bfloat16),  # s2b
        pltpu.SemaphoreType.DMA((D, 6)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
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
