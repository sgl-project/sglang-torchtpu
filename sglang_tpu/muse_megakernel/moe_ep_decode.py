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

from sglang_tpu.muse_megakernel.kernel import FP4_GROUP, FP4_PACK, _fp4_blockdiag_dot

TILE_ROWS = 8  # every expert's rows start on an 8-row tile
BD_KC = 256
BD_N_TILE = 256  # [128, 256] f32 accumulators: 32 vregs
BD_ARMS = (1, 2, 4)  # block-diagonal arms: an item of at most 4 slots
DENSE_ARMS = (8, 16, 32, 64)  # bf16-dequant arms: rows of the item's plain dot
W2_N_TILE = 512  # down-projection columns per dot below 4 slots ([32, 512] f32 at 2)
DQ_KC, DQ_TN = 256, 256  # one bf16-dequantized weight block: 32 vregs


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


def kperm(n: int) -> jax.Array:
    """The kernel's order of n indices: in each 256-block, position r * 16 + g holds index
    g * 16 + r, so a 16-row bf16 vreg of dequantized weights spans 16 scale groups and takes
    the block's [16, N] scales as they are (an involution)."""
    p = jnp.arange(n)
    return p - p % BD_KC + (p % 16) * 16 + (p % BD_KC) // 16


def _permute_fp4_rows(words: jax.Array) -> jax.Array:
    """u32 words [..., K/8, N] (nibble j of word w is row 8w + j) with rows in kperm order."""
    *lead, kw, n = words.shape
    shifts = jnp.arange(0, 32, 4, dtype=jnp.uint32)[:, None]
    codes = ((words[..., :, None, :] >> shifts) & 0xF).reshape(*lead, kw * FP4_PACK, n)
    codes = codes[..., kperm(kw * FP4_PACK), :].reshape(*lead, kw, FP4_PACK, n)
    return jnp.sum(codes << shifts, axis=-2, dtype=jnp.uint32)


def kernel_weights(
    w13: jax.Array, s13: jax.Array, w2: jax.Array, s2: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """ep_moe_call's weight layout from the natural one: hidden rows of w13 and intermediate
    rows of w2 in kperm order, and the gate / up columns in kperm order too, so h comes out
    in the order w2's rows are in. Block scales keep their group order."""
    cols = kperm(w13.shape[-1])
    return _permute_fp4_rows(w13)[..., cols], s13[..., cols], _permute_fp4_rows(w2), s2


def kernel_x(x: jax.Array) -> jax.Array:
    """ep_moe_call's x layout: hidden columns in kperm order."""
    return x[..., kperm(x.shape[-1])]


def _dequant_dot(x, wref, sref, K, N):
    """x [R, K] bf16 (kperm order) @ an NVFP4 slab dequantized to bf16 per [DQ_KC, DQ_TN]
    block -> f32 [R, N]; one dequant serves every row (code x scale rounded to bf16)."""
    cols = []
    for n0 in range(0, N, DQ_TN):
        acc = None
        for k0 in range(0, K, DQ_KC):
            w = pltpu.bitcast(
                wref[pl.ds(k0 // FP4_PACK, DQ_KC // FP4_PACK), pl.ds(n0, DQ_TN)],
                jnp.float4_e2m1fn,
            ).astype(jnp.bfloat16)
            s = sref[pl.ds(k0 // FP4_GROUP, DQ_KC // FP4_GROUP), pl.ds(n0, DQ_TN)]
            w = (w.reshape(FP4_GROUP, DQ_KC // FP4_GROUP, DQ_TN) * s[None]).reshape(
                DQ_KC, DQ_TN
            )
            part = jnp.dot(x[:, k0 : k0 + DQ_KC], w, preferred_element_type=jnp.float32)
            acc = part if acc is None else acc + part
        cols.append(acc)
    return cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=1)


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

            def bd_arm(A):
                """At most 4 slots: an A-row block-diagonal dot (A * 16 LHS rows)."""

                def run():
                    rows = pl.ds(0, A)
                    xt = xbuf[rows, :]
                    g = _fp4_blockdiag_dot(
                        xt,
                        wg.at[slot],
                        sg.at[slot],
                        He,
                        CI,
                        BD_KC,
                        n_tile=BD_N_TILE,
                        kperm=True,
                    )
                    u = _fp4_blockdiag_dot(
                        xt,
                        wu.at[slot],
                        su.at[slot],
                        He,
                        CI,
                        BD_KC,
                        n_tile=BD_N_TILE,
                        kperm=True,
                    )
                    h = (g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16)
                    if A >= 4:  # _fp4_blockdiag_dot tiles N itself from 4 rows up
                        acc[rows, :] += _fp4_blockdiag_dot(
                            h,
                            w2b.at[slot],
                            s2b.at[slot],
                            CI,
                            He,
                            BD_KC,
                            kperm=True,
                            n_tile=BD_N_TILE,
                        )
                        return
                    for n0 in range(0, He, W2_N_TILE):
                        cols = pl.ds(n0, W2_N_TILE)
                        acc[rows, cols] += _fp4_blockdiag_dot(
                            h,
                            w2b.at[slot, :, cols],
                            s2b.at[slot, :, cols],
                            CI,
                            W2_N_TILE,
                            BD_KC,
                            kperm=True,
                        )

                return run

            def dense_arm(R):
                """Up to R slots: dequantize each weight block once for all R rows."""

                def run():
                    xt = xbuf[pl.ds(0, R), :].astype(jnp.bfloat16)
                    g = _dequant_dot(xt, wg.at[slot], sg.at[slot], He, CI)
                    u = _dequant_dot(xt, wu.at[slot], su.at[slot], He, CI)
                    h = (g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16)
                    for n0 in range(0, He, DQ_TN):
                        cols = pl.ds(n0, DQ_TN)
                        acc[pl.ds(0, R), cols] += _dequant_dot(
                            h, w2b.at[slot, :, cols], s2b.at[slot, :, cols], CI, DQ_TN
                        )

                return run

            dense = tuple(R for R in DENSE_ARMS if R <= RI)
            arms = [bd_arm(A) for A in BD_ARMS] + [dense_arm(R) for R in dense]
            bounds = BD_ARMS + dense[:-1]
            jax.lax.switch(sum((n > b).astype(jnp.int32) for b in bounds), arms)

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
