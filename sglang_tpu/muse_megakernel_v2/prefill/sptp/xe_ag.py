# SPDX-License-Identifier: Apache-2.0
"""The expert input's AG in row chunks, each chunk dequantized to bf16 by a Pallas pass as it lands
(SPTP_XE_AG_CHUNKS). XLA dequantized the whole [T, He] after the AG into a transposed layout and a
SparseCore layout conversion: ~40 us / layer, exposed between the AG and the MoE."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def xe_ag_dequant_call(
    rows: int, cols: int, tp: int, n: int, axis: str, tiled: int = 0
):
    """(xe8 [Ts, cols] e4m3 own rows, xs_all [rows, 1] f32 every row's scale) -> bf16 [rows, cols]
    in the rank-major row order of a tiled AG; n all-gathers of Ts / n rows, dequantized per chunk.

    tiled: bf16 [rows, cols / 128, 128], the fused-RS MoE's lhs layout (one row = 32 sublane
    rows). Its [rows, cols] input reshape is otherwise a SparseCore layout conversion (~22 us)."""
    Ts = rows // tp
    sub = Ts // n
    nk = cols // 128
    if Ts % n or sub % 32 or cols % 256:
        raise ValueError(f"xe AG: {Ts} own rows in {n} chunks, {cols} cols")

    def kernel(x_ref, s_ref, *rest):
        # (aliased out in HBM,) out, (tiled 2: packed scratch)
        o_ref = rest[-2] if tiled == 2 else rest[-1]
        if not tiled:
            o_ref[...] = (x_ref[...].astype(jnp.float32) * s_ref[...]).astype(
                jnp.bfloat16
            )
            return
        # u32 row (r, i) of the tiled output packs bf16 rows 2i (low half) and 2i + 1 of row r
        nh = nk // 2
        ou = o_ref.bitcast(jnp.uint32).reshape(sub * nh, 128)
        sc = s_ref[...]

        def bits(j):
            v = x_ref[:, j * 128 : (j + 1) * 128].astype(jnp.float32) * sc
            v = v.astype(jnp.bfloat16).astype(jnp.float32)
            return pltpu.bitcast(v, jnp.uint32)

        if tiled == 3:
            # packed column blocks [nh, sub, 128] -> [sub, nh, 128] by a major/sublane einshape
            w = jnp.stack(
                [
                    (bits(2 * i + 1) & jnp.uint32(0xFFFF0000)) | (bits(2 * i) >> 16)
                    for i in range(nh)
                ]
            )
            ou[...] = pltpu.einshape("irl->ril", w).reshape(sub * nh, 128)
            return
        if tiled == 1:
            for i in range(nh):
                w = (bits(2 * i + 1) & jnp.uint32(0xFFFF0000)) | (bits(2 * i) >> 16)
                ou[pl.ds(i, sub, stride=nh), :] = w
            return
        # tiled 2: packed column blocks i-major in VMEM, then each output vreg (8 u32 rows of
        # one row r) is one stride-sub load
        w_ref = rest[-1]
        for i in range(nh):
            w_ref[pl.ds(i * sub, sub), :] = (
                bits(2 * i + 1) & jnp.uint32(0xFFFF0000)
            ) | (bits(2 * i) >> 16)

        def row(r, carry):
            for h in range(nh // 8):
                ou[pl.ds(pl.multiple_of(r * nh + h * 8, 8), 8), :] = w_ref[
                    pl.ds(h * 8 * sub + r, 8, stride=sub), :
                ]
            return carry

        jax.lax.fori_loop(0, sub, row, 0, unroll=8)

    def chunk(j, x8, s_all, out):
        # gathered chunk block r = rank r's rows [j * sub, (j + 1) * sub): global block r * n + j
        in_specs = [
            pl.BlockSpec((sub, cols), lambda r: (r, 0)),
            pl.BlockSpec((sub, 1), lambda r: (r * n + j, 0)),
        ]
        args = [x8, s_all]
        aliases = {}
        if out is not None:
            in_specs.append(pl.BlockSpec(memory_space=pl.ANY))
            args.append(out)
            aliases = {2: 0}
        return pl.pallas_call(
            kernel,
            grid=(tp,),
            in_specs=in_specs,
            out_specs=(
                pl.BlockSpec((sub, nk, 128), lambda r: (r * n + j, 0, 0))
                if tiled
                else pl.BlockSpec((sub, cols), lambda r: (r * n + j, 0))
            ),
            out_shape=jax.ShapeDtypeStruct(
                (rows, nk, 128) if tiled else (rows, cols), jnp.bfloat16
            ),
            input_output_aliases=aliases,
            scratch_shapes=(
                [pltpu.VMEM((sub * nk // 2, 128), jnp.uint32)] if tiled == 2 else []
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            name=f"muse_sptp_xe_ag_dequant_t{rows}_n{n}{f'_tl{tiled}' if tiled else ''}",
        )(*args)

    def kernel3d(x_ref, s_ref, o_ref):
        o_ref[...] = (x_ref[...].astype(jnp.float32) * s_ref[...][:, :, None]).astype(
            jnp.bfloat16
        )

    def call3d(xe8, xs_all):
        # tiled 4: own rows to [Ts, nk, 128] before the AG (1/tp of the relayout), then an
        # elementwise dequant of the gathered tiles
        g = jax.lax.all_gather(xe8.reshape(Ts, nk, 128), axis, axis=0, tiled=True)
        tm = 256
        return pl.pallas_call(
            kernel3d,
            grid=(rows // tm,),
            in_specs=[
                pl.BlockSpec((tm, nk, 128), lambda i: (i, 0, 0)),
                pl.BlockSpec((tm, 1), lambda i: (i, 0)),
            ],
            out_specs=pl.BlockSpec((tm, nk, 128), lambda i: (i, 0, 0)),
            out_shape=jax.ShapeDtypeStruct((rows, nk, 128), jnp.bfloat16),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            name=f"muse_sptp_xe_dequant3d_t{rows}",
        )(g, xs_all)

    def call(xe8, xs_all):
        if tiled == 4:
            return call3d(xe8, xs_all)
        out = None
        for j in range(n):
            g = jax.lax.all_gather(
                xe8[j * sub : (j + 1) * sub], axis, axis=0, tiled=True
            )
            out = chunk(j, g, xs_all, out)
        return out

    return call
