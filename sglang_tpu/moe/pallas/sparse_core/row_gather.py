# SPDX-License-Identifier: Apache-2.0
"""SparseCore gather of whole 32-bit rows by pure DMA.

``row_gather(x, indices)[i] == x[indices[i]]`` for a 32-bit ``x [T, W]``. Each vector
subcore Indirect-gathers 16 rows per stream into its TileSPMEM and DMAs that buffer
straight to the output rows (``dsa_gather``'s nope pattern): no vector work at all, so
the gather runs at HBM rate. ``ragged_gather_v2`` instead gathers the 32-bit word that
packs a 16-bit row pair (or four 8-bit rows) and repacks rows on the vector unit, which
makes it compute-bound: on tpu7x a 65,536-row bf16 dispatch took ~1.08 ms that this
kernel moves as an fp8 container in ~0.28 ms.

Callers lay 16-bit / 8-bit data out so that one row is one run of 32-bit words (see
``gmm_v2.prequantize_lhs_fp8``).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def _row_gather_kernel(
    in_hbm,
    idx_hbm,
    out_hbm,
    sem,
    *,
    num_row_subchunks: int,
    num_streams: int,
    col_chunks: int,
    exact_rows: int | None,
):
    sc = pltpu.get_tpu_info().sparse_core
    lanes = sc.num_lanes
    num_cores = lax.axis_size(("core", "subcore"))
    core = lax.axis_index(("core", "subcore"))
    row_chunk = lanes * num_row_subchunks
    num_blocks = pl.cdiv(idx_hbm.shape[0], row_chunk * num_cores)
    cw = in_hbm.shape[1] // col_chunks

    def outer(idx_ref):
        b = pl.program_id(0)
        sub_base = (b * num_cores + core) * num_row_subchunks

        def body(*gather_bufs):
            r = pl.program_id(0)
            c = pl.program_id(1)
            first_row = (sub_base + r * num_streams) * lanes

            def copy_out():
                copies = []
                for s, buf in enumerate(gather_bufs):
                    cp = pltpu.make_async_copy(
                        buf,
                        out_hbm.at[
                            pl.ds(first_row + s * lanes, lanes), pl.ds(c * cw, cw)
                        ],
                        sem.at[s],
                    )
                    cp.start()
                    copies.append(cp)
                for cp in copies:
                    cp.wait()

            if exact_rows is None:
                copy_out()
            else:
                # A step's rows lie all below or all past exact_rows (the caller checks
                # exact_rows % (num_streams * lanes) == 0); past it there is no output.
                pl.when(first_row < exact_rows)(copy_out)

        # Several independent Indirect gathers per step keep reads in flight.
        in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(lanes), cw),
                lambda r, c, s=s: (
                    idx_ref[pl.ds((r * num_streams + s) * lanes, lanes)],
                    c,
                ),
            )
            for s in range(num_streams)
        )
        pltpu.emit_pipeline(
            body,
            grid=(num_row_subchunks // num_streams, col_chunks),
            in_specs=in_specs,
        )(*([in_hbm] * num_streams))

    pltpu.emit_pipeline(
        outer,
        grid=(num_blocks,),
        in_specs=(pl.BlockSpec((row_chunk,), lambda b: (b * num_cores + core,)),),
    )(idx_hbm)


def _col_chunks(width: int, max_words: int) -> int:
    """Fewest 128-aligned column chunks of at most max_words words."""
    for chunks in range(1, width // 128 + 1):
        if width % chunks == 0 and (width // chunks) % 128 == 0:
            if width // chunks <= max_words:
                return chunks
    return width // 128


@functools.partial(
    jax.jit, static_argnames=("num_row_subchunks", "num_streams", "max_chunk_words")
)
def row_gather(
    x: jax.Array,
    indices: jax.Array,
    num_row_subchunks: int = 8,
    num_streams: int = 2,
    max_chunk_words: int = 768,
) -> jax.Array:
    """``x[indices]`` for 32-bit ``x [T, W]`` (W a multiple of 128), on the SparseCore.

    ``max_chunk_words`` bounds one stream buffer to 16 rows x that many words (two
    buffers per stream). Rows are DMA'd as is; ``indices`` must lie in ``[0, T)``.
    """
    assert x.ndim == 2 and indices.ndim == 1, (x.shape, indices.shape)
    assert jax.dtypes.itemsize_bits(x.dtype) == 32, x.dtype
    width = x.shape[1]
    assert width % 128 == 0, width
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        return x[indices]
    lanes = sc.num_lanes
    num_cores = sc.num_cores * sc.num_subcores
    n = indices.shape[0]
    # Small gathers: fewer subchunks per block so the padding stays small.
    while num_row_subchunks > num_streams and n <= lanes * num_cores * (
        num_row_subchunks // 2
    ):
        num_row_subchunks //= 2
    step_rows = lanes * num_streams
    if n % step_rows:
        num_streams = 1
        step_rows = lanes
    assert num_row_subchunks % num_streams == 0
    block = lanes * num_row_subchunks * num_cores
    pad = -n % block
    if pad:
        indices = jnp.pad(indices, (0, pad))
    exact = n % step_rows == 0
    out_rows = n if exact else n + pad
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
    )
    out = pl.kernel(
        functools.partial(
            _row_gather_kernel,
            num_row_subchunks=num_row_subchunks,
            num_streams=num_streams,
            col_chunks=_col_chunks(width, max_chunk_words),
            exact_rows=n if (exact and pad) else None,
        ),
        out_type=jax.ShapeDtypeStruct((out_rows, width), x.dtype),
        mesh=mesh,
        scratch_types=[pltpu.SemaphoreType.DMA((num_streams,))],
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name="sc_row_gather",
    )(x, indices.astype(jnp.int32))
    return out if exact else out[:n]
