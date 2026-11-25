# SPDX-License-Identifier: Apache-2.0
"""SparseCore gather of whole 32-bit rows by pure DMA.

``row_gather(x, indices)[i] == x[indices[i]]`` for a 32-bit ``x [T, W]``. Each vector
subcore Indirect-gathers 16 rows per stream into its TileSPMEM and DMAs that buffer
straight to the output rows (``dsa_gather``'s nope pattern): no vector work at all, so
the gather runs at HBM rate. ``ragged_gather_v2`` instead gathers 32-bit words packing
narrower rows and repacks them on the vector unit, making large gathers compute-bound.

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
    in_hbm: jax.Ref,
    idx_hbm: jax.Ref,
    out_hbm: jax.Ref,
    sem: jax.Ref,
    *,
    end_hbm: jax.Ref | None = None,
    end_vmem: jax.Ref | None = None,
    num_row_subchunks: int,
    num_streams: int,
    col_chunks: int,
    exact_rows: int | None,
):
    """Pure-DMA SparseCore row-gather kernel body.

    Args:
        in_hbm: Input 32-bit table in HBM, shape ``[T, W]``.
        idx_hbm: Padded int32 row indices in HBM.
        out_hbm: Output buffer in HBM, shape ``[out_rows, W]``.
        sem: DMA semaphores, one per concurrent gather stream.
        end_hbm: Optional dynamic row-count bound ``i32[1]`` in HBM.
        end_vmem: Scratch VMEM buffer for loading ``end_hbm`` (required with it).
        num_row_subchunks: Number of ``sc.num_lanes``-row subchunks per outer block.
        num_streams: Number of concurrent indirect DMA streams per inner step.
        col_chunks: Number of 128-aligned column chunks across ``W``.
        exact_rows: True output row count when ``out_hbm`` is unpadded on the tail.
    """
    sc = pltpu.get_tpu_info().sparse_core
    lanes = sc.num_lanes
    num_cores = lax.axis_size(("core", "subcore"))
    core = lax.axis_index(("core", "subcore"))
    row_chunk = lanes * num_row_subchunks
    block_size = row_chunk * num_cores
    if end_hbm is not None:
        cp = pltpu.make_async_copy(end_hbm, end_vmem.at[:1], sem.at[0])
        cp.start()
        cp.wait()
        end = jnp.minimum(end_vmem[...][0], exact_rows or idx_hbm.shape[0])
        num_blocks = jnp.where(end <= 0, 0, pl.cdiv(end, block_size))
    else:
        num_blocks = pl.cdiv(idx_hbm.shape[0], block_size)
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


def _row_gather_bounded_kernel(
    in_hbm: jax.Ref,
    idx_hbm: jax.Ref,
    end_hbm: jax.Ref,
    out_hbm: jax.Ref,
    end_vmem: jax.Ref,
    sem: jax.Ref,
    **kwargs,
):
    """Adapter for ``pl.kernel`` when ``gather_end`` is an operand + scratch."""
    _row_gather_kernel(
        in_hbm, idx_hbm, out_hbm, sem, end_hbm=end_hbm, end_vmem=end_vmem, **kwargs
    )


def _col_chunks(width: int, max_words: int) -> int:
    """Fewest 128-aligned column chunks of at most max_words words."""
    for chunks in range(1, width // 128 + 1):
        if width % chunks == 0 and (width // chunks) % 128 == 0:
            if width // chunks <= max_words:
                return chunks
    return width // 128


# Max 16-row (sc.num_lanes) subchunks per outer block per subcore; halves on small
# gathers to keep padding small.
_MAX_ROW_SUBCHUNKS = 8
_STREAMS = 2
# Fits 2 streams x 2 double-buffers x 16 lanes x 768 u32 = 192 KiB (75% of the
# 256 KiB SparseCore VMEM floor, 128-word aligned).
_MAX_CHUNK_WORDS = 768


@jax.jit
def row_gather(
    x: jax.Array,
    indices: jax.Array,
    gather_end: jax.Array | None = None,
) -> jax.Array:
    """``x[indices]`` for 32-bit ``x [T, W]`` (W a multiple of 128), on the SparseCore.

    When ``gather_end`` (dynamic ``i32[1]`` or scalar) is given, only blocks covering
    ``[0, gather_end)`` are gathered (rows past ``gather_end`` are left uninitialized).
    Rows are DMA'd as is; ``indices`` must lie in ``[0, T)``.
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
    num_row_subchunks = _MAX_ROW_SUBCHUNKS
    num_streams = _STREAMS
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
    has_end = gather_end is not None
    kernel_args = [x, indices.astype(jnp.int32)]
    scratch_types = [pltpu.SemaphoreType.DMA((num_streams,))]
    if has_end:
        end_arr = jnp.asarray(gather_end, dtype=jnp.int32).reshape(1)
        kernel_args.append(end_arr)
        scratch_types.insert(0, pltpu.VMEM((16,), jnp.int32))
    body = _row_gather_bounded_kernel if has_end else _row_gather_kernel
    out = pl.kernel(
        functools.partial(
            body,
            num_row_subchunks=num_row_subchunks,
            num_streams=num_streams,
            col_chunks=_col_chunks(width, _MAX_CHUNK_WORDS),
            exact_rows=n if (exact and pad) else None,
        ),
        out_type=jax.ShapeDtypeStruct((out_rows, width), x.dtype),
        mesh=mesh,
        scratch_types=scratch_types,
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name="sc_row_gather",
    )(*kernel_args)
    return out if exact else out[:n]
