"""Independent logical-to-physical mapping for all DSA score paths."""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


@functools.partial(jax.jit, static_argnames=("page_size",))
def map_sparsecore(indices, page_table, row_sequences, valid_lengths, *, page_size=64):
    m, k = indices.shape
    if indices.dtype != jnp.int32 or page_table.dtype != jnp.int32:
        raise TypeError("int32 indices and page table required")
    if page_table.ndim != 2 or row_sequences.shape != (m,) or valid_lengths.shape != (m,):
        raise ValueError("page table [sequences,pages], row metadata [rows] required")
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    if k % lanes:
        raise ValueError("K must be lane aligned")
    nseq, pages = page_table.shape
    pad = (-pages) % lanes
    table = jnp.pad(page_table, ((0, 0), (0, pad)))
    meta = jnp.broadcast_to(
        jnp.stack((row_sequences, valid_lengths), axis=1)[:, :, None], (m, 2, lanes)
    ).astype(jnp.int32)
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, pl.cdiv(m, info.num_subcores)),
    )

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((m, k), jnp.int32),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((pages + pad,), jnp.int32),
            pltpu.VMEM((k,), jnp.int32),
            pltpu.VMEM((k,), jnp.int32),
            pltpu.VMEM((2, lanes), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="dsa_physical_mapping_sparsecore",
    )
    def kernel(ids_ref, table_ref, meta_ref, out_ref, tbuf, ibuf, obuf, mbuf, sem):
        first = jax.lax.axis_index("core") * mesh.num_subcores + jax.lax.axis_index("subcore")

        @pl.loop(first, m, step=mesh.num_cores * mesh.num_subcores)
        def row(r):
            pltpu.async_copy(meta_ref.at[r], mbuf, sem).wait()
            seq = jnp.min(mbuf[0])
            length = jnp.min(mbuf[1])
            pltpu.async_copy(table_ref.at[jnp.clip(seq, 0, nseq - 1)], tbuf, sem).wait()
            pltpu.async_copy(ids_ref.at[r], ibuf, sem).wait()

            @plsc.parallel_loop(0, k, lanes)
            def map_chunk(i):
                ids = ibuf[pl.ds(i, lanes)]
                page = ids // page_size
                valid = (ids >= 0) & (ids < length) & (page < pages) & (seq >= 0) & (seq < nseq)
                physical = plsc.load_gather(tbuf, [jnp.clip(page, 0, pages - 1)], mask=valid)
                valid = valid & (physical >= 0)
                obuf[pl.ds(i, lanes)] = jnp.where(valid, physical * page_size + ids % page_size, -1)

            pltpu.async_copy(obuf, out_ref.at[r], sem).wait()

    return kernel(indices, table, meta)


def map_logical_topk(
    indices, page_table, row_sequences, valid_lengths, *, page_size=64, interpret=False
):
    """Resolve selected positions once, using exact arbitrary page-table gathers."""
    if indices.ndim != 2 or page_table.ndim != 2:
        raise ValueError("indices and page table must be rank 2")
    rows, k = indices.shape
    if row_sequences.shape != (rows,) or valid_lengths.shape != (rows,):
        raise ValueError("row metadata must have one entry per top-k row")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if not rows or not k or not all(page_table.shape):
        return jnp.full(indices.shape, -1, jnp.int32)
    if not interpret and jax.default_backend() == "tpu" and k % 128 == 0:
        return map_sparsecore(
            indices, page_table, row_sequences, valid_lengths, page_size=page_size
        )
    seq = jnp.clip(row_sequences, 0, page_table.shape[0] - 1)
    logical = jnp.maximum(indices, 0)
    page = logical // page_size
    physical = page_table[seq[:, None], jnp.clip(page, 0, page_table.shape[1] - 1)]
    valid = (
        (indices >= 0)
        & (logical < valid_lengths[:, None])
        & (page < page_table.shape[1])
        & (physical >= 0)
        & (row_sequences[:, None] >= 0)
        & (row_sequences[:, None] < page_table.shape[0])
    )
    return jnp.where(valid, physical * page_size + logical % page_size, -1).astype(jnp.int32)
