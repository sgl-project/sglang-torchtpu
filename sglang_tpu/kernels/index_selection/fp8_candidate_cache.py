# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare and gather packed FP8 candidate caches for cross-layer reuse."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from .packed_fp8_cache import PAGE_RECORDS, PAGE_ROWS, PackedFP8Layout
from .tuned_block_sizes import PAGED_MQA_GATHER_CONFIG


def canonical_candidate_blocks(
    candidates: jax.Array,
    table: jax.Array,
    offsets: jax.Array,
    *,
    block: int,
    width: int,
    cache_pages: int,
    page_size: int = PAGE_RECORDS,
) -> tuple[jax.Array, jax.Array]:
    """Sort only the candidate rows, never a full-context histogram.

    This single metadata algorithm uses O(requests * candidates) intermediates;
    the page table stays in HBM and only selected page IDs are gathered. There
    is no context-sized SparseCore VMEM allocation or backend-dependent route.
    Empty requests produce fully initialized -1 rows.
    """
    blocks = pl.cdiv(width, block)
    physical = resolve_candidate_blocks(
        candidates,
        table,
        sparse_block_kv=block,
        cache_pages=cache_pages,
        page_size=page_size,
    )
    parts = max(1, block // page_size)
    needed = (
        candidates[..., None] * block + jnp.arange(parts) * min(block, page_size)
        < width
    )
    valid = jnp.all(
        (physical.reshape(*candidates.shape, parts) >= 0) | ~needed, axis=-1
    )
    valid &= (candidates >= 0) & (candidates < blocks)
    valid &= (offsets[1:] > offsets[:-1])[:, None]
    canonical = jnp.sort(jnp.where(valid, candidates, blocks), axis=-1)
    unique = jnp.diff(canonical, axis=-1, prepend=-1) != 0
    canonical = jnp.sort(jnp.where(unique, canonical, blocks), axis=-1)
    canonical = jnp.where(canonical < blocks, canonical, -1)
    return canonical, resolve_candidate_blocks(
        canonical,
        table,
        sparse_block_kv=block,
        cache_pages=cache_pages,
        page_size=page_size,
    )


@jax.jit(static_argnames=("sparse_block_kv", "cache_pages", "page_size"))
def resolve_candidate_blocks(
    sparse_blocks: jax.Array,
    block_table: jax.Array,
    *,
    sparse_block_kv: int,
    cache_pages: int,
    page_size: int = PAGE_RECORDS,
) -> jax.Array:
    """Resolve selected block IDs without copying the full table into VMEM.

    Blocks spanning pages expand into page-sized fragments in candidate order.
    Missing/out-of-range pages produce -1, never a clamped physical page ID.
    """
    if min(sparse_block_kv, page_size) < 1 or (
        max(sparse_block_kv, page_size) % min(sparse_block_kv, page_size)
    ):
        raise ValueError("Candidate blocks and pages must divide one another")
    if sparse_block_kv > page_size:
        parts = sparse_block_kv // page_size
        sparse_blocks = jnp.where(
            sparse_blocks[..., None] >= 0,
            sparse_blocks[..., None] * parts + jnp.arange(parts),
            -1,
        ).reshape(sparse_blocks.shape[0], sparse_blocks.shape[1] * parts)
        sparse_block_kv = page_size
    if block_table.shape[1] == 0:
        return jnp.full(sparse_blocks.shape, -1, jnp.int32)
    per_page = page_size // sparse_block_kv
    limit = block_table.shape[1] * per_page
    safe = jnp.clip(sparse_blocks, 0, limit - 1)
    pages = jnp.take_along_axis(block_table, safe // per_page, axis=1)
    valid = (
        (sparse_blocks >= 0)
        & (sparse_blocks < limit)
        & (pages >= 0)
        & (pages < cache_pages)
    )
    return jnp.where(valid, pages * per_page + safe % per_page, -1)


def gather_candidate_keys(
    banks: tuple[jax.Array, ...],
    physical: jax.Array,
    offsets: jax.Array,
    *,
    block: int,
    layout: PackedFP8Layout,
) -> jax.Array:
    """Gather tightly packed pages, initializing invalid and empty rows to zero.

    Each request's fragment count must fill whole 128-key output pages. The
    ring writes every output tile, including empty requests; ignored source
    DMAs leave zeros, not stale bytes from an earlier ring-buffer iteration.
    """
    layout.validate(tuple(bank.shape[0] for bank in banks))
    if block < 1 or PAGE_RECORDS % block or layout.page_size % block:
        raise ValueError("Gather fragments must divide source and compact pages")
    per_page = PAGE_RECORDS // block
    if physical.ndim != 2 or physical.shape[1] % per_page:
        raise ValueError("Each request must fill whole compact FP8 pages")
    if physical.size == 0:
        return jnp.zeros((0, PAGE_RECORDS), jnp.int32)
    physical = jnp.where(
        (offsets[1:] > offsets[:-1])[:, None]
        & (physical >= 0)
        & (physical < layout.pages * (layout.page_size // block)),
        physical,
        -1,
    )
    info = plsc.get_sparse_core_info()
    lanes, units = info.num_lanes, info.num_cores * info.num_subcores
    group = max(lanes, min(128, 1024 // block))
    buffers = min(
        2,
        info.vmem_capacity_bytes
        // (group * PAGE_ROWS * (block + len(banks)) * 4 + 4096),
    )
    group, buffers = PAGED_MQA_GATHER_CONFIG.get((block, len(banks)), (group, buffers))
    if buffers < 1 or group % per_page:
        raise ValueError("Gather tile must hold whole pages and at least one buffer")
    distance = buffers - 1
    count, per_page = physical.size, PAGE_RECORDS // block
    source_per_page = layout.page_size // block
    per_unit = pl.cdiv(count, units * group) * group
    scratch_words = buffers * group * PAGE_ROWS * (block + len(banks)) + lanes
    metadata_rows = min(
        per_unit,
        (info.vmem_capacity_bytes // 4 - scratch_words - 4096) // group * group,
    )
    if metadata_rows < group:
        raise ValueError("Packed FP8 gather scratch exceeds SparseCore VMEM")
    metadata_steps = metadata_rows // group
    ids = jnp.pad(
        physical.reshape(-1), (0, units * per_unit - count), constant_values=-1
    )
    ids = ids.reshape(-1, units, group).transpose(1, 0, 2).reshape(units, per_unit)
    ids = jnp.pad(ids, ((0, 0), (0, -per_unit % metadata_rows)), constant_values=-1)
    # All output tiles must be written, even when the last requests are empty.
    control = jnp.full((lanes,), per_unit // group, jnp.int32)
    span, steps = group * PAGE_ROWS, per_unit // group
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(
            (units * per_unit // per_page * PAGE_ROWS, PAGE_RECORDS), jnp.int32
        ),
        scratch_types=(
            pltpu.VMEM((metadata_rows,), jnp.int32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.VMEM((buffers, span, block), jnp.int32),
            pltpu.VMEM((buffers, len(banks), span), jnp.int32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((buffers, len(banks))),
            pltpu.SemaphoreType.DMA((buffers,)),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False, needs_layout_passes=False
        ),
        mesh=mesh,
        name="paged_mqa_candidate_gather",
    )
    def gather(
        pool,
        physical_ref,
        control_ref,
        out,
        selected,
        control_vmem,
        words,
        read_ids,
        metadata_sem,
        read_sem,
        write_sem,
    ):
        unit = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        pltpu.async_copy(control_ref, control_vmem, metadata_sem).wait()
        active = control_vmem[...][0]
        metadata = physical_ref.at[unit]
        if metadata_rows < per_unit:
            metadata = metadata.at[:metadata_rows]
        pltpu.async_copy(metadata, selected, metadata_sem).wait()

        def read(step, slot, wait=False):
            if not wait:
                index = step
                if metadata_rows < per_unit:
                    index = step % metadata_steps

                    @pl.when((step > 0) & (index == 0))
                    def next_metadata():
                        pltpu.async_copy(
                            physical_ref.at[unit, pl.ds(step * group, metadata_rows)],
                            selected,
                            metadata_sem,
                        ).wait()

                # SparseCore stores are lane vectors. Scatter explicit 2-D
                # indices rather than reshaping VMEM or issuing a 2-D Swap.
                @plsc.parallel_loop(0, span * block, lanes, unroll=8)
                def clear(i):
                    index = i + jnp.arange(lanes)
                    plsc.store_scatter(
                        words.at[slot],
                        [index // block, index % block],
                        jnp.zeros((lanes,), jnp.int32),
                    )

                @plsc.parallel_loop(0, span, lanes, unroll=8)
                def addresses(i):
                    segment = i + jnp.arange(lanes)
                    field = segment // per_page % PAGE_ROWS
                    column = (
                        segment // (PAGE_ROWS * per_page) * per_page
                        + segment % per_page
                    )
                    chosen = plsc.load_gather(selected, [index * group + column])
                    safe = jnp.maximum(chosen, 0)
                    source = (
                        (
                            layout.page_offset
                            + safe // source_per_page * layout.page_stride
                        )
                        * per_page
                        + field * source_per_page
                        + safe % source_per_page
                    )
                    bank_base = 0
                    for bank, ref in enumerate(pool):
                        size = ref.shape[0] * per_page
                        valid = (
                            (chosen >= 0)
                            & (source >= bank_base)
                            & (source < bank_base + size)
                        )
                        read_ids[slot, bank, pl.ds(i, lanes)] = jnp.where(
                            valid, source - bank_base, -1
                        )
                        bank_base += size

            for bank, ref in enumerate(pool):
                copy = pltpu.make_async_copy(
                    ref.reshape(-1, block).at[
                        plsc.Indices(read_ids.at[slot, bank], ignored_value=-1)
                    ],
                    words.at[slot],
                    read_sem.at[slot, bank],
                )
                (copy.wait if wait else copy.start)()

        def write(step, wait=False):
            slot = step % buffers
            start = (step * units + unit) * group
            copy = pltpu.make_async_copy(
                words.at[slot],
                out.reshape(-1, block).at[pl.ds(start * PAGE_ROWS, span)],
                write_sem.at[slot],
            )
            (copy.wait if wait else copy.start)()

        for step in range(min(distance, steps)):
            pl.when(step < active)(
                lambda step=step: read(jnp.int32(step), jnp.int32(step))
            )

        @pl.loop(0, active)
        def body(step):
            slot = step % buffers
            if buffers == 1:
                read(step, slot)
            read(step, slot, wait=True)
            write(step)
            if buffers == 1:
                write(step, wait=True)

            @pl.when((distance > 0) & (step + distance < active))
            def prefetch():
                future = (step + distance) % buffers
                pl.when(step + distance >= buffers)(
                    lambda: write(step + distance - buffers, wait=True)
                )
                read(step + distance, future)

        for last in range(buffers if buffers > 1 else 0):
            pl.when(last < active)(
                lambda last=last: write(active - last - 1, wait=True)
            )

    return gather(banks, ids, control)[: count // per_page * PAGE_ROWS]
