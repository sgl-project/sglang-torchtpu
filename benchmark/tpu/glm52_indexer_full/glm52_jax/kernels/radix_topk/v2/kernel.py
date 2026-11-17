"""RadixTopKV2 with one SparseCore subcore per score row.

Default workload: N=128K, K=2K, B=16K, B2=32K. Scan B elements at a time,
filter against the last exact Kth key, and compact only when the candidate
count reaches K+B2 or the scan ends. Each subcore owns independent rows;
there are no cross-subcore histograms, barriers, or candidate merges.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def _ordered_bits(keys: jax.Array) -> jax.Array:
    """Match v1's FP32 radix ordering, including signed zero."""
    bits = keys.view(jnp.uint32)
    sign = (bits.view(jnp.int32) >> 31).view(jnp.uint32)
    return bits ^ (sign | jnp.uint32(1 << 31))


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "block_size",
        "reconstruct_threshold",
        "reverse_scan",
        "indices_only",
        "debug",
        "page_size",
    ),
)
@jax.named_scope("radix_topk_v2")
def radix_topk_v2(
    scores: jax.Array,
    *,
    valid_lengths: jax.Array | None = None,
    page_table: jax.Array | None = None,
    page_size: int = 64,
    k: int = 2048,
    block_size: int = 16384,
    reconstruct_threshold: int = 32768,
    reverse_scan: bool = False,
    indices_only: bool = False,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array] | jax.Array:
    """Select exact, unordered TopK using one subcore for the entire row.

    Args:
      scores: FP32 matrix (batch, N); a partial final block is filled with -inf in scratch.
      valid_lengths: Optional per-row valid prefix lengths. Unfilled outputs
        are (-inf, -1) when fewer than K positions are valid.
      k: K, default 2K. Must be a multiple of the subcore's lane count.
      block_size: B, default 16K, in elements.
      reconstruct_threshold: B2, default 32K, in elements. Compact after a
        block when candidate_count >= K+B2, not after every block.
      reverse_scan: Visit blocks from the end of the padded row. Within each
        block DMA and lane order stay forward; returned indices remain original.
      indices_only: Return only int32 positions; selected scores are still
        maintained internally to update the threshold.
      debug: Enable Pallas compiler debug output.

    Candidate capacity is K+B2+B (50K elements with the defaults). Before
    the first compaction, every real input is accepted, including -inf; padding is always excluded. Afterwards
    only keys strictly above the retained Kth key are appended. Ties may differ
    from v1; values follow its radix bit order. Windows are sequential, with
    no random-order performance guarantee. Outputs have shape (batch, K).
    """
    if scores.ndim != 2 or scores.dtype != jnp.float32:
        raise ValueError("scores must be a rank-2 float32 array")
    batch_size, width = scores.shape
    if not 1 <= k <= width:
        raise ValueError("k must be in [1, N]")
    if block_size <= 0 or reconstruct_threshold <= 0:
        raise ValueError("block_size and reconstruct_threshold must be positive")
    if valid_lengths is not None and (
        valid_lengths.shape != (batch_size,) or valid_lengths.dtype != jnp.int32
    ):
        raise ValueError("valid_lengths must be an int32 vector with one entry per row")
    valid_width = width
    # HBM rows require 128-element alignment, but need not be B-aligned.
    # Padding to B here copies the entire score matrix on every pipeline step.
    # Preserve the existing path for unusual sub-128/non-tile-aligned blocks.
    input_alignment = 128 if block_size % 128 == 0 else block_size
    padding = (-width) % input_alignment
    if padding:
        scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=-jnp.inf)
        width += padding

    if page_table is not None:
        if page_table.ndim != 1 or page_table.dtype != jnp.int32 or not page_table.shape[0]:
            raise ValueError("page_table must be a nonempty int32 vector shared by these rows")
        if page_size < 1:
            raise ValueError("page_size must be positive")
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    if any(n % lanes for n in (k, block_size, reconstruct_threshold)):
        raise ValueError("k, block_size and reconstruct_threshold must be lane-aligned")
    lengths = (
        jnp.full((batch_size,), valid_width, jnp.int32)
        if valid_lengths is None
        else jnp.clip(valid_lengths, 0, valid_width)
    )
    lengths = jnp.broadcast_to(lengths[:, None], (batch_size, lanes))
    # HBM output rows need a full 128-element tile, including for k=16.
    output_width = pl.cdiv(k, 128) * 128
    capacity = max(k + reconstruct_threshold + block_size, output_width)
    histogram_size = 256  # Four exact 8-bit radix passes for FP32.
    # Keys + indices for candidates and selected K, plus one input DMA buffer.
    scratch_bytes = 8 * (capacity + k) + 4 * (block_size + histogram_size + lanes)
    mapping_scratch = ()
    if page_table is not None:
        page_count = page_table.shape[0]
        padded_pages = pl.cdiv(page_count, lanes) * lanes
        page_table = jnp.pad(page_table, (0, padded_pages - page_count), constant_values=-1)
        mapping_scratch = (pltpu.VMEM((padded_pages,), jnp.int32),)
        scratch_bytes += 4 * padded_pages
    if scratch_bytes > info.vmem_capacity_bytes:
        raise ValueError(
            f"RadixTopKV2 scratch needs {scratch_bytes} bytes per subcore, "
            f"but only {info.vmem_capacity_bytes} bytes are available"
        )
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, pl.cdiv(batch_size, info.num_subcores)) if batch_size else 1,
    )
    out_indices = jax.ShapeDtypeStruct((batch_size, output_width), jnp.int32)
    out_type = (
        out_indices
        if indices_only
        else (jax.ShapeDtypeStruct((batch_size, output_width), scores.dtype), out_indices)
    )

    @pl.kernel(
        out_type=out_type,
        mesh=mesh,
        scratch_types=mapping_scratch
        + (
            pltpu.VMEM((block_size,), jnp.float32),
            pltpu.VMEM((capacity,), jnp.float32),
            pltpu.VMEM((capacity,), jnp.int32),
            pltpu.VMEM((k,), jnp.float32),
            pltpu.VMEM((k,), jnp.int32),
            pltpu.VMEM((histogram_size,), jnp.int32),
            pltpu.VMEM((lanes,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="radix_topk_v2_mapped" if page_table is not None else "radix_topk_v2",
        debug=debug,
    )
    def kernel(scores_ref, lengths_ref, *outputs_and_scratch):
        if page_table is not None:
            page_table_ref, *outputs_and_scratch = outputs_and_scratch
        if indices_only:
            out_scores_ref = None
            out_indices_ref, *scratch = outputs_and_scratch
        else:
            out_scores_ref, out_indices_ref, *scratch = outputs_and_scratch
        if page_table is not None:
            page_buffer, *scratch = scratch
        block, candidates, candidate_ids, selected, selected_ids, histogram, row_length, sem = (
            scratch
        )
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)

        def compact(num_candidates):
            """In-place exact radix selection; retain K pairs in candidates."""

            @plsc.parallel_loop(0, k, lanes)
            def clear_selected(i):
                selected[pl.ds(i, lanes)] = jnp.full(lanes, -jnp.inf, jnp.float32)
                selected_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            # State: current digit, active candidates, already accepted outputs.
            def cond(state):
                digit, remaining, found = state
                # If all remaining candidates are needed, no more digits are
                # necessary. After the final digit, survivors are tied.
                return (digit >= 0) & (remaining > k - found)

            def radix_pass(state):
                digit, remaining, found = state
                limit = pl.cdiv(remaining, lanes) * lanes

                @plsc.parallel_loop(0, histogram_size, lanes)
                def clear_histogram(i):
                    histogram[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

                def digits_for(keys):
                    return (_ordered_bits(keys) >> (digit * 8).astype(jnp.uint32)) & jnp.uint32(255)

                @plsc.parallel_loop(0, limit, lanes)
                def fill_histogram(i):
                    digits = digits_for(candidates[pl.ds(i, lanes)])
                    valid = i + lane_ids < remaining
                    counts, mask = plsc.scan_count(digits, mask=valid)
                    plsc.addupdate_scatter(histogram, [digits], counts, mask=mask)

                # Locate the bucket containing the (remaining - needed + 1)th
                # smallest key. All higher buckets are guaranteed winners.
                below = remaining - (k - found)
                init = (jnp.zeros(lanes, jnp.int32), jnp.zeros(lanes, jnp.int32))

                @plsc.parallel_loop(0, histogram_size, lanes, carry=init)
                def find_pivot(i, carry):
                    pivot, prefix = carry
                    cumulative = jnp.cumsum(histogram[pl.ds(i, lanes)])
                    offset = plsc.all_reduce_ffs(prefix + cumulative > below)
                    return pivot + offset, prefix + cumulative[-1]

                pivot, _ = find_pivot

                @plsc.parallel_loop(0, limit, lanes, carry=(jnp.int32(0), found))
                def filter_candidates(i, carry):
                    survivors, outputs = carry
                    keys = candidates[pl.ds(i, lanes)]
                    ids = candidate_ids[pl.ds(i, lanes)]
                    digits = digits_for(keys)
                    valid = i + lane_ids < remaining
                    accept = valid & (digits > pivot)
                    survive = valid & (digits == pivot)
                    plsc.store_compressed(selected.at[pl.ds(outputs, lanes)], keys, mask=accept)
                    plsc.store_compressed(selected_ids.at[pl.ds(outputs, lanes)], ids, mask=accept)
                    plsc.store_compressed(
                        candidates.at[pl.ds(survivors, lanes)], keys, mask=survive
                    )
                    plsc.store_compressed(
                        candidate_ids.at[pl.ds(survivors, lanes)], ids, mask=survive
                    )
                    return (
                        survivors + plsc.all_reduce_population_count(survive)[0],
                        outputs + plsc.all_reduce_population_count(accept)[0],
                    )

                survivors, outputs = filter_candidates
                return digit - 1, survivors, outputs

            _, remaining, found = jax.lax.while_loop(
                cond, radix_pass, (jnp.int32(3), num_candidates, jnp.int32(0))
            )
            needed = jnp.minimum(remaining, k - found)

            @plsc.parallel_loop(0, pl.cdiv(needed, lanes) * lanes, lanes)
            def postfill(i):
                mask = i + lane_ids < needed
                plsc.store_compressed(
                    selected.at[pl.ds(found + i, lanes)],
                    candidates[pl.ds(i, lanes)],
                    mask=mask,
                )
                plsc.store_compressed(
                    selected_ids.at[pl.ds(found + i, lanes)],
                    candidate_ids[pl.ds(i, lanes)],
                    mask=mask,
                )

            @plsc.parallel_loop(0, k, lanes, carry=jnp.full(lanes, jnp.uint32(0xFFFFFFFF)))
            def retain_topk(i, minimum):
                keys = selected[pl.ds(i, lanes)]
                candidates[pl.ds(i, lanes)] = keys
                candidate_ids[pl.ds(i, lanes)] = selected_ids[pl.ds(i, lanes)]
                return jnp.minimum(minimum, _ordered_bits(keys))

            return jnp.min(retain_topk)

        subcore = jax.lax.axis_index("subcore")
        core = jax.lax.axis_index("core")
        first_row = core * mesh.num_subcores + subcore
        row_stride = mesh.num_cores * mesh.num_subcores

        @pl.loop(first_row, batch_size, step=row_stride)
        def process_row(row):
            pltpu.async_copy(lengths_ref.at[row], row_length, sem).wait()
            length = jnp.min(row_length[:])
            num_blocks = pl.cdiv(length, block_size)

            @plsc.parallel_loop(0, output_width, lanes)
            def clear_outputs(i):
                candidates[pl.ds(i, lanes)] = jnp.full(lanes, -jnp.inf, jnp.float32)
                candidate_ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            # Carry: candidate count, last Kth ordered key, threshold initialized.
            def scan_block(block_id, state):
                count, threshold, have_threshold = state
                # block_id counts visits, so final reconstruction is independent
                # of physical scan direction. Never reverse or copy the score row.
                source_block = num_blocks - 1 - block_id if reverse_scan else block_id
                offset = source_block * block_size

                def load_full_block():
                    copy = pltpu.async_copy(
                        scores_ref.at[row, pl.ds(offset, block_size)], block, sem
                    )
                    copy.wait()

                tail_size = width % block_size
                if tail_size:

                    def load_tail_block():
                        # Only the final physical block can be short. Keep
                        # DMA sizes static and mask its scratch padding via
                        # the original valid length when appending candidates.
                        @plsc.parallel_loop(0, block_size, lanes)
                        def clear_input(i):
                            block[pl.ds(i, lanes)] = jnp.full(lanes, -jnp.inf, jnp.float32)

                        pltpu.async_copy(
                            scores_ref.at[row, pl.ds(width - tail_size, tail_size)],
                            block.at[pl.ds(0, tail_size)],
                            sem,
                        ).wait()

                    if width < block_size:
                        load_tail_block()
                    else:
                        jax.lax.cond(
                            offset == width - tail_size,
                            load_tail_block,
                            load_full_block,
                        )
                else:
                    load_full_block()

                @plsc.parallel_loop(0, block_size, lanes, carry=count)
                def append_candidates(i, count):
                    keys = block[pl.ds(i, lanes)]
                    ids = offset + i + lane_ids
                    # Padded -inf must not displace real -inf, especially when
                    # reverse scanning starts with fewer than K valid elements.
                    accept = (ids < length) & (~have_threshold | (_ordered_bits(keys) > threshold))
                    plsc.store_compressed(candidates.at[pl.ds(count, lanes)], keys, mask=accept)
                    plsc.store_compressed(candidate_ids.at[pl.ds(count, lanes)], ids, mask=accept)
                    return count + plsc.all_reduce_population_count(accept)[0]

                count = append_candidates
                should_compact = (count >= k + reconstruct_threshold) | (block_id == num_blocks - 1)

                def reconstruct():
                    return jnp.int32(k), compact(count), jnp.bool_(True)

                return jax.lax.cond(
                    should_compact, reconstruct, lambda: (count, threshold, have_threshold)
                )

            jax.lax.fori_loop(
                0,
                num_blocks,
                scan_block,
                (jnp.int32(0), jnp.uint32(0), jnp.bool_(False)),
            )
            if page_table is not None:
                pltpu.async_copy(page_table_ref, page_buffer, sem).wait()

                @plsc.parallel_loop(0, output_width, lanes)
                def map_ids(i):
                    ids = candidate_ids[pl.ds(i, lanes)]
                    page = ids // page_size
                    valid = (ids >= 0) & (ids < length) & (page < page_count)
                    physical = plsc.load_gather(
                        page_buffer, [jnp.clip(page, 0, page_count - 1)], mask=valid
                    )
                    candidate_ids[pl.ds(i, lanes)] = jnp.where(
                        valid & (physical >= 0), physical * page_size + ids % page_size, -1
                    )

            if not indices_only:
                pltpu.async_copy(candidates.at[:output_width], out_scores_ref.at[row], sem).wait()
            pltpu.async_copy(candidate_ids.at[:output_width], out_indices_ref.at[row], sem).wait()

    result = (
        kernel(scores, lengths, page_table) if page_table is not None else kernel(scores, lengths)
    )
    if output_width != k:
        result = jax.tree.map(lambda x: x[:, :k], result)
    return result


__all__ = ["radix_topk_v2"]
