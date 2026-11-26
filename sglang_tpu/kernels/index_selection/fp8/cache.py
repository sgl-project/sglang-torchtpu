# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write packed FP8 index keys with an in-page FP32 scale per key."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.kernels.index_selection.fp8.pages import (
    TOKENS_PER_ROW,
    WORD_BYTES,
    page_rows,
)

# Stage up to 128 touched pages, stopping at the sorted invalid-record tail.
PACKED_FP8_WRITE_BATCH = 128


def quantize_index_rows(values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return E4M3 values, FP32 scales and E8M0 scale bytes per vector."""
    # One power-of-two scale covers each post-RoPE vector, independent of layout.
    values = values.astype(jnp.float32)
    amax = jnp.maximum(jnp.abs(values).max(axis=-1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    scales = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    quantized = (values / scales[..., None]).astype(jnp.float8_e4m3fn)
    return quantized, scales, scale_bytes


def write_index_keys(
    cache: jax.Array, records: jax.Array, keys: jax.Array
) -> jax.Array:
    """Quantize only new keys and update their slots; valid slots are unique."""
    _, fields, _, page_size = cache.shape
    keys = keys.reshape(records.size, (fields - 1) * WORD_BYTES)
    quantized, scales, _ = quantize_index_rows(keys)
    bits = jax.lax.bitcast_convert_type(quantized, jnp.uint8).astype(jnp.uint32)
    bits = bits.reshape(records.size, fields - 1, WORD_BYTES)
    words = bits[:, :, 0] | (bits[:, :, 1] << 8)
    words |= (bits[:, :, 2] << 16) | (bits[:, :, 3] << 24)
    values = jnp.concatenate(
        (
            words.astype(jnp.int32),
            jax.lax.bitcast_convert_type(scales, jnp.int32)[:, None],
        ),
        axis=1,
    )
    return write_packed_pages(cache, records // page_size, records % page_size, values)


def write_packed_pages(
    cache: jax.Array,
    pages: jax.Array,
    columns: jax.Array,
    values: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Write int32 `values` [N, fields] at unique (page, column) slots; -1 skips."""
    count = values.shape[0]
    if not count:
        return cache
    num_pages = cache.shape[0]
    batch = min(count, PACKED_FP8_WRITE_BATCH)
    valid = (pages >= 0) & (pages < num_pages)
    pages, order = jax.lax.sort_key_val(
        jnp.where(valid, pages, num_pages), jnp.arange(count, dtype=jnp.int32)
    )
    columns, values = columns[order], values[order]
    # The final sentinel terminates the last page's merge without an end scan.
    pages = jnp.pad(pages, (0, -count % batch + 1), constant_values=num_pages)
    columns = jnp.pad(columns, (0, -count % batch))
    values = jnp.pad(values.T, ((0, 0), (0, -count % TOKENS_PER_ROW)))

    return _write_sorted_packed_pages(
        cache, pages, columns, values, count=count, interpret=interpret
    )


def _write_sorted_packed_pages(cache, pages, columns, values, *, count, interpret):
    num_pages, fields, _, page_size = cache.shape
    rows, parts = page_rows(cache.shape), page_size // TOKENS_PER_ROW
    batch = min(count, PACKED_FP8_WRITE_BATCH)

    def kernel(page_ref, column_ref, value_ref, _, output, staged, sem):
        words = output.bitcast(jnp.int32).reshape(num_pages * rows, TOKENS_PER_ROW)

        def first(record):
            page = page_ref[record]
            return (page < num_pages) & (
                (record == 0) | (page != page_ref[jnp.maximum(record - 1, 0)])
            )

        def each_valid(action):
            def body(slot):
                action(slot, 0)
                return slot + 1

            # Sorting puts invalid records last; decode writes only compressed
            # token boundaries, so most input rows need no page work.
            jax.lax.while_loop(
                lambda slot: (
                    (slot < batch)
                    & (page_ref[pl.program_id(0) * batch + slot] < num_pages)
                ),
                body,
                jnp.int32(0),
            )

        def copies(*, write, wait):
            def body(slot, carry):
                record = pl.program_id(0) * batch + slot

                @pl.when(first(record))
                def _():
                    hbm = words.at[pl.ds(page_ref[record] * rows, rows), :]
                    vmem = staged.at[slot]
                    src, dst = (vmem, hbm) if write else (hbm, vmem)
                    copy = pltpu.make_async_copy(src, dst, sem)
                    if wait:
                        copy.wait()
                    else:
                        copy.start()

                return carry

            each_valid(body)

        copies(write=False, wait=False)
        copies(write=False, wait=True)

        def merge(slot, carry):
            start = pl.program_id(0) * batch + slot

            @pl.when(first(start))
            def _():
                def update(record):
                    word = jnp.take_along_axis(
                        value_ref[
                            :,
                            pl.ds(
                                record // TOKENS_PER_ROW * TOKENS_PER_ROW,
                                TOKENS_PER_ROW,
                            ),
                        ],
                        jnp.full((fields, TOKENS_PER_ROW), record % TOKENS_PER_ROW),
                        axis=1,
                        mode="fill",
                    )
                    # Row f * parts + h holds field f of part h.
                    if parts > 1:
                        word = jnp.tile(word, (1, parts)).reshape(rows, TOKENS_PER_ROW)
                    positions = jnp.arange(rows * TOKENS_PER_ROW).reshape(
                        rows, TOKENS_PER_ROW
                    )
                    staged[slot] = jnp.where(
                        positions % page_size == column_ref[record],
                        word,
                        staged[slot],
                    )
                    return record + 1

                jax.lax.while_loop(
                    lambda record: page_ref[record] == page_ref[start], update, start
                )

            return carry

        each_valid(merge)
        copies(write=True, wait=False)
        copies(write=True, wait=True)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(pl.cdiv(count, batch),),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((batch, rows, TOKENS_PER_ROW), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={3: 0},
        interpret=pltpu.InterpretParams() if interpret else False,
        name="write_packed_fp8_pages",
    )(pages, columns, values, cache)
