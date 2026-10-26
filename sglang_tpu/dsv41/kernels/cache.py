# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Revision: bdb38700dd25959c029c7d77f00cee91ccd3a0bc
# Sources: models/vllm/deepseek_v41/v2/cache.py and
# kernels/{mla/paged_swa,deepseek_v41/csa2_cache,deepseek_v41/csa2_metadata}.py
# SPDX-License-Identifier: Apache-2.0
"""V4.1 cache layouts, publication, gather and device address preparation.

The flat INT32 pool owns storage throughout. Index halfplanes, BF16 SWA pages
and FP8 CSA2 words/scales retain their distinct layouts and quantization policies.
"""

import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROW_WORDS = 128

# Kernel target


def is_tpu_kernel_target():
    try:
        return pltpu.get_tpu_info().generation is not None
    except (ValueError, NotImplementedError, AssertionError):
        return False


# Index word-row publication


def _permute(x: jax.Array, source: jax.Array) -> jax.Array:
    """``x[..., source]`` as an exact 0/1 matmul (finite ``x``; subnormals flush).

    Lane-stride slices and lane gathers are slow on TPU; the MXU is not.
    """
    permutation = (jnp.arange(x.shape[-1])[:, None] == source[None]).astype(x.dtype)
    return jnp.einsum(
        "...d,de->...e",
        x,
        permutation,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


def piece_order(elements: int) -> jax.Array:
    """Element order of `gather_cache_pieces`: per word row, even then odd."""
    width = min(elements // 2, ROW_WORDS)
    return jnp.concatenate(
        [
            start + 2 * jnp.arange(width) + parity
            for start in range(0, elements, 2 * width)
            for parity in (0, 1)
        ]
    )


def split_halves(x: jax.Array) -> jax.Array:
    """Reorder the last axis as `piece_order` (``split`` reads)."""
    return _permute(x, piece_order(x.shape[-1]))


def merge_sub_rows(
    cache: jax.Array, records: jax.Array, values: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Turn unique sub-row records into full word-row writes.

    Each valid record yields its row: the current row with every segment
    written in this batch replaced. Records sharing a row yield identical
    rows, so writing them in any order is equivalent to the lane scatter.
    Returns ``(row records, int32 [n, ROW_WORDS])``; invalid records stay -1.
    """
    width = values.shape[1]
    per_row = ROW_WORDS // width
    valid = (records >= 0) & (records < cache.shape[0] * per_row)
    rows = jnp.where(valid, records // per_row, -1)
    segments = records % per_row
    merged = cache[jnp.maximum(rows, 0)]
    # [n, n]: which batch records land in each record's row. Batches are a
    # step's tokens (<= max_num_batched_tokens), so this stays small.
    same_row = (rows[:, None] == rows[None, :]) & valid[:, None] & valid[None, :]
    pieces = []
    for segment in range(per_row):
        writers = same_row & (segments[None, :] == segment)
        source = jnp.take(values, jnp.argmax(writers, axis=1), axis=0)
        pieces.append(
            jnp.where(
                writers.any(axis=1)[:, None],
                source,
                merged[:, segment * width : (segment + 1) * width],
            )
        )
    return rows, jnp.concatenate(pieces, axis=1)


def _pack_bf16_words(values: jax.Array) -> jax.Array:
    """Pack BF16 ``[n, 2 * width]`` into int32 ``[n, width]`` via `split_halves`.

    A direct ``bitcast_convert_type(values.reshape(n, -1, 2), int32)`` lowers on
    TPU to stride-2 lane slices ``[0:511:2]`` and ``[1:512:2]`` (~6.0 ms/chunk
    across 21 SWA/compressor writes). Reordering via `split_halves` (MXU 0/1
    matmul) places even and odd elements into contiguous 128-lane halves first.
    """
    half_width = min(values.shape[1] // 2, ROW_WORDS)
    split = split_halves(values).reshape(values.shape[0], -1, 2, half_width)
    even = jax.lax.bitcast_convert_type(
        split[:, :, 0, :].astype(jnp.float32), jnp.int32
    )
    odd = jax.lax.bitcast_convert_type(split[:, :, 1, :].astype(jnp.float32), jnp.int32)
    return (jax.lax.shift_right_logical(even, 16) | (odd & jnp.int32(-65536))).reshape(
        values.shape[0], values.shape[1] // 2
    )


def scatter_cache_values(
    cache: jax.Array,
    records: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """Write unique physical records, preserving the shared owner and padding.

    Full word rows use DSV4-style aliased DMA. Sub-row writes use independent
    (row, lane) coordinates, so two records sharing a row cannot overwrite
    each other through concurrent read/modify/write of the entire word row.
    """
    assert values.dtype in (jnp.bfloat16, jnp.float32, jnp.int32)
    if isinstance(cache, tuple):
        cache_lo, cache_hi = cache
        record_words = (
            values.shape[1] // 2 if values.dtype == jnp.bfloat16 else values.shape[1]
        )
        bank_records = cache_lo.shape[0] * ROW_WORDS // record_words
        records_lo = jnp.where(records < bank_records, records, -1)
        records_hi = jnp.where(records >= bank_records, records - bank_records, -1)
        return cast(
            Any,
            (
                scatter_cache_values(cache_lo, records_lo, values),
                scatter_cache_values(cache_hi, records_hi, values),
            ),
        )
    assert cache.dtype == jnp.int32 and cache.ndim == 2 and cache.shape[1] == ROW_WORDS
    if not values.shape[0]:
        return cache
    words = (
        values
        if values.dtype == jnp.int32
        else (
            _pack_bf16_words(values)
            if (
                values.dtype == jnp.bfloat16
                and is_tpu_kernel_target()
                and values.shape[1] >= 2 * ROW_WORDS
                and values.shape[1] % (2 * ROW_WORDS) == 0
            )
            else jax.lax.bitcast_convert_type(
                values.reshape(values.shape[0], -1, 2)
                if values.dtype == jnp.bfloat16
                else values,
                jnp.int32,
            )
        )
    )
    width = words.shape[1]
    if width < ROW_WORDS and is_tpu_kernel_target():
        # An elementwise (row, lane) scatter is slow on TPU. Rebuild whole word
        # rows instead and reuse the full-row DMA below.
        records, words = merge_sub_rows(cache, records, words)
        width = ROW_WORDS
    if width < ROW_WORDS or not is_tpu_kernel_target():
        columns = jnp.arange(width)[None]
        if width < ROW_WORDS:
            per_row = ROW_WORDS // width
            rows = records[:, None] // per_row
            lanes = (records[:, None] % per_row) * width + columns
        else:
            rows = records[:, None] * (width // ROW_WORDS) + columns // ROW_WORDS
            lanes = jnp.broadcast_to(columns % ROW_WORDS, rows.shape)
        rows = jnp.where(records[:, None] >= 0, rows, cache.shape[0])
        return cache.at[rows, lanes].set(words, mode="drop")

    chunks = width // ROW_WORDS
    # Records per grid step whose row DMAs are in flight together. A serial
    # start/wait per 512 B row is DMA-latency bound.
    block = 64
    count = records.shape[0]
    records = jnp.pad(records, (0, -count % block), constant_values=-1)

    def kernel(record_ref, values_ref, _input_ref, output_ref, sem):
        def copies(index):
            token = pl.program_id(0) * block + index
            record = record_ref[token]
            return (record >= 0) & (record < cache.shape[0] // chunks), [
                pltpu.make_async_copy(
                    values_ref.at[
                        pl.ds((token * chunks + chunk) * ROW_WORDS, ROW_WORDS)
                    ],
                    output_ref.at[record * chunks + chunk, :],
                    sem,
                )
                for chunk in range(chunks)
            ]

        def run(method):
            def body(index, carry):
                valid, row_copies = copies(index)

                @pl.when(valid)
                def _():
                    for copy in row_copies:
                        getattr(copy, method)()

                return carry

            jax.lax.fori_loop(0, block, body, 0)

        run("start")
        run("wait")

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(records.shape[0] // block,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={2: 0},
        # Flatten only the small update batch. A partial DMA from a tiled 2D
        # source can straddle its padded minor dimension; the cache stays 2D.
    )(records, words.reshape(-1), cache)


def scatter_index_keys(
    cache: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Index-only half-plane RMW, retaining the existing aliased pool writer."""
    from sglang_tpu.kernels.index_selection.key_layout import merge_record_rows

    if isinstance(cache, tuple):
        boundary = cache[0].shape[0] * 2
        return (
            scatter_index_keys(
                cache[0], jnp.where(records < boundary, records, -1), values
            ),
            scatter_index_keys(
                cache[1], jnp.where(records >= boundary, records - boundary, -1), values
            ),
        )
    rows, words = merge_record_rows(cache, records, values)
    return scatter_cache_values(cache, rows, words)


# BF16 sliding-window pages


def swa_word_pages(
    raw: jax.Array | tuple[jax.Array, jax.Array], page_size: int
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """View shared INT32 [rows,128] owners as plane-major SWA pages.

    The allocator's physical page stride/offset must map to whole 2*S-row
    pages. Banks split on page boundaries. This does not change ownership or
    allocate a second cache; verify bitcast/alias lowering on the target TPU.
    """
    if type(page_size) is not int or (
        page_size not in (8, 16, 32, 64) and (page_size < 128 or page_size % 128)
    ):
        raise ValueError(
            "SWA page size must be 8,16,32,64 or a positive multiple of 128"
        )
    if isinstance(raw, tuple):
        return tuple(swa_word_pages(bank, page_size) for bank in raw)
    if (
        raw.ndim != 2
        or raw.shape[1] != 128
        or raw.dtype != jnp.int32
        or raw.shape[0] % (2 * page_size)
    ):
        raise ValueError("SWA owner must be INT32 [rows,128] with whole physical pages")
    return raw.reshape(-1, 2, page_size, 128)


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("page_size",))
def write_swa_word_pool(
    raw: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
    *,
    page_size: int,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Publish SWA while preserving the shared flat word-row owner interface."""
    swa_word_pages(raw, page_size)
    return _publish_swa_sc(raw, records, values, page_size)


def _publish_swa_sc(raw, records, values, page_size):
    """Pack each current token once, then scatter both planes across all banks."""
    from jax.experimental.pallas import tpu_sc as plsc

    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != jnp.bfloat16:
        raise ValueError("SWA values must be BF16 [N,512]")
    if records.shape != values.shape[:1] or records.dtype != jnp.int32:
        raise ValueError("SWA records must be INT32 [N]")
    if not records.size:
        return raw
    banks = raw if isinstance(raw, tuple) else (raw,)
    bits = jax.lax.bitcast_convert_type(values, jnp.uint16).astype(jnp.uint32)
    bits = bits.reshape(-1, 2, 2, 128)
    words = (bits[:, :, 0] | (bits[:, :, 1] << 16)).reshape(-1, 128)
    words = jax.lax.bitcast_convert_type(words, jnp.int32)
    physical = (records // page_size * (2 * page_size) + records % page_size)[
        :, None
    ] + jnp.arange(2, dtype=jnp.int32)[None, :] * page_size
    physical = jnp.where(records[:, None] >= page_size, physical, -1).reshape(-1)
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    padding = -physical.size % (lanes * cores)
    ids = jnp.pad(physical, (0, padding), constant_values=-1)
    words = jnp.pad(words, ((0, padding), (0, 0)))
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    owners = tuple(jax.new_ref(bank) for bank in banks)

    def kernel(ids_ref, values_ref, out_refs, sem):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        def store(selected, word):
            copies = []
            base = 0
            for bank in range(len(banks)):
                valid = (selected[...] >= base) & (
                    selected[...] < base + banks[bank].shape[0]
                )
                indices = plsc.Indices(
                    jnp.where(valid, selected[...] - base, -1), ignored_value=-1
                )
                copy = pltpu.make_async_copy(
                    word, out_refs[bank].at[indices], sem.at[bank]
                )
                copy.start()
                copies.append(copy)
                base += banks[bank].shape[0]
            for copy in copies:
                copy.wait()

        pltpu.emit_pipeline(
            store,
            grid=(ids.size // (lanes * cores),),
            in_specs=(
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                pl.BlockSpec((lanes, 128), lambda b: (b * cores + core, 0)),
            ),
        )(ids_ref, values_ref)

    pl.kernel(
        kernel,
        mesh=mesh,
        scratch_types=(pltpu.SemaphoreType.DMA((len(banks),)),),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name="publish_swa_word_pool",
    )(ids, words, owners)
    result = tuple(owner[...] for owner in owners)
    return result if isinstance(raw, tuple) else result[0]


# FP8 compressed KV publication and gather


@jax.jit
def encode_csa2_kv(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Post-RoPE BF16 [N,512] -> INT32 [N,128], FP32 [N] scales.

    Byte g of word lane encodes feature g*128+lane. Quantize the unique
    compressed records once, before publishing them, not after query gather.
    """
    if latent.ndim != 2 or latent.shape[1] != 512 or latent.dtype != jnp.bfloat16:
        raise ValueError("CSA2 latents must be BF16 [N,512]")
    values = latent.astype(jnp.float32)
    scales = jnp.maximum(jnp.abs(values).max(axis=1), 1e-4) / 448
    quantized = (values / scales[:, None]).astype(jnp.float8_e4m3fn)
    planes = jax.lax.bitcast_convert_type(quantized, jnp.uint8).reshape(-1, 4, 128)
    planes = planes.astype(jnp.uint32)
    words = (
        planes[:, 0] | (planes[:, 1] << 8) | (planes[:, 2] << 16) | (planes[:, 3] << 24)
    )
    return jax.lax.bitcast_convert_type(words, jnp.int32), scales


@jax.jit
def gather_csa2_kv_sc(
    words: jax.Array | tuple[jax.Array, jax.Array],
    scales: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """SparseCore producer; invalid rows are zeroed once before any reuse."""
    banks = words if isinstance(words, tuple) else (words,)
    scale_banks = scales if isinstance(scales, tuple) else (scales,)
    if len(banks) not in (1, 2) or len(banks) != len(scale_banks):
        raise ValueError("CSA2 requires matching one or two word/scale banks")
    for bank, scale in zip(banks, scale_banks, strict=True):
        if bank.ndim != 2 or bank.shape[1] != 128 or bank.dtype != jnp.int32:
            raise ValueError("CSA2 word banks must be INT32 [records,128]")
        if (
            not bank.shape[0]
            or scale.shape != bank.shape[:1]
            or scale.dtype != jnp.float32
        ):
            raise ValueError("CSA2 scale banks must be nonempty FP32 [records]")
    if records.ndim != 2 or records.shape[1] != 512 or records.dtype != jnp.int32:
        raise ValueError("CSA2 selection IDs must be INT32 [T,512]")
    if not records.size:
        return jnp.zeros((*records.shape, 128), jnp.int32), jnp.zeros(
            records.shape, jnp.float32
        )
    return _gather_sc_banked(banks, scale_banks, records)


def _gather_sc_banked(words, scales, records):
    """Masked DMA into shared VMEM; no concatenated HBM cache.

    Two streams follow the latest epic banked gather's v7x DMA concurrency
    bound. Complementary bank masks fill disjoint rows of each staging slot.
    """
    from jax.experimental.pallas import tpu_sc as plsc

    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes = info.num_lanes
    cores = info.num_cores * info.num_subcores
    streams, subchunks = (4 if len(words) == 1 else 2), 512 // lanes
    block = cores * lanes * subchunks
    count = records.size
    padding = -count % block
    indices = jnp.pad(records.reshape(-1), (0, padding), constant_values=-1)
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )

    def kernel(
        word_refs,
        scale_refs,
        ids_ref,
        word_out,
        scale_out,
        word_buf,
        scale_buf,
        read_sem,
        write_sem,
    ):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        split = word_refs[0].shape[0]
        steps = subchunks // streams

        def outer(ids):
            base = (pl.program_id(0) * cores + core) * subchunks * lanes

            def gather(step, wait=False):
                slot = step % 2
                for stream in range(streams):
                    selected = ids[pl.ds((step * streams + stream) * lanes, lanes)]
                    if not wait:
                        word_buf[slot, stream] = jnp.zeros((lanes, 128), jnp.int32)
                        scale_buf[slot, stream] = jnp.zeros((lanes,), jnp.float32)
                    for bank in range(len(words)):
                        own = (selected >= 0) & (
                            selected >= split if bank else selected < split
                        )
                        local = selected - split if bank else selected
                        indexed = plsc.Indices(
                            jnp.where(own, local, -1), ignored_value=-1
                        )
                        for kind, (src, dst) in enumerate(
                            ((word_refs[bank], word_buf), (scale_refs[bank], scale_buf))
                        ):
                            copy = pltpu.make_async_copy(
                                src.at[indexed],
                                dst.at[slot, stream],
                                read_sem.at[slot, bank, stream, kind],
                            )
                            if wait:
                                copy.wait()
                            else:
                                copy.start()

            def step(index, _):
                slot = index % 2
                gather(index, wait=True)

                @pl.when(index + 1 < steps)
                def prefetch():
                    gather(index + 1)

                copies = []
                for stream in range(streams):
                    offset = base + (index * streams + stream) * lanes
                    for kind, (src, dst) in enumerate(
                        ((word_buf, word_out), (scale_buf, scale_out))
                    ):
                        copy = pltpu.make_async_copy(
                            src.at[slot, stream],
                            dst.at[pl.ds(offset, lanes)],
                            write_sem.at[stream, kind],
                        )
                        copy.start()
                        copies.append(copy)
                for copy in copies:
                    copy.wait()

            gather(jnp.int32(0))
            jax.lax.fori_loop(0, steps, step, None)

        pltpu.emit_pipeline(
            outer,
            grid=((count + padding) // block,),
            in_specs=(
                pl.BlockSpec((lanes * subchunks,), lambda b: (b * cores + core,)),
            ),
        )(ids_ref)

    out, scale_out = cast(
        tuple[jax.Array, jax.Array],
        pl.kernel(
            kernel,
            out_type=(
                jax.ShapeDtypeStruct((count + padding, 128), jnp.int32),
                jax.ShapeDtypeStruct((count + padding,), jnp.float32),
            ),
            scratch_types=(
                pltpu.VMEM((2, streams, lanes, 128), jnp.int32),
                pltpu.VMEM((2, streams, lanes), jnp.float32),
                pltpu.SemaphoreType.DMA((2, len(words), streams, 2)),
                pltpu.SemaphoreType.DMA((streams, 2)),
            ),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=False,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            mesh=mesh,
            name="csa2_banked_word_scale_gather",
        )(
            words,
            # pyrefly: ignore[bad-argument-count]
            scales,
            indices,
        ),
    )
    return out[:count].reshape(*records.shape, 128), scale_out[:count].reshape(
        records.shape
    )


@functools.partial(jax.jit, donate_argnums=(0, 1))
def publish_csa2_kv(
    words: jax.Array | tuple[jax.Array, jax.Array],
    scales: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    latent: jax.Array,
) -> tuple[
    jax.Array | tuple[jax.Array, jax.Array], jax.Array | tuple[jax.Array, jax.Array]
]:
    """Quantize unique current records once and publish words plus scalar scales.

    A pair of banks has concatenated record addressing. Negative records skip
    both stores. Valid record IDs must be unique and in range. Arrays retain
    their shared owner's physical row geometry; no full-cache concatenation
    or intermediate expanded KV is produced.
    """
    from jax.experimental.pallas import tpu_sc as plsc

    banked = isinstance(words, tuple)
    banks = words if banked else (words,)
    scale_banks = scales if isinstance(scales, tuple) else (scales,)
    if len(banks) not in (1, 2) or len(banks) != len(scale_banks):
        raise ValueError("Publication requires matching one or two banks")
    for bank, scale_bank in zip(banks, scale_banks, strict=True):
        if bank.ndim != 2 or bank.shape[1] != 128 or bank.dtype != jnp.int32:
            raise ValueError("Publication words must be INT32 [records,128]")
        if scale_bank.shape != bank.shape[:1] or scale_bank.dtype != jnp.float32:
            raise ValueError("Publication scales must be FP32 [records]")
    if records.dtype != jnp.int32 or records.shape != latent.shape[:1]:
        raise ValueError("Publication records must be INT32 [N]")
    encoded, encoded_scales = encode_csa2_kv(latent)
    if not records.size:
        return words, scales
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    padding = -records.size % (lanes * cores)
    ids = jnp.pad(records, (0, padding), constant_values=-1)
    values = jnp.pad(encoded, ((0, padding), (0, 0)))
    value_scales = jnp.pad(encoded_scales, (0, padding))
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    word_refs = tuple(jax.new_ref(b) for b in banks)
    scale_refs = tuple(jax.new_ref(b) for b in scale_banks)

    def kernel(ids_ref, values_ref, scales_ref, out_words, out_scales, sem):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        def store(selected, word, scale):
            copies = []
            base = 0
            for bank in range(len(banks)):
                valid = (selected[...] >= base) & (
                    selected[...] < base + banks[bank].shape[0]
                )
                indices = plsc.Indices(
                    jnp.where(valid, selected[...] - base, -1), ignored_value=-1
                )
                for kind, (src, dst) in enumerate(
                    ((word, out_words[bank]), (scale, out_scales[bank]))
                ):
                    copy = pltpu.make_async_copy(
                        src, dst.at[indices], sem.at[bank, kind]
                    )
                    copy.start()
                    copies.append(copy)
                base += banks[bank].shape[0]
            for copy in copies:
                copy.wait()

        pltpu.emit_pipeline(
            store,
            grid=(ids.size // (lanes * cores),),
            in_specs=(
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                pl.BlockSpec((lanes, 128), lambda b: (b * cores + core, 0)),
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
            ),
        )(ids_ref, values_ref, scales_ref)

    pl.kernel(
        kernel,
        mesh=mesh,
        scratch_types=(pltpu.SemaphoreType.DMA((len(banks), 2)),),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name="csa2_publish_words_scales",
    )(
        ids,
        # pyrefly: ignore[bad-argument-count]
        values,
        value_scales,
        word_refs,
        scale_refs,
    )
    out_words, out_scales = (
        tuple(r[...] for r in word_refs),
        tuple(r[...] for r in scale_refs),
    )
    return (out_words, out_scales) if banked else (out_words[0], out_scales[0])


# Device address preparation


def _requests(starts: jax.Array, tokens: int) -> jax.Array:
    return jnp.repeat(
        jnp.arange(starts.shape[0] - 1, dtype=jnp.int32),
        starts[1:] - starts[:-1],
        total_repeat_length=tokens,
    )


@functools.partial(
    jax.jit,
    static_argnames=("page_size", "page_stride", "row_offset", "non_causal"),
)
def prepare_swa_metadata(
    block_table: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    minimum_visible: jax.Array,
    *,
    page_size: int,
    page_stride: int,
    row_offset: int,
    non_causal: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return request page metadata, explicit bounds, and physical write records.

    Block table entries are allocator block IDs. Stride/offset are in BF16
    512-feature record units and must preserve whole word pages. Starts and
    positions obey csa2's contiguous-request contract; noncausal requests have
    at most 64 queries. Minimum visibility carries replay/window-start state.
    Bounds are passed to csa2 ONLY for noncausal calls; causal visibility stays
    query-dependent. Padding records are -1. Empty requests have empty bounds.
    """
    tokens = positions.shape[0]
    requests = block_table.shape[0]
    if (
        block_table.ndim != 2
        or block_table.shape[1] < 1
        or starts.shape != (requests + 1,)
        or minimum_visible.shape != (requests,)
        or not tokens
        or page_size not in (8, 16, 32, 64)
        and (page_size < 128 or page_size % 128)
        or page_stride % page_size
        or row_offset % page_size
    ):
        raise ValueError("CSA2 requires nonempty, whole-page aligned SWA metadata")
    first = positions[jnp.minimum(starts[:-1], tokens - 1)]
    count = starts[1:] - starts[:-1]
    left = jnp.maximum(first - (128 if non_causal else 127), 0)
    right = first + count
    left = jnp.where(count > 0, left, 0)
    right = jnp.where(count > 0, right, 0)
    first_page = left // page_size
    # An unaligned first page can add one page to ceil((queries+history)/S).
    columns = (tokens + 128 + page_size - 1) // page_size + 1
    logical_pages = first_page[:, None] + jnp.arange(columns, dtype=jnp.int32)
    blocks = jnp.take_along_axis(
        block_table, jnp.minimum(logical_pages, block_table.shape[1] - 1), axis=1
    )
    physical = blocks * (page_stride // page_size) + row_offset // page_size
    visible = (
        (count[:, None] > 0)
        & (logical_pages < block_table.shape[1])
        & (logical_pages * page_size < right[:, None])
    )
    metadata = jnp.concatenate(
        (
            (first_page * page_size)[:, None],
            minimum_visible[:, None],
            jnp.where(visible, physical, 0),
        ),
        axis=1,
    ).astype(jnp.int32)
    req = _requests(starts, tokens)
    current_blocks = block_table[
        req, jnp.minimum(positions // page_size, block_table.shape[1] - 1)
    ]
    records = row_offset + current_blocks * page_stride + positions % page_size
    records = jnp.where(jnp.arange(tokens) < starts[-1], records, -1)
    return metadata, jnp.stack((left, right), axis=1), records.astype(jnp.int32)


@functools.partial(jax.jit, static_argnames=("page_size",))
def prepare_main_records(
    selected: jax.Array,
    physical_pages: jax.Array,
    starts: jax.Array,
    *,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Resolve already-compacted logical selections; preserve stage gather reuse.

    Page IDs address contiguous records in concatenated physical banks. Caller
    guarantees all nonnegative selections are in range and form a valid prefix.
    Returns fixed [T,512] physical IDs and authoritative [T] valid-prefix counts.
    This operation belongs to the first gather for a source/index group only.
    """
    if selected.ndim != 2 or selected.shape[1] > 512 or page_size < 1:
        raise ValueError("CSA2 main selection width must be at most 512")
    tokens, width = selected.shape
    req = _requests(starts, tokens)
    valid = (selected >= 0) & (jnp.arange(tokens)[:, None] < starts[-1])
    logical = jnp.maximum(selected, 0)
    if is_tpu_kernel_target():
        from sglang_tpu.dsv4.core_attention import sparse_mla

        # Preserve the epic optimized page lookup. Ordinary JAX indexing is
        # especially expensive for single-request, long-context prefill.
        padded = jnp.pad(logical, ((0, 0), (0, 512 - width)))
        pages = sparse_mla.gather_page_ids(
            physical_pages.reshape(-1),
            padded // page_size,
            req,
            physical_pages.shape[0],
        )[:, :width]
    else:
        pages = physical_pages[req[:, None], logical // page_size]
    records = jnp.where(valid, pages * page_size + logical % page_size, -1)
    records = jnp.pad(records, ((0, 0), (0, 512 - width)), constant_values=-1)
    return records.astype(jnp.int32), jnp.sum(valid, axis=1, dtype=jnp.int32)
