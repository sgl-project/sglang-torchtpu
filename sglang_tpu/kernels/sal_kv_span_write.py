# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-torchtpu (Apache-2.0), commit 1af8022e6:
# kernels/experimental/pcp_streaming_rpa/kernel.py (_copy_hnd_token_slice,
# write_captured_kv_to_local_cache) and wrapper.py (_pack_kv_for_cache).
# Copyright 2026 Google LLC
"""Span-addressed K/V writes into a SEQ_ALONG_LANE paged cache.

The cache is [pages, 2 * kv_heads, packed_head_dim, packing, page_size] with the
token on the lane axis, so a span of slots is a lane range of one page. Each
span is copied by aligned DMA when source, destination and length sit on
128-token tiles, else stitched tile by tile on packed 32-bit words, leaving
bytes outside the span untouched.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.batched_rpa import configs, wrapper

TPU_KV_DMA_ALIGNMENT = 128


def _copy_hnd_token_slice(
    src_ref,
    dst_ref,
    source_start,
    destination_start,
    length,
    sem,
    *,
    source_hbm,
    destination_hbm,
):
    """Copy HND tokens using aligned DMA and bit-preserving VMEM stitching.

    Both refs have [2*heads, packed_dim, packing, tokens] layout. Partial
    destination tiles are read-modify-written sequentially: other segments
    may share a tile, and bytes outside this slice must survive unchanged.
    Only three 128-token scratch tiles are needed, independent of cache size.
    """
    alignment = TPU_KV_DMA_ALIGNMENT
    aligned = (
        (source_start % alignment == 0)
        & (destination_start % alignment == 0)
        & (length % alignment == 0)
    )

    @pl.when(aligned & (length > 0))
    def _aligned_copy():
        src = pl.multiple_of(source_start, alignment)
        dst = pl.multiple_of(destination_start, alignment)
        size = pl.multiple_of(length, alignment)
        op = pltpu.make_async_copy(
            src_ref.at[:, :, :, pl.ds(src, size)],
            dst_ref.at[:, :, :, pl.ds(dst, size)],
            sem,
        )
        op.start()
        op.wait()

    @pl.when(~aligned & (length > 0))
    def _partial_copy():
        shape = (*src_ref.shape[:-1], alignment)

        def _stitch(src0, src1, dest):
            first = (destination_start // alignment) * alignment
            count = pl.cdiv(destination_start % alignment + length, alignment)

            @pl.loop(0, count)
            def _tile(i):
                dst = pl.multiple_of(first + i * alignment, alignment)
                begin = jnp.maximum(destination_start, dst)
                end = jnp.minimum(destination_start + length, dst + alignment)
                source = source_start + begin - destination_start
                src = pl.multiple_of((source // alignment) * alignment, alignment)
                next_src = pl.multiple_of(
                    jnp.minimum(src + alignment, src_ref.shape[-1] - alignment),
                    alignment,
                )

                def _read(ref, offset, scratch, hbm):
                    if hbm:
                        op = pltpu.make_async_copy(
                            ref.at[:, :, :, pl.ds(offset, alignment)], scratch, sem
                        )
                        op.start()
                        op.wait()
                    else:
                        scratch[...] = ref[:, :, :, pl.ds(offset, alignment)]

                _read(src_ref, src, src0, source_hbm)
                _read(src_ref, next_src, src1, source_hbm)
                _read(dst_ref, dst, dest, destination_hbm)
                # Operate on packed bits, without any FP8 conversion or
                # arithmetic on possibly uninitialized neighbouring tokens.
                a = src0.bitcast(jnp.uint32).reshape(-1, alignment)[...]
                b = src1.bitcast(jnp.uint32).reshape(-1, alignment)[...]
                old = dest.bitcast(jnp.uint32).reshape(-1, alignment)
                lane = lax.broadcasted_iota(jnp.int32, a.shape, 1)
                relative = source - src - (begin - dst)
                a = pltpu.roll(a, -relative, axis=1)
                b = pltpu.roll(b, -relative, axis=1)
                incoming = jnp.where(lane + relative < alignment, a, b)
                valid = (lane >= begin - dst) & (lane < end - dst)
                old[...] = jnp.where(valid, incoming, old[...])
                if destination_hbm:
                    op = pltpu.make_async_copy(
                        dest, dst_ref.at[:, :, :, pl.ds(dst, alignment)], sem
                    )
                    op.start()
                    op.wait()
                else:
                    dst_ref[:, :, :, pl.ds(dst, alignment)] = dest[...]

        pl.run_scoped(
            _stitch,
            pltpu.VMEM(shape, src_ref.dtype),
            pltpu.VMEM(shape, src_ref.dtype),
            pltpu.VMEM(shape, src_ref.dtype),
        )


def _write_spans_kernel(
    num_segments_ref,
    segment_descriptors_ref,
    captured_kv_ref,
    kv_cache_in_ref,
    kv_cache_out_ref,
    dma_sem,
):
    """Scatter HND segments, preserving partial tiles and earlier segments."""
    del kv_cache_in_ref
    num_segments = jnp.clip(num_segments_ref[0], 0, segment_descriptors_ref.shape[1])
    page_size = kv_cache_out_ref.shape[-1]

    @pl.loop(0, num_segments)
    def _write_segment(segment_idx):
        source_start = segment_descriptors_ref[0, segment_idx]
        destination_start = segment_descriptors_ref[1, segment_idx]
        segment_len = segment_descriptors_ref[2, segment_idx]
        page = destination_start // page_size
        offset = destination_start % page_size
        _copy_hnd_token_slice(
            captured_kv_ref,
            kv_cache_out_ref.at[page],
            source_start,
            offset,
            segment_len,
            dma_sem,
            source_hbm=True,
            destination_hbm=True,
        )


def write_spans_seq_along_lane(
    kv_cache: jax.Array,
    captured_kv: jax.Array,
    segment_descriptors: jax.Array,
    num_segments: jax.Array,
) -> jax.Array:
    """Write captured_kv token spans into kv_cache with one donated Pallas call.

    Args:
      kv_cache: [pages, 2 * kv_heads, packed_head_dim, packing, page_size].
      captured_kv: [2 * kv_heads, packed_head_dim, packing, num_tokens], tokens
        padded to a multiple of 128.
      segment_descriptors: [3, segments] int32 rows of (source token,
        destination flat slot, length); a span never crosses a page.
      num_segments: Number of leading descriptors to apply.

    Returns:
      The updated cache, aliased to kv_cache.
    """
    if captured_kv.shape[:-1] != kv_cache.shape[1:-1]:
        raise ValueError(
            "SEQ_ALONG_LANE captured KV prefix must match the paged cache "
            f"head layout: {captured_kv.shape[:-1]} vs {kv_cache.shape[1:-1]}."
        )
    if captured_kv.shape[-1] % TPU_KV_DMA_ALIGNMENT:
        raise ValueError(
            f"Captured KV tokens {captured_kv.shape[-1]} must be a multiple of "
            f"{TPU_KV_DMA_ALIGNMENT}."
        )
    if segment_descriptors.ndim != 2 or segment_descriptors.shape[0] != 3:
        raise ValueError(
            "KV writeback segment descriptors must have shape [3, segments], "
            f"got {segment_descriptors.shape}."
        )

    num_segments = jnp.asarray(num_segments, dtype=jnp.int32).reshape((1,))
    segment_descriptors = jnp.asarray(segment_descriptors, dtype=jnp.int32)
    return pl.pallas_call(
        _write_spans_kernel,
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(pltpu.SemaphoreType.DMA,),
            grid=(1,),
        ),
        input_output_aliases={3: 0},
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes
        ),
        name="sal_kv_span_write",
    )(num_segments, segment_descriptors, captured_kv, kv_cache)


def pack_kv_seq_along_lane(k: jax.Array, v: jax.Array, *, page_size: int) -> jax.Array:
    """[tokens, kv_heads, head_dim] K/V, already in the cache dtype, as captured KV.

    Returns [2 * kv_heads, packed_head_dim, packing, tokens] with K/V
    interleaved per head and tokens zero-padded to a multiple of page_size.
    """
    if page_size <= 0 or page_size % TPU_KV_DMA_ALIGNMENT:
        raise ValueError(
            f"SEQ_ALONG_LANE page_size {page_size} must be a positive multiple "
            f"of {TPU_KV_DMA_ALIGNMENT}."
        )
    dummy_q = jnp.zeros_like(k)
    _, packed_kv = wrapper.prepare_inputs(
        dummy_q,
        k,
        v,
        k.dtype,
        k.dtype,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    padded_tokens = math.ceil(packed_kv.shape[-1] / page_size) * page_size
    return jnp.pad(
        packed_kv,
        ((0, 0), (0, 0), (0, 0), (0, padded_tokens - packed_kv.shape[-1])),
        constant_values=0,
    )
