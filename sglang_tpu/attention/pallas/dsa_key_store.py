# SPDX-License-Identifier: Apache-2.0
"""Decode-sized index-key writer of the DSA indexer, as one Pallas kernel.

For a decode step the XLA path that stores a token's index key costs several
microseconds per layer although it moves one 256-byte record: the key
LayerNorm and the FP8 quantization each end in a scalar reduction that the
next fusion reads back through scalar memory (a ~1.5-3 us bubble each), and
the whole-row scatter into the 4 GB cache is a dynamic-update-slice fusion of
its own. This kernel does the UE8M0 FP8 quantization and a read-modify-write
of each token's cache row by DMA in one launch.

Bit-exact with the XLA path (``quantize_fp8_ue8m0`` + ``_store_keys``): the
scale exponent is derived from the bits of ``abs_max`` instead of
``frexp(abs_max / 448)``. With ``abs_max = 1.f * 2**E`` and
``448 = 1.75 * 2**8``, ``ceil(log2(abs_max / 448))`` is ``E - 8`` when
``1.f <= 1.75`` and ``E - 7`` otherwise. Dividing by a power-of-two scale is
exact, so multiplying by its inverse gives the same values.

The kernel starts every token's row read at once, overlaps them with the
quantization, merges the records (tokens of one request that share a row are
merged in order, and only the last writes the row back), then writes. Its
scratch and scalar work grow with the token count, so it is only meant for
decode-sized batches; ``_store_keys`` stays the path for prefill.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# 448 (float8_e4m3fn max) = 1.75 * 2**8: mantissa bits of 1.75.
_FP8_MAX_MANTISSA = 0x600000
_FP8_MAX = 448.0
_TOKENS_PER_ROW = 4


def _ue8m0_quantize(x):
    """``quantize_fp8_ue8m0`` for f32 ``x[T, D]``: (fp8 values, exponent i32[T, 1])."""
    abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    bits = lax.bitcast_convert_type(abs_max, jnp.int32)
    biased = bits >> 23
    exp = biased - (127 + 8) + ((bits & 0x7FFFFF) > _FP8_MAX_MANTISSA).astype(jnp.int32)
    # frexp(0) has exponent 0 (scale 1); a subnormal abs_max is clipped to the
    # smallest normal exponent like the XLA path's clip.
    exp = jnp.where(biased == 0, jnp.where(bits == 0, 0, -126), exp)
    exp = jnp.clip(exp, -126, 127)
    inv_scale = lax.bitcast_convert_type((127 - exp) << 23, jnp.float32)
    values = jnp.clip(x * inv_scale, -_FP8_MAX, _FP8_MAX).astype(jnp.float8_e4m3fn)
    return values, exp


def _kernel(
    kv_lens_ref,  # SMEM i32[S]
    page_indices_ref,  # SMEM i32[S * pages_per_seq]
    cu_q_lens_ref,  # SMEM i32[S + 1]
    distribution_ref,  # SMEM i32[3]
    key_hbm,  # HBM [T, D]
    cache_in_ref,  # HBM, aliased with cache_ref
    cache_ref,  # HBM u8[P, page_size // 4, 4, record_bytes]
    key_ref,  # VMEM [T, D]
    rec_ref,  # VMEM u32[T, record_bytes]: one byte per lane
    rows_ref,  # VMEM u8[T, 1, 4, record_bytes]: each token's cache row
    page_sm,  # SMEM i32[T]
    row_sm,  # SMEM i32[T]
    shift_sm,  # SMEM i32[T]
    in_sems,  # DMA[T + 1]: row reads, then the key read
    out_sems,  # DMA[T]
    *,
    page_size,
    pages_per_seq,
):
    del cache_in_ref
    head_dim = key_ref.shape[-1]
    record_bytes = rows_ref.shape[-1]
    num_tokens = rows_ref.shape[0]

    def _row(t):
        return cache_ref.at[pl.ds(page_sm[t], 1), pl.ds(row_sm[t], 1)]

    def _read(t):
        return pltpu.make_async_copy(_row(t), rows_ref.at[pl.ds(t, 1)], in_sems.at[t])

    def _write(t):
        return pltpu.make_async_copy(rows_ref.at[pl.ds(t, 1)], _row(t), out_sems.at[t])

    # 1. Find each real token's (page, row, byte) and start reading its row.
    #    distribution[2] is the number of real requests; padded rows are skipped.
    def _seq(seq, carry):
        q_end = cu_q_lens_ref[seq + 1]

        def _token(t, c):
            pos = kv_lens_ref[seq] - (q_end - t)
            page_idx = jnp.minimum(pos // page_size, pages_per_seq - 1)
            page_sm[t] = page_indices_ref[seq * pages_per_seq + page_idx]
            slot = pos % page_size
            row_sm[t] = slot // _TOKENS_PER_ROW
            shift_sm[t] = (slot % _TOKENS_PER_ROW) * 8
            _read(t).start()
            return c

        lax.fori_loop(cu_q_lens_ref[seq], q_end, _token, carry)
        return carry

    num_real = jnp.minimum(cu_q_lens_ref[distribution_ref[2]], num_tokens)
    lax.fori_loop(0, distribution_ref[2], _seq, 0)

    # 2. Meanwhile build the records.
    key_copy = pltpu.make_async_copy(key_hbm, key_ref, in_sems.at[num_tokens])
    key_copy.start()
    key_copy.wait()
    x = key_ref[...].astype(jnp.float32)
    values, exp = _ue8m0_quantize(x)
    # One record per token, one byte per u32 lane: key bytes, scale byte, zeros.
    key_bytes = lax.bitcast_convert_type(values, jnp.uint8).astype(jnp.uint32)
    lane = lax.broadcasted_iota(jnp.int32, (num_tokens, record_bytes - head_dim), 1)
    tail = jnp.where(lane == 0, (exp + 127).astype(jnp.uint32), jnp.uint32(0))
    rec_ref[...] = jnp.concatenate([key_bytes, tail], axis=-1)

    # 3. Merge each record into its row. Consecutive tokens of a request can
    #    share a row: the later one starts from the earlier one's merged row,
    #    and only the last token of a row writes it back.
    def _same_row(a, b):
        return (page_sm[a] == page_sm[b]) & (row_sm[a] == row_sm[b])

    def _merge(t, carry):
        _read(t).wait()

        @pl.when((t > 0) & _same_row(jnp.maximum(t - 1, 0), t))
        def _():
            rows_ref[pl.ds(t, 1)] = rows_ref[pl.ds(t - 1, 1)]

        words = pltpu.bitcast(rows_ref[t, 0], jnp.uint32)  # [1, record_bytes]
        rec = rec_ref[pl.ds(t, 1), :]
        shift = shift_sm[t].astype(jnp.uint32)
        words = (words & ~(jnp.uint32(0xFF) << shift)) | (rec << shift)
        rows_ref[t, 0] = pltpu.bitcast(words, jnp.uint8)

        @pl.when(
            (t == num_real - 1) | ~_same_row(t, jnp.minimum(t + 1, num_tokens - 1))
        )
        def _():
            _write(t).start()

        return carry

    lax.fori_loop(0, num_real, _merge, 0)

    def _drain(t, carry):
        @pl.when(
            (t == num_real - 1) | ~_same_row(t, jnp.minimum(t + 1, num_tokens - 1))
        )
        def _():
            _write(t).wait()

        return carry

    lax.fori_loop(0, num_real, _drain, 0)


def store_keys_pallas(
    cache_k,  # u8[num_pages, page_size // 4, 4, record_bytes]
    key,  # [T, D]: the normed key
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
):
    """Quantize ``key`` and store each real token's record; returns the cache."""
    num_tokens, head_dim = key.shape
    num_pages, rows_per_page, tokens_per_row, record_bytes = cache_k.shape
    assert tokens_per_row == _TOKENS_PER_ROW, cache_k.shape
    num_seqs = kv_lens.shape[0]
    kernel = functools.partial(
        _kernel,
        page_size=rows_per_page * tokens_per_row,
        pages_per_seq=page_indices.shape[0] // num_seqs,
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(cache_k.shape, cache_k.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[
                pltpu.VMEM((num_tokens, head_dim), key.dtype),
                pltpu.VMEM((num_tokens, record_bytes), jnp.uint32),
                pltpu.VMEM((num_tokens, 1, tokens_per_row, record_bytes), jnp.uint8),
                pltpu.SMEM((num_tokens,), jnp.int32),
                pltpu.SMEM((num_tokens,), jnp.int32),
                pltpu.SMEM((num_tokens,), jnp.int32),
                pltpu.SemaphoreType.DMA((num_tokens + 1,)),
                pltpu.SemaphoreType.DMA((num_tokens,)),
            ],
        ),
        input_output_aliases={5: 0},
        name="dsa_index_key_store",
    )(
        kv_lens.astype(jnp.int32),
        page_indices.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        distribution.astype(jnp.int32),
        key,
        cache_k,
    )
