# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk/streamindex_topk.py
"""TPU-Friendly StreamIndex Top-K kernel."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from sglang_tpu.dsv4.sparsecore_topk import (
    BITS_NEG_INF,
    sparsecore_topk,
)
from sglang_tpu.dsv4.streamindex_topk import (
    bref_override,
    config,
    metadata,
)

MlaCase = config.MlaCase
KVLayout = config.KVLayout
DEFAULT_BUFFER_COUNT = config.DEFAULT_BUFFER_COUNT
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
DCP_AXIS_NAME = "dcp"

cp_local_to_global = metadata.cp_local_to_global
cp_local_length = metadata.cp_local_length
cp_owner_rank = metadata.cp_owner_rank
cp_global_to_local = metadata.cp_global_to_local
cp_rank_as_data = metadata.cp_rank_as_data


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def kernel(
    # Prefetch
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [3] (start_seq, end_seq, chunk_token_start)
    metadata_ref,  # MetadataRef
    cp_rank_ref,  # [1]
    # Input
    q_hbm_ref,  # Shape: [max_num_tokens, num_q_heads, head_dim], Memory: HBM
    indexer_weights_hbm_ref,  # Shape: [max_num_tokens, num_q_heads], Memory: HBM
    cache_kv_hbm_ref,  # HEAD_ALONG_SUBLANE: [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    # SEQ_ALONG_LANE: [total_num_pages, kv_sublane_groups, kv_packing, page_size]
    # HBM scores, [chunk_tokens, num_sublanes_total, 128]: (out,), or
    # (donated_in, out) after the first pass. Only the last is ever read.
    *score_refs,
    compression_ratio: int,
    static_q_len: int | None,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
    buffer_count: int = 3,
    case: MlaCase = MlaCase.MIXED,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    cp_size: int = 1,
    interleave_c: int = 1,
    chunk_idx: int = 0,
    causal_offset: int = 0,
):
    """Executes the n-buffered Pallas scoring pipeline for StreamIndex Top-K."""
    scores_hbm_ref = score_refs[-1]
    buffer_cnt = buffer_count
    seq_along_lane = kv_layout == KVLayout.SEQ_ALONG_LANE

    _, num_q_heads, head_dim = q_hbm_ref.shape

    if seq_along_lane:
        (_, kv_sublane_groups, kv_packing, page_size) = cache_kv_hbm_ref.shape
        kv_head_dim_groups = head_dim // kv_packing
        page_size_per_kv_packing = page_size // kv_packing
    else:
        (
            _,
            page_size_per_kv_packing,
            kv_packing,
            _,
        ) = cache_kv_hbm_ref.shape
        page_size = page_size_per_kv_packing * kv_packing
        kv_sublane_groups = None
        kv_head_dim_groups = None

    max_num_seqs = seq_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    pages_per_seq = num_page_indices // max_num_seqs

    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_p * page_size
    num_sublanes_bkv = bkv_sz // 128
    shift_comp = (compression_ratio - 1).bit_length()

    step_offset = sum(metadata_ref.num_steps[m] for m in range(chunk_idx))

    def seq_tile(p_id):
        """Sequence tile scheduled at pipeline step `p_id`."""
        return metadata_ref.batch_tile_idx[step_offset + p_id]

    def bq_block(p_id):
        """Query block scheduled at pipeline step `p_id`."""
        return metadata_ref.bq_idx[step_offset + p_id]

    def bkv_block(p_id):
        """KV block scheduled at pipeline step `p_id`."""
        return metadata_ref.bkv_idx[step_offset + p_id]

    q_spec = pl.BlockSpec(
        (pl.BoundedSlice(seq_batch_size * bq_sz), num_q_heads, head_dim),
        lambda p_id: (seq_tile(p_id), bq_block(p_id), 0),
        pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
    )
    weights_spec = pl.BlockSpec(
        (pl.BoundedSlice(seq_batch_size * bq_sz), num_q_heads),
        lambda p_id: (seq_tile(p_id), bq_block(p_id), 0),
        pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
    )
    if seq_along_lane:
        kv_spec = pl.BlockSpec(
            (
                pl.BoundedSlice(seq_batch_size),
                kv_sublane_groups,
                kv_packing,
                bkv_sz,
            ),
            lambda p_id: (seq_tile(p_id), 0, 0, bkv_block(p_id)),
            pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
        )
    else:
        kv_spec = pl.BlockSpec(
            (
                pl.BoundedSlice(seq_batch_size),
                bkv_sz_per_kv_packing,
                kv_packing,
                cache_kv_hbm_ref.shape[-1],
            ),
            lambda p_id: (seq_tile(p_id), 0, bkv_block(p_id), 0),
            pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
        )
    out_spec = pl.BlockSpec(
        (pl.BoundedSlice(seq_batch_size * bq_sz), num_sublanes_bkv, 128),
        lambda p_id: (seq_tile(p_id), bq_block(p_id), bkv_block(p_id), 0),
        pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
    )

    q_alloc = bref_override.StreamIndexQBufferedRef.input(
        spec=q_spec,
        dtype_or_type=q_hbm_ref,
        buffer_count=buffer_cnt,
        use_lookahead=True,
        bq_sz=bq_sz,
        seq_batch_size=seq_batch_size,
        chunk_tokens=scores_hbm_ref.shape[0],
        chunk_idx=chunk_idx,
    )
    weights_alloc = bref_override.StreamIndexQBufferedRef.input(
        spec=weights_spec,
        dtype_or_type=indexer_weights_hbm_ref,
        buffer_count=buffer_cnt,
        use_lookahead=True,
        bq_sz=bq_sz,
        seq_batch_size=seq_batch_size,
        chunk_tokens=scores_hbm_ref.shape[0],
        chunk_idx=chunk_idx,
    )
    if seq_along_lane:
        kv_alloc = bref_override.StreamIndexKVSeqAlongLaneBufferedRef.input(
            spec=kv_spec,
            dtype_or_type=cache_kv_hbm_ref,
            buffer_count=buffer_cnt,
            use_lookahead=True,
            bkv_p=bkv_p,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
        )
    else:
        kv_alloc = bref_override.StreamIndexKVBufferedRef.input(
            spec=kv_spec,
            dtype_or_type=cache_kv_hbm_ref,
            buffer_count=buffer_cnt,
            use_lookahead=True,
            bkv_p=bkv_p,
            page_size_per_kv_packing=page_size_per_kv_packing,
            kv_packing=kv_packing,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
        )
    o_alloc = bref_override.StreamIndexOBufferedRef.output(
        spec=out_spec,
        dtype_or_type=scores_hbm_ref,
        buffer_count=2,
        use_lookahead=False,
        bq_sz=bq_sz,
        num_sublanes=num_sublanes_bkv,
        seq_batch_size=seq_batch_size,
        chunk_idx=chunk_idx,
    )

    def transpose_to_head_major(bq_rows, n_tok):
        """[n_tok*H, D] head-minor rows -> [H*n_tok, D] head-major rows."""
        return bq_rows.transpose(1, 0, 2).reshape(num_q_heads * n_tok, head_dim)

    def reduce_heads(bq_rows, bkv, bq_weights, head_major=False):
        """relu(q @ k) weighted-summed over heads."""
        n_tok = bq_weights.shape[0]
        w = bq_weights.astype(jnp.float32)

        def qk(rows):
            if seq_along_lane:
                return jnp.einsum(
                    "nd,dm->nm", rows, bkv, preferred_element_type=jnp.float32
                )
            return jnp.einsum(
                "nd,md->nm", rows, bkv, preferred_element_type=jnp.float32
            )

        if n_tok == 1:
            st = qk(bq_rows).reshape(n_tok, num_q_heads, -1)
            st = jnp.maximum(st, 0.0) * w[:, :, None]
            return st.sum(axis=1)

        rows_hm = bq_rows if head_major else transpose_to_head_major(bq_rows, n_tok)
        st = qk(rows_hm).reshape(num_q_heads, n_tok, -1)
        acc = jnp.maximum(st[0], 0.0) * w[:, 0:1]
        for h in range(1, num_q_heads):
            relu_st = jnp.maximum(st[h], 0.0)
            acc = acc + relu_st * w[:, h : h + 1]
        return acc

    def compute_scores(
        bq_vec,
        bkv_vec,
        scale_val_vec,
        bq_weights_vec,
        bq_pos_compressed_vec,
        bkv_idx,
        batch_start_seq_idx,
        cp_rank=0,
        q_is_head_major=False,
    ):
        assert len(bq_vec) == seq_batch_size
        assert len(bkv_vec) == seq_batch_size
        assert len(scale_val_vec) == seq_batch_size
        assert len(bq_weights_vec) == seq_batch_size
        assert len(bq_pos_compressed_vec) == seq_batch_size
        ret = []

        for batch_idx in range(seq_batch_size):
            bq = bq_vec[batch_idx].reshape(-1, head_dim)
            bkv = bkv_vec[batch_idx]
            scale_val = scale_val_vec[batch_idx]
            bq_weights = bq_weights_vec[batch_idx]
            bq_pos_compressed = bq_pos_compressed_vec[batch_idx]

            s_summed = reduce_heads(bq, bkv, bq_weights, head_major=q_is_head_major)
            s_summed = s_summed * scale_val
            k_local = bkv_idx * bkv_sz + lax.broadcasted_iota(
                jnp.int32, s_summed.shape, 1
            )
            k_span = cp_local_to_global(k_local, cp_rank, cp_size, interleave_c)
            seq_len = seq_lens_ref[batch_start_seq_idx + batch_idx]
            kv_len = seq_len >> shift_comp

            valid_mask = k_span < kv_len
            causal_mask = k_span <= bq_pos_compressed[:, None]
            mask = jnp.logical_and(valid_mask, causal_mask)
            s_summed = jnp.where(mask, s_summed, -jnp.inf)
            ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))
        return jnp.concatenate(ret, axis=0)

    def step_body(q_vmem, weights_vmem, bkv_vmem, scores_vmem, q_hm_ref=None):
        p_id = pl.program_id(0)
        batch_start_seq_idx = seq_tile(p_id)
        bq_idx = bq_block(p_id)
        bkv_idx = bkv_block(p_id)
        cp_rank = cp_rank_ref[0]

        if q_hm_ref is not None:

            @pl.when(bkv_idx == 0)
            def _():
                q_hm_ref[...] = transpose_to_head_major(q_vmem[...], bq_sz)

            bq_vec = [q_hm_ref[...]]
        else:
            q_data = q_vmem[...].reshape(seq_batch_size, -1, head_dim)
            bq_vec = [q_data[batch_idx] for batch_idx in range(seq_batch_size)]
        w_data = weights_vmem[...].reshape(seq_batch_size, -1, num_q_heads)
        bq_weights_vec = [w_data[batch_idx] for batch_idx in range(seq_batch_size)]

        chunk_start = start_end_seq_idx_ref[2]
        bq_pos_compressed_vec = []
        for batch_idx in range(seq_batch_size):
            s_l = seq_lens_ref[batch_start_seq_idx + batch_idx]
            q_s = cu_q_lens_ref[batch_start_seq_idx + batch_idx]
            q_e = cu_q_lens_ref[batch_start_seq_idx + batch_idx + 1]
            q_l = q_e - q_s
            block_start = bq_idx * bq_sz + jnp.maximum(0, chunk_start - q_s)
            q_pos = s_l - q_l + block_start + jnp.arange(bq_sz, dtype=jnp.int32)
            bq_pos_compressed_vec.append((q_pos - causal_offset) >> shift_comp)

        bkvs = []
        bkv_scales = []
        for batch_idx in range(seq_batch_size):
            if seq_along_lane:
                key_bytes = bkv_vmem[batch_idx, :kv_head_dim_groups]
                fp8_val = pltpu.bitcast(
                    key_bytes.reshape(head_dim, bkv_sz), jnp.float8_e4m3fn
                )
                scale_bytes = bkv_vmem[
                    batch_idx, kv_head_dim_groups : kv_head_dim_groups + 1
                ].reshape(kv_packing, bkv_sz)[0:1]
                scale_val = pltpu.bitcast(scale_bytes, jnp.float8_e8m0fnu).astype(
                    jnp.bfloat16
                )
                bkvs.append(fp8_val)
                bkv_scales.append(scale_val)
            else:
                bkv = bkv_vmem[batch_idx, :bkv_sz_per_kv_packing][...]
                flat_bkv = bkv.reshape(-1, bkv.shape[-1])
                fp8_val = pltpu.bitcast(flat_bkv[:, :head_dim], jnp.float8_e4m3fn)
                scale_val = pltpu.bitcast(
                    flat_bkv[:, head_dim : head_dim + 1].T, jnp.float8_e8m0fnu
                ).astype(jnp.bfloat16)
                bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
                bkv_scales.append(scale_val)

        scores = compute_scores(
            bq_vec,
            bkvs,
            bkv_scales,
            bq_weights_vec,
            bq_pos_compressed_vec,
            bkv_idx,
            batch_start_seq_idx,
            q_is_head_major=q_hm_ref is not None,
            cp_rank=cp_rank,
        )
        scores_vmem[...] = pltpu.bitcast(scores, jnp.int32)

    def _run_all(q_hm_ref=None):
        @pl.with_scoped(final_allocs=(q_alloc, weights_alloc, kv_alloc, o_alloc))
        def _run_pipeline(final_allocs):
            q_buf, weights_buf, kv_buf, o_buf = final_allocs
            pipeline_fn = pltpu.emit_pipeline(
                functools.partial(step_body, q_hm_ref=q_hm_ref),
                grid=(metadata_ref.num_steps[chunk_idx],),
                in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
                out_specs=o_buf.spec,
            )
            window_refs = (cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref)
            pipeline_fn(
                (q_hbm_ref, *window_refs),
                (indexer_weights_hbm_ref, *window_refs),
                (cache_kv_hbm_ref, page_indices_ref, metadata_ref),
                (scores_hbm_ref, *window_refs),
                allocations=(q_buf, weights_buf, kv_buf, o_buf),
            )

        _run_pipeline()

    if seq_batch_size == 1 and bq_sz > 1:
        pl.run_scoped(
            _run_all,
            pltpu.VMEM((num_q_heads * bq_sz, head_dim), q_hbm_ref.dtype),
        )
    else:
        _run_all()


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return q


def prepare_index_weights(
    index_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads],
    q_dtype,
):
    _, actual_num_q_heads = index_weights.shape
    index_weights = index_weights.astype(jnp.float32)
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q_dtype))
    index_weights = jnp.pad(
        index_weights,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    return index_weights


def prepare_outputs(out):
    if out.ndim == 3:
        out = out.reshape(out.shape[0], -1)
    return out


KV_PACKING = 4


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    actual_head_dim: int,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
) -> tuple[int, ...]:
    """Returns the packed uint8 index KV cache shape for a given layout."""
    kv_layout = KVLayout.parse(kv_layout)
    head_dim = align_to(actual_head_dim, 128)
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if page_size % 128 != 0:
            raise ValueError(
                "SEQ_ALONG_LANE requires page_size to be a multiple of the 128-lane"
                f" register width, got {page_size=}."
            )
        return (
            total_num_pages,
            head_dim // KV_PACKING + 1,
            KV_PACKING,
            page_size,
        )
    if page_size % KV_PACKING != 0:
        raise ValueError(
            "HEAD_ALONG_SUBLANE requires page_size to be a multiple of"
            f" {KV_PACKING}, got {page_size=}."
        )
    record_width = head_dim + head_dim // 128
    return (
        total_num_pages,
        page_size // KV_PACKING,
        KV_PACKING,
        align_to(record_width, 128),
    )


def convert_cache_to_seq_along_lane(
    cache_kv: jax.Array,  # uint8[pages, page_size // 4, 4, width]
    actual_head_dim: int,
) -> jax.Array:  # uint8[pages, head_dim // 4 + 1, 4, page_size]
    """Repacks a HEAD_ALONG_SUBLANE index cache into the SEQ_ALONG_LANE layout."""
    total_num_pages, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    head_dim = align_to(actual_head_dim, 128)

    flat = cache_kv.reshape(total_num_pages, page_size, -1)
    keys = flat[:, :, :head_dim]
    scales = flat[:, :, head_dim : head_dim + 1]
    scales = jnp.pad(scales, ((0, 0), (0, 0), (0, KV_PACKING - 1)), constant_values=0)
    rows = jnp.concatenate([keys, scales], axis=-1)
    rows = rows.swapaxes(1, 2)
    return rows.reshape(
        total_num_pages,
        (head_dim + KV_PACKING) // KV_PACKING,
        KV_PACKING,
        page_size,
    )


def _effective_row_lengths(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    num_tokens: int,
    num_positions: int,
    compression_ratio: int,
    cp_rank: jax.Array | int = 0,
    cp_size: int = 1,
    interleave_c: int = 1,
    causal_offset: int = 0,
) -> jax.Array:  # i32[num_tokens]
    """Visible compressed KV positions per token row of the score matrix."""
    max_num_seqs = seq_lens.shape[0]
    num_seqs = distribution[2]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_mask = token_ids[:, None] >= cu_q_lens[None, 1 : max_num_seqs + 1]
    seq_mask = jnp.where(jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False)
    seq_ids = jnp.sum(seq_mask, axis=1)

    q_start = cu_q_lens[seq_ids]
    q_len = cu_q_lens[seq_ids + 1] - q_start
    seq_len = seq_lens[seq_ids]
    token_pos = seq_len - q_len + (token_ids - q_start)
    kv_len = seq_len // compression_ratio
    visible = jnp.minimum(kv_len, (token_pos - causal_offset) // compression_ratio + 1)
    visible = cp_local_length(visible, cp_rank, cp_size, interleave_c)
    valid = token_ids < cu_q_lens[num_seqs]
    return jnp.where(valid, jnp.minimum(visible, num_positions), 0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "buffer_count",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "enable_early_exit",
        "kv_layout",
        "chunk_tokens",
        "cp_size",
        "interleave_size",
        "return_scores",
        "causal_offset",
    ),
)
def streamindex_topk(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    indexer_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    cache_kv: jax.Array,
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    buffer_count: tuple[int, int, int] | int | None = DEFAULT_BUFFER_COUNT,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    chunk_tokens: int | None = None,
    cp_size: int = 1,
    cp_rank: jax.Array | int = 0,
    interleave_size: int = 1,
    return_scores: bool = False,
    causal_offset: int = 0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """StreamIndex Top-K retrieval.

    A query at position ``p`` sees compressed rows ``<= (p - causal_offset) //
    compression_ratio``. The default 0 lets a token see the row it falls in;
    ``compression_ratio - 1`` limits it to rows already closed (GLM-5.3-Flash's
    pooled indexer)."""
    kv_layout = KVLayout.parse(kv_layout)

    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if cp_size > 1:
        if interleave_size % compression_ratio != 0:
            raise ValueError(
                f"interleave_size ({interleave_size}) must be a multiple of "
                f"compression_ratio ({compression_ratio}) for the CP chunk "
                "boundary to fall on a compressed-row boundary."
            )
        if enable_early_exit:
            raise NotImplementedError(
                "enable_early_exit is not supported with cp_size > 1."
            )
    if enable_early_exit and causal_offset:
        raise NotImplementedError(
            "causal_offset is not supported with enable_early_exit."
        )
    if enable_early_exit and return_scores:
        raise NotImplementedError(
            "return_scores is not supported with enable_early_exit."
        )
    interleave_c = interleave_size // compression_ratio if cp_size > 1 else 1

    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )
    if compression_ratio < 1 or (compression_ratio & (compression_ratio - 1)) != 0:
        raise ValueError("compression_ratio must be a power of 2.")

    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    if buffer_count is None:
        buffer_counts = list(DEFAULT_BUFFER_COUNT)
    elif isinstance(buffer_count, int):
        buffer_counts = [buffer_count for _ in range(3)]
    else:
        buffer_counts = list(buffer_count)

    if len(buffer_counts) != 3:
        raise ValueError(
            f"buffer_count must be a 3-tuple or a single integer, got {buffer_count}"
        )

    if chunk_tokens is not None and chunk_tokens % decode_req_batch_size:
        raise ValueError(
            f"chunk_tokens ({chunk_tokens}) must be a multiple of"
            f" decode_req_batch_size ({decode_req_batch_size}), or a cut can"
            " fall inside a decode block and split it across two chunks."
        )

    max_num_seqs = seq_lens.shape[0]

    original_dtype = q.dtype
    actual_head_dim = q.shape[-1]

    prepared_indexer_weights = prepare_index_weights(indexer_weights, original_dtype)
    q = prepare_q_inputs(q)
    head_dim = q.shape[-1]

    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        total_num_pages, kv_sublane_groups, kv_packing, page_size = cache_kv.shape
        expected_shape = get_kv_cache_shape(
            total_num_pages, page_size, actual_head_dim, kv_layout
        )
        if cache_kv.shape != expected_shape:
            raise ValueError(
                "SEQ_ALONG_LANE expects cache_kv of shape"
                f" {expected_shape}, got {cache_kv.shape}."
            )
        if kv_sublane_groups * kv_packing != head_dim + kv_packing:
            raise ValueError(
                f"Inconsistent SEQ_ALONG_LANE cache: {kv_sublane_groups=},"
                f" {kv_packing=} does not match {head_dim=}."
            )
    else:
        _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
        page_size = page_size_per_kv_packing * kv_packing
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    for bkv_p in num_kv_pages_per_blocks:
        bkv_sz = page_size * bkv_p
        if bkv_sz % 128 != 0:
            raise ValueError(
                f"bkv_sz ({page_size} * {bkv_p} = {bkv_sz}) must be a multiple of 128."
            )

    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128
        for bkv_p in num_kv_pages_per_blocks
    )

    def _block_sizes(num_queries_per_block, bkv_p, static_q_len):
        """Query and KV block sizes a pass runs with."""
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz = page_size * bkv_p
        return bq_sz, bkv_sz

    def run_scores_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        scores_init,
        seq_lens,
        page_indices,
        cu_q_lens,
        start_seq_idx,
        end_seq_idx,
        static_q_len,
        num_kv_pages_per_block,
        num_queries_per_block,
        buffer_count,
        seq_batch_size,
        out_dtype,
        pass_meta,
        case=MlaCase.MIXED,
        chunk_token_start=None,
        chunk_tokens=None,
        scheduling_group_id=None,
        chunk_idx=0,
    ):
        max_num_tokens, num_q_heads, head_dim = q.shape
        if seq_batch_size > 1:
            assert static_q_len == 1

        chunked = chunk_token_start is not None
        if not chunked:
            chunk_tokens = max_num_tokens

        bkv_p = num_kv_pages_per_block
        bq_sz, bkv_sz = _block_sizes(num_queries_per_block, bkv_p, static_q_len)

        hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
        aliased = scores_init is not None
        in_specs = [
            hbm_spec,  # q
            hbm_spec,  # prepared_indexer_weights
            hbm_spec,  # cache_kv
        ]
        if aliased:
            in_specs.append(hbm_spec)  # scores_init, aliased to out
        out_specs = hbm_spec

        scratch_shapes = []

        chunk_start = 0 if chunk_token_start is None else chunk_token_start
        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx, chunk_start], jnp.int32),
            pass_meta,
            jnp.asarray([cp_rank], jnp.int32),
        )

        num_scalar_prefetch = len(scalar_prefetches)
        num_flat_prefetches = len(jax.tree_util.tree_leaves(scalar_prefetches))
        input_output_aliases = {num_flat_prefetches + 3: 0} if aliased else {}

        layout_tag = "" if kv_layout == KVLayout.HEAD_ALONG_SUBLANE else "-sal"
        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}{layout_tag}"
        if chunked:
            scope_name += f"-c{max_num_tokens // chunk_tokens}"
        pallas_kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    kernel,
                    compression_ratio=compression_ratio,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    seq_batch_size=seq_batch_size,
                    buffer_count=buffer_count,
                    case=case,
                    kv_layout=kv_layout,
                    cp_size=cp_size,
                    interleave_c=interleave_c,
                    chunk_idx=chunk_idx,
                    causal_offset=causal_offset,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=num_scalar_prefetch,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=(1,),
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=jax.ShapeDtypeStruct(
                    shape=(chunk_tokens, num_sublanes_total, 128),
                    dtype=out_dtype,
                ),
                input_output_aliases=input_output_aliases,
                name=scope_name,
            )
        )
        scores = pallas_kernel(
            *scalar_prefetches,
            q,
            prepared_indexer_weights,
            cache_kv,
            *((scores_init,) if aliased else ()),
        )
        if scheduling_group_id is not None:
            scores = xla_metadata.set_xla_metadata(
                scores, _scheduling_group_id=scheduling_group_id
            )
        return scores

    def _pass_specs(decode_batch_end):
        """`run_scores_kernel` kwargs for the three passes over the batch."""
        decode_cfg = dict(
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            buffer_count=buffer_counts[0],
            static_q_len=1,
            case=MlaCase.DECODE,
        )
        return (
            dict(
                start_seq_idx=jnp.array(0),
                end_seq_idx=decode_batch_end,
                seq_batch_size=decode_req_batch_size,
                **decode_cfg,
            ),
            dict(
                start_seq_idx=decode_batch_end,
                end_seq_idx=distribution[1],
                seq_batch_size=1,
                **decode_cfg,
            ),
            dict(
                num_kv_pages_per_block=num_kv_pages_per_blocks[2],
                num_queries_per_block=num_queries_per_blocks[2],
                buffer_count=buffer_counts[2],
                start_seq_idx=distribution[1],
                end_seq_idx=distribution[2],
                static_q_len=None,
                seq_batch_size=1,
                case=MlaCase.MIXED,
            ),
        )

    def _pass_metadata(spec, chunk_starts, chunk_tokens):
        """One schedule per pass, holding every chunk's steps back to back."""
        bq_sz, bkv_sz = _block_sizes(
            spec["num_queries_per_block"],
            spec["num_kv_pages_per_block"],
            spec["static_q_len"],
        )
        return metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=spec["start_seq_idx"],
            end_seq_idx=spec["end_seq_idx"],
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            pages_per_seq=pages_per_seq,
            compression_ratio=compression_ratio,
            static_q_len=spec["static_q_len"],
            seq_batch_size=spec["seq_batch_size"],
            page_size=page_size,
            max_num_tokens=q.shape[0],
            chunk_token_start=chunk_starts,
            chunk_tokens=chunk_tokens,
            cp_rank=cp_rank,
            cp_size=cp_size,
            interleave_c=interleave_c,
        )

    def _scores_for_chunk(
        passes,
        chunk_idx,
        chunk_token_start,
        chunk_tokens,
        out_dtype,
        scheduling_group_id,
    ):
        """Scores for one window of the flat token axis, in a private buffer."""
        scores = None
        for spec, pass_meta in passes:
            scores = run_scores_kernel(
                q,
                prepared_indexer_weights,
                cache_kv,
                scores,
                seq_lens,
                page_indices,
                cu_q_lens,
                pass_meta=pass_meta,
                chunk_idx=chunk_idx,
                chunk_token_start=chunk_token_start,
                chunk_tokens=chunk_tokens,
                out_dtype=out_dtype,
                scheduling_group_id=scheduling_group_id,
                **spec,
            )
        return scores

    def _common_path(_):
        out_dtype = jnp.int32
        max_num_tokens = q.shape[0]

        decode_batch_end = (
            distribution[0] // decode_req_batch_size * decode_req_batch_size
        )

        num_positions = max(num_sublanes_total * 128, k)
        eff_row_lengths = _effective_row_lengths(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            num_tokens=max_num_tokens,
            num_positions=num_positions,
            compression_ratio=compression_ratio,
            cp_rank=cp_rank,
            cp_size=cp_size,
            interleave_c=interleave_c,
            causal_offset=causal_offset,
        )

        if chunk_tokens is None or max_num_tokens % chunk_tokens:
            num_chunks, chunk_len = 1, max_num_tokens
        else:
            num_chunks, chunk_len = max_num_tokens // chunk_tokens, chunk_tokens

        group_base = 0
        if num_chunks > 1:
            group_base = config.reserve_scheduling_group_ids(num_chunks - 1)

        need_scores = cp_size > 1 or return_scores

        chunk_starts = (
            (None,)
            if num_chunks == 1
            else tuple(m * chunk_len for m in range(num_chunks))
        )
        passes = tuple(
            (spec, _pass_metadata(spec, chunk_starts, chunk_len))
            for spec in _pass_specs(decode_batch_end)
        )
        topk_idxs = []
        topk_scores_list = []
        for m in range(num_chunks):
            tc_group = None if m == 0 else group_base + m - 1
            sc_group = None if m == num_chunks - 1 else group_base + m
            sc2_group = None if m >= num_chunks - 2 else group_base + m + 1

            start = m * chunk_len
            scores = _scores_for_chunk(
                passes,
                m,
                chunk_starts[m],
                chunk_len,
                out_dtype,
                tc_group,
            )
            scores = scores.reshape(chunk_len, -1)
            if scores.shape[1] < k:
                scores = jnp.pad(
                    scores,
                    ((0, 0), (0, k - scores.shape[1])),
                    constant_values=BITS_NEG_INF,
                )

            chunk_topk_idxs, chunk_topk_scores = sparsecore_topk(
                scores,
                k,
                row_lengths=eff_row_lengths[start : start + chunk_len],
                write_empty_rows=True,
                scheduling_group_id=sc_group,
                stage2_scheduling_group_id=sc2_group,
                return_scores=True,
            )
            topk_idxs.append(chunk_topk_idxs)
            if need_scores:
                topk_scores_list.append(chunk_topk_scores)

        out = topk_idxs[0] if len(topk_idxs) == 1 else jnp.concatenate(topk_idxs)
        out = out[:max_num_tokens, :k]

        if cp_size == 1 and not return_scores:
            return out

        scores_out = (
            topk_scores_list[0]
            if len(topk_scores_list) == 1
            else jnp.concatenate(topk_scores_list)
        )
        scores_out = scores_out[:max_num_tokens, :k]

        filled = out >= 0
        global_idxs = jnp.where(
            filled,
            cp_local_to_global(out, cp_rank, cp_size, interleave_c),
            -1,
        )
        if not return_scores:
            return global_idxs
        return global_idxs, scores_out

    def _fast_path(_):
        token_idx = jnp.arange(q.shape[0])
        seq_idx = jnp.minimum(
            jnp.searchsorted(cu_q_lens[1:], token_idx, side="right"),
            seq_lens.shape[0] - 1,
        )
        seq_len = seq_lens[seq_idx]
        q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
        q_start = cu_q_lens[seq_idx]
        q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
        max_valid_idx = jnp.minimum(
            seq_len // compression_ratio - 1,
            q_abs_pos // compression_ratio,
        )
        max_valid_idx = jnp.where(token_idx < cu_q_lens[-1], max_valid_idx, -1)
        s_idx = jnp.arange(k, dtype=jnp.int32)[None, :]
        return jnp.where(s_idx <= max_valid_idx[:, None], s_idx, -1)

    if not enable_early_exit:
        return _common_path(None)
    return jax.lax.cond(
        jnp.max(seq_lens) // compression_ratio <= k,
        _fast_path,
        _common_path,
        None,
    )


def _select_owned_winners(
    merged: jax.Array,
    dcp_size: int,
    interleave_c: int,
    dcp_rank: jax.Array | int,
) -> jax.Array:
    """Keep the winners this rank owns, left-packed as rank-local indices."""
    num_rows, width = merged.shape
    mine = (merged >= 0) & (cp_owner_rank(merged, dcp_size, interleave_c) == dcp_rank)
    local = cp_global_to_local(jnp.maximum(merged, 0), dcp_size, interleave_c)

    slot = jnp.cumsum(mine, axis=1, dtype=jnp.int32) - mine
    rows = jnp.arange(num_rows, dtype=jnp.int32)[:, None]

    out = jnp.full((num_rows, width + 1), -1, dtype=jnp.int32)
    out = out.at[rows, jnp.where(mine, slot, width)].set(jnp.where(mine, local, -1))
    return out[:, :width]


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "k",
        "compression_ratio",
        "dcp_size",
        "interleave_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "buffer_count",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "chunk_tokens",
        "dcp_axis_name",
    ),
)
def streamindex_topk_dcp(
    q: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    mesh,
    k: int,
    compression_ratio: int,
    dcp_size: int,
    interleave_size: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    buffer_count: tuple[int, int, int] | int | None = DEFAULT_BUFFER_COUNT,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    chunk_tokens: int | None = None,
    dcp_axis_name: str = DCP_AXIS_NAME,
) -> jax.Array:
    """Exact global top-k over a DCP-sharded KV cache, delivered rank-local."""
    if dcp_size <= 1:
        raise ValueError(
            f"streamindex_topk_dcp requires dcp_size > 1, got {dcp_size}; "
            "use streamindex_topk for the unsharded case."
        )
    if dcp_axis_name not in mesh.axis_names:
        raise ValueError(f"mesh {mesh.axis_names} has no {dcp_axis_name!r} axis.")
    if mesh.shape[dcp_axis_name] != dcp_size:
        raise ValueError(
            f"dcp_size={dcp_size} does not match mesh axis"
            f" {dcp_axis_name!r}={mesh.shape[dcp_axis_name]}."
        )
    if q.shape[0] % dcp_size != 0:
        raise ValueError(
            f"padded_num_tokens={q.shape[0]} must be divisible by dcp_size={dcp_size}."
        )
    interleave_c = interleave_size // compression_ratio

    def _local(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        dcp_rank = cp_rank_as_data(dcp_axis_name, dcp_size)
        local_page_indices = jnp.mod(page_indices, jnp.int32(cache_kv.shape[0]))

        idxs, scores = streamindex_topk(
            q,
            indexer_weights,
            cache_kv,
            seq_lens,
            local_page_indices,
            cu_q_lens,
            distribution,
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            buffer_count=buffer_count,
            vmem_limit_bytes=vmem_limit_bytes,
            decode_req_batch_size=decode_req_batch_size,
            enable_early_exit=False,
            chunk_tokens=chunk_tokens,
            cp_size=dcp_size,
            cp_rank=dcp_rank,
            interleave_size=interleave_size,
            return_scores=True,
        )

        idxs = lax.all_gather(idxs, dcp_axis_name, axis=1, tiled=True)
        scores = lax.all_gather(scores, dcp_axis_name, axis=1, tiled=True)

        slots = sparsecore_topk(
            scores,
            k,
            row_lengths=jnp.full((scores.shape[0],), scores.shape[1], dtype=jnp.int32),
            write_empty_rows=True,
        )
        merged = jnp.take_along_axis(idxs, jnp.maximum(slots, 0), axis=1)
        merged = jnp.where(slots >= 0, merged, -1)

        return _select_owned_winners(merged, dcp_size, interleave_c, dcp_rank)

    replicated = P()
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            replicated,  # q
            replicated,  # indexer_weights
            P(dcp_axis_name),  # cache_kv
            replicated,  # seq_lens
            replicated,  # page_indices
            replicated,  # cu_q_lens
            replicated,  # distribution
        ),
        out_specs=P(dcp_axis_name),
        check_vma=False,
    )(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )
