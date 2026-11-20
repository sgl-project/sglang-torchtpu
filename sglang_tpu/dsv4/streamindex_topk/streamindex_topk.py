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

import dataclasses
import functools
import math
from typing import Any

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
    sparsecore_topk_dcp_merge,
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


@dataclasses.dataclass(frozen=True)
class _BlockedPrefillConfig:
    """Multi-token (prefill / mixed) scoring layout and tiling configuration.

    The head reduction runs over blocks of `tok_blk` tokens and `lane_chunk` KV
    positions. Heads are dealt round-robin into one group per MXU (head `h` goes
    to group `h % G`), and the cached queries are ordered `(head_group,
    token_block, head, token)`: each chunk's `q @ k` is one matmul per head
    group, which the compiler spreads over the MXUs, and every MXU emits its rows
    in the exact order the per-head accumulation consumes them. The
    lane-broadcast indexer weights are cached once per q block instead of being
    rebuilt every step.
    """

    enabled: bool
    tok_blk: int
    num_tok_blks: int
    num_head_groups: int
    heads_per_group: int
    rows_per_group_tok_blk: int
    rows_per_group: int
    lane_chunk: int
    num_lane_chunks: int
    tiles_per_chunk: int
    strided_score_store: bool
    score_pitch: int
    bf16_reduce: bool
    wb_dtype: Any

    @classmethod
    def create(
        cls,
        *,
        bq_sz: int,
        bkv_sz: int,
        num_q_heads: int,
        q_dtype: Any,
        seq_batch_size: int,
    ) -> "_BlockedPrefillConfig":
        """Computes the blocked-prefill layout parameters for a kernel pass."""
        tok_blk = math.gcd(bq_sz, 32)
        num_tok_blks = bq_sz // tok_blk
        num_head_groups = math.gcd(num_q_heads, pltpu.get_tpu_info().num_mxus)
        heads_per_group = num_q_heads // num_head_groups

        lane_chunk = math.gcd(bkv_sz, 256)
        num_lane_chunks = bkv_sz // lane_chunk
        tiles_per_chunk = lane_chunk // 128

        q_sublane_tiling = 8 * get_dtype_packing(q_dtype)
        enabled = seq_batch_size == 1 and bq_sz > 1 and tok_blk % q_sublane_tiling == 0

        # Scores are written straight into the [bq, nsub, 128] block with strided
        # stores when every token owns whole (8, 128) tiles of it.
        num_sublanes_bkv = bkv_sz // 128
        strided_score_store = num_sublanes_bkv % 8 == 0
        # Rows between consecutive tokens of the VMEM score block. The sublanes of
        # one store must fall in distinct VMEM banks (row mod 32), so with a pitch
        # of nsub rows, a multiple of 16 for the usual block sizes, every strided
        # store splits into 2-row pieces (1-row at nsub = 32), and a bundle holding
        # two stores cannot also pop the MXU. A pitch of 8 mod 16 rows lets each
        # store cover 4 rows; the padding rows are never copied out.
        score_pitch = num_sublanes_bkv
        if enabled and strided_score_store and score_pitch % 16 == 0:
            score_pitch += 8

        # Reduced-precision head reduction (used for FP8 queries):
        # relu(q @ k) is rounded to bf16 and the weights are held in bf16, so each
        # weighted term is an exact bf16 x bf16 product accumulated in f32. The
        # compiler lowers that pattern to one multiply-accumulate per packed bf16
        # register, which halves the VPU work of the reduction; see
        # `reduce_token_block`. It needs whole packed (16, 128) tiles per token
        # block.
        bf16_reduce = get_dtype_bitwidth(q_dtype) == 8 and tok_blk % 16 == 0
        wb_dtype = jnp.bfloat16 if bf16_reduce else jnp.float32

        return cls(
            enabled=enabled,
            tok_blk=tok_blk,
            num_tok_blks=num_tok_blks,
            num_head_groups=num_head_groups,
            heads_per_group=heads_per_group,
            rows_per_group_tok_blk=heads_per_group * tok_blk,
            rows_per_group=heads_per_group * bq_sz,
            lane_chunk=lane_chunk,
            num_lane_chunks=num_lane_chunks,
            tiles_per_chunk=tiles_per_chunk,
            strided_score_store=strided_score_store,
            score_pitch=score_pitch,
            bf16_reduce=bf16_reduce,
            wb_dtype=wb_dtype,
        )

    def cache_row(self, h: int, tb: int) -> int:
        """First `q_hm_ref` row of head `h` in token block `tb`."""
        g, hg = h % self.num_head_groups, h // self.num_head_groups
        return (
            g * self.rows_per_group
            + tb * self.rows_per_group_tok_blk
            + hg * self.tok_blk
        )


def kernel(
    *refs,
    page_table_in_hbm: bool,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
    buffer_count: int = 3,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    chunk_idx: int = 0,
):
    """Executes the n-buffered Pallas scoring pipeline for StreamIndex Top-K.

    Refs: scalar prefetch (seq_lens [max_num_seqs], page table when it is
    prefetched [max_num_seqs * pages_per_seq], cu_q_lens [max_num_seqs + 1],
    start/end seq idx + chunk token start [3], MetadataRef), then HBM inputs q
    [max_num_tokens, num_q_heads, head_dim], indexer weights
    [max_num_tokens, num_q_heads], cache_kv (HEAD_ALONG_SUBLANE [total_num_pages,
    page_size_per_kv_packing, kv_packing, lkv_dim] or SEQ_ALONG_LANE
    [total_num_pages, kv_sublane_groups, kv_packing, page_size]), the page table
    [max_num_seqs, pages_per_seq] when it stays in HBM, the HBM scores
    [chunk_tokens, num_sublanes_total, 128] as (out,) or (donated_in, out) after
    the first pass (only the last is ever read), and, with the table in HBM, an
    SMEM scratch holding one page-table row per sequence in flight.
    """
    if page_table_in_hbm:
        (
            seq_lens_ref,
            cu_q_lens_ref,
            start_end_seq_idx_ref,
            metadata_ref,
            q_hbm_ref,
            indexer_weights_hbm_ref,
            cache_kv_hbm_ref,
            page_table_ref,
            *score_refs,
            page_rows_ref,
        ) = refs
        pages_per_seq = page_table_ref.shape[1]
    else:
        (
            seq_lens_ref,
            page_table_ref,
            cu_q_lens_ref,
            start_end_seq_idx_ref,
            metadata_ref,
            q_hbm_ref,
            indexer_weights_hbm_ref,
            cache_kv_hbm_ref,
            *score_refs,
        ) = refs
        page_rows_ref = None
        pages_per_seq = page_table_ref.shape[0] // seq_lens_ref.shape[0]
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

    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_p * page_size
    num_sublanes_bkv = bkv_sz // 128

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

    prefill_cfg = _BlockedPrefillConfig.create(
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        num_q_heads=num_q_heads,
        q_dtype=q_hbm_ref.dtype,
        seq_batch_size=seq_batch_size,
    )

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
        (pl.BoundedSlice(seq_batch_size * bq_sz), prefill_cfg.score_pitch, 128),
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

    num_lane_tiles_bkv = bkv_sz // 128

    def seq_along_lane_block_rows(bkv_vmem):
        return bkv_vmem.bitcast(jnp.uint32).reshape(
            seq_batch_size, kv_sublane_groups * num_lane_tiles_bkv, 128
        )

    def seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx):
        """[head_dim, 128] FP8 keys of lane tile `lane_tile_idx`."""
        words = rows[
            batch_idx,
            pl.ds(lane_tile_idx, kv_head_dim_groups, stride=num_lane_tiles_bkv),
            :,
        ]
        return pltpu.bitcast(words, jnp.float8_e4m3fn)

    def seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx):
        """[1, 128] f32 dequantization scales of lane tile `lane_tile_idx`."""
        row = kv_head_dim_groups * num_lane_tiles_bkv + lane_tile_idx
        words = rows[batch_idx, pl.ds(row, 1), :]
        return pltpu.bitcast((words & 0xFF) << 23, jnp.float32)

    def transpose_to_head_major(bq_rows, n_tok):
        """[n_tok*H, D] head-minor rows -> [H*n_tok, D] head-major rows."""
        return bq_rows.transpose(1, 0, 2).reshape(num_q_heads * n_tok, head_dim)

    hm_tok = align_to(bq_sz, 8)

    def pad_tokens(bq_rows, n_tok):
        """[n_tok, H, D] -> [hm_tok, H, D], zero rows appended."""
        if n_tok == hm_tok:
            return bq_rows
        pad = jnp.zeros((hm_tok - n_tok, *bq_rows.shape[1:]), bq_rows.dtype)
        return jnp.concatenate([bq_rows, pad], axis=0)

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

        if head_major:
            rows_hm, rows_tok = bq_rows, hm_tok
            w = jnp.pad(w, ((0, rows_tok - n_tok), (0, 0)))
        else:
            rows_hm, rows_tok = transpose_to_head_major(bq_rows, n_tok), n_tok
        st = qk(rows_hm).reshape(num_q_heads, rows_tok, -1)
        acc = jnp.maximum(st[0], 0.0) * w[:, 0:1]
        for h in range(1, num_q_heads):
            relu_st = jnp.maximum(st[h], 0.0)
            acc = acc + relu_st * w[:, h : h + 1]
        return acc[:n_tok]

    def compute_scores(
        bq_vec,
        bkv_vec,
        scale_val_vec,
        bq_weights_vec,
        q_is_head_major=False,
    ):
        assert len(bq_vec) == seq_batch_size
        assert len(bkv_vec) == seq_batch_size
        assert len(scale_val_vec) == seq_batch_size
        assert len(bq_weights_vec) == seq_batch_size
        ret = []

        for batch_idx in range(seq_batch_size):
            bq = bq_vec[batch_idx].reshape(-1, head_dim)
            bkv = bkv_vec[batch_idx]
            scale_val = scale_val_vec[batch_idx]
            bq_weights = bq_weights_vec[batch_idx]

            s_summed = reduce_heads(bq, bkv, bq_weights, head_major=q_is_head_major)
            # Positions a token cannot see are left unmasked, as in
            # `blocked_prefill_step`: the top-k reads each row only up to its
            # `row_lengths` entry, the causal and end-of-sequence limit.
            s_summed = s_summed * scale_val
            ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))
        return jnp.concatenate(ret, axis=0)

    def step_body(q_vmem, weights_vmem, bkv_vmem, scores_vmem, q_hm_ref=None):
        bkv_idx = bkv_block(pl.program_id(0))

        if q_hm_ref is not None:

            @pl.when(bkv_idx == 0)
            def _():
                q_hm_ref[...] = transpose_to_head_major(
                    pad_tokens(q_vmem[...], bq_sz), hm_tok
                )

            bq_vec = [q_hm_ref[...]]
        elif bq_sz > 1:
            q_data = q_vmem[...].reshape(seq_batch_size, bq_sz, num_q_heads, head_dim)
            bq_vec = [
                transpose_to_head_major(pad_tokens(q_data[batch_idx], bq_sz), hm_tok)
                for batch_idx in range(seq_batch_size)
            ]
        else:
            q_data = q_vmem[...].reshape(seq_batch_size, -1, head_dim)
            bq_vec = [q_data[batch_idx] for batch_idx in range(seq_batch_size)]
        w_data = weights_vmem[...].reshape(seq_batch_size, -1, num_q_heads)
        bq_weights_vec = [w_data[batch_idx] for batch_idx in range(seq_batch_size)]

        bkvs = []
        bkv_scales = []
        for batch_idx in range(seq_batch_size):
            if seq_along_lane:
                rows = seq_along_lane_block_rows(bkv_vmem)
                lane_tiles = range(num_lane_tiles_bkv)
                fp8_val = jnp.concatenate(
                    [
                        seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx)
                        for lane_tile_idx in lane_tiles
                    ],
                    axis=1,
                )
                scale_val = jnp.concatenate(
                    [
                        seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx)
                        for lane_tile_idx in lane_tiles
                    ],
                    axis=1,
                )
                bkvs.append(fp8_val)
                bkv_scales.append(scale_val)
            else:
                bkv = bkv_vmem[batch_idx, :bkv_sz_per_kv_packing][...]
                flat_bkv = bkv.reshape(-1, bkv.shape[-1])
                fp8_val = pltpu.bitcast(flat_bkv[:, :head_dim], jnp.float8_e4m3fn)
                scale_val = pltpu.bitcast(
                    flat_bkv[:, head_dim : head_dim + 1].T, jnp.float8_e8m0fnu
                ).astype(jnp.float32)
                bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
                bkv_scales.append(scale_val)

        scores = compute_scores(
            bq_vec,
            bkvs,
            bkv_scales,
            bq_weights_vec,
            q_is_head_major=q_hm_ref is not None or bq_sz > 1,
        )
        scores_vmem[...] = pltpu.bitcast(scores, jnp.int32)

    def blocked_prefill_step(
        q_vmem, weights_vmem, bkv_vmem, scores_vmem, q_hm_ref, wb_ref
    ):
        """Scores one (q block, kv block) step of the multi-token path."""
        bkv_idx = bkv_block(pl.program_id(0))
        tok_blk = prefill_cfg.tok_blk
        num_tok_blks = prefill_cfg.num_tok_blks
        num_head_groups = prefill_cfg.num_head_groups
        heads_per_group = prefill_cfg.heads_per_group
        rows_per_group_tok_blk = prefill_cfg.rows_per_group_tok_blk
        rows_per_group = prefill_cfg.rows_per_group
        lane_chunk = prefill_cfg.lane_chunk
        num_lane_chunks = prefill_cfg.num_lane_chunks
        tiles_per_chunk = prefill_cfg.tiles_per_chunk
        score_pitch = prefill_cfg.score_pitch

        @pl.when(bkv_idx == 0)
        def _():
            w = weights_vmem[...].astype(jnp.float32)
            for tb in range(num_tok_blks):
                t0 = tb * tok_blk
                q_rows = transpose_to_head_major(q_vmem[pl.ds(t0, tok_blk)], tok_blk)
                for h in range(num_q_heads):
                    q_hm_ref[pl.ds(prefill_cfg.cache_row(h, tb), tok_blk)] = q_rows[
                        h * tok_blk : (h + 1) * tok_blk
                    ]
                    wb_ref[tb, h] = jnp.broadcast_to(
                        w[t0 : t0 + tok_blk, h : h + 1], (tok_blk, 128)
                    ).astype(wb_ref.dtype)

        if seq_along_lane:
            rows = seq_along_lane_block_rows(bkv_vmem)

            def load_chunk(c):
                tiles = range(c * tiles_per_chunk, (c + 1) * tiles_per_chunk)
                keys = jnp.concatenate(
                    [
                        seq_along_lane_key_tile(rows, 0, lane_tile_idx)
                        for lane_tile_idx in tiles
                    ],
                    1,
                )
                scales = jnp.concatenate(
                    [
                        seq_along_lane_scale_tile(rows, 0, lane_tile_idx)
                        for lane_tile_idx in tiles
                    ],
                    1,
                )
                return keys, scales

            def qk(q_rows, keys):
                return jnp.einsum(
                    "nd,dm->nm", q_rows, keys, preferred_element_type=jnp.float32
                )

        else:
            groups_per_chunk = lane_chunk // kv_packing

            def load_chunk(c):
                blk = bkv_vmem[0, pl.ds(c * groups_per_chunk, groups_per_chunk)]
                flat = blk.reshape(lane_chunk, blk.shape[-1])
                keys = pltpu.bitcast(flat[:, :head_dim], jnp.float8_e4m3fn)
                scales = pltpu.bitcast(
                    flat[:, head_dim : head_dim + 1].T, jnp.float8_e8m0fnu
                ).astype(jnp.float32)
                return keys, scales

            def qk(q_rows, keys):
                return jnp.einsum(
                    "nd,md->nm", q_rows, keys, preferred_element_type=jnp.float32
                )

        def reduce_token_block(sts, tb):
            acc = None
            for i in range(heads_per_group):
                for g, st in enumerate(sts):
                    r0 = tb * rows_per_group_tok_blk + i * tok_blk
                    st_h = st[r0 : r0 + tok_blk]
                    w_h = wb_ref[tb, i * num_head_groups + g]
                    if tiles_per_chunk > 1:
                        w_h = jnp.concatenate([w_h] * tiles_per_chunk, axis=1)
                    if prefill_cfg.bf16_reduce:
                        r = jnp.maximum(st_h.astype(jnp.bfloat16), 0)
                        term = r.astype(jnp.float32) * w_h.astype(jnp.float32)
                    else:
                        term = jnp.maximum(st_h, 0.0) * w_h
                    acc = term if acc is None else acc + term
            assert acc is not None
            return acc

        scores_rows = (
            scores_vmem.reshape(bq_sz * score_pitch, 128)
            if prefill_cfg.strided_score_store
            else None
        )
        chunk_scores = []
        for c in range(num_lane_chunks):
            keys, scales = load_chunk(c)
            sts = [
                qk(q_hm_ref[pl.ds(g * rows_per_group, rows_per_group)], keys)
                for g in range(num_head_groups)
            ]
            tb_scores = []
            for tb in range(num_tok_blks):
                acc = reduce_token_block(sts, tb) * scales
                if scores_rows is None:
                    tb_scores.append(acc)
                    continue
                for jj in range(tiles_per_chunk):
                    lane_tile = c * tiles_per_chunk + jj
                    scores_rows[
                        pl.ds(
                            tb * tok_blk * score_pitch + lane_tile,
                            tok_blk,
                            stride=score_pitch,
                        )
                    ] = pltpu.bitcast(acc[:, jj * 128 : (jj + 1) * 128], jnp.int32)
            if scores_rows is None:
                chunk_scores.append(jnp.concatenate(tb_scores, axis=0))

        if scores_rows is None:
            s_summed = jnp.concatenate(chunk_scores, axis=1)
            scores_vmem[...] = pltpu.bitcast(
                s_summed.reshape(bq_sz, num_sublanes_bkv, 128), jnp.int32
            )

    def _run_all(q_hm_ref=None, wb_ref=None):
        if prefill_cfg.enabled:
            body = functools.partial(
                blocked_prefill_step, q_hm_ref=q_hm_ref, wb_ref=wb_ref
            )
        else:
            body = functools.partial(step_body, q_hm_ref=q_hm_ref)

        @pl.with_scoped(final_allocs=(q_alloc, weights_alloc, kv_alloc, o_alloc))
        def _run_pipeline(final_allocs):
            q_buf, weights_buf, kv_buf, o_buf = final_allocs
            pipeline_fn = pltpu.emit_pipeline(
                body,
                grid=(metadata_ref.num_steps[chunk_idx],),
                in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
                out_specs=o_buf.spec,
            )
            window_refs = (cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref)
            pipeline_fn(
                (q_hbm_ref, *window_refs),
                (indexer_weights_hbm_ref, *window_refs),
                (cache_kv_hbm_ref, page_table_ref, metadata_ref, page_rows_ref),
                (scores_hbm_ref, *window_refs),
                allocations=(q_buf, weights_buf, kv_buf, o_buf),
            )

        _run_pipeline()

    if prefill_cfg.enabled:
        pl.run_scoped(
            _run_all,
            pltpu.VMEM((num_q_heads * bq_sz, head_dim), q_hbm_ref.dtype),
            pltpu.VMEM(
                (
                    prefill_cfg.num_tok_blks,
                    num_q_heads,
                    prefill_cfg.tok_blk,
                    128,
                ),
                prefill_cfg.wb_dtype,
            ),
        )
    elif seq_batch_size == 1 and bq_sz > 1:
        pl.run_scoped(
            _run_all,
            pltpu.VMEM((num_q_heads * hm_tok, head_dim), q_hbm_ref.dtype),
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
    if actual_head_dim % 64 != 0:
        unpacked = actual_head_dim - max(1, actual_head_dim // 128)
        if unpacked > 0 and unpacked % 64 == 0:
            actual_head_dim = unpacked
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
    causal_offset: int | None = None,
) -> jax.Array:  # i32[num_tokens]
    """Visible compressed KV positions per token row of the score matrix.

    Compressed position ``s`` covers raw tokens ``[s*r, (s+1)*r)``. A query at
    position ``token_pos`` sees only complete groups ``(s+1)*r <= token_pos + 1``
    unless overridden by ``causal_offset``.
    """
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
    if causal_offset is None:
        visible = jnp.minimum(kv_len, (token_pos + 1) // compression_ratio)
    else:
        visible = jnp.minimum(
            kv_len, (token_pos - causal_offset) // compression_ratio + 1
        )
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
        "decode_query_size",
        "uniform_q_len",
        "enable_early_exit",
        "kv_layout",
        "chunk_tokens",
        "overlap_chunks",
        "cp_size",
        "interleave_size",
        "return_scores",
        "causal_offset",
        "segments",
        "row_count",
        "page_table_in_hbm",
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
    decode_query_size: int = 1,
    uniform_q_len: int | None = None,
    enable_early_exit: bool = False,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    chunk_tokens: int | None = None,
    overlap_chunks: bool = True,
    cp_size: int = 1,
    cp_rank: jax.Array | int = 0,
    interleave_size: int = 1,
    return_scores: bool = False,
    causal_offset: int | None = None,
    segments: str = "all",
    row_start: jax.Array | None = None,
    row_count: int | None = None,
    page_table_in_hbm: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """StreamIndex Top-K retrieval.

    ``page_table_in_hbm`` keeps the page table in HBM and copies one sequence's
    row into SMEM at its first KV block. Off, the table is a scalar-prefetch
    operand, max_num_seqs x pages_per_seq int32 in SMEM for the whole program,
    which passes the 1 MiB scoped SMEM near a 1M context.

    ``decode_query_size`` is the number of query rows of every decode-range
    sequence (fixed-width speculative verify). A batch whose decode rows are
    not all that wide runs entirely on the mixed pass.

    ``uniform_q_len`` is the fixed query length of every sequence in
    ``[distribution[0], distribution[1])`` (e.g. a speculative verify segment
    behind true 1-token decodes). When ``None``, ``[0, distribution[1])`` are
    all treated as decode sequences and the uniform-prefill pass is omitted.

    A query at position ``p`` sees compressed rows ``< (p + 1) //
    compression_ratio`` by default (only closed groups), or ``<= (p -
    causal_offset) // compression_ratio`` when ``causal_offset`` is given.

    ``chunk_tokens`` scores the queries in row chunks of that size. With
    ``overlap_chunks`` the chunks carry XLA scheduling groups that overlap one
    chunk's scores with the previous chunk's top-k; without, they run in order.

    ``segments`` names the passes ``distribution`` can populate, as for the
    MLA kernel: ``"decode"`` promises ``distribution[1] == distribution[2]``
    (a decode forward) and drops the mixed pass; ``"mixed"`` promises
    ``distribution[1] == 0`` (an extend forward) and drops the decode passes.

    ``row_count`` (static) with ``row_start`` (traced i32 scalar) scores and
    selects only the query rows ``[row_start, row_start + row_count)`` of the
    flat token axis and returns ``i32[row_count, k]``: the DSA indexer's
    token split, where each TP rank picks the rows of its own token slice.
    Every row's causal range still comes from the full-batch metadata, and the
    SparseCore top-k takes the same branch as an unsplit call (it is chosen
    from the longest row of the whole batch), so each row equals the same row
    of the unsplit result. The window may run past ``max_num_tokens``; rows
    past it come back as ``-1``. ``row_count`` must be a multiple of
    ``decode_req_batch_size * decode_query_size`` and is incompatible with
    ``chunk_tokens``, ``cp_size > 1``, ``return_scores`` and early exit.
    """
    kv_layout = KVLayout.parse(kv_layout)
    if row_count is not None:
        if row_start is None:
            raise ValueError("row_count needs row_start.")
        if (
            chunk_tokens is not None
            or cp_size > 1
            or return_scores
            or enable_early_exit
        ):
            raise NotImplementedError(
                "row_count is not supported with chunk_tokens, cp_size > 1, "
                "return_scores or enable_early_exit."
            )
        if row_count % (decode_req_batch_size * decode_query_size):
            raise ValueError(
                f"row_count ({row_count}) must be a multiple of"
                f" decode_req_batch_size * decode_query_size"
                f" ({decode_req_batch_size * decode_query_size})."
            )
    if decode_query_size > 1:
        if enable_early_exit or cp_size > 1:
            raise NotImplementedError(
                "decode_query_size > 1 is not supported with enable_early_exit "
                "or cp_size > 1."
            )
        is_decode = jnp.arange(seq_lens.shape[0]) < distribution[1]
        uniform = jnp.all(~is_decode | (jnp.diff(cu_q_lens) == decode_query_size))
        distribution = jnp.where(
            uniform, distribution, jnp.array([0, 0, 1], jnp.int32) * distribution[2]
        )

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
    if segments not in ("all", "decode", "mixed"):
        raise ValueError(f"segments must be all, decode or mixed, got {segments!r}")
    if (
        enable_early_exit
        and causal_offset is not None
        and causal_offset != compression_ratio - 1
    ):
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

    decode_block_tokens = decode_req_batch_size * decode_query_size
    if chunk_tokens is not None and chunk_tokens % decode_block_tokens:
        raise ValueError(
            f"chunk_tokens ({chunk_tokens}) must be a multiple of"
            f" decode_req_batch_size * decode_query_size ({decode_block_tokens}),"
            " or a cut can fall inside a decode block and split it across two"
            " chunks."
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
            assert static_q_len is not None

        chunked = chunk_token_start is not None
        if not chunked:
            chunk_tokens = max_num_tokens

        bkv_p = num_kv_pages_per_block
        bq_sz, bkv_sz = _block_sizes(num_queries_per_block, bkv_p, static_q_len)

        hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
        aliased = scores_init is not None
        # q, prepared_indexer_weights, cache_kv, [page table], [scores_init]
        in_specs = [hbm_spec] * (4 if page_table_in_hbm else 3)
        if aliased:
            in_specs.append(hbm_spec)  # scores_init, aliased to out
        out_specs = hbm_spec

        chunk_start = 0 if chunk_token_start is None else chunk_token_start
        scalar_prefetches = (
            seq_lens,
            *(() if page_table_in_hbm else (page_indices,)),
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx, chunk_start], jnp.int32),
            pass_meta,
        )
        if page_table_in_hbm:
            hbm_inputs = (page_indices.reshape(max_num_seqs, pages_per_seq),)
            # One page-table row per sequence in flight.
            scratch_shapes = [pltpu.SMEM((seq_batch_size, pages_per_seq), jnp.int32)]
        else:
            hbm_inputs = ()
            scratch_shapes = []

        num_scalar_prefetch = len(scalar_prefetches)
        num_flat_prefetches = len(jax.tree_util.tree_leaves(scalar_prefetches))
        input_output_aliases = (
            {num_flat_prefetches + len(in_specs) - 1: 0} if aliased else {}
        )

        layout_tag = "" if kv_layout == KVLayout.HEAD_ALONG_SUBLANE else "-sal"
        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}{layout_tag}"
        if chunked:
            scope_name += f"-c{max_num_tokens // chunk_tokens}"
        pallas_kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    kernel,
                    page_table_in_hbm=page_table_in_hbm,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    seq_batch_size=seq_batch_size,
                    buffer_count=buffer_count,
                    kv_layout=kv_layout,
                    chunk_idx=chunk_idx,
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
            *hbm_inputs,
            *((scores_init,) if aliased else ()),
        )
        if scheduling_group_id is not None:
            scores = xla_metadata.set_xla_metadata(
                scores, _scheduling_group_id=scheduling_group_id
            )
        return scores

    def _pass_specs(decode_batch_end):
        """`run_scores_kernel` kwargs for the passes over the batch."""
        decode_cfg = dict(
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=(
                decode_query_size
                if decode_query_size > 1
                else num_queries_per_blocks[0]
            ),
            buffer_count=buffer_counts[0],
            static_q_len=decode_query_size,
            case=MlaCase.DECODE,
        )
        uniform = (
            (
                dict(
                    num_kv_pages_per_block=num_kv_pages_per_blocks[1],
                    num_queries_per_block=num_queries_per_blocks[1],
                    buffer_count=buffer_counts[1],
                    start_seq_idx=distribution[0],
                    end_seq_idx=distribution[1],
                    static_q_len=uniform_q_len,
                    seq_batch_size=1,
                    case=MlaCase.PREFILL,
                ),
            )
            if uniform_q_len is not None
            else ()
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
                end_seq_idx=distribution[0 if uniform else 1],
                seq_batch_size=1,
                **decode_cfg,
            ),
            *uniform,
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

        # A decode graph whose rows come in whole groups of
        # decode_req_batch_size can score the last, partly padded group on the
        # batched pass too: a padded row is a one-token sequence on valid pages
        # (as the backend builds it), and its scores are never read, because
        # its row length is 0. The tail pass is then always empty, and it is
        # not built: as an empty pass it still cost a launch and its first KV
        # prefetch (1.5 us per indexer layer at bs4 on TPU7x).
        batched_only = (
            segments == "decode"
            and decode_query_size == 1
            and uniform_q_len is None
            and max_num_seqs >= decode_req_batch_size
            and max_num_seqs % decode_req_batch_size == 0
        )
        if batched_only:
            decode_batch_end = (
                (distribution[0] + decode_req_batch_size - 1)
                // decode_req_batch_size
                * decode_req_batch_size
            )
        else:
            decode_batch_end = (
                distribution[0] // decode_req_batch_size * decode_req_batch_size
            )

        specs = _pass_specs(decode_batch_end)
        if batched_only:
            specs = specs[:1]
        else:
            if max_num_seqs < decode_req_batch_size:
                # The batched decode pass covers whole groups of
                # decode_req_batch_size sequences, so with fewer it never has
                # work. It would still prefetch its first KV blocks.
                specs = specs[1:]
            if segments == "decode":
                specs = specs[:-1]
            elif segments == "mixed":
                specs = specs[-1:]

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

        windowed = row_count is not None
        if windowed:
            num_chunks, chunk_len = 1, row_count
        elif chunk_tokens is None or max_num_tokens % chunk_tokens:
            num_chunks, chunk_len = 1, max_num_tokens
        else:
            num_chunks, chunk_len = max_num_tokens // chunk_tokens, chunk_tokens

        overlap = overlap_chunks and num_chunks > 1
        group_base = (
            config.reserve_scheduling_group_ids(num_chunks - 1) if overlap else 0
        )

        need_scores = cp_size > 1 or return_scores

        if windowed:
            # One chunk at a run-time start: the kernel and its schedule read
            # the chunk start as data (scalar prefetch), so one compiled
            # program serves every TP rank.
            window_start = jnp.asarray(row_start, jnp.int32).reshape(())
            chunk_starts = (window_start,)
            # Row lengths of the window, from the whole batch's metadata; rows
            # past max_num_tokens get length 0 (written as -1). A dynamic
            # slice of the zero-padded lengths, not an XLA gather: the server
            # runs with xla_tpu_offload_gather_to_sparsecore=true, and an
            # offloaded gather here would hand the SparseCore top-k wrong row
            # lengths (rows with too few, or no, valid picks). The start is
            # clamped to [0, max_num_tokens], so it never wraps.
            padded_row_lengths = jnp.pad(eff_row_lengths, (0, row_count))
            window_row_lengths = lax.dynamic_slice_in_dim(
                padded_row_lengths,
                jnp.clip(window_start, 0, max_num_tokens),
                row_count,
            )
            # The SparseCore top-k picks a width branch from the longest row;
            # take it over the whole batch, as an unsplit call does.
            branch_max_len = jnp.max(eff_row_lengths)
        else:
            chunk_starts = (
                (None,)
                if num_chunks == 1
                else tuple(m * chunk_len for m in range(num_chunks))
            )
            branch_max_len = None
        passes = tuple(
            (spec, _pass_metadata(spec, chunk_starts, chunk_len)) for spec in specs
        )
        topk_idxs = []
        topk_scores_list = []
        for m in range(num_chunks):
            tc_group = None if not overlap or m == 0 else group_base + m - 1
            sc_group = None if not overlap or m == num_chunks - 1 else group_base + m
            sc2_group = (
                None if not overlap or m >= num_chunks - 2 else group_base + m + 1
            )

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

            row_lengths = (
                window_row_lengths
                if windowed
                else eff_row_lengths[start : start + chunk_len]
            )
            if need_scores:
                chunk_topk_idxs, chunk_topk_scores = sparsecore_topk(
                    scores,
                    k,
                    row_lengths=row_lengths,
                    write_empty_rows=True,
                    scheduling_group_id=sc_group,
                    stage2_scheduling_group_id=sc2_group,
                    return_scores=True,
                    branch_max_len=branch_max_len,
                )
                topk_scores_list.append(chunk_topk_scores)
            else:
                chunk_topk_idxs = sparsecore_topk(
                    scores,
                    k,
                    row_lengths=row_lengths,
                    write_empty_rows=True,
                    scheduling_group_id=sc_group,
                    stage2_scheduling_group_id=sc2_group,
                    return_scores=False,
                    branch_max_len=branch_max_len,
                )
            topk_idxs.append(chunk_topk_idxs)

        out = topk_idxs[0] if len(topk_idxs) == 1 else jnp.concatenate(topk_idxs)
        out = out[: chunk_len if windowed else max_num_tokens, :k]

        if not return_scores:
            return out

        scores_out = (
            topk_scores_list[0]
            if len(topk_scores_list) == 1
            else jnp.concatenate(topk_scores_list)
        )
        scores_out = scores_out[:max_num_tokens, :k]
        return out, scores_out

    def _fast_path(_):
        row_lengths = _effective_row_lengths(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            num_tokens=q.shape[0],
            num_positions=k,
            compression_ratio=compression_ratio,
            causal_offset=causal_offset,
        )
        s_idx = jnp.arange(k, dtype=jnp.int32)[None, :]
        return jnp.where(s_idx < row_lengths[:, None], s_idx, -1)

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
        "kv_layout",
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
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
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
            kv_layout=kv_layout,
            chunk_tokens=chunk_tokens,
            cp_size=dcp_size,
            cp_rank=dcp_rank,
            interleave_size=interleave_size,
            return_scores=True,
        )

        scores = lax.all_gather(scores, dcp_axis_name, axis=1, tiled=True)

        return sparsecore_topk_dcp_merge(
            scores,
            idxs,
            k,
            dcp_rank=dcp_rank,
            dcp_size=dcp_size,
        )

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
