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
"""Wrapper for RPA kernel to match expected interface.

Select this kernel with `--attention-backend pallas_tpu_rpa_batched`. It reads
the token-major KV cache by default; set SGLANG_TPU_USE_SEQ_ALONG_LANE_KV=1 for
the seq-along-lane layout, which also switches the KV pool that allocates it.

Compared to the default RPA kernel, this kernel does the following:

1. Batches multiple sequences together to replace per-request flash_attention loops.

2. Enables triple-buffering via Pallas emit_pipeline

3. Precomputes expensive metadata upfront (e.g., page locations and bounds clipping) via
scheduler.py kernel. Kernel is calculated once and ammortized across different layers in a model.

Note: batched_rpa is build on top / derived from RPA3.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang.srt.environ import envs
from sglang_tpu.attention.pallas.batched_rpa import (
    configs,
    kernel,
    schedule,
    utils,
)


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int = 128,
) -> tuple[jax.Array, jax.Array]:

    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    _, actual_num_kv_heads, _ = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = utils.get_dtype_packing(q_dtype)
    kv_packing = utils.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head, q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_q_head_dim = utils.align_to(actual_head_dim, num_lanes)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        aligned_kv_head_dim = utils.align_to(actual_head_dim, num_sublanes * kv_packing)
    else:
        aligned_kv_head_dim = utils.align_to(actual_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    o_hbm_alias_q_hbm = (
        jnp.pad(
            q.reshape(
                total_q_tokens,
                actual_num_kv_heads,
                num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
                (0, aligned_q_head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            total_q_tokens,
            actual_num_kv_heads,
            aligned_num_q_heads_per_kv_head // q_packing,
            q_packing,
            aligned_q_head_dim,
        )
        .swapaxes(0, 1)
    )

    # Pad keys and values head_dim
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2_aligned = utils.align_to(actual_num_kv_heads_x2, kv_packing)

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_lanes = pltpu.get_tpu_info().num_lanes
        align_tokens = max(num_lanes, page_size)
        padded_total_tokens = utils.align_to(total_q_tokens, align_tokens)
        new_kv_hbm = (
            jnp.pad(
                jnp.concatenate([k, v], axis=-1).reshape(
                    total_q_tokens, actual_num_kv_heads_x2, actual_head_dim
                ),
                (
                    (0, padded_total_tokens - total_q_tokens),
                    (0, 0),
                    (0, aligned_kv_head_dim - actual_head_dim),
                ),
                constant_values=0,
            )
            .reshape(
                padded_total_tokens,
                actual_num_kv_heads_x2,
                aligned_kv_head_dim // kv_packing,
                kv_packing,
            )
            .transpose(1, 2, 3, 0)
        )
    else:
        new_kv_hbm = jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(
                total_q_tokens, actual_num_kv_heads_x2, actual_head_dim
            ),
            (
                (0, 0),
                (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
                (0, aligned_kv_head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            total_q_tokens,
            num_kv_heads_x2_aligned // kv_packing,
            kv_packing,
            aligned_kv_head_dim,
        )
    return o_hbm_alias_q_hbm, new_kv_hbm


def prepare_attention_sink(
    attention_sink: jax.Array, num_kv_heads: int, q_dtype: jnp.dtype
) -> jax.Array:
    """Lays out per-q-head sink logits as rows of the kernel's l/m scratch.

    Rows of a KV head are token-major with the aligned q heads of a KV head
    inside each token, so the table repeats them over whole f32 sublane tiles:
    [num_kv_heads, period, num_lanes], period a multiple of the aligned count.
    """
    num_q_heads_per_kv_head = attention_sink.shape[0] // num_kv_heads
    aligned_num_q_heads_per_kv_head = utils.align_to(
        num_q_heads_per_kv_head, utils.get_dtype_packing(q_dtype)
    )
    tpu_info = pltpu.get_tpu_info()
    period = math.lcm(aligned_num_q_heads_per_kv_head, tpu_info.num_sublanes)

    sink = attention_sink.astype(jnp.float32).reshape(
        num_kv_heads, num_q_heads_per_kv_head
    )
    # Padded q heads get no sink weight; finite so that sink - m never sees -inf.
    sink = jnp.pad(
        sink,
        ((0, 0), (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head)),
        constant_values=-1e30,
    )
    sink = jnp.tile(sink, (1, period // aligned_num_q_heads_per_kv_head))
    return jnp.broadcast_to(
        sink[:, :, None], (num_kv_heads, period, tpu_info.num_lanes)
    )


def prepare_outputs(out: jax.Array) -> jax.Array:
    kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype=None,
    kv_packing: int | None = None,
    kv_layout: configs.KVLayout | None = None,
):
    if kv_layout is None:
        kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes

    if kv_packing is None:
        kv_packing = utils.get_dtype_packing(kv_dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        return (
            total_num_pages,
            actual_num_kv_heads * 2,
            utils.align_to(actual_head_dim, num_sublanes * kv_packing) // kv_packing,
            kv_packing,
            page_size,
        )
    return (
        total_num_pages,
        page_size,
        utils.align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        utils.align_to(actual_head_dim, num_lanes),
    )


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill."""

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    mxu_column_size = tpu_info.mxu_column_size

    # Calculate aligned model dimensions.
    aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)
    aligned_num_q_heads_per_kv_head = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
    )
    aligned_num_q_heads = aligned_num_q_heads_per_kv_head * model_cfgs.num_kv_heads

    bkv_stride = pl.cdiv(model_cfgs.num_kv_heads * 2, serve_cfgs.packing_kv)
    if utils.has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

    q_bytes = jnp.dtype(serve_cfgs.dtype_q).itemsize
    kv_bytes = jnp.dtype(serve_cfgs.dtype_kv).itemsize
    out_bytes = jnp.dtype(serve_cfgs.dtype_out).itemsize

    def calculate_vmem_usage(
        batch_size: int, n_buffer: int, bq_sz: int, bkv_sz: int
    ) -> int:
        """Given tile size, calculate VMEM usage of the kernel."""

        # Step 1: Calculate buffer sizes.

        # Calculate size bq & bkv arrays for a single buffer.
        bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            bkv_array_size = (
                (bkv_sz + 2 * serve_cfgs.page_size)
                * aligned_num_kv_heads_x2
                * aligned_head_dim
            )
        else:
            bkv_array_size = bkv_sz * aligned_num_kv_heads_x2 * aligned_head_dim

        # Get output buffer size as well - which has same size as query size.
        bo_array_size = bq_array_size

        # Convert to bytes.
        bq_bytes = bq_array_size * q_bytes
        bkv_bytes = bkv_array_size * kv_bytes
        bo_bytes = bo_array_size * out_bytes

        # Account for multiple buffers. For output, we always use double buffer.
        bq_bytes *= n_buffer
        bkv_bytes *= n_buffer
        bo_bytes *= 2

        # Sum up all buffer memory usage.
        buffer_bytes = bq_bytes + bkv_bytes + bo_bytes
        if serve_cfgs.return_lse:
            # f32 rows of lanes: two output buffers and the packed LSE math per
            # (KV head, token), and the per-(token, q head) staging; plus m per
            # lane. Packed-array counts fit to gpt-oss DCP2/DCP8 compiles; fewer
            # let the DCP8 chunk pass (8 q over 8 KV heads, sinks) overflow VMEM.
            num_kv_heads = model_cfgs.num_kv_heads
            num_packed = 8 if model_cfgs.has_attention_sink else 3
            lse_rows = (2 + num_packed) * num_kv_heads + aligned_num_q_heads_per_kv_head
            lse_bytes = lse_rows * num_lanes * 4
            lse_bytes += (
                num_kv_heads * aligned_num_q_heads_per_kv_head * num_lanes * out_bytes
            )
            buffer_bytes += lse_bytes * bq_sz

        # Step 2: Calculate worst case memory usage during computation.

        # Calculate the size of loaded bq and bkv size.
        loaded_bq_size = bq_sz * model_cfgs.num_q_heads * aligned_head_dim
        loaded_bkv_size = bkv_sz * model_cfgs.num_kv_heads * aligned_head_dim

        # Calculate peak memory requirement of output - which is attention weight.
        qk_size = bq_sz * bkv_sz * model_cfgs.num_q_heads

        # Convert to bytes.
        loaded_bq_bytes = loaded_bq_size * q_bytes
        loaded_bkv_bytes = loaded_bkv_size * kv_bytes
        qk_bytes = qk_size * out_bytes

        # Sum up all compute memory usage.
        compute_bytes = loaded_bq_bytes + loaded_bkv_bytes + qk_bytes

        # Step 3: Sum up all memory usage.
        total_bytes = buffer_bytes + compute_bytes

        # Account for batch size.
        total_bytes *= batch_size

        return total_bytes

    def calculate_compute_buffer_time(
        batch_size: int, bq_c_sz: int, bkv_sz: int
    ) -> int:
        """Calculate computational complexity of a single compute block."""

        num_k_rows = pl.cdiv(bkv_sz, mxu_column_size)
        num_k_cols = pl.cdiv(model_cfgs.head_dim, mxu_column_size)
        num_k = num_k_rows * num_k_cols
        num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads

        return batch_size * num_muls

    # Even if we loose some potential performance, we want to avoid OOM at all
    # costs. Therefore, we conservatively only use 80% of the VMEM budget.
    capped_vmem_limit_bytes = vmem_limit_bytes * 0.8

    def fit_lanes_and_buffers(
        batch_size: int, n_buffer: int, bq_sz: int, bkv_sz: int
    ) -> tuple[int, int]:
        # Lanes go first; fewer buffers is the last resort. Neither drops below one.
        while (
            batch_size > 1
            and calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
            > capped_vmem_limit_bytes
        ):
            batch_size -= 1
        while (
            n_buffer > 1
            and calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
            > capped_vmem_limit_bytes
        ):
            n_buffer -= 1
        if (
            calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
            > capped_vmem_limit_bytes
        ):
            raise ValueError(
                f"Cannot fit a {bq_sz=}, {bkv_sz=} tile within 80% of"
                f" {vmem_limit_bytes=}."
            )
        return batch_size, n_buffer

    def find_best_block_sizes(
        max_batch_size: int, max_n_buffer: int, fixed_bq_sz: int | None = None
    ) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        bkv_sz = bkv_stride = mxu_column_size
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
        else:
            bq_sz = fixed_bq_sz
            bq_stride = 0
        # Step 1: Lower batch_size and/or n_buffer if even the smallest bq and bkv
        # size can trigger OOM.
        batch_size, n_buffer = fit_lanes_and_buffers(
            max_batch_size, max_n_buffer, bq_sz, bkv_sz
        )

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        while (
            calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
            < capped_vmem_limit_bytes
        ):
            # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
            # size. When using causal masking, if bq size is larger than bkv size,
            # entire kv tile can be masked out for some query tokens. Similarly, if
            # bkv size is larger than bq size, entire query tile can be masked out for
            # some kv tokens.
            bkv_sz += bkv_stride
            bq_sz += bq_stride

        # Rollback one step since the last attempted value triggered OOM.
        bkv_sz -= bkv_stride
        bq_sz -= bq_stride

        # Indicates OOM was triggered from the starting bkv size.
        if bkv_sz == 0:
            raise ValueError("Cannot find block sizes that fit within VMEM limit.")

        # Step 3: Given current tile size, calculate compute tile size.
        if fixed_bq_sz is not None:
            return configs.BlockSizes(
                bq_sz=bq_sz,
                bq_c_sz=bq_sz,
                bkv_sz=bkv_sz,
                batch_size=batch_size,
                n_buffer=n_buffer,
            )

        # Fixed threshold value based on hardware spec.
        # TODO(kyuyeunk): Use different threshold based on hardware and precision.
        threshold = 1500
        # In flash_attention, each compute sub-tile reshapes Q to
        # (bq_c_sz * aligned_num_q_heads_per_kv_head, head_dim), so keep
        # bq_c_sz a power-of-2 divisor >= 1 full sublane tile (num_lanes)
        # where possible, and return the divisor that satisfies threshold
        # rather than the previous divisor that exceeded it.
        min_bq_c_sz = min(bq_sz, max(1, num_lanes // aligned_num_q_heads_per_kv_head))
        max_num_bq_c = min(bq_sz, 32)
        best_bq_c_sz = bq_sz
        for num_bq_c in range(1, max_num_bq_c + 1):
            bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)
            if bq_c_rem != 0:
                continue
            if (
                bq_c_sz > min_bq_c_sz
                and (bq_c_sz & (bq_c_sz - 1)) != 0
                and bq_c_sz != bq_sz
            ):
                continue
            best_bq_c_sz = bq_c_sz
            if (
                calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz) <= threshold
                or bq_c_sz <= min_bq_c_sz
            ):
                break

        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=best_bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=n_buffer,
        )

    # Default to triple buffer as its almost always beneficial.
    n_buffer = 3
    # Fixed value based on experimental results.
    decode_batch_size = 8
    # One cell for prefill, not two. The kernel walks cells serially inside the
    # body, so a second cell adds no parallel compute -- only DMA overlap, which
    # prefill is not waiting on (a 32k context is a few MB of KV at fp8). What it
    # does cost is half the VMEM per cell, and that VMEM buys more as tile.
    # Measured on v7-8, Qwen3-Coder-30B at tp8, one 32k MIXED step with a 4k
    # chunk: two cells picks bq=bkv=1280 and runs 1194us, one cell picks
    # bq=bkv=2048 and runs 880us. bq=2048 with two cells does not fit VMEM at
    # all. The search below is otherwise unchanged -- it grows bq and bkv in
    # lockstep, which lands on the measured optimum once the budget is a whole
    # cell's worth.
    prefill_batch_size = 1

    def cover_whole_pages(block_sizes: configs.BlockSizes) -> configs.BlockSizes:
        # The schedule fetches block k's cached pages from k * cdiv(bkv_sz, page),
        # while the kernel labels its keys from k * bkv_sz; they agree on whole pages.
        page = serve_cfgs.page_size
        if block_sizes.bkv_sz % page == 0:
            return block_sizes
        if block_sizes.bkv_sz > page:
            # The q block stays as searched; the kernel does not need bq_sz <= bkv_sz.
            return dataclasses.replace(
                block_sizes, bkv_sz=block_sizes.bkv_sz // page * page
            )
        # Growing the k block to one page needs VMEM the search did not budget.
        # Like the search, keep the buffers and halve the q block first; a mixed
        # q block stops at 2 tokens, since the kernel reads bq_sz == 1 as decode.
        min_bq_sz = min(block_sizes.bq_sz, 2)
        bq_sz = block_sizes.bq_sz
        while (
            bq_sz > min_bq_sz
            and calculate_vmem_usage(1, block_sizes.n_buffer, bq_sz, page)
            > capped_vmem_limit_bytes
        ):
            bq_sz = max(min_bq_sz, bq_sz // 2)
        try:
            batch_size, n_buffer = fit_lanes_and_buffers(
                block_sizes.batch_size, block_sizes.n_buffer, bq_sz, page
            )
        except ValueError as e:
            raise ValueError(
                f"A one-page k block ({page=}) does not fit within VMEM limit;"
                " a smaller page size may."
            ) from e
        return dataclasses.replace(
            block_sizes,
            bq_sz=bq_sz,
            bq_c_sz=math.gcd(block_sizes.bq_c_sz, bq_sz),
            bkv_sz=page,
            batch_size=batch_size,
            n_buffer=n_buffer,
        )

    decode_block_sizes = cover_whole_pages(
        find_best_block_sizes(decode_batch_size, n_buffer, 1)
    )
    prefill_block_sizes = find_best_block_sizes(prefill_batch_size, n_buffer)

    # Operator escape hatch. The search above works from a VMEM estimate that
    # does not model everything the compiled program allocates, so on a shape or
    # a chip where it is wrong the failure is a startup allocation error with no
    # way out short of a code patch. Rounded up to a whole page: a k block below
    # one page yields an empty DMA schedule leaf and fails Mosaic compilation.
    forced_bkv = envs.SGLANG_TPU_FORCE_RPA_PREFILL_BKV.get()
    if forced_bkv is not None:
        page = serve_cfgs.page_size
        aligned = max(page, -(-int(forced_bkv) // page) * page)
        prefill_block_sizes = dataclasses.replace(prefill_block_sizes, bkv_sz=aligned)
    else:
        prefill_block_sizes = cover_whole_pages(prefill_block_sizes)

    return decode_block_sizes, prefill_block_sizes


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "skip_kv_update",
        "kv_layout",
        "cache_only",
        "skip_kv_write",
        "return_lse",
        "cp_group_size",
        "attention_scope",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    skip_kv_update: bool = False,
    kv_layout: configs.KVLayout | None = None,
    cache_only: bool = False,
    skip_kv_write: bool = False,
    return_lse: bool = False,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    kv_lens_global: jax.Array | None = None,
    attention_scope: configs.AttentionScope | int = configs.AttentionScope.FULL,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Perform batched ragged paged attention.

    Args:
        queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
        keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
        values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
        kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Stores existing kv cache data where k & vs are
            concatenated along num kv heads dim.
        kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
            page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
            sequence.
        cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
            sequences. distribution[2] represents total number of sequences.
        attention_sink: [num_q_heads]. Per-head sink logit, treated as one extra
            key with a zero value for every query (same semantics as RPA v3).
        sm_scale: Softmax scale value.
        sliding_window: Size of sliding window (also known as local attention). kvs
            outside of the window is not fetched from hbm and masked out during
            computation.
        soft_cap: Cap values of softmax inputs.
        mask_value: Value to use for causal masking. Defaults to smallest
            representable value of the activation dtype.
        q_scale: Quantization scale value of queries.
        k_scale: Quantization scale value of keys.
        v_scale: Quantization scale value of values.
        chunk_prefill_size: Not used.
        decode_block_sizes: Kernel block size to use during decode.
        prefill_block_sizes: Kernel block size to use during prefill.
        vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
            size of the hardware.
        debug_mode: Not used.
        out_dtype: Dtype of output. Defaults to dtype of queries.
        use_causal_mask: Not used.
        skip_kv_update: Cross-layer KV cache sharing. When True, this (shared)
            layer reads its entire K/V span from the cache slot written by its
            target layer earlier in the same forward pass, ignores its own k/v
            inputs, and does not write the cache back.
        cache_only: Read all kv_lens[s] keys of every sequence from the cache;
            keys/values are ignored, and the queries follow every key, so there
            is no causal mask. kv_lens[s] may be 0 or below q_len. Requires
            HEAD_ALONG_SUBLANE.
        skip_kv_write: Never write the cache. Without cache_only, the last q_len
            keys of each sequence still come from keys/values.
        return_lse: Also return lse[token, q head] = log(sum of exp(logits)),
            natural log, float32, with exp(sink) in the sum when attention_sink
            is given; -inf for a row that attended no key and has no sink.
        cp_group_size: Size of a decode-context-parallel group whose ranks each
            hold the positions p with p % cp_group_size == cp_rank, local key j
            being global position j * cp_group_size + cp_rank. The sliding
            window then runs in global positions. Requires cache_only.
        cp_rank: [1] int32. This rank's index in the group, traced so that one
            program serves every rank.
        kv_lens_global: [max_num_seqs] int32. Global kv length of each sequence;
            its queries sit at global positions kv_lens_global[s] - q_len onward.

    Returns:
        out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
        new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Result of new kv cache where k & vs are
            concatenated along num kv heads dim.
        lse: [max_num_tokens, num_q_heads] float32, only with return_lse.
    """

    if kv_layout is None:
        kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE

    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if chunk_prefill_size is not None:
        raise ValueError("Specifying chunk prefill size is not supported.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")

    if out_dtype is None:
        out_dtype = queries.dtype
    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    max_num_seqs = kv_lens.shape[0]
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache.shape[4]
    else:
        page_size = kv_cache.shape[1]

    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    num_kv_heads = keys.shape[1]
    num_page_indices = page_indices.shape[0]

    if attention_sink is not None and attention_sink.shape != (num_q_heads,):
        raise ValueError(f"Expected {attention_sink.shape=} to be ({num_q_heads},).")

    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
        has_attention_sink=attention_sink is not None,
    )
    if isinstance(attention_scope, int):
        _scope_map = {
            0: configs.AttentionScope.FULL,
            1: configs.AttentionScope.CACHE_ONLY,
            2: configs.AttentionScope.NEW_TOKENS_ONLY,
            3: configs.AttentionScope.STRIPED,
        }
        attention_scope = _scope_map.get(attention_scope, configs.AttentionScope.FULL)
    if cache_only:
        attention_scope = configs.AttentionScope.CACHE_ONLY

    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
        kv_layout=kv_layout,
        cache_only=cache_only,
        skip_kv_write=skip_kv_write,
        return_lse=return_lse,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
    )

    q_hbm, new_kv_hbm = prepare_inputs(
        queries,
        keys,
        values,
        queries.dtype,
        kv_cache.dtype,
        kv_layout=kv_layout,
        page_size=page_size,
    )
    sink_table = (
        None
        if attention_sink is None
        else prepare_attention_sink(attention_sink, num_kv_heads, queries.dtype)
    )

    # Explicit block sizes for both modes are used as given, even where the
    # search would refuse.
    default_decode = default_prefill = None
    if decode_block_sizes is None or prefill_block_sizes is None:
        default_decode, default_prefill = calculate_block_sizes(
            model_cfgs, serve_cfgs, vmem_limit_bytes
        )

    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    lse_hbm = None
    if return_lse:
        # Lane g of a (KV head, token) row is q head g; tokens that no q block
        # reaches keep -inf.
        lse_hbm = jnp.full(
            (num_kv_heads, queries.shape[0], 1, pltpu.get_tpu_info().num_lanes),
            -jnp.inf,
            dtype=jnp.float32,
        )

    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
        lse_hbm: jax.Array | None,
    ):
        if mode == configs.RpaCase.DECODE:
            effective_blocks = decode_block_sizes or default_decode
        else:
            effective_blocks = prefill_block_sizes or default_prefill

        cfgs = configs.RpaConfigs(
            block=effective_blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            cp_rank=cp_rank,
            kv_lens_global=kv_lens_global,
        )

        schedule_hbm = schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            cfgs=cfgs,
            skip_kv_update=skip_kv_update,
            cp_rank=cp_rank,
            kv_lens_global=kv_lens_global,
        )
        return kernel.rpa_kernel(
            cu_q_lens,
            kv_lens,
            page_indices,
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            kv_cache,
            sink_table,
            cfgs=cfgs,
            lse_hbm=lse_hbm,
            cp_rank=cp_rank,
            kv_lens_global=kv_lens_global,
        )

    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.DECODE, q_hbm, kv_cache, lse_hbm
    )
    if serve_cfgs.attention_scope != configs.AttentionScope.STRIPED:
        o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
            configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache, lse_hbm
        )

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    if not return_lse:
        return o_hbm, kv_cache
    # [kv_heads, tokens, 1, lanes] -> [tokens, num_q_heads]
    lse = lse_hbm[:, :, 0, :num_q_heads_per_kv_head].swapaxes(0, 1)
    return o_hbm, kv_cache, lse.reshape(queries.shape[0], num_q_heads)
