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

2. Triple-buffers its inputs with an explicit pipeline (`pipeline.py`): three
input slots and two output slots for every kernel.

3. Precomputes expensive metadata upfront (e.g., page locations and bounds clipping) via
the schedule kernel (`scheduler.py`), driven by a pluggable scheduler class. The
schedule is calculated once per call.

Note: batched_rpa is build on top / derived from RPA3.
"""

import dataclasses
import functools
import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.batched_rpa_pcp import configs, kernel
from sglang_tpu.attention.pallas.batched_rpa_pcp import scheduler as scheduler_lib
from sglang_tpu.attention.pallas.batched_rpa_pcp import scheduler_cp, utils
from sglang_tpu.envs import envs


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    k_scale: jax.Array | None = None,
    v_scale: jax.Array | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int = 128,
    lane_fold_kv_head_dim: bool = False,
    token_major_q: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Lays out q and the new k/v for the kernels.

    The new k/v may have more tokens than q (paged new K/V, see
    `ServingConfigs.paged_new_kv`). With `token_major_q` the queries stay
    `[tokens, kv_heads, ...]` instead of `[kv_heads, tokens, ...]`.
    """

    total_q_tokens, actual_num_q_heads, actual_q_head_dim = q.shape
    total_kv_tokens, actual_num_kv_heads, actual_kv_head_dim = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = utils.get_dtype_packing(q_dtype)
    kv_packing = utils.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head, q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_q_head_dim = utils.align_to(actual_q_head_dim, num_lanes)

    # Compute aligned kv head dimension, accounting for per-token scale if
    # present.
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if k_scale is not None and v_scale is not None:
            scale_bits = jax.dtypes.itemsize_bits(k_scale.dtype)
            kv_bits = jax.dtypes.itemsize_bits(kv_dtype)
            num_scale_channels = max(1, scale_bits // kv_bits)
            aligned_kv_head_dim = utils.align_to(
                actual_kv_head_dim + num_scale_channels, num_sublanes * kv_packing
            )
        else:
            aligned_kv_head_dim = utils.align_to(
                actual_kv_head_dim, num_sublanes * kv_packing
            )
        num_lane_folds_in_head_dim = 1
    else:
        num_lane_folds_in_head_dim = configs.num_lane_folds_in_head_dim(
            actual_kv_head_dim,
            kv_dtype,
            kv_layout,
            num_lanes,
            enabled=lane_fold_kv_head_dim,
        )
        aligned_kv_head_dim = utils.align_to(
            actual_kv_head_dim // num_lane_folds_in_head_dim, num_lanes
        )

    # queries: (T, H, D) -> (T, H_kv, G, D); kv-head-major unless token_major_q.
    o_hbm_alias_q_hbm = jnp.pad(
        q.reshape(
            total_q_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head,
            actual_q_head_dim,
        ),
        (
            (0, 0),
            (0, 0),
            (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
            (0, aligned_q_head_dim - actual_q_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        actual_num_kv_heads,
        aligned_num_q_heads_per_kv_head // q_packing,
        q_packing,
        aligned_q_head_dim,
    )
    if not token_major_q:
        o_hbm_alias_q_hbm = o_hbm_alias_q_hbm.swapaxes(0, 1)

    # Pad keys and values head_dim
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2_aligned = utils.align_to(
        actual_num_kv_heads_x2 * num_lane_folds_in_head_dim, kv_packing
    )

    if k_scale is not None and v_scale is not None:
        k_scale = k_scale.reshape(total_kv_tokens, actual_num_kv_heads)
        v_scale = v_scale.reshape(total_kv_tokens, actual_num_kv_heads)

        scale_bits = jax.dtypes.itemsize_bits(k_scale.dtype)
        kv_bits = jax.dtypes.itemsize_bits(k.dtype)
        num_scale_channels = max(1, scale_bits // kv_bits)

        # Bitcast the scale factors to the kv dtype and reshape scale factors to
        # (T, H_kv, num_scale_channels).
        k_scale_split = jax.lax.bitcast_convert_type(k_scale, k.dtype).reshape(
            total_kv_tokens, actual_num_kv_heads, num_scale_channels
        )
        v_scale_split = jax.lax.bitcast_convert_type(v_scale, v.dtype).reshape(
            total_kv_tokens, actual_num_kv_heads, num_scale_channels
        )

        # k/v have shape (T, H_kv, D) -> (T, H_kv, D + num_scale_channels)
        k = jnp.concatenate([k, k_scale_split], axis=-1)
        v = jnp.concatenate([v, v_scale_split], axis=-1)

        actual_kv_head_dim += num_scale_channels

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_lanes = pltpu.get_tpu_info().num_lanes
        align_tokens = max(num_lanes, page_size)
        padded_total_tokens = utils.align_to(total_kv_tokens, align_tokens)
        new_kv_hbm = (
            jnp.pad(
                jnp.concatenate([k, v], axis=-1).reshape(
                    total_kv_tokens, actual_num_kv_heads_x2, actual_kv_head_dim
                ),
                (
                    (0, padded_total_tokens - total_kv_tokens),
                    (0, 0),
                    (0, aligned_kv_head_dim - actual_kv_head_dim),
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
        kv = jnp.concatenate([k, v], axis=-1).reshape(
            total_kv_tokens, actual_num_kv_heads_x2, actual_kv_head_dim
        )
        if num_lane_folds_in_head_dim > 1:
            actual_kv_head_dim //= num_lane_folds_in_head_dim
            kv = (
                kv.reshape(
                    total_kv_tokens,
                    actual_num_kv_heads_x2,
                    num_lane_folds_in_head_dim,
                    actual_kv_head_dim,
                )
                .transpose(0, 2, 1, 3)
                .reshape(total_kv_tokens, -1, actual_kv_head_dim)
            )
            actual_num_kv_heads_x2 *= num_lane_folds_in_head_dim
        new_kv_hbm = jnp.pad(
            kv,
            (
                (0, 0),
                (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
                (0, aligned_kv_head_dim - actual_kv_head_dim),
            ),
            constant_values=0,
        ).reshape(
            total_kv_tokens,
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
    use_per_token_scale: bool = False,
    scale_dtype: jnp.dtype | None = None,
    lane_fold_kv_head_dim: bool = False,
):
    """Shape of the KV cache the kernel reads and writes.

    With the defaults this is the layout of SGLang's KV pools
    (`sglang_tpu.kv_pool`). `lane_fold_kv_head_dim` selects google3's
    lane-folded fp8 HEAD_ALONG_SUBLANE layout instead; see
    `configs.num_lane_folds_in_head_dim`.
    """
    if kv_layout is None:
        kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes

    if kv_packing is None:
        kv_packing = utils.get_dtype_packing(kv_dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_scale_channels = 0
        if use_per_token_scale:
            assert scale_dtype is not None
            kv_bits = jax.dtypes.itemsize_bits(kv_dtype)
            scale_bits = jax.dtypes.itemsize_bits(scale_dtype)
            num_scale_channels = max(1, scale_bits // kv_bits)
        base_dim = actual_head_dim + num_scale_channels
        return (
            total_num_pages,
            actual_num_kv_heads * 2,
            utils.align_to(base_dim, num_sublanes * kv_packing) // kv_packing,
            kv_packing,
            page_size,
        )
    fold = 1
    if lane_fold_kv_head_dim:
        fold = configs.num_lane_folds_in_head_dim(
            actual_head_dim, kv_dtype, kv_layout, num_lanes, enabled=True
        )
    return (
        total_num_pages,
        page_size,
        utils.align_to(actual_num_kv_heads * 2 * fold, kv_packing) // kv_packing,
        kv_packing,
        utils.align_to(actual_head_dim // fold, num_lanes),
    )


def fold_kv_cache_head_dim(
    kv_cache: jax.Array,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    chip_version: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X,
) -> jax.Array:
    """Converts an unfolded HEAD_ALONG_SUBLANE cache to the lane-folded layout
    (`lane_fold_kv_head_dim=True`)."""
    num_lanes = pltpu.get_tpu_info_for_chip(chip_version, 1).num_lanes
    kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE
    fold = configs.num_lane_folds_in_head_dim(
        actual_head_dim, kv_cache.dtype, kv_layout, num_lanes, enabled=True
    )
    if fold == 1:
        return kv_cache
    num_pages, page_size = kv_cache.shape[:2]
    num_kv_heads_x2 = actual_num_kv_heads * 2
    folded_head_dim = actual_head_dim // fold
    folded_shape = get_kv_cache_shape(
        num_pages,
        page_size,
        actual_num_kv_heads,
        actual_head_dim,
        kv_cache.dtype,
        kv_layout=kv_layout,
        lane_fold_kv_head_dim=True,
    )
    kv = kv_cache.reshape(num_pages, page_size, -1, kv_cache.shape[-1])
    kv = kv[:, :, :num_kv_heads_x2, :actual_head_dim]
    kv = (
        kv.reshape(num_pages, page_size, num_kv_heads_x2, fold, folded_head_dim)
        .transpose(0, 1, 3, 2, 4)
        .reshape(num_pages, page_size, num_kv_heads_x2 * fold, folded_head_dim)
    )
    kv = jnp.pad(
        kv,
        (
            (0, 0),
            (0, 0),
            (0, folded_shape[2] * folded_shape[3] - num_kv_heads_x2 * fold),
            (0, folded_shape[4] - folded_head_dim),
        ),
    )
    return kv.reshape(folded_shape)


def unfold_kv_cache_head_dim(
    kv_cache: jax.Array,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    chip_version: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X,
) -> jax.Array:
    """Inverse of `fold_kv_cache_head_dim`."""
    num_lanes = pltpu.get_tpu_info_for_chip(chip_version, 1).num_lanes
    kv_layout = configs.KVLayout.HEAD_ALONG_SUBLANE
    fold = configs.num_lane_folds_in_head_dim(
        actual_head_dim, kv_cache.dtype, kv_layout, num_lanes, enabled=True
    )
    if fold == 1:
        return kv_cache
    num_pages, page_size, _, kv_packing, _ = kv_cache.shape
    num_kv_heads_x2 = actual_num_kv_heads * 2
    folded_head_dim = actual_head_dim // fold
    kv = kv_cache.reshape(num_pages, page_size, -1, kv_cache.shape[-1])
    kv = kv[:, :, : num_kv_heads_x2 * fold, :folded_head_dim]
    kv = (
        kv.reshape(num_pages, page_size, fold, num_kv_heads_x2, folded_head_dim)
        .transpose(0, 1, 3, 2, 4)
        .reshape(num_pages, page_size, num_kv_heads_x2, actual_head_dim)
    )
    num_kv_heads_x2_aligned = utils.align_to(num_kv_heads_x2, kv_packing)
    aligned_head_dim = utils.align_to(actual_head_dim, num_lanes)
    kv = jnp.pad(
        kv,
        (
            (0, 0),
            (0, 0),
            (0, num_kv_heads_x2_aligned - num_kv_heads_x2),
            (0, aligned_head_dim - actual_head_dim),
        ),
    )
    return kv.reshape(
        num_pages,
        page_size,
        num_kv_heads_x2_aligned // kv_packing,
        kv_packing,
        aligned_head_dim,
    )


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_query_size: int = 1,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill.

    This is SGLang's search, not google3's: one prefill cell, whole-page K
    blocks, a K block that may grow one step past the Q block, and smaller Q
    tiles at 128+ query heads. Every kernel has `configs.NUM_INPUT_SLOTS` input
    and `configs.NUM_OUTPUT_SLOTS` output buffers, so the search only trades
    batch lanes and tile sizes against VMEM.
    """

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    mxu_column_size = tpu_info.mxu_column_size

    # Calculate aligned model dimensions.
    aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)
    num_lane_folds_in_head_dim = configs.num_lane_folds_in_head_dim(
        model_cfgs.head_dim,
        serve_cfgs.dtype_kv,
        serve_cfgs.kv_layout,
        num_lanes,
        enabled=serve_cfgs.lane_fold_kv_head_dim,
    )
    if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        aligned_kv_head_dim = utils.align_to(
            model_cfgs.head_dim, tpu_info.num_sublanes * serve_cfgs.packing_kv
        )
    else:
        aligned_kv_head_dim = utils.align_to(
            model_cfgs.head_dim // num_lane_folds_in_head_dim, num_lanes
        )
    aligned_num_q_heads_per_kv_head = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
    )
    aligned_num_q_heads = aligned_num_q_heads_per_kv_head * model_cfgs.num_kv_heads
    high_head_count = aligned_num_q_heads >= num_lanes

    bkv_stride = pl.cdiv(
        model_cfgs.num_kv_heads * 2 * num_lane_folds_in_head_dim,
        serve_cfgs.packing_kv,
    )
    if utils.has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

    q_bytes = jnp.dtype(serve_cfgs.dtype_q).itemsize
    kv_bytes = jnp.dtype(serve_cfgs.dtype_kv).itemsize
    out_bytes = jnp.dtype(serve_cfgs.dtype_out).itemsize

    def calculate_vmem_usage(batch_size: int, bq_sz: int, bkv_sz: int) -> int:
        """Given tile size, calculate VMEM usage of the kernel."""

        # Step 1: Calculate buffer sizes.

        # Calculate size bq & bkv arrays for a single buffer.
        bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            kv_head_width = aligned_kv_head_dim if high_head_count else aligned_head_dim
            bkv_array_size = (
                (bkv_sz + 2 * serve_cfgs.page_size)
                * aligned_num_kv_heads_x2
                * kv_head_width
            )
        else:
            bkv_array_size = bkv_sz * aligned_num_kv_heads_x2 * aligned_kv_head_dim

        # Get output buffer size as well - which has same size as query size.
        bo_array_size = bq_array_size

        # Convert to bytes.
        bq_bytes = bq_array_size * q_bytes
        bkv_bytes = bkv_array_size * kv_bytes
        bo_bytes = bo_array_size * out_bytes

        # Account for multiple buffers. These counts are fixed by the pipeline; see
        # `configs.NUM_INPUT_SLOTS` / `configs.NUM_OUTPUT_SLOTS`.
        bq_bytes *= configs.NUM_INPUT_SLOTS
        bkv_bytes *= configs.NUM_INPUT_SLOTS
        bo_bytes *= configs.NUM_OUTPUT_SLOTS

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

        if high_head_count:
            # When unsharded query heads span >= 128 lanes (e.g. 128 Q / 16 KV
            # heads on PP=8), the m and l scratch carries add two 128-lane tiles
            # per (KV head, token, q-head-per-kv) row beyond what qk_size proxies
            # at smaller head counts. Counted at 4 bytes so the tiles still fit
            # if the carries move to float32.
            carry_rows = (
                model_cfgs.num_kv_heads * bq_sz * aligned_num_q_heads_per_kv_head
            )
            total_bytes += carry_rows * (2 * num_lanes) * 4

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

    def fit_lanes(batch_size: int, bq_sz: int, bkv_sz: int) -> int:
        # Lanes are the only thing left to give up: the buffer count is fixed.
        while (
            batch_size > 1
            and calculate_vmem_usage(batch_size, bq_sz, bkv_sz)
            > capped_vmem_limit_bytes
        ):
            batch_size -= 1
        if calculate_vmem_usage(batch_size, bq_sz, bkv_sz) > capped_vmem_limit_bytes:
            raise ValueError(
                f"Cannot fit a {bq_sz=}, {bkv_sz=} tile within 80% of"
                f" {vmem_limit_bytes=}."
            )
        return batch_size

    def find_best_block_sizes(
        max_batch_size: int, fixed_bq_sz: int | None = None
    ) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        bkv_sz = bkv_stride = mxu_column_size
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
            # Every kernel has three input buffers, so a shape whose smallest
            # square tile overflows VMEM gets a smaller q block (bkv_sz stays at
            # the MXU width): first down to a whole sublane tile of query rows,
            # then, as the last resort, down to 2 tokens (the kernel reads
            # bq_sz == 1 as decode). This is the 128+ query-head case (bq_sz ==
            # mxu_column_size overflows on PP=8), and any other shape with large
            # per-token buffers, e.g. many KV heads on SEQ_ALONG_LANE with LSE.
            min_auto_bq = max(16, num_lanes // aligned_num_q_heads_per_kv_head)
            while (
                bq_sz > min_auto_bq
                and calculate_vmem_usage(1, bq_sz, bkv_sz) > capped_vmem_limit_bytes
            ):
                bq_sz = max(min_auto_bq, bq_sz // 2)
                bq_stride = 0
            while (
                bq_sz > 2
                and calculate_vmem_usage(1, bq_sz, bkv_sz) > capped_vmem_limit_bytes
            ):
                bq_sz = max(2, bq_sz // 2)
                bq_stride = 0
        else:
            bq_sz = fixed_bq_sz
            bq_stride = 0
        # Step 1: Lower batch_size if even the smallest bq and bkv size can
        # trigger OOM.
        batch_size = fit_lanes(max_batch_size, bq_sz, bkv_sz)

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        if fixed_bq_sz is not None or bq_stride > 0:
            while (
                calculate_vmem_usage(batch_size, bq_sz, bkv_sz)
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

        # When query-side buffers fill VMEM (e.g. 64 or 128 query heads on one chip
        # under PP with TP=1), lockstep growth stops at bkv_sz == MXU width (with
        # bq_sz <= MXU width) and VMEM left over. Spend it on one 2x step of keys
        # (bkv_sz = 2 * MXU width): each q block takes half the KV steps, and the
        # causal diagonal still ends on a k block edge. Only full-attention layers
        # with at least 8 q heads per KV head measured faster: with fewer, the
        # bq_c_sz that Step 3 picks for the wider block slows bf16 KV, and a sliding
        # window reads more keys outside the window. Also skip the step when
        # sequences in this pool are too short to fill the wider block, and when it
        # is not a whole number of pages (cover_whole_pages would cut it after Step 3
        # picks bq_c_sz for it).
        if (
            fixed_bq_sz is None
            and bq_sz <= mxu_column_size
            and bkv_sz == mxu_column_size
            and model_cfgs.sliding_window is None
            and model_cfgs.num_q_heads_per_kv_head >= 8
        ):
            grown_bkv_sz = 2 * bkv_sz
            if (
                grown_bkv_sz <= serve_cfgs.pages_per_seq * serve_cfgs.page_size
                and grown_bkv_sz % serve_cfgs.page_size == 0
                and calculate_vmem_usage(batch_size, bq_sz, grown_bkv_sz)
                < capped_vmem_limit_bytes
            ):
                bkv_sz = grown_bkv_sz

        # Step 3: Given current tile size, calculate compute tile size.
        if fixed_bq_sz is not None:
            return configs.BlockSizes(
                bq_sz=bq_sz,
                bq_c_sz=bq_sz,
                bkv_sz=bkv_sz,
                batch_size=batch_size,
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
        )

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
        # Like the search, halve the q block first; a mixed q block stops at 2
        # tokens, since the kernel reads bq_sz == 1 as decode.
        min_bq_sz = min(block_sizes.bq_sz, 2)
        bq_sz = block_sizes.bq_sz
        while (
            bq_sz > min_bq_sz
            and calculate_vmem_usage(1, bq_sz, page) > capped_vmem_limit_bytes
        ):
            bq_sz = max(min_bq_sz, bq_sz // 2)
        try:
            batch_size = fit_lanes(block_sizes.batch_size, bq_sz, page)
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
        )

    decode_block_sizes = cover_whole_pages(
        find_best_block_sizes(decode_batch_size, decode_query_size)
    )
    prefill_block_sizes = find_best_block_sizes(prefill_batch_size)

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


# ---------------------------------------------------------------------------
# Block sizes of the PCP ring / fused-PCP kernels (google3 cl/985434096).
#
# `calculate_block_sizes` above is SGLang's search and serves every other
# kernel, including the DCP ones. The classes below are google3's calculators
# with their own VMEM model; only the ring cache phase
# (`RingBlockSizeCalculator`) and the paged new-token phase of PCP
# (`CPBlockSizeCalculator`) use them. `BaseBlockSizeCalculator` is their shared
# base and is not used on its own here.
# ---------------------------------------------------------------------------


class BaseBlockSizeCalculator:
    """Chooses kernel block sizes that fit the VMEM budget.

    Subclass and override `prefill_block_sizes` to change how the budget is
    spent; the VMEM model and the compute-tile search are shared.
    """

    decode_batch_size = 8
    prefill_batch_size = 2

    def __init__(
        self,
        model_cfgs: configs.ModelConfigs,
        serve_cfgs: configs.ServingConfigs,
        vmem_limit_bytes: int,
        decode_query_size: int = 1,
    ):
        self.model_cfgs = model_cfgs
        self.serve_cfgs = serve_cfgs
        self.vmem_limit_bytes = vmem_limit_bytes
        self.decode_query_size = decode_query_size

        tpu_info = pltpu.get_tpu_info()
        self.tpu_info = tpu_info
        num_lanes = tpu_info.num_lanes
        num_sublanes = tpu_info.num_sublanes
        self.mxu_column_size = tpu_info.mxu_column_size

        # Calculate aligned model dimensions.
        self.aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)

        aligned_num_q_heads_per_kv_head = utils.align_to(
            model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
        )
        self.aligned_num_q_heads_per_kv_head = aligned_num_q_heads_per_kv_head
        self.aligned_num_q_heads = (
            aligned_num_q_heads_per_kv_head * model_cfgs.num_kv_heads
        )

        self.num_lane_folds_in_head_dim = configs.num_lane_folds_in_head_dim(
            model_cfgs.head_dim,
            serve_cfgs.dtype_kv,
            serve_cfgs.kv_layout,
            num_lanes,
            enabled=serve_cfgs.lane_fold_kv_head_dim,
        )
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            self.aligned_num_kv_heads_x2 = model_cfgs.num_kv_heads * 2
        else:
            bkv_stride = pl.cdiv(
                model_cfgs.num_kv_heads * 2 * self.num_lane_folds_in_head_dim,
                serve_cfgs.packing_kv,
            )
            if utils.has_bank_conflicts(bkv_stride):
                bkv_stride += 1
            self.aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

        self.q_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_q) / 8
        self.kv_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_kv) / 8
        self.out_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_out) / 8

        is_packed_fp4 = serve_cfgs.dtype_kv == jnp.uint8
        physical_kv_head_dim = model_cfgs.head_dim
        if is_packed_fp4:
            physical_kv_head_dim = physical_kv_head_dim // 2

        packing_kv = serve_cfgs.packing_kv
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            base_dim = physical_kv_head_dim + serve_cfgs.scale_channels
            self.aligned_kv_head_dim_val = utils.align_to(
                base_dim, num_sublanes * packing_kv
            )
        else:
            self.aligned_kv_head_dim_val = utils.align_to(
                model_cfgs.head_dim // self.num_lane_folds_in_head_dim, num_lanes
            )

    @property
    def capped_vmem_limit_bytes(self) -> float:
        # Even if we loose some potential performance, we want to avoid OOM at
        # all costs. Therefore, we conservatively only use 80% of the VMEM
        # budget.
        return self.vmem_limit_bytes * 0.8

    @property
    def max_bkv_sz(self) -> int:
        max_seq_len = self.serve_cfgs.pages_per_seq * self.serve_cfgs.page_size
        return min(max_seq_len, 4096)

    @property
    def bkv_stride_sz(self) -> int:
        """Granularity of the K block search.

        SGLang: K blocks are whole pages (the schedule fetches block k's cached
        pages from `k * cdiv(bkv_sz, page_size)`), so the MXU column size is
        rounded up to a whole number of pages.
        """
        return utils.align_to(self.mxu_column_size, self.serve_cfgs.page_size)

    def lse_buffer_bytes(self, bq_sz: int) -> int:
        """VMEM bytes of the LSE output buffers and staging for `bq_sz`.

        SGLang's LSE is one float32 row of lanes per (KV head, token) in each
        output slot (`RpaConfigs.lse_vmem_shape`), staged through one float32
        row per (token, aligned q head) of a KV head (`lse_stage_shape`).
        """
        if not self.serve_cfgs.return_lse:
            return 0
        num_lanes = self.tpu_info.num_lanes
        slots = configs.NUM_OUTPUT_SLOTS * self.model_cfgs.num_kv_heads * num_lanes
        stage = self.aligned_num_q_heads_per_kv_head * num_lanes
        return (slots + stage) * 4 * bq_sz

    def calculate_vmem_usage(self, batch_size: int, bq_sz: int, bkv_sz: int) -> int:
        """Given tile size, calculate VMEM usage of the kernel."""
        model_cfgs = self.model_cfgs
        serve_cfgs = self.serve_cfgs

        # Step 1: Calculate buffer sizes.
        bq_array_size = bq_sz * self.aligned_num_q_heads * self.aligned_head_dim
        vmem_bkv_sz = bkv_sz
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            # Allocate extra buffer to perform lane stitching. Must match
            # RpaConfigs.kv_vmem_lanes.
            vmem_bkv_sz += 2 * self.tpu_info.num_lanes
        bkv_array_size = (
            vmem_bkv_sz * self.aligned_num_kv_heads_x2 * self.aligned_kv_head_dim_val
        )
        bo_array_size = bq_array_size

        bq_bytes = bq_array_size * self.q_bytes
        bkv_bytes = bkv_array_size * self.kv_bytes
        bo_bytes = bo_array_size * self.out_bytes

        # Account for multiple buffers. These counts are fixed by the pipeline; see
        # `configs.NUM_INPUT_SLOTS` / `configs.NUM_OUTPUT_SLOTS`.
        bq_bytes *= configs.NUM_INPUT_SLOTS
        bkv_bytes *= configs.NUM_INPUT_SLOTS
        bo_bytes *= configs.NUM_OUTPUT_SLOTS

        # Sum up all buffer memory usage (scales are co-located inside bkv_bytes).
        buffer_bytes = bq_bytes + bkv_bytes + bo_bytes
        buffer_bytes += self.lse_buffer_bytes(bq_sz)

        # Step 2: Calculate worst case memory usage during computation.
        loaded_bq_size = bq_sz * model_cfgs.num_q_heads * self.aligned_head_dim
        loaded_bkv_size = bkv_sz * model_cfgs.num_kv_heads * self.aligned_head_dim
        qk_size = bq_sz * bkv_sz * model_cfgs.num_q_heads

        loaded_bq_bytes = loaded_bq_size * self.q_bytes
        loaded_bkv_bytes = loaded_bkv_size * self.kv_bytes
        qk_bytes = qk_size * self.out_bytes

        loaded_scale_bytes = 0
        if serve_cfgs.per_token_scale:
            scale_bits = jax.dtypes.itemsize_bits(serve_cfgs.per_token_scale_dtype)
            scale_bytes = scale_bits // 8
            loaded_scale_bytes = 2 * bkv_sz * model_cfgs.num_kv_heads * scale_bytes

        compute_bytes = (
            loaded_bq_bytes + loaded_bkv_bytes + qk_bytes + loaded_scale_bytes
        )

        total_bytes = buffer_bytes + compute_bytes
        total_bytes *= batch_size

        # Online-softmax scratch (m, l, and the output accumulator)
        lm_array_size = (
            model_cfgs.num_kv_heads * bq_sz * self.aligned_num_q_heads_per_kv_head
        )
        acc_array_size = (
            model_cfgs.num_kv_heads * bq_sz * self.aligned_num_q_heads_per_kv_head
        )
        # m stays in the compute dtype; l and acc are carried in float32.
        scratch_bytes = (
            lm_array_size * self.out_bytes + lm_array_size * 4 + acc_array_size * 4
        )
        total_bytes += scratch_bytes

        return int(total_bytes)

    def fits(self, batch_size: int, bq_sz: int, bkv_sz: int) -> bool:
        return (
            self.calculate_vmem_usage(batch_size, bq_sz, bkv_sz)
            <= self.capped_vmem_limit_bytes
        )

    def calculate_compute_buffer_time(
        self, batch_size: int, bq_c_sz: int, bkv_sz: int
    ) -> int:
        """Calculate approximate time it takes to compute a given tile."""
        model_cfgs = self.model_cfgs

        num_k_rows = pl.cdiv(bkv_sz, self.mxu_column_size)
        num_k_cols = pl.cdiv(model_cfgs.head_dim, self.mxu_column_size)
        num_k = num_k_rows * num_k_cols
        num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads

        return batch_size * num_muls

    def compute_tile_size(self, batch_size: int, bq_sz: int, bkv_sz: int) -> int:
        """Largest divisor of bq_sz whose compute block stays under threshold."""
        tpu_info = self.tpu_info
        is_8bit = jnp.dtype(self.serve_cfgs.dtype_q).itemsize == 1

        match tpu_info.generation:
            case 8 if configs.is_sfs(tpu_info):
                threshold = 800
            case 7:
                threshold = 1500
            case _:
                threshold = 1500

        if is_8bit:
            flops_ratio = tpu_info.fp8_ops_per_second // tpu_info.bf16_ops_per_second
            threshold *= flops_ratio
        num_bq_c = 1
        last_valid_bq_c_sz = bq_c_sz = bq_sz
        bq_c_rem = 0

        while (
            self.calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz) > threshold
            or bq_c_rem != 0
        ) and num_bq_c < bq_sz:
            if bq_c_rem == 0:
                last_valid_bq_c_sz = bq_c_sz
            num_bq_c += 1
            bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)

        return last_valid_bq_c_sz

    def find_best_block_sizes(
        self,
        max_batch_size: int,
        fixed_bq_sz: int | None = None,
        max_bkv_sz: int | None = None,
        max_bq_sz: int | None = None,
    ) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""
        capped_vmem_limit_bytes = self.capped_vmem_limit_bytes

        bkv_sz = bkv_stride = self.bkv_stride_sz
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
        else:
            bq_sz = fixed_bq_sz
            bq_stride = 0
        batch_size = max_batch_size

        # Step 1: Lower batch_size if even the smallest bq and bkv size can trigger
        # OOM.
        while (
            batch_size > 1
            and self.calculate_vmem_usage(batch_size, bq_sz, bkv_sz)
            > capped_vmem_limit_bytes
        ):
            batch_size -= 1

        # NOTE: as in google3, a smallest tile that is still over the cap at
        # batch_size 1 is returned as is; `CPBlockSizeCalculator` re-checks
        # single-lane tiles against its own cap and the fused-PCP wrapper
        # checks the tiles it ends up with (`pcp_fused_configs`).

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        while (
            self.calculate_vmem_usage(batch_size, bq_sz, bkv_sz)
            < capped_vmem_limit_bytes
            and bkv_sz <= self.max_bkv_sz
        ):
            # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
            # size. When using causal masking, if bq size is larger than bkv size,
            # entire kv tile can be masked out for some query tokens. Similarly, if
            # bkv size is larger than bq size, entire query tile can be masked out for
            # some kv tokens.
            bkv_sz += bkv_stride
            bq_sz += bq_stride

        # Rollback one step since the last attempted value triggered OOM.
        if bkv_sz > bkv_stride:
            bkv_sz -= bkv_stride
            if fixed_bq_sz is None:
                bq_sz -= bq_stride

        # Applied after the rollback, not inside the growth loop: the rollback
        # assumes the loop exited on the VMEM budget, so a cap in the loop
        # condition would roll back a step that was never taken. Caps must be
        # multiples of bkv_stride to keep KV tiles page-aligned. bq and bkv are
        # capped independently (see configs.SFS_PREFILL_MAX_*).
        if max_bkv_sz is not None and bkv_sz > max_bkv_sz:
            bkv_sz = max_bkv_sz
            if fixed_bq_sz is None:
                bq_sz = max_bkv_sz
        if fixed_bq_sz is None and max_bq_sz is not None and bq_sz > max_bq_sz:
            bq_sz = max_bq_sz

        # Step 2b: Step 2 grows bq and bkv together, so when bq hits the VMEM limit
        # bkv stops too. With many q heads per chip (e.g. Qwen3-32B at TP=2, 32 q
        # heads) that happens at 256, while a larger bkv still fits. Keep growing
        # bkv alone, up to a cap, while it fits.
        max_seq_len = self.serve_cfgs.pages_per_seq * self.serve_cfgs.page_size
        if fixed_bq_sz is None:
            step2b_max_bkv_sz = 512
            if max_bkv_sz is not None:
                step2b_max_bkv_sz = min(step2b_max_bkv_sz, max_bkv_sz)
            while (
                bkv_sz + bkv_stride <= max_seq_len
                and bkv_sz + bkv_stride <= step2b_max_bkv_sz
                and self.calculate_vmem_usage(batch_size, bq_sz, bkv_sz + bkv_stride)
                < capped_vmem_limit_bytes
            ):
                bkv_sz += bkv_stride

        # Step 3: Given current tile size, calculate compute tile size.
        # For decode mode (fixed_bq_sz is not None), query chunking (bq_c_sz < bq_sz)
        # is unnecessary because the query tile size is already very small.
        bq_c_sz = (
            bq_sz
            if fixed_bq_sz is not None
            else self.compute_tile_size(batch_size, bq_sz, bkv_sz)
        )
        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
        )

    def decode_block_sizes(self) -> configs.BlockSizes:
        return self.find_best_block_sizes(
            self.decode_batch_size, self.decode_query_size
        )

    def prefill_block_sizes(self) -> configs.BlockSizes:
        return self.find_best_block_sizes(self.prefill_batch_size)

    def calculate(self) -> tuple[configs.BlockSizes, configs.BlockSizes]:
        return self.decode_block_sizes(), self.prefill_block_sizes()


class CPBlockSizeCalculator(BaseBlockSizeCalculator):
    """Block sizes for the context-parallel phases (`cp_group_size` set).

    Uses a tile-aware VMEM model: compute-time temporaries (loaded Q/KV, the QK
    tile) scale with the compute tile `bq_c_sz`, while the Q/O pipeline buffers,
    the LSE buffers and the online-softmax scratch are sized by the full
    `bq_sz`. The scratch is modeled with its real shapes (`m` in the compute
    dtype, `l`/`acc` in float32), plus the live full-block copies the kernel
    keeps until the epilogue (observed as ~bq_sz-sized register spill slots:
    ~1.5x the state for batch_size=1, ~2.5-3x for batch_size=2).

    The model was calibrated in google3 against google3's kernel body on v7x;
    it has not been re-measured against this repo's kernel.
    """

    @property
    def capped_vmem_limit_bytes(self) -> float:
        # The model above tracks the compiler's VMEM usage closely (calibrated on
        # v7x against CP kernels known to compile / OOM), so a 15% headroom for
        # compiler temporaries suffices.
        return self.vmem_limit_bytes * 0.85

    def softmax_state_bytes(self, bq_sz: int) -> int:
        """VMEM bytes of the online-softmax state (m, l, acc) for `bq_sz`.

        Matches `RpaConfigs.lm_scratch_shape` / `acc_scratch_shape`: m and l are
        [num_kv_heads, bq_sz * q_per_kv, num_lanes] and acc is
        [num_kv_heads, bq_sz * q_per_kv, head_dim]; `l` and `acc` are float32 so
        block contributions are not rounded away once `l` grows large at long
        context, `m` stays in the compute dtype (it cancels exactly).
        """
        rows = (
            self.model_cfgs.num_kv_heads * bq_sz * self.aligned_num_q_heads_per_kv_head
        )
        lm_array_size = rows * self.tpu_info.num_lanes
        acc_array_size = rows * self.aligned_head_dim
        return int(
            lm_array_size * self.out_bytes  # m
            + lm_array_size * 4  # l
            + acc_array_size * 4  # acc
        )

    def calculate_vmem_usage(self, batch_size: int, bq_sz: int, bkv_sz: int) -> int:
        bq_c_sz = self.compute_tile_size(batch_size, bq_sz, bkv_sz)
        used = super().calculate_vmem_usage(batch_size, bq_c_sz, bkv_sz)
        q_diff = (
            (bq_sz - bq_c_sz)
            * self.aligned_num_q_heads
            * self.aligned_head_dim
            * (
                configs.NUM_INPUT_SLOTS * self.q_bytes
                + configs.NUM_OUTPUT_SLOTS * self.out_bytes
            )
            * batch_size
        )
        lse_diff = (
            self.lse_buffer_bytes(bq_sz) - self.lse_buffer_bytes(bq_c_sz)
        ) * batch_size
        state = self.softmax_state_bytes(bq_sz)
        scratch_diff = state - self.softmax_state_bytes(bq_c_sz)
        live_state = state * (batch_size + 0.5)
        stitch_bytes = 0
        if (
            self.serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE
            and self.serve_cfgs.attention_scope != configs.AttentionScope.CACHE_ONLY
        ):
            # Lane stitching of new tokens (`stitch_utils._stitch_prefill_lane`)
            # materializes the whole KV slot a few times (load, roll, select,
            # store); the ring's cache phase has no new tokens to stitch.
            vmem_bkv_sz = bkv_sz + 2 * self.tpu_info.num_lanes
            kv_slot_bytes = (
                vmem_bkv_sz
                * self.aligned_num_kv_heads_x2
                * self.aligned_kv_head_dim_val
                * self.kv_bytes
            )
            stitch_bytes = 4 * kv_slot_bytes * batch_size
        return int(used + q_diff + lse_diff + scratch_diff + live_state + stitch_bytes)

    # Single-lane (batch_size=1) CP kernels are modelled tightly enough to run at
    # the ring's 95% cap (see RingBlockSizeCalculator); measured in google3 on
    # v7x for the PCP=8 new-token phase: q256-k512 (model 60.5 MB) compiles,
    # q256-k1024 (model 62.9 MB, real 69.1 MB) does not. Multi-lane tiles keep
    # the 85% cap.
    single_lane_vmem_fraction = 0.95
    single_lane_max_bkv_sz = 512

    def prefill_block_sizes(self) -> configs.BlockSizes:
        blocks = super().prefill_block_sizes()
        if blocks.batch_size != 1:
            return blocks
        cap = self.vmem_limit_bytes * self.single_lane_vmem_fraction
        max_seq_len = self.serve_cfgs.pages_per_seq * self.serve_cfgs.page_size
        bkv_sz = blocks.bkv_sz
        while (
            bkv_sz * 2 <= self.single_lane_max_bkv_sz
            and bkv_sz * 2 <= self.max_bkv_sz
            and bkv_sz * 2 <= max_seq_len
            and self.calculate_vmem_usage(1, blocks.bq_sz, bkv_sz * 2) < cap
        ):
            bkv_sz *= 2
        if bkv_sz == blocks.bkv_sz:
            return blocks
        return configs.BlockSizes(
            bq_sz=blocks.bq_sz,
            bq_c_sz=self.compute_tile_size(1, blocks.bq_sz, bkv_sz),
            bkv_sz=bkv_sz,
            batch_size=1,
        )


class RingBlockSizeCalculator(CPBlockSizeCalculator):
    """Block sizes for the PCP ring's cache phase (`ring_axis_name` set).

    Each Q block walks every local KV block around the ring, so the number of ring
    steps is proportional to num_q_blocks * num_kv_blocks. The search tries bq_sz
    up to the local query length and power-of-two bkv_sz up to
    min(max_bkv_sz, local_kv_len), keeps the pairs that fit in VMEM, and picks
    the one with the least modelled time (`ring_step_time`: the larger of the
    tile compute and the KV hop, times the number of steps), then the fewest
    steps, the least padding, and the largest bkv_sz and bq_sz.
    """

    # One lane per step: the rotation moves the whole buffer in a single copy,
    # so every lane of a step would otherwise have to be on the same hop.
    ring_batch_size = 1

    @property
    def capped_vmem_limit_bytes(self) -> float:
        # For the single-lane ring kernels the model matches the compiler's VMEM
        # usage within a few MB on v7x (google3 measurements): q512-k1024
        # (PCP=4, 16 q heads/KV) and q256-k1024 (PCP=8, 2 KV heads) sit at
        # ~59-60 MB of 64 MB.
        return self.vmem_limit_bytes * 0.95

    @property
    def max_bkv_sz(self) -> int:
        # Capping bkv_sz at 1024 matches the maximum TPU attention tile size
        # and prevents compiler register spills while maximizing compute per step.
        return min(super().max_bkv_sz, 1024)

    # SGLang: cost model of one ring step, fitted on v7x (PCP4 x TP2 and PCP8,
    # Qwen3-30B-A3B, fp8 KV, bkv_sz 1024). A step computes a full
    # bq_sz x bkv_sz tile for every query head (padded Q rows cost as much as
    # real ones) and, overlapped with it, forwards the bkv_sz-token KV block to
    # the next rank; it takes the longer of the two. Measured: the tile at
    # ~370 TFLOP/s per core, the hop at ~60 GB/s (1 MiB in ~17 us). So PCP4 x
    # TP2 (16 q heads, 2 KV heads) is compute-bound at bq_sz 512 and bq_sz 768
    # only adds padding to a 4096-row chunk (9% slower), while PCP8 (32 q heads,
    # 4 KV heads) is DMA-bound below bq_sz ~384 (bq_sz 256: 21% slower).
    ring_tile_flops_per_s = 370e12
    ring_hop_bytes_per_s = 60e9

    def ring_step_time(self, bq_sz: int, bkv_sz: int) -> float:
        """Modelled seconds of one ring step (tile compute vs KV hop)."""
        model_cfgs = self.model_cfgs
        flops = 4 * bq_sz * bkv_sz * model_cfgs.num_q_heads * model_cfgs.head_dim
        hop_bytes = (
            bkv_sz * 2 * model_cfgs.num_kv_heads * model_cfgs.head_dim * self.kv_bytes
        )
        return max(
            flops / self.ring_tile_flops_per_s, hop_bytes / self.ring_hop_bytes_per_s
        )

    def prefill_block_sizes(self) -> configs.BlockSizes:
        serve_cfgs = self.serve_cfgs
        batch_size = self.ring_batch_size
        step_sz = self.mxu_column_size
        # SGLang: K blocks are whole pages (see `bkv_stride_sz`).
        min_bkv_sz = self.bkv_stride_sz

        max_bq = utils.align_to(max(serve_cfgs.total_q_tokens, step_sz), step_sz)

        local_kv_tokens = (
            serve_cfgs.pages_per_seq * serve_cfgs.page_size
            if serve_cfgs.pages_per_seq > 0
            else serve_cfgs.num_page_indices * serve_cfgs.page_size
        )
        total_kv = max(local_kv_tokens, min_bkv_sz)
        max_bkv = min(
            self.max_bkv_sz, max(utils.align_to(total_kv, min_bkv_sz), min_bkv_sz)
        )

        bkv_candidates = []
        bkv = min_bkv_sz
        while bkv <= max(max_bkv, min_bkv_sz):
            bkv_candidates.append(bkv)
            bkv *= 2

        # Q tiles only need lane alignment (rows = bq_sz * q_heads_per_kv_head),
        # so search from one lane tile up; the fewest-steps score still prefers the
        # largest bq_sz that fits, and 128 is the fallback when the float32
        # softmax state does not fit at mxu_column_size.
        bq_step = min(step_sz, self.tpu_info.num_lanes)
        bq_candidates = []
        bq = bq_step
        while bq <= max_bq:
            bq_candidates.append(bq)
            bq += bq_step

        valid_configs = []
        total_q = max(serve_cfgs.total_q_tokens, 1)
        for bq_sz in bq_candidates:
            for bkv_sz in bkv_candidates:
                if self.fits(batch_size, bq_sz, bkv_sz):
                    n_q = pl.cdiv(total_q, bq_sz)
                    n_kv = pl.cdiv(total_kv, bkv_sz)
                    total_steps = n_q * n_kv
                    padding = (n_q * bq_sz - total_q) + (n_kv * bkv_sz - total_kv)
                    # SGLang: rank by the modelled time of the cache phase.
                    # The KV extent is the page-table capacity rather than a
                    # real length, so it enters as total_kv / bkv_sz (no
                    # ceil), and only the Q padding counts as work. Prioritize:
                    # 1. Minimize the modelled time (`ring_step_time`)
                    # 2. Minimize ring schedule steps (n_q * n_kv)
                    # 3. Minimize padding, then maximize bkv_sz and bq_sz
                    est = n_q * (total_kv / bkv_sz) * self.ring_step_time(bq_sz, bkv_sz)
                    score = (round(est, 6), total_steps, padding, -bkv_sz, -bq_sz)
                    valid_configs.append((score, bq_sz, bkv_sz))

        if not valid_configs:
            raise ValueError("Cannot find block sizes that fit within VMEM limit.")

        valid_configs.sort(key=lambda x: x[0])
        _, best_bq_sz, best_bkv_sz = valid_configs[0]

        return configs.BlockSizes(
            bq_sz=best_bq_sz,
            bq_c_sz=self.compute_tile_size(batch_size, best_bq_sz, best_bkv_sz),
            bkv_sz=best_bkv_sz,
            batch_size=batch_size,
        )


def _scheduler_for_scope(
    attention_scope: configs.AttentionScope, cp_group_size: int | None
) -> scheduler_lib.BaseScheduler:
    """The scheduler that implements `attention_scope` (and `cp_group_size`)."""
    if cp_group_size is None:
        return {
            configs.AttentionScope.FULL: scheduler_lib.BaseScheduler,
            configs.AttentionScope.CACHE_ONLY: scheduler_lib.CacheOnlyScheduler,
            configs.AttentionScope.NEW_TOKENS_ONLY: scheduler_lib.NewTokensOnlyScheduler,
        }[attention_scope]()
    return {
        configs.AttentionScope.CACHE_ONLY: scheduler_cp.StripedCacheOnlyScheduler,
        configs.AttentionScope.NEW_TOKENS_ONLY: scheduler_cp.StripedChunkScheduler,
        configs.AttentionScope.STRIPED: scheduler_cp.StripedDecodeScheduler,
    }[attention_scope](cp_group_size=cp_group_size)


@jax.jit(
    static_argnames=(
        "scheduler",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "use_per_token_scale",
        "per_token_scale_dtype",
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
        "update_kv_cache",
        "kv_layout",
        "cache_only",
        "skip_kv_write",
        "return_lse",
        "decode_query_size",
        "cp_group_size",
        "attention_scope",
        "rpa_cases",
        "quantize_p",
        "lane_fold_kv_head_dim",
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
    decode_query_size: int = 1,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    kv_lens_global: jax.Array | None = None,
    attention_scope: configs.AttentionScope | int = configs.AttentionScope.FULL,
    rpa_cases: tuple[configs.RpaCase, ...] | configs.RpaCase | None = None,
    scheduler: scheduler_lib.BaseScheduler | None = None,
    extra_scalars: Sequence[jax.Array] = (),
    use_per_token_scale: bool = False,
    per_token_scale_dtype: jnp.dtype | None = None,
    dynamic_k_scale: jax.Array | None = None,
    dynamic_v_scale: jax.Array | None = None,
    update_kv_cache: bool | None = None,
    quantize_p: bool = False,
    lane_fold_kv_head_dim: bool = False,
    q_positions: jax.Array | None = None,
    kv_cache_lens: jax.Array | None = None,
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
        distribution: [3]. Cumulative sum of the number of mixed, prefill and
            decode sequences: the batch lists its mixed (extend) sequences
            first, ``[0, distribution[0])``, and its decode sequences last,
            ``[distribution[1], distribution[2])``; distribution[2] is the
            total number of sequences. A decode-only batch is ``[0, 0, n]``,
            an extend-only batch ``[n, n, n]``. The prefill-only segment
            ``[distribution[0], distribution[1])`` is handled by the mixed
            kernel call.
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
        kv_layout: Physical layout of `kv_cache`.
        cache_only: Read all kv_lens[s] keys of every sequence from the cache;
            keys/values are ignored, and the queries follow every key, so there
            is no causal mask. kv_lens[s] may be 0 or below q_len. Requires
            HEAD_ALONG_SUBLANE.
        skip_kv_write: Never write the cache. Without cache_only, the last q_len
            keys of each sequence still come from keys/values.
        return_lse: Also return lse[token, q head] = log(sum of exp(logits)),
            natural log, float32, with exp(sink) in the sum when attention_sink
            is given; -inf for a row that attended no key and has no sink.
        decode_query_size: Number of query tokens per sequence in the DECODE
            range (1 by default; higher for speculative decoding).
        cp_group_size: Size of a decode-context-parallel group whose ranks each
            hold the pages p with p % cp_group_size == cp_rank, local page j
            being global page j * cp_group_size + cp_rank. The sliding window
            then runs in global positions. Requires a non-FULL attention_scope.
        cp_rank: [1] int32. This rank's index in the group, traced so that one
            program serves every rank.
        kv_lens_global: [max_num_seqs] int32. Global kv length of each sequence;
            its queries sit at global positions kv_lens_global[s] - q_len onward.
        attention_scope: Which keys the call attends; see
            ``configs.AttentionScope``. NEW_TOKENS_ONLY with cp_group_size
            writes each multi-token sequence's new tokens into the owner
            rank's stripe slots (``kv_lens`` and ``page_indices`` being the
            global lengths and this rank's stripe table).
        rpa_cases: Which of the DECODE / MIXED launches to run (both when
            None); a case whose ``distribution`` range is empty can be skipped.
        scheduler: Schedule strategy instance (``scheduler.BaseScheduler`` or a
            subclass), for callers that drive the kernel through google3's
            interface. When given it replaces the scheduler that
            ``attention_scope`` / ``cp_group_size`` select, and ``kv_lens``,
            ``extra_scalars`` mean what that scheduler says. None (default)
            derives the scheduler from the arguments above.
        extra_scalars: Dynamic scalar arrays forwarded to an explicit
            ``scheduler`` on the host and prefetched into SMEM in the kernels.
        use_per_token_scale: Whether to use per-token quantization
            (SEQ_ALONG_LANE only).
        per_token_scale_dtype: Dtype of per-token scale.
        dynamic_k_scale: Per-token quantization scale value of keys.
        dynamic_v_scale: Per-token quantization scale value of values.
        update_kv_cache: If set, overrides skip_kv_update
            (update_kv_cache=False is the same as skip_kv_update=True).
        quantize_p: If True, cast the softmax probabilities down to the kv cache
            dtype before the pv matmul.
        lane_fold_kv_head_dim: `kv_cache` uses the lane-folded fp8
            HEAD_ALONG_SUBLANE layout of ``get_kv_cache_shape(...,
            lane_fold_kv_head_dim=True)``. Off by default: SGLang's KV pools
            keep the unfolded layout.
        q_positions: [max_num_seqs] int32, only with an explicit ``scheduler``.
            Replaces the query offsets the scheduler derives (the global
            position of each sequence's first query).
        kv_cache_lens: [max_num_seqs] int32, only with an explicit
            ``scheduler``. Replaces the cached lengths the scheduler derives;
            the new length becomes ``kv_lens - kv_cache_lens``, so a sequence
            may have more new keys than queries. Together with
            ``extra_scalars=(cp_rank, new_kv_page_indices[, new_kv_write_rows])``
            and a ``scheduler_cp.StripedNewTokensScheduler`` this is the
            new-token phase of PCP as a kernel launch of its own; the cache
            phase is a launch with ``scheduler_cp.RingCacheScheduler``. Both
            must run inside ``jax.shard_map``; see
            ``ragged_paged_attention_pcp_fused`` for the argument layout and for
            the single-kernel version, which is what serving uses.

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

    # update_kv_cache (as passed by tpu-inference) overrides skip_kv_update when
    # set; update_kv_cache=False (KV-share) is the same as skip_kv_update=True.
    if update_kv_cache is not None:
        skip_kv_update = not update_kv_cache

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

    if k_scale is not None and dynamic_k_scale is not None:
        raise ValueError(
            "Only one of k_scale or dynamic_k_scale can be set. Got"
            f" {k_scale=} and {dynamic_k_scale=}"
        )
    if v_scale is not None and dynamic_v_scale is not None:
        raise ValueError(
            "Only one of v_scale or dynamic_v_scale can be set. Got"
            f" {v_scale=} and {dynamic_v_scale=}"
        )

    if scheduler is None:
        if isinstance(attention_scope, int):
            _scope_map = {
                0: configs.AttentionScope.FULL,
                1: configs.AttentionScope.CACHE_ONLY,
                2: configs.AttentionScope.NEW_TOKENS_ONLY,
                3: configs.AttentionScope.STRIPED,
            }
            attention_scope = _scope_map.get(
                attention_scope, configs.AttentionScope.FULL
            )
        if cache_only:
            attention_scope = configs.AttentionScope.CACHE_ONLY
    else:
        # google3's interface: the scheduler object carries the scope.
        attention_scope = scheduler.scope
        cp_group_size = scheduler.get_cp_group_size()
        cache_only = attention_scope == configs.AttentionScope.CACHE_ONLY
    if scheduler is None and (q_positions is not None or kv_cache_lens is not None):
        raise ValueError("q_positions and kv_cache_lens need an explicit scheduler.")

    # PCP ring attention: the ring cache phase and the paged new-token phase.
    ring_axis_name = getattr(scheduler, "ring_axis_name", None)
    paged_new_kv = (
        isinstance(scheduler, scheduler_cp.StripedNewTokensScheduler)
        and len(extra_scalars) > 1
    )
    if ring_axis_name is not None or paged_new_kv:
        _refuse_unsupported_pcp_features(
            kv_layout=kv_layout,
            sliding_window=sliding_window,
            attention_sink=attention_sink,
            soft_cap=None,
            paged_new_kv=paged_new_kv,
        )

    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        per_token_scale=use_per_token_scale,
        per_token_scale_dtype=per_token_scale_dtype,
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
        return_lse=return_lse,
        skip_kv_update=skip_kv_update,
        quantize_p=quantize_p,
        cache_only=cache_only,
        skip_kv_write=skip_kv_write,
        lane_fold_kv_head_dim=lane_fold_kv_head_dim,
        ring_axis_name=ring_axis_name,
        ring_mesh_axis_names=getattr(scheduler, "ring_mesh_axis_names", None),
        paged_new_kv=paged_new_kv,
        float32_softmax_state=ring_axis_name is not None or paged_new_kv,
    )

    if scheduler is None:
        # SGLang's interface: the scope, the group size and the two CP operands
        # select the scheduler. The kernels only need the rank; the global
        # lengths are folded into the per-sequence lengths on the host.
        configs.validate_cp_inputs(serve_cfgs, kv_lens, cp_rank, kv_lens_global)
        scheduler = _scheduler_for_scope(attention_scope, cp_group_size)
        extra_scalars = () if cp_group_size is None else (cp_rank,)
    elif cp_group_size is not None:
        # Stand-ins for the static checks; the scheduler defines both operands.
        cp_rank = extra_scalars[0]
        if kv_lens_global is None:
            kv_lens_global = kv_lens

    q_hbm, new_kv_hbm = prepare_inputs(
        queries,
        keys,
        values,
        queries.dtype,
        kv_cache.dtype,
        dynamic_k_scale,
        dynamic_v_scale,
        kv_layout=kv_layout,
        page_size=page_size,
        lane_fold_kv_head_dim=lane_fold_kv_head_dim,
    )
    sink_table = (
        None
        if attention_sink is None
        else prepare_attention_sink(attention_sink, num_kv_heads, queries.dtype)
    )

    # Explicit block sizes for both modes are used as given, even where the
    # search would refuse.
    default_decode = default_prefill = None
    if ring_axis_name is not None or paged_new_kv:
        # The PCP phases only launch MIXED and size it with google3's CP
        # calculators.
        if prefill_block_sizes is None:
            calculator_cls = (
                RingBlockSizeCalculator
                if ring_axis_name is not None
                else CPBlockSizeCalculator
            )
            default_prefill = calculator_cls(
                model_cfgs, serve_cfgs, vmem_limit_bytes
            ).prefill_block_sizes()
    elif decode_block_sizes is None or prefill_block_sizes is None:
        default_decode, default_prefill = calculate_block_sizes(
            model_cfgs,
            serve_cfgs,
            vmem_limit_bytes,
            decode_query_size=decode_query_size,
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

    # Compute per-sequence length parameters for the kernel.
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    computed_cache_lens, kv_new_lens, q_offsets = scheduler.get_kv_lens(
        kv_lens,
        q_lens,
        page_size,
        extra_scalars=extra_scalars,
        kv_lens_global=kv_lens_global,
    )
    if kv_cache_lens is not None:
        computed_cache_lens = kv_cache_lens
        kv_new_lens = kv_lens - kv_cache_lens
    if q_positions is not None:
        q_offsets = q_positions
    kv_cache_lens = computed_cache_lens

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
        if cfgs.ring_enabled:
            _validate_ring_configs(cfgs)
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sink=attention_sink,
            cp_rank=cp_rank,
            kv_lens_global=kv_lens_global,
        )

        schedule_hbm = scheduler_lib.generate_rpa_metadata(
            cu_q_lens,
            q_offsets,
            kv_cache_lens,
            kv_new_lens,
            distribution,
            page_indices,
            cfgs=cfgs,
            scheduler=scheduler,
            extra_scalars=extra_scalars,
        )
        return kernel.rpa_kernel(
            cu_q_lens,
            q_offsets,
            kv_cache_lens,
            kv_new_lens,
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            kv_cache,
            lse_hbm,
            sink_table,
            cfgs=cfgs,
            scheduler=scheduler,
            extra_scalars=extra_scalars,
        )

    if ring_axis_name is not None or paged_new_kv:
        active_cases = (configs.RpaCase.MIXED,)
    elif rpa_cases is None:
        active_cases = (configs.RpaCase.DECODE, configs.RpaCase.MIXED)
    elif isinstance(rpa_cases, configs.RpaCase):
        active_cases = (rpa_cases,)
    else:
        active_cases = tuple(rpa_cases)

    o_hbm_alias_q_hbm = q_hbm
    if configs.RpaCase.DECODE in active_cases:
        o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
            configs.RpaCase.DECODE, o_hbm_alias_q_hbm, kv_cache, lse_hbm
        )
    if configs.RpaCase.MIXED in active_cases:
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


# ---------------------------------------------------------------------------
# PCP ring attention (google3 cl/985434096): the fused kernel's wrapper.
# ---------------------------------------------------------------------------


def _refuse_unsupported_pcp_features(
    *,
    kv_layout: configs.KVLayout,
    sliding_window: int | None,
    attention_sink: jax.Array | None,
    soft_cap: float | None,
    paged_new_kv: bool,
):
    """Features the PCP ring / paged new-token kernels do not implement."""
    if sliding_window is not None:
        raise NotImplementedError(
            "PCP ring attention does not support a sliding window: the ring"
            " walks whole stripes and has no window in global positions."
        )
    if attention_sink is not None:
        raise NotImplementedError(
            "PCP ring attention does not support attention sinks."
        )
    if soft_cap is not None:
        raise NotImplementedError(
            "The fused PCP kernel does not support soft-capped logits."
        )
    if paged_new_kv and kv_layout != configs.KVLayout.HEAD_ALONG_SUBLANE:
        raise NotImplementedError(
            "Paged new K/V (the PCP new-token phase) supports HEAD_ALONG_SUBLANE"
            f" only, got {kv_layout}."
        )


def _validate_ring_configs(cfgs: configs.RpaConfigs):
    """Static requirements of a ring (cache phase) kernel."""
    cp_group_size = cfgs.serve.cp_group_size
    if cp_group_size is None or cp_group_size < 2:
        raise ValueError(f"The PCP ring needs cp_group_size >= 2, got {cp_group_size}.")
    if cfgs.batch_size != 1:
        raise ValueError(
            "The PCP ring rotates one KV block per step and needs batch_size == 1,"
            f" got {cfgs.batch_size}."
        )
    if cfgs.max_steps_ub % cp_group_size != 0:
        # A schedule chunk has `max_steps_ub` steps (a multiple of 128) and the
        # pipeline restarts its slot rotation at every chunk, so a chunk must
        # end on the last hop of a rotation.
        raise ValueError(
            f"The PCP ring needs cp_group_size to divide {cfgs.max_steps_ub=};"
            f" use a power of two, got {cp_group_size}."
        )


@dataclasses.dataclass(frozen=True)
class PcpFusedConfigs:
    """Static configuration of one fused PCP kernel: both phases."""

    cfgs_cache: configs.RpaConfigs
    cfgs_new: configs.RpaConfigs
    cache_scheduler: scheduler_cp.RingCacheScheduler
    new_scheduler: scheduler_cp.StripedNewTokensScheduler


def pcp_fused_configs(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_q: jnp.dtype,
    dtype_kv: jnp.dtype,
    page_size: int,
    cp_group_size: int,
    total_q_tokens: int,
    num_cache_rows: int,
    cache_pages_per_row: int,
    num_new_rows: int,
    new_pages_per_row: int,
    ring_axis_name: str = "pcp",
    ring_mesh_axis_names: tuple[str, ...] | None = None,
    sm_scale: float = 1.0,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    out_dtype: jnp.dtype | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    token_major_q: bool = False,
    update_kv_cache: bool = False,
    cache_prefill_block_sizes: configs.BlockSizes | None = None,
    new_prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    collective_id: int = 3,
    cache_lens_cp_page_aligned: bool = False,
) -> PcpFusedConfigs:
    """Resolves the schedulers, block sizes and configs of a fused PCP kernel.

    Everything here is static (shapes, dtypes, flags), so the same call gives
    the host what `ragged_paged_attention_pcp_fused` will run with: it is the
    one place the fused kernel's block sizes are chosen. Needs the TPU info
    (a TPU, or `sglang_tpu.pallas_export.tpu_info_for_cpu_export`).

    Block sizes: the ring picks its tile (`RingBlockSizeCalculator`); the
    new-token phase shares the Q block and the (larger) KV window, so it keeps
    bq and uses the smaller of its own (`CPBlockSizeCalculator`) and the
    ring's bkv.

    `cache_lens_cp_page_aligned` (SGLang): the caller guarantees that every
    row's cached length (`kv_cache_lens`) is a multiple of page_size * cp,
    i.e. whole stripes, as SGLang's ring mode does (its allocator page and
    prefill chunk boundaries are page_size * cp tokens). The new-token phase
    then skips the mask of cached keys (see
    `ServingConfigs.cached_lens_block_aligned`).
    """
    if kv_layout != configs.KVLayout.HEAD_ALONG_SUBLANE:
        raise NotImplementedError(
            f"Fused PCP supports HEAD_ALONG_SUBLANE only, got {kv_layout}."
        )
    if num_cache_rows < 1 or num_new_rows < 1:
        raise ValueError(
            f"Fused PCP needs at least one row per phase, got {num_cache_rows=},"
            f" {num_new_rows=}."
        )
    if cache_pages_per_row < 1 or new_pages_per_row < 1:
        raise ValueError(
            "Fused PCP needs page tables with at least one column, got"
            f" {cache_pages_per_row=}, {new_pages_per_row=}."
        )
    dtype_q = jnp.dtype(dtype_q)
    dtype_kv = jnp.dtype(dtype_kv)
    out_dtype = dtype_q if out_dtype is None else jnp.dtype(out_dtype)
    if mask_value is None:
        mask_value = float(jnp.finfo(out_dtype).min)
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sm_scale=sm_scale,
        mask_value=mask_value,
    )
    cache_scheduler = scheduler_cp.RingCacheScheduler(
        cp_group_size=cp_group_size,
        ring_axis_name=ring_axis_name,
        ring_mesh_axis_names=ring_mesh_axis_names,
        collective_id=collective_id,
    )
    new_scheduler = scheduler_cp.StripedNewTokensScheduler(cp_group_size=cp_group_size)

    def _serve_cfgs(scheduler, num_seqs, pages_per_row, paged_new_kv):
        return configs.ServingConfigs(
            num_seqs=num_seqs,
            num_page_indices=num_seqs * pages_per_row,
            total_q_tokens=total_q_tokens,
            dtype_q=dtype_q,
            dtype_kv=dtype_kv,
            dtype_out=out_dtype,
            page_size=page_size,
            scale_k=k_scale,
            scale_v=v_scale,
            kv_layout=kv_layout,
            cp_group_size=cp_group_size,
            attention_scope=scheduler.scope,
            ring_axis_name=getattr(scheduler, "ring_axis_name", None),
            ring_mesh_axis_names=getattr(scheduler, "ring_mesh_axis_names", None),
            paged_new_kv=paged_new_kv,
            # Phase 1 hands its LSE over inside the kernel; no LSE output.
            return_lse=False,
            skip_kv_update=False,
            skip_kv_write=not update_kv_cache,
            token_major_q=token_major_q,
            float32_softmax_state=True,
        )

    serve_cache = _serve_cfgs(
        cache_scheduler, num_cache_rows, cache_pages_per_row, False
    )
    serve_new = _serve_cfgs(new_scheduler, num_new_rows, new_pages_per_row, True)

    cache_blocks = cache_prefill_block_sizes
    if cache_blocks is None:
        cache_blocks = RingBlockSizeCalculator(
            model_cfgs, serve_cache, vmem_limit_bytes
        ).prefill_block_sizes()
    new_blocks = new_prefill_block_sizes
    if new_blocks is None:
        new_calc = CPBlockSizeCalculator(model_cfgs, serve_new, vmem_limit_bytes)
        bkv_new = min(new_calc.prefill_block_sizes().bkv_sz, cache_blocks.bkv_sz)
        new_blocks = configs.BlockSizes(
            bq_sz=cache_blocks.bq_sz,
            bq_c_sz=new_calc.compute_tile_size(1, cache_blocks.bq_sz, bkv_new),
            bkv_sz=bkv_new,
            batch_size=1,
        )

    if (
        cache_lens_cp_page_aligned
        and (page_size * cp_group_size) % new_blocks.bkv_sz == 0
    ):
        # Cached lengths are whole stripes (multiples of page_size * cp), so
        # also multiples of the new-token phase's bkv_sz: skip its start mask.
        serve_new = dataclasses.replace(serve_new, cached_lens_block_aligned=True)

    def _cfgs(blocks, serve_cfgs):
        return configs.RpaConfigs(
            block=blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=configs.RpaCase.MIXED,
        )

    cfgs_cache = _cfgs(cache_blocks, serve_cache)
    cfgs_new = _cfgs(new_blocks, serve_new)
    _validate_ring_configs(cfgs_cache)
    for name, blocks in (("cache", cache_blocks), ("new", new_blocks)):
        if blocks.bkv_sz % page_size or blocks.bq_sz % blocks.bq_c_sz:
            raise ValueError(
                f"Fused PCP {name}-phase blocks need bkv_sz to be a whole number"
                f" of pages and bq_c_sz to divide bq_sz, got {blocks}."
            )
    if (
        cache_blocks.bq_sz != new_blocks.bq_sz
        or new_blocks.batch_size != 1
        or new_blocks.bkv_sz > cache_blocks.bkv_sz
    ):
        raise ValueError(
            "Fused PCP needs one bq_sz for both phases, batch_size == 1 and"
            f" bkv_sz(new) <= bkv_sz(cache), got {cache_blocks=}, {new_blocks=}."
        )
    return PcpFusedConfigs(
        cfgs_cache=cfgs_cache,
        cfgs_new=cfgs_new,
        cache_scheduler=cache_scheduler,
        new_scheduler=new_scheduler,
    )


def pcp_fused_schedule_capacity(fused: PcpFusedConfigs) -> tuple[int, int]:
    """Steps the (cache phase, new-token phase) schedules of a fused kernel hold.

    A phase whose schedule needs more steps than this overflows the schedule
    buffer: the kernel then silently drops the excess steps. Same accounting
    as `sglang_tpu.pp_attention_capacity` uses for the other kernels.
    """
    return tuple(
        max(
            0,
            cfgs.max_steps_ub * cfgs.max_schedule_size_multiplier
            - configs.NUM_INPUT_SLOTS
            - 1,
        )
        for cfgs in (fused.cfgs_cache, fused.cfgs_new)
    )


def pcp_fused_schedule_steps(
    fused: PcpFusedConfigs,
    *,
    cache_cu_q_lens,
    cache_kv_lens,
    num_cache_rows_live: int,
    new_cu_q_lens,
    new_kv_lens,
    kv_cache_lens,
    q_positions,
    num_new_rows_live: int,
    new_kv_write_rows=None,
) -> tuple[int, int]:
    """Steps the (cache phase, new-token phase) schedules of a batch need.

    Host-side mirror of `RingCacheScheduler` and `StripedNewTokensScheduler`
    (paged new K/V); the arguments are the host (NumPy / list) values of the
    operands of `ragged_paged_attention_pcp_fused` with the same names, and
    the live row counts are `cache_distribution[1]` / `new_distribution[1]`.
    """
    cfgs_cache, cfgs_new = fused.cfgs_cache, fused.cfgs_new
    cp_group_size = cfgs_cache.serve.cp_group_size
    page_size = cfgs_cache.serve.page_size
    super_page = cp_group_size * page_size
    bq_sz = cfgs_cache.bq_sz

    def cdiv(a, b):
        return -(-int(a) // int(b))

    cache_steps = 0
    for row in range(int(num_cache_rows_live)):
        q_len = int(cache_cu_q_lens[row + 1]) - int(cache_cu_q_lens[row])
        cache_len = int(cache_kv_lens[row]) - q_len
        # Rank 0's stripe: `utils.cp_local_cache_len(cache_len, cp, 0, page)`.
        local_len = (cache_len // super_page) * page_size + min(
            page_size, cache_len % super_page
        )
        cache_steps += (
            cdiv(q_len, bq_sz) * cdiv(local_len, cfgs_cache.bkv_sz) * cp_group_size
        )

    writes = not cfgs_new.serve.skip_kv_write
    bkv_sz = cfgs_new.bkv_sz
    new_steps = 0
    for row in range(int(num_new_rows_live)):
        q_len = int(new_cu_q_lens[row + 1]) - int(new_cu_q_lens[row])
        cache_len = int(kv_cache_lens[row])
        num_q = cdiv(q_len, bq_sz)
        num_k = cdiv(int(new_kv_lens[row]), bkv_sz)
        start_k = cache_len // bkv_sz
        row_writes = writes and (
            new_kv_write_rows is None or int(new_kv_write_rows[row]) != 0
        )
        for q_idx in range(num_q):
            q_sz = min(bq_sz, q_len - q_idx * bq_sz)
            last_pos = int(q_positions[row]) + q_idx * bq_sz + q_sz - 1
            end_k = min(num_k, last_pos // bkv_sz + 1)
            if row_writes and q_idx == num_q - 1:
                end_k = num_k
            new_steps += max(0, end_k - start_k)
    return cache_steps, new_steps


@functools.lru_cache(maxsize=64)
def _cached_pcp_fused_configs(tpu_info, **kwargs) -> PcpFusedConfigs:
    """`pcp_fused_configs` memoized on its (static) arguments and the chip.

    Resolving the block sizes takes ~10 ms of host time, and
    `check_pcp_fused_schedule` runs on every ring-mode extend batch before the
    forward is launched, so an uncached call adds that to every prefill chunk.
    `tpu_info` is only part of the key: the block sizes depend on the chip.
    """
    del tpu_info
    return pcp_fused_configs(**kwargs)


def check_pcp_fused_schedule(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_q,
    dtype_kv,
    page_size: int,
    cp_group_size: int,
    total_q_tokens: int,
    total_kv_tokens: int,
    cache_cu_q_lens,
    cache_kv_lens,
    cache_pages_per_row: int,
    new_cu_q_lens,
    new_kv_lens,
    kv_cache_lens,
    q_positions,
    new_pages_per_row: int,
    new_kv_pages_per_row: int,
    num_cache_rows_live: int,
    num_new_rows_live: int,
    new_kv_write_rows=None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    update_kv_cache: bool = False,
    cache_prefill_block_sizes: configs.BlockSizes | None = None,
    new_prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    cache_lens_cp_page_aligned: bool = False,
) -> None:
    """Raises ValueError if a batch does not fit the fused PCP kernel.

    Host-side: static shapes plus the host values of the length operands; no
    device is touched. Checks, with the block sizes the kernel will pick for
    these shapes (`pcp_fused_configs`), that neither phase's schedule
    overflows, and the static layout rules of `keys` / `values` and of the row
    starts. `cache_cu_q_lens` has `num_cache_rows + 1` entries and
    `new_cu_q_lens` has `num_new_rows + 1` (the operands' static sizes).
    """
    num_cache_rows = len(cache_cu_q_lens) - 1
    num_new_rows = len(new_cu_q_lens) - 1
    if total_kv_tokens % page_size or total_kv_tokens < total_q_tokens:
        raise ValueError(
            f"Fused PCP needs a whole number of pages of new K/V and at least as"
            f" many as queries, got {total_kv_tokens=}, {total_q_tokens=},"
            f" {page_size=}."
        )
    if new_kv_pages_per_row < 1:
        raise ValueError(f"Expected {new_kv_pages_per_row=} >= 1.")
    for name, cu, live in (
        ("cache", cache_cu_q_lens, num_cache_rows_live),
        ("new", new_cu_q_lens, num_new_rows_live),
    ):
        starts = [int(cu[row]) for row in range(int(live))]
        if any(start % 16 for start in starts):
            raise ValueError(
                f"Fused PCP {name}-phase rows must start at multiples of 16"
                f" query tokens, got starts {starts}."
            )
        if int(live) and int(cu[int(live)]) > total_q_tokens:
            raise ValueError(
                f"Fused PCP {name}-phase rows cover {int(cu[int(live)])} query"
                f" tokens, more than {total_q_tokens=}."
            )
    if cache_lens_cp_page_aligned:
        stripe = page_size * cp_group_size
        bad = [
            int(kv_cache_lens[row])
            for row in range(int(num_new_rows_live))
            if int(kv_cache_lens[row]) % stripe
        ]
        if bad:
            raise ValueError(
                "Fused PCP with cache_lens_cp_page_aligned needs cached lengths"
                f" in whole stripes of {stripe} tokens, got {bad}."
            )
    fused = _cached_pcp_fused_configs(
        pltpu.get_tpu_info(),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        page_size=page_size,
        cp_group_size=cp_group_size,
        total_q_tokens=total_q_tokens,
        num_cache_rows=num_cache_rows,
        cache_pages_per_row=cache_pages_per_row,
        num_new_rows=num_new_rows,
        new_pages_per_row=new_pages_per_row,
        kv_layout=kv_layout,
        update_kv_cache=update_kv_cache,
        cache_prefill_block_sizes=cache_prefill_block_sizes,
        new_prefill_block_sizes=new_prefill_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        cache_lens_cp_page_aligned=cache_lens_cp_page_aligned,
    )
    needed = pcp_fused_schedule_steps(
        fused,
        cache_cu_q_lens=cache_cu_q_lens,
        cache_kv_lens=cache_kv_lens,
        num_cache_rows_live=num_cache_rows_live,
        new_cu_q_lens=new_cu_q_lens,
        new_kv_lens=new_kv_lens,
        kv_cache_lens=kv_cache_lens,
        q_positions=q_positions,
        num_new_rows_live=num_new_rows_live,
        new_kv_write_rows=new_kv_write_rows,
    )
    capacity = pcp_fused_schedule_capacity(fused)
    for name, steps, cap, cfgs in zip(
        ("cache (ring)", "new-token"),
        needed,
        capacity,
        (fused.cfgs_cache, fused.cfgs_new),
    ):
        if steps > cap:
            raise ValueError(
                f"The fused PCP {name} phase needs {steps} schedule steps but"
                f" holds {cap} (bq_sz={cfgs.bq_sz}, bkv_sz={cfgs.bkv_sz},"
                f" {cp_group_size=})."
            )


@jax.jit(
    static_argnames=(
        "cp_group_size",
        "ring_axis_name",
        "ring_mesh_axis_names",
        "sm_scale",
        "mask_value",
        "k_scale",
        "v_scale",
        "out_dtype",
        "kv_layout",
        "token_major_q",
        "update_kv_cache",
        "cache_prefill_block_sizes",
        "new_prefill_block_sizes",
        "vmem_limit_bytes",
        "collective_id",
        "sliding_window",
        "soft_cap",
        "cache_lens_cp_page_aligned",
    ),
)
def ragged_paged_attention_pcp_fused(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    *,
    # Cache (ring) phase inputs, as passed to `ragged_paged_attention` with a
    # `RingCacheScheduler`.
    cache_kv_lens: jax.Array,
    cache_page_indices: jax.Array,
    cache_cu_q_lens: jax.Array,
    cache_distribution: jax.Array,
    # New-token phase inputs, as passed with a `StripedNewTokensScheduler`.
    new_kv_lens: jax.Array,
    new_page_indices: jax.Array,
    new_cu_q_lens: jax.Array,
    new_distribution: jax.Array,
    new_kv_page_indices: jax.Array,
    q_positions: jax.Array,
    kv_cache_lens: jax.Array,
    cp_rank: jax.Array,
    cp_group_size: int,
    ring_axis_name: str,
    ring_mesh_axis_names: tuple[str, ...] | None,
    new_kv_write_rows: jax.Array | None = None,
    sm_scale: float = 1.0,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    out_dtype: jnp.dtype | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    token_major_q: bool = False,
    update_kv_cache: bool = False,
    cache_prefill_block_sizes: configs.BlockSizes | None = None,
    new_prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    collective_id: int = 3,
    attention_sink: jax.Array | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    cache_lens_cp_page_aligned: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """PCP prefill in one kernel: ring cache phase + new-token phase.

    Prefill context parallelism over `cp_group_size` ranks whose KV caches are
    page-striped (rank r holds the pages p of a sequence with
    p % cp_group_size == r, as under decode context parallelism). Each rank
    holds some of a chunk's queries and attends (1) the cached prefix, by
    rotating every rank's stripe around the ring (remote DMAs of KV blocks),
    and (2) the chunk's own tokens, whose K/V the caller has all-gathered, in
    one `pallas_call`: the new-token phase continues the cache phase's online
    softmax (see `kernel.rpa_kernel_pcp_fused`), so there is no second launch,
    no LSE output and no merge.

    Must be called inside `jax.shard_map` over a mesh with the ring axis, by
    every rank, with identical static arguments. It contains no collective of
    its own.

    Rows ("sequences") of the two phases index the same query array but are
    independent partitions of it: the hand-off between the phases is stored
    per query token. Every row must start at a multiple of 16 query tokens.

    Args:
      queries: [T_q, num_q_heads, head_dim]. This rank's queries.
      keys, values: [T_kv, num_kv_heads, head_dim] in the cache dtype. The new
        K/V of the chunk, read only through `new_kv_page_indices`; T_kv is a
        whole number of pages and at least T_q.
      kv_cache: This rank's page stripe of the cache, in the layout of
        `get_kv_cache_shape`.
      cache_kv_lens: [S_c] int32. Per cache row, the global cached length plus
        the row's query length (the scheduler subtracts the latter).
      cache_page_indices: [S_c * pages] int32, pages >= 1. Stripe page table of
        each cache row. Entries up to rank 0's stripe length must be valid on
        every rank.
      cache_cu_q_lens: [S_c + 1] int32. Cumulative query lengths of the cache
        rows.
      cache_distribution: [3] int32, SGLang order; `[n, n, n]` for n live rows.
      new_kv_lens: [S_n] int32. Per new row, the global length after the
        chunk: cached length plus the number of new tokens of its sequence.
      new_page_indices: [S_n * pages] int32. Stripe page table of each new row
        (writeback destination); unused unless `update_kv_cache`.
      new_cu_q_lens: [S_n + 1] int32. Cumulative query lengths of the new rows.
      new_distribution: [3] int32, `[n, n, n]` for n live rows.
      new_kv_page_indices: [S_n * new_pages] int32. Per new row, page j of its
        sequence's new tokens (counted from the first new token) is page
        `new_kv_page_indices[row, j]` of keys/values. Every entry must be a
        valid page of keys/values.
      q_positions: [S_n] int32. Global position of each new row's first query.
      kv_cache_lens: [S_n] int32. Global cached length of each new row.
      cp_rank: [1] int32. This rank's index in the group. Must increase by one
        (mod cp_group_size) along the ring axis.
      cp_group_size: Number of ranks; a power of two.
      ring_axis_name: Mesh axis the ring runs on.
      ring_mesh_axis_names: All axis names of the mesh, in mesh order; None
        when the ring axis is the only one.
      new_kv_write_rows: [S_n] int32 or None, only with `update_kv_cache`.
        Nonzero entries mark the rows that write their sequence's new K/V;
        None marks every row. See `StripedNewTokensScheduler`.
      sm_scale: Softmax scale.
      mask_value: Value of masked logits; defaults to the smallest value of the
        output dtype.
      k_scale, v_scale: Static quantization scales of the cache dtype (same
        meaning as in `ragged_paged_attention`).
      out_dtype: Output dtype; defaults to the queries' dtype.
      kv_layout: HEAD_ALONG_SUBLANE only.
      token_major_q: Keep q / out token-major in HBM; see
        `ServingConfigs.token_major_q`.
      update_kv_cache: False (default): `kv_cache` is read-only and the kernel
        writes nothing; the caller writes each rank's owned pages of the new
        K/V itself, before or after this call. True: the kernel writes them
        and returns the cache; the cache is then aliased by a kernel that
        contains remote DMAs, which XLA may copy on every call.
      cache_prefill_block_sizes, new_prefill_block_sizes: Explicit block sizes
        of the two phases (same bq_sz, batch_size 1, bkv_sz(new) <=
        bkv_sz(cache)).
      vmem_limit_bytes: VMEM limit of the kernel.
      collective_id: Pallas collective id of the kernel (barrier semaphore);
        unique among the kernels of the process that synchronize devices.
      attention_sink, sliding_window, soft_cap: Not supported; must be None.
      cache_lens_cp_page_aligned: SGLang. The caller guarantees every
        `kv_cache_lens` entry is a multiple of page_size * cp_group_size; see
        `pcp_fused_configs`.

    Returns:
      `out` [T_q, num_q_heads, head_dim], or `(out, kv_cache)` with
      `update_kv_cache`. Query rows that no new row covers keep the query
      values.
    """
    _refuse_unsupported_pcp_features(
        kv_layout=kv_layout,
        sliding_window=sliding_window,
        attention_sink=attention_sink,
        soft_cap=soft_cap,
        paged_new_kv=True,
    )
    if out_dtype is None:
        out_dtype = queries.dtype

    page_size = kv_cache.shape[1]
    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    num_kv_heads = keys.shape[1]
    num_cache_rows = cache_kv_lens.shape[0]
    num_new_rows = new_kv_lens.shape[0]
    for name, pages, rows in (
        ("cache_page_indices", cache_page_indices, num_cache_rows),
        ("new_page_indices", new_page_indices, num_new_rows),
        ("new_kv_page_indices", new_kv_page_indices, num_new_rows),
    ):
        if pages.ndim != 1 or pages.shape[0] % rows or pages.shape[0] < rows:
            raise ValueError(
                f"Expected {name} to hold at least one page per row for {rows}"
                f" rows, got shape {pages.shape}."
            )

    fused = pcp_fused_configs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        page_size=page_size,
        cp_group_size=cp_group_size,
        total_q_tokens=queries.shape[0],
        num_cache_rows=num_cache_rows,
        cache_pages_per_row=cache_page_indices.shape[0] // num_cache_rows,
        num_new_rows=num_new_rows,
        new_pages_per_row=new_page_indices.shape[0] // num_new_rows,
        ring_axis_name=ring_axis_name,
        ring_mesh_axis_names=ring_mesh_axis_names,
        sm_scale=sm_scale,
        mask_value=mask_value,
        k_scale=k_scale,
        v_scale=v_scale,
        out_dtype=out_dtype,
        kv_layout=kv_layout,
        token_major_q=token_major_q,
        update_kv_cache=update_kv_cache,
        cache_prefill_block_sizes=cache_prefill_block_sizes,
        new_prefill_block_sizes=new_prefill_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        collective_id=collective_id,
        cache_lens_cp_page_aligned=cache_lens_cp_page_aligned,
    )
    cfgs_cache, cfgs_new = fused.cfgs_cache, fused.cfgs_new

    # Static checks. The new-token phase's are the kernel's usual ones; the
    # cache phase shares the arrays and only adds its own operands.
    cfgs_new.validate_inputs(
        q=queries,
        k=keys,
        v=values,
        kv_cache=kv_cache,
        kv_lens=new_kv_lens,
        page_indices=new_page_indices,
        cu_q_lens=new_cu_q_lens,
        distribution=new_distribution,
        cp_rank=cp_rank,
        kv_lens_global=new_kv_lens,
    )
    int_operands = {
        "cache_kv_lens": (cache_kv_lens, (num_cache_rows,)),
        "cache_cu_q_lens": (cache_cu_q_lens, (num_cache_rows + 1,)),
        "cache_distribution": (cache_distribution, (3,)),
        "cache_page_indices": (cache_page_indices, cache_page_indices.shape),
        "new_kv_page_indices": (new_kv_page_indices, new_kv_page_indices.shape),
        "q_positions": (q_positions, (num_new_rows,)),
        "kv_cache_lens": (kv_cache_lens, (num_new_rows,)),
    }
    if new_kv_write_rows is not None:
        if not update_kv_cache:
            raise ValueError("new_kv_write_rows requires update_kv_cache=True.")
        int_operands["new_kv_write_rows"] = (new_kv_write_rows, (num_new_rows,))
    for name, (operand, shape) in int_operands.items():
        if operand.dtype != jnp.int32 or operand.shape != shape:
            raise ValueError(
                f"Expected int32 {name} of shape {shape}, got {operand.dtype}"
                f" {operand.shape}."
            )

    q_hbm, new_kv_hbm = prepare_inputs(
        queries,
        keys,
        values,
        queries.dtype,
        kv_cache.dtype,
        kv_layout=kv_layout,
        page_size=page_size,
        token_major_q=token_major_q,
    )

    def _phase(
        scheduler,
        cfgs,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        extra_scalars,
        q_offsets_override,
        kv_cache_lens_override,
    ):
        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        computed_cache_lens, kv_new_lens, q_offsets = scheduler.get_kv_lens(
            kv_lens, q_lens, page_size, extra_scalars=extra_scalars
        )
        if kv_cache_lens_override is not None:
            computed_cache_lens = kv_cache_lens_override
            kv_new_lens = kv_lens - kv_cache_lens_override
        if q_offsets_override is not None:
            q_offsets = q_offsets_override
        schedule_hbm = scheduler_lib.generate_rpa_metadata(
            cu_q_lens,
            q_offsets,
            computed_cache_lens,
            kv_new_lens,
            distribution,
            page_indices,
            cfgs=cfgs,
            scheduler=scheduler,
            extra_scalars=extra_scalars,
        )
        scalars = (
            cu_q_lens,
            q_offsets,
            computed_cache_lens,
            kv_new_lens,
            extra_scalars,
        )
        return schedule_hbm, scalars

    schedule_cache, cache_scalars = _phase(
        fused.cache_scheduler,
        cfgs_cache,
        cache_kv_lens,
        cache_page_indices,
        cache_cu_q_lens,
        cache_distribution,
        (cp_rank,),
        None,
        None,
    )
    new_extra_scalars = (cp_rank, new_kv_page_indices)
    if new_kv_write_rows is not None:
        new_extra_scalars += (new_kv_write_rows,)
    schedule_new, new_scalars = _phase(
        fused.new_scheduler,
        cfgs_new,
        new_kv_lens,
        new_page_indices,
        new_cu_q_lens,
        new_distribution,
        new_extra_scalars,
        q_positions,
        kv_cache_lens,
    )

    o_hbm, kv_cache_out = kernel.rpa_kernel_pcp_fused(
        cache_scalars,
        new_scalars,
        schedule_cache,
        schedule_new,
        q_hbm,
        new_kv_hbm,
        kv_cache,
        cfgs_cache=cfgs_cache,
        cfgs_new=cfgs_new,
        cache_scheduler=fused.cache_scheduler,
        new_scheduler=fused.new_scheduler,
    )

    o_hbm = prepare_outputs(o_hbm)
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    if not token_major_q:
        o_hbm = o_hbm.swapaxes(1, 0)
    o_hbm = o_hbm.reshape(queries.shape)
    if update_kv_cache:
        return o_hbm, kv_cache_out
    return o_hbm
