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

from sglang_tpu.attention.pallas.batched_rpa import (
    configs,
    kernel,
    schedule,
    utils,
)
from sglang_tpu.envs import envs


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


# The q and KV refs pipeline with lookahead, which needs a second buffer.
_MIN_N_BUFFER = 2

# Smallest q block the search shrinks to before dropping buffers; the smallest
# measured to compile and run on the batched kernel.
_MIN_SHRUNK_BQ_SZ = 128

# Headroom below the VMEM limit for _spill_bytes: the fit under-predicts Mosaic's
# usage by up to 4.8 MiB (1.4 MiB on transposed tiles, none on decode), and
# Mosaic's limit is 0.06 MiB under 64 MiB; the rest is slack for unfitted shapes.
_VMEM_MARGIN_BYTES = 6 << 20


def _carry_bytes(cfgs: configs.RpaConfigs) -> int:
    """The m, l and acc scratch: f32, shared by the batch's lanes."""
    tpu_info = pltpu.get_tpu_info()
    if cfgs.use_transposed:
        # m is one row per KV head, padded to a sublane tile.
        return (
            math.prod(cfgs.mt_scratch_shape) * tpu_info.num_sublanes * 4
            + math.prod(cfgs.acct_scratch_shape) * 4
            + math.prod(cfgs.qt_scratch_shape) * jnp.dtype(cfgs.serve.dtype_kv).itemsize
        )
    *acc_rows, acc_width = cfgs.acc_scratch_shape
    acc_bytes = math.prod(acc_rows) * utils.align_to(acc_width, tpu_info.num_lanes) * 4
    return 2 * math.prod(cfgs.lm_scratch_shape) * 4 + acc_bytes


def _lse_bytes(cfgs: configs.RpaConfigs) -> int:
    """f32 rows of lanes: two output buffers and the packed LSE math per (KV head,
    token), and the per-(token, q head) staging; plus m per lane. Packed-array
    counts fit to gpt-oss DCP2/DCP8 compiles; fewer let the DCP8 chunk pass
    (8 q over 8 KV heads, sinks) overflow VMEM."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_kv_heads = cfgs.model.num_kv_heads
    q_heads = cfgs.aligned_num_q_heads_per_kv_head
    num_packed = 8 if cfgs.model.has_attention_sink else 3
    lse_rows = (2 + num_packed) * num_kv_heads + q_heads
    per_token = lse_rows * num_lanes * 4
    per_token += (
        num_kv_heads * q_heads * num_lanes * jnp.dtype(cfgs.serve.dtype_out).itemsize
    )
    return per_token * cfgs.bq_sz * cfgs.batch_size


def _spill_bytes(cfgs: configs.RpaConfigs) -> int:
    """VMEM Mosaic adds on top of buffers and carries (register spills, temporaries).

    Least-squares fit on tpu7x to Mosaic's reported usage of the v11 kernel over
    ~150 tiles of 12 geometries (both KV layouts, fp8/bf16 KV, 1-16 KV heads,
    sinks, windows); refit when the kernel's compute changes.
    """
    num_kv_heads = cfgs.model.num_kv_heads
    if cfgs.mode == configs.RpaCase.DECODE:
        spill = (48 << 10) * cfgs.batch_size * num_kv_heads
        if (
            cfgs.serve.kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE
            and jnp.dtype(cfgs.serve.dtype_kv).itemsize == 1
        ):
            # Packed fp8 K is unpacked to a bf16 copy of the tile.
            spill += int(
                1.1
                * cfgs.batch_size
                * cfgs.bkv_sz
                * num_kv_heads
                * cfgs.aligned_kv_head_dim
                * 2
            )
        return spill
    q_rows = (
        cfgs.batch_size
        * num_kv_heads
        * cfgs.bq_sz
        * cfgs.aligned_num_q_heads_per_kv_head
    )
    rows_f32 = q_rows * cfgs.aligned_q_head_dim * 4
    scores_f32 = q_rows * cfgs.bkv_sz * 4
    if cfgs.use_transposed:
        keys_bf16 = (
            cfgs.batch_size * num_kv_heads * cfgs.bkv_sz * cfgs.aligned_q_head_dim * 2
        )
        spill = 0.48 * rows_f32 + 1.17 * keys_bf16 + 0.043 * scores_f32
        lanes = cfgs.bq_c_sz * cfgs.aligned_num_q_heads_per_kv_head
        if not cfgs.serve.no_max and lanes > pltpu.get_tpu_info().num_lanes:
            # The running max over a compute block wider than a vreg spills its
            # bf16 scores: about half an f32 copy of a KV head's q-block scores.
            spill += 0.48 * scores_f32 / num_kv_heads
        if cfgs.model.sliding_window is not None:
            # The window mask: about one f32 copy of a KV head's q-block scores.
            spill += 0.91 * scores_f32 / num_kv_heads
        return int(spill)
    chunk_scores_f32 = scores_f32 // cfgs.bq_sz * cfgs.bq_c_sz
    spill = (
        1.41 * rows_f32
        + 0.23 * scores_f32
        + 0.83 * chunk_scores_f32
        + 2.59 * chunk_scores_f32 / num_kv_heads
    )
    if cfgs.model.has_attention_sink:
        spill += 0.78 * rows_f32
    return int(spill)


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill."""

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    mxu_column_size = tpu_info.mxu_column_size

    aligned_num_q_heads_per_kv_head = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
    )
    q_bytes = jnp.dtype(serve_cfgs.dtype_q).itemsize
    kv_bytes = jnp.dtype(serve_cfgs.dtype_kv).itemsize
    out_bytes = jnp.dtype(serve_cfgs.dtype_out).itemsize
    usable_vmem_bytes = vmem_limit_bytes - _VMEM_MARGIN_BYTES

    def compute_chunk_size(batch_size: int, bq_sz: int, bkv_sz: int) -> int:
        # Each compute sub-tile reshapes Q to (bq_c_sz * aligned q heads per KV
        # head, head_dim): a power-of-2 divisor of bq_sz, at least a lane tile.
        # Fixed threshold based on hardware spec.
        # TODO(kyuyeunk): Use different threshold based on hardware and precision.
        threshold = 1500
        min_bq_c_sz = min(bq_sz, max(1, num_lanes // aligned_num_q_heads_per_kv_head))
        best_bq_c_sz = bq_sz
        for num_bq_c in range(1, min(bq_sz, 32) + 1):
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
            num_muls = (
                batch_size
                * bq_c_sz
                * pl.cdiv(bkv_sz, mxu_column_size)
                * pl.cdiv(model_cfgs.head_dim, mxu_column_size)
                * model_cfgs.num_q_heads
            )
            if num_muls <= threshold or bq_c_sz <= min_bq_c_sz:
                break
        return best_bq_c_sz

    def calculate_vmem_usage(
        batch_size: int, n_buffer: int, bq_sz: int, bkv_sz: int
    ) -> int:
        """VMEM of the tile as compiled: buffers, carries and spills."""
        cfgs = configs.RpaConfigs(
            block=configs.BlockSizes(
                bq_sz=bq_sz,
                bq_c_sz=compute_chunk_size(batch_size, bq_sz, bkv_sz),
                bkv_sz=bkv_sz,
                batch_size=batch_size,
                n_buffer=n_buffer,
            ),
            model=model_cfgs,
            serve=serve_cfgs,
            # The search runs the decode kernel with one-token q blocks only.
            mode=configs.RpaCase.DECODE if bq_sz == 1 else configs.RpaCase.MIXED,
            vmem_limit_bytes=vmem_limit_bytes,
        )
        q_tile = math.prod(cfgs.q_vmem_shape)
        total_bytes = n_buffer * q_tile * q_bytes + 2 * q_tile * out_bytes
        total_bytes += n_buffer * math.prod(cfgs.kv_vmem_shape) * kv_bytes
        total_bytes += _carry_bytes(cfgs) + _spill_bytes(cfgs)
        if serve_cfgs.return_lse:
            total_bytes += _lse_bytes(cfgs)
        return total_bytes

    def fits(batch_size: int, n_buffer: int, bq_sz: int, bkv_sz: int) -> bool:
        return (
            calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
            <= usable_vmem_bytes
        )

    def fit_tile(
        batch_size: int, n_buffer: int, bq_sz: int, bkv_sz: int, min_bq_sz: int
    ) -> tuple[int, int, int]:
        # Lanes first, then halve the q block, and fewer buffers last. Buffers stop
        # at two: a single-buffered lookahead ref skips the pipeline's prologue
        # fetch, so block 0 is never fetched and the kernel hangs.
        while batch_size > 1 and not fits(batch_size, n_buffer, bq_sz, bkv_sz):
            batch_size -= 1
        while bq_sz > min_bq_sz and not fits(batch_size, n_buffer, bq_sz, bkv_sz):
            bq_sz = max(min_bq_sz, bq_sz // 2)
        while n_buffer > _MIN_N_BUFFER and not fits(
            batch_size, n_buffer, bq_sz, bkv_sz
        ):
            n_buffer -= 1
        if not fits(batch_size, n_buffer, bq_sz, bkv_sz):
            raise ValueError(
                f"Cannot fit a {bq_sz=}, {bkv_sz=} tile with {_MIN_N_BUFFER} buffers"
                f" within {vmem_limit_bytes=} less {_VMEM_MARGIN_BYTES=}."
            )
        return batch_size, bq_sz, n_buffer

    def find_best_block_sizes(
        max_batch_size: int, max_n_buffer: int, fixed_bq_sz: int | None = None
    ) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        bkv_sz = bkv_stride = mxu_column_size
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
            min_bq_sz = min(bq_sz, _MIN_SHRUNK_BQ_SZ)
        else:
            bq_sz = min_bq_sz = fixed_bq_sz
            bq_stride = 0
        # Step 1: Lower batch_size, then bq_sz, then n_buffer if even the smallest
        # tile can trigger OOM.
        batch_size, bq_sz, n_buffer = fit_tile(
            max_batch_size, max_n_buffer, bq_sz, bkv_sz, min_bq_sz
        )
        if bq_sz != bkv_sz and fixed_bq_sz is None:
            # A shrunk q block did not fit at the MXU width; keep it.
            bq_stride = 0
            bkv_stride = 0

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
        # size. When using causal masking, if bq size is larger than bkv size,
        # entire kv tile can be masked out for some query tokens. Similarly, if
        # bkv size is larger than bq size, entire query tile can be masked out for
        # some kv tokens.
        # A q block past the call's padded query tokens only adds masked rows.
        max_bq_sz = max(
            mxu_column_size, utils.align_to(serve_cfgs.total_q_tokens, mxu_column_size)
        )
        while (
            (bq_stride or bkv_stride)
            and bq_sz + bq_stride <= max_bq_sz
            and fits(batch_size, n_buffer, bq_sz + bq_stride, bkv_sz + bkv_stride)
        ):
            bkv_sz += bkv_stride
            bq_sz += bq_stride

        # When query-side buffers fill VMEM (e.g. 64 query heads on one chip under
        # PP with TP=1), lockstep growth stops at its first step, or the q block
        # shrinks, with VMEM left over. Spend it on one more k block (bkv_sz = 2 *
        # MXU width): each q block takes half the KV steps, and the causal
        # diagonal still ends on a k block edge. Only full-attention layers with at
        # least 8 q heads per KV head measured faster: with fewer, the bq_c_sz that
        # Step 3 picks for the wider block slows bf16 KV, and a sliding window reads
        # more keys outside the window. Wider ratios are not measured. Also skip the
        # step when sequences in this pool are too short to fill the wider block,
        # and when it is not a whole number of pages (cover_whole_pages would cut it
        # after Step 3 picks bq_c_sz for it).
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
                and fits(batch_size, n_buffer, bq_sz, grown_bkv_sz)
            ):
                bkv_sz = grown_bkv_sz

        # Step 3: the compute tile; a fixed bq_sz is its own compute tile.
        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=(
                bq_sz
                if fixed_bq_sz is not None
                else compute_chunk_size(batch_size, bq_sz, bkv_sz)
            ),
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
        # Refit like the search; a mixed q block stops at 2 tokens, since the
        # kernel reads bq_sz == 1 as decode.
        try:
            batch_size, bq_sz, n_buffer = fit_tile(
                block_sizes.batch_size,
                block_sizes.n_buffer,
                block_sizes.bq_sz,
                page,
                min_bq_sz=min(block_sizes.bq_sz, 2),
            )
        except ValueError as e:
            raise ValueError(
                f"A one-page k block ({page=}) does not fit within VMEM limit;"
                " a smaller page size may."
            ) from e
        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=compute_chunk_size(batch_size, bq_sz, page),
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
        "decode_query_size",
        "cp_group_size",
        "attention_scope",
        "rpa_cases",
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
            an extend-only batch ``[n, n, n]``.
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
        decode_query_size=decode_query_size,
        cache_only=cache_only,
        skip_kv_write=skip_kv_write,
        return_lse=return_lse,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
        skip_full_tile_mask=envs.SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK.get(),
        # The sink term and the LSE read the running max.
        no_max=(
            envs.SGLANG_TPU_RPA_NO_MAX.get()
            and attention_sink is None
            and not return_lse
        ),
        # Base-2 m and l would reach the soft cap, the sink term and the LSE.
        exp2_softmax=(
            envs.SGLANG_TPU_RPA_EXP2_SOFTMAX.get()
            and soft_cap is None
            and attention_sink is None
            and not return_lse
        ),
        ones_v=(
            envs.SGLANG_TPU_RPA_ONES_V.get()
            and kv_layout == configs.KVLayout.SEQ_ALONG_LANE
            and head_dim <= 64
            and attention_sink is None
            and not return_lse
        ),
        transposed=(
            envs.SGLANG_TPU_RPA_TRANSPOSED.get()
            and kv_layout == configs.KVLayout.SEQ_ALONG_LANE
            and head_dim <= 64
            and q_scale is None
            and soft_cap is None
            and attention_sink is None
            and not return_lse
            and not cache_only
            and cp_group_size is None
        ),
        skip_pad_q=envs.SGLANG_TPU_RPA_SKIP_PAD_Q.get(),
        skip_pad_q_group=envs.SGLANG_TPU_RPA_SKIP_PAD_Q_GROUP.get(),
        out_head_norm_eps=envs.SGLANG_TPU_RPA_OUT_HEAD_NORM_EPS.get(),
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
    blocks_env = (
        envs.SGLANG_TPU_RPA_PREFILL_BLOCKS
        if sliding_window is None
        else envs.SGLANG_TPU_RPA_SWA_PREFILL_BLOCKS
    ).get()
    if prefill_block_sizes is None and blocks_env:
        # The search grows q and k blocks together until VMEM is full; a q block that
        # divides the prefill chunk leaves no padded queries, and smaller tiles waste
        # less of a sliding-window band.
        bq_sz, bkv_sz = (int(v) for v in blocks_env.split(":"))
        prefill_block_sizes = dataclasses.replace(
            default_prefill,
            bq_sz=bq_sz,
            bq_c_sz=math.gcd(bq_sz, default_prefill.bq_c_sz),
            bkv_sz=bkv_sz,
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

    if rpa_cases is None:
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
