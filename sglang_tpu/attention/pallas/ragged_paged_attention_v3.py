# Adapted from https://github.com/sgl-project/sglang-jax/blob/febb350c/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py
# which was adapted from https://github.com/vllm-project/tpu-inference/releases/tag/v0.11.1
# Copyright 2025 The tpu-inference Authors. All rights reserved.
# Copyright 2025 The SGLang Team. All rights reserved.
"""TPU-Friendly Ragged Paged Attention kernel (v3).

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.

Key optimizations over v2:
- Split kernel into 3 separate pallas_calls (DECODE/PREFILL/MIXED)
- l/m/acc re-initialization per bq block
- Sliding window precise skipping via per-bq-block start/end indices
- Block size API (d_block_sizes/p_block_sizes/m_block_sizes)
- KV cache update timing moved to last bq iteration

sglang-jax specific features:
- custom_mask support for speculative decoding
- attention_sink support for streaming inference
- xai_temperature support for Grok-style models
- cu_kv_lens-based page_indices offset computation
"""

import functools
import inspect as _inspect
import logging
import math
from enum import Enum

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.util import (
    align_to,
    cdiv,
    get_dtype_bitwidth,
    get_dtype_packing,
    get_tpu_version,
    next_power_of_2,
)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
DEFAULT_VMEM_LIMIT_BYTES = 120 * 1024 * 1024  # 120MB
logger = logging.getLogger(__name__)

_COMPILER_PARAMS_SUPPORTS_SEMAPHORE = (
    "disable_semaphore_checks" in _inspect.signature(pltpu.CompilerParams).parameters
)


def _semaphore_kwargs(disable_semaphore_checks: bool) -> dict:
    if _COMPILER_PARAMS_SUPPORTS_SEMAPHORE:
        return {"disable_semaphore_checks": disable_semaphore_checks}
    return {}


class RpaCase(Enum):
    """Represents the different cases for Ragged Paged Attention.

    - DECODE: Sequences are in decode-only mode (q_len = 1).
    - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
    - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
    """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(self, distribution):
        assert distribution.shape == (3,)
        if self == RpaCase.DECODE:
            return 0, distribution[0]
        elif self == RpaCase.PREFILL:
            return distribution[0], distribution[1]
        elif self == RpaCase.MIXED:
            return distribution[1], distribution[2]
        else:
            raise ValueError(f"Unsupported RPA case: {self}")


def ref_ragged_paged_attention(
    queries: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[padded_batch_size]
    page_indices: jax.Array,  # i32[padded_batch_size, max_pages_per_seq]
    cu_q_lens: jax.Array,  # i32[padded_batch_size + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    custom_mask: jax.Array = None,  # [pattern_total_kv_len]
    causal: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    attention_sink: jax.Array | float | None = None,
):
    """Reference implementation for ragged paged attention."""
    if not causal and custom_mask is not None:
        assert custom_mask.size > jnp.cumsum(kv_lens)[-1], (
            f"use custom_mask, custom_mask length {custom_mask.size=} must larger than total kv length {jnp.cumsum(kv_lens)[-1]=}"
        )
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    _, _, num_kv_heads, head_dim = k_pages.shape
    num_q_heads = queries.shape[1]
    assert num_q_heads % num_kv_heads == 0
    num_query_per_kv = num_q_heads // num_kv_heads
    outputs = []
    mask_len_list = []
    for i in range(num_seqs[0]):
        kv_len = kv_lens[i]
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        mask_len_list.append(q_len * kv_len)
    mask_lens = jnp.array(mask_len_list, dtype=jnp.int32)
    cu_mask_lens = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(mask_lens)]
    )

    for i in range(num_seqs[0]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        indices = page_indices[i]
        q = queries[q_start:q_end]
        k = k_pages[indices, :, :, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        v = v_pages[indices, :, :, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        if k_scale is not None:
            k = k.astype(jnp.float32) * k_scale
            k = k.astype(q.dtype)
        if v_scale is not None:
            v = v.astype(jnp.float32) * v_scale
            v = v.astype(q.dtype)
        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)
        attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
        attn *= sm_scale
        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        if causal:
            mask = q_span < kv_span
        elif custom_mask is None:
            # Bidirectional: every query sees all kv_len keys.
            mask = jnp.zeros(attn.shape, dtype=jnp.bool_)
        else:
            mask_start = cu_mask_lens[i]
            mask_end = cu_mask_lens[i + 1]
            mask = custom_mask[mask_start:mask_end]
            mask = (
                jnp.repeat(jnp.expand_dims(mask, axis=0), num_q_heads, axis=0).reshape(
                    num_q_heads, q_len, kv_len
                )
                < 1
            )
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        if xai_temperature_len is not None:
            prefix_len = kv_len - q_len
            qidx = jnp.arange(prefix_len, kv_len)
            xai_temperature_scale = 1.0 / jnp.log2(float(xai_temperature_len))
            _qtemp = jnp.log2(qidx.astype(jnp.float32)) * xai_temperature_scale
            xai_temperature_reg = jnp.where(qidx > xai_temperature_len, _qtemp, 1.0)
            attn = attn * xai_temperature_reg[None, :, None]

        attn += jnp.where(mask, mask_value, 0.0)
        if attention_sink is not None:
            sink = jnp.asarray(attention_sink, dtype=jnp.float32)
            if sink.ndim == 0:
                sink = jnp.full((num_q_heads,), sink)
            sink_logits = jnp.broadcast_to(
                sink.reshape(num_q_heads, 1, 1),
                (num_q_heads, q_len, 1),
            )
            attn = jnp.concatenate([sink_logits, attn], axis=-1)
            attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
            attn = attn[..., 1:]
        else:
            attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)


def get_smem_estimate_bytes(max_num_seqs, pages_per_seq):
    total_bits = (
        # kv_lens_ref: i32[max_num_seqs]
        align_to(max_num_seqs, 128) * 32
        +
        # page_indices_ref: i32[max_num_seqs * pages_per_seq]
        align_to(max_num_seqs * pages_per_seq, 128) * 32
        +
        # cu_q_lens_ref: i32[max_num_seqs + 1]
        align_to(max_num_seqs + 1, 128) * 32
        +
        # cu_kv_lens_ref: i32[max_num_seqs + 1]
        align_to(max_num_seqs + 1, 128) * 32
        +
        # cu_seq_mask_lens: i32[max_num_seqs + 1]
        align_to(max_num_seqs + 1, 128) * 32
        +
        # distribution_ref: i32[3]
        128 * 32
        +
        # sem_ids_ref: i32[3]
        128 * 32
        +
        # bo_ids_ref: i32[4]
        128 * 32
        +
        # bkv_update_ids_ref: i32[6]
        128 * 32
    )
    return cdiv(total_bits, 8)


def get_vmem_estimate_bytes(
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    bq_sz,
    bkv_sz,
    q_dtype,
    kv_dtype,
    use_custom_mask=False,
    bkv_csz=None,
):
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    out_bits = get_dtype_bitwidth(q_dtype) if q_dtype != jnp.float32 else 32
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    bkv_stride = cdiv(actual_num_kv_heads * 2, kv_packing)
    if has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    head_dim = align_to(actual_head_dim, 128)

    total_bits = (
        # bkv_x2_ref
        (2 * bkv_sz * bkv_stride * kv_packing * head_dim) * (32 // kv_packing)
        +
        # bq_x2_ref + bo_x2_ref
        2
        * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim)
        * (32 // q_packing)
        +
        # l_ref + m_ref (out_dtype, not float32)
        2 * (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * out_bits
        +
        # acc_ref (out_dtype)
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) * out_bits
    )

    if use_custom_mask:
        # bkvmask_double_buf: (2, bq_sz, bkv_sz, head_dim) in int32
        total_bits += 2 * bq_sz * bkv_sz * head_dim * 32

    # Attention compute intermediates (f32). The for-loop over kv_heads is
    # statically unrolled by the compiler, so all heads' intermediates coexist
    # in VMEM.  Per head: QK^T scores (bq*qpkv, bkv_csz) + PV result (bq*qpkv, hd)
    # in f32, counted once (e.g. gpt-oss-120b at 256 query rows: 86.0 MiB
    # estimated, 81.7 MiB reported by the compiler).
    # Use bkv_csz (compute size) since that determines the matmul tile size.
    compute_bkv = bkv_csz if bkv_csz is not None else bkv_sz
    compute_bits = (
        actual_num_kv_heads
        * bq_sz
        * num_q_heads_per_kv_head
        * (compute_bkv + head_dim)
        * 32  # f32
    )
    total_bits += compute_bits

    return cdiv(total_bits, 8)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )


_STATIONARY_Q_V2_OPTS = frozenset(
    {
        "stationary_q_m0_ring",
        "stationary_q_m0_f32e",
        "stationary_q_m0_ring_fuse",
        "stationary_q_m0_f32e_fuse",
    }
)


def _stationary_q_v2_process(
    *,
    seq_idx,
    start_seq_idx,
    end_seq_idx,
    q_len,
    kv_len,
    kv_q_gap,
    cur_seq_start_bkv_idx,
    next_seq_start_bkv_idx,
    actual_bq_sz,
    num_bq,
    bq_sz,
    bkv_sz,
    bkv_stride,
    kv_dtype,
    num_slots,
    use_causal_mask,
    sliding_window,
    sliding_window_chunk_skip,
    _sw_off,
    sm_scale,
    k_scale,
    v_scale,
    probe,
    prefill_opts,
    sem_ids_ref,
    bq_x2_ref,
    bkv_x2_ref,
    bo_x2_ref,
    l_ref,
    acc_ref,
    q_t_ref,
    k_sub_ref,
    vt_sub_ref,
    mask_lt_ref,
    mask_ut_ref,
    cu_q_lens_ref,
    bo_carry_ref,
    q_sc_ref=None,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    get_next_bq_ids,
    get_next_bkv_ids,
    start_fetch_bq,
    wait_fetch_bq,
    start_fetch_bkv,
    wait_fetch_bkv,
    start_update_kv_cache,
    wait_send_bo,
    start_send_bo,
    start_send_bo_a3,
    start_fetch_bq_stage=None,
    wait_fetch_bq_stage=None,
    start_send_bo_staged=None,
    interpret: bool = False,
    cu_qgrps_ref=None,
    prefix_topk_ref=None,
    prefix_topk_slices: int = 0,
    prefix_topk_sink_slices: int = 0,
    prefix_q_grp_sz: int = 1024,
    prefix_max_bkv: int = 16,
    prefix_topk_tail_local: int = 0,
):
    """Stationary-Q v2 prefill engine: unpacked-KV VMEM ring, exact 256-slice unrolling,
    precomputed bf16 mask bias tiles, SW-2049 Q-unpack/epilogue fusion, direct 2D A3
    carry-buffer output, and optional f32 exp2 / Wave-7 FP8 QK(PV) & GQA VREG batching."""
    tq_sz = 256
    sq_kv_heads = k_sub_ref.shape[0]
    sq_pairs = 4 * sq_kv_heads
    sq_q_heads = 8 * sq_kv_heads
    use_mk128 = k_sub_ref.shape[2] == 128
    use_k64 = k_sub_ref.shape[3] == 64
    use_a3_out = bo_carry_ref is not None and "a3_out_layout" in prefill_opts
    use_fp8_qk_dyn = "fp8_qk_dyn" in prefill_opts
    use_fp8_qkpv_fast = "fp8_qkpv_fast" in prefill_opts
    use_fp8_qkpv_dyn = "fp8_qkpv_dyn" in prefill_opts
    use_fp8_qkpv_norm = "fp8_qkpv_norm" in prefill_opts
    use_fp8_qkpv = (
        ("fp8_qkpv" in prefill_opts)
        or ("fp8_qkpv_s1" in prefill_opts)
        or ("fp8_qkpv_s0" in prefill_opts)
        or use_fp8_qkpv_norm
    )
    qkpv_shift = (
        0.0
        if "fp8_qkpv_s0" in prefill_opts
        else (1.5 if "fp8_qkpv_s1" in prefill_opts else 2.0)
    )
    use_fp8_qk = (
        use_fp8_qkpv_fast
        or use_fp8_qkpv_dyn
        or use_fp8_qkpv
        or use_fp8_qk_dyn
        or ("fp8_qk" in prefill_opts)
    )
    use_fp8_kv = use_fp8_qk or ("fp8_kv" in prefill_opts)
    use_vt_fp8 = (
        use_fp8_qkpv_fast
        or use_fp8_qkpv_dyn
        or use_fp8_qkpv
        or ("fp8_kv" in prefill_opts)
    )
    use_gqa4_vreg = ("gqa4_vreg" in prefill_opts) and not use_mk128
    use_gqa2_vreg = (
        ("gqa2_vreg" in prefill_opts) and not use_gqa4_vreg and not use_mk128
    )
    use_vreg_acc1 = (
        ("vreg_acc1" in prefill_opts)
        and not use_mk128
        and not (use_gqa2_vreg or use_gqa4_vreg)
    )
    use_vreg_acc = (
        ("vreg_acc" in prefill_opts)
        and not use_mk128
        and not use_vreg_acc1
        and not (use_gqa2_vreg or use_gqa4_vreg)
    )
    use_sw_short_v2 = "sw_short_v2" in prefill_opts
    use_sw_short_fuse = ("sw_short_fuse" in prefill_opts) or use_sw_short_v2
    use_sw_unalign_fuse = "sw_unalign_fuse" in prefill_opts
    use_sw_tail_clamp = ("sw_tail_clamp" in prefill_opts) and use_mk128
    use_sub8_l = "sub8_l" in prefill_opts
    use_la2 = "la2" in prefill_opts
    use_unroll_up = ("unroll_up" in prefill_opts) and not (
        "gqa8" in prefill_opts and kv_pack_dense2
    )
    mrb_depth = (
        1
        if "mrb1" in prefill_opts
        else (4 if "mrb4" in prefill_opts else (3 if "mrb3" in prefill_opts else 2))
    )
    use_f32_exp = (
        use_fp8_qkpv_dyn
        or use_fp8_qk_dyn
        or "f32_exp" in prefill_opts
        or "stationary_q_m0_f32e" in prefill_opts
        or "stationary_q_m0_f32e_fuse" in prefill_opts
    )
    use_fuse_qe = (
        "stationary_q_m0_ring_fuse" in prefill_opts
        or "stationary_q_m0_f32e_fuse" in prefill_opts
    )
    use_tq_loop = "tq_unroll" not in prefill_opts
    use_staged_qo = (
        "staged_qo" in prefill_opts
        and bq_x2_ref.shape[2] == tq_sz
        and bq_sz >= 512
        and bq_sz % tq_sz == 0
    )
    num_s = bkv_sz // 256
    ring_total_slices = (
        (k_sub_ref.shape[1] // 2) if use_mk128 else k_sub_ref.shape[1]
    )
    ring_bkv_blocks = ring_total_slices // num_s
    ring_covers_all = ring_bkv_blocks > 1
    ring_s_mask = ring_total_slices - 1
    ring_h_mask = 2 * ring_total_slices - 1

    s_scale = sm_scale * (k_scale if k_scale is not None else 1.0)
    exp_scale = float(s_scale * math.log2(math.e))
    v_scale_val = float(v_scale) if v_scale is not None else 1.0
    seq_end_bkv = cdiv(kv_len, bkv_sz)

    @pl.when(seq_idx == start_seq_idx)
    def _init_static_mask_tiles():
        if "poison_vmem_nan" in prefill_opts:
            nan_f32 = jnp.float32(jnp.nan)
            nan_bf16 = jnp.bfloat16(jnp.nan)
            nan_f8 = lax.bitcast_convert_type(jnp.uint8(0x7F), jnp.float8_e4m3fn)
            l_ref[...] = jnp.full(l_ref.shape, nan_f32, jnp.float32)
            acc_ref[...] = jnp.full(acc_ref.shape, nan_f32, jnp.float32)
            q_t_ref[...] = jnp.full(
                q_t_ref.shape,
                nan_f8 if q_t_ref.dtype == jnp.float8_e4m3fn else nan_bf16,
                q_t_ref.dtype,
            )
            k_sub_ref[...] = jnp.full(
                k_sub_ref.shape,
                nan_f8 if k_sub_ref.dtype == jnp.float8_e4m3fn else nan_bf16,
                k_sub_ref.dtype,
            )
            vt_sub_ref[...] = jnp.full(
                vt_sub_ref.shape,
                nan_f8 if vt_sub_ref.dtype == jnp.float8_e4m3fn else nan_bf16,
                vt_sub_ref.dtype,
            )
        if use_mk128:
            q_io_h = lax.broadcasted_iota(jnp.int32, (1, 128, tq_sz), 2)
            k_128 = lax.broadcasted_iota(jnp.int32, (1, 128, tq_sz), 1)
            if kv_pack_bd4:
                k_io_0 = jnp.where(k_128 < 64, 4 * k_128, 4 * (k_128 - 64) + 2)
                k_io_1 = k_io_0 + 1
            elif kv_pack_dense2:
                k_io_0 = 2 * k_128
                k_io_1 = 2 * k_128 + 1
            else:
                k_io_0 = k_128
                k_io_1 = k_128 + 128
            if mask_lt_ref is not None:
                mask_lt_ref[0:1] = jnp.where(
                    q_io_h >= k_io_0, jnp.bfloat16(0.0), jnp.bfloat16(-jnp.inf)
                )
                mask_lt_ref[1:2] = jnp.where(
                    q_io_h >= k_io_1, jnp.bfloat16(0.0), jnp.bfloat16(-jnp.inf)
                )
            if mask_ut_ref is not None:
                ut0_c = (q_io_h < k_io_0) if sliding_window == 2048 else (q_io_h <= k_io_0)
                ut1_c = (q_io_h < k_io_1) if sliding_window == 2048 else (q_io_h <= k_io_1)
                mask_ut_ref[0:1] = jnp.where(
                    ut0_c, jnp.bfloat16(0.0), jnp.bfloat16(-jnp.inf)
                )
                mask_ut_ref[1:2] = jnp.where(
                    ut1_c, jnp.bfloat16(0.0), jnp.bfloat16(-jnp.inf)
                )
            return
        q_io = lax.broadcasted_iota(jnp.int32, (1, 256, tq_sz), 2)
        k_raw = lax.broadcasted_iota(jnp.int32, (1, 256, tq_sz), 1)
        if kv_pack_bd4:
            k_h = k_raw % 128
            k_ev = jnp.where(k_h < 64, 4 * k_h, 4 * (k_h - 64) + 2)
            k_io = jnp.where(k_raw < 128, k_ev, k_ev + 1)
        elif kv_pack_dense2:
            k_io = jnp.where(k_raw < 128, 2 * k_raw, 2 * (k_raw - 128) + 1)
        else:
            k_io = k_raw
        if mask_lt_ref is not None:
            mask_lt_ref[...] = jnp.where(
                q_io >= k_io, jnp.bfloat16(0.0), jnp.bfloat16(-jnp.inf)
            )
        if mask_ut_ref is not None:
            ut_cond = (q_io < k_io) if sliding_window == 2048 else (q_io <= k_io)
            mask_ut_ref[...] = jnp.where(
                ut_cond, jnp.bfloat16(0.0), jnp.bfloat16(-jnp.inf)
            )

    if ring_covers_all:
        sem_ids_ref[3] = cur_seq_start_bkv_idx

    use_bf16_pop = (
        ("bf16_pop" in prefill_opts)
        and not use_f32_exp
        and not (use_fp8_qk and use_fp8_qkpv)
    )

    def _kq_dot(k_3d, q_3d):
        qk_f = lax.dot(
            k_3d,
            q_3d,
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )
        return qk_f.astype(jnp.bfloat16) if use_bf16_pop else qk_f

    def _vp_dot(vt_3d, p_3d):
        return lax.dot(
            vt_3d,
            p_3d,
            dimension_numbers=(([2], [1]), ([0], [0])),
            preferred_element_type=jnp.float32,
        )

    def _vp_scaled(vt_3d, p_3d, p_sc=None):
        vp = _vp_dot(vt_3d, p_3d)
        if p_sc is not None:
            vp = vp * jnp.tile(p_sc, (1, 8, 1))
        return vp

    def _psum_w(p, w_cols: int):
        if use_sub8_l:
            return jnp.sum(
                p.reshape((1, p.shape[1] // 8, 8, w_cols)),
                axis=1,
                dtype=jnp.bfloat16,
            ).astype(jnp.float32)
        sub_shape = (
            (1, 16, 8, w_cols) if p.shape[1] == 128 else (1, 16, 16, w_cols)
        )
        p_sum_sub = jnp.sum(
            p.reshape(sub_shape),
            axis=1,
            dtype=jnp.bfloat16,
        )
        return jnp.broadcast_to(
            jnp.sum(
                p_sum_sub,
                axis=1,
                keepdims=True,
                dtype=jnp.float32,
            ),
            (1, 8, w_cols),
        )

    def _psum(p):
        return _psum_w(p, tq_sz)

    def _quant_qt_fp8(qt_bf16):
        """Quantize [64, W] bf16 Q -> (qt_f8 [64, W], q_exp_sc_8 [8, W] f32)."""
        w_q = qt_bf16.shape[1]
        qf = qt_bf16.astype(jnp.float32)
        m8 = jnp.max(jnp.abs(qf).reshape(8, 8, w_q), axis=0)
        q_amax = jnp.max(m8, axis=0, keepdims=True)
        q_amax_8 = jnp.broadcast_to(q_amax, (8, w_q))
        q_inv = jnp.where(
            q_amax == 0.0,
            jnp.float32(0.0),
            jnp.float32(448.0) / jnp.maximum(q_amax, jnp.float32(1e-12)),
        )
        qt_f8 = jnp.clip(qf * q_inv, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        q_exp_sc_8 = jnp.where(
            q_amax_8 == 0.0,
            jnp.float32(exp_scale),
            q_amax_8 * jnp.float32(exp_scale / 448.0),
        )
        return qt_f8, q_exp_sc_8

    def _sm_eval_w(qk_f32, bias_spec, w_cols: int, q_exp_256=None):
        n_rep = w_cols // 256
        m_rows = qk_f32.shape[1]
        if use_f32_exp:
            qk_sc = (
                qk_f32 * q_exp_256
                if q_exp_256 is not None
                else qk_f32 * exp_scale
            )
            if use_fp8_qkpv:
                qk_sc = qk_sc - jnp.float32(qkpv_shift)
            if bias_spec in ("ut", "ut0", "ut1"):
                m_u = (
                    mask_ut_ref[0:1]
                    if bias_spec == "ut0"
                    else (
                        mask_ut_ref[1:2]
                        if bias_spec == "ut1"
                        else mask_ut_ref[...]
                    )
                ).astype(jnp.float32)
                qk_sc = qk_sc + (
                    m_u if n_rep == 1 else jnp.concatenate([m_u] * n_rep, axis=2)
                )
            elif bias_spec in ("lt", "lt0", "lt1"):
                m_l = (
                    mask_lt_ref[0:1]
                    if bias_spec == "lt0"
                    else (
                        mask_lt_ref[1:2]
                        if bias_spec == "lt1"
                        else mask_lt_ref[...]
                    )
                ).astype(jnp.float32)
                qk_sc = qk_sc + (
                    m_l if n_rep == 1 else jnp.concatenate([m_l] * n_rep, axis=2)
                )
            elif bias_spec != "none":
                bs = (
                    bias_spec
                    if n_rep == 1
                    else jnp.concatenate([bias_spec] * n_rep, axis=2)
                )
                if bs.dtype == jnp.bfloat16:
                    qk_sc = qk_sc + bs.astype(jnp.float32)
                else:
                    qk_sc = jnp.where(bs, qk_sc, -jnp.inf)
            if use_fp8_qkpv_dyn:
                m_sub = jnp.max(qk_sc.reshape(1, m_rows // 8, 8, w_cols), axis=1)
                m_col = jnp.max(m_sub, axis=1, keepdims=True)
                # Clamp -inf/extreme negative slices >= -110 so m_bias >= -118.75 avoids f32 FTZ.
                m_safe = jnp.maximum(m_col, jnp.float32(-110.0))
                m_bias = m_safe - jnp.float32(8.75)
                p_f32 = jnp.exp2(qk_sc - m_bias)
                p_f8 = jnp.minimum(p_f32, jnp.float32(448.0)).astype(jnp.float8_e4m3fn)
                p_scale_8 = jnp.broadcast_to(jnp.exp2(m_bias), (1, 8, w_cols))
                p_sum_sub = jnp.sum(
                    p_f8.astype(jnp.float32).reshape(1, m_rows // 8, 8, w_cols),
                    axis=1,
                    dtype=jnp.float32,
                )
                ps_unscaled = jnp.broadcast_to(
                    jnp.sum(p_sum_sub, axis=1, keepdims=True, dtype=jnp.float32),
                    (1, 8, w_cols),
                )
                return p_f8, ps_unscaled * p_scale_8, p_scale_8
            if use_fp8_qkpv:
                p_f32 = jnp.exp2(qk_sc)
                p_f8 = jnp.minimum(p_f32, jnp.float32(448.0)).astype(jnp.float8_e4m3fn)
                p_for_sum = (
                    p_f8.astype(jnp.float32) if use_fp8_qkpv_norm else p_f32
                )
                p_sum_sub = jnp.sum(
                    p_for_sum.reshape(1, m_rows // 8, 8, w_cols),
                    axis=1,
                    dtype=jnp.float32,
                )
                ps = jnp.broadcast_to(
                    jnp.sum(p_sum_sub, axis=1, keepdims=True, dtype=jnp.float32),
                    (1, 8, w_cols),
                )
                return p_f8, ps, None
            p = jnp.exp2(qk_sc).astype(jnp.bfloat16)
        else:
            if use_fp8_qk and use_fp8_qkpv:
                qk = (qk_f32 - jnp.float32(qkpv_shift)).astype(jnp.bfloat16)
            else:
                qk = qk_f32.astype(jnp.bfloat16)
            if bias_spec in ("ut", "ut0", "ut1"):
                m_u = (
                    mask_ut_ref[0:1]
                    if bias_spec == "ut0"
                    else (
                        mask_ut_ref[1:2]
                        if bias_spec == "ut1"
                        else mask_ut_ref[...]
                    )
                )
                qk = qk + (
                    m_u if n_rep == 1 else jnp.concatenate([m_u] * n_rep, axis=2)
                )
            elif bias_spec in ("lt", "lt0", "lt1"):
                m_l = (
                    mask_lt_ref[0:1]
                    if bias_spec == "lt0"
                    else (
                        mask_lt_ref[1:2]
                        if bias_spec == "lt1"
                        else mask_lt_ref[...]
                    )
                )
                qk = qk + (
                    m_l if n_rep == 1 else jnp.concatenate([m_l] * n_rep, axis=2)
                )
            elif bias_spec != "none":
                bs = (
                    bias_spec
                    if n_rep == 1
                    else jnp.concatenate([bias_spec] * n_rep, axis=2)
                )
                if bs.dtype == jnp.bfloat16:
                    qk = qk + bs
                else:
                    qk = jnp.where(bs, qk, -jnp.inf)
            if use_fp8_qkpv_fast:
                m_sub = jnp.max(qk.reshape(1, m_rows // 8, 8, w_cols), axis=1)
                m_col = jnp.max(m_sub, axis=1, keepdims=True)
                m_safe = jnp.maximum(m_col, jnp.bfloat16(-110.0))
                m_bias = m_safe - jnp.bfloat16(8.75)
                p_bf = jnp.exp2(qk - m_bias)
                p_f8 = jnp.minimum(p_bf, jnp.bfloat16(448.0)).astype(jnp.float8_e4m3fn)
                p_scale_8 = jnp.broadcast_to(
                    jnp.exp2(m_bias.astype(jnp.float32)), (1, 8, w_cols)
                )
                ps = _psum_w(p_f8.astype(jnp.bfloat16), w_cols) * p_scale_8
                return p_f8, ps, p_scale_8
            p = jnp.exp2(qk)
            if use_fp8_qkpv:
                p_f8 = jnp.minimum(p, jnp.bfloat16(448.0)).astype(jnp.float8_e4m3fn)
                ps = _psum_w(
                    p_f8.astype(jnp.bfloat16) if use_fp8_qkpv_norm else p, w_cols
                )
                return p_f8, ps, None
        return p, _psum_w(p, w_cols), None

    def _sm_eval(qk_f32, bias_spec, q_exp_256=None):
        return _sm_eval_w(qk_f32, bias_spec, tq_sz, q_exp_256)

    def _write_unpacked_kv_slice(dst_s, kv_u32):
        if use_k64:
            if use_fp8_kv:
                k_f = pltpu.bitcast(kv_u32.astype(jnp.uint8), kv_dtype)
                k_sub_ref[0, dst_s] = k_f[:, :64]
                if sq_kv_heads > 1:
                    k_sub_ref[1, dst_s] = k_f[:, 64:]
            else:
                k_b = pltpu.bitcast(kv_u32.astype(jnp.uint8), kv_dtype).astype(
                    jnp.bfloat16
                )
                k_sub_ref[0, dst_s] = k_b[:, :64]
                if sq_kv_heads > 1:
                    k_sub_ref[1, dst_s] = k_b[:, 64:]
        elif use_fp8_kv:
            k_sub_ref[0, dst_s] = pltpu.bitcast(kv_u32.astype(jnp.uint8), kv_dtype)
            if sq_kv_heads > 1:
                kv_u32_hi = pltpu.roll(kv_u32, shift=64, axis=1)
                k_sub_ref[1, dst_s] = pltpu.bitcast(
                    kv_u32_hi.astype(jnp.uint8), kv_dtype
                )
        else:
            k_b = pltpu.bitcast(kv_u32.astype(jnp.uint8), kv_dtype).astype(jnp.bfloat16)
            k_sub_ref[0, dst_s] = k_b
            if sq_kv_heads > 1:
                k_sub_ref[1, dst_s] = pltpu.roll(k_b, shift=64, axis=1)
        v_bf16 = pltpu.bitcast(
            (kv_u32 >> jnp.uint32(8)).astype(jnp.uint8), kv_dtype
        ).astype(jnp.bfloat16)
        vt_128 = jnp.swapaxes(v_bf16, 0, 1)
        if use_vt_fp8:
            vt_sub_ref[0, dst_s] = vt_128[:64, :].astype(kv_dtype)
            if sq_kv_heads > 1:
                vt_sub_ref[1, dst_s] = vt_128[64:, :].astype(kv_dtype)
        else:
            vt_sub_ref[0, dst_s] = vt_128[:64, :]
            if sq_kv_heads > 1:
                vt_sub_ref[1, dst_s] = vt_128[64:, :]

    def _bkv_slice_u32(kv_u32_all, bkv_sem_idx, row_start, n_rows):
        if interpret:
            u8 = lax.bitcast_convert_type(
                bkv_x2_ref[bkv_sem_idx, pl.ds(row_start, n_rows), 0, :, :],
                jnp.uint8,
            ).astype(jnp.uint32)
            return (
                u8[:, 0, :]
                | (u8[:, 1, :] << jnp.uint32(8))
                | (u8[:, 2, :] << jnp.uint32(16))
                | (u8[:, 3, :] << jnp.uint32(24))
            )
        return kv_u32_all[pl.ds(row_start * bkv_stride, n_rows, bkv_stride)]

    def _roll64(arr_u32):
        return jnp.roll(arr_u32, 64, axis=1) if interpret else pltpu.roll(arr_u32, shift=64, axis=1)

    def _unpack_full_bkv_block(bkv_sem_idx, slot_base_s=0):
        if kv_pack_bd4:
            kv_u32_all = (
                None
                if interpret
                else bkv_x2_ref.bitcast(jnp.uint32)
                .at[bkv_sem_idx]
                .reshape((bkv_sz // 4) * bkv_stride, 128)
            )

            def _emit_bd4_slice(s_loc):
                pair_u32 = _bkv_slice_u32(kv_u32_all, bkv_sem_idx, s_loc * 64, 64)
                lo = pair_u32 & jnp.uint32(0xFFFF)
                hi = pair_u32 >> jnp.uint32(16)
                even_128 = jnp.concatenate([lo, hi], axis=0)
                odd_128 = jnp.concatenate([_roll64(lo), _roll64(hi)], axis=0)
                if use_mk128:
                    _write_unpacked_kv_slice(2 * (slot_base_s + s_loc), even_128)
                    _write_unpacked_kv_slice(2 * (slot_base_s + s_loc) + 1, odd_128)
                else:
                    _write_unpacked_kv_slice(
                        slot_base_s + s_loc, jnp.concatenate([even_128, odd_128], axis=0)
                    )

            if use_mk128 and not use_unroll_up:
                @pl.loop(0, num_s, unroll=False)
                def _up_mk_bd4(s_loc):
                    _emit_bd4_slice(s_loc)

                return
            for s_loc in range(num_s):
                _emit_bd4_slice(s_loc)
            return

        if kv_pack_dense2:
            kv_u32_all = (
                None
                if interpret
                else bkv_x2_ref.bitcast(jnp.uint32)
                .at[bkv_sem_idx]
                .reshape((bkv_sz // 2) * bkv_stride, 128)
            )
            if use_mk128:
                if use_unroll_up:
                    for s_loc in range(num_s):
                        pair_u32 = _bkv_slice_u32(
                            kv_u32_all, bkv_sem_idx, s_loc * 128, 128
                        )
                        _write_unpacked_kv_slice(
                            2 * (slot_base_s + s_loc),
                            pair_u32 & jnp.uint32(0xFFFF),
                        )
                        _write_unpacked_kv_slice(
                            2 * (slot_base_s + s_loc) + 1,
                            pair_u32 >> jnp.uint32(16),
                        )
                    return

                @pl.loop(0, num_s, unroll=False)
                def _up_mk_d2(s_loc):
                    pair_u32 = _bkv_slice_u32(
                        kv_u32_all, bkv_sem_idx, s_loc * 128, 128
                    )
                    _write_unpacked_kv_slice(
                        2 * (slot_base_s + s_loc),
                        pair_u32 & jnp.uint32(0xFFFF),
                    )
                    _write_unpacked_kv_slice(
                        2 * (slot_base_s + s_loc) + 1,
                        pair_u32 >> jnp.uint32(16),
                    )

                return

            for s_loc in range(num_s):
                pair_u32 = _bkv_slice_u32(
                    kv_u32_all, bkv_sem_idx, s_loc * 128, 128
                )
                kv_u32 = jnp.concatenate(
                    [
                        pair_u32 & jnp.uint32(0xFFFF),
                        pair_u32 >> jnp.uint32(16),
                    ],
                    axis=0,
                )
                _write_unpacked_kv_slice(slot_base_s + s_loc, kv_u32)

            return

        kv_u32_all = (
            None
            if interpret
            else bkv_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx]
            .reshape(bkv_sz * bkv_stride, 128)
        )
        if use_mk128:
            if use_unroll_up:
                for h_loc in range(2 * num_s):
                    kv_u32_h = _bkv_slice_u32(
                        kv_u32_all, bkv_sem_idx, h_loc * 128, 128
                    )
                    _write_unpacked_kv_slice(2 * slot_base_s + h_loc, kv_u32_h)
                return

            @pl.loop(0, 2 * num_s, unroll=False)
            def _up_mk_nod2(h_loc):
                kv_u32_h = _bkv_slice_u32(
                    kv_u32_all, bkv_sem_idx, h_loc * 128, 128
                )
                _write_unpacked_kv_slice(2 * slot_base_s + h_loc, kv_u32_h)

            return

        for s_loc in range(num_s):
            kv_u32 = _bkv_slice_u32(kv_u32_all, bkv_sem_idx, s_loc * 256, 256)
            _write_unpacked_kv_slice(slot_base_s + s_loc, kv_u32)

    def _unpack_bkv_block(bkv_sem_idx, bkv_block_idx, slot_base_s, valid_keys):
        if kv_pack_bd4:
            kv_u32_all = (
                None
                if interpret
                else bkv_x2_ref.bitcast(jnp.uint32)
                .at[bkv_sem_idx]
                .reshape((bkv_sz // 4) * bkv_stride, 128)
            )

            @pl.when(valid_keys == bkv_sz)
            def _up_full_bd4():
                _unpack_full_bkv_block(bkv_sem_idx, slot_base_s)

            @pl.when(valid_keys < bkv_sz)
            def _up_partial_bd4():
                s_lim = cdiv(valid_keys, 256)
                r_64 = lax.broadcasted_iota(jnp.int32, (64, 128), 0)

                @pl.loop(0, s_lim, unroll=False)
                def _up_one_bd4(s_loc):
                    pair_u32 = _bkv_slice_u32(kv_u32_all, bkv_sem_idx, s_loc * 64, 64)
                    lo = pair_u32 & jnp.uint32(0xFFFF)
                    hi = pair_u32 >> jnp.uint32(16)
                    tok0 = s_loc * 256 + 4 * r_64
                    u0 = jnp.where(tok0 < valid_keys, lo, jnp.uint32(0))
                    u1 = jnp.where((tok0 + 1) < valid_keys, _roll64(lo), jnp.uint32(0))
                    u2 = jnp.where((tok0 + 2) < valid_keys, hi, jnp.uint32(0))
                    u3 = jnp.where((tok0 + 3) < valid_keys, _roll64(hi), jnp.uint32(0))
                    even_128 = jnp.concatenate([u0, u2], axis=0)
                    odd_128 = jnp.concatenate([u1, u3], axis=0)
                    if use_mk128:
                        _write_unpacked_kv_slice(2 * (slot_base_s + s_loc), even_128)
                        _write_unpacked_kv_slice(
                            2 * (slot_base_s + s_loc) + 1, odd_128
                        )
                    else:
                        _write_unpacked_kv_slice(
                            slot_base_s + s_loc,
                            jnp.concatenate([even_128, odd_128], axis=0),
                        )

            return

        if kv_pack_dense2:
            kv_u32_all = (
                None
                if interpret
                else bkv_x2_ref.bitcast(jnp.uint32)
                .at[bkv_sem_idx]
                .reshape((bkv_sz // 2) * bkv_stride, 128)
            )

            @pl.when(valid_keys == bkv_sz)
            def _up_full_d2():
                _unpack_full_bkv_block(bkv_sem_idx, slot_base_s)

            @pl.when(valid_keys < bkv_sz)
            def _up_partial_d2():
                s_lim = cdiv(valid_keys, 256)

                @pl.loop(0, s_lim, unroll=False)
                def _up_one_d2(s_loc):
                    pair_start = s_loc * 128
                    pair_u32 = _bkv_slice_u32(
                        kv_u32_all, bkv_sem_idx, pair_start, 128
                    )
                    r_128 = lax.broadcasted_iota(jnp.int32, (128, 128), 0)
                    tok_even = s_loc * 256 + 2 * r_128
                    lo_u32 = jnp.where(
                        tok_even < valid_keys,
                        pair_u32 & jnp.uint32(0xFFFF),
                        jnp.uint32(0),
                    )
                    hi_u32 = jnp.where(
                        (tok_even + 1) < valid_keys,
                        pair_u32 >> jnp.uint32(16),
                        jnp.uint32(0),
                    )
                    if use_mk128:
                        _write_unpacked_kv_slice(
                            2 * (slot_base_s + s_loc), lo_u32
                        )
                        _write_unpacked_kv_slice(
                            2 * (slot_base_s + s_loc) + 1, hi_u32
                        )
                    else:
                        kv_u32 = jnp.concatenate([lo_u32, hi_u32], axis=0)
                        _write_unpacked_kv_slice(slot_base_s + s_loc, kv_u32)

            return

        kv_u32_all = (
            None
            if interpret
            else bkv_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx]
            .reshape(bkv_sz * bkv_stride, 128)
        )

        @pl.when(valid_keys == bkv_sz)
        def _up_full():
            _unpack_full_bkv_block(bkv_sem_idx, slot_base_s)

        @pl.when(valid_keys < bkv_sz)
        def _up_partial():
            if use_mk128:
                # Must unpack BOTH 128-row halves of every touched 256-key slice
                # (`2 * cdiv(valid_keys, 256)`), because `_run_slice_range` evaluates
                # `s_hi = cdiv(valid_keys, 256)` full 256-key slices (`2 * s_hi` halves)
                # and `0.0 * NaN = NaN` if the odd second half retains dirty VMEM bits
                # from prior Unified decode kernels!
                h_lim = 2 * cdiv(valid_keys, 256)

                @pl.loop(0, h_lim, unroll=False)
                def _up_one_mk(h_loc):
                    bkv_start = h_loc * 128
                    kv_u32_h = _bkv_slice_u32(
                        kv_u32_all, bkv_sem_idx, bkv_start, 128
                    )
                    k_pos = bkv_start + lax.broadcasted_iota(
                        jnp.int32, (128, 128), 0
                    )
                    kv_u32_h = jnp.where(
                        k_pos < valid_keys, kv_u32_h, jnp.uint32(0)
                    )
                    _write_unpacked_kv_slice(2 * slot_base_s + h_loc, kv_u32_h)

                return
            s_lim = cdiv(valid_keys, 256)

            @pl.loop(0, s_lim, unroll=False)
            def _up_one(s_loc):
                bkv_start = s_loc * 256
                kv_u32 = _bkv_slice_u32(kv_u32_all, bkv_sem_idx, bkv_start, 256)
                k_pos = bkv_start + lax.broadcasted_iota(jnp.int32, (256, 128), 0)
                kv_u32 = jnp.where(k_pos < valid_keys, kv_u32, jnp.uint32(0))
                _write_unpacked_kv_slice(slot_base_s + s_loc, kv_u32)

    @pl.loop(0, num_bq, unroll=False)
    def compute_with_bq_v2(bq_idx):
        bq_sem_idx = sem_ids_ref[0]
        next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
            seq_idx, bq_idx, bq_sem_idx
        )
        processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
        start_bkv_idx = jnp.int32(0)
        if sliding_window is not None:
            start_bkv_idx = jnp.maximum(processed_q_len - _sw_off, 0) // bkv_sz
        effective_kv_len = (
            jnp.minimum(kv_len, processed_q_len + actual_bq_sz)
            if use_causal_mask
            else kv_len
        )
        end_bkv_idx = cdiv(effective_kv_len, bkv_sz)
        valid_bq_rows = jnp.clip(q_len - bq_idx * actual_bq_sz, 0, actual_bq_sz)

        @pl.when(next_seq_idx < end_seq_idx)
        def prefetch_next_bq():
            if not use_staged_qo:
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

        def _decode_q_pair(u_s, s, lane_is_lo):
            if use_fp8_qk and not use_fp8_qk_dyn:
                q_enc_sc = float(1.0 if use_f32_exp else exp_scale)
                even_f = pltpu.bitcast(u_s << jnp.uint32(16), jnp.float32) * q_enc_sc
                odd_f = (
                    pltpu.bitcast(u_s & jnp.uint32(0xFFFF0000), jnp.float32) * q_enc_sc
                )
                if s < 4:
                    pair_q_f = jnp.where(
                        lane_is_lo,
                        even_f,
                        pltpu.roll(odd_f, shift=64, axis=1),
                    )
                else:
                    pair_q_f = jnp.where(
                        lane_is_lo,
                        pltpu.roll(even_f, shift=64, axis=1),
                        odd_f,
                    )
                return jnp.clip(
                    jnp.swapaxes(pair_q_f, 0, 1), -448.0, 448.0
                ).astype(jnp.float8_e4m3fn)
            if use_f32_exp:
                even_bf16 = pltpu.bitcast(u_s << jnp.uint32(16), jnp.float32).astype(
                    jnp.bfloat16
                )
                odd_bf16 = pltpu.bitcast(
                    u_s & jnp.uint32(0xFFFF0000), jnp.float32
                ).astype(jnp.bfloat16)
            else:
                even_bf16 = (
                    pltpu.bitcast(u_s << jnp.uint32(16), jnp.float32) * exp_scale
                ).astype(jnp.bfloat16)
                odd_bf16 = (
                    pltpu.bitcast(u_s & jnp.uint32(0xFFFF0000), jnp.float32) * exp_scale
                ).astype(jnp.bfloat16)
            if s < 4:
                pair_q_bf16 = jnp.where(
                    lane_is_lo,
                    even_bf16,
                    pltpu.roll(odd_bf16, shift=64, axis=1),
                )
            else:
                pair_q_bf16 = jnp.where(
                    lane_is_lo,
                    pltpu.roll(even_bf16, shift=64, axis=1),
                    odd_bf16,
                )
            return jnp.swapaxes(pair_q_bf16, 0, 1)

        def _store_qt_pair(pair_q_t, s, t_sl):
            if use_fp8_qk_dyn:
                q0_f8, q0_sc8 = _quant_qt_fp8(pair_q_t[:64, :])
                q1_f8, q1_sc8 = _quant_qt_fp8(pair_q_t[64:, :])
                q_t_ref[2 * s, :, t_sl] = q0_f8
                q_t_ref[2 * s + 1, :, t_sl] = q1_f8
                q_sc_ref[2 * s, :, t_sl] = q0_sc8
                q_sc_ref[2 * s + 1, :, t_sl] = q1_sc8
            else:
                q_t_ref[2 * s, :, t_sl] = pair_q_t[:64, :]
                q_t_ref[2 * s + 1, :, t_sl] = pair_q_t[64:, :]

        def _bq_pair_u32(q_u32_ref, sem_q, src_tq_start, s_pair):
            if interpret:
                u16 = lax.bitcast_convert_type(
                    bq_x2_ref[
                        sem_q, 0, pl.ds(src_tq_start, tq_sz), s_pair, :, :
                    ],
                    jnp.uint16,
                ).astype(jnp.uint32)
                return u16[:, 0, :] | (u16[:, 1, :] << jnp.uint32(16))
            return q_u32_ref[pl.ds(src_tq_start * sq_pairs + s_pair, tq_sz, sq_pairs)]

        def _unpack_one_tq_qt_from_ref(
            q_u32_ref, sem_q, src_tq_start, dst_tq_start, valid_tq, lane_is_lo
        ):
            t_sl = pl.ds(dst_tq_start, tq_sz)

            @pl.when(valid_tq == tq_sz)
            def _up_tq_full():
                for s in range(sq_pairs):
                    u_s = _bq_pair_u32(q_u32_ref, sem_q, src_tq_start, s)
                    pair_q_t = _decode_q_pair(u_s, (s // 4) * 4, lane_is_lo)
                    _store_qt_pair(pair_q_t, s, t_sl)

            @pl.when(jnp.logical_and(valid_tq > 0, valid_tq < tq_sz))
            def _up_tq_partial():
                q_row_io = lax.broadcasted_iota(jnp.int32, (tq_sz, 128), 0)
                row_ok = q_row_io < valid_tq
                for s in range(sq_pairs):
                    u_s = _bq_pair_u32(q_u32_ref, sem_q, src_tq_start, s)
                    u_s = jnp.where(row_ok, u_s, jnp.uint32(0))
                    pair_q_t = _decode_q_pair(u_s, (s // 4) * 4, lane_is_lo)
                    _store_qt_pair(pair_q_t, s, t_sl)

        def unpack_bq_to_qt_v2(sem_q):
            q_u32_ref = (
                None
                if interpret
                else bq_x2_ref.bitcast(jnp.uint32)
                .at[sem_q, 0]
                .reshape(bq_sz * sq_pairs, 128)
            )
            is_lo = lax.broadcasted_iota(jnp.int32, (tq_sz, 128), 1) < 64
            for tq_idx in range(bq_sz // tq_sz):
                tq_start = tq_idx * tq_sz
                valid_tq = jnp.clip(valid_bq_rows - tq_start, 0, tq_sz)
                if tq_idx == 0:
                    _unpack_one_tq_qt_from_ref(
                        q_u32_ref, sem_q, tq_start, tq_start, valid_tq, is_lo
                    )
                else:

                    @pl.when(tq_start < valid_bq_rows)
                    def _up_tail_tq(tq_start=tq_start, valid_tq=valid_tq):
                        _unpack_one_tq_qt_from_ref(
                            q_u32_ref, sem_q, tq_start, tq_start, valid_tq, is_lo
                        )

        def _wait_and_unpack_cur_bq(sem_q):
            if not use_staged_qo:
                wait_fetch_bq(seq_idx, bq_idx, sem_q)
                if probe != "nocompute":
                    unpack_bq_to_qt_v2(sem_q)
                return
            q_row_base = cu_q_lens_ref[seq_idx] + bq_idx * actual_bq_sz
            n_q_stages = cdiv(valid_bq_rows, tq_sz)
            is_lo = lax.broadcasted_iota(jnp.int32, (tq_sz, 128), 1) < 64

            @pl.when(n_q_stages > 0)
            def _do_staged_q():
                sz0 = jnp.minimum(tq_sz, valid_bq_rows)
                start_fetch_bq_stage(q_row_base, sz0, 0)

                @pl.loop(0, n_q_stages, unroll=False)
                def _q_st(st_idx):
                    st_slot = st_idx & 1
                    st_q_start = st_idx * tq_sz
                    st_sz = jnp.minimum(tq_sz, valid_bq_rows - st_q_start)
                    next_st = st_idx + 1

                    @pl.when(next_st < n_q_stages)
                    def _pf_next_q():
                        nxt_slot = next_st & 1
                        nxt_q_start = next_st * tq_sz
                        nxt_sz = jnp.minimum(tq_sz, valid_bq_rows - nxt_q_start)
                        start_fetch_bq_stage(q_row_base + nxt_q_start, nxt_sz, nxt_slot)

                    wait_fetch_bq_stage(q_row_base + st_q_start, st_sz, st_slot)
                    if probe != "nocompute":
                        q_u32_stage = (
                            None
                            if interpret
                            else bq_x2_ref.bitcast(jnp.uint32)
                            .at[st_slot, 0]
                            .reshape(tq_sz * sq_pairs, 128)
                        )
                        _unpack_one_tq_qt_from_ref(
                            q_u32_stage, st_slot, 0, st_q_start, st_sz, is_lo
                        )

        def _expand_mk128_specs(bias_specs):
            out_sp = []
            for bs in bias_specs:
                if bs == "none":
                    out_sp.extend(("none", "none"))
                elif bs == "lt":
                    out_sp.extend(("lt0", "lt1"))
                elif bs == "ut":
                    out_sp.extend(("ut0", "ut1"))
                elif isinstance(bs, tuple):
                    out_sp.extend(bs)
                else:
                    out_sp.extend((bs[:, :128, :], bs[:, 128:, :]))
            return tuple(out_sp)

        def _eval_mk128_head(
            kv_h,
            qt_1,
            qsc_1,
            h_base,
            n_slices,
            bias_specs,
            max_h_idx=None,
            h_indices=None,
        ):
            n_h = len(h_indices) if h_indices is not None else 2 * n_slices
            z_acc = jnp.zeros((1, 64, tq_sz), jnp.float32)
            kv_sl = (
                pl.ds(kv_h, 1)
                if not isinstance(kv_h, int)
                else slice(kv_h, kv_h + 1)
            )

            def _h_ring(h_idx):
                h_eff = (
                    jnp.minimum(h_idx, max_h_idx)
                    if max_h_idx is not None
                    else h_idx
                )
                return (h_eff & ring_h_mask) if ring_covers_all else h_eff

            def _h_sel(off):
                if h_indices is not None:
                    return h_indices[off]
                return _h_ring(h_base + off)

            def _read_k128(h_idx):
                return (
                    k_sub_ref[kv_h, h_idx]
                    if use_k64
                    else k_sub_ref[kv_h, h_idx][:, :64]
                )

            h_specs = (
                ("none",) * n_h
                if h_indices is not None
                else _expand_mk128_specs(bias_specs)
            )
            is_contig_ind = (
                h_indices is not None
                and use_k64
                and not interpret
                and len(h_indices) > 0
                and all(
                    isinstance(v, int) and v == idx_i
                    for idx_i, v in enumerate(h_indices)
                )
            )
            is_contig_pair_ring = (
                h_indices is None
                and sliding_window is None
                and use_k64
                and not interpret
                and max_h_idx is None
                and n_h >= 4
                and (n_h % 2) == 0
            )
            g_cat = 1
            if use_la2 and h_indices is not None:
                for _g in ((8, 4, 2) if is_contig_ind else (4, 2)):
                    if (n_h % _g) == 0 and n_h >= 2 * _g:
                        g_cat = _g
                        break
            elif use_la2 and is_contig_pair_ring:
                g_cat = 2
            if g_cat > 1:
                acc_cur = z_acc
                l_cur = jnp.zeros((1, 8, tq_sz), jnp.float32)
                k_flat_s2 = (
                    k_sub_ref.at[kv_h].reshape(k_sub_ref.shape[1] * 128, 64)
                    if (is_contig_ind or is_contig_pair_ring)
                    else None
                )

                def _kdot_blk(b0):
                    if is_contig_ind:
                        k_cat = k_flat_s2[pl.ds(b0 * 128, g_cat * 128)][None, :, :]
                    elif is_contig_pair_ring:
                        # h_base = 2*s_start and b0 are even -> _h_ring(h_base+b0) is even,
                        # so (h_even, h_even+1) are contiguous 256 rows with zero wraparound.
                        h_even = _h_ring(h_base + b0)
                        k_cat = k_flat_s2[pl.ds(h_even * 128, 256)][None, :, :]
                    else:
                        k_cat = jnp.concatenate(
                            [_read_k128(_h_sel(b0 + j)) for j in range(g_cat)],
                            axis=0,
                        )[None, :, :]
                    return _kq_dot(k_cat, qt_1)

                qk_cur = _kdot_blk(0)
                for b0 in range(g_cat, n_h, g_cat):
                    qk_nxt = _kdot_blk(b0)
                    pb = b0 - g_cat
                    for j in range(g_cat):
                        qk_j = qk_cur[:, j * 128 : (j + 1) * 128, :]
                        p_j, ps_j, psc_j = _sm_eval(qk_j, h_specs[pb + j], qsc_1)
                        l_cur = l_cur + ps_j
                        acc_cur = acc_cur + _vp_scaled(
                            vt_sub_ref[kv_sl, _h_sel(pb + j)], p_j, psc_j
                        )
                    qk_cur = qk_nxt
                pb = n_h - g_cat
                for j in range(g_cat):
                    qk_j = qk_cur[:, j * 128 : (j + 1) * 128, :]
                    p_j, ps_j, psc_j = _sm_eval(qk_j, h_specs[pb + j], qsc_1)
                    l_cur = l_cur + ps_j
                    acc_cur = acc_cur + _vp_scaled(
                        vt_sub_ref[kv_sl, _h_sel(pb + j)], p_j, psc_j
                    )
                return acc_cur, l_cur
            if use_la2 and n_h >= 2:
                h_r0 = _h_sel(0)
                h_r1 = _h_sel(1)
                qk_0 = _kq_dot(_read_k128(h_r0)[None, :, :], qt_1)
                qk_cur = _kq_dot(_read_k128(h_r1)[None, :, :], qt_1)
                p_prev, l_cur, psc_prev = _sm_eval(qk_0, h_specs[0], qsc_1)
                acc_cur = z_acc
                for h_off in range(2, n_h):
                    h_next = _h_sel(h_off)
                    h_vp = _h_sel(h_off - 2)
                    qk_next = _kq_dot(_read_k128(h_next)[None, :, :], qt_1)
                    acc_cur = acc_cur + _vp_scaled(
                        vt_sub_ref[kv_sl, h_vp], p_prev, psc_prev
                    )
                    p_prev, ps_c, psc_prev = _sm_eval(
                        qk_cur, h_specs[h_off - 1], qsc_1
                    )
                    l_cur = l_cur + ps_c
                    qk_cur = qk_next
                h_vp_penult = _h_sel(n_h - 2)
                h_vp_last = _h_sel(n_h - 1)
                acc_cur = acc_cur + _vp_scaled(
                    vt_sub_ref[kv_sl, h_vp_penult], p_prev, psc_prev
                )
                p_last, ps_last, psc_last = _sm_eval(
                    qk_cur, h_specs[n_h - 1], qsc_1
                )
                l_cur = l_cur + ps_last
                acc_cur = acc_cur + _vp_scaled(
                    vt_sub_ref[kv_sl, h_vp_last], p_last, psc_last
                )
                return acc_cur, l_cur

            h_r0 = _h_sel(0)
            qk_0 = _kq_dot(_read_k128(h_r0)[None, :, :], qt_1)
            p_prev, l_cur, psc_prev = _sm_eval(qk_0, h_specs[0], qsc_1)
            acc_cur = z_acc
            for h_off in range(1, n_h):
                h_c = _h_sel(h_off)
                h_p = _h_sel(h_off - 1)
                qk_c = _kq_dot(_read_k128(h_c)[None, :, :], qt_1)
                acc_cur = acc_cur + _vp_scaled(
                    vt_sub_ref[kv_sl, h_p], p_prev, psc_prev
                )
                p_prev, ps_c, psc_prev = _sm_eval(qk_c, h_specs[h_off], qsc_1)
                l_cur = l_cur + ps_c
            h_rl = _h_sel(n_h - 1)
            acc_cur = acc_cur + _vp_scaled(
                vt_sub_ref[kv_sl, h_rl], p_prev, psc_prev
            )
            return acc_cur, l_cur

        def _run_ring_chunk(
            s_start,
            n_slices,
            first_vis_cond,
            bias_specs,
            t_sl,
            *,
            static_first=False,
            topk_loader=None,
        ):
            if use_mk128:
                h_base = 2 * s_start

                def _run_one_mk_head(kv_h, g, h_ind):
                    g_sl = pl.ds(g, 1) if not isinstance(g, int) else slice(g, g + 1)
                    qt_1 = q_t_ref[g_sl, :, t_sl]
                    qsc_1 = (
                        jnp.tile(q_sc_ref[g_sl, :, t_sl], (1, 16, 1))
                        if use_fp8_qk_dyn
                        else None
                    )
                    acc_1, l_1 = _eval_mk128_head(
                        kv_h,
                        qt_1,
                        qsc_1,
                        h_base,
                        n_slices,
                        bias_specs,
                        h_indices=h_ind,
                    )
                    if static_first:
                        l_ref[g_sl, :, t_sl] = l_1
                        acc_ref[g_sl, :, t_sl] = acc_1
                    else:
                        l_ref[g_sl, :, t_sl] = jnp.where(
                            first_vis_cond,
                            l_1,
                            l_ref[g_sl, :, t_sl] + l_1,
                        )
                        acc_ref[g_sl, :, t_sl] = jnp.where(
                            first_vis_cond,
                            acc_1,
                            acc_ref[g_sl, :, t_sl] + acc_1,
                        )

                if n_slices <= 8 and topk_loader is None:
                    for kv_h in range(sq_kv_heads):
                        for g_off in range(8):
                            _run_one_mk_head(kv_h, kv_h * 8 + g_off, None)
                else:
                    for kv_h in range(sq_kv_heads):
                        h_ind = topk_loader(kv_h) if topk_loader is not None else None

                        @pl.loop(0, 4, unroll=False)
                        def _mk_pair_loop(gp_off, kv_h=kv_h, h_ind=h_ind):
                            g0 = kv_h * 8 + 2 * gp_off
                            _run_one_mk_head(kv_h, g0, h_ind)
                            _run_one_mk_head(kv_h, g0 + 1, h_ind)

                return

            if use_gqa2_vreg or use_gqa4_vreg:
                pairs_per_grp = 2 if use_gqa4_vreg else 1
                n_heads_grp = 2 * pairs_per_grp
                w_grp = n_heads_grp * tq_sz
                for kv_h in range(sq_kv_heads):
                    for gp in range(4 // pairs_per_grp):
                        g_base = kv_h * 8 + n_heads_grp * gp
                        qt_grp = jnp.concatenate(
                            [
                                q_t_ref[g : g + 1, :, t_sl]
                                for g in range(g_base, g_base + n_heads_grp)
                            ],
                            axis=2,
                        )
                        if use_fp8_qk_dyn:
                            qsc_grp_8 = jnp.concatenate(
                                [
                                    q_sc_ref[g : g + 1, :, t_sl]
                                    for g in range(g_base, g_base + n_heads_grp)
                                ],
                                axis=2,
                            )
                            q_exp_grp = jnp.tile(qsc_grp_8, (1, 32, 1))
                        else:
                            q_exp_grp = None
                        l_grp = jnp.zeros((1, 8, w_grp), jnp.float32)
                        acc_grp = jnp.zeros((1, 64, w_grp), jnp.float32)
                        mrb_g = []
                        rdy_g = None

                        def _adv_grp(q_exp_grp=q_exp_grp):
                            nonlocal rdy_g, l_grp
                            qk_g, loc_s, vt_s = mrb_g.pop(0)
                            p_g, ps_g, p_sc_g = _sm_eval_w(
                                qk_g, bias_specs[loc_s], w_grp, q_exp_grp
                            )
                            l_grp = l_grp + ps_g
                            rdy_g = (p_g, vt_s, p_sc_g)

                        def _fl_grp():
                            nonlocal rdy_g, acc_grp
                            p_g, vt_s, p_sc_g = rdy_g
                            acc_grp = acc_grp + _vp_scaled(vt_s, p_g, p_sc_g)
                            rdy_g = None

                        for loc_s in range(n_slices):
                            s_ring = (
                                (s_start + loc_s) & ring_s_mask
                                if ring_covers_all
                                else (s_start + loc_s)
                            )
                            ks = k_sub_ref[kv_h, s_ring][:, :64][None, :, :]
                            vts = vt_sub_ref[kv_h : kv_h + 1, s_ring]
                            qk_g = _kq_dot(ks, qt_grp)
                            if rdy_g is not None:
                                _fl_grp()
                            mrb_g.append((qk_g, loc_s, vts))
                            if len(mrb_g) >= mrb_depth:
                                _adv_grp()
                        while mrb_g:
                            if rdy_g is not None:
                                _fl_grp()
                            _adv_grp()
                        if rdy_g is not None:
                            _fl_grp()

                        for h_off in range(n_heads_grp):
                            g = g_base + h_off
                            c0 = h_off * tq_sz
                            c1 = c0 + tq_sz
                            l_add = l_grp[:, :, c0:c1]
                            a_add = acc_grp[:, :, c0:c1]
                            if static_first:
                                l_ref[g : g + 1, :, t_sl] = l_add
                                acc_ref[g : g + 1, :, t_sl] = a_add
                            else:
                                l_ref[g : g + 1, :, t_sl] = jnp.where(
                                    first_vis_cond,
                                    l_add,
                                    l_ref[g : g + 1, :, t_sl] + l_add,
                                )
                                acc_ref[g : g + 1, :, t_sl] = jnp.where(
                                    first_vis_cond,
                                    a_add,
                                    acc_ref[g : g + 1, :, t_sl] + a_add,
                                )
                return

            if use_vreg_acc1 and n_slices >= 4:
                for kv_h in range(sq_kv_heads):
                    for g_off in range(8):
                        g = kv_h * 8 + g_off
                        qt_1 = q_t_ref[g : g + 1, :, t_sl]
                        qsc_1 = (
                            jnp.tile(q_sc_ref[g : g + 1, :, t_sl], (1, 32, 1))
                            if use_fp8_qk_dyn
                            else None
                        )
                        l_1 = jnp.zeros((1, 8, tq_sz), jnp.float32)
                        acc_1 = jnp.zeros((1, 64, tq_sz), jnp.float32)
                        mrb_1 = []
                        rdy_1 = None

                        def _adv_1():
                            nonlocal rdy_1, l_1
                            qk_1, loc_s, vt_s, qs_1 = mrb_1.pop(0)
                            p_1, ps_1, psc_1 = _sm_eval(qk_1, bias_specs[loc_s], qs_1)
                            l_1 = l_1 + ps_1
                            rdy_1 = (p_1, vt_s, psc_1)

                        def _fl_1():
                            nonlocal rdy_1, acc_1
                            p_1, vt_s, psc_1 = rdy_1
                            acc_1 = acc_1 + _vp_scaled(vt_s, p_1, psc_1)
                            rdy_1 = None

                        for loc_s in range(n_slices):
                            s_ring = (
                                (s_start + loc_s) & ring_s_mask
                                if ring_covers_all
                                else (s_start + loc_s)
                            )
                            ks = k_sub_ref[kv_h, s_ring][:, :64][None, :, :]
                            vts = vt_sub_ref[kv_h : kv_h + 1, s_ring]
                            qk_1 = _kq_dot(ks, qt_1)
                            if rdy_1 is not None:
                                _fl_1()
                            mrb_1.append((qk_1, loc_s, vts, qsc_1))
                            if len(mrb_1) >= mrb_depth:
                                _adv_1()
                        while mrb_1:
                            if rdy_1 is not None:
                                _fl_1()
                            _adv_1()
                        if rdy_1 is not None:
                            _fl_1()

                        if static_first:
                            l_ref[g : g + 1, :, t_sl] = l_1
                            acc_ref[g : g + 1, :, t_sl] = acc_1
                        else:
                            l_ref[g : g + 1, :, t_sl] = jnp.where(
                                first_vis_cond,
                                l_1,
                                l_ref[g : g + 1, :, t_sl] + l_1,
                            )
                            acc_ref[g : g + 1, :, t_sl] = jnp.where(
                                first_vis_cond,
                                acc_1,
                                acc_ref[g : g + 1, :, t_sl] + acc_1,
                            )
                return

            if use_vreg_acc and n_slices >= 4:
                for kv_h in range(sq_kv_heads):
                    for gp in range(4):
                        g0 = kv_h * 8 + 2 * gp
                        g1 = g0 + 1
                        qt0 = q_t_ref[g0 : g0 + 1, :, t_sl]
                        qt1 = q_t_ref[g1 : g1 + 1, :, t_sl]
                        qsc0 = (
                            jnp.tile(q_sc_ref[g0 : g0 + 1, :, t_sl], (1, 32, 1))
                            if use_fp8_qk_dyn
                            else None
                        )
                        qsc1 = (
                            jnp.tile(q_sc_ref[g1 : g1 + 1, :, t_sl], (1, 32, 1))
                            if use_fp8_qk_dyn
                            else None
                        )
                        l0_reg = jnp.zeros((1, 8, tq_sz), jnp.float32)
                        l1_reg = jnp.zeros((1, 8, tq_sz), jnp.float32)
                        acc0_reg = jnp.zeros((1, 64, tq_sz), jnp.float32)
                        acc1_reg = jnp.zeros((1, 64, tq_sz), jnp.float32)
                        mrb_r = []
                        rdy_r = None

                        def _adv_r():
                            nonlocal rdy_r, l0_reg, l1_reg
                            qk0, qk1, loc_s, vt_s, qs0, qs1 = mrb_r.pop(0)
                            p0, ps0, psc0 = _sm_eval(qk0, bias_specs[loc_s], qs0)
                            p1, ps1, psc1 = _sm_eval(qk1, bias_specs[loc_s], qs1)
                            l0_reg = l0_reg + ps0
                            l1_reg = l1_reg + ps1
                            rdy_r = (p0, p1, vt_s, psc0, psc1)

                        def _fl_r():
                            nonlocal rdy_r, acc0_reg, acc1_reg
                            p0, p1, vt_s, psc0, psc1 = rdy_r
                            acc0_reg = acc0_reg + _vp_scaled(vt_s, p0, psc0)
                            acc1_reg = acc1_reg + _vp_scaled(vt_s, p1, psc1)
                            rdy_r = None

                        for loc_s in range(n_slices):
                            s_ring = (
                                (s_start + loc_s) & ring_s_mask
                                if ring_covers_all
                                else (s_start + loc_s)
                            )
                            ks = k_sub_ref[kv_h, s_ring][:, :64][None, :, :]
                            vts = vt_sub_ref[kv_h : kv_h + 1, s_ring]
                            qk0 = _kq_dot(ks, qt0)
                            qk1 = _kq_dot(ks, qt1)
                            if rdy_r is not None:
                                _fl_r()
                            mrb_r.append((qk0, qk1, loc_s, vts, qsc0, qsc1))
                            if len(mrb_r) >= mrb_depth:
                                _adv_r()
                        while mrb_r:
                            if rdy_r is not None:
                                _fl_r()
                            _adv_r()
                        if rdy_r is not None:
                            _fl_r()

                        if static_first:
                            l_ref[g0 : g0 + 1, :, t_sl] = l0_reg
                            l_ref[g1 : g1 + 1, :, t_sl] = l1_reg
                            acc_ref[g0 : g0 + 1, :, t_sl] = acc0_reg
                            acc_ref[g1 : g1 + 1, :, t_sl] = acc1_reg
                        else:
                            l_ref[g0 : g0 + 1, :, t_sl] = jnp.where(
                                first_vis_cond,
                                l0_reg,
                                l_ref[g0 : g0 + 1, :, t_sl] + l0_reg,
                            )
                            l_ref[g1 : g1 + 1, :, t_sl] = jnp.where(
                                first_vis_cond,
                                l1_reg,
                                l_ref[g1 : g1 + 1, :, t_sl] + l1_reg,
                            )
                            acc_ref[g0 : g0 + 1, :, t_sl] = jnp.where(
                                first_vis_cond,
                                acc0_reg,
                                acc_ref[g0 : g0 + 1, :, t_sl] + acc0_reg,
                            )
                            acc_ref[g1 : g1 + 1, :, t_sl] = jnp.where(
                                first_vis_cond,
                                acc1_reg,
                                acc_ref[g1 : g1 + 1, :, t_sl] + acc1_reg,
                            )
                return

            mrb_q = []
            ready_p = None

            def _adv2():
                nonlocal ready_p
                qk0, qk1, g0, g1, loc_s, vt_s, qsc0, qsc1 = mrb_q.pop(0)
                p0, ps0, psc0 = _sm_eval(qk0, bias_specs[loc_s], qsc0)
                p1, ps1, psc1 = _sm_eval(qk1, bias_specs[loc_s], qsc1)
                if loc_s == 0:
                    l_ref[g0 : g0 + 1, :, t_sl] = (
                        ps0
                        if static_first
                        else jnp.where(
                            first_vis_cond,
                            ps0,
                            l_ref[g0 : g0 + 1, :, t_sl] + ps0,
                        )
                    )
                    l_ref[g1 : g1 + 1, :, t_sl] = (
                        ps1
                        if static_first
                        else jnp.where(
                            first_vis_cond,
                            ps1,
                            l_ref[g1 : g1 + 1, :, t_sl] + ps1,
                        )
                    )
                else:
                    l_ref[g0 : g0 + 1, :, t_sl] = l_ref[g0 : g0 + 1, :, t_sl] + ps0
                    l_ref[g1 : g1 + 1, :, t_sl] = l_ref[g1 : g1 + 1, :, t_sl] + ps1
                ready_p = (p0, p1, g0, g1, loc_s, vt_s, psc0, psc1)

            def _fl2():
                nonlocal ready_p
                p0, p1, g0, g1, loc_s, vt_s, psc0, psc1 = ready_p
                vp0 = _vp_scaled(vt_s, p0, psc0)
                vp1 = _vp_scaled(vt_s, p1, psc1)
                if loc_s == 0:
                    acc_ref[g0 : g0 + 1, :, t_sl] = (
                        vp0
                        if static_first
                        else jnp.where(
                            first_vis_cond,
                            vp0,
                            acc_ref[g0 : g0 + 1, :, t_sl] + vp0,
                        )
                    )
                    acc_ref[g1 : g1 + 1, :, t_sl] = (
                        vp1
                        if static_first
                        else jnp.where(
                            first_vis_cond,
                            vp1,
                            acc_ref[g1 : g1 + 1, :, t_sl] + vp1,
                        )
                    )
                else:
                    acc_ref[g0 : g0 + 1, :, t_sl] = acc_ref[g0 : g0 + 1, :, t_sl] + vp0
                    acc_ref[g1 : g1 + 1, :, t_sl] = acc_ref[g1 : g1 + 1, :, t_sl] + vp1
                ready_p = None

            for kv_h in range(sq_kv_heads):
                for gp in range(4):
                    g0 = kv_h * 8 + 2 * gp
                    g1 = g0 + 1
                    qt0 = q_t_ref[g0 : g0 + 1, :, t_sl]
                    qt1 = q_t_ref[g1 : g1 + 1, :, t_sl]
                    qsc0 = (
                        jnp.tile(q_sc_ref[g0 : g0 + 1, :, t_sl], (1, 32, 1))
                        if use_fp8_qk_dyn
                        else None
                    )
                    qsc1 = (
                        jnp.tile(q_sc_ref[g1 : g1 + 1, :, t_sl], (1, 32, 1))
                        if use_fp8_qk_dyn
                        else None
                    )
                    for loc_s in range(n_slices):
                        s_ring = (
                            (s_start + loc_s) & ring_s_mask
                            if ring_covers_all
                            else (s_start + loc_s)
                        )
                        ks = k_sub_ref[kv_h, s_ring][:, :64][None, :, :]
                        vts = vt_sub_ref[kv_h : kv_h + 1, s_ring]
                        qk0 = _kq_dot(ks, qt0)
                        qk1 = _kq_dot(ks, qt1)
                        if ready_p is not None:
                            _fl2()
                        mrb_q.append((qk0, qk1, g0, g1, loc_s, vts, qsc0, qsc1))
                        if len(mrb_q) >= mrb_depth:
                            _adv2()
            while mrb_q:
                if ready_p is not None:
                    _fl2()
                _adv2()
            if ready_p is not None:
                _fl2()

        def _make_dyn_mask(q_first, key_pos_base, max_valid_kv):
            if use_mk128:
                q_pos_h = q_first + lax.broadcasted_iota(
                    jnp.int32, (1, 128, tq_sz), 2
                )
                k_128 = lax.broadcasted_iota(jnp.int32, (1, 128, tq_sz), 1)
                if kv_pack_bd4:
                    k_e = jnp.where(k_128 < 64, 4 * k_128, 4 * (k_128 - 64) + 2)
                    kp0 = key_pos_base + k_e
                    kp1 = key_pos_base + (k_e + 1)
                elif kv_pack_dense2:
                    kp0 = key_pos_base + 2 * k_128
                    kp1 = key_pos_base + (2 * k_128 + 1)
                else:
                    kp0 = key_pos_base + k_128
                    kp1 = key_pos_base + (k_128 + 128)
                m0 = kp0 < max_valid_kv
                m1 = kp1 < max_valid_kv
                if use_causal_mask:
                    m0 = jnp.logical_and(m0, q_pos_h >= kp0)
                    m1 = jnp.logical_and(m1, q_pos_h >= kp1)
                if sliding_window is not None:
                    m0 = jnp.logical_and(m0, q_pos_h < kp0 + sliding_window)
                    m1 = jnp.logical_and(m1, q_pos_h < kp1 + sliding_window)
                return (m0, m1)
            q_pos = q_first + lax.broadcasted_iota(jnp.int32, (1, 256, tq_sz), 2)
            k_raw = lax.broadcasted_iota(jnp.int32, (1, 256, tq_sz), 1)
            if kv_pack_bd4:
                k_h = k_raw % 128
                k_ev = jnp.where(k_h < 64, 4 * k_h, 4 * (k_h - 64) + 2)
                k_off = jnp.where(k_raw < 128, k_ev, k_ev + 1)
            elif kv_pack_dense2:
                k_off = jnp.where(k_raw < 128, 2 * k_raw, 2 * (k_raw - 128) + 1)
            else:
                k_off = k_raw
            kp = key_pos_base + k_off
            msk = kp < max_valid_kv
            if use_causal_mask:
                msk = jnp.logical_and(msk, q_pos >= kp)
            if sliding_window is not None:
                msk = jnp.logical_and(msk, q_pos < kp + sliding_window)
            return msk

        def _run_slice_range(
            q_first,
            s_lo,
            s_hi,
            s_unm_lo,
            s_unm_hi,
            global_first_s,
            base_kv_pos,
            max_valid_kv,
            t_sl,
        ):
            @pl.when(s_lo >= s_hi)
            def _zero_empty():
                l_ref[:, :, t_sl] = jnp.zeros((sq_q_heads, 8, tq_sz), jnp.float32)
                acc_ref[:, :, t_sl] = jnp.zeros(
                    (sq_q_heads, acc_ref.shape[1], tq_sz), jnp.float32
                )

            # 1. Left-masked boundary slices (SW left edge when not handled by sw9_fast)
            @pl.loop(s_lo, s_unm_lo, unroll=False)
            def _left_masked(s_cur):
                msk = _make_dyn_mask(q_first, base_kv_pos + s_cur * 256, max_valid_kv)
                _run_ring_chunk(
                    s_cur,
                    1,
                    (base_kv_pos // 256 + s_cur) == global_first_s,
                    (msk,),
                    t_sl,
                )

            # 2. Contiguous unmasked slices [s_unm_lo, s_unm_hi), binary-decomposed (8, 4, 2)
            #    plus paired (unmasked, causal-lt) when odd unmasked is followed by 1 aligned lt slice.
            #    For causal SW <= 2049, rem_unm <= 7 mathematically for all q_first, so omit dead _unm8 (-10.9k IMEM bundles!).
            n_unm = s_unm_lo
            rem_unm = s_unm_hi - s_unm_lo
            if (
                use_causal_mask
                and sliding_window is not None
                and sliding_window <= 2049
                and sliding_window_chunk_skip
            ):
                s_after8 = n_unm
            else:
                n8 = rem_unm // 8

                @pl.loop(0, n8, unroll=False)
                def _unm8(i8):
                    s_cur = s_unm_lo + i8 * 8
                    _run_ring_chunk(
                        s_cur,
                        8,
                        (base_kv_pos // 256 + s_cur) == global_first_s,
                        ("none",) * 8,
                        t_sl,
                    )

                s_after8 = n_unm + n8 * 8
            has4 = (rem_unm & 4) != 0

            @pl.when(has4)
            def _unm4():
                _run_ring_chunk(
                    s_after8,
                    4,
                    (base_kv_pos // 256 + s_after8) == global_first_s,
                    ("none",) * 4,
                    t_sl,
                )

            s_after4 = s_after8 + jnp.where(has4, jnp.int32(4), jnp.int32(0))
            has2 = (rem_unm & 2) != 0

            @pl.when(has2)
            def _unm2():
                _run_ring_chunk(
                    s_after4,
                    2,
                    (base_kv_pos // 256 + s_after4) == global_first_s,
                    ("none",) * 2,
                    t_sl,
                )

            s_after2 = s_after4 + jnp.where(has2, jnp.int32(2), jnp.int32(0))
            has1 = (rem_unm & 1) != 0
            n_right = s_hi - s_unm_hi
            can_lt_diag = jnp.bool_(False)
            if use_causal_mask and mask_lt_ref is not None:
                diag_kp = base_kv_pos + s_unm_hi * 256
                can_lt_diag = jnp.logical_and(
                    n_right == 1,
                    jnp.logical_and(
                        q_first == diag_kp,
                        diag_kp + 256 <= max_valid_kv,
                    ),
                )
                if sliding_window is not None:
                    can_lt_diag = jnp.logical_and(
                        can_lt_diag, q_first + tq_sz <= diag_kp + sliding_window
                    )

            pair_odd_lt = jnp.logical_and(has1, can_lt_diag)

            @pl.when(pair_odd_lt)
            def _run_paired_odd_lt():
                _run_ring_chunk(
                    s_after2,
                    2,
                    (base_kv_pos // 256 + s_after2) == global_first_s,
                    ("none", "lt"),
                    t_sl,
                )

            @pl.when(jnp.logical_and(has1, jnp.logical_not(pair_odd_lt)))
            def _unm1():
                _run_ring_chunk(
                    s_after2,
                    1,
                    (base_kv_pos // 256 + s_after2) == global_first_s,
                    ("none",),
                    t_sl,
                )

            # 3. Right-masked boundary slices (causal diagonal / sequence tail)
            @pl.when(jnp.logical_and(can_lt_diag, jnp.logical_not(has1)))
            def _right_lt1():
                _run_ring_chunk(
                    s_unm_hi,
                    1,
                    (base_kv_pos // 256 + s_unm_hi) == global_first_s,
                    ("lt",),
                    t_sl,
                )

            s_dyn_right_lo = jnp.where(can_lt_diag, s_hi, s_unm_hi)

            @pl.loop(s_dyn_right_lo, s_hi, unroll=False)
            def _right_dyn(s_cur):
                msk = _make_dyn_mask(q_first, base_kv_pos + s_cur * 256, max_valid_kv)
                _run_ring_chunk(
                    s_cur,
                    1,
                    (base_kv_pos // 256 + s_cur) == global_first_s,
                    (msk,),
                    t_sl,
                )

        bo_sem_idx = sem_ids_ref[2]
        sem_ids_ref[2] = (
            lax.rem(bo_sem_idx + 1, num_slots)
            if num_slots > 2
            else lax.select(bo_sem_idx == 0, 1, 0)
        )
        out_u32_ref = (
            None
            if (use_a3_out or use_staged_qo or interpret)
            else bo_x2_ref.bitcast(jnp.uint32)
            .at[bo_sem_idx, 0]
            .reshape(bq_sz * sq_pairs, 128)
        )
        is_lo_ep = lax.broadcasted_iota(jnp.int32, (tq_sz, 128), 1) < 64
        mask_lo_u32 = jnp.where(is_lo_ep, jnp.uint32(0xFFFFFFFF), jnp.uint32(0))
        mask_hi_u32 = jnp.where(is_lo_ep, jnp.uint32(0), jnp.uint32(0xFFFFFFFF))

        def _pair_col_f32(a0, a1, l0, l1):
            if use_sub8_l:
                l0 = jnp.broadcast_to(jnp.sum(l0, axis=0, keepdims=True), (8, tq_sz))
                l1 = jnp.broadcast_to(jnp.sum(l1, axis=0, keepdims=True), (8, tq_sz))
            inv_l0 = jnp.where(
                l0 > 0.0,
                pltpu.reciprocal(l0, approx=True) * v_scale_val,
                0.0,
            )
            inv_l1 = jnp.where(
                l1 > 0.0,
                pltpu.reciprocal(l1, approx=True) * v_scale_val,
                0.0,
            )
            o0_t = (
                (a0.reshape(8, 8, tq_sz) * inv_l0[None, :, :])
                .reshape(64, tq_sz)
                .astype(jnp.bfloat16)
            )
            o1_t = (
                (a1.reshape(8, 8, tq_sz) * inv_l1[None, :, :])
                .reshape(64, tq_sz)
                .astype(jnp.bfloat16)
            )
            pair_t = jnp.concatenate([o0_t, o1_t], axis=0)
            return jnp.swapaxes(pair_t, 0, 1).astype(jnp.float32) + 0.0

        def _pack_pair_out(a0, a1, l0, l1, s):
            if use_sub8_l:
                l0 = jnp.broadcast_to(jnp.sum(l0, axis=0, keepdims=True), (8, tq_sz))
                l1 = jnp.broadcast_to(jnp.sum(l1, axis=0, keepdims=True), (8, tq_sz))
            inv_l0 = jnp.where(
                l0 > 0.0,
                pltpu.reciprocal(l0, approx=True) * v_scale_val,
                0.0,
            )
            inv_l1 = jnp.where(
                l1 > 0.0,
                pltpu.reciprocal(l1, approx=True) * v_scale_val,
                0.0,
            )
            o0_t = (
                (a0.reshape(8, 8, tq_sz) * inv_l0[None, :, :])
                .reshape(64, tq_sz)
                .astype(jnp.bfloat16)
            )
            o1_t = (
                (a1.reshape(8, 8, tq_sz) * inv_l1[None, :, :])
                .reshape(64, tq_sz)
                .astype(jnp.bfloat16)
            )
            pair_t = jnp.concatenate([o0_t, o1_t], axis=0)
            pair = jnp.swapaxes(pair_t, 0, 1).astype(jnp.float32)
            rolled = pltpu.roll(pair, shift=64, axis=1)
            even = jnp.where(is_lo_ep, pair, rolled)
            odd = jnp.where(is_lo_ep, rolled, pair)
            lo_u32 = pltpu.bitcast(even, jnp.uint32)
            hi_u32 = pltpu.bitcast(odd, jnp.uint32)
            packed_u32 = lax.shift_right_logical(lo_u32, jnp.uint32(16)) | (
                hi_u32 & jnp.uint32(0xFFFF0000)
            )
            return packed_u32 & (mask_lo_u32 if s < 4 else mask_hi_u32)

        def _write_bo_pair_u32(target_u32_ref, sem_o, dst_tq_start, s_pair, val_u32):
            if interpret:
                lo16 = (val_u32 & jnp.uint32(0xFFFF)).astype(jnp.uint16)
                hi16 = (val_u32 >> jnp.uint32(16)).astype(jnp.uint16)
                pair_bf16 = lax.bitcast_convert_type(
                    jnp.stack([lo16, hi16], axis=1), jnp.bfloat16
                )
                bo_x2_ref[
                    sem_o, 0, pl.ds(dst_tq_start, tq_sz), s_pair, :, :
                ] = pair_bf16
                return
            target_u32_ref[pl.ds(dst_tq_start * sq_pairs + s_pair, tq_sz, sq_pairs)] = (
                val_u32
            )

        def _epilogue_one_tq(tq_start, t_sl):
            for s in range(sq_pairs):
                h0 = 2 * s
                h1 = 2 * s + 1
                l0 = l_ref[h0, :, t_sl]
                l1 = l_ref[h1, :, t_sl]
                a0 = acc_ref[h0, :, t_sl]
                a1 = acc_ref[h1, :, t_sl]
                _write_bo_pair_u32(
                    out_u32_ref,
                    bo_sem_idx,
                    tq_start,
                    s,
                    _pack_pair_out(a0, a1, l0, l1, (s // 4) * 4),
                )

        def _emit_unfused_epilogue_and_send():
            if use_staged_qo:
                n_o_stages = cdiv(valid_bq_rows, tq_sz)
                row_iota_st = lax.broadcasted_iota(jnp.int32, (tq_sz + 16, 128), 0)
                zero_tail_st = jnp.zeros((16, 128), jnp.float32)

                @pl.loop(0, n_o_stages, unroll=False)
                def _o_st(st_idx):
                    st_slot = st_idx & 1
                    wait_send_bo(st_slot)
                    st_q_start = st_idx * tq_sz
                    t_sl = pl.ds(st_q_start, tq_sz)
                    st_valid = jnp.minimum(tq_sz, valid_bq_rows - st_q_start)
                    if use_a3_out:
                        g0_st = cu_q_lens_ref[seq_idx] + bq_idx * actual_bq_sz + st_q_start
                        aligned_g0_st = (g0_st // 16) * 16
                        lead_st = g0_st - aligned_g0_st
                        total_rows_st = lead_st + st_valid
                        full_rows_st = (total_rows_st // 16) * 16

                        for s in range(sq_pairs):
                            h0 = 2 * s
                            h1 = 2 * s + 1
                            c_sl = slice(s * 128, (s + 1) * 128)
                            col_s = jnp.concatenate(
                                [
                                    _pair_col_f32(
                                        acc_ref[h0, :, t_sl],
                                        acc_ref[h1, :, t_sl],
                                        l_ref[h0, :, t_sl],
                                        l_ref[h1, :, t_sl],
                                    ),
                                    zero_tail_st,
                                ],
                                axis=0,
                            )
                            col_s = jnp.where(row_iota_st < st_valid, col_s, 0.0)

                            @pl.when(lead_st == 0)
                            def _store_al_st(c_sl=c_sl, col_s=col_s):
                                bo_x2_ref[st_slot, :, c_sl] = col_s.astype(
                                    jnp.bfloat16
                                )

                            @pl.when(lead_st > 0)
                            def _store_sh_st(c_sl=c_sl, col_s=col_s):
                                shifted = col_s
                                for sh in (1, 2, 4, 8):
                                    shifted = jnp.where(
                                        (lead_st & sh) != 0,
                                        pltpu.roll(shifted, shift=sh, axis=0),
                                        shifted,
                                    )
                                carry_s = bo_carry_ref[:, c_sl].astype(jnp.float32)
                                carry_pad = jnp.concatenate(
                                    [
                                        carry_s,
                                        jnp.zeros((tq_sz, 128), jnp.float32),
                                    ],
                                    axis=0,
                                )
                                merged = jnp.where(
                                    row_iota_st < lead_st, carry_pad, shifted
                                )
                                bo_x2_ref[st_slot, :, c_sl] = merged.astype(
                                    jnp.bfloat16
                                )

                        bo_carry_ref[...] = bo_x2_ref[
                            st_slot,
                            pl.ds(pl.multiple_of((total_rows_st // 16) * 16, 16), 16),
                            :,
                        ]
                        start_send_bo_a3(aligned_g0_st, full_rows_st, st_slot)
                    else:
                        out_u32_st = (
                            None
                            if interpret
                            else bo_x2_ref.bitcast(jnp.uint32)
                            .at[st_slot, 0]
                            .reshape(tq_sz * sq_pairs, 128)
                        )
                        for s in range(sq_pairs):
                            h0 = 2 * s
                            h1 = 2 * s + 1
                            _write_bo_pair_u32(
                                out_u32_st,
                                st_slot,
                                0,
                                s,
                                _pack_pair_out(
                                    acc_ref[h0, :, t_sl],
                                    acc_ref[h1, :, t_sl],
                                    l_ref[h0, :, t_sl],
                                    l_ref[h1, :, t_sl],
                                    (s // 4) * 4,
                                ),
                            )

                        g0_st = cu_q_lens_ref[seq_idx] + bq_idx * actual_bq_sz + st_q_start
                        start_send_bo_staged(g0_st, st_valid, st_slot)
                return

            if use_a3_out:
                g0_ep = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
                valid_ep = jnp.minimum(bq_sz, cu_q_lens_ref[seq_idx + 1] - g0_ep)
                aligned_g0_ep = (g0_ep // 16) * 16
                lead_ep = g0_ep - aligned_g0_ep
                total_rows_ep = lead_ep + valid_ep
                full_rows_ep = (total_rows_ep // 16) * 16
                row_iota_ep = lax.broadcasted_iota(jnp.int32, (bq_sz + 16, 128), 0)
                zero_tail_16 = jnp.zeros((16, 128), jnp.float32)

                for s in range(sq_pairs):
                    h0 = 2 * s
                    h1 = 2 * s + 1
                    c_sl = slice(s * 128, (s + 1) * 128)
                    parts_s = []
                    for tq_idx in range(bq_sz // tq_sz):
                        tq_start = tq_idx * tq_sz
                        t_sl = pl.ds(tq_start, tq_sz)
                        parts_s.append(
                            _pair_col_f32(
                                acc_ref[h0, :, t_sl],
                                acc_ref[h1, :, t_sl],
                                l_ref[h0, :, t_sl],
                                l_ref[h1, :, t_sl],
                            )
                        )
                    col_s = jnp.concatenate([*parts_s, zero_tail_16], axis=0)
                    col_s = jnp.where(row_iota_ep < valid_ep, col_s, 0.0)

                    @pl.when(lead_ep == 0)
                    def _store_aligned_v2(c_sl=c_sl, col_s=col_s):
                        bo_x2_ref[bo_sem_idx, :, c_sl] = col_s.astype(
                            jnp.bfloat16
                        )

                    @pl.when(lead_ep > 0)
                    def _store_shifted_v2(c_sl=c_sl, col_s=col_s):
                        shifted = col_s
                        for sh in (1, 2, 4, 8):
                            shifted = jnp.where(
                                (lead_ep & sh) != 0,
                                pltpu.roll(shifted, shift=sh, axis=0),
                                shifted,
                            )
                        carry_s = bo_carry_ref[:, c_sl].astype(jnp.float32)
                        carry_pad = jnp.concatenate(
                            [
                                carry_s,
                                jnp.zeros((bq_sz, 128), jnp.float32),
                            ],
                            axis=0,
                        )
                        merged = jnp.where(
                            row_iota_ep < lead_ep, carry_pad, shifted
                        )
                        bo_x2_ref[bo_sem_idx, :, c_sl] = merged.astype(
                            jnp.bfloat16
                        )

                bo_carry_ref[...] = bo_x2_ref[
                    bo_sem_idx,
                    pl.ds(pl.multiple_of((total_rows_ep // 16) * 16, 16), 16),
                    :,
                ]
                start_send_bo_a3(aligned_g0_ep, full_rows_ep, bo_sem_idx)
                return

            for tq_idx in range(bq_sz // tq_sz):
                tq_start = tq_idx * tq_sz
                t_sl = pl.ds(tq_start, tq_sz)
                if tq_idx == 0:
                    _epilogue_one_tq(tq_start, t_sl)
                else:

                    @pl.when(tq_start < valid_bq_rows)
                    def _epi_tail_tq(tq_start=tq_start, t_sl=t_sl):
                        _epilogue_one_tq(tq_start, t_sl)

            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        allow_sw_fuse = (sliding_window == 2049) or (
            sliding_window == 2048 and "sw2048_fuse" in prefill_opts
        )
        can_fuse_sw9 = (
            use_fuse_qe
            and bq_sz == 256
            and allow_sw_fuse
            and use_causal_mask
            and sliding_window_chunk_skip
            and mask_ut_ref is not None
            and mask_lt_ref is not None
            and ring_covers_all
        )

        def _send_fused_bo(valid_rows=None):
            if use_a3_out:
                g0_f = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
                aligned_g0_f = (g0_f // 16) * 16
                lead_f = g0_f - aligned_g0_f
                if valid_rows is None:
                    full_rows_f = tq_sz
                    row_iota_f = lax.broadcasted_iota(
                        jnp.int32, (tq_sz + 16, 128), 0
                    )
                    zero_tail_f = jnp.zeros((16, 128), jnp.float32)

                    @pl.when(lead_f == 0)
                    def _zero_carry_f():
                        bo_carry_ref[...] = jnp.zeros_like(bo_carry_ref)

                    @pl.when(lead_f > 0)
                    def _shift_and_merge_carry_f():
                        @pl.loop(0, sq_pairs, unroll=False)
                        def _sh_f_pair(s_f):
                            c_sl_f = pl.ds(pl.multiple_of(s_f * 128, 128), 128)
                            c_f = jnp.concatenate(
                                [
                                    bo_x2_ref[
                                        bo_sem_idx,
                                        :tq_sz,
                                        c_sl_f,
                                    ].astype(jnp.float32),
                                    zero_tail_f,
                                ],
                                axis=0,
                            )
                            for sh in (1, 2, 4, 8):
                                c_f = jnp.where(
                                    (lead_f & sh) != 0,
                                    pltpu.roll(c_f, shift=sh, axis=0),
                                    c_f,
                                )
                            carry_f = bo_carry_ref[:, c_sl_f].astype(jnp.float32)
                            carry_pad_f = jnp.concatenate(
                                [
                                    carry_f,
                                    jnp.zeros((tq_sz, 128), jnp.float32),
                                ],
                                axis=0,
                            )
                            merged_f = jnp.where(
                                row_iota_f < lead_f, carry_pad_f, c_f
                            )
                            bo_x2_ref[bo_sem_idx, :, c_sl_f] = merged_f.astype(
                                jnp.bfloat16
                            )

                        bo_carry_ref[...] = bo_x2_ref[
                            bo_sem_idx, pl.ds(tq_sz, 16), :
                        ]

                    start_send_bo_a3(aligned_g0_f, full_rows_f, bo_sem_idx)
                    return

                total_rows_t = lead_f + valid_rows
                full_rows_t = (total_rows_t // 16) * 16
                row_iota_t = lax.broadcasted_iota(
                    jnp.int32, (tq_sz + 16, 128), 0
                )
                zero_tail_t = jnp.zeros((16, 128), jnp.float32)

                @pl.loop(0, sq_pairs, unroll=False)
                def _sh_t_pair(s_t):
                    c_sl_t = pl.ds(pl.multiple_of(s_t * 128, 128), 128)
                    c_t = jnp.concatenate(
                        [
                            bo_x2_ref[
                                bo_sem_idx, :tq_sz, c_sl_t
                            ].astype(jnp.float32),
                            zero_tail_t,
                        ],
                        axis=0,
                    )
                    c_t = jnp.where(row_iota_t < valid_rows, c_t, 0.0)

                    @pl.when(lead_f == 0)
                    def _t_al():
                        bo_x2_ref[bo_sem_idx, :, c_sl_t] = c_t.astype(
                            jnp.bfloat16
                        )

                    @pl.when(lead_f > 0)
                    def _t_sh():
                        sh_t = c_t
                        for sh in (1, 2, 4, 8):
                            sh_t = jnp.where(
                                (lead_f & sh) != 0,
                                pltpu.roll(sh_t, shift=sh, axis=0),
                                sh_t,
                            )
                        carry_t = bo_carry_ref[:, c_sl_t].astype(jnp.float32)
                        carry_pad_t = jnp.concatenate(
                            [
                                carry_t,
                                jnp.zeros((tq_sz, 128), jnp.float32),
                            ],
                            axis=0,
                        )
                        merged_t = jnp.where(
                            row_iota_t < lead_f, carry_pad_t, sh_t
                        )
                        bo_x2_ref[bo_sem_idx, :, c_sl_t] = merged_t.astype(
                            jnp.bfloat16
                        )

                bo_carry_ref[...] = bo_x2_ref[
                    bo_sem_idx, pl.ds(pl.multiple_of(full_rows_t, 16), 16), :
                ]
                start_send_bo_a3(aligned_g0_f, full_rows_t, bo_sem_idx)
            else:
                start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        def _write_sw9_pair_out(a0, a1, l0, l1, s_pair, kv_h_tag=0):
            if use_a3_out:
                c_sl_sw = pl.ds(pl.multiple_of(s_pair * 128, 128), 128)
                bo_x2_ref[
                    bo_sem_idx,
                    :tq_sz,
                    c_sl_sw,
                ] = _pair_col_f32(a0, a1, l0, l1).astype(jnp.bfloat16)
            else:
                _write_bo_pair_u32(
                    out_u32_ref,
                    bo_sem_idx,
                    0,
                    s_pair,
                    _pack_pair_out(a0, a1, l0, l1, kv_h_tag * 4),
                )

        def _get_bq_u32_ref():
            return (
                None
                if interpret
                else bq_x2_ref.bitcast(jnp.uint32)
                .at[bq_sem_idx, 0]
                .reshape(256 * sq_pairs, 128)
            )

        def _run_sw_mk128_fused(
            s_lo, n_slices, bias_specs, valid_rows=None, max_h_idx=None
        ):
            q_u32_ref = _get_bq_u32_ref()
            h_base_sw = 2 * s_lo
            q_row_ok = (
                (lax.broadcasted_iota(jnp.int32, (tq_sz, 128), 0) < valid_rows)
                if valid_rows is not None
                else None
            )
            for kv_h in range(sq_kv_heads):
                for gp in range(4):
                    s_pair = kv_h * 4 + gp
                    u_s = _bq_pair_u32(q_u32_ref, bq_sem_idx, 0, s_pair)
                    if q_row_ok is not None:
                        u_s = jnp.where(q_row_ok, u_s, jnp.uint32(0))
                    pair_q_t = _decode_q_pair(u_s, kv_h * 4, is_lo_ep)
                    if use_fp8_qk_dyn:
                        q0_f8, q0_sc8 = _quant_qt_fp8(pair_q_t[:64, :])
                        q1_f8, q1_sc8 = _quant_qt_fp8(pair_q_t[64:, :])
                        qt_pair = (q0_f8[None, :, :], q1_f8[None, :, :])
                        qsc_pair = (
                            jnp.tile(q0_sc8[None, :, :], (1, 16, 1)),
                            jnp.tile(q1_sc8[None, :, :], (1, 16, 1)),
                        )
                    else:
                        qt_pair = (
                            pair_q_t[:64, :][None, :, :],
                            pair_q_t[64:, :][None, :, :],
                        )
                        qsc_pair = (None, None)
                    acc0, l0 = _eval_mk128_head(
                        kv_h,
                        qt_pair[0],
                        qsc_pair[0],
                        h_base_sw,
                        n_slices,
                        bias_specs,
                        max_h_idx=max_h_idx,
                    )
                    acc1, l1 = _eval_mk128_head(
                        kv_h,
                        qt_pair[1],
                        qsc_pair[1],
                        h_base_sw,
                        n_slices,
                        bias_specs,
                        max_h_idx=max_h_idx,
                    )
                    _write_sw9_pair_out(
                        acc0[0], acc1[0], l0[0], l1[0], s_pair, kv_h
                    )

        def _sw10_unalign_bias_specs(w_off):
            q_rel_w = (
                lax.broadcasted_iota(jnp.int32, (1, 128, tq_sz), 2) + w_off
            )
            k_128 = lax.broadcasted_iota(jnp.int32, (1, 128, tq_sz), 1)
            if kv_pack_bd4:
                k_r0 = jnp.where(k_128 < 64, 4 * k_128, 4 * (k_128 - 64) + 2)
                k_r1 = k_r0 + 1
            elif kv_pack_dense2:
                k_r0 = 2 * k_128
                k_r1 = 2 * k_128 + 1
            else:
                k_r0 = k_128
                k_r1 = k_128 + 128
            z_bf16 = jnp.bfloat16(0.0)
            neg_inf = jnp.bfloat16(-jnp.inf)

            def _pair(c0, c1):
                return (
                    jnp.where(c0, z_bf16, neg_inf),
                    jnp.where(c1, z_bf16, neg_inf),
                )

            ut_s0 = _pair(q_rel_w <= k_r0, q_rel_w <= k_r1)
            ut_s1 = _pair(q_rel_w <= (k_r0 + 256), q_rel_w <= (k_r1 + 256))
            if sliding_window == 2048:
                lt_s8 = _pair(q_rel_w > k_r0, q_rel_w > k_r1)
                lt_s9 = _pair(q_rel_w > (k_r0 + 256), q_rel_w > (k_r1 + 256))
            else:
                lt_s8 = _pair(q_rel_w >= k_r0, q_rel_w >= k_r1)
                lt_s9 = _pair(q_rel_w >= (k_r0 + 256), q_rel_w >= (k_r1 + 256))
            return (ut_s0, ut_s1) + ("none",) * 6 + (lt_s8, lt_s9)

        def _run_sw10_fused_qe(s_lo, w_off, valid_rows=None, max_h_idx=None):
            _run_sw_mk128_fused(
                s_lo,
                10,
                _sw10_unalign_bias_specs(w_off),
                valid_rows=valid_rows,
                max_h_idx=max_h_idx,
            )

        def _run_sw9_fused_qe(s_lo):
            bias_specs = ("ut",) + ("none",) * 7 + ("lt",)
            if use_mk128:
                _run_sw_mk128_fused(s_lo, 9, bias_specs)
                return
            q_u32_ref = _get_bq_u32_ref()
            t_sl = pl.ds(0, tq_sz)

            if use_gqa2_vreg or use_gqa4_vreg:
                pairs_per_grp = 2 if use_gqa4_vreg else 1
                w_grp = 2 * pairs_per_grp * tq_sz
                for kv_h in range(sq_kv_heads):
                    for grp_i in range(4 // pairs_per_grp):
                        qt_parts = []
                        for p_off in range(pairs_per_grp):
                            sp = kv_h * 4 + grp_i * pairs_per_grp + p_off
                            u_sp = _bq_pair_u32(q_u32_ref, bq_sem_idx, 0, sp)
                            pq_t = _decode_q_pair(u_sp, kv_h * 4, is_lo_ep)
                            qt_parts.append(pq_t[:64, :])
                            qt_parts.append(pq_t[64:, :])
                        qt_cat = jnp.concatenate(qt_parts, axis=1)
                        if use_fp8_qk_dyn:
                            qt_f8, qsc_8 = _quant_qt_fp8(qt_cat)
                            qt_dot = qt_f8[None, :, :]
                            q_exp_256 = jnp.tile(qsc_8[None, :, :], (1, 32, 1))
                        else:
                            qt_dot = qt_cat[None, :, :]
                            q_exp_256 = None

                        l_grp = jnp.zeros((1, 8, w_grp), jnp.float32)
                        acc_grp = jnp.zeros((1, 64, w_grp), jnp.float32)
                        mrb_g = []
                        rdy_g = None

                        def _adv_g(q_exp_256=q_exp_256):
                            nonlocal rdy_g, l_grp
                            qk_g, loc_s, vt_s = mrb_g.pop(0)
                            p_g, ps_g, p_sc_g = _sm_eval_w(
                                qk_g, bias_specs[loc_s], w_grp, q_exp_256
                            )
                            l_grp = l_grp + ps_g
                            rdy_g = (p_g, vt_s, p_sc_g)

                        def _fl_g():
                            nonlocal rdy_g, acc_grp
                            p_g, vt_s, p_sc_g = rdy_g
                            acc_grp = acc_grp + _vp_scaled(vt_s, p_g, p_sc_g)
                            rdy_g = None

                        for loc_s in range(9):
                            s_ring = (s_lo + loc_s) & ring_s_mask
                            ks = k_sub_ref[kv_h, s_ring][:, :64][None, :, :]
                            vts = vt_sub_ref[kv_h : kv_h + 1, s_ring]
                            qk_g = _kq_dot(ks, qt_dot)
                            if rdy_g is not None:
                                _fl_g()
                            mrb_g.append((qk_g, loc_s, vts))
                            if len(mrb_g) >= 2:
                                _adv_g()
                        while mrb_g:
                            if rdy_g is not None:
                                _fl_g()
                            _adv_g()
                        if rdy_g is not None:
                            _fl_g()

                        for p_off in range(pairs_per_grp):
                            sp = kv_h * 4 + grp_i * pairs_per_grp + p_off
                            c0 = (2 * p_off) * tq_sz
                            c1 = (2 * p_off + 1) * tq_sz
                            c2 = (2 * p_off + 2) * tq_sz
                            _write_sw9_pair_out(
                                acc_grp[0, :, c0:c1],
                                acc_grp[0, :, c1:c2],
                                l_grp[0, :, c0:c1],
                                l_grp[0, :, c1:c2],
                                sp,
                                kv_h,
                            )
                return

            if use_vreg_acc or use_vreg_acc1:
                sw_mrb = 1 if use_vreg_acc1 and mrb_depth == 1 else 2
                for kv_h in range(sq_kv_heads):
                    for gp in range(4):
                        s_pair = kv_h * 4 + gp
                        u_s = _bq_pair_u32(q_u32_ref, bq_sem_idx, 0, s_pair)
                        pair_q_t = _decode_q_pair(u_s, kv_h * 4, is_lo_ep)
                        if use_fp8_qk_dyn:
                            q0_f8, q0_sc8 = _quant_qt_fp8(pair_q_t[:64, :])
                            q1_f8, q1_sc8 = _quant_qt_fp8(pair_q_t[64:, :])
                            qt_pair = (q0_f8[None, :, :], q1_f8[None, :, :])
                            qsc_pair = (
                                jnp.tile(q0_sc8[None, :, :], (1, 32, 1)),
                                jnp.tile(q1_sc8[None, :, :], (1, 32, 1)),
                            )
                        else:
                            qt_pair = (
                                pair_q_t[:64, :][None, :, :],
                                pair_q_t[64:, :][None, :, :],
                            )
                            qsc_pair = (None, None)

                        acc_out = []
                        l_out = []
                        for h_sub in range(2):
                            qt_1h = qt_pair[h_sub]
                            qsc_1h = qsc_pair[h_sub]
                            l_1h = jnp.zeros((1, 8, tq_sz), jnp.float32)
                            acc_1h = jnp.zeros((1, 64, tq_sz), jnp.float32)
                            mrb_1 = []
                            rdy_1 = None

                            def _adv_1h(qsc_1h=qsc_1h):
                                nonlocal rdy_1, l_1h
                                qk_1, loc_s, vt_s = mrb_1.pop(0)
                                p_1, ps_1, psc_1 = _sm_eval(qk_1, bias_specs[loc_s], qsc_1h)
                                l_1h = l_1h + ps_1
                                rdy_1 = (p_1, vt_s, psc_1)

                            def _fl_1h():
                                nonlocal rdy_1, acc_1h
                                p_1, vt_s, psc_1 = rdy_1
                                acc_1h = acc_1h + _vp_scaled(vt_s, p_1, psc_1)
                                rdy_1 = None

                            for loc_s in range(9):
                                s_ring = (s_lo + loc_s) & ring_s_mask
                                ks = k_sub_ref[kv_h, s_ring][:, :64][None, :, :]
                                vts = vt_sub_ref[kv_h : kv_h + 1, s_ring]
                                qk_1 = _kq_dot(ks, qt_1h)
                                if rdy_1 is not None:
                                    _fl_1h()
                                mrb_1.append((qk_1, loc_s, vts))
                                if len(mrb_1) >= sw_mrb:
                                    _adv_1h()
                            while mrb_1:
                                if rdy_1 is not None:
                                    _fl_1h()
                                _adv_1h()
                            if rdy_1 is not None:
                                _fl_1h()
                            acc_out.append(acc_1h[0])
                            l_out.append(l_1h[0])

                        _write_sw9_pair_out(
                            acc_out[0], acc_out[1], l_out[0], l_out[1], s_pair, kv_h
                        )
                return

            mrb_q = []
            ready_p = None

            def _adv_f():
                nonlocal ready_p
                qk0, qk1, s_pair, kv_h_tag, loc_s, vt_s, qsc0, qsc1 = mrb_q.pop(0)
                g0 = 2 * s_pair
                g1 = g0 + 1
                p0, ps0, psc0 = _sm_eval(qk0, bias_specs[loc_s], qsc0)
                p1, ps1, psc1 = _sm_eval(qk1, bias_specs[loc_s], qsc1)
                if loc_s == 0:
                    l_ref[g0 : g0 + 1, :, t_sl] = ps0
                    l_ref[g1 : g1 + 1, :, t_sl] = ps1
                else:
                    l_ref[g0 : g0 + 1, :, t_sl] = l_ref[g0 : g0 + 1, :, t_sl] + ps0
                    l_ref[g1 : g1 + 1, :, t_sl] = l_ref[g1 : g1 + 1, :, t_sl] + ps1
                ready_p = (p0, p1, s_pair, kv_h_tag, loc_s, vt_s, psc0, psc1)

            def _fl_f():
                nonlocal ready_p
                p0, p1, s_pair, kv_h_tag, loc_s, vt_s, psc0, psc1 = ready_p
                g0 = 2 * s_pair
                g1 = g0 + 1
                vp0 = _vp_scaled(vt_s, p0, psc0)
                vp1 = _vp_scaled(vt_s, p1, psc1)
                if loc_s == 0:
                    acc_ref[g0 : g0 + 1, :, t_sl] = vp0
                    acc_ref[g1 : g1 + 1, :, t_sl] = vp1
                elif loc_s < 8:
                    acc_ref[g0 : g0 + 1, :, t_sl] = acc_ref[g0 : g0 + 1, :, t_sl] + vp0
                    acc_ref[g1 : g1 + 1, :, t_sl] = acc_ref[g1 : g1 + 1, :, t_sl] + vp1
                else:
                    a0 = acc_ref[g0, :, t_sl] + vp0[0]
                    a1 = acc_ref[g1, :, t_sl] + vp1[0]
                    l0 = l_ref[g0, :, t_sl]
                    l1 = l_ref[g1, :, t_sl]
                    _write_sw9_pair_out(a0, a1, l0, l1, s_pair, kv_h_tag)
                ready_p = None

            for kv_h in range(sq_kv_heads):
                for gp in range(4):
                    s_pair = kv_h * 4 + gp
                    u_s = _bq_pair_u32(q_u32_ref, bq_sem_idx, 0, s_pair)
                    pair_q_t = _decode_q_pair(u_s, kv_h * 4, is_lo_ep)
                    if use_fp8_qk_dyn:
                        q0_f8, q0_sc8 = _quant_qt_fp8(pair_q_t[:64, :])
                        q1_f8, q1_sc8 = _quant_qt_fp8(pair_q_t[64:, :])
                        qt0 = q0_f8[None, :, :]
                        qt1 = q1_f8[None, :, :]
                        qsc0 = jnp.tile(q0_sc8[None, :, :], (1, 32, 1))
                        qsc1 = jnp.tile(q1_sc8[None, :, :], (1, 32, 1))
                    else:
                        qt0 = pair_q_t[:64, :][None, :, :]
                        qt1 = pair_q_t[64:, :][None, :, :]
                        qsc0 = None
                        qsc1 = None
                    for loc_s in range(9):
                        s_ring = (s_lo + loc_s) & ring_s_mask
                        ks = k_sub_ref[kv_h, s_ring][:, :64][None, :, :]
                        vts = vt_sub_ref[kv_h : kv_h + 1, s_ring]
                        qk0 = _kq_dot(ks, qt0)
                        qk1 = _kq_dot(ks, qt1)
                        if ready_p is not None:
                            _fl_f()
                        mrb_q.append((qk0, qk1, s_pair, kv_h, loc_s, vts, qsc0, qsc1))
                        if len(mrb_q) >= mrb_depth:
                            _adv_f()
            while mrb_q:
                if ready_p is not None:
                    _fl_f()
                _adv_f()
            if ready_p is not None:
                _fl_f()

        if ring_covers_all:
            prev_unpacked = sem_ids_ref[3]

            @pl.loop(prev_unpacked, end_bkv_idx, unroll=False)
            def _fetch_and_unpack_ring(fetch_bkv_idx):
                bkv_sem_idx = sem_ids_ref[1]
                is_last_seq_bkv = (fetch_bkv_idx + 1) == seq_end_bkv
                next_seq_bkv = lax.select(is_last_seq_bkv, seq_idx + 1, seq_idx)
                next_bkv_to_fetch = lax.select(
                    is_last_seq_bkv, next_seq_start_bkv_idx, fetch_bkv_idx + 1
                )
                next_bkv_sem_idx = (
                    lax.rem(bkv_sem_idx + 1, num_slots)
                    if num_slots > 2
                    else lax.select(bkv_sem_idx == 0, 1, 0)
                )

                @pl.when(next_seq_bkv < end_seq_idx)
                def _pf_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_bkv, next_bkv_to_fetch, next_bkv_sem_idx)

                offset, update_sz = wait_fetch_bkv(seq_idx, fetch_bkv_idx, bkv_sem_idx)

                @pl.when(update_sz > 0)
                def _upd_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                if probe != "nocompute":
                    slot_base_s = (fetch_bkv_idx & (ring_bkv_blocks - 1)) * num_s
                    seq_bkv_valid = jnp.clip(kv_len - fetch_bkv_idx * bkv_sz, 0, bkv_sz)
                    _unpack_bkv_block(
                        bkv_sem_idx, fetch_bkv_idx, slot_base_s, seq_bkv_valid
                    )

            @pl.when(end_bkv_idx > prev_unpacked)
            def _advance_unpacked_cursor():
                sem_ids_ref[3] = end_bkv_idx

            if can_fuse_sw9 and probe != "nocompute":
                wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)
                wait_send_bo(bo_sem_idx)
                q_first_0 = processed_q_len
                is_sw9_fast_0 = jnp.logical_and(
                    valid_bq_rows == tq_sz,
                    jnp.logical_and(
                        (q_first_0 & 255) == 0,
                        jnp.logical_and(
                            q_first_0 >= 2048,
                            q_first_0 + tq_sz <= kv_len,
                        ),
                    ),
                )
                sw_left_0 = q_first_0 - _sw_off
                is_sw10_fast_0 = (
                    jnp.logical_and(
                        valid_bq_rows == tq_sz,
                        jnp.logical_and(
                            (q_first_0 & 255) != 0,
                            jnp.logical_and(
                                (sw_left_0 & 255) != 0,
                                jnp.logical_and(
                                    q_first_0 >= 2048,
                                    q_first_0 + tq_sz <= kv_len,
                                ),
                            ),
                        ),
                    )
                    if (use_sw_unalign_fuse and use_mk128)
                    else jnp.bool_(False)
                )
                is_sw_short_0 = (
                    jnp.logical_and(
                        valid_bq_rows == tq_sz,
                        jnp.logical_and(
                            (q_first_0 & 255) == 0,
                            jnp.logical_and(
                                q_first_0 < 2048,
                                q_first_0 + tq_sz <= kv_len,
                            ),
                        ),
                    )
                    if use_sw_short_fuse
                    else jnp.bool_(False)
                )
                is_sw_tail_0 = (
                    jnp.logical_and(
                        valid_bq_rows < tq_sz,
                        jnp.logical_and(
                            valid_bq_rows > 0,
                            jnp.logical_and(
                                q_first_0 >= 2048,
                                q_first_0 + valid_bq_rows <= kv_len,
                            ),
                        ),
                    )
                    if use_sw_tail_clamp
                    else jnp.bool_(False)
                )

                @pl.when(is_sw9_fast_0)
                def _do_fused_sw9():
                    s_lo_0 = (q_first_0 - 2048) // 256
                    _run_sw9_fused_qe(s_lo_0)
                    _send_fused_bo()

                if use_sw_unalign_fuse or use_sw_tail_clamp:

                    @pl.when(jnp.logical_or(is_sw10_fast_0, is_sw_tail_0))
                    def _do_fused_sw10():
                        w_off_0 = sw_left_0 & 255
                        s_lo_0 = sw_left_0 // 256
                        if use_sw_tail_clamp:
                            max_h_0 = 2 * cdiv(effective_kv_len, 256) - 1
                            _run_sw10_fused_qe(
                                s_lo_0,
                                w_off_0,
                                valid_rows=valid_bq_rows,
                                max_h_idx=max_h_0,
                            )
                            _send_fused_bo(valid_bq_rows)
                        else:
                            _run_sw10_fused_qe(s_lo_0, w_off_0)
                            _send_fused_bo()

                if use_sw_short_fuse:

                    @pl.when(is_sw_short_0)
                    def _do_fused_sw_short():
                        n_unm_0 = q_first_0 // 256
                        if use_sw_short_v2:
                            t_sl0 = pl.ds(0, tq_sz)
                            unpack_bq_to_qt_v2(bq_sem_idx)
                            has4_s = (n_unm_0 & 4) != 0
                            has2_s = (n_unm_0 & 2) != 0
                            has1_s = (n_unm_0 & 1) != 0

                            @pl.when(has4_s)
                            def _sws_4():
                                _run_ring_chunk(
                                    0,
                                    4,
                                    jnp.bool_(True),
                                    ("none",) * 4,
                                    t_sl0,
                                    static_first=True,
                                )

                            s_a4 = jnp.where(has4_s, jnp.int32(4), jnp.int32(0))

                            @pl.when(has2_s)
                            def _sws_2():
                                _run_ring_chunk(
                                    s_a4, 2, s_a4 == 0, ("none",) * 2, t_sl0
                                )

                            s_a2 = s_a4 + jnp.where(has2_s, jnp.int32(2), jnp.int32(0))

                            @pl.when(has1_s)
                            def _sws_1():
                                _run_ring_chunk(
                                    s_a2, 2, s_a2 == 0, ("none", "lt"), t_sl0
                                )

                            @pl.when(jnp.logical_not(has1_s))
                            def _sws_lt_even():
                                _run_ring_chunk(
                                    n_unm_0, 1, n_unm_0 == 0, ("lt",), t_sl0
                                )

                            for s_p in range(sq_pairs):
                                h0_p = 2 * s_p
                                h1_p = h0_p + 1
                                _write_sw9_pair_out(
                                    acc_ref[h0_p, :, t_sl0],
                                    acc_ref[h1_p, :, t_sl0],
                                    l_ref[h0_p, :, t_sl0],
                                    l_ref[h1_p, :, t_sl0],
                                    s_p,
                                    s_p // 4,
                                )
                            _send_fused_bo()
                            return
                        q_u32_ref = (
                            bq_x2_ref.bitcast(jnp.uint32)
                            .at[bq_sem_idx, 0]
                            .reshape(256 * sq_pairs, 128)
                        )
                        if use_mk128:
                            n_unm_h = 2 * n_unm_0
                            h_lt0 = (2 * n_unm_0) & ring_h_mask
                            h_lt1 = (2 * n_unm_0 + 1) & ring_h_mask
                            for kv_h in range(sq_kv_heads):

                                @pl.loop(0, 4, unroll=False)
                                def _sws_mk_gp(gp, kv_h=kv_h):
                                    s_pair = kv_h * 4 + gp
                                    u_s = q_u32_ref[pl.ds(s_pair, tq_sz, sq_pairs)]
                                    pair_q_t = _decode_q_pair(
                                        u_s, kv_h * 4, is_lo_ep
                                    )
                                    if use_fp8_qk_dyn:
                                        q0_f8, q0_sc8 = _quant_qt_fp8(pair_q_t[:64, :])
                                        q1_f8, q1_sc8 = _quant_qt_fp8(pair_q_t[64:, :])
                                        qt_pair = (q0_f8[None, :, :], q1_f8[None, :, :])
                                        qsc_pair = (
                                            jnp.tile(q0_sc8[None, :, :], (1, 16, 1)),
                                            jnp.tile(q1_sc8[None, :, :], (1, 16, 1)),
                                        )
                                    else:
                                        qt_pair = (
                                            pair_q_t[:64, :][None, :, :],
                                            pair_q_t[64:, :][None, :, :],
                                        )
                                        qsc_pair = (None, None)
                                    acc_out = []
                                    l_out = []
                                    for h_sub in range(2):
                                        qt_1h = qt_pair[h_sub]
                                        qsc_1h = qsc_pair[h_sub]
                                        z_acc = jnp.zeros((1, 64, tq_sz), jnp.float32)
                                        z_l = jnp.zeros((1, 8, tq_sz), jnp.float32)

                                        def _unm_h_step(
                                            h_idx,
                                            carry,
                                            kv_h=kv_h,
                                            qt_1h=qt_1h,
                                            qsc_1h=qsc_1h,
                                        ):
                                            acc_c, l_c = carry
                                            h_r = h_idx & ring_h_mask
                                            ks_c = (
                                                k_sub_ref[kv_h, h_r]
                                                if use_k64
                                                else k_sub_ref[kv_h, h_r][:, :64]
                                            )[None, :, :]
                                            vts_c = vt_sub_ref[kv_h : kv_h + 1, h_r]
                                            qk_c = _kq_dot(ks_c, qt_1h)
                                            p_c, ps_c, psc_c = _sm_eval(
                                                qk_c, "none", qsc_1h
                                            )
                                            return (
                                                acc_c + _vp_scaled(vts_c, p_c, psc_c),
                                                l_c + ps_c,
                                            )

                                        acc_1h, l_1h = lax.fori_loop(
                                            0, n_unm_h, _unm_h_step, (z_acc, z_l)
                                        )
                                        for h_lt, b_lt in (
                                            (h_lt0, "lt0"),
                                            (h_lt1, "lt1"),
                                        ):
                                            qk_lt = _kq_dot(
                                                (
                                                    k_sub_ref[kv_h, h_lt]
                                                    if use_k64
                                                    else k_sub_ref[kv_h, h_lt][:, :64]
                                                )[None, :, :],
                                                qt_1h,
                                            )
                                            p_lt, ps_lt, psc_lt = _sm_eval(
                                                qk_lt, b_lt, qsc_1h
                                            )
                                            acc_1h = acc_1h + _vp_scaled(
                                                vt_sub_ref[kv_h : kv_h + 1, h_lt],
                                                p_lt,
                                                psc_lt,
                                            )
                                            l_1h = l_1h + ps_lt
                                        acc_out.append(acc_1h[0])
                                        l_out.append(l_1h[0])
                                    _write_sw9_pair_out(
                                        acc_out[0],
                                        acc_out[1],
                                        l_out[0],
                                        l_out[1],
                                        s_pair,
                                        kv_h,
                                    )

                            _send_fused_bo()
                        else:
                            s_lt_ring = n_unm_0 & ring_s_mask
                            for kv_h in range(sq_kv_heads):
                                ks_lt = k_sub_ref[kv_h, s_lt_ring][:, :64][None, :, :]
                                vts_lt = vt_sub_ref[kv_h : kv_h + 1, s_lt_ring]
                                for gp in range(4):
                                    s_pair = kv_h * 4 + gp
                                    u_s = q_u32_ref[pl.ds(s_pair, tq_sz, sq_pairs)]
                                    pair_q_t = _decode_q_pair(u_s, kv_h * 4, is_lo_ep)
                                    if use_fp8_qk_dyn:
                                        q0_f8, q0_sc8 = _quant_qt_fp8(pair_q_t[:64, :])
                                        q1_f8, q1_sc8 = _quant_qt_fp8(pair_q_t[64:, :])
                                        qt_pair = (q0_f8[None, :, :], q1_f8[None, :, :])
                                        qsc_pair = (
                                            jnp.tile(q0_sc8[None, :, :], (1, 32, 1)),
                                            jnp.tile(q1_sc8[None, :, :], (1, 32, 1)),
                                        )
                                    else:
                                        qt_pair = (
                                            pair_q_t[:64, :][None, :, :],
                                            pair_q_t[64:, :][None, :, :],
                                        )
                                        qsc_pair = (None, None)

                                    acc_out = []
                                    l_out = []
                                    for h_sub in range(2):
                                        qt_1h = qt_pair[h_sub]
                                        qsc_1h = qsc_pair[h_sub]
                                        z_acc = jnp.zeros((1, 64, tq_sz), jnp.float32)
                                        z_l = jnp.zeros((1, 8, tq_sz), jnp.float32)

                                        def _unm_step(
                                            s_idx,
                                            carry,
                                            kv_h=kv_h,
                                            qt_1h=qt_1h,
                                            qsc_1h=qsc_1h,
                                        ):
                                            acc_c, l_c = carry
                                            s_r = s_idx & ring_s_mask
                                            ks_c = k_sub_ref[kv_h, s_r][:, :64][
                                                None, :, :
                                            ]
                                            vts_c = vt_sub_ref[kv_h : kv_h + 1, s_r]
                                            qk_c = _kq_dot(ks_c, qt_1h)
                                            p_c, ps_c, psc_c = _sm_eval(
                                                qk_c, "none", qsc_1h
                                            )
                                            return (
                                                acc_c + _vp_scaled(vts_c, p_c, psc_c),
                                                l_c + ps_c,
                                            )

                                        acc_1h, l_1h = lax.fori_loop(
                                            0, n_unm_0, _unm_step, (z_acc, z_l)
                                        )
                                        qk_lt = _kq_dot(ks_lt, qt_1h)
                                        p_lt, ps_lt, psc_lt = _sm_eval(
                                            qk_lt, "lt", qsc_1h
                                        )
                                        acc_1h = acc_1h + _vp_scaled(
                                            vts_lt, p_lt, psc_lt
                                        )
                                        l_1h = l_1h + ps_lt
                                        acc_out.append(acc_1h[0])
                                        l_out.append(l_1h[0])

                                    _write_sw9_pair_out(
                                        acc_out[0],
                                        acc_out[1],
                                        l_out[0],
                                        l_out[1],
                                        s_pair,
                                        kv_h,
                                    )
                            _send_fused_bo()

                @pl.when(
                    jnp.logical_not(
                        jnp.logical_or(
                            jnp.logical_or(
                                is_sw9_fast_0,
                                jnp.logical_or(is_sw10_fast_0, is_sw_tail_0),
                            ),
                            is_sw_short_0,
                        )
                    )
                )
                def _do_unfused_fallback():
                    unpack_bq_to_qt_v2(bq_sem_idx)
                    strip_kv_hi = jnp.minimum(kv_len, q_first_0 + tq_sz)
                    s_hi_0 = cdiv(strip_kv_hi, 256)
                    s_lo_0 = jnp.minimum(
                        jnp.maximum(q_first_0 - sliding_window + 1, 0) // 256,
                        s_hi_0,
                    )
                    s_unm_lo_0 = jnp.clip(
                        cdiv(
                            jnp.maximum(q_first_0 + tq_sz - sliding_window, 0),
                            256,
                        ),
                        s_lo_0,
                        s_hi_0,
                    )
                    s_unm_hi_0 = jnp.clip(
                        jnp.minimum(kv_len, q_first_0 + 1) // 256,
                        s_unm_lo_0,
                        s_hi_0,
                    )
                    _run_slice_range(
                        q_first_0,
                        s_lo_0,
                        s_hi_0,
                        s_unm_lo_0,
                        s_unm_hi_0,
                        s_lo_0,
                        0,
                        kv_len,
                        pl.ds(0, tq_sz),
                    )
                    _emit_unfused_epilogue_and_send()

                return

            if can_fuse_sw9:
                wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)
            else:
                _wait_and_unpack_cur_bq(bq_sem_idx)

            if probe != "nocompute":

                def _exec_tq_ring(tq_idx):
                    tq_start = tq_idx * tq_sz
                    t_sl = pl.ds(tq_start, tq_sz)

                    @pl.when(tq_start < valid_bq_rows)
                    def _do_tq():
                        q_first = processed_q_len + tq_start
                        strip_kv_hi = (
                            jnp.minimum(kv_len, q_first + tq_sz)
                            if use_causal_mask
                            else kv_len
                        )
                        s_hi = cdiv(strip_kv_hi, 256)
                        if sliding_window is not None and sliding_window_chunk_skip:
                            s_lo = jnp.minimum(
                                jnp.maximum(q_first - sliding_window + 1, 0) // 256,
                                s_hi,
                            )
                            s_unm_lo = jnp.clip(
                                cdiv(
                                    jnp.maximum(q_first + tq_sz - sliding_window, 0),
                                    256,
                                ),
                                s_lo,
                                s_hi,
                            )
                        else:
                            s_lo = jnp.int32(0)
                            s_unm_lo = jnp.int32(0)
                        max_unm_end = (
                            jnp.minimum(kv_len, q_first + 1)
                            if use_causal_mask
                            else kv_len
                        )
                        s_unm_hi = jnp.clip(max_unm_end // 256, s_unm_lo, s_hi)

                        if (
                            allow_sw_fuse
                            and use_causal_mask
                            and sliding_window_chunk_skip
                            and mask_ut_ref is not None
                            and mask_lt_ref is not None
                        ):
                            is_sw9_fast = jnp.logical_and(
                                (q_first & 255) == 0,
                                jnp.logical_and(
                                    q_first >= 2048,
                                    q_first + tq_sz <= kv_len,
                                ),
                            )

                            @pl.when(is_sw9_fast)
                            def _sw9_fast():
                                _run_ring_chunk(
                                    s_lo,
                                    9,
                                    jnp.bool_(True),
                                    ("ut",) + ("none",) * 7 + ("lt",),
                                    t_sl,
                                    static_first=True,
                                )

                            @pl.when(jnp.logical_not(is_sw9_fast))
                            def _sw9_general():
                                _run_slice_range(
                                    q_first,
                                    s_lo,
                                    s_hi,
                                    s_unm_lo,
                                    s_unm_hi,
                                    s_lo,
                                    0,
                                    kv_len,
                                    t_sl,
                                )
                        else:
                            _run_slice_range(
                                q_first,
                                s_lo,
                                s_hi,
                                s_unm_lo,
                                s_unm_hi,
                                s_lo,
                                0,
                                kv_len,
                                t_sl,
                            )

                if use_tq_loop and (bq_sz // tq_sz) > 1:
                    pl.loop(0, cdiv(valid_bq_rows, tq_sz), unroll=False)(_exec_tq_ring)
                else:
                    for _tqi in range(bq_sz // tq_sz):
                        _exec_tq_ring(_tqi)

        else:
            # Streaming fallback for >32K-context full-attention sequences
            @pl.when(start_bkv_idx >= end_bkv_idx)
            def _init_empty_stream():
                l_ref[...] = jnp.zeros_like(l_ref)
                acc_ref[...] = jnp.zeros_like(acc_ref)

            use_prefix_fast = (
                "prefix_fast" in prefill_opts
                and sliding_window is None
                and use_causal_mask
            )
            tail_bkv_start = (
                jnp.clip(processed_q_len // bkv_sz, start_bkv_idx, end_bkv_idx)
                if use_prefix_fast
                else start_bkv_idx
            )
            use_prefix_topk_inner = (
                prefix_topk_ref is not None
                and cu_qgrps_ref is not None
                and prefix_topk_slices > 0
            )
            n_qgrps_per_bq = actual_bq_sz // prefix_q_grp_sz
            n_i32_topk = max(prefix_topk_slices // 4, 1)

            def _make_od_unpacker(cur_sem_idx):
                phys_r = (
                    (bkv_sz // 4)
                    if kv_pack_bd4
                    else ((bkv_sz // 2) if kv_pack_dense2 else bkv_sz)
                )
                kv_u32_all_od = (
                    None
                    if interpret
                    else bkv_x2_ref.bitcast(jnp.uint32)
                    .at[cur_sem_idx]
                    .reshape(phys_r * bkv_stride, 128)
                )

                def _unpack_one_h(kv_h_i, dst_h, src_h):
                    if kv_pack_dense2:
                        s_loc = src_h >> 1
                        is_hi = (src_h & 1) != 0
                        pair_u32 = _bkv_slice_u32(
                            kv_u32_all_od, cur_sem_idx, s_loc * 128, 128
                        )
                        kv_u32 = jnp.where(
                            is_hi,
                            pair_u32 >> jnp.uint32(16),
                            pair_u32 & jnp.uint32(0xFFFF),
                        )
                    elif kv_pack_bd4:
                        s_loc = src_h >> 1
                        is_hi = (src_h & 1) != 0
                        pair_u32 = _bkv_slice_u32(
                            kv_u32_all_od, cur_sem_idx, s_loc * 64, 64
                        )
                        lo = pair_u32 & jnp.uint32(0xFFFF)
                        hi = pair_u32 >> jnp.uint32(16)
                        kv_u32 = jnp.where(
                            is_hi,
                            jnp.concatenate([_roll64(lo), _roll64(hi)], axis=0),
                            jnp.concatenate([lo, hi], axis=0),
                        )
                    else:
                        kv_u32 = _bkv_slice_u32(
                            kv_u32_all_od, cur_sem_idx, src_h * 128, 128
                        )
                    col_sl = slice(0, 64) if kv_h_i == 0 else slice(64, 128)
                    k_sub_ref[kv_h_i, dst_h] = pltpu.bitcast(
                        kv_u32[:, col_sl].astype(jnp.uint8), kv_dtype
                    )
                    v_bf16 = pltpu.bitcast(
                        (kv_u32[:, col_sl] >> jnp.uint32(8)).astype(jnp.uint8),
                        kv_dtype,
                    ).astype(jnp.bfloat16)
                    vt_sub_ref[kv_h_i, dst_h] = jnp.swapaxes(v_bf16, 0, 1)

                return _unpack_one_h

            def _load_topk_indices(kv_h, cur_bkv_idx, row_off):
                qgrp_id = (
                    cu_qgrps_ref[seq_idx]
                    + bq_idx * n_qgrps_per_bq
                    + (row_off // prefix_q_grp_sz)
                )
                base_off = (
                    (qgrp_id * prefix_max_bkv + cur_bkv_idx) * sq_kv_heads
                    + kv_h
                ) * n_i32_topk
                inds = []
                for j4 in range(n_i32_topk):
                    pk = prefix_topk_ref[base_off + j4]
                    for sh in (0, 8, 16, 24):
                        inds.append((pk >> sh) & 0xFF)
                return tuple(inds)

            if use_prefix_fast:

                @pl.loop(start_bkv_idx, tail_bkv_start, unroll=False)
                def _stream_prefix_bkv(bkv_idx):
                    bkv_sem_idx = sem_ids_ref[1]
                    (
                        next_seq_idx_kv,
                        _,
                        next_bkv_idx,
                        next_bkv_sem_idx,
                    ) = get_next_bkv_ids(
                        seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx
                    )

                    @pl.when(next_seq_idx_kv < end_seq_idx)
                    def _pf_prefix_bkv():
                        sem_ids_ref[1] = next_bkv_sem_idx
                        start_fetch_bkv(next_seq_idx_kv, next_bkv_idx, next_bkv_sem_idx)

                    @pl.when(bkv_idx == start_bkv_idx)
                    def _wait_up_bq_prefix():
                        _wait_and_unpack_cur_bq(bq_sem_idx)
                        if (
                            use_mk128
                            and "gqa8" in prefill_opts
                            and not use_fp8_qk_dyn
                            and probe != "nocompute"
                        ):
                            n_tq_p = cdiv(valid_bq_rows, 256)

                            @pl.loop(0, n_tq_p, unroll=False)
                            def _pack_q_gqa8(i256):
                                t_sl_p = pl.ds(pl.multiple_of(i256 * 256, 256), 256)
                                sl_lo = pl.ds(pl.multiple_of(i256 * 256, 256), 128)
                                sl_hi = pl.ds(
                                    pl.multiple_of(i256 * 256 + 128, 128), 128
                                )
                                for kv_h in range(sq_kv_heads):
                                    h_sl = slice(kv_h * 8, (kv_h + 1) * 8)
                                    qh8_lo = q_t_ref[h_sl, :, sl_lo]
                                    qh8_hi = q_t_ref[h_sl, :, sl_hi]
                                    slots_p = []
                                    for q_half in (qh8_lo, qh8_hi):
                                        for sub_c in range(4):
                                            c0 = sub_c * 32
                                            slots_p.append(
                                                jnp.concatenate(
                                                    [
                                                        q_half[g_i, :, c0 : c0 + 32]
                                                        for g_i in range(8)
                                                    ],
                                                    axis=1,
                                                )
                                            )
                                    q_t_ref[h_sl, :, t_sl_p] = jnp.stack(
                                        slots_p, axis=0
                                    )

                    wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                    if probe != "nocompute":
                        is_first_bkv = bkv_idx == start_bkv_idx
                        use_sparse_od = (
                            use_prefix_topk_inner
                            and use_mk128
                            and use_k64
                            and not use_fp8_qk_dyn
                            and not use_vt_fp8
                            and (prefix_q_grp_sz % 256 == 0)
                            and ("gqa8" in prefill_opts or sq_kv_heads == 1)
                        )
                        if not use_sparse_od:
                            _unpack_full_bkv_block(bkv_sem_idx)
                        elif prefix_topk_sink_slices <= 0:

                            @pl.when(is_first_bkv)
                            def _up_dense_first():
                                _unpack_full_bkv_block(bkv_sem_idx)

                        _unpack_single_h_kv = (
                            _make_od_unpacker(bkv_sem_idx) if use_sparse_od else None
                        )

                        def _exec_prefix_pass(static_first: bool, with_topk: bool):
                            if use_sparse_od and with_topk:
                                tq_per_qgrp = prefix_q_grp_sz // 256
                                n_qgrps_v = cdiv(valid_bq_rows, prefix_q_grp_sz)
                                static_h_ind = tuple(range(prefix_topk_slices))
                                if "gqa8" in prefill_opts:

                                    @pl.loop(0, n_qgrps_v, unroll=False)
                                    def _exec_od_qg_g8(qg_i):
                                        qgrp_row = qg_i * prefix_q_grp_sz
                                        for kv_h in range(sq_kv_heads):
                                            h_ind = _load_topk_indices(
                                                kv_h, bkv_idx, qgrp_row
                                            )
                                            for k_i in range(prefix_topk_slices):
                                                _unpack_single_h_kv(
                                                    kv_h, k_i, h_ind[k_i]
                                                )

                                            @pl.loop(0, tq_per_qgrp, unroll=False)
                                            def _od_i256(loc_256, kv_h=kv_h):
                                                i256 = qg_i * tq_per_qgrp + loc_256
                                                t_sl_p = pl.ds(
                                                    pl.multiple_of(i256 * 256, 256),
                                                    256,
                                                )
                                                for half_i in range(2):
                                                    q128_start = (
                                                        i256 * 256 + half_i * 128
                                                    )

                                                    @pl.when(
                                                        q128_start < valid_bq_rows
                                                    )
                                                    def _run_half_od():
                                                        for sub_c in range(4):
                                                            r_idx = (
                                                                kv_h * 8
                                                                + half_i * 4
                                                                + sub_c
                                                            )
                                                            r_sl = slice(
                                                                r_idx, r_idx + 1
                                                            )
                                                            acc_8h, l_8h = (
                                                                _eval_mk128_head(
                                                                    kv_h,
                                                                    q_t_ref[
                                                                        r_sl, :, t_sl_p
                                                                    ],
                                                                    None,
                                                                    0,
                                                                    num_s,
                                                                    ("none",) * num_s,
                                                                    h_indices=static_h_ind,
                                                                )
                                                            )
                                                            if static_first:
                                                                l_ref[
                                                                    r_sl, :, t_sl_p
                                                                ] = l_8h
                                                                acc_ref[
                                                                    r_sl, :, t_sl_p
                                                                ] = acc_8h
                                                            else:
                                                                l_ref[
                                                                    r_sl, :, t_sl_p
                                                                ] = (
                                                                    l_ref[
                                                                        r_sl, :, t_sl_p
                                                                    ]
                                                                    + l_8h
                                                                )
                                                                acc_ref[
                                                                    r_sl, :, t_sl_p
                                                                ] = (
                                                                    acc_ref[
                                                                        r_sl, :, t_sl_p
                                                                    ]
                                                                    + acc_8h
                                                                )

                                    return

                                @pl.loop(0, n_qgrps_v, unroll=False)
                                def _exec_od_qg_nog8(qg_i):
                                    qgrp_row = qg_i * prefix_q_grp_sz
                                    for kv_h in range(sq_kv_heads):
                                        h_ind = _load_topk_indices(
                                            kv_h, bkv_idx, qgrp_row
                                        )
                                        for k_i in range(prefix_topk_slices):
                                            _unpack_single_h_kv(kv_h, k_i, h_ind[k_i])

                                    @pl.loop(0, tq_per_qgrp, unroll=False)
                                    def _od_tq(loc_tq):
                                        tq_start = (
                                            qg_i * prefix_q_grp_sz + loc_tq * tq_sz
                                        )
                                        t_sl = pl.ds(
                                            pl.multiple_of(tq_start, tq_sz), tq_sz
                                        )

                                        @pl.when(tq_start < valid_bq_rows)
                                        def _run_od_tq():
                                            _run_ring_chunk(
                                                0,
                                                num_s,
                                                False,
                                                ("none",) * num_s,
                                                t_sl,
                                                static_first=static_first,
                                                topk_loader=lambda _kvh: static_h_ind,
                                            )

                                return

                            if use_mk128 and "gqa8" in prefill_opts and not use_fp8_qk_dyn:
                                n_tq_p = cdiv(valid_bq_rows, 256)

                                @pl.loop(0, n_tq_p, unroll=False)
                                def _exec_prefix_gqa8(i256):
                                    t_sl_p = pl.ds(
                                        pl.multiple_of(i256 * 256, 256), 256
                                    )
                                    for kv_h in range(sq_kv_heads):
                                        for half_i in range(2):
                                            q128_start = i256 * 256 + half_i * 128

                                            @pl.when(q128_start < valid_bq_rows)
                                            def _run_half_g8():
                                                h_ind = (
                                                    _load_topk_indices(
                                                        kv_h, bkv_idx, q128_start
                                                    )
                                                    if with_topk
                                                    else None
                                                )
                                                for sub_c in range(4):
                                                    r_idx = (
                                                        kv_h * 8
                                                        + half_i * 4
                                                        + sub_c
                                                    )
                                                    r_sl = slice(r_idx, r_idx + 1)
                                                    qt_8h = q_t_ref[
                                                        r_sl, :, t_sl_p
                                                    ]
                                                    acc_8h, l_8h = _eval_mk128_head(
                                                        kv_h,
                                                        qt_8h,
                                                        None,
                                                        0,
                                                        num_s,
                                                        ("none",) * num_s,
                                                        h_indices=h_ind,
                                                    )
                                                    if static_first:
                                                        l_ref[
                                                            r_sl, :, t_sl_p
                                                        ] = l_8h
                                                        acc_ref[
                                                            r_sl, :, t_sl_p
                                                        ] = acc_8h
                                                    elif with_topk:
                                                        l_ref[r_sl, :, t_sl_p] = (
                                                            l_ref[
                                                                r_sl, :, t_sl_p
                                                            ]
                                                            + l_8h
                                                        )
                                                        acc_ref[
                                                            r_sl, :, t_sl_p
                                                        ] = (
                                                            acc_ref[
                                                                r_sl, :, t_sl_p
                                                            ]
                                                            + acc_8h
                                                        )
                                                    else:
                                                        l_ref[r_sl, :, t_sl_p] = (
                                                            jnp.where(
                                                                is_first_bkv,
                                                                l_8h,
                                                                l_ref[
                                                                    r_sl,
                                                                    :,
                                                                    t_sl_p,
                                                                ]
                                                                + l_8h,
                                                            )
                                                        )
                                                        acc_ref[
                                                            r_sl, :, t_sl_p
                                                        ] = jnp.where(
                                                            is_first_bkv,
                                                            acc_8h,
                                                            acc_ref[
                                                                r_sl, :, t_sl_p
                                                            ]
                                                            + acc_8h,
                                                        )

                            else:

                                def _exec_prefix_tq(_tqi):
                                    tq_start = _tqi * tq_sz
                                    t_sl = pl.ds(tq_start, tq_sz)

                                    @pl.when(tq_start < valid_bq_rows)
                                    def _do_prefix_tq():
                                        _run_ring_chunk(
                                            0,
                                            num_s,
                                            False if with_topk else is_first_bkv,
                                            ("none",) * num_s,
                                            t_sl,
                                            static_first=static_first,
                                            topk_loader=(
                                                (
                                                    lambda kv_h: _load_topk_indices(
                                                        kv_h, bkv_idx, tq_start
                                                    )
                                                )
                                                if with_topk
                                                else None
                                            ),
                                        )

                                if use_tq_loop and (bq_sz // tq_sz) > 1:
                                    pl.loop(
                                        0, cdiv(valid_bq_rows, tq_sz), unroll=False
                                    )(_exec_prefix_tq)
                                else:
                                    for _tqi in range(bq_sz // tq_sz):
                                        _exec_prefix_tq(_tqi)

                        if use_prefix_topk_inner:

                            @pl.when(is_first_bkv)
                            def _exec_prefix_dense_sink():
                                _exec_prefix_pass(
                                    static_first=True,
                                    with_topk=(prefix_topk_sink_slices > 0),
                                )

                            @pl.when(jnp.logical_not(is_first_bkv))
                            def _exec_prefix_topk_sparse():
                                _exec_prefix_pass(static_first=False, with_topk=True)

                        else:
                            _exec_prefix_pass(static_first=False, with_topk=False)

                if (
                    use_mk128
                    and "gqa8" in prefill_opts
                    and not use_fp8_qk_dyn
                    and probe != "nocompute"
                ):

                    @pl.when(tail_bkv_start > start_bkv_idx)
                    def _unpack_prefix_gqa8():
                        n_tq_p = cdiv(valid_bq_rows, 256)

                        @pl.loop(0, n_tq_p, unroll=False)
                        def _up_acc_l_g8(i256):
                            t_sl_p = pl.ds(pl.multiple_of(i256 * 256, 256), 256)
                            sl_lo = pl.ds(pl.multiple_of(i256 * 256, 256), 128)
                            sl_hi = pl.ds(pl.multiple_of(i256 * 256 + 128, 128), 128)
                            has_hi = (i256 * 256 + 128) < valid_bq_rows
                            for kv_h in range(sq_kv_heads):
                                h_sl = slice(kv_h * 8, (kv_h + 1) * 8)
                                a_blk = acc_ref[h_sl, :, t_sl_p]
                                l_blk = l_ref[h_sl, :, t_sl_p]
                                acc_ref[h_sl, :, sl_lo] = jnp.concatenate(
                                    [
                                        jnp.swapaxes(
                                            a_blk[sc].reshape(64, 8, 32), 0, 1
                                        )
                                        for sc in range(4)
                                    ],
                                    axis=2,
                                )
                                l_ref[h_sl, :, sl_lo] = jnp.concatenate(
                                    [
                                        jnp.swapaxes(
                                            l_blk[sc].reshape(8, 8, 32), 0, 1
                                        )
                                        for sc in range(4)
                                    ],
                                    axis=2,
                                )

                                @pl.when(has_hi)
                                def _up_acc_l_hi():
                                    acc_ref[h_sl, :, sl_hi] = jnp.concatenate(
                                        [
                                            jnp.swapaxes(
                                                a_blk[4 + sc].reshape(64, 8, 32),
                                                0,
                                                1,
                                            )
                                            for sc in range(4)
                                        ],
                                        axis=2,
                                    )
                                    l_ref[h_sl, :, sl_hi] = jnp.concatenate(
                                        [
                                            jnp.swapaxes(
                                                l_blk[4 + sc].reshape(8, 8, 32),
                                                0,
                                                1,
                                            )
                                            for sc in range(4)
                                        ],
                                        axis=2,
                                    )

                        @pl.when(end_bkv_idx > tail_bkv_start)
                        def _up_q_tail_g8():

                            @pl.loop(0, n_tq_p, unroll=False)
                            def _up_q_g8_loop(i256):
                                t_sl_p = pl.ds(
                                    pl.multiple_of(i256 * 256, 256), 256
                                )
                                sl_lo = pl.ds(
                                    pl.multiple_of(i256 * 256, 256), 128
                                )
                                sl_hi = pl.ds(
                                    pl.multiple_of(i256 * 256 + 128, 128), 128
                                )
                                has_hi = (i256 * 256 + 128) < valid_bq_rows
                                for kv_h in range(sq_kv_heads):
                                    h_sl = slice(kv_h * 8, (kv_h + 1) * 8)
                                    q_blk = q_t_ref[h_sl, :, t_sl_p]
                                    q_t_ref[h_sl, :, sl_lo] = jnp.concatenate(
                                        [
                                            jnp.swapaxes(
                                                q_blk[sc].reshape(64, 8, 32),
                                                0,
                                                1,
                                            )
                                            for sc in range(4)
                                        ],
                                        axis=2,
                                    )

                                    @pl.when(has_hi)
                                    def _up_q_hi():
                                        q_t_ref[h_sl, :, sl_hi] = (
                                            jnp.concatenate(
                                                [
                                                    jnp.swapaxes(
                                                        q_blk[4 + sc].reshape(
                                                            64, 8, 32
                                                        ),
                                                        0,
                                                        1,
                                                    )
                                                    for sc in range(4)
                                                ],
                                                axis=2,
                                            )
                                        )

            @pl.loop(tail_bkv_start, end_bkv_idx, unroll=False)
            def _stream_bkv(bkv_idx):
                bkv_sem_idx = sem_ids_ref[1]
                (
                    next_seq_idx_kv,
                    _,
                    next_bkv_idx,
                    next_bkv_sem_idx,
                ) = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx
                )
                processed_kv_len = bkv_idx * bkv_sz

                @pl.when(next_seq_idx_kv < end_seq_idx)
                def _pf_stream_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx_kv, next_bkv_idx, next_bkv_sem_idx)

                @pl.when(bkv_idx == start_bkv_idx)
                def _wait_up_bq_stream():
                    _wait_and_unpack_cur_bq(bq_sem_idx)

                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)
                is_last_visit = bq_idx == num_bq - 1

                @pl.when(jnp.logical_and(update_sz > 0, is_last_visit))
                def _upd_stream_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                if probe != "nocompute":
                    effective_bkv_sz = jnp.clip(
                        effective_kv_len - processed_kv_len, 0, bkv_sz
                    )
                    _unpack_bkv_block(bkv_sem_idx, bkv_idx, 0, effective_bkv_sz)

                    def _exec_stream_tq(_tqi):
                        tq_start = _tqi * tq_sz
                        t_sl = pl.ds(tq_start, tq_sz)

                        @pl.when(tq_start < valid_bq_rows)
                        def _do_stream_tq():
                            q_first = processed_q_len + tq_start
                            strip_kv_hi = (
                                jnp.minimum(effective_kv_len, q_first + tq_sz)
                                if use_causal_mask
                                else effective_kv_len
                            )
                            loc_hi = cdiv(
                                jnp.clip(strip_kv_hi - processed_kv_len, 0, bkv_sz),
                                256,
                            )
                            max_unm_end = (
                                jnp.minimum(effective_kv_len, q_first + 1)
                                if use_causal_mask
                                else effective_kv_len
                            )
                            loc_unm_hi = jnp.clip(
                                jnp.maximum(max_unm_end - processed_kv_len, 0) // 256,
                                0,
                                loc_hi,
                            )
                            global_first_s = start_bkv_idx * num_s
                            if use_prefix_fast:
                                if (
                                    use_prefix_topk_inner
                                    and prefix_topk_tail_local > 0
                                    and not use_fp8_qk_dyn
                                ):
                                    q_grp_abs = (
                                        processed_q_len
                                        + (tq_start // prefix_q_grp_sz)
                                        * prefix_q_grp_sz
                                    )
                                    dense_unm_lo = jnp.clip(
                                        (
                                            q_grp_abs
                                            - prefix_topk_tail_local
                                            - processed_kv_len
                                        )
                                        // 256,
                                        0,
                                        loc_unm_hi,
                                    )
                                    do_tail_tk = jnp.logical_and(
                                        bkv_idx > start_bkv_idx,
                                        dense_unm_lo >= (2 * prefix_topk_slices),
                                    )

                                    @pl.when(do_tail_tk)
                                    def _do_sparse_tail_blk():
                                        _run_ring_chunk(
                                            0,
                                            num_s,
                                            False,
                                            ("none",) * num_s,
                                            t_sl,
                                            topk_loader=lambda kv_h: _load_topk_indices(
                                                kv_h, bkv_idx, tq_start
                                            ),
                                        )
                                        _run_slice_range(
                                            q_first,
                                            dense_unm_lo,
                                            loc_hi,
                                            dense_unm_lo,
                                            loc_unm_hi,
                                            global_first_s,
                                            processed_kv_len,
                                            effective_kv_len,
                                            t_sl,
                                        )

                                    @pl.when(jnp.logical_not(do_tail_tk))
                                    def _do_dense_tail_blk():
                                        _run_slice_range(
                                            q_first,
                                            jnp.int32(0),
                                            loc_hi,
                                            jnp.int32(0),
                                            loc_unm_hi,
                                            global_first_s,
                                            processed_kv_len,
                                            effective_kv_len,
                                            t_sl,
                                        )
                                else:
                                    _run_slice_range(
                                        q_first,
                                        jnp.int32(0),
                                        loc_hi,
                                        jnp.int32(0),
                                        loc_unm_hi,
                                        global_first_s,
                                        processed_kv_len,
                                        effective_kv_len,
                                        t_sl,
                                    )
                            else:
                                bkv_all_unm = loc_unm_hi == num_s

                                @pl.when(bkv_all_unm)
                                def _full_bkv_fast():
                                    _run_ring_chunk(
                                        0,
                                        num_s,
                                        (bkv_idx * num_s) == global_first_s,
                                        ("none",) * num_s,
                                        t_sl,
                                    )

                                @pl.when(jnp.logical_not(bkv_all_unm))
                                def _partial_bkv():
                                    _run_slice_range(
                                        q_first,
                                        jnp.int32(0),
                                        loc_hi,
                                        jnp.int32(0),
                                        loc_unm_hi,
                                        global_first_s,
                                        processed_kv_len,
                                        effective_kv_len,
                                        t_sl,
                                    )

                    if use_tq_loop and (bq_sz // tq_sz) > 1:
                        pl.loop(
                            0, cdiv(valid_bq_rows, tq_sz), unroll=False
                        )(_exec_stream_tq)
                    else:
                        for _tqi in range(bq_sz // tq_sz):
                            _exec_stream_tq(_tqi)

        if not use_staged_qo:
            wait_send_bo(bo_sem_idx)
        _emit_unfused_epilogue_and_send()


def _prefix_score_kernel(
    # Prefetch (6 scalar prefetches)
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    cu_qgrps_ref,
    distribution_ref,
    # Input
    q_hbm_ref,
    kv_cache_hbm_ref,
    # Output
    sc_hbm_out,  # [max_q_grps, max_bkv, 8, 128] f32
    # Scratch
    bq_stage_ref,  # [2, 1, 256, sq_pairs, 2, 128] bf16
    bkv_x2_ref,  # [2, phys_bkv_rows, bkv_stride, 4, 128] fp8
    sems,  # [3, 2] DMA semaphores (0=bkv, 1=bq, 2=sc)
    k_k1_ref,  # [sq_kv_heads, num_h_per_bkv, 128, 64] fp8
    q_k1_bf16_ref,  # [n_qgrps_per_bq, sq_q_heads, 64, w_q] bf16
    q_k1_f8_ref,  # [n_qgrps_per_bq, sq_q_heads, 64, w_q] fp8
    q_sc_k1_ref,  # [n_qgrps_per_bq, sq_kv_heads, 8, 128] f32
    sc_k1_ref,  # [16, 8, w_red] bf16
    sc_out_vmem,  # [2, n_qgrps_per_bq, 8, 128] f32
    *,
    bkv_sz: int,
    bq_sz: int,
    q_grp_sz: int,
    qstride: int,
    exp_scale: float,
    case: RpaCase,
    static_q_len: int | None = None,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    interpret: bool = False,
    topk_slices: int = 0,
    topk_in_kernel: bool = False,
    topk_gqa_batch: int = 1,
    topk_k_pair: int = 1,
    topk_sink_slices: int = 0,
    topk_tail_local: int = 0,
    stage2_bq_sz: int | None = None,
):
    """Stage-1 prefix All-QK scorer (read-only on Q and KV cache)."""
    start_seq_idx, end_seq_idx = case.get_range(distribution_ref)
    num_page_indices = page_indices_ref.shape[0]
    (
        _,
        phys_page_size,
        num_kv_heads_x2_per_kv_packing,
        _,
        _,
    ) = kv_cache_hbm_ref.shape
    if kv_pack_bd4:
        kv_pack_dense2 = False
    page_size = (
        phys_page_size * 4
        if kv_pack_bd4
        else (phys_page_size * 2 if kv_pack_dense2 else phys_page_size)
    )
    bkv_p = bkv_sz // page_size
    bkv_stride = bkv_x2_ref.shape[2]
    phys_bkv_rows = bkv_x2_ref.shape[1]
    kv_dtype = kv_cache_hbm_ref.dtype
    sq_kv_heads = k_k1_ref.shape[0]
    sq_pairs = 4 * sq_kv_heads
    num_h_per_bkv = k_k1_ref.shape[1]
    num_s = num_h_per_bkv // 2
    n_s_chunks = num_h_per_bkv // 8
    w_q = q_k1_bf16_ref.shape[3]
    w_red = sc_k1_ref.shape[2]
    eff_bq = bq_sz if static_q_len is None else min(bq_sz, static_q_len)
    s2_bq_sz = eff_bq if stage2_bq_sz is None else stage2_bq_sz
    n_qgrps_per_bq = eff_bq // q_grp_sz

    sc_k1_ref[...] = jnp.zeros(sc_k1_ref.shape, jnp.bfloat16)
    sc_out_vmem[...] = jnp.zeros(
        sc_out_vmem.shape, jnp.int32 if topk_in_kernel else jnp.float32
    )

    cache_hbm_shape = kv_cache_hbm_ref.shape
    flat_cache_hbm_ref = (
        None
        if interpret
        else kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )
    )

    def _async_copy(src, dst, sem, wait: bool):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _roll64(arr):
        return (
            jnp.roll(arr, 64, axis=1)
            if interpret
            else pltpu.roll(arr, shift=64, axis=1)
        )

    def _fetch_bkv_k1(seq_idx, bkv_idx, slot, *, wait: bool):
        sem = sems.at[0, slot]
        vmem_ref = bkv_x2_ref.at[slot, :, :num_kv_heads_x2_per_kv_packing]
        if not wait:
            start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
            page_indices_offset = start_kv_page_idx + bkv_idx * bkv_p
            for i in range(bkv_p):
                page_idx = jnp.minimum(
                    page_indices_offset + i, num_page_indices - 1
                )
                src_c = (
                    kv_cache_hbm_ref.at[page_indices_ref[page_idx]]
                    if interpret
                    else flat_cache_hbm_ref.at[
                        pl.ds(
                            page_indices_ref[page_idx] * phys_page_size,
                            phys_page_size,
                        )
                    ]
                )
                _async_copy(
                    src_c,
                    vmem_ref.at[pl.ds(i * phys_page_size, phys_page_size)],
                    sem,
                    False,
                )
        else:
            c_dst = vmem_ref.at[pl.ds(0, bkv_p * phys_page_size)]
            _async_copy(c_dst, c_dst, sem, True)

    def _unpack_k_only(slot):
        def _write_k_slice(dst_h, kv_u32):
            k_k1_ref[0, dst_h] = pltpu.bitcast(
                kv_u32[:, :64].astype(jnp.uint8), kv_dtype
            )
            if sq_kv_heads > 1:
                k_k1_ref[1, dst_h] = pltpu.bitcast(
                    kv_u32[:, 64:].astype(jnp.uint8), kv_dtype
                )

        def _bkv_slice_u32(kv_u32_all, row_start, n_rows):
            if interpret:
                u8 = lax.bitcast_convert_type(
                    bkv_x2_ref[slot, pl.ds(row_start, n_rows), 0, :, :],
                    jnp.uint8,
                ).astype(jnp.uint32)
                return (
                    u8[:, 0, :]
                    | (u8[:, 1, :] << jnp.uint32(8))
                    | (u8[:, 2, :] << jnp.uint32(16))
                    | (u8[:, 3, :] << jnp.uint32(24))
                )
            return kv_u32_all[pl.ds(row_start * bkv_stride, n_rows, bkv_stride)]

        if kv_pack_bd4:
            kv_u32_all = (
                None
                if interpret
                else bkv_x2_ref.bitcast(jnp.uint32)
                .at[slot]
                .reshape((bkv_sz // 4) * bkv_stride, 128)
            )

            @pl.loop(0, num_s, unroll=False)
            def _up_bd4(s_loc):
                pair_u32 = _bkv_slice_u32(kv_u32_all, s_loc * 64, 64)
                lo = pair_u32 & jnp.uint32(0xFFFF)
                hi = pair_u32 >> jnp.uint32(16)
                _write_k_slice(2 * s_loc, jnp.concatenate([lo, hi], axis=0))
                _write_k_slice(
                    2 * s_loc + 1,
                    jnp.concatenate([_roll64(lo), _roll64(hi)], axis=0),
                )

            return

        if kv_pack_dense2:
            kv_u32_all = (
                None
                if interpret
                else bkv_x2_ref.bitcast(jnp.uint32)
                .at[slot]
                .reshape((bkv_sz // 2) * bkv_stride, 128)
            )

            @pl.loop(0, num_s, unroll=False)
            def _up_d2(s_loc):
                pair_u32 = _bkv_slice_u32(kv_u32_all, s_loc * 128, 128)
                _write_k_slice(2 * s_loc, pair_u32 & jnp.uint32(0xFFFF))
                _write_k_slice(2 * s_loc + 1, pair_u32 >> jnp.uint32(16))

            return

        kv_u32_all = (
            None
            if interpret
            else bkv_x2_ref.bitcast(jnp.uint32)
            .at[slot]
            .reshape(bkv_sz * bkv_stride, 128)
        )

        @pl.loop(0, num_h_per_bkv, unroll=False)
        def _up_nod2(h_loc):
            _write_k_slice(h_loc, _bkv_slice_u32(kv_u32_all, h_loc * 128, 128))

    def _tile_amax_f32(x):
        m8 = jnp.max(jnp.abs(x).reshape(8, 8, w_q), axis=0)
        m1 = jnp.max(m8, axis=0, keepdims=True)
        if w_q == 512:
            m1 = jnp.maximum(m1[:, :256], m1[:, 256:])
            m1 = jnp.maximum(m1[:, :128], m1[:, 128:])
        elif w_q == 256:
            m1 = jnp.maximum(m1[:, :128], m1[:, 128:])
        return jnp.max(m1)

    lane_is_lo_128 = lax.broadcasted_iota(jnp.int32, (128, 128), 1) < 64
    lane_is_lo_256 = lax.broadcasted_iota(jnp.int32, (256, 128), 1) < 64
    r_128_io = lax.broadcasted_iota(jnp.int32, (128, 128), 0)
    r_256_io = lax.broadcasted_iota(jnp.int32, (256, 128), 0)
    q_h_iota = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    col_iota_8w = lax.broadcasted_iota(jnp.int32, (8, w_q), 1)
    col_iota_64w = lax.broadcasted_iota(jnp.int32, (64, w_q), 1)
    tk_iota_128 = lax.broadcasted_iota(jnp.int32, (8, 128), 1)
    tk_w = max(128, num_h_per_bkv)
    tk_iota_w = lax.broadcasted_iota(jnp.int32, (8, tk_w), 1)
    q_h_iota_w = lax.broadcasted_iota(jnp.int32, (8, tk_w), 0)
    tk_valid_mask = tk_iota_w < num_h_per_bkv
    bkv_start_k1 = 0 if topk_sink_slices > 0 else 1
    if topk_sink_slices > 0:
        sink_pin_vals = (
            jnp.float32(1e30)
            - tk_iota_w.astype(jnp.float32) * jnp.float32(1e25)
        )

    @pl.loop(start_seq_idx, end_seq_idx)
    def _k1_seq(seq_idx):
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start
        kv_len = kv_lens_ref[seq_idx]
        kv_q_gap = kv_len - q_len
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        @pl.loop(0, num_bq, unroll=False)
        def _k1_bq(bq_idx):
            processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
            effective_kv_len = jnp.minimum(kv_len, processed_q_len + actual_bq_sz)
            end_bkv_idx = cdiv(effective_kv_len, bkv_sz)
            valid_bq_rows = jnp.clip(q_len - bq_idx * actual_bq_sz, 0, actual_bq_sz)
            valid_qgrps = cdiv(valid_bq_rows, q_grp_sz)
            last_qgrp_start = (
                processed_q_len + jnp.maximum(valid_qgrps - 1, 0) * q_grp_sz
            )
            tail_bkv_start = (
                jnp.clip(
                    jnp.maximum(
                        last_qgrp_start // bkv_sz,
                        cdiv(jnp.maximum(last_qgrp_start - topk_tail_local, 0), bkv_sz),
                    ),
                    0,
                    end_bkv_idx,
                )
                if topk_tail_local > 0
                else jnp.clip(last_qgrp_start // bkv_sz, 0, end_bkv_idx)
            )

            @pl.when((tail_bkv_start > bkv_start_k1) & (valid_qgrps > 0))
            def _k1_run_bq():
                q_row_base = q_start + bq_idx * actual_bq_sz
                if qstride in (4, 8, 16):
                    stage_q_rows = 128 * qstride
                    n_stages = cdiv(valid_bq_rows, stage_q_rows)
                    stages_per_qgrp = max(1, q_grp_sz // stage_q_rows)
                    inv_qstride = jnp.float32(1.0 / float(qstride))

                    @pl.when(n_stages > 0)
                    def _k1_load_q_wide():
                        sz0 = jnp.minimum(stage_q_rows, valid_bq_rows)
                        _async_copy(
                            q_hbm_ref.at[:, pl.ds(q_row_base, sz0)],
                            bq_stage_ref.at[0, :, pl.ds(0, sz0)],
                            sems.at[1, 0],
                            False,
                        )

                        @pl.loop(0, n_stages, unroll=False)
                        def _k1_q_stage_wide(st_idx):
                            st_slot = st_idx & 1
                            st_q_start = st_idx * stage_q_rows
                            st_sz = jnp.minimum(
                                stage_q_rows, valid_bq_rows - st_q_start
                            )
                            next_st = st_idx + 1

                            @pl.when(next_st < n_stages)
                            def _k1_pf_q_wide():
                                nxt_slot = next_st & 1
                                nxt_q = next_st * stage_q_rows
                                nxt_sz = jnp.minimum(
                                    stage_q_rows, valid_bq_rows - nxt_q
                                )
                                _async_copy(
                                    q_hbm_ref.at[
                                        :, pl.ds(q_row_base + nxt_q, nxt_sz)
                                    ],
                                    bq_stage_ref.at[
                                        nxt_slot, :, pl.ds(0, nxt_sz)
                                    ],
                                    sems.at[1, nxt_slot],
                                    False,
                                )

                            _async_copy(
                                q_hbm_ref.at[
                                    :, pl.ds(q_row_base + st_q_start, st_sz)
                                ],
                                bq_stage_ref.at[st_slot, :, pl.ds(0, st_sz)],
                                sems.at[1, st_slot],
                                True,
                            )
                            g_idx = st_q_start // q_grp_sz
                            q_u32_stage = (
                                None
                                if interpret
                                else bq_stage_ref.bitcast(jnp.uint32)
                                .at[st_slot, 0]
                                .reshape(stage_q_rows * sq_pairs, 128)
                            )
                            sub_c0 = pl.multiple_of(
                                (st_q_start % q_grp_sz) // qstride, 128
                            )

                            def _read_u32_strided_q(off_q, s_p):
                                if interpret:
                                    u16 = lax.bitcast_convert_type(
                                        bq_stage_ref[
                                            st_slot,
                                            0,
                                            pl.ds(off_q, 128, qstride),
                                            s_p,
                                        ],
                                        jnp.uint16,
                                    ).astype(jnp.uint32)
                                    return u16[:, 0, :] | (
                                        u16[:, 1, :] << jnp.uint32(16)
                                    )
                                return q_u32_stage[
                                    pl.ds(
                                        off_q * sq_pairs + s_p,
                                        128,
                                        qstride * sq_pairs,
                                    )
                                ]

                            for s_p in range(sq_pairs):

                                def _dec_128_q(u_r, s_idx_p=s_p):
                                    ef = pltpu.bitcast(
                                        u_r << jnp.uint32(16), jnp.float32
                                    )
                                    of = pltpu.bitcast(
                                        u_r & jnp.uint32(0xFFFF0000),
                                        jnp.float32,
                                    )
                                    if s_idx_p < 4:
                                        return jnp.where(
                                            lane_is_lo_128, ef, _roll64(of)
                                        )
                                    return jnp.where(
                                        lane_is_lo_128, _roll64(ef), of
                                    )

                                acc_pq = jnp.zeros((128, 128), jnp.float32)
                                for off_q in range(qstride):
                                    ok_q = (qstride * r_128_io + off_q) < st_sz
                                    u_q = jnp.where(
                                        ok_q,
                                        _read_u32_strided_q(off_q, s_p),
                                        jnp.uint32(0),
                                    )
                                    acc_pq = acc_pq + _dec_128_q(u_q)
                                pooled_t = jnp.swapaxes(
                                    (inv_qstride * acc_pq).astype(jnp.bfloat16),
                                    0,
                                    1,
                                )
                                q_k1_bf16_ref[
                                    0, 2 * s_p, :, pl.ds(sub_c0, 128)
                                ] = pooled_t[:64, :]
                                q_k1_bf16_ref[
                                    0, 2 * s_p + 1, :, pl.ds(sub_c0, 128)
                                ] = pooled_t[64:, :]

                            quant_now = (
                                jnp.bool_(True)
                                if stages_per_qgrp == 1
                                else jnp.logical_or(
                                    (st_idx % stages_per_qgrp)
                                    == (stages_per_qgrp - 1),
                                    st_idx == (n_stages - 1),
                                )
                            )

                            @pl.when(quant_now)
                            def _quant_grp_now():
                                valid_g_rows = jnp.clip(
                                    valid_bq_rows - g_idx * q_grp_sz,
                                    0,
                                    q_grp_sz,
                                )
                                valid_cols = cdiv(valid_g_rows, qstride)
                                col_ok_64w = col_iota_64w < valid_cols
                                for kv_h in range(sq_kv_heads):

                                    def _quant_qh4(q_h, sc_acc):
                                        qf = jnp.where(
                                            col_ok_64w,
                                            q_k1_bf16_ref[
                                                0, kv_h * 8 + q_h
                                            ].astype(jnp.float32),
                                            jnp.float32(0.0),
                                        )
                                        amax = _tile_amax_f32(qf)
                                        q_inv = jnp.where(
                                            amax == 0.0,
                                            jnp.float32(0.0),
                                            jnp.float32(448.0)
                                            / jnp.maximum(
                                                amax, jnp.float32(1e-12)
                                            ),
                                        )
                                        q_k1_f8_ref[g_idx, kv_h * 8 + q_h] = (
                                            jnp.clip(
                                                qf * q_inv, -448.0, 448.0
                                            ).astype(jnp.float8_e4m3fn)
                                        )
                                        return jnp.where(
                                            q_h_iota == q_h,
                                            amax * jnp.float32(exp_scale / 448.0),
                                            sc_acc,
                                        )

                                    q_sc_k1_ref[g_idx, kv_h] = lax.fori_loop(
                                        0,
                                        8,
                                        _quant_qh4,
                                        jnp.zeros((8, 128), jnp.float32),
                                        unroll=False,
                                    )
                else:
                    n_stages = cdiv(valid_bq_rows, 256)

                    @pl.when(n_stages > 0)
                    def _k1_load_q():
                        sz0 = jnp.minimum(256, valid_bq_rows)
                        _async_copy(
                            q_hbm_ref.at[:, pl.ds(q_row_base, sz0)],
                            bq_stage_ref.at[0, :, pl.ds(0, sz0)],
                            sems.at[1, 0],
                            False,
                        )

                        @pl.loop(0, n_stages, unroll=False)
                        def _k1_q_stage(st_idx):
                            st_slot = st_idx & 1
                            st_q_start = st_idx * 256
                            st_sz = jnp.minimum(256, valid_bq_rows - st_q_start)
                            next_st = st_idx + 1

                            @pl.when(next_st < n_stages)
                            def _k1_pf_q():
                                nxt_slot = next_st & 1
                                nxt_q = next_st * 256
                                nxt_sz = jnp.minimum(256, valid_bq_rows - nxt_q)
                                _async_copy(
                                    q_hbm_ref.at[:, pl.ds(q_row_base + nxt_q, nxt_sz)],
                                    bq_stage_ref.at[nxt_slot, :, pl.ds(0, nxt_sz)],
                                    sems.at[1, nxt_slot],
                                    False,
                                )

                            _async_copy(
                                q_hbm_ref.at[:, pl.ds(q_row_base + st_q_start, st_sz)],
                                bq_stage_ref.at[st_slot, :, pl.ds(0, st_sz)],
                                sems.at[1, st_slot],
                                True,
                            )
                            g_idx = st_q_start // q_grp_sz
                            q_u32_stage = (
                                None
                                if interpret
                                else bq_stage_ref.bitcast(jnp.uint32)
                                .at[st_slot, 0]
                                .reshape(256 * sq_pairs, 128)
                            )
                            if qstride == 2:
                                sub_c0 = pl.multiple_of((st_q_start % q_grp_sz) // 2, 128)
                                ev_ok = (2 * r_128_io) < st_sz
                                od_ok = (2 * r_128_io + 1) < st_sz
                                for s_p in range(sq_pairs):
                                    if interpret:
                                        u16_e = lax.bitcast_convert_type(
                                            bq_stage_ref[st_slot, 0, pl.ds(0, 128, 2), s_p],
                                            jnp.uint16,
                                        ).astype(jnp.uint32)
                                        u_ev = u16_e[:, 0, :] | (u16_e[:, 1, :] << jnp.uint32(16))
                                        u16_o = lax.bitcast_convert_type(
                                            bq_stage_ref[st_slot, 0, pl.ds(1, 128, 2), s_p],
                                            jnp.uint16,
                                        ).astype(jnp.uint32)
                                        u_od = u16_o[:, 0, :] | (u16_o[:, 1, :] << jnp.uint32(16))
                                    else:
                                        u_ev = q_u32_stage[pl.ds(s_p, 128, 2 * sq_pairs)]
                                        u_od = q_u32_stage[
                                            pl.ds(sq_pairs + s_p, 128, 2 * sq_pairs)
                                        ]
                                    u_ev = jnp.where(ev_ok, u_ev, jnp.uint32(0))
                                    u_od = jnp.where(od_ok, u_od, jnp.uint32(0))

                                    def _dec_128(u_r, s_idx_p=s_p):
                                        ef = pltpu.bitcast(u_r << jnp.uint32(16), jnp.float32)
                                        of = pltpu.bitcast(
                                            u_r & jnp.uint32(0xFFFF0000), jnp.float32
                                        )
                                        if s_idx_p < 4:
                                            return jnp.where(
                                                lane_is_lo_128, ef, _roll64(of)
                                            )
                                        return jnp.where(lane_is_lo_128, _roll64(ef), of)

                                    pooled_t = jnp.swapaxes(
                                        (0.5 * (_dec_128(u_ev) + _dec_128(u_od))).astype(
                                            jnp.bfloat16
                                        ),
                                        0,
                                        1,
                                    )
                                    q_k1_bf16_ref[
                                        g_idx, 2 * s_p, :, pl.ds(sub_c0, 128)
                                    ] = pooled_t[:64, :]
                                    q_k1_bf16_ref[
                                        g_idx, 2 * s_p + 1, :, pl.ds(sub_c0, 128)
                                    ] = pooled_t[64:, :]
                            else:
                                sub_c0 = pl.multiple_of(st_q_start % q_grp_sz, 256)
                                r_ok = r_256_io < st_sz
                                for s_p in range(sq_pairs):
                                    if interpret:
                                        u16_f = lax.bitcast_convert_type(
                                            bq_stage_ref[st_slot, 0, :, s_p], jnp.uint16
                                        ).astype(jnp.uint32)
                                        u_f = u16_f[:, 0, :] | (u16_f[:, 1, :] << jnp.uint32(16))
                                    else:
                                        u_f = q_u32_stage[pl.ds(s_p, 256, sq_pairs)]
                                    u_f = jnp.where(r_ok, u_f, jnp.uint32(0))
                                    ef = pltpu.bitcast(
                                        u_f << jnp.uint32(16), jnp.float32
                                    ).astype(jnp.bfloat16)
                                    of = pltpu.bitcast(
                                        u_f & jnp.uint32(0xFFFF0000), jnp.float32
                                    ).astype(jnp.bfloat16)
                                    if s_p < 4:
                                        p_bf = jnp.where(lane_is_lo_256, ef, _roll64(of))
                                    else:
                                        p_bf = jnp.where(lane_is_lo_256, _roll64(ef), of)
                                    p_t = jnp.swapaxes(p_bf, 0, 1)
                                    q_k1_bf16_ref[
                                        g_idx, 2 * s_p, :, pl.ds(sub_c0, 256)
                                    ] = p_t[:64, :]
                                    q_k1_bf16_ref[
                                        g_idx, 2 * s_p + 1, :, pl.ds(sub_c0, 256)
                                    ] = p_t[64:, :]

                        @pl.loop(0, valid_qgrps, unroll=False)
                        def _k1_quant_grp(g_idx):
                            valid_g_rows = jnp.clip(
                                valid_bq_rows - g_idx * q_grp_sz, 0, q_grp_sz
                            )
                            valid_cols = cdiv(valid_g_rows, qstride)
                            col_ok_64w = col_iota_64w < valid_cols
                            for kv_h in range(sq_kv_heads):

                                def _quant_qh(q_h, sc_acc):
                                    qf = q_k1_bf16_ref[g_idx, kv_h * 8 + q_h].astype(
                                        jnp.float32
                                    )
                                    qf = jnp.where(col_ok_64w, qf, jnp.float32(0.0))
                                    amax = _tile_amax_f32(qf)
                                    q_inv = jnp.where(
                                        amax == 0.0,
                                        jnp.float32(0.0),
                                        jnp.float32(448.0)
                                        / jnp.maximum(amax, jnp.float32(1e-12)),
                                    )
                                    q_k1_f8_ref[g_idx, kv_h * 8 + q_h] = jnp.clip(
                                        qf * q_inv, -448.0, 448.0
                                    ).astype(jnp.float8_e4m3fn)
                                    h_sc = amax * jnp.float32(exp_scale / 448.0)
                                    return jnp.where(q_h_iota == q_h, h_sc, sc_acc)

                                q_sc_k1_ref[g_idx, kv_h] = lax.fori_loop(
                                    0,
                                    8,
                                    _quant_qh,
                                    jnp.zeros((8, 128), jnp.float32),
                                    unroll=False,
                                )

                def _sc_dma(target_bkv, sc_slot, *, wait: bool):
                    cp_sem = sems.at[2, sc_slot]

                    @pl.loop(0, valid_qgrps, unroll=False)
                    def _sc_g(g_idx):
                        flat_g = (
                            cu_qgrps_ref[seq_idx] + bq_idx * n_qgrps_per_bq + g_idx
                        )
                        _async_copy(
                            sc_out_vmem.at[sc_slot, g_idx],
                            sc_hbm_out.at[flat_g, target_bkv],
                            cp_sem,
                            wait,
                        )

                _fetch_bkv_k1(seq_idx, bkv_start_k1, 0, wait=False)

                @pl.loop(bkv_start_k1, tail_bkv_start, unroll=False)
                def _k1_bkv(bkv_idx):
                    bkv_slot = (bkv_idx - bkv_start_k1) & 1
                    nxt_bkv = bkv_idx + 1

                    @pl.when(nxt_bkv < tail_bkv_start)
                    def _pf_nxt_bkv():
                        _fetch_bkv_k1(seq_idx, nxt_bkv, 1 - bkv_slot, wait=False)

                    _fetch_bkv_k1(seq_idx, bkv_idx, bkv_slot, wait=True)
                    _unpack_k_only(bkv_slot)

                    sc_slot = (bkv_idx - bkv_start_k1) & 1

                    @pl.when(bkv_idx >= bkv_start_k1 + 2)
                    def _wait_prev_sc():
                        _sc_dma(bkv_idx - 2, sc_slot, wait=True)

                    neg_col = jnp.bfloat16(-1e4)

                    prefix_min_qgrp = jnp.maximum(
                        cdiv((bkv_idx + 1) * bkv_sz - processed_q_len, q_grp_sz),
                        0,
                    )
                    min_qgrp = (
                        jnp.minimum(
                            prefix_min_qgrp,
                            jnp.maximum(
                                cdiv(
                                    bkv_idx * bkv_sz
                                    + topk_tail_local
                                    + 256
                                    - processed_q_len,
                                    q_grp_sz,
                                ),
                                0,
                            ),
                        )
                        if topk_tail_local > 0
                        else prefix_min_qgrp
                    )

                    @pl.loop(min_qgrp, valid_qgrps, unroll=False)
                    def _score_grp(g_idx):
                        valid_g_rows = jnp.clip(
                            valid_bq_rows - g_idx * q_grp_sz, 0, q_grp_sz
                        )
                        valid_cols = cdiv(valid_g_rows, qstride)
                        col_ok_8w = col_iota_8w < valid_cols
                        sc_pair_f32 = jnp.full(
                            (8, tk_w), jnp.float32(-1e30), dtype=jnp.float32
                        )
                        if topk_gqa_batch > 1 and (8 % topk_gqa_batch == 0):
                            col_ok_1w = lax.broadcasted_iota(jnp.int32, (1, w_q), 1) < valid_cols
                            for kv_h in range(sq_kv_heads):
                                head_chunks = [[] for _ in range(8)]
                                q_batches = [
                                    jnp.concatenate(
                                        [
                                            q_k1_f8_ref[g_idx, kv_h * 8 + qh]
                                            for qh in range(b0, b0 + topk_gqa_batch)
                                        ],
                                        axis=1,
                                    )[None, :, :]
                                    for b0 in range(0, 8, topk_gqa_batch)
                                ]
                                k_flat_h = (
                                    None
                                    if interpret
                                    else k_k1_ref.at[kv_h].reshape(
                                        num_h_per_bkv * 128, 64
                                    )
                                )
                                for g_c in range(n_s_chunks):
                                    m_sub = [[] for _ in range(8)]
                                    if topk_k_pair in (2, 4, 8):
                                        for sub_p in range(8 // topk_k_pair):
                                            if interpret:
                                                k_mh = jnp.concatenate(
                                                    [
                                                        k_k1_ref[
                                                            kv_h,
                                                            g_c * 8
                                                            + topk_k_pair * sub_p
                                                            + kk,
                                                        ]
                                                        for kk in range(topk_k_pair)
                                                    ],
                                                    axis=0,
                                                )[None, :, :]
                                            else:
                                                k_mh = k_flat_h[
                                                    pl.ds(
                                                        (
                                                            g_c * 8
                                                            + topk_k_pair * sub_p
                                                        )
                                                        * 128,
                                                        topk_k_pair * 128,
                                                    )
                                                ][None, :, :]
                                            for bi, q_blk in enumerate(q_batches):
                                                qk = lax.dot(
                                                    k_mh,
                                                    q_blk,
                                                    dimension_numbers=(
                                                        ([2], [1]),
                                                        ([0], [0]),
                                                    ),
                                                    preferred_element_type=jnp.float32,
                                                ).astype(jnp.bfloat16)
                                                m_kq = jnp.max(
                                                    jnp.max(
                                                        qk.reshape(
                                                            (
                                                                topk_k_pair,
                                                                16,
                                                                8,
                                                                topk_gqa_batch * w_q,
                                                            )
                                                        ),
                                                        axis=1,
                                                    ),
                                                    axis=1,
                                                )
                                                for hi in range(topk_gqa_batch):
                                                    mh = m_kq[
                                                        :, hi * w_q : (hi + 1) * w_q
                                                    ]
                                                    for kk in range(topk_k_pair):
                                                        m_sub[
                                                            bi * topk_gqa_batch + hi
                                                        ].append(mh[kk : kk + 1])
                                    else:
                                        for sub_s in range(8):
                                            k_1h = k_k1_ref[kv_h, g_c * 8 + sub_s][
                                                None, :, :
                                            ]
                                            for bi, q_blk in enumerate(q_batches):
                                                qk = lax.dot(
                                                    k_1h,
                                                    q_blk,
                                                    dimension_numbers=(
                                                        ([2], [1]),
                                                        ([0], [0]),
                                                    ),
                                                    preferred_element_type=jnp.float32,
                                                ).astype(jnp.bfloat16)
                                                m_q = jnp.max(
                                                    jnp.max(
                                                        qk.reshape(
                                                            (1, 16, 8, topk_gqa_batch * w_q)
                                                        ),
                                                        axis=1,
                                                    ),
                                                    axis=1,
                                                )
                                                for hi in range(topk_gqa_batch):
                                                    mh = m_q[
                                                        :, hi * w_q : (hi + 1) * w_q
                                                    ]
                                                    m_sub[bi * topk_gqa_batch + hi].append(
                                                        jnp.where(col_ok_1w, mh, neg_col)
                                                    )
                                    for qh in range(8):
                                        m_8w = jnp.where(
                                            col_ok_1w,
                                            jnp.concatenate(m_sub[qh], axis=0),
                                            neg_col,
                                        )
                                        if w_q == 512:
                                            m_8w = jnp.maximum(
                                                m_8w[:, :256], m_8w[:, 256:]
                                            )
                                        head_chunks[qh].append(
                                            jnp.max(
                                                m_8w.astype(jnp.float32), axis=1
                                            )
                                        )
                                row_vecs = [
                                    jnp.concatenate(head_chunks[qh], axis=0)
                                    for qh in range(8)
                                ]
                                if num_h_per_bkv < tk_w:
                                    pad_tail = jnp.zeros(
                                        (tk_w - num_h_per_bkv,), jnp.float32
                                    )
                                    row_vecs = [
                                        jnp.concatenate([rv, pad_tail], axis=0)
                                        for rv in row_vecs
                                    ]
                                m_8 = jnp.stack(row_vecs, axis=0)
                                m_8 = m_8 * q_sc_k1_ref[g_idx, kv_h][:, :1]
                                mean_8 = jnp.sum(
                                    m_8, axis=1, keepdims=True
                                ) * jnp.float32(1.0 / float(num_h_per_bkv))
                                s_vec = jnp.max(m_8 - mean_8, axis=0)
                                sc_pair_f32 = jnp.where(
                                    q_h_iota_w == kv_h, s_vec[None, :], sc_pair_f32
                                )
                        else:
                            for kv_h in range(sq_kv_heads):

                                def _qh_score(q_h, m_acc):
                                    q_1h = q_k1_f8_ref[g_idx, kv_h * 8 + q_h][
                                        None, :, :
                                    ]
                                    for g_c in range(n_s_chunks):
                                        m_parts = []
                                        for sub_s in range(8):
                                            k_1h = k_k1_ref[kv_h, g_c * 8 + sub_s][
                                                None, :, :
                                            ]
                                            qk = lax.dot(
                                                k_1h,
                                                q_1h,
                                                dimension_numbers=(
                                                    ([2], [1]),
                                                    ([0], [0]),
                                                ),
                                                preferred_element_type=jnp.float32,
                                            ).astype(jnp.bfloat16)
                                            m_q = jnp.max(
                                                jnp.max(
                                                    qk.reshape((1, 16, 8, w_q)), axis=1
                                                ),
                                                axis=1,
                                            )
                                            m_parts.append(m_q)
                                        m_8w = jnp.where(
                                            col_ok_8w,
                                            jnp.concatenate(m_parts, axis=0),
                                            neg_col,
                                        )
                                        if w_q == 512:
                                            m_8w = jnp.maximum(
                                                m_8w[:, :256], m_8w[:, 256:]
                                            )
                                        sc_k1_ref[g_c] = m_8w
                                    sc_128 = jnp.max(
                                        sc_k1_ref[...]
                                        .reshape((128, w_red))
                                        .astype(jnp.float32),
                                        axis=1,
                                    )
                                    return jnp.where(
                                        q_h_iota == q_h, sc_128[None, :], m_acc
                                    )

                                m_8 = lax.fori_loop(
                                    0,
                                    8,
                                    _qh_score,
                                    jnp.zeros((8, 128), jnp.float32),
                                    unroll=False,
                                )
                                m_8 = m_8 * q_sc_k1_ref[g_idx, kv_h]
                                mean_8 = jnp.sum(
                                    m_8, axis=1, keepdims=True
                                ) * jnp.float32(1.0 / float(num_h_per_bkv))
                                s_vec = jnp.max(m_8 - mean_8, axis=0)
                                sc_pair_f32 = jnp.where(
                                    q_h_iota == kv_h, s_vec[None, :], sc_pair_f32
                                )
                        if topk_tail_local > 0:
                            q_off_s2 = bq_idx * actual_bq_sz + g_idx * q_grp_sz
                            q_grp_abs = kv_q_gap + q_off_s2
                            processed_q_len_s2 = (
                                kv_q_gap + (q_off_s2 // s2_bq_sz) * s2_bq_sz
                            )
                            tail_bkv_start_s2 = processed_q_len_s2 // bkv_sz
                            is_tail_blk = bkv_idx >= tail_bkv_start_s2
                            dense_unm_lo = jnp.clip(
                                (q_grp_abs - topk_tail_local - bkv_idx * bkv_sz)
                                // 256,
                                0,
                                num_s,
                            )
                            sc_pair_f32 = jnp.where(
                                jnp.logical_or(
                                    jnp.logical_not(is_tail_blk),
                                    tk_iota_w < (2 * dense_unm_lo),
                                ),
                                sc_pair_f32,
                                jnp.float32(-1e30),
                            )
                        if topk_sink_slices > 0:
                            sc_pair_f32 = jnp.where(
                                jnp.logical_and(
                                    bkv_idx == 0, tk_iota_w < topk_sink_slices
                                ),
                                sink_pin_vals,
                                sc_pair_f32,
                            )
                        if topk_in_kernel:
                            x_tk = jnp.where(
                                tk_valid_mask, sc_pair_f32, jnp.float32(-1e30)
                            )
                            x_i32 = lax.bitcast_convert_type(x_tk, jnp.int32)
                            s_ord = jnp.where(
                                x_i32 >= 0, x_i32, x_i32 ^ jnp.int32(0x7FFFFFFF)
                            )
                            cur_k = (s_ord & jnp.int32(-256)) | (
                                jnp.int32(255) - tk_iota_w
                            )
                            min_k = jnp.int32(-2147483648)
                            out_tk = jnp.zeros((8, 128), jnp.int32)
                            n_tk_w = topk_slices // 4

                            def _tk_word_1pass(w_idx, state):
                                ck, co = state
                                word = jnp.zeros((8,), jnp.uint32)
                                for b_i in range(4):
                                    best = jnp.max(ck, axis=1)
                                    idx = jnp.int32(255) - (best & jnp.int32(0xFF))
                                    ck = jnp.where(
                                        tk_iota_w == idx[:, None], min_k, ck
                                    )
                                    word = word | (
                                        idx.astype(jnp.uint32) << jnp.uint32(8 * b_i)
                                    )
                                co = jnp.where(
                                    tk_iota_128 == w_idx,
                                    word[:, None].astype(jnp.int32),
                                    co,
                                )
                                return ck, co

                            if n_tk_w <= 8:
                                for w_idx in range(n_tk_w):
                                    cur_k, out_tk = _tk_word_1pass(
                                        w_idx, (cur_k, out_tk)
                                    )
                            else:
                                _, out_tk = lax.fori_loop(
                                    0,
                                    n_tk_w,
                                    _tk_word_1pass,
                                    (cur_k, out_tk),
                                    unroll=False,
                                )
                            sc_out_vmem[sc_slot, g_idx] = out_tk
                        else:
                            sc_out_vmem[sc_slot, g_idx] = sc_pair_f32

                    _sc_dma(bkv_idx, sc_slot, wait=False)

                _sc_dma(
                    tail_bkv_start - 1,
                    (tail_bkv_start - 1 - bkv_start_k1) & 1,
                    wait=True,
                )

                @pl.when(tail_bkv_start > bkv_start_k1 + 1)
                def _wait_sc_penult():
                    _sc_dma(
                        tail_bkv_start - 2,
                        (tail_bkv_start - 2 - bkv_start_k1) & 1,
                        wait=True,
                    )


def _ragged_paged_attention_kernel(*args, **kwargs):
    # distribution_ref is at index 5 (after kv_lens, page_indices, cu_q_lens,
    # cu_kv_lens, cu_seq_mask_lens).
    extra_sp_count = kwargs.pop("extra_sp_count", 0)
    cu_qgrps_ref = args[9] if extra_sp_count == 2 else None
    prefix_topk_ref = args[10] if extra_sp_count == 2 else None
    loop_args = args[:9] + args[9 + extra_sp_count :]

    distribution_ref = loop_args[5]
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(distribution_ref)

    @pl.loop(start_seq_idx, end_seq_idx)
    def _(seq_idx):
        return _ragged_paged_attention_kernel_loop(
            seq_idx,
            *loop_args,
            cu_qgrps_ref=cu_qgrps_ref,
            prefix_topk_ref=prefix_topk_ref,
            **kwargs,
        )

    if (
        "a3_out_layout" in kwargs.get("prefill_opts", ())
        and len(loop_args) > 30
        and loop_args[30] is not None
        and not kwargs.get("debug_mode", False)
    ):
        cu_q_lens_ref = loop_args[2]
        o_hbm_ref = loop_args[15]
        bo_x2_ref = loop_args[20]
        sems = loop_args[21]
        bo_carry_ref = loop_args[30]
        max_tok = o_hbm_ref.shape[0]
        bq_sz_ep = bo_x2_ref.shape[1] - 16
        total_q = jnp.where(
            end_seq_idx > start_seq_idx,
            cu_q_lens_ref[end_seq_idx],
            jnp.int32(0),
        )
        aligned_end = pl.multiple_of((total_q // 16) * 16, 16)
        rem_end = total_q - aligned_end

        @pl.when(rem_end > 0)
        def _flush_carry_tail():
            cp = pltpu.make_async_copy(
                bo_carry_ref.at[pl.ds(0, 16), :],
                o_hbm_ref.at[pl.ds(aligned_end, 16), :],
                sems.at[2, 0],
            )
            cp.start()
            cp.wait()

        zero_start = pl.multiple_of(cdiv(total_q, 16) * 16, 16)

        @pl.when(zero_start < max_tok)
        def _zero_pad_rows():
            bo_x2_ref[0, :bq_sz_ep, :] = jnp.zeros(
                (bq_sz_ep, o_hbm_ref.shape[1]), o_hbm_ref.dtype
            )
            rem_total = max_tok - zero_start
            n_full_z = rem_total // bq_sz_ep

            @pl.loop(0, n_full_z, unroll=False)
            def _z_blk(i):
                st_blk = pl.multiple_of(zero_start + i * bq_sz_ep, 16)
                cp = pltpu.make_async_copy(
                    bo_x2_ref.at[0, pl.ds(0, bq_sz_ep), :],
                    o_hbm_ref.at[pl.ds(st_blk, bq_sz_ep), :],
                    sems.at[2, 0],
                )
                cp.start()
                cp.wait()

            rem_z = rem_total - n_full_z * bq_sz_ep

            @pl.when(rem_z > 0)
            def _z_rem():
                st = pl.multiple_of(zero_start + n_full_z * bq_sz_ep, 16)
                rem_z_al = pl.multiple_of((rem_z // 16) * 16, 16)
                cp = pltpu.make_async_copy(
                    bo_x2_ref.at[0, pl.ds(0, rem_z_al), :],
                    o_hbm_ref.at[pl.ds(st, rem_z_al), :],
                    sems.at[2, 0],
                )
                cp.start()
                cp.wait()


def _ragged_paged_attention_kernel_loop(
    seq_idx,
    # Prefetch (9 scalar prefetches)
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [flat page indices]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    cu_kv_lens_ref,  # [max_num_seqs + 1]
    cu_seq_mask_lens,  # [max_num_seqs + 1]
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4]
    bkv_update_ids_ref,  # [6]
    # Input
    q_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_hbm_ref,  # [max_num_tokens, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    custom_mask_ref,  # [flatten_total_kv_len, head_dim] or None
    zero_mask_ref,  # [bkv_sz, head_dim] or None
    attention_sink_ref,  # [actual_num_kv_heads, num_q_heads_per_kv_head, 128] or None
    # Output
    o_hbm_ref,  # same shape as q_hbm_ref
    updated_kv_cache_hbm_ref,  # same shape as kv_cache_hbm_ref
    # Scratch
    bkvmask_ref,  # [2, bq_sz, bkv_sz, head_dim] or None
    bkv_x2_ref,  # [2, bkv_sz, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, ...]
    sems,  # [5, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128]
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128]
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim]
    q_t_ref=None,  # [16, 64, bq_sz] bf16 when stationary_q is active
    k_sub_ref=None,  # [2, bkv_sz // 256, 256, 128] bf16 when stationary_q is active
    vt_sub_ref=None,  # [2, bkv_sz // 256, 64, 256] bf16 when stationary_q is active
    mask_lt_ref=None,  # [1, 256, 256] bf16 causal diagonal bias for stationary_q v2
    mask_ut_ref=None,  # [1, 256, 256] bf16 SW-2049 left-edge bias for stationary_q v2
    bo_carry_ref=None,  # [16, 1024] bf16 when a3_out_layout is active
    q_sc_ref=None,  # [16, 8, bq_sz] f32 per-Q-token scale when fp8_qk is active
    bnew_kv_ref=None,  # [num_slots, bkv_sz, 1, 4, 128] fp8 staging when kv_pack_dense2 is active
    *,
    cu_qgrps_ref=None,
    prefix_topk_ref=None,
    prefix_topk_slices: int = 0,
    prefix_topk_sink_slices: int = 0,
    prefix_q_grp_sz: int = 1024,
    prefix_max_bkv: int = 16,
    prefix_topk_tail_local: int = 0,
    causal: bool = True,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    static_q_len: int | None = None,
    bq_sz,  # bq fetch size
    bkv_sz,  # bkv prefetch size
    bq_csz,  # bq compute size
    bkv_csz,  # bkv compute size
    case: RpaCase = RpaCase.MIXED,
    skip_kv_mask: bool = False,
    sliding_window_chunk_skip: bool = True,
    window_aligned_kv: bool = False,
    zero_kv_tail: bool = False,
    ring: int = 0,
    probe: str | None = None,
    prefill_opts: frozenset = frozenset(),
    debug_mode: bool = False,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    interpret: bool = False,
    valid_lane_dim: int | None = None,
):
    use_a3_out = bo_carry_ref is not None and "a3_out_layout" in prefill_opts
    if use_a3_out:
        assert o_hbm_ref.shape == (q_hbm_ref.shape[1], 128 * q_hbm_ref.shape[2])
    else:
        assert q_hbm_ref.shape == o_hbm_ref.shape
    assert q_hbm_ref.shape[-1] == kv_cache_hbm_ref.shape[-1]

    use_causal_mask = causal
    if case == RpaCase.DECODE and static_q_len == 1:
        # A single query row sees every cached key; wider DECODE rows
        # (speculative verification) keep the causal mask.
        use_causal_mask = False

    out_dtype = q_hbm_ref.dtype if q_t_ref is not None else acc_ref.dtype
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    (
        total_num_pages,
        phys_page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_hbm_ref.shape
    if kv_pack_bd4:
        kv_pack_dense2 = False
    page_size = (
        phys_page_size * 4
        if kv_pack_bd4
        else (phys_page_size * 2 if kv_pack_dense2 else phys_page_size)
    )
    bkv_stride = bkv_x2_ref.shape[2]
    num_page_indices = page_indices_ref.shape[0]
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert head_dim % 128 == 0
    assert bkv_sz % page_size == 0
    assert bkv_sz % bkv_csz == 0, f"bkv_sz={bkv_sz} not divisible by bkv_csz={bkv_csz}"
    bkv_p = bkv_sz // page_size
    start_seq_idx, end_seq_idx = case.get_range(distribution_ref)
    # ring > 0: DECODE with one query row and one KV block per sequence keeps
    # ring - 1 sequences' q and KV fetches in flight, one buffer slot per
    # sequence; the two-slot pipeline fetches only one sequence ahead.
    num_slots = (
        3 if (case != RpaCase.DECODE and "mixed_s3" in prefill_opts) else (ring or 2)
    )
    assert (
        not ring
        or (case == RpaCase.DECODE and static_q_len == 1)
        or "mixed_s3" in prefill_opts
    )
    assert bkv_x2_ref.shape[0] == num_slots

    def ring_slot(seq):
        return (seq - start_seq_idx) % num_slots

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]
    kv_q_gap = kv_len - q_len
    cur_seq_start_bkv_idx = 0
    next_seq_start_bkv_idx = 0
    # window_aligned_kv: a sequence's KV blocks start at the page holding the
    # first key its first query row sees, not at token 0, so a window-sized
    # block covers the window (block-aligned blocks straddle it and fetch and
    # compute up to twice the keys). Block indices are then relative to that
    # base, and every block index starts at 0.
    assert not window_aligned_kv or (
        sliding_window is not None and case == RpaCase.DECODE
    )

    def kv_base(seq_idx):
        if not window_aligned_kv:
            return 0
        q_len = cu_q_lens_ref[seq_idx + 1] - cu_q_lens_ref[seq_idx]
        first_key = jnp.maximum(kv_lens_ref[seq_idx] - q_len - sliding_window + 1, 0)
        return first_key // page_size * page_size

    _sw_off = (
        (sliding_window - 1)
        if (sliding_window is not None and q_t_ref is not None)
        else sliding_window
    )
    if sliding_window is not None and not window_aligned_kv:
        cur_seq_start_bkv_idx = jnp.maximum(kv_q_gap - _sw_off, 0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, end_seq_idx - 1)
        next_q_start = cu_q_lens_ref[next_seq_idx]
        next_q_end = cu_q_lens_ref[next_seq_idx + 1]
        next_q_len = next_q_end - next_q_start
        next_kv_len = kv_lens_ref[next_seq_idx]
        next_kv_q_gap = next_kv_len - next_q_len
        next_seq_start_bkv_idx = jnp.maximum(next_kv_q_gap - _sw_off, 0) // bkv_sz

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    def flash_attention_step1_qk_softmax(
        q,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
        k,  # [bkv_csz, head_dim]
        v,  # [bkv_csz, head_dim]
        l_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        m_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        *,
        processed_q_len,
        processed_kv_len,
        effective_kv_len,
        xai_temperature_reg=None,
        custom_mask_data=None,
        apply_mask=True,
    ):
        assert len(q.shape) == 2
        assert q.shape[0] % num_q_heads_per_kv_head == 0
        assert q.shape[1] == head_dim
        actual_bq_csz = q.shape[0] // num_q_heads_per_kv_head
        assert k.shape == (bkv_csz, head_dim)
        assert v.shape == (bkv_csz, head_dim)
        assert l_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert m_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert k.dtype == v.dtype

        # Follow FlashAttention-2 forward pass.
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        s = jnp.matmul(q, k.T, preferred_element_type=jnp.float32)

        s_scale = sm_scale
        if k_scale is not None:
            s_scale *= k_scale
        if q_scale is not None:
            s_scale *= q_scale

        # fold_scale: s stays unscaled (m too), the scale rides in exp2's multiplier.
        fold_scale = "fold_scale" in prefill_opts
        if fold_scale:
            exp_scale = s_scale * math.log2(math.e)
        else:
            s *= s_scale

        # xai temperature scaling
        if xai_temperature_reg is not None:
            s = s * xai_temperature_reg[:, None]

        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)

        # Do not use int16 here unless every intermediate is proven to fit:
        # positions overflow at context lengths >= 32,768, and
        # k_span + sliding_window can overflow even earlier.
        int_ty = jnp.int32
        processed_q_len_int = processed_q_len.astype(int_ty)
        processed_kv_len_int = processed_kv_len.astype(int_ty)
        effective_kv_len_int = effective_kv_len.astype(int_ty)
        q_span = processed_q_len_int + (
            lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
        ).astype(int_ty)
        if kv_pack_bd4:
            qtr_c = bkv_csz // 4
            k_raw = lax.broadcasted_iota(int_ty, s.shape, 1)
            v_raw = lax.broadcasted_iota(int_ty, v.shape, 0)
            k_span = processed_kv_len_int + 4 * (k_raw % qtr_c) + (k_raw // qtr_c)
            v_span = processed_kv_len_int + 4 * (v_raw % qtr_c) + (v_raw // qtr_c)
        elif kv_pack_dense2:
            half_c = bkv_csz // 2
            k_raw = lax.broadcasted_iota(int_ty, s.shape, 1)
            v_raw = lax.broadcasted_iota(int_ty, v.shape, 0)
            k_span = processed_kv_len_int + jnp.where(
                k_raw < half_c, 2 * k_raw, 2 * (k_raw - half_c) + 1
            )
            v_span = processed_kv_len_int + jnp.where(
                v_raw < half_c, 2 * v_raw, 2 * (v_raw - half_c) + 1
            )
        else:
            k_span = processed_kv_len_int + lax.broadcasted_iota(int_ty, s.shape, 1)
            v_span = processed_kv_len_int + lax.broadcasted_iota(int_ty, v.shape, 0)

        mask = None
        if not apply_mask:
            pass
        elif use_causal_mask:
            assert not skip_kv_mask
            mask = mask_and(mask, q_span >= k_span)
        elif custom_mask_data is not None:
            # custom_mask_data: [actual_bq_csz, bkv_csz] int32, 1=keep
            custom_mask_expanded = jnp.repeat(
                custom_mask_data, num_q_heads_per_kv_head, axis=0
            )
            mask = mask_and(mask, custom_mask_expanded == 1)

        if not skip_kv_mask and apply_mask:
            mask = mask_and(mask, k_span < effective_kv_len_int)
            if probe != "novmask" and not (zero_kv_tail and case == RpaCase.DECODE):
                v = jnp.where(v_span < effective_kv_len_int, v, 0.0)

        if sliding_window is not None and apply_mask:
            mask = mask_and(mask, q_span < k_span + sliding_window)

        if mask is not None:
            s = jnp.where(mask, s, mask_value)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = m_ref[...].astype(jnp.float32)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        m_ref[...] = m_curr.astype(out_dtype)
        if fold_scale:
            p = jnp.exp2((s - broadcast_minor(m_curr, s.shape)) * exp_scale)
        else:
            p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        if fold_scale:
            exp_m_diff = jnp.exp2((m_prev - m_curr) * exp_scale)
        else:
            exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = l_ref[...].astype(jnp.float32)
        l_ref[...] = (exp_m_diff * l_prev + p_rowsum).astype(out_dtype)

        return p, v, exp_m_diff

    def flash_attention_step2_pv(
        p,  # [actual_bq_csz * num_q_heads_per_kv_head, bkv_csz]
        v,  # [bkv_csz, head_dim]
        exp_m_diff,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        o_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
    ):
        assert len(p.shape) == 2
        assert p.shape[0] % num_q_heads_per_kv_head == 0
        assert p.shape[1] == bkv_csz
        actual_bq_csz = p.shape[0] // num_q_heads_per_kv_head
        assert v.shape == (bkv_csz, head_dim)
        assert exp_m_diff.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert o_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, head_dim)
        pv = jnp.matmul(p, v, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale
        o_prev = o_ref[...].astype(jnp.float32)
        o_ref[...] = (broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv).astype(
            out_dtype
        )

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx, *, wait=False):
        if custom_mask_ref is None:
            return
        sem = sems.at[4, bkvmask_sem_idx]
        kvmask_vmem_ref = bkvmask_ref.at[bkvmask_sem_idx]

        kv_len = kv_lens_ref[seq_idx]
        mask_len = kv_len
        mask_start = bkvmask_idx * bkv_sz
        mask_left = mask_len - mask_start
        load_kvmask_sz = jnp.minimum(bkv_sz, mask_left)

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        load_q_sz = jnp.minimum(bq_sz, q_end - q_len_start)

        cur_seq_mask_start = cu_seq_mask_lens[seq_idx]
        cur_bq_mask_start = cur_seq_mask_start + bq_idx * bq_sz * kv_len

        def loop_body(i, _):
            start = cur_bq_mask_start + i * kv_len + mask_start
            _async_copy(
                custom_mask_ref.at[pl.ds(start, load_kvmask_sz)],
                kvmask_vmem_ref.at[i, pl.ds(0, load_kvmask_sz)],
                sem,
                wait,
            )
            _async_copy(
                zero_mask_ref.at[pl.ds(0, bkv_sz - load_kvmask_sz)],
                kvmask_vmem_ref.at[i, pl.ds(load_kvmask_sz, bkv_sz - load_kvmask_sz)],
                sem,
                wait,
            )

        lax.fori_loop(0, load_q_sz, loop_body, None, unroll=False)

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = (
            None
            if interpret
            else kv_cache_hbm_ref.reshape(
                cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
            )
        )
        kv_len = kv_lens_ref[seq_idx]
        base = kv_base(seq_idx)
        kv_len_start = base + bkv_idx * bkv_sz
        kv_p_start = base // page_size + bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        # sglang-jax: use cu_kv_lens for page_indices offset.
        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        if kv_pack_dense2 or kv_pack_bd4:
            pack_toks = 4 if kv_pack_bd4 else 2
            assert bnew_kv_ref is not None and bkv_stride == 1
            bnew_kv_rows = bnew_kv_ref.shape[1]
            pack_align_tok = 512 if kv_pack_bd4 else 256
            bnew_step_tok = 3584
            sem_new = sems.at[5, bkv_sem_idx]
            new_kv_len_start = q_end - kv_left_frm_new
            if not wait:
                wait_update_kv_cache(bkv_sem_idx)
                if probe != "nodma":

                    @pl.when(bkv_sz_frm_cache > 0)
                    def _do_cache_dmas_pk():
                        for i in range(bkv_p):
                            tok_sz = jnp.clip(
                                kv_left_frm_cache - i * page_size, 0, page_size
                            )
                            row_sz = cdiv(tok_sz, pack_toks)

                            @pl.when(row_sz > 0)
                            def _copy_pk_page(i=i, row_sz=row_sz):
                                page_idx = jnp.minimum(
                                    page_indices_offset + i, num_page_indices - 1
                                )
                                src_pk = (
                                    kv_cache_hbm_ref.at[
                                        page_indices_ref[page_idx], pl.ds(0, row_sz)
                                    ]
                                    if interpret
                                    else cache_hbm_ref.at[
                                        pl.ds(
                                            page_indices_ref[page_idx] * phys_page_size,
                                            row_sz,
                                        )
                                    ]
                                )
                                _async_copy(
                                    src_pk,
                                    vmem_ref.at[pl.ds(i * phys_page_size, row_sz)],
                                    sem,
                                    wait=False,
                                )

                @pl.when(bkv_sz_frm_new > 0)
                def _do_new_dma_pk():
                    if bnew_kv_rows == bkv_sz:
                        _async_copy(
                            kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                            bnew_kv_ref.at[
                                bkv_sem_idx, pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)
                            ],
                            sem_new,
                            wait=False,
                        )
                    else:
                        lead_0 = bkv_sz_frm_cache & (pack_align_tok - 1)
                        sz_0 = jnp.minimum(bnew_step_tok, bkv_sz_frm_new)
                        _async_copy(
                            kv_hbm_ref.at[pl.ds(new_kv_len_start, sz_0)],
                            bnew_kv_ref.at[bkv_sem_idx, pl.ds(lead_0, sz_0)],
                            sem_new,
                            wait=False,
                        )

            else:
                if probe != "nodma":

                    @pl.when(bkv_sz_frm_cache > 0)
                    def _wait_cache_dmas_pk():
                        cache_rows = cdiv(bkv_sz_frm_cache, pack_toks)
                        c_dst = vmem_ref.at[pl.ds(0, cache_rows)]
                        _async_copy(
                            src=c_dst,
                            dst=c_dst,
                            sem=sem,
                            wait=True,
                        )

                @pl.when(bkv_sz_frm_new > 0)
                def _wait_and_pack_new_pk():
                    if bnew_kv_rows == bkv_sz:
                        n_dst = bnew_kv_ref.at[
                            bkv_sem_idx, pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)
                        ]
                        _async_copy(
                            src=n_dst,
                            dst=n_dst,
                            sem=sem_new,
                            wait=True,
                        )
                        if not debug_mode:
                            total_valid = bkv_sz_frm_cache + bkv_sz_frm_new
                            if kv_pack_bd4:
                                if interpret:
                                    b_u8 = lax.bitcast_convert_type(
                                        bnew_kv_ref[bkv_sem_idx, :, 0, :, :], jnp.uint8
                                    )
                                    old_u8 = lax.bitcast_convert_type(
                                        bkv_x2_ref[bkv_sem_idx, :, 0, :, :], jnp.uint8
                                    )
                                    lo_h = (
                                        lax.broadcasted_iota(
                                            jnp.int32, (bkv_sz // 4, head_dim), 1
                                        )
                                        < 64
                                    )
                                    r_all = lax.broadcasted_iota(
                                        jnp.int32, (bkv_sz // 4, head_dim), 0
                                    )
                                    t0 = 4 * r_all + jnp.where(lo_h, 0, 1)
                                    t1 = 4 * r_all + jnp.where(lo_h, 2, 3)

                                    def _kv16_i(sl_u8):
                                        return sl_u8[:, 0, :].astype(jnp.uint32) | (
                                            sl_u8[:, 1, :].astype(jnp.uint32)
                                            << jnp.uint32(8)
                                        )

                                    n_lo = jnp.where(
                                        lo_h,
                                        _kv16_i(b_u8[0::4]),
                                        jnp.roll(_kv16_i(b_u8[1::4]), 64, axis=1),
                                    )
                                    n_hi = jnp.where(
                                        lo_h,
                                        _kv16_i(b_u8[2::4]),
                                        jnp.roll(_kv16_i(b_u8[3::4]), 64, axis=1),
                                    )
                                    o_lo = old_u8[:, 0, :].astype(jnp.uint32) | (
                                        old_u8[:, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                    )
                                    o_hi = old_u8[:, 2, :].astype(jnp.uint32) | (
                                        old_u8[:, 3, :].astype(jnp.uint32) << jnp.uint32(8)
                                    )
                                    lo_m = jnp.where(
                                        t0 < bkv_sz_frm_cache,
                                        o_lo,
                                        jnp.where(t0 < total_valid, n_lo, jnp.uint32(0)),
                                    )
                                    hi_m = jnp.where(
                                        t1 < bkv_sz_frm_cache,
                                        o_hi,
                                        jnp.where(t1 < total_valid, n_hi, jnp.uint32(0)),
                                    )
                                    out_u8 = jnp.stack(
                                        [
                                            (lo_m & jnp.uint32(0xFF)).astype(jnp.uint8),
                                            ((lo_m >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                                jnp.uint8
                                            ),
                                            (hi_m & jnp.uint32(0xFF)).astype(jnp.uint8),
                                            ((hi_m >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                                jnp.uint8
                                            ),
                                        ],
                                        axis=1,
                                    )
                                    bkv_x2_ref[bkv_sem_idx, :, 0, :, :] = (
                                        lax.bitcast_convert_type(out_u8, bkv_x2_ref.dtype)
                                    )
                                else:
                                    bnew_u32 = (
                                        bnew_kv_ref.bitcast(jnp.uint32)
                                        .at[bkv_sem_idx]
                                        .reshape(bkv_sz, head_dim)
                                    )
                                    dst_u32 = (
                                        bkv_x2_ref.bitcast(jnp.uint32)
                                        .at[bkv_sem_idx, :, :1]
                                        .reshape(bkv_sz // 4, head_dim)
                                    )
                                    row_chunk = 128 if (bkv_sz // 4) >= 128 else 64
                                    n_row_chunks = (bkv_sz // 4) // row_chunk
                                    r_iota = lax.broadcasted_iota(
                                        jnp.int32, (row_chunk, head_dim), 0
                                    )
                                    lo_h = (
                                        lax.broadcasted_iota(
                                            jnp.int32, (row_chunk, head_dim), 1
                                        )
                                        < 64
                                    )
                                    for c_idx in range(n_row_chunks):
                                        tok_base = c_idx * (4 * row_chunk)
                                        row_base = c_idx * row_chunk
                                        t0 = tok_base + 4 * r_iota + jnp.where(lo_h, 0, 1)
                                        t1 = tok_base + 4 * r_iota + jnp.where(lo_h, 2, 3)
                                        old_p = dst_u32[pl.ds(row_base, row_chunk)]
                                        n0 = bnew_u32[pl.ds(tok_base, row_chunk, 4)] & jnp.uint32(0xFFFF)
                                        n1 = pltpu.roll(
                                            bnew_u32[pl.ds(tok_base + 1, row_chunk, 4)] & jnp.uint32(0xFFFF),
                                            shift=64,
                                            axis=1,
                                        )
                                        n2 = bnew_u32[pl.ds(tok_base + 2, row_chunk, 4)] & jnp.uint32(0xFFFF)
                                        n3 = pltpu.roll(
                                            bnew_u32[pl.ds(tok_base + 3, row_chunk, 4)] & jnp.uint32(0xFFFF),
                                            shift=64,
                                            axis=1,
                                        )
                                        new_lo = jnp.where(lo_h, n0, n1)
                                        new_hi = jnp.where(lo_h, n2, n3)
                                        lo = jnp.where(
                                            t0 < bkv_sz_frm_cache,
                                            old_p & jnp.uint32(0xFFFF),
                                            jnp.where(
                                                t0 < total_valid, new_lo, jnp.uint32(0)
                                            ),
                                        )
                                        hi = jnp.where(
                                            t1 < bkv_sz_frm_cache,
                                            (old_p >> jnp.uint32(16)) & jnp.uint32(0xFFFF),
                                            jnp.where(
                                                t1 < total_valid, new_hi, jnp.uint32(0)
                                            ),
                                        )
                                        dst_u32[pl.ds(row_base, row_chunk)] = lo | (
                                            hi << jnp.uint32(16)
                                        )
                            elif interpret:
                                b_u8 = lax.bitcast_convert_type(
                                    bnew_kv_ref[bkv_sem_idx, :, 0, :, :], jnp.uint8
                                )
                                old_u8 = lax.bitcast_convert_type(
                                    bkv_x2_ref[bkv_sem_idx, :, 0, :, :], jnp.uint8
                                )
                                r_all = lax.broadcasted_iota(
                                    jnp.int32, (bkv_sz // 2, head_dim), 0
                                )
                                te = 2 * r_all
                                to = te + 1
                                n_lo = b_u8[0::2, 0, :].astype(jnp.uint32) | (
                                    b_u8[0::2, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                n_hi = b_u8[1::2, 0, :].astype(jnp.uint32) | (
                                    b_u8[1::2, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                o_lo = old_u8[:, 0, :].astype(jnp.uint32) | (
                                    old_u8[:, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                o_hi = old_u8[:, 2, :].astype(jnp.uint32) | (
                                    old_u8[:, 3, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                lo_m = jnp.where(
                                    te < bkv_sz_frm_cache,
                                    o_lo,
                                    jnp.where(te < total_valid, n_lo, jnp.uint32(0)),
                                )
                                hi_m = jnp.where(
                                    to < bkv_sz_frm_cache,
                                    o_hi,
                                    jnp.where(to < total_valid, n_hi, jnp.uint32(0)),
                                )
                                out_u8 = jnp.stack(
                                    [
                                        (lo_m & jnp.uint32(0xFF)).astype(jnp.uint8),
                                        ((lo_m >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                            jnp.uint8
                                        ),
                                        (hi_m & jnp.uint32(0xFF)).astype(jnp.uint8),
                                        ((hi_m >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                            jnp.uint8
                                        ),
                                    ],
                                    axis=1,
                                )
                                bkv_x2_ref[bkv_sem_idx, :, 0, :, :] = (
                                    lax.bitcast_convert_type(out_u8, bkv_x2_ref.dtype)
                                )
                            else:
                                bnew_u32 = (
                                    bnew_kv_ref.bitcast(jnp.uint32)
                                    .at[bkv_sem_idx]
                                    .reshape(bkv_sz, head_dim)
                                )
                                dst_u32 = (
                                    bkv_x2_ref.bitcast(jnp.uint32)
                                    .at[bkv_sem_idx, :, :1]
                                    .reshape(bkv_sz // 2, head_dim)
                                )
                                pair_chunk = 128 if (bkv_sz // 2) >= 128 else 64
                                n_pair_chunks = (bkv_sz // 2) // pair_chunk
                                r_iota = lax.broadcasted_iota(
                                    jnp.int32, (pair_chunk, head_dim), 0
                                )
                                for c_idx in range(n_pair_chunks):
                                    tok_base = c_idx * (2 * pair_chunk)
                                    pair_base = c_idx * pair_chunk
                                    t_even = tok_base + 2 * r_iota
                                    t_odd = t_even + 1
                                    old_p = dst_u32[pl.ds(pair_base, pair_chunk)]
                                    new_lo = bnew_u32[
                                        pl.ds(tok_base, pair_chunk, 2)
                                    ] & jnp.uint32(0xFFFF)
                                    new_hi = bnew_u32[
                                        pl.ds(tok_base + 1, pair_chunk, 2)
                                    ] & jnp.uint32(0xFFFF)
                                    lo = jnp.where(
                                        t_even < bkv_sz_frm_cache,
                                        old_p & jnp.uint32(0xFFFF),
                                        jnp.where(
                                            t_even < total_valid, new_lo, jnp.uint32(0)
                                        ),
                                    )
                                    hi = jnp.where(
                                        t_odd < bkv_sz_frm_cache,
                                        (old_p >> jnp.uint32(16)) & jnp.uint32(0xFFFF),
                                        jnp.where(
                                            t_odd < total_valid, new_hi, jnp.uint32(0)
                                        ),
                                    )
                                    dst_u32[pl.ds(pair_base, pair_chunk)] = lo | (
                                        hi << jnp.uint32(16)
                                    )
                        return

                    def _pack_new_chunk(cur_tok_start, cur_sz):
                        if debug_mode:
                            return
                        lead_c = cur_tok_start & (pack_align_tok - 1)
                        cur_blk_base = cur_tok_start // pack_align_tok
                        n_blks = cdiv(lead_c + cur_sz, pack_align_tok)
                        cur_tok_end = cur_tok_start + cur_sz
                        blk_rows_dst = pack_align_tok // pack_toks
                        if kv_pack_bd4:
                            if interpret:

                                @pl.loop(0, n_blks, unroll=False)
                                def _pk_bd4_i(b_loc):
                                    row_base = (cur_blk_base + b_loc) * 128
                                    tok_base = (cur_blk_base + b_loc) * 512
                                    bnew_base = b_loc * 512
                                    b_u8 = lax.bitcast_convert_type(
                                        bnew_kv_ref[
                                            bkv_sem_idx, pl.ds(bnew_base, 512), 0, :, :
                                        ],
                                        jnp.uint8,
                                    )
                                    old_u8 = lax.bitcast_convert_type(
                                        bkv_x2_ref[
                                            bkv_sem_idx, pl.ds(row_base, 128), 0, :, :
                                        ],
                                        jnp.uint8,
                                    )
                                    lo_h = (
                                        lax.broadcasted_iota(
                                            jnp.int32, (128, head_dim), 1
                                        )
                                        < 64
                                    )
                                    r_all = lax.broadcasted_iota(
                                        jnp.int32, (128, head_dim), 0
                                    )
                                    t0 = tok_base + 4 * r_all + jnp.where(lo_h, 0, 1)
                                    t1 = tok_base + 4 * r_all + jnp.where(lo_h, 2, 3)

                                    def _kv16_i(sl_u8):
                                        return sl_u8[:, 0, :].astype(jnp.uint32) | (
                                            sl_u8[:, 1, :].astype(jnp.uint32)
                                            << jnp.uint32(8)
                                        )

                                    n_lo = jnp.where(
                                        lo_h,
                                        _kv16_i(b_u8[0::4]),
                                        jnp.roll(_kv16_i(b_u8[1::4]), 64, axis=1),
                                    )
                                    n_hi = jnp.where(
                                        lo_h,
                                        _kv16_i(b_u8[2::4]),
                                        jnp.roll(_kv16_i(b_u8[3::4]), 64, axis=1),
                                    )
                                    o_lo = old_u8[:, 0, :].astype(jnp.uint32) | (
                                        old_u8[:, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                    )
                                    o_hi = old_u8[:, 2, :].astype(jnp.uint32) | (
                                        old_u8[:, 3, :].astype(jnp.uint32) << jnp.uint32(8)
                                    )
                                    lo_m = jnp.where(
                                        t0 < cur_tok_start,
                                        o_lo,
                                        jnp.where(t0 < cur_tok_end, n_lo, jnp.uint32(0)),
                                    )
                                    hi_m = jnp.where(
                                        t1 < cur_tok_start,
                                        o_hi,
                                        jnp.where(t1 < cur_tok_end, n_hi, jnp.uint32(0)),
                                    )
                                    out_u8 = jnp.stack(
                                        [
                                            (lo_m & jnp.uint32(0xFF)).astype(jnp.uint8),
                                            ((lo_m >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                                jnp.uint8
                                            ),
                                            (hi_m & jnp.uint32(0xFF)).astype(jnp.uint8),
                                            ((hi_m >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                                jnp.uint8
                                            ),
                                        ],
                                        axis=1,
                                    )
                                    bkv_x2_ref[
                                        bkv_sem_idx, pl.ds(row_base, 128), 0, :, :
                                    ] = lax.bitcast_convert_type(
                                        out_u8, bkv_x2_ref.dtype
                                    )
                            else:
                                bnew_u32 = (
                                    bnew_kv_ref.bitcast(jnp.uint32)
                                    .at[bkv_sem_idx]
                                    .reshape(bnew_kv_rows, head_dim)
                                )
                                dst_u32 = (
                                    bkv_x2_ref.bitcast(jnp.uint32)
                                    .at[bkv_sem_idx, :, :1]
                                    .reshape(bkv_sz // 4, head_dim)
                                )
                                r_iota = lax.broadcasted_iota(
                                    jnp.int32, (128, head_dim), 0
                                )
                                lo_h = (
                                    lax.broadcasted_iota(jnp.int32, (128, head_dim), 1)
                                    < 64
                                )

                                @pl.loop(0, n_blks, unroll=False)
                                def _pk_bd4(b_loc):
                                    row_base = (cur_blk_base + b_loc) * 128
                                    tok_base = (cur_blk_base + b_loc) * 512
                                    bnew_base = b_loc * 512
                                    t0 = tok_base + 4 * r_iota + jnp.where(lo_h, 0, 1)
                                    t1 = tok_base + 4 * r_iota + jnp.where(lo_h, 2, 3)
                                    old_p = dst_u32[pl.ds(row_base, 128)]
                                    n0 = bnew_u32[pl.ds(bnew_base, 128, 4)] & jnp.uint32(0xFFFF)
                                    n1 = pltpu.roll(
                                        bnew_u32[pl.ds(bnew_base + 1, 128, 4)] & jnp.uint32(0xFFFF),
                                        shift=64,
                                        axis=1,
                                    )
                                    n2 = bnew_u32[pl.ds(bnew_base + 2, 128, 4)] & jnp.uint32(0xFFFF)
                                    n3 = pltpu.roll(
                                        bnew_u32[pl.ds(bnew_base + 3, 128, 4)] & jnp.uint32(0xFFFF),
                                        shift=64,
                                        axis=1,
                                    )
                                    new_lo = jnp.where(lo_h, n0, n1)
                                    new_hi = jnp.where(lo_h, n2, n3)
                                    lo = jnp.where(
                                        t0 < cur_tok_start,
                                        old_p & jnp.uint32(0xFFFF),
                                        jnp.where(
                                            t0 < cur_tok_end, new_lo, jnp.uint32(0)
                                        ),
                                    )
                                    hi = jnp.where(
                                        t1 < cur_tok_start,
                                        (old_p >> jnp.uint32(16)) & jnp.uint32(0xFFFF),
                                        jnp.where(
                                            t1 < cur_tok_end, new_hi, jnp.uint32(0)
                                        ),
                                    )
                                    dst_u32[pl.ds(row_base, 128)] = lo | (
                                        hi << jnp.uint32(16)
                                    )
                        elif interpret:

                            @pl.loop(0, n_blks, unroll=False)
                            def _pk_d2_i(b_loc):
                                pair_base = (cur_blk_base + b_loc) * blk_rows_dst
                                tok_base = 2 * pair_base
                                bnew_base = b_loc * 256
                                b_u8 = lax.bitcast_convert_type(
                                    bnew_kv_ref[
                                        bkv_sem_idx, pl.ds(bnew_base, 256), 0, :, :
                                    ],
                                    jnp.uint8,
                                )
                                old_u8 = lax.bitcast_convert_type(
                                    bkv_x2_ref[
                                        bkv_sem_idx, pl.ds(pair_base, 128), 0, :, :
                                    ],
                                    jnp.uint8,
                                )
                                r_all = lax.broadcasted_iota(
                                    jnp.int32, (128, head_dim), 0
                                )
                                te = tok_base + 2 * r_all
                                to = te + 1
                                n_lo = b_u8[0::2, 0, :].astype(jnp.uint32) | (
                                    b_u8[0::2, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                n_hi = b_u8[1::2, 0, :].astype(jnp.uint32) | (
                                    b_u8[1::2, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                o_lo = old_u8[:, 0, :].astype(jnp.uint32) | (
                                    old_u8[:, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                o_hi = old_u8[:, 2, :].astype(jnp.uint32) | (
                                    old_u8[:, 3, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                lo_m = jnp.where(
                                    te < cur_tok_start,
                                    o_lo,
                                    jnp.where(te < cur_tok_end, n_lo, jnp.uint32(0)),
                                )
                                hi_m = jnp.where(
                                    to < cur_tok_start,
                                    o_hi,
                                    jnp.where(to < cur_tok_end, n_hi, jnp.uint32(0)),
                                )
                                out_u8 = jnp.stack(
                                    [
                                        (lo_m & jnp.uint32(0xFF)).astype(jnp.uint8),
                                        ((lo_m >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                            jnp.uint8
                                        ),
                                        (hi_m & jnp.uint32(0xFF)).astype(jnp.uint8),
                                        ((hi_m >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                            jnp.uint8
                                        ),
                                    ],
                                    axis=1,
                                )
                                bkv_x2_ref[
                                    bkv_sem_idx, pl.ds(pair_base, 128), 0, :, :
                                ] = lax.bitcast_convert_type(out_u8, bkv_x2_ref.dtype)
                        else:
                            bnew_u32 = (
                                bnew_kv_ref.bitcast(jnp.uint32)
                                .at[bkv_sem_idx]
                                .reshape(bnew_kv_rows, head_dim)
                            )
                            dst_u32 = (
                                bkv_x2_ref.bitcast(jnp.uint32)
                                .at[bkv_sem_idx, :, :1]
                                .reshape(bkv_sz // 2, head_dim)
                            )
                            r_iota = lax.broadcasted_iota(jnp.int32, (128, head_dim), 0)

                            @pl.loop(0, n_blks, unroll=False)
                            def _pk_d2(b_loc):
                                pair_base = (cur_blk_base + b_loc) * 128
                                tok_base = 2 * pair_base
                                bnew_base = b_loc * 256
                                t_even = tok_base + 2 * r_iota
                                t_odd = t_even + 1
                                old_p = dst_u32[pl.ds(pair_base, 128)]
                                new_lo = bnew_u32[pl.ds(bnew_base, 128, 2)] & jnp.uint32(0xFFFF)
                                new_hi = bnew_u32[pl.ds(bnew_base + 1, 128, 2)] & jnp.uint32(0xFFFF)
                                lo = jnp.where(
                                    t_even < cur_tok_start,
                                    old_p & jnp.uint32(0xFFFF),
                                    jnp.where(t_even < cur_tok_end, new_lo, jnp.uint32(0)),
                                )
                                hi = jnp.where(
                                    t_odd < cur_tok_start,
                                    (old_p >> jnp.uint32(16)) & jnp.uint32(0xFFFF),
                                    jnp.where(t_odd < cur_tok_end, new_hi, jnp.uint32(0)),
                                )
                                dst_u32[pl.ds(pair_base, 128)] = lo | (
                                    hi << jnp.uint32(16)
                                )

                    lead_0 = bkv_sz_frm_cache & (pack_align_tok - 1)
                    sz_0 = jnp.minimum(bnew_step_tok, bkv_sz_frm_new)
                    n_dst0 = bnew_kv_ref.at[bkv_sem_idx, pl.ds(lead_0, sz_0)]
                    _async_copy(src=n_dst0, dst=n_dst0, sem=sem_new, wait=True)
                    _pack_new_chunk(bkv_sz_frm_cache, sz_0)

                    n_new_chunks = cdiv(bkv_sz_frm_new, bnew_step_tok)

                    @pl.loop(1, n_new_chunks, unroll=False)
                    def _pack_more_new_chunks(c_idx):
                        c_off = c_idx * bnew_step_tok
                        cur_tok_start = bkv_sz_frm_cache + c_off
                        cur_sz = jnp.minimum(bnew_step_tok, bkv_sz_frm_new - c_off)
                        lead_c = cur_tok_start & (pack_align_tok - 1)
                        c_dst = bnew_kv_ref.at[bkv_sem_idx, pl.ds(lead_c, cur_sz)]
                        _async_copy(
                            kv_hbm_ref.at[pl.ds(new_kv_len_start + c_off, cur_sz)],
                            c_dst,
                            sem_new,
                            wait=False,
                        )
                        _async_copy(src=c_dst, dst=c_dst, sem=sem_new, wait=True)
                        _pack_new_chunk(cur_tok_start, cur_sz)

            return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            if probe != "nodma":

                @pl.when(bkv_sz_frm_cache > 0)
                def _do_cache_dmas():
                    for i in range(bkv_p):
                        sz = jnp.clip(kv_left_frm_cache - i * page_size, 0, page_size)

                        @pl.when(sz > 0)
                        def _copy_cache_page(i=i, sz=sz):
                            page_idx = jnp.minimum(
                                page_indices_offset + i, num_page_indices - 1
                            )
                            src_c = (
                                kv_cache_hbm_ref.at[
                                    page_indices_ref[page_idx], pl.ds(0, sz)
                                ]
                                if interpret
                                else cache_hbm_ref.at[
                                    pl.ds(page_indices_ref[page_idx] * page_size, sz)
                                ]
                            )
                            _async_copy(
                                src_c,
                                vmem_ref.at[pl.ds(i * page_size, sz)],
                                sem,
                                wait=False,
                            )

            @pl.when(bkv_sz_frm_new > 0)
            def _do_new_dma():
                new_kv_len_start = q_end - kv_left_frm_new
                _async_copy(
                    kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                    vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                    sem,
                    wait,
                )
        else:
            wait_sz = (
                bkv_sz_frm_new
                if probe == "nodma"
                else bkv_sz_frm_cache + bkv_sz_frm_new
            )

            @pl.when(wait_sz > 0)
            def _wait_bkv_dma():
                dst = vmem_ref.at[pl.ds(0, wait_sz)]
                _async_copy(
                    src=dst,
                    dst=dst,
                    sem=sem,
                    wait=True,
                )
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False):
        sem = sems.at[3, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]
        base = kv_base(seq_idx)
        bkv_id = (offset - base) // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - base // page_size - bkv_id * bkv_p
        # sglang-jax: use cu_kv_lens for page_indices offset.
        start_kv_page_idx = cdiv(cu_kv_lens_ref[seq_idx], page_size)
        page_indices_offset = start_kv_page_idx + kv_p_start

        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = (
            None
            if interpret
            else updated_kv_cache_hbm_ref.reshape(
                cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
            )
        )

        if kv_pack_dense2 or kv_pack_bd4:
            pack_toks = 4 if kv_pack_bd4 else 2

            def loop_body_pk(i, states):
                rem_sz, ign = states
                sz = jnp.minimum(page_size - ign, rem_sz)
                row_lo = ign // pack_toks
                n_rows = cdiv(ign + sz, pack_toks) - row_lo
                dst_pk = (
                    updated_kv_cache_hbm_ref.at[
                        page_indices_ref[page_indices_offset + i],
                        pl.ds(row_lo, n_rows),
                    ]
                    if interpret
                    else cache_hbm_ref.at[
                        pl.ds(
                            page_indices_ref[page_indices_offset + i] * phys_page_size
                            + row_lo,
                            n_rows,
                        )
                    ]
                )
                _async_copy(
                    vmem_ref.at[pl.ds((p_ignore + i) * phys_page_size + row_lo, n_rows)],
                    dst_pk,
                    sem,
                    wait,
                )
                return rem_sz - sz, 0

            if not wait:
                lax.fori_loop(
                    0,
                    kv_p_end - kv_p_start,
                    loop_body_pk,
                    (update_sz, ignore),
                    unroll=False,
                )
            else:
                total_rows = cdiv(offset + update_sz, pack_toks) - (offset // pack_toks)
                if interpret:
                    def wait_pk_i(i, states):
                        rem_sz, ign = states
                        sz = jnp.minimum(page_size - ign, rem_sz)
                        row_lo = ign // pack_toks
                        n_rows = cdiv(ign + sz, pack_toks) - row_lo
                        d_i = updated_kv_cache_hbm_ref.at[
                            page_indices_ref[page_indices_offset + i],
                            pl.ds(row_lo, n_rows),
                        ]
                        _async_copy(src=d_i, dst=d_i, sem=sem, wait=True)
                        return rem_sz - sz, 0
                    lax.fori_loop(0, kv_p_end - kv_p_start, wait_pk_i, (update_sz, ignore), unroll=False)
                else:
                    dst = cache_hbm_ref.at[pl.ds(0, total_rows)]
                    _async_copy(
                        src=dst,
                        dst=dst,
                        sem=sem,
                        wait=True,
                    )
            return

        def loop_body(i, states):
            update_sz, ignore = states
            sz = jnp.minimum(page_size - ignore, update_sz)
            dst_n = (
                updated_kv_cache_hbm_ref.at[
                    page_indices_ref[page_indices_offset + i],
                    pl.ds(ignore, sz),
                ]
                if interpret
                else cache_hbm_ref.at[
                    pl.ds(
                        page_indices_ref[page_indices_offset + i] * page_size + ignore,
                        sz,
                    )
                ]
            )
            _async_copy(
                vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                dst_n,
                sem,
                wait,
            )
            return update_sz - sz, 0

        if not wait:
            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),
                unroll=False,
            )
        else:
            if interpret:
                def wait_n_i(i, states):
                    rem_sz, ign = states
                    sz = jnp.minimum(page_size - ign, rem_sz)
                    d_i = updated_kv_cache_hbm_ref.at[
                        page_indices_ref[page_indices_offset + i],
                        pl.ds(ign, sz),
                    ]
                    _async_copy(src=d_i, dst=d_i, sem=sem, wait=True)
                    return rem_sz - sz, 0
                lax.fori_loop(0, kv_p_end - kv_p_start, wait_n_i, (update_sz, ignore), unroll=False)
            else:
                dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
                _async_copy(
                    src=dst,
                    dst=dst,
                    sem=sem,
                    wait=True,
                )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx):
        return _fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx)

    def wait_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx):
        return _fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx, wait=True)

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    use_staged_qo_loop = bool(
        q_t_ref is not None
        and (prefill_opts & _STATIONARY_Q_V2_OPTS)
        and "staged_qo" in prefill_opts
        and bq_x2_ref.shape[2] == 256
        and bq_sz >= 512
        and bq_sz % 256 == 0
    )

    def start_fetch_bq_stage(q_row_start, sz, bq_sem_idx):
        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_row_start, sz)],
            bq_x2_ref.at[bq_sem_idx, :, pl.ds(0, sz)],
            sems.at[1, bq_sem_idx],
            False,
        )

    def wait_fetch_bq_stage(q_row_start, sz, bq_sem_idx):
        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_row_start, sz)],
            bq_x2_ref.at[bq_sem_idx, :, pl.ds(0, sz)],
            sems.at[1, bq_sem_idx],
            True,
        )

    def start_send_bo_staged(o_row_start, sz, bo_sem_idx):
        @pl.when(sz > 0)
        def _():
            bo_ids_ref[bo_sem_idx] = o_row_start
            bo_ids_ref[bo_sem_idx + num_slots] = sz
            _async_copy(
                bo_x2_ref.at[bo_sem_idx, :, pl.ds(0, sz)],
                o_hbm_ref.at[:, pl.ds(o_row_start, sz)],
                sems.at[2, bo_sem_idx],
                False,
            )

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + num_slots] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def start_send_bo_a3(aligned_g0, full_rows, bo_sem_idx):
        @pl.when(full_rows > 0)
        def _():
            bo_ids_ref[bo_sem_idx] = aligned_g0
            bo_ids_ref[bo_sem_idx + num_slots] = full_rows
            al_g0 = pl.multiple_of((aligned_g0 // 16) * 16, 16)
            al_rows = pl.multiple_of((full_rows // 16) * 16, 16)
            _async_copy(
                bo_x2_ref.at[bo_sem_idx, pl.ds(0, al_rows), :],
                o_hbm_ref.at[pl.ds(al_g0, al_rows), :],
                sems.at[2, bo_sem_idx],
                wait=False,
            )

    def wait_send_bo(bo_sem_idx):
        if use_a3_out:
            old_rows = bo_ids_ref[bo_sem_idx + num_slots]

            @pl.when(old_rows > 0)
            def _():
                old_g0 = bo_ids_ref[bo_sem_idx]
                bo_ids_ref[bo_sem_idx] = -1
                bo_ids_ref[bo_sem_idx + num_slots] = 0
                al_old_g0 = pl.multiple_of((old_g0 // 16) * 16, 16)
                al_old_rows = pl.multiple_of((old_rows // 16) * 16, 16)
                _async_copy(
                    bo_x2_ref.at[bo_sem_idx, pl.ds(0, al_old_rows), :],
                    o_hbm_ref.at[pl.ds(al_old_g0, al_old_rows), :],
                    sems.at[2, bo_sem_idx],
                    wait=True,
                )

            return
        if use_staged_qo_loop:
            old_rows = bo_ids_ref[bo_sem_idx + num_slots]

            @pl.when(old_rows > 0)
            def _():
                old_g0 = bo_ids_ref[bo_sem_idx]
                bo_ids_ref[bo_sem_idx] = -1
                bo_ids_ref[bo_sem_idx + num_slots] = 0
                _async_copy(
                    bo_x2_ref.at[bo_sem_idx, :, pl.ds(0, old_rows)],
                    o_hbm_ref.at[:, pl.ds(old_g0, old_rows)],
                    sems.at[2, bo_sem_idx],
                    wait=True,
                )

            return
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + num_slots]

        @pl.when(jnp.logical_and(start_seq_idx <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + num_slots] = offset
        bkv_update_ids_ref[bkv_sem_idx + 2 * num_slots] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 2 * num_slots]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + num_slots]
            bkv_update_ids_ref[bkv_sem_idx + 2 * num_slots] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

    def strided_load(ref, start, sz, step, *, dtype=None):
        assert get_dtype_packing(ref.dtype) == 1
        assert len(ref.shape) == 2
        r, l = ref.shape  # noqa
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        assert sz % step == 0
        vec = jnp.concat(
            [ref[pl.ds(start + i, sz // step, step)] for i in range(folds)], axis=1
        )
        if dtype is not None:
            vec = pltpu.bitcast(vec, dtype)
        return vec

    def strided_store(ref, start, sz, step, val):
        assert get_dtype_packing(ref.dtype) == 1
        assert ref.dtype == val.dtype
        assert ref.shape == val.shape
        assert len(ref.shape) == 2
        r, l = ref.shape  # noqa
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        assert sz % step == 0
        for i in range(folds):
            ref[pl.ds(start + i, sz // step, step)] = val[:, i * 128 : (i + 1) * 128]

    def _mask_valid_lanes(k, v):
        if valid_lane_dim is not None and valid_lane_dim < head_dim:
            lane_ok = lax.broadcasted_iota(jnp.int32, k.shape, 1) < valid_lane_dim
            z = jnp.zeros_like(k)
            return jnp.where(lane_ok, k, z), jnp.where(lane_ok, v, z)
        return k, v

    def load_bq(bq_sem_idx, kv_head_idx, start, sz):
        if interpret:
            return bq_x2_ref[bq_sem_idx, kv_head_idx, pl.ds(start, sz)].reshape(
                sz * num_q_heads_per_kv_head, head_dim
            )
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx, kv_head_idx]
            .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, head_dim)
        )
        start *= num_q_heads_per_kv_head_per_packing
        sz *= num_q_heads_per_kv_head_per_packing
        return strided_load(q_ref, start, sz, 1, dtype=q_dtype)

    def load_bkv(bkv_sem_idx, kv_head_idx, start, sz):
        if kv_pack_bd4:
            assert kv_packing == 4 and kv_head_idx == 0 and bkv_stride == 1
            n_phys = sz // 4
            if interpret:
                raw_u8 = lax.bitcast_convert_type(
                    bkv_x2_ref[bkv_sem_idx, pl.ds(start // 4, n_phys), 0], jnp.uint8
                )
                k0 = lax.bitcast_convert_type(raw_u8[:, 0, :], kv_dtype)
                v0 = lax.bitcast_convert_type(raw_u8[:, 1, :], kv_dtype)
                k1 = jnp.roll(k0, 64, axis=1)
                v1 = jnp.roll(v0, 64, axis=1)
                k2 = lax.bitcast_convert_type(raw_u8[:, 2, :], kv_dtype)
                v2 = lax.bitcast_convert_type(raw_u8[:, 3, :], kv_dtype)
                k3 = jnp.roll(k2, 64, axis=1)
                v3 = jnp.roll(v2, 64, axis=1)
                k_all = jnp.concatenate([k0, k1, k2, k3], axis=0)
                v_all = jnp.concatenate([v0, v1, v2, v3], axis=0)
            else:
                kv_ref = (
                    bkv_x2_ref.bitcast(jnp.uint32)
                    .at[bkv_sem_idx]
                    .reshape(bkv_sz // 4, head_dim)
                )
                pair = kv_ref[pl.ds(start // 4, n_phys)]
                lo = pair & jnp.uint32(0xFFFF)
                hi = pair >> jnp.uint32(16)
                lo_r = pltpu.roll(lo, shift=64, axis=1)
                hi_r = pltpu.roll(hi, shift=64, axis=1)
                ks = [pltpu.bitcast(x.astype(jnp.uint8), kv_dtype) for x in (lo, lo_r, hi, hi_r)]
                vs = [
                    pltpu.bitcast((x >> jnp.uint32(8)).astype(jnp.uint8), kv_dtype)
                    for x in (lo, lo_r, hi, hi_r)
                ]
                k_all = jnp.concatenate(ks, axis=0)
                v_all = jnp.concatenate(vs, axis=0)
            lane_ok = lax.broadcasted_iota(jnp.int32, k_all.shape, 1) < 64
            z = jnp.zeros_like(k_all)
            return jnp.where(lane_ok, k_all, z), jnp.where(lane_ok, v_all, z)

        if kv_pack_dense2:
            assert kv_packing == 4 and kv_head_idx == 0 and bkv_stride == 1
            if interpret:
                raw_u8 = lax.bitcast_convert_type(
                    bkv_x2_ref[bkv_sem_idx, pl.ds(start // 2, sz // 2), 0], jnp.uint8
                )
                k_lo = lax.bitcast_convert_type(raw_u8[:, 0, :], kv_dtype)
                v_lo = lax.bitcast_convert_type(raw_u8[:, 1, :], kv_dtype)
                k_hi = lax.bitcast_convert_type(raw_u8[:, 2, :], kv_dtype)
                v_hi = lax.bitcast_convert_type(raw_u8[:, 3, :], kv_dtype)
                return _mask_valid_lanes(
                    jnp.concatenate([k_lo, k_hi], axis=0),
                    jnp.concatenate([v_lo, v_hi], axis=0),
                )
            kv_ref = (
                bkv_x2_ref.bitcast(jnp.uint32)
                .at[bkv_sem_idx]
                .reshape(bkv_sz // 2, head_dim)
            )
            pair = kv_ref[pl.ds(start // 2, sz // 2)]
            k_lo = pltpu.bitcast(pair.astype(jnp.uint8), kv_dtype)
            v_lo = pltpu.bitcast((pair >> jnp.uint32(8)).astype(jnp.uint8), kv_dtype)
            k_hi = pltpu.bitcast((pair >> jnp.uint32(16)).astype(jnp.uint8), kv_dtype)
            v_hi = pltpu.bitcast((pair >> jnp.uint32(24)).astype(jnp.uint8), kv_dtype)
            return _mask_valid_lanes(
                jnp.concatenate([k_lo, k_hi], axis=0),
                jnp.concatenate([v_lo, v_hi], axis=0),
            )

        if interpret:
            kv_sl = bkv_x2_ref[
                bkv_sem_idx, pl.ds(start, sz), :num_kv_heads_x2_per_kv_packing
            ].reshape(sz, num_kv_heads_x2_per_kv_packing * kv_packing, head_dim)
            return _mask_valid_lanes(
                kv_sl[:, 2 * kv_head_idx, :],
                kv_sl[:, 2 * kv_head_idx + 1, :],
            )

        start *= bkv_stride
        sz *= bkv_stride
        step = bkv_stride
        kv_ref = (
            bkv_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx]
            .reshape(bkv_sz * step, head_dim)
        )

        if kv_packing == 1:
            start += kv_head_idx * 2
            k = strided_load(kv_ref, start, sz, step, dtype=kv_dtype)
            v = strided_load(kv_ref, start + 1, sz, step, dtype=kv_dtype)
            k = pltpu.bitcast(k, kv_dtype)
            v = pltpu.bitcast(v, kv_dtype)
            return _mask_valid_lanes(k, v)

        num_kv_per_load = kv_packing // 2
        offset = kv_head_idx // num_kv_per_load
        kv_idx_in_load = kv_head_idx % num_kv_per_load
        kv = strided_load(kv_ref, start + offset, sz, step)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        k = kv >> (kv_idx_in_load * 2 * bitwidth)
        v = k >> bitwidth
        k = pltpu.bitcast(k.astype(repack_ty), kv_dtype)
        v = pltpu.bitcast(v.astype(repack_ty), kv_dtype)
        return _mask_valid_lanes(k, v)

    def zero_bkv_tail(bkv_sem_idx, valid_sz):
        # Zero buffer rows [valid_sz, bkv_sz): they hold earlier sequences' KV
        # or the prologue's zeros, and V must be finite where p is 0. Replaces
        # the per-chunk V select (measured ~0.6 us per 8k-key decode request).
        assert bkv_stride == 1 and bkv_sz % 8 == 0
        group = 8
        big = 64
        if kv_pack_bd4:
            phys_bkv_sz = bkv_sz // 4
            assert phys_bkv_sz % 8 == 0
            lo_h = lax.broadcasted_iota(jnp.int32, (group, head_dim), 1) < 64
            if interpret:
                cur_u8 = lax.bitcast_convert_type(
                    bkv_x2_ref[bkv_sem_idx, :, 0, :, :], jnp.uint8
                )
                lo_h_all = (
                    lax.broadcasted_iota(jnp.int32, (phys_bkv_sz, head_dim), 1) < 64
                )
                r_all = lax.broadcasted_iota(jnp.int32, (phys_bkv_sz, head_dim), 0)
                t0 = 4 * r_all + jnp.where(lo_h_all, 0, 1)
                t1 = 4 * r_all + jnp.where(lo_h_all, 2, 3)
                ok0 = (t0 < valid_sz)[:, None, :]
                ok1 = (t1 < valid_sz)[:, None, :]
                z_u8 = jnp.zeros((phys_bkv_sz, 2, head_dim), jnp.uint8)
                cur_u8 = cur_u8.at[:, :2, :].set(
                    jnp.where(ok0, cur_u8[:, :2, :], z_u8)
                )
                cur_u8 = cur_u8.at[:, 2:, :].set(
                    jnp.where(ok1, cur_u8[:, 2:, :], z_u8)
                )
                bkv_x2_ref[bkv_sem_idx, :, 0, :, :] = lax.bitcast_convert_type(
                    cur_u8, bkv_x2_ref.dtype
                )
                return
            rows = (
                bkv_x2_ref.bitcast(jnp.uint32)
                .at[bkv_sem_idx]
                .reshape(phys_bkv_sz, head_dim)
            )
            valid_rows = cdiv(valid_sz, 4)
            first = jnp.minimum(valid_rows // group * group, phys_bkv_sz - group)
            head = rows[pl.ds(pl.multiple_of(first, group), group)]
            row_id = first + lax.broadcasted_iota(jnp.int32, head.shape, 0)
            t0 = 4 * row_id + jnp.where(lo_h, 0, 1)
            t1 = 4 * row_id + jnp.where(lo_h, 2, 3)
            m0 = jnp.where(t0 < valid_sz, jnp.uint32(0x0000FFFF), jnp.uint32(0))
            m1 = jnp.where(t1 < valid_sz, jnp.uint32(0xFFFF0000), jnp.uint32(0))
            rows[pl.ds(pl.multiple_of(first, group), group)] = head & (m0 | m1)
            start = first + group
            num_big = jnp.maximum(phys_bkv_sz - start, 0) // big

            def zero_big_bd4(i, _):
                rows[pl.ds(pl.multiple_of(big_start + i * big, group), big)] = (
                    jnp.zeros((big, head_dim), jnp.uint32)
                )

            def zero_group_bd4(i, _):
                rows[pl.ds(pl.multiple_of(start + i * group, group), group)] = (
                    jnp.zeros((group, head_dim), jnp.uint32)
                )

            big_start = phys_bkv_sz - num_big * big
            lax.fori_loop(0, num_big, zero_big_bd4, None)
            lax.fori_loop(
                0, jnp.maximum(big_start - start, 0) // group, zero_group_bd4, None
            )
            return

        if kv_pack_dense2:
            phys_bkv_sz = bkv_sz // 2
            assert phys_bkv_sz % 8 == 0
            rows = (
                bkv_x2_ref.bitcast(jnp.uint32)
                .at[bkv_sem_idx]
                .reshape(phys_bkv_sz, head_dim)
            )
            valid_pairs = cdiv(valid_sz, 2)
            first = jnp.minimum(valid_pairs // group * group, phys_bkv_sz - group)
            head = rows[pl.ds(pl.multiple_of(first, group), group)]
            pair_id = first + lax.broadcasted_iota(jnp.int32, head.shape, 0)
            even_ok = (2 * pair_id) < valid_sz
            odd_ok = (2 * pair_id + 1) < valid_sz
            rows[pl.ds(pl.multiple_of(first, group), group)] = jnp.where(
                odd_ok,
                head,
                jnp.where(even_ok, head & jnp.uint32(0xFFFF), jnp.uint32(0)),
            )
            start = first + group
            num_big = jnp.maximum(phys_bkv_sz - start, 0) // big
            big_start = phys_bkv_sz - num_big * big

            def zero_big_d2(i, _):
                rows[pl.ds(pl.multiple_of(big_start + i * big, group), big)] = (
                    jnp.zeros((big, head_dim), jnp.uint32)
                )

            def zero_group_d2(i, _):
                rows[pl.ds(pl.multiple_of(start + i * group, group), group)] = (
                    jnp.zeros((group, head_dim), jnp.uint32)
                )

            lax.fori_loop(0, num_big, zero_big_d2, None)
            lax.fori_loop(
                0, jnp.maximum(big_start - start, 0) // group, zero_group_d2, None
            )
            return

        if interpret:
            cur_bkv = bkv_x2_ref[bkv_sem_idx]
            r_id = lax.broadcasted_iota(jnp.int32, cur_bkv.shape, 0)
            bkv_x2_ref[bkv_sem_idx] = jnp.where(
                r_id < valid_sz, cur_bkv, jnp.zeros_like(cur_bkv)
            )
            return

        rows = bkv_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(bkv_sz, head_dim)
        first = jnp.minimum(valid_sz // group * group, bkv_sz - group)
        head = rows[pl.ds(pl.multiple_of(first, group), group)]
        row_id = first + lax.broadcasted_iota(jnp.int32, head.shape, 0)
        rows[pl.ds(pl.multiple_of(first, group), group)] = jnp.where(
            row_id < valid_sz, head, jnp.zeros_like(head)
        )
        start = first + group
        num_big = jnp.maximum(bkv_sz - start, 0) // big
        big_start = bkv_sz - num_big * big

        def zero_big(i, _):
            rows[pl.ds(pl.multiple_of(big_start + i * big, group), big)] = jnp.zeros(
                (big, head_dim), jnp.uint32
            )

        def zero_group(i, _):
            rows[pl.ds(pl.multiple_of(start + i * group, group), group)] = jnp.zeros(
                (group, head_dim), jnp.uint32
            )

        lax.fori_loop(0, num_big, zero_big, None)
        lax.fori_loop(0, jnp.maximum(big_start - start, 0) // group, zero_group, None)

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])], axis=-1
        )[..., : shape[-1]]

    def mask_and(mask, new_mask):
        if mask is None:
            return new_mask
        return jnp.logical_and(mask, new_mask)

    def process(static_q_len=None):
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        # A short static width may not divide the compute tile; snap up to the
        # width rather than to a divisor, which can collapse to single rows.
        actual_bq_csz = min(bq_csz, actual_bq_sz)
        if actual_bq_sz % actual_bq_csz != 0:
            actual_bq_csz = actual_bq_sz

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = (
                lax.rem(bq_sem_idx + 1, num_slots)
                if num_slots > 2
                else lax.select(bq_sem_idx == 0, 1, 0)
            )
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *, num_bkv):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = (
                lax.rem(bkv_sem_idx + 1, num_slots)
                if num_slots > 2
                else lax.select(bkv_sem_idx == 0, 1, 0)
            )

            next_bq_start_bkv_idx = 0
            if sliding_window is not None and not window_aligned_kv:
                next_bq_start_bkv_idx = (
                    jnp.maximum(kv_q_gap + (bq_idx + 1) * actual_bq_sz - _sw_off, 0)
                    // bkv_sz
                )
            next_bkv_idx = lax.select(is_last_bkv, next_bq_start_bkv_idx, next_bkv_idx)
            next_bkv_idx = lax.select(is_last_bq, next_seq_start_bkv_idx, next_bkv_idx)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        if q_t_ref is not None and (prefill_opts & _STATIONARY_Q_V2_OPTS):
            _stationary_q_v2_process(
                seq_idx=seq_idx,
                start_seq_idx=start_seq_idx,
                end_seq_idx=end_seq_idx,
                q_len=q_len,
                kv_len=kv_len,
                kv_q_gap=kv_q_gap,
                cur_seq_start_bkv_idx=cur_seq_start_bkv_idx,
                next_seq_start_bkv_idx=next_seq_start_bkv_idx,
                actual_bq_sz=actual_bq_sz,
                num_bq=num_bq,
                bq_sz=bq_sz,
                bkv_sz=bkv_sz,
                bkv_stride=bkv_stride,
                kv_dtype=kv_dtype,
                num_slots=num_slots,
                use_causal_mask=use_causal_mask,
                sliding_window=sliding_window,
                sliding_window_chunk_skip=sliding_window_chunk_skip,
                _sw_off=_sw_off,
                sm_scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                probe=probe,
                prefill_opts=prefill_opts,
                sem_ids_ref=sem_ids_ref,
                bq_x2_ref=bq_x2_ref,
                bkv_x2_ref=bkv_x2_ref,
                bo_x2_ref=bo_x2_ref,
                l_ref=l_ref,
                acc_ref=acc_ref,
                q_t_ref=q_t_ref,
                k_sub_ref=k_sub_ref,
                vt_sub_ref=vt_sub_ref,
                mask_lt_ref=mask_lt_ref,
                mask_ut_ref=mask_ut_ref,
                cu_q_lens_ref=cu_q_lens_ref,
                bo_carry_ref=bo_carry_ref,
                q_sc_ref=q_sc_ref,
                kv_pack_dense2=kv_pack_dense2,
                kv_pack_bd4=kv_pack_bd4,
                get_next_bq_ids=get_next_bq_ids,
                get_next_bkv_ids=get_next_bkv_ids,
                start_fetch_bq=start_fetch_bq,
                wait_fetch_bq=wait_fetch_bq,
                start_fetch_bkv=start_fetch_bkv,
                wait_fetch_bkv=wait_fetch_bkv,
                start_update_kv_cache=start_update_kv_cache,
                wait_send_bo=wait_send_bo,
                start_send_bo=start_send_bo,
                start_send_bo_a3=start_send_bo_a3,
                start_fetch_bq_stage=start_fetch_bq_stage,
                wait_fetch_bq_stage=wait_fetch_bq_stage,
                start_send_bo_staged=start_send_bo_staged,
                interpret=interpret,
                cu_qgrps_ref=cu_qgrps_ref,
                prefix_topk_ref=prefix_topk_ref,
                prefix_topk_slices=prefix_topk_slices,
                prefix_topk_sink_slices=prefix_topk_sink_slices,
                prefix_q_grp_sz=prefix_q_grp_sz,
                prefix_max_bkv=prefix_max_bkv,
                prefix_topk_tail_local=prefix_topk_tail_local,
            )
            return

        if q_t_ref is not None:
            tq_sz = 256
            use_m0 = "stationary_q_m0" in prefill_opts
            use_bkvfast = "bkv_full_unroll" in prefill_opts
            num_s = bkv_sz // 256
            num_sub_per_bkv = bkv_sz // 512
            s_scale = sm_scale * (k_scale if k_scale is not None else 1.0)
            exp_scale = float(s_scale * math.log2(math.e))
            v_scale_val = float(v_scale) if v_scale is not None else 1.0

            def _kq_dot(k_3d, q_3d):
                return lax.dot(
                    k_3d,
                    q_3d,
                    dimension_numbers=(([2], [1]), ([0], [0])),
                    preferred_element_type=jnp.float32,
                )

            def _vp_dot(vt_3d, p_3d):
                return lax.dot(
                    vt_3d,
                    p_3d,
                    dimension_numbers=(([2], [1]), ([0], [0])),
                    preferred_element_type=jnp.float32,
                )

            @pl.loop(0, num_bq, unroll=False)
            def compute_with_bq_stationary_q(bq_idx):
                bq_sem_idx = sem_ids_ref[0]
                next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                    seq_idx, bq_idx, bq_sem_idx
                )
                processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
                start_bkv_idx = jnp.int32(0)
                base = 0
                if sliding_window is not None:
                    start_bkv_idx = jnp.maximum(processed_q_len - _sw_off, 0) // bkv_sz
                if use_causal_mask:
                    effective_kv_len = jnp.minimum(
                        kv_len, processed_q_len + actual_bq_sz
                    )
                else:
                    effective_kv_len = kv_len
                end_bkv_idx = cdiv(effective_kv_len - base, bkv_sz)

                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bq():
                    sem_ids_ref[0] = next_bq_sem_idx
                    start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

                def unpack_bq_to_qt(sem_q):
                    q_u32_ref = (
                        bq_x2_ref.bitcast(jnp.uint32)
                        .at[sem_q, 0]
                        .reshape(bq_sz * 8, 128)
                    )
                    lane = lax.broadcasted_iota(jnp.int32, (tq_sz, 128), 1)
                    is_lo = lane < 64
                    for tq_idx in range(bq_sz // tq_sz):
                        tq_start = tq_idx * tq_sz
                        t_sl = pl.ds(tq_start, tq_sz)
                        for s in range(8):
                            u_s = q_u32_ref[pl.ds(tq_start * 8 + s, tq_sz, 8)]
                            even_bf16 = (
                                pltpu.bitcast(u_s << jnp.uint32(16), jnp.float32)
                                * exp_scale
                            ).astype(jnp.bfloat16)
                            odd_bf16 = (
                                pltpu.bitcast(u_s & jnp.uint32(0xFFFF0000), jnp.float32)
                                * exp_scale
                            ).astype(jnp.bfloat16)
                            if s < 4:
                                pair_q_bf16 = jnp.where(
                                    is_lo,
                                    even_bf16,
                                    pltpu.roll(odd_bf16, shift=64, axis=1),
                                )
                            else:
                                pair_q_bf16 = jnp.where(
                                    is_lo,
                                    pltpu.roll(even_bf16, shift=64, axis=1),
                                    odd_bf16,
                                )
                            pair_q_t = jnp.swapaxes(pair_q_bf16, 0, 1)
                            q_t_ref[2 * s, :, t_sl] = pair_q_t[:64, :]
                            q_t_ref[2 * s + 1, :, t_sl] = pair_q_t[64:, :]

                @pl.when(start_bkv_idx >= end_bkv_idx)
                def init_empty_bq():
                    l_ref[...] = jnp.zeros_like(l_ref)
                    acc_ref[...] = jnp.zeros_like(acc_ref)

                @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
                def compute_with_bkv_stationary_q(bkv_idx):
                    bkv_sem_idx = sem_ids_ref[1]
                    (
                        next_seq_idx_kv,
                        next_bq_idx_kv,
                        next_bkv_idx,
                        next_bkv_sem_idx,
                    ) = get_next_bkv_ids(
                        seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx
                    )
                    processed_kv_len = bkv_idx * bkv_sz

                    @pl.when(next_seq_idx_kv < end_seq_idx)
                    def prefetch_next_bkv():
                        sem_ids_ref[1] = next_bkv_sem_idx
                        start_fetch_bkv(next_seq_idx_kv, next_bkv_idx, next_bkv_sem_idx)

                    @pl.when(bkv_idx == start_bkv_idx)
                    def wait_and_unpack_cur_bq():
                        wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)
                        unpack_bq_to_qt(bq_sem_idx)

                    offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                    is_last_visit = bq_idx == num_bq - 1
                    if sliding_window is not None:
                        next_bq_start_bkv_idx = (
                            jnp.maximum(processed_q_len + actual_bq_sz - _sw_off, 0)
                            // bkv_sz
                        )
                        is_last_visit = jnp.logical_or(
                            is_last_visit, bkv_idx < next_bq_start_bkv_idx
                        )

                    @pl.when(jnp.logical_and(update_sz > 0, is_last_visit))
                    def update_cur_bkv_to_cache():
                        start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                    effective_bkv_sz = jnp.clip(
                        effective_kv_len - processed_kv_len, 0, bkv_sz
                    )

                    if probe != "nocompute":
                        kv_u32_all = (
                            bkv_x2_ref.bitcast(jnp.uint32)
                            .at[bkv_sem_idx]
                            .reshape(bkv_sz * bkv_stride, 128)
                        )
                        full_bkv_valid = effective_bkv_sz == bkv_sz
                        bkv_s_hi = cdiv(effective_bkv_sz, 512) * 2
                        if sliding_window is not None and sliding_window_chunk_skip:
                            win_k0 = jnp.maximum(
                                processed_q_len - sliding_window + 1 - processed_kv_len,
                                0,
                            )
                            bkv_s_lo = jnp.minimum((win_k0 // 512) * 2, bkv_s_hi)
                        else:
                            bkv_s_lo = jnp.int32(0)

                        @pl.when(jnp.logical_and(full_bkv_valid, bkv_s_lo == 0))
                        def unpack_bkv_full():
                            for s_idx in range(num_s):
                                kv_u32 = kv_u32_all[
                                    pl.ds(s_idx * 256 * bkv_stride, 256, bkv_stride)
                                ]
                                k_b = pltpu.bitcast(
                                    kv_u32.astype(jnp.uint8), kv_dtype
                                ).astype(jnp.bfloat16)
                                v_bf16 = pltpu.bitcast(
                                    (kv_u32 >> jnp.uint32(8)).astype(jnp.uint8),
                                    kv_dtype,
                                ).astype(jnp.bfloat16)
                                k_sub_ref[0, s_idx] = k_b
                                k_sub_ref[1, s_idx] = pltpu.roll(k_b, shift=64, axis=1)
                                vt_128 = jnp.swapaxes(v_bf16, 0, 1)
                                vt_sub_ref[0, s_idx] = vt_128[:64, :]
                                vt_sub_ref[1, s_idx] = vt_128[64:, :]

                        @pl.when(
                            jnp.logical_not(
                                jnp.logical_and(full_bkv_valid, bkv_s_lo == 0)
                            )
                        )
                        def unpack_bkv_partial():
                            @pl.loop(bkv_s_lo, bkv_s_hi, unroll=False)
                            def _up_sub(s_idx):
                                bkv_start = s_idx * 256
                                kv_u32 = kv_u32_all[
                                    pl.ds(bkv_start * bkv_stride, 256, bkv_stride)
                                ]
                                k_pos = bkv_start + lax.broadcasted_iota(
                                    jnp.int32, (256, 128), 0
                                )
                                kv_u32 = jnp.where(
                                    k_pos < effective_bkv_sz, kv_u32, jnp.uint32(0)
                                )
                                k_b = pltpu.bitcast(
                                    kv_u32.astype(jnp.uint8), kv_dtype
                                ).astype(jnp.bfloat16)
                                v_bf16 = pltpu.bitcast(
                                    (kv_u32 >> jnp.uint32(8)).astype(jnp.uint8),
                                    kv_dtype,
                                ).astype(jnp.bfloat16)
                                k_sub_ref[0, s_idx] = k_b
                                k_sub_ref[1, s_idx] = pltpu.roll(k_b, shift=64, axis=1)
                                vt_128 = jnp.swapaxes(v_bf16, 0, 1)
                                vt_sub_ref[0, s_idx] = vt_128[:64, :]
                                vt_sub_ref[1, s_idx] = vt_128[64:, :]

                        def _psum(p):
                            p_sum16 = jnp.sum(
                                p.reshape(1, 16, 16, tq_sz),
                                axis=1,
                                dtype=jnp.bfloat16,
                            )
                            return jnp.broadcast_to(
                                jnp.sum(
                                    p_sum16,
                                    axis=1,
                                    keepdims=True,
                                    dtype=jnp.float32,
                                ),
                                (1, 8, tq_sz),
                            )

                        def _sm_m0(qk_f32, mask=None):
                            qk = qk_f32.astype(jnp.bfloat16)
                            if mask is not None:
                                qk = jnp.where(mask, qk, -jnp.inf)
                            p = jnp.exp2(qk)
                            return p, _psum(p)

                        def _sm_pilot_accum_f32(qk_f32, m_f32, mask=None):
                            diff = (
                                (qk_f32.reshape(1, 32, 8, tq_sz) - m_f32[:, None, :, :])
                                .reshape(1, 256, tq_sz)
                                .astype(jnp.bfloat16)
                            )
                            if mask is not None:
                                diff = jnp.where(mask, diff, -jnp.inf)
                            p = jnp.exp2(diff)
                            return p, _psum(p)

                        def _sm_pilot_first_f32(qk_f32):
                            m_f32 = jnp.broadcast_to(
                                jnp.max(qk_f32[:, :32, :], axis=1, keepdims=True),
                                (1, 8, tq_sz),
                            )
                            diff = (
                                (qk_f32.reshape(1, 32, 8, tq_sz) - m_f32[:, None, :, :])
                                .reshape(1, 256, tq_sz)
                                .astype(jnp.bfloat16)
                            )
                            p = jnp.exp2(diff)
                            return p, m_f32, _psum(p)

                        def _sm_online_masked_f32(qk_f32, m_prev_f32, l_prev, mask):
                            qk_masked = jnp.where(mask, qk_f32, -jnp.inf)
                            s_max = jnp.broadcast_to(
                                jnp.max(qk_masked, axis=1, keepdims=True),
                                (1, 8, tq_sz),
                            )
                            m_new = jnp.maximum(m_prev_f32, s_max)
                            m_safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
                            alpha = jnp.exp2(m_prev_f32 - m_safe)
                            diff = (
                                (
                                    qk_masked.reshape(1, 32, 8, tq_sz)
                                    - m_safe[:, None, :, :]
                                )
                                .reshape(1, 256, tq_sz)
                                .astype(jnp.bfloat16)
                            )
                            p = jnp.exp2(diff)
                            return p, m_new, alpha * l_prev + _psum(p), alpha

                        for tq_idx in range(bq_sz // tq_sz):
                            tq_start = tq_idx * tq_sz
                            t_sl = pl.ds(tq_start, tq_sz)

                            @pl.when(bq_idx * actual_bq_sz + tq_start < q_len)
                            def run_tq_stationary_q(tq_start=tq_start, t_sl=t_sl):
                                q_first = processed_q_len + tq_start
                                strip_kv_hi = (
                                    jnp.minimum(effective_kv_len, q_first + tq_sz)
                                    if use_causal_mask
                                    else effective_kv_len
                                )
                                sub_hi = cdiv(
                                    jnp.clip(strip_kv_hi - processed_kv_len, 0, bkv_sz),
                                    512,
                                )
                                if (
                                    sliding_window is not None
                                    and sliding_window_chunk_skip
                                ):
                                    win_fk = jnp.maximum(
                                        q_first - sliding_window + 1 - processed_kv_len,
                                        0,
                                    )
                                    sub_lo = jnp.minimum(win_fk // 512, sub_hi)
                                    global_first_sub = (
                                        jnp.maximum(q_first - sliding_window + 1, 0)
                                        // 512
                                    )
                                else:
                                    sub_lo = jnp.int32(0)
                                    global_first_sub = start_bkv_idx * num_sub_per_bkv

                                @pl.when(
                                    jnp.logical_and(
                                        bkv_idx == start_bkv_idx, sub_lo >= sub_hi
                                    )
                                )
                                def _zero_strip_if_skipped():
                                    if not use_m0:
                                        m_ref[:, :, t_sl] = jnp.full(
                                            (16, 8, tq_sz), -jnp.inf, jnp.float32
                                        )
                                    l_ref[:, :, t_sl] = jnp.zeros(
                                        (16, 8, tq_sz), jnp.float32
                                    )
                                    acc_ref[:, :, t_sl] = jnp.zeros(
                                        (16, 64, tq_sz), jnp.float32
                                    )

                                def _run_m0_slices(
                                    s_start, n_slices, first_vis_cond, masks=None
                                ):
                                    mrb_q = []
                                    ready_p = None

                                    def _adv():
                                        nonlocal ready_p
                                        qk0, qk1, g0, g1, loc_s, vt_s = mrb_q.pop(0)
                                        msk = (
                                            masks[loc_s] if masks is not None else None
                                        )
                                        p0, ps0 = _sm_m0(qk0, msk)
                                        p1, ps1 = _sm_m0(qk1, msk)
                                        if loc_s == 0:
                                            l_ref[g0 : g0 + 1, :, t_sl] = jnp.where(
                                                first_vis_cond,
                                                ps0,
                                                l_ref[g0 : g0 + 1, :, t_sl] + ps0,
                                            )
                                            l_ref[g1 : g1 + 1, :, t_sl] = jnp.where(
                                                first_vis_cond,
                                                ps1,
                                                l_ref[g1 : g1 + 1, :, t_sl] + ps1,
                                            )
                                        else:
                                            l_ref[g0 : g0 + 1, :, t_sl] = (
                                                l_ref[g0 : g0 + 1, :, t_sl] + ps0
                                            )
                                            l_ref[g1 : g1 + 1, :, t_sl] = (
                                                l_ref[g1 : g1 + 1, :, t_sl] + ps1
                                            )
                                        ready_p = (p0, p1, g0, g1, loc_s, vt_s)

                                    def _fl():
                                        nonlocal ready_p
                                        p0, p1, g0, g1, loc_s, vt_s = ready_p
                                        vp0 = _vp_dot(vt_s, p0)
                                        vp1 = _vp_dot(vt_s, p1)
                                        if loc_s == 0:
                                            acc_ref[g0 : g0 + 1, :, t_sl] = jnp.where(
                                                first_vis_cond,
                                                vp0,
                                                acc_ref[g0 : g0 + 1, :, t_sl] + vp0,
                                            )
                                            acc_ref[g1 : g1 + 1, :, t_sl] = jnp.where(
                                                first_vis_cond,
                                                vp1,
                                                acc_ref[g1 : g1 + 1, :, t_sl] + vp1,
                                            )
                                        else:
                                            acc_ref[g0 : g0 + 1, :, t_sl] = (
                                                acc_ref[g0 : g0 + 1, :, t_sl] + vp0
                                            )
                                            acc_ref[g1 : g1 + 1, :, t_sl] = (
                                                acc_ref[g1 : g1 + 1, :, t_sl] + vp1
                                            )
                                        ready_p = None

                                    for kv_h in range(2):
                                        for gp in range(4):
                                            g0 = kv_h * 8 + 2 * gp
                                            g1 = g0 + 1
                                            qt0 = q_t_ref[g0 : g0 + 1, :, t_sl]
                                            qt1 = q_t_ref[g1 : g1 + 1, :, t_sl]
                                            for loc_s in range(n_slices):
                                                s_idx = s_start + loc_s
                                                ks = k_sub_ref[kv_h, s_idx][:, :64][
                                                    None, :, :
                                                ]
                                                vts = vt_sub_ref[kv_h : kv_h + 1, s_idx]
                                                qk0 = _kq_dot(ks, qt0)
                                                qk1 = _kq_dot(ks, qt1)
                                                if ready_p is not None:
                                                    _fl()
                                                mrb_q.append(
                                                    (qk0, qk1, g0, g1, loc_s, vts)
                                                )
                                                if len(mrb_q) >= 2:
                                                    _adv()
                                    while mrb_q:
                                        if ready_p is not None:
                                            _fl()
                                        _adv()
                                    if ready_p is not None:
                                        _fl()

                                if use_bkvfast and use_m0:
                                    bkv_all_unm = (
                                        processed_kv_len + bkv_sz <= effective_kv_len
                                    )
                                    if use_causal_mask:
                                        bkv_all_unm = jnp.logical_and(
                                            bkv_all_unm,
                                            processed_kv_len + bkv_sz <= q_first + 1,
                                        )
                                    if sliding_window is not None:
                                        bkv_all_unm = jnp.logical_and(
                                            bkv_all_unm,
                                            q_first + tq_sz
                                            <= processed_kv_len + sliding_window,
                                        )
                                else:
                                    bkv_all_unm = jnp.bool_(False)

                                if use_bkvfast and use_m0:

                                    @pl.when(bkv_all_unm)
                                    def _run_full_bkv_unmasked():
                                        _run_m0_slices(
                                            0,
                                            num_s,
                                            (bkv_idx * num_sub_per_bkv)
                                            == global_first_sub,
                                            None,
                                        )

                                sub_loop_lo = jnp.where(bkv_all_unm, sub_hi, sub_lo)

                                @pl.loop(sub_loop_lo, sub_hi, unroll=False)
                                def run_sub512(sub_idx):
                                    s_base = sub_idx * 2
                                    sub_kv_len = processed_kv_len + sub_idx * 512
                                    is_first_vis = (
                                        bkv_idx * num_sub_per_bkv + sub_idx
                                    ) == global_first_sub
                                    sub_unmasked = sub_kv_len + 512 <= effective_kv_len
                                    if use_causal_mask:
                                        sub_unmasked = jnp.logical_and(
                                            sub_unmasked,
                                            sub_kv_len + 512 <= q_first + 1,
                                        )
                                    if sliding_window is not None:
                                        sw_cond = (
                                            q_first + tq_sz
                                            <= sub_kv_len + sliding_window
                                            if use_m0
                                            else q_first + tq_sz
                                            < sub_kv_len + sliding_window
                                        )
                                        sub_unmasked = jnp.logical_and(
                                            sub_unmasked, sw_cond
                                        )

                                    if use_m0:

                                        @pl.when(sub_unmasked)
                                        def _m0_unmasked():
                                            _run_m0_slices(
                                                s_base, 2, is_first_vis, None
                                            )

                                        @pl.when(jnp.logical_not(sub_unmasked))
                                        def _m0_masked():
                                            q_pos = q_first + lax.broadcasted_iota(
                                                jnp.int32, (1, 256, tq_sz), 2
                                            )
                                            k_iota = lax.broadcasted_iota(
                                                jnp.int32, (1, 256, tq_sz), 1
                                            )
                                            masks = []
                                            for loc_s in range(2):
                                                kp = sub_kv_len + loc_s * 256 + k_iota
                                                msk = kp < effective_kv_len
                                                if use_causal_mask:
                                                    msk = jnp.logical_and(
                                                        msk, q_pos >= kp
                                                    )
                                                if sliding_window is not None:
                                                    msk = jnp.logical_and(
                                                        msk,
                                                        q_pos < kp + sliding_window,
                                                    )
                                                masks.append(msk)
                                            _run_m0_slices(
                                                s_base, 2, is_first_vis, masks
                                            )

                                    else:
                                        branch_idx = jnp.where(
                                            sub_unmasked,
                                            jnp.where(
                                                is_first_vis,
                                                jnp.int32(0),
                                                jnp.int32(1),
                                            ),
                                            jnp.int32(2),
                                        )

                                        def _p_br0():
                                            mrb_q = []
                                            ready_p = None

                                            def _adv():
                                                nonlocal ready_p
                                                qk0, qk1, g0, g1, loc_s, vt_s = (
                                                    mrb_q.pop(0)
                                                )
                                                if loc_s == 0:
                                                    p0, m0_f, l0 = _sm_pilot_first_f32(
                                                        qk0
                                                    )
                                                    p1, m1_f, l1 = _sm_pilot_first_f32(
                                                        qk1
                                                    )
                                                    m_ref[g0 : g0 + 1, :, t_sl] = m0_f
                                                    m_ref[g1 : g1 + 1, :, t_sl] = m1_f
                                                    l_ref[g0 : g0 + 1, :, t_sl] = l0
                                                    l_ref[g1 : g1 + 1, :, t_sl] = l1
                                                else:
                                                    m0_f = m_ref[g0 : g0 + 1, :, t_sl]
                                                    m1_f = m_ref[g1 : g1 + 1, :, t_sl]
                                                    p0, ps0 = _sm_pilot_accum_f32(
                                                        qk0, m0_f
                                                    )
                                                    p1, ps1 = _sm_pilot_accum_f32(
                                                        qk1, m1_f
                                                    )
                                                    l_ref[g0 : g0 + 1, :, t_sl] = (
                                                        l_ref[g0 : g0 + 1, :, t_sl]
                                                        + ps0
                                                    )
                                                    l_ref[g1 : g1 + 1, :, t_sl] = (
                                                        l_ref[g1 : g1 + 1, :, t_sl]
                                                        + ps1
                                                    )
                                                ready_p = (
                                                    p0,
                                                    p1,
                                                    g0,
                                                    g1,
                                                    loc_s,
                                                    vt_s,
                                                )

                                            def _fl():
                                                nonlocal ready_p
                                                p0, p1, g0, g1, loc_s, vt_s = ready_p
                                                vp0 = _vp_dot(vt_s, p0)
                                                vp1 = _vp_dot(vt_s, p1)
                                                if loc_s == 0:
                                                    acc_ref[g0 : g0 + 1, :, t_sl] = vp0
                                                    acc_ref[g1 : g1 + 1, :, t_sl] = vp1
                                                else:
                                                    acc_ref[g0 : g0 + 1, :, t_sl] = (
                                                        acc_ref[g0 : g0 + 1, :, t_sl]
                                                        + vp0
                                                    )
                                                    acc_ref[g1 : g1 + 1, :, t_sl] = (
                                                        acc_ref[g1 : g1 + 1, :, t_sl]
                                                        + vp1
                                                    )
                                                ready_p = None

                                            for kv_h in range(2):
                                                for gp in range(4):
                                                    g0 = kv_h * 8 + 2 * gp
                                                    g1 = g0 + 1
                                                    qt0 = q_t_ref[g0 : g0 + 1, :, t_sl]
                                                    qt1 = q_t_ref[g1 : g1 + 1, :, t_sl]
                                                    for loc_s in range(2):
                                                        s_idx = s_base + loc_s
                                                        ks = k_sub_ref[kv_h, s_idx][
                                                            :, :64
                                                        ][None, :, :]
                                                        vts = vt_sub_ref[
                                                            kv_h : kv_h + 1, s_idx
                                                        ]
                                                        qk0 = _kq_dot(ks, qt0)
                                                        qk1 = _kq_dot(ks, qt1)
                                                        if ready_p is not None:
                                                            _fl()
                                                        mrb_q.append(
                                                            (
                                                                qk0,
                                                                qk1,
                                                                g0,
                                                                g1,
                                                                loc_s,
                                                                vts,
                                                            )
                                                        )
                                                        if len(mrb_q) >= 2:
                                                            _adv()
                                            while mrb_q:
                                                if ready_p is not None:
                                                    _fl()
                                                _adv()
                                            if ready_p is not None:
                                                _fl()

                                        def _p_br1():
                                            mrb_q = []
                                            ready_p = None

                                            def _adv():
                                                nonlocal ready_p
                                                qk0, qk1, g0, g1, vt_s = mrb_q.pop(0)
                                                m0_f = m_ref[g0 : g0 + 1, :, t_sl]
                                                m1_f = m_ref[g1 : g1 + 1, :, t_sl]
                                                p0, ps0 = _sm_pilot_accum_f32(qk0, m0_f)
                                                p1, ps1 = _sm_pilot_accum_f32(qk1, m1_f)
                                                l_ref[g0 : g0 + 1, :, t_sl] = (
                                                    l_ref[g0 : g0 + 1, :, t_sl] + ps0
                                                )
                                                l_ref[g1 : g1 + 1, :, t_sl] = (
                                                    l_ref[g1 : g1 + 1, :, t_sl] + ps1
                                                )
                                                ready_p = (p0, p1, g0, g1, vt_s)

                                            def _fl():
                                                nonlocal ready_p
                                                p0, p1, g0, g1, vt_s = ready_p
                                                acc_ref[g0 : g0 + 1, :, t_sl] = acc_ref[
                                                    g0 : g0 + 1, :, t_sl
                                                ] + _vp_dot(vt_s, p0)
                                                acc_ref[g1 : g1 + 1, :, t_sl] = acc_ref[
                                                    g1 : g1 + 1, :, t_sl
                                                ] + _vp_dot(vt_s, p1)
                                                ready_p = None

                                            for kv_h in range(2):
                                                for gp in range(4):
                                                    g0 = kv_h * 8 + 2 * gp
                                                    g1 = g0 + 1
                                                    qt0 = q_t_ref[g0 : g0 + 1, :, t_sl]
                                                    qt1 = q_t_ref[g1 : g1 + 1, :, t_sl]
                                                    for loc_s in range(2):
                                                        s_idx = s_base + loc_s
                                                        ks = k_sub_ref[kv_h, s_idx][
                                                            :, :64
                                                        ][None, :, :]
                                                        vts = vt_sub_ref[
                                                            kv_h : kv_h + 1, s_idx
                                                        ]
                                                        qk0 = _kq_dot(ks, qt0)
                                                        qk1 = _kq_dot(ks, qt1)
                                                        if ready_p is not None:
                                                            _fl()
                                                        mrb_q.append(
                                                            (
                                                                qk0,
                                                                qk1,
                                                                g0,
                                                                g1,
                                                                vts,
                                                            )
                                                        )
                                                        if len(mrb_q) >= 2:
                                                            _adv()
                                            while mrb_q:
                                                if ready_p is not None:
                                                    _fl()
                                                _adv()
                                            if ready_p is not None:
                                                _fl()

                                        def _p_br2():
                                            q_pos = q_first + lax.broadcasted_iota(
                                                jnp.int32, (1, 256, tq_sz), 2
                                            )
                                            k_iota = lax.broadcasted_iota(
                                                jnp.int32, (1, 256, tq_sz), 1
                                            )
                                            masks = []
                                            for loc_s in range(2):
                                                kp = sub_kv_len + loc_s * 256 + k_iota
                                                msk = kp < effective_kv_len
                                                if use_causal_mask:
                                                    msk = jnp.logical_and(
                                                        msk, q_pos >= kp
                                                    )
                                                if sliding_window is not None:
                                                    msk = jnp.logical_and(
                                                        msk,
                                                        q_pos < kp + sliding_window,
                                                    )
                                                masks.append(msk)
                                            m_neg_f = jnp.full(
                                                (1, 8, tq_sz), -jnp.inf, jnp.float32
                                            )

                                            mrb_q = []
                                            ready_p = None

                                            def _adv():
                                                nonlocal ready_p
                                                qk0, qk1, g0, g1, loc_s, vt_s = (
                                                    mrb_q.pop(0)
                                                )
                                                if loc_s == 0:
                                                    m0_in = jnp.where(
                                                        is_first_vis,
                                                        m_neg_f,
                                                        m_ref[g0 : g0 + 1, :, t_sl],
                                                    )
                                                    l0_in = jnp.where(
                                                        is_first_vis,
                                                        jnp.zeros(
                                                            (1, 8, tq_sz),
                                                            jnp.float32,
                                                        ),
                                                        l_ref[g0 : g0 + 1, :, t_sl],
                                                    )
                                                    m1_in = jnp.where(
                                                        is_first_vis,
                                                        m_neg_f,
                                                        m_ref[g1 : g1 + 1, :, t_sl],
                                                    )
                                                    l1_in = jnp.where(
                                                        is_first_vis,
                                                        jnp.zeros(
                                                            (1, 8, tq_sz),
                                                            jnp.float32,
                                                        ),
                                                        l_ref[g1 : g1 + 1, :, t_sl],
                                                    )
                                                else:
                                                    m0_in = m_ref[g0 : g0 + 1, :, t_sl]
                                                    l0_in = l_ref[g0 : g0 + 1, :, t_sl]
                                                    m1_in = m_ref[g1 : g1 + 1, :, t_sl]
                                                    l1_in = l_ref[g1 : g1 + 1, :, t_sl]
                                                p0, m0_n, l0_n, al0 = (
                                                    _sm_online_masked_f32(
                                                        qk0,
                                                        m0_in,
                                                        l0_in,
                                                        masks[loc_s],
                                                    )
                                                )
                                                p1, m1_n, l1_n, al1 = (
                                                    _sm_online_masked_f32(
                                                        qk1,
                                                        m1_in,
                                                        l1_in,
                                                        masks[loc_s],
                                                    )
                                                )
                                                m_ref[g0 : g0 + 1, :, t_sl] = m0_n
                                                l_ref[g0 : g0 + 1, :, t_sl] = l0_n
                                                m_ref[g1 : g1 + 1, :, t_sl] = m1_n
                                                l_ref[g1 : g1 + 1, :, t_sl] = l1_n
                                                ready_p = (
                                                    p0,
                                                    p1,
                                                    al0,
                                                    al1,
                                                    g0,
                                                    g1,
                                                    loc_s,
                                                    vt_s,
                                                )

                                            def _fl():
                                                nonlocal ready_p
                                                (
                                                    p0,
                                                    p1,
                                                    al0,
                                                    al1,
                                                    g0,
                                                    g1,
                                                    loc_s,
                                                    vt_s,
                                                ) = ready_p
                                                vp0 = _vp_dot(vt_s, p0)
                                                vp1 = _vp_dot(vt_s, p1)
                                                if loc_s == 0:
                                                    a0_in = jnp.where(
                                                        is_first_vis,
                                                        jnp.zeros(
                                                            (1, 64, tq_sz),
                                                            jnp.float32,
                                                        ),
                                                        acc_ref[g0 : g0 + 1, :, t_sl],
                                                    )
                                                    a1_in = jnp.where(
                                                        is_first_vis,
                                                        jnp.zeros(
                                                            (1, 64, tq_sz),
                                                            jnp.float32,
                                                        ),
                                                        acc_ref[g1 : g1 + 1, :, t_sl],
                                                    )
                                                else:
                                                    a0_in = acc_ref[
                                                        g0 : g0 + 1, :, t_sl
                                                    ]
                                                    a1_in = acc_ref[
                                                        g1 : g1 + 1, :, t_sl
                                                    ]
                                                acc_ref[g0 : g0 + 1, :, t_sl] = (
                                                    a0_in.reshape(1, 8, 8, tq_sz)
                                                    * al0[:, None, :, :]
                                                ).reshape(1, 64, tq_sz) + vp0
                                                acc_ref[g1 : g1 + 1, :, t_sl] = (
                                                    a1_in.reshape(1, 8, 8, tq_sz)
                                                    * al1[:, None, :, :]
                                                ).reshape(1, 64, tq_sz) + vp1
                                                ready_p = None

                                            for kv_h in range(2):
                                                for gp in range(4):
                                                    g0 = kv_h * 8 + 2 * gp
                                                    g1 = g0 + 1
                                                    qt0 = q_t_ref[g0 : g0 + 1, :, t_sl]
                                                    qt1 = q_t_ref[g1 : g1 + 1, :, t_sl]
                                                    for loc_s in range(2):
                                                        s_idx = s_base + loc_s
                                                        ks = k_sub_ref[kv_h, s_idx][
                                                            :, :64
                                                        ][None, :, :]
                                                        vts = vt_sub_ref[
                                                            kv_h : kv_h + 1, s_idx
                                                        ]
                                                        qk0 = _kq_dot(ks, qt0)
                                                        qk1 = _kq_dot(ks, qt1)
                                                        if ready_p is not None:
                                                            _fl()
                                                        mrb_q.append(
                                                            (
                                                                qk0,
                                                                qk1,
                                                                g0,
                                                                g1,
                                                                loc_s,
                                                                vts,
                                                            )
                                                        )
                                                        if len(mrb_q) >= 2:
                                                            _adv()
                                            while mrb_q:
                                                if ready_p is not None:
                                                    _fl()
                                                _adv()
                                            if ready_p is not None:
                                                _fl()

                                        lax.switch(
                                            branch_idx,
                                            [
                                                lambda: (_p_br0(), jnp.int32(0))[1],
                                                lambda: (_p_br1(), jnp.int32(0))[1],
                                                lambda: (_p_br2(), jnp.int32(0))[1],
                                            ],
                                        )

                bo_sem_idx = sem_ids_ref[2]
                sem_ids_ref[2] = (
                    lax.rem(bo_sem_idx + 1, num_slots)
                    if num_slots > 2
                    else lax.select(bo_sem_idx == 0, 1, 0)
                )
                wait_send_bo(bo_sem_idx)

                if use_a3_out:
                    g0_ep = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
                    valid_ep = jnp.minimum(bq_sz, cu_q_lens_ref[seq_idx + 1] - g0_ep)
                    aligned_g0_ep = (g0_ep // 16) * 16
                    lead_ep = g0_ep - aligned_g0_ep
                    total_rows_ep = lead_ep + valid_ep
                    full_rows_ep = (total_rows_ep // 16) * 16
                    row_iota_ep = lax.broadcasted_iota(jnp.int32, (bq_sz + 16, 128), 0)
                    zero_tail_16 = jnp.zeros((16, 128), jnp.float32)

                    for s in range(8):
                        h0 = 2 * s
                        h1 = 2 * s + 1
                        parts_s = []
                        for tq_idx in range(bq_sz // tq_sz):
                            tq_start = tq_idx * tq_sz
                            t_sl = pl.ds(tq_start, tq_sz)
                            l0 = l_ref[h0, :, t_sl]
                            l1 = l_ref[h1, :, t_sl]
                            inv_l0 = jnp.where(
                                l0 > 0.0,
                                pltpu.reciprocal(l0, approx=True) * v_scale_val,
                                0.0,
                            )
                            inv_l1 = jnp.where(
                                l1 > 0.0,
                                pltpu.reciprocal(l1, approx=True) * v_scale_val,
                                0.0,
                            )
                            o0_t = (
                                (
                                    acc_ref[h0, :, t_sl].reshape(8, 8, tq_sz)
                                    * inv_l0[None, :, :]
                                )
                                .reshape(64, tq_sz)
                                .astype(jnp.bfloat16)
                            )
                            o1_t = (
                                (
                                    acc_ref[h1, :, t_sl].reshape(8, 8, tq_sz)
                                    * inv_l1[None, :, :]
                                )
                                .reshape(64, tq_sz)
                                .astype(jnp.bfloat16)
                            )
                            pair_t = jnp.concatenate([o0_t, o1_t], axis=0)
                            parts_s.append(
                                jnp.swapaxes(pair_t, 0, 1).astype(jnp.float32) + 0.0
                            )
                        col_s = jnp.concatenate([*parts_s, zero_tail_16], axis=0)
                        col_s = jnp.where(row_iota_ep < valid_ep, col_s, 0.0)

                        @pl.when(lead_ep == 0)
                        def _store_aligned(s=s, col_s=col_s):
                            bo_x2_ref[bo_sem_idx, :, s * 128 : (s + 1) * 128] = (
                                col_s.astype(jnp.bfloat16)
                            )

                        @pl.when(lead_ep > 0)
                        def _store_shifted(s=s, col_s=col_s):
                            shifted = col_s
                            for sh in (1, 2, 4, 8):
                                shifted = jnp.where(
                                    (lead_ep & sh) != 0,
                                    pltpu.roll(shifted, shift=sh, axis=0),
                                    shifted,
                                )
                            carry_s = bo_carry_ref[:, s * 128 : (s + 1) * 128].astype(
                                jnp.float32
                            )
                            carry_pad = jnp.concatenate(
                                [
                                    carry_s,
                                    jnp.zeros((bq_sz, 128), jnp.float32),
                                ],
                                axis=0,
                            )
                            merged = jnp.where(
                                row_iota_ep < lead_ep, carry_pad, shifted
                            )
                            bo_x2_ref[bo_sem_idx, :, s * 128 : (s + 1) * 128] = (
                                merged.astype(jnp.bfloat16)
                            )

                    bo_carry_ref[...] = bo_x2_ref[
                        bo_sem_idx,
                        pl.ds(pl.multiple_of((total_rows_ep // 16) * 16, 16), 16),
                        :,
                    ]
                    start_send_bo_a3(aligned_g0_ep, full_rows_ep, bo_sem_idx)
                    return

                out_u32_ref = (
                    bo_x2_ref.bitcast(jnp.uint32)
                    .at[bo_sem_idx, 0]
                    .reshape(bq_sz * 8, 128)
                )
                lane_ep = lax.broadcasted_iota(jnp.int32, (tq_sz, 128), 1)
                is_lo_ep = lane_ep < 64
                mask_lo_u32 = jnp.where(is_lo_ep, jnp.uint32(0xFFFFFFFF), jnp.uint32(0))
                mask_hi_u32 = jnp.where(is_lo_ep, jnp.uint32(0), jnp.uint32(0xFFFFFFFF))

                for tq_idx in range(bq_sz // tq_sz):
                    tq_start = tq_idx * tq_sz
                    t_sl = pl.ds(tq_start, tq_sz)
                    for s in range(8):
                        h0 = 2 * s
                        h1 = 2 * s + 1
                        l0 = l_ref[h0, :, t_sl]
                        l1 = l_ref[h1, :, t_sl]
                        inv_l0 = jnp.where(
                            l0 > 0.0,
                            pltpu.reciprocal(l0, approx=True) * v_scale_val,
                            0.0,
                        )
                        inv_l1 = jnp.where(
                            l1 > 0.0,
                            pltpu.reciprocal(l1, approx=True) * v_scale_val,
                            0.0,
                        )
                        o0_t = (
                            (
                                acc_ref[h0, :, t_sl].reshape(8, 8, tq_sz)
                                * inv_l0[None, :, :]
                            )
                            .reshape(64, tq_sz)
                            .astype(jnp.bfloat16)
                        )
                        o1_t = (
                            (
                                acc_ref[h1, :, t_sl].reshape(8, 8, tq_sz)
                                * inv_l1[None, :, :]
                            )
                            .reshape(64, tq_sz)
                            .astype(jnp.bfloat16)
                        )
                        pair_t = jnp.concatenate([o0_t, o1_t], axis=0)
                        pair = jnp.swapaxes(pair_t, 0, 1).astype(jnp.float32)
                        rolled = pltpu.roll(pair, shift=64, axis=1)
                        even = jnp.where(is_lo_ep, pair, rolled)
                        odd = jnp.where(is_lo_ep, rolled, pair)
                        lo_u32 = pltpu.bitcast(even, jnp.uint32)
                        hi_u32 = pltpu.bitcast(odd, jnp.uint32)
                        packed_u32 = lax.shift_right_logical(lo_u32, jnp.uint32(16)) | (
                            hi_u32 & jnp.uint32(0xFFFF0000)
                        )
                        out_u32_ref[pl.ds(tq_start * 8 + s, tq_sz, 8)] = packed_u32 & (
                            mask_lo_u32 if s < 4 else mask_hi_u32
                        )

                start_send_bo(seq_idx, bq_idx, bo_sem_idx)

            return

        @pl.loop(0, num_bq, unroll=False)
        def compute_with_bq(bq_idx):
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)

            # Initialize l, m before bkv loop.
            if attention_sink_ref is not None:
                # Attention sink: m = sink logits, l = 1.0
                # (pretend we've already seen a virtual token with logit = sink_value).
                m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
                l_ref[...] = jnp.full_like(l_ref, 1.0)
                for kv_head_idx in range(actual_num_kv_heads):
                    sinks = attention_sink_ref[
                        kv_head_idx
                    ]  # [num_q_heads_per_kv_head, 128]
                    lm_start = 0
                    lm_size = actual_bq_sz * num_q_heads_per_kv_head
                    sink_tiled = jnp.tile(sinks, (actual_bq_sz, 1))
                    m_ref.at[kv_head_idx, pl.ds(lm_start, lm_size)][...] = (
                        sink_tiled.astype(out_dtype)
                    )
            else:
                l_ref[...] = jnp.full_like(l_ref, 0.0)
                m_ref[...] = jnp.full_like(m_ref, -jnp.inf)

            bq_sem_idx = ring_slot(seq_idx) if ring else sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
            start_bkv_idx = 0
            base = kv_base(seq_idx)
            if window_aligned_kv:
                # Decode has one query block: no later block revisits any KV block.
                next_bq_start_bkv_idx = 0
            elif sliding_window is not None:
                start_bkv_idx = (
                    jnp.maximum(processed_q_len - sliding_window, 0) // bkv_sz
                )
                # The next query block starts here: this query block is the
                # last one to visit the KV blocks before it.
                next_bq_start_bkv_idx = (
                    jnp.maximum(processed_q_len + actual_bq_sz - sliding_window, 0)
                    // bkv_sz
                )
            if use_causal_mask:
                effective_kv_len = jnp.minimum(kv_len, processed_q_len + actual_bq_sz)
            else:
                effective_kv_len = kv_len
            end_bkv_idx = cdiv(effective_kv_len - base, bkv_sz)

            # xai temperature computation
            xai_temperature_reg = None
            if xai_temperature_len is not None:
                prefix_len = kv_len - q_len
                local_q_offset = (
                    bq_idx * bq_sz
                    + lax.iota(jnp.int32, actual_bq_sz * num_q_heads_per_kv_head)
                    // num_q_heads_per_kv_head
                )
                absolute_q_position = prefix_len + local_q_offset
                xai_temperature_scale = 1.0 / jnp.log2(float(xai_temperature_len))
                _qtemp = (
                    jnp.log2(absolute_q_position.astype(jnp.float32))
                    * xai_temperature_scale
                )
                xai_temperature_reg = jnp.where(
                    absolute_q_position > xai_temperature_len, _qtemp, 1.0
                )

            ring_seq = seq_idx + num_slots - 1

            # Prefetch next bq
            @pl.when((ring_seq if ring else next_seq_idx) < end_seq_idx)
            def prefetch_next_bq():
                if ring:
                    start_fetch_bq(ring_seq, 0, ring_slot(ring_seq))
                    return
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
            def compute_with_bkv(bkv_idx):
                assert bkv_sz % kv_packing == 0

                # Get next bkv ids.
                bkv_sem_idx = ring_slot(seq_idx) if ring else sem_ids_ref[1]
                next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx = (
                    get_next_bkv_ids(
                        seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx
                    )
                )
                processed_kv_len = base + bkv_idx * bkv_sz

                # Prefetch next bkv
                @pl.when((ring_seq if ring else next_seq_idx) < end_seq_idx)
                def prefetch_next_bkv():
                    if ring:
                        start_fetch_bkv(ring_seq, 0, ring_slot(ring_seq))
                        return
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)
                    if custom_mask_ref is not None:
                        start_fetch_mask(
                            next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx
                        )

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == start_bkv_idx)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)
                if zero_kv_tail:
                    valid_bkv_rows = jnp.clip(
                        (effective_kv_len if case == RpaCase.DECODE else kv_len)
                        - processed_kv_len,
                        0,
                        bkv_sz,
                    )
                    zero_bkv_tail(bkv_sem_idx, valid_bkv_rows)

                # Wait for custom mask if applicable
                if custom_mask_ref is not None:
                    wait_fetch_mask(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Write this block's new tokens to the KV cache on the last
                # query block that visits it. Without a sliding window, that is
                # the sequence's last query block. With one, each query block
                # starts at its own window, so blocks left of the next query
                # block's window are not visited again: write them now, or the
                # new tokens there never reach the cache and a later prefix hit
                # inside this prompt reads stale K/V.
                is_last_visit = bq_idx == num_bq - 1
                if sliding_window is not None:
                    is_last_visit = jnp.logical_or(
                        is_last_visit, bkv_idx < next_bq_start_bkv_idx
                    )

                @pl.when(jnp.logical_and(update_sz > 0, is_last_visit))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                if debug_mode:
                    return

                # Load custom mask data for this block
                custom_mask_data = None
                if bkvmask_ref is not None:
                    custom_mask_data = bkvmask_ref[bkv_sem_idx, :actual_bq_sz, :, 0]

                # Flash attention with cur bkv and bq
                effective_bkv_sz = jnp.minimum(
                    effective_kv_len - processed_kv_len, bkv_sz
                )
                effective_bkv_sz = jnp.maximum(effective_bkv_sz, 0)

                # Use static loop bound to avoid potential Pallas issues with
                # dynamic loop bounds. The @pl.when guard skips invalid iterations.
                max_num_loops = bkv_sz // bkv_csz

                def compute_chunk(idx, *, interior):
                    bkv_start = idx * bkv_csz
                    chunk_has_keys = bkv_start < effective_bkv_sz
                    if sliding_window is not None and sliding_window_chunk_skip:
                        # The block loop above skips whole KV blocks left of the
                        # window, but a kept block is still computed from its
                        # first key. A compute chunk that ends before the
                        # window of this query block's first row is masked for
                        # every row (later rows' windows start later), so skip
                        # it: it cannot change the output. With a sink, m starts
                        # finite and the chunk's weights exp(mask_value - m) are
                        # 0; without one, what it accumulates is scaled by
                        # exp(mask_value - m) = 0 at the row's first visible key.
                        # Key k is visible from query q iff q < k + window.
                        window_start = processed_q_len - sliding_window + 1
                        chunk_end = processed_kv_len + bkv_start + bkv_csz
                        chunk_has_keys = jnp.logical_and(
                            chunk_has_keys, chunk_end > window_start
                        )

                    if probe == "nocompute":
                        chunk_has_keys = False

                    def sub_block(bq_start, apply_mask=True):
                        # Slice custom mask for this compute sub-block
                        cur_mask_data = None
                        if custom_mask_data is not None:
                            cur_mask_data = bkvmask_ref[
                                bkv_sem_idx,
                                pl.ds(bq_start, actual_bq_csz),
                                pl.ds(bkv_start, bkv_csz),
                                0,
                            ]

                        # Slice xai temperature for this compute sub-block
                        cur_xai_temp = None
                        if xai_temperature_reg is not None:
                            q_head_start = bq_start * num_q_heads_per_kv_head
                            q_head_sz = actual_bq_csz * num_q_heads_per_kv_head
                            cur_xai_temp = xai_temperature_reg[
                                q_head_start : q_head_start + q_head_sz
                            ]

                        for kv_head_idx in range(actual_num_kv_heads):
                            bk_c, bv_c = load_bkv(
                                bkv_sem_idx,
                                kv_head_idx,
                                bkv_start,
                                bkv_csz,
                            )
                            bq_c = load_bq(
                                bq_sem_idx, kv_head_idx, bq_start, actual_bq_csz
                            )

                            lm_slice_start = bq_start * num_q_heads_per_kv_head
                            lm_slice_size = actual_bq_csz * num_q_heads_per_kv_head
                            lm_slice = (
                                kv_head_idx,
                                pl.ds(lm_slice_start, lm_slice_size),
                            )

                            cur_p, cur_v, cur_exp_m_diff = (
                                flash_attention_step1_qk_softmax(
                                    bq_c,
                                    bk_c,
                                    bv_c,
                                    l_ref.at[lm_slice],
                                    m_ref.at[lm_slice],
                                    processed_q_len=processed_q_len + bq_start,
                                    processed_kv_len=processed_kv_len + bkv_start,
                                    effective_kv_len=effective_kv_len,
                                    xai_temperature_reg=cur_xai_temp,
                                    custom_mask_data=cur_mask_data,
                                    apply_mask=apply_mask,
                                )
                            )
                            flash_attention_step2_pv(
                                cur_p,
                                cur_v,
                                cur_exp_m_diff,
                                acc_ref.at[lm_slice],
                            )

                    def sub_block_maybe_masked(bq_start):
                        if "mask_skip" not in prefill_opts:
                            sub_block(bq_start)
                            return
                        # Every (row, key) of this sub-block is visible: skip the mask.
                        q_first = processed_q_len + bq_start
                        k_first = processed_kv_len + bkv_start
                        no_mask = k_first + bkv_csz <= effective_kv_len
                        if use_causal_mask:
                            no_mask &= k_first + bkv_csz - 1 <= q_first
                        if sliding_window is not None:
                            no_mask &= (
                                q_first + actual_bq_csz - 1 < k_first + sliding_window
                            )
                        pl.when(no_mask)(lambda: sub_block(bq_start, apply_mask=False))
                        pl.when(jnp.logical_not(no_mask))(lambda: sub_block(bq_start))

                    if interior:
                        for bq_start in range(0, actual_bq_sz, actual_bq_csz):
                            sub_block(bq_start, apply_mask=False)
                    else:

                        @pl.when(chunk_has_keys)
                        def _():
                            for bq_start in range(0, actual_bq_sz, actual_bq_csz):
                                sub_block_maybe_masked(bq_start)

                def run_chunks(interior):
                    unroll = 2 if interior else False

                    @pl.loop(0, max_num_loops, unroll=unroll)
                    def attention_loop(idx):
                        compute_chunk(idx, interior=interior)

                if "visible_kv_tile" in prefill_opts and probe is None:
                    # Every query sees every key in an interior tile. Prove this once
                    # for all its compute chunks. Keep the per-chunk BF16 state updates
                    # in their original order on both sides of the branch.
                    end = processed_kv_len + bkv_sz
                    complete = end <= effective_kv_len
                    if use_causal_mask:
                        complete &= end - 1 <= processed_q_len
                    if sliding_window is not None:
                        complete &= (
                            processed_q_len + actual_bq_sz - 1
                            < processed_kv_len + sliding_window
                        )
                    pl.when(complete)(lambda: run_chunks(True))
                    pl.when(jnp.logical_not(complete))(lambda: run_chunks(False))
                else:
                    run_chunks(False)

            # Load acc and calculate final output.
            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)  # noqa
            out = (
                acc * pltpu.reciprocal(l, approx=True)
                if (l.dtype == jnp.float32 and out_dtype != jnp.float32)
                else lax.div(acc, l)
            ).astype(out_dtype)

            # Wait for previous bo to be fully sent before storing new bo.
            if ring:
                bo_sem_idx = ring_slot(seq_idx)
            else:
                bo_sem_idx = sem_ids_ref[2]
                sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            if interpret:
                bo_x2_ref[bo_sem_idx] = out.reshape(bo_x2_ref.shape[1:])
                start_send_bo(seq_idx, bq_idx, bo_sem_idx)
                return
            out_ref = (
                bo_x2_ref.at[bo_sem_idx]
                .bitcast(jnp.int32)
                .reshape(
                    actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head_per_packing,
                    head_dim,
                )
            )
            out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
            strided_store(out_ref, 0, out_ref.shape[0], 1, out)

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    def ring_prologue():
        if interpret:
            zeros = jnp.zeros(bkv_x2_ref.shape[1:], bkv_x2_ref.dtype)
            for slot in range(num_slots):
                bkv_x2_ref[slot] = zeros
        else:
            bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32)
            zeros = jnp.zeros(bkv_x2_int32_ref.shape[1:], jnp.int32)
            for slot in range(num_slots):
                bkv_x2_int32_ref[slot] = zeros
        for slot in range(num_slots - 1):

            @pl.when(start_seq_idx + slot < end_seq_idx)
            def _():
                start_fetch_bq(start_seq_idx + slot, 0, slot)
                start_fetch_bkv(start_seq_idx + slot, 0, slot)

    def prologue():
        if use_a3_out:
            bo_carry_ref[...] = jnp.zeros_like(bo_carry_ref)
        if not use_staged_qo_loop:
            start_fetch_bq(seq_idx=start_seq_idx, bq_idx=0, bq_sem_idx=0)
        # Initialize bkv_x2_ref to zeros to avoid NaN issues.
        # Use bitcast to int32 and preserve actual shape (which may include bank conflict padding)
        if q_t_ref is None:
            if interpret:
                zeros = jnp.zeros(bkv_x2_ref.shape[1:], bkv_x2_ref.dtype)
                bkv_x2_ref[0] = zeros
            else:
                bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32)
                zeros = jnp.zeros(bkv_x2_int32_ref.shape[1:], jnp.int32)
                bkv_x2_int32_ref[0] = zeros
        start_fetch_bkv(
            seq_idx=start_seq_idx, bkv_idx=cur_seq_start_bkv_idx, bkv_sem_idx=0
        )
        if q_t_ref is None:
            for _sl in range(1, num_slots):
                if interpret:
                    bkv_x2_ref[_sl] = zeros
                else:
                    bkv_x2_int32_ref[_sl] = zeros
        if custom_mask_ref is not None:
            start_fetch_mask(start_seq_idx, 0, 0, 0)

    pl.when(seq_idx == start_seq_idx)(ring_prologue if ring else prologue)

    @pl.when(jnp.logical_and(start_seq_idx <= seq_idx, seq_idx < end_seq_idx))
    def pipeline():
        process(static_q_len=static_q_len)

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(num_slots):
            wait_send_bo(bo_sem_idx=i)
            wait_update_kv_cache(bkv_sem_idx=i)

    ### ------- Kernel end ------- ###


def has_bank_conflicts(stride, distance=24, num_banks=32):
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False


def merge_kv(
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)

    head_dim = align_to(actual_head_dim, 128)
    kv = jnp.pad(
        jnp.concat([k, v], axis=-1).reshape(
            max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
        ),
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim,
    )
    return kv


def prepare_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    attention_sink: jax.Array | float | None = None,  # f32[actual_num_q_heads]
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = (
        jnp.pad(
            q.reshape(
                max_num_tokens,
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head // q_packing,
            q_packing,
            head_dim,
        )
        .swapaxes(0, 1)
    )
    kv = merge_kv(k, v)

    if attention_sink is not None:
        sink = jnp.asarray(attention_sink, dtype=jnp.float32)
        if sink.ndim == 0:
            sink = jnp.full((actual_num_q_heads,), sink, dtype=jnp.float32)
        sink = sink[: actual_num_kv_heads * actual_num_q_heads_per_kv_head]
        sink = sink.reshape(actual_num_kv_heads, actual_num_q_heads_per_kv_head)
        if num_q_heads_per_kv_head > actual_num_q_heads_per_kv_head:
            sink = jnp.pad(
                sink,
                ((0, 0), (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head)),
            )
        attention_sink = sink.reshape(actual_num_kv_heads, num_q_heads_per_kv_head, 1)
        attention_sink = jnp.repeat(attention_sink, 128, axis=-1)

    return q, kv, attention_sink


def prepare_outputs(
    out,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
):
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = out.shape
    actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
    return (
        out.swapaxes(0, 1)
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_q_packing * q_packing,
            head_dim,
        )[:, :, :actual_num_q_heads_per_kv_head, :actual_head_dim]
        .reshape(max_num_tokens, actual_num_q_heads, actual_head_dim)
    )


def prepare_head_packed_outputs(
    out,  # [1, max_num_tokens, num_q_heads // q_packing, q_packing, head_pack * d]
    head_pack: int,
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    """-> [max_num_tokens, actual_num_q_heads * d]: q head h keeps the lanes of
    packed head h // (actual_num_q_heads // head_pack), one slice per packed head."""
    num_kv_heads, max_num_tokens, rows, q_packing, head_dim = out.shape
    if head_pack < 0:
        hp = -head_pack
        d = head_dim // hp
        if (
            num_kv_heads != 1
            or rows * q_packing != actual_num_q_heads
            or head_dim != actual_head_dim
            or head_dim % hp
        ):
            raise ValueError(f"no padded head output for {out.shape=}, {head_pack=}")
        return out[0, :, :, :, :d].reshape(max_num_tokens, rows * q_packing * d)
    d = head_dim // head_pack
    rows_per_head = rows // head_pack
    if (
        num_kv_heads != 1
        or rows * q_packing != actual_num_q_heads
        or head_dim != actual_head_dim
        or rows % head_pack
    ):
        raise ValueError(f"no head-packed output for {out.shape=}, {head_pack=}")
    parts = []
    for g in range(head_pack):
        rows_g = slice(g * rows_per_head, (g + 1) * rows_per_head)
        lanes_g = slice(g * d, (g + 1) * d)
        parts.append(
            out[0, :, rows_g, :, lanes_g].reshape(
                max_num_tokens, rows_per_head * q_packing * d
            )
        )
    return jnp.concatenate(parts, axis=-1)


def prepare_kv_cache_fused(
    kv_cache_fused: jax.Array,  # [total_num_pages, page_size, actual_num_kv_heads * 2, actual_head_dim]
):
    (
        total_num_pages,
        page_size,
        actual_num_kv_heads_interleaved_per_packing,
        packing,
        actual_head_dim,
    ) = kv_cache_fused.shape
    # assert actual_num_kv_heads_interleaved_per_packing % 2 == 0
    head_dim = align_to(actual_head_dim, 128)

    kv_cache_fused_processed = jnp.pad(
        kv_cache_fused,
        (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return kv_cache_fused_processed


def prepare_updated_kv_cache_fused(
    kv_cache_fused,  # [total_num_pages, page_size, num_kv_heads_interleaved // kv_packing, kv_packing, head_dim]
    actual_num_kv_heads: int,
    actual_head_dim: int,
):
    """Extract actual KV cache from processed fused format."""
    (
        total_num_pages,
        page_size,
        num_kv_heads_interleaved_packed,
        kv_packing,
        head_dim,
    ) = kv_cache_fused.shape

    actual_num_kv_heads_interleaved = actual_num_kv_heads * 2
    return kv_cache_fused[:, :, :actual_num_kv_heads_interleaved]


def static_validate_inputs(
    queries,
    keys,
    values,
    kv_cache_fused,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    *,
    causal: int = 1,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_query_size: int = 1,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    skip_kv_mask: bool = False,
    attention_sink=None,
    _kv_pack_dense2: bool = False,
    _kv_pack_bd4: bool = False,
):
    """Validate inputs to the RPA kernel statically."""
    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(
            f"Expected {q.shape[0]=} to be equal to {k.shape[0]=} and {v.shape[0]=}"
        )
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(
            f"Expected {q.shape[2]=} to be equal to {k.shape[2]=} and {v.shape[2]=}"
        )

    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]

    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(
            f"Expected {actual_num_q_heads=} to be divisible by {actual_num_kv_heads=}."
        )

    if kv_cache_fused is not None:
        kv_cache_processed = prepare_kv_cache_fused(kv_cache_fused)
        _, phys_page_size, _, kv_packing, head_dim = kv_cache_processed.shape
        if _kv_pack_bd4:
            page_size = phys_page_size * 4
        elif _kv_pack_dense2:
            page_size = phys_page_size * 2
        else:
            page_size = phys_page_size

        if not jnp.issubdtype(kv_cache_fused.dtype, jnp.floating):
            raise ValueError(
                f"Expected {kv_cache_fused.dtype=} to be a floating point."
            )

    if not (
        jnp.int32
        == kv_lens.dtype
        == page_indices.dtype
        == cu_q_lens.dtype
        == cu_kv_lens.dtype
        == distribution.dtype
    ):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {cu_kv_lens.dtype=}, {distribution.dtype=}"
        )

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}"
        )

    max_num_seqs = kv_lens.shape[0]
    if cu_q_lens.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if cu_kv_lens.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {cu_kv_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3,):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if decode_query_size < 1:
        raise ValueError(f"{decode_query_size=} must be positive.")

    use_causal_mask = causal == 1
    if skip_kv_mask and use_causal_mask:
        raise ValueError("Cannot skip kv mask when using causal mask.")

    def _validate_block_sizes(block_sizes, prefix):
        if block_sizes is None:
            return
        bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes
        if not (bq_csz > 0 and bq_sz % bq_csz == 0):
            raise ValueError(
                f"{prefix} {bq_csz=} and {bq_sz=} must satisfy (0 < bq_csz and bq_sz"
                " % bq_csz == 0)."
            )
        if not (bkv_csz > 0 and bkv_sz % bkv_csz == 0):
            raise ValueError(
                f"{prefix} {bkv_csz=} and {bkv_sz=} must satisfy (0 < bkv_csz and"
                " bkv_sz % bkv_csz == 0)."
            )
        if kv_cache_fused is not None:
            if bkv_sz % page_size != 0:
                raise ValueError(
                    f"{prefix} {bkv_sz=} must be divisible by {page_size=}."
                )
            if bkv_csz % page_size != 0:
                raise ValueError(
                    f"{prefix} {bkv_csz=} must be divisible by {page_size=}."
                )

    _validate_block_sizes(d_block_sizes, "decode")
    _validate_block_sizes(p_block_sizes, "prefill")
    _validate_block_sizes(m_block_sizes, "mixed")

    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale
    del xai_temperature_len
    del skip_kv_mask
    del attention_sink


def dynamic_validate_inputs(
    queries,
    keys,
    values,
    kv_cache_fused,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    **kwargs,
):
    """Runtime validation of dynamic tensor values."""
    i, j, k = distribution[0], distribution[1], distribution[2]
    if not (i <= j <= k):
        raise ValueError(f"distribution must satisfy i <= j <= k, got {i}, {j}, {k}")
    max_num_seqs = kv_lens.shape[0]
    if k > max_num_seqs:
        raise ValueError(f"distribution[2]={k} exceeds {max_num_seqs=}")
    max_num_tokens = queries.shape[0]
    if cu_q_lens[k] > max_num_tokens:
        raise ValueError(f"cu_q_lens[{k}]={cu_q_lens[k]} exceeds {max_num_tokens=}")

    kv_cache_processed = prepare_kv_cache_fused(kv_cache_fused)
    total_num_pages = kv_cache_processed.shape[0]
    phys_page_size = kv_cache_processed.shape[1]
    page_size = (
        phys_page_size * 4
        if kwargs.get("_kv_pack_bd4", False)
        else (
            phys_page_size * 2
            if kwargs.get("_kv_pack_dense2", False)
            else phys_page_size
        )
    )
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    for seq_idx in range(int(k)):
        q_len = int(cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx])
        kv_len = int(kv_lens[seq_idx])
        if q_len <= 0:
            raise ValueError(f"Sequence {seq_idx}: q_len={q_len} must be > 0")
        if q_len > kv_len:
            raise ValueError(f"Sequence {seq_idx}: q_len={q_len} > kv_len={kv_len}")
        page_cnt = cdiv(kv_len, page_size)
        if page_cnt > pages_per_seq:
            raise ValueError(
                f"Sequence {seq_idx}: page_cnt={page_cnt} > pages_per_seq={pages_per_seq}"
            )
        for p in range(page_cnt):
            page_idx = int(page_indices[seq_idx * pages_per_seq + p])
            if page_idx < 0 or page_idx >= total_num_pages:
                raise ValueError(
                    f"Sequence {seq_idx}, page {p}: index={page_idx} "
                    f"out of [0, {total_num_pages})"
                )


def get_default_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    max_num_seqs,
    pages_per_seq,
    *,
    case: RpaCase = RpaCase.MIXED,
    vmem_limit_bytes: int | None = None,
    use_custom_mask: bool = False,
    sliding_window: int | None = None,
    decode_query_size: int = 1,
):
    """Get (bq_sz, bkv_sz, bq_csz, bkv_csz) by some heuristic formulas.

    DECODE tiles hold ``decode_query_size`` query rows per sequence so a
    fixed-width speculative verification runs on the decode schedule: one
    query tile and the largest KV tile VMEM allows.
    """
    tpu_version = get_tpu_version()

    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = next_power_of_2(align_to(actual_num_kv_heads * 2, kv_packing))
    head_dim = align_to(head_dim, 128)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    num_q_heads_per_kv_head = next_power_of_2(actual_num_q_heads_per_kv_head)

    max_q = next_power_of_2(max_num_tokens)
    max_kv = pages_per_seq * page_size

    min_bkv_sz_to_peak = (
        16 * 1024 * 1024 * kv_packing // 4 // head_dim // num_kv_heads_x2
    )

    match tpu_version:
        case 5 | 6:
            if case == RpaCase.DECODE:
                bq_sz = decode_query_size
                bkv_sz = (
                    min(min_bkv_sz_to_peak, max_kv)
                    if sliding_window is None
                    else sliding_window
                )
                bq_csz = decode_query_size
                bkv_csz = bkv_sz
            else:
                bq_sz = min(1024 // num_q_heads_per_kv_head, max_q // 2)
                bkv_sz = min(1024, max_kv)
                bq_csz = min(512 // num_q_heads_per_kv_head, max_q)
                bkv_csz = (
                    min(512, align_to(max_kv // 2, page_size))
                    if max_kv > 1
                    else page_size
                )
        case 7:
            if case == RpaCase.DECODE:
                bq_sz = decode_query_size
                bkv_sz = (
                    min(min_bkv_sz_to_peak, max_kv)
                    if sliding_window is None
                    else sliding_window
                )
                bq_csz = decode_query_size
                bkv_csz = bkv_sz
            else:
                bq_sz = min(2048 // num_q_heads_per_kv_head, max_q // 2)
                bkv_sz = min(2048, max_kv // 2)
                bq_csz = min(1024 // num_q_heads_per_kv_head, max_q // 2)
                bkv_csz = (
                    min(512, align_to(max_kv // 2, page_size))
                    if max_kv > 1
                    else page_size
                )
        case _:
            raise NotImplementedError(f"Unsupported {tpu_version=}.")

    bkv_alignment = max(page_size, kv_packing)
    bq_sz = max(1, bq_sz)
    bkv_sz = align_to(bkv_sz, bkv_alignment)
    bq_csz = max(1, bq_csz)
    bkv_csz = align_to(bkv_csz, bkv_alignment)

    # A sliding window much shorter than a KV compute chunk: one chunk of
    # query rows needs only bq_csz + sliding_window - 1 keys. The kernel skips
    # KV blocks and compute chunks left of the window of a query block's
    # first row, so size the blocks to let that skip remove nearly all masked
    # work:
    #   - one query compute chunk per query block: with more, the later
    #     chunks also compute the keys only the first row still sees;
    #   - a window-sized KV compute chunk, the skip's granularity;
    #   - the default KV compute chunk as the KV block, to fetch few keys
    #     left of the window without paying per-block overhead on tiny blocks.
    # Measured on tpu7x for gpt-oss (window 128, 8 query heads per KV head,
    # 1 to 8 KV heads): sliding-window layers run 1.18-1.36x faster than with
    # the full-attention blocks, and 256-row query blocks take up to 22%
    # longer than 128-row ones. Gated on v7 until v5/v6 are measured.
    if (
        tpu_version == 7
        and case != RpaCase.DECODE
        and sliding_window is not None
        and bq_csz + sliding_window - 1 <= bkv_csz
    ):
        bq_sz = bq_csz
        bkv_sz = min(bkv_sz, bkv_csz)
        bkv_csz = min(align_to(next_power_of_2(sliding_window), bkv_alignment), bkv_sz)

    # Reduce block sizes if VMEM estimate exceeds limit.
    # Use a fraction of vmem_limit as margin for shapes the estimate has not
    # been checked against.
    if vmem_limit_bytes is not None:
        # Give higher vmem budget for prefill and mixed; a larger decode KV
        # block measured slower per token.
        _budget_frac = 0.15 if case == RpaCase.DECODE else 0.60
        vmem_budget = int(vmem_limit_bytes * _budget_frac)
        # The DECODE query tile never shrinks below the static row count.
        min_bq_sz = decode_query_size if case == RpaCase.DECODE else 1
        min_prefill_bkv_sz = (
            bkv_alignment if case == RpaCase.DECODE else max(bkv_csz, bkv_alignment)
        )
        while bq_sz > min_bq_sz or bkv_sz > bkv_alignment:
            # Once prefill shrinks bkv_sz to a single compute sub-tile
            # (bkv_sz == bkv_csz), PV reuses dead QK^T registers in XLA's
            # schedule; tpu7x VMEM dumps measure 38.35 MiB (60.0%) where the
            # static sum estimates 42.50 MiB (66.4%).
            eff_budget = (
                int(vmem_limit_bytes * 0.67)
                if case != RpaCase.DECODE and bkv_sz <= min_prefill_bkv_sz
                else vmem_budget
            )
            vmem_est = get_vmem_estimate_bytes(
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                head_dim,
                bq_sz,
                bkv_sz,
                q_dtype,
                kv_dtype,
                use_custom_mask=use_custom_mask,
                bkv_csz=bkv_csz,
            )
            if vmem_est <= eff_budget:
                break
            # A query tile already at its floor cannot shrink; the KV tile must,
            # or the loop never terminates. On high-head-count PP prefill
            # (e.g. 128 Q / 16 KV heads), bkv_x2 at bkv_sz=2048 alone consumes
            # 34 MiB of a 38.4 MiB budget; shrink bkv_sz toward bkv_csz once
            # bq_sz reaches 64 instead of collapsing bq_sz to 8.
            if (
                bkv_sz > bq_sz * bkv_alignment
                or bq_sz <= min_bq_sz
                or (
                    case != RpaCase.DECODE
                    and bq_sz <= 64
                    and bkv_sz > min_prefill_bkv_sz
                )
            ):
                bkv_sz = max(bkv_sz // 2, bkv_alignment)
                bkv_csz = min(bkv_csz, bkv_sz)
            else:
                bq_sz = max(bq_sz // 2, min_bq_sz)
                bq_csz = bq_sz
            bkv_sz = align_to(bkv_sz, bkv_alignment)
            bkv_csz = align_to(bkv_csz, bkv_alignment)

    # Ensure csz <= sz.
    bkv_csz = min(bkv_csz, bkv_sz)
    bq_csz = min(bq_csz, bq_sz)

    # Ensure bkv_sz is evenly divisible by bkv_csz. If not, fall back to
    # bkv_csz = bkv_sz (disabling the nested attention loop for this config).
    if bkv_csz > 0 and bkv_sz % bkv_csz != 0:
        bkv_csz = bkv_sz

    logger.info(
        "get_default_block_sizes: case=%s, tpu_v=%d, bq=%d, bkv=%d, "
        "max_q=%d, max_kv=%d, heads=%d, head_dim=%d, page=%d, "
        "custom_mask=%s, vmem_limit=%s",
        case.symbol,
        tpu_version,
        bq_sz,
        bkv_sz,
        max_q,
        max_kv,
        actual_num_kv_heads,
        head_dim,
        page_size,
        use_custom_mask,
        vmem_limit_bytes,
    )

    return {
        "bq_sz": bq_sz,
        "bkv_sz": bkv_sz,
        "bq_csz": bq_csz,
        "bkv_csz": bkv_csz,
    }


def get_vmem_limit():
    try:
        # The kernel's own VMEM limit: the whole capacity, which is also what
        # the scoped-VMEM setting of the benchmark and CI flags grants XLA.
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    except Exception:
        vmem_limit_bytes = DEFAULT_VMEM_LIMIT_BYTES
    return vmem_limit_bytes


@functools.partial(
    jax.jit,
    static_argnames=(
        "causal",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "xai_temperature_len",
        "chunk_prefill_size",
        "decode_query_size",
        "d_block_sizes",
        "p_block_sizes",
        "m_block_sizes",
        "vmem_limit_bytes",
        "out_dtype",
        "skip_kv_mask",
        "disable_semaphore_checks",
        "debug_mode",
        "_sliding_window_chunk_skip",
        "_decode_window_aligned_kv",
        "_decode_zero_kv_tail",
        "_mixed_zero_kv_tail",
        "_decode_ring",
        "_decode_probe",
        "_prefill_opts",
        "_fused_out_pack",
        "_prepared_inputs",
        "_skip_decode",
        "_a3_out_layout",
        "_skip_mixed",
        "_decode_fast",
        "_decode_out_norm_eps",
        "_decode_gate_in_values",
        "_decode_tp16_prepared",
        "_kv_pack_dense2",
        "_kv_pack_bd4",
        "_interpret",
        "_prefill_topk_slices",
        "_prefill_topk_qstride",
        "_prefill_topk_in_kernel",
        "_prefill_topk_gqa_batch",
        "_prefill_topk_k_pair",
        "_prefill_topk_k1_bq",
        "_prefill_topk_sink_slices",
        "_prefill_topk_tail_local",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache_fused"),
)
def ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache_fused: jax.Array,  # [total_num_pages, page_size, actual_num_kv_heads * 2, actual_head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[flat]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    cu_kv_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    custom_mask: jax.Array | None,
    attention_sink: jax.Array | None = None,
    *,
    causal: int = 1,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_query_size: int = 1,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    out_dtype=None,
    skip_kv_mask: bool = False,
    disable_semaphore_checks: bool = True,
    debug_mode: bool = False,
    # Private: False restores computing KV chunks left of the sliding window.
    # The output is identical; it exists only so benchmarks and tests can A/B
    # the skip.
    _sliding_window_chunk_skip: bool = True,
    # Private: DECODE sliding-window KV blocks start at the window's first page
    # (see window_aligned_kv in the kernel). Same output; size d_block_sizes'
    # bkv to cdiv(window + page - 1, page) pages to fetch one block per sequence.
    _decode_window_aligned_kv: bool = False,
    # Private: DECODE zeroes the KV buffer rows past each sequence's length once
    # per block instead of selecting V per compute chunk. Same output.
    _decode_zero_kv_tail: bool = False,
    # Private: MIXED non-stationary_q nod2 zeroes KV buffer rows past kv_len (True by default).
    _mixed_zero_kv_tail: bool = True,
    # Private: DECODE keeps this many sequences' q/KV buffers (ring - 1 fetches
    # ahead) when each sequence is one KV block; 0 keeps the two-slot pipeline.
    _decode_ring: int = 0,
    # Private, benchmark removal arms only (wrong output): "nocompute",
    # "nodma" (no cache page DMAs), "novmask" (no V tail zeroing). DECODE only.
    _decode_probe: str | None = None,
    # Private: MIXED/PREFILL launches only, a subset of {"fold_scale" (softmax
    # scale folded into exp2), "mask_skip" (no mask on fully visible sub-blocks)}.
    _prefill_opts: frozenset = frozenset(),
    # Private: >1 means the queries pack this many kv heads into one head
    # (attention_pallas_rpa_base._pack_qkv_heads); the output is then
    # [max_num_tokens, actual_num_q_heads * actual_head_dim // _fused_out_pack].
    _fused_out_pack: int = 1,
    # Private: queries are already prepare_inputs' [kv_heads, T, Hq / packing,
    # packing, 128] and keys its merged kv [T, 2 * kv_heads / packing, packing,
    # 128]; values is None. No attention sink.
    _prepared_inputs: bool = False,
    # Private: skip the 0-sequence DECODE sub-launch on pure extend forwards.
    _skip_decode: bool = False,
    # Private: Stationary-Q MIXED launch writes unpacked bf16[max_num_tokens, 1024]
    # directly from its epilogue (requires _skip_decode=True and chunk_prefill_size=None).
    _a3_out_layout: bool = False,
    # Private: skip the 0-sequence MIXED sub-launch on pure decode forwards.
    _skip_mixed: bool = False,
    # Private: dedicated Ghostfish single-token DECODE fast kernel mode string.
    _decode_fast: str = "",
    # Private: optional fused post-RPA per-head RMSNorm eps & gate_in_values for decode.
    _decode_out_norm_eps: float | None = None,
    _decode_gate_in_values: bool = False,
    # Private: TP=16 prepared QKV_EPI path (8Q/1KV, d=64) returning 512 cols.
    _decode_tp16_prepared: bool = False,
    # Private: dense 2-tokens-per-u32 KV cache packing (phys page rows = page_size // 2).
    _kv_pack_dense2: bool = False,
    # Private: 4-tokens-per-u32 TP=16 d=64 KV cache packing (phys page rows = page_size // 4).
    _kv_pack_bd4: bool = False,
    # Private: Pallas interpret configuration (e.g. pltpu.InterpretParams(uninitialized_memory="nan")).
    _interpret: object = False,
    # Private: Opt-in 2-kernel LC Full top-k prefix slice selection (0 = disabled).
    _prefill_topk_slices: int = 0,
    _prefill_topk_qstride: int = 2,
    _prefill_topk_in_kernel: bool = False,
    _prefill_topk_gqa_batch: int = 1,
    _prefill_topk_k_pair: int = 1,
    _prefill_topk_k1_bq: int = 0,
    _prefill_topk_sink_slices: int = 0,
    _prefill_topk_tail_local: int = 0,
):
    """Ragged paged attention with fused KV cache.

    Args:
      queries: concatenated all sequences' queries.
      keys: concatenated all sequences' keys.
      values: concatenated all sequences' values.
      kv_cache_fused: paged KV cache with head interleaving [K1,V1,K2,V2,...].
      kv_lens: padded kv lengths.
      page_indices: flattened page indices look-up table.
      cu_q_lens: cumulative sum of effective query lengths.
      cu_kv_lens: cumulative sum of effective key/value lengths.
      distribution: (i, j, k) decode/prefill/mixed sequence ranges.
      custom_mask: custom attention mask for speculative decoding.
      attention_sink: per-head sink logits for streaming inference.
      causal: 1 for causal mask. 0 for non-causal: apply ``custom_mask`` when
        given, else attend bidirectionally (every query sees all ``kv_lens``
        keys, still subject to ``sliding_window``).
      sm_scale: softmax scale applied to Q@K^T.
      sliding_window: sliding window size.
      soft_cap: logit soft cap.
      mask_value: mask value for masked positions.
      q_scale: query scale.
      k_scale: key scale.
      v_scale: value scale.
      xai_temperature_len: length-based temperature for Grok-style models.
      chunk_prefill_size: chunk prefill size.
      decode_query_size: static query rows per DECODE sequence (1 = ordinary
        decode). Speculative verification passes its fixed width so the
        decode range runs with static loop bounds and decode KV tiles. Active
        DECODE rows whose length differs are processed by MIXED instead.
      d_block_sizes: block sizes for decode (bq_sz, bkv_sz, bq_csz, bkv_csz).
      p_block_sizes: block sizes for prefill.
      m_block_sizes: block sizes for mixed.
      vmem_limit_bytes: vmem limit for the pallas kernel.
      debug_mode: if true, skip DMAs and flash attention.

    Returns:
      (output, updated_kv_cache_fused)
    """
    # The full-layer host table carries a proof only on profitable batches.
    # Its shape selects one of two warmed programs. Counts remain live tensors.
    shared_pages = None
    if _decode_fast and distribution.shape != (3,):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            get_shared_group_size,
        )

        if get_shared_group_size(_decode_fast) > 0:
            n_groups = distribution.shape[0] - 3
            if n_groups <= 0 or kv_lens.shape[0] % n_groups != 0:
                raise ValueError("unexpected full-attention decode distribution shape")
            shared_pages = distribution[3:]
            distribution = distribution[:3]
    q, k, v = queries, keys, values

    if vmem_limit_bytes is None:
        vmem_limit_bytes = get_vmem_limit()

    if _prepared_inputs:
        assert (
            (v is None or _decode_gate_in_values)
            and attention_sink is None
            and custom_mask is None
        )
        # Validate the shapes prepare_inputs would have been given.
        num_kv, num_tokens, rows, packing, dim = q.shape
        k = v = jax.ShapeDtypeStruct((num_tokens, num_kv, dim), k.dtype)
        q = jax.ShapeDtypeStruct((num_tokens, num_kv * rows * packing, dim), q.dtype)
    static_validate_inputs(
        q,
        k,
        v,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        causal=causal,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        xai_temperature_len=xai_temperature_len,
        chunk_prefill_size=chunk_prefill_size,
        decode_query_size=decode_query_size,
        d_block_sizes=d_block_sizes,
        p_block_sizes=p_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        skip_kv_mask=skip_kv_mask,
        attention_sink=attention_sink,
        _kv_pack_dense2=_kv_pack_dense2,
        _kv_pack_bd4=_kv_pack_bd4,
    )
    if _kv_pack_bd4:
        _kv_pack_dense2 = False

    actual_num_q_heads = q.shape[1]
    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    _eff_out_dtype = (
        (jnp.float32 if q.dtype == jnp.float32 else jnp.bfloat16)
        if out_dtype is None
        else out_dtype
    )
    _need_fused_post = _decode_out_norm_eps is not None or _decode_gate_in_values
    _use_decode_fast = False
    if _decode_fast and not _skip_decode and not debug_mode:
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            can_use_decode_fast,
            run_decode_fast_rpa,
        )

        _use_decode_fast = can_use_decode_fast(
            mode_str=_decode_fast,
            q=q,
            kv_cache_fused=kv_cache_fused,
            out_dtype=_eff_out_dtype,
            decode_query_size=decode_query_size,
            custom_mask=custom_mask,
            attention_sink=attention_sink,
            soft_cap=soft_cap,
            xai_temperature_len=xai_temperature_len,
            q_scale=q_scale,
            actual_num_q_heads=actual_num_q_heads,
            actual_num_kv_heads=actual_num_kv_heads,
            actual_head_dim=actual_head_dim,
            sliding_window=sliding_window,
            pages_per_seq=page_indices.shape[0] // max(kv_lens.shape[0], 1),
            pure_unpacked_decode=(
                _skip_mixed
                and chunk_prefill_size is None
                and not _prepared_inputs
                and _fused_out_pack == 1
            ),
            kv_pack_dense2=_kv_pack_dense2,
            need_fused_post=_need_fused_post,
            kv_pack_bd4=_kv_pack_bd4,
        )
        if shared_pages is not None and (
            sliding_window is not None
            or not _skip_mixed
            or chunk_prefill_size is not None
            or kv_lens.shape[0] % 8 != 0
            or q.shape[0] < kv_lens.shape[0]
            or kv_lens.shape[0] < 8
        ):
            shared_pages = None
        if _use_decode_fast and _skip_mixed and chunk_prefill_size is None:
            return run_decode_fast_rpa(
                queries,
                keys,
                values,
                kv_cache_fused,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                k_scale=k_scale,
                v_scale=v_scale,
                mode_str=_decode_fast,
                decode_query_size=decode_query_size,
                fused_out_pack=_fused_out_pack,
                prepared_inputs=_prepared_inputs,
                gate=values if _decode_gate_in_values else None,
                out_norm_eps=_decode_out_norm_eps,
                vmem_limit_bytes=vmem_limit_bytes,
                disable_semaphore_checks=disable_semaphore_checks,
                probe=_decode_probe,
                shared_pages=shared_pages,
                kv_pack_dense2=_kv_pack_dense2,
                kv_pack_bd4=_kv_pack_bd4,
                force_native_tp16=_decode_tp16_prepared,
            )

    if _prepared_inputs:
        q, kv = queries, keys
    else:
        q, kv, attention_sink = prepare_inputs(q, k, v, attention_sink)
    kv_cache_fused_processed = prepare_kv_cache_fused(kv_cache_fused)

    (
        _,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    phys_page_size = kv_cache_fused_processed.shape[1]
    page_size = (
        phys_page_size * 4
        if _kv_pack_bd4
        else (phys_page_size * 2 if _kv_pack_dense2 else phys_page_size)
    )
    num_kv_heads_x2_per_kv_packing = kv_cache_fused_processed.shape[2]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing
    if out_dtype is None:
        out_dtype = jnp.float32 if q.dtype == jnp.float32 else jnp.bfloat16

    # Prepare custom mask.
    if custom_mask is not None:
        if custom_mask.dtype == jnp.bool_:
            custom_mask = custom_mask.astype(jnp.int32)
        custom_mask = jnp.repeat(
            jnp.expand_dims(custom_mask, axis=1), repeats=head_dim, axis=1
        )

        # Prepare cu_seq_mask_lens for custom mask.
        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        seq_mask_lens = kv_lens * q_lens
        cu_seq_mask_lens = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_mask_lens)]
        )
    else:
        cu_seq_mask_lens = jnp.array([0])

    # Scalar prefetch init values.
    init_sem_ids = jnp.zeros((3,), jnp.int32)

    use_causal_mask = causal == 1
    use_custom_mask = custom_mask is not None
    if use_custom_mask and use_causal_mask:
        raise ValueError("custom_mask requires causal=0.")
    tpu_version = 0 if _interpret else get_tpu_version()

    def run_rpa_kernel(
        q,
        kv_cache,
        *,
        bq_sz,
        bkv_sz,
        bq_csz,
        bkv_csz,
        static_q_len=None,
        case: RpaCase = RpaCase.MIXED,
    ):
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache
            (
                pl.BlockSpec(memory_space=pltpu.HBM)
                if custom_mask is not None
                else None
            ),  # custom_mask
            pl.BlockSpec(memory_space=pltpu.HBM),  # zero_mask
            (
                pl.BlockSpec(memory_space=pltpu.VMEM)
                if attention_sink is not None
                else None
            ),  # attention_sink
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        bkv_stride = num_kv_heads_x2_per_kv_packing
        if has_bank_conflicts(bkv_stride):
            bkv_stride += 1

        # The ring needs exactly one KV block per sequence: the whole page table
        # (full attention) or a window-aligned block covering the window.
        one_block_per_seq = (
            bkv_sz >= pages_per_seq * page_size
            if sliding_window is None
            else _decode_window_aligned_kv
            and bkv_sz >= cdiv(sliding_window + page_size - 1, page_size) * page_size
        )
        ring = (
            _decode_ring
            if case == RpaCase.DECODE
            and static_q_len == 1
            and custom_mask is None
            and one_block_per_seq
            else 0
        )
        prefill_opts = frozenset()
        if case != RpaCase.DECODE:
            prefill_opts = frozenset(_prefill_opts)
            if (
                soft_cap is not None
                or xai_temperature_len is not None
                or attention_sink is not None
            ):
                prefill_opts -= {"fold_scale"}
            if custom_mask is not None:
                prefill_opts -= {"mask_skip", "visible_kv_tile"}

        is_tp16_sq = (
            actual_num_kv_heads == 1
            and num_q_heads_per_kv_head == 8
            and q_packing == 2
            and head_dim == 128
            and bool(prefill_opts & _STATIONARY_Q_V2_OPTS)
            and ("tp16_sq" in prefill_opts or "w9" in prefill_opts)
        )
        use_stationary_q = (
            case != RpaCase.DECODE
            and any(
                o in ("stationary_q_m0", "stationary_q_pilot", *_STATIONARY_Q_V2_OPTS)
                for o in prefill_opts
            )
            and actual_num_kv_heads == 1
            and (num_q_heads_per_kv_head == 16 or is_tp16_sq)
            and q_packing == 2
            and head_dim == 128
            and q.dtype == jnp.bfloat16
            and kv_cache.dtype == jnp.float8_e4m3fn
            and out_dtype == jnp.bfloat16
            and custom_mask is None
            and attention_sink is None
            and soft_cap is None
            and xai_temperature_len is None
            and q_scale is None
            and bkv_sz % 512 == 0
            and bq_sz % 256 == 0
        )
        sq_kv_heads = 1 if is_tp16_sq else 2
        sq_q_heads = 8 * sq_kv_heads
        a3_out_cols = 512 * sq_kv_heads
        use_a3_out = (
            use_stationary_q
            and _a3_out_layout
            and _skip_decode
            and chunk_prefill_size is None
            and max_num_tokens % 16 == 0
        )
        if use_stationary_q:
            prefill_opts -= {"fold_scale", "mask_skip"}
            if use_a3_out:
                prefill_opts = prefill_opts | {"a3_out_layout"}
            else:
                prefill_opts -= {"a3_out_layout"}
        else:
            prefill_opts -= {
                "stationary_q_m0",
                "stationary_q_pilot",
                "bkv_full_unroll",
                "a3_out_layout",
                "tq_loop",
                "tq_unroll",
                "sw2048_fuse",
                "fp8_kv",
                "fp8_qk",
                "fp8_qk_dyn",
                "fp8_qkpv",
                "fp8_qkpv_s1",
                "fp8_qkpv_s0",
                "fp8_qkpv_norm",
                "gqa2_vreg",
                "gqa4_vreg",
                "vreg_acc",
                "vreg_acc1",
                "mk128",
                "gqa8",
                "sub8_l",
                "bf16_pop",
                "la2",
                "unroll_up",
                "sw_short_v2",
                "poison_vmem_nan",
                "tp16_sq",
                "w9",
                "sw_short_fuse",
                "mrb1",
                "mrb3",
                "mrb4",
                "fp8_qkpv_fast",
                "f32_exp",
                "prefix_fast",
                "staged_qo",
                "sw_unalign_fuse",
                "sw_tail_clamp",
                *_STATIONARY_Q_V2_OPTS,
            }
        use_stationary_q_v2 = bool(
            use_stationary_q and (prefill_opts & _STATIONARY_Q_V2_OPTS)
        )
        if (_kv_pack_dense2 or _kv_pack_bd4) and use_stationary_q and not use_stationary_q_v2:
            raise NotImplementedError(
                "_kv_pack_dense2/_kv_pack_bd4 require stationary_q_v2 when stationary_q is enabled."
            )

        num_slots = (
            3
            if (case != RpaCase.DECODE and "mixed_s3" in prefill_opts)
            else (ring or 2)
        )

        bkv_phys_rows = (
            bkv_sz // 4
            if _kv_pack_bd4
            else (bkv_sz // 2 if _kv_pack_dense2 else bkv_sz)
        )
        bkv_double_buf = pltpu.VMEM(
            (num_slots, bkv_phys_rows, bkv_stride, *kv_cache.shape[3:]),
            kv_cache.dtype,
        )

        allow_sw_fuse_v2 = (sliding_window == 2049) or (
            sliding_window == 2048 and "sw2048_fuse" in prefill_opts
        )
        use_fuse_qe_v2 = bool(
            use_stationary_q_v2
            and allow_sw_fuse_v2
            and use_causal_mask
            and bq_sz == 256
            and (
                "stationary_q_m0_ring_fuse" in prefill_opts
                or "stationary_q_m0_f32e_fuse" in prefill_opts
            )
        )
        use_staged_qo = bool(
            use_stationary_q_v2
            and "staged_qo" in prefill_opts
            and not use_fuse_qe_v2
            and bq_sz >= 512
            and bq_sz % 256 == 0
            and not use_custom_mask
        )
        qo_stage_sz = 256 if use_staged_qo else bq_sz
        bq_double_buf = pltpu.VMEM(
            (num_slots, actual_num_kv_heads, qo_stage_sz, *q.shape[2:]),
            q.dtype,
        )
        bo_double_buf = (
            pltpu.VMEM((num_slots, qo_stage_sz + 16, a3_out_cols), q.dtype)
            if use_a3_out
            else (
                pltpu.VMEM(
                    (num_slots, actual_num_kv_heads, qo_stage_sz, *q.shape[2:]),
                    q.dtype,
                )
                if (use_fuse_qe_v2 or use_staged_qo)
                else bq_double_buf
            )
        )

        if not use_custom_mask:
            bkvmask_double_buf = None
        else:
            bkvmask_double_buf = pltpu.VMEM(
                (2, bq_sz, bkv_sz, head_dim),
                jnp.int32,
            )

        if use_stationary_q_v2:
            max_kv_tokens_v2 = pages_per_seq * page_size
            num_s_v2 = bkv_sz // 256
            if sliding_window is not None:
                min_ring_blks = cdiv(bq_sz + sliding_window - 1, bkv_sz) + 1
                ring_blks_v2 = max(4, next_power_of_2(min_ring_blks))
            elif max_kv_tokens_v2 <= 32768:
                ring_blks_v2 = max(2, next_power_of_2(cdiv(max_kv_tokens_v2, bkv_sz)))
            else:
                ring_blks_v2 = 1
            ring_slices_v2 = ring_blks_v2 * num_s_v2
            use_mk128_v2 = "mk128" in prefill_opts
            use_fp8_qk_dyn_v2 = "fp8_qk_dyn" in prefill_opts
            use_fp8_qkpv_dyn_v2 = (
                "fp8_qkpv_dyn" in prefill_opts or "fp8_qkpv_fast" in prefill_opts
            )
            use_fp8_qkpv_v2 = (
                ("fp8_qkpv" in prefill_opts)
                or ("fp8_qkpv_s1" in prefill_opts)
                or ("fp8_qkpv_s0" in prefill_opts)
                or ("fp8_qkpv_norm" in prefill_opts)
            )
            use_fp8_qk_v2 = (
                use_fp8_qkpv_dyn_v2
                or use_fp8_qkpv_v2
                or use_fp8_qk_dyn_v2
                or ("fp8_qk" in prefill_opts)
            )
            use_fp8_kv_v2 = use_fp8_qk_v2 or ("fp8_kv" in prefill_opts)
            use_vt_fp8_v2 = (
                use_fp8_qkpv_dyn_v2 or use_fp8_qkpv_v2 or ("fp8_kv" in prefill_opts)
            )
            l_scratch = pltpu.VMEM((sq_q_heads, 8, bq_sz), jnp.float32)
            m_scratch = None
            acc_scratch = pltpu.VMEM((sq_q_heads, 64, bq_sz), jnp.float32)
            mask_shape_v2 = (2, 128, 256) if use_mk128_v2 else (1, 256, 256)
            extra_scratches = [
                pltpu.VMEM(
                    (sq_q_heads, 64, bq_sz),
                    jnp.float8_e4m3fn if use_fp8_qk_v2 else jnp.bfloat16,
                ),
                pltpu.VMEM(
                    (sq_kv_heads, 2 * ring_slices_v2, 128, 64)
                    if use_mk128_v2
                    else (sq_kv_heads, ring_slices_v2, 256, 128),
                    jnp.float8_e4m3fn if use_fp8_kv_v2 else jnp.bfloat16,
                ),
                pltpu.VMEM(
                    (sq_kv_heads, 2 * ring_slices_v2, 64, 128)
                    if use_mk128_v2
                    else (sq_kv_heads, ring_slices_v2, 64, 256),
                    jnp.float8_e4m3fn if use_vt_fp8_v2 else jnp.bfloat16,
                ),
                (pltpu.VMEM(mask_shape_v2, jnp.bfloat16) if use_causal_mask else None),
                (
                    pltpu.VMEM(mask_shape_v2, jnp.bfloat16)
                    if (allow_sw_fuse_v2 and use_causal_mask)
                    else None
                ),
                (
                    pltpu.VMEM((16, a3_out_cols), jnp.bfloat16)
                    if use_a3_out
                    else None
                ),
                (
                    pltpu.VMEM((sq_q_heads, 8, bq_sz), jnp.float32)
                    if use_fp8_qk_dyn_v2
                    else None
                ),
            ]
        elif use_stationary_q:
            l_scratch = pltpu.VMEM((16, 8, bq_sz), jnp.float32)
            m_scratch = (
                None
                if "stationary_q_m0" in prefill_opts
                else pltpu.VMEM((16, 8, bq_sz), jnp.float32)
            )
            acc_scratch = pltpu.VMEM((16, 64, bq_sz), jnp.float32)
            extra_scratches = [
                pltpu.VMEM((16, 64, bq_sz), jnp.bfloat16),
                pltpu.VMEM((2, bkv_sz // 256, 256, 128), jnp.bfloat16),
                pltpu.VMEM((2, bkv_sz // 256, 64, 256), jnp.bfloat16),
                *(
                    [None, None, pltpu.VMEM((16, 1024), jnp.bfloat16)]
                    if use_a3_out
                    else []
                ),
            ]
        else:
            l_scratch = pltpu.VMEM(
                (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
                out_dtype,
            )
            m_scratch = l_scratch
            acc_scratch = pltpu.VMEM(
                (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
                out_dtype,
            )
            extra_scratches = []

        if _kv_pack_dense2 or _kv_pack_bd4:
            if len(extra_scratches) < 7:
                extra_scratches = list(extra_scratches) + [None] * (
                    7 - len(extra_scratches)
                )
            extra_scratches.append(
                pltpu.VMEM(
                    (num_slots, min(bkv_sz, 4096), 1, *kv_cache.shape[3:]),
                    kv_cache.dtype,
                )
            )

        scratch_shapes = [
            bkvmask_double_buf,
            bkv_double_buf,
            bq_double_buf,
            bo_double_buf,
            pltpu.SemaphoreType.DMA(
                (6 if (_kv_pack_dense2 or _kv_pack_bd4) else 5, num_slots)
            ),
            l_scratch,
            m_scratch,
            acc_scratch,
            *extra_scratches,
        ]

        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            cu_seq_mask_lens,
            distribution,
            jnp.zeros((4,), jnp.int32) if use_stationary_q_v2 else init_sem_ids,
            jnp.full((2 * num_slots,), -1, jnp.int32),  # bo_ids
            jnp.full((3 * num_slots,), -1, jnp.int32),  # bkv_update_ids
        )

        scope_name = (
            f"RPA{case.symbol}-p_{page_size}-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}"
        )
        if sliding_window is not None:
            scope_name += f"-sw_{sliding_window}"
            if not _sliding_window_chunk_skip:
                scope_name += "-noskip"
        window_aligned_kv = (
            _decode_window_aligned_kv
            and sliding_window is not None
            and case == RpaCase.DECODE
        )
        probe = (
            _decode_probe
            if (case == RpaCase.DECODE or _decode_probe in ("nocompute", "nodma"))
            else None
        )
        if window_aligned_kv:
            scope_name += "-wa"
        zero_kv_tail = (
            (
                (_decode_zero_kv_tail and case == RpaCase.DECODE)
                or (
                    _mixed_zero_kv_tail
                    and case == RpaCase.MIXED
                    and not use_stationary_q
                    and not _kv_pack_dense2
                    and not _kv_pack_bd4
                )
            )
            and bkv_stride == 1
            and custom_mask is None
        )
        if zero_kv_tail:
            scope_name += "-zt"
        if ring:
            scope_name += f"-ring{ring}"
        if probe:
            scope_name += f"-probe_{probe}"
        for opt in sorted(prefill_opts):
            scope_name += f"-{opt}"
        if _kv_pack_bd4:
            scope_name += "-bd4"
        elif _kv_pack_dense2:
            scope_name += "-d2"

        eff_bq_topk = bq_sz if static_q_len is None else min(bq_sz, static_q_len)
        num_h_per_bkv = 2 * (bkv_sz // 256)
        w_q_cap = (
            128
            if _prefill_topk_qstride == 16
            else (256 if _prefill_topk_qstride in (4, 8) else 512)
        )
        w_q_topk = (
            min(w_q_cap, eff_bq_topk // _prefill_topk_qstride)
            if (
                _prefill_topk_qstride in (1, 2, 4, 8, 16)
                and eff_bq_topk >= 256
                and eff_bq_topk % 256 == 0
            )
            else 0
        )
        q_grp_sz_topk = w_q_topk * _prefill_topk_qstride if w_q_topk > 0 else 1024
        max_q_grps_topk = cdiv(max_num_tokens, max(q_grp_sz_topk, 1)) + max_num_seqs
        max_bkv_topk = cdiv(pages_per_seq * page_size, bkv_sz)
        smem_ints_topk = (
            max_q_grps_topk
            * max_bkv_topk
            * sq_kv_heads
            * max(_prefill_topk_slices // 4, 1)
        )
        use_prefix_topk = bool(
            use_stationary_q_v2
            and "prefix_fast" in prefill_opts
            and sliding_window is None
            and use_causal_mask
            and use_mk128_v2
            and not use_fp8_qk_dyn_v2
            and not debug_mode
            and eff_bq_topk >= 256
            and eff_bq_topk % 256 == 0
            and bkv_sz >= 1024
            and (
                num_h_per_bkv <= 128
                or (
                    num_h_per_bkv <= 256
                    and _prefill_topk_in_kernel
                    and _prefill_topk_gqa_batch > 1
                    and 8 % _prefill_topk_gqa_batch == 0
                )
            )
            and (pages_per_seq * page_size) > bkv_sz
            and _prefill_topk_slices > 0
            and _prefill_topk_slices < num_h_per_bkv
            and _prefill_topk_slices % 4 == 0
            and _prefill_topk_qstride in (1, 2, 4, 8, 16)
            and w_q_topk in (128, 256, 512)
            and probe is None
            and smem_ints_topk <= 131072
        )

        extra_sp_count = 0
        prefix_topk_slices_eff = 0
        prefix_topk_sink_slices_eff = 0
        prefix_topk_tail_local_eff = 0
        if use_prefix_topk:
            topk_sink_slices_eff = (
                _prefill_topk_sink_slices
                if (0 < _prefill_topk_sink_slices <= _prefill_topk_slices)
                else 0
            )
            topk_tail_local_eff = (
                int(_prefill_topk_tail_local)
                if _prefill_topk_tail_local > 0
                else 0
            )
            q_lens_topk = cu_q_lens[1:] - cu_q_lens[:-1]
            qgrps_per_seq = (q_lens_topk + q_grp_sz_topk - 1) // q_grp_sz_topk
            cu_qgrps = jnp.concatenate(
                [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(qgrps_per_seq)]
            )
            w_red_topk = 256 if w_q_topk == 512 else w_q_topk
            k1_exp_scale = float(
                sm_scale * (k_scale if k_scale is not None else 1.0) * math.log2(math.e)
            )
            qgrp_aligned_max_q = (max_num_tokens // q_grp_sz_topk) * q_grp_sz_topk
            bq_sz_k1 = (
                max(bq_sz, min(_prefill_topk_k1_bq, qgrp_aligned_max_q))
                if (_prefill_topk_k1_bq > 0 and qgrp_aligned_max_q >= q_grp_sz_topk)
                else bq_sz
            )
            eff_bq_k1 = bq_sz_k1 if static_q_len is None else min(bq_sz_k1, static_q_len)
            k1_n_qgrps_per_bq = max(1, eff_bq_k1 // q_grp_sz_topk)
            topk_in_kernel_eff = bool(_prefill_topk_in_kernel)
            sc_out_dtype_k1 = jnp.int32 if topk_in_kernel_eff else jnp.float32
            k1_q_stage_rows = (
                128 * _prefill_topk_qstride
                if _prefill_topk_qstride in (4, 8, 16)
                else 256
            )
            q_k1_bf16_n = (
                1 if _prefill_topk_qstride in (4, 8, 16) else k1_n_qgrps_per_bq
            )
            k1_scratches = [
                pltpu.VMEM((2, 1, k1_q_stage_rows, *q.shape[2:]), q.dtype),
                pltpu.VMEM(
                    (2, bkv_phys_rows, bkv_stride, *kv_cache.shape[3:]),
                    kv_cache.dtype,
                ),
                pltpu.SemaphoreType.DMA((3, 2)),
                pltpu.VMEM(
                    (sq_kv_heads, num_h_per_bkv, 128, 64), kv_cache.dtype
                ),
                pltpu.VMEM(
                    (q_k1_bf16_n, sq_q_heads, 64, w_q_topk), jnp.bfloat16
                ),
                pltpu.VMEM(
                    (k1_n_qgrps_per_bq, sq_q_heads, 64, w_q_topk), kv_cache.dtype
                ),
                pltpu.VMEM((k1_n_qgrps_per_bq, sq_kv_heads, 8, 128), jnp.float32),
                pltpu.VMEM((16, 8, w_red_topk), jnp.bfloat16),
                pltpu.VMEM((2, k1_n_qgrps_per_bq, 8, 128), sc_out_dtype_k1),
            ]
            k1_kernel = pl.pallas_call(
                functools.partial(
                    _prefix_score_kernel,
                    bkv_sz=bkv_sz,
                    bq_sz=bq_sz_k1,
                    q_grp_sz=q_grp_sz_topk,
                    qstride=_prefill_topk_qstride,
                    exp_scale=k1_exp_scale,
                    case=case,
                    static_q_len=static_q_len,
                    kv_pack_dense2=_kv_pack_dense2,
                    kv_pack_bd4=_kv_pack_bd4,
                    interpret=bool(_interpret),
                    topk_slices=_prefill_topk_slices,
                    topk_in_kernel=topk_in_kernel_eff,
                    topk_gqa_batch=_prefill_topk_gqa_batch,
                    topk_k_pair=_prefill_topk_k_pair,
                    topk_sink_slices=topk_sink_slices_eff,
                    topk_tail_local=topk_tail_local_eff,
                    stage2_bq_sz=eff_bq_topk,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=6,
                    in_specs=[
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                    ],
                    out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
                    grid=(1,),
                    scratch_shapes=k1_scratches,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                    **(_semaphore_kwargs(disable_semaphore_checks)),
                ),
                out_shape=(
                    [
                        pltpu.HBM(
                            shape=(max_q_grps_topk, max_bkv_topk, 8, 128),
                            dtype=sc_out_dtype_k1,
                        )
                    ]
                    if tpu_version >= 7
                    else [
                        jax.ShapeDtypeStruct(
                            shape=(max_q_grps_topk, max_bkv_topk, 8, 128),
                            dtype=sc_out_dtype_k1,
                        )
                    ]
                ),
                interpret=_interpret,
                name=f"{scope_name}-ptk_score",
            )
            k1_sp = (kv_lens, page_indices, cu_q_lens, cu_kv_lens, cu_qgrps, distribution)
            if tpu_version >= 7:
                (sc_hbm_out,) = k1_kernel(
                    *k1_sp,
                    pltpu.with_memory_space_constraint(q, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv_cache, pltpu.HBM),
                )
            else:
                (sc_hbm_out,) = k1_kernel(*k1_sp, q, kv_cache)

            if topk_in_kernel_eff:
                prefix_topk_packed = sc_hbm_out[
                    :, :, :sq_kv_heads, : (_prefill_topk_slices // 4)
                ].reshape(-1)
            else:
                sc_active = sc_hbm_out[:, :, :sq_kv_heads, :num_h_per_bkv]
                _, top_idx = lax.top_k(sc_active, _prefill_topk_slices, is_stable=False)
                top_u8 = top_idx.astype(jnp.uint32).reshape(
                    max_q_grps_topk,
                    max_bkv_topk,
                    sq_kv_heads,
                    _prefill_topk_slices // 4,
                    4,
                )
                prefix_topk_packed = (
                    (
                        top_u8[..., 0]
                        | (top_u8[..., 1] << jnp.uint32(8))
                        | (top_u8[..., 2] << jnp.uint32(16))
                        | (top_u8[..., 3] << jnp.uint32(24))
                    )
                    .astype(jnp.int32)
                    .reshape(-1)
                )
            scalar_prefetches = scalar_prefetches + (cu_qgrps, prefix_topk_packed)
            extra_sp_count = 2
            prefix_topk_slices_eff = _prefill_topk_slices
            prefix_topk_sink_slices_eff = topk_sink_slices_eff
            prefix_topk_tail_local_eff = topk_tail_local_eff
            scope_name += (
                f"-ptk{_prefill_topk_slices}_qs{_prefill_topk_qstride}"
                + (
                    f"_ik_g{_prefill_topk_gqa_batch}_kp{_prefill_topk_k_pair}"
                    + (f"_k1b{bq_sz_k1}" if _prefill_topk_k1_bq > 0 else "")
                    if topk_in_kernel_eff
                    else ""
                )
                + (
                    f"_sk{topk_sink_slices_eff}"
                    if topk_sink_slices_eff > 0
                    else ""
                )
                + (
                    f"_tl{topk_tail_local_eff}"
                    if topk_tail_local_eff > 0
                    else ""
                )
            )

        extra_compiler_kwargs = (
            {"flags": {"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True}}
            if use_stationary_q
            else {}
        )

        o_out_shape = (max_num_tokens, a3_out_cols) if use_a3_out else q.shape
        q_arg_idx = len(scalar_prefetches)
        kvc_arg_idx = q_arg_idx + 2
        out_aliases = (
            {kvc_arg_idx: 1}
            if use_a3_out
            else {
                q_arg_idx: 0,  # q -> output
                kvc_arg_idx: 1,  # kv_cache -> updated_kv_cache
            }
        )
        valid_lane_dim = (
            head_dim // (-_fused_out_pack)
            if (_prepared_inputs and _fused_out_pack < 0)
            else (actual_head_dim if not _prepared_inputs else head_dim)
        )

        kernel = pl.pallas_call(
            functools.partial(
                _ragged_paged_attention_kernel,
                causal=use_causal_mask,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=mask_value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                xai_temperature_len=xai_temperature_len,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_sz=bkv_sz,
                bq_csz=bq_csz,
                bkv_csz=bkv_csz,
                case=case,
                skip_kv_mask=skip_kv_mask,
                sliding_window_chunk_skip=_sliding_window_chunk_skip,
                window_aligned_kv=window_aligned_kv,
                zero_kv_tail=zero_kv_tail,
                ring=ring,
                probe=probe,
                prefill_opts=prefill_opts,
                debug_mode=debug_mode,
                kv_pack_dense2=_kv_pack_dense2,
                kv_pack_bd4=_kv_pack_bd4,
                interpret=bool(_interpret),
                valid_lane_dim=valid_lane_dim,
                extra_sp_count=extra_sp_count,
                prefix_topk_slices=prefix_topk_slices_eff,
                prefix_topk_sink_slices=prefix_topk_sink_slices_eff,
                prefix_q_grp_sz=q_grp_sz_topk,
                prefix_max_bkv=max_bkv_topk,
                prefix_topk_tail_local=prefix_topk_tail_local_eff,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(1,),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
                **extra_compiler_kwargs,
                **(_semaphore_kwargs(disable_semaphore_checks)),
            ),
            out_shape=(
                [
                    pltpu.HBM(shape=o_out_shape, dtype=q.dtype),
                    pltpu.HBM(shape=kv_cache.shape, dtype=kv_cache.dtype),
                ]
                if tpu_version >= 7
                else [
                    jax.ShapeDtypeStruct(shape=o_out_shape, dtype=q.dtype),
                    jax.ShapeDtypeStruct(shape=kv_cache.shape, dtype=kv_cache.dtype),
                ]
            ),
            input_output_aliases=out_aliases,
            interpret=_interpret,
            name=scope_name,
        )

        zero_mask = jnp.zeros((bkv_sz, head_dim), dtype=jnp.int32)

        if tpu_version >= 7:

            @jax.jit
            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(
                    *scalar_prefetches,
                    pltpu.with_memory_space_constraint(q, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv_cache, pltpu.HBM),
                    (
                        pltpu.with_memory_space_constraint(custom_mask, pltpu.HBM)
                        if custom_mask is not None
                        else custom_mask
                    ),
                    zero_mask,
                    attention_sink,
                )

        else:

            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(
                    *scalar_prefetches,
                    q,
                    kv,
                    kv_cache,
                    custom_mask,
                    zero_mask,
                    attention_sink,
                )

        return run(scalar_prefetches, q, kv, kv_cache)

    def _prepare_block_sizes(block_sizes, case):
        if block_sizes is None:
            return get_default_block_sizes(
                q.dtype,
                kv_cache_fused_processed.dtype,
                actual_num_q_heads,
                actual_num_kv_heads,
                head_dim,
                page_size,
                max_num_tokens,
                max_num_seqs,
                pages_per_seq,
                case=case,
                vmem_limit_bytes=vmem_limit_bytes,
                use_custom_mask=use_custom_mask,
                sliding_window=sliding_window,
                decode_query_size=decode_query_size,
            )

        return {
            "bq_sz": block_sizes[0],
            "bkv_sz": block_sizes[1],
            "bq_csz": block_sizes[2],
            "bkv_csz": block_sizes[3],
        }

    if decode_query_size > 1:
        # Static DECODE loop bounds are valid only when every active DECODE row
        # has exactly decode_query_size queries. Otherwise the whole batch goes
        # to MIXED, the only dynamic-q range, without a host sync or recompile.
        seq_ids = jnp.arange(max_num_seqs)
        is_decode = seq_ids < distribution[0]
        uniform = jnp.all(~is_decode | (jnp.diff(cu_q_lens) == decode_query_size))
        distribution = jnp.where(
            uniform, distribution, jnp.array([0, 0, 1], jnp.int32) * distribution[2]
        )

    # When chunk_prefill_size is None, PREFILL pallas_call is skipped.
    # Remap prefill sequences to MIXED so they are still processed.
    if chunk_prefill_size is None:
        distribution = distribution.at[1].set(distribution[0])

    # Decode-only
    if not _skip_decode:
        if _use_decode_fast:
            o_dec_3d, kv_cache_fused_processed = run_decode_fast_rpa(
                queries,
                keys,
                values,
                kv_cache_fused_processed,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                k_scale=k_scale,
                v_scale=v_scale,
                mode_str=_decode_fast,
                decode_query_size=decode_query_size,
                fused_out_pack=1,
                prepared_inputs=_prepared_inputs,
                vmem_limit_bytes=vmem_limit_bytes,
                disable_semaphore_checks=disable_semaphore_checks,
                probe=_decode_probe,
                kv_pack_dense2=_kv_pack_dense2,
                kv_pack_bd4=_kv_pack_bd4,
            )
            o_dec_5d = o_dec_3d.reshape(1, *q.shape[1:])
            if _skip_mixed and chunk_prefill_size is None:
                q = o_dec_5d
            else:
                dec_tok_end = cu_q_lens[jnp.clip(distribution[0], 0, max_num_seqs)]
                tok_idx = jnp.arange(max_num_tokens, dtype=jnp.int32)[
                    None, :, None, None, None
                ]
                q = jnp.where(tok_idx < dec_tok_end, o_dec_5d, q)
        else:
            q, kv_cache_fused_processed = run_rpa_kernel(
                q,
                kv_cache_fused_processed,
                **_prepare_block_sizes(d_block_sizes, RpaCase.DECODE),
                static_q_len=decode_query_size,
                case=RpaCase.DECODE,
            )

    if chunk_prefill_size is not None:
        # Prefill-only
        q, kv_cache_fused_processed = run_rpa_kernel(
            q,
            kv_cache_fused_processed,
            **_prepare_block_sizes(p_block_sizes, RpaCase.PREFILL),
            static_q_len=chunk_prefill_size,
            case=RpaCase.PREFILL,
        )
    # Mixed
    if not _skip_mixed:
        q, kv_cache_fused_processed = run_rpa_kernel(
            q,
            kv_cache_fused_processed,
            **_prepare_block_sizes(m_block_sizes, RpaCase.MIXED),
            static_q_len=None,
            case=RpaCase.MIXED,
        )

    out_final = (
        q
        if q.ndim == 2
        else (
            prepare_outputs(q, actual_num_q_heads_per_kv_head, actual_head_dim)
            if _fused_out_pack == 1
            else prepare_head_packed_outputs(
                q, _fused_out_pack, actual_num_q_heads_per_kv_head, actual_head_dim
            )
        )
    )
    if out_final.ndim == 2 and (
        (_decode_tp16_prepared and out_final.shape[-1] == 1024)
        or (
            _decode_gate_in_values
            and values is not None
            and values.ndim == 2
            and out_final.shape[-1] == 2 * values.shape[-1]
        )
    ):
        # TP=16 prepared QKV_EPI packs 8x64 heads into the low 512 columns.
        out_final = out_final[:, : (512 if _decode_tp16_prepared else values.shape[-1])]
    if _decode_out_norm_eps is not None and out_final.ndim == 2:
        hd = actual_head_dim // max(abs(_fused_out_pack), 1)
        heads_f = out_final.reshape(out_final.shape[0], -1, hd).astype(jnp.float32)
        inv_r = lax.rsqrt(
            jnp.mean(heads_f * heads_f, axis=-1, keepdims=True)
            + float(_decode_out_norm_eps)
        )
        out_final = (heads_f * inv_r).astype(out_final.dtype).reshape(out_final.shape)
    if _decode_gate_in_values and values is not None and out_final.ndim == 2:
        sig_g = jax.nn.sigmoid(values.astype(jnp.float32)).astype(out_final.dtype)
        out_final = sig_g * out_final
    return (
        out_final,
        prepare_updated_kv_cache_fused(
            kv_cache_fused_processed, actual_num_kv_heads, actual_head_dim
        ),
    )


def write_kv_only_rpa_v3(
    kv_new: jax.Array,
    kv_cache_fused: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    interpret: bool = False,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
) -> jax.Array:
    """Write ``kv_new`` rows into ``kv_cache_fused`` along RPA v3 page tables without attention.

    ``kv_new`` has shape ``[T, kv_heads_packed, kv_packing, head_dim]`` (matching
    ``kv_cache_fused.shape[2:]`` and dtype). Used by L61 last-layer pruning to persist all ``T``
    tokens of K/V while running attention/MoE only on ``last_token_indices``.
    """
    if kv_new.shape[1:] != kv_cache_fused.shape[2:]:
        raise ValueError(
            f"kv_new trailing shape {kv_new.shape[1:]} != kv_cache trailing shape {kv_cache_fused.shape[2:]}"
        )
    if kv_new.dtype != kv_cache_fused.dtype:
        raise ValueError(
            f"kv_new dtype {kv_new.dtype} != kv_cache dtype {kv_cache_fused.dtype}"
        )
    phys_page_size = kv_cache_fused.shape[1]
    if kv_pack_bd4:
        kv_pack_dense2 = False
    page_size = (
        phys_page_size * 4
        if kv_pack_bd4
        else (phys_page_size * 2 if kv_pack_dense2 else phys_page_size)
    )
    num_page_indices = page_indices.shape[0]
    flat_cache = kv_cache_fused.reshape(
        (kv_cache_fused.shape[0] * phys_page_size, *kv_cache_fused.shape[2:])
    )

    if kv_pack_dense2 or kv_pack_bd4:
        pack_toks = 4 if kv_pack_bd4 else 2
        if kv_new.shape[1:] != (1, 4, 128) or kv_new.dtype != jnp.float8_e4m3fn:
            raise ValueError(
                f"kv_pack_dense2/bd4 requires kv_new trailing shape (1, 4, 128) fp8, got {kv_new.shape[1:]} {kv_new.dtype}"
            )

        def _kv_write_packed_kernel(
            kv_lens_ref,
            page_indices_ref,
            cu_q_lens_ref,
            cu_kv_lens_ref,
            distribution_ref,
            kv_new_ref,
            cache_in_ref,
            cache_out_ref,
            buf_vmem,
            pack_vmem,
            in_sems,
            cin_sems,
            out_sems,
            slot_src_smem,
            slot_sz_smem,
            slot_dst_smem,
            step_smem,
        ):
            slot_src_smem[0] = 0
            slot_src_smem[1] = 0
            slot_sz_smem[0] = 0
            slot_sz_smem[1] = 0
            step_smem[0] = 0
            n_seqs = jnp.clip(distribution_ref[2], 0, kv_lens_ref.shape[0])

            @pl.loop(0, n_seqs)
            def _seq_d2(seq_idx):
                q_start = cu_q_lens_ref[seq_idx]
                q_end = cu_q_lens_ref[seq_idx + 1]
                q_len = q_end - q_start
                kv_len = kv_lens_ref[seq_idx]

                @pl.when((q_len > 0) & (kv_len >= q_len))
                def _write_seq_d2():
                    kv_start = kv_len - q_len
                    p_start = kv_start // page_size
                    p_end = cdiv(kv_len, page_size)
                    seq_pg_base = cdiv(cu_kv_lens_ref[seq_idx], page_size)

                    @pl.loop(p_start, p_end)
                    def _page_d2(p):
                        step = step_smem[0]
                        slot = step % 2
                        other = 1 - slot
                        step_smem[0] = step + 1
                        prev_n = slot_sz_smem[slot]

                        @pl.when(prev_n > 0)
                        def _wait_prev_d2():
                            prev_src = slot_src_smem[slot]
                            prev_dst = slot_dst_smem[slot]
                            pltpu.make_async_copy(
                                src_ref=pack_vmem.at[slot, pl.ds(prev_src, prev_n)],
                                dst_ref=cache_out_ref.at[pl.ds(prev_dst, prev_n)],
                                sem=out_sems.at[slot],
                            ).wait()
                            slot_sz_smem[slot] = 0

                        tok_lo = jnp.maximum(kv_start, p * page_size)
                        tok_hi = jnp.minimum(kv_len, (p + 1) * page_size)
                        n_tok = tok_hi - tok_lo
                        off_lo = tok_lo - p * page_size
                        off_hi = tok_hi - p * page_size
                        pair_lo = off_lo // pack_toks
                        pair_hi = cdiv(off_hi, pack_toks)
                        n_pairs = pair_hi - pair_lo
                        src_row = q_start + (tok_lo - kv_start)
                        pg = page_indices_ref[
                            jnp.minimum(seq_pg_base + p, num_page_indices - 1)
                        ]
                        dst_row = pg * phys_page_size + pair_lo
                        need_cin = (off_lo % pack_toks) != 0

                        @pl.when(need_cin)
                        def _start_cin():
                            other_n = slot_sz_smem[other]

                            @pl.when(other_n > 0)
                            def _drain_other():
                                o_src = slot_src_smem[other]
                                o_dst = slot_dst_smem[other]
                                pltpu.make_async_copy(
                                    src_ref=pack_vmem.at[other, pl.ds(o_src, other_n)],
                                    dst_ref=cache_out_ref.at[pl.ds(o_dst, other_n)],
                                    sem=out_sems.at[other],
                                ).wait()
                                slot_sz_smem[other] = 0

                            pltpu.make_async_copy(
                                src_ref=cache_in_ref.at[pl.ds(dst_row, 1)],
                                dst_ref=pack_vmem.at[slot, pl.ds(pair_lo, 1)],
                                sem=cin_sems.at[slot],
                            ).start()

                        in_cp = pltpu.make_async_copy(
                            src_ref=kv_new_ref.at[pl.ds(src_row, n_tok)],
                            dst_ref=buf_vmem.at[slot, pl.ds(off_lo, n_tok)],
                            sem=in_sems.at[slot],
                        )
                        in_cp.start()
                        in_cp.wait()

                        @pl.when(need_cin)
                        def _wait_cin():
                            pltpu.make_async_copy(
                                src_ref=cache_in_ref.at[pl.ds(dst_row, 1)],
                                dst_ref=pack_vmem.at[slot, pl.ds(pair_lo, 1)],
                                sem=cin_sems.at[slot],
                            ).wait()

                        r_io = lax.broadcasted_iota(
                            jnp.int32, (phys_page_size, 128), 0
                        )
                        if kv_pack_bd4:
                            lo_h = (
                                lax.broadcasted_iota(
                                    jnp.int32, (phys_page_size, 128), 1
                                )
                                < 64
                            )
                            t0 = 4 * r_io + jnp.where(lo_h, 0, 1)
                            t1 = 4 * r_io + jnp.where(lo_h, 2, 3)
                            if interpret:
                                b_u8 = lax.bitcast_convert_type(
                                    buf_vmem[slot].reshape(page_size, 4, 128), jnp.uint8
                                )

                                def _kv16(sl_u8):
                                    return sl_u8[:, 0, :].astype(jnp.uint32) | (
                                        sl_u8[:, 1, :].astype(jnp.uint32)
                                        << jnp.uint32(8)
                                    )

                                new_lo = jnp.where(
                                    lo_h,
                                    _kv16(b_u8[0::4]),
                                    jnp.roll(_kv16(b_u8[1::4]), 64, axis=1),
                                )
                                new_hi = jnp.where(
                                    lo_h,
                                    _kv16(b_u8[2::4]),
                                    jnp.roll(_kv16(b_u8[3::4]), 64, axis=1),
                                )
                                p_u8 = lax.bitcast_convert_type(
                                    pack_vmem[slot].reshape(phys_page_size, 4, 128),
                                    jnp.uint8,
                                )
                                old_lo = p_u8[:, 0, :].astype(jnp.uint32) | (
                                    p_u8[:, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                old_hi = p_u8[:, 2, :].astype(jnp.uint32) | (
                                    p_u8[:, 3, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                lo = jnp.where(
                                    t0 < off_lo,
                                    old_lo,
                                    jnp.where(t0 < off_hi, new_lo, jnp.uint32(0)),
                                )
                                hi = jnp.where(
                                    t1 < off_lo,
                                    old_hi,
                                    jnp.where(t1 < off_hi, new_hi, jnp.uint32(0)),
                                )
                                out_u8 = jnp.stack(
                                    [
                                        (lo & jnp.uint32(0xFF)).astype(jnp.uint8),
                                        ((lo >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                            jnp.uint8
                                        ),
                                        (hi & jnp.uint32(0xFF)).astype(jnp.uint8),
                                        ((hi >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                            jnp.uint8
                                        ),
                                    ],
                                    axis=1,
                                )
                                pack_vmem[slot] = lax.bitcast_convert_type(
                                    out_u8, kv_new.dtype
                                ).reshape(phys_page_size, 1, 4, 128)
                            else:
                                b_u32 = (
                                    buf_vmem.bitcast(jnp.uint32)
                                    .at[slot]
                                    .reshape(page_size, 128)
                                )
                                p_u32 = (
                                    pack_vmem.bitcast(jnp.uint32)
                                    .at[slot]
                                    .reshape(phys_page_size, 128)
                                )
                                n0 = b_u32[pl.ds(0, phys_page_size, 4)] & jnp.uint32(0xFFFF)
                                n1 = pltpu.roll(
                                    b_u32[pl.ds(1, phys_page_size, 4)] & jnp.uint32(0xFFFF),
                                    shift=64,
                                    axis=1,
                                )
                                n2 = b_u32[pl.ds(2, phys_page_size, 4)] & jnp.uint32(0xFFFF)
                                n3 = pltpu.roll(
                                    b_u32[pl.ds(3, phys_page_size, 4)] & jnp.uint32(0xFFFF),
                                    shift=64,
                                    axis=1,
                                )
                                new_lo = jnp.where(lo_h, n0, n1)
                                new_hi = jnp.where(lo_h, n2, n3)
                                old_p = p_u32[...]
                                old_lo = old_p & jnp.uint32(0xFFFF)
                                old_hi = (old_p >> jnp.uint32(16)) & jnp.uint32(0xFFFF)
                                lo = jnp.where(
                                    t0 < off_lo,
                                    old_lo,
                                    jnp.where(t0 < off_hi, new_lo, jnp.uint32(0)),
                                )
                                hi = jnp.where(
                                    t1 < off_lo,
                                    old_hi,
                                    jnp.where(t1 < off_hi, new_hi, jnp.uint32(0)),
                                )
                                p_u32[...] = lo | (hi << jnp.uint32(16))
                        else:
                            t_even = 2 * r_io
                            t_odd = t_even + 1
                            if interpret:
                                b_u8 = lax.bitcast_convert_type(
                                    buf_vmem[slot].reshape(page_size, 4, 128), jnp.uint8
                                )
                                new_lo = b_u8[0::2, 0, :].astype(jnp.uint32) | (
                                    b_u8[0::2, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                new_hi = b_u8[1::2, 0, :].astype(jnp.uint32) | (
                                    b_u8[1::2, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                p_u8 = lax.bitcast_convert_type(
                                    pack_vmem[slot].reshape(phys_page_size, 4, 128),
                                    jnp.uint8,
                                )
                                old_lo = p_u8[:, 0, :].astype(jnp.uint32) | (
                                    p_u8[:, 1, :].astype(jnp.uint32) << jnp.uint32(8)
                                )
                                lo = jnp.where(
                                    t_even < off_lo,
                                    old_lo,
                                    jnp.where(t_even < off_hi, new_lo, jnp.uint32(0)),
                                )
                                hi = jnp.where(t_odd < off_hi, new_hi, jnp.uint32(0))
                                out_u8 = jnp.stack(
                                    [
                                        (lo & jnp.uint32(0xFF)).astype(jnp.uint8),
                                        ((lo >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                            jnp.uint8
                                        ),
                                        (hi & jnp.uint32(0xFF)).astype(jnp.uint8),
                                        ((hi >> jnp.uint32(8)) & jnp.uint32(0xFF)).astype(
                                            jnp.uint8
                                        ),
                                    ],
                                    axis=1,
                                )
                                pack_vmem[slot] = lax.bitcast_convert_type(
                                    out_u8, kv_new.dtype
                                ).reshape(phys_page_size, 1, 4, 128)
                            else:
                                b_u32 = (
                                    buf_vmem.bitcast(jnp.uint32)
                                    .at[slot]
                                    .reshape(page_size, 128)
                                )
                                p_u32 = (
                                    pack_vmem.bitcast(jnp.uint32)
                                    .at[slot]
                                    .reshape(phys_page_size, 128)
                                )
                                new_lo = b_u32[
                                    pl.ds(0, phys_page_size, 2)
                                ] & jnp.uint32(0xFFFF)
                                new_hi = b_u32[
                                    pl.ds(1, phys_page_size, 2)
                                ] & jnp.uint32(0xFFFF)
                                old_lo = p_u32[...] & jnp.uint32(0xFFFF)
                                lo = jnp.where(
                                    t_even < off_lo,
                                    old_lo,
                                    jnp.where(t_even < off_hi, new_lo, jnp.uint32(0)),
                                )
                                hi = jnp.where(t_odd < off_hi, new_hi, jnp.uint32(0))
                                p_u32[...] = lo | (hi << jnp.uint32(16))

                        pltpu.make_async_copy(
                            src_ref=pack_vmem.at[slot, pl.ds(pair_lo, n_pairs)],
                            dst_ref=cache_out_ref.at[pl.ds(dst_row, n_pairs)],
                            sem=out_sems.at[slot],
                        ).start()
                        slot_src_smem[slot] = pair_lo
                        slot_sz_smem[slot] = n_pairs
                        slot_dst_smem[slot] = dst_row

            for s in (0, 1):
                rem_n = slot_sz_smem[s]

                @pl.when(rem_n > 0)
                def _drain_d2(s=s, rem_n=rem_n):
                    rem_src = slot_src_smem[s]
                    rem_dst = slot_dst_smem[s]
                    pltpu.make_async_copy(
                        src_ref=pack_vmem.at[s, pl.ds(rem_src, rem_n)],
                        dst_ref=cache_out_ref.at[pl.ds(rem_dst, rem_n)],
                        sem=out_sems.at[s],
                    ).wait()

        tpu_version = 0 if interpret else get_tpu_version()
        out_shape = (
            pltpu.HBM(shape=flat_cache.shape, dtype=flat_cache.dtype)
            if tpu_version >= 7
            else jax.ShapeDtypeStruct(shape=flat_cache.shape, dtype=flat_cache.dtype)
        )
        pallas_kwargs_d2 = {
            "grid_spec": pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=5,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM((2, page_size, *kv_new.shape[1:]), kv_new.dtype),
                    pltpu.VMEM((2, phys_page_size, *kv_new.shape[1:]), kv_new.dtype),
                    pltpu.SemaphoreType.DMA((2,)),
                    pltpu.SemaphoreType.DMA((2,)),
                    pltpu.SemaphoreType.DMA((2,)),
                    pltpu.SMEM((2,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                    pltpu.SMEM((1,), jnp.int32),
                ],
            ),
            "out_shape": out_shape,
            "input_output_aliases": {6: 0},
            "name": f"RPA_write_kv_only_{'bd4' if kv_pack_bd4 else 'd2'}-p_{page_size}",
        }
        if interpret:
            pallas_kwargs_d2["interpret"] = pltpu.InterpretParams(
                detect_races=True, uninitialized_memory="nan"
            )
        else:
            pallas_kwargs_d2["compiler_params"] = pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                disable_bounds_checks=True,
            )

        kernel_d2 = pl.pallas_call(_kv_write_packed_kernel, **pallas_kwargs_d2)
        if tpu_version >= 7:
            updated_d2 = kernel_d2(
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                pltpu.with_memory_space_constraint(kv_new, pltpu.HBM),
                pltpu.with_memory_space_constraint(flat_cache, pltpu.HBM),
            )
        else:
            updated_d2 = kernel_d2(
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                kv_new,
                flat_cache,
            )
        return updated_d2.reshape(kv_cache_fused.shape)

    def _kv_write_kernel(
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        cu_kv_lens_ref,
        distribution_ref,
        kv_new_ref,
        cache_in_ref,
        cache_out_ref,
        buf_vmem,
        in_sems,
        out_sems,
        slot_sz_smem,
        slot_dst_smem,
        step_smem,
    ):
        del cache_in_ref
        slot_sz_smem[0] = 0
        slot_sz_smem[1] = 0
        step_smem[0] = 0
        n_seqs = jnp.clip(distribution_ref[2], 0, kv_lens_ref.shape[0])

        @pl.loop(0, n_seqs)
        def _seq(seq_idx):
            q_start = cu_q_lens_ref[seq_idx]
            q_end = cu_q_lens_ref[seq_idx + 1]
            q_len = q_end - q_start
            kv_len = kv_lens_ref[seq_idx]

            @pl.when((q_len > 0) & (kv_len >= q_len))
            def _write_seq():
                kv_start = kv_len - q_len
                p_start = kv_start // page_size
                p_end = cdiv(kv_len, page_size)
                seq_pg_base = cdiv(cu_kv_lens_ref[seq_idx], page_size)

                @pl.loop(p_start, p_end)
                def _page(p):
                    step = step_smem[0]
                    slot = step % 2
                    step_smem[0] = step + 1
                    prev_n = slot_sz_smem[slot]

                    @pl.when(prev_n > 0)
                    def _wait_prev():
                        prev_dst = slot_dst_smem[slot]
                        pltpu.make_async_copy(
                            src_ref=buf_vmem.at[slot, pl.ds(0, prev_n)],
                            dst_ref=cache_out_ref.at[pl.ds(prev_dst, prev_n)],
                            sem=out_sems.at[slot],
                        ).wait()

                    tok_lo = jnp.maximum(kv_start, p * page_size)
                    tok_hi = jnp.minimum(kv_len, (p + 1) * page_size)
                    n = tok_hi - tok_lo
                    src_row = q_start + (tok_lo - kv_start)
                    pg = page_indices_ref[
                        jnp.minimum(seq_pg_base + p, num_page_indices - 1)
                    ]
                    dst_row = pg * page_size + (tok_lo - p * page_size)

                    in_cp = pltpu.make_async_copy(
                        src_ref=kv_new_ref.at[pl.ds(src_row, n)],
                        dst_ref=buf_vmem.at[slot, pl.ds(0, n)],
                        sem=in_sems.at[slot],
                    )
                    in_cp.start()
                    in_cp.wait()
                    pltpu.make_async_copy(
                        src_ref=buf_vmem.at[slot, pl.ds(0, n)],
                        dst_ref=cache_out_ref.at[pl.ds(dst_row, n)],
                        sem=out_sems.at[slot],
                    ).start()
                    slot_sz_smem[slot] = n
                    slot_dst_smem[slot] = dst_row

        for s in (0, 1):
            rem_n = slot_sz_smem[s]

            @pl.when(rem_n > 0)
            def _drain(s=s, rem_n=rem_n):
                rem_dst = slot_dst_smem[s]
                pltpu.make_async_copy(
                    src_ref=buf_vmem.at[s, pl.ds(0, rem_n)],
                    dst_ref=cache_out_ref.at[pl.ds(rem_dst, rem_n)],
                    sem=out_sems.at[s],
                ).wait()

    tpu_version = 0 if interpret else get_tpu_version()
    out_shape = (
        pltpu.HBM(shape=flat_cache.shape, dtype=flat_cache.dtype)
        if tpu_version >= 7
        else jax.ShapeDtypeStruct(shape=flat_cache.shape, dtype=flat_cache.dtype)
    )
    pallas_kwargs = {
        "grid_spec": pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            grid=(1,),
            scratch_shapes=[
                pltpu.VMEM((2, page_size, *kv_new.shape[1:]), kv_new.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SMEM((2,), jnp.int32),
                pltpu.SMEM((2,), jnp.int32),
                pltpu.SMEM((1,), jnp.int32),
            ],
        ),
        "out_shape": out_shape,
        "input_output_aliases": {6: 0},
        "name": f"RPA_write_kv_only-p_{page_size}",
    }
    if interpret:
        pallas_kwargs["interpret"] = pltpu.InterpretParams(
            detect_races=True, uninitialized_memory="nan"
        )
    else:
        pallas_kwargs["compiler_params"] = pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        )

    kernel = pl.pallas_call(_kv_write_kernel, **pallas_kwargs)
    if tpu_version >= 7:
        updated = kernel(
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            pltpu.with_memory_space_constraint(kv_new, pltpu.HBM),
            pltpu.with_memory_space_constraint(flat_cache, pltpu.HBM),
        )
    else:
        updated = kernel(
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            kv_new,
            flat_cache,
        )
    return updated.reshape(kv_cache_fused.shape)
