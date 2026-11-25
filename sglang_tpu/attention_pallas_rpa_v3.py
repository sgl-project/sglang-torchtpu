# SPDX-License-Identifier: Apache-2.0
"""Pallas v3 ragged-paged-attention backend for the TPU torch_tpu platform.

Thin wrapper over :class:`BaseTpuPallasRpaAttnBackend` (see
``attention_pallas_rpa_base``) selecting the JAX/Pallas v3 kernel
(``sglang_tpu.attention.pallas.ragged_paged_attention_v3``). Selected via
``--attention-backend pallas_tpu_rpa_v3`` (the TPU platform default). Supports
attention sinks, bidirectional (``ENCODER_ONLY``) layers, and EAGLE speculative decoding (TARGET_VERIFY,
DRAFT_EXTEND_V2, multi-step draft decode). Target verification and the
decode-phase draft extension run on the kernel's DECODE range with a static
``decode_query_size``.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

import dataclasses
import functools
import logging
import os
from typing import Optional

import torch

from sglang_tpu.attention_pallas_rpa_base import (
    BaseTpuPallasRpaAttnBackend,
    PallasForwardMetadata,
    RpaKernelSpec,
    _build_pallas_kernel_op,
    _fused_kv_operands,
    _pool_kv_pack_heads,
    _unpack_out_heads,
    _zero_inactive_rpa_rows,
)
from sglang_tpu.envs import envs

logger = logging.getLogger(__name__)

__all__ = [
    "PallasForwardMetadata",
    "TpuPallasRpaV3AttnBackend",
]


def _import_v3_kernel():
    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
        ragged_paged_attention,
    )

    return ragged_paged_attention


_STATIONARY_Q_V2_MODES = (
    "m0_ring",
    "m0_f32e",
    "m0_ring_fuse",
    "m0_f32e_fuse",
)
_STATIONARY_Q_M0_MODES = ("m0", *_STATIONARY_Q_V2_MODES)
_STATIONARY_Q_MODES = (*_STATIONARY_Q_M0_MODES, "pilot")


def _env_block_sizes(full_env, sw_env, sliding_window):
    """The layer kind's "bq,bkv,bq_c,bkv_c" env as an int tuple, None when unset."""
    value = (sw_env if sliding_window else full_env).get()
    return tuple(int(x) for x in value.split(",")) if value else None


def _pool_kv_pack_dense2(pool) -> bool:
    """True when the fused pool packs 2 FP8 tokens per 32-bit row (``kv_pack_dense2``)."""
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    pools = (
        (pool.full_kv_pool, pool.swa_kv_pool)
        if isinstance(pool, SWAKVPool)
        else (pool,)
    )
    flags = {
        bool(getattr(p, "kv_pack_dense2", False))
        if isinstance(p, TpuFusedMHATokenToKVPool)
        else False
        for p in pools
    }
    if len(flags) != 1:
        raise NotImplementedError(
            f"SWA sub-pools set kv_pack_dense2 differently: {flags}"
        )
    return flags.pop()


def _pool_is_tp16_sq_64(pool) -> bool:
    """True when the fused pool holds 1 local KV head of dim 64 (Muse TP=16)."""
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    pools = (
        (pool.full_kv_pool, pool.swa_kv_pool)
        if isinstance(pool, SWAKVPool)
        else (pool,)
    )
    return bool(pools) and all(
        isinstance(p, TpuFusedMHATokenToKVPool)
        and getattr(p, "kv_pack_heads", 1) == 1
        and getattr(p, "head_num", 0) == 1
        and getattr(p, "head_dim", 0) == 64
        for p in pools
    )


def _invoke_v3_kernel(
    jax_rpa,
    *,
    queries,
    keys,
    values,
    kv_cache_fused,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    attention_sink,
    sm_scale,
    sliding_window,
    soft_cap,
    k_scale,
    v_scale,
    decode_query_size=1,
    use_causal_mask=True,
    fused_out_pack=1,
    prepared_inputs=False,
    skip_decode=False,
    a3_out_layout=False,
    skip_mixed=False,
    decode_out_norm_eps=None,
    decode_gate_in_values=False,
    decode_tp16_prepared=False,
    kv_pack_dense2=False,
):
    extra = {}
    long_prefill_tiles = False
    dense2_env = getattr(envs, "SGLANG_TPU_KV_PACK_DENSE2", None)
    use_dense2 = bool(
        kv_pack_dense2
        or (
            dense2_env is not None
            and dense2_env.get()
            and kv_cache_fused is not None
            and len(kv_cache_fused.shape) == 5
            and kv_cache_fused.shape[1] >= 8
            and kv_cache_fused.shape[2:] == (1, 4, 128)
        )
    )
    logical_page_size = (
        kv_cache_fused.shape[1] * 2
        if (use_dense2 and kv_cache_fused is not None)
        else (kv_cache_fused.shape[1] if kv_cache_fused is not None else 128)
    )
    if use_dense2:
        extra["_kv_pack_dense2"] = True
    allow_verify_q = decode_query_size == 1 or (
        decode_query_size > 1 and envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.get()
    )
    if (
        envs.SGLANG_TPU_MUSE_KERNEL_ENABLED.get()
        and sliding_window == 2048
        and allow_verify_q
    ):
        # Muse window-2048 decode (DECODE launch only): one 4096-key block instead
        # of two 2048 ones, measured 57.2->51.9 us at 16 seqs, 172->144 us at 64.
        extra["d_block_sizes"] = (decode_query_size, 4096, decode_query_size, 4096)
    decode_blocks = _env_block_sizes(
        envs.SGLANG_TPU_RPA_V3_DECODE_BLOCK_SIZES,
        envs.SGLANG_TPU_RPA_V3_DECODE_BLOCK_SIZES_SW,
        sliding_window,
    )
    if (
        envs.SGLANG_TPU_RPA_V3_DECODE_SW_WINDOW_ALIGNED.get()
        and sliding_window
        and allow_verify_q
    ):
        page_size = logical_page_size
        window_bkv = -(-(sliding_window + page_size - 1) // page_size) * page_size
        extra["d_block_sizes"] = (
            decode_query_size,
            window_bkv,
            decode_query_size,
            window_bkv,
        )
        extra["_decode_window_aligned_kv"] = True
    if decode_blocks and decode_query_size == 1:
        extra["d_block_sizes"] = decode_blocks
    if envs.SGLANG_TPU_RPA_V3_DECODE_ZERO_KV_TAIL.get():
        extra["_decode_zero_kv_tail"] = True
    if envs.SGLANG_TPU_RPA_V3_DECODE_RING.get():
        extra["_decode_ring"] = envs.SGLANG_TPU_RPA_V3_DECODE_RING.get()
    decode_fast_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST", None)
    df_verify_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY", None)
    need_fused_post = decode_out_norm_eps is not None or decode_gate_in_values
    df_raw = decode_fast_env.get().strip() if decode_fast_env is not None else ""
    df_raw_toks = (
        set(df_raw.lower().replace(",", "_").replace("-", "_").split("_"))
        if df_raw
        else set()
    )
    allow_df_dq = decode_query_size == 1 or (
        2 <= decode_query_size <= 8
        and (
            need_fused_post
            or (df_verify_env is not None and df_verify_env.get())
            or bool(df_raw_toks & {"multiq", "verify"})
        )
    )
    if allow_df_dq and (df_raw or need_fused_post):
        df_val = df_raw or "w7"
        w7_dec_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST_W7", None)
        if (
            w7_dec_env is not None
            and w7_dec_env.get()
            and df_val.lower() in ("1", "true", "auto", "on", "v2", "2", "streamdyn")
        ):
            df_val = "w7"
        w8_dec_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST_W8", None)
        if w8_dec_env is not None and w8_dec_env.get() and "w8" not in df_val.lower():
            df_val = f"{df_val}_w8"
        if decode_query_size > 1:
            vopts_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY_OPTS", None)
            vopts_raw = vopts_env.get().strip() if vopts_env is not None else ""
            if vopts_raw:
                if vopts_raw.lower() == "auto":
                    vchunk = "c1024" if decode_query_size <= 4 else "c512"
                    vopts_raw = f"{vchunk},noz,pbf16,g4k,foldcol,noadapt,verify"
                df_val = f"{df_val},{vopts_raw}"
        adapt_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_ADAPTIVE", None)
        if (
            adapt_env is not None
            and adapt_env.get()
            and "adapt" not in df_val.lower()
            and "noadapt" not in df_val.lower()
        ):
            df_val = f"{df_val}_adapt"
        extra["_decode_fast"] = df_val
    probe_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_PROBE", None)
    if probe_env is not None and probe_env.get().strip():
        extra["_decode_probe"] = probe_env.get().strip().lower()
    mixed_blocks = _env_block_sizes(
        envs.SGLANG_TPU_RPA_V3_MIXED_BLOCK_SIZES,
        envs.SGLANG_TPU_RPA_V3_MIXED_BLOCK_SIZES_SW,
        sliding_window,
    )
    prefill_mode = envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.get().strip().lower()
    is_prepared = prepared_inputs or (
        queries is not None and getattr(queries, "ndim", 0) == 5
    )
    num_q_tokens = (
        0
        if queries is None
        else (queries.shape[1] if is_prepared else queries.shape[0])
    )
    w7_prefill_env = getattr(envs, "SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS", None)
    w7_prefill_toks = (
        {
            t.strip()
            for t in w7_prefill_env.get().lower().replace(",", " ").split()
            if t.strip()
        }
        if w7_prefill_env is not None and w7_prefill_env.get().strip()
        else set()
    )
    lc_prefill_env = getattr(envs, "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS", None)
    lc_prefill_toks = (
        {
            t.strip()
            for t in lc_prefill_env.get().lower().replace(",", " ").split()
            if t.strip()
        }
        if lc_prefill_env is not None and lc_prefill_env.get().strip()
        else set()
    )
    raw_all_toks = w7_prefill_toks | lc_prefill_toks
    w9_prefill_env = getattr(envs, "SGLANG_TPU_RPA_V3_PREFILL_W9", None)
    use_w9_preset = ("w9" in raw_all_toks) or bool(
        w9_prefill_env is not None and w9_prefill_env.get()
    )
    tp16_sq_env = getattr(envs, "SGLANG_TPU_RPA_V3_TP16_STATIONARY_Q", None)
    tp16_sq_explicit_env = os.environ.get("SGLANG_TPU_RPA_V3_TP16_STATIONARY_Q")
    tp16_sq_env_on = True if tp16_sq_env is None else bool(tp16_sq_env.get())
    tp16_native_on = bool(envs.SGLANG_TPU_RPA_V3_TP16_NATIVE_PREFILL.get())
    a1_rpa_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT", None)
    allow_tp16_sq = "notp16_sq" not in raw_all_toks and (
        ("tp16_sq" in raw_all_toks)
        or bool(
            is_prepared
            and a1_rpa_env is not None
            and a1_rpa_env.get()
            and tp16_sq_env_on
        )
        or (
            tp16_sq_env_on
            and (
                use_w9_preset
                or is_prepared
                or tp16_sq_explicit_env is not None
                or not tp16_native_on
            )
        )
    )
    # stationary_q supports TP8 (16 Q / 2 packed KV) and opt-in TP16 (8 Q / 1 KV, d=64),
    # plus legacy TP16_NATIVE_PREFILL block-size routing when explicitly requested.
    is_tp8_sq_shape = False
    is_tp16_sq_shape = False
    is_tp16_legacy_native = False
    if queries is not None:
        if is_prepared:
            is_tp8_sq_shape = (
                queries.ndim == 5
                and queries.shape[0] == 1
                and (queries.shape[2] * queries.shape[3]) == 16
            )
            is_tp16_sq_shape = (
                queries.ndim == 5
                and queries.shape[0] == 1
                and (queries.shape[2] * queries.shape[3]) == 8
                and queries.shape[4] == 128
            )
        else:
            q_ndim = len(queries.shape)
            k_ndim = len(keys.shape) if keys is not None else 0
            is_tp8_sq_shape = (
                q_ndim == 3
                and k_ndim == 3
                and keys.shape[1] == 1
                and queries.shape[1] == 16
            )
            is_tp16_sq_shape = (
                q_ndim == 3
                and k_ndim == 3
                and keys.shape[1] == 1
                and queries.shape[1] == 8
                and queries.shape[2] == 64
            )
            is_tp16_legacy_native = (
                not allow_tp16_sq
                and envs.SGLANG_TPU_RPA_V3_TP16_NATIVE_PREFILL.get()
                and prefill_mode == "m0_ring_fuse"
                and q_ndim == 3
                and queries.shape[1:] == (8, 64)
                and k_ndim == 3
                and keys.shape[1:] == (1, 64)
            )
    is_sq_head_shape = (
        is_tp8_sq_shape
        or (allow_tp16_sq and is_tp16_sq_shape)
        or is_tp16_legacy_native
    )
    eff_skip_decode = bool(
        skip_decode
        or (
            is_prepared
            and not skip_mixed
            and envs.SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE.get()
        )
    )
    orig_num_q_tokens = num_q_tokens
    pad_short_tp16_sq = bool(
        eff_skip_decode
        and not skip_mixed
        and decode_query_size == 1
        and prefill_mode in _STATIONARY_Q_MODES
        and 0 < num_q_tokens < 256
        and allow_tp16_sq
        and is_tp16_sq_shape
        and soft_cap is None
        and attention_sink is None
    )
    is_mock_stub = False
    if pad_short_tp16_sq:
        import types

        pad_tok = 256 - orig_num_q_tokens
        is_mock_stub = isinstance(queries, types.SimpleNamespace)
        if is_mock_stub:
            q_sh = (
                (queries.shape[0], 256, *queries.shape[2:])
                if is_prepared
                else (256, *queries.shape[1:])
            )
            queries = types.SimpleNamespace(**{**vars(queries), "shape": q_sh})
            if isinstance(keys, types.SimpleNamespace) and hasattr(keys, "shape"):
                keys = types.SimpleNamespace(
                    **{**vars(keys), "shape": (256, *keys.shape[1:])}
                )
            if isinstance(values, types.SimpleNamespace) and hasattr(values, "shape"):
                values = types.SimpleNamespace(
                    **{**vars(values), "shape": (256, *values.shape[1:])}
                )
        else:
            import jax.numpy as jnp

            if is_prepared:
                assert queries.ndim == 5, f"Expected 5D prepared queries, got {queries.shape}"
                queries = jnp.pad(
                    queries, ((0, 0), (0, pad_tok), (0, 0), (0, 0), (0, 0))
                )
            else:
                assert queries.ndim == 3, f"Expected 3D unprepared queries, got {queries.shape}"
                queries = jnp.pad(queries, ((0, pad_tok), (0, 0), (0, 0)))
            if keys is not None:
                keys = jnp.pad(
                    keys, ((0, pad_tok), *((0, 0) for _ in range(keys.ndim - 1)))
                )
            if values is not None:
                values = jnp.pad(
                    values,
                    ((0, pad_tok), *((0, 0) for _ in range(values.ndim - 1))),
                )
        num_q_tokens = 256
    use_stationary_q_prefill = (
        not skip_mixed
        and prefill_mode in _STATIONARY_Q_MODES
        and num_q_tokens >= 256
        and is_sq_head_shape
        and soft_cap is None
        and attention_sink is None
    )
    if use_stationary_q_prefill:
        sq_blocks_env = getattr(
            envs, "SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES", None
        )
        sq_blocks_sw_env = getattr(
            envs, "SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES_SW", None
        )
        stationary_q_blocks = (
            _env_block_sizes(sq_blocks_env, sq_blocks_sw_env, sliding_window)
            if (sq_blocks_env is not None and sq_blocks_sw_env is not None)
            else None
        )
        if stationary_q_blocks and pad_short_tp16_sq and stationary_q_blocks[0] > 256:
            stationary_q_blocks = (
                256,
                stationary_q_blocks[1],
                min(stationary_q_blocks[2], 256),
                *stationary_q_blocks[3:],
            )
        per_seq_kv_capacity = 0
        if (
            page_indices is not None
            and kv_lens is not None
            and kv_cache_fused is not None
            and getattr(kv_lens, "shape", (0,)) > (0,)
        ):
            pages_per_seq = page_indices.shape[0] // kv_lens.shape[0]
            per_seq_kv_capacity = pages_per_seq * logical_page_size
        is_lc_full = (
            sliding_window is None
            and not is_tp16_legacy_native
            and prefill_mode in _STATIONARY_Q_M0_MODES
            and per_seq_kv_capacity > 32768
        )
        # Under w9 / scope_lc, keep LC_OPTS scoped to Full-Attn >32K so LC mrb1/vreg_acc1
        # never bleed onto SWA layers, while keeping default-OFF identical to cand_w9_base.
        scoped_lc_toks = (
            lc_prefill_toks
            if (is_lc_full or not (use_w9_preset or "scope_lc" in raw_all_toks))
            else {
                t
                for t in lc_prefill_toks
                if t.startswith("sw_") or t in ("w7", "w8", "w9", "tp16_sq")
            }
        )
        all_prefill_toks = w7_prefill_toks | scoped_lc_toks
        use_w8_preset = ("w8" in all_prefill_toks) and not use_w9_preset
        use_w7_preset = ("w7" in all_prefill_toks) or use_w8_preset or use_w9_preset
        is_tp16_sq_target = bool(allow_tp16_sq and is_tp16_sq_shape)
        eff_lc_toks = set(lc_prefill_toks)
        if use_w9_preset and is_lc_full:
            if not (eff_lc_toks & {"bq1024", "bq2048", "bq4096", "bq8192"}):
                eff_lc_toks.add("bq4096")
            tp8_bq8k = (
                not is_tp16_sq_target
                and "bq8192" in eff_lc_toks
                and num_q_tokens >= 8192
            )
            if not (
                eff_lc_toks
                & {"bkv4096", "bkv8192", "bkv16384", "bkv16k", "nobkv8192"}
            ) and not tp8_bq8k:
                eff_lc_toks.add("bkv8192")
            if not (eff_lc_toks & {"s3", "mixed_s3"}):
                eff_lc_toks.add("s2")
            if "noprefix_fast" not in eff_lc_toks:
                eff_lc_toks.add("prefix_fast")
            if "nostaged_qo" not in eff_lc_toks:
                eff_lc_toks.add("staged_qo")
        if stationary_q_blocks:
            bq_sz = stationary_q_blocks[0]
            extra["m_block_sizes"] = stationary_q_blocks
        else:
            if prefill_mode in _STATIONARY_Q_V2_MODES:
                bq_sz = (
                    512
                    if (
                        (
                            (
                                sliding_window is None
                                and (
                                    per_seq_kv_capacity > 32768
                                    or "full_bq512" in all_prefill_toks
                                    or use_w7_preset
                                )
                            )
                            or (
                                sliding_window is not None
                                and "sw_bq512" in all_prefill_toks
                            )
                        )
                        and num_q_tokens >= 512
                    )
                    else 256
                )
            else:
                bq_sz = 512 if num_q_tokens >= 512 else 256
            if (
                sliding_window is None
                and "full_bq1024" in all_prefill_toks
                and num_q_tokens >= 1024
            ):
                bq_sz = 1024
            if is_lc_full:
                if "bq8192" in eff_lc_toks and num_q_tokens >= 8192:
                    bq_sz = 8192
                elif (
                    "bq4096" in eff_lc_toks or "bq8192" in eff_lc_toks
                ) and num_q_tokens >= 4096:
                    bq_sz = 4096
                elif (
                    "bq2048" in eff_lc_toks
                    or "bq4096" in eff_lc_toks
                    or "bq8192" in eff_lc_toks
                ) and num_q_tokens >= 2048:
                    bq_sz = 2048
                elif (
                    "bq1024" in eff_lc_toks
                    or "bq2048" in eff_lc_toks
                    or "bq4096" in eff_lc_toks
                    or "bq8192" in eff_lc_toks
                ) and num_q_tokens >= 1024:
                    bq_sz = 1024
            if (
                is_lc_full
                and ("bkv16384" in eff_lc_toks or "bkv16k" in eff_lc_toks)
                and num_q_tokens >= 512
            ):
                bkv_sz = 16384
            elif is_lc_full and "bkv8192" in eff_lc_toks and num_q_tokens >= 512:
                bkv_sz = 8192
            elif (
                sliding_window is None
                and prefill_mode in _STATIONARY_Q_M0_MODES
                and num_q_tokens >= 512
                and (
                    num_q_tokens >= 16384
                    or per_seq_kv_capacity > 18432
                    or "full_bkv4k" in all_prefill_toks
                    or use_w7_preset
                )
            ):
                bkv_sz = 4096
            else:
                bkv_sz = 2048
            bkv_csz = max(256, logical_page_size)
            bkv_sz = max(bkv_sz, bkv_csz)
            if is_lc_full and not is_tp16_sq_target and bq_sz >= 8192 and bkv_sz >= 8192:
                raise ValueError(
                    f"TP=8 LC RPAm (bq_sz={bq_sz}, bkv_sz={bkv_sz}) exceeds 64 MiB VMEM; "
                    "use bq4096 or bkv4096 on TP=8."
                )
            extra["m_block_sizes"] = (bq_sz, bkv_sz, 256, bkv_csz)
        prefill_opts = {
            f"stationary_q_{prefill_mode}",
            "skip_empty_cdma",
            "mixed_s3",
        }
        if allow_tp16_sq and is_tp16_sq_shape:
            prefill_opts.add("tp16_sq")
        if is_lc_full and "s2" in eff_lc_toks:
            prefill_opts.discard("mixed_s3")
        if is_lc_full and "prefix_fast" in eff_lc_toks:
            prefill_opts.add("prefix_fast")
        if is_lc_full and (
            "staged_qo" in eff_lc_toks
            or (bq_sz >= 4096 and "nostaged_qo" not in eff_lc_toks)
        ):
            prefill_opts.add("staged_qo")
        if use_w7_preset and sliding_window is not None:
            prefill_opts.update(("fp8_qk", "mrb3"))
        if use_w8_preset and sliding_window is not None:
            prefill_opts.update(("vreg_acc", "sw_short_fuse"))
        if use_w9_preset:
            if "nofp8_qk" not in raw_all_toks and (
                sliding_window is not None
                or "nofull_fp8_qk" not in raw_all_toks
            ):
                prefill_opts.add("fp8_qk")
            if "nobf16_pop" not in raw_all_toks:
                prefill_opts.add("bf16_pop")
            if "nosub8_l" not in raw_all_toks:
                prefill_opts.add("sub8_l")
            if sliding_window is not None:
                if sliding_window == 2048 and "nosw2048_fuse" not in raw_all_toks:
                    prefill_opts.add("sw2048_fuse")
                if sliding_window == 2048 and "nosw_short_v2" not in raw_all_toks:
                    prefill_opts.add("sw_short_v2")
                if (
                    "sw_mk128" in raw_all_toks
                    or (
                        not (allow_tp16_sq and is_tp16_sq_shape)
                        and "nosw_mk128" not in raw_all_toks
                        and "nomk128" not in raw_all_toks
                    )
                ):
                    prefill_opts.add("mk128")
                if (
                    sliding_window in (2048, 2049)
                    and "mk128" in prefill_opts
                    and "nosw2048_fuse" not in raw_all_toks
                    and "nosw_unalign_fuse" not in raw_all_toks
                ):
                    prefill_opts.add("sw_unalign_fuse")
                if (
                    sliding_window in (2048, 2049)
                    and "mk128" in prefill_opts
                    and "nosw2048_fuse" not in raw_all_toks
                    and "nosw_tail_clamp" not in raw_all_toks
                ):
                    prefill_opts.add("sw_tail_clamp")
            else:
                if "nomk128" not in raw_all_toks:
                    prefill_opts.add("mk128")
                if not is_lc_full and "noprefix_fast" not in raw_all_toks:
                    prefill_opts.add("prefix_fast")
            if "mk128" in prefill_opts:
                if "nola2" not in raw_all_toks:
                    prefill_opts.add("la2")
                if "nounroll_up" not in raw_all_toks:
                    prefill_opts.add("unroll_up")
        for w7_opt in (
            "fp8_kv",
            "fp8_qk",
            "fp8_qk_dyn",
            "fp8_qkpv",
            "fp8_qkpv_dyn",
            "fp8_qkpv_fast",
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
            "sw_unalign_fuse",
            "sw_tail_clamp",
            "tp16_sq",
            "sw_short_fuse",
            "tq_unroll",
            "mrb1",
            "mrb3",
            "mrb4",
            "f32_exp",
            "staged_qo",
        ):
            if w7_opt in all_prefill_toks:
                prefill_opts.add(w7_opt)
            if sliding_window is not None and f"sw_{w7_opt}" in all_prefill_toks:
                prefill_opts.add(w7_opt)
            if sliding_window is None and f"full_{w7_opt}" in all_prefill_toks:
                prefill_opts.add(w7_opt)
        if "mk128" in prefill_opts:
            prefill_opts.difference_update(("vreg_acc", "vreg_acc1", "mrb1"))
        if (
            not (allow_tp16_sq and is_tp16_sq_shape)
            and envs.SGLANG_TPU_RPA_V3_TP16_NATIVE_PREFILL.get()
            and prefill_mode == "m0_ring_fuse"
            and not stationary_q_blocks
            and skip_decode
            and decode_query_size == 1
            and not is_prepared
            and use_causal_mask
            and queries.shape == (8192, 8, 64)
            and keys.shape == (8192, 1, 64)
            and str(queries.dtype) == "bfloat16"
            and str(kv_cache_fused.dtype) == "float8_e4m3fn"
            and logical_page_size == 128
            and sliding_window in (None, 2048, 2049)
        ):
            # Preserve 256-key softmax updates and their BF16 rounding boundaries.
            extra["m_block_sizes"] = (
                (512, 2048, 512, 256)
                if sliding_window is None
                else (256, 1024, 256, 256)
            )
            long_prefill_tiles = (
                sliding_window is None
                and per_seq_kv_capacity >= 65536 + 6144
                and attention_sink is None
                and soft_cap is None
                and envs.SGLANG_TPU_RPA_V3_TP16_LONG_PREFILL_TILES.get()
            )
            prefill_opts.update(("mask_skip", "visible_kv_tile"))
            if sm_scale == 0.125 and k_scale in (None, 1.0):
                prefill_opts.add("fold_scale")
        if sliding_window is None and prefill_mode in _STATIONARY_Q_M0_MODES:
            prefill_opts.add("bkv_full_unroll")
        sw2048_fuse_env = getattr(envs, "SGLANG_TPU_RPA_V3_PREFILL_SW2048_FUSE", None)
        if (
            sliding_window == 2048
            and sw2048_fuse_env is not None
            and sw2048_fuse_env.get()
        ):
            prefill_opts.add("sw2048_fuse")
    else:
        if mixed_blocks:
            extra["m_block_sizes"] = mixed_blocks
        prefill_opts = {
            opt
            for opt, env in (
                ("fold_scale", envs.SGLANG_TPU_RPA_V3_PREFILL_FOLD_SCALE),
                ("mask_skip", envs.SGLANG_TPU_RPA_V3_PREFILL_MASK_SKIP),
            )
            if env.get()
        }
    if eff_skip_decode:
        extra["_skip_decode"] = True
    if skip_mixed:
        extra["_skip_mixed"] = True
    if a3_out_layout != 0:
        if use_stationary_q_prefill and extra.get("_skip_decode", False):
            extra["_a3_out_layout"] = True
        else:
            fused_out_pack = a3_out_layout
    if prefill_opts:
        extra["_prefill_opts"] = frozenset(prefill_opts)
    if fused_out_pack != 1:
        extra["_fused_out_pack"] = fused_out_pack
    if prepared_inputs:
        extra["_prepared_inputs"] = True
    if decode_out_norm_eps is not None:
        extra["_decode_out_norm_eps"] = float(decode_out_norm_eps)
    if decode_gate_in_values:
        extra["_decode_gate_in_values"] = True
    if decode_tp16_prepared:
        extra["_decode_tp16_prepared"] = True
    run = functools.partial(
        jax_rpa,
        queries,
        keys,
        values,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        None,  # custom_mask
        attention_sink,
        causal=1 if use_causal_mask else 0,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        decode_query_size=decode_query_size,
        **extra,
    )

    if long_prefill_tiles:
        import jax
        import jax.numpy as jnp

        # Wider Q tiles amortize the long prefix scan. Keep the 512-row tile for
        # ragged batches and whenever rounding to 1024 would add padded Q work.
        live = cu_q_lens[1] - cu_q_lens[0]
        eligible = (
            (distribution[0] == 0)
            & (distribution[2] == 1)
            & (kv_lens[0] - live >= 65536)
            & (live >= 6144)
            & (2 * ((live + 1023) // 1024) == (live + 511) // 512)
        )
        tile = jnp.where(eligible, jnp.where(live % 2048 == 0, 2, 1), 0)
        lp_bkv_csz = max(256, logical_page_size)
        res = jax.lax.switch(
            tile,
            (
                run,
                functools.partial(run, m_block_sizes=(1024, 4096, 1024, lp_bkv_csz)),
                functools.partial(run, m_block_sizes=(2048, 4096, 2048, lp_bkv_csz)),
            ),
        )
    else:
        res = run()
    if pad_short_tp16_sq:
        if is_mock_stub:
            if isinstance(res, tuple) and len(res) == 2 and hasattr(res[0], "__getitem__"):
                out_final, updated_kv = res
                out_final = (
                    out_final[:, :orig_num_q_tokens]
                    if out_final.ndim == 5
                    else out_final[:orig_num_q_tokens]
                )
                return out_final, updated_kv
            return res
        assert isinstance(res, tuple) and len(res) == 2, (
            f"Expected (out, updated_kv) 2-tuple from jax_rpa, got {type(res)}"
        )
        out_final, updated_kv = res
        out_final = (
            out_final[:, :orig_num_q_tokens]
            if out_final.ndim == 5
            else out_final[:orig_num_q_tokens]
        )
        tok_ax = 1 if out_final.ndim == 5 else 0
        assert out_final.shape[tok_ax] == orig_num_q_tokens
        return out_final, updated_kv
    return res


RPA_V3_SPEC = RpaKernelSpec(
    op_name="sglang_tpu_pallas::ragged_paged_attention",
    import_kernel=_import_v3_kernel,
    invoke_kernel=_invoke_v3_kernel,
    supports_attention_sink=True,
    uses_cu_kv_lens=True,
    is_batched=False,
    supports_decode_query_size=True,
    supports_shared_decode=True,
)


def _fused_out_spec(pack: int) -> RpaKernelSpec:
    """RPA_V3_SPEC returning packed heads' own lanes (SGLANG_TPU_RPA_V3_PREFILL_FUSED_OUT)."""
    tag = f"pad{-pack}" if pack < 0 else str(pack)
    return dataclasses.replace(
        RPA_V3_SPEC,
        op_name=f"{RPA_V3_SPEC.op_name}_fused_out{tag}",
        invoke_kernel=functools.partial(_invoke_v3_kernel, fused_out_pack=pack),
        out_head_pack=abs(pack),
    )


def _prepared_spec(spec: RpaKernelSpec) -> RpaKernelSpec:
    """``spec`` taking q / merged kv already in the kernel's layout (SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT)."""
    return dataclasses.replace(
        spec,
        op_name=f"{spec.op_name}_prepared",
        invoke_kernel=functools.partial(spec.invoke_kernel, prepared_inputs=True),
        prepared_inputs=True,
    )


def _prepared_a3_layout_spec(pack: int) -> RpaKernelSpec:
    """Prepared Stationary-Q RPA_V3_SPEC writing unpacked bf16[T, Hq * d] directly."""
    tag = f"pad{-pack}" if pack < 0 else str(pack)
    return dataclasses.replace(
        RPA_V3_SPEC,
        op_name=f"{RPA_V3_SPEC.op_name}_prepared_a3_layout{tag}",
        invoke_kernel=functools.partial(
            _invoke_v3_kernel,
            prepared_inputs=True,
            skip_decode=True,
            a3_out_layout=pack,
        ),
        out_head_pack=abs(pack),
        prepared_inputs=True,
    )


def _skip_decode_spec(spec: RpaKernelSpec) -> RpaKernelSpec:
    """``spec`` skipping the 0-sequence DECODE sub-launch on pure extend forwards."""
    return dataclasses.replace(
        spec,
        op_name=f"{spec.op_name}_skip_decode",
        invoke_kernel=functools.partial(spec.invoke_kernel, skip_decode=True),
    )


def _skip_mixed_spec(spec: RpaKernelSpec) -> RpaKernelSpec:
    """``spec`` skipping the 0-sequence MIXED sub-launch on pure decode forwards."""
    return dataclasses.replace(
        spec,
        op_name=f"{spec.op_name}_skip_mixed",
        invoke_kernel=functools.partial(spec.invoke_kernel, skip_mixed=True),
    )


def _decode_gated_spec(spec: RpaKernelSpec, norm_eps: float) -> RpaKernelSpec:
    """Prepared decode ``spec`` fusing per-head RMSNorm and ``sigmoid(gate) * out``."""
    eps_tag = f"{norm_eps:.0e}".replace("-", "m").replace("+", "p")
    return dataclasses.replace(
        spec,
        op_name=f"{spec.op_name}_onorm_{eps_tag}_gate_v2",
        invoke_kernel=functools.partial(
            spec.invoke_kernel,
            decode_out_norm_eps=float(norm_eps),
            decode_gate_in_values=True,
        ),
        keep_values_when_prepared=True,
    )


def _tp16_prepared_spec(spec: RpaKernelSpec) -> RpaKernelSpec:
    """TP=16 (8Q/1KV, d=64) prepared decode ``spec`` returning 512 output columns."""
    return dataclasses.replace(
        spec,
        op_name=f"{spec.op_name}_tp16_sq",
        invoke_kernel=functools.partial(
            spec.invoke_kernel, decode_tp16_prepared=True
        ),
        out_head_pack=4,
    )


def _import_muse_batched_kernel():
    from sglang_tpu.attention.pallas.batched_rpa_muse.wrapper import (
        ragged_paged_attention,
    )

    return ragged_paged_attention


def _invoke_muse_batched_prefill_kernel(
    jax_rpa,
    *,
    queries,
    keys,
    values,
    kv_cache_fused,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    attention_sink,
    sm_scale,
    sliding_window,
    soft_cap,
    k_scale,
    v_scale,
    decode_query_size=1,
    use_causal_mask=True,
):
    """batched_rpa_muse on a v3 extend batch: same token-major KV cache, MIXED launch only."""
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.batched_rpa_muse import configs

    # Only causal extend rows reach here (TpuPallasRpaV3AttnBackend._kernel_op).
    del cu_kv_lens, decode_query_size, use_causal_mask
    return jax_rpa(
        queries,
        keys,
        values,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        # v3 extend is [0, n, n]; batched lists extend rows first: [n, n, n].
        jnp.broadcast_to(distribution[2], (3,)),
        attention_sink=attention_sink,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        rpa_cases=configs.RpaCase.MIXED,
    )


MUSE_BATCHED_PREFILL_SPEC = RpaKernelSpec(
    op_name="sglang_tpu_pallas::ragged_paged_attention_v3_muse_batched_prefill",
    import_kernel=_import_muse_batched_kernel,
    invoke_kernel=_invoke_muse_batched_prefill_kernel,
    supports_attention_sink=True,
    uses_cu_kv_lens=False,
)


class TpuPallasRpaV3AttnBackend(BaseTpuPallasRpaAttnBackend):
    """RPA v3 attention backend (``--attention-backend pallas_tpu_rpa_v3``)."""

    kernel_spec = RPA_V3_SPEC

    def __init__(self, model_runner, speculative_step_id=0):
        super().__init__(model_runner, speculative_step_id)
        self._batched_prefill_op = (
            _build_pallas_kernel_op(MUSE_BATCHED_PREFILL_SPEC)
            if envs.SGLANG_TPU_MUSE_KERNEL_BATCHED_PREFILL_ATTN.get()
            else None
        )
        pack = _pool_kv_pack_heads(self.token_to_kv_pool)
        self._kv_pack_dense2 = _pool_kv_pack_dense2(self.token_to_kv_pool)
        self._fused_out_op = (
            _build_pallas_kernel_op(_fused_out_spec(pack))
            if envs.SGLANG_TPU_RPA_V3_PREFILL_FUSED_OUT.get()
            and pack > 1
            and self._dcp == 1
            else None
        )
        self._skip_decode_op = (
            _build_pallas_kernel_op(
                _skip_decode_spec(
                    _fused_out_spec(pack)
                    if self._fused_out_op is not None
                    else RPA_V3_SPEC
                )
            )
            if envs.SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE.get() and self._dcp == 1
            else None
        )
        skip_mixed_env = getattr(envs, "SGLANG_TPU_RPA_V3_SKIP_EMPTY_MIXED", None)
        decode_fused_out_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FUSED_OUT", None)
        decode_fast_v2_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST_V2", None)
        want_skip_mixed = bool(skip_mixed_env is not None and skip_mixed_env.get())
        want_v2 = bool(decode_fast_v2_env is not None and decode_fast_v2_env.get())
        want_decode_fused_out = bool(
            pack > 1
            and (
                (decode_fused_out_env is not None and decode_fused_out_env.get())
                or (want_v2 and want_skip_mixed)
            )
        )
        self._want_decode_fast_v2 = want_v2
        if self._dcp == 1 and (want_skip_mixed or want_decode_fused_out):
            dec_spec = (
                _fused_out_spec(pack)
                if (self._fused_out_op is not None or want_decode_fused_out)
                else RPA_V3_SPEC
            )
            if want_skip_mixed:
                dec_spec = _skip_mixed_spec(dec_spec)
            self._decode_op = _build_pallas_kernel_op(dec_spec)
        else:
            self._decode_op = None
        decode_qkv_epi_env = getattr(envs, "SGLANG_TPU_MUSE_DECODE_QKV_EPILOGUE", None)
        decode_gate_fuse_env = getattr(
            envs, "SGLANG_TPU_MUSE_DECODE_ATTN_GATE_FUSE", None
        )
        is_tp16_sq_pool = _pool_is_tp16_sq_64(self.token_to_kv_pool)
        dec_pack = 2 if is_tp16_sq_pool else pack
        if (
            self._dcp == 1
            and (pack > 1 or is_tp16_sq_pool)
            and decode_qkv_epi_env is not None
            and decode_qkv_epi_env.get()
        ):
            dec_prep_spec = (
                _fused_out_spec(dec_pack)
                if (
                    self._fused_out_op is not None
                    or want_decode_fused_out
                    or is_tp16_sq_pool
                )
                else RPA_V3_SPEC
            )
            if is_tp16_sq_pool:
                dec_prep_spec = _tp16_prepared_spec(dec_prep_spec)
            self._decode_prepared_op = _build_pallas_kernel_op(
                _prepared_spec(_skip_mixed_spec(dec_prep_spec))
            )
            if decode_gate_fuse_env is not None and decode_gate_fuse_env.get():
                norm_eps = float(
                    getattr(
                        model_runner.model_config.hf_text_config, "rms_norm_eps", 1e-5
                    )
                )
                gated_base = _fused_out_spec(dec_pack)
                if is_tp16_sq_pool:
                    gated_base = _tp16_prepared_spec(gated_base)
                self._decode_prepared_gated_op = _build_pallas_kernel_op(
                    _decode_gated_spec(
                        _prepared_spec(_skip_mixed_spec(gated_base)),
                        norm_eps,
                    )
                )
            else:
                self._decode_prepared_gated_op = None
        else:
            self._decode_prepared_op = None
            self._decode_prepared_gated_op = None
        a1_rpa_layout_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT", None)
        a3_rpa_out_env = getattr(envs, "SGLANG_TPU_MUSE_KERNEL_A3_RPA_OUT", None)
        rpa_a3_layout_env = getattr(envs, "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT", None)
        prefill_mode = envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.get().strip().lower()
        use_rpa_a3_layout = (
            rpa_a3_layout_env is not None
            and rpa_a3_layout_env.get()
            and prefill_mode in _STATIONARY_Q_MODES
            and not (a3_rpa_out_env is not None and a3_rpa_out_env.get())
        )
        tp16_sq_backend_env = getattr(
            envs, "SGLANG_TPU_RPA_V3_TP16_STATIONARY_Q", None
        )
        is_tp16_sq_pool = _pool_is_tp16_sq_64(self.token_to_kv_pool) and bool(
            tp16_sq_backend_env is None or tp16_sq_backend_env.get()
        )
        prep_pack = -2 if is_tp16_sq_pool else pack
        self._prepared_op = (
            _build_pallas_kernel_op(
                _prepared_a3_layout_spec(prep_pack)
                if use_rpa_a3_layout
                else _prepared_spec(
                    _fused_out_spec(prep_pack)
                    if (
                        (self._fused_out_op is not None or is_tp16_sq_pool)
                        and not (a3_rpa_out_env is not None and a3_rpa_out_env.get())
                    )
                    else RPA_V3_SPEC
                )
            )
            if (a1_rpa_layout_env is not None and a1_rpa_layout_env.get())
            and (pack > 1 or is_tp16_sq_pool)
            and self._dcp == 1
            else None
        )
        # Prewarm custom_op outside torch.compile so Dynamo never traces custom_jax_kernel (#1454).
        try:
            self._write_kv_only_op = (
                _get_write_kv_only_op(self._kv_pack_dense2) if self._dcp == 1 else None
            )
        except ImportError:
            self._write_kv_only_op = None
        if prefill_mode in _STATIONARY_Q_MODES or self._skip_decode_op is not None:
            sq_blk = getattr(envs, "SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES", None)
            sq_blk_sw = getattr(
                envs, "SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES_SW", None
            )
            logger.info(
                "RPA v3 prefill config: mode=%s, skip_empty_decode=%s, "
                "stationary_q_blocks=%s, stationary_q_blocks_sw=%s",
                prefill_mode or "stock",
                envs.SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE.get(),
                sq_blk.get() if sq_blk is not None else "",
                sq_blk_sw.get() if sq_blk_sw is not None else "",
            )

    def _kernel_op(self, forward_batch, use_causal_mask):
        md = self.forward_metadata
        if (
            self._batched_prefill_op is not None
            and use_causal_mask
            # The metadata's distribution is v3's [0, n, n] exactly when it is
            # not decode-shaped; DCP and prefill CP run other kernels or writes.
            and forward_batch is not None
            and not forward_batch.forward_mode.is_decode()
            and md.decode_query_size == 1
            and self._dcp == 1
            and md.cp_write_segments is None
        ):
            return self._batched_prefill_op
        skip_decode_op = getattr(self, "_skip_decode_op", None)
        if (
            skip_decode_op is not None
            and forward_batch is not None
            and not forward_batch.forward_mode.is_decode()
            and md.decode_query_size == 1
            and self._dcp == 1
            and md.cp_write_segments is None
        ):
            return skip_decode_op
        decode_op = getattr(self, "_decode_op", None)
        allow_verify = (
            forward_batch is not None
            and forward_batch.forward_mode.is_target_verify()
            and envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.get()
        )
        if (
            decode_op is not None
            and forward_batch is not None
            and (
                (forward_batch.forward_mode.is_decode() and md.decode_query_size == 1)
                or allow_verify
            )
            and self._dcp == 1
            and md.cp_write_segments is None
        ):
            return decode_op
        if self._fused_out_op is not None:
            return self._fused_out_op
        return self._op

    def write_kv_only(self, kv, layer, forward_batch, chunk_meta=None):
        """Write merged fp8 ``kv`` ``[T, 1, 4, 128]`` into ``layer``'s KV pool without attention."""
        md = chunk_meta if chunk_meta is not None else self.forward_metadata
        if (
            self.kv_cache_quant_dtype is None
            or (chunk_meta is None and md.cp_write_segments is not None)
            or self._dcp != 1
            or layer.k_scale_float not in (None, 1.0)
            or layer.v_scale_float not in (None, 1.0)
        ):
            raise NotImplementedError("write_kv_only needs a scale-1 packed fp8 pool")
        kv_cache, kv_layer_idx = _fused_kv_operands(
            self.token_to_kv_pool, layer.layer_id
        )
        sliding_window = layer.sliding_window_size
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None
        elif sliding_window is not None:
            sliding_window += 1
        is_swa = sliding_window is not None
        if chunk_meta is not None:
            if is_swa and md.swa_page_indices is not None:
                group = self._swa_layer_groups.get(layer.layer_id, 0)
                swa_grp = getattr(md, "swa_group_pages", None)
                if swa_grp is None:
                    swa_grp = md.swa_group_page_indices
                page_indices = (
                    md.swa_page_indices
                    if group == 0 or swa_grp is None
                    else swa_grp[group - 1]
                )
            else:
                page_indices = md.page_indices
            kv_lens = md.kv_lens
        else:
            page_indices = self._layer_page_indices(layer.layer_id, is_swa_layer=is_swa)
            kv_lens = self._kv_lens(forward_batch)
        dense2 = bool(getattr(self, "_kv_pack_dense2", False))
        op = getattr(self, "_write_kv_only_op", None)
        if op is None:
            cached = _WRITE_KV_ONLY_OPS.get(dense2)
            op = cached if cached is not None else _get_write_kv_only_op(dense2)
        op(
            kv,
            kv_cache,
            kv_lens,
            page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
            kv_layer_idx,
        )

    def forward_prepared(self, q, kv, layer, forward_batch, chunk_meta=None):
        """Causal extend on q ``[1, T, Hq / 2, 2, 2 * d]`` and merged fp8 kv ``[T, 1, 4, 2 * d]``
        (A1's RPA layout, scale-1 fp8); returns ``[T, Hq * d]`` like ``_forward``."""
        md = chunk_meta if chunk_meta is not None else self.forward_metadata
        if (
            self._prepared_op is None
            or self.kv_cache_quant_dtype is None
            or (chunk_meta is None and md.cp_write_segments is not None)
            or layer.k_scale_float not in (None, 1.0)
            or layer.v_scale_float not in (None, 1.0)
        ):
            raise NotImplementedError("prepared RPA inputs need a packed fp8 pool")
        num_tokens = q.shape[1]
        kv_cache, kv_layer_idx = _fused_kv_operands(
            self.token_to_kv_pool, layer.layer_id
        )
        sliding_window = layer.sliding_window_size
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None
        elif sliding_window is not None:
            sliding_window += 1  # RPA counts the current token, SGLang does not.
        is_swa = sliding_window is not None
        if chunk_meta is not None:
            if is_swa and md.swa_page_indices is not None:
                group = self._swa_layer_groups.get(layer.layer_id, 0)
                swa_grp = getattr(md, "swa_group_pages", None)
                if swa_grp is None:
                    swa_grp = md.swa_group_page_indices
                page_indices = (
                    md.swa_page_indices
                    if group == 0 or swa_grp is None
                    else swa_grp[group - 1]
                )
            else:
                page_indices = md.page_indices
            kv_lens = md.kv_lens
            decode_query_size = 1
        else:
            page_indices = self._layer_page_indices(layer.layer_id, is_swa_layer=is_swa)
            kv_lens = self._kv_lens(forward_batch)
            decode_query_size = md.decode_query_size
        out = self._prepared_op(
            q,
            kv,
            kv,  # unused (values)
            kv_cache,
            kv_lens,
            page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
            None,
            float(layer.scaling),
            sliding_window,
            layer.logit_cap if layer.logit_cap else None,
            1.0,
            1.0,
            decode_query_size,
            use_causal_mask=True,
            layer_idx=kv_layer_idx,
        )
        if out.ndim == 2 and out.shape == (
            num_tokens,
            layer.tp_q_head_num * layer.qk_head_dim,
        ):
            return out
        out = out.reshape(num_tokens, layer.tp_q_head_num, -1)
        if out.shape[-1] != layer.qk_head_dim:
            out = _unpack_out_heads(
                out, out.shape[-1] // layer.qk_head_dim, layer.qk_head_dim
            )
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.qk_head_dim)

    def can_forward_decode_prepared(self, layer, forward_batch) -> bool:
        """True when decode (or opted-in verify) can consume D4's prepared Q and merged FP8 KV."""
        md = self.forward_metadata
        allow_verify = (
            forward_batch is not None
            and forward_batch.forward_mode.is_target_verify()
            and envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.get()
        )
        return (
            getattr(self, "_decode_prepared_op", None) is not None
            and md is not None
            and forward_batch is not None
            and (
                (forward_batch.forward_mode.is_decode() and md.decode_query_size == 1)
                or allow_verify
            )
            and self._dcp == 1
            and md.cp_write_segments is None
            and self.kv_cache_quant_dtype == torch.float8_e4m3fn
            and not getattr(layer, "is_cross_attention", False)
            and (
                getattr(layer, "tp_q_head_num", 0),
                getattr(layer, "tp_k_head_num", 0),
            )
            in ((16, 2), (8, 1))
            and getattr(layer, "qk_head_dim", 0) == 64
            and getattr(layer, "v_head_dim", 0) == 64
            and not getattr(layer, "logit_cap", 0)
        )

    def forward_decode_prepared(self, q, kv, layer, forward_batch):
        """Single-token decode on D4's prepared ``q`` ``[1, B, 8, 2, 128]`` and merged fp8 ``kv``."""
        md = self.forward_metadata
        if not self.can_forward_decode_prepared(layer, forward_batch):
            raise NotImplementedError(
                "decode prepared RPA inputs need SGLANG_TPU_MUSE_DECODE_QKV_EPILOGUE=1 "
                "on a packed fp8 16Q/2KV d=64 decode step"
            )
        num_tokens = q.shape[1]
        kv_cache, kv_layer_idx = _fused_kv_operands(
            self.token_to_kv_pool, layer.layer_id
        )
        sliding_window = getattr(layer, "sliding_window_size", None)
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None
        elif sliding_window is not None:
            sliding_window += 1
        page_indices = self._layer_page_indices(
            layer.layer_id, is_swa_layer=sliding_window is not None
        )
        k_scale = float(layer.k_scale_float if layer.k_scale_float is not None else 1.0)
        v_scale = float(layer.v_scale_float if layer.v_scale_float is not None else 1.0)
        out = self._decode_prepared_op(
            q,
            kv,
            kv,  # unused (values)
            kv_cache,
            self._kv_lens(forward_batch),
            page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            self._layer_distribution(is_swa_layer=sliding_window is not None),
            None,
            float(layer.scaling),
            sliding_window,
            None,
            k_scale,
            v_scale,
            md.decode_query_size,
            use_causal_mask=True,
            layer_idx=kv_layer_idx,
        )
        capture_mode = getattr(forward_batch, "capture_hidden_mode", None)
        if capture_mode is not None and capture_mode.is_full():
            out = _zero_inactive_rpa_rows(out, md.active_num_tokens)
        expected_out_cols = layer.tp_q_head_num * layer.v_head_dim
        if out.ndim == 2 and out.shape[-1] == 2 * expected_out_cols:
            out = out[:, :expected_out_cols]
        elif out.ndim == 3 and out.shape[1] == 2 * layer.tp_q_head_num:
            out = out[:, : layer.tp_q_head_num, :]
        out = out.reshape(num_tokens, layer.tp_q_head_num, -1)
        if out.shape[-1] != layer.qk_head_dim:
            out = _unpack_out_heads(
                out, out.shape[-1] // layer.qk_head_dim, layer.qk_head_dim
            )
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode_prepared_gated(self, q, kv, gate, layer, forward_batch):
        """Single-token decode fusing post-RPA per-head RMSNorm + ``sigmoid(gate) * out``."""
        gated_op = getattr(self, "_decode_prepared_gated_op", None)
        if gated_op is None or not self.can_forward_decode_prepared(
            layer, forward_batch
        ):
            return None
        md = self.forward_metadata
        num_tokens = q.shape[1]
        if gate.shape != (num_tokens, layer.tp_q_head_num * layer.v_head_dim):
            return None
        kv_cache, kv_layer_idx = _fused_kv_operands(
            self.token_to_kv_pool, layer.layer_id
        )
        sliding_window = getattr(layer, "sliding_window_size", None)
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None
        elif sliding_window is not None:
            sliding_window += 1
        page_indices = self._layer_page_indices(
            layer.layer_id, is_swa_layer=sliding_window is not None
        )
        k_scale = float(layer.k_scale_float if layer.k_scale_float is not None else 1.0)
        v_scale = float(layer.v_scale_float if layer.v_scale_float is not None else 1.0)
        out = gated_op(
            q,
            kv,
            gate.contiguous(),
            kv_cache,
            self._kv_lens(forward_batch),
            page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            self._layer_distribution(is_swa_layer=sliding_window is not None),
            None,
            float(layer.scaling),
            sliding_window,
            None,
            k_scale,
            v_scale,
            md.decode_query_size,
            use_causal_mask=True,
            layer_idx=kv_layer_idx,
        )
        capture_mode = getattr(forward_batch, "capture_hidden_mode", None)
        if capture_mode is not None and capture_mode.is_full():
            out = _zero_inactive_rpa_rows(out, md.active_num_tokens)
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs: Optional[int] = None
    ):
        # RPA-v3 rebuilds verify metadata and has no graph-capture buffers.
        pass


_WRITE_KV_ONLY_OPS: dict[bool, object] = {}


def _get_write_kv_only_op(kv_pack_dense2: bool = False):
    cached = _WRITE_KV_ONLY_OPS.get(kv_pack_dense2)
    if cached is not None:
        return cached

    import jax
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
        write_kv_only_rpa_v3,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    op_name = (
        "sglang_tpu_pallas::rpa_v3_write_kv_only_d2"
        if kv_pack_dense2
        else "sglang_tpu_pallas::rpa_v3_write_kv_only"
    )

    def _wrapped(
        kv_new: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        layer_idx: jax.Array | None = None,
    ) -> jax.Array:
        kv_cache_flat = kv_cache_fused
        if layer_idx is not None:
            num_blocks, layer_num = kv_cache_fused.shape[:2]
            kv_cache_flat = kv_cache_fused.reshape(
                (num_blocks * layer_num,) + kv_cache_fused.shape[2:]
            )
            page_indices = page_indices * layer_num + layer_idx
        updated = write_kv_only_rpa_v3(
            kv_new,
            kv_cache_flat,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            kv_pack_dense2=kv_pack_dense2,
        )
        return updated.reshape(kv_cache_fused.shape)

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=(),
        donate_argnums=(1,),
        input_output_aliases={1: 0},
        warn_deprecated=False,
    )

    def _custom_op_device_types():
        try:
            torch._C._dispatch_key_for_device("tpu")
            return ["tpu"]
        except RuntimeError:
            return None

    @torch.library.custom_op(
        op_name,
        mutates_args=("kv_cache_fused",),
        device_types=_custom_op_device_types(),
    )
    def write_kv_only_op(
        kv_new: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        cu_kv_lens: torch.Tensor,
        distribution: torch.Tensor,
        layer_idx: torch.Tensor | None = None,
    ) -> None:
        with tpu_info_for_cpu_export():
            jax_kernel(
                kv_new,
                kv_cache_fused,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                layer_idx,
            )

    @write_kv_only_op.register_fake
    def _(*args, **kwargs) -> None:
        return None

    _WRITE_KV_ONLY_OPS[kv_pack_dense2] = write_kv_only_op
    return write_kv_only_op
