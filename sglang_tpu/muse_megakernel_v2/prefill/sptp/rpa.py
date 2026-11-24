# SPDX-License-Identifier: Apache-2.0
"""Batched RPA for the SP + TP prefill with q already in the kernel's layout and the raw output.

batched_rpa.wrapper.ragged_paged_attention relayouts q [T, H, D] to [Hkv, T, G / pack, pack, D128]
and the output back, about 3 ms per chunk of XLA copies for Muse; the sptp qkvg card writes that
layout itself. This mirrors the wrapper's prefill path for this one case (SEQ_ALONG_LANE KV, no
sink, no lse, no context parallel, causal) by its helpers; it duplicates the wrapper's config
assembly, so a shared seam would replace it if the layout is adopted."""

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.batched_rpa import configs, kernel, schedule, utils
from sglang_tpu.attention.pallas.batched_rpa.wrapper import calculate_block_sizes
from sglang_tpu.envs import envs as tpu_envs


def new_kv(k, v, page_size: int):
    """wrapper.prepare_inputs' SEQ_ALONG_LANE new-KV operand of fp8 k, v [T, Hkv, D]."""
    t, hkv, d = k.shape
    pack = utils.get_dtype_packing(k.dtype)
    lanes = pltpu.get_tpu_info().num_lanes
    sub = pltpu.get_tpu_info().num_sublanes
    dk = utils.align_to(d, sub * pack)
    tp = utils.align_to(t, max(lanes, page_size))
    kv = jnp.pad(
        jnp.concatenate([k, v], axis=-1).reshape(t, 2 * hkv, d),
        ((0, tp - t), (0, 0), (0, dk - d)),
    )
    return kv.reshape(tp, 2 * hkv, dk // pack, pack).transpose(1, 2, 3, 0)


def rpa_prepared(
    q_hbm,
    k,
    v,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    num_q_heads: int,
    sm_scale: float,
    sliding_window: int | None,
):
    """q_hbm [Hkv, T, G / pack, pack, D128] bf16 (pad lanes zero), fp8 k / v [T, Hkv, D] ->
    (o [Hkv, T, G / pack, pack, D128] bf16, kv_cache)."""
    t, hkv, d = k.shape
    q_dtype = q_hbm.dtype
    queries = jax.ShapeDtypeStruct((t, num_q_heads, d), q_dtype)
    kv_layout = configs.KVLayout.SEQ_ALONG_LANE
    page_size = kv_cache.shape[4]
    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=hkv,
        head_dim=d,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=None,
        mask_value=jnp.finfo(q_dtype).min,
        has_attention_sink=False,
    )
    serve_cfgs = configs.ServingConfigs(
        num_seqs=kv_lens.shape[0],
        num_page_indices=page_indices.shape[0],
        total_q_tokens=t,
        dtype_q=q_dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=q_dtype,
        page_size=page_size,
        scale_q=None,
        scale_k=1.0,
        scale_v=1.0,
        kv_layout=kv_layout,
        cache_only=False,
        skip_kv_write=False,
        causal=True,
        return_lse=False,
        cp_group_size=None,
        skip_full_tile_mask=tpu_envs.SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK.get(),
        no_max=tpu_envs.SGLANG_TPU_RPA_NO_MAX.get(),
        exp2_softmax=tpu_envs.SGLANG_TPU_RPA_EXP2_SOFTMAX.get(),
        ones_v=tpu_envs.SGLANG_TPU_RPA_ONES_V.get() and d <= 64,
        transposed_key_chunk=tpu_envs.SGLANG_TPU_RPA_TRANSPOSED_KEY_CHUNK.get(),
        transposed_valu_exp2_every=tpu_envs.SGLANG_TPU_RPA_VALU_EXP2_EVERY.get(),
        transposed_fp8_p=tpu_envs.SGLANG_TPU_RPA_TRANSPOSED_FP8_P.get(),
        transposed_chunk_vt=tpu_envs.SGLANG_TPU_RPA_CHUNK_VT.get(),
        transposed_s_bf16=tpu_envs.SGLANG_TPU_RPA_S_BF16.get(),
        transposed=tpu_envs.SGLANG_TPU_RPA_TRANSPOSED.get() and d <= 64,
        skip_pad_q=tpu_envs.SGLANG_TPU_RPA_SKIP_PAD_Q.get(),
        skip_pad_q_group=tpu_envs.SGLANG_TPU_RPA_SKIP_PAD_Q_GROUP.get(),
        # the program normalizes each head itself
        out_head_norm_eps=None,
    )
    new_kv_hbm = new_kv(k, v, page_size)
    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs, serve_cfgs, vmem_limit_bytes
    )
    blocks_env = (
        tpu_envs.SGLANG_TPU_RPA_PREFILL_BLOCKS
        if sliding_window is None
        else tpu_envs.SGLANG_TPU_RPA_SWA_PREFILL_BLOCKS
    ).get()
    prefill_blocks = default_prefill
    if blocks_env:
        bq_sz, bkv_sz = (int(x) for x in blocks_env.split(":"))
        prefill_blocks = dataclasses.replace(
            default_prefill,
            bq_sz=bq_sz,
            bq_c_sz=math.gcd(bq_sz, default_prefill.bq_c_sz),
            bkv_sz=bkv_sz,
        )

    def run(mode, o, cache):
        cfgs = configs.RpaConfigs(
            block=default_decode if mode == configs.RpaCase.DECODE else prefill_blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        cfgs.validate_inputs(
            q=queries,
            k=k,
            v=v,
            kv_cache=cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            cp_rank=None,
            kv_lens_global=None,
        )
        sched = schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_lens,
            distribution,
            cfgs=cfgs,
            skip_kv_update=False,
            cp_rank=None,
            kv_lens_global=None,
        )
        return kernel.rpa_kernel(
            cu_q_lens,
            kv_lens,
            page_indices,
            sched,
            o,
            new_kv_hbm,
            cache,
            None,
            cfgs=cfgs,
            lse_hbm=None,
            cp_rank=None,
            kv_lens_global=None,
        )

    o, kv_cache, _ = run(configs.RpaCase.DECODE, q_hbm, kv_cache)
    o, kv_cache, _ = run(configs.RpaCase.MIXED, o, kv_cache)
    return o, kv_cache
