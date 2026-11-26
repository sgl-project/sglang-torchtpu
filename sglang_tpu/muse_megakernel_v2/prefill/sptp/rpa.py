# SPDX-License-Identifier: Apache-2.0
"""Batched RPA for the SP + TP prefill with q already in the kernel's layout and the raw output.

batched_rpa.wrapper.ragged_paged_attention relayouts q [T, H, D] to [Hkv, T, G / pack, pack, D128]
and the output back, about 3 ms per chunk of XLA copies for Muse; the sptp qkvg card writes that
layout itself. This mirrors the wrapper's prefill path for this one case (SEQ_ALONG_LANE KV, no
sink, no lse, no context parallel, causal) by its helpers; it duplicates the wrapper's config
assembly, so a shared seam would replace it if the layout is adopted."""

import dataclasses
import functools
import math
import os

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.batched_rpa import configs, kernel, schedule, utils
from sglang_tpu.attention.pallas.batched_rpa.wrapper import calculate_block_sizes
from sglang_tpu.envs import envs as tpu_envs


@dataclasses.dataclass(frozen=True)
class RpaOpts:
    """sptp-local kernel options (build-time env SPTP_RPA_*); the defaults are the shared kernel."""

    # transposed_tri: aligned diagonal / window-edge tiles score only their visible keys
    tri: bool = False
    # transposed_qt_bf16: Q^T transposed in bf16
    qt_bf16: bool = False
    # "bq:bkv" of global / SWA layers over SGLANG_TPU_RPA_(SWA_)PREFILL_BLOCKS
    global_blocks: str = ""
    swa_blocks: str = ""
    # new KV operand: 1 one stack + transpose; 2 a transpose of 32-bit words (4 fp8 head
    # lanes of one token, the operand's tiled layout) instead of concat / pad / relayout
    kv_stack: int = 0
    # no decode pass: the caller guarantees distribution[0] == 0 (a prefill-only batch,
    # as the sptp prefill program serves one chunk)
    prefill_only: bool = False
    # transposed_out_tok: o[0] holds the output token major (tok_rows)
    out_tok: bool = False

    @classmethod
    def from_env(cls) -> "RpaOpts":
        env = os.environ.get
        return cls(
            tri=env("SPTP_RPA_TRI", "1") == "1",
            qt_bf16=env("SPTP_RPA_QTB", "1") == "1",
            global_blocks=env("SPTP_RPA_GBLOCKS", "1024:1024"),
            swa_blocks=env("SPTP_RPA_SBLOCKS", ""),
            kv_stack=int(env("SPTP_RPA_KV_STACK", "0")),
            prefill_only=env("SPTP_RPA_PREFILL_ONLY", "1") == "1",
            out_tok=env("SPTP_RPA_OUT_TOK", "0") == "1",
        )


def tok_rows(o):
    """[T, Hkv * G * D] bf16 (head h = kv * G + g) of an RpaOpts(out_tok=True) output (an XLA
    relayout of slot 0; a Pallas consumer reads o[0] = [T, G / pack, pack, 128] itself)."""
    _, t, gp, pack, lanes = o.shape
    return o[0].reshape(t, gp * pack * lanes)


def new_kv_stacked(k, v, page_size: int):
    """new_kv by stack + transpose: [2 * Hkv (k0, v0, k1, v1), D / pack, pack, T]."""
    t, hkv, d = k.shape
    pack = utils.get_dtype_packing(k.dtype)
    lanes = pltpu.get_tpu_info().num_lanes
    sub = pltpu.get_tpu_info().num_sublanes
    assert d % (sub * pack) == 0 and t % max(lanes, page_size) == 0, (
        k.shape,
        page_size,
    )
    kv = jnp.stack([k, v], axis=2).reshape(t, 2 * hkv, d // pack, pack)
    return kv.transpose(1, 2, 3, 0)


def new_kv_words(k, v, page_size: int):
    """new_kv_stacked through 32-bit words: [T, 2 Hkv D] fp8 -> u32 [T, 2 Hkv D / 4] ->
    transposed -> fp8 [2 Hkv, D / 4, 4, T] (4 head lanes of one token per word, as tiled)."""
    t, hkv, d = k.shape
    pack = utils.get_dtype_packing(k.dtype)
    assert d % pack == 0 and t % max(pltpu.get_tpu_info().num_lanes, page_size) == 0
    kv = jnp.stack([k, v], axis=2).reshape(t, 2 * hkv * d // pack, pack)
    w = jax.lax.bitcast_convert_type(kv, jnp.uint32).T  # [2 Hkv D / 4, T]
    kv8 = jax.lax.bitcast_convert_type(w, k.dtype)  # [2 Hkv D / 4, T, 4]
    return kv8.reshape(2 * hkv, d // pack, t, pack).transpose(0, 1, 3, 2)


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


# jitted like the wrapper: one lowered computation per (heads, window) instead of a Mosaic
# compile per layer (inlined: ~530 s vs ~140 s for the 62-layer program)
def rpa_prepared(*args, opts: "RpaOpts | None" = None, **kw):
    """_rpa_prepared with the build-time env resolved here, so a jit cache entry never
    outlives an env change (warm benchmark workers)."""
    env = tuple(
        sorted((k, v) for k, v in os.environ.items() if k.startswith("SGLANG_TPU_RPA_"))
    )
    return _rpa_prepared(
        *args, opts=RpaOpts.from_env() if opts is None else opts, env=env, **kw
    )


@functools.partial(
    jax.jit,
    static_argnames=("num_q_heads", "sm_scale", "sliding_window", "opts", "env"),
)
def _rpa_prepared(
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
    opts: RpaOpts,
    env: tuple,
):
    """q_hbm [Hkv, T, G / pack, pack, D128] bf16 (pad lanes zero), fp8 k / v [T, Hkv, D] ->
    (o [Hkv, T, G / pack, pack, D128] bf16, kv_cache). v None: k is the new-KV operand
    (new_kv's layout, e.g. xag_qkvg kv_rpa)."""
    del env  # cache key only
    new_kv_hbm = None
    if v is None:  # k: the new-KV operand already built ([2 Hkv, D / pack, pack, T])
        new_kv_hbm = k
        h2, dp, pk, tk = k.shape
        k = v = jax.ShapeDtypeStruct((tk, h2 // 2, dp * pk), k.dtype)
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
        transposed_tri=opts.tri,
        transposed_qt_bf16=opts.qt_bf16,
        transposed_out_tok=opts.out_tok,
        transposed=tpu_envs.SGLANG_TPU_RPA_TRANSPOSED.get() and d <= 64,
        skip_pad_q=tpu_envs.SGLANG_TPU_RPA_SKIP_PAD_Q.get(),
        skip_pad_q_group=tpu_envs.SGLANG_TPU_RPA_SKIP_PAD_Q_GROUP.get(),
        # the program normalizes each head itself
        out_head_norm_eps=None,
    )
    if new_kv_hbm is None:
        new_kv_hbm = {0: new_kv, 1: new_kv_stacked, 2: new_kv_words}[opts.kv_stack](
            k, v, page_size
        )
    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs, serve_cfgs, vmem_limit_bytes
    )
    blocks_env = (
        opts.global_blocks if sliding_window is None else opts.swa_blocks
    ) or (
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

    o = q_hbm
    if not opts.prefill_only:
        o, kv_cache, _ = run(configs.RpaCase.DECODE, o, kv_cache)
    o, kv_cache, _ = run(configs.RpaCase.MIXED, o, kv_cache)
    return o, kv_cache
