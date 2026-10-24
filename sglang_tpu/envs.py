# SPDX-License-Identifier: Apache-2.0
"""Environment variables owned by the TPU plugin.

Declare new TPU-only variables here with the descriptor types from
`sglang.srt.environ`. Existing ``SGLANG_TPU_*`` entries stay in core until their
core readers move; never declare one name in both registries.
"""

from __future__ import annotations

from sglang.srt.environ import EnvBool, EnvField, EnvFloat, EnvInt, EnvStr

__all__ = ["TpuEnvs", "envs"]

# Core closes EnvField.__set_name__ after its own Envs, reopen it for this class.
EnvField._allow_set_name = True


class TpuEnvs:
    # Host heap (see host_heap_trim.py)
    SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED = EnvBool(True)
    # Seconds between periodic trims. 0 turns the periodic trim off.
    SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC = EnvFloat(0.0)
    # MoE: one fused kernel for single-token packed-MXFP4 decode
    # (moe/pallas/single_token_moe.py); False keeps the gmm_v2 path.
    SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL = EnvBool(True)
    # MoE: one kernel streaming the selected FP8 experts' weights for 1..8 tokens
    # (moe/pallas/moe_stream.py); False keeps the gmm_v2 path.
    SGLANG_TPU_ENABLE_STREAMING_MOE_KERNEL = EnvBool(False)
    # FP8 linear: one W8A8 kernel for decode-sized batches (kernels/fp8_linear_small_m.py);
    # False keeps the XLA path.
    SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL = EnvBool(False)
    # RPA v3: sum the softmax row on the MXU through a ones column in V's last
    # padding lane instead of on the vector ALU. Only applies when head_dim is
    # not a multiple of 128 (e.g. 64).
    SGLANG_TPU_ENABLE_RPA_PAD_LANE_ROWSUM = EnvBool(False)
    # Launch-time xprof capture (see profile_auto.py): each stage once, without
    # /start_profile. Keep the step count small: each chip's trace is hundreds
    # of MB.
    SGLANG_TPU_ENABLE_PROFILE_AUTO = EnvBool(False)
    SGLANG_TPU_PROFILE_AUTO_STEPS = EnvInt(5)
    # Kimi-K3: run attention-residual aggregation for decode-sized batches as
    # one Pallas kernel (pending add + mix + out norm) instead of the torch
    # reference (see kimi_attn_residual.py and hooks/attn_residual.py). Not
    # bit-exact with it: the kernel scores in f32, the torch path in bf16.
    SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL = EnvBool(False)
    # GLM-5.3-Flash: one fused Pallas kernel per mHC site instead of the torch
    # hc_pre / hc_post (see mhc.py and hooks/mhc.py).
    SGLANG_TPU_ENABLE_GLM5_NEXT_FUSED_MHC = EnvBool(False)
    # Drop Dynamo guards on nn.Module state (attributes, submodules, parameters)
    # for the decoder graph; input and global guards stay (see
    # decoder_compile_policy._guard_filter_options). Assumes module state is
    # fixed after warmup.
    SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER = EnvBool(False)

    # Speculative decoding
    # Route DFLASH/DSPARK to sglang_tpu.experimental.TpuBlockSpecWorker; SGLANG_TPU_HOST_PADDING_NO_D2H also does.
    SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER = EnvBool(False)
    # DFLASH draft linears, including the DFlash2 conv kernel projections, as fp8
    # weight-only (fp8_weight_linear.py) for decode-sized calls; prefill-sized calls
    # keep the bf16 weights. Changes the draft's proposals, not verification.
    SGLANG_TPU_DFLASH_FP8_DRAFT = EnvBool(False)

    # Muse Spark: whole-decoder Pallas megakernel for decode and DFlash target-verify
    # steps, with fp8 dense weight copies and an fp8 weight-only LM head.
    SGLANG_TPU_MUSE_MEGAKERNEL = EnvBool(False)
    # Largest megakernel step in rows (running requests x verify tokens, padded to the
    # TP size); larger steps run the native decoder. Steps above 32 rows use the
    # large-step paths (ops._BASE_STEP_MAX_ROWS).
    SGLANG_TPU_MUSE_MK_MAX_ROWS = EnvInt(32)
    # DFlash: verify only the first WIDTH tokens of each draft block once the decode batch
    # reaches MIN_BS requests (dflash_adaptive_verify.py); "" verifies the full block.
    SGLANG_TPU_DFLASH_ADAPTIVE_VERIFY = EnvStr("")
    # > 0: megakernel sliding-window layers on SEQ_ALONG_LANE pools use KV chunks of this many
    # tokens (a divisor of the 2 * inter_local-token chunk); their 2048-token window gains nothing.
    SGLANG_TPU_MUSE_MK_SAL_SWA_CHUNK_TOKENS = EnvInt(0)
    # Megakernel attention: a softmax stream past the request's last chunk skips its iteration
    # (MKConfig.attn_skip_idle_streams); and, with SWA chunks set, a full-attention layer's last
    # chunk is masked by key column only (MKConfig.attn_col_mask).
    SGLANG_TPU_MUSE_MK_ATTN_SKIP_IDLE = EnvBool(False)
    SGLANG_TPU_MUSE_MK_ATTN_COL_MASK = EnvBool(False)
    # Fused KV pool stores a rank's heads side by side in one 128-lane row when they
    # fit exactly (e.g. 2 x head_dim 64): half the bytes of a zero-padded row per head.
    SGLANG_TPU_KV_PACK_HEADS = EnvBool(False)
    # Fused KV pool and RPA v3 pad head_dim to this before it becomes the minor (lane)
    # axis. 64 halves an unpacked head_dim-64 pool; fp8 then reaches 256 B per token at
    # two KV heads per rank. Must stay >= the sublane x packing tile.
    SGLANG_TPU_KV_HEAD_DIM_ALIGN = EnvInt(128)
    # Fused-RS NVFP4 experts keep the checkpoint's e4m3 block scales and apply the
    # per-expert global scales after each dot: half the scale bytes of folded bf16.
    SGLANG_TPU_NVFP4_E4M3_SCALES = EnvBool(False)
    # Megakernel MoE: send each row's expert outputs to their reduce-scatter owner as
    # soon as the row's last expert is done (MKConfig.y_early).
    SGLANG_TPU_MK_Y_EARLY = EnvBool(False)
    # Megakernel MoE: run the experts owner block by owner block and send each block of
    # reduce-scatter rows to its owner as one DMA while the remaining experts run
    # (MKConfig.y_block_send; exact). Ignored where SGLANG_TPU_MK_Y_EARLY applies.
    SGLANG_TPU_MK_Y_BLOCK_SEND = EnvBool(True)
    # Megakernel attention on fp8 SEQ_ALONG_LANE pools: feed K (and V) to the MXU as fp8
    # instead of widening each chunk to bf16 on the VPU first. 0 = off, 1 = K only, 2 = K and V
    # (exact; measured at 32 rows / 100K context: 1 is -1.8%, 2 is +1% slower than 0).
    SGLANG_TPU_MK_SAL_FP8_DOT = EnvInt(0)
    # Profiling only: name megakernel regions in xprof; needs libtpu's
    # --xla_enable_custom_call_region_trace=true and slows fine-grained regions.
    SGLANG_TPU_MK_TRACE_REGIONS = EnvBool(False)
    # Tuning only: declared-scratch budget cut per row past 64 rows (KiB), for the VMEM sweep.
    SGLANG_TPU_MK_LARGE_STEP_BUDGET_PER_ROW_KB = EnvInt(None)
    # Tuning only: scoped VMEM limit (MiB) for steps past 64 rows (default 60; 64 MiB physical).
    SGLANG_TPU_MK_LARGE_STEP_VMEM_LIMIT_MB = EnvFloat(None)
    # Tuning only: force the 16-bit packed page tables on (1) or off (0); unset = by table size.
    SGLANG_TPU_MK_PACK_PAGE_TABLES = EnvInt(None)
    # Megakernel attention: fold a power-of-two attn_scale into the bf16 queries instead of
    # scaling every score (exact).
    SGLANG_TPU_MK_FOLD_ATTN_SCALE = EnvBool(False)
    # Muse megakernel: fold the f / o all-gathers into the next qkvg and the pool + router
    # matvecs (not bit-exact: the matvec input rounding moves; verify steps only).
    SGLANG_TPU_MUSE_MK_FOLD = EnvBool(True)
    # Raiden HiCache transfers move each run of pages that is consecutive on both the
    # device and the host as one segment instead of one segment per page.
    SGLANG_TPU_RAIDEN_COALESCE_PAGES = EnvBool(False)
    # Fused-RS expert TP passes each device only its own block of routed rows, so a
    # call is no longer capped at 1/ep_size of the tokens and the expert weights are
    # streamed once per chunk rather than once per capped call.
    SGLANG_TPU_FUSED_RS_ETP_LOCAL_ROWS = EnvBool(False)
    # Expert-TP fused-RS: sum the per-device partial rows in the kernel's tiled
    # (rows, hidden // 128, 128) layout instead of relayouting them to 2D first.
    SGLANG_TPU_FUSED_RS_ETP_TILED_REDUCE = EnvBool(False)
    # Expert-TP fused-RS: skip zeroing the output buffer, which the per-row writes
    # cover exactly once.
    SGLANG_TPU_FUSED_RS_ETP_SKIP_OUT_ZERO = EnvBool(False)
    # Fused-RS NVFP4 GMMs widen fp4 weights to bf16 through fp8 (bit-identical, fewer VPU ops).
    SGLANG_TPU_FP4_DEQUANT_VIA_FP8 = EnvBool(False)
    # Tuning only: comma list of fused-RS row buckets to keep (the tile_m bucket always stays).
    SGLANG_TPU_FUSED_RS_M_BUCKETS = EnvStr(None)
    # Fused-RS tiles (tile_m,k1,n1,k2,n2,w1_bufs,w2_bufs) for EP-layout prefill calls
    # (>= 32768 routed rows), overriding the tuned table.
    SGLANG_TPU_FUSED_RS_EP_PREFILL_TILES = EnvStr(None)
    # Fused-RS EP mode leaves each tile's output row writes in flight until its staging
    # slot is reused, as expert-TP mode does, instead of waiting for them after every tile.
    SGLANG_TPU_FUSED_RS_EP_DEFER_WRITES = EnvBool(False)
    # Expert-TP fused-RS with a post-expert norm: ep sum, norm and top-k combine in one Pallas
    # pass instead of three XLA fusions. Equal up to f32 summation order (bf16 casts kept).
    SGLANG_TPU_FUSED_RS_ETP_FUSED_COMBINE = EnvBool(False)
    # Fused-RS: rows issued per iteration of the per-row gather / write DMA loops.
    SGLANG_TPU_FUSED_RS_ROW_LOOP_UNROLL = EnvInt(1)
    # Fused-RS: output staging buffers a tile's deferred direct writes stay in flight
    # across (VMEM: slots x tile_m x hidden x 2 B). Measured on the Muse Spark expert-TP
    # call: 6 slots -1.5% per layer vs 3, 8 slots -2%.
    SGLANG_TPU_FUSED_RS_STAGING_SLOTS = EnvInt(3)
    # Batched RPA: run target-verify rows (speculative_num_draft_tokens per request) as MIXED
    # with a q block of that width; the prefill block search otherwise pads each request's
    # rows to a ~2048-row block (16 x 8 rows at 150K context: 42 -> 2.2 ms per layer).
    SGLANG_TPU_RPA_BATCHED_VERIFY_BLOCKS = EnvBool(False)
    # Raiden HiCache: per-rank host RAM budget (GB) for the full, SWA and draft host pools
    # together, replacing the free-RAM / local-ranks sample taken as each rank arrives.
    SGLANG_TPU_HICACHE_HOST_BUDGET_GB = EnvFloat(None)
    # Raiden HiCache: before building the host pools, release freed heap pages and wait for
    # every rank, so no rank allocates while a peer still holds its weight-loading buffers.
    SGLANG_TPU_HICACHE_TRIM_BEFORE_ALLOC = EnvBool(False)
    # Batched RPA prefill: tiles whose keys all precede their queries skip the causal
    # (and sliding-window) mask, and a power-of-two sm_scale is folded into q. Exact.
    SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK = EnvBool(False)
    # Batched RPA prefill softmax (the per-score vector chain, not the MXU, sets the
    # time): exp2 with log2e folded into the q scale, one VALU op per score fewer.
    # Off with an attention sink or an LSE output, which would see base-2 m and l;
    # the folded scale rounds q in bf16, so the output is not bit-identical.
    SGLANG_TPU_RPA_EXP2_SOFTMAX = EnvBool(False)
    # Row sums from the PV matmul: V tiles get ones rows in the lanes head_dim 64
    # leaves idle on the MXU, so the accumulator carries l and the vector row-sum
    # reduce and the separate l rescale go away. SEQ_ALONG_LANE, head_dim <= 64.
    SGLANG_TPU_RPA_ONES_V = EnvBool(False)
    # Prefill attention tiles with transposed scores (keys along sublanes) and
    # fp8 q; not bit-exact with the default tile. Mixed/prefill, SEQ_ALONG_LANE,
    # head_dim <= 64, no sink / LSE / soft cap / cache-only / context parallel.
    SGLANG_TPU_RPA_TRANSPOSED = EnvBool(False)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False
