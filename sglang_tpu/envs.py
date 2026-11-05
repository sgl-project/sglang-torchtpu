# SPDX-License-Identifier: Apache-2.0
"""Environment variables owned by the TPU plugin.

Declare new TPU-only variables here with the descriptor types from
`sglang.srt.environ`. Existing ``SGLANG_TPU_*`` entries stay in core until their
core readers move; never declare one name in both registries.
"""

from __future__ import annotations

from sglang.srt.environ import EnvBool, EnvField, EnvFloat, EnvInt, EnvStr, EnvTuple

__all__ = ["TpuEnvs", "envs"]

# Core closes EnvField.__set_name__ after its own Envs, reopen it for this class.
EnvField._allow_set_name = True


class TpuEnvs:
    # Opt-in, request-tagged benchmark admission (benchmark_batch_gate.py).
    # Zero preserves ordinary scheduling, including continuous batching.
    SGLANG_TPU_BENCHMARK_BATCH_SIZE = EnvInt(0)
    SGLANG_TPU_BENCHMARK_TIMEOUT_SEC = EnvFloat(600.0)
    SGLANG_TPU_RELEASE_CHECKPOINT_PAGE_CACHE = EnvBool(True)
    SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS = EnvBool(True)
    # Inkling short convolution on the Pallas kernel instead of XLA ops. Off: the Pallas
    # path gives wrong compiled output for Inkling2 (2026-10-30).
    SGLANG_TPU_ENABLE_INKLING_SCONV_PALLAS = EnvBool(False)
    # Inkling2: serve NVFP4 experts packed (W4A16 fused-RS) instead of block FP8.
    SGLANG_TPU_INKLING2_NVFP4_EXPERTS = EnvBool(False)
    # Inkling2 prefill: block AttnRes + the following RMSNorm as one Pallas kernel.
    SGLANG_TPU_INKLING2_ATTN_RES_PALLAS = EnvBool(True)
    # Host heap (see host_heap_trim.py)
    SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED = EnvBool(True)
    # Seconds between periodic trims. 0 turns the periodic trim off.
    SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC = EnvFloat(0.0)
    # Kimi-K3: run these RMSNorm-fed dense decode GEMVs as fp8 W8A8 with a static
    # input scale (see kimi_k3/static_fp8.py). Comma-separated subset of
    # moe_front, latent_up, kda_qkvg, mla_qkv_a, or "all" (every validated part:
    # all but mla_qkv_a). Empty keeps bf16.
    SGLANG_TPU_K3_STATIC_FP8 = EnvTuple(())
    # Run those static fp8 GEMVs as one Pallas kernel for decode-sized batches
    # (kernels/static_fp8_gemv.py); False keeps the XLA fp8 dot for every batch.
    SGLANG_TPU_ENABLE_K3_STATIC_FP8_PALLAS_GEMV = EnvBool(True)
    # Kimi-K3 single-token decode: split the replicated latent down / up-proj
    # and router gate GEMVs by output columns across the ranks of a host and
    # exchange the columns inside the host (host_split_gemv.py). Needs
    # SGLANG_TPU_K3_STATIC_FP8 moe_front and latent_up (checked at startup).
    SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV = EnvBool(False)
    # MoE: one fused kernel for single-token packed-MXFP4 decode
    # (moe/pallas/single_token_moe.py); False keeps the gmm_v2 path.
    SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL = EnvBool(True)
    # MoE: one kernel streaming the selected FP8 experts' weights for 1..8 tokens
    # (moe/pallas/moe_stream.py); False keeps the gmm_v2 path.
    SGLANG_TPU_ENABLE_STREAMING_MOE_KERNEL = EnvBool(False)
    # FP8 linear: one W8A8 kernel for decode-sized batches (kernels/fp8_linear_small_m.py);
    # False keeps the XLA path.
    SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL = EnvBool(False)
    # Sparse MLA: pooled-DSA batches of 512+ tokens (prefill) through the
    # query-blocked kernel (attention/pallas/mla_sparse_qblock.py), which fetches
    # each block's union of selected pages once; False keeps the per-query kernel.
    SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL = EnvBool(False)
    # Vision: run single-sequence "tpu" vision attention through block-skipping splash
    # (attention/vision_splash.py); False keeps the segment-id flash op.
    SGLANG_TPU_ENABLE_VISION_SPLASH_ATTENTION = EnvBool(False)
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
    # Two-level one-shot Pallas all-reduce for single-token TP decode tensors
    # (twolevel_allreduce.py): host-local exchange, then cross-host. Applies
    # when TP spans the whole world; everything else keeps XLA's all-reduce.
    SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE = EnvBool(False)
    # MoE router: one Pallas kernel for the biased sigmoid top-k of decode-sized
    # batches (kernels/router_topk.py); False keeps the generic top-k.
    SGLANG_TPU_ENABLE_ROUTER_TOPK_KERNEL = EnvBool(True)
    # GLM-5.3-Flash: single-request target steps (the MTP verify, or decode
    # without speculation) as one Pallas call per device (glm_flash_megakernel/);
    # other steps run the model as usual. ..._MAX_LEN caps the request length the
    # kernel takes (default: the context length).
    SGLANG_TPU_GLM_FLASH_MEGAKERNEL = EnvBool(False)
    SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN = EnvInt(None)
    # Pin each scheduler's main thread to one CPU of its rank's set and keep its
    # other threads off that physical core (hooks/cpu_pinning.py).
    SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING = EnvBool(False)
    # Kimi-K3: run decode steps and prefill chunks as one Pallas megakernel per step
    # (k3_megakernel); the kernel owns the weights, so this replaces sglang's path.
    SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL = EnvBool(False)
    # Kimi-K3 megakernel numeric relaxations (comma list, see k3_megakernel/config.py Relax);
    # empty = none: BF16 weights and f32 reduction wires (the default kernel options still
    # reorder some f32 sums). Each relaxation must pass the GSM8K / GPQA gates before use.
    SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX = EnvTuple(())
    # Kimi-K3 megakernel StackKernel options, "key=value;..." (values as Python literals, e.g.
    # "moe_ring=6;wire_relax=bf16_transport"); empty = the kernel defaults. Serving accepts only
    # k3_megakernel/ops.py SERVING_KERNEL_OPTIONS. wire_relax adds the wire relaxations
    # (bf16_transport, host_bf16, oproj_flat) without touching the weight layout.
    SGLANG_TPU_K3_MEGAKERNEL_KERNEL_OPTS = EnvStr("")
    # Kimi-K3 megakernel prefill kernel: "auto" (8-token rows blocks when --chunked-prefill-size is
    # a multiple of 8, else one stack pass per token), "rows" or "loop".
    SGLANG_TPU_K3_MEGAKERNEL_PREFILL = EnvStr("auto")
    # Kimi-K3 megakernel verify kernel (speculative serving): "rows" or "loop" (one stack pass per
    # row, bit-identical to decode steps; parity checks).
    SGLANG_TPU_K3_MEGAKERNEL_VERIFY = EnvStr("rows")
    # Kimi-K3 megakernel expert loading: reader processes per rank, and ranks of a host that may
    # hold their expert set in host memory at once (lower it when host RAM peaks during load).
    SGLANG_TPU_K3_MEGAKERNEL_EXPERT_LOAD_PROCS = EnvInt(8)
    SGLANG_TPU_K3_MEGAKERNEL_EXPERT_LOAD_SLOTS = EnvInt(4)
    # Dev only: with the megakernel on, keep sglang's decoder layers (and its prefill path) for
    # A/B comparisons; the default lets the kernel own the decoder weights.
    SGLANG_TPU_K3_MEGAKERNEL_KEEP_SGLANG_WEIGHTS = EnvBool(False)

    # Host padding is unconditional; this routes speculative batches to the
    # experimental spec worker, which keeps their metadata on the host as well.
    SGLANG_TPU_HOST_PADDING_NO_D2H = EnvBool(True)

    # Speculative decoding
    # Route DFLASH/DSPARK to sglang_tpu.experimental.TpuBlockSpecWorker and MTP
    # and EAGLE3 to TpuChainSpecWorker; SGLANG_TPU_HOST_PADDING_NO_D2H also does.
    SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER = EnvBool(True)
    # DFLASH draft linears, including the DFlash2 conv kernel projections, as fp8
    # weight-only (fp8_weight_linear.py) for decode-sized calls; prefill-sized calls
    # keep the bf16 weights. Changes the draft's proposals, not verification.
    SGLANG_TPU_DFLASH_FP8_DRAFT = EnvBool(False)

    # Muse Spark: whole-decoder Pallas megakernel for decode and DFlash target-verify
    # steps, with fp8 dense weight copies and an fp8 weight-only LM head.
    SGLANG_TPU_MUSE_MEGAKERNEL = EnvBool(False)
    # With the megakernel: prefill and other native-path steps run the dense linears and routers
    # on the kernel's fp8 K-major copies (per-column scales, as in decode) and the bf16
    # weights are freed for the KV pool.
    SGLANG_TPU_MUSE_SHARED_DENSE = EnvBool(False)
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
    # Allocates RPA v3 MHA KV as [num_blocks, layer_num, page_size, ...],
    # so all layers for a page form one row for single-DMA Raiden P/D transfers.
    # Both P/D servers must agree; tpu_platform refuses the configurations that
    # read layer by layer.
    SGLANG_TPU_USE_BLOCK_MAJOR_KV = EnvBool(False)
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
    # Fixed batch shape of MIXED (mixed-chunk) batches; unset = the largest bs bucket. Raise it
    # when more requests decode inside mixed chunks than any decode bucket holds.
    SGLANG_TPU_MIXED_MAX_BS = EnvInt(None)
    # Megakernel attention: fold a power-of-two attn_scale into the bf16 queries instead of
    # scaling every score (exact).
    SGLANG_TPU_MK_FOLD_ATTN_SCALE = EnvBool(False)
    # Muse megakernel: fold the f / o all-gathers into the next qkvg and the pool + router
    # matvecs (not bit-exact: the matvec input rounding moves; verify steps only).
    SGLANG_TPU_MUSE_MK_FOLD = EnvBool(True)
    # Megakernel KV write on SEQ_ALONG_LANE pools: all page read-modify-writes of a layer in
    # flight together, overlapped with the next layer's (MKConfig.kv_write_pipelined; exact).
    SGLANG_TPU_MUSE_MK_KV_WRITE_PIPELINED = EnvBool(False)
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
    # Fused-RS NVFP4 tiles of at most this many rows run the block-diagonal fp8 dot (e2m1
    # codes to the MXU as e4m3, block scales applied to per-group partial products) instead
    # of dequantizing the weight tile to bf16; 0 disables. Decode tiles hold 1-8 rows.
    SGLANG_TPU_FUSED_RS_BLOCKDIAG_ROWS = EnvInt(0)
    # K chunk of the block-diagonal fp8 dot; a multiple of the fp4 scale block that divides
    # the GMM K tiles. Rows per chunk = tile rows x chunk / 16.
    SGLANG_TPU_FUSED_RS_BLOCKDIAG_KC = EnvInt(256)
    # Fused-RS NVFP4: run W4A8 (block scales folded into the weights, activations requantized
    # to e4m3 per 256-wide block, fp8 MXU) instead of the W4A16 bf16 matmul.
    SGLANG_TPU_FUSED_RS_FP4_W4A8 = EnvBool(False)
    # Fused-RS EP mode: build the tile metadata in XLA before the kernel (as expert-TP does)
    # instead of the kernel's scalar prologue over every routed row.
    SGLANG_TPU_FUSED_RS_EP_PRECOMPUTE_META = EnvBool(True)
    # Fused-RS: issue the first tile's expert weight DMAs before the entry barrier.
    SGLANG_TPU_FUSED_RS_EARLY_WEIGHT_DMA = EnvBool(False)
    # Fused-RS EP mode: skip zeroing the output buffer; every (token, slot) row is written
    # exactly once by its expert's rank (masked slots are spread over real experts).
    SGLANG_TPU_FUSED_RS_EP_SKIP_OUT_ZERO = EnvBool(True)
    # Drop the post-scatter cross-rank barrier: send and receive semaphore waits already order
    # every DMA, the barrier only aligned kernel exit (exact; -10% on the 8192-token call).
    SGLANG_TPU_FUSED_RS_SKIP_EXIT_BARRIER = EnvBool(True)
    # Fused-RS EP mode: write each tile's output as one DMA run per owner rank (power-of-two
    # pieces) into a rank-ordered output instead of one DMA per row; the post-combine maps
    # ranks back to tokens. Measured: the per-row loop costs ~0.7 us per local row.
    SGLANG_TPU_FUSED_RS_EP_RUN_WRITE = EnvBool(False)
    # Fused-RS tiles (tile_m,k1,n1,k2,n2,w1_bufs,w2_bufs) for EP-layout prefill calls
    # (>= 32768 routed rows), overriding the tuned table.
    SGLANG_TPU_FUSED_RS_EP_PREFILL_TILES = EnvStr(None)
    # Fused-RS EP mode leaves each tile's output row writes in flight until its staging
    # slot is reused, as expert-TP mode does, instead of waiting for them after every tile.
    SGLANG_TPU_FUSED_RS_EP_DEFER_WRITES = EnvBool(True)
    # Expert-TP fused-RS with a post-expert norm: ep sum, norm and top-k combine in one Pallas
    # pass instead of three XLA fusions. Equal up to f32 summation order (bf16 casts kept).
    SGLANG_TPU_FUSED_RS_ETP_FUSED_COMBINE = EnvBool(False)
    # Fused-RS: rows issued per iteration of the per-row gather / write DMA loops.
    SGLANG_TPU_FUSED_RS_ROW_LOOP_UNROLL = EnvInt(1)
    # Fused-RS: output staging buffers a tile's deferred direct writes stay in flight
    # across (VMEM: slots x tile_m x hidden x 2 B). Measured on the Muse Spark expert-TP
    # call: 6 slots -1.5% per layer vs 3, 8 slots -2%.
    SGLANG_TPU_FUSED_RS_STAGING_SLOTS = EnvInt(3)
    # Fused-RS: when an expert's W1 and W2 each fit one DMA, they alternate between the two
    # weight buffers by expert parity and the next expert's are fetched at the start of a tile,
    # behind its whole compute, instead of one buffer refilled between matmuls (exact).
    SGLANG_TPU_FUSED_RS_WEIGHT_PINGPONG = EnvBool(False)
    # Expert-TP fused-RS: the two cores of a chip add their partials on-chip and send one
    # pair sum per row to another chip's owner, halving off-chip bytes (one extra bf16 rounding).
    SGLANG_TPU_FUSED_RS_ETP_PAIR_EXCHANGE = EnvBool(False)
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
    # Batched RPA (non-transposed tiles, e.g. verify blocks): softmax without the running
    # row max. Only for bounded scores: Muse's weightless per-head RMS-normed q and k keep
    # |s| <= ~44, so exp(s) fits; rounding differs from the max-subtracted form.
    SGLANG_TPU_RPA_NO_MAX = EnvBool(False)
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

    # PD disaggregation (raiden)
    # TCP connections per rank pair for raiden KV reads and resharding pushes,
    # not the pipeline path. Set the same value on prefill and decode. More can
    # raise throughput but opens more connections, and above 4 is unlikely to
    # help much. Benchmark first; in a large prefill-decode mesh, consider 1 or 2.
    SGLANG_TPU_RAIDEN_TRANSFER_PARALLELISM = EnvInt(4)

    # HiCache write_back on raiden (see hicache_write_back_copy_ahead.py): tokens
    # of the next eviction victims kept copied to host. Unset: four chunked-prefill
    # chunks. 0: evictions back up their victims synchronously, as upstream does.
    SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS = EnvInt(None)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False
