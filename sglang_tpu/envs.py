# SPDX-License-Identifier: Apache-2.0
"""Environment variables owned by the TPU plugin.

Declare new TPU-only variables here with the descriptor types from
`sglang.srt.environ`. Existing ``SGLANG_TPU_*`` entries stay in core until their
core readers move; never declare one name in both registries.
"""

from __future__ import annotations

from sglang.srt.environ import (
    EnvBool,
    EnvBoolWithAlias,
    EnvField,
    EnvFloat,
    EnvFloatWithAlias,
    EnvInt,
    EnvStr,
    EnvTuple,
)

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
    # Every N verify steps, TP ranks compare a digest of their committed tokens and
    # raise on mismatch: hash_uniforms replaces the accept broadcast. 0 disables.
    # Steps, not seconds, so every rank joins the same all-gather; ~1 min at bs 1.
    SGLANG_TPU_SPEC_RANK_CHECK_INTERVAL = EnvInt(8192)
    # TpuBlockSpecWorker's candidate acceptance by block verification (Sun et al.
    # 2024, Algorithm 2) instead of token-wise rejection; lossless either way.
    SGLANG_TPU_SPEC_BLOCK_VERIFICATION = EnvBool(False)
    # DFLASH draft linears, including the DFlash2 conv kernel projections, as fp8
    # weight-only (fp8_weight_linear.py) for decode-sized calls; prefill-sized calls
    # keep the bf16 weights. Changes the draft's proposals, not verification.
    SGLANG_TPU_DFLASH_FP8_DRAFT = EnvBool(False)

    # Muse Spark: whole-decoder Pallas megakernel for decode and DFlash target-verify
    # steps, with fp8 dense weight copies and an fp8 weight-only LM head.
    SGLANG_TPU_MUSE_MEGAKERNEL = EnvBool(False)
    # Fused KV pool stores a rank's heads side by side in one 128-lane row when they
    # fit exactly (e.g. 2 x head_dim 64): half the bytes of a zero-padded row per head.
    SGLANG_TPU_KV_PACK_HEADS = EnvBool(False)
    # Allocates RPA v3 MHA KV as [num_blocks, layer_num, page_size, ...],
    # so all layers for a page form one row for single-DMA Raiden P/D transfers.
    # Both P/D servers must agree; tpu_platform refuses the configurations that
    # read layer by layer.
    SGLANG_TPU_USE_BLOCK_MAJOR_KV = EnvBool(False)

    # Raiden PD disaggregation
    # Base control port. Each local rank reserves consecutive control/data
    # ports for full KV, SWA KV, and recurrent state.
    SGLANG_RAIDEN_KV_PORT = EnvInt(31000)
    # TCP connections per rank pair for raiden KV reads and resharding pushes,
    # not the pipeline path. Set the same value on prefill and decode. More can
    # raise throughput but opens more connections, and above 4 is unlikely to
    # help much. Benchmark first; in a large prefill-decode mesh, consider 1 or 2.
    SGLANG_TPU_RAIDEN_TRANSFER_PARALLELISM = EnvInt(4)
    # Threads pushing resharded KV off the listener thread.
    SGLANG_RAIDEN_PUSH_WORKERS = EnvInt(1)

    # Host staging budget per node, split across local ranks and engines.
    SGLANG_RAIDEN_HOST_POOL_GB = EnvFloat(128.0)
    # Maximum pages per request. 0 derives the limit from the context/window.
    SGLANG_RAIDEN_MAX_BLOCKS = EnvInt(0)
    SGLANG_RAIDEN_SWA_MAX_BLOCKS = EnvInt(0)
    # Context length fallback when --context-length is unset.
    SGLANG_RAIDEN_MAX_MODEL_LEN = EnvInt(8192)
    # Pinned-host budget for retraction backup. 0 keeps the row-copy path.
    SGLANG_RAIDEN_RETRACT_HOST_GB = EnvFloat(0.0)

    # Raiden: a DP or TP decode receives KV from a pipeline-parallel prefill.
    SGLANG_RAIDEN_PIPELINE_TRANSFER = EnvBool(False)
    # Raiden: stage each transfer (and reshard plan) in its own host-pool pages
    # instead of fixed max-length slots; asymmetric attention TP needs it.
    SGLANG_RAIDEN_DYNAMIC_HOST_STAGING = EnvBoolWithAlias(
        True, deprecated_name="TPU_RAIDEN_DYNAMIC_HOST_STAGING"
    )
    # Raiden: a decode starts a room's next read as soon as the prefill
    # announces it; 0 starts reads only on the next scheduler poll.
    SGLANG_RAIDEN_LISTENER_START_READ = EnvBool(True)
    # Raiden: deadline of each KV chunk transfer, not of the whole request; the
    # closing read counts as a chunk, and a non-chunked transfer is one chunk.
    # Counted from registration on the prefill and from start_read on the
    # decode. Set the same on both.
    SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S = EnvFloatWithAlias(
        60.0, deprecated_name="SGLANG_RAIDEN_TIMEOUT_S"
    )
    # Pipeline prefill: move pages final before the prefill ends (cached prefix,
    # finished chunks) as early transfers of at least PART_PAGES pages each.
    SGLANG_RAIDEN_PIPELINE_EARLY_PARTS = EnvBool(False)
    SGLANG_RAIDEN_PIPELINE_PART_PAGES = EnvInt(16)
    # Pipeline prefill: each group of STAGE_GROUP consecutive stages moves its
    # layers once its forwards of the final chunk complete, before sampling.
    SGLANG_RAIDEN_PIPELINE_STAGE_PARTS = EnvBool(False)
    SGLANG_RAIDEN_PIPELINE_STAGE_GROUP = EnvInt(1)
    # Decode: threads coordinating pipeline transfers, one client each.
    SGLANG_RAIDEN_PIPELINE_COORD_WORKERS = EnvInt(1)
    # Pipeline prefill: each stage registers its share of the closing transfer
    # at the final chunk's launch, the last stage once its tokens are copied.
    SGLANG_RAIDEN_PIPELINE_EARLY_CLOSE = EnvBool(False)

    # raiden KV-offload backend
    # Seconds a raiden DMA may stay pending, from each blocking wait or, when
    # polled, from issue. Past it the scheduler exits with TimeoutError, so keep
    # it far above a healthy transfer.
    SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S = EnvFloat(60.0)

    # HiCache write_back on raiden (see hicache_write_back_copy_ahead.py): tokens
    # of the next eviction victims kept copied to host. Unset: four chunked-prefill
    # chunks. 0: evictions back up their victims synchronously, as upstream does.
    SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS = EnvInt(None)

    # Beam Pallas kernels behind the model's pluggable ops (see beam_kernels.py).
    # False keeps every Beam op on its torch reference, for A/B and bisection.
    SGLANG_TPU_BEAM_KERNELS = EnvBool(True)
    # Comma-separated Beam ops kept on their torch reference while the rest use
    # Pallas, to bisect a single kernel: sandwich, rmsnorm, qk_norm, topk.
    SGLANG_DEBUG_TPU_BEAM_KERNELS_SKIP = EnvTuple(())

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False
