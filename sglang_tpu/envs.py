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
    # Parent-selected rendezvous port for the host-wide DP replica process group.
    SGLANG_TPU_DP_SESSION_PORT = EnvInt(None)
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
    # Kimi-K3: gather the TP-sharded logits by an all-reduce of a zero-padded
    # buffer instead of XLA's all-gather (kimi_k3/logits_gather.py); bit-identical.
    # XLA's all-gather returned wrong logits in the batched-prefill graph of
    # Kimi-K3.1 at TP32 (wrong first tokens), so this is on by default.
    SGLANG_TPU_LOGITS_GATHER_ALLREDUCE = EnvBool(True)
    # Kimi-K3 KDA decode: fence the static fp8 [q, k, v, g] GEMV behind the
    # b_proj / f_a / f_b GEMVs (an XLA optimization barrier, same values) so
    # they run while its weight prefetch is in flight (kimi_k3/order_op.py).
    SGLANG_TPU_K3_KDA_GATES_FIRST = EnvBool(True)
    # Single-token o_proj input fenced as a [1, K] row (kimi_k3/oproj_row.py).
    SGLANG_TPU_K3_OPROJ_ROW = EnvBool(False)
    # Single-token decode logits all-gathered by the one-shot Pallas kernel
    # instead of XLA's all-gather (kimi_k3/logits_gather.py); bit-identical.
    SGLANG_TPU_LOGITS_ONESHOT_GATHER = EnvBool(False)
    # Kimi-K3 single-token decode: split the replicated latent down / up-proj
    # and router gate GEMVs by output columns across the ranks of a host and
    # exchange the columns inside the host (host_split_gemv.py). Needs
    # SGLANG_TPU_K3_STATIC_FP8 moe_front and latent_up (checked at startup).
    SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV = EnvBool(False)
    # Kimi-K3 single-token decode with the host split: the latent up-proj
    # kernel (end of layer i's MoE) also copies layer i+1's KDA [q, k, v, g]
    # fp8 weight HBM -> VMEM during its host exchange, so the qkvg GEMV reads it
    # from VMEM instead of waiting for XLA's late prefetch (TP32: 4.2 us per
    # KDA layer). Same arithmetic (bit-identical). Weights larger than
    # SGLANG_TPU_K3_QKVG_PREFETCH_MAX_KIB are not prefetched.
    SGLANG_TPU_K3_QKVG_PREFETCH = EnvBool(False)
    SGLANG_TPU_K3_QKVG_PREFETCH_MAX_KIB = EnvInt(16384)
    # Kimi-K3 single-token decode: the host-split latent up-proj kernel also
    # does the MoE tail add (up + shared) + prefix and the next layer's
    # attention-side attention-residual aggregation (latent_up_tail.py). Same
    # ops, same order. Needs the host split and the attention-residual kernel.
    SGLANG_TPU_K3_LATENT_UP_ATTN_RES = EnvBool(False)
    # Decode Pallas kernels (static fp8 GEMV, host-split GEMV, single-token
    # MoE) request their computed working set as scoped VMEM instead of a
    # fixed 40-48 MiB, leaving XLA room to prefetch weights across them
    # (sglang_tpu/kernels/vmem_limit.py). Same kernel source, but a smaller
    # limit can change Mosaic's tiling of the single-token MoE kernel and with
    # it the f32 summation order (13L Kimi-K3 greedy differs; Kimi-K3.1 is
    # bit-identical), so it is off by default and set by the Kimi-K3.1 recipe.
    SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE = EnvBool(False)
    # MoE: one fused kernel for single-token packed-MXFP4 decode
    # (moe/pallas/single_token_moe.py); False keeps the gmm_v2 path.
    SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL = EnvBool(True)
    # MoE: one kernel streaming the selected FP8 experts' weights for 1..8 tokens
    # (moe/pallas/moe_stream.py); False keeps the gmm_v2 path.
    SGLANG_TPU_ENABLE_STREAMING_MOE_KERNEL = EnvBool(False)
    # FP8 linear: one W8A8 kernel for decode-sized batches (kernels/fp8_linear_small_m.py);
    # False keeps the XLA path.
    SGLANG_TPU_ENABLE_FP8_LINEAR_SMALL_M_KERNEL = EnvBool(False)
    # Sparse MLA: DSA extend batches of 512+ tokens (prefill, no speculative
    # decoding) through a query-blocked kernel, which fetches a block of queries'
    # pages once and masks each query's selection, instead of gathering every
    # query's rows: pooled DSA (GLM-5.3-Flash) attention/pallas/mla_sparse_qblock.py,
    # plain DSA (Kimi-K3.1) attention/pallas/mla_sparse_qblock_tokens.py. The
    # flag thus also switches every plain-DSA (index kpool == 1) model's prefill,
    # not only pooled DSA. Not bit-exact with the per-query kernel (summation
    # order); False keeps it.
    SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL = EnvBool(False)
    # DSA indexer token split for prefill (EXTEND / MIXED batches of at least
    # _MIN_TOKENS tokens; the minimum is also 32 rows per rank): every
    # attention-TP rank stores the index keys of all tokens but scores and
    # top-k-selects only its contiguous slice of ceil(T / attn_tp) query rows
    # (rounded up to 8); an int32 all-reduce over the attention-TP group puts
    # the [T, topk] rows together (attention_pallas_mla_sparse.py). Same rows
    # bit for bit; attention is unchanged. _MAX_ROWS caps the rows per rank
    # (the all-reduce buffer is [attn_tp, rows, topk] int32 per DSA layer);
    # larger batches keep the replicated path, as do decode and verify.
    SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT = EnvBool(False)
    SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT_MIN_TOKENS = EnvInt(1024)
    SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT_MAX_ROWS = EnvInt(512)
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
    # Serving-time decode forwards of a dynamo-compiled decoder run only
    # Dynamo's differentiating guards (tensor shapes/dtypes and every guard
    # that failed during warmup) instead of the full guard set, once warmup
    # froze compilation with fail_on_recompile. Picks the same precompiled
    # graph; saves ~0.1-0.3 ms of host time per decode step.
    SGLANG_TPU_DECODE_SKIP_GUARD_EVAL = EnvBool(False)
    # Compiled (dynamo + AOTAutograd) inference graphs whose input mutations
    # are plain data mutations (KV / recurrent state) use a lean AOTAutograd
    # runtime wrapper that copies all updated inputs back with one
    # torch._foreach_copy_ instead of torch's per-input epilogue
    # (aot_fast_runtime_wrapper.py). Same copies, so identical results.
    SGLANG_TPU_AOT_LEAN_RUNTIME_WRAPPER = EnvBool(False)
    # torch_tpu split executables with a single partition (no graph-splitting
    # collectives) call their compiled executable directly instead of through
    # the split FX GraphModule's Python forward (aot_fast_runtime_wrapper.py).
    SGLANG_TPU_SPLIT_EXEC_DIRECT_CALL = EnvBool(False)
    # 1: Dynamo installs nn.Module parameters / buffers as graph attributes
    # (install_free_tensors, narrowed to _parameters/_buffers); 2: every tensor
    # reached through an nn.Module (e.g. static-fp8 packed weights held as plain
    # attributes), still not globals / nonlocals / forward inputs. AOTAutograd
    # lifts them once instead of Dynamo's bytecode re-reading ~3000 attribute
    # chains per decode call (aot_fast_runtime_wrapper.py). A parameter object
    # replaced after compile is not noticed (in-place updates are).
    SGLANG_TPU_DYNAMO_PARAMS_AS_GRAPH_ATTRS = EnvInt(0)
    # TpuPagedTokenToKVPoolAllocator.alloc_decode: when no request crosses a
    # page boundary, out_cache_loc = last_loc + 1 without the full host formula.
    SGLANG_TPU_ALLOC_DECODE_FAST_PATH = EnvBool(False)
    # Two-level one-shot Pallas all-reduce for single-token TP decode tensors
    # (twolevel_allreduce.py): host-local exchange, then cross-host. Applies
    # when TP spans the whole world; everything else keeps XLA's all-reduce.
    SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE = EnvBool(False)
    # That all-reduce: the in-host phase waits only for the host's peers; the
    # cross-host barrier is checked just before the cross-host phase, so its
    # latency overlaps the in-host exchange (twolevel_allreduce._kernel). Same
    # sums. Only changes anything with more than one host.
    SGLANG_TPU_TWOLEVEL_ALLREDUCE_SPLIT_BARRIER = EnvBool(False)
    # That all-reduce: Kimi-K3's merged MoE reduce also applies the routed
    # latent RMSNorm in the kernel (one op instead of the reduce + two norm
    # fusions). The norm runs on the [1, H] row, which reproduces XLA's f32
    # summation order (bit-identical).
    SGLANG_TPU_TWOLEVEL_ALLREDUCE_FUSE_LATENT_NORM = EnvBool(True)
    # That all-reduce: the attention o_proj reduce of a Kimi-K3 attention-residual
    # layer also runs the MLP-side aggregation (kimi_attn_res_decode with the
    # pending add) in the same kernel. Same ops in the same order (exact).
    SGLANG_TPU_TWOLEVEL_ALLREDUCE_FUSE_ATTN_RES = EnvBool(True)
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
    # Kimi-K3 megakernel prefill kernel: "auto" (rows blocks when --chunked-prefill-size is a
    # multiple of the block, else one stack pass per token), "rows" or "loop".
    SGLANG_TPU_K3_MEGAKERNEL_PREFILL = EnvStr("auto")
    # Tokens per rows-mode prefill block, a multiple of 8. Each block streams every layer's
    # weights once, so a larger block amortizes that over more tokens; VMEM bounds it (93L TP32
    # fits 32, 13L TP8 fits 16). 8 is the previous behaviour.
    SGLANG_TPU_K3_MEGAKERNEL_PREFILL_BLOCK = EnvInt(8)
    # Kimi-K3 megakernel verify kernel (speculative serving): "rows" or "loop" (one stack pass per
    # row, bit-identical to decode steps; parity checks).
    SGLANG_TPU_K3_MEGAKERNEL_VERIFY = EnvStr("rows")
    # Kimi-K3 megakernel expert loading: reader processes per rank, and ranks of a host that may
    # hold their expert set in host memory at once (lower it when host RAM peaks during load).
    SGLANG_TPU_K3_MEGAKERNEL_EXPERT_LOAD_PROCS = EnvInt(8)
    SGLANG_TPU_K3_MEGAKERNEL_EXPERT_LOAD_SLOTS = EnvInt(4)
    # "layer": copy each MoE layer's experts into a preallocated device array (host peak ~2 layers
    # per rank); "whole": build the rank's whole expert set in host memory, upload once.
    SGLANG_TPU_K3_MEGAKERNEL_EXPERT_UPLOAD = EnvStr("layer")
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
    # PD on the experimental spec worker: the prefill ships the target's draft-input
    # hidden states (pd_hidden_pool.py) and the decode builds the draft KV and carry,
    # instead of the prefill running the draft over the prompt.
    SGLANG_TPU_PD_DECODE_DRAFT_PREFILL = EnvBool(False)
    # With it and a windowed draft, the prefill writes only the hidden rows a decode
    # can read: the window before the prompt end, or before each chunk end when a
    # prefix-sharing tree may hand that window to a later request.
    SGLANG_TPU_ENABLE_PD_HIDDEN_WINDOW_WRITE = EnvBool(True)
    # P/D prefix pull (disaggregation/prefix_pull.py): the least prefix gap, in
    # tokens, worth pulling from the decode instead of recomputing (one chunk).
    SGLANG_TPU_PD_PREFIX_PULL_MIN_TOKENS = EnvInt(4096)
    # Seconds between the cumulative prefix-gap / decode-window log lines; 0 = off.
    SGLANG_TPU_PD_PREFIX_GAP_LOG_INTERVAL_S = EnvFloat(60.0)
    # Pull prefixes from the decode: the decode locks the prefix it reports as
    # pullable until the request's KV lands, and the prefill fetches it.
    SGLANG_TPU_PD_PREFIX_PULL = EnvBool(False)
    # Decode: prompt-end SWA windows kept locked by the cache for the next turn's
    # pull, least recently used released first and under memory pressure; 0 = off.
    SGLANG_TPU_PD_DECODE_PROMPT_END_ANCHORS = EnvInt(0)
    # What a failed or timed-out prefix pull does. 1 (bring-up default): the
    # request fails with the reason, an error log and
    # sglang:pd_prefill_prefix_pull_failed_total{reason}; nothing is recomputed.
    # 0: every PP stage recomputes the prefix instead, an explicit reliability
    # path that logs a warning with the room, reason and recomputed tokens each
    # time and counts sglang:pd_prefill_prefix_pull_fallback_total{reason}.
    SGLANG_TPU_PD_PREFIX_PULL_STRICT = EnvBool(True)
    # DFLASH draft linears, including the DFlash2 conv kernel projections, as fp8
    # weight-only (fp8_weight_linear.py) for decode-sized calls; prefill-sized calls
    # keep the bf16 weights. Changes the draft's proposals, not verification.
    SGLANG_TPU_DFLASH_FP8_DRAFT = EnvBool(False)

    # Muse Spark: whole-decoder Pallas megakernel for decode and DFlash target-verify
    # steps, with fp8 dense weight copies and an fp8 weight-only LM head.
    SGLANG_TPU_MUSE_MEGAKERNEL = EnvBool(False)
    # Muse NVFP4: keep the BF16 experts of the first and last layers BF16, on fused-RS
    # or gmm_v2. By default they are quantized to FP8 on host at load, as are all
    # experts of a BF16 checkpoint. The Muse megakernel reads only fp4 or fp8 experts,
    # so it cannot be combined with this.
    SGLANG_TPU_ENABLE_MUSE_BF16_FIRST_LAST_LAYER_EXPERTS = EnvBool(False)
    # Allocates RPA v3 MHA KV as [num_blocks, layer_num, page_size, ...],
    # so all layers for a page form one row for single-DMA Raiden P/D transfers.
    # Both P/D servers must agree; tpu_platform refuses the configurations that
    # read layer by layer.
    SGLANG_TPU_USE_BLOCK_MAJOR_KV = EnvBool(False)
    # Vision encoder flash attention: q tile (0 = the kernel's default blocks).
    # The kernel keeps a single k step (block_k = kv_len, softmax over the whole
    # row); 256 is bit-exact with the default tile on tpu7x, 512+ differ at bf16
    # rounding level. Applied only while kv_len <=
    # SGLANG_TPU_VISION_FLASH_BLOCK_Q_MAX_SEQ: a 256-row q tile at the 8K-16K
    # image buckets halted the TPU during warmup.
    SGLANG_TPU_VISION_FLASH_BLOCK_Q = EnvInt(0)
    SGLANG_TPU_VISION_FLASH_BLOCK_Q_MAX_SEQ = EnvInt(4096)
    # Vision encoder flash attention: compute only the key chunks whose segments
    # meet each query block (band-skip kernel). Byte-identical to the dense
    # kernel at the shapes it covers (kv_len 3072-12288 at a 256-row q tile,
    # 16384 at the default 128); other shapes keep the dense kernel.
    SGLANG_TPU_VISION_FLASH_BAND_SKIP = EnvBool(False)

    # Largest decode batch (tokens) that `--moe-runner-backend pallas_tpu_fused_decode`
    # runs with the fused MoE decode kernel (`sglang_tpu/moe/pallas/fused_moe_decode`);
    # larger batches use gmm_v2.
    SGLANG_TPU_FUSED_MOE_DECODE_MAX_TOKENS = EnvInt(256)

    # DSA indexer, decode-sized batches: quantize and store the index keys with
    # one Pallas kernel (attention/pallas/dsa_key_store.py, same bytes as the
    # XLA path); False keeps the XLA whole-row scatter for every batch.
    SGLANG_TPU_DSA_PALLAS_KEY_STORE = EnvBool(True)
    # SparseCore top-k (dsv4/sparsecore_topk.py): when every row's valid prefix fits
    # one slice, run a single pass over the narrowest 4096-wide bucket instead of the
    # partitioned top-k; long-context pools pad rows to the max context. Selects the
    # same set of indices, but their order (and so the sparse-MLA summation order
    # downstream) can differ from the partitioned path.
    SGLANG_TPU_SCTOPK_SHORT_ROWS = EnvBool(True)

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

    # Host staging budget per node, split across local ranks.
    SGLANG_RAIDEN_HOST_POOL_GB = EnvFloat(24.0)
    # Host layer buffers across full-attention and SWA engines, split by layer
    # count with at least one each. 0 keeps full mirrors.
    # More buffers, or 0, make each slot larger. Raise SGLANG_RAIDEN_HOST_POOL_GB
    # with it to keep the slot count and concurrent transfers.
    SGLANG_TPU_RAIDEN_STAGING_LAYERS = EnvInt(8)
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

    # TPU PD KV transfer (disaggregation/tpu/conn.py)
    # Transfers between logged latency summaries; 0 disables.
    SGLANG_DEBUG_TPU_KV_STATS_EVERY = EnvInt(100)
    # Issue the prefill D2H asynchronously; 0 keeps the synchronous batched D2H.
    SGLANG_TPU_KV_ASYNC = EnvBool(True)
    # Parallel PUSH streams per transfer on each TP rank; values below 1 mean 1.
    SGLANG_TPU_KV_CHANNELS = EnvInt(2)

    # Multi-host node normalization (node_normalization.py)
    # Discovery store port; unset uses the --dist-init-addr port plus one.
    SGLANG_TPU_TOPOLOGY_BOOTSTRAP_PORT = EnvInt(None)

    # Pipeline-parallel prefill chunking
    SGLANG_TPU_PP_DYNAMIC_CHUNKS = EnvBool(True)
    SGLANG_TPU_PP_CHUNK_SLACK = EnvFloat(0.1)

    # PD disaggregation tracing
    SGLANG_TPU_ENABLE_DISAGGREGATION_PERFETTO_TRACING = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_ENABLE_PERFETTO_TRACING"
    )

    # Compilation, kernels, sampling and profiling
    SGLANG_TPU_CHECK_RECOMPILATION = EnvBool(False)
    # Also attribute eager compiles to sglang source lines. ~9 us per eager op.
    SGLANG_TPU_CHECK_RECOMPILATION_SITES = EnvBool(False)
    SGLANG_TPU_ALLOW_SPEC_RECOMPILE = EnvBool(False)
    # Serve input (prompt) logprobs on the compiled backend. Opt-in because it
    # is not free for anyone else: the compiled extend must return every prefill
    # row (CaptureHiddenMode.FULL), and capture_hidden_mode is part of the
    # graph's guard, so the mode is a property of the warmed variant rather than
    # of the request. Off keeps the extend graph and the serving path exactly as
    # they are today, and a request asking for prompt logprobs is rejected.
    SGLANG_TPU_ENABLE_PROMPT_LOGPROB = EnvBool(False)
    # Cut the eager input-logprob chunks of a forward holding one request at every
    # multiple of this many rows from the request's first row (hooks/logprob_grid.py),
    # so a prefill chunk larger than the grid hands the epilogue the same row sets as
    # grid-sized chunks. 0 registers no hook.
    SGLANG_TPU_LOGPROB_GRID_ROWS = EnvInt(0)
    # Put each request's logit row on ForwardBatch.logit_rows before every plain
    # EXTEND forward (compilation/logit_rows.py), for models that run their last
    # layer on those rows only. Off leaves the field None.
    SGLANG_TPU_FOUR_STAGE_LOGIT_ROWS = EnvBool(False)
    # Preserve FP32 expert scales and apply GMM partial scales in FP32.
    SGLANG_TPU_FP8_MOE_FP32_SCALES = EnvBool(False)
    # Store INT4 MoE group scales as f32 instead of bf16 (66 MB per layer per
    # device on Kimi-K2.6). vmem-fp8 always uses f32.
    SGLANG_TPU_INT4_MOE_FP32_SCALES = EnvBool(False)
    SGLANG_TPU_FP8_MOE_COMPACT_SCALES = EnvBool(False)
    SGLANG_TPU_FP8_MOE_COMPACT_W1_SCALES = EnvBool(False)
    SGLANG_TPU_FP8_MOE_COMPACT_W2_SCALES = EnvBool(False)
    # Reuse uploaded attention metadata across forwards; applies to all TPU batches.
    SGLANG_TPU_ENABLE_ATTN_METADATA_CACHE = EnvBool(True)
    # Read extend query lengths from a complete host mirror instead of the device.
    SGLANG_TPU_ENABLE_HOST_QUERY_LENS = EnvBool(True)
    # Opt-in runtime KV-tile promotion for compatible prefill backends.
    # Backends snapshot this flag at initialization; tile hints are tensor inputs.
    SGLANG_TPU_WORKLOAD_AWARE_KV_TILES = EnvBool(False)
    SGLANG_TPU_EAGER_MODE = EnvStr("DEFER_NEVER")
    # Physical TPU v7 TP8 measurements favor distributed TopK from batch 64.
    # Deployments can override this cutoff after measuring their topology.
    SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE = EnvInt(64)
    # Filter top-k/top-p/min-p with the sort-free binary-search cutoff instead
    # of the full-vocabulary torch sort. Disable to A/B the sampled
    # distribution against the reference path.
    SGLANG_TPU_ENABLE_BSEARCH_SAMPLING = EnvBool(True)
    # Vision-encoder precompile ladder. Unset uses the computed square-grid
    # ladder (Kimi-K3 with vision splash attention defaults to a ladder up to
    # 66564); set-but-empty disables precompile entirely.
    SGLANG_TPU_VIT_PRECOMPILE_SIZES = EnvTuple(tuple())
    SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES = EnvInt(4900)
    # Override the computed ladder's rung count, trading compiles against
    # padding waste. Unset leaves each encoder on the count it picked.
    SGLANG_TPU_VIT_NUM_PRECOMPILE_BUCKETS = EnvInt(None)
    # Text-only serving of a VL decoder: warm only the token-id ("text_ids") EXTEND graphs, not
    # the input_embeds ones. Each variant is a separate large graph (Kimi-K3.1 93L TP32: about
    # 100 GB of host RAM per host to compile them). Compilation is frozen after warmup, so
    # scheduler intake rejects multimodal requests with a 400 (sglang_tpu/patches.py).
    SGLANG_TPU_VL_TEXT_ONLY_EXTEND = EnvBool(False)
    # MoE: offload the gather in permute and unpermute to SparseCore.
    SGLANG_TPU_FUSED_MOE_SC_GATHER = EnvBool(True)
    # MoE: gather on the TensorCore below this many routed rows (tokens x topk);
    # 0 = SparseCore always. Measured crossover is 1024 (Kimi, R1, Qwen3-480B).
    SGLANG_TPU_FUSED_MOE_SC_GATHER_MIN_ROWS = EnvInt(0)
    # MoE: use onehot dense matmul for small permute and unpermute/combine
    # paths instead of gather. When set to 0, this is disabled. When set
    # to a positive integer, this is the threshold for tokens per rank below
    # which the dense matmul path is taken. Suggset to start with 1024 or 2048
    # and tune from there. Below it, batches of up to 2048 routed rows also
    # take the sort-free dense rank route, and one-token batches the
    # route-meta kernel.
    SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD = EnvInt(0)
    # Turn off EP-local expert reads (models that opt in read only their EP
    # rank's routed experts with --weight-loader-disable-mmap).
    SGLANG_TPU_DISABLE_EP_LOCAL_EXPERT_READS = EnvBool(False)
    # xprof scope per task. Values <= 0 trace all chips or SparseCores.
    SGLANG_TPU_PROFILE_NUM_CHIPS = EnvInt(1)
    SGLANG_TPU_PROFILE_NUM_SPARSE_CORES = EnvInt(1)
    # xprof python tracer. Off by default: it dominates host busy time.
    SGLANG_TPU_PROFILE_PYTHON_TRACER = EnvBool(False)
    # Read the RPA KV cache in the SEQ_ALONG_LANE layout (token index on the
    # lane axis) instead of the token-major fused one. num_kv_heads*2 is
    # unpadded there, so at one KV head per rank and fp8 KV it is 256 bytes per
    # token against 512, which is what lets long context and large batches fit.
    # Opt-in: it changes the KV pool's buffer shape.
    SGLANG_TPU_USE_SEQ_ALONG_LANE_KV = EnvBool(False)
    # Pin the RPA prefill/MIXED k-block instead of deriving it from the VMEM
    # budget. Escape hatch for a shape or a chip where that derivation is
    # wrong: too large and the kernel fails to allocate at startup, too small
    # and the step runs more blocks than it needs. Rounded up to a page
    # multiple. Unset (None) means the wrapper's own search decides.
    SGLANG_TPU_FORCE_RPA_PREFILL_BKV = EnvInt(None)
    # "bq:bq_c:bkv:n_buffer": prefill blocks for every layer, skipping the VMEM search, whose
    # estimate refuses tiles that fit at TP1 head counts (PP prefill); n_buffer >= 2.
    SGLANG_TPU_FORCE_RPA_PREFILL_BLOCKS = EnvStr(None)
    # Log batched RPA metadata per forward (debugging the PP profile core halt).
    SGLANG_TPU_DEBUG_RPA_METADATA = EnvBool(True)
    # Hack to workaround libtpu bug in 0.0.46 and 0.0.47 causing performance
    # regression in fused MoE with reduce scatter.
    # TODO: Remove once the issue gets resolved in libtpu version > 0.0.47
    # (likely 0.0.48).
    SGLANG_TPU_LIBTPU_WORKAROUND_FOR_ONLINE_QUANT_SCALE = EnvBool(True)
    # Share one page pool between full-attention and sliding-window KV on TPU
    # (equivalent to --enable-unified-memory; see unified_kv_pool.py).
    SGLANG_TPU_UNIFIED_KV_POOL = EnvBool(False)

    # Sampling
    # Keep the repetition-penalty state padded to the power-of-two logits row
    # bucket and run the scaling penalty and greedy argmax on the padded logits
    # (sampling/padded_logits.py). Same per-row ops, so the same tokens. Opt-in:
    # speculative decoding expands the penalties per request.
    SGLANG_TPU_PADDED_LOGITS_SAMPLING = EnvBool(False)
    # Shared compilation (--use-spmd-compile): a trace-cost profile JSON that
    # assigns each shared graph to the least-loaded rank (shared_compile_cost.py).
    # Empty: round-robin owners.
    SGLANG_TPU_SHARED_COMPILE_COST_PROFILE = EnvStr("")
    # With the megakernel: prefill and other native-path steps run the dense linears and routers
    # on the kernel's fp8 K-major copies (per-column scales, as in decode) and the bf16
    # weights are freed for the KV pool.
    SGLANG_TPU_MUSE_SHARED_DENSE = EnvBool(False)
    # Muse Spark prefill sequence parallelism: extend forwards of at least MIN_TOKENS tokens
    # keep the fp32 residual stream token-sharded over the TP group between layers.
    SGLANG_TPU_MUSE_PREFILL_SP = EnvBool(False)
    SGLANG_TPU_MUSE_PREFILL_SP_MIN_TOKENS = EnvInt(1024)
    # Muse Spark: gather prefill-sized column shards with one all-gather along the column
    # dim instead of a token-dim all-gather plus a transpose.
    SGLANG_TPU_MUSE_GATHER_COLUMNS_DIM = EnvBool(False)
    # Muse Spark rotary embedding by a one-lane roll on [tokens, heads * head_dim]
    # instead of a [..., head_dim // 2, 2] pair view (same arithmetic).
    SGLANG_TPU_MUSE_ROPE_ROLL = EnvBool(False)
    # With the megakernel: extend steps run the prefill program (muse_megakernel/prefill.py:
    # batched RPA, fused attention-output and post-MoE kernels, fused-RS MoE) on the
    # kernel's weights. Needs fp8 SEQ_ALONG_LANE pools and bf16 NVFP4 block scales.
    SGLANG_TPU_MUSE_MK_PREFILL = EnvBool(False)
    # Muse expert layout for the megakernel and the prefill program: tp | ep | auto
    # (auto: ep when the server's EP size > 1). muse_megakernel/moe_mode.py.
    SGLANG_TPU_MUSE_MOE_MODE = EnvStr("auto")
    # The prefill program's expert-input and post-expert-projection all-gathers in e4m3
    # (one scale per block): half their ICI bytes, ~2^-4 relative error per element.
    SGLANG_TPU_MUSE_MK_PREFILL_FP8_GATHER = EnvBool(False)
    # Largest megakernel step in rows (running requests x verify tokens, padded to the
    # TP size); larger steps run the native decoder. Steps above 32 rows run the
    # megakernel's row-tiled program (ops.BASE_STEP_MAX_ROWS) and pad to 16 rows.
    SGLANG_TPU_MUSE_MK_MAX_ROWS = EnvInt(32)
    # Muse megakernel: stack the NVFP4 expert slabs of this many consecutive layers into one
    # operand per slab (0 = one operand per layer); the layer params become views into it.
    # 15 measured 3-4% faster decode / verify steps (cc = 1, q = 1 and 8, 128k-512k).
    SGLANG_TPU_MUSE_MK_EXPERT_GROUP = EnvInt(0)
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
    # pages_swa windows DMA'd into SMEM instead of the whole table: "auto" (when the two page
    # tables exceed the SMEM budget), "1" or "0" (forced, e.g. to A/B the two paths)
    SGLANG_TPU_MUSE_MK_SWA_WINDOW_TABLE = EnvStr("auto")
    # Experimental: the pipelined SAL attention loop (MKConfig.attn_pipe) on steps of 33-64 rows
    # too, not only up to _BASE_STEP_MAX_ROWS (12-16 decoding requests at 4-token verify).
    SGLANG_TPU_MUSE_MK_LARGE_STEP_ATTN_PIPE = EnvBool(False)
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
    # Experimental: megakernel experts with more than this many live (row, slot) pairs (1, 2 or 4;
    # 0 off) run in a second loop after each owner group's few-row experts (MKConfig.expert_split;
    # exact). Steps of more than 24 rows only.
    SGLANG_TPU_MUSE_MK_EXPERT_SPLIT = EnvInt(0)
    # Megakernel row-tiled steps: an expert with more than 32 live (row, slot) pairs runs one
    # staged pass in a 64 / 128 / ... / R-row bucket, dequantizing its slabs once, instead of
    # 32- or 64-row passes that each dequantize them again (MKConfig.ex_one_pass; exact).
    # Unset: on with SGLANG_TPU_MUSE_MK_KPERM (-1.3 / -0.8 ms at 64 / 128 rows), off without it
    # (no step-time gain measured there).
    SGLANG_TPU_MUSE_MK_EXPERT_ONE_PASS = EnvBool(None)
    # Megakernel attention on fp8 SEQ_ALONG_LANE pools: feed K (and V) to the MXU as fp8
    # instead of widening each chunk to bf16 on the VPU first. 0 = off, 1 = K only, 2 = K and V
    # (exact). With the 2026-10-27 megakernel, 2 saves 0.09 / 0.20 / 0.40 ms per q = 1 step at
    # 128k / 256k / 512k and nothing at q = 8.
    SGLANG_TPU_MK_SAL_FP8_DOT = EnvInt(0)
    # Profiling only: name the megakernel's in-kernel regions for xprof (needs libtpu
    # --xla_enable_custom_call_region_trace=true and a device tracer level above 1);
    # 2 also names the per-chunk / per-expert regions, which distort the timing they show.
    SGLANG_TPU_MK_TRACE_REGIONS = EnvInt(0)
    # Profiling only: scoped VMEM limit (MiB) for every step while TRACE_REGIONS is on; the region
    # markers take stack VMEM (the 16-row step overflowed the default 60 MiB by 1.1 MiB).
    SGLANG_TPU_MK_TRACE_VMEM_LIMIT_MB = EnvFloat(None)
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
    # Muse megakernel Wave 1-4 optimizations (hoisted SAL refill, stack2 QK/PV, all-VMEM
    # collective/expert scratch on row-tiled steps, 2D PairColl collectives, vectorized routing).
    SGLANG_TPU_MUSE_MK_OPT = EnvBool(True)
    # Offline block-256 (16, 16)^T pre-shuffling of fp4 expert slabs + wpool/scales for
    # zero-shuffle NVFP4 dequant in the decode megakernel (and prefill MoE via MUSE_PF_MOE_KPERM).
    SGLANG_TPU_MUSE_MK_KPERM = EnvBool(False)
    # Decode megakernel (TP): fp4 experts requantized to one scale per this many K rows (0 off,
    # 64 / 128 / 256); the expert dots push raw e2m1 codes to the MXU (MKConfig.qblock).
    SGLANG_TPU_MUSE_MK_QBLOCK = EnvInt(0)
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
    # Fused-RS: when an expert's W1 and W2 each fit one DMA, they alternate between the two
    # weight buffers by expert parity and the next expert's are fetched at the start of a tile,
    # behind its whole compute, instead of one buffer refilled between matmuls (exact).
    SGLANG_TPU_FUSED_RS_WEIGHT_PINGPONG = EnvBool(False)
    # Expert-TP fused-RS: the two cores of a chip add their partials on-chip and send one
    # pair sum per row to another chip's owner, halving off-chip bytes (one extra bf16 rounding).
    SGLANG_TPU_FUSED_RS_ETP_PAIR_EXCHANGE = EnvBool(False)
    # Pair exchange: issue each tile's five runs (own, sibling, one per other chip) from a
    # per-tile table built with the routing metadata instead of walking all owners in-kernel.
    SGLANG_TPU_FUSED_RS_PAIR_RUN_TABLE = EnvBool(False)
    # Fused-RS: GMM2 stores each 256-column block straight into the row-DMA staging layout
    # (overlaps the relayout with the next block's matmul) instead of relayouting the tile after.
    SGLANG_TPU_FUSED_RS_GMM2_ROW_OUT = EnvBool(False)
    # Pair exchange with the run table on a 4-chip ring: send the diagonal chip's pair sums
    # through a neighbor chip, which adds them to its own run for that owner (no 2-hop
    # transfers; one more bf16 rounding). Relay direction alternates by tile.
    SGLANG_TPU_FUSED_RS_PAIR_RELAY = EnvBool(False)
    # Chip offset (in ep-rank chip order) of the diagonal chip on the ring.
    SGLANG_TPU_FUSED_RS_PAIR_RELAY_DIAG = EnvInt(2)
    # Pair exchange: add the sibling's partial only over the tile's valid rows (bit-identical).
    SGLANG_TPU_FUSED_RS_PAIR_WINDOW_ADD = EnvBool(False)
    # Pair exchange with the run table: add the sibling's partial only over the rows of this
    # core's three forwarded runs (the only pair sums it sends); bit-identical.
    SGLANG_TPU_FUSED_RS_PAIR_RUN_ADD = EnvBool(False)
    # Pair exchange combine: derive each owned rank's token and top-k weight from a stable sort
    # of this rank's own (token, slot) pairs instead of argsorting every row and gathering back.
    # Same arrays (bit-identical).
    SGLANG_TPU_FUSED_RS_PAIR_LOCAL_RANKS = EnvBool(False)
    # Batched RPA: run target-verify rows (speculative_num_draft_tokens per request) as MIXED
    # with a q block of that width; the prefill block search otherwise pads each request's
    # rows to a ~2048-row block (16 x 8 rows at 150K context: 42 -> 2.2 ms per layer).
    SGLANG_TPU_RPA_BATCHED_VERIFY_BLOCKS = EnvBool(False)
    # Batched RPA, mixed verify: keep the MIXED batch's order (prefill sequences, then the
    # decoding tails) and run the tails as the verify pass's MIXED range, instead of
    # permuting every layer's q/k/v rows to put the tails first and permuting the output back.
    SGLANG_TPU_MIXED_VERIFY_TAILS_LAST = EnvBool(False)
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
    # Transposed prefill tiles skip the compute blocks past the last query of a
    # partial q block (a sequence's last) instead of scoring padded rows. Exact.
    SGLANG_TPU_RPA_SKIP_PAD_Q = EnvBool(False)
    # Query tokens per iteration of that skip's loop (fewer, longer iterations overlap better).
    SGLANG_TPU_RPA_SKIP_PAD_Q_GROUP = EnvInt(128)
    # "bq:bkv": batched-RPA prefill q / k block sizes for full-attention layers and
    # for sliding-window layers, instead of the VMEM search's; unset keeps the search.
    SGLANG_TPU_RPA_PREFILL_BLOCKS = EnvStr(None)
    SGLANG_TPU_RPA_SWA_PREFILL_BLOCKS = EnvStr(None)
    # Batched RPA outputs come per-head RMS-normalized (weightless, this eps), as Muse
    # Spark's attention output norm; the model then skips its own. Muse only.
    SGLANG_TPU_RPA_OUT_HEAD_NORM_EPS = EnvFloat(None)
    # Transposed tiles with SGLANG_TPU_RPA_NO_MAX run an unshifted streaming softmax
    # (no running max, no alpha rescale) over key chunks of this many keys (the largest
    # divisor of the k block up to it); unset streams the whole k block up to 1536 keys.
    SGLANG_TPU_RPA_TRANSPOSED_KEY_CHUNK = EnvInt(None)
    # Rejected experiments of the unshifted / transposed tiles (kept for A/B): every Nth
    # key chunk's exp2 on the VALU, fp8 PV in the shifted tile, V^T per key chunk, bf16
    # QK scores. Rolled q-chunk loop: SGLANG_TPU_RPA_QCHUNK_LOOP (transposed.py).
    SGLANG_TPU_RPA_VALU_EXP2_EVERY = EnvInt(0)
    SGLANG_TPU_RPA_TRANSPOSED_FP8_P = EnvBool(False)
    SGLANG_TPU_RPA_CHUNK_VT = EnvBool(False)
    SGLANG_TPU_RPA_S_BF16 = EnvBool(False)
    # Pin the RPA prefill q block / q compute chunk of full-attention layers (bq_c is
    # reduced to a divisor of bq); unset uses calculate_block_sizes's power-of-2 search
    # (e.g. bq=1024, bkv=1536 on Muse Spark global layers and 512:512 on window=2048).
    SGLANG_TPU_FORCE_RPA_PREFILL_BQ = EnvInt(None)
    SGLANG_TPU_FORCE_RPA_PREFILL_BQ_C = EnvInt(None)
    # Batched RPA: in-kernel trace regions (step body, full / masked tile), visible in
    # xprof with libtpu's --xla_enable_custom_call_region_trace=true. Profiling only.
    SGLANG_TPU_RPA_TRACE_REGIONS = EnvBool(False)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False
