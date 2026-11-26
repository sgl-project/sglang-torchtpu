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
    # Chip box (XxYxZ) of each --dp-size replica of a multi-host launch, e.g. 2x2x4 for
    # one replica per z column of a 4x4x4 (node_normalization.order_hosts_by_replica).
    SGLANG_TPU_DP_REPLICA_SHAPE = EnvStr(None)
    # Opt-in, request-tagged benchmark admission (benchmark_batch_gate.py).
    # Zero preserves ordinary scheduling, including continuous batching.
    SGLANG_TPU_BENCHMARK_BATCH_SIZE = EnvInt(0)
    SGLANG_TPU_BENCHMARK_TIMEOUT_SEC = EnvFloat(600.0)
    SGLANG_TPU_RELEASE_CHECKPOINT_PAGE_CACHE = EnvBool(True)
    SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS = EnvBool(True)
    # Host heap (see host_heap_trim.py)
    SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED = EnvBool(True)
    # Opt-in warmup & runtime optimization: pin EXTEND/MIXED forwards to LogitsOutputMode.FULL_VOCAB
    # so distributed TopK / greedy / Gumbel-Max variants are only compiled for DECODE buckets.
    SGLANG_TPU_EXTEND_FULL_VOCAB_LOGITS = EnvBool(False)
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
    # DSA indexer token split, always on with attention TP > 1: prefill
    # (EXTEND / MIXED) batches of at least this many tokens (and at least 32
    # rows per rank) have every attention-TP rank store the index keys of all
    # tokens but score and top-k-select only its contiguous slice of
    # ceil(T / attn_tp) query rows (rounded up to 8); an int32 all-reduce over
    # the attention-TP group puts the [T, topk] rows together
    # (attention_pallas_mla_sparse.py). Same rows bit for bit; attention is
    # unchanged. The pooled (kpool) indexer (GLM-5.3-Flash) splits the same
    # way, and the ranks exchange their rows' picked pools ([T, topk // kpool])
    # and expand them to tokens afterwards. No cap on the rows per rank (the
    # all-reduce buffer, [attn_tp, rows, topk] int32 per DSA layer, grows with
    # the chunk); decode and verify keep the replicated path.
    SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT_MIN_TOKENS = EnvInt(1024)
    # Sparse MLA, on top of the indexer token split: each rank attends only the
    # rows it selected, with every head, so each selected cache row is gathered
    # by one rank instead of all; a Pallas all-to-all (kernels/oneshot_all_to_all.py)
    # moves queries and outputs, and the rows' all-reduce is skipped. Plain DSA
    # prefill batches of the per-query kernel; attention TP must span the world.
    SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT = EnvBool(False)
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
    # Serving-time decode forwards of a dynamo-compiled decoder run only
    # Dynamo's differentiating guards (tensor shapes/dtypes and every guard
    # that failed during warmup) instead of the full guard set, once warmup
    # froze compilation with fail_on_recompile. Picks the same precompiled
    # graph; saves ~0.1-0.3 ms of host time per decode step.
    SGLANG_TPU_DECODE_SKIP_GUARD_EVAL = EnvBool(False)
    # Extend skip_guard_eval_unsafe to speculative TARGET_VERIFY / DRAFT_EXTEND_V2
    # forwards (both target verify and block-draft forwards run under TARGET_VERIFY).
    SGLANG_TPU_VERIFY_SKIP_GUARD_EVAL = EnvBool(False)
    # Extend skip_guard_eval_unsafe to bucketed EXTEND forwards once warmup froze
    # compilation with fail_on_recompile.
    SGLANG_TPU_EXTEND_SKIP_GUARD_EVAL = EnvBool(False)
    # Fast-path serving calls to @compiled_op helpers: bypass signature.bind and
    # pytree stride loops on canonical positional tensors, and skip full Dynamo
    # guard evaluation once warmup froze compilation with fail_on_recompile.
    SGLANG_TPU_FAST_COMPILED_OP = EnvBool(False)
    # Extend FutureMap direct tensor handoff (zero index_copy_ / gather / event
    # record) to HostPaddedBlockDraftInput (DFLASH/DSPARK) and non-speculative
    # HostPaddingFutureMap (d=1), flushing only when batch composition changes.
    SGLANG_TPU_SPEC_DIRECT_HANDOFF = EnvBool(False)
    # Pack [full, swa, committed_lens] verify addressing into one H2D upload, fuse
    # full+SWA select_verify_rows with build_draft_block_inputs into one compiled op,
    # and emit greedy verify_tokens directly inside DFlashDraftModel.forward.
    SGLANG_TPU_DFLASH_FUSE_STEP_OPS = EnvBool(False)
    # Opt-in v2 step fusion extending DFLASH_FUSE_STEP_OPS: collapse Launch 1
    # (prepare_dflash_block_step + draft forward) and Launch 3 (accept +
    # write_draft_kv) into single compiled dispatches on TpuBlockSpecWorker.
    SGLANG_TPU_DFLASH_FUSE_STEP_OPS_V2 = EnvBool(False)
    # Run all-greedy TARGET_VERIFY and non-speculative DECODE/EXTEND steps with
    # distributed top-k width k=1 (LogitsOutputMode.DISTRIBUTED_GREEDY) instead of 64.
    SGLANG_TPU_SPEC_VERIFY_GREEDY_TOPK1 = EnvBool(False)
    # Clear running_batch immediately when all requests in it have finished under
    # overlap scheduling (eliminates 8k/1 max_new_tokens=1 zombie Req accumulation).
    SGLANG_TPU_FAST_FILTER_FINISHED_BATCH = EnvBool(False)
    # Reuse uploaded RPA decode/verify page_indices and swa_page_indices in O(1)
    # across target and draft forwards while req_to_token version and batch rows hold.
    SGLANG_TPU_RPA_CACHE_DECODE_PAGE_TABLE = EnvBool(False)
    # Skip unused eager decode orig_seq_lens+1 updates and fuse global top-k
    # sampler ops into fused_eager_ops programs on TPU.
    SGLANG_TPU_DECODE_FAST_HOST = EnvBool(False)
    # Deduplicate aliased accept_lens/block_accept_lens D2H copies and batch result
    # + kv_fence_tail D2H through GIL-releasing torch_tpu batch_transfer_d2h.
    SGLANG_TPU_FAST_RESULT_COPY = EnvBool(False)
    # Snapshot/restore only the ScheduleBatch fields mutated mid-forward on TPU
    # HostPaddedDraftInput decode steps instead of iterating all dataclass fields.
    SGLANG_TPU_FAST_FORWARD_ISOLATION = EnvBool(False)
    # Log per-phase host wall times (us) in TpuBlockSpecWorker every 32 decode steps.
    SGLANG_TPU_SPEC_STEP_TIMERS = EnvBool(False)
    # Filter finished requests from running_batch before prefill admission on single-queue
    # TPU overlap scheduling so freed slots admit replacements one decode step earlier.
    SGLANG_TPU_EAGER_FILTER_RUNNING_BATCH = EnvBool(False)
    # Admit 1-token prefill requests directly into ForwardMode.DECODE (via FutureMap)
    # instead of launching a separate padded EXTEND / MIXED step.
    SGLANG_TPU_PIGGYBACK_SINGLE_TOKEN_PREFILL = EnvBool(False)
    # Defer underfilled chunked prefills while decode concurrency is saturated so
    # multi-request arrivals pack prefill token buckets near 100% utilization.
    SGLANG_TPU_PACK_PREFILL_TO_BUCKET = EnvBool(False)
    SGLANG_TPU_PACK_PREFILL_TARGET_TOKENS = EnvInt(0)
    SGLANG_TPU_PACK_PREFILL_MAX_DELAY_STEPS = EnvInt(14)
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
    # GLM-5.3-Flash: target steps (the MTP verify, or decode without speculation)
    # as one Pallas call per device (glm_flash_megakernel/); other steps run the
    # model as usual. ..._MAX_LEN caps the request length the kernel takes
    # (default: the context length); ..._MAX_ROWS the rows (requests x draft
    # tokens) of one step; a larger step runs as several calls of at most
    # ..._CALL_ROWS rows each. With the context-parallel DSA, 16 rows is the largest
    # bucket that fits tpu7x's 64 MiB VMEM with 256-column dense tiles; steps with more
    # requests run on the model path (measured 2026-11-12).
    SGLANG_TPU_GLM_FLASH_MEGAKERNEL = EnvBool(False)
    SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN = EnvInt(None)
    SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_ROWS = EnvInt(16)
    SGLANG_TPU_GLM_FLASH_MEGAKERNEL_CALL_ROWS = EnvInt(32)
    # Experimental: the kpool indexer's StreamIndex page table stays in HBM with
    # one row per sequence in SMEM, so a context past ~262k fits the 1 MiB SMEM.
    SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM = EnvBool(False)
    # Pooled DSA indexer: keep each request's raw key|gate rows in a ring of pages per
    # request slot instead of one row per pool token; only the open pool is read back.
    SGLANG_TPU_DSA_KPOOL_RAW_RING = EnvBool(False)
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
    # DFLASH draft linears, including the DFlash2 conv kernel projections, as fp8
    # weight-only (fp8_weight_linear.py) for decode-sized calls; prefill-sized calls
    # keep the bf16 weights. Changes the draft's proposals, not verification.
    SGLANG_TPU_DFLASH_FP8_DRAFT = EnvBool(False)
    # Multiplier applied to max_running_requests * sliding_window when sizing the
    # compact SWA draft KV pool on TpuUnifiedSWAKVPool under hierarchical cache.
    SGLANG_TPU_SPEC_DRAFT_SWA_WINDOW_MULTIPLIER = EnvFloat(4.0)

    # Muse Spark: whole-decoder Pallas megakernel for decode and DFlash target-verify
    # steps, with fp8 dense weight copies and an fp8 weight-only LM head.
    SGLANG_TPU_MUSE_MEGAKERNEL = EnvBool(False)
    # Muse Spark prefill kernel set: fused_rs_muse GMM-RS, batched_rpa_muse prefill
    # attention (Wave-22 change_b) and the muse_prefill_kernels layer pipeline on
    # shapes they support.
    SGLANG_TPU_MUSE_KERNEL_ENABLED = EnvBool(False)
    # Native TP16/EP16 prefill with one KV head per rank (A1 v1, A3 v3).
    SGLANG_TPU_MUSE_KERNEL_TP16_PREFILL = EnvBool(False)
    SGLANG_TPU_MUSE_KERNEL_TP16_OWNER_A1 = EnvBool(False)
    # fused_rs_muse W4A16 fallback tiles: scale each K256 x MXU-width FP4 block with one
    # multiply instead of one per 16-row scale group. Read only by fused_rs_muse.
    SGLANG_TPU_MUSE_KERNEL_VECTORIZED_DEQUANT = EnvBool(False)
    # fused_rs_muse prefill tiles: "tile_m,tile_k1,tile_n1,tile_k2,tile_n2,w1_bufs,w2_bufs"
    # replaces the tuned tiles of FP4-expert calls with size_m >= 16384 (prefill). Empty
    # keeps the tuned table. For A/B; an over-budget tuple fails at TPU compile.
    SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES = EnvStr("")
    # Debug: also run each prefill-kernel layer natively and log the per-layer stream
    # difference (muse_prefill_kernels/integration.py). Doubles prefill cost.
    SGLANG_TPU_MUSE_KERNEL_DEBUG_COMPARE = EnvBool(False)
    # Muse prefill kernel A1: v1 = bf16 all-gather + one kernel; v5 = two bf16 half
    # all-gathers overlapped with the kernel; v6 = v5 on static x32 fp8 activations;
    # v5x4 / v5x8 = v5 in 4 / 8 parts (needs SparseCore all-gather offload to overlap).
    SGLANG_TPU_MUSE_KERNEL_A1_VARIANT = EnvStr("v1")
    # Muse prefill kernel A3: v3 = o_proj + reduce-scatter; v4_3d = bf16 all-to-all +
    # o_proj on the token owner; v16 = the same on static fp8 in one Pallas kernel.
    # v4_3d / v16 keep a replicated fp8 Wo: +3.3 GiB per device over 60 layers.
    SGLANG_TPU_MUSE_KERNEL_A3_VARIANT = EnvStr("v3")
    # Muse prefill kernels: A3 normalizes the attention output per head in its prologue
    # (v3 also reads its halves in place), and MoE dequant half 0 allocates its buffer.
    SGLANG_TPU_MUSE_KERNEL_FUSED_GLUE = EnvBool(False)
    # Muse prefill kernels: all-gather the fp32 stream out of the last kernel layer in bf16
    # (half the bytes); the native layer after it then reads the stream rounded to bf16 once.
    SGLANG_TPU_MUSE_KERNEL_EXIT_GATHER_BF16 = EnvBool(False)
    # pallas_tpu_rpa_v3 extend/prefill runs batched_rpa_muse (change_b on full-attention
    # layers when head_dim is 64, i.e. SGLANG_TPU_KV_PACK_HEADS off) on the same KV cache;
    # decode stays on v3.
    SGLANG_TPU_MUSE_KERNEL_BATCHED_PREFILL_ATTN = EnvBool(False)
    # fused_rs_muse routing: order rows by expert with a one-hot count, prefix sum and
    # scatter instead of a sort (same tensors). Only when rows x experts <= 1M (decode).
    SGLANG_TPU_MUSE_KERNEL_SORTFREE_METADATA = EnvBool(False)
    # fused_rs_muse NVFP4 decode tiles of at most this many rows take the block-diagonal
    # matmul (1, 2, 4 or 8; 0 turns it off); larger tiles take the row buckets.
    SGLANG_TPU_MUSE_KERNEL_BLOCKDIAG_MAX_ROWS = EnvInt(8)
    # fused_rs_muse without expert TP: build the GM tile map in the kernel's scalar
    # prologue (stock fused_rs) instead of XLA. Tiles differ only when a group plus its
    # sublane offset spans more than tile_m rows.
    SGLANG_TPU_MUSE_KERNEL_SKIP_PRECOMPUTED_METADATA = EnvBool(False)
    # fused_rs_muse block-scatter prefill: pass precomputed block-scatter metadata as a
    # 1D SMEM ref instead of 2D [slots, 26] (padded to 128 minor lanes) and skip index
    # packing when block_scatter is active (removes scalar div/mod at T=16384).
    SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META = EnvBool(False)
    # Muse prefill MoE zero-glue expert load balancing (muse_prefill_kernels/eplb.py):
    # "" = off; "perm" = zero-HBM router-column + expert bin-packed permutation;
    # "stripe" = permutation + R hot-expert replicas rotated per tile_m-token stripe (0 us glue);
    # "home" = permutation + R hot-expert replicas routed to each token's home chip (0 us glue);
    # "q_block" = permutation + R hot-expert replicas with 64-block full-tile water-filling.
    # Muse: JSON {"layers": {"<layer>": [256 checkpoint expert ids]}}; entry p is the checkpoint
    # expert loaded into slot p. Gate rows and e_score_correction_bias are permuted the same way,
    # so routing picks the same experts. Unlisted layers keep the checkpoint order.
    SGLANG_TPU_MUSE_KERNEL_EXPERT_PLACEMENT = EnvStr("")
    # Prefill EPLB modes: "", "perm", "stripe", "home", "q_block", "dyn_holder", "dyn_split".
    SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE = EnvStr("")
    # Replicated hot experts per EP rank for PREFILL_EPLB_MODE in {"stripe", "home", "q_block", "dyn_holder", "dyn_split"} (0..8).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_REPLICAS = EnvInt(0)
    # Muse prefill §4.5b cross-chip preloaded hot-expert replica slots per rank (0 = off; 1, 2, or 4).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_REPLICA_SLOTS = EnvInt(0)
    # Muse prefill §4.5b runtime planner mode ("whole" = paper L_home-c >= L_holder+c; "split" = Fig 14 Right tile-quantum split).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_REPLICA_MODE = EnvStr("whole")
    # Muse prefill §4.3 pure-EP dynamic intra-chip chipmate weight redistribution over die-to-die link.
    SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE = EnvBool(False)
    # Muse prefill §4.3 chipmate mode ("whole" = paper pair-max-tile expert move; "split" = outlier expert tile-quantum split).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE = EnvStr("whole")
    # Muse prefill §4.2b hot-expert tile interleaving with dedicated VMEM hot W1 bank on FP4 prefill.
    SGLANG_TPU_MUSE_KERNEL_PREFILL_HOT_INTERLEAVE = EnvBool(False)
    # Optional .json/.npy/.pt [num_layers, 256] expert token-count profile for prefill EPLB
    # (loaded at startup if present; written after warmup calibration if CALIB_FORWARDS > 0).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_STATS_PATH = EnvStr("")
    # Warmup EXTEND forwards to accumulate active-token expert counts before 1-shot rebalance
    # (0 = off; uses STATS_PATH if set).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_CALIB_FORWARDS = EnvInt(0)
    # Eligible non-dummy EXTEND forwards to skip before accumulating online EPLB counts.
    SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_SKIP_FORWARDS = EnvInt(0)
    # Minimum active EXTEND tokens (sum(extend_seq_lens)) required for an online EPLB sample;
    # filters out startup dummy precompile buckets and tiny HTTP warmup probes.
    SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MIN_CALIB_TOKENS = EnvInt(256)
    # Eligible EXTEND forwards per rolling skew check after initial EPLB calibration (0 = 1-shot only).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_REBALANCE_INTERVAL = EnvInt(16)
    # Mean per-layer max/mean rank tile skew threshold that triggers a rolling EPLB rebalance.
    SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_SKEW_THRESHOLD = EnvFloat(1.22)
    # Muse prefill MoE: mask padded tail tokens (tok >= sum(extend_seq_lens)) out of expert
    # routing and GMM-RS so pad rows neither execute expert matmuls nor pile onto 8 pad experts.
    SGLANG_TPU_MUSE_KERNEL_SKIP_PAD_MOE_TOKENS = EnvBool(False)
    # Opt-in 0-cost-when-off per-stage isfinite trap over real tokens in MusePrefillKernels.
    SGLANG_TPU_MUSE_NAN_TRAP = EnvBool(False)
    # fused_rs_muse NVFP4 short/medium prefill (4096 <= size_m <= 16384, i.e. T in {512, 1024, 2048}):
    # use half-N W1+W2-cached (128,4096,2048,4096,2048,2,2) tiles and skip multi-bucket W4A16 unroll
    # on block_scatter calls (decode size_m <= 2048 and T >= 4096 stay unchanged).
    SGLANG_TPU_MUSE_KERNEL_SHORT_PREFILL_TILES = EnvBool(False)
    # fused_rs_muse NVFP4 decode: the expert scales are constant over each 256-row K block
    # (SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES, required), so every decode tile sends the
    # e2m1 codes to the MXU as FP8 and scales each K256 partial product once (no BF16
    # dequant, no block-diagonal expansion).
    SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES = EnvBool(False)
    # fused_rs_muse NVFP4 decode, lossless: calls with at least this many routed rows
    # (1 = every decode call, 0 = off; decode is <= 4096) run every tile as BF16-dequant
    # dots per K256 chunk in a loop over 256-column strips instead of the unrolled W4A16
    # strips and blockdiag branches (a much smaller program). BLOCK256_SCALES implies them.
    SGLANG_TPU_MUSE_KERNEL_K256_DECODE_TILES = EnvInt(0)
    # fused_rs_muse FP4 decode tiles (routed rows <= 4096) as 7 comma-separated ints
    # tile_m,tile_k1,tile_n1,tile_k2,tile_n2,num_w1_bufs,num_w2_bufs; empty = tuned table.
    # Applies from K256_DECODE_TILES rows up when that is set. With K256 tiles,
    # 128,4096,2048,4096,2048,2,2 double-buffers half-N weights.
    SGLANG_TPU_MUSE_KERNEL_DECODE_GMM_TILES = EnvStr("")
    # Muse decode D1: ignore SGLANG_TPU_MUSE_KERNEL_EXPERT_PLACEMENT unless disaggregation_mode
    # is "prefill", preventing prefill-calibrated expert permutations from unbalancing decode.
    SGLANG_TPU_MUSE_KERNEL_DECODE_SKIP_EXPERT_PLACEMENT = EnvBool(False)
    # Muse decode D2a: requantize NVFP4 experts at load to compact [E, K/256, N] bf16 scales
    # instead of repeating each K256 scale 16x in the block-16 layout (-10.8 GiB HBM, -16x
    # scale DMA bytes, bit-exact with PROBE_BLOCK256_SCALES + BLOCK256_SCALES).
    SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES = EnvBool(False)
    # Muse decode D2b: column strip width in _block256_gmm{1,2}_tile (0 = default 256; >0 fixed
    # e.g. 512/1024; -1 = adaptive 1024/512/256; -2 = adaptive 512/256; -3 = adaptive 2048/1024/512/256;
    # -4 = adaptive 2048(m<=32)/1024(m<=64)/512; -5 = adaptive 2048(m<=16)/1024(m<=64)/512).
    SGLANG_TPU_MUSE_KERNEL_DECODE_STRIP_NC = EnvInt(0)
    # Muse decode D2b: dual-MXU / spill-free K256 schedule in _block256_gmm1_tile:
    # 0 = off; 1 = interleaved gate+up single K-loop; 2 = VMEM-staged silu(gate) before up;
    # 3 = hybrid (1 for m<=32, 2 for m>=64) + 32-row fast-path LHS/post-norm bucketing;
    # 4 = hybrid (1 for m<=16, 2 for m>=32) + 2-strip unroll + 16/32/64-row LHS/post-norm/flush;
    # 5 = all-vmem_stage + 2-strip unroll + 16/32/64-row LHS/post-norm/flush;
    # 6 = 2-bucket [16, 32] compact program looping >32-row tiles over 32-row chunks.
    SGLANG_TPU_MUSE_KERNEL_DECODE_DUAL_K256 = EnvInt(0)
    # Ordinary EP16 decode, compact FP4 K256, B16/B32 only.
    SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_EP16 = EnvBool(False)
    # Ordinary EP8 decode with native K16 scales and the source fused router.
    SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_NATIVE_EP8 = EnvBool(False)
    # Muse decode Fuse-All (Track 1): sparse fuse-all decode Pallas kernel for pure
    # EP8/EP16 with compact K256 or native K16 FP4 scales across B in {16,32,64,128,256}.
    SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL = EnvBool(False)
    # Muse decode symmetric EP(P/2)xTP2 (Track 2): unlocks hybrid EP4xTP2 (8 ranks) and
    # EP8xTP2 (16 ranks) expert sharding in MuseSparkMoE and runs the symmetric TP2
    # fuse-all decode Pallas kernel for both compact K256 and native K16 scales.
    SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2 = EnvBool(False)
    # Muse prefill hybrid EP(P/2)xTP2 unlock: enables sequence-parallel MusePrefillKernels
    # (A1 + RPA + A3 + B1 + B2 + paired-virtual EPxTP2 fused-RS GMM + D) when
    # tp == attn_tp_size == 2 * moe_ep_size (e.g. EP4xTP2 on 8 ranks, EP8xTP2 on 16 ranks).
    SGLANG_TPU_MUSE_KERNEL_EP_TP2_PREFILL = EnvBool(False)
    # Muse prefill finer 1024-token buckets (T in {3072, 5120, 6144, 7168}, t_shard % 128 == 0):
    # splits uneven A1 parts in 128-row multiples and uses 1-pass B2 when t_shard % 256 != 0.
    SGLANG_TPU_MUSE_KERNEL_FINER_PREFILL_BUCKETS = EnvBool(False)
    # Muse decode §4.5b cross-chip preloaded hot-expert replica slots per owner
    # (0 = off; 1, 2, or 4). Composes with DECODE_FUSE_ALL and DECODE_EP_TP2.
    SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS = EnvInt(0)
    # Optional [num_layers, 256] calibration counts file (.json/.npy/.pt/.jsonl) for
    # selecting preloaded hot-expert replicas when DECODE_REPLICA_SLOTS > 0.
    SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_STATS_PATH = EnvStr("")
    # Cost proxy for the in-kernel §4.5b home-vs-holder planner ("active", "tiles", "weighted").
    SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_COST_MODE = EnvStr("tiles")
    # Offline §4.5b replica candidate selection ("global_top" or "bottleneck_owner").
    SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SELECTION_MODE = EnvStr("global_top")
    # Offline §4.5b calibration weight objective ("decode_active", "decode_hybrid", "prefill_rows").
    SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_OBJECTIVE = EnvStr("decode_active")
    # Muse decode §4.3 pure-EP dynamic intra-chip chipmate weight redistribution
    # (requires DECODE_FUSE_ALL=1 on pure EP8/EP16).
    SGLANG_TPU_MUSE_KERNEL_DECODE_CHIPMATE_REBALANCE = EnvBool(False)
    # Allow Muse Fuse-All decode MoE to run on speculative TARGET_VERIFY batches
    # (e.g. DFlash verify batches of 128 or 256 tokens at c64).
    SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_TARGET_VERIFY = EnvBool(False)
    # Allow Muse decode QKV epilogue (D4) and RPA skip_mixed decode fast path
    # on uniform-width speculative TARGET_VERIFY batches (default off).
    SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST = EnvBool(False)
    # Allow pure-EP Muse Fuse-All decode MoE to coexist zero-copy with static prefill
    # Zero-Glue EPLB `[base_experts + R, ...]` slabs (PREFILL_EPLB_CALIB_FORWARDS must be 0).
    SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB = EnvBool(False)
    # Opt-in circular-ring Fuse-All decode MoE v2 (101eefc4d7) for NVFP4 layers
    # when DECODE_FUSE_ALL=1: gated to K16 (B>=32) and K256 (B=64); other batch
    # sizes stay on v1 sparse unless SGLANG_TPU_MUSE_KERNEL_DECODE_V2_BATCHES is set.
    SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_V2 = EnvBool(False)
    # Wave-8 granular overrides for fused_decode_sparse_v2 & HBM-wall breakers:
    SGLANG_TPU_MUSE_KERNEL_DECODE_V2_BATCHES = EnvStr("")
    SGLANG_TPU_MUSE_KERNEL_DECODE_V2_RING = EnvStr("")
    SGLANG_TPU_MUSE_KERNEL_DECODE_PAIR_RS = EnvInt(-1)
    SGLANG_TPU_MUSE_KERNEL_DECODE_VMEM_DEQ = EnvInt(0)
    SGLANG_TPU_MUSE_KERNEL_DECODE_FP4_UNPACK = EnvStr("")
    SGLANG_TPU_MUSE_KERNEL_DECODE_ASYNC_GATHER = EnvInt(-1)
    SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_NORM_SCATTER = EnvInt(-1)
    SGLANG_TPU_MUSE_KERNEL_DECODE_M_BUCKETS = EnvStr("")
    SGLANG_TPU_MUSE_KERNEL_DECODE_LHS_FP8 = EnvBool(False)
    SGLANG_TPU_MUSE_KERNEL_DECODE_V2_CONTIG_PARTS = EnvInt(0)
    SGLANG_TPU_MUSE_DECODE_TOPK_MASS = EnvFloat(0.0)
    SGLANG_TPU_MUSE_DECODE_MIN_K = EnvInt(2)
    SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_MIN_RANK = EnvInt(6)
    SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_MAX_WEIGHT = EnvFloat(0.105)
    SGLANG_TPU_MUSE_DECODE_COLD_W3_EXPERTS = EnvInt(0)
    SGLANG_TPU_MUSE_DECODE_COLD_W3_MODE = EnvStr("inplace")
    SGLANG_TPU_MUSE_DECODE_COLD_W3_ALPHA = EnvFloat(0.75)
    SGLANG_TPU_MUSE_DECODE_REPLICA_GREEDY_PEAK = EnvBool(False)
    SGLANG_TPU_MUSE_DECODE_REPLICA_CHIP_AWARE = EnvBool(False)
    # Opt-in boundary-layer FP8 Fuse-All decode MoE (101eefc4d7) for L0/L61
    # float8_e4m3fn experts when DECODE_FUSE_ALL=1: gated to B<=128.
    SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_FP8 = EnvBool(False)
    # Muse decode Wave-2: branchless scalar holder-candidate extraction + hoisted VREG
    # replica vectors inside fused_decode_sparse (eliminates dynamic SMEM/VPU R>=1 tax).
    SGLANG_TPU_MUSE_KERNEL_DECODE_STATIC_HOLDER_MASK = EnvBool(False)
    # Muse decode Wave-2: enable 2-stage chipmate-bridge All-Gather + BF16 RS wire at B=64
    # inside fused_decode_sparse.
    SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_BRIDGE64 = EnvBool(False)
    # Muse decode Wave-2: speculative first-expert HBM->VMEM weight prime at kernel entry
    # overlapped under in-kernel routing (0 = off; 1, 2, or 3 weight chunks primed).
    SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME = EnvInt(0)
    # Muse decode Wave-2: run L0 and L61 channelwise FP8 MoE boundary layers on the
    # single-launch stream3_half FP8 sparse decode kernel across B<=128 (prioritized
    # over DECODE_SPARSE_FP8).
    SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_FUSE_ALL = EnvBool(False)
    # Muse decode Wave-7: fast B=64 sparse decode execution mode inside fused_decode_sparse
    # (0 = stock; 1 = single-chunk 64x256 VPU routing + unrolled prologue/RS + m=8 fast bucket
    # + loop-free <=8-row gather/scatter; 2 = 1 + contiguous w1_2x_w2_1x + spec_prime;
    # 3 = dense-rows B=64 mode eliminating token pack/gather/switch/scatter).
    SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_B64 = EnvInt(0)
    # Muse decode Wave-7: L0/L61 FP8 sparse decode streaming variant ("" = stream3_half,
    # "w1_k2_w2_1x" = 3x16 MiB contiguous K-half streaming, "w1_k2_w8a8" = contiguous K-half + W8A8).
    SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_MODE = EnvStr("")
    # Muse Wave-7: load-time channelwise FP8 -> NVFP4 conversion for L0 & L61 boundary MoE layers:
    # 0 = off; 256 / 16 = decode-only NVFP4 shadow cache (EXTEND prefill stays exact FP8);
    # -256 / -16 = full in-place NVFP4 conversion (frees 3.0 GiB/rank FP8 weights).
    SGLANG_TPU_MUSE_QUANT_BOUNDARY_MOE_FP4 = EnvInt(0)
    # Muse decode Wave-7: prewarm & enable token-sharded [B // world, 4096] output ops
    # (return_sharded=True) on fused sparse/fp8 MoE decode for Bridge-2 overlapped XOR=1 AG.
    SGLANG_TPU_MUSE_KERNEL_DECODE_RETURN_SHARDED = EnvBool(False)
    # Muse Wave-8: decoupled m_buckets=(8, 32) at B<=64 inside fused_decode_sparse (bit-exact).
    SGLANG_TPU_MUSE_KERNEL_DECODE_M8 = EnvBool(False)
    # Muse Wave-8: hoist K256 scale strip read outside the 16-chunk K-loop in fused_decode_sparse.
    SGLANG_TPU_MUSE_KERNEL_DECODE_HOIST_SCALES = EnvBool(True)
    # Muse Wave-8: zero-extra-HBM K-contiguous 8-MiB part streaming (stream3_k2) in fused_decode_sparse.
    SGLANG_TPU_MUSE_KERNEL_DECODE_CONTIG_PARTS = EnvBool(False)
    # Muse Wave-8 HBM-wall breaker: dynamic Top-K routing (<0.5 = min normalized prob floor p_min,
    # e.g. 0.02; 0.5..1.0 = cumulative top-8 mass cap, e.g. 0.92; 0.0 = off).
    SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK = EnvFloat(0.0)
    # Muse Wave-8 HBM-wall breaker: batch-coupled "free-an-expert" tail pruning ("rows[:max_w[:min_slot]]",
    # e.g. "2:0.06:5" drops trailing hits only when doing so frees all rows of that expert; "" = off).
    SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS = EnvStr("")
    # Muse Wave-9 (#2407): bucket-scaled FREE pruning for tail decode buckets (B < 64):
    # "b64_ms16" (default) = max(1, round(free_rows * B / 64)) + min_slot >= 2 at B <= 16 (keeps min_slot at B >= 32);
    # "b64" = same row scaling, min_slot >= 2 at B <= 32;
    # "ge32" = disable FREE at B < 32, b64 scaling + min_slot >= 2 at B == 32; "off" = unscaled.
    SGLANG_TPU_MUSE_DECODE_FREE_SCALE_MODE = EnvStr("b64_ms16")
    # Muse Wave-9 muse_sparse granular kernel sub-knobs (all default-OFF = cand_w8_zi baseline):
    SGLANG_TPU_MUSE_KERNEL_DECODE_STRIP_MODE = EnvStr("wide")
    SGLANG_TPU_MUSE_KERNEL_DECODE_NC_SPLIT512 = EnvBool(False)
    SGLANG_TPU_MUSE_KERNEL_DECODE_K512_CHAIN = EnvBool(False)
    SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_RB16 = EnvInt(0)
    SGLANG_TPU_MUSE_KERNEL_DECODE_UNROLL_STRIPS = EnvBool(False)
    SGLANG_TPU_MUSE_KERNEL_DECODE_ROUTE_FAST = EnvBool(False)
    SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_PLAN = EnvInt(0)
    SGLANG_TPU_MUSE_KERNEL_DECODE_FAST_GS = EnvBool(False)
    SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2_PAIR_RS = EnvBool(False)
    # Opt-in B=16 (LC 8K/1K) sparse execution (m_buckets=(8, 16) vs dense) unlocking FREE/MIN_P.
    SGLANG_TPU_MUSE_KERNEL_DECODE_B16_SPARSE = EnvBool(False)
    # Opt-in native B=8 decode MoE compile bucket on TP=8 (ep=8).
    SGLANG_TPU_MUSE_KERNEL_DECODE_B8 = EnvBool(False)
    # Opt-in non-uniform per-layer FREE/MIN_P preset or spec ("" = uniform off).
    SGLANG_TPU_MUSE_DECODE_NONUNIFORM_FREE = EnvStr("")
    # Opt-in (#3035): allow SPEC_PRIME (>0) alongside ROUTE_FAST + FAST_PLAN=2 and optional home slot hint (-1=auto).
    SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME_FAST = EnvBool(False)
    SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_HOME_SLOT = EnvInt(-1)
    # Opt-in DFlash verify (B >= 128) routing-pruning & W4A8/tile overrides ("" / -1 / False = inherit decode):
    SGLANG_TPU_MUSE_DECODE_VERIFY_FREE_EXPERT_ROWS = EnvStr("")
    SGLANG_TPU_MUSE_DECODE_VERIFY_MIN_P_TOPK = EnvFloat(-1.0)
    # Opt-in d=1 scaled FREE/MIN_P at B >= 128 (e.g. c256 concurrency when speculative_algorithm is None).
    SGLANG_TPU_MUSE_DECODE_D1_LARGE_BATCH_FREE = EnvBool(False)
    # Opt-in load-aware rank-capped FREE tail shedding ("<cap>:<tail_rows>:<tail_w>", e.g. "11:2:0.32").
    SGLANG_TPU_MUSE_DECODE_FREE_RANK_CAP = EnvStr("")
    SGLANG_TPU_MUSE_KERNEL_DECODE_TILE_M128 = EnvBool(False)
    # Muse Wave-7B: allow DECODE_REPLICA_SLOTS > 0 alongside PREFILL_EPLB_STATS_PATH when
    # PrefillEplbManager._refresh_decode_replicas keeps _fused_decode_rep_* synchronized.
    SGLANG_TPU_MUSE_KERNEL_DECODE_SYNC_PREFILL_EPLB = EnvBool(False)
    # Muse decode Wave-5: extend decode/verify fused batch caps from 256 -> 384
    # (64 seqs x dq=6 DFlash d=6 verify & draft) across MuseSparkMoE fuse-all decode,
    # decode_norm_router, decode_qkv_epilogue, bridge_all_reduce, and DFlash FP8 draft.
    SGLANG_TPU_MUSE_DECODE_B384 = EnvBool(False)
    # Muse decode D6: row-tiled Pallas VMEM RMSNorm + gated-residual epilogue for decode (<= 256):
    # 0 = off; 1 = Pallas RMSNorm (canonical 2D layout); 2 = fused post-attn + pre-ffn and
    # post-ffn norm+residual; 3 = 2 + cross-layer next input_layernorm fused into post-ffn;
    # 4 = 2 + direct [tp*B, N/tp] post_expert_proj gather fused into post-ffn norm+residual;
    # 5 = 4 + cross-layer next input_layernorm fused into post-ffn.
    SGLANG_TPU_MUSE_KERNEL_DECODE_NORM_LAYOUT = EnvInt(0)
    # Muse decode D6 lane-parallel VPU 128-lane accumulation across strips (1 cross-lane
    # reduction instead of 8-16, matching XLA's 128-lane tree reduction).
    SGLANG_TPU_MUSE_KERNEL_DECODE_NORM_FAST = EnvBool(False)
    # Chipmate-bridge 2-stage two-shot TP8 all-reduce max payload per rank in KiB
    # (0 = off, keeps stock 512 KiB flat twoshot cap; e.g. 2048 enables bridge AR up to 2 MiB).
    SGLANG_TPU_BRIDGE_ALL_REDUCE_MAX_KB = EnvInt(0)
    # Muse decode D6b: use 1-hop Pallas oneshot_all_gather for MuseSparkMoE pre/post_expert_proj
    # on EP8 decode shards <= 128 KiB.
    SGLANG_TPU_MUSE_KERNEL_DECODE_ONESHOT_MOE_AG = EnvBool(False)
    # Muse attention load-time fused [q|k|v|gate] projection (zero extra HBM; releases separate
    # qkv_proj and gate_proj weights after MusePrefillKernels extracts its own copies).
    SGLANG_TPU_MUSE_FUSE_QKVG_PROJ = EnvBool(False)
    # Single-switch lossless decode/verify attention fusion combo: enables load-time [q|k|v|gate]
    # concat + XOR=1 chip-hierarchical o_proj AR (mode=2 bridge_all_reduce w/ nl>=2, else mode=1).
    SGLANG_TPU_MUSE_DECODE_ATTN_QKV_TPAR = EnvBool(False)
    # Muse decode/verify opt-in per-channel weight-only FP8 attention projections ([q|k|v|gate]
    # and o_proj) for B <= 256 (GSM8K-gated).
    SGLANG_TPU_MUSE_DECODE_FP8_ATTN_PROJ = EnvBool(False)
    # Muse decode/verify fused chip-hierarchical XOR=1 bridge o_proj all-reduce + norm/residual:
    # 0 = off; 1 = fused tp_ar_res_norm (o_proj AR + post_attn_norm + gated_res + pre_ffn_norm +
    # bf16 cast); 2 = standalone bridge_all_reduce on o_proj; 3 = full 8-op fused attention tail
    # (per-head norm + sigmoid(gate) + o_proj overlapped w/ bridge RS + AR + res_norm + bf16 cast).
    SGLANG_TPU_MUSE_DECODE_FUSED_O_PROJ_AR = EnvInt(0)
    # Muse decode full-layer MoE boundary fusion (Wave 2): fuses MuseSparkMoE
    # gate + pre_expert_proj + col-AG + pre_expert_norm + sparse MoE + post_expert_proj + col-AG
    # into a single TPU7x Pallas launch across EP in {8, 16}, B in {16,32,64,128,256},
    # compact K256 or native K16 FP4 weights, and R in {0, 1, 2, 4}.
    SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE = EnvBool(False)
    # Boundary fusion scope when SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE=1:
    # "full" = Stage B + Core + Stage A (default); "stage_a" = Core + Stage A only;
    # "stage_b" = Stage B + Core only (composes with DECODE_NORM_LAYOUT=4/5 tp_ag_res_norm).
    SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MODE = EnvStr("full")
    # Pre-pack fresh contiguous K-major [in, out] gate/pre/post projection buffers at load
    # (+0.97 GiB/rank, -11 us/layer & -20.3K bundles vs [out, in]); set 0 for +0 GiB HBM.
    SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_KMAJOR = EnvBool(True)
    # Maximum decode/verify token batch routed to full-layer MoE (default 128; B > MAX_BS
    # dispatches cleanly to fused_decode_sparse so B=256 avoids icache spillover).
    SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MAX_BS = EnvInt(128)
    # Muse decode D5: fused Pallas router top-k and direct decode metadata on decode (<= 256):
    # 0 = off; 1 = Pallas router_topk; 2 = 1 + unsorted-bijection (local_start, local_end, recv_cnt).
    SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_ROUTER = EnvInt(0)
    # fused_rs_muse NVFP4 prefill (routed rows > 4096): run every tile as the K256 dots in a
    # loop over 256-column strips (BF16 dequant; FP8 codes when BLOCK256_SCALES is also set)
    # instead of the unrolled W4A16 strips. Off keeps prefill on the W4A16 strips.
    SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_TILES = EnvBool(False)
    # fused_rs_muse W4A16 strips (BF16-dequantized NVFP4, decode and prefill): K rows per
    # MXU dot; 0 keeps the activation quant block (256), so each strip pops and adds
    # tile_k/256 f32 partials on the VALU. A multiple of 64 dividing tile_k.
    SGLANG_TPU_MUSE_KERNEL_W4A16_K_STEP = EnvInt(0)
    # fused_rs_muse prefill (routed rows > 4096): column strip step width in inner_kernel
    # (0 = default 128; >0 e.g. 256/512/1024/2048 amortizes VMEM LHS/scale loads across strips).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N = EnvInt(0)
    # fused_rs_muse NVFP4 K256 W4A8 prefill (routed rows > 4096, rhs_qbs == 256): LHS online FP8
    # quantization block size (0 = default 256; 512/1024/2048/4096 = coarser/full-row LHS FP8
    # keeping exact NVFP4 weights while collapsing inner K-loop VALU to single vfma).
    SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS = EnvInt(0)
    # fused_rs_muse NVFP4 K256 W4A8 prefill accumulator/schedule mode (0 = f32 single-strip;
    # 1 = bf16 2x packed VPU inner FMA; 2 = paired gate+up dual-MXU strip f32; 3 = paired
    # gate+up dual-MXU strip + bf16 2x packed VPU inner FMA).
    SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE = EnvInt(0)
    # fused_rs_muse K256_PREFILL_TILES=1 prefill: 0 = default BF16 LHS; 256 = per-(row, K256)
    # fp8xFP8 2x MXU dot; 4096 = per-(row, full-K) fp8xFP8 2x MXU dot.
    SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_FP8 = EnvInt(0)
    # fused_rs_muse K256_PREFILL_TILES=1 prefill: column strip width nc_cur (0 = default 256;
    # >0 e.g. 512/1024/2048).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_NC = EnvInt(0)
    # Muse prefill Pallas All-Gather (0 = off; 1 = 2-stage XOR=1 bridge on A1+exit; 2 = +MoE;
    # 3 = Dual-Port 1-Hop XY/XYZ-Torus + XOR=1 bridge across A1+exit+MoE; 4 = MoE/B2-only XY/XYZ-Torus
    # keeping A1, chunk, exit, and split Kernel D on overlapped SparseCore lax.all_gather).
    SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG = EnvInt(0)
    # Per-call-site opt-in overrides (all default off so XOR1_BRIDGE_AG=0 keeps SparseCore):
    # MoE-internal B2 ag_i8 all-gather in pipeline.py (0 = off; 1/2 = flat; 3 = xy/xyz torus).
    SGLANG_TPU_MUSE_KERNEL_XOR1_AG_MOE = EnvInt(0)
    # Chunk/A1 entry all-gather in integration.py (0 = off; 1/2 = flat; 3 = xy/xyz torus).
    SGLANG_TPU_MUSE_KERNEL_XOR1_AG_CHUNK = EnvInt(0)
    # Opt-in: also run xor1_bridge_all_gather inside split _d_with_next_in_gather (False keeps
    # SparseCore jax.lax.all_gather so it overlaps under Kernel D Part 2 on TensorCore).
    SGLANG_TPU_MUSE_KERNEL_XOR1_AG_UNDER_D = EnvBool(False)
    # Standalone pre-attn/stream gather_sm & gather_fp8 override (0 = off; 1/2 = flat; 3 = torus).
    SGLANG_TPU_MUSE_KERNEL_XOR1_AG_B2 = EnvInt(0)
    # Muse prefill Kernel B2: collapse grid=(num_m, 4) -> (num_m,) with 2-slot double-buffered
    # wpool_fp8 async DMA prefetch, eliminating 3/4 redundant garbage HBM output writes (-804 MB/fwd).
    SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE = EnvBool(False)
    # Muse prefill Wave-6 single-launch fused B1+B2 (post-attn norm + gated residual +
    # pre-ffn norm + router top-8 + VMEM-cached 32 MiB wpool_fp8 + pre_expert_norm +
    # packed-bf16-scale int8 quant): 0 = off; 1 = 1-pass bf16 wpool; 2 = 1-pass dynamic
    # per-row fp8xfp8 2x MXU wpool; 3 = 2-half bf16 wpool; 4 = 2-half fp8xfp8 wpool.
    SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2 = EnvInt(0)
    # fused_rs_muse FP4 calls with >= 16384 routed rows (prefill): run each tile's matmuls
    # at a smaller row bucket covering its valid rows (0 = off; 1 = full 8..tile_m buckets;
    # 2 = coarse 32/64/tile_m buckets with bucketed LHS copy & GMM2 output reshape).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS = EnvInt(0)
    # Muse prefill Wave-7 Pallas VMEM+MXU sort-free routing (0 = off; 1 = Pallas local
    # inv_perm_c + local_chip_counts; 2 = 1 + sort-free global block-scatter routing).
    SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING = EnvInt(0)
    # Muse prefill Wave-7: when PREFILL_EPLB_MODE is set with CALIB_FORWARDS=0 and empty
    # PREFILL_EPLB_STATS_PATH, fall back to DECODE_REPLICA_STATS_PATH for top-R rebalance.
    SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_AUTO_STATS = EnvBool(False)
    # Muse prefill Wave-7B gmm_v2_fused_rs fast LHS FP8 quantization
    # (0 = off; 1 = hot-VREG m-chunked 3D VMEM GMM1 quant + on-the-fly Strip-0
    # GMM2 quant). Bit-exact when enabled with FP4_LHS_QBS=4096.
    SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT = EnvInt(0)
    # Muse prefill Wave-8 gmm_v2_fused_rs SMEM u16x2 LHS index packing + local window
    # (0 = auto at size_m >= 262144 for T=32k/48k/64k; 1 = force on all prefill block_scatter).
    SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS = EnvInt(0)
    # Optional explicit 128-aligned local SMEM window row cap for gmm_v2_fused_rs (0 = auto).
    SGLANG_TPU_MUSE_KERNEL_GMM_SMEM_WINDOW_ROWS = EnvInt(0)
    # Muse prefill Wave-8 gmm_v2_fused_rs 2-tile-ahead gather + early reshape/RS before DMA waits
    # (0 = off; 1 = early reshape/RS + 2-ahead gather; 2 = early reshape/RS only; 3 = 2-ahead gather only).
    SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD = EnvInt(0)
    # Muse prefill Wave-8 gmm_v2_fused_rs 3-slot expert-lookahead weight ring (+12 MiB VMEM).
    SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD = EnvInt(0)
    # Muse prefill Wave-8 gmm_v2_fused_rs dedicated VMEM FP8 expansion cache for W2 (+16 MiB VMEM).
    SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE = EnvInt(0)
    # Muse prefill Wave-8 gmm_v2_fused_rs sender-rotated ICI direct-write schedule
    # (0 = off; 1 = XOR send order; 2 = XOR send order + within-expert rotated chip order).
    SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS = EnvInt(0)
    # Muse prefill Wave-9 gmm_v2_fused_rs intra-tile EP-chip interleaving for rotated RS
    # (0 = off; >0 = min full (tile_m // ep_size)-row blocks per chip required to interleave
    # full 256-row tiles across EP chips in rot_order, avoiding single-link ICI scatter stalls).
    SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS = EnvInt(0)
    # Muse prefill Wave-8 gmm_v2_fused_rs direct SiLU epilogue without VMEM/bf16 round-trip
    # (0 = off; 1 = single-pass FP32 gate*up with fused GMM2 amax reduction; 2 = fast BF16 sigmoid).
    SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU = EnvInt(0)
    # Muse prefill Wave-8 quantize MoE hidden states to FP8 once before/at AG instead of 8x in GMM1
    # (0 = off; 1 = FP8 hidden states + per-token f32 scale passed into gmm_v2_fused_rs;
    # 4/5 = direct FP8 from fused b1_b2; 6 = static FP8 prefill LHS with zero row-scale AG).
    SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8 = EnvInt(0)
    # Explicit static FP8 scale S for b1_b2 -> gmm_v2_fused_rs prefill LHS (0.0 = off;
    # > 0 in (0, 32] e.g. 8.0 or 4.0 folds 1/S into w1_global_scale with zero row-scale AG/dequant;
    # GMM_PREQUANT_LHS_FP8=6 defaults this to 8.0 when unset).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS = EnvFloat(0.0)
    # Muse Wave-8 NVFP4 wider compact block scales (0 = off; 512/1024/2048/4096 = requantize at load;
    # -512/-1024/-2048/-4096 = zero-reload in-kernel scale-stride perf probe).
    SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK = EnvInt(0)
    # Muse Wave-8 two-level power-of-2 FP4 scales s[kb,n] = c[n] * 2^p[kb,n] folded into FP8 exponent
    # (0 = off; 1 = MSE-calibrated two-level requant at load + K-chunked MXU dot; 2 = on-the-fly snap probe).
    SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2 = EnvInt(0)
    # K-chunk size for FP4_TWO_LEVEL_POW2 MXU dots (1024 = 4 pops/strip in VREGs; 4096 = 1 pop via VMEM).
    SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_K_STEP = EnvInt(1024)
    # Muse NVFP4: keep the BF16 experts of the first and last layers BF16, on fused-RS
    # or gmm_v2. By default they are quantized to FP8 on host at load, as are all
    # experts of a BF16 checkpoint. The Muse megakernel reads only fp4 or fp8 experts,
    # so it cannot be combined with this.
    SGLANG_TPU_ENABLE_MUSE_BF16_FIRST_LAST_LAYER_EXPERTS = EnvBool(False)
    # Fused KV pool stores a rank's heads side by side in one 128-lane row when they
    # fit exactly (e.g. 2 x head_dim 64): half the bytes of a zero-padded row per head.
    SGLANG_TPU_KV_PACK_HEADS = EnvBool(False)
    # When SGLANG_TPU_KV_PACK_HEADS=1 packs a rank's FP8 KV heads into one 128-lane row
    # (e.g. Muse 2x64), pack 2 consecutive tokens into each 32-bit (1, 4, 128) word
    # ([K_even, V_even, K_odd, V_odd]), halving KV pool HBM (512 -> 256 B/tok/layer/TC)
    # and halving KV page DMA bytes in RPAd_fast, RPAm, and write_kv_only_rpa_v3.
    SGLANG_TPU_KV_PACK_DENSE2 = EnvBool(False)
    # On TP=16 (1 KV head x head_dim 64 FP8, zero-padded to 128 lanes), pack 4 consecutive
    # tokens into each 32-bit (1, 4, 128) word ([K0|K1, V0|V1, K2|K3, V2|V3] across lanes
    # [0..63 | 64..127]), cutting KV pool HBM & page DMA by 4x (512 -> 128 B/tok/layer).
    SGLANG_TPU_KV_PACK_BD4 = EnvBool(False)
    # Enable vectorized batch4/batch2 swabatch group execution on dense2-packed KV pools
    # (also selectable via "d2grp" token in SGLANG_TPU_RPA_V3_DECODE_FAST).
    SGLANG_TPU_RPA_V3_DENSE2_GROUP = EnvBool(False)
    # Enable hoisted whole-batch new-KV precompute on shared16/32/64 LC Full decode
    # (also selectable via "hoistkv" token in SGLANG_TPU_RPA_V3_DECODE_FAST).
    SGLANG_TPU_RPA_V3_SHARED_HOISTKV = EnvBool(False)
    # Allow fused Step (c) (b1_moe_b2_rpad) on SWA layers during shared-prefix LC steps
    # (decode_shared_prefix_pages is Full-only; SWA uses standard per-seq distribution).
    SGLANG_TPU_MUSE_MK_STEP_C_LC_SWA = EnvBool(False)
    # Route uniform-width speculative TARGET_VERIFY (decode_query_size in 2..8, e.g.
    # DFlash dq=5/6) through RPAd_fast batched [16*dq, 128] single-MXU-pass verify.
    SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY = EnvBool(False)
    # Opt-in RPAd_fast mode override/suffix applied when decode_query_size > 1
    # (e.g. "auto" -> c1024,noz,pbf16,g4k,foldcol for dq<=4 and c512,... for dq>=5).
    SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY_OPTS = EnvStr("")
    # Wave-8 RPAd_fast inner-loop optimizations: swafast2 (zero upfront VMEM zeroing via
    # lazy-HWM sanitizer + per-seq 512/1024/2176 gears + vertical page_reduce), w7auto
    # c512 gear + pv_bf16 pass-through fix, and hoisted whole-batch new_kv precompute.
    SGLANG_TPU_RPA_V3_DECODE_FAST_W8 = EnvBool(False)
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
    # Expert-TP fused-RS as a two-level reduce-scatter: the two cores of a chip add their
    # partials on-chip, then send one chip sum per row to another chip's owner, halving
    # off-chip bytes (one extra bf16 rounding).
    SGLANG_TPU_ENABLE_FUSED_RS_ETP_TWOLEVEL = EnvBool(False)

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
    SGLANG_RAIDEN_HOST_POOL_GB = EnvFloat(64.0)
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
    # Raiden: seconds between the decode listener's sweeps, which settle reads
    # and start queued chunks while the scheduler is not draining. Must be
    # > 0 and below SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S.
    SGLANG_RAIDEN_LISTENER_SWEEP_INTERVAL_S = EnvFloat(1.0)
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

    # Pipeline parallelism: hand activations between stages point to point
    # (isend/irecv, pp_buffered.py) instead of the lockstep collective-permute wave.
    SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF = EnvBool(False)
    # Buffered hand-off: forwards a stage may queue for sending before it blocks.
    SGLANG_TPU_PP_BUFFERED_SEND_DEPTH = EnvInt(2)

    # raiden KV-offload backend
    # Seconds a raiden DMA may stay pending, from each blocking wait or, when
    # polled, from issue. Past it the scheduler exits with TimeoutError, so keep
    # it far above a healthy transfer.
    SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S = EnvFloat(60.0)

    # HiCache write_back on raiden (see hicache_write_back_copy_ahead.py): tokens
    # of the next eviction victims kept copied to host. Unset: four chunked-prefill
    # chunks. 0: evictions back up their victims synchronously, as upstream does.
    SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS = EnvInt(None)
    # Separate copy-ahead token budget for the SWA eviction order under write_back
    # (independent of the Full leaf window when set). Unset: 131072 on the unified
    # KV pool, else 1/4 of SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS.
    SGLANG_TPU_HICACHE_WRITE_BACK_SWA_COPY_AHEAD_TOKENS = EnvInt(None)

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
    # Store NVFP4 MoE group scales as f32 instead of bf16 on the non-fused-RS
    # gmm_v2 path (e.g. PP with TP=1) so gmm_v2 uses Mosaic 32-bit zero-stride
    # scale loads instead of VPU broadcast expansion.
    SGLANG_TPU_NVFP4_MOE_FP32_SCALES = EnvBool(False)
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
    # MoE under EP, off the dense permute, with the SparseCore gather and combine:
    # route on local expert ids (counting sort) and gather only this rank's rows.
    SGLANG_TPU_FUSED_MOE_EP_COUNTING_ROUTE = EnvBool(True)
    # MoE: use onehot dense matmul for small permute and unpermute/combine
    # paths instead of gather. When set to 0, this is disabled. When set
    # to a positive integer, the permute is dense below this many tokens per
    # rank and the combine below this many routed rows (tokens x topk).
    # Suggest to start with 1024 or 2048 and tune from there. Below it,
    # batches of up to 2048 routed rows also take the sort-free dense rank
    # route, and one-token batches the route-meta kernel.
    SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD = EnvInt(0)
    # MoE: where the permute is dense and the combine is the SparseCore gather
    # (threshold / topk <= tokens < threshold), permute with the gather too.
    # Set at startup by SGLANG_TPU_K3_STATIC_FP8 moe_front or latent_up: there a
    # one-hot permute feeding the SparseCore combine gave wrong tokens.
    SGLANG_TPU_FORCE_MOE_GATHER_PERMUTE_WITH_SC_COMBINE = EnvBool(False)
    # Turn off EP-local expert reads (models that opt in read only their EP
    # rank's routed experts with --weight-loader-disable-mmap or Run:AI).
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
    # RPA v3 MIXED launch (serving prefill) block sizes "bq,bkv,bq_c,bkv_c" in tokens,
    # for full-attention and sliding-window layers. Empty keeps the heuristic.
    SGLANG_TPU_RPA_V3_MIXED_BLOCK_SIZES = EnvStr("")
    SGLANG_TPU_RPA_V3_MIXED_BLOCK_SIZES_SW = EnvStr("")
    # RPA v3 DECODE launch (one query per sequence) block sizes, same format. Set,
    # it replaces the heuristic and the Muse window-2048 decode tile.
    SGLANG_TPU_RPA_V3_DECODE_BLOCK_SIZES = EnvStr("")
    SGLANG_TPU_RPA_V3_DECODE_BLOCK_SIZES_SW = EnvStr("")
    # RPA v3 DECODE on sliding-window layers: start each sequence's KV block at
    # the window's first page instead of a block-aligned token, one block of
    # cdiv(window + page - 1, page) pages (unless the _SW block env is set).
    SGLANG_TPU_RPA_V3_DECODE_SW_WINDOW_ALIGNED = EnvBool(False)
    # RPA v3 DECODE: zero the KV buffer rows past each sequence's length once per
    # block instead of masking V per compute chunk. Same output.
    SGLANG_TPU_RPA_V3_DECODE_ZERO_KV_TAIL = EnvBool(False)
    # RPA v3 DECODE: buffer slots in the KV/q fetch ring (N-1 sequences in flight)
    # when each sequence is one KV block (full layers at the default block, SW
    # with WINDOW_ALIGNED). 0 keeps the two-slot pipeline. Full slots are 4.3 MB.
    SGLANG_TPU_RPA_V3_DECODE_RING = EnvInt(0)
    # RPA v3 DECODE dedicated Ghostfish fast path (whole-batch VMEM Q/KV/O, static
    # whole-page DMA ring, epilogue-waited async KV cache writes, single-load VREG
    # KV chunks with uint32 pre-bitcast boundary masking). "" keeps stock kernel;
    # "1"/"auto" selects tuned default; token strings override slots/chunk/loop/pack.
    # "tp16" opts into Muse's native 8x64 SWA decode with SKIP_EMPTY_MIXED.
    # "shared16_host" hoists the full-layer sharing proof to decode metadata.
    # Supports packed TP8 / native TP16 and uses stock full RPAd on private rows.
    SGLANG_TPU_RPA_V3_DECODE_FAST = EnvStr("")
    # Skip the 0-sequence MIXED sub-launch of RPA v3 on pure decode forwards.
    SGLANG_TPU_RPA_V3_SKIP_EMPTY_MIXED = EnvBool(False)
    # Packed-head RPA v3 decode returns each head's own lanes directly ([B, Hq * d]),
    # dropping the post-kernel one-hot unpack on decode without changing prefill.
    SGLANG_TPU_RPA_V3_DECODE_FUSED_OUT = EnvBool(False)
    # Fused Pallas decode qkv epilogue (q/k RMSNorm + interleaved RoPE + fp8 cast
    # + KV_PACK layout).
    SGLANG_TPU_MUSE_DECODE_QKV_EPILOGUE = EnvBool(False)
    # Fused post-RPA decode attention output RMSNorm + sigmoid(gate) inside RPAd_fast
    # (when SGLANG_TPU_MUSE_DECODE_QKV_EPILOGUE=1 and SGLANG_TPU_RPA_V3_DECODE_FAST != "").
    SGLANG_TPU_MUSE_DECODE_ATTN_GATE_FUSE = EnvBool(False)
    # RPA v3 DECODE D3v2: enables pps<=136 context-slack support, dynamic max_n_pg slot
    # zeroing, zero-spill dynamic-window Full-Attn + 2-half SWA compute, coalesced page
    # DMAs, and in-VMEM fused_out_pack=2.
    SGLANG_TPU_RPA_V3_DECODE_FAST_V2 = EnvBool(False)
    # Wave-7 RPAd_fast: unlocks pps<=256 Full-Attn in-VMEM onorm+gate (streamdyn/ring4k)
    # and lazy 1-shot slot_uninit VMEM boundary sanitization (0 upfront zero pages on full SWA).
    SGLANG_TPU_RPA_V3_DECODE_FAST_W7 = EnvBool(False)
    # Adaptive RPAd_fast dispatch via lax.switch on max(kv_lens) between short <=1024 (c1024, mp=8)
    # and long windows (empty/False = auto under w7/adapt; True/False overrides).
    SGLANG_TPU_RPA_V3_DECODE_ADAPTIVE = EnvBool(False)
    # Unlock shared-prefix decode batches down to bs=8 (group_size=8 -> 128-row
    # MXU tile) for Long-Context c=8 workloads.
    SGLANG_TPU_RPA_SHARED_BS8 = EnvBool(False)
    # Optional RPAd_fast / RPAm diagnostic probe mode ("", "dense_kv", "nodma", "nocompute", "nokvwrite").
    SGLANG_TPU_RPA_V3_DECODE_PROBE = EnvStr("")
    # Wave-7 RPAm Stationary-Q prefill options (comma/underscore tokens: fp8_kv, fp8_qk,
    # fp8_qkpv, gqa2_vreg, gqa4_vreg, full_bq512, full_bkv4k). Empty = off.
    SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS = EnvStr("")
    # Long-Context RPAm Stationary-Q prefill options (comma/space tokens: bq1024, bq2048,
    # bq4096, bkv8192, prefix_fast, s2, plus fp8_kv/fp8_qk/fp8_qk_dyn/gqa2_vreg/gqa4_vreg/
    # f32_exp/mrb3/mrb4/tq_unroll for Full-Attn >32k context). Empty = off.
    SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS = EnvStr("")
    # Muse prefill kernel A3 (v16): sub-tile static FP8 gate prologue to [128, 128] in VREGs
    # (eliminates 4,764 spill stores at tm=1024 and drops the 2 MiB VMEM norm scratch).
    SGLANG_TPU_MUSE_KERNEL_A3_ZERO_SPILL_PROLOGUE = EnvBool(False)
    # RPA v3 MIXED/PREFILL launches (serving prefill), same output within kernel
    # tolerance: softmax scale folded into exp2; no mask on fully visible sub-blocks.
    SGLANG_TPU_RPA_V3_PREFILL_FOLD_SCALE = EnvBool(False)
    SGLANG_TPU_RPA_V3_PREFILL_MASK_SKIP = EnvBool(False)
    # Native Muse TP16 8k prefill tiles for the m0_ring_fuse recipe.
    SGLANG_TPU_RPA_V3_TP16_NATIVE_PREFILL = EnvBool(False)
    SGLANG_TPU_RPA_V3_TP16_LONG_PREFILL_TILES = EnvBool(False)
    # Wave-9 zero-spill M_k=128 Stationary-Q v2 prefill preset (TP8 + TP16).
    SGLANG_TPU_RPA_V3_PREFILL_W9 = EnvBool(False)
    # Native Muse TP16 (8 Q / 1 KV head, d=64) Stationary-Q v2 prefill engine
    # (default-ON: cures #2390 stock bf16 softmax NaN overflow; set =0 to kill-switch).
    SGLANG_TPU_RPA_V3_TP16_STATIONARY_Q = EnvBool(True)
    # Stationary-Q Dual-MXU prefill engine for packed-KV RPA v3 (16 Q heads on 1
    # 128-wide fp8 KV head): "" = off (stock kernel), "m0" = zero-shift exp2(qk)
    # with full-block unrolling on full-attention layers, "pilot" = scale-invariant
    # f32-shift pilot-max, "m0_ring" = sequence-persistent unpacked-KV VMEM ring +
    # 9-slice unrolled SW / 256-slice binary-unrolled Full + precomputed bf16 mask
    # tiles, "m0_f32e" = "m0_ring" + f32 exp2(qk), "m0_ring_fuse" / "m0_f32e_fuse" =
    # inline Q-unpack + epilogue overlapped under SW MXU.
    SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE = EnvStr("")
    # Optional "bq,bkv,bq_c,bkv_c" block-size overrides for Stationary-Q prefill
    # modes (SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE != "").
    SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES = EnvStr("")
    SGLANG_TPU_RPA_V3_STATIONARY_Q_BLOCK_SIZES_SW = EnvStr("")
    # Skip the 0-sequence DECODE sub-launch of RPA v3 on pure extend forwards.
    SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE = EnvBool(False)
    # Packed-head RPA v3 (SGLANG_TPU_KV_PACK_HEADS) returns each head's own lanes,
    # dropping the post-kernel relayout and one-hot unpack. Same output.
    SGLANG_TPU_RPA_V3_PREFILL_FUSED_OUT = EnvBool(False)
    # Stationary-Q RPA v3 prefill (SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE != "")
    # writes unpacked bf16[T, 1024] directly from its epilogue (16 MiB vs 32 MiB), dropping
    # prepare_outputs copy, KV_PACK pair-sum, and A3 attn_ref relayout copy. Exact.
    SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT = EnvBool(False)
    # Unlock mk128 on TP=16 SWA Stationary-Q RPAm instead of vreg_acc (default False).
    SGLANG_TPU_RPA_V3_TP16_SWA_MK128 = EnvBool(False)
    # Opt-in 2-kernel LC Full prefill top-k slice pruning (default 0 = disabled).
    # When > 0 (multiple of 4, < half-slices per bkv block), keeps the top-k
    # 128-key half-slices per prefix bkv block while keeping sink (bkv=0) and
    # causal tail dense.
    SGLANG_TPU_RPA_V3_PREFILL_TOPK_SLICES = EnvInt(0)
    # Query pooling stride for Stage-1 All-QK scoring (2 = pair-averaged Q, 1 = full Q).
    SGLANG_TPU_RPA_V3_PREFILL_TOPK_QSTRIDE = EnvInt(2)
    # Muse prefill kernel path with SGLANG_TPU_KV_PACK_HEADS: A1 (two-half bf16 variants)
    # writes q and kv in RPA v3's packed layout, dropping the XLA expand/repack glue.
    SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT = EnvBool(False)
    # Muse prefill kernel A3 (v16) with SGLANG_TPU_KV_PACK_HEADS: A3's prologue reads
    # RPA v3's packed output layout directly via diagonal head-lane selection, dropping
    # the post-RPA relayout and KV_PACK unpack glue. Same values.
    SGLANG_TPU_MUSE_KERNEL_A3_RPA_OUT = EnvBool(False)
    # Muse prefill kernel entry: column-sharded [V, H / tp] embedding lookup + all-to-all
    # + sharded RMSNorm (+0.38 GiB/rank), replacing the full-T VocabParallel all-reduce.
    SGLANG_TPU_MUSE_KERNEL_SP_ENTRY = EnvBool(False)
    # Muse prefill kernel RoPE: pre-build one 128-lane [max_pos, 128] cos/sin table and
    # gather T / tp rows per rank + SparseCore all-gather. Same values.
    SGLANG_TPU_MUSE_KERNEL_SHARDED_ROPE = EnvBool(False)
    # Muse prefill kernel A3 (v16): 2048-col Wo tiles (at T <= 8192), dynamic recv-sem
    # waits, pre-barrier local DMA, and 1024-row prologue tiles. Same values.
    SGLANG_TPU_MUSE_KERNEL_A3_OPT = EnvBool(False)
    # Muse prefill kernel A1 (multi-part bf16 variants): comma-separated per-rank row
    # chunk sizes summing to T/tp (e.g. "384,320,256,64" or "local+384,320,256,64") to
    # shrink the exposed SparseCore all-gather tail. Empty keeps equal chunks.
    SGLANG_TPU_MUSE_KERNEL_A1_CHUNKS = EnvStr("")
    # Muse prefill kernel A1 (multi-part bf16 variants): write outputs via 1-way async
    # HBM copies without empty-buffer input copies or HBM->VMEM garbage reads.
    SGLANG_TPU_MUSE_KERNEL_A1_1WAY = EnvBool(False)
    # Muse prefill A1 v5 / v5x4 / v5x8 (two-half body): uneven parts as comma-separated row
    # weights, e.g. "2,3,2,1" = rows/rank 256,384,256,128 at T=8192. "" or a shard that does
    # not split into 128-row multiples keeps the variant's equal parts. Same output bits.
    SGLANG_TPU_MUSE_KERNEL_A1_PARTS = EnvStr("")
    # Muse prefill A1 v5 / v5x4 / v5x8 (two-half body): kernel D of the layer before runs in
    # two pieces and the next layer's first A1 all-gather starts under the second piece (the
    # first kernel layer's A1 and the last layer's D are unchanged). Same output bits.
    SGLANG_TPU_MUSE_KERNEL_A1_AG_UNDER_D = EnvBool(False)
    # Muse prefill Causal Chunked Streaming (Option A): shard tokens in causal chunk-major
    # order across TP ranks so Kernel D[L] overlaps Chunk 0 AG, and A1[L+1] Chunk k +
    # Causal RPAm[L+1] Chunk k overlap Chunk k+1's SparseCore all-gather.
    SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM = EnvBool(False)
    # Optional comma-separated per-rank row sizes for causal chunk streaming (e.g.
    # "512,768,768" at T=16384 or "256,384,384" at T=8192). Empty auto-selects optimal 3-chunk.
    SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_SPEC = EnvStr("")
    # Wave-8 prefill A3 (v16 opt_wo): stage remote push-A2A sends and stagger 2x2 torus link order
    # (0 = off; 1 = defer t_half=1 sends until t_half=0 lands + parity-staggered 1-factorization d_order;
    # 2 = 1-in-flight step-staged sends + parity-staggered d_order; 3 = defer t_half=1 sends only).
    SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS = EnvInt(0)
    # Wave-8 prefill CCS + A3 (v16): run pallas_gate_static_fp8_quant per causal chunk right after
    # RPA(k) (overlapped under ag_{k+1}) and concatenate compact fp8 q [tp, c_k, 1024] (16 MiB vs 64-96 MiB).
    SGLANG_TPU_MUSE_KERNEL_CCS_CHUNK_A3_PROLOGUE = EnvBool(False)
    # Wave-8 prefill Kernel D schedule & epilogue fusion (bitmask: 0 = off; 1 = slab-first wunpool
    # DMA start + per-slab wait + 32-DMA-per-sub-dot finer _so + 256-col zero-spill Phase 3/4;
    # 2 = quadratic inv_rms_next in Phase 2 epilogue fusing Phases 3+4 into one sweep; 3 = 1 + 2).
    SGLANG_TPU_MUSE_KERNEL_D_OPT = EnvInt(0)
    # Wave-8 prefill Kernel D Phase-1 MXU combine: pack top-8 weights into 16x128 block-diagonal
    # w_pad tiles and replace VPU top-8 weighted sum + amax pass with MXU dots.
    SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE = EnvBool(False)
    # Wave-8 prefill Kernel B1 & B2 static input prepack (b1_vecs, b1_bias_t, b2_vecs at load).
    SGLANG_TPU_MUSE_KERNEL_B1_B2_PREPACK = EnvBool(False)
    # Wave-8 prefill Kernel B1 & B2 256-lane zero-spill sub-strips + prepacked statics + dead w_lo skip.
    SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST = EnvBool(False)
    # Wave-8 prefill MoE routing: force 0-sort MXU bucket-rank + scatter block-scatter routing.
    SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING = EnvBool(False)
    # Wave-8 prefill stream All-Gather: quantize full-stream gathers to per-row FP8 + packed bf16
    # scale before all-gather and fuse chunk-major unpermute on 1-byte FP8 before dequantization.
    SGLANG_TPU_MUSE_KERNEL_FP8_STREAM_GATHER = EnvBool(False)
    # Wave-9 prefill CCS fused AG+A1: fuse next-chunk all_gather(rest_in[k+1]) inside a1_chunk's
    # shard_map MLIR module so SparseCore/ICI AG overlaps directly under pallas_a1_chunk_1way
    # (0 = off, 1 = lax.all_gather / XOR1_BRIDGE_AG, 2 = force xor1_bridge_all_gather on CCS AG,
    #  3 = in-kernel overlapped 1-hop XOR=1 torus AG inside pallas_a1_chunk_1way).
    SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1 = EnvInt(0)
    # Wave-9 prefill FUSED_B1_B2=3/4 overlap: run dequant_half_gmm_layout(ag_i8_0) before waiting
    # on global_topk_u8's optimization_barrier so dequant_half0 overlaps global_topk_u8 + ag_i8_1.
    SGLANG_TPU_MUSE_KERNEL_OVERLAP_DEQ0_ROUTING_AG = EnvBool(False)
    # Muse prefill kernel A1: zero-copy 2D tiled all-gather, unsliced 2D RoPE, direct 4D
    # RPA KV layout, and fused kv_local strided-128 s8 pre-quantization.
    SGLANG_TPU_MUSE_KERNEL_A1_ZC = EnvBool(False)
    # Muse prefill Kernel D: co-schedule next-tile scalar gather DMA setup inside the
    # unconditional Phase-2 MXU basic block instead of an isolated scf.if SALU-only block.
    SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP = EnvBool(False)
    # Muse prefill kernels A1/A3: skip padded tail tiles (row >= sum(extend_seq_lens))
    # via traced int32[1] scalar prefetch with zero XLA recompiles.
    SGLANG_TPU_MUSE_KERNEL_SKIP_PAD_ATTN_TOKENS = EnvBool(False)
    # Muse prefill MoE routing: stable argsorts of expert ids and the inverse
    # permutation as one packed s32 sort each (key * n + index). Same result.
    SGLANG_TPU_MUSE_KERNEL_PACKED_SORTS = EnvBool(False)
    # Muse prefill kernel B1: the router weight's bf16 hi/lo split is made once at load
    # instead of per layer per forward; lo is 0, as XLA computes it on TPU (same bits).
    SGLANG_TPU_MUSE_KERNEL_ROUTER_HILO_PRE = EnvBool(False)
    # Muse prefill MoE with SGLANG_TPU_MUSE_KERNEL_FUSED_GLUE: both dequant halves write GMM's
    # [tok, 32, 128] layout directly, dropping XLA's relayout of the AG buffer. Same values.
    SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT = EnvBool(False)
    # Muse prefill kernels: also run Layers 0 & 61 (FP8 experts) on the SP kernel path
    # (A1 + RPA + A3 + B1 + B2 + fused_rs_muse FP8 GMM-RS + Kernel D), keeping all 62 layers
    # on the token-sharded SP spine.
    SGLANG_TPU_MUSE_KERNEL_FP8_LAYERS = EnvBool(False)
    # fused_rs_muse prefill FP8 tiles: "tile_m,tile_k1,tile_n1,tile_k2,tile_n2,w1_bufs,w2_bufs"
    # replaces the tuned tiles of FP8-expert calls with size_m >= 16384 (prefill).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_FP8_GMM_TILES = EnvStr("")
    # fused_rs_muse FP8 expert LHS quantization block size (0 = default 512, 4096 = full-row
    # single-dot W8A8 on 2x FP8 MXU, -1 = W8A16 bf16 @ fp8 single-dot).
    SGLANG_TPU_MUSE_KERNEL_FP8_LHS_QBS = EnvInt(0)
    # Muse prefill Kernel D token tile (64 or 128).
    SGLANG_TPU_MUSE_KERNEL_D_TOK_TILE = EnvInt(64)
    # Muse prefill Kernel D zero-spill 256-col sub-tiled schedule with HBM-streamed residual
    # and pre-packed static norm/gate vectors.
    SGLANG_TPU_MUSE_KERNEL_D_ZERO_SPILL = EnvBool(False)
    # Muse prefill Kernel D: pre-pack static norm/residual/scale vectors [8, 8192] once at load.
    SGLANG_TPU_MUSE_KERNEL_D_PREPACK_VECS = EnvBool(False)
    # Muse prefill MoE routing: sort-free 256-expert / 32-local-expert counting sort
    # replacing argsort on local_topk_ids and global_topk_ids.
    SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL = EnvBool(False)
    # Sort-free local inv_perm mode (0/1 = bitwise packed sort + 2-stage 16K pre-compacted merge,
    # 2 = lane-local vst.iar + MXU U_256, 3 = VALU 8-sum + MXU L_256).
    SGLANG_TPU_MUSE_KERNEL_SORTFREE_MODE = EnvInt(0)
    # Sort-free block-scatter lhs_indices (needs SORTFREE_PREFILL): 0 = per-rank sort + merge,
    # 1 = one flat sort, 2 = MXU bucket ranks + one scatter (no sort).
    SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER = EnvInt(0)
    # Muse prefill MoE B2 dequant: single-pass 2-half GMM-layout dequant without VREG spills.
    SGLANG_TPU_MUSE_KERNEL_DEQUANT_1PASS = EnvBool(False)
    # Muse prefill MoE B2: pack exact 16-bit bf16 row scales into columns [4096:4224] of the
    # SparseCore async int8 all-gather, eliminating both sync scale psums and mask_r reductions.
    SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE = EnvBool(False)
    # Muse prefill Exit: slice/owner-gather last_indices rows before final RMSNorm and use
    # Pallas FP8 LM head softcap.
    SGLANG_TPU_MUSE_KERNEL_EXIT_OWNER_GATHER = EnvBool(False)
    # Exact last-layer (L61) pruning on EXTEND when capture_hidden_mode == NULL and
    # not return_logprob: write L61 KV for all T rows, then slice Q and residual to
    # last_token_indices (padded to 128) before RPAm_61 -> A3_61 -> MoE_61 -> FinalNorm.
    SGLANG_TPU_MUSE_L61_PRUNE = EnvBool(False)
    # Gather the target's requested intermediate residuals for a DFlash draft.
    SGLANG_TPU_MUSE_KERNEL_CAPTURE_DFLASH = EnvBool(False)
    # Raise _OneshotLogitsGatherer shard cap from 8 MiB to 26 MiB so B=128 (12.33 MiB) and
    # B=256 (24.66 MiB) use 1-hop Pallas oneshot_all_gather instead of 8x zero-padded All-Reduce.
    SGLANG_TPU_MUSE_KERNEL_LOGITS_ONESHOT_26M = EnvBool(False)
    # Enable exact Pallas distributed TopK on pure greedy (is_all_greedy) batches (returns
    # lowest global token ID on ties and masks padded vocab columns to -inf).
    SGLANG_TPU_DISTRIBUTED_GREEDY_TOPK = EnvBool(False)
    # Near-lossless 2x fp8@fp8 MXU tier for Kernel B1->B2 (algebraic-cancel FP8 router_in).
    SGLANG_TPU_MUSE_KERNEL_B2_FP8_LHS = EnvBool(False)
    # Near-lossless 2x fp8@fp8 MXU tier for Kernel D wunpool (0 = off bf16@fp8,
    # 512/1024/4096 = K quantization block size for fp8@fp8 unpool).
    SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL = EnvInt(0)
    # Align Kernel D Phase 1 top-k combine to bf16 VMEM T(16, 128) tiles (c_step=16 in 32-row halves),
    # cutting VLD -24.7%, VST -45.9%, and spills -59.5% (-11.7 us/layer = -0.72 ms/fwd, 100% bit-exact).
    SGLANG_TPU_MUSE_KERNEL_D_T64_C16 = EnvBool(False)
    # Wave-6 prefill RPA: unlock stationary_q sw9_fused_qe fast path when production
    # Muse Spark passes sliding_window=2048 (config 2048 - 1 + 1) as well as 2049.
    SGLANG_TPU_RPA_V3_PREFILL_SW2048_FUSE = EnvBool(False)
    # Wave-6 prefill A1: override v6_s8str128_bf16kv default chunking ("512" ->
    # "local+512,512", "1024" -> "local+1024", "all512" -> "512,512") to avoid
    # row_off=384 tm=128 degradation, and skip dead x_s8_all quant when 1 chunk.
    SGLANG_TPU_MUSE_KERNEL_A1_FAST_CHUNKS = EnvStr("")
    # Wave-6 prefill Kernel D: promote 4slab tok_tile 64 -> 128 with c16h 16-col
    # combine + salu_overlap (100% MXU M=128 utilization, 8 wunpool passes vs 16).
    SGLANG_TPU_MUSE_KERNEL_D_T128_FAST = EnvBool(False)
    # Hack to workaround libtpu bug in 0.0.46 and 0.0.47 causing performance
    # regression in fused MoE with reduce scatter.
    # TODO: Remove once the issue gets resolved in libtpu version > 0.0.47
    # (likely 0.0.48).
    SGLANG_TPU_LIBTPU_WORKAROUND_FOR_ONLINE_QUANT_SCALE = EnvBool(True)
    # Share one page pool between full-attention and sliding-window KV on TPU
    # (equivalent to --enable-unified-memory; see unified_kv_pool.py).
    SGLANG_TPU_UNIFIED_KV_POOL = EnvBool(False)
    # Cap SWA resident host usage at this share of the TPU unified HiCache host
    # arena and reclaim Full KV before SWA below the cap (<= 0 disables the
    # policy and restores request-order reclaim; >= 1 disables the cap while
    # keeping Full-first reclaim).
    SGLANG_TPU_UNIFIED_HICACHE_SWA_SHARE = EnvFloat(0.3)

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
    # Use the decode's DRAM as the KV offload tier below the prefill's own RAM,
    # one switch on both roles: the prefill pushes cold cached branches into
    # the decode's host pool and recalls them when they return. Settings the
    # offload does not support are refused at startup
    # (disaggregation/decode_dram_offload/startup_validation.py).
    SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM = EnvBool(False)
    # Muse expert layout for the megakernel: tp | ep | auto
    # (auto: ep when the server's EP size > 1). muse_megakernel/common/moe_mode.py.
    SGLANG_TPU_MUSE_MOE_MODE = EnvStr("auto")
    # Largest megakernel step in rows (running requests x verify tokens, padded to the
    # TP size); larger steps run the native decoder. Steps above 32 rows run the
    # megakernel's row-tiled program (ops.BASE_STEP_MAX_ROWS) and pad to 16 rows.
    SGLANG_TPU_MUSE_MK_MAX_ROWS = EnvInt(32)
    # Muse megakernel: stack the NVFP4 expert slabs of this many consecutive layers into one
    # operand per slab (0 = one operand per layer); the layer params become views into it.
    # 15 measured 3-4% faster decode / verify steps (cc = 1, q = 1 and 8, 128k-512k).
    SGLANG_TPU_MUSE_MK_EXPERT_GROUP = EnvInt(0)
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
    # Megakernel attention on fp8 SEQ_ALONG_LANE pools: feed K (and V) to the MXU as fp8
    # instead of widening each chunk to bf16 on the VPU first. 0 = off, 1 = K only, 2 = K and V
    # (exact). With the 2026-10-27 megakernel, 2 saves 0.09 / 0.20 / 0.40 ms per q = 1 step at
    # 128k / 256k / 512k and nothing at q = 8.
    SGLANG_TPU_MK_SAL_FP8_DOT = EnvInt(0)
    # Profiling only: name the megakernel's in-kernel regions for xprof (needs libtpu
    # --xla_enable_custom_call_region_trace=true and a device tracer level above 1): 1 per
    # phase; 2 also per chunk / expert, which distorts the timing they show.
    SGLANG_TPU_MK_TRACE_REGIONS = EnvInt(0)
    # Megakernel attention: fold a power-of-two attn_scale into the bf16 queries instead of
    # scaling every score (exact).
    SGLANG_TPU_MK_FOLD_ATTN_SCALE = EnvBool(False)
    # Muse megakernel: fold the f / o all-gathers into the next qkvg and the pool + router
    # matvecs (not bit-exact: the matvec input rounding moves; verify steps only).
    SGLANG_TPU_MUSE_MK_FOLD = EnvBool(True)
    # Muse megakernel Wave 1-4 optimizations (hoisted SAL refill, stack2 QK/PV, all-VMEM
    # collective/expert scratch on row-tiled steps, pair-route collectives, vectorized routing).
    SGLANG_TPU_MUSE_MK_OPT = EnvBool(True)
    # Muse megakernels (decode, prefill): fp4 experts requantized at load to one bf16 scale per
    # this many K rows (0 off, 256), shared with the native fused-RS MoE (W4A16); the expert
    # dots push raw e2m1 codes to the MXU (MKConfig.qblock).
    SGLANG_TPU_MUSE_MK_QBLOCK = EnvInt(0)
    # Muse megakernel: extend and mixed steps run the prefill program (muse_megakernel/prefill/)
    # on the kernel's weights. Needs TP 8, fp8 SEQ_ALONG_LANE pools, pallas_tpu_rpa_batched
    # and tpu_compile_tokens in multiples of 128: the model load refuses anything else (the
    # pool dtype on the first extend step).
    SGLANG_TPU_MUSE_MK_PREFILL = EnvBool(False)
    # Prefill program: the expert-input and post-expert-projection all-gathers in e4m3, one
    # scale per block: half their ICI bytes, ~2^-4 relative error per element.
    SGLANG_TPU_MUSE_MK_PREFILL_FP8_GATHER = EnvBool(False)
    # Prefill program in the sequence-parallel layout (prefill/spep): each rank owns rows / tp of
    # the chunk between layers, instead of every rank holding every row (prefill/tp).
    SGLANG_TPU_MUSE_MK_PREFILL_SP = EnvBool(False)
    # SP program reads decode EP's weight layout, so prefill SP and decode EP share one copy of
    # the weights (w13 then runs unfused: silu * up after the gmm).
    SGLANG_TPU_MUSE_MK_SP_EP_LAYOUT = EnvBool(False)
    # Quality probe: requantize Muse fp4 experts at load to one bf16 scale per 256 K rows
    # (amax / 6, as MUSE_MK_QBLOCK) but store it repeated in the block-16 layout, so every
    # kernel serves the lossy weights unchanged. Not a speedup.
    SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES = EnvBool(False)
    # Kernel-level weight layouts (anvil / numerics only); serving raises on them.
    SGLANG_TPU_MUSE_MK_KPERM = EnvBool(False)
    SGLANG_TPU_NVFP4_E4M3_SCALES = EnvBool(False)
    # Wave-9 Per-Layer MoE-to-MoE (M2M) Decode Megakernel: fuses Bridge-2b(l-1) + QKV/RoPE
    # epilogue + RPAd(l) + post-RPA head-RMSNorm/gate + Bridge-1b(l) into one zero-spill
    # collective Pallas kernel across TP=8 (2x2x1) and TP=16 (2x2x2) with 0 extra weight HBM.
    SGLANG_TPU_MUSE_MK_M2M = EnvBool(False)
    SGLANG_TPU_MUSE_MK_M2M_MAX_FULL_CTX = EnvInt(4096)
    # Batched RPA prefill tiles (P/D prefill; the Muse Spark recipe sets all four).
    # Softmax exp2 with log2e folded into the q scale, one VALU op per score fewer.
    # Off with an attention sink or an LSE output, which would see base-2 m and l;
    # the folded scale rounds q in bf16, so the output is not bit-identical.
    SGLANG_TPU_RPA_EXP2_SOFTMAX = EnvBool(False)
    # Row sums from the PV matmul: V tiles get ones rows in the lanes head_dim 64
    # leaves idle on the MXU, so the accumulator carries l and the vector row-sum
    # reduce and the separate l rescale go away. SEQ_ALONG_LANE, head_dim <= 64.
    SGLANG_TPU_RPA_ONES_V = EnvBool(False)
    # Tiles whose keys all precede their queries skip the causal (and sliding-window)
    # mask, and a power-of-two sm_scale is folded into q. Exact.
    SGLANG_TPU_RPA_SKIP_FULL_TILE_MASK = EnvBool(False)
    # Prefill attention tiles with transposed scores (keys along sublanes) and
    # fp8 q; not bit-exact with the default tile. Mixed/prefill, SEQ_ALONG_LANE,
    # head_dim <= 64, no sink / LSE / soft cap / q scale / DCP attention scopes.
    SGLANG_TPU_RPA_TRANSPOSED = EnvBool(False)
    # Kernel-sweep knobs (anvil --env), read here so the compile cache keys them; unset = the
    # tuned default (dev_int). Muse megakernel tile / plan knobs (ops.py).
    SGLANG_TPU_MUSE_MK_DEV_ROW_TILE = EnvInt(None)
    SGLANG_TPU_MUSE_MK_DEV_TILED_PAIR_Y = EnvInt(None)
    SGLANG_TPU_MUSE_MK_DEV_EXPERT_CHUNK = EnvInt(None)
    SGLANG_TPU_MUSE_MK_DEV_TILE_LOOPS_BUDGET_MIB = EnvInt(None)
    SGLANG_TPU_MUSE_MK_DEV_YSEND_PLAN = EnvInt(None)
    SGLANG_TPU_MUSE_MK_DEV_TILED_VY_MIN_SLOTS = EnvInt(None)
    SGLANG_TPU_MUSE_MK_DEV_TILED_VY_MIN_DEPTH = EnvInt(None)
    # EP decode paradigm: gather (EP, rows replicated) | a2a (SP + EP, rows sharded) |
    # mixed (gather up to 128 rows, a2a above).
    SGLANG_TPU_MUSE_MK_EP_ARM = EnvStr("gather")
    # a2a arm: up to this many rows, pool / unpool read a 1/tp column slice of the weights for
    # every row (8 instead of 64 MiB per layer); above it, gathering every row costs more. 0: off.
    SGLANG_TPU_MUSE_MK_EP_COL_POOL_MAX_ROWS = EnvInt(16)
    # EP megakernel without Pallas bounds checks.
    SGLANG_TPU_EP_NO_BOUNDS = EnvBool(False)
    # Decode megakernels (TP and EP): rows up to which global-layer chunk loops are hoisted
    # per global layer. 0 (default): one shared loop with tree-walk KV refill, which keeps
    # the loops in IMEM overlays (q8 32k: TP -3 to -6.5%); 64 restores the previous hoisting.
    SGLANG_TPU_MUSE_MK_ATTN_HOIST_MAX_ROWS = EnvInt(0)
    # Muse Spark weight-only per-vocab-row FP8 LM head (shared across DECODE, EXTEND,
    # DFlash TARGET_VERIFY, and DFlash draft argmax with one single [H, Vp] fp8 copy).
    SGLANG_TPU_MUSE_FP8_LM_HEAD = EnvBool(False)
    # Single-kernel fused FP8 LM head + distributed exact argmax (k=1) / Top-K (k<=64)
    # with zero HBM logits materialization and 1-hop cyclic VMEM all-gather. Implies
    # SGLANG_TPU_MUSE_FP8_LM_HEAD and shares the same FP8 weight buffer across all call-sites.
    SGLANG_TPU_MUSE_FUSED_FP8_LM_HEAD_TOPK = EnvBool(False)
    # Lossless vertical-bitonic + 1-hop cyclic VMEM all-gather replacement for the
    # 3-hop HBM ring in Pallas distributed TopK (works on BF16/FP32 logits, B <= 256).
    SGLANG_TPU_MUSE_FAST_DISTRIBUTED_TOPK = EnvBool(False)
    # Wave-8 DFlash verify bridge + gate-fuse lift: enables Bridge 1b/2b, QKV epilogue,
    # and DECODE_ATTN_GATE_FUSE across dq=4/5/6 (B up to 384, incl. B=320 on d=4).
    SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT = EnvBool(False)
    # Optional explicit max token cap override for Wave-7/8 decode collective bridges (0 = default).
    SGLANG_TPU_MUSE_DECODE_BRIDGE_MAX_TOKENS = EnvInt(0)
    # Wave-8 Bridge 1b in-kernel router sigmoid + correction bias + Top-8 + expert counts
    # overlapped under pre_expert Stage-3 D2D wait (skips 256-col router Top-K inside muse_sparse).
    SGLANG_TPU_MUSE_DECODE_BRIDGE1_FUSE_TOPK = EnvBool(False)
    # Wave-8 Bridge 2b direct [1, B, 8, 2, 128] Q + [B, 1, 4, 128] FP8 KV + [B, 1024] Gate
    # fusion (fuses decode_qkv_epilogue directly into Bridge 2b & L0 dual_qkv_gate).
    SGLANG_TPU_MUSE_DECODE_BRIDGE2_DIRECT_QKV_EPI = EnvBool(False)
    # Wave-8 Bridge 2b VPU/MXU overlap: interleave my_half (W_qkv, W_gate) MXU with mate_half _epi.
    SGLANG_TPU_MUSE_DECODE_BRIDGE2_OVERLAP_EPI = EnvBool(False)
    # Wave-8 auto-route non-speculative greedy k=1 directly to DISTRIBUTED_GREEDY (tn=1024 argmax).
    SGLANG_TPU_MUSE_GREEDY_TOPK1_AUTO = EnvBool(False)
    # Wave-8 fast fused FP8 LM head Top-64: tn=1024 + 3-hop XOR=1->2->4 tree-butterfly merge.
    SGLANG_TPU_MUSE_FAST_FUSED_LM_HEAD_TOPK = EnvBool(False)
    # Wave-8 on-device Gumbel-max fused FP8 LM head for unconstrained temperature sampling
    # (exact multinomial categorical draw in 1 kernel with zero HBM logits & zero Top-64 sort).
    SGLANG_TPU_MUSE_FUSED_GUMBEL_MAX_LM_HEAD = EnvBool(False)
    # Muse Wave-7 prefill Kernel A1 dynamic per-row W8A8 FP8 (2x FP8xFP8 MXU with packed
    # hi+lo e4m3 row scale in cols 8192:8320, zero outlier clipping, and local_first SCAG overlap;
    # bypassed when A1_AG_UNDER_D=1).
    SGLANG_TPU_MUSE_PREFILL_A1_W8A8_DYN = EnvBool(False)
    # Wave-7 zero-spill HBM-under-ICI overlapped collective + FP8 dense projection bridges
    # (sglang_tpu/kernels/decode_collective_bridges.py). Default 0 = off.
    # 1 = Bridge 1b only (10-op attn -> MoE, B=8..128);
    # 2 = Bridge 2b only (7-op MoE -> next attn qkv/gate, L0 dual qkv/gate, L61 res-only);
    # 3 = Full Champion Dual Megabridges (B1b + B2b across B=8..128, L0 dual, L61 res-only);
    # 4 = Bridge 1b + Bridge 2a (keeps qkv/gate untouched for LOGO ablation);
    # 5 = Hybrid B1c (FP8 o_proj + XLA AR + 7-op post_attn_to_moe) + Bridge 2b;
    # 6 = Hybrid B1c + Bridge 2a.
    SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES = EnvInt(0)
    # Use BF16 router weights inside decode collective Bridge 1 (default False: exact FP32
    # router weights hide 100% under ICI at B=64 — 63.68 vs 63.68 us/L — with zero drift;
    # set True to save 248 MiB HBM across 62 layers).
    SGLANG_TPU_MUSE_DECODE_BRIDGE_ROUTER_BF16 = EnvBool(False)
    # Opt-in HBM saver for decode collective bridges: release replaced BF16 dense weights after
    # MusePrefillKernels captures prefill weights (falls back to FP8 buffers on short EXTEND).
    SGLANG_TPU_MUSE_DECODE_BRIDGE_FREE_BF16 = EnvBool(False)
    # Wave-8 prefill gmm_v2_gather_scatter inner_kernel scoped W4A8 VMEM slice & zero-spill sub-tile:
    # 0 = off; 1 = slice [rk0:rk1, start_n:end_n] directly from VMEM + explicit fp8 cast (avoids
    # monolithic 8 MiB uint32 SSA load); 2 = 1 + m_sub=128 sub-tiling at tile_m>=256;
    # 3 = 2 + mh_outer<=48-VREG zero-spill tiling + 32-row K-half flq LHS quant (0 FP8 VMEM RMW);
    # 8 = 1 + hoist_rhs_scale=True + K-half quantize_3d_lhs_tile_to_fp8 (m_step=256, smaller IMEM).
    SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8 = EnvInt(0)
    # Wave-8 prefill gmm_v2_fused_rs expert-outer Scaled-FP8 expansion:
    # 0 = off; 1 = per-strip FP4->Scaled-FP8 VMEM expansion + zero-spill k_chunk=256 FP8 MXU dot
    # in m_sub=128 sub-tiles (0 inner-K VPU scale muls); 2 = pair-batched (2 consecutive same-expert
    # gm tiles = 512 rows) strip-major execution reusing 1x VPU FP4->Scaled-FP8 expansion across
    # both gm tiles with 32-row K-half LHS quant and zero-spill sw3 single-tail fallback.
    SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8 = EnvInt(0)
    # Wave-8 prefill gmm_v2_fused_rs + Kernel D unit-RMS static-FP8 direct-write reduce-scatter:
    # 0.0 = off; >0 (recommended 4.0, exact 2^2 with 64*4=256<=448 zero-clipping guarantee after
    # in-kernel post_expert_norm) emits float8_e4m3fn[rows, 32, 128] over ICI RS (-50% RS & HBM
    # gather bytes, -4 MiB GMM VMEM), compatible with block_scatter + chipmate.
    SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS = EnvFloat(0.0)
    # Wave-8 prefill chipmate deferred weight send: defer donor TC's 24.4 MiB intra-chip remote
    # weight send until after gm_id==0 cold-start DMAs complete (eliminates 48.8 MiB HBM burst).
    SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND = EnvBool(False)
    # Wave-8 prefill chipmate borrowed expert slots per TC (1 = default intra-chip E_33;
    # 2 = two-round intra-chip rebalance over E_34; 3 = 3D hypercube butterfly xor=(1,2,4)
    # over E_35 on EP=8: slot 0 intra-chip xor=1, slot 1 1-hop ICI X xor=2, slot 2 1-hop ICI Y xor=4).
    SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS = EnvInt(1)
    # Wave-9 prefill gmm_v2 zero-RMW 32-row flq + fused GMM1->GMM2 activation rowmax:
    # 0 = off; 1 = 32-row concat zero-RMW FP8 store in GMM1 flq (1x gather read, 0 FP8 RMW);
    # 2 = 1 + fuse GMM2 activation rowmax into GMM1 Up epilogue (eliminates 2 MiB GMM2 read pass).
    SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW = EnvInt(0)
    # Wave-8 prefill Kernel A1 column-strip sub-tiling (0 = off; 256 or 512 = zero-spill
    # [tm, sub_n] f32 accumulators + progressive Gate/Q/KV 1-way async HBM writes).
    SGLANG_TPU_MUSE_KERNEL_A1_SUB_N = EnvInt(0)
    # Wave-8 prefill Kernel A1 single-butterfly 6-roll dual-head 64-lane RMSNorm in VREGs.
    SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM = EnvBool(False)
    # Wave-18 opt-in BF16 reverse all-to-all in Ulysses attention (A1_VARIANT="v10_sp"):
    # sends bf16 across rev-A2A and applies 1x final static-FP8 quant inside gate-fuse, matching v6.
    SGLANG_TPU_MUSE_KERNEL_ULYSSES_REV_BF16 = EnvBool(False)
    # Wave-8 prefill SP Entry: Pallas async remote-copy A2A + fused Pallas embed RMSNorm.
    SGLANG_TPU_MUSE_KERNEL_ENTRY_ASYNC_A2A = EnvBool(False)
    # Wave-8 prefill Kernel A3 (v16 opt_wo): unlock bn=2048 at T=16384 (num_t_half=2, 60 MiB VMEM).
    SGLANG_TPU_MUSE_KERNEL_A3_BN2048 = EnvBool(False)
    # Wave-8 prefill Kernel A3 local K-deep [t_shard, 8192] @ [8192, 8192] Wo GEMM (0 = off;
    # 1 = Pallas async A2A + VMEM-resident [t_shard, 8192] 1-pass Wo with 0 acc_vmem;
    # 2 = under CCS=1, overlap per-chunk prologue+A2A for k<K-1 and fuse last-chunk push-A2A into Wo).
    SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO = EnvInt(0)
    # Wave-8 prefill Kernel D run-coalesced variable-length HBM gather + Wsel[64, 512] MXU combine.
    SGLANG_TPU_MUSE_KERNEL_D_COAL_GATHER = EnvBool(False)
    # Wave-8 prefill Kernel D single-call in-kernel push All-Gather of next_attn_in[:rows0]
    # (eliminating the 2-call D split, 32 MiB/L wunpool reload, and 61 XLA all-gather ops).
    SGLANG_TPU_MUSE_KERNEL_D_IN_KERNEL_AG = EnvBool(False)
    # Wave-8 prefill native TPU7x FP8xNVFP4 (float4_e2m1fn) dense Wo in A3 (0 = off; 256 = qbs=256).
    SGLANG_TPU_MUSE_KERNEL_W4_WO = EnvInt(0)
    # Wave-8 prefill native TPU7x FP8xNVFP4 (float4_e2m1fn) dense wunpool in D (0 = off; 256 = qbs=256).
    SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL = EnvInt(0)
    # Wave-8 token-sharded stream [B//TP, 8192] + TP_chip=2 W_pre/W_post across Bridge 1b & 2b.
    SGLANG_TPU_MUSE_DECODE_TOKEN_SHARDED_BRIDGES = EnvBool(False)
    # Wave-9 chip-granularity stream [B//(TP//2), 8192] across Bridge 1b, Bridge 2b, Final Norm,
    # and Entry on BOTH TP=8 (B=32..384) and TP=16 (B=64..384), eliminating intra-chip stream
    # split/re-gather D2D barriers and concat loops.
    SGLANG_TPU_MUSE_DECODE_CHIP_SHARDED_BRIDGES = EnvBool(False)
    # Opt-in parallel orthogonal 3-slab XYZ reduce-scatter in TP=16 B=64 chip-sharded Bridge 1b.
    SGLANG_TPU_MUSE_DECODE_TP16_PARALLEL_RS = EnvBool(False)
    # Opt-in Bug C (4 KiB sq_pair swap, -50% VPU) + half-col 1-shot ag_exp + d2d2 at r_pair=8.
    SGLANG_TPU_MUSE_DECODE_BUG_C_ONESHOT = EnvBool(False)
    # Wave-9 opt-in TP_chip=2 W_post [4096, 4096] in chip-sharded Bridge 2b & L61 Final Norm on TP=8,
    # eliminating both tok_s2 and post_ici inter-chip ICI collectives (+0.97 GiB/rank HBM).
    SGLANG_TPU_MUSE_DECODE_CHIP_B2_TP2_WPOST = EnvBool(False)
    # Wave-9 opt-in Step (b) [chip_b1 + muse_sparse + chip_b2] decode superkernel (L0..L60)
    # + Step (a) [chip_b1 + muse_sparse] on L61 across TP=8/16 (eliminates inter-op XLA
    # barriers and keeps residual stream live in VMEM across MoE).
    SGLANG_TPU_MUSE_DECODE_B1_MOE_B2_SUPERKERNEL = EnvBool(False)
    # Wave-9 opt-in Step (c) [chip_b1 + muse_sparse(3d) + chip_b2 + RPAd] decode superkernel
    # on SWA / bounded-context layers (<= 20 ring slots*pages, B <= 64, EP=8/16): fuses next
    # layer RPAd directly behind Bridge 2b QKV/RoPE/KV epilogue and skips next self_attn.
    SGLANG_TPU_MUSE_DECODE_STEP_C_SUPERKERNEL = EnvBool(False)
    # Wave-8 fuse decode QKV RMSNorm + RoPE + KV FP8 quant epilogue directly into Bridge 2b & L0.
    SGLANG_TPU_MUSE_DECODE_BRIDGE_QKV_EPI = EnvBool(False)
    # Wave-9 unlock fused in-VMEM QKV/RoPE/FP8-KV epilogue on TP=16 (qkv_cols=640, 8Q/1KV) across
    # chip_b2, W8 Bridge 2b, L0 dual_qkv_epi_gate, and standalone decode_qkv_epilogue.
    SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI = EnvBool(False)
    # Wave-8 fuse final L61 model.norm into Bridge 2a and skip dead final stream HBM store.
    SGLANG_TPU_MUSE_DECODE_FUSE_FINAL_NORM = EnvBool(False)
    # Wave-8 Pallas owner-row gather / token-shard RS + RMSNorm replacing VocabParallelEmbedding AR.
    SGLANG_TPU_MUSE_DECODE_PALLAS_EMBED = EnvBool(False)
    # Wave-10 prefill (8, 128)-native zero-pad/zero-slice packing in pallas_local_inv_perm_and_counts.
    SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK = EnvBool(False)
    # Wave-10 prefill local-expert key compaction before lhs_indices sort (0 = off;
    # 1 = Pallas 2D (128,128) VMEM+MXU compaction + 32k/64k/full lax.switch power-of-2 sort;
    # 2 = pure-XLA 4096-chunk pre-compacted 32k sort + lax.cond exact fallback).
    SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT = EnvInt(0)
    # Wave-12 prefill zero-concat 2-chunk CCS A3 Wo: pass (q_part0_3d, q_part1_3d) directly into
    # _opt_wo_2chunk_kernel without torch.cat(q_parts, dim=1) (%pad_maximum_fusion).
    SGLANG_TPU_MUSE_KERNEL_A3_CCS_2CHUNK = EnvBool(False)
    # Wave-12 prefill enable 1-op Pallas VMEM+SMEM chipmate router during TorchTPU CPU export
    # (via get_tpu_info_or_none() is not None instead of jax.default_backend() == 'tpu').
    SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE = EnvBool(False)
    # Opt-in chunked-prefill exact token budget filling for TPU (schedule_policy.py).
    # 0 = off (default, page-ceiled compute budgets); 1 = exact token compute charge +
    # page-aligned chunk cuts; 2 = exact token compute charge + exact unaligned cuts;
    # >=64 = exact token compute charge + chunk cuts aligned to the token multiple (e.g. 4096).
    SGLANG_TPU_EXACT_CHUNK_FILL = EnvInt(0)
    # Wave-16 opt-in asymmetric power-of-2 B1+B2 chunk split (e.g. 512+1024 on TP=16,
    # 1024+2048 on TP=8 for 24K buckets) under GMM_PREQUANT_LHS_FP8=5 so every B2
    # all_gather is a power-of-2 size without non-pow2 SparseCore AG cliffs.
    SGLANG_TPU_MUSE_KERNEL_B2_POW2_CHUNKS = EnvBool(False)
    # Wave-16 opt-in double-buffered stream HBM->VMEM prefetch + direct (bm, 32, 128) FP8 store
    # inside pallas_fused_b1_b2_wpool_quant (-0.73..-0.96 ms/61L on 4c Ghostfish, 0-ULP bit-exact).
    SGLANG_TPU_MUSE_KERNEL_B2_DBL_STREAM = EnvBool(False)
    # Wave-18 opt-in multi-chunk pow2 B1+B2 + cat_sc pipeline ("auto" -> (512,1024,1024,512) at
    # t_local=3072, (512,512,512) at t_local=1536; or explicit CSV) overlapping SparseCore AG_i8.
    SGLANG_TPU_MUSE_KERNEL_B2_PIPE_CHUNKS = EnvStr("")

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False


def dev_int(field: EnvInt, default: int) -> int:
    """A kernel-sweep knob: its value if set, else the tuned default."""
    value = field.get()
    return default if value is None else value
