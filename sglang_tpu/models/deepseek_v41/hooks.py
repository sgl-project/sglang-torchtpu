# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 TPU runtime hooks (compress ratios 0/1/2).

Every hook here passes V4 (no ratio-1/2 layers, no Engram, no vision layers)
through to ``original``. They are appended after sglang_tpu/hooks/dsv4.py so
that, on shared targets, they are the outer wrapper.

build_tpu_dsv41_kv_pool, get_tpu_dsv41_bytes_per_swa_token,
get_tpu_dsv41_bytes_per_full_token
  Why: KVCacheConfigurator._build_dsv4_kv_pool hardcodes DeepSeekV4TokenToKVPool
       and the V4 TPU hook rebinds it to DSV4TPUTokenToKVPool, which rejects
       ratios 1/2; DSV4PoolConfigurator budgets FlashMLA bytes per token.
  How: AROUND the three targets, bodies in
       sglang_tpu.models.deepseek_v41.memory_pool (owned by the pool module).
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: KVCacheConfigurator and DSV4PoolConfigurator query
       current_platform for the DSV4 pool class and its byte sizing.

select_tpu_dsv41_loader
  Why: each ~100 GB Engram table is one tensor, far above a per-rank Run:AI
       buffer, and every rank keeps only its TP row range of it.
  How: AFTER get_model_loader swaps in DSV41RunaiModelStreamerLoader for a
       target model with Engram layers. Routed experts take the generic
       EP-local Run:AI read (sglang_tpu.ep_weight_shard); Engram tables leave
       the stream and are read as this rank's row range.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: EngramEmbedding loads a pre-sliced row range from a
       model-owned checkpoint tensor filter.

load_tpu_dsv41_engram_rows
  Why: EngramEmbedding._load_rows slices its TP row range out of the full
       ~100 GB table, so the loader must hand it the whole table. The V4.1
       loader reads only this rank's rows (one contiguous range, concurrent
       pread), which the original would slice a second time.
  How: AROUND EngramEmbedding._load_rows; tensors tagged with ROW_START_ATTR
       are copied as-is after a row-range check, others take the original path.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: EngramEmbedding accepts a pre-sliced row range.
trace_tpu_platform_fact
  Why: the V4.1 shifted-mHC loop in DeepseekV4Model reads
       get_platform().is_blackwell inside the compiled forward;
       PlatformContext.__getattr__ uses object.__getattribute__, which Dynamo
       cannot trace, so torch.compile fails during TPU warmup.
  How: AROUND PlatformContext.__getattr__ resolves the fact through a
       torch._dynamo.assume_constant_result helper; facts are fixed once the
       process starts, before any model is traced.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: PlatformContext is Dynamo-traceable or V4.1 forward reads
       platform facts outside the compiled region.
tpu_dsv41_hc_combine
  Why: the shifted-mHC forward (DeepseekV4DecoderLayer._hc_combine and the
       model's final hc_head) calls the Triton hc_combine with no device
       guard; Dynamo cannot trace a Triton launch on TPU.
  How: AROUND sglang.kernels.ops.layernorm.mhc.mhc_combine computes the same
       fp32 weighted sum of the hc_mult copies in torch for non-CUDA tensors.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: hc_combine dispatches on device like the other mHC kernels.

tpu_dsv41_engram_gate
  Why: the torch fallback of sglang.srt.layers.engram.engram_gate uses
       torch.copysign, which TorchTPU does not implement (aten::copysign.out).
  How: AROUND engram_gate keeps the original on CUDA; elsewhere computes the
       same gate via engram_host.signed_sqrt (int32 sign-bit check so -0.0
       preserves its sign, matching copysign), then the core's image-token mask.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: TorchTPU implements aten::copysign or engram_gate avoids it.

tpu_dsv41_engram_worker_forward
  Why: hashing on CPU from device input_ids/positions stalls every decode step
       on D2H copies and oversubscribes PyTorch's intra-op thread pool across
       8 scheduler processes. Core TpModelWorker.forward_batch_generation has
       no host-input preparation seam before ForwardBatch construction.
  How: AROUND TpModelWorker.forward_batch_generation captures the scheduled
       batch's host Engram windows (including folded encoder SWA replay prefixes
       and speculative extends) without waiting for step N-1's sampled token
       (overlap holes are resolved right before the compiled forward), records
       PendingTokens after non-speculative sampling, and clears per-step state
       in finally. Unsupported modes (warmup, non-folded two-pass SWA replay
       hits, beam search, PP/CP/DCP > 1) fall back cleanly.
  Upstream PR or issue: not filed (ported from sgl-project/sglang-torchtpu#2092).
  Remove when: upstream exposes a worker-level host-input preparation seam.

tpu_dsv41_engram_embed_init, tpu_dsv41_engram_embed_forward,
tpu_dsv41_hasher_forward
  Why: TPU serves Engram only from one host table per node shared by the DP
       ranks (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1, layout "shared"); the
       in-graph lookup of HBM tables returned wrong outputs under attention DP
       and its offloaded gather halted a SparseCore. _HostTable pins its
       mapping with cudaHostRegister, EngramEmbedding queries the current CUDA
       device for its host read order, and the core lookup reads the table
       from a GPU.
  How: AROUND EngramEmbedding.__init__ rejects any other placement on TPU and
       builds the table with cudaHostRegister and the CUDA device query
       stubbed out. The V4 compile policy and its forward shim
       (deepseek_v41/compile_policy.py) hash on the host and stage packed rows
       via double-buffered async raw DMA; inside the graph
       EngramHasher.forward returns the staged row ids and
       EngramEmbedding.forward unpacks and dequantizes the staged rows.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: the device can read host tables, or the in-graph HBM lookup
       is correct under attention DP.

allow_tpu_dsv41_encoder_replay
  Why: validate_deepseek_v41_features rejects --enable-encoder-swa-bounded-replay
       on hardware other than CUDA and gfx950 with one inline check, so the TPU
       request window and folded replay rows cannot be enabled. Every other check
       (prefill graphs, CP, ROCm DP, PD, mixed chunk, LoRA, radix sessions) must
       still apply.
  How: AROUND the validator; for a V4.1 model on TPU with the flag it runs the
       original with the module's get_platform reporting CUDA and not ROCm, the
       only two platform facts the validator reads. An unset
       --enable-decoder-swa-bounded-replay is first resolved off, since the
       validator defaults it on for CUDA and TPU attention has no decoder replay.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR3).
  Remove when: the validator asks current_platform whether it folds encoder
       SWA replay.

fold_tpu_dsv41_encoder_replay
  Why: encoder_swa_replay._fold_batch concatenates each hit's replay rows with
       the scheduled rows assuming one device and int64 slots: replay token ids
       go to runner.device and replay slots are read from req_to_token. On TPU
       req_to_token is host-resident (SGLANG_TPU_REQ_TO_TOKEN_ON_HOST), the
       DSpark spec worker stages input_ids and out_cache_loc on the host, and the
       compiled extend graph rejects an int64 out_cache_loc (it is int32).
  How: AROUND _fold_batch for the V4.1 TPU pool: runs the original against a
       runner view whose device is input_ids' device and whose req_to_token
       returns int64 slots on out_cache_loc's device, with out_cache_loc cast to
       int64 once up front and the slots read from req_pool_indices_cpu; then
       casts the folded out_cache_loc back to the scheduled dtype and moves
       scheduled_row and the Engram history to the runner's device, where the
       original places them. keep_rows and row_floor stay on the host: the
       attention metadata reads them there, and a device copy at the live row
       count would cost a readback per extend.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR3).
  Remove when: _fold_batch builds the replay rows on the scheduled tensors'
       devices and keeps out_cache_loc's dtype.

drop_tpu_dsv41_folded_rows
  Why: encoder_swa_replay.drop_folded_rows gathers the scheduled rows of a
       folded extend's hidden at their live count. On TPU that gather, and the
       draft KV writer's re-pad of its result, compile a program per extend
       length; the spec workers read the hidden at the token bucket.
  How: AROUND drop_folded_rows for host keep_rows (fold_tpu_dsv41_encoder_replay):
       gathers with a host-built index of the hidden's own row count whose head
       is keep_rows; the tail rows repeat the last row, and the spec workers write
       rows past the live tokens to slot 0, as for the target's padding rows.
  Upstream PR or issue: not filed (V4.1 TPU bring-up).
  Remove when: drop_folded_rows keeps the hidden's row count.

check_tpu_dsv41_decoder_replay (in allow_tpu_dsv41_encoder_replay)
  Why: validate_deepseek_v41_features rejects --enable-decoder-swa-bounded-replay
       under attention DP (input_ids_global is a DP-wide gather), but the TPU
       backend gives every DP rank the same padded decoder row bucket and
       re-gathers the decoder rows' DP-wide ids (decoder_replay.py); TPU plans
       pure extends only and needs encoder SWA replay under the radix cache.
  How: the validator wrapper runs the TPU rules (prefill graphs, mixed chunk,
       radix cache), then the original with a resolving view that reports the
       flag off, so every other check still applies.
  Upstream PR or issue: not filed (decoder CED prefill on TPU).
  Remove when: the core validator asks the platform about DP decoder tails.

decoder_replay_sync.hook_specs (decoder SWA bounded replay scheduler state)
  Why: the TPU backend needs each extend request's full fill length (the
       "prompt" row rule) and every DP rank's decoder row count (the shared
       decoder row bucket); core's ScheduleBatch, ForwardBatch and MLP sync
       carry neither.
  How: AFTER prepare_for_extend / convert_decode_to_extend / copy keep
       ScheduleBatch.decoder_fill_lens; AFTER ForwardBatch.init_new copies it
       and global_decoder_rows, and AFTER EagerRunner.load_batch carries them
       to the ForwardBatch it rebuilds from dataclass fields; while prepare_mlp_sync_batch_raw runs with a
       decoder-replay backend, the MLP-sync info tensors carry one more column
       (this rank's rows; idle slots 0), read back after all_gather /
       finalize_local and set on the synced batch by _update_gather_batch.
       AROUND DeepseekV4Model._forward_layers_hc_pre_from_prev stops the
       layer loop at the first decoder layer when no DP rank keeps a decoder
       row (bucket 0; every row is mid-prompt and unread), on every rank
       together, with zero DSpark aux rows for skipped capture layers.
       AROUND DeepseekV4Model.forward puts the DSpark capture (hidden,
       pre-head, aux rows; core leaves them in the tail layout) back in the
       full layout, and the causal-LM head of that forward reads an identity
       tail (AROUND DeepseekV4ForCausalLM.forward clears it).
       AROUND deepseek_v4._scatter_tail_rows zero-initializes the full layout
       for the TPU tail (core leaves rows outside the tail unset, but under the
       "prompt" rule the logits processor reads the last row of a chunk that
       kept no decoder rows); other tails go to core unchanged.
  Upstream PR or issue: not filed (decoder CED prefill on TPU).
  Remove when: core carries per-rank decoder rows in the MLP sync.

init_tpu_dsv41_pool_configurator
  Why: DSV4PoolConfigurator.__init__ counts the DSpark draft layers kept in a
       per-request SWA ring as a local; get_tpu_dsv41_swa_fixed_bytes prices those
       ring pages in the TPU layout and needs the count.
  How: AROUND __init__: after the original, set draft_ring_layers by core's rule
       and assert it agrees with the paged_draft_layers core kept.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR4).
  Remove when: the configurator exposes the draft ring layer count.

derive_tpu_dsv41_pool_sizes, build_tpu_dsv41_allocator, get_tpu_dsv41_swa_fixed_bytes
  Why: under --enable-encoder-swa-bounded-replay the core sizes the paged SWA pool
       at 0 tokens (SWA lives in the request window), which the TPU 8-page SWA
       floor in _derive_pool_sizes refuses; the allocator builder then takes the
       core PagedTokenToKVPoolAllocator, whose alloc paths launch Triton; and the
       configurator prices core's FlashMLA-packed window instead of the TPU
       window's bf16 pages.
  How: AROUND each target, bodies in sglang_tpu.models.deepseek_v41.memory_pool:
       skip the floor for a 0-token SWA size only, bind the module's
       PagedTokenToKVPoolAllocator to the TPU paged allocator, and swap the
       window bytes. Every other case calls the original unchanged.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR3).
  Remove when: the SWA floor skips pools without a paged SWA allocator, that
       allocator arm asks current_platform for its class, and the configurator
       asks the pool for its window bytes.

trace_tpu_env_default
  Why: EnvField._resolve_default calls callable(self.default) for lazily
       computed defaults (e.g. SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)),
       which Dynamo cannot trace; the V4.1 MoE DP path reads it in the forward.
  How: AROUND EnvField._resolve_default resolves the default through a
       torch._dynamo.assume_constant_result helper keyed by the env name;
       defaults depend only on the platform, fixed before any model is traced.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: callable env defaults are resolved at import or are traceable.

tpu_dsv41_dspark_attention, tpu_dsv41_dspark_forward,
tpu_dsv41_dspark_write_target_hidden_kv
  Why: DSparkAttention.forward and write_target_hidden_kv write the draft KV
       with the CUDA FlashMLA writer, and the attention inverse-rotates its
       output, which CSA2 already does; the draft forward returns no
       next_draft_tokens, which TpuBlockSpecWorker reads for greedy drafts.
  How: AROUND each when the draft runs on the CSA2 backend (dspark.py):
       attention goes through the backend's prepare_mqa_qkv and non-causal
       forward, then wo_a / wo_b; the KV write publishes into the CSA2 SWA
       pages; the forward appends the greedy Markov chain in-graph, as
       vllm-torchtpu models/vllm/deepseek_v41/v2/dspark.py does.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR2).
  Remove when: the V4 draft reads its KV write and inverse RoPE from the
       attention backend and returns draft tokens.

tpu_dsv41_dspark_compute_base_logits
  Why: under attention DP each rank's lm_head holds the full vocab
       (tp_size 1), but gather_and_crop_vocab gathers over the full TP group
       and keeps TP rank 0's rows, so every rank drafted rank 0's tokens.
  How: AROUND compute_base_logits projects this rank's rows without a gather
       when the head is unsharded.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR2).
  Remove when: compute_base_logits gathers over the lm_head's own group.

tpu_dsv41_vocab_parallel_load_weights, tpu_dsv41_vocab_parallel_lm_head
  Why: under attention DP at attention TP 1 every rank holds the whole LM head
       and reads all 1.32 GB of it at every target verify and DSpark draft call
       to score its own few rows; core has no head split over the DP ranks that
       keeps each rank's own rows (--enable-tp-lm-head-all-to-all shards over
       the TP group, which DSpark under DP rejects).
  How: with SGLANG_TPU_ENABLE_DSV41_VOCAB_PARALLEL_HEAD, AROUND
       DeepseekV4ForCausalLM.load_weights keeps only this rank's vocab slice of
       a V4.1 target's head (vocab_parallel_head.shard_lm_head), and AROUND
       LogitsProcessor._compute_lm_head_rows scores a split head through the
       vocab-parallel op; the DSpark draft's tpu_compute_base_logits shares it.
  Upstream PR or issue: not filed (V4.1 TPU serving).
  Remove when: core shards the LM head over the vocab across the DP ranks
       with each rank's logits rows returned to it.

tpu_dsv41_carry_mirrored_padding
  Why: host_input_mirror hands build_forward_metadata the host copies of a
       batch's positions, out_cache_loc and req_pool_indices only while the
       batch holds the exact device tensors they describe, and
       ForwardBatch._pad_inputs_to_size (DP MLP sync) rebuilds all three with
       torch.cat, a new tensor even at an unchanged size; every record a spec
       worker made before the runner fell back to a blocking device read.
  How: AROUND _pad_inputs_to_size moves each record to the padded tensor and
       zero-pads its host copy the same way (host_input_mirror.carry_padding).
  Upstream PR or issue: not filed (V4.1 TPU serving).
  Remove when: the padding returns a field unchanged when it has nothing to
       append, and records follow the fields they describe.

tpu_dsv41_flush_engram_at_length_read
  (registered only with SGLANG_TPU_DSV41_FLUSH_ENGRAM_AT_LENGTH_READ=1)
  Why: the queued Engram verify commits read the previous verify's outputs back.
       Flushed at the next verify's staging, they are the first device read
       after the draft launch; there the host has to wait for the draft.
  How: AFTER StepHostRows._build reads the step's sequence lengths (which
       already waited for the previous verify), apply the queue then; the
       verify staging finds it empty. Commit order is unchanged.
  Upstream PR or issue: not filed (V4.1 TPU serving).
  Remove when: the Engram history lives on the device and commits in-graph.

tpu_dsv41_carry_mirror_to_loaded_batch
  Why: host_input_mirror ties its records to the ForwardBatch object, and on
       TPU EagerRunner.load_batch hands the forward a dataclasses.replace copy;
       build_forward_metadata then finds no record and reads positions,
       out_cache_loc and req_pool_indices back from the device, blocking on
       the previous forward (three reads before every draft and verify).
  How: AROUND EagerRunner.load_batch moves the record to the returned copy
       (host_input_mirror.carry_batch); the copy holds the same tensors.
  Upstream PR or issue: not filed (V4.1 TPU serving).
  Remove when: records follow the fields they describe across batch copies.

tpu_dsv41_defer_engram_commit, tpu_dsv41_flush_engram_before_prebuilt
  Why: the target's Engram history lives on the host (engram_host), so
       committing a verify reads its outputs back and blocks the host for the
       whole verify; core commit_after_verify also writes padded rows into
       req slot 0.
  How: AROUND NgramEmbeddingManager.update_after_verify queues the commit;
       the next host Engram staging (compile_policy) applies the queue, and a
       PD prebuilt batch, which writes history rows in _prepare_engram_history
       without staging, applies it first.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR2).
  Remove when: the Engram history lives on the device and commits in-graph.

prepare_tpu_dsv41_wo_a_fp8, tpu_dsv41_project_wo_a
  Why: disable_tpu_dsv4_wo_a_fp8_gemm keeps wo_a in BF16 (~2.88 GB/rank across
       43 layers), making torch.einsum("tgd,grd->tgr") ~20.7% of decode_c8
       active time.
  How: AFTER DefaultModelLoader.postprocess_weights quantizes eligible V4.1
       wo_a weights to per-channel float8_e4m3fn ([G, R, D] + [G, R] f32
       scale) when SGLANG_TPU_DSV41_WO_A_FP8=1; AROUND MQALayer._project_wo_a
       dispatches to the exported XLA FP8 einsum op (ops/wo_a_fp8.py).
  Upstream PR or issue: not filed (V4.1 TPU kernel optimization).
  Remove when: MQALayer._project_wo_a dispatches through a backend FP8 grouped
       matmul seam.

prepare_tpu_dsv41_dummy_model
  Why: DummyModelLoader.load_model bypasses load_weights / postprocess_weights,
       so dummy-weight TPU benchmarks skip vocab-parallel LM head sharding,
       FP8 wo_a quantization, and fused EP MoE preparation.
  How: AFTER DummyModelLoader.load_model runs the same V4.1 post-load
       transforms on TPU.
  Upstream PR or issue: not filed (V4.1 TPU kernel optimization).
  Remove when: DummyModelLoader invokes the same post-load hook as
       DefaultModelLoader.

prepare_tpu_dsv41_fused_ep
  (registered only with SGLANG_TPU_DSV41_FUSED_EP=1, see fused_ep_hook_specs)
  Why: the fused EP kernel (fused_ep_moe_v2) needs its own expert layout,
       converted from the gmm_v2 FP4 expert tensors, which exist only after the
       quant methods' process_weights_after_loading. The platform's
       process_model_after_loading is only called from a model's
       post_load_weights (Kimi-K3 today), which runs inside load_weights,
       before DefaultModelLoader.postprocess_weights: too early.
  How: AFTER DefaultModelLoader.postprocess_weights (after
       prepare_model_for_fused_moe_decode, before precompile) converts the
       eligible layers' experts, builds the op and sets mlp.dp_local_forward,
       which NonCudaDecoderMixin._run_moe_ffn_dp_sync (deepseek_v4_platform_stage)
       calls with this rank's rows. Bodies in
       sglang_tpu.models.deepseek_v41.fused_ep_op.
  Upstream PR or issue: not filed.
  Remove when: the loader offers a platform step after quant-method weight
       processing.
"""

from __future__ import annotations

import torch


def build_tpu_dsv41_kv_pool(original, kvc, *args, **kwargs):
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        build_tpu_dsv41_kv_pool as _impl,
    )

    return _impl(original, kvc, *args, **kwargs)


def get_tpu_dsv41_bytes_per_swa_token(original, configurator) -> float:
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        get_tpu_dsv41_bytes_per_swa_token as _impl,
    )

    return _impl(original, configurator)


def get_tpu_dsv41_bytes_per_full_token(original, configurator) -> float:
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        get_tpu_dsv41_bytes_per_full_token as _impl,
    )

    return _impl(original, configurator)


def allow_tpu_dsv41_encoder_replay(original, server_args):
    import contextlib
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.arg_groups import deepseek_v4_hook as validator
    from sglang.srt.arg_groups.overrides import declare_resolution
    from sglang.srt.runtime_context import get_platform

    cfg = validator.resolving_view(server_args)
    if (
        not get_platform().is_tpu
        or validator.model_config_of(server_args).hf_config.model_type != "deepseek_v41"
    ):
        return original(server_args)
    if cfg.enable_decoder_swa_bounded_replay is None:
        # Under the CUDA spoof below the validator defaults decoder replay on;
        # TPU serves it only when asked for.
        declare_resolution(
            server_args,
            "allow_tpu_dsv41_encoder_replay",
            enable_decoder_swa_bounded_replay=False,
        )
    with contextlib.ExitStack() as stack:
        if cfg.enable_decoder_swa_bounded_replay:
            check_tpu_dsv41_decoder_replay(cfg)
            # Core's decoder-replay checks (DP attention, prefill graphs) are
            # replaced by the TPU rules above.
            view = validator.resolving_view
            stack.enter_context(
                patch.object(
                    validator,
                    "resolving_view",
                    lambda args: _WithoutDecoderReplay(view(args)),
                )
            )
        if cfg.enable_encoder_swa_bounded_replay:
            cuda = SimpleNamespace(is_cuda=True, is_hip=False)
            stack.enter_context(patch.object(validator, "get_platform", lambda: cuda))
        return original(server_args)


class _WithoutDecoderReplay:
    """A resolving view that reports --enable-decoder-swa-bounded-replay off."""

    def __init__(self, cfg):
        object.__setattr__(self, "_cfg", cfg)

    def __getattr__(self, name):
        if name == "enable_decoder_swa_bounded_replay":
            return False
        return getattr(object.__getattribute__(self, "_cfg"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_cfg"), name, value)


def check_tpu_dsv41_decoder_replay(cfg) -> None:
    """TPU rules of --enable-decoder-swa-bounded-replay beyond core's.

    The backend plans decoder rows for pure extends only, so MIXED batches are
    out. The radix cache would hand a hit decoder SWA rows that were never
    computed (only each earlier request's tail ran the decoder), so it needs
    encoder SWA replay, which keeps no SWA in the cache and rebuilds the window.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend

    if cfg.cuda_graph_config.prefill.backend != Backend.DISABLED:
        raise ValueError(
            "--enable-decoder-swa-bounded-replay cannot be combined with the "
            "prefill CUDA graph"
        )
    if cfg.enable_mixed_chunk:
        raise ValueError(
            "--enable-decoder-swa-bounded-replay on TPU cannot be combined with "
            "--enable-mixed-chunk"
        )
    if not cfg.disable_radix_cache and not cfg.enable_encoder_swa_bounded_replay:
        raise ValueError(
            "--enable-decoder-swa-bounded-replay with the radix cache requires "
            "--enable-encoder-swa-bounded-replay on TPU"
        )


class _SlotsOnDevice:
    """req_to_token view: each indexed slice as int64 on ``device``."""

    def __init__(self, table: torch.Tensor, device: torch.device):
        self._table = table
        self._device = device

    def __getitem__(self, key) -> torch.Tensor:
        # Cast where the table lives, then copy: a cast on the device after the
        # copy is a program per replay length.
        return self._table[key].to(torch.int64).to(self._device)


def fold_tpu_dsv41_encoder_replay(original, *, batch, runner, rows):
    """AROUND encoder_swa_replay._fold_batch: fold on the scheduled tensors'
    devices and keep out_cache_loc's dtype."""
    from copy import copy
    from types import SimpleNamespace

    import msgspec

    from sglang.srt.runtime_context import get_platform
    from sglang_tpu.models.deepseek_v41.memory_pool import DSV41TPUTokenToKVPool

    if not get_platform().is_tpu or not isinstance(
        runner.token_to_kv_pool, DSV41TPUTokenToKVPool
    ):
        return original(batch=batch, runner=runner, rows=rows)
    loc = batch.out_cache_loc
    scheduled = copy(batch)
    # One cast here makes the original's per-request .long() calls no-ops.
    scheduled.out_cache_loc = loc.to(torch.int64)
    slots_cpu = batch.req_pool_indices_cpu
    if slots_cpu is not None:
        # The original indexes the host req_to_token by each request's slot; a
        # device slot would be read back (and flush the device queue) per hit.
        scheduled.req_pool_indices = slots_cpu
    # The original reads only these three runner fields.
    view = SimpleNamespace(
        device=batch.input_ids.device,
        model=runner.model,
        req_to_token_pool=SimpleNamespace(
            req_to_token=_SlotsOnDevice(
                runner.req_to_token_pool.req_to_token, loc.device
            )
        ),
    )
    folded = original(batch=scheduled, runner=view, rows=rows)
    folded.batch.out_cache_loc = folded.batch.out_cache_loc.to(loc.dtype)
    folded.batch.req_pool_indices = batch.req_pool_indices
    if folded.batch.engram_history is not None:
        folded.batch.engram_history = folded.batch.engram_history.to(
            runner.device, non_blocking=True
        )

    # keep_rows and row_floor stay where the fold built them (the host, with the
    # spec worker's host-staged input_ids): the metadata reads them on the host and
    # drop_tpu_dsv41_folded_rows builds its gather index from keep_rows there.
    return msgspec.structs.replace(
        folded,
        scheduled_row=folded.scheduled_row.to(runner.device, non_blocking=True),
    )


def skip_tpu_noop_pad(original, forward_batch, tensor, size, *, value=0):
    """AROUND ForwardBatch._pad_tensor_to_size: on the V4.1 TPU pool, return a
    tensor already at ``size`` as is. The original concatenates a zero-row
    tensor, and on TPU that no-op cat is a program per row count, compiled at
    serving time on the DP rank whose batch sets the padded size. No V4.1 path
    modifies these batch fields in place, so dropping the copy is safe there;
    other models keep the original."""
    from sglang_tpu.models.deepseek_v41.memory_pool import dsv41_pool_active

    if tensor.shape[0] == size and dsv41_pool_active():
        return tensor
    return original(forward_batch, tensor, size, value=value)


def drop_tpu_dsv41_folded_rows(original, *, logits_output, folded, tail_lens=None):
    """AROUND encoder_swa_replay.drop_folded_rows: keep the hidden's row count.

    Decoder-SWA tail rows (``tail_lens``, passed by cores with decoder replay)
    take the original path; the argument is only forwarded when given."""
    hidden = logits_output.hidden_states
    keep_rows = folded.keep_rows
    if (
        hidden is None
        or tail_lens is not None
        or logits_output.hidden_states_token_indices is not None
        or keep_rows.device.type != "cpu"
        or hidden.shape[0] < folded.num_rows
    ):
        if tail_lens is not None:
            return original(
                logits_output=logits_output, folded=folded, tail_lens=tail_lens
            )
        return original(logits_output=logits_output, folded=folded)
    rows = hidden.shape[0]
    # The tail repeats the last row: those rows are past the live tokens, whose
    # draft KV rows the spec workers send to slot 0.
    index = torch.full((rows,), rows - 1, dtype=torch.int64)
    index[: keep_rows.shape[0]] = keep_rows
    index = index.to(hidden.device, non_blocking=True)
    logits_output.hidden_states = hidden[index]
    # Core drops the same rows from last_hidden_states (aux hidden capture).
    # It may still be bucket-padded when the hidden was trimmed to the live
    # folded rows; the index only reads rows both hold.
    last = getattr(logits_output, "last_hidden_states", None)
    if last is not None and last.shape[0] >= rows:
        logits_output.last_hidden_states = last[index]


def derive_tpu_dsv41_pool_sizes(original, kvc, *, config):
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        derive_tpu_dsv41_pool_sizes as _impl,
    )

    return _impl(original, kvc, config=config)


def build_tpu_dsv41_allocator(original, kvc, **kwargs):
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        build_tpu_dsv41_allocator as _impl,
    )

    return _impl(original, kvc, **kwargs)


def init_tpu_dsv41_pool_configurator(original, configurator, kvc) -> None:
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        init_tpu_dsv41_pool_configurator as _impl,
    )

    _impl(original, configurator, kvc)


def get_tpu_dsv41_swa_fixed_bytes(original, configurator) -> float:
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        get_tpu_dsv41_swa_fixed_bytes as _impl,
    )

    return _impl(original, configurator)


def select_tpu_dsv41_loader(result, load_config, model_config=None):
    from sglang.srt.model_loader.loader import RunaiModelStreamerLoader
    from sglang_tpu.ep_weight_shard import EPLocalRunaiModelStreamerLoader

    # The generic EP-local Run:AI hook may already have wrapped the stock loader.
    if (
        type(result) not in (RunaiModelStreamerLoader, EPLocalRunaiModelStreamerLoader)
        or model_config is None
        or load_config.draft_model_idx is not None
    ):
        return result
    from sglang_tpu.models.deepseek_v41.loader import (
        DSV41RunaiModelStreamerLoader,
        is_bundled_dspark_draft,
    )

    # Without Engram tables the generic EP-local loader does everything; a
    # bundled DSpark draft comes here to stream only its mtp.* tensors.
    if not (model_config.use_engram or is_bundled_dspark_draft(model_config)):
        return result

    return DSV41RunaiModelStreamerLoader(load_config)


def load_tpu_dsv41_engram_rows(original, embedding, param, loaded_weight):
    from sglang_tpu.models.deepseek_v41.engram_row_reader import load_local_rows

    return load_local_rows(original, embedding, param, loaded_weight)


_ORIGINAL_PLATFORM_GETATTR = None


@torch._dynamo.assume_constant_result
def _resolve_platform_fact(name: str):
    from sglang.srt.runtime_context import get_platform

    return _ORIGINAL_PLATFORM_GETATTR(get_platform(), name)


def trace_tpu_platform_fact(original, platform, name):
    """AROUND PlatformContext.__getattr__: facts are fixed per process, so
    Dynamo bakes them into the graph instead of tracing the override dict."""
    global _ORIGINAL_PLATFORM_GETATTR
    _ORIGINAL_PLATFORM_GETATTR = original
    return _resolve_platform_fact(name)


@torch._dynamo.assume_constant_result
def _resolve_env_default(name: str):
    from sglang.srt.environ import envs
    from sglang_tpu.envs import envs as tpu_envs

    # Class-level lookup: Envs instances refuse bare field access, and an
    # EnvField refuses bool(), so test membership explicitly.
    for owner in (type(envs), type(tpu_envs)):
        for cls in owner.__mro__:
            if name in vars(cls):
                default = vars(cls)[name].default
                return default() if callable(default) else default
    raise AttributeError(f"unknown env field {name}")


def trace_tpu_env_default(original, field):
    """AROUND EnvField._resolve_default: env defaults are fixed per process, so
    under Dynamo they are baked into the graph instead of tracing
    callable(default); eager reads take the original path."""
    if not torch.compiler.is_compiling():
        return original(field)
    return _resolve_env_default(field.name)


def tpu_dsv41_hc_combine(original, x_flat, pre, hc, out_dtype):
    """y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h], accumulated in fp32."""
    if x_flat.is_cuda:
        return original(x_flat, pre, hc, out_dtype)
    m = x_flat.shape[0]
    copies = x_flat.reshape(m, hc, -1).float()
    return (pre.float().unsqueeze(-1) * copies).sum(1).to(out_dtype)


def tpu_dsv41_engram_gate(
    original,
    x,
    kv,
    q_weight,
    k_weight,
    eps,
    clamp_value,
    input_ids=None,
    image_token_id=None,
):
    """engram_gate without torch.copysign (no TorchTPU kernel)."""
    if x.is_cuda:
        return original(
            x, kv, q_weight, k_weight, eps, clamp_value, input_ids, image_token_id
        )
    from sglang_tpu.models.deepseek_v41.engram_host import signed_sqrt

    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(signed_sqrt(dot, clamp_value))
    out = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
    if input_ids is not None:
        out = torch.where((input_ids == image_token_id)[:, None, None], x, out)
    return out


def tpu_dsv41_engram_worker_forward(original, worker, batch, *args, **kwargs):
    """AROUND TpModelWorker.forward_batch_generation: capture host Engram inputs."""
    runner = worker.model_runner
    if not runner.model_config.use_engram or runner.is_draft_worker:
        return original(worker, batch, *args, **kwargs)
    model = runner.model
    hasher = model.model.engram_hasher
    if hasher is None:
        return original(worker, batch, *args, **kwargs)
    from sglang.srt.runtime_context import get_exec, get_parallel
    from sglang_tpu.models.deepseek_v41 import engram_host

    coordinator = engram_host.get_or_create_coordinator(model, hasher, runner.device)
    if coordinator is None:
        return original(worker, batch, *args, **kwargs)
    is_verify = bool(kwargs.get("is_verify", False))
    spec_active = (
        batch is not None
        and batch.spec_algorithm is not None
        and not batch.spec_algorithm.is_none()
    )
    is_dllm_fn = getattr(worker, "is_dllm", None)
    is_dllm = bool(is_dllm_fn()) if callable(is_dllm_fn) else False
    attn_backend = getattr(runner, "attn_backend", None)
    kv_pool = getattr(runner, "token_to_kv_pool", None)
    prepared = (
        None
        if is_verify
        else coordinator.prepare_if_supported(
            batch,
            parallel=get_parallel(),
            enable_encoder_swa_bounded_replay=bool(
                get_exec().features.enable_encoder_swa_bounded_replay
            ),
            folds_encoder_swa_replay=bool(
                getattr(attn_backend, "folds_encoder_swa_replay", False)
            ),
            has_request_window=getattr(kv_pool, "request_window", None) is not None,
            is_dllm=is_dllm,
        )
    )
    try:
        result = original(worker, batch, *args, **kwargs)
        if prepared is not None and not spec_active:
            requests, positions = prepared
            coordinator.remember(requests, positions, result)
        elif is_verify or spec_active:
            coordinator.pending = None
        else:
            coordinator.remember_fallback_if_applicable(batch, result)
        return result
    finally:
        coordinator.clear_step()


def tpu_dsv41_hasher_forward(original, hasher, input_ids, forward_batch):
    """AROUND EngramHasher.forward: in the TPU graph, the host-staged row ids."""
    if input_ids.device.type == "cpu":
        # The V4 compile policy hashing on the host (engram_host.host_hash_ids):
        # the metadata may still carry the previous forward's staged ids.
        return original(hasher, input_ids, forward_batch)
    from sglang_tpu.models.deepseek_v41.engram_host import staged_inputs

    local_ids, _ = staged_inputs(forward_batch)
    if local_ids is not None:
        return local_ids
    return original(hasher, input_ids, forward_batch)


def tpu_dsv41_engram_embed_init(original, embedding, *args, **kwargs):
    """AROUND EngramEmbedding.__init__: build host tables without the CUDA pin."""
    if torch.cuda.is_available():
        return original(embedding, *args, **kwargs)
    from sglang.srt.environ import envs

    if not (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get()
        and envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get() == "shared"
    ):
        raise ValueError(
            "DeepSeek-V4.1 Engram on TPU needs the shared host table: set "
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 and "
            "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared"
        )
    from sglang_tpu.models.deepseek_v41.engram_host import host_table_without_cuda

    with host_table_without_cuda():
        return original(embedding, *args, **kwargs)


def tpu_dsv41_engram_embed_forward(
    original, embedding, indices, forward_batch=None, **kwargs
):
    """AROUND EngramEmbedding.forward: dequantize the host-staged rows."""
    if indices.is_cuda or embedding.host_table is None:
        return original(embedding, indices, forward_batch, **kwargs)
    from sglang_tpu.models.deepseek_v41.engram_host import (
        staged_inputs,
        staged_lookup,
    )

    # The core host-table lookup reads a raw host pointer from the device.
    _, staged = staged_inputs(forward_batch)
    if staged is None:
        raise RuntimeError(
            "DeepSeek-V4.1 Engram on TPU reads rows staged by the V4 compile "
            "policy; this forward did not stage them"
        )
    return staged_lookup(embedding, indices, staged)


def tpu_dsv41_dspark_attention(original, attn, positions, hidden_states, forward_batch):
    """AROUND DSparkAttention.forward: the draft on the CSA2 backend."""
    from sglang_tpu.models.deepseek_v41.dspark import (
        dspark_attention_forward,
        serves_dspark_draft,
    )

    if hidden_states.is_cuda or not serves_dspark_draft():
        return original(attn, positions, hidden_states, forward_batch)
    return dspark_attention_forward(attn, positions, hidden_states, forward_batch)


def tpu_dsv41_dspark_forward(original, model, input_ids, positions, *args, **kwargs):
    """AROUND DeepseekV4ForCausalLMDSpark.forward: add the greedy drafts."""
    from sglang_tpu.models.deepseek_v41.dspark import (
        dspark_forward,
        serves_dspark_draft,
    )

    if input_ids.is_cuda or not serves_dspark_draft():
        return original(model, input_ids, positions, *args, **kwargs)
    return dspark_forward(original, model, input_ids, positions, *args, **kwargs)


def tpu_dsv41_dspark_write_target_hidden_kv(original, model, **kwargs):
    """AROUND DeepseekV4ForCausalLMDSpark.write_target_hidden_kv: CSA2 SWA pages."""
    from sglang_tpu.models.deepseek_v41.dspark import write_target_hidden_kv

    if kwargs["main_hidden"].is_cuda:
        return original(model, **kwargs)
    return write_target_hidden_kv(model, **kwargs)


def tpu_dsv41_dspark_compute_base_logits(original, model, x):
    """AROUND DeepseekV4ForCausalLMDSpark.compute_base_logits: no gather of an
    unsharded head."""
    from sglang_tpu.models.deepseek_v41.dspark import tpu_compute_base_logits

    if x.is_cuda or model.lm_head is None or model.lm_head.tp_size != 1:
        return original(model, x)
    return tpu_compute_base_logits(model, x)


def tpu_dsv41_vocab_parallel_load_weights(original, model, weights, *args, **kwargs):
    """AROUND DeepseekV4ForCausalLM.load_weights: split a V4.1 target's LM head."""
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.deepseek_v41.vocab_parallel_head import (
        VocabParallelLMHead,
        shard_lm_head,
    )

    if isinstance(model.lm_head, VocabParallelLMHead):
        raise RuntimeError(
            "Cannot reload weights: the LM head is vocab-parallel and holds only "
            "this rank's vocab rows"
        )
    result = original(model, weights, *args, **kwargs)
    is_nextn = kwargs.get("is_nextn", args[0] if args else False)
    if (
        not torch.cuda.is_available()
        and not is_nextn
        and model.config.model_type == "deepseek_v41"
        and tpu_envs.SGLANG_TPU_ENABLE_DSV41_VOCAB_PARALLEL_HEAD.get()
    ):
        shard_lm_head(model)
    return result


def tpu_dsv41_vocab_parallel_lm_head(
    original, processor, hidden_states, lm_head, embedding_bias=None, *, out=None
):
    """AROUND LogitsProcessor._compute_lm_head_rows: score a split head.

    ``out`` is core's optional logits buffer; the split head returns its own tensor.
    """
    from sglang_tpu.models.deepseek_v41.vocab_parallel_head import (
        VocabParallelLMHead,
        vocab_parallel_logits,
    )

    if not isinstance(lm_head, VocabParallelLMHead):
        return original(processor, hidden_states, lm_head, embedding_bias, out=out)
    assert embedding_bias is None, "the vocab-parallel head has no bias"
    return vocab_parallel_logits(lm_head, hidden_states)


def tpu_dsv41_carry_mirrored_padding(original, forward_batch, *args, **kwargs):
    """AROUND ForwardBatch._pad_inputs_to_size: keep host_input_mirror records valid."""
    from sglang_tpu.compilation import host_input_mirror

    before = {
        name: getattr(forward_batch, name) for name in host_input_mirror.MIRRORED_FIELDS
    }
    result = original(forward_batch, *args, **kwargs)
    for name, old in before.items():
        host_input_mirror.carry_padding(forward_batch, name, old)
    return result


def tpu_dsv41_carry_mirror_to_loaded_batch(
    original, runner, forward_batch, *args, **kwargs
):
    """AROUND EagerRunner.load_batch: keep host_input_mirror records on the copy."""
    from sglang_tpu.compilation import host_input_mirror

    loaded = original(runner, forward_batch, *args, **kwargs)
    host_input_mirror.carry_batch(forward_batch, loaded)
    return loaded


def tpu_dsv41_flush_engram_at_length_read(original, step_rows, *args, **kwargs):
    """AROUND StepHostRows._build: flush the Engram commits after the length read."""
    from sglang_tpu.models.deepseek_v41 import engram_host

    # Idle DP ranks pass host lengths and read nothing; a flush there would add a wait.
    synced = step_rows._seq_lens_cpu is None
    rows = original(step_rows, *args, **kwargs)
    if synced and rows is not None:
        engram_host.flush_all_engram_commits()
    return rows


def tpu_dsv41_defer_engram_commit(
    original, manager, *, verify_ids_2d, req_pool_indices, commit_lens
):
    """AROUND NgramEmbeddingManager.update_after_verify: queue the host commit."""
    from sglang_tpu.models.deepseek_v41 import engram_host

    hasher = manager.engram_hasher
    if hasher is None or verify_ids_2d.is_cuda:
        return original(
            manager,
            verify_ids_2d=verify_ids_2d,
            req_pool_indices=req_pool_indices,
            commit_lens=commit_lens,
        )
    engram_host.defer_engram_commit(
        hasher,
        verify_ids_2d=verify_ids_2d,
        req_pool_indices=req_pool_indices,
        commit_lens=commit_lens,
    )


def tpu_dsv41_flush_engram_before_prebuilt(original, manager, batch):
    """AROUND NgramEmbeddingManager._prepare_engram_history: a PD prebuilt batch
    writes history rows directly, so queued verify commits land first."""
    from sglang_tpu.models.deepseek_v41 import engram_host

    hasher = manager.engram_hasher
    if batch.forward_mode.is_prebuilt() and engram_host.has_pending_engram_commits(
        hasher
    ):
        engram_host.flush_engram_commits(hasher)
    return original(manager, batch)


def prepare_tpu_dsv41_fused_ep(result, model, target_device):
    """AFTER DefaultModelLoader.postprocess_weights: convert the eligible V4.1
    MoE layers' experts to the fused EP operands and set their
    ``dp_local_forward``."""
    if target_device.type == "tpu":
        from sglang_tpu.models.deepseek_v41.fused_ep_op import (
            prepare_model_for_dsv41_fused_ep,
        )

        prepare_model_for_dsv41_fused_ep(model)
    return result


def prepare_tpu_dsv41_wo_a_fp8(result, model, target_device):
    """AFTER DefaultModelLoader.postprocess_weights: quantize V4.1 wo_a to per-channel FP8."""
    if (
        target_device.type == "tpu"
        and getattr(getattr(model, "config", None), "model_type", None) == "deepseek_v41"
    ):
        from sglang_tpu.models.deepseek_v41.ops.wo_a_fp8 import (
            prepare_model_for_dsv41_wo_a_fp8,
        )

        prepare_model_for_dsv41_wo_a_fp8(model)
    return result


def tpu_dsv41_project_wo_a(original, layer, o, *args, **kwargs):
    """AROUND MQALayer._project_wo_a: run per-channel FP8 grouped einsum on TPU."""
    wo_a = getattr(layer, "wo_a", None)
    scale = getattr(wo_a, "_tpu_wo_a_fp8_scale", None) if wo_a is not None else None
    if scale is None:
        return original(layer, o, *args, **kwargs)
    from sglang_tpu.models.deepseek_v41.ops.wo_a_fp8 import wo_a_fp8_matmul

    # On TPU CSA2, inverse RoPE is already applied by CSA2 (apply_mqa_inv_rope is identity).
    o_grouped = o.view(o.shape[0], layer.n_local_groups, -1)
    return wo_a_fp8_matmul(o_grouped, wo_a.weight, scale)


def prepare_tpu_dsv41_dummy_model(result, loader, *, model_config, device_config):
    """AFTER DummyModelLoader.load_model: apply V4.1 post-load transforms
    (vocab-parallel LM head sharding, FP8 wo_a, and fused EP MoE preparation)
    that DefaultModelLoader runs through load_weights / postprocess_weights."""
    from sglang_tpu.envs import envs as tpu_envs

    if (
        torch.cuda.is_available()
        or getattr(getattr(result, "config", None), "model_type", None) != "deepseek_v41"
    ):
        return result
    is_draft = result.__class__.__name__.endswith("DSpark") or result.__class__.__name__.endswith("NextN")
    if (
        not is_draft
        and tpu_envs.SGLANG_TPU_ENABLE_DSV41_VOCAB_PARALLEL_HEAD.get()
        and hasattr(result, "lm_head")
    ):
        from sglang_tpu.models.deepseek_v41.vocab_parallel_head import shard_lm_head

        shard_lm_head(result)
    from sglang_tpu.models.deepseek_v41.ops.wo_a_fp8 import (
        prepare_model_for_dsv41_wo_a_fp8,
    )

    prepare_model_for_dsv41_wo_a_fp8(result)
    if tpu_envs.SGLANG_TPU_DSV41_FUSED_EP.get():
        from sglang_tpu.models.deepseek_v41.fused_ep_op import (
            prepare_model_for_dsv41_fused_ep,
        )

        prepare_model_for_dsv41_fused_ep(result)
    return result


def fused_ep_hook_specs():
    """The fused EP hook, only when SGLANG_TPU_DSV41_FUSED_EP=1. Otherwise nothing
    is registered and the default path is untouched."""
    from sglang.srt.plugins.hook_registry import HookType
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_DSV41_FUSED_EP.get():
        return []
    # Never raise here: load_plugins logs a plugin import error and drops all TPU
    # hooks. check_dsv41_fused_ep_env rejects a missing SGLANG_SHARED_EXPERT_TP1.
    return [
        (
            "sglang.srt.model_loader.loader.DefaultModelLoader.postprocess_weights",
            prepare_tpu_dsv41_fused_ep,
            HookType.AFTER,
        ),
    ]


def engram_flush_hook_specs():
    """The early Engram commit flush, only with SGLANG_TPU_DSV41_FLUSH_ENGRAM_AT_LENGTH_READ=1."""
    from sglang.srt.plugins.hook_registry import HookType
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_DSV41_FLUSH_ENGRAM_AT_LENGTH_READ.get():
        return []
    return [
        (
            "sglang_tpu.experimental.speculative.host_verify_rows.StepHostRows._build",
            tpu_dsv41_flush_engram_at_length_read,
            HookType.AROUND,
        ),
    ]


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    # Declares the ForwardBatch fields the Engram hooks read.
    from sglang_tpu.models.deepseek_v41 import engram_host  # noqa: F401
    from sglang_tpu.models.deepseek_v41 import decoder_replay_sync

    kvc = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    cfg = "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator"
    dspark = "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark"
    ngram = (
        "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager"
        ".NgramEmbeddingManager"
    )
    return [
        (f"{kvc}._build_dsv4_kv_pool", build_tpu_dsv41_kv_pool, HookType.AROUND),
        (
            f"{cfg}._get_bytes_per_swa_token",
            get_tpu_dsv41_bytes_per_swa_token,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_bytes_per_full_token",
            get_tpu_dsv41_bytes_per_full_token,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_swa_fixed_bytes",
            get_tpu_dsv41_swa_fixed_bytes,
            HookType.AROUND,
        ),
        (f"{cfg}.__init__", init_tpu_dsv41_pool_configurator, HookType.AROUND),
        (f"{kvc}._derive_pool_sizes", derive_tpu_dsv41_pool_sizes, HookType.AROUND),
        (
            f"{kvc}._build_token_to_kv_pool_allocator",
            build_tpu_dsv41_allocator,
            HookType.AROUND,
        ),
        (
            "sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v41_features",
            allow_tpu_dsv41_encoder_replay,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.encoder_swa_replay._fold_batch",
            fold_tpu_dsv41_encoder_replay,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.encoder_swa_replay.drop_folded_rows",
            drop_tpu_dsv41_folded_rows,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.forward_batch_info.ForwardBatch."
            "_pad_tensor_to_size",
            skip_tpu_noop_pad,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_loader.loader.get_model_loader",
            select_tpu_dsv41_loader,
            HookType.AFTER,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding._load_rows",
            load_tpu_dsv41_engram_rows,
            HookType.AROUND,
        ),
        (
            "sglang.srt.runtime_context.PlatformContext.__getattr__",
            trace_tpu_platform_fact,
            HookType.AROUND,
        ),
        (
            "sglang.srt.environ.EnvField._resolve_default",
            trace_tpu_env_default,
            HookType.AROUND,
        ),
        (
            "sglang.kernels.ops.layernorm.mhc.mhc_combine",
            tpu_dsv41_hc_combine,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.engram_gate",
            tpu_dsv41_engram_gate,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation",
            tpu_dsv41_engram_worker_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramHasher.forward",
            tpu_dsv41_hasher_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding.__init__",
            tpu_dsv41_engram_embed_init,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding.forward",
            tpu_dsv41_engram_embed_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4_dspark.DSparkAttention.forward",
            tpu_dsv41_dspark_attention,
            HookType.AROUND,
        ),
        (
            f"{dspark}.forward",
            tpu_dsv41_dspark_forward,
            HookType.AROUND,
        ),
        (
            f"{dspark}.write_target_hidden_kv",
            tpu_dsv41_dspark_write_target_hidden_kv,
            HookType.AROUND,
        ),
        (
            f"{dspark}.compute_base_logits",
            tpu_dsv41_dspark_compute_base_logits,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights",
            tpu_dsv41_vocab_parallel_load_weights,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head_rows",
            tpu_dsv41_vocab_parallel_lm_head,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.forward_batch_info.ForwardBatch._pad_inputs_to_size",
            tpu_dsv41_carry_mirrored_padding,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.runner.eager_runner.EagerRunner.load_batch",
            tpu_dsv41_carry_mirror_to_loaded_batch,
            HookType.AROUND,
        ),
        (
            f"{ngram}.update_after_verify",
            tpu_dsv41_defer_engram_commit,
            HookType.AROUND,
        ),
        (
            f"{ngram}._prepare_engram_history",
            tpu_dsv41_flush_engram_before_prebuilt,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4.MQALayer._project_wo_a",
            tpu_dsv41_project_wo_a,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_loader.loader.DefaultModelLoader.postprocess_weights",
            prepare_tpu_dsv41_wo_a_fp8,
            HookType.AFTER,
        ),
        (
            "sglang.srt.model_loader.loader.DummyModelLoader.load_model",
            prepare_tpu_dsv41_dummy_model,
            HookType.AFTER,
        ),
        *decoder_replay_sync.hook_specs(),
        *engram_flush_hook_specs(),
    ] + fused_ep_hook_specs()
