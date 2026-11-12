# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 TPU runtime hooks (compress ratios 0/1/2).

Every hook here passes V4 (no ratio-1/2 layers, model_type != deepseek_v41)
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
  Why: RunaiModelStreamerLoader reads every routed expert on every rank; under
       EP8 each rank keeps 1/8 of them, so startup reads ~8x the bytes it needs.
       The loader has no model-owned tensor selection seam.
  How: AFTER get_model_loader swaps in DSV41RunaiModelStreamerLoader for the
       V4.1 target only; it inherits construction, loading and postprocessing.
  Upstream PR or issue: not filed (forked from sgl-project/sglang-torchtpu#1744).
  Remove when: the core loader accepts a model-owned checkpoint tensor filter.

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
  How: AROUND sglang.kernels.ops.layernorm.mhc.hc_combine computes the same
       fp32 weighted sum of the hc_mult copies in torch for non-CUDA tensors.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: hc_combine dispatches on device like the other mHC kernels.

tpu_dsv41_engram_gate
  Why: the torch fallback of sglang.srt.layers.engram.engram_gate uses
       torch.copysign, which TorchTPU does not implement (aten::copysign.out).
  How: AROUND engram_gate keeps the original on CUDA; elsewhere computes the
       same gate with torch.where(dot >= 0, |dot|', -|dot|') for the sign.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: TorchTPU implements aten::copysign or engram_gate avoids it.

tpu_dsv41_int32_graph_inputs
  Why: the compiled V4.1 forward takes input_ids, positions and
       req_pool_indices as int64 graph inputs; the image-token torch.where,
       the Engram hasher and the ratio-2 pair state split them into 64-bit
       halves, which TPU lowering rejects ("Unsupported conversion from
       vmreg/vreg to U64"). A cast inside the forward still leaves the
       int64 parameter in the graph.
  How: AROUND CompiledForwardShim._forward casts them to int32 before the
       compiled call for V4.1 on non-CUDA devices and restores the batch
       fields afterwards; token ids, positions and pool rows fit in int32.
       Engram rows are hashed and gathered here on the host (engram_host), so
       the graph takes staged rows instead of hashing and looking up tables.
       With attention DP the token embedding is looked up eagerly here and
       passed in as input_embeds (input_ids still go in for Engram and MoE
       routing), as vllm-torchtpu #1848 does: inside the TP1/DP8 compiled
       graph the unsharded BF16 embedding gather returns wrong rows, while the
       same lookup outside the graph is exact. Warmup takes this path too, so
       precompiled and serving graphs share one signature. The DSpark draft
       (no Engram, no .model) takes its own branch: same int32 casts and
       eager embedding, spread over hc_mult as its forward_embed does.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: the TPU batch builder produces int32 index tensors and the
       in-graph embedding gather is fixed in torch_tpu.

tpu_dsv41_engram_embed_init, tpu_dsv41_engram_embed_forward,
tpu_dsv41_hasher_forward
  Why: TPU serves Engram only from one host table per node shared by the DP
       ranks (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1, layout "shared"); the
       in-graph lookup of HBM tables returned wrong outputs under attention DP
       and its offloaded gather halted a SparseCore. _HostTable pins its
       mapping with cudaHostRegister, and the core lookup reads it from a GPU.
  How: AROUND EngramEmbedding.__init__ rejects any other placement on TPU and
       builds the table with cudaHostRegister stubbed out. The shim hook
       (tpu_dsv41_int32_graph_inputs) hashes on the host and stages raw rows;
       inside the graph EngramHasher.forward returns the staged row ids and
       EngramEmbedding.forward dequantizes the staged rows.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: the device can read host tables, or the in-graph HBM lookup
       is correct under attention DP.

trace_tpu_env_default
  Why: EnvField._resolve_default calls callable(self.default) for lazily
       computed defaults (e.g. SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)),
       which Dynamo cannot trace; the V4.1 MoE DP path reads it in the forward.
  How: AROUND EnvField._resolve_default resolves the default through a
       torch._dynamo.assume_constant_result helper keyed by the env name;
       defaults depend only on the platform, fixed before any model is traced.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: callable env defaults are resolved at import or are traceable.

tpu_dsv41_dspark_attention, tpu_dsv41_dspark_forward
  Why: DSparkAttention.forward writes its block KV with the CUDA FlashMLA
       writer and inverse-rotates its output, which CSA2 already does; the
       draft forward returns no next_draft_tokens, which TpuBlockSpecWorker
       reads for greedy drafts.
  How: AROUND both for the V4.1 draft off CUDA (dspark.py): attention goes
       through the backend's prepare_mqa_qkv and non-causal forward, then
       wo_a / wo_b; the forward appends the greedy Markov chain in-graph, as
       vllm-torchtpu models/vllm/deepseek_v41/v2/dspark.py does.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR2).
  Remove when: DSparkAttention reads its KV write and inverse RoPE from the
       attention backend and the V4 draft returns draft tokens.

tpu_dsv41_create_draft_worker
  Why: TpuBlockSpecWorker's DSPARK drafter accepts only the dense
       DSparkDraftModel; the V4.1 draft is the MoE DeepseekV4ForCausalLMDSpark.
  How: AROUND block_spec_worker._create_draft_worker returns
       TpuDsv41DsparkDraftWorker (dspark_drafter.py, imported lazily) for a
       V4.1 target; other targets keep the built-in drafter.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR2).
  Remove when: the block worker selects drafters by draft model class.

tpu_dsv41_commit_engram_history
  Why: the target's Engram history must advance past the tokens each verify
       commits; core does it in DSparkWorkerV2 (update_after_verify), which
       TpuBlockSpecWorker replaces, and a verify forward commits nothing.
  How: AROUND TpuBlockSpecWorker.commit appends the anchor and correct drafts
       to each live request's history (functional update + copy_).
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR2).
  Remove when: TpuSpecWorker commits the target's Engram history itself.
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


def select_tpu_dsv41_loader(result, load_config, model_config=None):
    from sglang.srt.model_loader.loader import RunaiModelStreamerLoader

    config = getattr(model_config, "hf_config", None)
    if (
        type(result) is not RunaiModelStreamerLoader
        or getattr(config, "model_type", None) != "deepseek_v41"
        or getattr(model_config, "is_draft_model", False)
        or load_config.draft_model_idx is not None
    ):
        return result
    from sglang_tpu.models.deepseek_v41.loader import DSV41RunaiModelStreamerLoader

    return DSV41RunaiModelStreamerLoader(load_config, model_config)


def load_tpu_dsv41_engram_rows(original, embedding, param, loaded_weight):
    from sglang_tpu.models.deepseek_v41.ep_local_expert_reader import load_local_rows

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


def tpu_dsv41_engram_gate(original, x, kv, q_weight, k_weight, eps, clamp_value):
    """engram_gate without torch.copysign (no TorchTPU kernel)."""
    if x.is_cuda:
        return original(x, kv, q_weight, k_weight, eps, clamp_value)
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(clamp_value).sqrt()
    gate = torch.sigmoid(torch.where(dot >= 0, magnitude, -magnitude))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def tpu_dsv41_hasher_forward(original, hasher, input_ids, forward_batch):
    """AROUND EngramHasher.forward: in the TPU graph, the host-staged row ids."""
    if forward_batch.tpu_engram_local_ids is not None:
        return forward_batch.tpu_engram_local_ids
    return original(hasher, input_ids, forward_batch)


_INT32_BATCH_FIELDS = ("input_ids", "positions", "req_pool_indices")


def _int32(t):
    if isinstance(t, torch.Tensor) and t.dtype == torch.int64 and not t.is_cuda:
        return t.to(torch.int32)
    return t


def tpu_dsv41_int32_graph_inputs(
    original, shim, fn, input_ids, positions, forward_batch, *args, **kwargs
):
    """AROUND CompiledForwardShim._forward: V4.1 index tensors enter as int32."""
    config = shim.policy.model_runner.model_config.hf_config
    if config.model_type != "deepseek_v41":
        return original(shim, fn, input_ids, positions, forward_batch, *args, **kwargs)
    model = shim.policy.model_runner.model
    from sglang_tpu.models.deepseek_v41.dspark import is_dsv41_dspark_draft

    if is_dsv41_dspark_draft(model):
        return _draft_int32_graph_inputs(
            original, shim, fn, input_ids, positions, forward_batch, *args, **kwargs
        )
    names = _INT32_BATCH_FIELDS
    hasher = model.model.engram_hasher
    host_tables = ()
    if hasher is not None and not torch.compiler.is_compiling():
        from sglang_tpu.models.deepseek_v41 import engram_host

        host_tables = engram_host.shared_host_tables(model)
    if host_tables:
        names = (*names, "tpu_engram_local_ids", "tpu_engram_staged")
    from sglang.srt.runtime_context import get_parallel

    embed_outside = (
        forward_batch.input_embeds is None
        and isinstance(input_ids, torch.Tensor)
        and not input_ids.is_cuda
        and not torch.compiler.is_compiling()
        and get_parallel().attn_dp_size > 1
    )
    if embed_outside:
        names = (*names, "input_embeds")
    saved = {name: getattr(forward_batch, name) for name in names}
    if host_tables:
        _stage_host_engram(hasher, host_tables, input_ids, forward_batch)
    if embed_outside:
        embed_tokens = shim.policy.model_runner.model.get_input_embeddings()
        with torch.no_grad():
            forward_batch.input_embeds = embed_tokens(input_ids)
    for name in _INT32_BATCH_FIELDS:
        setattr(forward_batch, name, _int32(saved[name]))
    try:
        return original(
            shim,
            fn,
            _int32(input_ids),
            _int32(positions),
            forward_batch,
            *args,
            **kwargs,
        )
    finally:
        for name, value in saved.items():
            setattr(forward_batch, name, value)


def _stage_host_engram(hasher, tables, input_ids, forward_batch) -> None:
    """Hash this rank's tokens and gather their Engram rows on the host."""
    from sglang_tpu.models.deepseek_v41 import engram_host

    engram_host.move_hasher_to_host(hasher)
    with torch.no_grad():
        hash_ids = engram_host.host_hash_ids(hasher, input_ids, forward_batch)
    num_tokens, num_layers, num_cols = hash_ids.shape
    device = input_ids.device
    forward_batch.tpu_engram_staged = engram_host.gather_staged_rows(
        hash_ids, tables, device
    )
    forward_batch.tpu_engram_local_ids = engram_host.local_row_ids(
        num_tokens, num_layers, num_cols, device
    )


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
    from sglang_tpu.models.deepseek_v41.engram_host import without_cuda_host_register

    with without_cuda_host_register():
        return original(embedding, *args, **kwargs)


def tpu_dsv41_engram_embed_forward(
    original, embedding, indices, forward_batch=None, **kwargs
):
    """AROUND EngramEmbedding.forward: dequantize the host-staged rows."""
    if indices.is_cuda or embedding.host_table is None:
        return original(embedding, indices, forward_batch, **kwargs)
    from sglang_tpu.models.deepseek_v41.engram_host import staged_lookup

    # The core host-table lookup reads a raw host pointer from the device.
    if forward_batch is None or forward_batch.tpu_engram_staged is None:
        raise RuntimeError(
            "DeepSeek-V4.1 Engram on TPU reads rows staged by the compiled-forward "
            "shim; this forward did not stage them"
        )
    return staged_lookup(embedding, indices, forward_batch)


def _draft_int32_graph_inputs(
    original, shim, fn, input_ids, positions, forward_batch, *args, **kwargs
):
    """The DSpark draft's int32 inputs and eager embedding; it has no Engram.

    The draft embeds with the target's table, so it hits the same in-graph
    gather bug; its forward takes input_embeds already spread over hc_mult.
    """
    model = shim.policy.model_runner.model
    names = (*_INT32_BATCH_FIELDS, "input_embeds")
    saved = {name: getattr(forward_batch, name) for name in names}
    if (
        forward_batch.input_embeds is None
        and isinstance(input_ids, torch.Tensor)
        and not input_ids.is_cuda
        and not torch.compiler.is_compiling()
        and input_ids.numel()
    ):
        with torch.no_grad():
            embeds = model.embed_tokens(input_ids)
        forward_batch.input_embeds = embeds.unsqueeze(1).repeat(1, model.hc_mult, 1)
    for name in _INT32_BATCH_FIELDS:
        setattr(forward_batch, name, _int32(saved[name]))
    try:
        return original(
            shim,
            fn,
            _int32(input_ids),
            _int32(positions),
            forward_batch,
            *args,
            **kwargs,
        )
    finally:
        for name, value in saved.items():
            setattr(forward_batch, name, value)


def tpu_dsv41_dspark_attention(original, attn, positions, hidden_states, forward_batch):
    """AROUND DSparkAttention.forward: the V4.1 draft on the CSA2 backend."""
    if not attn.is_dsv41 or hidden_states.is_cuda:
        return original(attn, positions, hidden_states, forward_batch)
    from sglang_tpu.models.deepseek_v41.dspark import dspark_attention_forward

    return dspark_attention_forward(attn, positions, hidden_states, forward_batch)


def tpu_dsv41_dspark_forward(original, model, input_ids, positions, *args, **kwargs):
    """AROUND DeepseekV4ForCausalLMDSpark.forward: add the greedy drafts."""
    from sglang_tpu.models.deepseek_v41.dspark import (
        dspark_forward,
        is_dsv41_dspark_draft,
    )

    if input_ids.is_cuda or not is_dsv41_dspark_draft(model):
        return original(model, input_ids, positions, *args, **kwargs)
    return dspark_forward(original, model, input_ids, positions, *args, **kwargs)


def tpu_dsv41_commit_engram_history(
    original, worker, batch, state, rows, verify_input, logits_output, accept
):
    """AROUND TpuBlockSpecWorker.commit: advance the target's Engram history."""
    result = original(worker, batch, state, rows, verify_input, logits_output, accept)
    model = worker.target_worker.model_runner.model
    inner = getattr(model, "model", None)
    hasher = getattr(inner, "engram_hasher", None)
    if hasher is None or accept.accept_lens.is_cuda:
        return result
    from sglang_tpu.models.deepseek_v41.dspark import commit_engram_history

    commit_engram_history(
        hasher,
        bonus_tokens=state.bonus_tokens,
        accept_tokens=accept.accept_tokens,
        accept_lens=accept.accept_lens,
        req_pool_indices=state.req_pool_indices,
    )
    return result


def tpu_dsv41_create_draft_worker(original, *, algorithm, target_runner):
    """AROUND block_spec_worker._create_draft_worker: the V4.1 DSpark drafter."""
    if (
        algorithm.is_dspark()
        and target_runner.model_config.hf_config.model_type == "deepseek_v41"
    ):
        from sglang_tpu.models.deepseek_v41.dspark_drafter import (
            TpuDsv41DsparkDraftWorker,
        )

        return TpuDsv41DsparkDraftWorker(target_runner)
    return original(algorithm=algorithm, target_runner=target_runner)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    # Declares the ForwardBatch fields the Engram hooks read.
    from sglang_tpu.models.deepseek_v41 import engram_host  # noqa: F401

    kvc = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    cfg = "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator"
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
            "sglang.kernels.ops.layernorm.mhc.hc_combine",
            tpu_dsv41_hc_combine,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.engram_gate",
            tpu_dsv41_engram_gate,
            HookType.AROUND,
        ),
        (
            "sglang_tpu.compilation.compiled_forward_shim.CompiledForwardShim._forward",
            tpu_dsv41_int32_graph_inputs,
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
            "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.forward",
            tpu_dsv41_dspark_forward,
            HookType.AROUND,
        ),
        (
            "sglang_tpu.experimental.speculative.block_spec_worker.TpuBlockSpecWorker.commit",
            tpu_dsv41_commit_engram_history,
            HookType.AROUND,
        ),
        (
            "sglang_tpu.experimental.speculative.block_spec_worker._create_draft_worker",
            tpu_dsv41_create_draft_worker,
            HookType.AROUND,
        ),
    ]
