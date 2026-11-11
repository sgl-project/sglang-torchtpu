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

tpu_dsv41_engram_hash_ids
  Why: compute_engram_hash_ids multiplies by 63-bit multipliers in int64; TPU
       lowering rejects 64-bit integer vectors ("Unsupported conversion from
       vmreg/vreg to U64") when the decoder graph is compiled.
  How: AROUND compute_engram_hash_ids keeps the original on CUDA; elsewhere
       runs ops/engram.py, the uint32-pair hash adapted from the vllm-torchtpu
       epic, on int32 tables staged at load (bit-exact, int32 row ids).
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: TPU lowers 64-bit integer arithmetic.

tpu_dsv41_int32_input_ids
  Why: with a vision tower, DeepseekV4ForCausalLM.forward compares int64
       input_ids against MM_PAD_SHIFT_VALUE and the model compares them with
       image_token_id inside the fused Engram torch.where; TPU lowering fails
       on the 64-bit compare ("Unsupported conversion from vmreg/vreg to U64").
  How: AROUND DeepseekV4ForCausalLM.forward casts V4.1 input_ids to int32 on
       non-CUDA devices; token ids and multimodal pad ids fit in int32.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: TPU lowers 64-bit integer compares.

tpu_dsv41_torch_hash_ids
  Why: EngramHasher._torch_hash_ids (the non-CUDA hash path) widens ids,
       history and positions to int64 before building the predecessor table,
       so 64-bit compares and gathers reach TPU lowering.
  How: AROUND _torch_hash_ids builds the same table in int32 on non-CUDA
       devices and calls compute_engram_hash_ids (itself hooked to the 32-bit
       op); token ids, positions and MM_PAD_SHIFT_VALUE fit in int32.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: TPU lowers 64-bit integer arithmetic.

trace_tpu_env_default
  Why: EnvField._resolve_default calls callable(self.default) for lazily
       computed defaults (e.g. SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)),
       which Dynamo cannot trace; the V4.1 MoE DP path reads it in the forward.
  How: AROUND EnvField._resolve_default resolves the default through a
       torch._dynamo.assume_constant_result helper keyed by the env name;
       defaults depend only on the platform, fixed before any model is traced.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: callable env defaults are resolved at import or are traceable.
"""

from __future__ import annotations

import torch


def _pool_hooks():
    from sglang_tpu.models.deepseek_v41 import memory_pool

    return (
        memory_pool.build_tpu_dsv41_kv_pool,
        memory_pool.get_tpu_dsv41_bytes_per_swa_token,
        memory_pool.get_tpu_dsv41_bytes_per_full_token,
    )


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


def tpu_dsv41_engram_hash_ids(
    original, tokens, blocked, pad_id, token_map, multipliers, primes, offsets
):
    if tokens.is_cuda:
        return original(
            tokens, blocked, pad_id, token_map, multipliers, primes, offsets
        )
    from sglang_tpu.models.deepseek_v41.ops.engram import engram_hash_ids

    return engram_hash_ids(tokens, blocked, pad_id)


def tpu_dsv41_torch_hash_ids(
    original, hasher, input_ids, positions, kmode, history, num_real, block, row, starts
):
    """EngramHasher._torch_hash_ids with an int32 predecessor table on TPU.

    The original widens ids, history and positions to int64 before hashing;
    token ids, positions and the multimodal pad shift all fit in int32.
    """
    if input_ids.is_cuda:
        return original(
            hasher, input_ids, positions, kmode, history, num_real, block, row, starts
        )
    from sglang.kernels.ops.embeddings.engram_hash import MODE_DECODE, MODE_VERIFY
    from sglang.srt.layers.engram import compute_engram_hash_ids
    from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE

    i32 = torch.int32
    n = hasher.max_ngram_size
    num_tokens = input_ids.shape[0]
    device = input_ids.device
    input_ids = input_ids.to(i32)
    shifts = torch.arange(n, device=device, dtype=i32)
    if kmode == MODE_DECODE:
        tokens = torch.cat([input_ids.unsqueeze(-1), history.to(i32).flip(-1)], dim=-1)
    else:
        t = torch.arange(num_real, device=device, dtype=i32)
        if kmode == MODE_VERIFY:
            row = t // block
            offset = t - row * block
        else:
            row = row.to(i32)
            offset = t - starts.to(i32)[row]
        from_batch = shifts.unsqueeze(0) <= offset.unsqueeze(-1)
        in_batch = input_ids[(t.unsqueeze(-1) - shifts).clamp_min(0)]
        hist_col = (n - 2 - (shifts.unsqueeze(0) - offset.unsqueeze(-1) - 1)).clamp(
            0, n - 2
        )
        from_hist = history.to(i32)[row].gather(1, hist_col.to(torch.int64))
        tokens = torch.where(from_batch, in_batch, from_hist)
    if num_real < num_tokens:
        tokens = torch.cat([tokens, tokens.new_zeros(num_tokens - num_real, n)])
    blocked = positions.to(i32).unsqueeze(-1) < shifts
    if num_real < num_tokens:
        blocked[num_real:] = True
    if hasher.image_token_id is not None:
        tokens = tokens.masked_fill(tokens >= MM_PAD_SHIFT_VALUE, hasher.image_token_id)
        blocked = (
            (blocked | (tokens == hasher.image_token_id))
            .to(i32)
            .cummax(-1)
            .values.bool()
        )
    hash_ids = compute_engram_hash_ids(
        tokens,
        blocked,
        hasher.pad_id,
        hasher.token_map,
        hasher.multipliers,
        hasher.primes,
        hasher.offsets,
    )
    return hash_ids, tokens


def tpu_dsv41_int32_input_ids(original, model, input_ids, *args, **kwargs):
    """AROUND DeepseekV4ForCausalLM.forward: V4.1 token ids fit in int32."""
    if (
        not input_ids.is_cuda
        and input_ids.dtype == torch.int64
        and model.config.model_type == "deepseek_v41"
    ):
        input_ids = input_ids.to(torch.int32)
    return original(model, input_ids, *args, **kwargs)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    kvc = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    cfg = "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator"
    build_pool, swa_bytes, full_bytes = _pool_hooks()
    return [
        (f"{kvc}._build_dsv4_kv_pool", build_pool, HookType.AROUND),
        (f"{cfg}._get_bytes_per_swa_token", swa_bytes, HookType.AROUND),
        (f"{cfg}._get_bytes_per_full_token", full_bytes, HookType.AROUND),
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
            "sglang.srt.layers.engram.compute_engram_hash_ids",
            tpu_dsv41_engram_hash_ids,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.engram_gate",
            tpu_dsv41_engram_gate,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.forward",
            tpu_dsv41_int32_input_ids,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramHasher._torch_hash_ids",
            tpu_dsv41_torch_hash_ids,
            HookType.AROUND,
        ),
    ]
