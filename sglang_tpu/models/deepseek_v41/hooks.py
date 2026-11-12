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
       An extend without scheduler-supplied engram_history gets the rows the
       hasher would gather, so the graph never guards on engram_history being None.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: the TPU batch builder produces int32 index tensors.

tpu_dsv41_torch_hash_ids
  Why: EngramHasher._torch_hash_ids (the non-CUDA hash path) widens ids,
       history and positions to int64 before building the predecessor table,
       so 64-bit compares and gathers reach TPU lowering.
  How: AROUND _torch_hash_ids builds the same table in int32 on non-CUDA
       devices and calls compute_engram_hash_ids (itself hooked to the 32-bit
       op); token ids, positions and MM_PAD_SHIFT_VALUE fit in int32.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: TPU lowers 64-bit integer arithmetic.

tpu_dsv41_engram_owned_rows
  Why: EngramEmbedding._owned_rows dequantizes with scale.float() on the
       float8_e8m0fnu scales; TPU lowering of that conversion fails in the
       Engram lookup fusion ("Unsupported conversion from vmreg/vreg to U64").
  How: AROUND _owned_rows for the device-table, non-CUDA path gathers the raw
       bytes and builds the power of two as (byte << 23) bitcast to fp32, as
       the vllm-torchtpu epic engram_hbm.dequant and #1743 do.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: TPU lowers float8_e8m0fnu to float32.

tpu_dsv41_hasher_forward
  Why: the extend path of EngramHasher.forward builds per-token request rows
       with repeat_interleave over int64 lengths, which lowers to a while loop
       on 64-bit counters that TPU lowering rejects.
  How: AROUND EngramHasher.forward on non-CUDA extend computes the rows in
       int32 as a compare-sum against request ends and hashes every padded row
       (no guard on the host lengths), then commits history for real rows only.
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
        # Cumulative OR over the n columns, unrolled: cummax lowers to a loop
        # with a 64-bit counter.
        stop = blocked | (tokens == hasher.image_token_id)
        cols = [stop[:, 0]]
        for k in range(1, n):
            cols.append(cols[-1] | stop[:, k])
        blocked = torch.stack(cols, dim=-1)
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


def tpu_dsv41_hasher_forward(original, hasher, input_ids, forward_batch):
    """EngramHasher.forward for TPU extend with int32 request rows.

    The original builds token rows with repeat_interleave over int64 lengths,
    which lowers to a loop of 64-bit counters; rows here are a compare-sum
    against each request's end (requests are contiguous in extend_start_loc).
    """
    num_tokens = input_ids.shape[0]
    if (
        input_ids.is_cuda
        or num_tokens == 0
        or not forward_batch.forward_mode.is_extend()
    ):
        return original(hasher, input_ids, forward_batch)
    from sglang.kernels.ops.embeddings.engram_hash import MODE_EXTEND

    i32 = torch.int32
    n = hasher.max_ngram_size
    req_slots = forward_batch.req_pool_indices
    bs = req_slots.shape[0]
    lens = forward_batch.extend_seq_lens.to(i32)
    starts = forward_batch.extend_start_loc.to(i32)
    # Hash every padded row: rows are independent and only real rows commit,
    # so the graph never specializes on the host lengths (extend_seq_lens_cpu).
    num_real = num_tokens
    ends = starts + lens
    t = torch.arange(num_real, device=input_ids.device, dtype=i32)
    # Bucket-padding requests have zero length (end 0) and must not count.
    passed = (t.unsqueeze(-1) >= ends.unsqueeze(0)) & (lens > 0).unsqueeze(0)
    row = passed.sum(-1).clamp(max=bs - 1).to(i32)
    if forward_batch.engram_history is not None:
        history = forward_batch.engram_history
    else:
        history = hasher.history[req_slots]
    hash_ids, tokens = hasher._torch_hash_ids(
        input_ids,
        forward_batch.positions,
        MODE_EXTEND,
        history,
        num_real,
        1,
        row,
        starts,
    )
    commit_rows = torch.where(lens > 0, req_slots.to(i32), hasher.pad_row)
    commit_last = (ends - 1).clamp(0, num_tokens - 1)
    out_loc = forward_batch.out_cache_loc
    if out_loc is not None:
        commit_rows = torch.where(
            out_loc[commit_last] == 0, hasher.pad_row, commit_rows
        )
    hasher.history[commit_rows] = (
        tokens[commit_last][:, : n - 1].flip(-1).to(hasher.history.dtype)
    )
    return hash_ids


_INT32_BATCH_FIELDS = ("input_ids", "positions", "req_pool_indices")


def _int32(t):
    if isinstance(t, torch.Tensor) and t.dtype == torch.int64 and not t.is_cuda:
        return t.to(torch.int32)
    return t


def _padded_history(hasher, history, req_pool_indices):
    """[bucket bs, n - 1] extend history, so the graph sees one shape and type.

    The scheduler's rows cover the real requests only (or are absent, when the
    hasher reads history[req_pool_indices]); bucket-padding requests extend
    no tokens, so their rows are never committed and their hashes are dropped.
    """
    rows = req_pool_indices.shape[0]
    if history is None:
        return hasher.history[req_pool_indices.to(torch.int64)]
    if history.shape[0] < rows:
        history = torch.cat(
            [history, history.new_zeros(rows - history.shape[0], history.shape[1])]
        )
    return history


def tpu_dsv41_int32_graph_inputs(
    original, shim, fn, input_ids, positions, forward_batch, *args, **kwargs
):
    """AROUND CompiledForwardShim._forward: V4.1 index tensors enter as int32."""
    config = shim.policy.model_runner.model_config.hf_config
    if config.model_type != "deepseek_v41":
        return original(shim, fn, input_ids, positions, forward_batch, *args, **kwargs)
    names = _INT32_BATCH_FIELDS
    hasher = shim.policy.model_runner.model.model.engram_hasher
    pin_history = hasher is not None and forward_batch.forward_mode.is_extend()
    if pin_history:
        names = (*names, "engram_history")
    saved = {name: getattr(forward_batch, name) for name in names}
    for name in _INT32_BATCH_FIELDS:
        setattr(forward_batch, name, _int32(saved[name]))
    if pin_history:
        forward_batch.engram_history = _padded_history(
            hasher, forward_batch.engram_history, forward_batch.req_pool_indices
        )
    from sglang_tpu.models.deepseek_v41 import nan_probe

    if nan_probe.ENABLED:
        nan_probe.before_forward(forward_batch.input_ids.device)
    try:
        out = original(
            shim,
            fn,
            _int32(input_ids),
            _int32(positions),
            forward_batch,
            *args,
            **kwargs,
        )
        if nan_probe.ENABLED:
            nan_probe.after_forward(forward_batch, out)
        return out
    finally:
        for name, value in saved.items():
            setattr(forward_batch, name, value)


def tpu_dsv41_engram_owned_rows(original, embedding, indices):
    """_owned_rows with an integer e8m0 dequant; bytes 0 and 255 aside, exact."""
    if indices.is_cuda or embedding.host_table is not None or embedding.rows == 0:
        return original(embedding, indices)
    local = indices - embedding.row_start
    owned = (local >= 0) & (local < embedding.rows)
    local = local.masked_fill(~owned, 0)
    raw = embedding.weight.view(torch.uint8)[local].view(torch.float8_e4m3fn)
    scale = embedding.scale.view(torch.uint8)[local].to(torch.int32)
    power = (scale << 23).view(torch.float32)
    rows = raw.float().unflatten(-1, (scale.shape[-1], -1))
    values = (rows * power.unsqueeze(-1)).flatten(-2)
    return values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType
    from sglang_tpu.models.deepseek_v41 import nan_probe

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
            "sglang.srt.layers.engram.EngramHasher._torch_hash_ids",
            tpu_dsv41_torch_hash_ids,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding._owned_rows",
            tpu_dsv41_engram_owned_rows,
            HookType.AROUND,
        ),
    ] + nan_probe.hook_specs()
