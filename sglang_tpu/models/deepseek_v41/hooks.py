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
       Into the uint8 tables of tpu_dsv41_engram_bytes_init it copies the
       checkpoint's FP8 / E8M0 bytes, not their values.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: EngramEmbedding accepts a pre-sliced row range.

tpu_dsv41_engram_bytes_init
  Why: reinterpreting an FP8 Engram table inside the compiled TPU forward
       (weight.view(torch.uint8)[rows]) lowers to a bitcast-convert of the
       whole table on every step: two 12.3 GB tables plus their scales per
       rank, 19.3 of the 41 ms decode forward on tpu7x (DP8, 1M context).
       The two lookup hooks below also need this rank's DP slot and first
       table row. Read as Python ints inside the compiled forward they become
       graph constants, so each rank traces its own program; rank-divergent
       programs can return wrong data on some ranks (see _all_reduce_gather in
       sglang_tpu/attention_pallas_rpa_dcp.py).
  How: AFTER EngramEmbedding.__init__ the device-table weight and scale are
       uint8 parameters, as vllm-torchtpu engram_hbm stores them; the lookup
       gathers uint8 rows and reinterprets only the gathered rows. On non-CUDA
       devices it also sets the plain tensor attributes tpu_dp_rank_mask,
       tpu_dp_rank_index and tpu_row_start (host tables and empty shards
       included), created eagerly on the model's build device, so the lookups
       read the rank as runtime graph inputs (_set_engram_rank_tensors).
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: an FP8 / E8M0 table can be gathered in the compiled forward
       without a whole-table conversion; the rank tensors go with
       tpu_dsv41_engram_owned_rows and tpu_dsv41_engram_dp_lookup.

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
       With attention DP the token embedding is looked up eagerly here and
       passed in as input_embeds (input_ids still go in for Engram and MoE
       routing), as vllm-torchtpu #1848 does: inside the TP1/DP8 compiled
       graph the unsharded BF16 embedding gather returns wrong rows, while the
       same lookup outside the graph is exact. Warmup takes this path too, so
       precompiled and serving graphs share one signature.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: the TPU batch builder produces int32 index tensors and the
       in-graph embedding gather is fixed in torch_tpu.

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
       the vllm-torchtpu epic engram_hbm.dequant and #1743 do. With uint8
       tables it gathers the rows first and reinterprets only those. The
       table offset is the device tensor tpu_row_start, not the Python int
       row_start, so every rank traces one program.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: TPU lowers float8_e8m0fnu to float32.

tpu_dsv41_engram_dp_lookup
  Why: under attention DP, EngramEmbedding._dp_sharded_lookup all-gathers the
       ranks' hash ids into a fresh buffer and reduce-scatters the rows into
       another. Inside the compiled decode forward on torch_tpu 0.2.0 /
       libtpu 0.0.50 that returned wrong data (NaN logits from the first
       decode step) once the whole-table Engram conversion above was gone;
       the same fault class is documented in
       sglang_tpu/attention_pallas_rpa_dcp.py.
  How: AROUND _dp_sharded_lookup on non-CUDA devices: this rank's ids go into
       a zero [world * rows] buffer that is all-reduced, and the looked-up
       values are all-reduced before this rank's block is kept. Exact up to
       the sign of zero, as the core reduce-scatter: the other id blocks are
       zero and every table row has one owner. The block is placed with the
       one-hot tpu_dp_rank_mask and kept with index_select on
       tpu_dp_rank_index, device tensors rather than attn_dp_rank, so every
       rank traces one program and issues the same collectives (the
       _all_reduce_gather / own_heads pattern of attention_pallas_rpa_dcp.py).
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: all_gather_into_tensor / reduce_scatter_tensor into fresh
       buffers are correct inside compiled TPU graphs.

tpu_dsv41_hasher_forward
  Why: the extend path of EngramHasher.forward builds per-token request rows
       with repeat_interleave over int64 lengths, which lowers to a while loop
       on 64-bit counters that TPU lowering rejects.
  How: AROUND EngramHasher.forward on non-CUDA extend computes the rows in
       int32 as a compare-sum against request ends and hashes every padded row
       (no guard on the host lengths), then commits history for real rows only.
       Extend and decode commit history as history.copy_(history.index_put(..)),
       the functional-update-then-copy_ form vllm-torchtpu uses for its Engram
       slot cache; the in-place index_put of the original is not written back
       from the compiled TPU graph, so decode read all-zero predecessors.
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

tpu_dsv41_fp32_moe_sum
  Why: with attention DP 8, MoE EP 8 and attn_tp 1, every rank computes MoE
       partials (its EP-routed experts plus its share of the TP8-sharded shared
       expert) for every token of the DP-gathered buffer, and _post_experts_sum
       adds the 8 partials with a BF16 all-reduce. The BF16 summation order
       depends on where a token sits in that buffer, so one greedy prompt gave
       different tokens and logprobs on different DP ranks. Summed in FP32 and
       rounded to BF16 once, all 8 ranks were bit-identical on TPU, GSM8K
       unchanged. The earlier sglang_tpu V4.1 model summed each token in FP32.
  How: AROUND sglang.srt.layers.moe.utils._post_experts_sum, which
       post_experts_all_reduce and sum_post_experts_output call through the
       module global. A BF16 input that will be reduced (reduce_ep or
       reduce_tp) goes through the original as .float() and comes back with
       .to(torch.bfloat16); any other call is passed through. The all-reduce
       then carries FP32, twice the bytes. Hooks register on TPU for every
       model, so the hook also requires the process's target model to be V4.1
       (process_model_config().hf_config.model_type, the check
       select_tpu_dsv41_loader makes), read through
       torch._dynamo.assume_constant_result: one constant, the same on every
       rank, decided at trace time outside the compiled graph.
  Upstream PR or issue: none yet; candidate seam: a precision option for the
       post-experts reduction.
  Remove when: core offers that option, or the TPU all-reduce sums BF16 in an
       order that does not depend on the rank.
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
    from sglang_tpu.models.deepseek_v41.ep_local_expert_reader import (
        ROW_START_ATTR,
        load_local_rows,
    )

    if param.dtype == torch.uint8 and loaded_weight.dtype != torch.uint8:
        # Byte tables (tpu_dsv41_engram_bytes_init): copy the checkpoint's FP8 / E8M0
        # bytes, not their values.
        row_start = getattr(loaded_weight, ROW_START_ATTR, None)
        loaded_weight = loaded_weight.view(torch.uint8)
        if row_start is not None:
            setattr(loaded_weight, ROW_START_ATTR, row_start)
    return load_local_rows(original, embedding, param, loaded_weight)


def tpu_dsv41_engram_dp_lookup(original, embedding, indices, forward_batch):
    """AROUND EngramEmbedding._dp_sharded_lookup on non-CUDA devices: gather the DP
    ranks' hash ids and return this rank's rows with all-reduces instead of an
    all-gather and a reduce-scatter into fresh buffers (see the compiled-gather
    fault documented in sglang_tpu/attention_pallas_rpa_dcp.py). Exact up to the
    sign of zero: every id block but this rank's is zero, and each table row
    has exactly one owner.

    The rank is data, not structure: the one-hot tpu_dp_rank_mask picks the block
    this rank's ids fill and index_select on tpu_dp_rank_index keeps this rank's
    rows, as _all_reduce_gather / own_heads do in attention_pallas_rpa_dcp.py. The
    traced graph is then the same on every rank (attn_dp_rank sliced in as a
    Python int would give each rank its own program)."""
    if indices.is_cuda:
        return original(embedding, indices, forward_batch)
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.dp_attention import get_global_dp_buffer_len
    from sglang.srt.runtime_context import get_parallel

    world = get_parallel().attn_dp_size
    rows = get_global_dp_buffer_len()
    local_rows = indices.shape[0]
    if rows == 0:
        return embedding._empty(indices)
    if rows != world * local_rows or embedding.tp_size != world:
        return original(embedding, indices, forward_batch)
    mask = embedding.tpu_dp_rank_mask.view(world, *([1] * indices.dim()))
    padded = torch.where(mask, indices.unsqueeze(0), indices.new_zeros(()))
    ids_global = tensor_model_parallel_all_reduce(
        padded.reshape(rows, *indices.shape[1:])
    )
    values = embedding._owned_rows(ids_global).reshape(rows, -1)
    values = tensor_model_parallel_all_reduce(values)
    local = values.reshape(world, local_rows, -1).index_select(
        0, embedding.tpu_dp_rank_index
    )
    return local.reshape(*indices.shape, embedding.dim)


def _set_engram_rank_tensors(embedding) -> None:
    """Give the Engram lookups this rank's position as device tensors.

    tpu_dp_rank_mask   bool[attn_dp_size], True at attn_dp_rank (DP lookup block).
    tpu_dp_rank_index  int64[1], attn_dp_rank (DP lookup block kept).
    tpu_row_start      int32[] (int64[] past the int32 range), row_start (table
                       offset in _owned_rows); TPU hash ids are int32, and a 0-d
                       operand keeps the ids' dtype.

    A Python int read inside the compiled forward is a graph constant, so the
    rank-dependent attn_dp_rank and row_start would give each rank its own
    program. Tensors that exist before tracing are runtime graph inputs instead,
    so they are created here, eagerly, with no device argument: inside model
    construction the default device is the loader's target device, where the ids
    arrive (also for host tables, whose own storage is host memory).

    Plain attributes, not parameters or buffers: load_weights fills only
    named_parameters(), and they stay out of state_dict(), post-load staging and
    the startup storage manifest. Dynamo lifts them as graph inputs. The direct
    tracer would need them in model_extra_tensor_attrs, but DeepSeek-V4.1 TPU
    attention rejects that tracer.
    """
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    world, rank = parallel.attn_dp_size, parallel.attn_dp_rank
    rank_mask = torch.tensor([r == rank for r in range(world)], dtype=torch.bool)
    if rank_mask.is_cuda:
        return
    embedding.tpu_dp_rank_mask = rank_mask
    embedding.tpu_dp_rank_index = torch.tensor([rank], dtype=torch.int64)
    fits = embedding.row_start + embedding.rows <= torch.iinfo(torch.int32).max
    embedding.tpu_row_start = torch.tensor(
        embedding.row_start, dtype=torch.int32 if fits else torch.int64
    )


def tpu_dsv41_engram_bytes_init(result, embedding, *args, **kwargs):
    """AFTER EngramEmbedding.__init__: hold the device tables as uint8 bytes, and
    set the rank tensors the lookups read (_set_engram_rank_tensors).

    The non-CUDA lookup (tpu_dsv41_engram_owned_rows) gathers uint8 rows and
    reinterprets only those rows. Reinterpreting an FP8 table inside the compiled
    TPU forward lowers to a bitcast-convert of the whole table on every step
    (two 12.3 GB tables per rank), and FP8-typed gathers gave NaN in the decode
    graph. vllm-torchtpu stores the same tables as uint8 (engram_hbm.py).

    The rank tensors are set before the host-table / empty-shard return: the DP
    lookup (tpu_dsv41_engram_dp_lookup) runs for those tables too.
    """
    _set_engram_rank_tensors(embedding)
    if embedding.host_table is not None or embedding.rows == 0:
        return result
    for name in ("weight", "scale"):
        old = getattr(embedding, name)
        if old.is_cuda or old.dtype == torch.uint8:
            continue
        new = torch.nn.Parameter(
            torch.empty(old.shape, dtype=torch.uint8, device=old.device),
            requires_grad=False,
        )
        new.weight_loader = old.weight_loader
        setattr(embedding, name, new)
    return result


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


@torch._dynamo.assume_constant_result
def _serves_dsv41() -> bool:
    """Whether this process's target model is DeepSeek-V4.1 (draft workers
    included: process_model_config is built from the served model path).

    Fixed per process and equal on every rank; under Dynamo it is evaluated once
    at trace time and baked in, as ModelConfig is not traceable."""
    from sglang.srt.runtime_context import process_model_config

    config = process_model_config().hf_config
    return getattr(config, "model_type", None) == "deepseek_v41"


def tpu_dsv41_fp32_moe_sum(original, hidden_states, *, reduce_ep, reduce_tp):
    """AROUND _post_experts_sum: sum V4.1's BF16 MoE partials in FP32.

    The ops are .float(), the original reduction, .to(torch.bfloat16): one
    rounding to BF16 after the full sum instead of one per partial added. Summed
    in FP32, 8 BF16 partials are exact (so their order does not matter) unless
    their exponents span more than about 13 binades."""
    if (
        hidden_states.dtype != torch.bfloat16
        or not (reduce_ep or reduce_tp)
        or not _serves_dsv41()
    ):
        return original(hidden_states, reduce_ep=reduce_ep, reduce_tp=reduce_tp)
    summed = original(hidden_states.float(), reduce_ep=reduce_ep, reduce_tp=reduce_tp)
    return summed.to(torch.bfloat16)


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
    mode = forward_batch.forward_mode
    if input_ids.is_cuda or num_tokens == 0:
        return original(hasher, input_ids, forward_batch)
    if mode.is_decode():
        return _tpu_hasher_decode(hasher, input_ids, forward_batch)
    if not mode.is_extend():
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
    _commit_history(hasher, commit_rows, tokens[commit_last][:, : n - 1].flip(-1))
    return hash_ids


def _commit_history(hasher, rows, values):
    """Write history rows the way vllm-torchtpu writes its Engram slot cache:
    a functional update copied back into the resident tensor
    (``cache.copy_(_write_op(cache, slots, values))``). An in-place index_put
    on the plain attribute inside the compiled graph is not written back on
    TPU, so decode read all-zero predecessors. Padding rows go to pad_row."""
    history = hasher.history
    rows = rows.to(torch.int64)
    history.copy_(history.index_put((rows,), values.to(history.dtype)))


def _tpu_hasher_decode(hasher, input_ids, forward_batch):
    """EngramHasher.forward decode on TPU: same hash and commit as the
    original torch path, with the history write from ``_commit_history``."""
    from sglang.kernels.ops.embeddings.engram_hash import MODE_DECODE

    n = hasher.max_ngram_size
    req_slots = forward_batch.req_pool_indices
    hash_ids, tokens = hasher._torch_hash_ids(
        input_ids,
        forward_batch.positions,
        MODE_DECODE,
        hasher.history[req_slots.to(torch.int64)],
        input_ids.shape[0],
        1,
        None,
        None,
    )
    rows = req_slots.to(torch.int32)
    out_loc = forward_batch.out_cache_loc
    if out_loc is not None:
        rows = torch.where(out_loc == 0, hasher.pad_row, rows)
    _commit_history(hasher, rows, tokens[:, : n - 1].flip(-1))
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
    if embed_outside:
        embed_tokens = shim.policy.model_runner.model.get_input_embeddings()
        with torch.no_grad():
            forward_batch.input_embeds = embed_tokens(input_ids)
    for name in _INT32_BATCH_FIELDS:
        setattr(forward_batch, name, _int32(saved[name]))
    if pin_history:
        forward_batch.engram_history = _padded_history(
            hasher, forward_batch.engram_history, forward_batch.req_pool_indices
        )
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


def tpu_dsv41_engram_owned_rows(original, embedding, indices):
    """_owned_rows with an integer e8m0 dequant; bytes 0 and 255 aside, exact.

    The offset is the device tensor tpu_row_start (_set_engram_rank_tensors), not
    the Python int row_start, so the traced graph is the same on every rank.
    embedding.rows stays a Python int: it is the table's leading dimension, a
    static shape either way. It is equal on every rank when num_embeddings is a
    multiple of tp_size; otherwise ranks differ by one row in the table shape
    itself, and the lookup is still exact."""
    if indices.is_cuda or embedding.host_table is not None or embedding.rows == 0:
        return original(embedding, indices)
    local = indices - embedding.tpu_row_start
    owned = (local >= 0) & (local < embedding.rows)
    local = local.masked_fill(~owned, 0)
    if embedding.weight.dtype == torch.uint8:
        # Byte tables: gather rows, then reinterpret only the gathered rows.
        raw = embedding.weight[local].view(torch.float8_e4m3fn)
        scale = embedding.scale[local].to(torch.int32)
    else:
        raw = embedding.weight.view(torch.uint8)[local].view(torch.float8_e4m3fn)
        scale = embedding.scale.view(torch.uint8)[local].to(torch.int32)
    power = (scale << 23).view(torch.float32)
    rows = raw.float().unflatten(-1, (scale.shape[-1], -1))
    values = (rows * power.unsqueeze(-1)).flatten(-2)
    return values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

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
            "sglang.srt.layers.engram.EngramEmbedding.__init__",
            tpu_dsv41_engram_bytes_init,
            HookType.AFTER,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding._dp_sharded_lookup",
            tpu_dsv41_engram_dp_lookup,
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
        (
            "sglang.srt.layers.moe.utils._post_experts_sum",
            tpu_dsv41_fp32_moe_sum,
            HookType.AROUND,
        ),
    ]
