# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 TPU runtime hooks (compress ratios 0/1/2).

Pool and loader hooks pass V4 (no ratio-1/2 layers, model_type != deepseek_v41)
through to ``original``. The Engram row hook supports the shared Hasher's
packed-sequence metadata on TPU, including V4 users of that same contract.
Hooks are appended after sglang_tpu/hooks/dsv4.py so shared targets are wrapped
on the outside. Hasher-hook registrations require the neutral prepare_extend_rows
and prepare_lookback_blocked seams carried with this adaptation.

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

prepare_tpu_dsv41_engram_rows
  Why: With the required associative-scan workaround, libtpu 0.0.50 aborts
       lowering the INT64 prefix scan in Engram's repeat_interleave.
  How: AROUND the row-preparation seam; count the already-published packed
       sequence starts instead of recomputing their prefix sum on device.
  Upstream PR or issue: sgl-project/sglang-torchtpu#2101;
       google-pytorch/torch_tpu#2898 tracks the scan-emitter workaround.
  Remove when: the native INT64 scan compiles correctly with that workaround,
       or the backend gains a general exact row-expansion implementation.

prepare_tpu_dsv41_lookback_blocked
  Why: The same libtpu workaround also aborts the INT32 cummax scan used for
       Engram image barriers, independently reproduced after cutting row scan.
  How: AROUND the boolean lookback seam; inclusive prefix OR by static doubling
       preserves every image barrier without scan or cumulative-index outputs.
  Upstream PR or issue: sgl-project/sglang-torchtpu#2101;
       google-pytorch/torch_tpu#2898 tracks the scan-emitter workaround.
  Remove when: native boolean-prefix lowering is correct with the workaround.

barrier_tpu_dsv41_engram_output
  Why: libtpu 0.0.50 cannot choose a tiled BF16 image-select fusion window when
       both adjacent HC parameter-generation DAGs remain connected.
  How: AROUND Engram.forward preserves its lookup/collective/empty handling,
       then fences only nonempty compiled TPU outputs with the public barrier.
       The exact closed-HC reproducer fails without this identity operation and
       compiles with it; image selection and BF16 rounding remain unchanged.
  Upstream PR or issue: sgl-project/sglang-torchtpu#2101; reproduced under the
       deployed pins requiring the scan-emitter workaround tracked in
       google-pytorch/torch_tpu#2898 (a separate compiler issue).
  Remove when: the original closed-HC reproducer compiles correctly under the
       deployed TorchTPU/libtpu pins without the output barrier.
"""

from __future__ import annotations


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


def prepare_tpu_dsv41_engram_rows(
    original, hasher, lens, starts, *, bs, device, output_size
):
    import torch

    if lens.device.type != "tpu" or output_size is None or bs > 2**31 - 1:
        return original(
            hasher, lens, starts, bs=bs, device=device, output_size=output_size
        )
    # Packed starts include duplicate offsets for empty requests. Counting all
    # starts <= t selects the last such offset, exactly skipping zero lengths.
    tokens = torch.arange(output_size, device=device, dtype=torch.int64)
    return ((tokens[:, None] >= starts[None, :]).sum(1, dtype=torch.int32) - 1).to(
        torch.int64
    )


def prepare_tpu_dsv41_lookback_blocked(original, hasher, mask):
    import torch

    if mask.device.type != "tpu":
        return original(hasher, mask)
    result, shift = mask.bool(), 1
    while shift < mask.shape[-1]:
        result = torch.cat(
            (result[..., :shift], result[..., shift:] | result[..., :-shift]), -1
        )
        shift *= 2
    return result


def barrier_tpu_dsv41_engram_output(original, engram, *args, **kwargs):
    import torch

    result = original(engram, *args, **kwargs)
    if (
        result.device.type == "tpu"
        and torch.compiler.is_compiling()
        and result.numel() != 0
    ):
        return torch.tpu.ops.optimization_barrier([result])[0]
    return result


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
            "sglang.srt.layers.engram.EngramHasher.prepare_extend_rows",
            prepare_tpu_dsv41_engram_rows,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramHasher.prepare_lookback_blocked",
            prepare_tpu_dsv41_lookback_blocked,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.Engram.forward",
            barrier_tpu_dsv41_engram_output,
            HookType.AROUND,
        ),
    ]
