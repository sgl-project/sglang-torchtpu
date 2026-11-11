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
    ]
