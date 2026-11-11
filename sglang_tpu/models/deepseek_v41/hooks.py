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
"""

from __future__ import annotations


def _pool_hooks():
    from sglang_tpu.models.deepseek_v41 import memory_pool

    return (
        memory_pool.build_tpu_dsv41_kv_pool,
        memory_pool.get_tpu_dsv41_bytes_per_swa_token,
        memory_pool.get_tpu_dsv41_bytes_per_full_token,
    )


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    kvc = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    cfg = "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator"
    build_pool, swa_bytes, full_bytes = _pool_hooks()
    return [
        (f"{kvc}._build_dsv4_kv_pool", build_pool, HookType.AROUND),
        (f"{cfg}._get_bytes_per_swa_token", swa_bytes, HookType.AROUND),
        (f"{cfg}._get_bytes_per_full_token", full_bytes, HookType.AROUND),
    ]
