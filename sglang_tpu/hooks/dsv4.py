# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 TPU KV pool construction and memory-budget sizing hooks.

build_tpu_dsv4_kv_pool, get_tpu_dsv4_bytes_per_swa_token,
get_tpu_dsv4_bytes_per_full_token, get_tpu_dsv4_c128_state_fixed_bytes,
get_tpu_dsv4_fixed_c4_state_bytes
  Why: KVCacheConfigurator._build_dsv4_kv_pool and DSV4PoolConfigurator hardcode
       the GPU/NPU DeepSeekV4TokenToKVPool class and FlashMLA/PA_ND byte-per-token
       and compressor-state formulas. On TPU, DeepSeek-V4 uses 4D page-major
       uint8 Pallas buffers (DSV4TPUTokenToKVPool) with 1024 B/token SWA/HCA,
       640 B/token CSA, 256 B/token Indexer K, and uniform 32-page per-request
       ring + 2048-page scratch compressor state pools.
  How: AROUND KVCacheConfigurator._build_dsv4_kv_pool binds
       DeepSeekV4TokenToKVPool to DSV4TPUTokenToKVPool for the call and registers
       the per-request C4/C128 state-shift reset callback on ReqToTokenPool.
       AROUND DSV4PoolConfigurator._get_bytes_per_swa_token,
       _get_bytes_per_full_token, _get_c128_state_fixed_bytes, and
       _fixed_c4_state_bytes delegate the byte budget to DSV4TPUTokenToKVPool.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1119.
  Remove when: KVCacheConfigurator and DSV4PoolConfigurator query
       current_platform for the DSV4 KV pool class and its byte-sizing methods.
"""

from __future__ import annotations

from unittest.mock import patch


def build_tpu_dsv4_kv_pool(original, kvc, *args, **kwargs):
    import sglang.srt.mem_cache.kv_cache_configurator as kvc_mod
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    with patch.object(kvc_mod, "DeepSeekV4TokenToKVPool", DSV4TPUTokenToKVPool):
        pool = original(kvc, *args, **kwargs)
    req_to_token_pool = kwargs["req_to_token_pool"]
    if not kvc.is_draft_worker and not pool._unified_kv:
        req_to_token_pool.register_on_alloc_rows(pool.clear_c4_req_states)
    return pool


def get_tpu_dsv4_bytes_per_swa_token(original, configurator) -> float:
    del original
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    return DSV4TPUTokenToKVPool.bytes_per_swa_token(
        num_layers_total=configurator.num_layers_total
    )


def get_tpu_dsv4_bytes_per_full_token(original, configurator) -> float:
    del original
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    swa_ratio = (
        0
        if configurator._unified or configurator.swa_cap_tokens is not None
        else configurator.swa_ratio
    )
    return DSV4TPUTokenToKVPool.bytes_per_full_token(
        swa_ratio=swa_ratio,
        bytes_per_swa_token=configurator.bytes_per_swa_token,
        c4_shrink_factor=configurator.c4_shrink_factor,
        num_c4_layers=configurator.num_layers(4),
        num_c128_layers=configurator.num_layers(128),
    )


def get_tpu_dsv4_c128_state_fixed_bytes(
    original, configurator, max_running_requests: int
) -> int:
    del original
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    return DSV4TPUTokenToKVPool.c128_state_fixed_bytes(
        num_req_slots=configurator._get_num_req_slots(max_running_requests),
        attn_head_dim=configurator.attn_head_dim,
        num_c128_layers=configurator.num_layers(128),
    )


def get_tpu_dsv4_fixed_c4_state_bytes(
    original, configurator, max_running_requests: int
) -> int:
    del original
    from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool

    return DSV4TPUTokenToKVPool.c4_state_fixed_bytes(
        num_req_slots=configurator._get_num_req_slots(max_running_requests),
        attn_head_dim=configurator.attn_head_dim,
        indexer_head_dim=configurator.indexer_head_dim,
        num_c4_layers=configurator.num_layers(4),
    )


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    kvc = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    cfg = "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator"
    return [
        (
            f"{kvc}._build_dsv4_kv_pool",
            build_tpu_dsv4_kv_pool,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_bytes_per_swa_token",
            get_tpu_dsv4_bytes_per_swa_token,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_bytes_per_full_token",
            get_tpu_dsv4_bytes_per_full_token,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_c128_state_fixed_bytes",
            get_tpu_dsv4_c128_state_fixed_bytes,
            HookType.AROUND,
        ),
        (
            f"{cfg}._fixed_c4_state_bytes",
            get_tpu_dsv4_fixed_c4_state_bytes,
            HookType.AROUND,
        ),
    ]
