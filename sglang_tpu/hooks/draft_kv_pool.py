# SPDX-License-Identifier: Apache-2.0
"""The draft runner's MHA KV pool follows the draft's own attention backend.

draft_mha_pool_class, draft_cell_size_per_token
  Why: KVCacheConfigurator._tpu_mha_pool_class asks the platform, which keys the
       pool class on the process attention backend. A DFlash-family draft on its
       own backend (a dense DSpark draft on pallas_tpu_rpa_v3 beside a
       pallas_tpu_mla Kimi-K3 target) then gets the generic MHA pool instead of
       the fused layout its RPA backend and the draft KV writer address. The
       target's KV budget likewise reserves the generic K + V bytes for the
       draft, while the fused pool pads head_dim to 128 lanes and the KV-head
       pair to the dtype packing (2x for the K3 draft's head_dim 64).
  How: AROUND KVCacheConfigurator._tpu_mha_pool_class: a draft runner with a
       resolved draft attention backend takes that backend's pool class.
       AROUND dflash_utils.dflash_draft_cell_size_per_token: when that pool is
       the fused layout, reserve its padded bytes per token.
  Upstream PR or issue: not filed yet.
  Remove when: the configurator asks the platform for the pool class, and its
       bytes per token, of the runner's own attention backend.
"""

from __future__ import annotations


def _draft_pool_cls(backend: str | None):
    if backend is None:
        return None
    from sglang_tpu.tpu_platform import _mha_kv_pool_cls

    return _mha_kv_pool_cls(backend)


def draft_mha_pool_class(original, configurator):
    pool_cls = _draft_pool_cls(
        configurator.speculative_draft_attention_backend
        if configurator.is_draft_worker
        else None
    )
    return original(configurator) if pool_cls is None else pool_cls


def draft_cell_size_per_token(
    original,
    *,
    draft_model_config,
    draft_num_layers: int,
    draft_kv_cache_dtype,
    tp_size: int,
) -> int:
    generic = original(
        draft_model_config=draft_model_config,
        draft_num_layers=draft_num_layers,
        draft_kv_cache_dtype=draft_kv_cache_dtype,
        tp_size=tp_size,
    )
    from sglang.srt.runtime_context import get_spec
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    pool_cls = _draft_pool_cls(get_spec().speculative_draft_attention_backend)
    if (
        pool_cls is None
        or not issubclass(pool_cls, TpuFusedMHATokenToKVPool)
        or draft_num_layers <= 0
    ):
        return generic
    per_layer = pool_cls.kv_bytes_per_token_per_layer(
        num_kv_heads=draft_model_config.get_num_kv_heads(tp_size),
        head_dim=draft_model_config.head_dim,
        v_head_dim=draft_model_config.v_head_dim,
        store_dtype=draft_kv_cache_dtype,
    )
    # Never below the generic count: the fused pool only adds padding.
    return max(generic, int(per_layer) * int(draft_num_layers))
