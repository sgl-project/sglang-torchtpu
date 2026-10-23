# SPDX-License-Identifier: Apache-2.0
"""The KV pool budget counts the TPU split DSA pool's real bytes per token.

size_tpu_dsa_kv_pool
  Why: DefaultPoolConfigurator sizes a DSA pool with the GPU layout (an
       FP8-scaled MLA latent and a 132-byte index key per layer). The TPU split
       pool stores a rope row padded to one lane, a 256-byte index-key record
       and, for a pooled indexer, a bf16 key + gate per token. Sized from free
       memory, the pool then oversubscribes HBM (2.1x for GLM-5.3-Flash) and the
       first compile runs out of memory.
  How: AFTER DefaultPoolConfigurator.__init__: when the platform's DSA pool is
       TpuSplitMLATokenToKVPool, recount the cell size from the pool's own
       bytes_per_token for the layers this runner allocates, plus the EAGLE /
       MTP draft's layers the target reserves for.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1072.
  Remove when: the configurator asks the platform for the DSA pool's bytes per
       token (get_mla_kv_bytes_per_token covers non-DSA MLA only).
"""

from __future__ import annotations


def _split_pool_bytes(kvc, *, layer_ids, all_index_layers: bool) -> int:
    from sglang.srt.configs.model_config import (
        dsa_layer_skips_topk,
        get_dsa_index_head_dim,
        get_dsa_index_kpool,
    )
    from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool

    hf_config = kvc.model_config.hf_config
    return TpuSplitMLATokenToKVPool.bytes_per_token(
        index_head_dim=get_dsa_index_head_dim(hf_config),
        # As the pool builders: a draft computes its own top-k.
        index_key_layers=[
            all_index_layers or not dsa_layer_skips_topk(hf_config, layer_id)
            for layer_id in layer_ids
        ],
        index_kpool=get_dsa_index_kpool(hf_config),
    )


def _attention_layer_ids(kvc):
    """The layers the pool allocates: every layer, or a hybrid model's
    full-attention layers (HybridLinearKVPool), as DefaultPoolConfigurator counts."""
    from sglang.srt.configs.hybrid_arch import mambaish_config

    layers = kvc.layer_info
    owned = range(layers.start_layer, layers.end_layer)
    if mambaish := mambaish_config(kvc.model_config):
        return [i for i in mambaish.full_attention_layer_ids if i in owned]
    return owned


def size_tpu_dsa_kv_pool(result: None, configurator, kvc, *args, **kwargs) -> None:
    from sglang.srt.configs.model_config import is_deepseek_dsa
    from sglang.srt.platforms import current_platform
    from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool

    if (
        not is_deepseek_dsa(kvc.model_config.hf_config)
        or kvc.kv_cache_dtype_str == "mxint8"
        or not issubclass(
            current_platform.get_dsa_kv_pool_cls(), TpuSplitMLATokenToKVPool
        )
    ):
        return result
    cell_size = _split_pool_bytes(
        kvc,
        layer_ids=_attention_layer_ids(kvc),
        all_index_layers=kvc.is_draft_worker,
    )
    draft_layers = kvc.spec_aux_config.eagle_draft_num_layers
    if (
        (kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone())
        and not kvc.is_draft_worker
        and draft_layers
    ):
        cell_size += _split_pool_bytes(
            kvc, layer_ids=range(int(draft_layers)), all_index_layers=True
        )
    configurator._cell_size = cell_size
    return result
