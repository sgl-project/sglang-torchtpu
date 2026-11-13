# SPDX-License-Identifier: Apache-2.0
"""Inkling2 attention backend and KV pool on TPU.

inkling2_attn_backend
  Why: attn_backend_wrapper builds Inkling's conv-state sidecar from the CUDA
       InklingShortConvAttnBackend (Triton imports at module top); the TPU sconv
       backend must stay out of core.
  How: AROUND attention_registry.attn_backend_wrapper returns the TPU hybrid
       (RPA v3 for every layer + TpuInklingSconvBackend) for Inkling2 models and
       defers to the original otherwise.
  Upstream PR or issue: none (private Inkling2 port).
  Remove when: attn_backend_wrapper selects the conv-state sidecar through a
       platform hook.

inkling2_swa_kv_pool
  Why: the TPU branch of KVCacheConfigurator._build_token_to_kv_pool skips
       models with conv / mamba state, so Inkling2 (hybrid SWA + conv state)
       falls through to plain MHATokenToKVPool sub-pools, which RPA v3 cannot read.
  How: AROUND KVCacheConfigurator._build_hybrid_swa_kv_pool builds the TPU
       fused SWA pool for Inkling2 and defers to the original otherwise.
  Upstream PR or issue: none (private Inkling2 port).
  Remove when: the TPU pool branch accepts hybrid-SWA models with conv state.
"""

from __future__ import annotations

from typing import Any, Callable


def is_inkling2(hf_config: Any) -> bool:
    from sglang.tml.config import Inkling2MMConfig, Inkling2ModelConfig

    return isinstance(hf_config, (Inkling2MMConfig, Inkling2ModelConfig))


def inkling2_attn_backend(original: Callable, runner: Any, full_attn_backend: Any):
    if not is_inkling2(runner.model_config.hf_config):
        return original(runner, full_attn_backend)
    from sglang_tpu.attention_inkling_sconv_pallas import build_inkling_hybrid_backend

    return build_inkling_hybrid_backend(runner, full_attn_backend)


def inkling2_swa_kv_pool(
    original: Callable,
    configurator: Any,
    *,
    full_max_total_num_tokens: Any,
    swa_max_total_num_tokens: Any,
    mha_pool_class: Any,
):
    if not is_inkling2(configurator.model_config.hf_config):
        return original(
            configurator,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
            mha_pool_class=mha_pool_class,
        )
    return configurator._build_tpu_swa_kv_pool(
        full_max_total_num_tokens=full_max_total_num_tokens,
        swa_max_total_num_tokens=swa_max_total_num_tokens,
    )
