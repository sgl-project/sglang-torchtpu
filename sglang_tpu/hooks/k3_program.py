# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 megakernel programs (``--speculative-algorithm K3_PROGRAM``).

allow_k3_program_spec
  Why: core rejects every speculative algorithm for Kimi-K3 on TPU (no KDA
       target-verify path); K3_PROGRAM verifies DSpark drafts inside the decode
       program and never runs sglang's KDA verify.
  How: AROUND sglang.srt.arg_groups.kimi_k3_hook.apply_kimi_k3_spec_backend_defaults
       returns without the check under K3_PROGRAM; any other algorithm keeps it.
  Upstream PR or issue: <to be filed> (a platform seam for which speculative
       algorithms a hybrid model supports).
  Remove when: that seam exists.

k3_program_kv_layer_ids
  Why: the unified pools register the DSpark draft's five KV pools after the
       24 MLA pools, and HybridLinearKVPool derives one layer id per entry from
       the full-attention layers alone, which P/D pairing rejects (29 entries,
       24 ids).
  How: AROUND sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.get_kv_layer_ids
       returns K3UnifiedKVPool's ids (the MLA layers, then the draft band
       after the 93 layers) when that is the full-attention pool.
  Upstream PR or issue: <to be filed> (a pool-reported layer id list).
  Remove when: HybridLinearKVPool asks its full-attention pool for the ids.

attach_k3_program
  Why: the program owns the decoder weights (pre-sharded, read at load) and
       must be resident before sglang sizes the KV pools from free HBM; the
       model has no seam for a whole-model replacement.
  How: AFTER KimiK3LinearForCausalLM.post_load_weights builds the program for
       the server's P/D role and loads its weights (sglang_program). The
       decoder layers are never built: hooks/k3_megakernel's weight
       ownership covers K3_PROGRAM (skip_k3_decoder_layers,
       load_k3_non_layer_weights).
  Upstream PR or issue: <to be filed> (a model-level seam to replace a decoder
       stack's layers and forward).
  Remove when: that seam exists.

size_k3_program_pools
  Why: DefaultPoolConfigurator sizes the full-attention pool from the MLA
       latent's bytes per token on every rank (about 27.6 KB for Kimi-K3); the
       unified pools hold a rank's share of each logical block (1160 bytes a
       token in bf16), so sized from free memory the pool would come out about
       24 times too small.
  How: AFTER DefaultPoolConfigurator.__init__ sets the cell size from
       K3UnifiedKVPool.bytes_per_token under K3_PROGRAM (the state slots are
       costed through K3StatePool.cache_bytes_per_request).
  Upstream PR or issue: sgl-project/sglang-torchtpu#1072 (the configurator
       asking the platform pool for its bytes per token).
  Remove when: that lands.
"""

from __future__ import annotations

from typing import Any


def k3_program_selected(server_args: Any) -> bool:
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang_tpu.k3_programs.sglang_worker import ALGORITHM

    algorithm = resolving_view(server_args).speculative_algorithm
    return str(algorithm or "").upper() == ALGORITHM


def allow_k3_program_spec(original, server_args: Any, *args, **kwargs) -> None:
    """AROUND apply_kimi_k3_spec_backend_defaults."""
    if k3_program_selected(server_args):
        return None
    return original(server_args, *args, **kwargs)


def k3_program_kv_layer_ids(original, pool: Any, *args, **kwargs) -> list[int]:
    """AROUND HybridLinearKVPool.get_kv_layer_ids."""
    from sglang_tpu.k3_programs.sglang_pools import K3UnifiedKVPool

    full = getattr(pool, "full_kv_pool", None)
    if isinstance(full, K3UnifiedKVPool):
        return full.get_kv_layer_ids()
    return original(pool, *args, **kwargs)


def k3_program_active() -> bool:
    """Whether this server runs K3_PROGRAM (from the runtime context; no
    published configuration, as before server arguments resolve: no)."""
    from sglang.srt.runtime_context import get_context, get_spec
    from sglang_tpu.k3_programs.sglang_worker import ALGORITHM

    if not get_context().is_config_namespace_published("spec"):
        return False
    return str(get_spec().speculative_algorithm or "").upper() == ALGORITHM


def attach_k3_program(result: None, causal_lm: Any, *args, **kwargs) -> None:
    """AFTER KimiK3LinearForCausalLM.post_load_weights."""
    if not k3_program_active():
        return
    from sglang_tpu.k3_programs.sglang_program import build_from_server

    causal_lm.model._k3_program = build_from_server()
    causal_lm.model._derived_weight_cache_error = (
        "Online weight updates are not supported with K3_PROGRAM: the program "
        "reads pre-sharded weights at load time. Restart the server to change "
        "weights."
    )


def size_k3_program_pools(result: None, configurator, kvc, *args, **kwargs) -> None:
    """AFTER DefaultPoolConfigurator.__init__."""
    if not k3_program_active():
        return result
    from sglang.srt.runtime_context import get_model
    from sglang_tpu.k3_programs.sglang_pools import K3UnifiedKVPool

    configurator._cell_size = K3UnifiedKVPool.bytes_per_token(
        get_model().kv_cache_dtype
    )
    return result


def hook_specs() -> list[tuple]:
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.arg_groups.kimi_k3_hook.apply_kimi_k3_spec_backend_defaults",
            allow_k3_program_spec,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.get_kv_layer_ids",
            k3_program_kv_layer_ids,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights",
            attach_k3_program,
            HookType.AFTER,
        ),
        (
            "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator.__init__",
            size_k3_program_pools,
            HookType.AFTER,
        ),
    ]
