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
    ]
