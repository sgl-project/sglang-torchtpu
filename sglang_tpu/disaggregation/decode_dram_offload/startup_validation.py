"""Refuse settings the KV offload to decode DRAM does not support.

The offload moves full-attention KV between the two roles' HiCache host
pools, rank to rank. Anything it cannot
move or agree on is refused here with the setting to change, at engine start
on either role. A pair whose engines disagree fails its requests instead (the
raiden channel, ``raiden/dram_offload_channel.py``). Nothing degrades to "no offload"
silently.
"""

from __future__ import annotations

import inspect
from typing import Any, List

from sglang_tpu.envs import envs as tpu_envs

_SWITCH = tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM


class UnsupportedDramOffloadConfig(ValueError):
    """The switch is on with a setting the offload does not support."""


def _refuse(problems: List[str], *, where: str) -> None:
    if problems:
        raise UnsupportedDramOffloadConfig(
            f"{_SWITCH.name}=1 is not supported on this {where}: "
            + "; ".join(problems)
            + f" (change the setting or unset {_SWITCH.name})"
        )


def missing_tpu_sync_host_api() -> List[str]:
    """Host-to-host read parameters the installed tpu-sync lacks."""
    from tpu_sync.api.torch import kv_cache_manager

    manager = kv_cache_manager.KVCacheManager
    register = inspect.signature(manager.register_read).parameters
    start = inspect.signature(manager.start_read).parameters
    return [
        name
        for name, parameters, parameter in (
            ("register_read(source=)", register, "source"),
            ("start_read(destination=)", start, "destination"),
        )
        if parameter not in parameters
    ]


def validate_engine(
    *, role: str, scheduler: Any, kv_manager: Any, tree_cache: Any
) -> None:
    """Raise UnsupportedDramOffloadConfig unless this engine can take part.

    ``role`` is "prefill" or "decode". Called once the transfer manager holds
    the model's KV pools, before the engine serves a request.
    """
    if not _SWITCH.get():
        return
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import get_device, get_disagg, get_memory
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

    where = f"{role} engine"
    # Without the raiden manager none of the engine facts below exist.
    if not isinstance(kv_manager, RaidenKVManager):
        _refuse(["--disaggregation-transfer-backend must be raiden"], where=where)
    memory = get_memory()
    server_args = scheduler.server_args
    python_tree = isinstance(tree_cache, UnifiedRadixCache) and isinstance(
        tree_cache.tree_core, UnifiedTreeCore
    )
    hicache_on = python_tree and tree_cache.cache_controller is not None
    requirements = [
        (
            not server_args.disable_radix_cache,
            "the radix cache must stay on (--disable-radix-cache is set)",
        ),
        (
            # It lets a request leave bootstrap before its recall is settled.
            get_disagg().optimistic_prefill_attempts == 0,
            "optimistic prefill is not supported (--optimistic-prefill-attempts, "
            "which --disaggregation-decode-allocation-policy prefill_complete sets)",
        ),
        (
            # A sleeping engine steps on requests only: it would not see its
            # peer's pushes or settle its own.
            not get_device().sleep_on_idle,
            "--sleep-on-idle is not supported",
        ),
        (
            role != "decode" or get_disagg().disaggregation_decode_enable_radix_cache,
            "the decode needs --disaggregation-decode-enable-radix-cache",
        ),
        (
            python_tree,
            "the Python unified radix tree is required (the Rust tree core is not supported)",
        ),
        (
            hicache_on,
            "--enable-hierarchical-cache is required on both roles",
        ),
        (
            memory.hicache_io_backend == "raiden",
            "--hicache-io-backend must be raiden",
        ),
        (
            memory.hicache_storage_backend is None,
            "--hicache-storage-backend (an L3 tier) is not supported",
        ),
        (
            not memory.enable_unified_memory,
            "the unified KV pool (--enable-unified-memory) is not supported",
        ),
        (
            kv_manager.pp_size == 1,
            f"pipeline parallelism is not supported (--pp-size {kv_manager.pp_size})",
        ),
        (
            kv_manager.pp_size > 1 or not kv_manager._pipeline_enabled,
            "the Raiden pipeline transfer is not supported "
            f"({tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.name} is set)",
        ),
        (
            kv_manager.attn_cp_size == 1,
            "context parallelism is not supported",
        ),
        (
            kv_manager.attn_dp_size == 1 and kv_manager.system_dp_size == 1,
            "data parallelism is not supported",
        ),
        (
            not kv_manager._state_kv_caches,
            "models with recurrent state (GDN/Mamba) are not supported",
        ),
        (
            scheduler.spec_algorithm.is_none(),
            "speculative decoding is not supported",
        ),
        (
            not kv_manager._swa_kv_caches,
            "hybrid sliding-window models are not supported: only "
            "full-attention KV is offloaded, and such a model cannot resume "
            "from it without its window",
        ),
    ]
    problems = [reason for ok, reason in requirements if not ok]
    missing = missing_tpu_sync_host_api()
    if missing:
        problems.append(
            "the installed tpu-sync has no host-to-host read API (missing "
            + ", ".join(missing)
            + ")"
        )
    _refuse(problems, where=where)
