"""Refuse settings the KV offload to decode DRAM does not support.

The offload moves full-attention KV between the two roles' HiCache host
pools, rank to rank. Anything it cannot
move or agree on is refused here with the setting to change, at engine start
on either role, or for the pair when the first request connects two engines
that disagree. Nothing degrades to "no offload" silently.
"""

from __future__ import annotations

import inspect
from typing import Any, List, Optional, Tuple

from sglang_tpu.envs import envs as tpu_envs

_SWITCH = tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM

_HOW_TO_TURN_OFF = f"change the setting or unset {_SWITCH.name}"


class UnsupportedSpillConfig(ValueError):
    """The switch is on with a setting the offload does not support."""


def _refuse(problems: List[str], *, where: str) -> None:
    if problems:
        raise UnsupportedSpillConfig(
            f"{_SWITCH.name}=1 is not supported on this {where}: "
            + "; ".join(problems)
            + f" ({_HOW_TO_TURN_OFF})"
        )


def _failed(requirements: List[Tuple[bool, str]]) -> List[str]:
    return [reason for ok, reason in requirements if not ok]


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
            ("start_read(local_host_block_ids=)", start, "local_host_block_ids"),
        )
        if parameter not in parameters
    ]


def validate_engine(
    *, role: str, scheduler: Any, kv_manager: Any, tree_cache: Any
) -> None:
    """Raise UnsupportedSpillConfig unless this engine can take part.

    ``role`` is "prefill" or "decode". Called once the transfer manager holds
    the model's KV pools, before the engine serves a request.
    """
    if not _SWITCH.get():
        return
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import get_disagg, get_memory
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

    where = f"{role} engine"
    # Without the raiden manager none of the engine facts below exist.
    _refuse(
        _failed(
            [
                (
                    isinstance(kv_manager, RaidenKVManager),
                    "--disaggregation-transfer-backend must be raiden",
                )
            ]
        ),
        where=where,
    )
    memory = get_memory()
    server_args = scheduler.server_args
    python_tree = isinstance(tree_cache, UnifiedRadixCache) and isinstance(
        tree_cache.tree_core, UnifiedTreeCore
    )
    hicache_on = python_tree and tree_cache.cache_controller is not None
    hybrid_swa = bool(kv_manager._swa_kv_caches)
    requirements = [
        (
            not server_args.disable_radix_cache,
            "the radix cache must stay on (--disable-radix-cache is set)",
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
            kv_manager.pp_size == 1 and not kv_manager._pipeline_enabled,
            "pipeline parallelism and the pipeline transfer are not supported",
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
            not hybrid_swa,
            "hybrid sliding-window models are not supported: only "
            "full-attention KV is offloaded, and such a model cannot resume "
            "from it without its window",
        ),
    ]
    problems = _failed(requirements)
    missing = missing_tpu_sync_host_api()
    if missing:
        problems.append(
            "the installed tpu-sync has no host-to-host read API (missing "
            + ", ".join(missing)
            + ")"
        )
    _refuse(problems, where=where)


def peer_switch_mismatch(*, peer_role: str, peer_enabled: bool) -> Optional[str]:
    """Why a pair cannot run: the switch is on for one role only."""
    mine = _SWITCH.get()
    if mine == bool(peer_enabled):
        return None
    on, off = (
        ("this engine", f"its {peer_role}")
        if mine
        else (f"its {peer_role}", "this engine")
    )
    return (
        f"{_SWITCH.name} is on for {on} and off for {off}; "
        "set it to the same value on both roles"
    )


def resharding_refusal() -> Optional[str]:
    """Why a prefill with the switch on refuses a decode that asks for a
    resharded (different attention TP) transfer, or None with the switch off."""
    if not _SWITCH.get():
        return None
    return (
        f"{_SWITCH.name}=1 does not support resharding: the decode "
        "runs at a different attention TP size than this prefill; use equal "
        f"attention TP on both roles or unset {_SWITCH.name}"
    )
