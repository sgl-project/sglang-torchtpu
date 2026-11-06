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


def _args_problems(*, role: str, scheduler: Any) -> List[str]:
    """Settings known before the transfer manager exists."""
    from sglang.srt.runtime_context import get_device, get_disagg, get_memory

    memory = get_memory()
    server_args = scheduler.server_args
    requirements = [
        (
            server_args.disaggregation_transfer_backend == "raiden",
            "--disaggregation-transfer-backend must be raiden",
        ),
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
            server_args.enable_hierarchical_cache,
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
            server_args.pp_size == 1,
            f"pipeline parallelism is not supported (--pp-size {server_args.pp_size})",
        ),
        (
            # A decode opts in to consume from a pipeline prefill with the env.
            role != "decode" or not tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.get(),
            "the Raiden pipeline transfer is not supported "
            f"({tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.name} is set)",
        ),
        (
            server_args.dp_size == 1 and server_args.attn_dp_size == 1,
            "data parallelism is not supported",
        ),
        (
            scheduler.spec_algorithm.is_none(),
            "speculative decoding is not supported",
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
    return problems


def _manager_problems(*, kv_manager: Any, tree_cache: Any) -> List[str]:
    """Facts only the transfer manager and the tree hold."""
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

    if not isinstance(kv_manager, RaidenKVManager):
        return ["--disaggregation-transfer-backend must be raiden"]
    python_tree = isinstance(tree_cache, UnifiedRadixCache) and isinstance(
        tree_cache.tree_core, UnifiedTreeCore
    )
    requirements = [
        (
            python_tree,
            "the Python unified radix tree is required (the Rust tree core is not supported)",
        ),
        (
            python_tree and tree_cache.cache_controller is not None,
            "--enable-hierarchical-cache is required on both roles",
        ),
        (
            kv_manager.attn_cp_size == 1,
            "context parallelism is not supported",
        ),
        (
            not kv_manager._state_kv_caches,
            "models with recurrent state (GDN/Mamba) are not supported",
        ),
        (
            not kv_manager._swa_kv_caches,
            "hybrid sliding-window models are not supported: only "
            "full-attention KV is offloaded, and such a model cannot resume "
            "from it without its window",
        ),
    ]
    return [reason for ok, reason in requirements if not ok]


def validate_engine_args(*, role: str, scheduler: Any) -> None:
    """Raise UnsupportedDramOffloadConfig for a setting known from the server
    arguments alone. Called before the transfer manager is built: a refusal
    then leaves no transport thread behind, and the engine exits at once
    (a pipeline transfer started before the check held the engine's log
    pipe for minutes after the refusal)."""
    if not _SWITCH.get():
        return
    _refuse(_args_problems(role=role, scheduler=scheduler), where=f"{role} engine")


def validate_engine(*, role: str, kv_manager: Any, tree_cache: Any) -> None:
    """Raise UnsupportedDramOffloadConfig for a setting only the transfer
    manager and the tree show. ``role`` is "prefill" or "decode". Called once
    the manager holds the model's KV pools, after ``validate_engine_args``."""
    if not _SWITCH.get():
        return
    _refuse(
        _manager_problems(kv_manager=kv_manager, tree_cache=tree_cache),
        where=f"{role} engine",
    )
