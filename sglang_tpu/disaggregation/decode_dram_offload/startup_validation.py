"""Refuse settings the KV offload to decode DRAM does not support.

The offload moves full-attention KV (and a hybrid sliding-window model's last
window) between the two roles' HiCache host pools, rank to rank. Anything it
cannot move or agree on is refused here with the setting to change, at engine
start on either role, before the transfer manager is built. A pair whose
engines disagree fails its requests instead (the raiden channel,
``raiden/dram_offload_channel.py``). Nothing degrades to "no offload" silently.
"""

from __future__ import annotations

import inspect
from typing import Any, List

from sglang_tpu.envs import envs as tpu_envs

_SWITCH = tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM


class UnsupportedDramOffloadConfig(ValueError):
    """The switch is on with a setting the offload does not support."""


def _missing_tpu_sync_host_api() -> List[str]:
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


def _problems(*, role: str, scheduler: Any) -> List[str]:
    from sglang.srt.configs.hybrid_arch import mambaish_config
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import (
        get_device,
        get_disagg,
        get_memory,
        get_parallel,
    )

    memory = get_memory()
    server_args = scheduler.server_args
    model_config = scheduler.model_config
    tree_cache = scheduler.tree_cache
    python_tree = isinstance(tree_cache, UnifiedRadixCache) and isinstance(
        tree_cache.tree_core, UnifiedTreeCore
    )
    # The pool configurator picks the hybrid pools by these predicates
    # (mem_cache/kv_cache_configurator.py), so they name exactly the engines
    # whose transfer manager would hold sliding-window or recurrent-state
    # tensors besides the full-attention KV.
    hybrid_swa = model_config.is_hybrid_swa
    # A pipeline prefill and the tensor-parallel decode it feeds; there the
    # draft sidecar pools can move with the pages (stage_prefill_ops.py).
    pipeline_pairing = (
        server_args.pp_size > 1
        if role == "prefill"
        else bool(tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.get())
    )
    draft_moves = pipeline_pairing and tpu_envs.SGLANG_PD_OFFLOAD_ALLOW_DRAFT.get()
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
            python_tree,
            "the Python unified radix tree is required (the Rust tree core is not supported)",
        ),
        (
            # A pipeline prefill pushes to a tensor-parallel decode, which
            # keeps each stage's pages in that stage's layout
            # (stage_layout_store.py); a decode pipeline has no such store.
            role != "decode" or server_args.pp_size == 1,
            "a pipeline-parallel decode is not supported "
            f"(--pp-size {server_args.pp_size})",
        ),
        (
            # The stages agree through the pipeline's ring, the ranks of a
            # stage would need a second agreement inside it.
            server_args.pp_size == 1
            or int(getattr(get_parallel(), "attn_tp_size", 1)) == 1,
            "a pipeline-parallel prefill needs attention TP 1 (--tp-size 1) "
            f"(--pp-size {server_args.pp_size})",
        ),
        (
            server_args.dp_size == 1 and server_args.attn_dp_size == 1,
            "data parallelism is not supported",
        ),
        (
            get_parallel().attn_cp_size == 1,
            "context parallelism is not supported",
        ),
        (
            # The draft KV of a cached prefix lives in HiCache's draft sidecar
            # pool, by the same host slots as the pages. Neither pairing
            # moves the sidecar, so a recalled prefix would hand the draft
            # model bytes that were never transferred.
            scheduler.spec_algorithm.is_none() or draft_moves,
            "speculative decoding is not supported"
            + (
                f" (a pipeline prefill and its decode can opt in with "
                f"{tpu_envs.SGLANG_PD_OFFLOAD_ALLOW_DRAFT.name}=1 on both roles)"
                if pipeline_pairing
                else ""
            ),
        ),
        (
            mambaish_config(model_config) is None,
            "models with recurrent state (GDN/Mamba) are not supported",
        ),
        (
            # Their compressor state rings travel with neither page set.
            not model_config.is_hybrid_swa_compress,
            "hybrid sliding-window models with compressed window state are "
            "not supported",
        ),
        (
            not hybrid_swa or int(server_args.page_size) > 1,
            "a hybrid sliding-window model needs --page-size > 1",
        ),
        (
            not hybrid_swa or not python_tree or tree_cache.tree_core.has_swa_host_pool,
            "a hybrid sliding-window model needs a host tier of SWA windows, "
            "which this cache was built without",
        ),
    ]
    problems = [reason for ok, reason in requirements if not ok]
    missing = _missing_tpu_sync_host_api()
    if missing:
        problems.append(
            "the installed tpu-sync has no host-to-host read API (missing "
            + ", ".join(missing)
            + ")"
        )
    return problems


def validate_engine_args(*, role: str, scheduler: Any) -> None:
    """Raise UnsupportedDramOffloadConfig for a setting the offload does not
    support. ``role`` is "prefill" or "decode". Called before the transfer
    manager is built: a refusal then leaves no transport thread behind, and
    the engine exits at once (a pipeline transfer started before the check
    held the engine's log pipe for minutes after the refusal)."""
    problems = _problems(role=role, scheduler=scheduler)
    if problems:
        raise UnsupportedDramOffloadConfig(
            f"{_SWITCH.name}=1 is not supported on this {role} engine: "
            + "; ".join(problems)
            + f" (change the setting or unset {_SWITCH.name})"
        )
