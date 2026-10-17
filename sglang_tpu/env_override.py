# SPDX-License-Identifier: Apache-2.0
"""TPU env-var overrides.

Call ``apply_tpu_env_overrides()`` from TPU setup before ``torch_tpu`` is
imported. libtpu reads ``LIBTPU_INIT_ARGS`` at load time.
"""

from __future__ import annotations

import os
import sys

from sglang_tpu._logging import get_logger

logger = get_logger(__name__)

_LIBTPU_INIT_ARGS = "LIBTPU_INIT_ARGS"

_TPU_PREMAPPED_BUFFER_SIZE = "TPU_PREMAPPED_BUFFER_SIZE"
# Default size for libtpu's premapped (pinned, DMA-mapped) host staging pool.
_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
# PID of the process that wrote the default after it had imported torch. In
# that process torch_tpu has already read the variable as unset. Spawned
# workers import this module again, so they start with None.
_premapped_default_writer_pid: int | None = None

# Mirror legacy compilation-cache env vars to their renamed counterparts.
# TODO(wyzhang): Remove after the TorchTPU upgrade completes.
_COMPILATION_CACHE_ENV_RENAMES = (
    (
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE",
        "TORCH_TPU_TIER2_COMPILATION_CACHE",
    ),
    (
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
    ),
)

# Append instead of setdefault so caller-supplied flags are preserved.
_SPARSE_CORE_OFFLOAD_OVERRIDES = (
    # TODO(wyzhang): drop after the torchtpu fix.
    # SparseCore offload for the all-reduce after embed lookup can return
    # wrong values inside compiled regions on TP=8 dense Qwen.
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
)

_ASSOCIATIVE_SCAN_OVERRIDES = (
    # TODO: drop once google-pytorch/torch_tpu#2898 is fixed.
    # libtpu >= 0.0.44 miscompiles `scan(..., is_associative=true)`, so every
    # cumulative op (cumsum/cumprod/cummax/cummin/logcumsumexp) silently returns
    # wrong values when the scanned length is not a multiple of 1024.
    "--xla_tpu_use_associative_scan_emitter=false",
)

_VMEM_PREFETCH_OVERRIDES = (
    # TODO: drop once the XLA:TPU memory-space-assignment scheduling issue is
    # fixed. With more than one outstanding VMEM prefetch, sufficiently large
    # compiled programs (e.g. the full 36-layer gpt-oss forward at TP=8)
    # deadlock the device on their first execute when the KV pool is small:
    # every core enters the program and none completes. Serializing the
    # prefetches avoids the deadlock; memory-space assignment itself stays
    # enabled. Applied only for small explicit KV pools (see
    # apply_small_kv_pool_vmem_prefetch_override) so unaffected deployments
    # keep the default prefetch schedule bit-for-bit. The gf (ghostfish)
    # flag namespace is the one tpu7x's compiler backend reads.
    "--xla_gf_vmem_max_outstanding_prefetches=1",
)

_PREFILL_CP_SPARSE_CORE_OVERRIDES = (
    # TODO(yinlin09): drop once libtpu is fixed. In the prefill context-parallel
    # forward, which all-gathers K/V on every layer, the SparseCore gather
    # offload returns wrong tokens on some prompts and the SparseCore
    # all-gather offload halts a SparseCore core. The wrong tokens and a
    # SparseCore halt here first appear in libtpu 0.0.46 and remain in 0.0.49.
    "--xla_tpu_offload_gather_to_sparsecore=false",
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false",
)

_DECODE_CP_SPARSE_CORE_OVERRIDES = (
    # TODO(yinlin09): drop once libtpu is fixed. With decode context parallelism
    # the SparseCore all-gather offload returns a wrong first token after long
    # prompts (libtpu 0.0.47, 8192-token prefill bucket); every validated DCP run
    # also kept the SparseCore gather offload off.
    "--xla_tpu_offload_gather_to_sparsecore=false",
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false",
)

# Smallest KV pool (in tokens) observed safe with the default prefetch depth
# on the largest supported dense stack (36-layer gpt-oss at TP=8): 4096 pages
# of 64 tokens. Pools at or above this never deadlocked; below it the first
# EXTEND execute hangs the device.
_VMEM_PREFETCH_SAFE_MIN_TOKENS = 4096 * 64
# At the smallest pools, 34 layers were clean while 36 hung; models below
# this layer count never deadlocked at any pool size and keep the default
# prefetch schedule.
_VMEM_PREFETCH_MIN_LAYERS = 35


def _flag_name(flag: str) -> str:
    return flag.split("=", 1)[0]


def _override_libtpu_flags(flags: tuple[str, ...]) -> list[str]:
    """Append ``flags`` to LIBTPU_INIT_ARGS; caller-supplied values win.

    A flag already present under the same name — even with a different
    value — is left untouched, so users can override any default here.
    """
    existing = os.environ.get(_LIBTPU_INIT_ARGS, "")
    existing_names = {_flag_name(f) for f in existing.split()}
    to_append = [f for f in flags if _flag_name(f) not in existing_names]
    if not to_append:
        return []

    os.environ[_LIBTPU_INIT_ARGS] = f"{existing} {' '.join(to_append)}".strip()
    return to_append


def _set_renamed_compilation_cache_envs() -> list[str]:
    added = []
    for old_name, new_name in _COMPILATION_CACHE_ENV_RENAMES:
        if old_name in os.environ and new_name not in os.environ:
            os.environ[new_name] = os.environ[old_name]
            added.append(new_name)
    return added


def _set_default_premapped_buffer_size() -> bool:
    """Default ``TPU_PREMAPPED_BUFFER_SIZE`` for processes spawned after this call.

    torch_tpu reads the variable once, when ``import torch`` autoloads it, and
    if it is unset writes ``"0"`` to the C environment, which turns the pool
    off. ``os.environ`` does not see that write, but child processes inherit
    it. So a worker that sets the variable itself is too late. The launcher
    has to set it before it spawns workers; the ``os.environ`` write also
    replaces the inherited ``"0"`` through ``putenv``. A value the user set,
    including ``0``, is kept. Returns whether the default was written.
    """
    global _premapped_default_writer_pid
    if _TPU_PREMAPPED_BUFFER_SIZE in os.environ:
        return False
    os.environ[_TPU_PREMAPPED_BUFFER_SIZE] = str(_DEFAULT_TPU_PREMAPPED_BUFFER_SIZE)
    if "torch" in sys.modules:
        _premapped_default_writer_pid = os.getpid()
    return True


def premapped_buffer_size_seen_by_torch_tpu() -> str | None:
    """``TPU_PREMAPPED_BUFFER_SIZE`` as torch_tpu read it in this process.

    Returns None when torch_tpu read it unset and turned the pool off. That
    includes the case where this process wrote the default only after
    importing torch, even though ``os.environ`` now holds the default.
    """
    if _premapped_default_writer_pid == os.getpid():
        return None
    return os.environ.get(_TPU_PREMAPPED_BUFFER_SIZE)


def apply_tpu_env_overrides() -> None:
    """Apply TPU env overrides."""
    if _set_default_premapped_buffer_size():
        logger.info(
            "TPU env_override set %s=%d for spawned workers.",
            _TPU_PREMAPPED_BUFFER_SIZE,
            _DEFAULT_TPU_PREMAPPED_BUFFER_SIZE,
        )

    added = _set_renamed_compilation_cache_envs()
    if added:
        logger.info(
            "TPU env_override set renamed compilation-cache env var(s): %s",
            " ".join(added),
        )

    appended = _override_libtpu_flags(
        _SPARSE_CORE_OFFLOAD_OVERRIDES + _ASSOCIATIVE_SCAN_OVERRIDES
    )
    if appended:
        logger.info(
            "TPU env_override appended %d LIBTPU_INIT_ARGS flag(s): %s",
            len(appended),
            " ".join(appended),
        )
    else:
        logger.debug(
            "TPU env_override: LIBTPU_INIT_ARGS already has the override flags"
        )


def apply_small_kv_pool_vmem_prefetch_override(
    *, max_total_tokens: int | None, num_hidden_layers: int
) -> None:
    """Serialize VMEM prefetches when the deadlock conditions hold.

    Called with the resolved ``--max-total-tokens`` and the model's layer
    count before workers spawn, so they inherit the flags. A ``None`` or
    large pool — or a model below the measured layer threshold — leaves the
    default prefetch schedule untouched.
    """
    if max_total_tokens is None or max_total_tokens >= _VMEM_PREFETCH_SAFE_MIN_TOKENS:
        return
    if num_hidden_layers < _VMEM_PREFETCH_MIN_LAYERS:
        return
    appended = _override_libtpu_flags(_VMEM_PREFETCH_OVERRIDES)
    if appended:
        logger.info(
            "TPU env_override: KV pool of %d tokens is below %d; serializing "
            "VMEM prefetches (%s)",
            max_total_tokens,
            _VMEM_PREFETCH_SAFE_MIN_TOKENS,
            " ".join(appended),
        )


def apply_prefill_cp_sparse_core_override(*, attn_cp_size: int) -> None:
    """Turn off the SparseCore offloads that break prefill context parallelism.

    Called with the resolved ``--attn-cp-size`` before workers spawn, so they
    inherit the flags; without context parallelism the offloads keep the
    libtpu defaults.
    """
    if attn_cp_size <= 1:
        return
    appended = _override_libtpu_flags(_PREFILL_CP_SPARSE_CORE_OVERRIDES)
    if appended:
        logger.info(
            "TPU env_override: prefill context parallelism (--attn-cp-size %d); "
            "turning off SparseCore offloads (%s)",
            attn_cp_size,
            " ".join(appended),
        )


def apply_decode_cp_sparse_core_override(*, dcp_size: int) -> None:
    """Turn off the SparseCore offloads that break decode context parallelism.

    Called with the resolved ``--dcp-size`` before workers spawn, so they
    inherit the flags; without decode context parallelism the offloads keep
    the libtpu defaults.
    """
    if dcp_size <= 1:
        return
    appended = _override_libtpu_flags(_DECODE_CP_SPARSE_CORE_OVERRIDES)
    if appended:
        logger.info(
            "TPU env_override: decode context parallelism (--dcp-size %d); "
            "turning off SparseCore offloads (%s)",
            dcp_size,
            " ".join(appended),
        )
