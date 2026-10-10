# SPDX-License-Identifier: Apache-2.0
"""TPU env-var overrides.

Call ``apply_tpu_env_overrides()`` from TPU setup before ``torch_tpu`` is
imported. libtpu reads ``LIBTPU_INIT_ARGS`` at load time.
"""

from __future__ import annotations

import os

from sglang_tpu._logging import get_logger

logger = get_logger(__name__)

_LIBTPU_INIT_ARGS = "LIBTPU_INIT_ARGS"

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


def apply_tpu_env_overrides() -> None:
    """Apply TPU env overrides."""
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
