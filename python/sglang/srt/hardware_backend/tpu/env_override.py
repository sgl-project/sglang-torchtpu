# SPDX-License-Identifier: Apache-2.0
"""TPU env-var overrides.

Call ``apply_tpu_env_overrides()`` from TPU setup before ``torch_tpu`` is
imported. libtpu reads ``LIBTPU_INIT_ARGS`` at load time.
"""

from __future__ import annotations

import os

from sglang.srt.hardware_backend.tpu._logging import get_logger

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

_SPARSE_CORE_OFFLOAD_OVERRIDES = (
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
    "--xla_tpu_offload_gather_to_sparsecore=false",
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false",
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
)

_ASSOCIATIVE_SCAN_OVERRIDES = (
    # TODO: drop once google-pytorch/torch_tpu#2898 is fixed.
    # libtpu >= 0.0.44 miscompiles `scan(..., is_associative=true)`, so every
    # cumulative op (cumsum/cumprod/cummax/cummin/logcumsumexp) silently returns
    # wrong values when the scanned length is not a multiple of 1024.
    "--xla_tpu_use_associative_scan_emitter=false",
)


def _override_libtpu_flags(flags: tuple[str, ...]) -> list[str]:
    existing = os.environ.get(_LIBTPU_INIT_ARGS, "")
    existing_flags = set(existing.split())
    to_append = [f for f in flags if f not in existing_flags]
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
