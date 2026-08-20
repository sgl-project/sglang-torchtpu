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

# Append instead of setdefault so caller-supplied flags are preserved.
_SPARSE_CORE_OFFLOAD_OVERRIDES = (
    # TODO(wyzhang): drop after the torchtpu fix.
    # SparseCore offload for the all-reduce after embed lookup can return
    # wrong values inside compiled regions on TP=8 dense Qwen.
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
)

_SMEM_NEGOTIATION_OVERRIDES = (
    # Let libtpu size SMEM to what a kernel's scheduling metadata actually
    # needs. The Gated-DeltaNet kernel keeps its per-tile plan
    # (p_id_to_s_idx / _r_base / _r_size / _is_first_tile / _is_last_tile) in
    # SMEM, sized by the padded token count, so a long prefill overflows the
    # static budget and the arrays spill. Measured on v7-8 at Qwen3.5-397B
    # per-rank geometry (2 kq / 8 v heads, head dim 128), a 32768-token
    # prefill costs 11.96 ms/layer without this flag and 1.93 ms with it --
    # 6.2x, or roughly 450 ms per step across the 45 linear layers. Shapes up
    # to 16384 tokens fit the static budget and are unaffected.
    "--xla_tpu_use_dynamic_smem_negotiation=true",
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
        _SPARSE_CORE_OFFLOAD_OVERRIDES
        + _ASSOCIATIVE_SCAN_OVERRIDES
        + _SMEM_NEGOTIATION_OVERRIDES
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
