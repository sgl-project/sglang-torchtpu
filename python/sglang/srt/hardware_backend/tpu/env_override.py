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

# Append instead of setdefault so caller-supplied flags are preserved.
_SPARSE_CORE_OFFLOAD_OVERRIDES = (
    # TODO(wyzhang): drop after the torchtpu fix.
    # SparseCore offload for the all-reduce after embed lookup can return
    # wrong values inside compiled regions on TP=8 dense Qwen.
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
)

# libtpu 0.0.42.1 compiles the batched RPA attention kernel (--attention-backend
# CUSTOM) ~7x slower without dynamic SMEM negotiation: it can't fit the kernel's
# scheduling metadata in the default SMEM budget and falls back to a slow codegen
# path. Enabling it restores fast compilation. Compile-time only; drop once the
# upstream batched-RPA SMEM optimization lands. See google-pytorch/torchtpu-vllm#370.
_SMEM_NEGOTIATION_OVERRIDES = (
    "--xla_tpu_use_dynamic_smem_negotiation=true",
)


def _override_libtpu_flags(flags: tuple[str, ...]) -> list[str]:
    existing = os.environ.get(_LIBTPU_INIT_ARGS, "")
    existing_flags = set(existing.split())
    to_append = [f for f in flags if f not in existing_flags]
    if not to_append:
        return []

    os.environ[_LIBTPU_INIT_ARGS] = f"{existing} {' '.join(to_append)}".strip()
    return to_append


def apply_tpu_env_overrides() -> None:
    """Apply TPU env overrides."""
    appended = _override_libtpu_flags(
        _SPARSE_CORE_OFFLOAD_OVERRIDES + _SMEM_NEGOTIATION_OVERRIDES
    )
    if appended:
        logger.info(
            "TPU env_override appended %d LIBTPU_INIT_ARGS flag(s): %s",
            len(appended),
            " ".join(appended),
        )
    else:
        logger.debug("TPU env_override: LIBTPU_INIT_ARGS already has the override flags")
