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
    appended = _override_libtpu_flags(_SPARSE_CORE_OFFLOAD_OVERRIDES)
    if appended:
        logger.info(
            "TPU env_override appended %d LIBTPU_INIT_ARGS flag(s): %s",
            len(appended),
            " ".join(appended),
        )
    else:
        logger.debug("TPU env_override: LIBTPU_INIT_ARGS already has SC-off flags")
