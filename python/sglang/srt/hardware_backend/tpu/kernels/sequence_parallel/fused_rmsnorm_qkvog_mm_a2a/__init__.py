# SPDX-License-Identifier: Apache-2.0

"""Fused SP prefill RMSNorm, replicated-weight QKVOG matmul, and A2A."""

from ._schedule import supports_token_count

_ENTRYPOINT_EXPORTS = {
    "fused_rmsnorm_qkvog_mm_a2a",
    "jax_rmsnorm_qkvog_mm_a2a",
    "pack_qkvog_weight_by_destination",
}


def __getattr__(name: str):
    """Import JAX only when a kernel entrypoint is actually requested."""
    if name not in _ENTRYPOINT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .fused_rmsnorm_qkvog_mm_a2a_entrypoint import (
        fused_rmsnorm_qkvog_mm_a2a,
        jax_rmsnorm_qkvog_mm_a2a,
        pack_qkvog_weight_by_destination,
    )

    exports = {
        "fused_rmsnorm_qkvog_mm_a2a": fused_rmsnorm_qkvog_mm_a2a,
        "jax_rmsnorm_qkvog_mm_a2a": jax_rmsnorm_qkvog_mm_a2a,
        "pack_qkvog_weight_by_destination": pack_qkvog_weight_by_destination,
    }
    globals().update(exports)
    return exports[name]


__all__ = [
    "fused_rmsnorm_qkvog_mm_a2a",
    "jax_rmsnorm_qkvog_mm_a2a",
    "pack_qkvog_weight_by_destination",
    "supports_token_count",
]
