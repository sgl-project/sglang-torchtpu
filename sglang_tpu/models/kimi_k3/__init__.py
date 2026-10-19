# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3-specific TPU code.

The static fp8 part selector lives here, apart from the binder, so the server
can validate SGLANG_TPU_K3_STATIC_FP8 at startup without importing the fp8
linear code.
"""

from __future__ import annotations

from sglang_tpu.envs import envs as tpu_envs

STATIC_FP8_PARTS = frozenset({"moe_front", "latent_up", "kda_qkvg", "mla_qkv_a"})
# What "all" selects: every validated part. mla_qkv_a has no accuracy data yet
# and its error lands in the cached KV latent, so it must be named explicitly.
STATIC_FP8_ALL_PARTS = STATIC_FP8_PARTS - {"mla_qkv_a"}


def static_fp8_parts() -> frozenset[str]:
    """The GEMVs SGLANG_TPU_K3_STATIC_FP8 selects; raises on unknown names."""
    parts = set(tpu_envs.SGLANG_TPU_K3_STATIC_FP8.get())
    unknown = parts - STATIC_FP8_PARTS - {"all"}
    if unknown:
        raise ValueError(
            f"SGLANG_TPU_K3_STATIC_FP8: unknown {sorted(unknown)}; expected a "
            f"subset of {sorted(STATIC_FP8_PARTS)} or 'all'"
        )
    if "all" in parts:
        parts = (parts - {"all"}) | STATIC_FP8_ALL_PARTS
    # The host split slices the static fp8 copies of these two GEMVs.
    missing = {"moe_front", "latent_up"} - parts
    if tpu_envs.SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV.get() and missing:
        raise ValueError(
            "SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV needs SGLANG_TPU_K3_STATIC_FP8 "
            f"to include {sorted(missing)}"
        )
    return frozenset(parts)
