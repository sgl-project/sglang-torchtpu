# SPDX-License-Identifier: Apache-2.0
"""Muse Spark prefill program, SP + TP prototype: prefill/spep's sequence-parallel dense path
with prefill/tp's expert-TP fused-RS MoE (not wired into serving)."""

from sglang_tpu.muse_megakernel_v2.prefill.sptp.program import (  # noqa: F401
    build_forward,
    n_expert_refs,
)
