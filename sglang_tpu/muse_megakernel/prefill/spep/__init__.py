# SPDX-License-Identifier: Apache-2.0
"""Muse Spark prefill program in the sequence-parallel layout (SGLANG_TPU_MUSE_MK_PREFILL_SP)."""

from sglang_tpu.muse_megakernel.prefill.spep.program import (  # noqa: F401
    build_forward,
    n_expert_refs,
)
