# SPDX-License-Identifier: Apache-2.0
"""The PP prefill stage's experts: sglang_tpu's fused MoE (sort, gather, gmm_v2 with fused SiLU,
gmm_v2, un-permute), called as the native Muse layer calls it (no combine)."""

import jax
import jax.numpy as jnp

from sglang_tpu.muse_megakernel.prefill_pp.config import BF16, PPConfig


def experts(c: PPConfig, xe, ids, tw, ex, use_sc_gather: int = 1):
    """[T, top_k, He] bf16 slot outputs of xe [T, He] bf16 routed by ids / tw [T, >= top_k]
    through ex = (w13, w13_scale, w2, w2_scale)."""
    # fused_moe_op loads torch_tpu at import; deviceless compiles of the rest go without it
    from sglang_tpu.moe.pallas.fused_moe_op import _fused_moe_jax

    w13, s13, w2, s2 = ex
    return _fused_moe_jax(
        xe, w13, w2, tw[:, : c.top_k].astype(BF16), ids[:, : c.top_k], s13, s2, None, None,
        c.top_k, "silu", no_combine=True, use_sc_gather=use_sc_gather,
    )  # fmt: skip


def slot_experts(c: PPConfig, slot: int, is_fp8, xe, ids, tw, ex_fp4, ex_fp8):
    """One slot's experts. A slot that holds the stage's fp8 layer on some stage picks the set
    at run time; its SparseCore gather cannot sit in a conditional, so there it gathers in XLA.
    Elsewhere the call is the served one."""
    if slot not in c.fp8_slots:
        return experts(c, xe, ids, tw, ex_fp4)
    return jax.lax.cond(
        is_fp8,
        lambda: experts(c, xe, ids, tw, ex_fp8, use_sc_gather=0),
        lambda: experts(c, xe, ids, tw, ex_fp4, use_sc_gather=0),
    )


def no_experts(c: PPConfig):
    """MUSE_PP_SKIP=experts: zeros (timing ablation only)."""
    return jnp.zeros((c.rows, c.top_k, c.expert_hidden), BF16)
