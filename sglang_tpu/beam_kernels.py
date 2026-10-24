# SPDX-License-Identifier: Apache-2.0
"""Install the Beam Pallas kernels behind the model's pluggable ops.

``python/sglang/srt/models/beam.py`` exposes its Beam-specific blocks as
``BaseFusedOp`` subclasses whose ``forward_native`` is the reference math, and
resolves an accelerated forward from ``BaseFusedOp``'s registry under the
platform dispatch key ("tpu"). This module fills that registry from
``TpuSRTPlatform.init_backend()``, before any model is built, so core gains no
TPU branch; the Pallas ops themselves are built only for models that use them.

| model op                         | kernel                                    |
|----------------------------------|-------------------------------------------|
| ``BeamSandwichResidualRMSNormOp``| ``tpu_sandwich_residual_rmsnorm`` (K1)    |
| ``BeamRMSNormOp``                | ``tpu_rmsnorm`` (K1)                      |
| ``BeamQKNormPositionOp``         | ``tpu_fused_qk_norm_position`` (K2)       |
| ``BeamGroupedTopKOp``            | ``tpu_biased_grouped_topk`` (K4)          |

``BeamAttentionGateOp`` has no kernel; XLA fuses its native elementwise form.

Each adapter takes the kernel only inside its validated envelope (bf16 rows,
at least one token, head_dim 128, router groups of whole sublane tiles) and
otherwise calls the op's ``forward_native``, so an unusual shape degrades to the
reference instead of failing. ``SGLANG_TPU_BEAM_KERNELS=0`` skips installation
entirely (reference path everywhere) for A/B and bisection.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_DISPATCH_KEY = "tpu"
_installed = False


def _sandwich(op, branch, residual, post_weight, next_weight, eps, scales):
    if branch.dtype != torch.bfloat16 or branch.dim() != 2 or branch.shape[0] == 0:
        return op.forward_native(
            branch, residual, post_weight, next_weight, eps, scales
        )
    from sglang_tpu.sandwich_residual_norm import tpu_sandwich_residual_rmsnorm

    return tpu_sandwich_residual_rmsnorm(
        branch, residual.to(branch.dtype), post_weight, next_weight, eps, scales
    )


def _rmsnorm(op, x, weight, eps):
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[0] == 0:
        return op.forward_native(x, weight, eps)
    from sglang_tpu.sandwich_residual_norm import tpu_rmsnorm

    return tpu_rmsnorm(x, weight, eps)


def _qk_norm_position(
    op,
    qkv,
    positions,
    q_weight,
    k_weight,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    cos_sin_cache,
    ssmax_scale,
):
    if qkv.dtype != torch.bfloat16 or qkv.shape[0] == 0 or head_dim != 128:
        return op.forward_native(
            qkv,
            positions,
            q_weight,
            k_weight,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            cos_sin_cache,
            ssmax_scale,
        )
    from sglang_tpu.qk_norm_rope import tpu_fused_qk_norm_position

    if cos_sin_cache is not None:
        q, k, _ = tpu_fused_qk_norm_position(
            qkv,
            q_weight,
            k_weight,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
        )
    else:
        q, k, _ = tpu_fused_qk_norm_position(
            qkv,
            q_weight,
            k_weight,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_scale=ssmax_scale,
        )
    return q, k


def _grouped_topk(
    op,
    hidden_states,
    gating_output,
    correction_bias,
    topk,
    renormalize,
    num_expert_group,
    topk_group,
):
    from sglang_tpu.grouped_topk_router import (
        can_use_tpu_biased_grouped_topk,
        tpu_biased_grouped_topk,
    )

    if gating_output.shape[0] == 0 or not can_use_tpu_biased_grouped_topk(
        num_experts=gating_output.shape[1],
        topk=topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    ):
        return op.forward_native(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
        )
    return tpu_biased_grouped_topk(
        hidden_states,
        gating_output.float(),
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
    )


def build_beam_pallas_ops() -> None:
    """Build the Beam Pallas custom ops (imports JAX). Idempotent."""
    from sglang_tpu.grouped_topk_router import register_tpu_biased_grouped_topk_op
    from sglang_tpu.qk_norm_rope import register_tpu_qk_norm_position_ops
    from sglang_tpu.sandwich_residual_norm import (
        register_tpu_sandwich_residual_norm_ops,
    )

    register_tpu_sandwich_residual_norm_ops()
    register_tpu_qk_norm_position_ops()
    register_tpu_biased_grouped_topk_op()


def _prebuild_for_beam_models(model: torch.nn.Module) -> None:
    from sglang.srt.models.beam import _BeamFusedOp

    if any(isinstance(m, _BeamFusedOp) for m in model.modules()):
        build_beam_pallas_ops()


def register_beam_kernels() -> None:
    """Route Beam's pluggable ops to the Pallas adapters. Idempotent.

    Cheap enough for ``init_backend()``: no JAX import and no op build. The ops
    are built by a pre-compile hook, before torch.compile traces a model that
    contains Beam ops, so other models never pay for them. In eager mode the
    wrappers build them on first call.
    """
    global _installed
    if _installed:
        return
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_BEAM_KERNELS.get():
        logger.info("Beam Pallas kernels disabled (SGLANG_TPU_BEAM_KERNELS=0)")
        _installed = True
        return

    from sglang.kernels.fused_op import BaseFusedOp
    from sglang.srt.models.beam import (
        BeamGroupedTopKOp,
        BeamQKNormPositionOp,
        BeamRMSNormOp,
        BeamSandwichResidualRMSNormOp,
    )
    from sglang_tpu.compilation.pre_compile_hooks import register_pre_compile_hook

    skip = set(tpu_envs.SGLANG_DEBUG_TPU_BEAM_KERNELS_SKIP.get())
    for name, op_cls, fn in (
        ("sandwich", BeamSandwichResidualRMSNormOp, _sandwich),
        ("rmsnorm", BeamRMSNormOp, _rmsnorm),
        ("qk_norm", BeamQKNormPositionOp, _qk_norm_position),
        ("topk", BeamGroupedTopKOp, _grouped_topk),
    ):
        if name in skip:
            logger.info("Beam %s op kept on its torch reference", name)
            continue
        BaseFusedOp.register_oot_forward(op_cls, fn, _DISPATCH_KEY)
    register_pre_compile_hook(_prebuild_for_beam_models)
    _installed = True
