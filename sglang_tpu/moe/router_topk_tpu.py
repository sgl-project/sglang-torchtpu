# SPDX-License-Identifier: Apache-2.0
"""Fused Pallas router top-k for small batches on TPU.

``biased_topk_impl`` (sigmoid scores + correction bias, then top-k) lowers on
TPU to an XLA sort over all experts plus a gather and reductions: 7 us per
MoE layer for Kimi-K3's top-16 of 896 at batch 1. This kernel does sigmoid,
the biased choice, k rounds of argmax-and-mask, the weight gather and the
renormalization in VMEM (sglang_tpu/kernels/router_topk.py); this module
registers it as an op and routes eligible ``biased_topk_impl`` calls to it.

Ties pick the lowest expert id, which ``torch.topk`` does not promise; the
selected set differs from the sort only on exactly tied scores.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

from threading import Lock
from typing import Callable

import torch

# Batch sizes the kernel serves; larger batches keep the sort.
MAX_TOKENS = 64
# Expert counts validated against biased_topk_impl on TPU: Kimi-K3 (top-16 of
# 896) and GLM-5.3-Flash (top-8 of 288). Other sigmoid routers (e.g. Kimi-K2.x
# with 384) keep the sort until they are measured.
VALIDATED_NUM_EXPERTS = frozenset({288, 896})

_OP: list = []
_OP_LOCK = Lock()
# Plain bool: Dynamo traces the eligibility check.
_OP_READY = False


def _build_op() -> Callable:
    import jax  # noqa: F401  (force loud failure if JAX is missing)

    from sglang_tpu.kernels.router_topk import router_topk_jax
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    def _wrapped(
        logits: jax.Array,
        bias: jax.Array,
        k: int,
        renormalize: bool,
        eps: float,
        scale: float,
    ) -> tuple[jax.Array, jax.Array]:
        return router_topk_jax(logits, bias, k, renormalize, eps, scale)

    def _fake(logits, bias, k, renormalize, eps, scale):
        t = logits.shape[0]
        return (
            logits.new_empty((t, k), dtype=torch.float32),
            logits.new_empty((t, k), dtype=torch.int32),
        )

    op = cpu_export_jax_op("sglang_tpu_pallas::router_topk", _wrapped)
    op.register_fake(_fake)
    return op


def register_router_topk_op() -> None:
    """Register the op when enabled; must run before torch.compile traces the
    model. Left unregistered, every call keeps the generic top-k."""
    global _OP_READY
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_ENABLE_ROUTER_TOPK_KERNEL.get():
        return
    with _OP_LOCK:
        if not _OP:
            _OP.append(_build_op())
        _OP_READY = True


def _eligible(gating_output, correction_bias, topk, num_fused_shared_experts) -> bool:
    return (
        _OP_READY
        and num_fused_shared_experts == 0
        and correction_bias is not None
        and correction_bias.dim() == 1
        # The kernel adds scores + bias in f32, as torch does for an f32 bias
        # (Kimi-K3 casts its bias to f32 post-load). A bf16 bias would round
        # the sum to bf16 in torch, so it keeps the sort.
        and correction_bias.dtype == torch.float32
        and gating_output.dim() == 2
        and 0 < gating_output.shape[0] <= MAX_TOKENS
        and gating_output.shape[1] in VALIDATED_NUM_EXPERTS
        and topk <= 128
    )


def _run(
    gating_output,
    correction_bias,
    topk,
    renormalize,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
):
    from sglang.srt.layers.moe.topk import _RENORMALIZE_SUM_EPSILON

    scale = (
        float(routed_scaling_factor)
        if renormalize
        and apply_routed_scaling_factor_on_output
        and routed_scaling_factor is not None
        else 1.0
    )
    return _OP[0](
        gating_output,
        correction_bias,
        topk,
        bool(renormalize),
        float(_RENORMALIZE_SUM_EPSILON),
        scale,
    )


def biased_topk_around(
    original,
    hidden_states,
    gating_output,
    correction_bias,
    topk,
    renormalize,
    scoring_func="sigmoid",
    num_fused_shared_experts=0,
    routed_scaling_factor=None,
    num_token_non_padded=None,
    expert_location_dispatch_info=None,
    apply_routed_scaling_factor_on_output=False,
):
    """AROUND hook for ``topk.biased_topk_impl``."""
    if not (
        scoring_func == "sigmoid"
        and expert_location_dispatch_info is None
        and _eligible(gating_output, correction_bias, topk, num_fused_shared_experts)
    ):
        return original(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            scoring_func,
            num_fused_shared_experts,
            routed_scaling_factor,
            num_token_non_padded,
            expert_location_dispatch_info,
            apply_routed_scaling_factor_on_output,
        )
    return _run(
        gating_output,
        correction_bias,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


def biased_grouped_topk_around(
    original,
    hidden_states,
    gating_output,
    correction_bias,
    topk,
    renormalize,
    num_expert_group=None,
    topk_group=None,
    num_fused_shared_experts=0,
    routed_scaling_factor=None,
    apply_routed_scaling_factor_on_output=False,
):
    """AROUND hook for ``topk.biased_grouped_topk_impl``: with one expert group
    the group mask keeps every expert, so it is the plain biased top-k."""
    if not (
        num_expert_group in (None, 1)
        and topk_group in (None, 1)
        and _eligible(gating_output, correction_bias, topk, num_fused_shared_experts)
    ):
        return original(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
    return _run(
        gating_output,
        correction_bias,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
