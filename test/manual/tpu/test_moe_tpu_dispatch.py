"""Numerics tests for the TPU-tensorized moe_forward_native patch.

Calls `_tpu_tensorized_moe_forward` directly with CPU tensors so the test
runs on any machine (no TPU required). Compares against the reference
`fused_moe_native.moe_forward_native` from sglang upstream.

Cases (all requested by review on PR #3):
  * -1 sentinel masking under EP (expert remapped to -1 contributes 0)
  * multiple token-count shapes
  * silu activation
  * gelu activation
  * silu + gemm1_alpha is not None (GPT-OSS path — exercises the
    interleaved [::2]/[1::2] slicing; layout matches the reference
    `swiglu_gpt_oss_sigmoid_alpha`).

Usage:
    python test/manual/tpu/test_moe_tpu_dispatch.py
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.tpu.patches import (
    _tpu_tensorized_moe_forward,
)
from sglang.srt.layers.moe import fused_moe_native
from sglang.test.test_utils import CustomTestCase


def _make_layer(num_experts: int, hidden: int, inter: int, *, interleaved: bool):
    """Build a duck-typed layer holding w13_weight / w2_weight.

    interleaved=True lays w13 out as [gate0, up0, gate1, up1, ...] along
    the output dim (GPT-OSS convention used by swiglu_gpt_oss_sigmoid_alpha).
    interleaved=False uses [gate_half | up_half] (standard SwiGLU,
    SiluAndMul.forward_native).
    """
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, inter, hidden) * 0.02
    w3 = torch.randn(num_experts, inter, hidden) * 0.02

    if interleaved:
        # [E, inter, hidden] each; interleave along inter to [E, 2*inter, hidden].
        w13 = torch.empty(num_experts, 2 * inter, hidden)
        w13[:, 0::2, :] = w1
        w13[:, 1::2, :] = w3
    else:
        w13 = torch.cat([w1, w3], dim=1)

    w2 = torch.randn(num_experts, hidden, inter) * 0.02
    return SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        num_experts=num_experts,
        moe_ep_size=1,
    )


def _make_topk(num_tokens: int, num_experts: int, topk: int, *, with_neg1: bool):
    """Random top-k assignment. with_neg1=True flips ~30% of slots to -1
    (the EP non-local-expert sentinel)."""
    torch.manual_seed(1)
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, topk), dim=-1)
    if with_neg1:
        mask = torch.rand(num_tokens, topk) < 0.3
        topk_ids = torch.where(
            mask, torch.full_like(topk_ids, -1), topk_ids
        )
    return topk_weights, topk_ids


def _ref_with_neg1(layer, x, topk_weights, topk_ids, moe_runner_config):
    """Reference behavior under -1 masking: drop the -1 slots, run the
    reference path on the kept slots, then scatter results back with zero
    contribution from -1 slots. The reference moe_forward_native does NOT
    handle -1 itself; the standard dispatcher zeros those contributions
    via the post-MoE all-reduce. Mirror that here so the comparison stays
    apples-to-apples."""
    safe_ids = torch.where(
        topk_ids >= 0, topk_ids, torch.zeros_like(topk_ids)
    ).to(torch.long)
    valid = (topk_ids >= 0).to(topk_weights.dtype)
    masked_weights = topk_weights * valid

    return fused_moe_native.moe_forward_native(
        layer,
        x,
        (masked_weights, safe_ids, None),
        moe_runner_config,
    )


def _make_config(activation: str, *, gemm1_alpha=None, gemm1_clamp_limit=None):
    return SimpleNamespace(
        activation=activation,
        gemm1_alpha=gemm1_alpha,
        gemm1_clamp_limit=gemm1_clamp_limit,
        apply_router_weight_on_input=False,
        num_fused_shared_experts=0,
    )


class TestMoeTpuDispatchNumerics(CustomTestCase):
    """Numerics: _tpu_tensorized_moe_forward vs reference moe_forward_native."""

    NUM_EXPERTS = 8
    HIDDEN = 64
    INTER = 32
    TOPK = 2
    TOL = 5e-5

    def _check(
        self,
        *,
        num_tokens,
        activation,
        with_neg1,
        interleaved,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
    ):
        layer = _make_layer(
            self.NUM_EXPERTS, self.HIDDEN, self.INTER, interleaved=interleaved
        )
        x = torch.randn(num_tokens, self.HIDDEN) * 0.1
        topk_weights, topk_ids = _make_topk(
            num_tokens, self.NUM_EXPERTS, self.TOPK, with_neg1=with_neg1
        )
        cfg = _make_config(
            activation,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

        out = _tpu_tensorized_moe_forward(
            layer, x, topk_weights, topk_ids, cfg
        )

        if with_neg1:
            ref = _ref_with_neg1(layer, x, topk_weights, topk_ids, cfg)
        else:
            ref = fused_moe_native.moe_forward_native(
                layer, x, (topk_weights, topk_ids, None), cfg
            )

        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        diff = (out - ref).abs().max().item()
        self.assertLess(diff, self.TOL, f"max diff {diff} > {self.TOL}")

    def test_silu_small_token_count(self):
        self._check(
            num_tokens=7,
            activation="silu",
            with_neg1=False,
            interleaved=False,
        )

    def test_silu_power_of_two_token_count(self):
        self._check(
            num_tokens=16,
            activation="silu",
            with_neg1=False,
            interleaved=False,
        )

    def test_silu_ragged_token_count(self):
        self._check(
            num_tokens=13,
            activation="silu",
            with_neg1=False,
            interleaved=False,
        )

    def test_silu_single_token(self):
        self._check(
            num_tokens=1,
            activation="silu",
            with_neg1=False,
            interleaved=False,
        )

    def test_gelu(self):
        self._check(
            num_tokens=12,
            activation="gelu",
            with_neg1=False,
            interleaved=False,
        )

    def test_silu_gemm1_alpha_gpt_oss(self):
        # GPT-OSS path: interleaved w13 layout; gemm1_alpha + clamp limit.
        # Reference doesn't accept this config directly via moe_forward_native
        # (it bakes the swiglu variant inline), so compare to a hand-rolled
        # equivalent built from the same SwiGLU-OSS formula.
        layer = _make_layer(
            self.NUM_EXPERTS, self.HIDDEN, self.INTER, interleaved=True
        )
        num_tokens = 10
        x = torch.randn(num_tokens, self.HIDDEN) * 0.1
        topk_weights, topk_ids = _make_topk(
            num_tokens, self.NUM_EXPERTS, self.TOPK, with_neg1=False
        )
        cfg = _make_config("silu", gemm1_alpha=1.702, gemm1_clamp_limit=7.0)
        out = _tpu_tensorized_moe_forward(
            layer, x, topk_weights, topk_ids, cfg
        )

        # Hand-rolled reference: per-expert loop, interleaved split, exact
        # formula from swiglu_gpt_oss_sigmoid_alpha.
        ref = torch.zeros_like(out)
        for t in range(num_tokens):
            for k in range(self.TOPK):
                e = int(topk_ids[t, k])
                gate_up = layer.w13_weight[e] @ x[t]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(max=cfg.gemm1_clamp_limit)
                up = up.clamp(
                    min=-cfg.gemm1_clamp_limit, max=cfg.gemm1_clamp_limit
                )
                hidden = (
                    gate * torch.sigmoid(gate * cfg.gemm1_alpha) * (up + 1)
                )
                contrib = layer.w2_weight[e] @ hidden
                ref[t] += topk_weights[t, k] * contrib

        diff = (out - ref).abs().max().item()
        self.assertLess(diff, self.TOL, f"max diff {diff} > {self.TOL}")

    def test_neg1_masking_under_ep(self):
        # Some topk slots flipped to -1 (EP non-local-expert sentinel);
        # TPU path must mask those to zero contribution.
        self._check(
            num_tokens=12,
            activation="silu",
            with_neg1=True,
            interleaved=False,
        )

    def test_fused_shared_under_ep_raises(self):
        layer = _make_layer(
            self.NUM_EXPERTS, self.HIDDEN, self.INTER, interleaved=False
        )
        layer.moe_ep_size = 2
        cfg = _make_config("silu")
        cfg.num_fused_shared_experts = 1
        x = torch.randn(4, self.HIDDEN)
        topk_weights, topk_ids = _make_topk(
            4, self.NUM_EXPERTS, self.TOPK, with_neg1=False
        )
        with self.assertRaises(NotImplementedError):
            _tpu_tensorized_moe_forward(
                layer, x, topk_weights, topk_ids, cfg
            )


if __name__ == "__main__":
    unittest.main()
