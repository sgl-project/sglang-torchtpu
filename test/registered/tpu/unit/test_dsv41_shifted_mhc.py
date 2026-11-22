# SPDX-License-Identifier: Apache-2.0
"""Shifted mHC TorchTPU bridge against the unfused DeepSeek-V4.1 computation."""

import importlib.util
import os
import sys
from types import SimpleNamespace

# TorchTPU owns the device; JAX exports the custom op from the CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.models.deepseek_v41.kernels.mhc import shifted_post_pre_rms as kernel
from sglang_tpu.models.deepseek_v41.ops import mhc as ops
from sglang_tpu.pallas_export import get_pallas_tpu_device_kind

register_tpu_ci(
    est_time=300,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

M, D = 4, 5120
RMS_EPS, HC_EPS, NORM_EPS, SINKHORN_ITERS = 1e-6, 1e-6, 1e-6, 20
RTOLS = (2**-7, 1e-4, 1e-4, 2**-6, 1e-4)


def test_weight_packing_matches_source_layout():
    weight = torch.randn(24, M * D, generator=torch.Generator().manual_seed(23))
    packed = ops.pack_mix_weight(weight)
    jax_packed = kernel.pack_mix_weight(jnp.asarray(weight.numpy()))
    assert packed.shape == (M * D, 128) and packed.is_contiguous()
    assert packed.dtype == torch.bfloat16
    np.testing.assert_array_equal(
        packed.float().numpy(), np.asarray(jax_packed, np.float32)
    )


def test_direct_weight_update_refreshes_the_retained_buffer(monkeypatch):
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.hc_attn_fn = torch.nn.Parameter(
        torch.zeros((M * (M + 2), M * D)), requires_grad=False
    )
    model.layer.norm = torch.nn.Module()
    model.layer.attn_hc = SimpleNamespace(
        fn=model.layer.hc_attn_fn, norm=model.layer.norm
    )
    retained = ops.pack_mix_weight(model.layer.hc_attn_fn)
    model.layer.norm.register_buffer(
        "_tpu_shifted_mhc_mix_weight", retained, persistent=False
    )
    monkeypatch.setattr(ops, "_shifted_op", object())
    model.layer.hc_attn_fn.fill_(0.1)
    ops.refresh_shifted_mhc_after_direct_update(
        model, [("layer.hc_attn_fn", model.layer.hc_attn_fn)]
    )
    assert model.layer.norm._tpu_shifted_mhc_mix_weight is retained
    torch.testing.assert_close(retained, ops.pack_mix_weight(model.layer.hc_attn_fn))


@pytest.fixture(scope="module")
def tpu():
    if importlib.util.find_spec("torch_tpu") is None:
        pytest.skip("requires TorchTPU (#2367)")
    import torch_tpu  # noqa: F401

    if get_pallas_tpu_device_kind() != "TPU7x":
        pytest.skip("the imported Pallas kernel requires TPU7x (vllm-torchtpu #2876)")
    ops.register_shifted_mhc_op()
    return torch.device("tpu")


def _unfused(y, residual, pre, post, comb, fn, scale, base, norm):
    from sglang.srt.models.deepseek_v4 import _hc_split_sinkhorn_torch

    updated = (
        post[..., None] * y[:, None, :]
        + (comb[..., None] * residual[:, :, None, :]).sum(dim=1)
    ).to(y.dtype)
    flat = updated.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, True) + RMS_EPS)
    next_pre, next_post, next_comb = _hc_split_sinkhorn_torch(
        mixes[:, None, :], scale, base, M, SINKHORN_ITERS, HC_EPS
    )
    combined = (pre[..., None] * updated).sum(dim=1).to(y.dtype).float()
    normalized = (
        combined * torch.rsqrt(combined.square().mean(-1, True) + NORM_EPS) * norm
    ).to(y.dtype)
    return (
        updated,
        next_post[:, 0, :, None],
        next_comb[:, 0],
        normalized,
        next_pre[:, 0],
    )


@pytest.mark.parametrize(
    ("tokens", "check_elements"),
    [(16, True), (240, True), (64, False), (512, False), (1024, False), (2048, False)],
)
def test_fused_boundary_matches_unfused_torch(tpu, tokens, check_elements):
    from sglang.srt.models.deepseek_v4_mhc import HcConfig, HcNextBoundary, HcSubLayer

    rng = torch.Generator().manual_seed(42)

    def normal(*shape):
        return torch.randn(shape, generator=rng)

    y = (normal(tokens, D) * 0.5).bfloat16()
    residual = (normal(tokens, M, D) * 0.5).bfloat16()
    if check_elements:
        # Use noncancelling activations for the per-element rounding budget.
        # The signed cases use the end-to-end relative L2 budget because BF16
        # rounding of the intermediate residual can dominate near-zero values.
        y, residual = y.abs(), residual.abs()
    pre = normal(tokens, M).sigmoid() + HC_EPS
    post = normal(tokens, M).sigmoid() * 2
    comb = normal(tokens, M, M).softmax(-1)
    fn = normal(M * (M + 2), M * D) * 0.01
    scale = torch.tensor([0.1, 0.2, 0.3])
    base = normal(M * (M + 2)) * 0.1
    norm = (1 + 0.1 * normal(D)).bfloat16()
    ref = _unfused(y, residual, pre, post, comb, fn, scale, base, norm)

    norm_module = torch.nn.Module()
    norm_module.register_parameter("weight", torch.nn.Parameter(norm.to(tpu)))
    norm_module.variance_epsilon = NORM_EPS
    norm_module.register_buffer(
        "_tpu_shifted_mhc_mix_weight", ops.pack_mix_weight(fn).to(tpu), persistent=False
    )
    cfg = HcConfig(M, SINKHORN_ITERS, HC_EPS, RMS_EPS, D, True, False)
    next_hc = HcSubLayer(cfg, fn.to(tpu), scale.to(tpu), base.to(tpu), norm_module)
    next_boundary = HcNextBoundary(norm_module, False, True, next_hc)
    coeffs = tuple(tensor.to(tpu) for tensor in (pre, post, comb))

    state = ops.fused_boundary(
        next_hc, y.to(tpu), residual.to(tpu), coeffs, next_boundary, next_hc
    )
    assert state is not None and state.pre is coeffs[0]
    next_pre, next_post, next_comb = state.stats
    actual = (
        state.residual,
        next_post[..., None],
        next_comb,
        state.input.rows,
        next_pre,
    )
    for actual_tensor, expected, rtol in zip(actual, ref, RTOLS, strict=True):
        got = actual_tensor.cpu().float()
        expected = expected.float()
        diff = (got - expected).abs()
        if check_elements:
            bound = torch.maximum(got.abs(), expected.abs()) * rtol + 1e-6
            assert (diff <= bound).all(), (rtol, (diff - bound).amax().item())
        error = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected)
        assert error <= 2e-3, (rtol, error.item())


@pytest.mark.parametrize("tokens", [0, 15])
def test_unsupported_token_bucket_falls_back(tpu, tokens):
    cfg = SimpleNamespace(mult=M, hidden=D)
    hc = SimpleNamespace(cfg=cfg)
    y = torch.empty((tokens, D), device=tpu, dtype=torch.bfloat16)
    assert ops.fused_boundary(hc, y, None, None, None, None) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
