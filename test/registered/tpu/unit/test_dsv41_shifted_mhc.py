# SPDX-License-Identifier: Apache-2.0
"""Shifted mHC TorchTPU bridge against the unfused DeepSeek-V4.1 computation."""

import importlib.util
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.envs import envs
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
    with jax.default_device(jax.devices("cpu")[0]):
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


def _case(tokens, *, noncancelling=False, near_zero=False, seed=42):
    rng = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.randn(shape, generator=rng)

    y = (normal(tokens, D) * 0.5).bfloat16()
    residual = (normal(tokens, M, D) * 0.5).bfloat16()
    if noncancelling:
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
    if near_zero:
        y, residual = y / 512, residual / 512
    if tokens == 2048:
        # The real prefill graph can pad a 2044-row workload to this bucket.
        y[-4:].zero_()
        residual[-4:].zero_()
    return y, residual, pre, post, comb, fn, scale, base, norm


def _boundary(fn, scale, base, norm, device):
    from sglang.srt.models.deepseek_v4_mhc import HcConfig, HcNextBoundary, HcSubLayer

    norm_module = torch.nn.Module()
    norm_module.register_parameter("weight", torch.nn.Parameter(norm.to(device)))
    norm_module.variance_epsilon = NORM_EPS
    norm_module.register_buffer(
        "_tpu_shifted_mhc_mix_weight",
        ops.pack_mix_weight(fn).to(device),
        persistent=False,
    )
    cfg = HcConfig(M, SINKHORN_ITERS, HC_EPS, RMS_EPS, D, True, False)
    next_hc = HcSubLayer(
        cfg, fn.to(device), scale.to(device), base.to(device), norm_module
    )
    next_boundary = HcNextBoundary(norm_module, False, True, next_hc)
    return next_hc, next_boundary


def _outputs(state):
    next_pre, next_post, next_comb = state.stats
    return state.residual, next_post[..., None], next_comb, state.input.rows, next_pre


OUTPUT_NAMES = ("residual", "next_post", "next_comb", "normalized", "next_pre")


def _check_outputs(actual, reference, *, check_elements=False):
    for name, actual_tensor, expected, rtol in zip(
        OUTPUT_NAMES, actual, reference, RTOLS, strict=True
    ):
        assert actual_tensor.shape == expected.shape, name
        assert actual_tensor.dtype == expected.dtype, name
        got = actual_tensor.cpu().float()
        expected = expected.float()
        assert got.isfinite().all(), name
        diff = (got - expected).abs()
        if check_elements:
            bound = torch.maximum(got.abs(), expected.abs()) * rtol + 1e-6
            assert (diff <= bound).all(), (name, (diff - bound).amax().item())
        denom = torch.linalg.vector_norm(expected).clamp_min(1e-12)
        error = torch.linalg.vector_norm(diff) / denom
        assert error <= 2e-3, (name, error.item())


@pytest.mark.parametrize(
    ("tokens", "check_elements"),
    [
        (16, True),
        (240, True),
        (32, False),
        (48, False),
        (64, False),
        (96, False),
        (288, False),
        (384, False),
        (512, False),
        (1024, False),
        (2048, False),
        (8192, False),
    ],
)
def test_fused_boundary_matches_unfused_torch(tpu, tokens, check_elements):
    y, residual, pre, post, comb, fn, scale, base, norm = _case(
        tokens, noncancelling=check_elements
    )
    ref = _unfused(y, residual, pre, post, comb, fn, scale, base, norm)
    hc, boundary = _boundary(fn, scale, base, norm, tpu)
    coeffs = tuple(tensor.to(tpu) for tensor in (pre, post, comb))
    with envs.SGLANG_TPU_ENABLE_DSV41_SHIFTED_MHC.override(True):
        state = ops.fused_boundary(
            hc, y.to(tpu), residual.to(tpu), coeffs, boundary, hc
        )
    assert state is not None and state.pre is coeffs[0]
    _check_outputs(_outputs(state), ref, check_elements=check_elements)


def test_direct_trace_chains_boundaries_and_reuses_updated_packed_buffer(tpu):
    from sglang_tpu.compilation.direct_compile import direct_compile

    y, residual, pre, post, comb, fn, scale, base, norm = _case(288, near_zero=True)
    hc, boundary = _boundary(fn, scale, base, norm, tpu)
    norm_module = hc.norm

    def chain(y, residual, pre, post, comb, scale, base):
        live_hc = hc._replace(scale=scale, base=base)
        outputs = []
        for _ in range(3):
            state = ops.fused_boundary(
                live_hc, y, residual, (pre, post, comb), boundary, live_hc
            )
            assert state is not None and state.pre is pre
            outputs.append(_outputs(state))
            y, residual = state.input.rows, state.residual
            pre, post, comb = state.stats
        return tuple(outputs)

    compiled = direct_compile(
        chain,
        model=norm_module,
        specialization_key=lambda *args: args[0].shape,
        name="test/shifted_mhc_chain",
        spmd_safe=True,
    )
    inputs = tuple(
        tensor.to(tpu) for tensor in (y, residual, pre, post, comb, scale, base)
    )

    def check(weight):
        with envs.SGLANG_TPU_ENABLE_DSV41_SHIFTED_MHC.override(True):
            results = compiled(*inputs)
        ref_args = (y, residual, pre, post, comb)
        for actual in results:
            ref = _unfused(*ref_args, weight, scale, base, norm)
            _check_outputs(actual, ref)
            streams, next_post, next_comb, normalized, next_pre = ref
            ref_args = (normalized, streams, next_pre, next_post.squeeze(-1), next_comb)
        return results

    before = check(fn)
    retained = norm_module._tpu_shifted_mhc_mix_weight
    updated_fn = -fn
    ops._refresh_packed_buffer(hc._replace(fn=updated_fn.to(tpu)))
    assert norm_module._tpu_shifted_mhc_mix_weight is retained
    after = check(updated_fn)
    assert len(compiled.cache) == 1
    assert not torch.equal(before[0][-1].cpu(), after[0][-1].cpu())


@pytest.mark.parametrize("tokens", [0, 8, 15, 40])
def test_unsupported_token_bucket_falls_back(tpu, tokens):
    cfg = SimpleNamespace(mult=M, hidden=D)
    hc = SimpleNamespace(cfg=cfg)
    y = torch.empty((tokens, D), device=tpu, dtype=torch.bfloat16)
    with envs.SGLANG_TPU_ENABLE_DSV41_SHIFTED_MHC.override(True):
        assert ops.fused_boundary(hc, y, None, None, None, None) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
