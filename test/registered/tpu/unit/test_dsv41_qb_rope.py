# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import torch

from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.models.deepseek_v41.ops.qb_rope import (
    register_qb_rope_op,
    supports_qb_rope,
)

register_tpu_ci(
    est_time=120,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


def _module():
    linear = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(linear)
    linear.quant_method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    linear.quant_method.use_marlin = False
    linear.quant_method.use_mxfp8 = False
    linear.quant_method.block_fp8_as_mxfp8 = False
    linear.quant_method.block_quant = True
    linear.weight_block_size = None
    linear.gather_output = False
    linear.output_size = linear.output_size_per_partition = 32768
    linear.bias = None
    linear.weight = torch.empty(1280, 32768, device="meta", dtype=torch.float8_e4m3fn)
    linear.weight_scale = torch.empty(32768, 1, device="meta")
    return SimpleNamespace(wq_b=linear, head_dim=512, n_local_heads=64)


def test_fallback_formats_and_linear_behavior():
    layer = _module()
    x = torch.empty(6, 1280, device="meta", dtype=torch.bfloat16)
    pos = torch.empty(6, device="meta", dtype=torch.int64)
    table = torch.empty(2**19, 128, device="meta")
    assert supports_qb_rope(layer, x, pos, table)
    assert not supports_qb_rope(layer, x[:0], pos[:0], table)
    assert not supports_qb_rope(
        layer, x[:1].expand(2049, -1), pos[:1].expand(2049), table
    )
    assert not supports_qb_rope(layer, x.float(), pos, table)
    assert not supports_qb_rope(layer, x, pos[:1], table)
    assert not supports_qb_rope(layer, x, pos, table[:, :32])
    linear = layer.wq_b
    for owner, name, value in (
        (linear, "weight", linear.weight.T),
        (linear, "weight", linear.weight.T.contiguous().T),
        (linear, "weight_scale", linear.weight_scale[:1]),
        (linear, "weight_scale", linear.weight_scale.T),
        (linear, "weight_block_size", [128, 128]),
        (linear, "bias", linear.weight_scale),
        (linear, "output_size", linear.output_size * 2),
        (linear, "gather_output", True),
        (linear.quant_method, "use_mxfp8", True),
        (linear, "quant_method", object()),
    ):
        original = getattr(owner, name)
        setattr(owner, name, value)
        assert not supports_qb_rope(layer, x, pos, table)
        setattr(owner, name, original)
    linear.register_forward_hook(lambda *_: None)
    assert not supports_qb_rope(layer, x, pos, table)


@pytest.mark.skipif(
    importlib.util.find_spec("torch_tpu") is None, reason="requires TorchTPU"
)
@pytest.mark.parametrize(("tokens", "packed"), [(7, False), (96, True)])
def test_qb_rope_matches_channelwise_linear_and_rope(tokens, packed):
    import torch_tpu  # noqa: F401 - registers the TPU device

    from sglang_tpu.dsv4.dsv4_attention_backend import apply_tpu_dsv4_rope
    from sglang_tpu.fp8_linear import apply_tpu_fp8_linear
    from sglang_tpu.pallas_export import get_pallas_tpu_device_kind

    if get_pallas_tpu_device_kind() != "TPU7x":
        pytest.skip("requires TPU7x")
    generator = torch.Generator().manual_seed(137)
    x = torch.randn(tokens, 1280, generator=generator).bfloat16()
    x[0] = 0
    x[1] *= 1e-11
    weight = (torch.randn(1280, 32768, generator=generator) * 28).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(32768, 1, generator=generator) * 0.03
    angles = torch.randn(4096, 32, generator=generator)
    table = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if packed:
        table = table.reshape(-1, 128)
    positions = torch.randint(4096, (tokens,), generator=generator, dtype=torch.int32)
    positions[:3] = torch.tensor([4095, 4094, 0])
    x, weight, scale, positions, table = (
        t.to("tpu") for t in (x, weight, scale, positions, table)
    )
    expected = apply_tpu_dsv4_rope(
        apply_tpu_fp8_linear(x, weight, scale).reshape(tokens, 64, 512),
        positions,
        table,
    )
    actual = register_qb_rope_op()(x, weight, scale, positions, table)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
