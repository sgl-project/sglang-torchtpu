"""Unit tests for the TPU INT4 MoE load-time transforms.

The offset-binary nibble unpack must invert the checkpoint packing exactly, and
the vmem-fp8 preparation must keep the packed words, produce normalized group
and channel scales once, and refuse unsupported group sizes before mutating.
"""

import unittest

import torch
from torch.nn.parameter import Parameter

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.w4a16_moe import (
    _unpack_int4_offset_binary,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_EXPERTS = 2
K = 64
N = 16
GROUP_SIZE = 16


def _pack_offset_binary(q: torch.Tensor) -> torch.Tensor:
    """[E, K, N] int4 values in [-8, 7] -> [E, K//8, N] int32, LSB-first."""
    e, k, n = q.shape
    nibbles = (q.to(torch.int64) + 8).reshape(e, k // 8, 8, n)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64).view(1, 1, 8, 1)
    words = (nibbles << shifts).sum(dim=2)
    # Wrap to signed int32 (nibble 7 can set the sign bit).
    return words.to(torch.int32)


class _FakeMoELayer(torch.nn.Module):
    def __init__(self, packed13, scale13, packed2, scale2):
        super().__init__()
        self.w13_weight_packed = Parameter(packed13, requires_grad=False)
        self.w13_weight_scale = Parameter(scale13, requires_grad=False)
        self.w2_weight_packed = Parameter(packed2, requires_grad=False)
        self.w2_weight_scale = Parameter(scale2, requires_grad=False)


class TestUnpackInt4OffsetBinary(CustomTestCase):
    def test_roundtrip_inverts_packing(self):
        torch.manual_seed(0)
        q = torch.randint(-8, 8, (NUM_EXPERTS, K, N))
        unpacked = _unpack_int4_offset_binary(_pack_offset_binary(q))
        self.assertEqual(unpacked.dtype, torch.int8)
        torch.testing.assert_close(unpacked.to(q.dtype), q)


class TestVmemFp8Preparation(CustomTestCase):
    def test_packed_storage_scales_and_idempotence(self):
        from sglang.srt.environ import envs
        from sglang_tpu.w4a16_moe import (
            prepare_tpu_w4a16_moe_weights,
        )

        torch.manual_seed(4)
        q = torch.randint(-8, 8, (2, 64, 16))
        q[:, :, 0] = 0
        scales = (torch.rand(2, 2, 16) * 0.1).bfloat16()
        packed = _pack_offset_binary(q)
        layer = _FakeMoELayer(
            packed.clone(), scales.clone(), packed.clone(), scales.clone()
        )
        dequant = q.bfloat16() * scales.repeat_interleave(32, dim=1)
        channel = (dequant.float().abs().amax(dim=1, keepdim=True) / 448).clamp_min(
            1e-12
        )
        normalized = scales.float() / channel
        with envs.SGLANG_TPU_INT4_MOE_MODE.override("vmem-fp8"):
            prepare_tpu_w4a16_moe_weights(layer, 32)
            for prefix in ("w13", "w2"):
                weight = getattr(layer, prefix + "_weight_packed")
                group_scale = getattr(layer, prefix + "_weight_scale")
                channel_scale = getattr(layer, prefix + "_vmem_channel_scale")
                self.assertEqual(weight.dtype, torch.int32)
                self.assertEqual(weight.numel(), packed.numel())
                torch.testing.assert_close(weight, packed ^ -2004318072, rtol=0, atol=0)
                torch.testing.assert_close(
                    group_scale, normalized.unsqueeze(2), rtol=0, atol=0
                )
                torch.testing.assert_close(channel_scale, channel, rtol=0, atol=0)
                self.assertIsNone(group_scale.data._base)
                self.assertTrue(torch.isfinite(group_scale).all())
            before = layer.w13_weight_packed.clone()
            prepare_tpu_w4a16_moe_weights(layer, 32)
            torch.testing.assert_close(layer.w13_weight_packed, before, rtol=0, atol=0)

    def test_unsupported_groups_fail_before_mutation(self):
        from sglang.srt.environ import envs
        from sglang_tpu.w4a16_moe import (
            prepare_tpu_w4a16_moe_weights,
        )

        packed = torch.zeros(2, 8, 16, dtype=torch.int32)
        scales = torch.ones(2, 4, 16, dtype=torch.bfloat16)
        layer = _FakeMoELayer(
            packed.clone(), scales.clone(), packed.clone(), scales.clone()
        )
        with envs.SGLANG_TPU_INT4_MOE_MODE.override("vmem-fp8"):
            with self.assertRaisesRegex(ValueError, "group size 32"):
                prepare_tpu_w4a16_moe_weights(layer, 16)
        torch.testing.assert_close(layer.w13_weight_packed, packed, rtol=0, atol=0)


class TestScaleStorage(CustomTestCase):
    def _layer(self):
        torch.manual_seed(2)
        q = torch.randint(-8, 8, (2, 64, 16))
        scales = (torch.rand(2, 2, 16) * 0.1).bfloat16()
        packed = _pack_offset_binary(q)
        layer = _FakeMoELayer(
            packed.clone(), scales.clone(), packed.clone(), scales.clone()
        )
        return layer, scales

    def test_default_stores_f32_scales(self):
        from sglang_tpu.envs import envs
        from sglang_tpu.w4a16_moe import (
            prepare_tpu_w4a16_moe_weights,
        )

        layer, scales = self._layer()
        with envs.SGLANG_TPU_INT4_MOE_FP32_SCALES.override(True):
            prepare_tpu_w4a16_moe_weights(layer, 32)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        torch.testing.assert_close(
            layer.w2_weight_scale, scales.float().unsqueeze(2), rtol=0, atol=0
        )

    def test_opt_out_keeps_bf16_scales(self):
        from sglang_tpu.envs import envs
        from sglang_tpu.w4a16_moe import (
            prepare_tpu_w4a16_moe_weights,
        )

        layer, scales = self._layer()
        with envs.SGLANG_TPU_INT4_MOE_FP32_SCALES.override(False):
            prepare_tpu_w4a16_moe_weights(layer, 32)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.bfloat16)
        self.assertEqual(tuple(layer.w13_weight_scale.shape), (2, 2, 1, 16))
        torch.testing.assert_close(
            layer.w2_weight_scale, scales.unsqueeze(2), rtol=0, atol=0
        )

    def test_vmem_fp8_ignores_the_opt_out(self):
        from sglang.srt.environ import envs
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.w4a16_moe import (
            prepare_tpu_w4a16_moe_weights,
        )

        layer, _ = self._layer()
        with (
            tpu_envs.SGLANG_TPU_INT4_MOE_FP32_SCALES.override(False),
            envs.SGLANG_TPU_INT4_MOE_MODE.override("vmem-fp8"),
        ):
            prepare_tpu_w4a16_moe_weights(layer, 32)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)


class TestResolveInt4MoeMode(CustomTestCase):
    def test_unset_and_empty_mean_the_default_w4a16_path(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.tpu_moe_utils import resolve_int4_moe_mode

        with envs.SGLANG_TPU_INT4_MOE_MODE.override(None):
            self.assertIsNone(resolve_int4_moe_mode())
        with envs.SGLANG_TPU_INT4_MOE_MODE.override(""):
            self.assertIsNone(resolve_int4_moe_mode())

    def test_named_modes_pass_through(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.tpu_moe_utils import resolve_int4_moe_mode

        for mode in ("vmem-bf16", "vmem-fp8"):
            with self.subTest(mode=mode):
                with envs.SGLANG_TPU_INT4_MOE_MODE.override(mode):
                    self.assertEqual(resolve_int4_moe_mode(), mode)

    def test_unknown_mode_raises(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.tpu_moe_utils import resolve_int4_moe_mode

        with envs.SGLANG_TPU_INT4_MOE_MODE.override("w4a8"):
            with self.assertRaisesRegex(ValueError, "Unsupported TPU INT4 MoE mode"):
                resolve_int4_moe_mode()


if __name__ == "__main__":
    unittest.main()
