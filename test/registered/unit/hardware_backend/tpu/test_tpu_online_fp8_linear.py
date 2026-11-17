"""Online fp8 quantization of dense linear weights on TPU (#1767)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.quantization import fp8
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.fp8_linear import (
    _set_channelwise_fp8_weight,
    prepare_tpu_online_fp8_linear_weights,
)
from sglang_tpu.moe.glm_a2a_fused_moe_v2 import _prepare_shared

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FP8 = torch.float8_e4m3fn


def _online_fp8_method() -> Fp8LinearMethod:
    """An Fp8LinearMethod for an unquantized checkpoint, as `Fp8Config()` builds."""
    method = object.__new__(Fp8LinearMethod)
    method.block_quant = False
    method.use_mxfp8 = False
    method.is_checkpoint_fp8_serialized = False
    method.cutlass_fp8_supported = False
    method.use_marlin = False
    return method


def _quantize_on_tpu(weight: torch.Tensor) -> nn.Module:
    layer = nn.Module()
    layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
    with patch.object(fp8, "_is_tpu", True):
        _online_fp8_method().process_weights_after_loading(layer)
    return layer


def _replica_qkv_shards() -> tuple[torch.Tensor, torch.Tensor]:
    """Two TP ranks' fused q|k|v shards that share one replicated kv head.

    Each rank holds different q rows. Rank 0's q rows have the larger amax,
    so a shard-wide scale would differ between the replicas.
    """
    generator = torch.Generator().manual_seed(0)
    kv = torch.randn(2 * 16, 32, generator=generator)
    q_rank0 = torch.randn(32, 32, generator=generator) * 8.0
    q_rank1 = torch.randn(32, 32, generator=generator)
    return torch.cat([q_rank0, kv]), torch.cat([q_rank1, kv])


class TestTpuOnlineFp8Linear(CustomTestCase):
    def test_replicated_kv_rows_are_byte_identical_across_ranks(self):
        shard0, shard1 = _replica_qkv_shards()

        layer0 = _quantize_on_tpu(shard0)
        layer1 = _quantize_on_tpu(shard1)

        # Post-load layout is a contiguous [K, N]; kv columns follow the 32 q
        # columns.
        self.assertTrue(layer0.weight.is_contiguous())
        self.assertTrue(layer1.weight.is_contiguous())
        kv0 = layer0.weight.view(torch.uint8)[:, 32:]
        kv1 = layer1.weight.view(torch.uint8)[:, 32:]
        self.assertTrue(torch.equal(kv0, kv1))
        self.assertTrue(torch.equal(layer0.weight_scale[32:], layer1.weight_scale[32:]))

    def test_per_tensor_scale_would_split_the_replicas(self):
        # Guards the premise of the test above: without per-row scales the
        # same kv rows get different fp8 bytes on the two ranks.
        shard0, shard1 = _replica_qkv_shards()

        q0, _ = fp8.input_to_float8(shard0, dtype=_FP8)
        q1, _ = fp8.input_to_float8(shard1, dtype=_FP8)

        self.assertFalse(
            torch.equal(q0[32:].view(torch.uint8), q1[32:].view(torch.uint8))
        )

    def test_layer_keeps_post_load_layout_and_row_scales(self):
        weight = torch.tensor([[1.0, -2.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])

        layer = _quantize_on_tpu(weight)

        self.assertEqual(layer.weight.dtype, _FP8)
        self.assertEqual(tuple(layer.weight.shape), (4, 2))
        # apply_tpu_fp8_linear reads a contiguous [K, N] base tensor, not a
        # transposed view of the [N, K] weight.
        self.assertTrue(layer.weight.is_contiguous())
        self.assertIsNone(layer.weight.data._base)
        self.assertTrue(layer.weight_scale.is_contiguous())
        self.assertEqual(tuple(layer.weight_scale.shape), (2, 1))
        self.assertIsNone(layer.input_scale)
        fp8_max = torch.finfo(_FP8).max
        expected_scales = torch.tensor([[2.0 / fp8_max], [1e-12]])
        self.assertTrue(torch.equal(layer.weight_scale, expected_scales))
        # An all-zero row keeps the finite floor scale and quantizes to zeros.
        self.assertEqual(layer.weight[:, 1].float().tolist(), [0.0] * 4)
        self.assertEqual(
            layer.weight[:, 0].float().tolist(), [224.0, -448.0, 112.0, 0.0]
        )

    def test_channelwise_round_trip_is_within_fp8_precision(self):
        weight = torch.randn(64, 128, generator=torch.Generator().manual_seed(1))

        layer = nn.Module()
        layer.weight = nn.Parameter(weight.clone(), requires_grad=False)

        prepare_tpu_online_fp8_linear_weights(layer, dtype=_FP8)

        dequant = layer.weight.t().float() * layer.weight_scale
        relative_error = (dequant - weight).abs().max() / weight.abs().max()
        self.assertLess(relative_error.item(), 0.07)


class TestGlmSharedChannelwiseGather(CustomTestCase):
    def test_down_shards_keep_their_dequantized_values_after_replication(self):
        """Different TP channel scales must survive shared-weight replication."""
        down_weights = [
            torch.tensor([[448.0, -224.0], [112.0, 448.0]]).to(_FP8),
            torch.tensor([[224.0, 448.0], [-448.0, 112.0]]).to(_FP8),
        ]
        down_scales = [torch.tensor([0.01, 0.03]), torch.tensor([0.04, 0.015])]
        expected_down = torch.cat(
            [weight.float() * scale for weight, scale in zip(down_weights, down_scales)]
        )
        reference = nn.Module()
        _set_channelwise_fp8_weight(reference, expected_down.t(), _FP8)
        gate_up = [
            (torch.arange(8).reshape(2, 4) + rank * 8).to(_FP8) for rank in range(2)
        ]
        gate_up_scales = [torch.arange(1, 5).float() * (rank + 1) for rank in range(2)]
        expected_gate_up = torch.stack(
            [
                torch.cat(
                    [weight[:, branch * 2 : (branch + 1) * 2] for weight in gate_up],
                    dim=1,
                )
                for branch in range(2)
            ]
        )
        for rank in range(2):
            with self.subTest(rank=rank):
                peer = 1 - rank
                # Only the collective boundary is simulated. Every returned
                # tensor includes the actual local value produced by the binder.
                peers = iter(
                    [
                        gate_up[peer][:, :2],
                        gate_up[peer][:, 2:],
                        gate_up_scales[peer][:2],
                        gate_up_scales[peer][2:],
                        down_scales[peer],
                        reference.weight.data[peer * 2 : (peer + 1) * 2],
                    ]
                )

                def gather(value, dim):
                    other = next(peers)
                    return torch.cat(
                        [value, other] if rank == 0 else [other, value], dim=dim
                    )

                shared = SimpleNamespace(
                    gate_up_proj=SimpleNamespace(
                        weight=gate_up[rank],
                        weight_scale=gate_up_scales[rank],
                        weight_scale_inv=None,
                    ),
                    down_proj=SimpleNamespace(
                        weight=down_weights[rank],
                        weight_scale=down_scales[rank],
                        weight_scale_inv=None,
                    ),
                )
                moe = SimpleNamespace(shared_experts=shared, _shared_expert_tp1=False)
                parallel = SimpleNamespace(
                    tp_size=2, tp_group=SimpleNamespace(all_gather=gather)
                )
                self.assertTrue(_prepare_shared(moe, parallel))
                actual_down = moe._a2a_moe_v2_sw2.float() * moe._a2a_moe_v2_ss2
                relative_l2 = torch.linalg.vector_norm(
                    actual_down - expected_down
                ) / torch.linalg.vector_norm(expected_down)
                self.assertLess(float(relative_l2), 0.04)
                self.assertTrue(torch.isfinite(actual_down).all())
                torch.testing.assert_close(
                    moe._a2a_moe_v2_sw13.float(),
                    expected_gate_up.float(),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
