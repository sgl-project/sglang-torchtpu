# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TPU fused-RS Torch wrapper."""

import importlib.util
import os
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.pallas import fused_moe_rs_op
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestFusedMoeRsOp(CustomTestCase):
    @staticmethod
    def _configure_export_env():
        from sglang.srt.hardware_backend.tpu import pallas_export

        os.environ["WORLD_SIZE"] = "8"
        pallas_export.configure_jax_export_env(8)

    def test_padding_satisfies_ep_and_routed_row_alignment(self):
        cases = (
            # num_tokens, topk, ep_size, padded_tokens
            (8, 1, 8, 16),
            (4, 2, 4, 8),
            (16, 8, 16, 16),
            (17, 8, 16, 32),
        )
        for num_tokens, topk, ep_size, expected in cases:
            with self.subTest(num_tokens=num_tokens, topk=topk, ep_size=ep_size):
                actual = fused_moe_rs_op.fused_moe_rs_padded_tokens(
                    num_tokens, topk, ep_size
                )
                self.assertEqual(actual, expected)
                self.assertEqual(actual % ep_size, 0)
                self.assertEqual((actual * topk) % 16, 0)

    def test_prepare_sets_physical_rank_metadata(self):
        layer = torch.nn.Module()
        layer.num_local_experts = 16
        layer.w13_weight = torch.empty(1)

        with (
            patch.dict(
                os.environ,
                {"TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false"},
            ),
            patch.object(
                fused_moe_rs_op,
                "_get_physical_rank_metadata",
                return_value=(4, (0, 1, 6, 7, 2, 3, 4, 5)),
            ),
            patch(
                "sglang.srt.distributed.get_tp_group",
                return_value=Mock(rank=2),
            ),
            patch.object(fused_moe_rs_op, "_build_fused_moe_rs_op") as build,
            patch.object(
                fused_moe_rs_op, "_build_fused_moe_rs_gather_op"
            ) as build_gather,
        ):
            fused_moe_rs_op.prepare_fused_moe_rs(layer)
            fused_moe_rs_op.prepare_fused_moe_rs(layer)
            self.assertEqual(
                os.environ["TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"],
                "false",
            )

        torch.testing.assert_close(
            layer._tpu_fused_rs_physical_order,
            torch.tensor([0, 1, 6, 7, 2, 3, 4, 5], dtype=torch.int32),
        )
        torch.testing.assert_close(
            layer._tpu_fused_rs_group_offset,
            torch.tensor([[64, 4]], dtype=torch.int32),
        )
        self.assertIn("_tpu_fused_rs_physical_order", layer._buffers)
        self.assertIn("_tpu_fused_rs_group_offset", layer._buffers)
        self.assertIn("_tpu_fused_rs_physical_order", layer._non_persistent_buffers_set)
        self.assertIn("_tpu_fused_rs_group_offset", layer._non_persistent_buffers_set)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(build_gather.call_count, 2)

    def test_gather_uses_separate_compiled_collective(self):
        physical_order_to_tp_rank = torch.tensor([0, 1, 6, 7, 2, 3, 4, 5])
        token_shard = torch.zeros(1, 1)
        expected = torch.arange(8).reshape(8, 1)

        with (
            patch.object(
                fused_moe_rs_op,
                "fused_moe_rs_call",
                return_value=token_shard,
            ) as call,
            patch.object(
                fused_moe_rs_op,
                "fused_moe_rs_gather_call",
                return_value=expected,
            ) as gather_call,
        ):
            output = fused_moe_rs_op.fused_moe_rs_gather(
                hidden_states=torch.zeros(8, 1),
                w1=torch.empty(1),
                w2=torch.empty(1),
                w1_scale=torch.empty(1),
                w2_scale=torch.empty(1),
                topk_weights=torch.empty(1),
                topk_ids=torch.empty(1),
                post_expert_norm_weight=torch.empty(1),
                group_offset=torch.empty(1),
                topk=1,
                activation="silu",
                has_post_expert_norm=False,
                fp8_direct_write=True,
                physical_order_to_tp_rank=physical_order_to_tp_rank,
            )

        torch.testing.assert_close(output, expected)
        self.assertNotIn("physical_order_to_tp_rank", call.call_args.kwargs)
        gather_call.assert_called_once_with(token_shard, physical_order_to_tp_rank)

    @unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
    def test_sender_counts_are_transposed_by_physical_destination(self):
        import jax.numpy as jnp

        from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import (
            _compute_sender_payload,
            _transpose_sender_payloads,
        )

        sender_payload = _compute_sender_payload(
            jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
            jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
            expert_shard_id=jnp.int32(0),
            physical_device_id=jnp.int32(0),
            ep_size=4,
        )
        self.assertEqual(sender_payload.tolist(), [0, 0, 1, 0, 0])

        all_sender_payloads = jnp.asarray(
            [
                [2, 5, 6, 7, 8],
                [0, 9, 10, 11, 12],
                [3, 13, 14, 15, 16],
                [1, 17, 18, 19, 20],
            ],
            dtype=jnp.int32,
        )
        recv_counts = _transpose_sender_payloads(
            all_sender_payloads, physical_device_id=jnp.int32(1), ep_size=4
        )
        self.assertEqual(recv_counts.tolist(), [10, 18, 6, 14])

    @unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
    def test_decode_routing_exchanges_exact_sender_counts(self):
        self._configure_export_env()
        import jax.numpy as jnp

        from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import (
            _compute_rs_routing,
        )

        topk_indices = jnp.zeros((64, 8), dtype=jnp.int32)
        gathered_payloads = (
            jnp.zeros((8, 9), dtype=jnp.int32)
            .at[:, 0]
            .set(jnp.arange(8, dtype=jnp.int32))
        )
        with patch("jax.lax.all_gather", return_value=gathered_payloads) as all_gather:
            _, _, _, peer_recv_counts = _compute_rs_routing(
                topk_indices,
                num_experts=160,
                topk=8,
                ep_size=8,
                ep_axis_name="model",
                physical_device_id=jnp.int32(0),
                expert_shard_id=jnp.int32(0),
            )

        all_gather.assert_called_once()
        self.assertEqual(peer_recv_counts.tolist(), [0] * 8)

    @unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
    def test_prefill_routing_transposes_sender_counts_by_physical_destination(self):
        self._configure_export_env()
        import jax.numpy as jnp

        from sglang.srt.layers.moe.pallas.fused_rs.fused_moe import (
            _compute_rs_routing,
        )

        topk_indices = (jnp.arange(512 * 8, dtype=jnp.int32) % 160).reshape(512, 8)
        gathered_payloads = jnp.asarray(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                [1, 9, 10, 11, 12, 13, 14, 15, 16],
                [2, 17, 18, 19, 20, 21, 22, 23, 24],
                [3, 25, 26, 27, 28, 29, 30, 31, 32],
                [4, 33, 34, 35, 36, 37, 38, 39, 40],
                [5, 41, 42, 43, 44, 45, 46, 47, 48],
                [6, 49, 50, 51, 52, 53, 54, 55, 56],
                [7, 57, 58, 59, 60, 61, 62, 63, 64],
            ],
            dtype=jnp.int32,
        )
        with patch("jax.lax.all_gather", return_value=gathered_payloads) as all_gather:
            _, _, _, peer_recv_counts = _compute_rs_routing(
                topk_indices,
                num_experts=160,
                topk=8,
                ep_size=8,
                ep_axis_name="model",
                physical_device_id=jnp.int32(1),
                expert_shard_id=jnp.int32(0),
            )

        all_gather.assert_called_once()
        self.assertEqual(peer_recv_counts.tolist(), [2, 10, 18, 26, 34, 42, 50, 58])

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_unmocked_build_fused_moe_rs_op_exports_mlir(self):
        self._configure_export_env()
        op = fused_moe_rs_op._build_fused_moe_rs_op()
        gather_op = fused_moe_rs_op._build_fused_moe_rs_gather_op()
        self.assertIsNotNone(op)
        self.assertIsNotNone(gather_op)
        self.assertTrue(hasattr(torch.ops.sglang_tpu_pallas, "fused_moe_rs"))
        self.assertTrue(hasattr(torch.ops.sglang_tpu_pallas, "fused_moe_rs_gather"))

    @unittest.skipUnless(
        importlib.util.find_spec("torch_tpu"),
        "requires TorchTPU",
    )
    def test_dynamo_faketensor_export_fused_moe_rs(self):
        self._configure_export_env()

        class FusedMoeRsModule(torch.nn.Module):
            def forward(
                self,
                hidden_states: torch.Tensor,
                w1: torch.Tensor,
                w2: torch.Tensor,
                w1_scale: torch.Tensor,
                w2_scale: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                post_expert_norm_weight: torch.Tensor,
            ) -> torch.Tensor:
                return fused_moe_rs_op.fused_moe_rs_gather(
                    hidden_states=hidden_states,
                    w1=w1,
                    w2=w2,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    post_expert_norm_weight=post_expert_norm_weight,
                    group_offset=torch.tensor(
                        [[0, 0]], dtype=torch.int32, device=hidden_states.device
                    ),
                    topk=8,
                    activation="silu",
                    has_post_expert_norm=False,
                    fp8_direct_write=True,
                    physical_order_to_tp_rank=torch.arange(
                        8, dtype=torch.int32, device=hidden_states.device
                    ),
                )

        fused_moe_rs_op._build_fused_moe_rs_op()
        fused_moe_rs_op._build_fused_moe_rs_gather_op()
        mod = FusedMoeRsModule()
        tokens = 128
        hidden_size = 6144
        num_local_experts = 20
        inter_size = 2560
        topk = 8
        example_inputs = (
            torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="meta"),
            torch.randn(
                num_local_experts,
                hidden_size,
                inter_size * 2,
                dtype=torch.float8_e4m3fn,
                device="meta",
            ),
            torch.randn(
                num_local_experts,
                inter_size,
                hidden_size,
                dtype=torch.float8_e4m3fn,
                device="meta",
            ),
            torch.randn(
                num_local_experts,
                1,
                1,
                inter_size * 2,
                dtype=torch.float32,
                device="meta",
            ),
            torch.randn(
                num_local_experts,
                1,
                1,
                hidden_size,
                dtype=torch.float32,
                device="meta",
            ),
            torch.randn(tokens, topk, dtype=torch.float32, device="meta"),
            torch.zeros(tokens, topk, dtype=torch.int32, device="meta"),
            torch.ones(1, hidden_size, dtype=torch.bfloat16, device="meta"),
        )
        exported = torch.export.export(mod, example_inputs)
        self.assertIsNotNone(exported)


if __name__ == "__main__":
    unittest.main()
