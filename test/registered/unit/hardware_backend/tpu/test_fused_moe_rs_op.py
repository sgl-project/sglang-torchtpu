# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TPU fused-RS Torch wrapper."""

import importlib.util
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.moe.pallas import fused_moe_rs_op

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestFusedMoeRsOp(CustomTestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_native_execution_preserves_mesh_assignment_and_local_shapes(self):
        from torch_tpu._internal.pallas import pallas

        from sglang.test.cpu_test_utils import get_cpu_export_mesh

        device_ids = (8, 9, 12, 13, 14, 15, 10, 11, 0, 1, 4, 5, 6, 7, 2, 3)
        mesh = get_cpu_export_mesh(("data", "model"), (1, 16), device_ids)

        def tensor(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device="meta")

        inputs = (
            tensor((256, 4096)),
            tensor((16, 4096, 8192), torch.float8_e4m3fn),
            tensor((16, 4096, 4096), torch.float8_e4m3fn),
            tensor((16, 1, 1, 8192)),
            tensor((16, 1, 1, 4096)),
            tensor((256, 8)),
            tensor((256, 8), torch.int32),
            tensor((1,)),
            tensor((2,), torch.int32),
            8,
            "silu",
            False,
            False,
        )

        def placeholder(aval, mesh=None, partition_spec=None):
            return tensor(
                pallas.get_local_shape(aval.shape, mesh, partition_spec),
                pallas.JAX_TO_TORCH_DTYPE_MAP[aval.dtype],
            )

        with (
            patch.object(fused_moe_rs_op, "_FUSED_MOE_RS_OP", None),
            patch.object(fused_moe_rs_op, "_get_mesh", return_value=mesh),
            patch.object(torch.library, "custom_op", return_value=Mock()) as register,
            patch.object(pallas, "torch_placeholder", side_effect=placeholder),
            patch.object(pallas.tpu_torch_pallas, "register_custom_kernel"),
            patch.object(
                pallas.tpu_torch_pallas,
                "call_custom_kernel",
                side_effect=lambda *_, **kwargs: kwargs["output_shapes"],
            ) as execute,
        ):
            fused_moe_rs_op._build_fused_moe_rs_op()
            native_call = register.call_args.args[1]
            placeholders = pallas.jax_placeholders(
                inputs,
                mesh=native_call.mesh,
                partition_specs=native_call.input_partition_specs,
            )
            self.assertEqual(len(placeholders), len(inputs))
            for original, converted in zip(inputs, placeholders):
                if isinstance(original, torch.Tensor):
                    self.assertEqual(converted.shape, original.shape)
            output = native_call(*inputs)

        self.assertEqual(output.shape, (16, 4096))
        self.assertEqual(output.dtype, torch.bfloat16)
        execute.assert_called_once()
        self.assertEqual(
            execute.call_args.kwargs["device_assignment"], list(device_ids)
        )

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

    def test_prepare_sets_logical_expert_ownership(self):
        layer = torch.nn.Module()
        layer.num_local_experts = 16
        layer.w13_weight = torch.empty(1)

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=Mock(moe_ep_rank=2),
            ),
            patch.object(fused_moe_rs_op, "_build_fused_moe_rs_op") as build,
        ):
            fused_moe_rs_op.prepare_fused_moe_rs(layer)
            fused_moe_rs_op.prepare_fused_moe_rs(layer)

        torch.testing.assert_close(
            layer._tpu_fused_rs_group_offset,
            torch.tensor([32, 2], dtype=torch.int32),
        )
        self.assertIn("_tpu_fused_rs_group_offset", layer._buffers)
        self.assertIn("_tpu_fused_rs_group_offset", layer._non_persistent_buffers_set)
        self.assertEqual(build.call_count, 2)

    def test_gather_preserves_logical_rank_order(self):
        rank_order_shards = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]])
        tp_group = Mock(
            world_size=8,
            all_gather=Mock(return_value=rank_order_shards),
        )
        token_shard = torch.zeros(2, 1)

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=Mock(tp_group=tp_group),
            ),
            patch.object(
                fused_moe_rs_op,
                "fused_moe_rs_call",
                return_value=token_shard,
            ),
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
                world_size=8,
            )

        torch.testing.assert_close(output, torch.arange(8).reshape(8, 1))
        tp_group.all_gather.assert_called_once_with(token_shard, dim=0)

    def test_keep_token_shard_skips_gather(self):
        token_shard = torch.tensor([[4], [5]], dtype=torch.bfloat16)
        tp_group = Mock(world_size=8)

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=Mock(tp_group=tp_group),
            ),
            patch.object(
                fused_moe_rs_op,
                "fused_moe_rs_call",
                return_value=token_shard,
            ),
        ):
            output = fused_moe_rs_op.fused_moe_rs_gather(
                hidden_states=torch.zeros(16, 1),
                w1=torch.empty(1),
                w2=torch.empty(1),
                w1_scale=torch.empty(1),
                w2_scale=torch.empty(1),
                topk_weights=torch.ones(16, 1),
                topk_ids=torch.zeros(16, 1, dtype=torch.int32),
                post_expert_norm_weight=torch.empty(1),
                group_offset=torch.empty(1),
                topk=8,
                activation="silu",
                has_post_expert_norm=False,
                fp8_direct_write=False,
                world_size=8,
                keep_token_shard=True,
            )

        torch.testing.assert_close(output, token_shard)
        tp_group.all_gather.assert_not_called()

    def test_keep_token_shard_pads_each_logical_rank_block(self):
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(8, 1)
        topk_weights = hidden_states.clone()
        topk_ids = torch.arange(8, dtype=torch.int32).reshape(8, 1)
        token_shard = torch.tensor([[5], [0]], dtype=torch.bfloat16)

        with patch.object(
            fused_moe_rs_op,
            "fused_moe_rs_call",
            return_value=token_shard,
        ) as fused:
            output = fused_moe_rs_op.fused_moe_rs_gather(
                hidden_states=hidden_states,
                w1=torch.empty(1),
                w2=torch.empty(1),
                w1_scale=torch.empty(1),
                w2_scale=torch.empty(1),
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                post_expert_norm_weight=torch.empty(1),
                group_offset=torch.empty(1),
                topk=1,
                activation="silu",
                has_post_expert_norm=False,
                fp8_direct_write=False,
                world_size=8,
                keep_token_shard=True,
            )

        expected = torch.tensor(
            [
                [0],
                [0],
                [1],
                [0],
                [2],
                [0],
                [3],
                [0],
                [4],
                [0],
                [5],
                [0],
                [6],
                [0],
                [7],
                [0],
            ],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(fused.call_args.kwargs["hidden_states"], expected)
        torch.testing.assert_close(fused.call_args.kwargs["topk_weights"], expected)
        torch.testing.assert_close(
            fused.call_args.kwargs["topk_ids"], expected.to(torch.int32)
        )
        torch.testing.assert_close(output, torch.tensor([[5]], dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
