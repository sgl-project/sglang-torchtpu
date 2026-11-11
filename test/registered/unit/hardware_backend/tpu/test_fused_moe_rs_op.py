# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TPU fused-RS Torch wrapper."""

import importlib.util
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.moe.pallas import fused_moe_rs_op

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


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

    def test_token_padding_reuses_the_first_token_experts(self):
        """A zero-padded expert id would route every padding row to expert 0, making its
        owner read that expert's weights on every call at small batch."""
        import jax.numpy as jnp

        hidden = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        weights = jnp.ones((2, 3), dtype=jnp.float32)
        ids = jnp.array([[5, 9, 2], [7, 1, 4]], dtype=jnp.int32)
        padded_hidden, padded_weights, padded_ids = fused_moe_rs_op.pad_moe_tokens(
            hidden, weights, ids, padded_tokens=5
        )
        self.assertEqual(padded_ids.shape, (5, 3))
        for row in range(2, 5):
            self.assertEqual(list(padded_ids[row]), list(ids[0]))
        # Padding contributes nothing: zero activations and zero routing weights.
        self.assertEqual(float(padded_hidden[2:].sum()), 0.0)
        self.assertEqual(float(padded_weights[2:].sum()), 0.0)

    def test_token_padding_is_a_no_op_when_already_aligned(self):
        import jax.numpy as jnp

        ids = jnp.array([[1, 2]], dtype=jnp.int32)
        hidden = jnp.zeros((1, 4), dtype=jnp.float32)
        out = fused_moe_rs_op.pad_moe_tokens(hidden, hidden, ids, padded_tokens=1)
        self.assertIs(out[2], ids)

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
            torch.tensor([32], dtype=torch.int32),
        )
        self.assertIn("_tpu_fused_rs_group_offset", layer._buffers)
        self.assertIn("_tpu_fused_rs_group_offset", layer._non_persistent_buffers_set)
        self.assertEqual(build.call_count, 2)

    def test_expert_tp_needs_the_explicit_backend(self):
        """A TP-only MoE on the auto backend must keep its own path, not expert TP."""
        for backend, expected in (
            (MoeRunnerBackend.AUTO, False),
            (MoeRunnerBackend.PALLAS_TPU_FUSED_RS, True),
        ):
            with patch(
                "sglang.srt.layers.moe.get_moe_runner_backend", return_value=backend
            ):
                self.assertEqual(
                    fused_moe_rs_op.is_fused_moe_rs_expert_tp(1, 8, 8), expected
                )

    def test_prepare_expert_tp_offsets_by_tp_rank(self):
        layer = torch.nn.Module()
        layer.num_local_experts = 256
        layer.w13_weight = torch.empty(1)
        parallel = Mock(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=8,
            moe_tp_rank=3,
            tp_size=8,
            tp_rank=3,
        )

        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.layers.moe.get_moe_runner_backend",
                return_value=MoeRunnerBackend.PALLAS_TPU_FUSED_RS,
            ),
            patch.object(fused_moe_rs_op, "_build_fused_moe_rs_op"),
            patch("sglang_tpu.oneshot_collectives.get_oneshot_all_gather_op"),
        ):
            fused_moe_rs_op.prepare_fused_moe_rs(layer)

        self.assertTrue(layer.__dict__["_tpu_fused_rs_expert_tp"])
        # Rank 3's slices are virtual experts [768, 1024).
        torch.testing.assert_close(
            layer._tpu_fused_rs_group_offset, torch.tensor([768], dtype=torch.int32)
        )

    def test_expert_tp_routing_matches_virtual_expert_sort(self):
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.moe.pallas.fused_rs.fused_moe import (
            _compute_expert_tp_routing,
            _compute_rs_routing,
        )

        num_experts, topk, ep_size, tokens = 16, 4, 4, 6
        rng = np.random.default_rng(0)
        ids = np.stack(
            [rng.choice(num_experts, topk, replace=False) for _ in range(tokens)]
        ).astype(np.int32)
        virtual_ids = (
            ids[:, :, None] + np.arange(ep_size, dtype=np.int32) * num_experts
        ).reshape(tokens, topk * ep_size)

        expected = _compute_rs_routing(
            jnp.asarray(virtual_ids),
            num_experts=num_experts * ep_size,
            topk=topk * ep_size,
        )
        actual = _compute_expert_tp_routing(
            jnp.asarray(ids), num_experts=num_experts, topk=topk, ep_size=ep_size
        )
        for name, a, e in zip(
            ("lhs_indices", "group_sizes", "output_indices", "topk_slot_indices"),
            actual,
            expected,
        ):
            with self.subTest(name=name):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(e))

    def test_expert_tp_gather_chunks_under_the_routed_row_limit(self):
        self.assertEqual(fused_moe_rs_op.fused_moe_rs_expert_tp_max_tokens(8, 8), 2048)
        calls = []

        def fake_call(**kwargs):
            rows = kwargs["hidden_states"].shape[0]
            calls.append(rows)
            padded = fused_moe_rs_op.fused_moe_rs_padded_tokens(rows, 8, 8)
            shard = torch.arange(padded, dtype=torch.float32)[:, None]
            return shard[: padded // 8] + kwargs["hidden_states"][0, 0]

        def fake_all_gather(shard, dim):
            # Rank-order shards of a padded chunk: row i holds chunk_start + i.
            base = shard[0, 0]
            return base + torch.arange(shard.shape[0] * 8, dtype=torch.float32)[:, None]

        tp_group = Mock(world_size=8, all_gather=Mock(side_effect=fake_all_gather))
        tokens = 5001
        hidden = torch.arange(tokens, dtype=torch.float32)[:, None]
        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=Mock(tp_group=tp_group),
            ),
            patch.object(fused_moe_rs_op, "fused_moe_rs_call", side_effect=fake_call),
        ):
            output = fused_moe_rs_op.fused_moe_rs_gather(
                hidden_states=hidden.contiguous(),
                w1=torch.empty(1),
                w2=torch.empty(1),
                w1_scale=torch.empty(1),
                w2_scale=torch.empty(1),
                topk_weights=torch.zeros(tokens, 8),
                topk_ids=torch.zeros(tokens, 8, dtype=torch.int32),
                post_expert_norm_weight=torch.empty(1),
                group_offset=torch.empty(1),
                topk=8,
                activation="silu",
                has_post_expert_norm=False,
                fp8_direct_write=False,
                world_size=8,
                expert_tp=True,
            )

        self.assertEqual(calls, [2048, 2048, 905])
        torch.testing.assert_close(
            output, torch.arange(tokens, dtype=torch.float32)[:, None]
        )

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
